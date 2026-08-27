package router

// Relay — one discrimen routing through another.
//
// A relay is a second discrimen, somewhere else, whose fleet this one may
// route to. It is NOT a provider row pointed at the other router's port. The
// difference is the whole point of the file, and it is three things:
//
//	The upstream keeps the slots. A relayed request is dispatched to the other
//	router, which acquires the worker slot, queues, ranks and spills exactly as
//	it does for its own traffic. Two routers registering the SAME workers
//	directly would each keep their own idea of how busy those GPUs are, and both
//	would be wrong by the size of the other's queue. One pipeline, one accounting.
//
//	The profile crosses instead of being re-measured. Quality, capacity,
//	context, capabilities and the thinking dialect were already measured
//	upstream, by the same graded benchmark this binary carries — 130 questions
//	and several hundred thousand output tokens of somebody's GPU time. Running
//	it again from here would measure the same workers a second time to learn the
//	same numbers. "Measure, don't trust" is kept: they ARE measured, and the
//	bench_version on the wire is what says the two measurements are the same
//	measurement. A version mismatch falls back to profiling here.
//
//	Relayed traffic leaves no prompts upstream. A relay key marks its caller,
//	and a marked caller's request bodies are dropped before the log row is
//	written (see redactForRelay). The row itself stays — which backend served,
//	what it cost, how long it took — because that is capacity accounting rather
//	than content, and a per-key budget that stopped counting would be a budget
//	nobody could enforce.
//
// TRUST. A relay is a router you run. The downstream adopts the upstream's
// measurements sight unseen, so a hostile or merely careless upstream can claim
// quality 100 and be believed. That is the right trade between two halves of
// one fleet and the wrong one for a stranger's endpoint — for those there is
// /admin/providers, which measures what it is told.
//
// DIRECTION. Everything here is one of two halves, and a router can be both at
// once:
//
//	northbound (the relayed-TO side) — the relay key flag, GET /relay/fleet,
//	the loop guard, and the log redaction.
//
//	southbound (the relaying side) — the relay config rows, the refresh loop
//	that expands one relay into one backend per upstream worker, and the profile
//	import that certifies them without a probe.

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// ── Identity and the loop guard ─────────────────────────────────────────────

// relayHopHeader carries the chain of router ids a request has already passed
// through, oldest first. A router that finds its OWN id in the chain is looking
// at a request it already routed, so the fleets are cyclic and no amount of
// forwarding will find a worker.
const relayHopHeader = "X-LLM-Relay"

// relayMaxHops bounds the chain independently of the cycle check, for the shape
// the cycle check cannot see: a chain of distinct routers, each relaying to the
// next, that is simply too long to be anybody's intent. Four is past every
// topology worth having (a site, its peer, and a shared upstream is three) and
// far short of a request that spends its whole deadline in transit.
const relayMaxHops = 4

// settingRouterID is where this router's own identity lives. It is generated on
// first use and persisted, rather than configured: nothing an operator could
// type is more useful than a value that is unique by construction, and an
// environment variable that two deployments copy is precisely how a cycle
// becomes undetectable.
const settingRouterID = "router_id"

// routerID returns this router's stable identity, generating and persisting one
// on first call. Cached in memory afterwards.
//
// A failure to persist is NOT fatal and NOT retried into a second id: the
// process keeps the id it generated for as long as it runs, so the loop guard
// works within this process's lifetime even on a read-only volume. It would
// only be a new id after a restart, which costs nothing but the ability to
// recognise a cycle that predates the restart.
func (r *Router) routerID() string {
	if id, ok := r.selfID.Load().(string); ok && id != "" {
		return id
	}
	r.selfIDOnce.Do(func() {
		id := ""
		if r.logs != nil {
			if stored, err := r.logs.LoadSetting(context.Background(), settingRouterID); err == nil {
				id = strings.TrimSpace(stored)
			}
		}
		if id == "" {
			id = generatedRouterID()
			if r.logs != nil {
				if err := r.logs.SaveSetting(context.Background(), settingRouterID, id); err != nil {
					log.Printf("persist router id failed: %v (the relay loop guard will forget it on restart)", err)
				}
			}
		}
		r.selfID.Store(id)
	})
	id, _ := r.selfID.Load().(string)
	return id
}

// generatedRouterID mints an identity for a router that has none.
//
// The hostname fallback exists for the one case crypto/rand can fail, and is a
// fallback rather than the primary because two containers of the same image
// routinely share a hostname — and two routers with the same id would refuse
// each other's traffic as a loop.
func generatedRouterID() string {
	if tok, err := randomToken(8); err == nil {
		return "r-" + tok
	}
	if host, err := os.Hostname(); err == nil && strings.TrimSpace(host) != "" {
		return "r-" + strings.TrimSpace(host)
	}
	return "r-unnamed"
}

// relayChain reads the hop chain off a request. Empty for anything that did not
// arrive through a relay, which is every ordinary client.
func relayChain(req *http.Request) []string {
	raw := req.Header.Get(relayHopHeader)
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	out := []string{}
	for _, part := range strings.Split(raw, ",") {
		if hop := strings.TrimSpace(part); hop != "" {
			out = append(out, hop)
		}
	}
	return out
}

// refuseRelayLoop answers a request that has already been here, or one that has
// been forwarded too many times, and reports whether it did.
//
// 508 rather than a 400 or a 503: the condition is "this request has visited
// this router before", which is what Loop Detected means, and it is neither the
// caller's malformed input nor a capacity problem that retrying could clear. A
// client that retries a 508 gets the same answer, which is the truth.
func (r *Router) refuseRelayLoop(w http.ResponseWriter, req *http.Request) bool {
	chain := relayChain(req)
	if len(chain) == 0 {
		return false
	}
	self := r.routerID()
	for _, hop := range chain {
		if hop == self {
			log.Printf("relay loop refused: chain %s already contains this router (%s)", strings.Join(chain, "→"), self)
			writeError(w, http.StatusLoopDetected,
				"relay loop: this router (%s) already appears in the relay chain %s", self, strings.Join(chain, ","))
			return true
		}
	}
	if len(chain) >= relayMaxHops {
		log.Printf("relay hop limit refused: chain %s is %d hops", strings.Join(chain, "→"), len(chain))
		writeError(w, http.StatusLoopDetected,
			"relay chain too long: %d hops (limit %d)", len(chain), relayMaxHops)
		return true
	}
	return false
}

// stampRelayChain appends this router to the hop chain on a request being
// forwarded to a relay, preserving whatever chain the request arrived with.
//
// Only relay rows get the header. An ordinary worker has no use for it, and a
// strict endpoint that rejects unknown headers is a problem this does not need
// to create.
func (r *Router) stampRelayChain(out *http.Request, in *http.Request, backend *Backend) {
	if !isRelayRow(backend) {
		return
	}
	chain := []string{}
	if in != nil {
		chain = relayChain(in)
	}
	chain = append(chain, r.routerID())
	out.Header.Set(relayHopHeader, strings.Join(chain, ","))
}

// ── Northbound: what a relay may see ────────────────────────────────────────

// relayModelEntry is ONE UPSTREAM WORKER as the upstream describes it — not one
// per model. Every worker crosses separately, carrying its own measured numbers,
// so the downstream ranks remote workers against local ones as peers and picks
// the best of the whole union.
//
// This replaced a per-MODEL summary that merged endpoints together (quality and
// context minimised, speed averaged, slots summed). The merge was lossy in the
// one direction that mattered: workers differ — different quantisations, cards
// and context windows — and a single blended row hid exactly the differences the
// ranker exists to exploit. It also forced a rule per field about which
// direction to combine in, all of which is now simply deleted.
//
// ADDRESSING. ID is the upstream's own worker id, and it is what makes this
// work without any new protocol: the downstream stores it as the row's ServedID,
// patchForwardedBody stamps it into the outbound "model" field, and upstream
// backendServesModel already resolves a worker id as a model name. So naming the
// worker IS the routing — no pin header, no second mechanism. Model stays the
// real model name, for display only.
type relayModelEntry struct {
	ID    string `json:"id"`
	Model string `json:"model"`
	// Quality is the thinking-mode benchmark score; QualityNoThink is the same
	// benchmark with thinking forced off. Both cross, because selection reads
	// whichever matches the mode it is about to serve (see qualityFor). Sending
	// only Quality left every imported row at QualityNoThink=0, which made a
	// no-think request fall back to the THINKING score and over-rate the whole
	// relayed fleet on exactly the requests the two-score benchmark exists to
	// separate. Zero means "not measured upstream", not "scored zero".
	Quality        int      `json:"quality"`
	QualityNoThink int      `json:"quality_nothink,omitempty"`
	BenchVersion   int      `json:"bench_version"`
	ContextK       int      `json:"context_k"`
	Features       []string `json:"features"`
	Thinking       bool     `json:"thinking"`
	BaselineTPS    float64  `json:"baseline_tps,omitempty"`
	ObservedTPS    float64  `json:"observed_tps,omitempty"`
	PrefillTPS     float64  `json:"observed_prefill_tps,omitempty"`
	TTFTMillis     int64    `json:"ttft_ms,omitempty"`
	MaxConcurrency int      `json:"max_concurrency"`
	ActiveRequests int      `json:"active_requests"`
}

// relayFleetResponse is the whole of GET /relay/fleet.
//
// bench_version rides at the top level as well as on every entry because it is
// the compatibility statement, not a per-model fact: a downstream reads it once
// to decide whether any of these quality numbers mean what its own do.
type relayFleetResponse struct {
	RouterID     string            `json:"router_id"`
	BenchVersion int               `json:"bench_version"`
	Models       []relayModelEntry `json:"models"`
}

// handleRelayFleet publishes the fleet a relay credential may reach.
//
// Client scope, and then the relay flag on top: an ordinary client key must not
// be able to read the fleet's measured capacity and live occupancy, which is
// most of what /backends was moved behind the admin gate to protect. An admin
// key passes for the same reason it passes everywhere else — there is no
// authority a relay has that an admin does not.
func (r *Router) handleRelayFleet(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	ident, ok := r.requireClient(w, req)
	if !ok {
		return
	}
	// A downstream announces itself on the fetch as well as on its traffic, so a
	// pair of routers configured to relay to each other finds out here — at
	// discovery, once, with a message — rather than on every request forever.
	if r.refuseRelayLoop(w, req) {
		return
	}
	if !ident.Relay && ident.Role != roleAdmin {
		writeJSON(w, http.StatusForbidden, validationError{
			Message: "this endpoint needs a relay key (POST /admin/keys with {\"role\":\"client\",\"relay\":true})",
		})
		return
	}
	writeJSON(w, http.StatusOK, relayFleetResponse{
		RouterID:     r.routerID(),
		BenchVersion: benchmarkVersion,
		Models:       relayFleetFor(ident, r.registry.eligible()),
	})
}

// relayFleetFor renders the eligible fleet as one entry per WORKER, restricted
// to what this credential may reach.
//
// The restriction runs through allowsBackend, and what is published as the
// address is the worker's own id. Both follow from the same requirement: every
// name in this response has to be one the downstream can send back and have
// accepted, by a key that gates discovery and traffic through two different
// tests. Discovery runs through allowsBackend, which asks what a worker answers
// to; the traffic path compares the spelling the client sent to the allow-list
// literally. There used to be a relayModelName here that resolved that by
// publishing the allow-list's OWN spelling; it went when the id became the
// address, and mayNameWorker closes the same gap from the other side — a key
// that may be SERVED BY a worker may also NAME it. See
// TestRelayPublishedIDSurvivesTheTrafficGate.
func relayFleetFor(ident *identity, backends []*Backend) []relayModelEntry {
	out := make([]relayModelEntry, 0, len(backends))
	for _, b := range backends {
		if isEmbeddingsOnly(b) || !ident.allowsBackend(b) {
			continue
		}
		out = append(out, relayModelEntry{
			ID:             b.ID,
			Model:          b.Model,
			Quality:        b.Quality,
			QualityNoThink: b.QualityNoThink,
			BenchVersion:   benchmarkVersion,
			ContextK:       b.ContextK,
			Features:       append([]string(nil), b.Features...),
			Thinking:       b.Thinking,
			BaselineTPS:    b.BaselineTPS,
			ObservedTPS:    liveTPS(b),
			PrefillTPS:     b.ObservedPrefillTPS,
			TTFTMillis:     b.Certification.TTFTMillis,
			MaxConcurrency: effectiveSlots(b),
			ActiveRequests: b.ActiveRequests,
		})
	}
	// Id-sorted so a downstream diffing two fetches sees content changes and
	// not Go's randomised map order.
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// effectiveSlots is what one endpoint contributes to a pool's concurrency. An
// endpoint that declared no ceiling is not infinite capacity — it is capacity
// nobody measured — so it counts as uncappedNominalSlots rather than as zero,
// which would make a whole uncapped fleet report a pool of none.
//
// This USED to be "the same nominal figure the ranker already assumes for it",
// and for most rows it still is. It is no longer true across the board:
// queueSlots now returns 0 — no queue priced at all — for a row that is manual
// AND non-local, i.e. a hosted API somebody entered by hand and deliberately
// left the ceiling blank on, because charging a provider fronting thousands of
// customers a queue penalty off THIS router's four in-flight requests pushed
// traffic off a fast paid endpoint under mild local load.
//
// The divergence is deliberate rather than an oversight to reconcile. queueSlots
// answers "how long will this request wait here", where "nobody has told us and
// nobody sensibly could" is honestly answered by pricing no wait. This answers
// "how many requests may the downstream have in flight against this row", where
// the same silence cannot be answered by "as many as you like": the downstream
// builds a slot channel from this number (setRelayLoad → syncSlotsLocked), and
// a nil channel there means unbounded dispatch across a WAN at one upstream
// worker. Four is a bound the downstream can hold and the upstream can survive.
func effectiveSlots(b *Backend) int {
	if b.MaxConcurrency > 0 {
		return b.MaxConcurrency
	}
	return uncappedNominalSlots
}

// ── Northbound: what a relay leaves behind ──────────────────────────────────

// redactForRelay drops the request and response bodies from a log row written
// for a relay caller, and reports the row that should actually be stored.
//
// What survives is deliberate. Which endpoint served, what it cost, how long it
// took and which key spent it are capacity and billing facts about this
// router's own hardware, and an operator who could not see them could not
// answer "what is my fleet doing" — the question relayed traffic makes most
// pressing, because it is the traffic they did not send. The prompt and the
// answer are the other party's content, and the reason the flag exists.
//
// The error string is kept: it is written by this router about its own
// dispatch, and a relay whose requests are failing is exactly the case an
// operator has to be able to diagnose from here.
func redactForRelay(entry RequestLog, ident *identity) RequestLog {
	if ident == nil || !ident.Relay {
		return entry
	}
	entry.Input = ""
	entry.Output = ""
	return entry
}

// learnFromRelay reports whether an outcome from this caller may teach the
// online tier adapter and the background judge.
//
// It may not. Both learn about a tier THIS router chose for a prompt it
// classified, and a relayed request was classified downstream: the difficulty
// score attached to it is the downstream's, computed against the downstream's
// fleet, and the downstream is already learning from the same outcome. Feeding
// it here would count one signal twice and would move bins that describe a
// fleet this router cannot see. Same reasoning that keeps a named-model route
// and the expert panel out of the adapter.
func learnFromRelay(ident *identity) bool {
	return ident == nil || !ident.Relay
}

// validateRelayFlag refuses the relay flag on a role that cannot use it.
//
// A worker key never reaches the OpenAI surface, so marking one as a relay would
// suppress logging for traffic it can never send and grant a fleet endpoint it
// can never call — a setting with no effect, which is worse than an error
// because it reads as one that worked. An admin key is refused for the opposite
// reason: it already reaches /relay/fleet, and the flag's other half would
// silently stop recording the prompts of the most privileged credential there
// is. A relay is a client. Issue it one.
func validateRelayFlag(relay bool, role string) *validationError {
	if !relay || role == roleClient {
		return nil
	}
	return &validationError{
		Message: fmt.Sprintf("relay is only valid on a %s key (this one is %s)", roleClient, role),
		Param:   "relay",
	}
}

// ── Southbound: the relay config ────────────────────────────────────────────

// Relay is an upstream router this one may route through. Name is the local
// handle — it prefixes the backend ids the relay expands into, so it appears in
// /v1/models and in X-LLM-Backend-ID and is worth keeping short.
type Relay struct {
	Name      string    `json:"name"`
	URL       string    `json:"url"`
	APIKey    string    `json:"api_key,omitempty"`
	Enabled   bool      `json:"enabled"`
	UpdatedAt time.Time `json:"updated_at,omitempty"`
	// Everything below is measured, not configured — the state of the last
	// refresh, so /admin/relays answers "is this working" without a second call.
	Models       int       `json:"models"`
	BenchVersion int       `json:"bench_version,omitempty"`
	RTTMillis    float64   `json:"rtt_ms,omitempty"`
	LastFetch    time.Time `json:"last_fetch,omitempty"`
	LastError    string    `json:"last_error,omitempty"`
	RouterID     string    `json:"remote_router_id,omitempty"`
}

// relayKey is the lookup form of a relay name, lowercased for the same reason a
// group name is: it is a name this router owns, and a client reading it out of
// /v1/models should not have to guess its capitalisation.
func relayKey(name string) string { return strings.ToLower(strings.TrimSpace(name)) }

// relayBackendID is the id a relayed model gets in this router's registry.
//
// The separator is a colon rather than a slash: these ids reach the registry
// through path-addressed routes (/backends/{id}, /debug/backends/{id}/chat) and
// a slash would split one id into two path segments.
func relayBackendID(relay, model string) string { return relay + ":" + model }

// relayStore holds the configured relays and the last fleet each one reported.
// The database is canonical across restarts; this is the copy the refresh loop
// and the admin API read.
//
// The zero value works, for the same reason groupStore's does: a Router built
// by hand in a test must not have to remember to construct it.
type relayStore struct {
	mu     sync.RWMutex
	byName map[string]Relay
	fleets map[string][]relayModelEntry
}

func (s *relayStore) put(rel Relay) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.byName == nil {
		s.byName = map[string]Relay{}
	}
	// Preserve the measured half across a config write: an operator renaming a
	// relay or toggling it has learned nothing new about its round trip.
	if prev, ok := s.byName[relayKey(rel.Name)]; ok {
		rel.Models, rel.BenchVersion = prev.Models, prev.BenchVersion
		rel.RTTMillis, rel.LastFetch = prev.RTTMillis, prev.LastFetch
		rel.LastError, rel.RouterID = prev.LastError, prev.RouterID
	}
	s.byName[relayKey(rel.Name)] = rel
}

func (s *relayStore) lookup(name string) (Relay, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	rel, ok := s.byName[relayKey(name)]
	return rel, ok
}

func (s *relayStore) remove(name string) bool {
	key := relayKey(name)
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.byName[key]; !ok {
		return false
	}
	delete(s.byName, key)
	delete(s.fleets, key)
	return true
}

// list returns the relays name-sorted, api keys scrubbed. Nothing reads a relay
// key back out: it exists to be sent upstream, and an admin page that displayed
// it would be one screenshot away from leaking the other router's fleet.
func (s *relayStore) list() []Relay {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Relay, 0, len(s.byName))
	for _, rel := range s.byName {
		rel.APIKey = ""
		out = append(out, rel)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// enabled returns the relays the refresh loop should fetch, api keys intact.
func (s *relayStore) enabled() []Relay {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := []Relay{}
	for _, rel := range s.byName {
		if rel.Enabled {
			out = append(out, rel)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// noteFetch records the outcome of one refresh against a relay.
func (s *relayStore) noteFetch(name string, fleet *relayFleetResponse, rtt float64, err error) {
	key := relayKey(name)
	s.mu.Lock()
	defer s.mu.Unlock()
	rel, ok := s.byName[key]
	if !ok {
		return
	}
	rel.LastFetch = time.Now()
	if err != nil {
		rel.LastError = err.Error()
		s.byName[key] = rel
		return
	}
	rel.LastError = ""
	rel.RouterID = fleet.RouterID
	rel.BenchVersion = fleet.BenchVersion
	rel.Models = len(fleet.Models)
	// A smoothed round trip rather than the latest sample: this number is added
	// to every latency estimate the relay's rows produce, and one slow fetch
	// must not reprice the whole upstream fleet for the next fifteen seconds.
	if rel.RTTMillis <= 0 {
		rel.RTTMillis = rtt
	} else {
		rel.RTTMillis = 0.7*rel.RTTMillis + 0.3*rtt
	}
	s.byName[key] = rel
	if s.fleets == nil {
		s.fleets = map[string][]relayModelEntry{}
	}
	s.fleets[key] = fleet.Models
}

// ── Southbound: persistence ─────────────────────────────────────────────────

func (s *LogStore) SaveRelay(ctx context.Context, rel Relay) error {
	sealed, err := s.box.seal(rel.APIKey)
	if err != nil {
		return fmt.Errorf("encrypt relay api key: %w", err)
	}
	_, err = s.db.ExecContext(ctx, `INSERT INTO router_relays (name, url, api_key, enabled, updated_at)
		VALUES (?, ?, ?, ?, ?)
		ON CONFLICT(name) DO UPDATE SET url=excluded.url, api_key=excluded.api_key,
			enabled=excluded.enabled, updated_at=excluded.updated_at`,
		relayKey(rel.Name), rel.URL, sealed, boolInt(rel.Enabled),
		time.Now().UTC().Format(time.RFC3339Nano))
	return err
}

func (s *LogStore) LoadRelays(ctx context.Context) ([]Relay, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT name, url, api_key, enabled, updated_at FROM router_relays ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Relay{}
	for rows.Next() {
		var rel Relay
		var sealed, updated string
		var enabled int
		if err := rows.Scan(&rel.Name, &rel.URL, &sealed, &enabled, &updated); err != nil {
			return nil, err
		}
		rel.Enabled = enabled != 0
		rel.UpdatedAt, _ = time.Parse(time.RFC3339Nano, updated)
		if sealed != "" {
			plain, err := s.box.open(sealed)
			if err != nil {
				// Same rule as a backend's persisted key: an undecryptable
				// credential must not block startup. The relay will fail its next
				// fetch with a 401, which is visible on /admin/relays.
				log.Printf("decrypt relay api key for %q failed: %v (re-enter the key to restore it)", rel.Name, err)
			} else {
				rel.APIKey = plain
			}
		}
		out = append(out, rel)
	}
	return out, rows.Err()
}

func (s *LogStore) DeleteRelay(ctx context.Context, name string) error {
	res, err := s.db.ExecContext(ctx, `DELETE FROM router_relays WHERE name = ?`, relayKey(name))
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// loadRelays populates the in-memory store at startup.
func (r *Router) loadRelays(ctx context.Context) {
	if r.logs == nil {
		return
	}
	saved, err := r.logs.LoadRelays(ctx)
	if err != nil {
		log.Printf("load persisted relays failed: %v", err)
		return
	}
	for _, rel := range saved {
		r.relays.put(rel)
	}
	if len(saved) > 0 {
		log.Printf("loaded %d relay(s)", len(saved))
	}
}

// ── Southbound: the refresh loop ────────────────────────────────────────────

// relayRefreshLoop keeps the derived backend rows in step with what each
// upstream is currently serving.
//
// It runs on its own ticker rather than inside healthLoop, at the same interval,
// because one unreachable relay across a WAN would otherwise hold a health tick
// open for the whole local fleet. The first pass runs immediately: a restart
// should not leave the relayed half of the fleet missing for fifteen seconds.
func (r *Router) relayRefreshLoop() {
	interval := 15 * time.Second
	if r.cfg != nil && r.cfg.HealthInterval > 0 {
		interval = r.cfg.HealthInterval
	}
	r.refreshRelays()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for range ticker.C {
		r.refreshRelays()
	}
}

// refreshRelays reconciles every enabled relay, then removes whatever is left
// over.
//
// Serialised, because three things call it: the ticker, an admin write (which
// wants its change visible before it answers) and an admin delete. Two passes
// overlapping would each compute a set of live ids from a different view of the
// config and then prune against it — so the older pass could delete rows the
// newer one had just created, and the fleet would flicker for a tick. The lock
// is held across the network fetches, which is fine: this is the only thing that
// touches these rows, nothing else waits on it, and a refresh that is still
// running is a refresh that does not need to be started again.
func (r *Router) refreshRelays() {
	r.relayRefresh.Lock()
	defer r.relayRefresh.Unlock()
	live := map[string]bool{}
	for _, rel := range r.relays.enabled() {
		for _, id := range r.refreshRelay(rel) {
			live[id] = true
		}
	}
	r.pruneRelayRows(live)
}

// refreshRelay fetches one upstream's fleet and reconciles this router's rows
// against it, returning the backend ids that should now exist for it.
//
// A failed fetch returns the ids the relay ALREADY has rather than none. That is
// the difference between a WAN blip and a decommissioning: dropping the rows on
// the first failed poll would deregister a healthy upstream fleet because one
// request timed out, and the rows are health-checked independently anyway — an
// upstream that is really gone fails its own health check and stops being
// eligible, without anything here having to guess.
func (r *Router) refreshRelay(rel Relay) []string {
	fleet, rtt, err := r.fetchRelayFleet(rel)
	r.relays.noteFetch(rel.Name, fleet, rtt, err)
	if err != nil {
		log.Printf("relay %s: fetch fleet failed: %v", rel.Name, err)
		return r.relayRowIDs(rel.Name)
	}
	if fleet.RouterID != "" && fleet.RouterID == r.routerID() {
		log.Printf("relay %s: refuses to relay to itself (remote router id %s is this router)", rel.Name, fleet.RouterID)
		return nil
	}
	ids := make([]string, 0, len(fleet.Models))
	for _, entry := range fleet.Models {
		// An upstream that predates the per-worker fleet sends no id, only a
		// pooled model name. Fall back to it rather than skipping the entry:
		// the alternative is that upgrading this side first takes the whole
		// relayed fleet dark until the other side catches up. The name still
		// addresses the pool, which is exactly the old behaviour.
		if entry.ID == "" {
			entry.ID = entry.Model
		}
		if entry.ID == "" {
			continue
		}
		id := relayBackendID(rel.Name, entry.ID)
		ids = append(ids, id)
		r.applyRelayEntry(rel, id, sanitizeRelayEntry(rel.Name, entry), fleet.BenchVersion)
	}
	return ids
}

// sanitizeRelayEntry brings one upstream worker's numbers back inside the ranges
// this router's own fields are defined over, before anything downstream of here
// believes them.
//
// This is NOT a retreat from the trust statement at the top of the file. A relay
// is a router you run and its MEASUREMENTS are adopted sight unseen: a peer that
// says quality 78 is believed, and that is the whole point of importing a profile
// instead of re-running 130 questions on somebody else's GPUs. What is checked
// here is narrower and different in kind — that the numbers are inside the domain
// the fields have everywhere else in this binary. "Quality 78" is a claim. "Quality
// 100000" is not a claim about anything; it is a units error, a version skew, or a
// field that meant something else on the other side of the wire.
//
// The registration path already enforces exactly these ranges, and refuses the row
// outright when they are broken (normalizeRegistration: "quality must be 0..100",
// "max_concurrency %d exceeds the %d maximum"). The relay path went round it:
// applyRelayEntry builds a registration carrying only id/url/model/credential and
// applies every measured field afterwards through applyProfileIfGen and
// setRelayLoad, neither of which range-checks. So an upstream — or, far likelier, a
// version-skewed upstream whose field meant something else — could write values
// into this registry that no local code path can produce.
//
// max_concurrency is the second half of a bug whose first half is already fixed,
// and it is worth being precise about which half this is, because the dangerous
// half is not here. syncSlotsLocked clamps the number it builds a SLOT CHANNEL
// from: filling that channel one token at a time under the registry write lock is
// ~12.6ns each, so 1e9 of them stops the router routing for thirteen seconds.
// That was the outage, and that is fixed.
//
// What was left behind is that the clamp protects the channel and nothing else —
// b.MaxConcurrency keeps whatever the peer said. Three things then read the raw
// figure, and none of them is the outage:
//
//   - effectiveSlots republishes it, so the absurd number propagates to the next
//     router down the chain rather than stopping at the one that received it.
//     Each hop re-clamps its own channel and passes the raw figure on again.
//   - syncSlotsLocked logs its "clamping to" line BEFORE the no-change early
//     return, and setRelayLoad calls it on every refresh (applyProfileIfGen too,
//     whenever the two routers' benchmark versions match), so one bad row writes
//     a log line or two every fifteen seconds, forever.
//   - isFull reads it, and the dashboard renders it as "active / max" beside
//     every worker, so the fleet reads as having a billion slots somewhere. The
//     route preview publishes it as well.
//
// What it does NOT do, despite looking as though it should: distort the ranker.
// queueSlots hands it to loadPenalty, which takes min(n, slots) for the batch
// share and (n-slots)/slots for the queue — and for any occupancy a real fleet
// reaches, min(n, 1e9) and min(n, 4096) are the same n and both queues are zero.
// The clamp buys an honest number and a quiet log, not a routing fix. Said here
// because the obvious next step, on reading the paragraph above, is to go looking
// for the mis-ranking, and there isn't one.
//
// Clamping rather than dropping the row: a fleet that has gone dark is worse than
// a fleet priced conservatively, and the same reasoning keeps a failed fetch's rows
// in place a few lines above. Silent when the numbers are ordinary, which is
// always; a clamp that fires is logged once per refresh because it means the two
// routers disagree about what a field means, and that is worth finding out about.
func sanitizeRelayEntry(relay string, e relayModelEntry) relayModelEntry {
	clampInt := func(field string, v, lo, hi int) int {
		if v < lo || v > hi {
			log.Printf("relay %s: worker %s reported %s=%d, outside the %d..%d this router defines it over — clamped",
				relay, e.ID, field, v, lo, hi)
			if v < lo {
				return lo
			}
			return hi
		}
		return v
	}
	// The benchmark's own 0-100 scale, the same bound normalizeRegistration
	// enforces. Zero already means "not measured upstream" to qualityFor, so a
	// negative collapsing to it says the true thing.
	e.Quality = clampInt("quality", e.Quality, 0, benchmarkQualityScale)
	e.QualityNoThink = clampInt("quality_nothink", e.QualityNoThink, 0, benchmarkQualityScale)
	e.MaxConcurrency = clampInt("max_concurrency", e.MaxConcurrency, 0, maxDeclarableConcurrency)
	// Floored but NOT capped, unlike the ceiling above. Occupancy is a count of
	// what is happening rather than a declaration about what may, and an uncapped
	// provider row upstream really can hold more requests at once than any slot
	// channel would be built for; capping it would under-report a genuinely
	// saturated upstream, which is the one direction occupancy must not err in.
	// setRelayLoad floors it again — doing it here as well means the entry is
	// coherent before it is split across two consumers rather than after.
	if e.ActiveRequests < 0 {
		log.Printf("relay %s: worker %s reported active_requests=%d — read as idle", relay, e.ID, e.ActiveRequests)
		e.ActiveRequests = 0
	}
	// A negative rate or latency is not a slow worker, it is a broken field.
	// Zeroing reads as "not measured", which every consumer already handles:
	// liveTPS falls through, prefillSeconds takes the fleet constant.
	e.BaselineTPS = clampNonNegative(relay, e.ID, "baseline_tps", e.BaselineTPS)
	e.ObservedTPS = clampNonNegative(relay, e.ID, "observed_tps", e.ObservedTPS)
	e.PrefillTPS = clampNonNegative(relay, e.ID, "observed_prefill_tps", e.PrefillTPS)
	if e.TTFTMillis < 0 {
		log.Printf("relay %s: worker %s reported ttft_ms=%d — treated as unmeasured", relay, e.ID, e.TTFTMillis)
		e.TTFTMillis = 0
	}
	if e.ContextK < 0 {
		log.Printf("relay %s: worker %s reported context_k=%d — treated as undeclared", relay, e.ID, e.ContextK)
		e.ContextK = 0
	}
	return e
}

// clampNonNegative floors one imported rate at zero — "not measured" — and says
// so when it fires.
func clampNonNegative(relay, id, field string, v float64) float64 {
	if v < 0 {
		log.Printf("relay %s: worker %s reported %s=%v — treated as unmeasured", relay, id, field, v)
		return 0
	}
	return v
}

// fetchRelayFleet is one GET /relay/fleet, timed.
//
// The elapsed time it returns is an UPPER bound on the network round trip — it
// includes the upstream building the response, which is a registry snapshot and
// therefore small. Erring high is the safe direction for a number that exists to
// stop this router preferring a remote fleet it has under-priced.
func (r *Router) fetchRelayFleet(rel Relay) (*relayFleetResponse, float64, error) {
	base := strings.TrimSuffix(strings.TrimSpace(rel.URL), "/")
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/relay/fleet", nil)
	if err != nil {
		return nil, 0, err
	}
	if rel.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+rel.APIKey)
	}
	// Announce ourselves on the fetch as well as on the traffic, so an upstream
	// that is (mis)configured to relay back to us can refuse at discovery time
	// rather than at the first request.
	req.Header.Set(relayHopHeader, r.routerID())
	start := time.Now()
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	rtt := float64(time.Since(start).Microseconds()) / 1000
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, rtt, fmt.Errorf("GET %s/relay/fleet: HTTP %d", base, resp.StatusCode)
	}
	var out relayFleetResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, rtt, fmt.Errorf("decode relay fleet: %w", err)
	}
	return &out, rtt, nil
}

// applyRelayEntry brings one upstream WORKER into this router's registry, as its
// own row. Not one row per model: two upstream endpoints serving the same weights
// differ in quantisation, card and context window, and a blended row would hide
// exactly the differences the ranker exists to exploit (see relayModelEntry).
//
// The REGISTRATION it upserts is deliberately minimal and deliberately stable:
// id, url, model, credential, health path. Everything the upstream measures —
// quality, capacity, context, features, thinking — is applied afterwards through
// applyProfileIfGen, exactly as a probe's results are. Folding those into the
// registration instead would make every drift in the upstream's live throughput
// a content change, and upsert treats a content change as a re-registration:
// certification reset, profile generation bumped, slot channel rebuilt, fifteen
// seconds later again forever.
func (r *Router) applyRelayEntry(rel Relay, id string, entry relayModelEntry, fleetBench int) {
	reg := BackendRegistration{
		ID:     id,
		URL:    strings.TrimSuffix(strings.TrimSpace(rel.URL), "/"),
		Model:  entry.Model,
		APIKey: rel.APIKey,
		// The upstream's own /health, which is public and cheap, and whose round
		// trip is also the liveness signal for the WAN path this row rides.
		HealthPath: "/health",
		Source:     sourceRelay,
		Relay:      rel.Name,
	}
	if err := normalizeRegistration(&reg); err != nil {
		log.Printf("relay %s: skip model %q: %v", rel.Name, entry.Model, err)
		return
	}
	backend, created := r.registry.upsert(reg)
	if created {
		log.Printf("relay %s: registered %s (model %s, q=%d%%, ctx %dk)",
			rel.Name, id, entry.Model, entry.Quality, entry.ContextK)
	}
	// The model spelling to put in a forwarded body. Ordinarily this is read off
	// the endpoint's own /v1/models by queryModelInfo, which a relay row never
	// runs — and without it patchForwardedBody would leave the CLIENT's spelling
	// in place and the upstream would auto-route a request this router had
	// already chosen a model for. It is the name the upstream published, so it is
	// a name the upstream accepts, which is the whole contract of relayFleetFor.
	r.registry.setModelMeta(id, ModelMeta{ServedID: entry.ID})
	prof := relayProfile(entry)
	mismatch := fleetBench != benchmarkVersion
	if mismatch {
		// A quality measured against a different question set is not a quality on
		// this router's scale, and adopting it would put a number nobody can
		// compare into the one place the whole tier mechanism reads. It cannot be
		// re-measured here either — benchmarking a relay means spending the
		// upstream's GPUs to grade the upstream's own workers — so the row takes
		// the same conservative tier a worker holds while its benchmark is still
		// running: routable, and only for easy traffic, until the two routers are
		// on the same build.
		prof.Quality = provisionalQuality
		// The no-think score is graded against the same question set, so it is
		// exactly as incomparable — drop it rather than let a number from
		// another scale reach qualityFor.
		prof.QualityNoThink = 0
		prof.BenchVersion = 0
	}
	r.registry.applyProfileIfGen(id, 0, prof)
	r.registry.setRelayLoad(id, entry.ActiveRequests, entry.MaxConcurrency, r.relayRTT(rel.Name))
	// Certify on first sight only. finishCertification clears the proxy-failure
	// counter, so calling it every refresh would disable the circuit breaker that
	// takes a misbehaving upstream out of rotation.
	if backend != nil && !backend.Certification.Ready {
		message := fmt.Sprintf("imported from relay %q: worker %s, q=%d%%, benchmark v%d",
			rel.Name, entry.ID, entry.Quality, fleetBench)
		if mismatch {
			message = fmt.Sprintf(
				"imported from relay %q, but it benchmarks with v%d and this router with v%d — capacity and capabilities adopted, quality held at the provisional %d%% until the two match",
				rel.Name, fleetBench, benchmarkVersion, provisionalQuality)
			log.Printf("relay %s: benchmark version mismatch (remote v%d, local v%d) — %s held at provisional quality %d%%; upgrade one side to compare tiers",
				rel.Name, fleetBench, benchmarkVersion, id, provisionalQuality)
		}
		r.registry.finishCertification(id, true, map[string]Check{
			"relay": {OK: true, Message: message},
		}, entry.BaselineTPS, entry.TTFTMillis, "")
	}
}

// relayProfile turns what the upstream reported into the same WorkerProfile a
// local probe would have produced, so the import lands through exactly the path
// a measurement does.
//
// Every latency term here is the ENDPOINT's, measured on the upstream's own
// LAN, with the link between the two routers excluded. The link is added back
// once, at the point of use, by prefillSeconds — and Registry.observe strips it
// back out of this router's own samples, so the field means the same thing
// whichever of the two filled it in.
//
// It is worth saying why, because the obvious alternative was tried and is
// wrong. Folding the round trip into the imported TTFT works only while the
// prefill RATE is empty, because prefillSeconds reads the rate in preference to
// the TTFT and never looks at it again — and a rate has nowhere to put a
// constant offset. Leaving the rate empty to force the TTFT path costs far more
// than the link is worth: without a rate, a remote model is priced at a FLAT
// first-token latency no matter how long the prompt is, which on a fleet whose
// local workers prefill at thousands of tokens a second makes the far one look
// unbeatable on exactly the long-context prompts it is worst at. And nothing
// fixes it later — observe() only samples non-thinking turns, so a reasoning
// worker never measures a rate here at all (see applyProfileIfGen, which seeds
// a local worker's rate from the probe for that same reason).
//
// The thinking dialect is the one value that is NOT imported, and cannot be. It
// describes how to spell the thinking gate to the endpoint that finally serves
// the request, and this row's immediate peer is a router, not that endpoint —
// which speaks the client-facing OpenAI spelling and translates onward. So the
// dialect is reasoning_effort by construction, and setting it any other way
// would have this router write a chat-template gate that the upstream, per its
// own escape-hatch rule, would forward verbatim to an endpoint that may not
// speak it.
func relayProfile(entry relayModelEntry) *WorkerProfile {
	return &WorkerProfile{
		Model:           entry.Model,
		Quality:         entry.Quality,
		QualityNoThink:  entry.QualityNoThink,
		BenchVersion:    entry.BenchVersion,
		ContextK:        entry.ContextK,
		MaxConcurrency:  entry.MaxConcurrency,
		BaselineTPS:     entry.BaselineTPS,
		PrefillTPS:      entry.PrefillTPS,
		TTFTMillis:      entry.TTFTMillis,
		Features:        append([]string(nil), entry.Features...),
		Thinking:        entry.Thinking,
		ThinkingDialect: thinkingDialectEffort,
		MeasuredAt:      time.Now(),
	}
}

// relayRTT is the smoothed round trip to one relay, in milliseconds.
func (r *Router) relayRTT(name string) float64 {
	rel, ok := r.relays.lookup(name)
	if !ok {
		return 0
	}
	return rel.RTTMillis
}

// relayRowIDs lists the registry ids currently derived from one relay, or from
// every relay when name is "".
func (r *Router) relayRowIDs(name string) []string {
	out := []string{}
	for _, b := range r.registry.snapshot() {
		if isRelayRow(b) && (name == "" || b.Relay == name) {
			out = append(out, b.ID)
		}
	}
	return out
}

// pruneRelayRows removes derived rows whose model is no longer served upstream,
// or whose relay has been deleted or disabled.
//
// Derived rows are never persisted (see Main), so this is the only thing that
// removes them and there is no stored row to delete alongside. Their cached
// profiles go too: the id will only ever come back with an imported profile
// anyway, and leaving a stale one behind would have a re-enabled relay certify
// from numbers nobody measured this year.
func (r *Router) pruneRelayRows(live map[string]bool) {
	for _, b := range r.registry.snapshot() {
		if !isRelayRow(b) || live[b.ID] {
			continue
		}
		if r.registry.remove(b.ID) {
			log.Printf("relay %s: %s no longer served upstream — deregistered", b.Relay, b.ID)
		}
		if r.logs != nil {
			if err := r.logs.DeleteWorkerProfile(context.Background(), b.ID); err != nil {
				log.Printf("delete cached profile for %s failed: %v", b.ID, err)
			}
		}
	}
}

// ── Southbound: admin CRUD ──────────────────────────────────────────────────

// relaySpec is the write shape. Pointers for the same reason providerSpec uses
// them: disabling a relay and not mentioning it are different instructions.
type relaySpec struct {
	Name    *string `json:"name"`
	URL     *string `json:"url"`
	APIKey  *string `json:"api_key"`
	Enabled *bool   `json:"enabled"`
}

func (r *Router) handleAdminRelays(w http.ResponseWriter, req *http.Request) {
	if !r.requireAdmin(w, req) {
		return
	}
	switch req.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, map[string]any{
			"relays":           r.relays.list(),
			"bench_version":    benchmarkVersion,
			"router_id":        r.routerID(),
			"relay_hop_header": relayHopHeader,
			"relay_max_hops":   relayMaxHops,
			"derived_backends": len(r.relayRowIDs("")),
		})
	case http.MethodPost:
		r.writeRelay(w, req, "")
	default:
		methodNotAllowed(w)
	}
}

func (r *Router) handleAdminRelayByName(w http.ResponseWriter, req *http.Request) {
	if !r.requireAdmin(w, req) {
		return
	}
	name := strings.Trim(strings.TrimPrefix(req.URL.Path, "/admin/relays/"), "/")
	if name == "" {
		r.handleAdminRelays(w, req)
		return
	}
	switch req.Method {
	case http.MethodGet:
		rel, ok := r.relays.lookup(name)
		if !ok {
			writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("no relay %q", name)})
			return
		}
		rel.APIKey = ""
		writeJSON(w, http.StatusOK, rel)
	case http.MethodPatch, http.MethodPut:
		r.writeRelay(w, req, name)
	case http.MethodDelete:
		r.deleteRelay(w, req, name)
	default:
		methodNotAllowed(w)
	}
}

// writeRelay creates a relay (name == "") or edits one, applying only the fields
// the caller actually sent.
func (r *Router) writeRelay(w http.ResponseWriter, req *http.Request, name string) {
	var spec relaySpec
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 1<<20)).Decode(&spec); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: %s", err)
		return
	}
	rel := Relay{Enabled: true}
	if name != "" {
		existing, ok := r.relays.lookup(name)
		if !ok {
			writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("no relay %q", name)})
			return
		}
		rel = existing
	}
	if spec.Name != nil && name == "" {
		rel.Name = *spec.Name
		// SaveRelay upserts, so without this a POST naming an existing relay would
		// silently replace its URL and credential instead of saying it is already
		// there. PATCH is how you change one.
		if _, taken := r.relays.lookup(rel.Name); taken {
			writeJSON(w, http.StatusConflict, validationError{
				Message: fmt.Sprintf("relay %q already exists; PATCH it to change its url or key", relayKey(rel.Name)),
				Param:   "name",
			})
			return
		}
	} else if spec.Name != nil && !strings.EqualFold(*spec.Name, rel.Name) {
		// Renaming would orphan every derived backend id, which carries the name.
		// Delete and recreate says the same thing and says it visibly.
		writeJSON(w, http.StatusConflict, validationError{
			Message: "a relay's name cannot be changed; delete it and add another", Param: "name",
		})
		return
	}
	assign(&rel.URL, spec.URL)
	assign(&rel.APIKey, spec.APIKey)
	assign(&rel.Enabled, spec.Enabled)
	if err := validateRelay(rel, r); err != nil {
		writeJSON(w, http.StatusBadRequest, *err)
		return
	}
	rel.Name = relayKey(rel.Name)
	rel.UpdatedAt = time.Now()
	if err := r.logs.SaveRelay(req.Context(), rel); err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	r.relays.put(rel)
	log.Printf("relay %q %s (url=%s enabled=%v)", rel.Name, mapBool(name == "", "added", "updated"), rel.URL, rel.Enabled)
	// Reconcile immediately rather than at the next tick, so the call that
	// created a relay is the call an operator can check the result of.
	go r.refreshRelays()
	out := rel
	out.APIKey = ""
	writeJSON(w, mapStatus(name == ""), out)
}

func (r *Router) deleteRelay(w http.ResponseWriter, req *http.Request, name string) {
	if err := r.logs.DeleteRelay(req.Context(), name); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("no relay %q", name)})
			return
		}
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	r.relays.remove(name)
	// Take its rows out now: waiting for the next tick would leave a deleted
	// relay's models routable, which is not what deleting it means. Only ITS
	// rows, and without a full reconcile — that would re-fetch every other relay
	// over the network while this admin request waits on it.
	removed := 0
	for _, id := range r.relayRowIDs(relayKey(name)) {
		if r.registry.remove(id) {
			removed++
		}
		if r.logs != nil {
			if err := r.logs.DeleteWorkerProfile(req.Context(), id); err != nil {
				log.Printf("delete cached profile for %s failed: %v", id, err)
			}
		}
	}
	log.Printf("relay %q removed (%d backend(s) deregistered)", relayKey(name), removed)
	writeJSON(w, http.StatusOK, map[string]any{"deleted": relayKey(name), "backends_removed": removed})
}

// validateRelay refuses a relay that could not work, or that would collide with
// a name the router already answers to.
func validateRelay(rel Relay, r *Router) *validationError {
	name := relayKey(rel.Name)
	if name == "" {
		return &validationError{Message: "name is required", Param: "name"}
	}
	if strings.ContainsAny(name, ":/ ") {
		return &validationError{
			Message: "a relay name may not contain ':', '/' or a space — it prefixes the backend ids the relay creates",
			Param:   "name",
		}
	}
	if autoModelNames[name] || isExpertModel(name) {
		return &validationError{Message: fmt.Sprintf("%q is a name the router owns", name), Param: "name"}
	}
	if strings.TrimSpace(rel.URL) == "" {
		return &validationError{Message: "url is required", Param: "url"}
	}
	if !strings.HasPrefix(rel.URL, "http://") && !strings.HasPrefix(rel.URL, "https://") {
		return &validationError{Message: "url must be an http:// or https:// base url", Param: "url"}
	}
	if r != nil {
		if _, taken := r.groups.lookup(name); taken {
			return &validationError{Message: fmt.Sprintf("%q is already a group name", name), Param: "name"}
		}
	}
	return nil
}

// mapStatus is 201 for a create and 200 for an edit.
func mapStatus(created bool) int {
	if created {
		return http.StatusCreated
	}
	return http.StatusOK
}

// relayHealthLine is the /health entry describing the relayed half of the fleet,
// or empty when none is configured.
//
// A benchmark mismatch is reported here rather than only in the log, because it
// is the one relay fault that leaves everything apparently working: the models
// are present, healthy and routable, and only their tier is being decided by a
// number this router had to measure for itself.
func (r *Router) relayHealthLine() map[string]any {
	relays := r.relays.list()
	if len(relays) == 0 {
		return nil
	}
	out := map[string]any{"configured": len(relays)}
	models, failing, mismatched := 0, 0, 0
	for _, rel := range relays {
		if !rel.Enabled {
			continue
		}
		models += rel.Models
		if rel.LastError != "" {
			failing++
		}
		if rel.BenchVersion != 0 && rel.BenchVersion != benchmarkVersion {
			mismatched++
		}
	}
	out["models"] = models
	if failing > 0 {
		out["unreachable"] = failing
	}
	if mismatched > 0 {
		out["benchmark_mismatch"] = mismatched
		out["benchmark_version"] = benchmarkVersion
	}
	return out
}
