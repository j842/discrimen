package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// TestRelayCarriesNoThinkQuality covers the score that used to be dropped on the
// floor. Sending only Quality left every imported row at QualityNoThink=0, so
// qualityFor fell back to the THINKING score for no-think requests and over-rated
// the whole relayed fleet — `ask -ls` showed it as "q=? qt=84".
func TestRelayCarriesNoThinkQuality(t *testing.T) {
	reg := newTestRegistry()
	relayReady(t, reg, "a", "m", 84, 128, 4, "chat")
	relayReady(t, reg, "b", "m", 70, 128, 2, "chat")
	// Through the registry map: eligible() hands out snapshots, so mutating the
	// pointer upsert returned would not be seen.
	reg.backends["a"].QualityNoThink = 40
	reg.backends["b"].QualityNoThink = 55

	got := relayFleetFor(&identity{Role: roleClient, Relay: true}, reg.eligible())
	by := map[string]relayModelEntry{}
	for _, e := range got {
		by[e.ID] = e
	}
	// Each worker carries its OWN no-think score — no merging.
	if by["a"].QualityNoThink != 40 || by["b"].QualityNoThink != 55 {
		t.Errorf("quality_nothink = %d/%d, want 40/55 per worker",
			by["a"].QualityNoThink, by["b"].QualityNoThink)
	}

	// And it has to survive the trip into a WorkerProfile, or selection never
	// sees it however well the wire carries it.
	if prof := relayProfile(by["a"]); prof.QualityNoThink != 40 {
		t.Errorf("profile quality_nothink = %d, want 40", prof.QualityNoThink)
	}

	// A worker nobody scored no-think reports 0, which qualityFor reads as
	// "not measured" and falls back to Quality for.
	reg.backends["b"].QualityNoThink = 0
	got = relayFleetFor(&identity{Role: roleClient, Relay: true}, reg.eligible())
	for _, e := range got {
		if e.ID == "b" && e.QualityNoThink != 0 {
			t.Errorf("unmeasured worker published quality_nothink = %d, want 0", e.QualityNoThink)
		}
	}
}

// An upstream that predates the per-worker fleet sends entries with a model and
// no id. Upgrading the downstream first must not take the relayed fleet dark.
func TestRelayToleratesUpstreamWithoutWorkerIDs(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		writeJSON(w, http.StatusOK, relayFleetResponse{
			RouterID: "r-old", BenchVersion: benchmarkVersion,
			// No ID — the shape an older upstream emits.
			Models: []relayModelEntry{{Model: "m", Quality: 70, BenchVersion: benchmarkVersion, MaxConcurrency: 2}},
		})
	}))
	defer upstream.Close()

	r := adminRouter(t)
	r.relays.put(Relay{Name: "up", URL: upstream.URL, Enabled: true})
	r.refreshRelays()
	if got := len(r.relayRowIDs("up")); got != 1 {
		t.Fatalf("registered %d rows, want 1 — an id-less upstream must still relay", got)
	}
}

// ── Defect: the relay import went round every range check ──────────────────
//
// normalizeRegistration refuses a registration whose quality is outside 0..100
// or whose max_concurrency is above maxDeclarableConcurrency. The relay path
// never showed it those fields: applyRelayEntry registers id/url/model/credential
// only and applies everything measured afterwards, through applyProfileIfGen and
// setRelayLoad, neither of which range-checks. So a peer could write numbers into
// this registry that no local code path can produce.
//
// max_concurrency is the second half of a bug whose first half is already fixed:
// syncSlotsLocked clamps the number it builds a slot CHANNEL from (1e9 tokens at
// ~12.6ns each, under the registry write lock, is thirteen seconds of not
// routing), but b.MaxConcurrency keeps the raw figure. effectiveSlots then
// republishes it to the next router down, so the number propagates along the
// whole chain instead of stopping at the router that received it — which is what
// the second half of this test pins.
func TestRelayEntryIsClampedToTheRangesItsFieldsAreDefinedOver(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, relayFleetResponse{
			RouterID: "r-skewed", BenchVersion: benchmarkVersion,
			Models: []relayModelEntry{{
				ID: "w", Model: "m", BenchVersion: benchmarkVersion,
				Features: []string{"chat"}, BaselineTPS: 100,
				// A units error, a version skew, or a field that meant something
				// else on the other side of the wire. None of them is a claim
				// about a worker that this router can act on.
				Quality:        100_000,
				QualityNoThink: -5,
				MaxConcurrency: 1_000_000_000,
				ActiveRequests: -3,
			}},
		})
	}))
	defer upstream.Close()

	r := adminRouter(t)
	r.relays.put(Relay{Name: "up", URL: upstream.URL, Enabled: true})
	r.refreshRelays()

	b := r.registry.get(relayBackendID("up", "w"))
	if b == nil {
		t.Fatalf("row was not registered; rows: %v", r.relayRowIDs(""))
	}
	if b.Quality > benchmarkQualityScale || b.Quality < 0 {
		t.Errorf("quality = %d, want it inside 0..%d — the scale normalizeRegistration enforces on every other path",
			b.Quality, benchmarkQualityScale)
	}
	if b.QualityNoThink < 0 {
		t.Errorf("quality_nothink = %d, want 0 (\"not measured\") rather than a negative score", b.QualityNoThink)
	}
	if b.MaxConcurrency > maxDeclarableConcurrency || b.MaxConcurrency < 0 {
		t.Errorf("max_concurrency = %d, want it inside 0..%d", b.MaxConcurrency, maxDeclarableConcurrency)
	}
	if b.RemoteActive < 0 {
		t.Errorf("remote occupancy = %d, want it floored at zero", b.RemoteActive)
	}
	// And it must not be handed on. Whatever this router accepted is what it
	// publishes to anyone relaying through IT, so an unclamped figure walks the
	// whole chain, re-clamping each hop's slot channel and propagating the raw
	// number again.
	if got := effectiveSlots(b); got > maxDeclarableConcurrency {
		t.Errorf("republished max_concurrency = %d — the next router down inherits the same bad number", got)
	}
}

// A rate or a latency below zero is not a slow worker, it is a broken field, and
// zero is the value every consumer already reads as "not measured": liveTPS falls
// through to the next source, prefillSeconds takes the fleet constant.
func TestSanitizeRelayEntryFloorsNegativeMeasurements(t *testing.T) {
	got := sanitizeRelayEntry("up", relayModelEntry{
		ID: "w", Model: "m",
		BaselineTPS: -100, ObservedTPS: -1, PrefillTPS: -2800,
		TTFTMillis: -250, ContextK: -128,
	})
	if got.BaselineTPS != 0 || got.ObservedTPS != 0 || got.PrefillTPS != 0 {
		t.Errorf("negative rates survived: %+v", got)
	}
	if got.TTFTMillis != 0 || got.ContextK != 0 {
		t.Errorf("negative latency/context survived: ttft=%d ctx=%d", got.TTFTMillis, got.ContextK)
	}
	// An ordinary entry passes through byte for byte: this validates a domain, it
	// does not second-guess a measurement.
	ordinary := relayModelEntry{
		ID: "w", Model: "m", Quality: 82, QualityNoThink: 71, BenchVersion: benchmarkVersion,
		ContextK: 116, Features: []string{"chat"}, Thinking: true,
		BaselineTPS: 108, ObservedTPS: 96, PrefillTPS: 2816, TTFTMillis: 250,
		MaxConcurrency: 4, ActiveRequests: 1,
	}
	if out := sanitizeRelayEntry("up", ordinary); !reflect.DeepEqual(out, ordinary) {
		t.Errorf("an in-range entry was altered:\n got %+v\nwant %+v", out, ordinary)
	}
}

// relayReady builds a registry row in the state a certified backend is in, so
// the fleet endpoint and the ranker see something eligible.
func relayReady(t *testing.T, reg *Registry, id, model string, quality, ctxK, conc int, features ...string) *Backend {
	t.Helper()
	// Through normalizeRegistration, not around it: it is what settles the TTL,
	// and a row with none is expired the instant it is created — so eligible()
	// would return nothing and every assertion below would pass vacuously.
	r := BackendRegistration{
		ID: id, URL: "http://" + id, Model: model,
		Quality: quality, ContextK: ctxK, MaxConcurrency: conc,
		Features: features, BaselineTPS: 100,
	}
	if err := normalizeRegistration(&r); err != nil {
		t.Fatalf("normalizeRegistration(%s): %v", id, err)
	}
	b, _ := reg.upsert(r)
	reg.finishCertification(id, true, map[string]Check{}, 100, 200, "")
	return b
}

// ── Northbound ──────────────────────────────────────────────────────────────

func TestRelayFleetNeedsRelayKey(t *testing.T) {
	r := adminRouter(t)
	relayReady(t, r.registry, "w1", "qwen3.8", 80, 128, 4, "chat")

	// A plain client key reaches /v1/chat/completions and must NOT reach here:
	// the fleet's measured capacity and live occupancy is what /backends was put
	// behind the admin gate to protect.
	issueKey(t, r, "sk-plain-client", apiKey{Role: roleClient, Name: "plain"})
	req := httptest.NewRequest(http.MethodGet, "/relay/fleet", nil)
	req.Header.Set("Authorization", "Bearer sk-plain-client")
	if rec := serveAdmin(r, req); rec.Code != http.StatusForbidden {
		t.Fatalf("plain client key: want 403, got %d: %s", rec.Code, rec.Body.String())
	}

	// A relay key does.
	issueKey(t, r, "sk-relay", apiKey{Role: roleClient, Name: "downstream", Relay: true})
	req = httptest.NewRequest(http.MethodGet, "/relay/fleet", nil)
	req.Header.Set("Authorization", "Bearer sk-relay")
	rec := serveAdmin(r, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("relay key: want 200, got %d: %s", rec.Code, rec.Body.String())
	}
	var fleet relayFleetResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &fleet); err != nil {
		t.Fatalf("decode: %v — %s", err, rec.Body.String())
	}
	if fleet.BenchVersion != benchmarkVersion {
		t.Fatalf("bench_version = %d, want %d", fleet.BenchVersion, benchmarkVersion)
	}
	if fleet.RouterID == "" {
		t.Fatal("router_id is empty — the loop guard has nothing to recognise")
	}
	if len(fleet.Models) != 1 || fleet.Models[0].Model != "qwen3.8" {
		t.Fatalf("models = %+v, want one entry for qwen3.8", fleet.Models)
	}
}

// The allow-list is what limits a relay to part of a fleet, and it has to hold
// on the fleet endpoint as well as on the traffic — otherwise a downstream
// discovers models it will then be refused.
func TestRelayFleetRespectsAllowList(t *testing.T) {
	r := adminRouter(t)
	relayReady(t, r.registry, "w1", "shared-model", 80, 128, 4, "chat")
	relayReady(t, r.registry, "w2", "private-model", 90, 128, 2, "chat")

	ident := &identity{Role: roleClient, Relay: true, Models: []string{"shared-model"}}
	got := relayFleetFor(ident, r.registry.eligible())
	if len(got) != 1 || got[0].Model != "shared-model" {
		t.Fatalf("relayFleetFor = %+v, want only shared-model", got)
	}
}

// Discovery and traffic are gated by two different tests, and they have to
// agree. Discovery runs through allowsBackend (what a worker answers to); the
// traffic path runs through allowsModel, which matches the allow-list
// LITERALLY. Publishing the worker id would once have been discovered and then
// refused — which is why the fleet used to publish the allow-list's own
// spelling instead. Now the id IS the address, so mayNameWorker closes the gap:
// a key that may be SERVED BY a worker may also NAME it.
func TestRelayPublishedIDSurvivesTheTrafficGate(t *testing.T) {
	r := adminRouter(t)
	relayReady(t, r.registry, "w-endpoint-id", "Qwen3.8-27B-Uncensored-NInfer", 80, 128, 4, "chat")

	// Whichever spelling the operator allow-listed, the published id must be
	// usable by that same key.
	for _, spelling := range []string{
		"Qwen3.8-27B-Uncensored-NInfer", // the model id
		"w-endpoint-id",                 // the endpoint id
		"qwen3.8uncensoredninfer",       // the published alias
	} {
		ident := &identity{Role: roleClient, Relay: true, Models: []string{spelling}}
		got := relayFleetFor(ident, r.registry.eligible())
		if len(got) != 1 {
			t.Fatalf("allow-list %q: got %d entries, want 1", spelling, len(got))
		}
		if got[0].ID != "w-endpoint-id" {
			t.Errorf("allow-list %q published id %q, want the worker id", spelling, got[0].ID)
		}
		// The gap that used to bite: the downstream will send this id as the
		// model, and the same key has to accept it.
		if !ident.allowsModel(got[0].ID) && !r.mayNameWorker(ident, got[0].ID) {
			t.Errorf("allow-list %q: published id %q is refused by the traffic gate", spelling, got[0].ID)
		}
	}

	// A key restricted to something else still cannot name it — mayNameWorker
	// widens the spelling, not the permission.
	other := &identity{Role: roleClient, Relay: true, Models: []string{"some-other-model"}}
	if r.mayNameWorker(other, "w-endpoint-id") {
		t.Error("a key with no claim on this worker was allowed to name it")
	}
}

// Two endpoints serving the SAME model stay two entries, each carrying its own
// measured numbers. This is the point of the per-worker fleet: the downstream
// ranks them against its local workers individually, and a blended row would
// hide exactly the differences the ranker exists to exploit.
func TestRelayFleetPublishesEachWorkerSeparately(t *testing.T) {
	reg := newTestRegistry()
	relayReady(t, reg, "a", "m", 80, 128, 4, "chat", "tools", "vision")
	relayReady(t, reg, "b", "m", 60, 32, 2, "chat", "tools")
	reg.backends["a"].Thinking = true
	reg.incActive("a", 1)
	reg.incActive("b", 2)

	got := relayFleetFor(&identity{Role: roleClient, Relay: true}, reg.eligible())
	if len(got) != 2 {
		t.Fatalf("want one entry per worker, got %d: %+v", len(got), got)
	}
	by := map[string]relayModelEntry{}
	for _, e := range got {
		by[e.ID] = e
	}
	a, b := by["a"], by["b"]
	if a.ID == "" || b.ID == "" {
		t.Fatalf("entries are not keyed by worker id: %+v", got)
	}
	// Both name the same model — the model is display, the id is the address.
	if a.Model != "m" || b.Model != "m" {
		t.Errorf("models = %q/%q, want both \"m\"", a.Model, b.Model)
	}
	// Nothing is merged: each worker reports itself.
	if a.Quality != 80 || b.Quality != 60 {
		t.Errorf("quality = %d/%d, want 80/60 unmerged", a.Quality, b.Quality)
	}
	if a.ContextK != 128 || b.ContextK != 32 {
		t.Errorf("context_k = %d/%d, want 128/32 unmerged", a.ContextK, b.ContextK)
	}
	if strings.Join(a.Features, ",") != "chat,tools,vision" {
		t.Errorf("a features = %v, want its own, not an intersection", a.Features)
	}
	if !a.Thinking || b.Thinking {
		t.Errorf("thinking = %v/%v, want a's true and b's false — no union", a.Thinking, b.Thinking)
	}
	if a.MaxConcurrency != 4 || b.MaxConcurrency != 2 {
		t.Errorf("max_concurrency = %d/%d, want 4/2 unsummed", a.MaxConcurrency, b.MaxConcurrency)
	}
	if a.ActiveRequests != 1 || b.ActiveRequests != 2 {
		t.Errorf("active_requests = %d/%d, want 1/2 unsummed", a.ActiveRequests, b.ActiveRequests)
	}
}

// The worker id is the ADDRESS: the downstream stores it as ServedID, it is
// stamped into the outbound "model", and backendServesModel resolves it back to
// that exact worker upstream. Without this the two halves disagree and a relayed
// request lands on whichever worker the upstream felt like.
func TestRelayEntryAddressesTheExactWorker(t *testing.T) {
	reg := newTestRegistry()
	relayReady(t, reg, "w-endpoint-id", "Qwen3.8-27B", 80, 128, 4, "chat")

	got := relayFleetFor(&identity{Role: roleClient, Relay: true}, reg.eligible())
	if len(got) != 1 {
		t.Fatalf("want 1 entry, got %d", len(got))
	}
	if got[0].ID != "w-endpoint-id" {
		t.Fatalf("id = %q, want the worker id", got[0].ID)
	}
	// The round trip that matters: the published id must resolve upstream.
	if !backendServesModel(reg.backends["w-endpoint-id"], got[0].ID) {
		t.Fatal("upstream would not resolve its own published id back to the worker")
	}
}

// An embeddings worker is never relayed: the classifier that needs one needs it
// locally, and a remote one would put the WAN inside every routing decision.
func TestRelayFleetSkipsEmbeddings(t *testing.T) {
	reg := newTestRegistry()
	relayReady(t, reg, "emb", "bge-small", 50, 8, 4, "embeddings")
	if got := relayFleetFor(&identity{Role: roleClient, Relay: true}, reg.eligible()); len(got) != 0 {
		t.Fatalf("relayFleetFor = %+v, want no entries", got)
	}
}

func TestRedactForRelay(t *testing.T) {
	entry := RequestLog{
		Input: "the prompt", Output: "the answer", Error: "dispatch failed",
		BackendID: "w1", BackendModel: "m", StatusCode: 200, DurationMillis: 12, KeyID: "3",
	}
	kept := redactForRelay(entry, &identity{Role: roleClient})
	if kept.Input != "the prompt" || kept.Output != "the answer" {
		t.Fatal("an ordinary client's bodies must be logged unchanged")
	}
	got := redactForRelay(entry, &identity{Role: roleClient, Relay: true})
	if got.Input != "" || got.Output != "" {
		t.Fatalf("relay bodies survived redaction: in=%q out=%q", got.Input, got.Output)
	}
	// Everything that is capacity accounting rather than content stays, or an
	// operator cannot answer "what is my fleet doing".
	if got.BackendID != "w1" || got.StatusCode != 200 || got.DurationMillis != 12 || got.KeyID != "3" {
		t.Fatalf("redaction removed accounting fields: %+v", got)
	}
	if got.Error != "dispatch failed" {
		t.Error("the router's own error string must survive — it is how a failing relay is diagnosed")
	}
}

func TestLearnFromRelay(t *testing.T) {
	if !learnFromRelay(nil) || !learnFromRelay(&identity{Role: roleClient}) {
		t.Error("ordinary callers must still teach the adapter")
	}
	if learnFromRelay(&identity{Role: roleClient, Relay: true}) {
		t.Error("a relayed outcome must not teach: the downstream classified it and is already learning from it")
	}
}

func TestValidateRelayFlag(t *testing.T) {
	if err := validateRelayFlag(true, roleClient); err != nil {
		t.Errorf("client + relay should be allowed: %v", err)
	}
	for _, role := range []string{roleWorker, roleAdmin} {
		if err := validateRelayFlag(true, role); err == nil {
			t.Errorf("%s + relay should be refused", role)
		}
	}
	if err := validateRelayFlag(false, roleWorker); err != nil {
		t.Errorf("an unflagged worker key is fine: %v", err)
	}
}

// The relay flag has to survive the round trip through SQLite, or a restart
// quietly turns a relay back into a logged client.
func TestRelayKeyPersists(t *testing.T) {
	r := adminRouter(t)
	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/keys",
		`{"name":"downstream","role":"client","relay":true,"models":["m1","m2"]}`))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create: %d %s", rec.Code, rec.Body.String())
	}
	var created struct {
		Key    apiKey `json:"key"`
		Secret string `json:"secret"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !created.Key.Relay {
		t.Fatal("created key is not marked as a relay")
	}
	got, ok := r.logs.LookupAPIKey(t.Context(), created.Secret)
	if !ok || !got.Relay {
		t.Fatalf("looked-up key lost the relay flag: ok=%v key=%+v", ok, got)
	}
	// And the identity the request path sees carries it.
	req := httptest.NewRequest(http.MethodGet, "/relay/fleet", nil)
	req.Header.Set("Authorization", "Bearer "+created.Secret)
	if ident := r.identify(req); ident == nil || !ident.Relay {
		t.Fatalf("identify lost the relay flag: %+v", ident)
	}
	// Clearing it is an edit, not a reissue.
	rec = serveAdmin(r, adminReq(http.MethodPatch, "/admin/keys/"+strconv.FormatInt(created.Key.ID, 10), `{"relay":false}`))
	if rec.Code != http.StatusOK {
		t.Fatalf("patch: %d %s", rec.Code, rec.Body.String())
	}
	if got, _ := r.logs.LookupAPIKey(t.Context(), created.Secret); got.Relay {
		t.Fatal("relay flag survived being switched off")
	}
}

func TestRelayKeyRefusedOnWorkerRole(t *testing.T) {
	r := adminRouter(t)
	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/keys", `{"role":"worker","relay":true}`))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400 for a worker relay key, got %d: %s", rec.Code, rec.Body.String())
	}
}

// ── The loop guard ──────────────────────────────────────────────────────────

func TestRefuseRelayLoop(t *testing.T) {
	r := adminRouter(t)
	self := r.routerID()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	if r.refuseRelayLoop(httptest.NewRecorder(), req) {
		t.Error("a request with no chain is not a loop")
	}

	req.Header.Set(relayHopHeader, "r-somewhere-else")
	if r.refuseRelayLoop(httptest.NewRecorder(), req) {
		t.Error("a chain of other routers is not a loop")
	}

	req.Header.Set(relayHopHeader, "r-a, "+self+" ,r-b")
	rec := httptest.NewRecorder()
	if !r.refuseRelayLoop(rec, req) {
		t.Fatal("a chain containing this router must be refused")
	}
	if rec.Code != http.StatusLoopDetected {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusLoopDetected)
	}

	// Distinct routers, but too many of them.
	hops := make([]string, relayMaxHops)
	for i := range hops {
		hops[i] = "r-" + strconv.Itoa(i)
	}
	req.Header.Set(relayHopHeader, strings.Join(hops, ","))
	rec = httptest.NewRecorder()
	if !r.refuseRelayLoop(rec, req) || rec.Code != http.StatusLoopDetected {
		t.Fatalf("a chain at the hop limit must be refused, got %d", rec.Code)
	}
}

// The chain has to GROW, or a two-router cycle is invisible to the second hop.
func TestStampRelayChain(t *testing.T) {
	r := adminRouter(t)
	relay := &Backend{BackendRegistration: BackendRegistration{ID: "up:m", Source: sourceRelay, Relay: "up"}}
	local := &Backend{BackendRegistration: BackendRegistration{ID: "w1", Source: sourceBeacon}}

	in := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	in.Header.Set(relayHopHeader, "r-first")

	out := httptest.NewRequest(http.MethodPost, "http://up/v1/chat/completions", nil)
	r.stampRelayChain(out, in, relay)
	if got, want := out.Header.Get(relayHopHeader), "r-first,"+r.routerID(); got != want {
		t.Fatalf("chain = %q, want %q", got, want)
	}

	// A local worker gets no header at all — a strict endpoint should not be sent
	// a field it has no use for.
	plain := httptest.NewRequest(http.MethodPost, "http://w1/v1/chat/completions", nil)
	r.stampRelayChain(plain, in, local)
	if got := plain.Header.Get(relayHopHeader); got != "" {
		t.Fatalf("a local worker was sent a relay chain: %q", got)
	}

	// A router-originated call (judge, expert member) starts the chain itself.
	origin := httptest.NewRequest(http.MethodPost, "http://up/v1/chat/completions", nil)
	r.stampRelayChain(origin, nil, relay)
	if got := origin.Header.Get(relayHopHeader); got != r.routerID() {
		t.Fatalf("origin chain = %q, want %q", got, r.routerID())
	}
}

func TestRouterIDIsStableAndPersisted(t *testing.T) {
	r := adminRouter(t)
	first := r.routerID()
	if first == "" || r.routerID() != first {
		t.Fatalf("router id is not stable: %q then %q", first, r.routerID())
	}
	// A second Router over the SAME store must recognise the same identity, or a
	// restart forgets every cycle it had learned to refuse.
	again := &Router{cfg: r.cfg, registry: newTestRegistry(), logs: r.logs}
	if got := again.routerID(); got != first {
		t.Fatalf("router id after restart = %q, want the persisted %q", got, first)
	}
}

// ── Southbound ──────────────────────────────────────────────────────────────

func TestRelayProfileImport(t *testing.T) {
	entry := relayModelEntry{
		ID: "m", Model: "m", Quality: 77, BenchVersion: benchmarkVersion, ContextK: 128,
		Features: []string{"chat", "tools"}, Thinking: true,
		BaselineTPS: 90, PrefillTPS: 4000, TTFTMillis: 300,
		MaxConcurrency: 6, ActiveRequests: 2,
	}
	prof := relayProfile(entry)
	// Both latency terms describe the far ENDPOINT, link excluded — prefillSeconds
	// is the one place the link is added. Importing the prefill rate is what stops
	// a remote model being priced at a flat first-token latency however long the
	// prompt is, which is the shape that makes it look unbeatable on exactly the
	// long-context prompts it is worst at.
	if prof.TTFTMillis != 300 {
		t.Errorf("ttft = %d, want the upstream's own 300 with no link folded in", prof.TTFTMillis)
	}
	if prof.PrefillTPS != 4000 {
		t.Errorf("prefill rate = %v, want the upstream's measured 4000", prof.PrefillTPS)
	}
	if prof.ThinkingDialect != thinkingDialectEffort {
		t.Errorf("dialect = %q, want %q: the peer is a router, which speaks the client-facing spelling",
			prof.ThinkingDialect, thinkingDialectEffort)
	}
	if prof.Quality != 77 || prof.MaxConcurrency != 6 || prof.ContextK != 128 {
		t.Errorf("measured values did not cross: %+v", prof)
	}
}

// The end-to-end southbound path against a stub upstream: discover, register,
// import, certify — and prune when the upstream stops serving the model.
func TestRelayRefreshRegistersAndPrunes(t *testing.T) {
	models := []relayModelEntry{{
		ID: "qwen3.8-uncensored", Model: "qwen3.8-uncensored", Quality: 82, BenchVersion: benchmarkVersion,
		ContextK: 116, Features: []string{"chat", "tools", "vision"}, Thinking: true,
		BaselineTPS: 108, PrefillTPS: 2816, TTFTMillis: 250,
		MaxConcurrency: 4, ActiveRequests: 1,
	}}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/relay/fleet" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if req.Header.Get("Authorization") != "Bearer sk-upstream" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		writeJSON(w, http.StatusOK, relayFleetResponse{
			RouterID: "r-upstream", BenchVersion: benchmarkVersion, Models: models,
		})
	}))
	defer upstream.Close()

	r := adminRouter(t)
	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/relays",
		`{"name":"work","url":"`+upstream.URL+`","api_key":"sk-upstream"}`))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create relay: %d %s", rec.Code, rec.Body.String())
	}
	// The create kicks a refresh in the background; run one synchronously so the
	// assertions do not race it.
	r.refreshRelays()

	id := relayBackendID("work", "qwen3.8-uncensored")
	b := r.registry.get(id)
	if b == nil {
		t.Fatalf("relay row %q was not registered; rows: %v", id, r.relayRowIDs(""))
	}
	if !isRelayRow(b) || b.Relay != "work" {
		t.Fatalf("row is not marked as relayed: source=%q relay=%q", b.Source, b.Relay)
	}
	if !b.Certification.Ready {
		t.Fatalf("relay row was not certified: %+v", b.Certification)
	}
	if b.Quality != 82 || b.ContextK != 116 || b.MaxConcurrency != 4 {
		t.Fatalf("imported values did not land: q=%d ctx=%d conc=%d", b.Quality, b.ContextK, b.MaxConcurrency)
	}
	if b.ServedID != "qwen3.8-uncensored" {
		t.Fatalf("served id = %q — without it the forwarded body keeps the client's spelling and the upstream re-routes", b.ServedID)
	}
	if b.RemoteActive != 1 {
		t.Fatalf("remote occupancy = %d, want the upstream's 1", b.RemoteActive)
	}
	// Over a real HTTP round trip, so a wrong or duplicated json tag on the wire
	// shape is caught here rather than by a fleet that silently prices every
	// remote model at a flat first-token latency. This is the field `ask -l`
	// renders as pp=, and the one the long-prompt estimate is built from.
	if b.ObservedPrefillTPS != 2816 {
		t.Fatalf("prefill rate = %v, want the upstream's measured 2816 — without it a long prompt is priced flat",
			b.ObservedPrefillTPS)
	}
	if b.Certification.TTFTMillis != 250 {
		t.Fatalf("certified ttft = %d, want the endpoint's own 250 with the link kept separate",
			b.Certification.TTFTMillis)
	}
	if b.URL != upstream.URL {
		t.Fatalf("url = %q, want the upstream router %q", b.URL, upstream.URL)
	}
	// Nothing derived is persisted: the relay row is the source of truth.
	saved, err := r.logs.LoadBackendRegistrations(t.Context())
	if err != nil {
		t.Fatalf("LoadBackendRegistrations: %v", err)
	}
	for _, reg := range saved {
		if reg.ID == id {
			t.Fatal("a derived relay row was persisted; it must be rebuilt from the relay config")
		}
	}

	// A second pass with the same content must not churn the row — a
	// re-registration would reset its certification every refresh.
	before := r.registry.get(id).profileGen
	r.refreshRelays()
	if after := r.registry.get(id).profileGen; after != before {
		t.Fatalf("an unchanged refresh re-registered the row (gen %d → %d)", before, after)
	}

	// Stop serving it upstream, and it goes.
	models = nil
	r.refreshRelays()
	if r.registry.get(id) != nil {
		t.Fatal("a model the upstream stopped serving is still registered")
	}
}

// Deleting a relay has to take its models out of routing in the same call.
// Waiting for the next tick would leave a deleted relay serving traffic, which
// is not what deleting it means.
func TestRelayDeleteDeregistersItsRows(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, relayFleetResponse{
			RouterID: "r-up", BenchVersion: benchmarkVersion,
			Models: []relayModelEntry{
				{ID: "m1", Model: "m1", Quality: 70, BenchVersion: benchmarkVersion, MaxConcurrency: 2},
				{ID: "m2", Model: "m2", Quality: 75, BenchVersion: benchmarkVersion, MaxConcurrency: 2},
			},
		})
	}))
	defer upstream.Close()

	r := adminRouter(t)
	r.relays.put(Relay{Name: "up", URL: upstream.URL, Enabled: true})
	if err := r.logs.SaveRelay(t.Context(), Relay{Name: "up", URL: upstream.URL, Enabled: true}); err != nil {
		t.Fatalf("SaveRelay: %v", err)
	}
	// A second relay whose rows must survive the delete.
	r.relays.put(Relay{Name: "other", URL: upstream.URL, Enabled: true})
	r.refreshRelays()
	if got := len(r.relayRowIDs("up")); got != 2 {
		t.Fatalf("registered %d rows for up, want 2", got)
	}

	rec := serveAdmin(r, adminReq(http.MethodDelete, "/admin/relays/up", ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("delete: %d %s", rec.Code, rec.Body.String())
	}
	if got := r.relayRowIDs("up"); len(got) != 0 {
		t.Fatalf("a deleted relay is still serving %v", got)
	}
	if got := len(r.relayRowIDs("other")); got != 2 {
		t.Fatalf("the delete took another relay's rows with it: %d left, want 2", got)
	}
}

// A relay that answers with its own id as ours is a cycle at discovery time.
func TestRelayRefusesItself(t *testing.T) {
	r := adminRouter(t)
	self := r.routerID()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, relayFleetResponse{
			RouterID: self, BenchVersion: benchmarkVersion,
			Models: []relayModelEntry{{ID: "m", Model: "m", Quality: 50, BenchVersion: benchmarkVersion}},
		})
	}))
	defer upstream.Close()

	r.relays.put(Relay{Name: "mirror", URL: upstream.URL, Enabled: true})
	r.refreshRelays()
	if ids := r.relayRowIDs(""); len(ids) != 0 {
		t.Fatalf("registered %v from a relay that is this router", ids)
	}
}

// A failed fetch is a WAN blip, not a decommissioning: the rows stay and the
// health loop decides whether they are usable.
func TestRelayKeepsRowsWhenFetchFails(t *testing.T) {
	fail := false
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if fail {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		writeJSON(w, http.StatusOK, relayFleetResponse{
			RouterID: "r-up", BenchVersion: benchmarkVersion,
			Models: []relayModelEntry{{ID: "m", Model: "m", Quality: 70, BenchVersion: benchmarkVersion, MaxConcurrency: 2}},
		})
	}))
	defer upstream.Close()

	r := adminRouter(t)
	r.relays.put(Relay{Name: "up", URL: upstream.URL, Enabled: true})
	r.refreshRelays()
	if r.registry.get(relayBackendID("up", "m")) == nil {
		t.Fatal("first refresh registered nothing")
	}
	fail = true
	r.refreshRelays()
	if r.registry.get(relayBackendID("up", "m")) == nil {
		t.Fatal("one failed poll deregistered a healthy upstream fleet")
	}
	if rel, _ := r.relays.lookup("up"); rel.LastError == "" {
		t.Error("the failure was not recorded on the relay")
	}
}

// A different benchmark makes the quality number incomparable. The row stays
// routable — at the same provisional tier an unproven worker holds — rather
// than adopting a score from another scale or claiming a local re-measurement
// that a relay can never do.
// A relay row has to arrive MEASURED, and a quality score is not that.
//
// Everything else about an upstream worker already crossed the wire — speed,
// capacity, context, capabilities, and the benchmark score — but the score is a
// scalar, and routing ranks on the per-question rows behind it. So a relay row
// arrived carrying q=89 and no evidence, which is exactly the state automatic
// routing refuses to choose: last in the fleet, behind workers measured to be
// worse, for as long as it existed. The rows are the fix; the fleet response
// carries them when the downstream asks.
func TestRelayImportsTheEvidenceBehindTheQualityScore(t *testing.T) {
	graded := []relayObservation{
		{QID: "q1", Thinking: true, Correct: true, LatencyMS: 900},
		{QID: "q2", Thinking: true, Correct: true, LatencyMS: 900},
		{QID: "q3", Thinking: true, Correct: false, LatencyMS: 900},
		{QID: "q4", Thinking: true, Correct: true, LatencyMS: 900},
	}
	var fetches, asked int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		fetches++
		entry := relayModelEntry{
			ID: "up-qwen", Model: "qwen", Quality: 89, BenchVersion: benchmarkVersion,
			ContextK: 128, Features: []string{"chat"}, Thinking: true,
			BaselineTPS: 100, MaxConcurrency: 4,
		}
		// The rows ride along only when asked for: they are ~400 per worker, and
		// the poll that carries everything else runs every fifteen seconds.
		if wantsRelayObservations(req) {
			asked++
			entry.Observations = graded
		}
		writeJSON(w, http.StatusOK, relayFleetResponse{
			RouterID: "r-up", BenchVersion: benchmarkVersion, Models: []relayModelEntry{entry},
		})
	}))
	defer upstream.Close()

	r := adminRouter(t)
	r.outcomes = newOutcomeMatrix(nil)
	r.relays.put(Relay{Name: "work", URL: upstream.URL, Enabled: true})
	r.refreshRelays()

	id := relayBackendID("work", "up-qwen")
	b := r.registry.get(id)
	if b == nil {
		t.Fatalf("relay row %q was not registered", id)
	}
	// Filed under the identity THIS registry computes for the row. The upstream's
	// own hash for the same worker is a different string and would be an address
	// nothing here ever looks up.
	rate, measured := r.outcomes.bankRates(true)[modelHash(b)].rate()
	if !measured {
		t.Fatal("the relay row is still unmeasured after importing its graded answers — " +
			"automatic routing will not choose it")
	}
	if rate != 0.75 {
		t.Errorf("imported hit rate = %.2f, want 0.75 (three of four)", rate)
	}

	// The point of all of it: ranked on its own record, ahead of a local worker
	// nothing has measured. Before the import both were unmeasured and the relay
	// row could only ever be reached by failover.
	got, reason, _ := r.outcomes.chooseByOutcome([]*Backend{testBackend("local-fresh", 1000), b},
		vec(0, 0, 1), true, jobCost{outputTokens: 200, mode: thinkingOn})
	if len(got) == 0 || got[0].ID != id {
		t.Errorf("the relayed worker did not lead an unmeasured local one (reason %q)", reason)
	}

	// Asked once, not on every poll. The fleet fetch runs every fifteen seconds
	// for the life of the router; the evidence is needed once per worker.
	r.refreshRelays()
	r.refreshRelays()
	if fetches < 3 {
		t.Fatalf("test setup: %d fetches, want at least 3", fetches)
	}
	if asked != 1 {
		t.Errorf("asked for the graded rows on %d of %d fetches, want exactly 1", asked, fetches)
	}
}

func TestRelayBenchmarkMismatchHoldsProvisionalQuality(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, relayFleetResponse{
			RouterID: "r-up", BenchVersion: benchmarkVersion + 1,
			Models: []relayModelEntry{{
				ID: "m", Model: "m", Quality: 99, BenchVersion: benchmarkVersion + 1,
				ContextK: 64, MaxConcurrency: 3, Features: []string{"chat"},
				// Graded against the other router's question set, so these rows are
				// on the same incomparable scale as the 99 above.
				Observations: []relayObservation{
					{QID: "other-bank-q1", Thinking: true, Correct: true},
					{QID: "other-bank-q2", Thinking: true, Correct: true},
				},
			}},
		})
	}))
	defer upstream.Close()

	r := adminRouter(t)
	r.outcomes = newOutcomeMatrix(nil)
	r.relays.put(Relay{Name: "up", URL: upstream.URL, Enabled: true})
	r.refreshRelays()

	// The rows go through the same gate as the score they add up to. Importing
	// them would file a hit rate measured on another question set under the one
	// identity the ranker reads.
	if b := r.registry.get(relayBackendID("up", "m")); b != nil && r.outcomes.hasBenchEvidence(modelHash(b)) {
		t.Error("graded answers from a different benchmark version were imported as evidence on this router's scale")
	}

	b := r.registry.get(relayBackendID("up", "m"))
	if b == nil {
		t.Fatal("a version mismatch must not stop the models being usable")
	}
	if b.Quality != provisionalQuality {
		t.Fatalf("quality = %d, want the provisional %d rather than an incomparable 99", b.Quality, provisionalQuality)
	}
	// Capacity and capabilities are facts about the deployment, not the question
	// set, so they still cross.
	if b.ContextK != 64 || b.MaxConcurrency != 3 {
		t.Fatalf("deployment facts were dropped with the score: ctx=%d conc=%d", b.ContextK, b.MaxConcurrency)
	}
	if line := r.relayHealthLine(); line == nil || line["benchmark_mismatch"] != 1 {
		t.Fatalf("/health does not report the mismatch: %+v", line)
	}
}

func TestRelayAdminCRUD(t *testing.T) {
	r := adminRouter(t)
	if rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/relays", `{"name":"work"}`)); rec.Code != http.StatusBadRequest {
		t.Fatalf("a relay with no url: want 400, got %d", rec.Code)
	}
	if rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/relays", `{"name":"a:b","url":"http://x"}`)); rec.Code != http.StatusBadRequest {
		t.Fatalf("a name carrying the id separator: want 400, got %d", rec.Code)
	}
	if rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/relays", `{"name":"work","url":"x"}`)); rec.Code != http.StatusBadRequest {
		t.Fatalf("a url that is not a base url: want 400, got %d", rec.Code)
	}
	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/relays", `{"name":"Work","url":"http://up:8585","api_key":"sk-x"}`))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create: %d %s", rec.Code, rec.Body.String())
	}
	// The secret never comes back out.
	if strings.Contains(rec.Body.String(), "sk-x") {
		t.Fatalf("the relay credential was echoed: %s", rec.Body.String())
	}
	// SaveRelay upserts, so a second POST must be refused rather than silently
	// replacing the url and credential of the one already there.
	if rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/relays", `{"name":"work","url":"http://other:8585"}`)); rec.Code != http.StatusConflict {
		t.Fatalf("re-creating an existing relay: want 409, got %d: %s", rec.Code, rec.Body.String())
	}
	if rel, _ := r.relays.lookup("work"); rel.URL != "http://up:8585" {
		t.Fatalf("the refused create changed the relay anyway: url=%q", rel.URL)
	}
	rec = serveAdmin(r, adminReq(http.MethodGet, "/admin/relays", ""))
	if !strings.Contains(rec.Body.String(), `"work"`) {
		t.Fatalf("list does not carry the lowercased name: %s", rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "sk-x") {
		t.Fatalf("list leaked the relay credential: %s", rec.Body.String())
	}
	// Renaming would orphan every derived backend id, which carries the name.
	if rec := serveAdmin(r, adminReq(http.MethodPatch, "/admin/relays/work", `{"name":"other"}`)); rec.Code != http.StatusConflict {
		t.Fatalf("rename: want 409, got %d", rec.Code)
	}
	if rec := serveAdmin(r, adminReq(http.MethodPatch, "/admin/relays/work", `{"enabled":false}`)); rec.Code != http.StatusOK {
		t.Fatalf("disable: %d %s", rec.Code, rec.Body.String())
	}
	if rel, _ := r.relays.lookup("work"); rel.Enabled {
		t.Fatal("the relay is still enabled")
	}
	if rel, _ := r.relays.lookup("work"); rel.APIKey != "sk-x" {
		t.Fatal("an edit that did not mention the credential dropped it")
	}
	// It survives a restart, credential included.
	restored := &Router{cfg: r.cfg, registry: newTestRegistry(), logs: r.logs}
	restored.loadRelays(t.Context())
	got, ok := restored.relays.lookup("work")
	if !ok || got.URL != "http://up:8585" || got.APIKey != "sk-x" || got.Enabled {
		t.Fatalf("relay did not survive a restart intact: ok=%v %+v", ok, got)
	}
	if rec := serveAdmin(r, adminReq(http.MethodDelete, "/admin/relays/work", "")); rec.Code != http.StatusOK {
		t.Fatalf("delete: %d %s", rec.Code, rec.Body.String())
	}
	if _, ok := r.relays.lookup("work"); ok {
		t.Fatal("the relay is still configured after a delete")
	}
}

// A beacon must not be able to squat a derived id: it would be taken over here
// and pruned again by the next refresh, and the two would fight over it forever.
func TestRegisterRefusesRelayID(t *testing.T) {
	r := registerRouter(t)
	r.registry.upsert(BackendRegistration{
		ID: "work:m", URL: "http://up", Model: "m", Source: sourceRelay, Relay: "work",
	})
	req := httptest.NewRequest(http.MethodPost, "/backends/register",
		strings.NewReader(`{"id":"work:m","url":"http://elsewhere:8000","model":"m"}`))
	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Fatalf("want 409, got %d: %s", rec.Code, rec.Body.String())
	}
	if b := r.registry.get("work:m"); b == nil || !isRelayRow(b) {
		t.Fatal("the beacon took the relay row over anyway")
	}
}

// ── The ranker ──────────────────────────────────────────────────────────────

// Occupancy is the fleet's, not our share of it — otherwise a relay looks idle
// while the upstream's own traffic is saturating the same GPUs.
func TestOccupancyPrefersUpstreamCount(t *testing.T) {
	local := &Backend{ActiveRequests: 2}
	if got := occupancy(local); got != 2 {
		t.Fatalf("local occupancy = %d, want 2", got)
	}
	busy := &Backend{ActiveRequests: 1, RemoteActive: 5}
	if got := occupancy(busy); got != 5 {
		t.Fatalf("relay occupancy = %d, want the upstream's 5", got)
	}
	// Max, not sum: our in-flight requests are already inside the upstream's
	// count, so adding them would charge each one twice.
	burst := &Backend{ActiveRequests: 7, RemoteActive: 5}
	if got := occupancy(burst); got != 7 {
		t.Fatalf("burst occupancy = %d, want 7 — a burst since the last poll is not yet in the remote figure", got)
	}
}

// Two workers the upstream measures identically, one of them across a 90ms
// link: the far one must rank worse, or the link is free in every comparison
// against a local worker.
func TestRelayLatencyIncludesTheLink(t *testing.T) {
	entry := relayModelEntry{
		ID: "m", Model: "m", Quality: 80, BenchVersion: benchmarkVersion, ContextK: 128,
		Features: []string{"chat"}, BaselineTPS: 100, PrefillTPS: 4000, TTFTMillis: 250,
		MaxConcurrency: 4,
	}
	build := func(id string, rtt float64) *Backend {
		reg := newTestRegistry()
		reg.upsert(BackendRegistration{ID: id, URL: "http://" + id, Model: "m"})
		reg.applyProfileIfGen(id, 0, relayProfile(entry))
		reg.setRelayLoad(id, 0, entry.MaxConcurrency, rtt)
		reg.finishCertification(id, true, map[string]Check{}, entry.BaselineTPS, entry.TTFTMillis, "")
		return reg.get(id)
	}
	near, far := build("local", 0), build("relayed", 90)
	job := jobCost{promptTokens: 1000, outputTokens: 500}
	if got, want := expectedLatency(far, job)-expectedLatency(near, job), 0.09; got < want*0.9 || got > want*1.1 {
		t.Fatalf("the link cost %.4fs in the estimate, want ~%.2fs (far=%v near=%v)",
			got, want, expectedLatency(far, job), expectedLatency(near, job))
	}
}

// The prefill rate has to SCALE with the prompt, which is the whole reason it is
// imported rather than left to the flat TTFT fallback. Without it a remote model
// costs the same for a 200-token prompt and a 100k one, and wins every
// long-context comparison against a local worker it is nowhere near as fast as.
func TestRelayPrefillScalesWithThePrompt(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "far", URL: "http://far", Model: "m"})
	reg.applyProfileIfGen(id0(t, reg, "far"), 0, relayProfile(relayModelEntry{
		ID: "m", Model: "m", Quality: 80, BenchVersion: benchmarkVersion, ContextK: 256,
		BaselineTPS: 100, PrefillTPS: 2800, TTFTMillis: 300, MaxConcurrency: 4,
	}))
	reg.setRelayLoad("far", 0, 4, 137)
	b := reg.get("far")

	short := prefillSeconds(b, 200)
	long := prefillSeconds(b, 100_000)
	if long <= short*10 {
		t.Fatalf("prefill does not scale with the prompt: 200 tokens %.3fs vs 100k tokens %.3fs", short, long)
	}
	// 100k at 2800 tok/s is ~35.7s, plus the 0.137s link.
	if want := 100_000/2800.0 + 0.137; long < want*0.99 || long > want*1.01 {
		t.Fatalf("100k prompt priced at %.3fs, want ~%.3fs", long, want)
	}
}

// id0 returns the id it was given, after asserting the row exists — a readable
// way to keep the upsert and the profile application on one line each.
func id0(t *testing.T, reg *Registry, id string) string {
	t.Helper()
	if reg.get(id) == nil {
		t.Fatalf("no backend %q", id)
	}
	return id
}
