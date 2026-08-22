package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

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

// Discovery and traffic are gated by two different tests — allowsBackend, which
// accepts any spelling a worker answers to, and allowsModel, which compares the
// client's word to the allow-list literally. Publish a name outside the list and
// the downstream is refused by the key that just advertised it.
func TestRelayFleetPublishesASayableName(t *testing.T) {
	reg := newTestRegistry()
	relayReady(t, reg, "w-endpoint-id", "Qwen3.8-27B-Uncensored-NInfer", 80, 128, 4, "chat")

	// Every spelling backendServesModel accepts has to round-trip: whichever one
	// the operator wrote is what comes back.
	for _, spelling := range []string{
		"Qwen3.8-27B-Uncensored-NInfer", // the model id
		"w-endpoint-id",                 // the endpoint id
		"qwen3.8uncensoredninfer",       // the published alias
	} {
		ident := &identity{Role: roleClient, Relay: true, Models: []string{spelling}}
		got := relayFleetFor(ident, reg.eligible())
		if len(got) != 1 {
			t.Fatalf("allow-list %q: got %d entries, want 1", spelling, len(got))
		}
		if got[0].Model != spelling {
			t.Errorf("allow-list %q published as %q — the downstream would ask for a name this key refuses",
				spelling, got[0].Model)
		}
		// And prove the refusal is real: the published name must pass the gate the
		// traffic path applies.
		if !ident.allowsModel(got[0].Model) {
			t.Errorf("published %q, which allowsModel refuses", got[0].Model)
		}
	}

	// No allow-list refuses nothing, so the raw model id is the right answer.
	got := relayFleetFor(&identity{Role: roleClient, Relay: true}, reg.eligible())
	if len(got) != 1 || got[0].Model != "Qwen3.8-27B-Uncensored-NInfer" {
		t.Fatalf("unrestricted key: got %+v, want the raw model id", got)
	}
}

// Several endpoints serving one model become one entry, and each field combines
// in the direction its USE requires — see relayModelEntry.
func TestRelayFleetAggregatesEndpoints(t *testing.T) {
	reg := newTestRegistry()
	relayReady(t, reg, "a", "m", 80, 128, 4, "chat", "tools", "vision")
	relayReady(t, reg, "b", "m", 60, 32, 2, "chat", "tools")
	reg.backends["a"].Thinking = true // b cannot think; the union still can
	reg.incActive("a", 1)
	reg.incActive("b", 2)

	got := relayFleetFor(&identity{Role: roleClient, Relay: true}, reg.eligible())
	if len(got) != 1 {
		t.Fatalf("want one aggregated entry, got %d: %+v", len(got), got)
	}
	e := got[0]
	if e.Endpoints != 2 {
		t.Errorf("endpoints = %d, want 2", e.Endpoints)
	}
	if e.Quality != 60 {
		t.Errorf("quality = %d, want the minimum 60 — a request may land on either", e.Quality)
	}
	if e.ContextK != 32 {
		t.Errorf("context_k = %d, want the minimum 32", e.ContextK)
	}
	if strings.Join(e.Features, ",") != "chat,tools" {
		t.Errorf("features = %v, want the intersection [chat tools]", e.Features)
	}
	if !e.Thinking {
		t.Error("thinking = false, want the union true — the upstream hard-filters within the model")
	}
	if e.MaxConcurrency != 6 {
		t.Errorf("max_concurrency = %d, want the sum 6", e.MaxConcurrency)
	}
	if e.ActiveRequests != 3 {
		t.Errorf("active_requests = %d, want the sum 3", e.ActiveRequests)
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
		Model: "m", Quality: 77, BenchVersion: benchmarkVersion, ContextK: 128,
		Features: []string{"chat", "tools"}, Thinking: true,
		BaselineTPS: 90, PrefillTPS: 4000, TTFTMillis: 300,
		MaxConcurrency: 6, ActiveRequests: 2, Endpoints: 2,
	}
	prof := relayProfile(entry, 90)
	if prof.TTFTMillis != 390 {
		t.Errorf("ttft = %d, want the upstream's 300 plus the 90ms link", prof.TTFTMillis)
	}
	if prof.PrefillTPS != 0 {
		t.Errorf("prefill rate = %v, want it left unset — a rate cannot carry a link's latency", prof.PrefillTPS)
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
		Model: "qwen3.8-uncensored", Quality: 82, BenchVersion: benchmarkVersion,
		ContextK: 116, Features: []string{"chat", "tools", "vision"}, Thinking: true,
		BaselineTPS: 108, TTFTMillis: 250, MaxConcurrency: 4, ActiveRequests: 1, Endpoints: 1,
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
				{Model: "m1", Quality: 70, BenchVersion: benchmarkVersion, MaxConcurrency: 2, Endpoints: 1},
				{Model: "m2", Quality: 75, BenchVersion: benchmarkVersion, MaxConcurrency: 2, Endpoints: 1},
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
			Models: []relayModelEntry{{Model: "m", Quality: 50, BenchVersion: benchmarkVersion, Endpoints: 1}},
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
			Models: []relayModelEntry{{Model: "m", Quality: 70, BenchVersion: benchmarkVersion, MaxConcurrency: 2, Endpoints: 1}},
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
func TestRelayBenchmarkMismatchHoldsProvisionalQuality(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, relayFleetResponse{
			RouterID: "r-up", BenchVersion: benchmarkVersion + 1,
			Models: []relayModelEntry{{
				Model: "m", Quality: 99, BenchVersion: benchmarkVersion + 1,
				ContextK: 64, MaxConcurrency: 3, Endpoints: 1, Features: []string{"chat"},
			}},
		})
	}))
	defer upstream.Close()

	r := adminRouter(t)
	r.relays.put(Relay{Name: "up", URL: upstream.URL, Enabled: true})
	r.refreshRelays()

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
		Model: "m", Quality: 80, BenchVersion: benchmarkVersion, ContextK: 128,
		Features: []string{"chat"}, BaselineTPS: 100, TTFTMillis: 250,
		MaxConcurrency: 4, Endpoints: 1,
	}
	build := func(id string, rtt float64) *Backend {
		reg := newTestRegistry()
		reg.upsert(BackendRegistration{ID: id, URL: "http://" + id, Model: "m"})
		prof := relayProfile(entry, rtt)
		reg.applyProfileIfGen(id, 0, prof)
		reg.finishCertification(id, true, map[string]Check{}, entry.BaselineTPS, prof.TTFTMillis, "")
		return reg.get(id)
	}
	near, far := build("local", 0), build("relayed", 90)
	job := jobCost{promptTokens: 1000, outputTokens: 500}
	if expectedLatency(far, job) <= expectedLatency(near, job) {
		t.Fatalf("a relayed worker was priced as though it were in the next rack: far=%v near=%v",
			expectedLatency(far, job), expectedLatency(near, job))
	}
}

