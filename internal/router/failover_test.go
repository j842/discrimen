package router

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// A 5xx with nothing written to the client is a ROUTING failure, not an answer:
// the caller asked the router to choose, and the worker it chose could not
// respond. These pin that the request moves, and — just as important — that it
// does NOT move in the cases where moving would break a guarantee.

// failoverFleet stands up two upstreams and a router whose plan lists them in
// order, so "the next candidate" is unambiguous.
func failoverFleet(t *testing.T, first, second http.HandlerFunc) (*Router, *dispatch, *httptest.ResponseRecorder, *http.Request) {
	t.Helper()
	a := httptest.NewServer(first)
	t.Cleanup(a.Close)
	b := httptest.NewServer(second)
	t.Cleanup(b.Close)

	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "first", URL: a.URL, Model: "m", TTLSeconds: 3600, MaxConcurrency: 2})
	reg.upsert(BackendRegistration{ID: "second", URL: b.URL, Model: "m", TTLSeconds: 3600, MaxConcurrency: 2})
	r := &Router{
		cfg: &Config{BackendIdleTimeout: 5 * time.Second}, registry: reg,
		client: &http.Client{}, streamClient: &http.Client{},
	}
	backend := reg.get("first")
	slot := make(chan struct{}, 1)
	escalated := false
	d := &dispatch{
		backend: &backend, slot: &slot,
		body: []byte(`{"messages":[]}`), raw: []byte(`{"messages":[]}`),
		plan: &routePlan{
			auto:       true,
			route:      "route:outcome:p=0.90,n=8",
			candidates: []*Backend{reg.get("first"), reg.get("second")},
		},
		job: nominalJob(), log: &RequestLog{BackendID: "first"},
		output: &strings.Builder{}, escalated: &escalated,
	}
	return r, d, httptest.NewRecorder(),
		post("/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`, "")
}

func jsonAnswer(w http.ResponseWriter, text string) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"` + text + `"}}]}`))
}

// 503 on the first worker → the second serves it, and the client sees one 200.
func TestFailoverServesFromTheNextCandidate(t *testing.T) {
	r, d, rec, req := failoverFleet(t,
		func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"Loading model"}`))
		},
		func(w http.ResponseWriter, _ *http.Request) { jsonAnswer(w, "hello") })

	r.dispatchBuffered(rec, req, d)

	if rec.Code != http.StatusOK {
		t.Fatalf("client saw %d, want 200 — the 503 should never have reached it", rec.Code)
	}
	if got := rec.Header().Get("X-LLM-Backend-ID"); got != "second" {
		t.Errorf("X-LLM-Backend-ID = %q, want \"second\" — the header must name who actually answered", got)
	}
	if got := rec.Header().Get("X-LLM-Failover"); !strings.Contains(got, "first->second") {
		t.Errorf("X-LLM-Failover = %q, want first->second", got)
	}
	if !strings.Contains(rec.Body.String(), "hello") {
		t.Errorf("body did not come from the second worker: %s", rec.Body.String())
	}
}

// A 4xx is the REQUEST's problem and will fail identically everywhere. Moving it
// would burn a second worker's slot to produce the same error.
func TestNoFailoverOnClientError(t *testing.T) {
	secondCalled := false
	r, d, rec, req := failoverFleet(t,
		func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"malformed"}`))
		},
		func(w http.ResponseWriter, _ *http.Request) { secondCalled = true; jsonAnswer(w, "hello") })

	r.dispatchBuffered(rec, req, d)

	if secondCalled {
		t.Error("a 4xx was failed over; every worker will reject it identically")
	}
	if rec.Code != http.StatusBadRequest {
		t.Errorf("client saw %d, want the original 400", rec.Code)
	}
}

// A pin or a named model is an INSTRUCTION. Answering from somewhere else
// silently breaks exactly the guarantee the caller asked for.
func TestNoFailoverWhenTheCallerChose(t *testing.T) {
	secondCalled := false
	r, d, rec, req := failoverFleet(t,
		func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) },
		func(w http.ResponseWriter, _ *http.Request) { secondCalled = true; jsonAnswer(w, "hello") })
	d.plan.auto = false // pinned / named / debug

	r.dispatchBuffered(rec, req, d)

	if secondCalled {
		t.Error("a caller-chosen worker was failed over; the pin was not honoured")
	}
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("client saw %d, want the 503 the pinned worker gave", rec.Code)
	}
}

// Every candidate down → the client gets a 503 that says when to come back.
// Without the hint a client cannot tell "retry shortly" from "give up", retries
// immediately, and makes the saturation worse.
func TestAllCandidatesDownReturns503WithRetryAfter(t *testing.T) {
	down := func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) }
	r, d, rec, req := failoverFleet(t, down, down)
	r.cfg.HealthInterval = 3 * time.Second // what retryAfterUnavailable derives the hint from
	// With every candidate exhausted the ladder IS the last remaining move, so it
	// runs — shortened here so the test does not spend 17 seconds proving it.
	restore := proxyRetryDelays
	proxyRetryDelays = []time.Duration{time.Millisecond, time.Millisecond}
	t.Cleanup(func() { proxyRetryDelays = restore })

	r.dispatchBuffered(rec, req, d)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("client saw %d, want 503", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Error("503 carries no Retry-After; the invariant is that this router's 503 always says when to come back")
	}
}

// A worker whose 5xx survives failover must show WHY on /backends. An empty
// LastError on a failing worker reads as healthy.
func TestUpstreamErrorSnippet(t *testing.T) {
	got := upstreamErrorSnippet(503, []byte("Loading model\nplease wait"))
	if !strings.Contains(got, "503") || !strings.Contains(got, "Loading model") {
		t.Errorf("snippet = %q, want the status and the upstream's message", got)
	}
	if strings.Contains(got, "\n") {
		t.Errorf("snippet is multi-line: %q", got)
	}
	long := upstreamErrorSnippet(500, []byte(strings.Repeat("x", 5000)))
	if len(long) > upstreamSnippetMax+40 {
		t.Errorf("snippet not bounded: %d chars", len(long))
	}
	if empty := upstreamErrorSnippet(502, nil); !strings.Contains(empty, "502") {
		t.Errorf("empty body should still record the status, got %q", empty)
	}
}

// An upstream that says WHEN to come back knows better than a fixed ladder.
func TestRetryAfterHint(t *testing.T) {
	h := http.Header{}
	if d := retryAfterHint(h); d != 0 {
		t.Errorf("absent header = %v, want 0", d)
	}
	h.Set("Retry-After", "5")
	if d := retryAfterHint(h); d != 5*time.Second {
		t.Errorf("seconds form = %v, want 5s", d)
	}
	h.Set("Retry-After", "not-a-number")
	if d := retryAfterHint(h); d != 0 {
		t.Errorf("unparseable = %v, want 0", d)
	}
	h.Set("Retry-After", http.TimeFormat[:0]+time.Now().Add(30*time.Second).UTC().Format(http.TimeFormat))
	if d := retryAfterHint(h); d <= 0 || d > 31*time.Second {
		t.Errorf("date form = %v, want ~30s", d)
	}
}

// nextCandidates preserves the matrix's ordering — "expected correct, fastest
// first" — so the next candidate is already the right one to try.
func TestNextCandidatesPreservesOrder(t *testing.T) {
	a, b, c := testBackend("a", 1), testBackend("b", 2), testBackend("c", 3)
	plan := &routePlan{candidates: []*Backend{a, b, c}}
	got := nextCandidates(plan, map[string]bool{"b": true})
	if len(got) != 2 || got[0].ID != "a" || got[1].ID != "c" {
		t.Errorf("got %v, want [a c] in order", idsOf(got))
	}
	if all := nextCandidates(plan, map[string]bool{"a": true, "b": true, "c": true}); len(all) != 0 {
		t.Errorf("everything tried should leave nothing, got %v", idsOf(all))
	}
	if nextCandidates(nil, nil) != nil {
		t.Error("a nil plan should yield nothing rather than panic")
	}
}

func idsOf(bs []*Backend) []string {
	out := make([]string, 0, len(bs))
	for _, b := range bs {
		out = append(out, b.ID)
	}
	return out
}

// Escalation, the tier adapter and the judge all used to gate on the route
// string containing a literal "route:d=". That held only while every auto route
// produced that one shape — and the moment the outcome matrix began emitting
// "route:outcome:…", all three silently switched off for the path carrying most
// traffic. The judge going dark was the worst of it: recordJudgedOutcome lives
// inside maybeJudge, so the matrix's own feedback loop was open, closing only
// when the embeddings worker was down and the tier path took over.
//
// These pin the structural replacement.
func TestAutoRouteIsStructuralNotStringSniffed(t *testing.T) {
	// A matrix route carries no "d=" and must still count as router-chosen.
	matrix := &routePlan{auto: true, route: "route:outcome:p=0.85,n=12"}
	if _, ok := parseRouteScore(matrix.route); ok {
		t.Fatal("test premise wrong: the matrix route parses as a tier score")
	}
	if !matrix.auto {
		t.Error("a matrix route must still be auto — escalation and judging hang off this")
	}

	// A tier route keeps both properties.
	tier := &routePlan{auto: true, route: "route:d=0.62,q>=62"}
	if score, ok := parseRouteScore(tier.route); !ok || score != 0.62 {
		t.Errorf("tier route score = %.2f ok=%v, want 0.62 true", score, ok)
	}
	if !tier.auto {
		t.Error("a tier route must be auto")
	}

	// A client-named model is NOT auto however it was ranked: the caller made
	// the choice, so escalating or judging it second-guesses an instruction.
	named := &routePlan{auto: false, route: "model:outcome:p=0.85,n=12"}
	if named.auto {
		t.Error("a named-model route must not be auto")
	}
	// The old string gate agreed on this case, which is why it survived so long.
	if _, ok := parseRouteScore("model:d=0.62,q>=62"); ok {
		t.Error("parseRouteScore accepted a named-model route")
	}
}

// Every branch that lets the ROUTER choose the worker must mark the plan auto.
// The classifier being unavailable does not make the pick the caller's — and
// omitting it switched off escalation, both failover paths and the judge in
// exactly the degraded mode where they matter most.
func TestEveryRouterChosenBranchIsAuto(t *testing.T) {
	reg := newTestRegistry()
	readyBackend(reg, "tiny", 20, 200, 2)
	req := &ChatRequest{Messages: []Message{{Role: "user", Content: "hi"}}}

	// No classifier at all — the degraded mode, which reaches the third branch.
	bare := &Router{cfg: &Config{DefaultMaxTokens: 4096}, registry: reg,
		sessions: newSessionTracker(time.Hour, 16)}
	plan, err := bare.planRoute(req, 0, false)
	if err != nil {
		t.Fatalf("planRoute: %v", err)
	}
	if !plan.auto {
		t.Errorf("route %q from the no-classifier branch is not marked auto, so escalation, "+
			"both failover paths and the judge are all switched off", plan.route)
	}

	// And a client-named model is NOT auto on that same branch: the caller chose.
	named := &ChatRequest{Model: "tiny", Messages: req.Messages}
	if p, err := bare.planRoute(named, 0, false); err == nil && p.auto {
		t.Errorf("route %q for a client-named model is marked auto", p.route)
	}
}

// ── Defect: a failover target that ALSO fails was recorded nowhere ─────────
//
// The comment on failover records this being fixed for the worker the request
// was moved AWAY from. The worker it was moved TO had the same hole, on the
// other side of the same call: redispatch's reject path released the slot and
// returned with no noteFailedAttempt, no noteProxyResult, and no error recorded
// (requestBufferedWithDelays writes setError for a TRANSPORT error only, never
// for a 5xx status).
//
// Measured shape: worker A is wedged, worker B is wedged too. Every request 503s
// on A, fails over to B, takes a 503 there and returns A's error. A accrues
// proxyFailures and is ejected after three; B stays at proxyFailures=0 with a
// blank LastError and renders on /backends as a healthy worker. B only starts
// being judged once A is gone and it becomes primary.
func TestFailoverTargetThatAlsoFailsIsRecorded(t *testing.T) {
	down := func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"CUDA out of memory"}`))
	}
	r, d, rec, req := failoverFleet(t, down, down)
	// Every candidate is exhausted, so the ladder runs; shortened so the test does
	// not spend 17 seconds proving it.
	restore := proxyRetryDelays
	proxyRetryDelays = []time.Duration{time.Millisecond}
	t.Cleanup(func() { proxyRetryDelays = restore })

	r.dispatchBuffered(rec, req, d)

	target := r.registry.get("second")
	if target.proxyFailures == 0 {
		t.Error("the failover target 503'd and accrued nothing: proxyFailures stays 0, so its breaker " +
			"can never trip however long it stays wedged")
	}
	if target.LastError == "" {
		t.Fatal("the failover target renders on /backends with a blank LastError, which reads as healthy")
	}
	if !strings.Contains(target.LastError, "CUDA out of memory") {
		t.Errorf("LastError = %q, want the upstream's own message", target.LastError)
	}
	// The worker the request was moved away from is still recorded too — this
	// must not have moved the hole rather than closed it.
	if from := r.registry.get("first"); from.proxyFailures == 0 {
		t.Error("the worker that was failed over lost its own strike")
	}
}

// The streamed path had the identical hole: a re-dial that 503s released the
// slot and continued, recording nothing against the worker that refused it.
func TestStreamFailoverTargetThatAlsoFailsIsRecorded(t *testing.T) {
	down := func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusServiceUnavailable) }
	r, d, _, req := failoverFleet(t, down, down)

	// What dispatchStreaming hands streamFailover: the first worker answered 503
	// at the dial, before a single byte reached the client.
	first := &http.Response{StatusCode: http.StatusServiceUnavailable, Body: http.NoBody}
	if _, _, moved := r.streamFailover(context.Background(), req, d, first, nil); moved {
		t.Fatal("test premise wrong: both workers are down, nothing should have moved")
	}

	target := r.registry.get("second")
	if target.proxyFailures == 0 {
		t.Error("the streamed failover target refused the dial and accrued nothing")
	}
	if target.LastError == "" {
		t.Error("the streamed failover target keeps a blank LastError, which reads as healthy on /backends")
	}
}

// An empty 200 is a failure by the router's own rule (writeBuffered scores it as
// one), and noteFailedAttempt has to reach the SAME verdict — otherwise an
// escalation target that returned nothing was recorded as a success, resetting
// its consecutive-failure run. That is exactly the laundering writeBuffered was
// fixed for, on the path where the empty answer is never written to anyone.
func TestNoteFailedAttemptTreatsAnEmpty200AsAFailure(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "w", URL: "http://w", Model: "m", TTLSeconds: 3600})
	r := &Router{registry: reg}

	r.noteFailedAttempt("w", bufferedResult{statusCode: http.StatusOK, body: []byte(emptyAnswer)})
	if b := reg.get("w"); b.proxyFailures != 1 || b.LastError == "" {
		t.Errorf("an empty 200 scored as a success: proxyFailures=%d LastError=%q", b.proxyFailures, b.LastError)
	}
	// A 4xx is the request's problem, not the worker's — unchanged, and the same
	// reading writeBuffered gives it.
	r.noteFailedAttempt("w", bufferedResult{statusCode: http.StatusBadRequest, body: []byte(`{"error":"nope"}`)})
	if b := reg.get("w"); b.proxyFailures != 0 {
		t.Errorf("a 4xx was charged to the worker: proxyFailures=%d", b.proxyFailures)
	}
	// And a 429 from the streaming path, which carries no body, keeps its
	// existing reading rather than being caught by the emptiness test.
	r.noteFailedAttempt("w", bufferedResult{statusCode: http.StatusTooManyRequests})
	if b := reg.get("w"); b.proxyFailures != 0 {
		t.Errorf("a bodyless 429 was charged to the worker: proxyFailures=%d", b.proxyFailures)
	}
}

// ── Defect: redispatch's in-flight increment was not defer-protected ───────

// Every other incActive site in the codebase pairs with a defer. Here the
// decrement and the slot release sat on ordinary statement paths, so a panic
// inside requestBufferedWithDelays or accept() left the target permanently +1
// and its slot permanently consumed. proxyToBackend's own defers do not cover
// it: they unwind through *d.backend / *d.slot, which have not been swapped yet,
// so they release the OLD worker a second time.
//
// ActiveRequests only ever clamps at zero and never corrects downward, so a
// max_concurrency:1 worker that takes one such hit looks saturated — and is
// ranked last — for the rest of the process's life.
func TestRedispatchReleasesTheTargetOnPanic(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		jsonAnswer(w, "hello")
	}))
	t.Cleanup(srv.Close)

	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "from", URL: srv.URL, Model: "m", TTLSeconds: 3600, MaxConcurrency: 1})
	reg.upsert(BackendRegistration{ID: "target", URL: srv.URL, Model: "m", TTLSeconds: 3600, MaxConcurrency: 1})
	r := &Router{cfg: &Config{}, registry: reg, client: &http.Client{}, streamClient: &http.Client{}}

	backend := reg.get("from")
	slot := make(chan struct{}, 1)
	escalated := false
	d := &dispatch{
		backend: &backend, slot: &slot,
		body: []byte(`{"messages":[]}`), raw: []byte(`{"messages":[]}`),
		plan: &routePlan{auto: true, route: "route:outcome:p=0.90,n=8",
			candidates: []*Backend{reg.get("from"), reg.get("target")}},
		job: nominalJob(), log: &RequestLog{BackendID: "from"},
		output: &strings.Builder{}, escalated: &escalated,
	}
	req := post("/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`, "")

	var recovered any
	func() {
		defer func() { recovered = recover() }()
		r.redispatch(req, d, bufferedResult{}, []*Backend{reg.get("target")},
			func(bufferedResult) bool { panic("deciding whether to accept blew up") }, nil)
	}()
	if recovered == nil {
		t.Fatal("test premise wrong: accept() did not panic")
	}

	if b := reg.get("target"); b.ActiveRequests != 0 {
		t.Errorf("target left at active_requests=%d; ActiveRequests never corrects downward, so this "+
			"worker is ranked as permanently saturated", b.ActiveRequests)
	}
	if _, ok := reg.tryAcquireSlot("target"); !ok {
		t.Error("target's only concurrency slot was consumed by the panic and never returned — no request " +
			"can ever be dispatched to it again")
	}
	// The worker the request came from still holds what it held: ownership never
	// transferred, so nothing of its accounting may have been unwound.
	if b := reg.get("from"); b.ActiveRequests != 0 {
		t.Errorf("the ORIGINAL worker's accounting was disturbed by a failed redispatch: active_requests=%d", b.ActiveRequests)
	}
	if *d.backend == nil || (*d.backend).ID != "from" {
		t.Error("a panicking redispatch swapped the caller's backend anyway")
	}
}

// An upstream's Retry-After is advice about the upstream, not permission to
// abandon the caller.
func TestRetryAfterIsCapped(t *testing.T) {
	if maxRetryAfterWait > 10*time.Second {
		t.Errorf("maxRetryAfterWait is %s — long enough to hold a caller's slot hostage", maxRetryAfterWait)
	}
	h := http.Header{}
	h.Set("Retry-After", "600")
	if hint := retryAfterHint(h); hint <= maxRetryAfterWait {
		t.Fatalf("test premise wrong: hint %s already under the cap", hint)
	}
}

// The retry ladder must honour the caller's declared budget. It read
// req.Context().Deadline(), which net/http never sets, so the check could never
// fire and the ladder slept its full length against any budget.
func TestRemainingBudget(t *testing.T) {
	none := &dispatch{}
	if none.remainingBudget() != 0 {
		t.Error("no declared budget should read as unbounded (0)")
	}
	live := &dispatch{budget: time.Second, start: time.Now()}
	if b := live.remainingBudget(); b <= 0 || b > time.Second {
		t.Errorf("remaining = %v, want just under 1s", b)
	}
	expired := &dispatch{budget: time.Millisecond, start: time.Now().Add(-time.Hour)}
	if b := expired.remainingBudget(); b <= 0 {
		t.Errorf("an expired budget returned %v; it must stay positive-but-tiny so the "+
			"ladder returns immediately rather than reading as unbounded", b)
	}
}
