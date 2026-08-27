package router

import (
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
