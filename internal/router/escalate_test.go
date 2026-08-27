package router

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// The split between "empty" and "truncated" is what decides whether a request is
// worth a second generation, so it is graded directly.
func TestClassifyResponse(t *testing.T) {
	cases := []struct {
		name     string
		body     string
		streamed bool
		want     inadequacy
	}{
		{"good answer", `{"choices":[{"message":{"content":"42"},"finish_reason":"stop"}]}`, false, responseOK},
		{"empty content", `{"choices":[{"message":{"content":""},"finish_reason":"stop"}]}`, false, responseEmpty},
		{"null content", `{"choices":[{"message":{"content":null},"finish_reason":"stop"}]}`, false, responseEmpty},
		{"vLLM empty tool_calls array", `{"choices":[{"message":{"content":"","tool_calls":[]}}]}`, false, responseEmpty},
		{"tool call is an answer", `{"choices":[{"message":{"content":"","tool_calls":[{"id":"a"}]}}]}`, false, responseOK},
		{"reasoning only still counts", `{"choices":[{"message":{"content":"","reasoning_content":"hmm"}}]}`, false, responseOK},
		{"truncated but non-empty", `{"choices":[{"message":{"content":"the answer is"},"finish_reason":"length"}]}`, false, responseTruncated},
		// The escalatable case: the model burned its whole budget on reasoning and
		// never answered. Emptiness has to win over the length flag, or this is
		// misfiled as a caller's too-tight ceiling and never repaired.
		{"empty AND length-capped", `{"choices":[{"message":{"content":""},"finish_reason":"length"}]}`, false, responseEmpty},

		{"stream with content", "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n", true, responseOK},
		{"stream with nothing", "data: {\"choices\":[{\"delta\":{}}]}\n", true, responseEmpty},
		{"stream tool call", "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0}]}}]}\n", true, responseOK},
		{"stream truncated", "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"length\"}]}\n", true, responseTruncated},
	}
	for _, tc := range cases {
		if got := classifyResponse([]byte(tc.body), tc.streamed); got != tc.want {
			t.Errorf("%s: classifyResponse=%d want %d", tc.name, got, tc.want)
		}
		// The boolean the adapter consumes must be unchanged by the split.
		if wantBool := tc.want != responseOK; responseInadequate([]byte(tc.body), tc.streamed) != wantBool {
			t.Errorf("%s: responseInadequate disagrees with classifyResponse", tc.name)
		}
	}
}

// ── Defect: the two failure modes ranked candidates differently ────────────
//
// A 5xx went through failover → nextCandidates, which preserves the matrix's
// ordering. An empty 200 went through escalate → betterCandidates, which kept a
// strict `Quality >` predicate and then RE-SORTED what survived it. Same request,
// same plan.candidates, two different rankers depending on how the worker failed.
//
// Two consequences, both live: if the matrix routed to the worker holding the
// fleet's highest bank score, betterCandidates returned nothing and escalation
// never fired at all — the client got the empty answer even where the matrix had
// confident evidence that another worker gets this topic right. And a worker the
// matrix had ranked LAST, because it predicts WRONG here, could be picked as the
// repair purely on its score against someone else's question bank.

// orderedFleet stands up two workers whose PLAN order deliberately contradicts
// the retired quality scalar: the matrix put "primary" first and "rescue"
// second, while "rescue" carries the lower bank score. Any ranker that consults
// Quality therefore disagrees with the plan, which is what makes the two failure
// modes comparable.
func orderedFleet(t *testing.T, primary, rescue http.HandlerFunc) (*Router, *dispatch, *httptest.ResponseRecorder, *http.Request) {
	t.Helper()
	a := httptest.NewServer(primary)
	t.Cleanup(a.Close)
	b := httptest.NewServer(rescue)
	t.Cleanup(b.Close)

	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "primary", URL: a.URL, Model: "m", Quality: 90,
		BaselineTPS: 100, MaxConcurrency: 2, TTLSeconds: 3600, Features: []string{"chat"}})
	reg.upsert(BackendRegistration{ID: "rescue", URL: b.URL, Model: "m", Quality: 30,
		BaselineTPS: 100, MaxConcurrency: 2, TTLSeconds: 3600, Features: []string{"chat"}})

	r := &Router{
		cfg: &Config{EscalateInline: true, BackendIdleTimeout: 5 * time.Second}, registry: reg,
		client: &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
	}
	backend := reg.get("primary")
	slot := make(chan struct{}, 1)
	escalated := false
	d := &dispatch{
		backend: &backend, slot: &slot,
		body: []byte(`{"messages":[]}`), raw: []byte(`{"messages":[]}`),
		plan: &routePlan{
			auto:       true,
			route:      "route:outcome:p=0.90,n=8",
			candidates: []*Backend{reg.get("primary"), reg.get("rescue")},
		},
		job: nominalJob(), log: &RequestLog{BackendID: "primary"},
		output: &strings.Builder{}, escalated: &escalated,
	}
	return r, d, httptest.NewRecorder(),
		post("/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`, "")
}

// The two failure modes must differ only in POLICY — what counts as a failure,
// how many hops are allowed, whether the deadline permits it — never in which
// worker is next.
func TestEmptyAnswerAndFailoverPickTheSameCandidate(t *testing.T) {
	empty := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(emptyAnswer))
	}
	down := func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"Loading model"}`))
	}
	answers := func(w http.ResponseWriter, _ *http.Request) { jsonAnswer(w, "the capital is Paris") }

	// An empty 200 from the plan's first choice.
	rEmpty, dEmpty, recEmpty, reqEmpty := orderedFleet(t, empty, answers)
	rEmpty.dispatchBuffered(recEmpty, reqEmpty, dEmpty)
	viaEscalation := recEmpty.Header().Get("X-LLM-Backend-ID")

	// A 5xx from the plan's first choice. Same plan, same order, same request.
	rDown, dDown, recDown, reqDown := orderedFleet(t, down, answers)
	rDown.dispatchBuffered(recDown, reqDown, dDown)
	viaFailover := recDown.Header().Get("X-LLM-Backend-ID")

	if viaFailover != "rescue" {
		t.Fatalf("test premise wrong: 5xx failover served from %q, want rescue", viaFailover)
	}
	if viaEscalation != viaFailover {
		t.Fatalf("the same plan produced %q for an empty answer and %q for a 5xx — the empty-answer "+
			"path is ranking candidates for itself instead of walking the plan", viaEscalation, viaFailover)
	}
	if got := recEmpty.Header().Get("X-LLM-Escalated"); got != "primary->rescue" {
		t.Errorf("X-LLM-Escalated = %q, want primary->rescue", got)
	}
	if !strings.Contains(recEmpty.Body.String(), "Paris") {
		t.Errorf("client kept the empty answer: %s", recEmpty.Body.String())
	}
}

// The plan's ORDER is the answer, not a starting point to re-sort. A candidate
// list the matrix built for this prompt, in this thinking mode, must reach both
// movers unchanged.
func TestEscalationWalksThePlanOrder(t *testing.T) {
	plan := &routePlan{candidates: []*Backend{
		{BackendRegistration: BackendRegistration{ID: "matrix-first", Quality: 10, BaselineTPS: 5}},
		{BackendRegistration: BackendRegistration{ID: "matrix-second", Quality: 99, BaselineTPS: 100}},
	}}
	// Everything a ranker could reach for — bank score, throughput — points the
	// other way, so an order-preserving filter is the only thing that yields this.
	got := nextCandidates(plan, map[string]bool{})
	if len(got) != 2 || got[0].ID != "matrix-first" {
		t.Fatalf("plan order not preserved: %v", ids(got))
	}
	from := plan.candidates[0]
	next := nextCandidates(plan, map[string]bool{from.ID: true})
	if len(next) != 1 || next[0].ID != "matrix-second" {
		t.Fatalf("after the plan's first choice failed, next = %v, want [matrix-second]", ids(next))
	}
}

// cannedWorker is a chat backend that returns a canned body.
func cannedWorker(t *testing.T, body string, hits *atomic.Int64) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if hits != nil {
			hits.Add(1)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func escalationRouter(t *testing.T, cheapBody, goodBody string, cheapHits, goodHits *atomic.Int64) (*Router, *tierAdapter) {
	t.Helper()
	cheap := cannedWorker(t, cheapBody, cheapHits)
	good := cannedWorker(t, goodBody, goodHits)

	reg := newTestRegistry()
	for _, w := range []struct {
		id      string
		url     string
		quality int
	}{{"cheap", cheap.URL, 30}, {"good", good.URL, 90}} {
		reg.upsert(BackendRegistration{
			ID: w.id, URL: w.url, Model: "default", Quality: w.quality,
			BaselineTPS: 100, MaxConcurrency: 2, TTLSeconds: 3600, Features: []string{"chat"},
		})
		reg.finishCertification(w.id, true, map[string]Check{}, 100, 10, "")
	}

	cfg := &Config{
		DefaultMaxTokens: 4096, AutoDifficulty: true, EscalateInline: true,
		DifficultyBands: defaultDifficultyBands, DifficultyTemp: 0.10,
		DifficultyTimeout: time.Second, DifficultyCacheSize: 16, DifficultyMaxChars: 4000,
		AdaptBins: 10, AdaptMaxBias: 0.3, AdaptLRUp: 0.04, AdaptLRDown: 0.01,
	}
	dir := t.TempDir()
	adapter := newTierAdapter(cfg, dir+"/tier_adapter.json")
	logs, err := openLogStore(dir+"/logs.sqlite", 16384, "")
	if err != nil {
		t.Fatalf("open log store: %v", err)
	}
	t.Cleanup(func() { logs.Close() })
	r := &Router{
		cfg: cfg, registry: reg, classifier: testClassifier(fakeEmbed), adapter: adapter,
		client: &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
		logs: logs, sessions: newSessionTracker(time.Hour, 16),
	}
	return r, adapter
}

const emptyAnswer = `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[]},"finish_reason":"stop"}]}`
const realAnswer = `{"choices":[{"message":{"role":"assistant","content":"the capital is Paris"},"finish_reason":"stop"}]}`

func TestEscalationRepairsAnEmptyAnswer(t *testing.T) {
	var cheapHits, goodHits atomic.Int64
	r, adapter := escalationRouter(t, emptyAnswer, realAnswer, &cheapHits, &goodHits)

	rec := runChat(t, r, `{"model":"default","stream":false,"messages":[{"role":"user","content":"say hello"}]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "Paris") {
		t.Fatalf("client got the empty answer instead of the repaired one: %s", rec.Body.String())
	}
	if cheapHits.Load() != 1 || goodHits.Load() != 1 {
		t.Fatalf("want exactly one attempt each, got cheap=%d good=%d", cheapHits.Load(), goodHits.Load())
	}
	if got := rec.Header().Get("X-LLM-Escalated"); got != "cheap->good" {
		t.Fatalf("escalation not reported: %q", got)
	}
	if got := rec.Header().Get("X-LLM-Backend-ID"); got != "good" {
		t.Fatalf("route headers should name the worker that actually answered, got %q", got)
	}
	// The repair must not teach the adapter that the cheap tier was fine.
	waitFor(t, func() bool {
		for _, b := range adapter.snapshot() {
			if b > 0 {
				return true
			}
		}
		return false
	}, "adapter never learned the region needed a better model")
}

// A truncated answer is the caller's own token ceiling; a bigger model hits the
// same wall and bills twice for it.
func TestEscalationSkipsTruncatedAnswers(t *testing.T) {
	const truncated = `{"choices":[{"message":{"role":"assistant","content":"the capital is"},"finish_reason":"length"}]}`
	var cheapHits, goodHits atomic.Int64
	r, _ := escalationRouter(t, truncated, realAnswer, &cheapHits, &goodHits)

	rec := runChat(t, r, `{"model":"default","stream":false,"messages":[{"role":"user","content":"say hello"}]}`)
	if goodHits.Load() != 0 {
		t.Fatalf("truncation must not escalate, but the better worker was called %d times", goodHits.Load())
	}
	if !strings.Contains(rec.Body.String(), "the capital is") {
		t.Fatalf("original answer should be returned unchanged: %s", rec.Body.String())
	}
}

// If the better worker is also empty, the caller gets the original response and
// the escalation never loops.
func TestEscalationDoesNotLoop(t *testing.T) {
	var cheapHits, goodHits atomic.Int64
	r, _ := escalationRouter(t, emptyAnswer, emptyAnswer, &cheapHits, &goodHits)

	rec := runChat(t, r, `{"model":"default","stream":false,"messages":[{"role":"user","content":"say hello"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d", rec.Code)
	}
	if cheapHits.Load() != 1 || goodHits.Load() != 1 {
		t.Fatalf("exactly one hop allowed: cheap=%d good=%d", cheapHits.Load(), goodHits.Load())
	}
	if rec.Header().Get("X-LLM-Escalated") != "" {
		t.Fatal("a failed escalation must not be reported as one")
	}
	// The slot and the active-request count must come back to the worker that
	// actually served, not be stranded on the escalation target.
	for _, id := range []string{"cheap", "good"} {
		if b := r.registry.get(id); b.ActiveRequests != 0 {
			t.Fatalf("%s left with active_requests=%d after a failed escalation", id, b.ActiveRequests)
		}
	}
}

func TestEscalationDisabled(t *testing.T) {
	var cheapHits, goodHits atomic.Int64
	r, _ := escalationRouter(t, emptyAnswer, realAnswer, &cheapHits, &goodHits)
	r.cfg.EscalateInline = false

	runChat(t, r, `{"model":"default","stream":false,"messages":[{"role":"user","content":"say hello"}]}`)
	if goodHits.Load() != 0 {
		t.Fatalf("escalation is off; better worker must not be called (%d times)", goodHits.Load())
	}
}

// A pinned worker is an explicit client choice — answering from a different model
// would be a worse surprise than the empty reply.
func TestEscalationSkipsPinnedRequests(t *testing.T) {
	var cheapHits, goodHits atomic.Int64
	r, _ := escalationRouter(t, emptyAnswer, realAnswer, &cheapHits, &goodHits)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"default","stream":false,"messages":[{"role":"user","content":"say hello"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-LLM-Backend-ID", "cheap")
	rec := httptest.NewRecorder()
	r.handleChatCompletions(rec, req)

	if goodHits.Load() != 0 {
		t.Fatalf("a pinned request must never be escalated (better worker called %d times)", goodHits.Load())
	}
}

// An open tool loop is the one place a second opinion cannot be asked for. The
// better model never emitted the tool call this turn is answering, so it is
// handed an orphan tool_call_id and usually refuses the request outright — the
// caller pays for a second generation, on a paid endpoint, and still gets the
// empty answer. Acquisition already spends sessionLockWait keeping the loop on
// one worker; escalating would undo that in one hop.
func TestEscalationSkipsAnOpenToolLoop(t *testing.T) {
	var cheapHits, goodHits atomic.Int64
	r, _ := escalationRouter(t, emptyAnswer, realAnswer, &cheapHits, &goodHits)

	// The loop's earlier turns were served by "cheap", which is what makes the
	// session locked rather than merely sticky.
	key, _ := sessionKeyFor(convo(sys("agent"), usr("say hello")))
	r.sessions.remember(key, "cheap")

	rec := runChat(t, r, `{"model":"default","stream":false,"messages":[`+
		`{"role":"system","content":"agent"},{"role":"user","content":"say hello"},`+
		`{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"ls"}}]},`+
		`{"role":"tool","tool_call_id":"c1","content":"a.txt"}]}`)

	if goodHits.Load() != 0 {
		t.Fatalf("a mid-tool-loop turn must not be escalated (better worker called %d times)", goodHits.Load())
	}
	if cheapHits.Load() != 1 {
		t.Fatalf("the incumbent should have served exactly once, got %d", cheapHits.Load())
	}
	if got := rec.Header().Get("X-LLM-Escalated"); got != "" {
		t.Fatalf("escalation happened inside a tool loop: %q", got)
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("the original answer should still be returned, got %d", rec.Code)
	}
}

func runChat(t *testing.T, r *Router, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	r.handleChatCompletions(rec, req)
	return rec
}

// waitFor polls cond, which is satisfied by the request's background
// bookkeeping goroutine (adapter/judge/log insert).
func waitFor(t *testing.T, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal(msg)
}

// A caller that declared a deadline has already spent part of it on the failed
// generation. Escalating past that budget turns a bad answer into no answer.
func TestEscalationRespectsADeclaredDeadline(t *testing.T) {
	var cheapHits, goodHits atomic.Int64
	r, _ := escalationRouter(t, emptyAnswer, realAnswer, &cheapHits, &goodHits)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"default","stream":false,"messages":[{"role":"user","content":"say hello"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-LLM-Deadline-MS", "1") // already gone by the time we'd escalate
	rec := httptest.NewRecorder()
	r.handleChatCompletions(rec, req)

	if goodHits.Load() != 0 {
		t.Fatalf("escalation must not overrun a declared deadline (better worker called %d times)", goodHits.Load())
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("the original answer should still be returned, got %d", rec.Code)
	}

	// With room to spare, the same request does escalate.
	cheapHits.Store(0)
	goodHits.Store(0)
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"default","stream":false,"messages":[{"role":"user","content":"say hello"}]}`))
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("X-LLM-Deadline-MS", "600000")
	rec2 := httptest.NewRecorder()
	r.handleChatCompletions(rec2, req2)
	if goodHits.Load() != 1 {
		t.Fatalf("a generous deadline should still allow the repair, better worker called %d times", goodHits.Load())
	}
}

// ── Unknown-parameter backstop ──────────────────────────────────────────────

// rejectedField is what decides whether a 400 is safe to retry around, so it is
// graded directly against the shapes servers actually emit.
//
// The request carries a thinking gate and a reasoning level the ROUTER injected
// (neither is in the client body) plus a temperature and a response_format the
// CLIENT sent — which is the distinction the whole backstop turns on.
func TestRejectedField(t *testing.T) {
	const request = `{"model":"m","messages":[],"chat_template_kwargs":{"enable_thinking":true},` +
		`"reasoning_effort":"medium","temperature":0.7,"response_format":{"type":"json_schema"}}`
	const client = `{"model":"m","messages":[],"temperature":0.7,"response_format":{"type":"json_schema"}}`
	cases := []struct {
		name   string
		status int
		body   string
		want   string
	}{
		{"OpenAI prose", http.StatusBadRequest,
			`{"error":{"message":"Unrecognized request argument supplied: chat_template_kwargs","type":"invalid_request_error","param":null,"code":null}}`,
			"chat_template_kwargs"},
		{"OpenAI param, when the endpoint fills it in", http.StatusBadRequest,
			`{"error":{"message":"Unknown parameter.","param":"chat_template_kwargs","type":"invalid_request_error"}}`,
			"chat_template_kwargs"},
		{"pydantic detail[].loc, which is what vLLM validation looks like", http.StatusUnprocessableEntity,
			`{"detail":[{"type":"extra_forbidden","loc":["body","chat_template_kwargs"],"msg":"Extra inputs are not permitted"}]}`,
			"chat_template_kwargs"},
		{"gateway phrasing", http.StatusBadRequest,
			`{"error":{"message":"Unknown field: chat_template_kwargs"}}`, "chat_template_kwargs"},

		// Everything below must NOT be retried.
		{"an invalid VALUE is not an unknown field — stripping it silently changes the request", http.StatusBadRequest,
			`{"error":{"message":"temperature must be <= 2","param":"temperature"}}`, ""},
		{"a field the request does not carry identifies nothing", http.StatusBadRequest,
			`{"error":{"message":"Unrecognized request argument supplied: logit_bias"}}`, ""},
		{"a field that carries the caller's meaning is never dropped", http.StatusBadRequest,
			`{"error":{"message":"Unknown model: m","param":"model"}}`, ""},
		// The defect this list was inverted for: the endpoint names a field the
		// CLIENT sent, and the router has no business removing it however plainly
		// the rejection is worded. Both spellings of the rejection, because a
		// machine-readable param bypasses the prose scan entirely.
		{"a client's response_format is not ours to drop, however the reject is worded", http.StatusBadRequest,
			`{"error":{"message":"Unrecognized request argument supplied: response_format"}}`, ""},
		{"…including when the endpoint names it in error.param", http.StatusBadRequest,
			`{"error":{"message":"Unknown parameter.","param":"response_format","type":"invalid_request_error"}}`, ""},
		{"…and a client's temperature likewise", http.StatusBadRequest,
			`{"error":{"message":"Unrecognized request argument supplied: temperature"}}`, ""},
		{"two candidates is ambiguous, and picking one would be a guess", http.StatusBadRequest,
			`{"error":{"message":"unknown fields: chat_template_kwargs, reasoning_effort"}}`, ""},
		{"a 500 is the endpoint's problem, not the request's", http.StatusInternalServerError,
			`{"error":{"message":"Unrecognized request argument supplied: chat_template_kwargs"}}`, ""},
		{"an unparseable error body", http.StatusBadRequest, `<html>unknown chat_template_kwargs</html>`, "chat_template_kwargs"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := rejectedField(tc.status, []byte(tc.body), []byte(request), []byte(client))
			if tc.want == "" {
				if ok {
					t.Fatalf("retried on %q", got)
				}
				return
			}
			if !ok || got != tc.want {
				t.Fatalf("field = %q (ok=%v), want %q", got, ok, tc.want)
			}
		})
	}
}

// strictWorker is an endpoint that refuses any request carrying `field` — what a
// provider validating its input does — recording each body it saw.
func strictWorker(t *testing.T, field string, seen *[]string) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		body, _ := io.ReadAll(req.Body)
		*seen = append(*seen, string(body))
		w.Header().Set("Content-Type", "application/json")
		if bytes.Contains(body, []byte(`"`+field+`"`)) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = fmt.Fprintf(w, `{"error":{"message":"Unrecognized request argument supplied: %s","type":"invalid_request_error"}}`, field)
			return
		}
		_, _ = io.WriteString(w, realAnswer)
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

// routerFor is a router with one ready worker at url and a real log store.
func routerFor(t *testing.T, url string) *Router {
	t.Helper()
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{
		ID: "strict", URL: url, Model: "m", Quality: 50,
		BaselineTPS: 100, MaxConcurrency: 2, TTLSeconds: 3600, Features: []string{"chat"},
	})
	reg.finishCertification("strict", true, map[string]Check{}, 100, 10, "")

	dir := t.TempDir()
	logs, err := openLogStore(dir+"/logs.sqlite", 16384, "")
	if err != nil {
		t.Fatalf("open log store: %v", err)
	}
	t.Cleanup(func() { logs.Close() })
	return &Router{
		cfg:      &Config{DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second},
		registry: reg, client: &http.Client{Timeout: 5 * time.Second},
		streamClient: &http.Client{}, logs: logs,
	}
}

// TestStripAndRetry: an endpoint that refuses a field the ROUTER injected gets
// the request again without it, once, and — the retry having worked — the
// verdict sticks so later requests never pay for it again.
func TestStripAndRetry(t *testing.T) {
	var seen []string
	r := routerFor(t, strictWorker(t, "chat_template_kwargs", &seen))
	// requirements is a router-only field: it is stripped on the way south, and
	// what reaches the endpoint in its place is the chat_template_kwargs gate the
	// router wrote from it. The client asked for no such field, which is what
	// makes it the router's to withdraw.
	body := `{"model":"m","messages":[{"role":"user","content":"say hello"}],` +
		`"requirements":{"thinking":"off"}}`

	rec := runChat(t, r, body)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want the retry to have succeeded: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "Paris") {
		t.Fatalf("client did not get the retried answer: %s", rec.Body.String())
	}
	if len(seen) != 2 {
		t.Fatalf("want one rejection and one retry, got %d requests", len(seen))
	}
	if !strings.Contains(seen[0], "chat_template_kwargs") {
		t.Errorf("first attempt should have carried the field: %s", seen[0])
	}
	if strings.Contains(seen[1], "chat_template_kwargs") {
		t.Errorf("retry still carried the refused field: %s", seen[1])
	}
	// Remembered against the backend, and visible in /backends.
	if got := r.registry.get("strict").RejectedFields; len(got) != 1 || got[0] != "chat_template_kwargs" {
		t.Fatalf("rejected field not remembered: %v", got)
	}

	// The next request omits it up front — one worker hit, no rejection.
	seen = nil
	if rec := runChat(t, r, body); rec.Code != http.StatusOK {
		t.Fatalf("second request = %d: %s", rec.Code, rec.Body.String())
	}
	if len(seen) != 1 {
		t.Fatalf("want a single clean request, got %d", len(seen))
	}
	if strings.Contains(seen[0], "chat_template_kwargs") {
		t.Errorf("a learned field was sent again: %s", seen[0])
	}

	// A re-registration is a different deployment, so what the old one refused
	// says nothing about it.
	r.registry.upsert(BackendRegistration{
		ID: "strict", URL: "http://elsewhere", Model: "m", TTLSeconds: 3600, Features: []string{"chat"},
	})
	if got := r.registry.get("strict").RejectedFields; len(got) != 0 {
		t.Errorf("re-registration kept the old deployment's verdicts: %v", got)
	}
}

// One retry, never a loop: an endpoint that refuses every request gets exactly
// two, and the caller gets the rejection rather than an unbounded hunt for a
// body it will accept.
func TestStripAndRetryIsBounded(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		body, _ := io.ReadAll(req.Body)
		seen = append(seen, string(body))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		// Names whichever field the body still carries, so a looping
		// implementation would keep going until nothing was left.
		field := "chat_template_kwargs"
		if !bytes.Contains(body, []byte(field)) {
			field = "top_k"
		}
		_, _ = fmt.Fprintf(w, `{"error":{"message":"Unrecognized request argument supplied: %s"}}`, field)
	}))
	defer srv.Close()

	r := routerFor(t, srv.URL)

	rec := runChat(t, r, `{"model":"m","messages":[{"role":"user","content":"hi"}],`+
		`"requirements":{"thinking":"off"},"top_k":40}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want the endpoint's own 400 passed through", rec.Code)
	}
	if len(seen) != 2 {
		t.Fatalf("want exactly one retry, got %d requests", len(seen))
	}
}

// A streaming request is never retried: bytes are already on the wire and there
// is no rewinding them (the same rule proxyRetryDelays records).
func TestStripAndRetrySkipsStreaming(t *testing.T) {
	var seen []string
	r := routerFor(t, strictWorker(t, "chat_template_kwargs", &seen))
	runChat(t, r, `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}],`+
		`"requirements":{"thinking":"off"}}`)
	if len(seen) != 1 {
		t.Fatalf("a streamed request must be forwarded once, got %d", len(seen))
	}
}

// ── Defect: the backstop must never negotiate away the CALLER's request ─────

// A field the client sent is not the router's to withdraw. The realistic case is
// a json_schema response_format on a model that only does json_object: the
// endpoint says 400, and stripping the field turns a visible rejection into
// free-form prose returned with a 200 — for this request, and for every later
// one to that backend, streamed ones included, where none of this code runs.
func TestStripAndRetryNeverDropsAClientField(t *testing.T) {
	var seen []string
	r := routerFor(t, strictWorker(t, "response_format", &seen))
	body := `{"model":"m","messages":[{"role":"user","content":"say hello"}],` +
		`"response_format":{"type":"json_schema","json_schema":{"name":"r","schema":{}}}}`

	rec := runChat(t, r, body)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want the endpoint's 400 handed straight to the caller: %s", rec.Code, rec.Body.String())
	}
	if len(seen) != 1 {
		t.Fatalf("the request must be forwarded once and not retried without the field, got %d requests: %v", len(seen), seen)
	}
	if !strings.Contains(seen[0], "response_format") {
		t.Errorf("the client's field never reached the endpoint: %s", seen[0])
	}
	if got := r.registry.get("strict").RejectedFields; len(got) != 0 {
		t.Fatalf("a client's field was blacklisted for every later request: %v", got)
	}
	// And the next request still carries it, rather than being quietly answered
	// as free-form prose.
	seen = nil
	runChat(t, r, body)
	if len(seen) != 1 || !strings.Contains(seen[0], "response_format") {
		t.Fatalf("a later request lost the client's response_format: %v", seen)
	}
}

// The verdict is evidence, not a hypothesis: a retry that is refused just as
// firmly proves nothing, and recording it anyway strips the field from every
// later request on the strength of a guess.
func TestStripAndRetryLearnsOnlyFromASuccessfulRetry(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		body, _ := io.ReadAll(req.Body)
		seen = append(seen, string(body))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		// The first rejection names the router's injected field; the retry, with
		// that field gone, is refused for an unrelated reason.
		if bytes.Contains(body, []byte("chat_template_kwargs")) {
			_, _ = io.WriteString(w, `{"error":{"message":"Unrecognized request argument supplied: chat_template_kwargs"}}`)
			return
		}
		_, _ = io.WriteString(w, `{"error":{"message":"this model is not available on your plan"}}`)
	}))
	defer srv.Close()

	r := routerFor(t, srv.URL)
	rec := runChat(t, r, `{"model":"m","messages":[{"role":"user","content":"hi"}],`+
		`"requirements":{"thinking":"off"}}`)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want a 400", rec.Code)
	}
	if len(seen) != 2 {
		t.Fatalf("want the one bounded retry, got %d requests", len(seen))
	}
	if got := r.registry.get("strict").RejectedFields; len(got) != 0 {
		t.Fatalf("learned %v from a retry that was refused as well", got)
	}
}

// A learned verdict is about an endpoint as it was, and endpoints get
// redeployed. Past the TTL the field goes out again to find out.
func TestRejectedFieldVerdictIsRetested(t *testing.T) {
	var seen []string
	r := routerFor(t, strictWorker(t, "chat_template_kwargs", &seen))
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"requirements":{"thinking":"off"}}`

	if rec := runChat(t, r, body); rec.Code != http.StatusOK {
		t.Fatalf("first request = %d: %s", rec.Code, rec.Body.String())
	}
	if got := r.registry.get("strict").RejectedFields; len(got) != 1 {
		t.Fatalf("nothing was learned: %v", got)
	}
	// Within the TTL the verdict stands: one clean request, no rejection.
	seen = nil
	runChat(t, r, body)
	if len(seen) != 1 {
		t.Fatalf("a fresh verdict should cost one request, got %d", len(seen))
	}

	old := rejectedFieldTTL
	rejectedFieldTTL = time.Nanosecond
	defer func() { rejectedFieldTTL = old }()

	seen = nil
	if rec := runChat(t, r, body); rec.Code != http.StatusOK {
		t.Fatalf("re-tested request = %d: %s", rec.Code, rec.Body.String())
	}
	if len(seen) != 2 || !strings.Contains(seen[0], "chat_template_kwargs") {
		t.Fatalf("an aged-out verdict must be re-tested by sending the field again, got %v", seen)
	}
	// Still refused, so it is learned again — with a fresh clock.
	if got := r.registry.get("strict").RejectedFields; len(got) != 1 || got[0] != "chat_template_kwargs" {
		t.Fatalf("a re-test that failed should keep the verdict: %v", got)
	}
}
