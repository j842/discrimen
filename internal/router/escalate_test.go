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

func TestBetterCandidates(t *testing.T) {
	from := &Backend{BackendRegistration: BackendRegistration{ID: "mid", Quality: 50}}
	worse := &Backend{BackendRegistration: BackendRegistration{ID: "tiny", Quality: 20}}
	equal := &Backend{BackendRegistration: BackendRegistration{ID: "twin", Quality: 50}}
	better := &Backend{BackendRegistration: BackendRegistration{ID: "big", Quality: 90, BaselineTPS: 100}}
	best := &Backend{BackendRegistration: BackendRegistration{ID: "huge", Quality: 99, BaselineTPS: 5}}

	got := betterCandidates([]*Backend{from, worse, equal, better, best}, from, nominalJob())
	if len(got) != 2 {
		t.Fatalf("only strictly-better workers qualify, got %v", ids(got))
	}
	// Among the better ones, soonest-to-finish leads.
	if got[0].ID != "big" {
		t.Fatalf("want the fastest better worker first, got %v", ids(got))
	}
	if n := betterCandidates([]*Backend{from, worse, equal}, from, nominalJob()); n != nil {
		t.Fatalf("nothing better ⇒ no escalation, got %v", ids(n))
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
func TestRejectedField(t *testing.T) {
	const request = `{"model":"m","messages":[],"chat_template_kwargs":{"enable_thinking":true},"temperature":0.7}`
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
		{"two candidates is ambiguous, and picking one would be a guess", http.StatusBadRequest,
			`{"error":{"message":"unknown fields: chat_template_kwargs, temperature"}}`, ""},
		{"a 500 is the endpoint's problem, not the request's", http.StatusInternalServerError,
			`{"error":{"message":"Unrecognized request argument supplied: chat_template_kwargs"}}`, ""},
		{"an unparseable error body", http.StatusBadRequest, `<html>unknown chat_template_kwargs</html>`, "chat_template_kwargs"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := rejectedField(tc.status, []byte(tc.body), []byte(request))
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

// TestStripAndRetry: an endpoint that refuses a field the router sent gets the
// request again without it, once, and the verdict sticks so later requests never
// pay for it again.
func TestStripAndRetry(t *testing.T) {
	var seen []string
	r := routerFor(t, strictWorker(t, "chat_template_kwargs", &seen))
	// The client pins the kwargs gate itself (the escape hatch), so the router
	// forwards it verbatim and the strict endpoint refuses the request.
	body := `{"model":"m","messages":[{"role":"user","content":"say hello"}],` +
		`"chat_template_kwargs":{"enable_thinking":false}}`

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
		`"chat_template_kwargs":{"enable_thinking":false},"top_k":40}`)
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
		`"chat_template_kwargs":{"enable_thinking":false}}`)
	if len(seen) != 1 {
		t.Fatalf("a streamed request must be forwarded once, got %d", len(seen))
	}
}
