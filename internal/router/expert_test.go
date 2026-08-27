package router

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ── Fixture ─────────────────────────────────────────────────────────────────

// panelWorker is one fake model in the fleet. It answers with `answer` unless it
// is handed a synthesis prompt, which it answers with `synthesis` — the same
// worker is routinely both a member and the synthesiser, and the two calls have
// to be told apart to assert what each was sent.
type panelWorker struct {
	id        string
	model     string
	quality   int
	contextK  int
	slots     int
	answer    string // raw completion body; "" ⇒ use answerBody(id)
	synthesis string
	// status non-2xx ⇒ the worker refuses everything. A 400 rather than a 500,
	// which is a refusal either way and skips the 5xx retry ladder these tests
	// have no reason to sit through.
	status int
	price  float64

	hits   atomic.Int64
	synths atomic.Int64
	mu     sync.Mutex
	bodies []string
}

// seen returns the request bodies this worker was sent, in order.
func (p *panelWorker) seen() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.bodies...)
}

// lastSynthesis is the body of the synthesis call this worker was sent.
func (p *panelWorker) lastSynthesis(t *testing.T) string {
	t.Helper()
	for _, b := range p.seen() {
		if strings.Contains(b, "Candidate answers:") {
			return b
		}
	}
	t.Fatalf("worker %s was never asked to synthesise (saw %d requests)", p.id, len(p.seen()))
	return ""
}

// answerBody is an ordinary completion carrying `text`, with a usage block —
// every endpoint that reports one is what makes the summed usage assertable.
func answerBody(text string, prompt, completion int) string {
	return fmt.Sprintf(`{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],`+
		`"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}}`, text, prompt, completion, prompt+completion)
}

// expertFleet stands up one httptest endpoint per worker and a router over them.
func expertFleet(t *testing.T, workers ...*panelWorker) *Router {
	t.Helper()
	reg := newTestRegistry()
	for _, w := range workers {
		worker := w
		if worker.answer == "" {
			worker.answer = answerBody("answer from "+worker.id, 10, 5)
		}
		if worker.synthesis == "" {
			worker.synthesis = answerBody("the synthesised answer", 100, 20)
		}
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			body, _ := io.ReadAll(req.Body)
			worker.mu.Lock()
			worker.bodies = append(worker.bodies, string(body))
			worker.mu.Unlock()
			worker.hits.Add(1)
			w.Header().Set("Content-Type", "application/json")
			if worker.status != 0 && worker.status != http.StatusOK {
				w.WriteHeader(worker.status)
				_, _ = io.WriteString(w, `{"error":{"message":"no"}}`)
				return
			}
			reply := worker.answer
			if strings.Contains(string(body), "Candidate answers:") {
				worker.synths.Add(1)
				reply = worker.synthesis
			}
			if strings.Contains(string(body), `"stream":true`) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, sseOf(reply))
				return
			}
			_, _ = io.WriteString(w, reply)
		}))
		t.Cleanup(srv.Close)
		reg.upsert(BackendRegistration{
			ID: worker.id, URL: srv.URL, Model: worker.model, Quality: worker.quality,
			ContextK: worker.contextK, BaselineTPS: 100, MaxConcurrency: worker.slots,
			TTLSeconds: 3600, Features: []string{"chat", "tools"}, Thinking: true,
			Provider: "local", Source: sourceBeacon,
			InputPricePerMtok: worker.price, OutputPricePerMtok: worker.price,
		})
		reg.finishCertification(worker.id, true, map[string]Check{}, 100, 10, "")
	}
	return &Router{
		cfg: &Config{DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second, LogMaxBodyBytes: 16384,
			BackendIdleTimeout: 10 * time.Second},
		registry: reg, logs: newTestLogStore(t),
		client: &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
	}
}

// sseOf renders a buffered completion as the stream a stream:true request gets,
// content and reasoning split across deltas the way a real worker sends them.
func sseOf(completion string) string {
	var raw map[string]any
	if json.Unmarshal([]byte(completion), &raw) != nil {
		return "data: [DONE]\n\n"
	}
	content, reasoning, _ := completionText(raw)
	var sb strings.Builder
	if reasoning != "" {
		fmt.Fprintf(&sb, "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":%q}}]}\n\n", reasoning)
	}
	fmt.Fprintf(&sb, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":%q}}]}\n\n", content)
	if u, ok := raw["usage"].(map[string]any); ok {
		payload, _ := json.Marshal(map[string]any{"choices": []any{}, "usage": u})
		fmt.Fprintf(&sb, "data: %s\n\n", payload)
	}
	sb.WriteString("data: [DONE]\n\n")
	return sb.String()
}

// askExpert sends a chat request naming the ensemble.
func askExpert(t *testing.T, r *Router, extra string) *httptest.ResponseRecorder {
	t.Helper()
	body := `{"model":"expert","messages":[{"role":"user","content":"what is the capital of France?"}]` + extra + `}`
	return runChat(t, r, body)
}

// threeModels is the ordinary fleet these tests route over: three distinct
// models, the last of them the best measured worker and therefore the
// synthesiser.
func threeModels() (*panelWorker, *panelWorker, *panelWorker) {
	return &panelWorker{id: "w-alpha", model: "alpha", quality: 40},
		&panelWorker{id: "w-beta", model: "beta", quality: 55},
		&panelWorker{id: "w-gamma", model: "gamma", quality: 90}
}

// ── The happy path ──────────────────────────────────────────────────────────

// TestExpertAsksEveryModelAndSynthesises is the whole feature in one request:
// every model answers, the best worker writes the reply, and the client sees
// only that reply.
func TestExpertAsksEveryModelAndSynthesises(t *testing.T) {
	alpha, beta, gamma := threeModels()
	r := expertFleet(t, alpha, beta, gamma)

	rec := askExpert(t, r, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	// Every model, and every one of them asked the SAME question.
	for _, w := range []*panelWorker{alpha, beta, gamma} {
		if w.hits.Load() == 0 {
			t.Fatalf("%s was never asked — the panel is not every model", w.id)
		}
		if !strings.Contains(w.seen()[0], "the capital of France") {
			t.Errorf("%s was asked something else: %s", w.id, w.seen()[0])
		}
	}
	if gamma.synths.Load() != 1 {
		t.Fatalf("the best worker synthesised %d times, want exactly 1", gamma.synths.Load())
	}
	if !strings.Contains(rec.Body.String(), "the synthesised answer") {
		t.Fatalf("client did not get the synthesis: %s", rec.Body.String())
	}
	// None of the panel's working reaches the client.
	for _, leak := range []string{"answer from w-alpha", "Candidate", "candidate"} {
		if strings.Contains(rec.Body.String(), leak) {
			t.Errorf("the panel leaked %q into the client's answer: %s", leak, rec.Body.String())
		}
	}

	// The synthesiser was given every answer, unattributed.
	prompt := gamma.lastSynthesis(t)
	for _, w := range []*panelWorker{alpha, beta, gamma} {
		if !strings.Contains(prompt, "answer from "+w.id) {
			t.Errorf("%s's answer is missing from the synthesis prompt", w.id)
		}
	}
	for _, name := range []string{"alpha", "beta", "gamma"} {
		if strings.Contains(prompt, `Candidate 1 ---\n`+name) {
			t.Errorf("the synthesis prompt attributes an answer to %q", name)
		}
	}

	// Observability: the route, what happened, and who wrote the answer.
	if got := rec.Header().Get("X-LLM-Route"); got != routeExpert {
		t.Errorf("X-LLM-Route = %q, want %q", got, routeExpert)
	}
	if got := rec.Header().Get("X-LLM-Expert"); got != "members=3,skipped=0,synth=w-gamma" {
		t.Errorf("X-LLM-Expert = %q", got)
	}
	if got := rec.Header().Get("X-LLM-Backend-Model"); got != "gamma" {
		t.Errorf("X-LLM-Backend-Model = %q, want the synthesiser's model", got)
	}
}

// TestExpertTakesOneWorkerPerModel: two workers serving the same model are one
// opinion, not two, and the best-ranked of them speaks for it.
func TestExpertTakesOneWorkerPerModel(t *testing.T) {
	slow := &panelWorker{id: "twin-slow", model: "beta", quality: 55}
	fast := &panelWorker{id: "twin-fast", model: "beta", quality: 55}
	other := &panelWorker{id: "w-gamma", model: "gamma", quality: 90}
	r := expertFleet(t, slow, fast, other)
	// Same model and the same measured quality, so what separates the two is the
	// completion-time ranking — which is the question the panel is asking.
	r.registry.finishCertification("twin-slow", true, map[string]Check{}, 5, 2000, "")

	members := expertMembers(r.registry.eligible(), nominalJob())
	if len(members) != 2 {
		t.Fatalf("panel = %v, want one member per distinct model", ids(members))
	}
	seen := map[string]bool{}
	for _, m := range members {
		if seen[m.Model] {
			t.Fatalf("panel asked model %q twice: %v", m.Model, ids(members))
		}
		seen[m.Model] = true
		if m.Model == "beta" && m.ID != "twin-fast" {
			t.Errorf("model beta is spoken for by %q, want the better-ranked worker", m.ID)
		}
	}

	if rec := askExpert(t, r, ""); rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if slow.hits.Load() != 0 {
		t.Errorf("the second worker for the same model was asked as well (%d times)", slow.hits.Load())
	}
}

// ── Tolerating failures ─────────────────────────────────────────────────────

// TestExpertDropsAFailedMember: a member that refuses is not an error, it is one
// fewer opinion.
func TestExpertDropsAFailedMember(t *testing.T) {
	alpha, beta, gamma := threeModels()
	beta.status = http.StatusBadRequest
	r := expertFleet(t, alpha, beta, gamma)

	rec := askExpert(t, r, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("one broken member must not fail the request, got %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("X-LLM-Expert"); got != "members=2,skipped=1,synth=w-gamma" {
		t.Errorf("X-LLM-Expert = %q, want the failure counted as a skip", got)
	}
	if prompt := gamma.lastSynthesis(t); strings.Contains(prompt, "w-beta") {
		t.Error("the failed member reached the synthesis prompt")
	}
}

// TestExpertSkipsASaturatedMember: a busy worker costs its own model's answer
// and nothing else. The panel must not queue behind it.
func TestExpertSkipsASaturatedMember(t *testing.T) {
	withExpertSlotWait(t, 100*time.Millisecond)
	alpha, beta, gamma := threeModels()
	beta.slots = 1
	r := expertFleet(t, alpha, beta, gamma)

	held, ok := r.registry.tryAcquireSlot("w-beta")
	if !ok {
		t.Fatal("could not saturate the member")
	}
	defer r.registry.releaseSlot(held)

	start := time.Now()
	rec := askExpert(t, r, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("the panel waited %s on a saturated member — it must skip, not queue", elapsed)
	}
	if beta.hits.Load() != 0 {
		t.Errorf("the saturated worker was called %d times", beta.hits.Load())
	}
	if got := rec.Header().Get("X-LLM-Expert"); got != "members=2,skipped=1,synth=w-gamma" {
		t.Errorf("X-LLM-Expert = %q, want the saturated member counted as a skip", got)
	}
}

// TestExpertReturnsASingleAnswerUnsynthesised: with one usable answer there is
// nothing to gather, so the answer is returned as it stands rather than paid for
// twice.
func TestExpertReturnsASingleAnswerUnsynthesised(t *testing.T) {
	alpha, beta, gamma := threeModels()
	alpha.status = http.StatusBadRequest
	beta.status = http.StatusBadRequest
	r := expertFleet(t, alpha, beta, gamma)

	rec := askExpert(t, r, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if gamma.synths.Load() != 0 {
		t.Error("a single answer was sent for synthesis — there was nothing to gather")
	}
	if gamma.hits.Load() != 1 {
		t.Errorf("the surviving member was called %d times, want 1", gamma.hits.Load())
	}
	if !strings.Contains(rec.Body.String(), "answer from w-gamma") {
		t.Fatalf("the surviving answer was not returned: %s", rec.Body.String())
	}
	if got := rec.Header().Get("X-LLM-Expert"); got != "members=1,skipped=2,synth=none" {
		t.Errorf("X-LLM-Expert = %q", got)
	}
}

// TestExpertWithNoUsableAnswers: nothing answered, so there is nothing to
// return — the OpenAI envelope, and a Retry-After the client can act on.
func TestExpertWithNoUsableAnswers(t *testing.T) {
	alpha, beta, gamma := threeModels()
	for _, w := range []*panelWorker{alpha, beta, gamma} {
		w.status = http.StatusBadRequest
	}
	r := expertFleet(t, alpha, beta, gamma)

	rec := askExpert(t, r, "")
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Retry-After"); got == "" {
		t.Error("a 503 without a Retry-After tells the client nothing")
	}
	body := errorEnvelopeOf(t, rec)
	if body.Type != "service_unavailable" {
		t.Errorf("error type = %q", body.Type)
	}
	if got := rec.Header().Get("X-LLM-Expert"); got != "members=0,skipped=3,synth=none" {
		t.Errorf("X-LLM-Expert = %q", got)
	}
}

// TestExpertReturnsTheBestAnswerWhenSynthesisFails: N generations are already
// paid for, so a synthesiser that refuses — or that answers with nothing, which
// is the same thing to a client — costs the synthesis and not the answer.
func TestExpertReturnsTheBestAnswerWhenSynthesisFails(t *testing.T) {
	t.Run("an empty synthesis", func(t *testing.T) {
		alpha, beta, gamma := threeModels()
		gamma.synthesis = `{"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"stop"}]}`
		r := expertFleet(t, alpha, beta, gamma)

		rec := askExpert(t, r, "")
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
		}
		if !strings.Contains(rec.Body.String(), "answer from w-gamma") {
			t.Fatalf("an empty synthesis threw away three answers: %s", rec.Body.String())
		}
		if got := rec.Header().Get("X-LLM-Expert"); got != "members=3,skipped=0,synth=none" {
			t.Errorf("X-LLM-Expert = %q", got)
		}
	})

	t.Run("a synthesiser that refuses", func(t *testing.T) {
		alpha, beta, _ := threeModels()
		// The best worker in the fleet refuses everything, so it is both a member
		// that produces nothing and the synthesiser that cannot be used.
		broken := &panelWorker{id: "w-delta", model: "delta", quality: 99, status: http.StatusBadRequest}
		r := expertFleet(t, alpha, beta, broken)

		rec := askExpert(t, r, "")
		if rec.Code != http.StatusOK {
			t.Fatalf("a refusing synthesiser must not lose the answers, got %d: %s", rec.Code, rec.Body.String())
		}
		if !strings.Contains(rec.Body.String(), "answer from w-beta") {
			t.Fatalf("want the best surviving answer, got %s", rec.Body.String())
		}
		if got := rec.Header().Get("X-LLM-Expert"); got != "members=2,skipped=1,synth=none" {
			t.Errorf("X-LLM-Expert = %q", got)
		}
	})
}

// ── Tools ───────────────────────────────────────────────────────────────────

// TestExpertFallsBackForTools: N models produce N incompatible tool calls, and a
// synthesiser inventing one would be worse than any of them. The request is
// routed normally and says so.
func TestExpertFallsBackForTools(t *testing.T) {
	alpha, beta, gamma := threeModels()
	r := expertFleet(t, alpha, beta, gamma)

	rec := askExpert(t, r, `,"tools":[{"type":"function","function":{"name":"ls"}}]`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	total := alpha.hits.Load() + beta.hits.Load() + gamma.hits.Load()
	if total != 1 {
		t.Errorf("a tools request fanned out to %d workers — it must route normally", total)
	}
	if got := rec.Header().Get("X-LLM-Expert"); got != "fallback=tools" {
		t.Errorf("X-LLM-Expert = %q, want the fallback to be visible", got)
	}
	if got := rec.Header().Get("X-LLM-Route"); got == routeExpert {
		t.Error("the fallback still reported itself as an ensemble")
	}
}

// TestExpertFallsBackMidToolLoop: this turn's tool result belongs to the model
// that emitted the matching tool call; handing it to the whole fleet produces a
// fleet of refusals.
func TestExpertFallsBackMidToolLoop(t *testing.T) {
	alpha, beta, gamma := threeModels()
	r := expertFleet(t, alpha, beta, gamma)

	rec := runChat(t, r, `{"model":"expert","messages":[`+
		`{"role":"user","content":"list the files"},`+
		`{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"ls"}}]},`+
		`{"role":"tool","tool_call_id":"c1","content":"a.txt"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("X-LLM-Expert"); got != "fallback=tool-loop" {
		t.Errorf("X-LLM-Expert = %q", got)
	}
	if total := alpha.hits.Load() + beta.hits.Load() + gamma.hits.Load(); total != 1 {
		t.Errorf("a mid-loop turn reached %d workers", total)
	}
}

// TestExpertIsChatOnly: /v1/completions has no conversation to put to a panel
// and no synthesis shape to answer in, so it routes normally.
func TestExpertIsChatOnly(t *testing.T) {
	alpha, beta, gamma := threeModels()
	r := expertFleet(t, alpha, beta, gamma)

	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/v1/completions", `{"model":"expert","prompt":"hi"}`, ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if total := alpha.hits.Load() + beta.hits.Load() + gamma.hits.Load(); total != 1 {
		t.Errorf("/v1/completions fanned out to %d workers", total)
	}
	if got := rec.Header().Get("X-LLM-Route"); got != routeCompletions {
		t.Errorf("X-LLM-Route = %q, want %q", got, routeCompletions)
	}
}

// ── Streaming ───────────────────────────────────────────────────────────────

// TestExpertStreamsTheSynthesis: the fan-out has to be buffered, but the answer
// the client reads is generated live — so a streaming client gets a real stream.
func TestExpertStreamsTheSynthesis(t *testing.T) {
	alpha, beta, gamma := threeModels()
	r := expertFleet(t, alpha, beta, gamma)

	rec := askExpert(t, r, `,"stream":true`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want an event stream", ct)
	}
	stream := rec.Body.String()
	if !strings.Contains(stream, "the synthesised answer") || !strings.Contains(stream, "data: [DONE]") {
		t.Fatalf("not a usable stream: %s", stream)
	}
	// The members were asked buffered, whatever the client wanted.
	for _, w := range []*panelWorker{alpha, beta} {
		for _, body := range w.seen() {
			if strings.Contains(body, `"stream":true`) {
				t.Errorf("%s was asked to stream: %s", w.id, body)
			}
		}
	}
	if got := rec.Header().Get("X-LLM-Expert"); got != "members=3,skipped=0,synth=w-gamma" {
		t.Errorf("X-LLM-Expert = %q", got)
	}
}

// TestExpertStreamsASingleAnswer: with nothing to synthesise there is no live
// generation left to relay, but a stream:true client still has to be handed a
// stream — a JSON body would break its SDK outright.
func TestExpertStreamsASingleAnswer(t *testing.T) {
	alpha, beta, gamma := threeModels()
	alpha.status = http.StatusBadRequest
	beta.status = http.StatusBadRequest
	r := expertFleet(t, alpha, beta, gamma)

	rec := askExpert(t, r, `,"stream":true`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want an event stream", ct)
	}
	stream := rec.Body.String()
	if !strings.Contains(stream, "answer from w-gamma") || !strings.HasSuffix(stream, "data: [DONE]\n\n") {
		t.Fatalf("not a usable stream: %s", stream)
	}
}

// ── The working is stripped ─────────────────────────────────────────────────

// TestExpertFeedsTheSynthesiserContentOnly: a member's reasoning trace is its
// working, not its answer, and a synthesiser reading five traces is grading
// process instead of substance.
func TestExpertFeedsTheSynthesiserContentOnly(t *testing.T) {
	alpha, beta, gamma := threeModels()
	alpha.answer = `{"choices":[{"message":{"role":"assistant","content":"Paris is the capital.",` +
		`"reasoning_content":"SECRET WORKING: let me think about France"},"finish_reason":"stop"}]}`
	// The other dialect, and the case where the parser left the answer in the
	// reasoning field because the model never closed its thinking block.
	beta.answer = `{"choices":[{"message":{"role":"assistant","content":null,` +
		`"reasoning":"the answer is Paris"},"finish_reason":"stop"}]}`
	r := expertFleet(t, alpha, beta, gamma)

	rec := askExpert(t, r, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	prompt := gamma.lastSynthesis(t)
	if strings.Contains(prompt, "SECRET WORKING") {
		t.Error("a member's reasoning trace reached the synthesiser")
	}
	if !strings.Contains(prompt, "Paris is the capital.") {
		t.Error("the member's content is missing from the synthesis prompt")
	}
	// Content empty and reasoning holding the whole answer is the one case where
	// the trace IS the answer; dropping it would drop the member.
	if !strings.Contains(prompt, "the answer is Paris") {
		t.Error("an answer that arrived only as reasoning was dropped")
	}
}

// TestExpertSynthesisRunsWithThinkingOff: the synthesiser is choosing between
// finished answers, and its own reasoning block would be one more thing to strip.
func TestExpertSynthesisRunsWithThinkingOff(t *testing.T) {
	alpha, beta, gamma := threeModels()
	r := expertFleet(t, alpha, beta, gamma)

	// The client asked for thinking; the synthesis call still has it off.
	rec := askExpert(t, r, `,"reasoning_effort":"high"`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	prompt := gamma.lastSynthesis(t)
	if !strings.Contains(prompt, `"enable_thinking":false`) {
		t.Errorf("the synthesis call did not force thinking off: %s", truncate(prompt, 400))
	}
	if strings.Contains(prompt, `"reasoning_effort"`) {
		t.Errorf("the client's reasoning_effort survived into the synthesis call: %s", truncate(prompt, 400))
	}
}

// TestExpertStripsReasoningFromTheReply: whatever the synthesiser emits, the
// client asked one question and gets one answer.
func TestExpertStripsReasoningFromTheReply(t *testing.T) {
	alpha, beta, gamma := threeModels()
	gamma.synthesis = `{"choices":[{"message":{"role":"assistant","content":"Paris.",` +
		`"reasoning_content":"SYNTH WORKING"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	r := expertFleet(t, alpha, beta, gamma)

	rec := askExpert(t, r, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "SYNTH WORKING") {
		t.Fatalf("the synthesiser's reasoning reached the client: %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "Paris.") {
		t.Fatalf("the answer was lost with the reasoning: %s", rec.Body.String())
	}

	// And on the streamed path, where the trace arrives as deltas.
	rec = askExpert(t, r, `,"stream":true`)
	if strings.Contains(rec.Body.String(), "SYNTH WORKING") {
		t.Fatalf("the synthesiser's reasoning was streamed to the client: %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "Paris.") {
		t.Fatalf("the streamed answer was lost: %s", rec.Body.String())
	}
}

// ── Accounting ──────────────────────────────────────────────────────────────

// TestExpertSumsUsage: the client is billed for what an ensemble actually
// spends, which is every member plus the synthesis and not the synthesis alone.
func TestExpertSumsUsage(t *testing.T) {
	alpha, beta, gamma := threeModels()
	for _, w := range []*panelWorker{alpha, beta, gamma} {
		w.answer = answerBody("answer from "+w.id, 10, 5)
	}
	gamma.synthesis = answerBody("the synthesised answer", 70, 30)
	r := expertFleet(t, alpha, beta, gamma)

	rec := askExpert(t, r, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	var out struct {
		Usage struct {
			Prompt     int `json:"prompt_tokens"`
			Completion int `json:"completion_tokens"`
			Total      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("reply is not JSON: %v", err)
	}
	// Three members at 10+5, plus the synthesis at 70+30.
	if out.Usage.Prompt != 100 || out.Usage.Completion != 45 || out.Usage.Total != 145 {
		t.Errorf("usage = %+v, want the sum over three members and the synthesis", out.Usage)
	}
}

// TestExpertChargesTheKeyForTheWholeEnsemble: a budgeted key must see the real
// cost of the route it chose, or the budget is not one.
func TestExpertChargesTheKeyForTheWholeEnsemble(t *testing.T) {
	alpha, beta, gamma := threeModels()
	gamma.synthesis = answerBody("the synthesised answer", 70, 30)
	r := expertFleet(t, alpha, beta, gamma)
	key := issueKey(t, r, clientSecret, apiKey{Role: roleClient, Name: "budgeted", TokenBudget: 1_000_000})

	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/v1/chat/completions",
		`{"model":"expert","messages":[{"role":"user","content":"hi"}]}`, clientSecret))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if got := waitForTokens(t, r, clientSecret, 145); got != 145 {
		t.Errorf("charged %d tokens to key %d, want 145 (three members plus the synthesis)", got, key.ID)
	}
}

// TestExpertRestrictsThePanelToTheKeysModels: an ensemble reaches every model at
// once, so a key issued for one worker must not reach the fleet by naming a
// route. A key allowed one model degenerates to a panel of one, which is right.
func TestExpertRestrictsThePanelToTheKeysModels(t *testing.T) {
	alpha, beta, gamma := threeModels()
	r := expertFleet(t, alpha, beta, gamma)
	issueKey(t, r, clientSecret, apiKey{Role: roleClient, Name: "narrow", Models: []string{"expert", "beta"}})

	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/v1/chat/completions",
		`{"model":"expert","messages":[{"role":"user","content":"hi"}]}`, clientSecret))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if alpha.hits.Load() != 0 || gamma.hits.Load() != 0 {
		t.Errorf("the panel reached models off the allow-list (alpha=%d gamma=%d)", alpha.hits.Load(), gamma.hits.Load())
	}
	if got := rec.Header().Get("X-LLM-Expert"); got != "members=1,skipped=0,synth=none" {
		t.Errorf("X-LLM-Expert = %q, want a panel of one", got)
	}
	if !strings.Contains(rec.Body.String(), "answer from w-beta") {
		t.Fatalf("the one allowed model did not answer: %s", rec.Body.String())
	}
}

// ── The router must not learn from an ensemble ──────────────────────────────

// An ensemble is not a routing decision the router can learn from: N models'
// work says nothing about where ONE request should have gone. The judge is the
// only learner left, and it is gated on plan.auto — which an expert plan never
// sets, so the ensemble is excluded structurally rather than by the shape of its
// route string.
//
// It used to be excluded because "expert" does not parse as "route:d=", which
// held only while every learner sniffed that prefix. The tier adapter that read
// it has since been removed and the judge moved to plan.auto; this pins the
// property that outlived both.
func TestAnEnsembleIsNotARouterChoice(t *testing.T) {
	alpha, beta, gamma := threeModels()
	r := expertFleet(t, alpha, beta, gamma)

	req := &ChatRequest{Model: expertModel, Messages: []Message{{Role: "user", Content: "say hello"}}}
	plan, err := r.planRoute(req, 0, false)
	if err != nil {
		t.Fatalf("planRoute: %v", err)
	}
	if plan.route != routeExpert {
		t.Fatalf("test premise wrong: route = %q, want %q", plan.route, routeExpert)
	}
	if plan.auto {
		t.Error("an expert plan is marked auto, so the judge would record an ensemble's " +
			"answer as evidence about the one worker that happened to synthesise it")
	}
}

// ── The reserved name ───────────────────────────────────────────────────────

// TestExpertIsPublishedAsAModel: a client cannot select a route it cannot see,
// and an SDK checks a model exists before using it.
func TestExpertIsPublishedAsAModel(t *testing.T) {
	alpha, beta, gamma := threeModels()
	r := expertFleet(t, alpha, beta, gamma)

	var entry map[string]any
	for _, m := range r.modelCatalogue() {
		if m["id"] == expertModel {
			entry = m
		}
	}
	if entry == nil {
		t.Fatalf("the ensemble is not on the menu: %v", r.modelCatalogue())
	}
	if entry["owned_by"] != routerOwner {
		t.Errorf("owned_by = %v, want %q — the ensemble is the router's", entry["owned_by"], routerOwner)
	}
	if entry["expert"] != true {
		t.Errorf("the menu row does not say what it is: %v", entry)
	}
	// And the single-model fetch resolves it, however it is capitalised.
	for _, name := range []string{"expert", "EXPERT"} {
		rec := httptest.NewRecorder()
		r.routes().ServeHTTP(rec, authGet("/v1/models/"+name, ""))
		if rec.Code != http.StatusOK {
			t.Fatalf("GET /v1/models/%s = %d: %s", name, rec.Code, rec.Body.String())
		}
		var one map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &one); err != nil || one["id"] != expertModel {
			t.Errorf("GET /v1/models/%s returned %v (err=%v)", name, one, err)
		}
	}
}

// TestExpertNameCannotBeTakenOver: a name the router owns is not available to a
// group an operator creates or to a worker that registers under it.
func TestExpertNameCannotBeTakenOver(t *testing.T) {
	r := groupRouter(t)
	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, adminReq(http.MethodPost, "/admin/groups", `{"name":"Expert","members":["qwen3"]}`))
	if rec.Code != http.StatusConflict {
		t.Fatalf("creating a group called expert = %d, want 409: %s", rec.Code, rec.Body.String())
	}

	// A worker that registers a model called "expert" does not capture the name:
	// the ensemble resolves ahead of it, and the menu says so.
	impostor := &panelWorker{id: "w-impostor", model: expertModel, quality: 99}
	other := &panelWorker{id: "w-beta", model: "beta", quality: 50}
	fleet := expertFleet(t, impostor, other)
	rows := 0
	for _, m := range fleet.modelCatalogue() {
		if m["id"] == expertModel {
			rows++
			if m["owned_by"] != routerOwner {
				t.Errorf("the menu points expert at a worker: %v", m)
			}
		}
	}
	if rows != 1 {
		t.Errorf("expert appears %d times on the menu", rows)
	}
	rec = askExpert(t, fleet, "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if other.hits.Load() == 0 {
		t.Error("the impostor captured the name — the rest of the fleet was never asked")
	}
	// The worker stays reachable by its own id.
	if rec := runChat(t, fleet, `{"model":"w-impostor","messages":[{"role":"user","content":"hi"}]}`); rec.Code != http.StatusOK {
		t.Errorf("the shadowed worker became unreachable: %d", rec.Code)
	}
}

// TestExpertAliasIsNeverShadowed: an alias is a server-side reduction of a model
// id, so a worker could reduce to "expert" without ever registering the name.
func TestExpertAliasIsNeverShadowed(t *testing.T) {
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", Model: "expert-8B-Q4_K_M"}}
	if a := backendAlias(b); a != "" {
		t.Errorf("backendAlias = %q, want an alias that does not shadow the router's own name", a)
	}
}

// ── /v1/route-preview ───────────────────────────────────────────────────────

// TestRoutePreviewDescribesAnExpertRoute: the preview exists so a decision can
// be inspected without paying for it, and this is the one route where the cost
// of finding out is N generations.
func TestRoutePreviewDescribesAnExpertRoute(t *testing.T) {
	alpha, beta, gamma := threeModels()
	r := expertFleet(t, alpha, beta, gamma)
	issueKey(t, r, adminSecret, apiKey{Role: roleAdmin, Name: "admin"})
	issueKey(t, r, clientSecret, apiKey{Role: roleClient, Name: "client"})

	preview := func(token, body string) previewResponse {
		t.Helper()
		rec := httptest.NewRecorder()
		r.routes().ServeHTTP(rec, post("/v1/route-preview", body, token))
		if rec.Code != http.StatusOK {
			t.Fatalf("preview = %d: %s", rec.Code, rec.Body.String())
		}
		var out previewResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("preview is not JSON: %v", err)
		}
		return out
	}
	const ask = `{"model":"expert","messages":[{"role":"user","content":"hi"}]}`

	got := preview(clientSecret, ask)
	if got.Route != routeExpert {
		t.Errorf("route = %q, want %q", got.Route, routeExpert)
	}
	if got.Expert == nil || got.Expert.Members != 3 {
		t.Errorf("expert section = %+v, want three members", got.Expert)
	}
	// The client half must not become an inventory of the fleet.
	if len(got.Candidates) != 0 || got.WouldServe != "" {
		t.Errorf("the client view names workers: candidates=%v would_serve=%q", got.Candidates, got.WouldServe)
	}

	full := preview(adminSecret, ask)
	if len(full.Candidates) != 3 {
		t.Errorf("the admin view shows %d members, want the panel", len(full.Candidates))
	}

	// A request the ensemble declines says so rather than looking ordinary.
	fell := preview(clientSecret, `{"model":"expert","tools":[{"type":"function","function":{"name":"ls"}}],`+
		`"messages":[{"role":"user","content":"hi"}]}`)
	if fell.Expert == nil || fell.Expert.Fallback != "tools" {
		t.Errorf("fallback preview = %+v", fell.Expert)
	}
	if fell.Route == routeExpert {
		t.Error("a declined ensemble previewed as one")
	}

	// No ensemble asked for ⇒ no expert section at all.
	if plain := preview(clientSecret, `{"model":"beta","messages":[{"role":"user","content":"hi"}]}`); plain.Expert != nil {
		t.Errorf("a plain request reported an ensemble: %+v", plain.Expert)
	}
}

// ── Units ───────────────────────────────────────────────────────────────────

// TestPickSynthesiser is the rule that decides who writes the answer: the
// highest MEASURED quality worker that can still hold the prompt.
func TestPickSynthesiser(t *testing.T) {
	small := &Backend{BackendRegistration: BackendRegistration{ID: "small", Quality: 99, ContextK: 4}}
	big := &Backend{BackendRegistration: BackendRegistration{ID: "big", Quality: 70, ContextK: 128}}
	unknown := &Backend{BackendRegistration: BackendRegistration{ID: "unknown", Quality: 60}}
	panel := []*Backend{small, big, unknown}

	if got := pickSynthesiser(panel, 2, nominalJob()); got == nil || got.ID != "small" {
		t.Errorf("a prompt everything fits goes to the best model, got %v", got)
	}
	if got := pickSynthesiser(panel, 64, nominalJob()); got == nil || got.ID != "big" {
		t.Errorf("a worker that cannot hold the prompt cannot synthesise it, got %v", got)
	}
	// An undeclared context is not evidence of a small one — same rule as
	// admitReason.
	if got := pickSynthesiser([]*Backend{small, unknown}, 64, nominalJob()); got == nil || got.ID != "unknown" {
		t.Errorf("an undeclared context was read as too small, got %v", got)
	}
	if got := pickSynthesiser([]*Backend{small}, 64, nominalJob()); got != nil {
		t.Errorf("nothing fits, so there is no synthesiser, got %v", got.ID)
	}
}

// TestExpertSynthesisPromptSaysWhatItMustSay. The prompt IS the feature, so the
// instructions it cannot do without are pinned here: a synthesiser that mentions
// the panel, prefaces the answer, or averages two contradictory claims has
// produced something worse than the best member's reply.
func TestExpertSynthesisPromptSaysWhatItMustSay(t *testing.T) {
	prompt := expertSynthesisSystem([]expertAnswer{
		{backend: &Backend{BackendRegistration: BackendRegistration{ID: "a", Model: "alpha"}}, text: "first answer"},
		{backend: &Backend{BackendRegistration: BackendRegistration{ID: "b", Model: "beta"}}, text: "second answer"},
	})
	for _, must := range []string{
		"unattributed", // judge answers, not names
		"their order carries no meaning",
		"single best answer",        // one answer, not a comparison
		"answer correctly yourself", // not limited to picking the least bad
		"Never split the difference",
		"Do not mention these candidates",
		"Do not preface",
	} {
		if !strings.Contains(prompt, must) {
			t.Errorf("the synthesis prompt no longer says %q", must)
		}
	}
	// The answers are there, and the models that wrote them are not.
	if !strings.Contains(prompt, "first answer") || !strings.Contains(prompt, "second answer") {
		t.Error("the candidate answers are missing from the prompt")
	}
	for _, leak := range []string{"alpha", "beta"} {
		if strings.Contains(prompt, leak) {
			t.Errorf("the prompt attributes an answer to %q", leak)
		}
	}
}

func TestExpertFallbackReasons(t *testing.T) {
	cases := []struct {
		name string
		req  *ChatRequest
		want string
	}{
		{"an ordinary chat", &ChatRequest{Messages: []Message{usr("hi")}}, ""},
		{"tools", &ChatRequest{Messages: []Message{usr("hi")}, Tools: json.RawMessage(`[{"type":"function"}]`)}, "tools"},
		{"a null tools field is not tools", &ChatRequest{Messages: []Message{usr("hi")}, Tools: json.RawMessage(`null`)}, ""},
		{"mid-tool-loop", &ChatRequest{Messages: []Message{usr("hi"), {Role: "tool", ToolCallID: "c1"}}}, "tool-loop"},
		{"no conversation at all", &ChatRequest{}, "non-chat"},
	}
	for _, tc := range cases {
		if got := expertFallback(tc.req); got != tc.want {
			t.Errorf("%s: expertFallback = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestBufferedBodyForcesTheNonStreamedShape(t *testing.T) {
	got := string(bufferedBody([]byte(`{"model":"expert","stream":true,"stream_options":{"include_usage":true},"messages":[]}`)))
	if !strings.Contains(got, `"stream":false`) {
		t.Errorf("stream not forced off: %s", got)
	}
	if strings.Contains(got, "stream_options") {
		t.Errorf("stream_options survived into a non-streamed request: %s", got)
	}
	// A body it cannot parse is returned untouched rather than mangled.
	if got := string(bufferedBody([]byte(`{not json`))); got != `{not json` {
		t.Errorf("an unparseable body was rewritten: %s", got)
	}
}

func TestUsageTallyChargesForSilence(t *testing.T) {
	free := &Backend{BackendRegistration: BackendRegistration{ID: "free"}}
	paid := &Backend{BackendRegistration: BackendRegistration{ID: "paid", Provider: "openai",
		InputPricePerMtok: 1_000_000, OutputPricePerMtok: 2_000_000}}
	job := jobCost{promptTokens: 7, outputTokens: 3}

	var tally usageTally
	tally.add(free, usageCount{prompt: 10, completion: 5, total: 15}, job)
	if tally.total != 15 || tally.charged != 15 || tally.paid {
		t.Fatalf("reported usage mis-tallied: %+v", tally)
	}
	// An endpoint that says nothing is not free: it is charged what routing sized
	// the job at, and the client-visible sum is untouched by the estimate.
	tally.add(paid, usageCount{}, job)
	if tally.total != 15 {
		t.Errorf("an estimate leaked into the reported usage: %+v", tally)
	}
	if tally.charged != 25 {
		t.Errorf("charged = %d, want the estimate added (15 + 7 + 3)", tally.charged)
	}
	if !tally.paid || tally.cost != 7+6 {
		t.Errorf("paid=%v cost=%v, want the declared prices charged", tally.paid, tally.cost)
	}
}

func TestRewriteSSELine(t *testing.T) {
	members := usageCount{prompt: 100, completion: 50, total: 150}

	// An ordinary delta passes through byte for byte.
	const delta = `data: {"choices":[{"delta":{"content":"hi"}}]}`
	if got, _ := rewriteSSELine([]byte(delta), members); string(got) != delta {
		t.Errorf("an ordinary frame was rewritten: %s", got)
	}
	if got, _ := rewriteSSELine([]byte("data: [DONE]"), members); string(got) != "data: [DONE]" {
		t.Errorf("the terminator was rewritten: %s", got)
	}
	// vLLM writes "usage":null on every chunk; there is nothing there to restate,
	// and matching it would decode and re-encode every token of every stream.
	const nullUsage = `data: {"choices":[{"delta":{"content":"hi"}}],"usage":null}`
	if got, used := rewriteSSELine([]byte(nullUsage), members); string(got) != nullUsage || used != (usageCount{}) {
		t.Errorf("a null usage was treated as a usage block: %s", got)
	}

	// A reasoning delta loses the trace.
	got, _ := rewriteSSELine([]byte(`data: {"choices":[{"delta":{"reasoning_content":"working"}}]}`), members)
	if strings.Contains(string(got), "working") {
		t.Errorf("a reasoning delta reached the client: %s", got)
	}

	// A usage frame is restated as the ensemble's total.
	got, used := rewriteSSELine([]byte(`data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`), members)
	if used != (usageCount{prompt: 10, completion: 4, total: 14}) {
		t.Errorf("the synthesiser's own usage was misread: %+v", used)
	}
	var frame struct {
		Usage usageCountJSON `json:"usage"`
	}
	if err := json.Unmarshal(got[len("data: "):], &frame); err != nil {
		t.Fatalf("rewritten frame is not JSON: %v", err)
	}
	if frame.Usage.Total != 164 || frame.Usage.Prompt != 110 || frame.Usage.Completion != 54 {
		t.Errorf("streamed usage = %+v, want the ensemble total", frame.Usage)
	}
}

// usageCountJSON reads a usage block off the wire, for the streamed assertions.
type usageCountJSON struct {
	Prompt     int `json:"prompt_tokens"`
	Completion int `json:"completion_tokens"`
	Total      int `json:"total_tokens"`
}

// withExpertSlotWait shortens the per-member slot grace so the saturation test
// runs in milliseconds rather than seconds.
func withExpertSlotWait(t *testing.T, d time.Duration) {
	t.Helper()
	prev := expertSlotWait
	expertSlotWait = d
	t.Cleanup(func() { expertSlotWait = prev })
}
