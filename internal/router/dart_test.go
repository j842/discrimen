package router

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// The single most dangerous mistake in this file: sampling the drafts greedily.
// At temperature 0 both drafts are the same string by construction, so the gate
// fires on every prompt, serves an unchecked answer, and every metric looks
// excellent. This pins the constant that prevents it.
func TestDraftsMustBeSampled(t *testing.T) {
	if draftTemperature <= 0 {
		t.Fatalf("draftTemperature is %v — at zero the drafts are identical by "+
			"construction, so agreement measures nothing while appearing perfect",
			draftTemperature)
	}
	if draftCount < 2 {
		t.Fatalf("draftCount is %d; agreement needs at least two samples", draftCount)
	}
}

// Multiple choice must be excluded: with a handful of options two independent
// drafts agree by chance often enough to destroy the signal.
func TestLooksMultipleChoice(t *testing.T) {
	mcq := []string{
		"What is the value of x?\n\nA) 1\nB) 2\nC) 3\nD) 4",
		"Pick one:\n(A) north\n(B) south\n(C) east",
		"Question text\nA. alpha\nB. beta\nC. gamma",
	}
	for _, p := range mcq {
		if !looksMultipleChoice(p) {
			t.Errorf("multiple choice not detected:\n%s", p)
		}
	}
	notMCQ := []string{
		"What is 17 times 23?",
		"Summarise this email and tell me if I need to reply.",
		"Write a function that reverses a linked list.",
		// A single lettered line is a list item, not an option set.
		"Steps:\nA) start the server\nthen check the logs",
		// Prose mentioning options must not trip it.
		"Explain why option A is better than the alternatives.",
	}
	for _, p := range notMCQ {
		if looksMultipleChoice(p) {
			t.Errorf("false positive on non-MCQ prompt: %q", p)
		}
	}
}

// Two samples of the same answer differ in formatting, not content. Exact match
// on a normalised form is the cheap path and must not be defeated by casing,
// markdown or a full stop.
func TestNormaliseDraft(t *testing.T) {
	same := []struct{ a, b string }{
		{"42", "42."},
		{"The answer is 42", "the answer is 42"},
		{"**42**", "42"},
		{" 42 \n", "42"},
		{"the  answer   is 42", "the answer is 42"},
	}
	for _, c := range same {
		if normaliseDraft(c.a) != normaliseDraft(c.b) {
			t.Errorf("%q and %q should normalise alike, got %q and %q",
				c.a, c.b, normaliseDraft(c.a), normaliseDraft(c.b))
		}
	}
	if normaliseDraft("42") == normaliseDraft("43") {
		t.Error("different answers normalised to the same string")
	}
}

// The gate must decline rather than run in every case where its evidence would
// be meaningless or its answer unrecallable.
func mcqRequest() *ChatRequest {
	return &ChatRequest{Messages: []Message{{Role: "user",
		Content: "Pick:\nA) one\nB) two\nC) three"}}}
}

func TestDraftGateDeclines(t *testing.T) {
	r := &Router{cfg: &Config{DraftGating: true}}
	b := testBackend("w", 100)
	msg := []Message{{Role: "user", Content: "What is 17 times 23?"}}

	cases := []struct {
		name string
		req  *ChatRequest
		tr   thinkingResolution
	}{
		{"streamed request cannot be recalled", &ChatRequest{Stream: true, Messages: msg},
			thinkingResolution{autoDecided: true}},
		{"caller demanded thinking", &ChatRequest{Messages: msg},
			thinkingResolution{hardThink: true}},
		{"caller demanded no thinking", &ChatRequest{Messages: msg},
			thinkingResolution{patch: true, noThink: true}},
		{"router did not choose the mode", &ChatRequest{Messages: msg},
			thinkingResolution{}},
		{"multiple choice", mcqRequest(), thinkingResolution{autoDecided: true}},
		{"no user text", &ChatRequest{Messages: nil}, thinkingResolution{autoDecided: true}},
	}
	for _, c := range cases {
		if v := r.draftGate(t.Context(), b, c.req, c.tr); v.Ran {
			t.Errorf("%s: gate ran when it should have declined", c.name)
		}
	}
	// ...and declines entirely when disabled, which is the default.
	off := &Router{cfg: &Config{DraftGating: false}}
	if v := off.draftGate(t.Context(), b, &ChatRequest{Messages: msg},
		thinkingResolution{autoDecided: true}); v.Ran {
		t.Error("gate ran while disabled")
	}
}

// An agreed draft must come back in the ordinary chat-completion shape, or a
// gated request would look different to every client.
func TestDraftAnswerBody(t *testing.T) {
	body := string(draftAnswerBody("qwen", "391", 120, 8))
	for _, want := range []string{`"object":"chat.completion"`, `"model":"qwen"`,
		`"role":"assistant"`, `"content":"391"`, `"finish_reason":"stop"`,
		// The drafts' cost must be reported: per-key budgeting reads this block,
		// and omitting it would make every gated request free.
		`"prompt_tokens":120`, `"completion_tokens":8`, `"total_tokens":128`} {
		if !strings.Contains(body, want) {
			t.Errorf("draft response missing %s:\n%s", want, body)
		}
	}
}

// A failed draft is not a disagreement — it is no measurement, and must not be
// reported as evidence that the prompt was hard.
func TestEmptyDraftIsNotDisagreement(t *testing.T) {
	r := &Router{}
	if sim, agreed := r.draftsAgree(t.Context(), []string{"only one"}); agreed || sim != 0 {
		t.Error("a single draft cannot agree with anything")
	}
	// Identical drafts agree without needing an embedder at all, which is what
	// keeps the cheap path cheap.
	if sim, agreed := r.draftsAgree(t.Context(), []string{"42", "42."}); !agreed || sim != 1 {
		t.Errorf("identical answers did not agree (sim=%.2f)", sim)
	}
	// Different drafts with no embedder available must fail closed — erring
	// towards thinking costs latency, erring the other way serves an unchecked
	// answer.
	if _, agreed := r.draftsAgree(t.Context(), []string{"42", "17"}); agreed {
		t.Error("different drafts agreed with no embedder to compare them")
	}
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

// ── the gate wired into the request path ───────────────────────────────────

// draftFleet stands up one upstream whose replies are scripted, so the number of
// generations and the mode each was asked in can both be asserted.
type draftUpstream struct {
	replies  []string      // one per call, last repeats
	thinking []interface{} // enable_thinking seen on each call
	bodies   []string
}

func newDraftRouter(t *testing.T, up *draftUpstream) (*Router, *dispatch, *httptest.ResponseRecorder, *http.Request) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		raw, _ := io.ReadAll(req.Body)
		up.bodies = append(up.bodies, string(raw))
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		var think interface{}
		if kw, ok := parsed["chat_template_kwargs"].(map[string]any); ok {
			think = kw["enable_thinking"]
		}
		up.thinking = append(up.thinking, think)
		reply := up.replies[len(up.replies)-1]
		if len(up.thinking)-1 < len(up.replies) {
			reply = up.replies[len(up.thinking)-1]
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"` + reply +
			`"}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	t.Cleanup(srv.Close)

	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "w", URL: srv.URL, Model: "m", TTLSeconds: 3600, MaxConcurrency: 4})
	r := &Router{cfg: &Config{DraftGating: true, BackendIdleTimeout: 5 * time.Second},
		registry: reg, client: &http.Client{}, streamClient: &http.Client{}, benchClient: &http.Client{}}
	backend := reg.get("w")
	slot := make(chan struct{}, 1)
	escalated := false
	chatReq := &ChatRequest{Messages: []Message{{Role: "user", Content: "What is 17 times 23?"}}}
	d := &dispatch{
		backend: &backend, slot: &slot,
		body: []byte(`{"messages":[]}`), raw: []byte(`{"messages":[]}`),
		plan:    &routePlan{auto: true, route: "route:outcome:p=0.9,n=8", candidates: []*Backend{reg.get("w")}},
		chatReq: chatReq, job: nominalJob(),
		// The classifier guessed THINKING; the gate is about to test that guess.
		tr:     thinkingResolution{patch: true, enable: true, softThink: true, autoDecided: true},
		log:    &RequestLog{BackendID: "w"},
		output: &strings.Builder{}, escalated: &escalated,
	}
	return r, d, httptest.NewRecorder(),
		post("/v1/chat/completions", `{"messages":[{"role":"user","content":"What is 17 times 23?"}]}`, "")
}

// Agreement means the prompt was easy: the draft IS the answer, and no third
// generation happens. That saving is the entire justification for the gate.
func TestDraftGateAgreementServesTheDraft(t *testing.T) {
	up := &draftUpstream{replies: []string{"391", "391."}}
	r, d, rec, req := newDraftRouter(t, up)

	r.dispatchBuffered(rec, req, d)

	if len(up.thinking) != draftCount {
		t.Errorf("upstream called %d times, want %d — agreement must skip the real generation",
			len(up.thinking), draftCount)
	}
	for i, th := range up.thinking {
		if th != false {
			t.Errorf("draft %d asked with enable_thinking=%v, want false", i, th)
		}
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "391") {
		t.Errorf("the draft was not served: %s", rec.Body.String())
	}
	if g := rec.Header().Get("X-LLM-Draft-Gate"); !strings.HasPrefix(g, "agreed") {
		t.Errorf("X-LLM-Draft-Gate = %q, want agreed…", g)
	}
	// The drafts cost real tokens and the caller must be billed for them.
	if !strings.Contains(rec.Body.String(), `"total_tokens":30`) {
		t.Errorf("usage does not report both drafts' cost: %s", rec.Body.String())
	}
}

// Disagreement is evidence the prompt is NOT easy: the request proceeds, and it
// proceeds with thinking ON regardless of which way the classifier leaned.
func TestDraftGateDisagreementForcesThinking(t *testing.T) {
	up := &draftUpstream{replies: []string{"391", "17", "391 exactly"}}
	r, d, rec, req := newDraftRouter(t, up)

	r.dispatchBuffered(rec, req, d)

	if len(up.thinking) != draftCount+1 {
		t.Fatalf("upstream called %d times, want %d (two drafts then the real generation)",
			len(up.thinking), draftCount+1)
	}
	if got := up.thinking[draftCount]; got != true {
		t.Errorf("the real generation asked with enable_thinking=%v, want true — disagreement "+
			"is the evidence that this prompt needs a scratchpad", got)
	}
	if g := rec.Header().Get("X-LLM-Draft-Gate"); !strings.HasPrefix(g, "disagreed") {
		t.Errorf("X-LLM-Draft-Gate = %q, want disagreed…", g)
	}
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "391 exactly") {
		t.Errorf("the real answer was not served: %d %s", rec.Code, rec.Body.String())
	}
}

// Disabled is the default, and must cost exactly one generation.
func TestDraftGateDisabledCostsOneGeneration(t *testing.T) {
	up := &draftUpstream{replies: []string{"391"}}
	r, d, rec, req := newDraftRouter(t, up)
	r.cfg.DraftGating = false

	r.dispatchBuffered(rec, req, d)

	if len(up.thinking) != 1 {
		t.Errorf("upstream called %d times with the gate off, want 1", len(up.thinking))
	}
	if rec.Header().Get("X-LLM-Draft-Gate") != "" {
		t.Error("the gate stamped a header while disabled")
	}
}

// A mode the CALLER specified is an instruction, not a hypothesis. The gate must
// not spend two generations second-guessing it.
func TestDraftGateRespectsAnExplicitMode(t *testing.T) {
	up := &draftUpstream{replies: []string{"391"}}
	r, d, rec, req := newDraftRouter(t, up)
	d.tr = thinkingResolution{patch: true, enable: false, noThink: true} // explicit off

	r.dispatchBuffered(rec, req, d)

	if len(up.thinking) != 1 {
		t.Errorf("upstream called %d times, want 1 — an explicit mode must not be gated", len(up.thinking))
	}
}
