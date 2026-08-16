package router

import (
	"strings"
	"testing"
	"time"
)

// TestJudgeGraderPrefersFree: the judge grades a sampled fraction of ordinary
// traffic forever, so the moment the best model in the fleet is a metered one
// that becomes a standing spend. It must reach for the best FREE model first
// and only fall through when none of them outranks the worker being graded.
func TestJudgeGraderPrefersFree(t *testing.T) {
	reg := newTestRegistry()
	registerPriced(t, reg, "cheapie", 3, 4, 0, 0)
	registerPriced(t, reg, "free-mid", 7, 4, 0, 0)
	registerPriced(t, reg, "paid-top", 9, 4, 3, 15)
	r := &Router{registry: reg}

	// A free model outranks the worker that served → grade for nothing.
	if got, paid := r.judgeGrader(reg.get("cheapie")); got == nil || got.ID != "free-mid" || paid {
		t.Errorf("grading a cheap answer: got %v (paid=%v), want free-mid for free", got, paid)
	}
	// Nothing free outranks it → the paid model, flagged so the caller charges.
	if got, paid := r.judgeGrader(reg.get("free-mid")); got == nil || got.ID != "paid-top" || !paid {
		t.Errorf("no free model is better: got %v (paid=%v), want paid-top flagged paid", got, paid)
	}
	// With the paid model gone there is simply nothing better to grade with; the
	// best free model comes back and maybeJudge's own guard drops the sample.
	reg.remove("paid-top")
	if got, paid := r.judgeGrader(reg.get("free-mid")); got == nil || got.ID != "free-mid" || paid {
		t.Errorf("free-only fleet: got %v (paid=%v), want free-mid for free", got, paid)
	}
	// An embeddings-only worker is never a grader, whatever its quality.
	reg.upsert(BackendRegistration{ID: "emb", URL: "http://emb", Model: "e", Quality: 100,
		TTLSeconds: 3600, Features: []string{"embeddings"}})
	reg.finishCertification("emb", true, map[string]Check{}, 0, 0, "")
	if got, _ := r.judgeGrader(reg.get("cheapie")); got == nil || got.ID != "free-mid" {
		t.Errorf("embeddings worker used as a grader: %v", got)
	}
}

// TestJudgePaidBudget: the cap is a rolling token allowance, so an exhausted
// window pauses paid grading and a rolled-over one resumes it.
func TestJudgePaidBudget(t *testing.T) {
	var b judgeBudget
	if !b.allow() {
		t.Fatal("a fresh window must allow the first call")
	}
	b.charge(judgePaidTokenBudget - 1)
	if !b.allow() {
		t.Fatal("one token short of the cap is still inside it")
	}
	b.charge(1)
	if b.allow() {
		t.Fatal("a spent allowance must refuse")
	}
	// Roll the window over by hand; the allowance comes back and the spend resets.
	b.resetAt = time.Now().Add(-time.Second)
	if !b.allow() {
		t.Fatal("a rolled-over window must allow again")
	}
	if b.spent != 0 {
		t.Errorf("spend carried across the window boundary: %d", b.spent)
	}
}

func TestJudgeCallTokens(t *testing.T) {
	cases := []struct {
		name string
		raw  map[string]any
		want int
	}{
		{"the endpoint's own total wins", map[string]any{"usage": map[string]any{
			"total_tokens": 321.0, "prompt_tokens": 300.0, "completion_tokens": 21.0}}, 321},
		{"summed when no total is published", map[string]any{"usage": map[string]any{
			"prompt_tokens": 300.0, "completion_tokens": 21.0}}, 321},
		{"silence is charged the ceiling", map[string]any{}, judgeMaxCallTokens},
		{"an empty usage block is silence too", map[string]any{"usage": map[string]any{}}, judgeMaxCallTokens},
	}
	for _, c := range cases {
		if got := judgeCallTokens(c.raw); got != c.want {
			t.Errorf("%s: judgeCallTokens = %d, want %d", c.name, got, c.want)
		}
	}
}

func TestParseJudgeVerdict(t *testing.T) {
	cases := []struct {
		in      string
		wantBad bool
		wantOK  bool
	}{
		{"GOOD", false, true},
		{"BAD", true, true},
		{"good", false, true},
		{"  bad\n", true, true},
		{"<think>hmm, mostly right</think> GOOD", false, true},
		{"The answer is BAD — it's wrong.", true, true},
		{"meh", false, false},                     // neither word
		{"GOOD parts but also BAD", false, false}, // both → ambiguous, ignored
	}
	for _, c := range cases {
		bad, ok := parseJudgeVerdict(c.in)
		if bad != c.wantBad || ok != c.wantOK {
			t.Errorf("parseJudgeVerdict(%q) = (%v,%v), want (%v,%v)", c.in, bad, ok, c.wantBad, c.wantOK)
		}
	}
}

func TestExtractAnswer(t *testing.T) {
	nonStream := []byte(`{"choices":[{"message":{"content":"The capital is Paris."}}]}`)
	if got := extractAnswer(nonStream, false); got != "The capital is Paris." {
		t.Errorf("non-stream: got %q", got)
	}
	// Streamed: content deltas concatenated, reasoning_content ignored.
	stream := []byte(
		`data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}` + "\n" +
			`data: {"choices":[{"delta":{"content":"4"}}]}` + "\n" +
			`data: {"choices":[{"delta":{"content":"2"}}]}` + "\n" +
			`data: [DONE]` + "\n")
	if got := extractAnswer(stream, true); got != "42" {
		t.Errorf("stream: got %q want 42", got)
	}
}

func TestLastUserText(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "reply"},
		{Role: "user", Content: "second"},
	}
	if got := lastUserText(msgs); got != "second" {
		t.Errorf("lastUserText = %q want second", got)
	}
	mm := []Message{{Role: "user", Content: []any{
		map[string]any{"type": "text", "text": "describe this"},
		map[string]any{"type": "image_url"},
	}}}
	if got := strings.TrimSpace(lastUserText(mm)); got != "describe this" {
		t.Errorf("multimodal lastUserText = %q want 'describe this'", got)
	}
}
