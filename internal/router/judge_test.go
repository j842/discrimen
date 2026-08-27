package router

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestJudgeGraderPrefersFree: the judge grades a sampled fraction of ordinary
// traffic forever, so the moment the model doing the grading is a metered one
// that becomes a standing spend. It must reach for a FREE grader first and fall
// through to a paid one only when the served worker is the only free chat worker
// registered.
//
// The middle case below used to expect the paid model, on the rule "nothing free
// OUTRANKS the worker being graded". That rule is gone (see judgeGrader), and it
// had to go in both directions at once: keeping it while making the fleet's best
// worker gradeable would have put every sample of the busiest worker in the
// fleet on someone's invoice, since by definition nothing free outranks it.
func TestJudgeGraderPrefersFree(t *testing.T) {
	reg := newTestRegistry()
	registerPriced(t, reg, "cheapie", 3, 4, 0, 0)
	registerPriced(t, reg, "free-mid", 7, 4, 0, 0)
	registerPriced(t, reg, "paid-top", 9, 4, 3, 15)
	r := &Router{registry: reg}

	// The strongest free worker that is not the one being graded.
	if got, paid := r.judgeGrader(reg.get("cheapie")); got == nil || got.ID != "free-mid" || paid {
		t.Errorf("grading a cheap answer: got %v (paid=%v), want free-mid for free", got, paid)
	}
	// Including when the served worker IS the best free one: the runner-up grades
	// it, for nothing, rather than the fleet paying to mark its own homework.
	if got, paid := r.judgeGrader(reg.get("free-mid")); got == nil || got.ID != "cheapie" || paid {
		t.Errorf("grading the best FREE worker: got %v (paid=%v), want cheapie for free", got, paid)
	}
	// And the paid worker is graded free too, which is the ordinary case.
	if got, paid := r.judgeGrader(reg.get("paid-top")); got == nil || got.ID != "free-mid" || paid {
		t.Errorf("grading the paid worker: got %v (paid=%v), want free-mid for free", got, paid)
	}
	// A paid grader only when there is no other free chat worker at all.
	reg.remove("cheapie")
	if got, paid := r.judgeGrader(reg.get("free-mid")); got == nil || got.ID != "paid-top" || !paid {
		t.Errorf("sole free worker: got %v (paid=%v), want paid-top flagged paid", got, paid)
	}
	// With nothing else registered there is no second opinion to be had, and the
	// answer is nil rather than the worker itself — handing back the served worker
	// is what made maybeJudge's guard look like a sampling decision when it was
	// really "this worker is never judged".
	reg.remove("paid-top")
	if got, paid := r.judgeGrader(reg.get("free-mid")); got != nil || paid {
		t.Errorf("one-worker fleet: got %v (paid=%v), want no grader", got, paid)
	}
	// An embeddings-only worker is never a grader, whatever its quality.
	registerPriced(t, reg, "cheapie", 3, 4, 0, 0)
	reg.upsert(BackendRegistration{ID: "emb", URL: "http://emb", Model: "e", Quality: 100,
		TTLSeconds: 3600, Features: []string{"embeddings"}})
	reg.finishCertification("emb", true, map[string]Check{}, 0, 0, "")
	if got, _ := r.judgeGrader(reg.get("free-mid")); got == nil || got.ID != "cheapie" {
		t.Errorf("embeddings worker used as a grader: %v", got)
	}
}

// ── Defect: the judge could never grade the fleet's best worker ─────────────

// judgeGrader ranked by the retired quality scalar and maybeJudge then refused
// any grader that did not OUTRANK the worker being graded. That test is
// unsatisfiable for the fleet's own best worker: nothing outranks it, so it was
// handed itself and the sample was dropped — permanently, in both thinking
// modes.
//
// recordJudgedOutcome is the only route by which production traffic reaches the
// outcome matrix (the bank is LiveBench maths; production is agent loops and
// chat), so that worker ended up with no evidence about real traffic at all. On
// a real prompt it therefore had no prediction, landed in the matrix's
// `unmeasured` band, and ranked BEHIND any cheap worker with a known rate at or
// above outcomeCorrectFloor. A blind spot on exactly one worker — the busiest —
// chosen by a number routing no longer consults.
func TestJudgeGradesTheFleetsBestWorker(t *testing.T) {
	reg := newTestRegistry()
	registerPriced(t, reg, "small", 30, 4, 0, 0)
	registerPriced(t, reg, "mid", 60, 4, 0, 0)
	top := registerPriced(t, reg, "flagship", 95, 4, 0, 0)
	r := &Router{registry: reg}

	got, paid := r.judgeGrader(top)
	if got == nil {
		t.Fatal("no grader for the fleet's best worker: its traffic is never judged, so the " +
			"matrix learns nothing about the worker serving most of it")
	}
	if got.ID == top.ID {
		t.Fatalf("the best worker was handed ITSELF as a grader (%s); maybeJudge drops every one of its samples", got.ID)
	}
	// The runner-up, not merely "someone". Grading is verification rather than
	// generation, so it does not need to outrank the answerer — but among the
	// workers that can do it, the strongest is still the right one to ask.
	if got.ID != "mid" {
		t.Errorf("grader = %s, want the runner-up (mid)", got.ID)
	}
	if paid {
		t.Error("a free runner-up was available; grading the best worker must not reach for a metered model")
	}
}

// End to end, because the unit above only proves a grader was NAMED: a sampled
// answer from the fleet's best worker is actually graded, and the verdict lands
// in the outcome matrix as an observation against that worker. This is the loop
// the quality gate held open.
func TestJudgeRecordsAnOutcomeForTheBestWorker(t *testing.T) {
	var graded atomic.Int64
	judge := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		graded.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"BAD"}}],"usage":{"total_tokens":40}}`)
	}))
	t.Cleanup(judge.Close)
	embed := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":[{"embedding":[1,0,0,0],"index":0}]}`)
	}))
	t.Cleanup(embed.Close)

	reg := newTestRegistry()
	// The flagship is both the fleet's best worker and the one that served, which
	// is the combination that used to be unjudgeable.
	reg.upsert(BackendRegistration{ID: "flagship", URL: "http://flagship", Model: "m", Quality: 95,
		MaxConcurrency: 4, TTLSeconds: 3600, Features: []string{"chat"}})
	reg.finishCertification("flagship", true, map[string]Check{}, 50, 10, "")
	reg.upsert(BackendRegistration{ID: "runner-up", URL: judge.URL, Model: "m", Quality: 60,
		MaxConcurrency: 4, TTLSeconds: 3600, Features: []string{"chat"}})
	reg.finishCertification("runner-up", true, map[string]Check{}, 50, 10, "")
	reg.upsert(BackendRegistration{ID: "emb", URL: embed.URL, Model: "e",
		MaxConcurrency: 4, TTLSeconds: 3600, Features: []string{"embeddings"}})
	reg.finishCertification("emb", true, map[string]Check{}, 0, 0, "")

	r := &Router{
		cfg: &Config{JudgeSampleRate: 1}, registry: reg,
		client:   &http.Client{Timeout: 5 * time.Second},
		judgeSem: make(chan struct{}, judgeMaxConcurrent),
		outcomes: newOutcomeMatrix(nil),
	}

	r.maybeJudge([]Message{{Role: "user", Content: "what is the capital of France?"}}, false,
		reg.get("flagship"), "route:outcome:p=0.90,n=8",
		`{"choices":[{"message":{"content":"Berlin."}}]}`, false, 1200)

	waitFor(t, func() bool {
		for _, id := range r.outcomes.backendsWithEvidence() {
			if id == "flagship" {
				return true
			}
		}
		return false
	}, "the fleet's best worker accumulated no judged observation — recordJudgedOutcome is the only "+
		"evidence the matrix will ever hold about the traffic actually being served")

	if n := graded.Load(); n != 1 {
		t.Errorf("judge calls = %d, want exactly one", n)
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
