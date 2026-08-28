package router

import (
	"testing"
	"time"
)

// The whole point of the timing columns is that a MISSING measurement is
// distinguishable from a measured one. These pin that, because the failure is
// silent: a RequestLog built by some future route without setting Thinking
// would record "thinking off" as a fact, and the two regimes it pools have
// output lengths that differ by more than an order of magnitude.
func TestRequestLogZeroValueMeansUnrecorded(t *testing.T) {
	var entry RequestLog
	if entry.Thinking != thinkingUnknown {
		t.Errorf("zero-value Thinking is %d, want thinkingUnknown (%d) — a log built without "+
			"setting it must not claim a measurement", entry.Thinking, thinkingUnknown)
	}
	if entry.Concurrency != 0 {
		t.Errorf("zero-value Concurrency is %d, want 0 meaning unrecorded", entry.Concurrency)
	}
	if entry.TTFTMillis != 0 {
		t.Errorf("zero-value TTFTMillis is %d, want 0 meaning unmeasured", entry.TTFTMillis)
	}
	// thinkingOff must NOT be the zero value, or the guarantee above is vacuous.
	if thinkingOff == thinkingUnknown {
		t.Fatal("thinkingOff equals thinkingUnknown, so 'not recorded' and 'thinking was off' are indistinguishable")
	}
}

func TestThinkingLogValue(t *testing.T) {
	cases := []struct {
		name string
		tr   thinkingResolution
		want thinkingMode
	}{
		// noThink is the field selection itself uses to choose between a worker's
		// two quality scores, so the log must agree with the routing decision.
		{"explicit off", thinkingResolution{noThink: true}, thinkingOff},
		{"patched on", thinkingResolution{patch: true, enable: true}, thinkingOn},
		{"hard require", thinkingResolution{hardThink: true}, thinkingOn},
		{"auto prefers thinking", thinkingResolution{softThink: true}, thinkingOn},
		// Nothing decided: the worker's own default applies and we did not observe
		// which it is. Reading `enable` here would wrongly record "off".
		{"undecided", thinkingResolution{}, thinkingUnknown},
		{"enable set but not patched", thinkingResolution{enable: true}, thinkingUnknown},
		// noThink wins over a stale enable: it is the resolved decision.
		{"off beats enable", thinkingResolution{noThink: true, enable: true}, thinkingOff},
	}
	for _, tc := range cases {
		if got := thinkingLogValue(tc.tr); got != tc.want {
			t.Errorf("%s: thinkingLogValue = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// TTFT is only meaningful on a streamed reply. A buffered one is written in a
// single Write at the end, where first-byte and last-byte are the same instant,
// and reporting that would restate the duration as if it were a prefill
// measurement.
func TestReplyMeterTTFTUnmeasuredUntilWritten(t *testing.T) {
	m := &replyMeter{}
	start := time.Now()
	if got := m.ttftMillis(start); got != 0 {
		t.Errorf("ttftMillis before any write = %d, want 0 (unmeasured)", got)
	}
	if _, err := m.Write([]byte("data: {}\n")); err != nil {
		t.Fatal(err)
	}
	if m.first.IsZero() {
		t.Error("first-byte time not recorded on the first Write")
	}
	// An empty write is not a first token and must not start the clock.
	m2 := &replyMeter{}
	if _, err := m2.Write(nil); err != nil {
		t.Fatal(err)
	}
	if !m2.first.IsZero() {
		t.Error("an empty Write started the TTFT clock")
	}
}

// benchSelfGrades exists to keep a question the grader cannot score out of the
// emitted set. For a code-exec question the string probes it uses are not just
// uninformative but actively misleading — checkAnswer against an empty Expect
// returns true for ANY text — so these pin the checks that replaced them.
func TestSelfGradesCodeExec(t *testing.T) {
	tests := []benchCase{{Input: "1", Output: "1", Testtype: "functional"}}
	_ = tests
	base := func() poolQuestion {
		return poolQuestion{
			ID: "q", Task: "coding_completion", Match: benchMatchCodeExec, Prompt: "p",
			Code: &poolCode{Class: "Solution", Func: "f", PublicJSON: `[{"input":"1","output":"1"}]`, Prefix: "class Solution:"},
		}
	}
	if !benchSelfGrades(base()) {
		t.Error("a well-formed completion question was rejected")
	}
	// The bug this guards: a completion question with no prefix is graded as a
	// bare fragment, which never compiles, so the task measures formatting.
	q := base()
	q.Code.Prefix = ""
	if benchSelfGrades(q) {
		t.Error("a coding_completion question with no Prefix was accepted — it would grade fragments standalone")
	}
	// LCB_generation answers are whole programs and need no prefix.
	q = base()
	q.Task, q.Code.Prefix = "LCB_generation", ""
	if !benchSelfGrades(q) {
		t.Error("a generation question was rejected for having no prefix, which it does not need")
	}
	// Nothing to run the answer against.
	q = base()
	q.Code.PublicJSON, q.Code.HasPrivate = "", false
	if benchSelfGrades(q) {
		t.Error("a code question with no test cases was accepted")
	}
	q = base()
	q.Code = nil
	if benchSelfGrades(q) {
		t.Error("a code-exec question with no payload was accepted")
	}
}

// An empty Expect makes checkAnswer return true for every answer, so one
// malformed question would lift the whole fleet's score rather than being
// discarded. The shapes benchSelfGrades probes with are built FROM Expect, so
// they cannot catch this on their own.
func TestSelfGradesRejectsEmptyExpect(t *testing.T) {
	q := poolQuestion{ID: "q", Task: "math_comp", Match: "numeric", Prompt: "p", Expect: ""}
	if benchSelfGrades(q) {
		t.Error("a question with an empty Expect was accepted — it grades every answer correct")
	}
	if !benchSelfGrades(poolQuestion{ID: "q", Task: "math_comp", Match: "numeric", Prompt: "p", Expect: "42"}) {
		t.Error("a well-formed numeric question was rejected")
	}
}

// A truncated profile must still be representative. The question set is authored
// easy-to-hard, so dispatching in index order and stopping early would score a
// worker on easy tiers alone — reporting a weak worker as strong, which is the
// direction that does damage (it draws hard traffic it cannot answer).
func TestStratifiedOrderKeepsPrefixesBalanced(t *testing.T) {
	var qs []benchmarkQ
	for tier := 1; tier <= 4; tier++ {
		for i := 0; i < 10; i++ {
			qs = append(qs, benchmarkQ{Tier: tier})
		}
	}
	order := benchStratifiedOrder(qs)
	if len(order) != len(qs) {
		t.Fatalf("order has %d entries, want %d — questions would be silently dropped", len(order), len(qs))
	}
	seen := map[int]bool{}
	for _, i := range order {
		if seen[i] {
			t.Fatalf("index %d dispatched twice", i)
		}
		seen[i] = true
	}
	// Every tier represented within the first round.
	tiers := map[int]int{}
	for _, i := range order[:4] {
		tiers[qs[i].Tier]++
	}
	if len(tiers) != 4 {
		t.Errorf("first 4 dispatched cover tiers %v, want one of each", tiers)
	}
	// And a half-length prefix is still balanced.
	half := map[int]int{}
	for _, i := range order[:len(order)/2] {
		half[qs[i].Tier]++
	}
	for tier, n := range half {
		if n < 4 || n > 6 {
			t.Errorf("half-length prefix has %d of tier %d, want ~5 — the sample is skewed", n, tier)
		}
	}
}

func TestStratifiedOrderHandlesRaggedTiers(t *testing.T) {
	// Tiers with very different counts must not lose questions.
	qs := []benchmarkQ{{Tier: 1}, {Tier: 2}, {Tier: 2}, {Tier: 2}, {Tier: 9}}
	order := benchStratifiedOrder(qs)
	if len(order) != len(qs) {
		t.Fatalf("ragged tiers: got %d indices, want %d", len(order), len(qs))
	}
	seen := map[int]bool{}
	for _, i := range order {
		seen[i] = true
	}
	if len(seen) != len(qs) {
		t.Errorf("ragged tiers: %d distinct indices, want %d", len(seen), len(qs))
	}
	if len(benchStratifiedOrder(nil)) != 0 {
		t.Error("empty input should produce an empty order, not a hang")
	}
}

// The per-mode split exists because assuming the two modes behave alike is what
// the goal explicitly forbids. These pin that an unmeasured mode falls back
// rather than reporting zero, and that generated LENGTH never pools the modes.
func TestPerModeEstimates(t *testing.T) {
	b := &Backend{ObservedTPS: 100, ObservedTPSThink: 40, ObservedGenThink: 3000, ObservedGenNoThink: 150}
	if got := liveTPSFor(b, thinkingOn); got != 40 {
		t.Errorf("thinking tps = %v, want the thinking-specific 40", got)
	}
	// No no-think measurement yet: fall back to the pooled figure, not to zero.
	if got := liveTPSFor(b, thinkingOff); got != 100 {
		t.Errorf("unmeasured no-think tps = %v, want pooled 100", got)
	}
	if got := liveTPSFor(b, thinkingUnknown); got != 100 {
		t.Errorf("unknown-mode tps = %v, want pooled 100", got)
	}
	// Generated length must NOT fall back to a pooled value — there isn't one,
	// because averaging a 150-token answer with a 3000-token trace describes
	// neither. An unknown mode uses the caller's nominal estimate instead.
	job := jobCost{outputTokens: 256, mode: thinkingUnknown}
	if got := expectedGenTokens(b, job); got != 256 {
		t.Errorf("unknown mode gen = %v, want the nominal 256", got)
	}
	if got := expectedGenTokens(b, jobCost{outputTokens: 256, mode: thinkingOn}); got != 3000 {
		t.Errorf("thinking gen = %v, want measured 3000", got)
	}
	// The caller's ceiling binds even against a measurement.
	if got := expectedGenTokens(b, jobCost{outputTokens: 256, mode: thinkingOn, maxTokens: 200}); got != 200 {
		t.Errorf("gen with a 200-token ceiling = %v, want 200", got)
	}
}

// The profile budget and the no-think pass interact, and got this wrong once:
// runNoThinkQualityBenchmark indexes into the mixed pass's results positionally
// and refuses to run if the lengths disagree, so dropping skipped entries would
// have silently disabled no-think scoring on every truncated profile.
func TestSkippedResultsStayIndexAligned(t *testing.T) {
	if len(benchmarkQuestions) == 0 {
		t.Skip("no questions compiled in")
	}
	// A BenchResult recorded before the field existed must read as HAVING run —
	// those profiles did. That is why the field is "Skipped" and not "Ran".
	var old BenchResult
	if old.Skipped {
		t.Error("zero-value BenchResult reads as skipped; an existing cached profile would score as unasked")
	}
	var out benchOutcome
	if out.skipped {
		t.Error("zero-value benchOutcome reads as skipped")
	}
	// A skipped question is neither a pass nor a miss: it must not appear as a
	// failure, which is what would drag a slow worker's score down for questions
	// the budget never let it see.
	skipped := BenchResult{Tier: 5, Skipped: true}
	if skipped.Pass || skipped.Errored || skipped.Slow || skipped.Loose {
		t.Error("a skipped result carries a failure flag")
	}
}

// A skipped question must leave the category breakdown alone. Counting it in
// Total but never in Passed reads as a miss, so a truncated profile would report
// a model as weak in whichever categories the budget ran out in — and it runs
// out most on the slow workers, which is where a wrong reading does most harm.
func TestCategorySummaryIgnoresSkipped(t *testing.T) {
	var s benchCatScore
	s.add(BenchResult{Tier: 5, Pass: true})
	s.add(BenchResult{Tier: 5, Skipped: true})
	s.add(BenchResult{Tier: 5, Skipped: true})
	s.finish()
	if s.Total != 1 {
		t.Errorf("Total = %d, want 1 — skipped questions entered the denominator", s.Total)
	}
	if s.Percent != 100 {
		t.Errorf("Percent = %d, want 100 — two unasked questions were scored as misses", s.Percent)
	}
	// A genuine failure still counts, or the fix would hide real weaknesses.
	var s2 benchCatScore
	s2.add(BenchResult{Tier: 5, Pass: true})
	s2.add(BenchResult{Tier: 5})
	s2.finish()
	if s2.Total != 2 || s2.Percent != 50 {
		t.Errorf("Total=%d Percent=%d, want 2/50 — an ordinary miss must still count", s2.Total, s2.Percent)
	}
}
