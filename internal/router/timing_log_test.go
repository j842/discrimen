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
