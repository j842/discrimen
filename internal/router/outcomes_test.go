package router

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"
)

// vec builds a simple unit-ish vector so similarity is easy to reason about in
// tests: two questions sharing a leading component are "about the same topic".
func vec(xs ...float64) []float64 { return xs }

func obs(qid, backend string, thinking, correct bool, ms int64, source string) Observation {
	return Observation{QID: qid, Backend: backend, Thinking: thinking, Correct: correct,
		LatencyMS: ms, Source: source, At: time.Unix(0, 0)}
}

func newTestMatrix(t *testing.T) *outcomeMatrix {
	t.Helper()
	return newOutcomeMatrix(nil) // no DB: persistence is exercised separately
}

// The central claim: a prediction reflects how a worker did on questions LIKE
// this one, and workers are ranked by that rather than by a global score.
func TestPredictUsesNearbyQuestions(t *testing.T) {
	m := newTestMatrix(t)
	// Two topics, far apart in embedding space.
	m.setVector("maths1", vec(1, 0))
	m.setVector("maths2", vec(0.99, 0.14))
	m.setVector("code1", vec(0, 1))
	m.setVector("code2", vec(0.14, 0.99))

	ctx := context.Background()
	// "mathsbot" is good at maths and bad at code; "codebot" the reverse. A
	// single quality scalar cannot express this, which is the whole point.
	_ = m.record(ctx, []Observation{
		obs("maths1", "mathsbot", true, true, 100, obsSourceBench),
		obs("maths2", "mathsbot", true, true, 100, obsSourceBench),
		obs("code1", "mathsbot", true, false, 100, obsSourceBench),
		obs("code2", "mathsbot", true, false, 100, obsSourceBench),
		obs("maths1", "codebot", true, false, 100, obsSourceBench),
		obs("maths2", "codebot", true, false, 100, obsSourceBench),
		obs("code1", "codebot", true, true, 100, obsSourceBench),
		obs("code2", "codebot", true, true, 100, obsSourceBench),
	})

	mathsPrompt, codePrompt := vec(1, 0.05), vec(0.05, 1)
	if p := m.predict(mathsPrompt, "mathsbot", true); p.Correct < 0.9 {
		t.Errorf("mathsbot on a maths prompt = %.2f, want ~1.0", p.Correct)
	}
	if p := m.predict(mathsPrompt, "codebot", true); p.Correct > 0.1 {
		t.Errorf("codebot on a maths prompt = %.2f, want ~0.0", p.Correct)
	}
	if p := m.predict(codePrompt, "codebot", true); p.Correct < 0.9 {
		t.Errorf("codebot on a code prompt = %.2f, want ~1.0", p.Correct)
	}
	// Both have an identical OVERALL hit rate of 50%, so a scalar would rank
	// them equal on every prompt. This is the improvement being claimed.
	rm, _ := m.summary("mathsbot", true)
	rc, _ := m.summary("codebot", true)
	if math.Abs(rm-rc) > 0.001 {
		t.Fatalf("test setup: overall rates should be equal, got %.2f and %.2f", rm, rc)
	}
}

// Thinking-on and thinking-off are separate models and must not share evidence.
func TestPredictSeparatesThinkingModes(t *testing.T) {
	m := newTestMatrix(t)
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(0.99, 0.14))
	_ = m.record(context.Background(), []Observation{
		obs("q1", "w", true, true, 100, obsSourceBench),
		obs("q2", "w", true, true, 100, obsSourceBench),
		obs("q1", "w", false, false, 100, obsSourceBench),
		obs("q2", "w", false, false, 100, obsSourceBench),
	})
	on := m.predict(vec(1, 0.05), "w", true)
	off := m.predict(vec(1, 0.05), "w", false)
	if on.Correct < 0.9 {
		t.Errorf("thinking-on = %.2f, want ~1.0", on.Correct)
	}
	if off.Correct > 0.1 {
		t.Errorf("thinking-off = %.2f, want ~0.0 — the modes leaked into each other", off.Correct)
	}
}

// A prompt unlike anything profiled must report that, not average over
// irrelevant questions. Most real traffic is this case.
func TestPredictReportsLowConfidenceForUnfamiliarPrompts(t *testing.T) {
	m := newTestMatrix(t)
	m.setVector("q1", vec(1, 0, 0))
	m.setVector("q2", vec(0.99, 0.14, 0))
	_ = m.record(context.Background(), []Observation{
		obs("q1", "w", true, true, 100, obsSourceBench),
		obs("q2", "w", true, true, 100, obsSourceBench),
	})
	far := m.predict(vec(0, 0, 1), "w", true)
	if far.known() {
		t.Errorf("an orthogonal prompt produced a usable prediction (conf=%.2f support=%.1f)",
			far.Confidence, far.Support)
	}
	near := m.predict(vec(1, 0.05, 0), "w", true)
	if !near.known() {
		t.Errorf("a close prompt produced no usable prediction (conf=%.2f support=%.1f)",
			near.Confidence, near.Support)
	}
}

// A worker with no observations at all must be unknown rather than zero — zero
// would read as "reliably wrong" and exclude a newly registered worker forever.
func TestPredictUnknownWorkerIsNotZero(t *testing.T) {
	m := newTestMatrix(t)
	m.setVector("q1", vec(1, 0))
	_ = m.record(context.Background(), []Observation{obs("q1", "known", true, true, 100, obsSourceBench)})
	p := m.predict(vec(1, 0), "never-profiled", true)
	if p.known() {
		t.Error("an unprofiled worker produced a usable prediction")
	}
	if p.Correct != 0 || p.Observations != 0 {
		t.Errorf("unprofiled worker = {correct %.2f obs %d}, want the zero value", p.Correct, p.Observations)
	}
}

// Judged evidence counts, but less than a graded bank answer: a judge that
// shares a blind spot with the worker it grades will agree with it.
func TestJudgedEvidenceIsDiscounted(t *testing.T) {
	m := newTestMatrix(t)
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(0.99, 0.14))
	// One bench failure, one judged success. The bench evidence must dominate.
	_ = m.record(context.Background(), []Observation{
		obs("q1", "w", true, false, 100, obsSourceBench),
		obs("q2", "w", true, true, 100, obsSourceJudge),
	})
	p := m.predict(vec(1, 0.05), "w", true)
	if p.Correct >= 0.5 {
		t.Errorf("hit rate %.2f — judged evidence was not discounted against bench evidence", p.Correct)
	}
}

// A re-profile supersedes its predecessor for the same (question, worker, mode,
// source); it must not accumulate, or a worker's history drags its estimate.
func TestReprofileSupersedes(t *testing.T) {
	m := newTestMatrix(t)
	m.setVector("q1", vec(1, 0))
	ctx := context.Background()
	_ = m.record(ctx, []Observation{obs("q1", "w", true, false, 100, obsSourceBench)})
	_ = m.record(ctx, []Observation{obs("q1", "w", true, true, 100, obsSourceBench)})
	if p := m.predict(vec(1, 0), "w", true); p.Correct != 1 {
		t.Errorf("hit rate %.2f after a corrected re-profile, want 1.0 — the old result survived", p.Correct)
	}
	// ...but bench and judge evidence coexist: different evidence about
	// different traffic.
	_ = m.record(ctx, []Observation{obs("q1", "w", true, false, 100, obsSourceJudge)})
	if p := m.predict(vec(1, 0), "w", true); p.Correct == 1 {
		t.Error("judged evidence replaced bench evidence instead of joining it")
	}
}

// forget must remove a worker entirely, so a delete or a grader change does not
// leave stale evidence behind.
func TestForgetRemovesAWorker(t *testing.T) {
	m := newTestMatrix(t)
	m.setVector("q1", vec(1, 0))
	ctx := context.Background()
	_ = m.record(ctx, []Observation{
		obs("q1", "gone", true, true, 100, obsSourceBench),
		obs("q1", "kept", true, true, 100, obsSourceBench),
	})
	_ = m.forget(ctx, "gone")
	if p := m.predict(vec(1, 0), "gone", true); p.Observations != 0 {
		t.Error("a forgotten worker still has evidence")
	}
	if p := m.predict(vec(1, 0), "kept", true); p.Observations == 0 {
		t.Error("forget removed the wrong worker")
	}
}

func testBackend(id string, tps float64) *Backend {
	b := &Backend{}
	b.ID = id
	b.BaselineTPS = tps
	b.MaxConcurrency = 4
	return b
}

// The routing rule in one test: among workers predicted to answer correctly,
// the FASTEST wins — not the most capable. That is the whole policy, and it is
// what replaces "meet this quality target".
func TestChooseFastestAmongCorrect(t *testing.T) {
	m := newTestMatrix(t)
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(0.99, 0.14))
	ctx := context.Background()
	// slow+accurate, fast+accurate, fast+wrong.
	_ = m.record(ctx, []Observation{
		obs("q1", "slow-good", true, true, 5000, obsSourceBench),
		obs("q2", "slow-good", true, true, 5000, obsSourceBench),
		obs("q1", "fast-good", true, true, 500, obsSourceBench),
		obs("q2", "fast-good", true, true, 500, obsSourceBench),
		obs("q1", "fast-bad", true, false, 100, obsSourceBench),
		obs("q2", "fast-bad", true, false, 100, obsSourceBench),
	})
	cands := []*Backend{testBackend("slow-good", 10), testBackend("fast-good", 100), testBackend("fast-bad", 500)}
	got, reason := m.chooseByOutcome(cands, vec(1, 0.05), true, jobCost{outputTokens: 200, mode: thinkingOn})
	if len(got) == 0 {
		t.Fatal("no candidates returned")
	}
	if got[0].ID != "fast-good" {
		t.Errorf("picked %q, want fast-good — fastest among those predicted correct (reason %q)", got[0].ID, reason)
	}
	for _, b := range got {
		if b.ID == "fast-bad" {
			t.Error("a worker predicted to be WRONG survived the filter, despite being fastest")
		}
	}
}

// When nothing similar has been profiled — the common case for real traffic —
// fall back to best overall leaning speed, and SAY so in the reason.
func TestFallbackWhenNothingSimilarIsKnown(t *testing.T) {
	m := newTestMatrix(t)
	m.setVector("q1", vec(1, 0, 0))
	ctx := context.Background()
	_ = m.record(ctx, []Observation{
		obs("q1", "strong", true, true, 4000, obsSourceBench),
		obs("q1", "weak", true, false, 200, obsSourceBench),
	})
	// A prompt orthogonal to everything profiled.
	cands := []*Backend{testBackend("strong", 10), testBackend("weak", 500)}
	got, reason := m.chooseByOutcome(cands, vec(0, 0, 1), true, jobCost{outputTokens: 200, mode: thinkingOn})
	if len(got) == 0 {
		t.Fatal("fallback returned nothing — an unfamiliar prompt must still route")
	}
	if !strings.Contains(reason, "fallback") {
		t.Errorf("reason %q does not record that this was a fallback", reason)
	}
	// "weak" is far behind on overall hit rate (0 vs 1), so the margin must NOT
	// admit it: leaning towards speed is not the same as ignoring accuracy.
	if got[0].ID != "strong" {
		t.Errorf("fallback picked %q; a worker %0.f%% behind on overall hit rate is outside the speed margin",
			got[0].ID, outcomeSpeedMargin*100)
	}
}

// Leaning towards speed means: among COMPARABLE workers, the fastest wins.
// Comparable is defined by outcomeSpeedMargin — a worker further behind than
// that is not traded for speed, which the previous test pins.
func TestFallbackPrefersSpeedAmongComparableWorkers(t *testing.T) {
	m := newTestMatrix(t)
	ctx := context.Background()
	var recs []Observation
	for i := 0; i < 8; i++ {
		qid := "q" + string(rune('a'+i))
		m.setVector(qid, vec(1, float64(i)/100, 0))
		// slow is right on all 8; quick on 7 of 8 — a 0.125 gap, inside the margin.
		recs = append(recs, obs(qid, "slow", true, true, 9000, obsSourceBench))
		recs = append(recs, obs(qid, "quick", true, i != 3, 300, obsSourceBench))
	}
	_ = m.record(ctx, recs)

	slowRate, _ := m.summary("slow", true)
	quickRate, _ := m.summary("quick", true)
	if slowRate-quickRate > outcomeSpeedMargin {
		t.Fatalf("test setup: gap %.3f exceeds the margin %.2f", slowRate-quickRate, outcomeSpeedMargin)
	}
	// An orthogonal prompt, so the matrix has no usable neighbours and the
	// fallback decides.
	cands := []*Backend{testBackend("slow", 10), testBackend("quick", 400)}
	got, reason := m.chooseByOutcome(cands, vec(0, 0, 1), true, jobCost{outputTokens: 200, mode: thinkingOn})
	if !strings.Contains(reason, "fallback") {
		t.Fatalf("expected the fallback path, got reason %q", reason)
	}
	if got[0].ID != "quick" {
		t.Errorf("picked %q; 'quick' is within %.0f%% on accuracy and far faster",
			got[0].ID, outcomeSpeedMargin*100)
	}
}

// When the fleet is bad at something, routing must still return a candidate —
// the most likely to be right, not the fastest. A fast wrong answer helps nobody.
func TestNoCandidateClearsTheFloor(t *testing.T) {
	m := newTestMatrix(t)
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(0.99, 0.14))
	ctx := context.Background()
	_ = m.record(ctx, []Observation{
		obs("q1", "a", true, false, 200, obsSourceBench),
		obs("q2", "a", true, true, 200, obsSourceBench), // 0.5, at the floor
		obs("q1", "b", true, false, 100, obsSourceBench),
		obs("q2", "b", true, false, 100, obsSourceBench), // 0.0
	})
	cands := []*Backend{testBackend("a", 50), testBackend("b", 500)}
	got, reason := m.chooseByOutcome(cands, vec(1, 0.05), true, jobCost{outputTokens: 200, mode: thinkingOn})
	if len(got) == 0 {
		t.Fatal("returned nothing; a hard prompt must still route somewhere")
	}
	if got[0].ID != "a" {
		t.Errorf("picked %q, want the more accurate 'a' (reason %q)", got[0].ID, reason)
	}
}

// The display summary must separate the two modes and must not fold judged
// production answers into the bank score — the bank is the fixed instrument, and
// mixing a moving sample of real traffic into it would make the headline drift
// for reasons unrelated to the worker.
func TestSummariseSeparatesModesAndSources(t *testing.T) {
	m := newTestMatrix(t)
	ctx := context.Background()
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(0.99, 0.14))
	_ = m.record(ctx, []Observation{
		obs("q1", "w", true, true, 100, obsSourceBench),
		obs("q2", "w", true, false, 100, obsSourceBench),
		obs("q1", "w", false, false, 100, obsSourceBench),
		obs("q2", "w", false, false, 100, obsSourceBench),
		obs("jq", "w", true, true, 100, obsSourceJudge),
	})
	topics := func(qid string) string {
		if qid == "q1" {
			return "maths"
		}
		return "coding"
	}
	on := m.summarise("w", true, topics)
	if on.Total != 2 || on.Correct != 1 {
		t.Errorf("thinking summary = %d/%d, want 1/2 — judged rows leaked into the bank score", on.Correct, on.Total)
	}
	if on.Judged != 1 {
		t.Errorf("judged count = %d, want 1", on.Judged)
	}
	off := m.summarise("w", false, topics)
	if off.Correct != 0 || off.Total != 2 {
		t.Errorf("no-think summary = %d/%d, want 0/2 — the modes leaked", off.Correct, off.Total)
	}
	// The per-topic breakdown is the strengths-and-weaknesses map.
	if len(on.ByTopic) != 2 {
		t.Fatalf("expected two topics, got %d", len(on.ByTopic))
	}
	for _, ts := range on.ByTopic {
		switch ts.Topic {
		case "maths":
			if ts.Rate != 1 {
				t.Errorf("maths rate %.2f, want 1.0", ts.Rate)
			}
		case "coding":
			if ts.Rate != 0 {
				t.Errorf("coding rate %.2f, want 0.0", ts.Rate)
			}
		}
	}
	// A worker with no evidence must read as INSUFFICIENT, not as a zero rate —
	// zero means "answered and was wrong".
	none := m.summarise("never-seen", true, topics)
	if !none.Insufficient {
		t.Error("a worker with no observations does not report insufficient")
	}
}

// Judged production questions are bounded; bank questions never are. Without
// that the matrix grows without limit and every routed request scans it.
func TestPruneJudgedKeepsBankQuestions(t *testing.T) {
	m := newTestMatrix(t)
	ctx := context.Background()
	m.setVector("bank1", vec(1, 0))
	_ = m.record(ctx, []Observation{obs("bank1", "w", true, true, 10, obsSourceBench)})
	for i := 0; i < 20; i++ {
		qid := "j" + string(rune('a'+i))
		m.setVector(qid, vec(1, float64(i)/100))
		o := obs(qid, "w", true, true, 10, obsSourceJudge)
		o.At = time.Unix(int64(i), 0)
		_ = m.record(ctx, []Observation{o})
	}
	m.pruneJudged(ctx, 5)
	if p := m.predict(vec(1, 0), "w", true); p.Observations == 0 {
		t.Error("pruning removed the bank question")
	}
	m.mu.RLock()
	judged := 0
	for _, list := range m.obs {
		for _, o := range list {
			if o.Source == obsSourceJudge {
				judged++
			}
		}
	}
	m.mu.RUnlock()
	if judged > 5 {
		t.Errorf("%d judged questions survived a cap of 5", judged)
	}
	// The oldest went first.
	m.mu.RLock()
	_, keptNewest := m.obs["j"+string(rune('a'+19))]
	_, keptOldest := m.obs["j"+string(rune('a'+0))]
	m.mu.RUnlock()
	if !keptNewest {
		t.Error("the newest judged question was pruned")
	}
	if keptOldest {
		t.Error("the oldest judged question survived")
	}
}
