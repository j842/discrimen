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

// noExploration parks the exploration counter where the next opportunities are
// not multiples of outcomeExploreEvery, so a test asserting on the ORDER the
// matrix produces is not occasionally handed the deliberately-suboptimal one.
// See outcomeExploreEvery — the counter is package-level because the policy is
// "one request in N" across the whole router, not per matrix.
func noExploration(t *testing.T) {
	t.Helper()
	outcomeExploreTick.Store(0)
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
	// The predicted-wrong worker is ranked LAST, not removed. Removing it would
	// leave failover and escalation with nowhere to go at exactly the moment the
	// chosen worker fails — and an unmeasured worker is not a bad one either.
	if len(got) != len(cands) {
		t.Errorf("%d candidates in, %d out — the list must be reordered, not narrowed",
			len(cands), len(got))
	}
	if got[len(got)-1].ID != "fast-bad" {
		t.Errorf("last candidate is %q, want fast-bad ranked last despite being fastest",
			got[len(got)-1].ID)
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

// Ranking, not filtering, is what keeps failover viable. A single measured
// worker used to reduce a whole fleet to one candidate.
func TestChooseNeverNarrowsTheCandidateList(t *testing.T) {
	noExploration(t)
	m := newTestMatrix(t)
	ctx := context.Background()
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(0.99, 0.14))
	_ = m.record(ctx, []Observation{
		obs("q1", "measured", true, true, 100, obsSourceBench),
		obs("q2", "measured", true, true, 100, obsSourceBench),
	})
	cands := []*Backend{testBackend("measured", 100), testBackend("fresh-a", 50), testBackend("fresh-b", 50)}
	got, _ := m.chooseByOutcome(cands, vec(1, 0.05), true, jobCost{outputTokens: 200, mode: thinkingOn})
	if len(got) != 3 {
		t.Fatalf("3 candidates in, %d out — unmeasured workers were evicted rather than ranked", len(got))
	}
	if got[0].ID != "measured" {
		t.Errorf("first is %q, want the measured worker", got[0].ID)
	}
	// Same for the fallback path, where nothing is measured at all.
	empty := newTestMatrix(t)
	all, _ := empty.chooseByOutcome(cands, vec(1, 0), true, jobCost{outputTokens: 200, mode: thinkingOn})
	if len(all) != 3 {
		t.Errorf("fallback returned %d of 3 candidates", len(all))
	}
}

// The confidence gate must be able to BIND. It used to be compared against the
// same constant that admits a neighbour in the first place — and the mean of
// values each >= X is >= X — so the check could never be false and known()
// silently reduced to "at least two observations", while the file's comments
// rested their honesty on a figure that gated nothing.
func TestConfidenceGateCanActuallyFail(t *testing.T) {
	if outcomeMinConfidence <= outcomeMinSimilarity {
		t.Fatalf("outcomeMinConfidence (%.2f) must exceed outcomeMinSimilarity (%.2f), or "+
			"the gate is vacuous by construction", outcomeMinConfidence, outcomeMinSimilarity)
	}
	m := newTestMatrix(t)
	ctx := context.Background()
	// Two neighbours that only just clear the admission floor.
	far1 := normalize([]float64{1, 1.2})
	far2 := normalize([]float64{1, 1.25})
	m.setVector("far1", far1)
	m.setVector("far2", far2)
	_ = m.record(ctx, []Observation{
		obs("far1", "w", true, true, 10, obsSourceBench),
		obs("far2", "w", true, true, 10, obsSourceBench),
	})
	p := m.predict([]float64{1, 0}, "w", true)
	if p.Observations >= outcomeMinObservations && p.known() {
		t.Errorf("barely-admitted neighbours produced a KNOWN prediction (conf=%.3f) — "+
			"the gate still cannot distinguish close evidence from distant evidence", p.Confidence)
	}
	// ...while genuinely close neighbours still qualify.
	m2 := newTestMatrix(t)
	m2.setVector("near1", normalize([]float64{1, 0.02}))
	m2.setVector("near2", normalize([]float64{1, 0.05}))
	_ = m2.record(ctx, []Observation{
		obs("near1", "w", true, true, 10, obsSourceBench),
		obs("near2", "w", true, true, 10, obsSourceBench),
	})
	if near := m2.predict([]float64{1, 0}, "w", true); !near.known() {
		t.Errorf("close neighbours produced an unusable prediction (conf=%.3f)", near.Confidence)
	}
}

// A vector from a DIFFERENT embedding space must not produce a prediction. dot()
// truncates to the shorter slice, so a 768-dim query against 384-dim bank
// vectors still scores a plausible cosine — measured at 0.866, clearing the
// admission floor and yielding a fully confident routing decision from garbage.
// Bank vectors are embedded once and never re-derived, so an embedding-model
// swap puts the matrix and the classifier in different spaces.
func TestPredictRejectsAForeignEmbeddingSpace(t *testing.T) {
	m := newTestMatrix(t)
	ctx := context.Background()
	m.setVector("q1", []float64{1, 0, 0, 0})
	m.setVector("q2", []float64{0.99, 0.14, 0, 0})
	_ = m.record(ctx, []Observation{
		obs("q1", "w", true, true, 10, obsSourceBench),
		obs("q2", "w", true, true, 10, obsSourceBench),
	})
	if p := m.predict([]float64{1, 0, 0, 0}, "w", true); !p.known() {
		t.Fatal("test premise wrong: a same-space query should predict")
	}
	wrongDim := []float64{1, 0, 0, 0, 0, 0, 0, 0}
	if p := m.predict(wrongDim, "w", true); p.Observations != 0 {
		t.Errorf("a %d-dim query against %d-dim vectors produced a prediction (correct=%.2f conf=%.2f)",
			len(wrongDim), 4, p.Correct, p.Confidence)
	}
}

// ── Rebuilding the matrix from data already on disk ─────────────────────────

// worker_profiles already holds every graded answer, in the exact shape
// observationsFromMixed consumes, for every worker ever profiled. Nothing read
// it: the only writer of bench observations was a profile completing in THIS
// process, so an empty observations table meant no routing evidence at all and
// the only way back was to re-profile the whole fleet — hours, every worker at
// once. Measured on the live fleet: "0 questions, 392 vectors, 0 observations"
// beside a worker holding a complete 392-result profile.
func TestBackfillRecoversObservationsFromStoredProfiles(t *testing.T) {
	ctx := context.Background()
	logs := newTestLogStore(t)
	r := &Router{logs: logs, outcomes: newOutcomeMatrix(logs.db)}

	// A profile of the shape profileBackend writes: the mixed pass (easy tiers
	// asked thinking-off, hard tiers thinking-on) plus a no-think pass.
	prof := &WorkerProfile{
		Model:        "m",
		BenchVersion: benchmarkVersion,
		MeasuredAt:   time.Unix(1000, 0).UTC(),
		BenchResults: []BenchResult{
			{Tier: 1, Prompt: "2+2?", Expect: "4", Pass: true, LatencyMS: 100},
			{Tier: benchHardTier, Prompt: "prove it", Expect: "qed", Pass: false, LatencyMS: 9000},
			{Tier: 1, Prompt: "skipped one", Expect: "x", Skipped: true},
			{Tier: 1, Prompt: "errored one", Expect: "y", Errored: true},
		},
		// The no-think pass re-asks the SAME questions with thinking off, so its
		// easy tiers collide with the mixed pass's — same question, same mode,
		// same source — and supersede rather than accumulate.
		BenchResultsNoThink: []BenchResult{
			{Tier: 1, Prompt: "2+2?", Expect: "4", Pass: true, LatencyMS: 40},
			{Tier: benchHardTier, Prompt: "prove it", Expect: "qed", Pass: false, LatencyMS: 900},
		},
	}
	if err := logs.SaveWorkerProfile(ctx, "w1", prof); err != nil {
		t.Fatalf("SaveWorkerProfile: %v", err)
	}
	// A second worker whose profile was graded against a retired question set.
	stale := &WorkerProfile{Model: "m", BenchVersion: benchmarkVersion - 1, MeasuredAt: time.Unix(1000, 0).UTC(),
		BenchResults: []BenchResult{{Tier: 1, Prompt: "old", Expect: "o", Pass: true, LatencyMS: 10}}}
	if err := logs.SaveWorkerProfile(ctx, "w2", stale); err != nil {
		t.Fatalf("SaveWorkerProfile: %v", err)
	}

	if err := r.backfillOutcomesFromProfiles(ctx); err != nil {
		t.Fatalf("backfill: %v", err)
	}

	// Thinking-on: only the hard-tier question was asked in that mode. The
	// skipped and errored ones say nothing about the model and must not be
	// reconstructed as misses.
	rate, on := r.outcomes.summary("w1", true)
	if on != 1 || rate != 0 {
		t.Errorf("thinking evidence = %d rows at %.2f, want 1 at 0.00 — the mixed pass was not split "+
			"by tier, or a skipped question was recorded as a miss", on, rate)
	}
	// Thinking-off: the two distinct questions of the no-think pass, one of which
	// the mixed pass also asked thinking-off.
	if _, off := r.outcomes.summary("w1", false); off != 2 {
		t.Errorf("no-think evidence = %d rows, want 2 — the no-think pass was not recorded, or the "+
			"mixed pass's easy tier was filed as thinking evidence", off)
	}
	// A profile graded under a different question set is not evidence about this one.
	if _, n := r.outcomes.summary("w2", false); n != 0 {
		t.Errorf("a BenchVersion %d profile contributed %d rows to a v%d matrix",
			stale.BenchVersion, n, benchmarkVersion)
	}

	// Idempotent: a second run files nothing, because recordIfNewer refuses to
	// replace a row that is already at least as fresh.
	before := r.outcomes.String()
	if err := r.backfillOutcomesFromProfiles(ctx); err != nil {
		t.Fatalf("second backfill: %v", err)
	}
	if after := r.outcomes.String(); after != before {
		t.Errorf("backfill is not idempotent: %q then %q", before, after)
	}

	// ...and it wrote through to SQLite rather than only into memory, so the next
	// restart does not need it again.
	restarted := newOutcomeMatrix(logs.db)
	if err := restarted.load(ctx); err != nil {
		t.Fatalf("load: %v", err)
	}
	if _, n := restarted.summary("w1", false); n != 2 {
		t.Errorf("after a restart the backfilled rows number %d, want 2 — the backfill did not persist", n)
	}
}

// A backfill reads profiles that may be OLDER than what is already recorded — a
// re-profile that has since landed, or a week of judged traffic. It must not
// walk those backwards.
func TestBackfillDoesNotClobberFresherEvidence(t *testing.T) {
	ctx := context.Background()
	logs := newTestLogStore(t)
	r := &Router{logs: logs, outcomes: newOutcomeMatrix(logs.db)}

	// The stored profile says the worker got it wrong, and is dated long ago.
	prof := &WorkerProfile{Model: "m", BenchVersion: benchmarkVersion, MeasuredAt: time.Unix(1000, 0).UTC(),
		BenchResultsNoThink: []BenchResult{{Tier: 1, Prompt: "2+2?", Expect: "4", Pass: false, LatencyMS: 100}}}
	if err := logs.SaveWorkerProfile(ctx, "w", prof); err != nil {
		t.Fatalf("SaveWorkerProfile: %v", err)
	}
	// A newer observation says otherwise.
	fresh := obs(benchQuestionQID("2+2?", "4"), "w", false, true, 100, obsSourceBench)
	fresh.At = time.Unix(99999, 0).UTC()
	if err := r.outcomes.record(ctx, []Observation{fresh}); err != nil {
		t.Fatalf("record: %v", err)
	}

	if err := r.backfillOutcomesFromProfiles(ctx); err != nil {
		t.Fatalf("backfill: %v", err)
	}
	rate, n := r.outcomes.summary("w", false)
	if n != 1 || rate != 1 {
		t.Errorf("after backfill = %d rows at %.2f, want 1 at 1.00 — a stale profile overwrote fresher evidence", n, rate)
	}
}

// ── Judged evidence has to survive a restart as something queryable ─────────

// load() restored every judged observation while m.vecs came back empty, and the
// only refill (ensureBankVectors) walks benchmarkQuestions. neighboursOf scans
// m.vecs, so a qid with no vector can never be a neighbour: the judged half of
// the matrix survived restarts as ballast that still consumed the
// maxJudgedQuestions cap and still inflated the dashboard's Judged count. The
// observations table stores a content hash and no text, so the vector cannot be
// re-derived — it has to be persisted.
func TestJudgedVectorsSurviveARestart(t *testing.T) {
	ctx := context.Background()
	logs := newTestLogStore(t)
	m := newOutcomeMatrix(logs.db)

	jq := judgedQID("how do I rotate a wireguard key?")
	if err := m.setJudgedVector(ctx, jq, vec(1, 0, 0)); err != nil {
		t.Fatalf("setJudgedVector: %v", err)
	}
	j2 := judgedQID("how do I rotate a tls certificate?")
	if err := m.setJudgedVector(ctx, j2, vec(0.999, 0.02, 0)); err != nil {
		t.Fatalf("setJudgedVector: %v", err)
	}
	if err := m.record(ctx, []Observation{
		obs(jq, "w", true, true, 100, obsSourceJudge),
		obs(j2, "w", true, true, 100, obsSourceJudge),
	}); err != nil {
		t.Fatalf("record: %v", err)
	}
	if p := m.predict(vec(1, 0.01, 0), "w", true); p.Observations == 0 {
		t.Fatal("test premise wrong: the judged rows are unreachable even before a restart")
	}

	restarted := newOutcomeMatrix(logs.db)
	if err := restarted.load(ctx); err != nil {
		t.Fatalf("load: %v", err)
	}
	if !restarted.hasVector(jq) {
		t.Error("the judged question has no vector after a restart — its observations are unqueryable ballast")
	}
	if p := restarted.predict(vec(1, 0.01, 0), "w", true); p.Observations != 2 {
		t.Errorf("a prompt like a judged one found %d observations after a restart, want 2", p.Observations)
	}
}

// A restored vector from a retired embedding space must be DROPPED, not used:
// dot() truncates to the shorter slice, so a stale vector still scores a
// plausible cosine against a query in the new space, and dimensionMatchesLocked
// only samples one entry — with a mixed map it answers at random.
func TestRestoredVectorsFromAForeignSpaceAreDropped(t *testing.T) {
	ctx := context.Background()
	logs := newTestLogStore(t)
	m := newOutcomeMatrix(logs.db)

	old := judgedQID("asked under the old embedder")
	if err := m.setJudgedVector(ctx, old, vec(1, 0, 0, 0)); err != nil {
		t.Fatalf("setJudgedVector: %v", err)
	}
	if err := m.record(ctx, []Observation{obs(old, "w", true, true, 100, obsSourceJudge)}); err != nil {
		t.Fatalf("record: %v", err)
	}

	restarted := newOutcomeMatrix(logs.db)
	if err := restarted.load(ctx); err != nil {
		t.Fatalf("load: %v", err)
	}
	if !restarted.hasVector(old) {
		t.Fatal("test premise wrong: the judged vector did not come back at all")
	}
	// The bank now embeds in a different number of dimensions.
	restarted.setVector("bank1", vec(1, 0, 0, 0, 0, 0))
	if restarted.hasVector(old) {
		t.Error("a vector from the retired embedding space survived alongside the new one")
	}
	again := newOutcomeMatrix(logs.db)
	if err := again.load(ctx); err != nil {
		t.Fatalf("reload: %v", err)
	}
	if again.hasVector(old) {
		t.Error("the retired vector came back from disk on the next restart")
	}
}

// Pruning bounds the judged observations; the vectors must go with them, or the
// cap bounds one table while the other grows forever.
func TestPruneJudgedDropsStoredVectors(t *testing.T) {
	ctx := context.Background()
	logs := newTestLogStore(t)
	m := newOutcomeMatrix(logs.db)
	for i := 0; i < 6; i++ {
		qid := judgedQID("question " + string(rune('a'+i)))
		if err := m.setJudgedVector(ctx, qid, vec(1, float64(i)/100)); err != nil {
			t.Fatalf("setJudgedVector: %v", err)
		}
		o := obs(qid, "w", true, true, 10, obsSourceJudge)
		o.At = time.Unix(int64(i), 0)
		if err := m.record(ctx, []Observation{o}); err != nil {
			t.Fatalf("record: %v", err)
		}
	}
	m.pruneJudged(ctx, 2)

	var stored int
	if err := logs.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM observation_vectors`).Scan(&stored); err != nil {
		t.Fatalf("count vectors: %v", err)
	}
	if stored != 2 {
		t.Errorf("%d stored vectors survived a cap of 2 — the vector table is unbounded", stored)
	}
}

// ── Above the floor, predicted correctness must still count ─────────────────

// The able band was sorted PURELY on Seconds, so accuracy was weightless above
// the floor, and outcomeSpeedMargin — the constant that implements "leaning
// towards speed" — applied only in the fallback, which made the primary policy
// MORE speed-dominant than the fallback it supersedes. Reproduced: a worker at
// p=0.50 with two observations beat one at p=1.00 on a 3% speed edge.
func TestAbleBandWeighsCorrectnessNotJustSpeed(t *testing.T) {
	noExploration(t)
	m := newTestMatrix(t)
	ctx := context.Background()
	var recs []Observation
	for i := 0; i < 8; i++ {
		qid := "q" + string(rune('a'+i))
		m.setVector(qid, vec(1, float64(i)/200))
		recs = append(recs, obs(qid, "accurate", true, true, 1000, obsSourceBench))
		// "marginal" gets half of them right — on the floor, and admitted to the
		// band by the >= comparison.
		recs = append(recs, obs(qid, "marginal", true, i%2 == 0, 1000, obsSourceBench))
	}
	if err := m.record(ctx, recs); err != nil {
		t.Fatalf("record: %v", err)
	}
	// A 3% speed edge to the marginal worker, which under a pure speed sort was
	// all it took.
	cands := []*Backend{testBackend("accurate", 100), testBackend("marginal", 103)}
	got, reason := m.chooseByOutcome(cands, vec(1, 0.01), true, jobCost{outputTokens: 200, mode: thinkingOn})
	if len(got) == 0 {
		t.Fatal("no candidates returned")
	}
	if got[0].ID != "accurate" {
		t.Errorf("picked %q — a 3%% speed edge outranked a 2x difference in predicted correctness (reason %q)",
			got[0].ID, reason)
	}

	// ...but leaning towards speed survives: two workers of COMPARABLE predicted
	// correctness still sort fastest first.
	m2 := newTestMatrix(t)
	var tied []Observation
	for i := 0; i < 8; i++ {
		qid := "q" + string(rune('a'+i))
		m2.setVector(qid, vec(1, float64(i)/200))
		tied = append(tied, obs(qid, "slower", true, true, 1000, obsSourceBench))
		tied = append(tied, obs(qid, "quicker", true, i != 3, 1000, obsSourceBench)) // 7/8, inside the margin
	}
	if err := m2.record(ctx, tied); err != nil {
		t.Fatalf("record: %v", err)
	}
	got2, reason2 := m2.chooseByOutcome([]*Backend{testBackend("slower", 10), testBackend("quicker", 400)},
		vec(1, 0.01), true, jobCost{outputTokens: 200, mode: thinkingOn})
	if got2[0].ID != "quicker" {
		t.Errorf("picked %q; within %.0f%% on predicted correctness the FASTER worker must win (reason %q)",
			got2[0].ID, outcomeSpeedMargin*100, reason2)
	}
}

// outcomeMinObservations is a floor of two, so a worker that got ONE of TWO
// nearby questions right scored exactly 0.50 and cleared outcomeCorrectFloor on
// the >= comparison. Evidence that thin cannot claim the floor on its own.
func TestThinEvidenceCannotClaimTheCorrectnessFloor(t *testing.T) {
	ctx := context.Background()
	m := newTestMatrix(t)
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(0.999, 0.02))
	if err := m.record(ctx, []Observation{
		obs("q1", "thin", true, true, 100, obsSourceBench),
		obs("q2", "thin", true, false, 100, obsSourceBench),
	}); err != nil {
		t.Fatalf("record: %v", err)
	}
	p := m.predict(vec(1, 0.01), "thin", true)
	if !p.known() {
		t.Fatal("test premise wrong: two close bench observations should be a known prediction")
	}
	if p.supportedCorrect() > outcomeCorrectFloor {
		t.Errorf("one right out of two nearby questions scored %.3f, above the %.2f floor — thin "+
			"evidence still claims the able band on a marginal rate", p.supportedCorrect(), outcomeCorrectFloor)
	}
	// The same claim on a dozen observations is a real measurement and is treated
	// as one: this discounts THIN evidence, not evidence.
	m2 := newTestMatrix(t)
	var many []Observation
	for i := 0; i < 12; i++ {
		qid := "q" + string(rune('a'+i))
		m2.setVector(qid, vec(1, float64(i)/400))
		many = append(many, obs(qid, "solid", true, true, 100, obsSourceBench))
	}
	if err := m2.record(ctx, many); err != nil {
		t.Fatalf("record: %v", err)
	}
	if p := m2.predict(vec(1, 0.01), "solid", true); p.supportedCorrect() <= outcomeCorrectFloor {
		t.Errorf("twelve nearby correct answers scored %.3f — the discount is punishing evidence, not thinness",
			p.supportedCorrect())
	}
}

// ── The live estimate must not be frozen out by a recorded median ───────────

// outcomeSeconds took max(live, historical), so whenever the recorded median
// dominated the sort key became a CONSTANT per (worker, neighbourhood): it
// stopped moving with queue depth, prompt length or the session discount, and
// every live-load mechanism below it was dead for that comparison. The median is
// also biased, because bench latencies are timed at each worker's OWN measured
// max batch size. Concrete inversion: an idle 8-slot worker (median 42s, live
// estimate 6s) lost to a saturated 2-slot worker (median 20s, live estimate 18s).
func TestOutcomeSecondsPrefersAnIdleWorkerOverASaturatedOne(t *testing.T) {
	// Equally fast per token; they differ in capacity and in how busy they are.
	idle := testBackend("idle", 100)
	idle.MaxConcurrency = 8
	idle.ActiveRequests = 0
	idle.Certification.TTFTMillis = 100

	busy := testBackend("busy", 100)
	busy.MaxConcurrency = 2
	busy.ActiveRequests = 12 // several deep past its slots
	busy.Certification.TTFTMillis = 100

	job := jobCost{outputTokens: 200, mode: thinkingOff}
	// The idle worker's median is the LARGER of the two: it batches eight wide,
	// so its per-question wall clock is inflated relative to the two-slot
	// worker's. That is precisely the case max() got backwards.
	idlePred := prediction{Correct: 1, Confidence: 0.9, Support: 2, Observations: 2, MedianLatencyMS: 42000}
	busyPred := prediction{Correct: 1, Confidence: 0.9, Support: 2, Observations: 2, MedianLatencyMS: 20000}

	idleSecs := outcomeSeconds(idle, idlePred, job)
	busySecs := outcomeSeconds(busy, busyPred, job)
	if idleSecs >= busySecs {
		t.Errorf("idle worker priced at %.1fs against a saturated one at %.1fs — a recorded median is "+
			"still overruling live queue depth", idleSecs, busySecs)
	}

	// The estimate must keep MOVING with live state: loading the idle worker up
	// has to raise its number. Under max() it did not, whenever the median won.
	loaded := *idle
	loaded.ActiveRequests = 40
	if after := outcomeSeconds(&loaded, idlePred, job); after <= idleSecs {
		t.Errorf("queueing 40 requests moved the estimate from %.1fs to %.1fs — the sort key is a "+
			"constant per (worker, neighbourhood) again", idleSecs, after)
	}
	// ...and with the PROMPT, which the recorded median knows nothing about.
	idle.ObservedPrefillTPS = 500
	longPrompt := job
	longPrompt.promptTokens = 40000
	if long := outcomeSeconds(idle, idlePred, longPrompt); long <= outcomeSeconds(idle, idlePred, job) {
		t.Error("a 40k-token prompt did not change the estimate — the live prefill term is being discarded")
	}
}

// The median still has to be USED — it is the only thing that knows how long
// this kind of question makes a worker generate for. A worker whose neighbours
// ran long must price higher than an identical worker whose neighbours were
// quick.
func TestOutcomeSecondsStillUsesTheRecordedLength(t *testing.T) {
	b := testBackend("w", 100)
	b.Certification.TTFTMillis = 100
	job := jobCost{outputTokens: 200, mode: thinkingOff}
	short := outcomeSeconds(b, prediction{MedianLatencyMS: 1000}, job)
	long := outcomeSeconds(b, prediction{MedianLatencyMS: 30000}, job)
	if long <= short {
		t.Errorf("a 30s median priced at %.2fs against a 1s median at %.2fs — the matrix's only "+
			"contribution, the answer length, was dropped", long, short)
	}
	// With no median at all the generic estimate stands.
	if none := outcomeSeconds(b, prediction{}, job); none != expectedLatency(b, job) {
		t.Errorf("with no recorded median the estimate is %.2fs, want the generic %.2fs",
			none, expectedLatency(b, job))
	}
}

// ── The judge discount has to actually discount something ──────────────────

// obsJudgeWeight was applied to both the numerator and the denominator of a
// weighted mean, so it cancels whenever every contributing row is judged — the
// steady state for real traffic. Measured: one correct and one incorrect
// observation gave Correct=0.500, Confidence=1.000, Observations=2 and
// known()=true whether the source was all-judge or all-bench, so two GOOD
// verdicts from a single-token LLM-judge call carried the routing authority of
// two exact-match bench grades.
func TestJudgedEvidenceAloneCannotSatisfyTheObservationFloor(t *testing.T) {
	ctx := context.Background()
	build := func(source string) prediction {
		m := newTestMatrix(t)
		m.setVector("q1", vec(1, 0))
		m.setVector("q2", vec(0.999, 0.02))
		if err := m.record(ctx, []Observation{
			obs("q1", "w", true, true, 100, source),
			obs("q2", "w", true, false, 100, source),
		}); err != nil {
			t.Fatalf("record: %v", err)
		}
		return m.predict(vec(1, 0.01), "w", true)
	}
	bench, judged := build(obsSourceBench), build(obsSourceJudge)
	if !bench.known() {
		t.Fatal("test premise wrong: two nearby bench observations must be a known prediction")
	}
	if judged.known() {
		t.Errorf("two judged rows (correct=%.3f conf=%.3f obs=%d) gate routing exactly as two bench rows do",
			judged.Correct, judged.Confidence, judged.Observations)
	}
	if judged.evidence() >= bench.evidence() {
		t.Errorf("judged evidence weighs %.2f against bench %.2f — the discount still cancels",
			judged.evidence(), bench.evidence())
	}
	// Four judged rows ARE two bench rows' worth, so judged evidence still
	// accumulates into something usable. Discounted, not discarded — it is the
	// only evidence that will ever cover the traffic actually being served.
	m := newTestMatrix(t)
	var four []Observation
	for i := 0; i < 4; i++ {
		qid := "q" + string(rune('a'+i))
		m.setVector(qid, vec(1, float64(i)/400))
		four = append(four, obs(qid, "w", true, true, 100, obsSourceJudge))
	}
	if err := m.record(ctx, four); err != nil {
		t.Fatalf("record: %v", err)
	}
	if p := m.predict(vec(1, 0.01), "w", true); !p.known() {
		t.Errorf("four judged rows (evidence %.2f) still do not gate routing — judged evidence is "+
			"being discarded rather than discounted", p.evidence())
	}
}

// Support is computed on every prediction and was reported nowhere, so an
// operator reading X-Llm-Route could not tell whether a route rested on one
// strong neighbour or a dozen weak ones.
func TestRouteReasonReportsSupport(t *testing.T) {
	noExploration(t)
	m := newTestMatrix(t)
	ctx := context.Background()
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(0.999, 0.02))
	m.setVector("q3", vec(0.998, 0.03))
	if err := m.record(ctx, []Observation{
		obs("q1", "w", true, true, 100, obsSourceBench),
		obs("q2", "w", true, true, 100, obsSourceBench),
		obs("q3", "w", true, true, 100, obsSourceBench),
	}); err != nil {
		t.Fatalf("record: %v", err)
	}
	_, reason := m.chooseByOutcome([]*Backend{testBackend("w", 100)}, vec(1, 0.01), true,
		jobCost{outputTokens: 200, mode: thinkingOn})
	if !strings.Contains(reason, "sup=") {
		t.Errorf("route reason %q does not report the support behind the prediction", reason)
	}
}

// ── The fallback scanned the whole matrix once per candidate ────────────────

// summary() walks the entire observation map — the bank plus up to
// maxJudgedQuestions judged rows — and the fallback called it once per
// candidate, taking the read lock each time. Seven workers meant seven full
// scans per routed request: the same shape, on the same hot path, that
// neighboursOf was split out of predict to remove at a measured 13ms a request.
func TestFallbackScansTheMatrixOnce(t *testing.T) {
	noExploration(t)
	m := newTestMatrix(t)
	ctx := context.Background()
	m.setVector("q1", vec(1, 0, 0))
	var recs []Observation
	cands := make([]*Backend, 0, 7)
	for i := 0; i < 7; i++ {
		id := "w" + string(rune('a'+i))
		recs = append(recs, obs("q1", id, true, true, 100, obsSourceBench))
		cands = append(cands, testBackend(id, float64(10*(i+1))))
	}
	if err := m.record(ctx, recs); err != nil {
		t.Fatalf("record: %v", err)
	}
	// A prompt orthogonal to everything profiled, so the fallback decides.
	m.fullScans.Store(0)
	got, reason := m.chooseByOutcome(cands, vec(0, 0, 1), true, jobCost{outputTokens: 200, mode: thinkingOn})
	if !strings.Contains(reason, "fallback") {
		t.Fatalf("expected the fallback path, got reason %q", reason)
	}
	if len(got) != len(cands) {
		t.Fatalf("%d candidates in, %d out", len(cands), len(got))
	}
	if n := m.fullScans.Load(); n > 1 {
		t.Errorf("%d full scans of the observation map for %d candidates, want 1 — the fallback is "+
			"rescanning per candidate again", n, len(cands))
	}
}

// ── Exploration: the one place the router does not pick its best ────────────

// An unmeasured worker ranks behind every measured one, and the only way to stop
// being unmeasured on real traffic is to be served. Without exploration a worker
// that is never tried never earns evidence, and the matrix's first impression of
// a fleet becomes permanent.
func TestExplorationOccasionallyTriesAnUnmeasuredWorker(t *testing.T) {
	noExploration(t)
	m := newTestMatrix(t)
	ctx := context.Background()
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(0.999, 0.02))
	if err := m.record(ctx, []Observation{
		obs("q1", "measured", true, true, 10, obsSourceBench),
		obs("q2", "measured", true, true, 10, obsSourceBench),
	}); err != nil {
		t.Fatalf("record: %v", err)
	}
	// The measured worker is both known-correct and faster, so nothing but
	// exploration will ever put the fresh one first.
	cands := []*Backend{testBackend("measured", 500), testBackend("fresh", 10)}

	const rounds = outcomeExploreEvery * 2
	explored := 0
	reasons := map[string]int{}
	for i := 0; i < rounds; i++ {
		got, reason := m.chooseByOutcome(cands, vec(1, 0.01), true, jobCost{outputTokens: 200, mode: thinkingOn})
		if len(got) != 2 {
			t.Fatalf("round %d returned %d candidates", i, len(got))
		}
		reasons[reason]++
		if got[0].ID == "fresh" {
			explored++
			if !strings.Contains(reason, "explore") {
				t.Errorf("round %d promoted the unmeasured worker but reported %q — an exploration that "+
					"cannot be distinguished in X-Llm-Route cannot be measured", i, reason)
			}
		}
	}
	if explored == 0 {
		t.Fatalf("%d routed requests and the unmeasured worker was never tried, so it can never earn "+
			"evidence (reasons seen: %v)", rounds, reasons)
	}
	// Bounded: exploration is a deliberate, priced cost, not a policy.
	if explored > rounds/outcomeExploreEvery+1 {
		t.Errorf("%d of %d requests explored, want about 1 in %d", explored, rounds, outcomeExploreEvery)
	}
}

// Exploration must not invent a candidate: with nothing unmeasured there is
// nothing to try, and the ranking stands.
func TestExplorationDoesNothingWithoutAnUnmeasuredCandidate(t *testing.T) {
	noExploration(t)
	m := newTestMatrix(t)
	ctx := context.Background()
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(0.999, 0.02))
	var recs []Observation
	for _, id := range []string{"a", "b"} {
		recs = append(recs, obs("q1", id, true, true, 10, obsSourceBench))
		recs = append(recs, obs("q2", id, true, true, 10, obsSourceBench))
	}
	if err := m.record(ctx, recs); err != nil {
		t.Fatalf("record: %v", err)
	}
	cands := []*Backend{testBackend("a", 10), testBackend("b", 500)}
	for i := 0; i < outcomeExploreEvery*2; i++ {
		got, reason := m.chooseByOutcome(cands, vec(1, 0.01), true, jobCost{outputTokens: 200, mode: thinkingOn})
		if strings.Contains(reason, "explore") {
			t.Fatalf("round %d explored with every candidate already measured (%q)", i, reason)
		}
		if got[0].ID != "b" {
			t.Fatalf("round %d picked %q, want the faster measured worker", i, got[0].ID)
		}
	}
}
