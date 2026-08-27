package router

import (
	"context"
	"testing"
	"time"
)

// A backfilled observation must be filed under the identity a LIVE query will
// compute, or it is written to an address nothing ever asks for.
//
// This failed in production. A legacy profile (no stored ModelHash) fell back to
// unfingerprintedModelHash(id, model), and modelHash only takes that path for a
// worker with no served id, no parameter count and no size — which no real worker
// is. Measured on the fleet after a deploy: 497 observations recovered, every
// worker reporting total=0. The backfill logged success and rescued nothing.
func TestBackfilledEvidenceIsFiledWhereRoutingLooksForIt(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	benchmarkQuestions = []benchmarkQ{
		{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Match: "numeric"},
		{Tier: 1, Prompt: "Capital of France?", Expect: "Paris", Match: "contains"},
	}

	ctx := context.Background()
	logs := newTestLogStore(t)
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "w", URL: "http://w", Model: "gemma", MaxConcurrency: 1, TTLSeconds: 3600})
	// A registered worker always carries SOMETHING to fingerprint — a served id at
	// the very least. That is exactly what makes the unfingerprinted fallback
	// unreachable as a match target.
	reg.setModelMeta("w", ModelMeta{ServedID: "default", ModelParams: 26e9, Engine: engineLlamaCPP})
	live := reg.get("w")

	// A profile from before the fingerprint was recorded: no ModelHash.
	prof := &WorkerProfile{
		Model: "gemma", BenchVersion: benchmarkVersion, MeasuredAt: time.Unix(1000, 0).UTC(),
		BenchResults: []BenchResult{
			{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Pass: true, LatencyMS: 10},
			{Tier: 1, Prompt: "Capital of France?", Expect: "Paris", Pass: true, LatencyMS: 12},
		},
	}
	if prof.ModelHash != "" {
		t.Fatal("fixture wrong: this must be a legacy profile")
	}
	if err := logs.SaveWorkerProfile(ctx, "w", prof); err != nil {
		t.Fatalf("SaveWorkerProfile: %v", err)
	}

	r := &Router{logs: logs, registry: reg, outcomes: newOutcomeMatrix(logs.db)}
	if err := r.backfillOutcomesFromProfiles(ctx); err != nil {
		t.Fatalf("backfill: %v", err)
	}

	// The question routing actually asks: what does THIS worker know?
	rate, n := r.outcomes.summary(modelHash(live), false)
	if n != 2 {
		t.Fatalf("the worker sees %d of its own 2 recovered observations — they were filed "+
			"under an identity no live query computes", n)
	}
	if rate != 1 {
		t.Errorf("recovered hit rate %.2f, want 1.00", rate)
	}
	// And the cache can be hit, which is the point of recovering them at all.
	if _, _, ok := r.outcomes.cachedVerdict(benchQuestionQID(benchmarkQuestions[0]), modelHash(live), "w", false); !ok {
		t.Error("a recovered answer does not hit the permacache, so the profile re-asks it")
	}
}

// A profile whose worker is GONE keeps the unfingerprinted form. Nothing can
// compute its real fingerprint, and inventing one would attribute a
// decommissioned worker's results to whichever model is asked about next.
func TestALegacyProfileForAnAbsentWorkerStaysUnfingerprinted(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	benchmarkQuestions = []benchmarkQ{{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Match: "numeric"}}

	ctx := context.Background()
	logs := newTestLogStore(t)
	prof := &WorkerProfile{
		Model: "gemma", BenchVersion: benchmarkVersion, MeasuredAt: time.Unix(1000, 0).UTC(),
		BenchResults: []BenchResult{{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Pass: true, LatencyMS: 10}},
	}
	if err := logs.SaveWorkerProfile(ctx, "ghost", prof); err != nil {
		t.Fatalf("SaveWorkerProfile: %v", err)
	}
	r := &Router{logs: logs, registry: newTestRegistry(), outcomes: newOutcomeMatrix(logs.db)}
	if err := r.backfillOutcomesFromProfiles(ctx); err != nil {
		t.Fatalf("backfill: %v", err)
	}
	if _, n := r.outcomes.summary(unfingerprintedModelHash("ghost", "gemma"), false); n != 1 {
		t.Errorf("an absent worker's legacy profile was filed elsewhere: %d rows", n)
	}
}
