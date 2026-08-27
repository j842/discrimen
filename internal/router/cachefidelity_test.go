package router

import (
	"strings"
	"testing"
	"time"
)

// A cache hit must return the SAME verdict the run produced, including half
// credit. Observation.Correct is a bool and a loose result is Pass=false, so a
// cache that stored only Correct would hand back a plain miss — and the same
// worker, replying identically, would score LOWER on every re-profile than it did
// on the first. That is two scores that no longer mean the same thing, which is
// the failure the grading rules exist to prevent, reintroduced by the cache.
func TestLooseCreditSurvivesTheCache(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	benchmarkQuestions = []benchmarkQ{{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Match: "numeric"}}

	loose := []BenchResult{{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Pass: false, Loose: true, LatencyMS: 50}}
	rows := observationsFromMixed("model-x", "worker-a", loose, time.Unix(0, 0))
	if len(rows) != 1 {
		t.Fatalf("got %d observations, want 1", len(rows))
	}
	if !rows[0].Loose {
		t.Fatal("half credit was not recorded — the observation carries only Correct")
	}

	m := newOutcomeMatrix(nil)
	if err := m.record(t.Context(), rows); err != nil {
		t.Fatalf("record: %v", err)
	}
	hit, _, ok := m.cachedVerdict(benchQuestionQID(benchmarkQuestions[0]), "model-x", "worker-a", false)
	if !ok {
		t.Fatal("no cache hit")
	}
	if !hit.Loose {
		t.Error("the cached verdict lost its half credit — a re-profile would score this answer as a plain miss")
	}
}

// A context window is a property of the DEPLOYMENT, not of the weights.
//
// modelHash carries ModelCtxTrain — what the model was trained at — not the -c
// the server was started with, and the context probe measures each host
// separately. So two deployments of one model share every cached verdict while
// having genuinely different windows. A window miss must therefore count for the
// worker that hit it and NOT enter the model's shared record: otherwise one 32K
// deployment files a wrong answer that every 128K deployment of the same weights
// reads as real, and none of them ever asks the question again.
func TestAWindowMissIsNotEvidenceAboutTheModel(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	long := strings.Repeat("padding text for a long context question ", 3000)
	benchmarkQuestions = []benchmarkQ{{Tier: 9, Prompt: long + " How many entries?", Expect: "3000", Match: "numeric"}}

	small := &Backend{}
	small.ID, small.Model, small.ContextK = "small-box", "gemma", 8
	small.ServedID, small.ModelParams, small.ModelSizeBytes, small.Engine = "gemma", 26e9, 17e9, engineLlamaCPP

	r := &Router{cfg: &Config{}, benchClient: nil}
	out := r.benchOne(small, benchmarkQuestions[0], false, nil)

	if !out.unsupported {
		t.Fatal("an over-long prompt was not marked unsupported")
	}
	if out.pass {
		t.Error("a worker that cannot accept the prompt was scored as passing it")
	}
	if out.errd || out.skipped {
		t.Error("a window miss must count for this worker, not vanish from its denominator")
	}

	// It counts for the worker...
	res := BenchResult{Tier: 9, Prompt: benchmarkQuestions[0].Prompt, Expect: "3000",
		Pass: out.pass, Errored: out.errd, Skipped: out.skipped, Unsupported: out.unsupported}
	// ...and contributes nothing to the model's shared record.
	if rows := observationsFromMixed(modelHash(small), small.ID, []BenchResult{res}, time.Unix(0, 0)); len(rows) != 0 {
		t.Errorf("a window miss produced %d model-level observations, want 0 — a big-window "+
			"deployment of the same weights would read it as a genuine wrong answer", len(rows))
	}
}
