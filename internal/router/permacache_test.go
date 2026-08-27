package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// The permacache has to actually skip generations, and be seen to.
//
// Everything about it is silent by construction: a cache that never hits still
// produces a correct profile, just slowly, and a cache that hits WRONGLY produces
// a plausible score from somebody else's answers. So this drives the real
// benchmark twice against a counting worker and asserts the second run asks
// nothing — and then that changing the grader makes it ask again.
func TestTheBenchmarkReusesAnAnswerItAlreadyHas(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	benchmarkQuestions = []benchmarkQ{
		{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Match: "numeric"},
		{Tier: 1, Prompt: "Capital of France?", Expect: "Paris", Match: "contains"},
	}

	var asked atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		asked.Add(1)
		var body struct {
			Messages []struct{ Content string } `json:"messages"`
		}
		_ = json.NewDecoder(req.Body).Decode(&body)
		answer := "4"
		if len(body.Messages) > 0 && len(body.Messages[0].Content) > 0 && body.Messages[0].Content[0] == 'C' {
			answer = "Paris"
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]any{"role": "assistant", "content": answer}, "finish_reason": "stop",
			}},
		})
	}))
	defer srv.Close()

	m := newOutcomeMatrix(nil)
	r := &Router{cfg: &Config{}, benchClient: &http.Client{}, outcomes: m}
	b := &Backend{}
	b.ID, b.Model, b.URL = "w", "m", srv.URL
	b.ServedID, b.ModelParams, b.ModelSizeBytes, b.Engine = "gemma", 26e9, 17e9, engineLlamaCPP

	// First run: cold. Every question costs a generation, and the results are
	// filed under the model.
	score, ok, _, _, details := r.runQualityBenchmark(b, 2)
	if !ok || score != 100 {
		t.Fatalf("cold run scored %d ok=%v, want 100", score, ok)
	}
	if got := asked.Load(); got != 2 {
		t.Fatalf("cold run asked %d questions, want 2", got)
	}
	rows := observationsFromMixed(modelHash(b), b.ID, details, time.Unix(0, 0))
	if len(rows) != 2 {
		t.Fatalf("cold run produced %d observations, want 2", len(rows))
	}
	if err := m.record(t.Context(), rows); err != nil {
		t.Fatalf("record: %v", err)
	}

	// Second run: every question is already answered by THIS model, so nothing is
	// asked and the score is identical.
	asked.Store(0)
	score2, ok2, _, _, details2 := r.runQualityBenchmark(b, 2)
	if !ok2 || score2 != score {
		t.Errorf("cached run scored %d, want %d — a cache hit changed the verdict", score2, score)
	}
	if got := asked.Load(); got != 0 {
		t.Errorf("cached run asked %d questions, want 0 — the permacache never hit", got)
	}
	for _, d := range details2 {
		if !d.Pass {
			t.Errorf("cached result lost its verdict: %+v", d)
		}
	}

	// Fix the numeric grader: only its questions get a new qid, so only THAT
	// question is re-asked. This is the whole point — the old global
	// benchmarkVersion re-ran the entire bank to fix one grader.
	asked.Store(0)
	graderVersions["numeric"]++
	defer func() { graderVersions["numeric"]-- }()
	if _, _, _, _, d3 := r.runQualityBenchmark(b, 2); len(d3) != 2 {
		t.Fatalf("post-grader-fix run produced %d results, want 2", len(d3))
	}
	if got := asked.Load(); got != 1 {
		t.Errorf("after bumping ONLY the numeric grader, %d questions were re-asked, want 1 — "+
			"a grader fix should not invalidate the questions it does not grade", got)
	}
}

// A hit from another deployment of the same weights carries a real verdict and a
// meaningless duration: same model, different box, different load. Taking the
// latency would attribute one machine's speed to another.
func TestACachedLatencyDoesNotTravelBetweenWorkers(t *testing.T) {
	m := newOutcomeMatrix(nil)
	q := benchmarkQ{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Match: "numeric"}
	qid := benchQuestionQID(q)

	same := &Backend{}
	same.ID, same.Model = "host-a", "gemma"
	same.ServedID, same.ModelParams, same.ModelSizeBytes, same.Engine = "gemma", 26e9, 17e9, engineLlamaCPP

	if err := m.record(t.Context(), []Observation{{
		QID: qid, ModelHash: modelHash(same), Backend: "host-a",
		Thinking: false, Correct: true, LatencyMS: 1234, Source: obsSourceBench, At: time.Unix(0, 0),
	}}); err != nil {
		t.Fatalf("record: %v", err)
	}

	if hit, sameWorker, ok := m.cachedVerdict(qid, modelHash(same), "host-a", false); !ok || !sameWorker || hit.LatencyMS != 1234 {
		t.Errorf("the worker that measured it: ok=%v same=%v latency=%d, want true/true/1234", ok, sameWorker, hit.LatencyMS)
	}
	// The same weights on another host.
	other := *same
	other.ID = "host-b"
	if hit, sameWorker, ok := m.cachedVerdict(qid, modelHash(&other), "host-b", false); !ok || sameWorker {
		t.Errorf("another deployment: ok=%v same=%v, want true/false — the verdict travels, the timing does not", ok, sameWorker)
	} else if !hit.Correct {
		t.Error("the verdict did not travel to another deployment of the same model")
	}
}

// A question is its text AND its grading. Same prompt, different match mode or a
// bumped grader, means a different question — otherwise a grader fix would keep
// serving the verdict it was written to correct.
func TestQuestionIdentityCoversItsGrading(t *testing.T) {
	base := benchmarkQ{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Match: "numeric"}
	if benchQuestionQID(base) != benchQuestionQID(base) {
		t.Fatal("qid is not stable for an unchanged question")
	}

	other := base
	other.Match = "contains"
	if benchQuestionQID(base) == benchQuestionQID(other) {
		t.Error("the same text graded two different ways shares one identity")
	}

	edited := base
	edited.Expect = "5"
	if benchQuestionQID(base) == benchQuestionQID(edited) {
		t.Error("changing the expected answer left the identity unchanged")
	}

	before := benchQuestionQID(base)
	graderVersions["numeric"]++
	after := benchQuestionQID(base)
	graderVersions["numeric"]--
	if before == after {
		t.Error("bumping the grader version left the identity unchanged — a grader fix would keep " +
			"serving the cached verdict it was written to correct")
	}
	if benchQuestionQID(base) != before {
		t.Error("the identity did not come back when the version did")
	}
}
