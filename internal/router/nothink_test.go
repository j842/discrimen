package router

// The two-score quality model: a worker is benchmarked both in the mixed mode
// the router serves when it chooses (Quality) and with thinking disabled on
// every question (QualityNoThink), and selection reads the score for the mode
// the request will actually be served in. The regression this exists for
// (2026-08-24): a 35B MoE benchmarked q=84 in its thinking mode was handed
// hard requirements.thinking="off" traffic against that score and wrote
// deterministic garbage — the no-think client was talking to a model the
// benchmark had never measured.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestQualityFor(t *testing.T) {
	moe := mkBackend("moe", 84, 100, 2, 0)
	moe.QualityNoThink = 40
	if q := qualityFor(moe, false); q != 84 {
		t.Fatalf("thinking-mode request must read the mixed score, got %d", q)
	}
	if q := qualityFor(moe, true); q != 40 {
		t.Fatalf("no-think request must read the no-think score, got %d", q)
	}
	// A profile cached before the field existed (or a provisional worker) has
	// zero — the fallback IS the pre-two-score behaviour, so nothing regresses
	// on a fleet of old profiles.
	legacy := mkBackend("legacy", 70, 100, 2, 0)
	if q := qualityFor(legacy, true); q != 70 {
		t.Fatalf("unmeasured no-think score must fall back to Quality, got %d", q)
	}
}

func TestResolveThinkingNoThink(t *testing.T) {
	r := &Router{} // explicit branches never touch classifier/cfg
	cases := []struct {
		name    string
		req     *ChatRequest
		noThink bool
	}{
		{"requirements off", &ChatRequest{Requirements: &Requirements{Thinking: "off"}}, true},
		{"requirements on", &ChatRequest{Requirements: &Requirements{Thinking: "on"}}, false},
		{"effort none", &ChatRequest{ReasoningEffort: "none"}, true},
		{"effort high", &ChatRequest{ReasoningEffort: "high"}, false},
		{"kwarg pinned off", &ChatRequest{ChatTemplateKwargs: map[string]any{"enable_thinking": false}}, true},
		{"kwarg pinned on", &ChatRequest{ChatTemplateKwargs: map[string]any{"enable_thinking": true}}, false},
		// Nothing explicit and no classifier: the mode is unknown, and the mixed
		// score is the best available estimate — noThink must stay false.
		{"unknown mode", &ChatRequest{}, false},
	}
	for _, c := range cases {
		if tr := r.resolveThinking(c.req, "route", nil); tr.noThink != c.noThink {
			t.Errorf("%s: noThink=%v, want %v", c.name, tr.noThink, c.noThink)
		}
	}
}

func TestRankByDifficultyReadsModeQuality(t *testing.T) {
	// The MoE outranks the dense worker in thinking mode and collapses below
	// the bar without it; the dense worker is the same model either way.
	moe := mkBackend("moe", 84, 100, 2, 0)
	moe.QualityNoThink = 40
	dense := mkBackend("dense", 70, 100, 2, 0)
	dense.QualityNoThink = 70

	// Target 80: in thinking mode only the MoE clears it; in no-think mode
	// neither does, and the below-bar order is closest-quality-first — which is
	// the dense worker once the MoE is read at its collapsed score.
	got := rankByDifficulty([]*Backend{dense, moe}, 80, jobCost{}, false)
	if got[0].ID != "moe" {
		t.Fatalf("thinking-mode request: only the MoE clears q>=80, got %s first", got[0].ID)
	}
	got = rankByDifficulty([]*Backend{moe, dense}, 80, jobCost{}, true)
	if got[0].ID != "dense" {
		t.Fatalf("no-think request must rank the collapsed MoE below the dense worker, got %s first", got[0].ID)
	}
}

func TestAutoTargetQualityClampsPerMode(t *testing.T) {
	moe := mkBackend("moe", 84, 100, 2, 0)
	moe.QualityNoThink = 40
	fleet := []*Backend{moe}
	if q := autoTargetQuality(fleet, 0.9, false); q != 84 {
		t.Fatalf("thinking clamp: got %d, want 84", q)
	}
	// A no-think request cannot be barred above the best no-think ability —
	// otherwise a hard bar set by the thinking-mode fleet strands it above
	// every worker's real ability in the mode it will be served in.
	if q := autoTargetQuality(fleet, 0.9, true); q != 40 {
		t.Fatalf("no-think clamp: got %d, want 40", q)
	}
}

func TestQualityFloorPreferenceReadsModeQuality(t *testing.T) {
	moe := mkBackend("moe", 84, 100, 2, 0)
	moe.QualityNoThink = 40
	dense := mkBackend("dense", 70, 100, 2, 0)
	dense.QualityNoThink = 70
	pref := qualityFloorPreference([]*Backend{moe, dense}, 60, true, 0)
	if pref.keep == nil {
		t.Fatal("expected a bounded preference (one worker above the no-think bar)")
	}
	if pref.keep(moe) {
		t.Fatal("the collapsed MoE must not be in the above-bar set for a no-think request")
	}
	if !pref.keep(dense) {
		t.Fatal("the dense worker clears the no-think bar and must be preferred")
	}
}

func TestWorkerProfileNoThinkJSONCompat(t *testing.T) {
	// A profile persisted before the field existed must load with zero (→
	// qualityFor falls back), and a new profile must round-trip both scores.
	var old WorkerProfile
	if err := json.Unmarshal([]byte(`{"model":"m","quality":80}`), &old); err != nil {
		t.Fatal(err)
	}
	if old.QualityNoThink != 0 {
		t.Fatalf("legacy profile: QualityNoThink=%d, want 0", old.QualityNoThink)
	}
	p := WorkerProfile{Model: "m", Quality: 84, QualityNoThink: 40}
	data, _ := json.Marshal(p)
	var back WorkerProfile
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back.QualityNoThink != 40 {
		t.Fatalf("round trip lost the no-think score: %d", back.QualityNoThink)
	}
}

// TestNoThinkBenchmarkMergesEasyTiers proves the two load-bearing properties of
// the no-think pass: it re-asks ONLY the hard tiers (the easy tiers already ran
// thinking-off in the mixed pass and are carried over, not re-billed), and it
// asks them with thinking actually disabled.
func TestNoThinkBenchmarkMergesEasyTiers(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	benchmarkQuestions = []benchmarkQ{
		{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Match: "numeric"},
		{Tier: 1, Prompt: "What is 3+3?", Expect: "6", Match: "numeric"},
		{Tier: 5, Prompt: "What is 17*23?", Expect: "391", Match: "numeric"},
		{Tier: 5, Prompt: "What is 19*29?", Expect: "551", Match: "numeric"},
	}
	answers := map[string]benchmarkQ{}
	for _, q := range benchmarkQuestions {
		answers[q.Prompt] = q
	}

	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		hits.Add(1)
		var body struct {
			Kwargs   map[string]any `json:"chat_template_kwargs"`
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(req.Body).Decode(&body)
		think, _ := body.Kwargs["enable_thinking"].(bool)
		prompt := body.Messages[0].Content
		prompt = strings.TrimSuffix(prompt, " /no_think")
		prompt = strings.TrimSuffix(prompt, " Give the number only.")
		q, ok := answers[prompt]
		answer := "unknown question"
		switch {
		case !ok:
		case q.Tier >= benchHardTier && !think:
			answer = "999999" // the collapsed no-think model: confidently wrong
		default:
			answer = q.Expect
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message":       map[string]any{"role": "assistant", "content": answer},
				"finish_reason": "stop",
			}},
		})
	}))
	defer srv.Close()

	r := &Router{benchClient: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "moe", URL: srv.URL, Model: "m"}}

	mixed, ok, _, _, details := r.runQualityBenchmark(b, 2)
	if !ok || mixed != 100 {
		t.Fatalf("mixed run: score=%d ok=%v, want 100 (hard tiers answered thinking-on)", mixed, ok)
	}
	mixedRequests := hits.Load()
	if mixedRequests != 4 {
		t.Fatalf("mixed run asked %d questions, want 4", mixedRequests)
	}

	nt, ntOK, _, ntResults := r.runNoThinkQualityBenchmark(b, 2, details)
	// The per-question record is what lets the category breakdown split the
	// no-think half; a score with no results behind it can only ever be shown
	// as one number.
	if ntOK && len(ntResults) != len(benchmarkQuestions) {
		t.Errorf("no-think details = %d, want one per question (%d)", len(ntResults), len(benchmarkQuestions))
	}
	if !ntOK {
		t.Fatal("no-think pass should have scored")
	}
	if got := hits.Load() - mixedRequests; got != 2 {
		t.Fatalf("no-think pass re-asked %d questions, want only the 2 hard ones", got)
	}
	// Easy tiers pass (carried from the mixed run), hard tiers fail — half the
	// base bucket, which the weighted score renders as 50.
	if nt != 50 {
		t.Fatalf("no-think score=%d, want 50 (easy carried over, hard failed)", nt)
	}
}

func TestNeedsNoThinkBackfill(t *testing.T) {
	// A stored run only backfills if it lines up with TODAY's bank position for
	// position — the backfill zips the two by index. A length match used to be
	// enough because benchmarkVersion was bumped for any question-set change;
	// content-addressed grading ended that, so the prompts are compared.
	aligned := make([]BenchResult, len(benchmarkQuestions))
	for i, q := range benchmarkQuestions {
		aligned[i] = BenchResult{Tier: q.Tier, Prompt: q.Prompt, Expect: q.Expect}
	}
	edited := append([]BenchResult(nil), aligned...)
	edited[len(edited)-1].Prompt = "a question that is no longer in the bank"

	cases := []struct {
		name string
		p    WorkerProfile
		want bool
	}{
		{"thinking worker with stored results", WorkerProfile{Thinking: true, BenchResults: aligned}, true},
		{"already has the score", WorkerProfile{Thinking: true, QualityNoThink: 40, BenchResults: aligned}, false},
		{"non-thinking worker needs nothing", WorkerProfile{BenchResults: aligned}, false},
		{"no stored results to merge from", WorkerProfile{Thinking: true}, false},
		{"right length, wrong questions", WorkerProfile{Thinking: true, BenchResults: edited}, false},
	}
	for _, c := range cases {
		if got := needsNoThinkBackfill(&c.p); got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}

// THE ALIGNMENT IS CHECKED, NOT ASSUMED. runNoThinkQualityBenchmark carries the
// mixed pass's easy-tier answers over BY INDEX, and its `mixed` argument can come
// from a profile read off disk. Under the old rules a stored run at the current
// benchmarkVersion had to be the current question set; graded answers are
// content-addressed now, so a question can be edited without a version bump and a
// same-length run is no longer the same exam. Zipping them anyway would attach one
// question's stored verdict to another's prompt — and observationsFrom resolves a
// row by its prompt, so the wrong verdict would be filed under the new question's
// qid and permacached there.
func TestNoThinkPassRefusesAMisalignedMixedRun(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	benchmarkQuestions = []benchmarkQ{
		{Tier: 1, Prompt: "easy-1", Expect: "1", Match: "numeric"},
		{Tier: 5, Prompt: "hard-1", Expect: "42", Match: "numeric"},
	}

	var asked atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		asked.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]any{"role": "assistant", "content": "42"}, "finish_reason": "stop",
			}},
		})
	}))
	defer srv.Close()

	r := &Router{benchClient: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL, Model: "m"}}

	// A run of the right LENGTH whose first question is not the bank's first
	// question any more — a question edited under an unchanged benchmarkVersion.
	stale := []BenchResult{
		{Tier: 1, Prompt: "easy-1-as-it-used-to-be-worded", Expect: "1", Pass: true, LatencyMS: 5},
		{Tier: 5, Prompt: "hard-1", Expect: "42", Pass: true, LatencyMS: 9},
	}
	if score, ok, _, details := r.runNoThinkQualityBenchmark(b, 1, stale); ok {
		t.Errorf("a mixed run that does not match the bank scored %d with ok=true (%d results); "+
			"ok=true is what persists a profile and writes the observations", score, len(details))
	}
	if got := asked.Load(); got != 0 {
		t.Errorf("%d questions were asked against a misaligned run, want 0", got)
	}

	// The same run, correctly aligned, still works — the guard must not be a
	// blanket refusal.
	good := []BenchResult{
		{Tier: 1, Prompt: "easy-1", Expect: "1", Pass: true, LatencyMS: 5},
		{Tier: 5, Prompt: "hard-1", Expect: "42", Pass: true, LatencyMS: 9},
	}
	if _, ok, _, _ := r.runNoThinkQualityBenchmark(b, 1, good); !ok {
		t.Error("an aligned mixed run was refused")
	}
}
