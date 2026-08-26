package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// Every question has to land somewhere. benchCatUnknown is a real bucket that
// renders on the dashboard, and this is what keeps it empty: a new tier, or a
// LiveBench refresh in a format benchLiveMarkers does not know, fails here
// rather than filing a hundred questions under whatever they happened to be
// beside.
func TestBenchCategoryPlacesEveryQuestion(t *testing.T) {
	counts := map[string]int{}
	for _, q := range benchmarkQuestions {
		cat := benchCategoryOf(q.Tier, q.Prompt)
		counts[cat]++
		if cat == benchCatUnknown {
			t.Errorf("tier %d question is unclassified — add its tier to benchTierCategory, or its "+
				"LiveBench format to benchLiveMarkers: %s", q.Tier, benchSnippet(q.Prompt))
		}
	}
	// The set is meant to span three abilities. A category that has emptied out is
	// a benchmark that stopped measuring something, which is worth knowing about
	// even though nothing here is broken by it.
	for _, cat := range []string{benchCatCoding, benchCatMaths, benchCatReasoning} {
		if counts[cat] == 0 {
			t.Errorf("no question in the whole set is categorised %q — the breakdown has a column that "+
				"can only ever read 0/0", cat)
		}
	}
	t.Logf("category split of %d questions: %v", len(benchmarkQuestions), counts)
}

// Tier is the fallback rule, so every tier the set actually uses needs a charter
// entry — unless every question in it is recognised by a LiveBench marker first,
// which is true of tiers that only generated questions landed in.
func TestBenchCategoryCoversEveryTier(t *testing.T) {
	needsTier := map[int]string{}
	for _, q := range benchmarkQuestions {
		if benchCategoryOfLive(q.Prompt) != "" || benchCodeRe.MatchString(q.Prompt) {
			continue // settled before the tier table is consulted
		}
		if _, ok := benchTierCategory[q.Tier]; !ok {
			needsTier[q.Tier] = benchSnippet(q.Prompt)
		}
	}
	for tier, sample := range needsTier {
		t.Errorf("tier %d has hand-authored questions and no entry in benchTierCategory — say what the "+
			"tier measures: %s", tier, sample)
	}
}

// The marker table is the whole basis for categorising the generated half of the
// set, and it matches PROSE. That is safe only for as long as LiveBench keeps
// pasting the same answer-format instructions onto its items, which is exactly
// the kind of thing that changes in a refresh without anybody noticing. Every
// emitted question must be recognised, and no hand-authored one may be.
func TestBenchCategoryRecognisesEveryLiveQuestion(t *testing.T) {
	live := map[string]bool{}
	for _, q := range benchmarkQuestionsLive {
		live[q.Prompt] = true
		if benchCategoryOfLive(q.Prompt) == "" {
			t.Errorf("generated question in tier %d matches no marker in benchLiveMarkers, so it will be "+
				"categorised by its tier — which for a generated question is a measured pass rate and "+
				"means nothing: %s", q.Tier, benchSnippet(q.Prompt))
		}
	}
	for _, q := range benchmarkQuestions {
		if live[q.Prompt] {
			continue
		}
		if cat := benchCategoryOfLive(q.Prompt); cat != "" {
			t.Errorf("hand-authored tier-%d question matches a LiveBench marker and would be filed as %q: %s",
				q.Tier, cat, benchSnippet(q.Prompt))
		}
	}
}

// And the ground truth for the markers: the pool `bench emit` selects from
// carries LiveBench's own category on every question, so the table can be
// checked against it rather than against this file's opinion of it. Developer-
// side only — benchdata/pool.json is not in the image — so a missing pool skips.
func TestBenchCategoryMatchesTheLiveBenchPool(t *testing.T) {
	pool, err := benchLoadPool()
	if err != nil {
		t.Skipf("no question pool to check against: %v", err)
	}
	// LiveBench names its maths dataset "math"; this file spells it the way the
	// rest of the repo does.
	want := map[string]string{"math": benchCatMaths, "reasoning": benchCatReasoning, "coding": benchCatCoding}
	bad := map[string]int{}
	for _, q := range pool.Questions {
		expect, known := want[q.Category]
		if !known {
			bad["unknown pool category "+q.Category]++
			continue
		}
		if got := benchCategoryOfLive(q.Prompt); got != expect {
			bad[q.Task+": got "+got+", want "+expect]++
		}
	}
	for what, n := range bad {
		t.Errorf("%d pool question(s) — %s. Add the task's answer-format boilerplate to benchLiveMarkers.", n, what)
	}
}

// The hand-authored half is placed by its tier's charter, with a code override
// for the traces tier 4 mixes in with arithmetic. These are the placements worth
// pinning: each one is a rule this file would otherwise get wrong.
func TestBenchCategoryFilesHandAuthoredQuestions(t *testing.T) {
	cases := []struct {
		name   string
		tier   int
		prompt string
		want   string
	}{
		{"a code trace in the arithmetic tier is coding, not maths", 4,
			"What does this C++ program print? Give only the output.\n#include <iostream>\nint main(){ std::cout << 5 / 2 * 2.0; }", benchCatCoding},
		{"a shell gotcha is coding", 12,
			"What does this bash script print?\n\nf() { local x=$(false); echo \"$?\"; }\nf\n\nGive the number only.", benchCatCoding},
		{"a Go trace is coding, and bare 'go' is not what matched", 12,
			"This Go program prints one number. What is it?\n\npackage main\n", benchCatCoding},
		{"an arithmetic word problem in tier 4 is maths", 4,
			"A book has 250 pages. Tom reads 40% of it on Monday and another 30 pages on Tuesday. How many pages are left?", benchCatMaths},
		{"number theory is maths", 5,
			"What is the sum of all three-digit positive integers that are divisible by 7?", benchCatMaths},
		{"a misleading classic is reasoning", 6,
			"A notebook and a pen cost 220 cents in total. The notebook costs 200 cents more than the pen. How many cents does the pen cost?", benchCatReasoning},
		{"an invented operator is reasoning even though the answer is a number", 9,
			"Define the operator a # b as follows: if a is even, a # b = a/2 + b; if a is odd, a # b = 3a - b.", benchCatReasoning},
		{"a world-model trap is reasoning", 10,
			"Kofi puts a tray of twelve ice cubes into his freezer at 9am. How many ice cubes are in the tray?", benchCatReasoning},
		{"a control question is neither maths nor reasoning", 1,
			"What is the capital of France? Answer with one word.", benchCatGeneral},
		{"a reasoning trap that mentions going somewhere is not coding", 6,
			"How many different ways can you go from A to B?", benchCatReasoning},
	}
	for _, c := range cases {
		if got := benchCategoryOf(c.tier, c.prompt); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

// The two modes are the reason this view exists — a worker measured at 93
// thinking-on and 41 thinking-off is not 93 at anything a no-think client asks
// it. The breakdown has to carry both, aligned per question, and must report
// "not stored" as ABSENT rather than as zero.
func TestBenchCategoryBreakdownSplitsThinkingModes(t *testing.T) {
	coding := "This Go program prints one number. What is it?"
	maths := "What is the sum of all three-digit positive integers that are divisible by 7?"
	think := []BenchResult{
		{Tier: 12, Prompt: coding, Pass: true},
		{Tier: 12, Prompt: coding, Pass: true},
		{Tier: 5, Prompt: maths, Pass: true},
		{Tier: 5, Prompt: maths, Loose: true},
	}
	nothink := []BenchResult{
		{Tier: 12, Prompt: coding},
		{Tier: 12, Prompt: coding},
		{Tier: 5, Prompt: maths, Pass: true},
		{Tier: 5, Prompt: maths, Pass: true},
	}
	rows := benchCategoryBreakdown(think, nothink)
	if len(rows) != 2 {
		t.Fatalf("expected coding and maths rows, got %d: %+v", len(rows), rows)
	}
	// Canonical order, not map order: coding first.
	if rows[0].Category != benchCatCoding || rows[1].Category != benchCatMaths {
		t.Fatalf("rows are out of canonical order: %q then %q", rows[0].Category, rows[1].Category)
	}
	if rows[0].Think.Percent != 100 || rows[0].NoThink.Percent != 0 {
		t.Errorf("coding: think %d%%, no-think %d%% — want 100 and 0",
			rows[0].Think.Percent, rows[0].NoThink.Percent)
	}
	// A loose pass is half a point, the same as the headline score gives it.
	if rows[1].Think.Passed != 1.5 || rows[1].Think.Percent != 75 || rows[1].Think.Loose != 1 {
		t.Errorf("maths think: %+v — want 1.5 of 2 (75%%) with one loose", rows[1].Think)
	}
	if rows[1].NoThink.Percent != 100 {
		t.Errorf("maths no-think: %d%% — want 100", rows[1].NoThink.Percent)
	}
	if len(rows[0].Tiers) != 1 || rows[0].Tiers[0].Tier != 12 || rows[0].Tiers[0].NoThink == nil {
		t.Errorf("coding tier detail is wrong: %+v", rows[0].Tiers)
	}

	// No no-think run stored: absent, not zero. The distinction is the whole
	// point — a worker really can score 0 with thinking off.
	rows = benchCategoryBreakdown(think, nil)
	for _, row := range rows {
		if row.NoThink != nil {
			t.Errorf("%s reports a no-think score from a run that never happened: %+v", row.Category, row.NoThink)
		}
		for _, tier := range row.Tiers {
			if tier.NoThink != nil {
				t.Errorf("%s tier %d reports a no-think score that was never measured", row.Category, tier.Tier)
			}
		}
	}
	// Misaligned lengths mean the two runs graded different sets; zipping them by
	// index would produce numbers that look exactly as plausible as correct ones.
	rows = benchCategoryBreakdown(think, nothink[:2])
	for _, row := range rows {
		if row.NoThink != nil {
			t.Errorf("%s zipped a no-think run of the wrong length", row.Category)
		}
	}
}

// GET /backends/{id}/benchmark is where the dashboard reads all of this, so the
// two halves of the contract are checked here: the breakdown is present, and
// nothink_results_stored says which of "not measured" and "measured badly" the
// missing no-think figures mean. A UI that guessed between them would put a
// confident 0% against a worker nobody has run the pass on.
func TestServeBenchmarkCarriesTheCategoryBreakdown(t *testing.T) {
	r := &Router{cfg: &Config{}, registry: newTestRegistry(), logs: newTestLogStore(t)}
	issueKey(t, r, adminSecret, apiKey{Role: roleAdmin, Name: "admin"})
	prof := &WorkerProfile{
		Model: "test-model", Quality: 93, QualityNoThink: 41, Thinking: true,
		BenchVersion: benchmarkVersion, QualityNoThinkDetail: "t1=4/4 t12=11/28",
		BenchResults: []BenchResult{
			{Tier: 12, Prompt: "This Go program prints one number. What is it?", Pass: true},
			{Tier: 5, Prompt: "What is the sum of all three-digit positive integers that are divisible by 7?", Pass: true},
		},
	}
	if err := r.logs.SaveWorkerProfile(t.Context(), "w", prof); err != nil {
		t.Fatalf("SaveWorkerProfile: %v", err)
	}

	get := func() map[string]any {
		t.Helper()
		req := httptest.NewRequest(http.MethodGet, "/backends/w/benchmark", nil)
		req.Header.Set("Authorization", "Bearer "+adminSecret)
		rec := httptest.NewRecorder()
		r.handleBackendByID(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET /backends/w/benchmark: %d %s", rec.Code, rec.Body.String())
		}
		var out map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("decode: %v", err)
		}
		return out
	}

	out := get()
	cats, _ := out["categories"].([]any)
	if len(cats) != 2 {
		t.Fatalf("expected a coding and a maths row, got %v", out["categories"])
	}
	first, _ := cats[0].(map[string]any)
	if first["category"] != benchCatCoding {
		t.Errorf("categories are not in canonical order: %v", first["category"])
	}
	if _, present := first["nothink"]; present {
		t.Error("a no-think score was published for a run that never stored one")
	}
	if out["nothink_results_stored"] != false {
		t.Error("nothink_results_stored should be false when the no-think per-question results are absent")
	}
	if out["quality_nothink_detail"] != "t1=4/4 t12=11/28" {
		t.Errorf("the per-tier no-think line is missing: %v — it is the finest split that exists "+
			"when the per-question results were not kept", out["quality_nothink_detail"])
	}

	// With the per-question no-think results stored, the same endpoint splits
	// them by category. This is what the field on WorkerProfile is for.
	prof.BenchResultsNoThink = []BenchResult{
		{Tier: 12, Prompt: prof.BenchResults[0].Prompt},
		{Tier: 5, Prompt: prof.BenchResults[1].Prompt, Pass: true},
	}
	if err := r.logs.SaveWorkerProfile(t.Context(), "w", prof); err != nil {
		t.Fatalf("SaveWorkerProfile: %v", err)
	}
	out = get()
	if out["nothink_results_stored"] != true {
		t.Fatal("nothink_results_stored is false with an aligned no-think run stored")
	}
	cats, _ = out["categories"].([]any)
	coding, _ := cats[0].(map[string]any)
	nt, _ := coding["nothink"].(map[string]any)
	if nt == nil || nt["percent"] != float64(0) {
		t.Errorf("coding no-think score is %v — want an explicit 0%%, which is a measurement, not an absence", nt)
	}
}

// benchCategoryOfLive reports the category a LiveBench marker settles, or "" for
// a prompt no marker matches. Test-only: the production path folds this into
// benchCategoryOf, but the tests above need to tell "recognised as generated"
// apart from "fell through to the tier table".
func benchCategoryOfLive(prompt string) string {
	for _, m := range benchLiveMarkers {
		if strings.Contains(prompt, m.marker) {
			return m.category
		}
	}
	return ""
}

// Category counts, printed rather than asserted: what the split actually is
// today is useful to see in a test log, and pinning the numbers would fail every
// time a question is added.
func TestBenchCategoryReportsTheSplit(t *testing.T) {
	byCat := map[string]map[int]int{}
	for _, q := range benchmarkQuestions {
		cat := benchCategoryOf(q.Tier, q.Prompt)
		if byCat[cat] == nil {
			byCat[cat] = map[int]int{}
		}
		byCat[cat][q.Tier]++
	}
	for _, cat := range benchCategoryOrder {
		tiers, ok := byCat[cat]
		if !ok {
			continue
		}
		keys := make([]int, 0, len(tiers))
		total := 0
		for tier, n := range tiers {
			keys = append(keys, tier)
			total += n
		}
		sort.Ints(keys)
		parts := make([]string, 0, len(keys))
		for _, tier := range keys {
			parts = append(parts, "t"+strconv.Itoa(tier)+"="+strconv.Itoa(tiers[tier]))
		}
		t.Logf("%-12s %3d  %s", cat, total, strings.Join(parts, " "))
	}
}
