package router

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// The loader has to swallow whatever spelling a published dataset uses, because
// the alternative is a conversion script — and a conversion script is exactly
// what silently drops the difficulty labels the optimality metric needs.
func TestArenaLoadDatasetToleratesFieldSpellings(t *testing.T) {
	jsonl := strings.Join([]string{
		`{"id":"a1","question":"What is 2+2?","answer":"4","category":"maths","level":"easy"}`,
		`{"uid":7,"prompt":"Capital of France?","gold":"Paris","domain":"geography","difficulty":"EASY"}`,
		`{"input":"Pick one","target":"B","choices":["red","blue"],"subject":"trivia"}`,
		``,
		`{"text":"no answer here"}`,
	}, "\n")
	qs, err := arenaLoadDataset(writeTemp(t, "d.jsonl", jsonl))
	if err != nil {
		t.Fatal(err)
	}
	if len(qs) != 3 {
		t.Fatalf("want 3 usable questions (the answerless one is skipped), got %d", len(qs))
	}
	if qs[0].ID != "a1" || qs[0].Prompt != "What is 2+2?" || qs[0].Expect != "4" ||
		qs[0].Domain != "maths" || qs[0].Difficulty != "easy" {
		t.Fatalf("field mapping wrong: %+v", qs[0])
	}
	if qs[0].Match != "numeric" {
		t.Fatalf("a bare number should infer numeric matching, got %q", qs[0].Match)
	}
	if qs[1].ID != "7" || qs[1].Difficulty != "easy" {
		t.Fatalf("numeric ids and mixed-case difficulty should normalise: %+v", qs[1])
	}
	if qs[1].Match != "final-contains" {
		t.Fatalf("free text should infer final-contains, got %q", qs[1].Match)
	}
	// Choices have to reach the model or the item isn't gradable.
	if qs[2].Match != "mcq" {
		t.Fatalf("a lone letter with choices should infer mcq, got %q", qs[2].Match)
	}
	if !strings.Contains(qs[2].Prompt, "A. red") || !strings.Contains(qs[2].Prompt, "B. blue") {
		t.Fatalf("choices must be appended to the prompt: %q", qs[2].Prompt)
	}
}

func TestArenaLoadDatasetJSONArray(t *testing.T) {
	qs, err := arenaLoadDataset(writeTemp(t, "d.json",
		`[{"prompt":"one","expect":"1"},{"prompt":"two","expect":"2"}]`))
	if err != nil {
		t.Fatal(err)
	}
	if len(qs) != 2 || qs[1].Prompt != "two" {
		t.Fatalf("array form not parsed: %+v", qs)
	}
}

func TestArenaLoadDatasetRejectsUnusableFile(t *testing.T) {
	if _, err := arenaLoadDataset(writeTemp(t, "d.jsonl", `{"note":"nothing gradable"}`)); err == nil {
		t.Fatal("a dataset with no gradable items must be an error, not an empty run")
	}
}

func TestArenaPerturbationsPreserveMeaning(t *testing.T) {
	const prompt = "What is the capital of France?"
	for _, p := range arenaPerturbations {
		got := p.apply(prompt)
		if got == prompt {
			t.Errorf("%s changed nothing", p.kind)
		}
		if len(got) < len(prompt) {
			t.Errorf("%s shortened the prompt: %q", p.kind, got)
		}
	}
	// Deterministic, so robustness numbers are comparable between runs.
	if arenaTypo(prompt) != arenaTypo(prompt) {
		t.Fatal("perturbations must be deterministic")
	}
	if arenaTypo("tiny") != "tiny" {
		t.Fatal("a too-short prompt should be left alone rather than mangled")
	}
}

// The report is the deliverable, so its arithmetic is tested against a
// hand-worked example covering all four oracle verdicts.
func TestArenaMetricsOracleVerdicts(t *testing.T) {
	mk := func(pass bool, secs float64, oracle map[string]arenaWorkerResult) *arenaOutcome {
		return &arenaOutcome{
			Question: arenaQuestion{Prompt: "q", Difficulty: "easy"},
			Pass:     pass, Seconds: secs, Oracle: oracle,
		}
	}
	var m arenaMetrics
	// Optimal: right, and nothing correct was cheaper.
	arenaAccumulate([]*arenaMetrics{&m}, mk(true, 1.0, map[string]arenaWorkerResult{
		"tiny": {Pass: true, Seconds: 1.0}, "big": {Pass: true, Seconds: 9.0},
	}))
	// Overspend: right, but a cheaper worker was also right.
	arenaAccumulate([]*arenaMetrics{&m}, mk(true, 9.0, map[string]arenaWorkerResult{
		"tiny": {Pass: true, Seconds: 1.0}, "big": {Pass: true, Seconds: 9.0},
	}))
	// Undershoot: wrong, but some worker was right.
	arenaAccumulate([]*arenaMetrics{&m}, mk(false, 1.0, map[string]arenaWorkerResult{
		"tiny": {Pass: false, Seconds: 1.0}, "big": {Pass: true, Seconds: 9.0},
	}))
	// Unanswerable: nobody got it — not a routing failure.
	arenaAccumulate([]*arenaMetrics{&m}, mk(false, 1.0, map[string]arenaWorkerResult{
		"tiny": {Pass: false, Seconds: 1.0}, "big": {Pass: false, Seconds: 9.0},
	}))

	if m.n != 4 || m.passed != 2 {
		t.Fatalf("n=%d passed=%d want 4/2", m.n, m.passed)
	}
	if !m.haveOracle {
		t.Fatal("oracle data present but not recorded")
	}
	if m.oracleAnswered != 3 || m.unanswerable != 1 {
		t.Fatalf("answerable=%d unanswerable=%d want 3/1", m.oracleAnswered, m.unanswerable)
	}
	if m.optimal != 1 || m.overspend != 1 || m.undershoot != 1 {
		t.Fatalf("optimal=%d overspend=%d undershoot=%d want 1/1/1", m.optimal, m.overspend, m.undershoot)
	}
	// Oracle cost is the cheapest CORRECT worker per answerable question: 1+1+9.
	if m.oracleSeconds != 11 {
		t.Fatalf("oracle cost %v want 11", m.oracleSeconds)
	}
	// And what the router spent on those same three: 1+9+1.
	if m.answerableSeconds != 11 {
		t.Fatalf("router cost on answerable questions %v want 11", m.answerableSeconds)
	}
}

// An erroring worker is not evidence that a question is answerable.
func TestArenaMetricsIgnoresErroredOracleRuns(t *testing.T) {
	var m arenaMetrics
	arenaAccumulate([]*arenaMetrics{&m}, &arenaOutcome{
		Question: arenaQuestion{Prompt: "q"},
		Pass:     false, Seconds: 2,
		Oracle: map[string]arenaWorkerResult{
			"big": {Pass: true, Seconds: 3, Errored: true},
		},
	})
	if m.oracleAnswered != 0 || m.unanswerable != 1 {
		t.Fatalf("an errored oracle run must not count as answerable: answered=%d unanswerable=%d",
			m.oracleAnswered, m.unanswerable)
	}
}

// The optimality tolerance is relative AND absolute; both halves matter.
func TestArenaOptimalityTolerance(t *testing.T) {
	optimal := func(routerSecs, oracleSecs float64) bool {
		var m arenaMetrics
		arenaAccumulate([]*arenaMetrics{&m}, &arenaOutcome{
			Question: arenaQuestion{Prompt: "q"}, Pass: true, Seconds: routerSecs,
			Oracle: map[string]arenaWorkerResult{"tiny": {Pass: true, Seconds: oracleSecs}},
		})
		return m.optimal == 1
	}
	if !optimal(10.8, 10.0) {
		t.Fatal("an 8% overshoot on a 10s job is variance, not overspend")
	}
	if optimal(15.0, 10.0) {
		t.Fatal("a 50% overshoot on a 10s job is real overspend")
	}
	// The absolute floor: without it a fleet whose workers all answer in
	// milliseconds scores 0% optimal on scheduling jitter alone.
	if !optimal(0.20, 0.01) {
		t.Fatal("a 0.19s difference is jitter and must not count as overspend")
	}
	if optimal(3.0, 0.01) {
		t.Fatal("3s against 0.01s is a genuine routing difference")
	}
}

func TestArenaRobustnessCountsStableDecisions(t *testing.T) {
	var m arenaMetrics
	arenaAccumulate([]*arenaMetrics{&m}, &arenaOutcome{
		Question: arenaQuestion{Prompt: "q"}, BackendID: "big", TargetQuality: 70,
		Perturbed: []arenaPerturbation{
			{Kind: "lowercase", WouldServe: "big", Target: 70},
			{Kind: "polite", WouldServe: "tiny", Target: 20},
			{Kind: "typo", WouldServe: "big", Target: 70},
		},
	})
	if m.haveRobust != 3 || m.stableServe != 2 || m.stableTier != 2 {
		t.Fatalf("robustness counts wrong: n=%d serve=%d tier=%d", m.haveRobust, m.stableServe, m.stableTier)
	}
}

func TestPercentile(t *testing.T) {
	xs := []float64{5, 1, 4, 2, 3}
	if got := percentile(xs, 0); got != 1 {
		t.Fatalf("p0=%v want 1", got)
	}
	if got := percentile(xs, 1); got != 5 {
		t.Fatalf("p100=%v want 5", got)
	}
	if got := percentile(xs, 0.5); got != 3 {
		t.Fatalf("p50=%v want 3", got)
	}
	if got := percentile(nil, 0.5); got != 0 {
		t.Fatalf("empty p50=%v want 0", got)
	}
}
