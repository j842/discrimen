package router

import "testing"

// The compact line is what `ask -l` renders per worker. Two properties matter:
// the thinking-on→off arrow must appear when both runs were stored, and must be
// ABSENT (not "→0%") when the no-think run's per-question results were not —
// a worker can genuinely score zero with thinking off.
func TestBenchCategorySummaryShape(t *testing.T) {
	think := []BenchResult{
		{Tier: 12, Prompt: "### Instructions: You are an expert Python programmer.", Pass: true},
		{Tier: 12, Prompt: "### Instructions: You are an expert Python programmer.", Pass: false},
		{Tier: 5, Prompt: "What is 2+2? Give the number only.", Pass: true},
	}
	nothink := []BenchResult{
		{Tier: 12, Prompt: "### Instructions: You are an expert Python programmer.", Pass: false},
		{Tier: 12, Prompt: "### Instructions: You are an expert Python programmer.", Pass: false},
		{Tier: 5, Prompt: "What is 2+2? Give the number only.", Pass: true},
	}
	both := benchCategorySummary(think, nothink)
	if !contains(both, "→") {
		t.Errorf("both runs stored but no arrow: %q", both)
	}
	if !contains(both, "coding") {
		t.Errorf("coding category missing: %q", both)
	}
	only := benchCategorySummary(think, nil)
	if contains(only, "→") {
		t.Errorf("no-think absent but arrow rendered — 'not measured' must not look like 'scored zero': %q", only)
	}
	if benchCategorySummary(nil, nil) != "" {
		t.Error("no results should render nothing, not an empty header")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// An OUTAGE is not a weakness. A question that errored is one we could not
// grade — the transport failed after every retry, the worker was abandoned as
// busy, the code-exec sidecar could not start the program — and counting it in
// Total but never in Passed is arithmetically identical to marking it wrong.
// The sibling Skipped rule has always excluded it; this one did not, so a
// sandbox that was down for a run reported the worker as bad at coding.
func TestBenchCatScoreIgnoresErrored(t *testing.T) {
	var s benchCatScore
	s.add(BenchResult{Tier: 12, Pass: true})
	s.add(BenchResult{Tier: 12, Errored: true})
	s.add(BenchResult{Tier: 12, Errored: true})
	s.finish()
	if s.Total != 1 {
		t.Errorf("Total = %d, want 1 — ungradeable questions entered the denominator", s.Total)
	}
	if s.Percent != 100 {
		t.Errorf("Percent = %d, want 100 — two outages were scored as misses", s.Percent)
	}
	if s.Errored != 2 || s.asked() != 3 {
		t.Errorf("Errored=%d asked=%d, want 2 and 3 — the unmeasured questions have to stay countable, "+
			"or 1-of-3 renders exactly like 1-of-1", s.Errored, s.asked())
	}

	// Too slow to be usable is a real verdict about the worker, and stays in the
	// denominator. benchmark.go draws the line in the same place; if this ever
	// flips, a worker could earn a high category score by never answering in time.
	var slow benchCatScore
	slow.add(BenchResult{Tier: 12, Pass: true})
	slow.add(BenchResult{Tier: 12, Slow: true})
	slow.finish()
	if slow.Total != 2 || slow.Percent != 50 {
		t.Errorf("Total=%d Percent=%d, want 2 and 50 — a too-slow answer must still count against the worker",
			slow.Total, slow.Percent)
	}
}

// The defect that made this worth pinning was not the arithmetic on its own: it
// was that `ask -l` showed a category percentage next to a headline Quality
// derived from the SAME run by the OPPOSITE rule, so a sandbox outage read as
// "coding 0%" beside a Quality of 75. This recomputes benchmark.go's rule
// (`if !res.errd { count[res.tier]++ }`, a loose pass worth half) beside this
// file's and requires them to agree.
func TestCategoryRateAgreesWithTheHeadlineScore(t *testing.T) {
	coding := "This Go program prints one number. What is it?"
	results := []BenchResult{
		{Tier: 12, Prompt: coding, Pass: true},
		{Tier: 12, Prompt: coding, Pass: true},
		{Tier: 12, Prompt: coding, Loose: true},   // half credit, in the denominator
		{Tier: 12, Prompt: coding},                // plain miss, in the denominator
		{Tier: 12, Prompt: coding, Slow: true},    // usability fail, in the denominator
		{Tier: 12, Prompt: coding, Errored: true}, // outage, out of both
		{Tier: 12, Prompt: coding, Errored: true},
		{Tier: 12, Prompt: coding, Skipped: true}, // never asked, out of both
	}

	// benchmark.go's arithmetic, written out rather than called: the point is
	// that the two agree, and sharing an implementation would prove nothing.
	var headlineTotal int
	var headlinePassed float64
	for _, r := range results {
		if r.Skipped || r.Errored {
			continue
		}
		headlineTotal++
		switch {
		case r.Pass:
			headlinePassed++
		case r.Loose:
			headlinePassed += 0.5
		}
	}
	wantPct := int(100*headlinePassed/float64(headlineTotal) + 0.5)

	rows := benchCategoryBreakdown(results, nil)
	if len(rows) != 1 || rows[0].Category != benchCatCoding {
		t.Fatalf("expected one coding row, got %+v", rows)
	}
	got := rows[0].Think
	if got.Total != headlineTotal {
		t.Errorf("category denominator %d, headline denominator %d — the same run counted two ways",
			got.Total, headlineTotal)
	}
	if got.Percent != wantPct {
		t.Errorf("category reports %d%%, the headline rule gives %d%% — two numbers from one run, "+
			"shown side by side, computed by opposite rules", got.Percent, wantPct)
	}
	if got.Errored != 2 || got.Skipped != 1 || got.asked() != len(results) {
		t.Errorf("unmeasured questions are not accounted for: %+v", got)
	}
}

// A percentage with no denominator invites the reader to supply one, and the one
// they supply is the whole category. "coding 50%" measured on two of fifteen
// questions is not the same claim as "coding 50%" measured on all fifteen, and
// an operator choosing a worker off this line deserves to be told which it is.
func TestBenchCategorySummaryReportsPartialCoverage(t *testing.T) {
	coding := "This Go program prints one number. What is it?"
	maths := "What is the sum of all three-digit positive integers that are divisible by 7?"
	results := []BenchResult{
		{Tier: 12, Prompt: coding, Pass: true},
		{Tier: 12, Prompt: coding},
		{Tier: 12, Prompt: coding, Errored: true},
		{Tier: 12, Prompt: coding, Skipped: true},
		{Tier: 5, Prompt: maths, Pass: true},
		{Tier: 5, Prompt: maths, Pass: true},
	}
	line := benchCategorySummary(results, nil)
	if !contains(line, "coding 50% (2/4 measured)") {
		t.Errorf("partial coverage is not reported: %q — 50%% of two questions renders identically to "+
			"50%% of four", line)
	}
	if contains(line, "maths 100% (") {
		t.Errorf("a fully measured category carries a coverage suffix it does not need: %q", line)
	}
}

// The worst form of the outage-as-weakness bug: every question in a category
// errored, so there is no measurement at all, and Percent's zero is the zero
// value of an int rather than a score anyone earned. Rendering it as "0%" is
// what a reader would act on — and would be reading a sandbox outage as a model
// that cannot code.
func TestBenchCategorySummaryNeverRendersAnOutageAsZero(t *testing.T) {
	coding := "This Go program prints one number. What is it?"
	maths := "What is the sum of all three-digit positive integers that are divisible by 7?"
	results := []BenchResult{
		{Tier: 12, Prompt: coding, Errored: true},
		{Tier: 12, Prompt: coding, Errored: true},
		{Tier: 5, Prompt: maths, Pass: true},
	}
	line := benchCategorySummary(results, nil)
	if contains(line, "coding 0%") {
		t.Errorf("an unmeasured category rendered as a score of zero: %q", line)
	}
	if !contains(line, "coding n/a (0/2 measured)") {
		t.Errorf("an unmeasured category must say so: %q", line)
	}

	// Same rule on the far side of the arrow: a no-think run that was STORED but
	// whose questions all errored is still not a zero.
	nothink := []BenchResult{
		{Tier: 12, Prompt: coding, Pass: true},
		{Tier: 12, Prompt: coding, Pass: true},
		{Tier: 5, Prompt: maths, Errored: true},
	}
	line = benchCategorySummary(results, nothink)
	if contains(line, "maths 100%→0%") {
		t.Errorf("an unmeasured no-think category rendered as a score of zero: %q", line)
	}
	if !contains(line, "maths 100%→n/a") {
		t.Errorf("no-think side must report the absence of a measurement: %q", line)
	}
}
