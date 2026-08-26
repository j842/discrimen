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
