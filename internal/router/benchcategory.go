package router

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
)

// A per-CATEGORY view of the quality benchmark: what a worker is good at, as
// opposed to how hard the questions it answered were.
//
// The tiers already published by the benchmark are a difficulty ladder, and they
// answer "how far up does this worker get". They cannot answer the question an
// operator actually asks before handing a worker a job — "can it read code?" —
// because difficulty and ability are orthogonal axes: tier 12 is programming,
// but a LiveBench maths item calibrated into tier 12 sits there too, and tier 4
// deliberately mixes compiler gotchas in with arithmetic word problems. Reading
// t4=12/25 tells you nothing about which half was missed.
//
// The categories are the three abilities the question set actually spans, plus
// a bucket for the control questions that measure none of them:
//
//	coding     read a program exactly and say what it does
//	maths      arithmetic, number theory, combinatorics, competition maths
//	reasoning  traps, world models, rules invented in the prompt, constraint puzzles
//	general    tiers 1-2, the controls and the floor — deliberately not discriminators
//
// WHY THE CATEGORY IS DERIVED AND NOT STORED. benchmarkQ carries no Category
// field and this file does not add one. Half the question set is generated
// (benchmark_data_live.go, written by `bench emit` from LiveBench), so a stored
// field would have to be threaded through the generator, and the generator's
// output is part of the PROFILE CACHE KEY: LoadWorkerProfile keys on
// benchmarkVersion, and anything that changes the question set without changing
// that version silently compares workers graded on different sets against one
// absolute 0-100 scale (see benchgen.go's "WHY THE OUTPUT IS COMMITTED"). A
// derived category has none of that exposure — it changes no bytes, invalidates
// no profile, and can be corrected in a patch release without re-benchmarking a
// fleet. The cost is that this file has to recognise questions from their text,
// which is what the rest of it is about.

const (
	benchCatCoding    = "coding"
	benchCatMaths     = "maths"
	benchCatReasoning = "reasoning"
	benchCatGeneral   = "general"
	// benchCatUnknown is a question this file could not place. It is deliberately
	// a VISIBLE bucket rather than a silent fold into one of the four: a question
	// landing here means a new tier or a new LiveBench task arrived and nobody
	// said what it measures, and a row labelled "unclassified" on the dashboard is
	// the right way to find that out. TestBenchCategoryPlacesEveryQuestion fails
	// CI first, so it should never reach a deployment.
	benchCatUnknown = "unclassified"
)

// benchCategoryOrder is the display order: the three abilities strongest-signal
// first, controls last. Sorting alphabetically would put the meaningless bucket
// at the top of the table.
var benchCategoryOrder = []string{benchCatCoding, benchCatMaths, benchCatReasoning, benchCatGeneral, benchCatUnknown}

// benchLiveMarkers identify a question sourced from LiveBench, and therefore its
// category, by the answer-format boilerplate its task appends to every item.
//
// Prose matching looks fragile and is the most robust signal available here.
// These prompts are MACHINE-GENERATED: every math_comp AMC item ends with the
// same "duplicate that letter five times" instruction, every spatial item with
// the same "**bold**" instruction, because a benchmark harness pastes them on.
// Nothing in benchmark_data.go's hand-authored set says any of it, so a match is
// conclusive in both directions.
//
// The alternative was reading the task name out of the comment `bench emit`
// writes above each question — which is a comment, gone by compile time, and
// unreachable from a running router. The other alternative, keying on the
// question's LiveBench id, needs benchdata/pool.json, which is a developer-side
// file that the container never sees.
//
// ORDER MATTERS, and one measured collision is why: an LCB_generation coding
// problem is about "people standing in a line", which is also the zebra_puzzle
// marker. Coding is checked first because its marker is a header the generator
// puts at the very start of the prompt, so it cannot be a coincidence, whereas
// the others can appear inside a question's own text.
//
// TestBenchCategoryRecognisesEveryLiveQuestion checks this table against the
// pool the generator emits from, so a LiveBench format change fails CI rather
// than quietly filing a hundred maths questions under whatever tier they landed
// in.
var benchLiveMarkers = []struct {
	marker   string
	category string
}{
	{"### Instructions: You are an expert Python programmer.", benchCatCoding}, // LCB_generation, coding_completion
	{"duplicate that letter five times", benchCatMaths},                        // math_comp, AMC half
	{"an integer consisting of exactly 3 digits", benchCatMaths},               // math_comp, AIME half
	{"masked out using the tag <missing", benchCatMaths},                       // olympiad
	{"put your answer in **bold** as a single ", benchCatReasoning},            // spatial (…integer / …phrase)
	{"people standing in a line", benchCatReasoning},                           // zebra_puzzle, both premise formats
}

// benchCodeRe spots a hand-authored question that asks the model to READ a
// program. It exists because tier is not enough: tier 12 is entirely code, but
// tier 4 deliberately mixes compiler and shell gotchas in with arithmetic word
// problems, and filing those under maths would hide the one measurement the
// coding category exists to make.
//
// Only languages the set actually uses, and the one-letter names ("Go", "C")
// only in "X program" — a bare \bgo\b matches half the reasoning tier ("how many
// ways can you go…") and a bare \bc\b matches an MCQ option, either of which
// would file world-model traps as coding. Matching a language name is otherwise
// enough precisely because this regex only ever sees hand-authored prompts: the
// LiveBench markers above are checked first, so a maths question that mentions
// Python cannot reach here.
//
// The "c++" branch is separate because it cannot end in \b: the character before
// the following space is '+', which is not a word character, so \b never
// matches there and every C++ trace in tier 4 was silently filed as maths.
var benchCodeRe = regexp.MustCompile(`(?i)\b(bash|python|golang|kotlin|javascript|typescript)\b|\bc\+\+|\b(?:go|c) program\b`)

// benchTierCategory is what each HAND-AUTHORED tier measures, taken from the
// charter benchmark_data.go writes above each one. Tier is the right signal for
// this half of the set and the wrong one for the generated half: these tiers
// were AUTHORED as ability bands ("traps", "unrecallable", "programming"), while
// a generated question's tier is a measured fleet pass rate with no ability
// meaning at all.
//
// Three of these are judgements rather than facts, and are recorded as such:
//
//	tier 4  is mostly multi-step arithmetic, so it is maths — but it also holds
//	        two chemistry recall items and three irrelevant-clause traps. The code
//	        items in it are caught by benchCodeRe before this table is consulted.
//	tier 8  is "frontier maths" and is mostly modular arithmetic and probability,
//	        but the marble-in-a-glass and portrait riddles in it are world-model
//	        traps wearing a maths tier.
//	tier 9  is arithmetic-SHAPED — invented operators, factorial bases — and is
//	        reasoning anyway, because what it measures is executing a rule stated
//	        in the prompt. Filing it by the shape of its answers rather than by
//	        what makes it hard would make the maths column meaningless.
//
// A tier absent from this table yields benchCatUnknown rather than a guess. Adding
// tier 13 should mean adding a line here, and TestBenchCategoryCoversEveryTier is
// what says so.
var benchTierCategory = map[int]string{
	1:  benchCatGeneral,   // controls — every working model passes
	2:  benchCatGeneral,   // floor — the weakest model fails, every competent one passes
	3:  benchCatReasoning, // careful-reading traps
	4:  benchCatMaths,     // multi-step arithmetic (+ the code gotchas benchCodeRe takes)
	5:  benchCatMaths,     // number theory, multi-step word problems
	6:  benchCatReasoning, // misleading-classic traps
	7:  benchCatMaths,     // combinatorics, sequences, digit problems
	8:  benchCatMaths,     // modular arithmetic, probability
	9:  benchCatReasoning, // rules invented in the prompt, priors to override, traces to audit
	10: benchCatReasoning, // SimpleBench-style world-model traps
	11: benchCatMaths,     // budget-bounded insight: digit/base enumeration with a closed form
	12: benchCatCoding,    // programming — trace a short program, give its exact output
}

// benchCategoryOf places one question. Three rules, in the order they are
// checked and for the reasons above: a LiveBench format marker settles the
// generated half outright, a language name settles a hand-authored code trace,
// and what is left is hand-authored and takes its tier's charter.
func benchCategoryOf(tier int, prompt string) string {
	for _, m := range benchLiveMarkers {
		if strings.Contains(prompt, m.marker) {
			return m.category
		}
	}
	if benchCodeRe.MatchString(prompt) {
		return benchCatCoding
	}
	if cat, ok := benchTierCategory[tier]; ok {
		return cat
	}
	return benchCatUnknown
}

// benchCatScore is one category's (or one tier-within-a-category's) tally in one
// thinking mode.
//
// Strict and Loose are kept apart and Passed folds them the way the headline
// score does — a loose pass is the right answer in the wrong format and earns
// half a point (v37, see checkAnswerLoose). Reporting only the total would hide
// the thing the loose tally was added to expose: a worker whose category score
// is really a formatting cost rather than a knowledge one.
type benchCatScore struct {
	Strict  int     `json:"strict"`  // graded correct outright
	Loose   int     `json:"loose"`   // right answer, ignored the requested format — half credit
	Total   int     `json:"total"`   // every question in the category, failures and errors included
	Passed  float64 `json:"passed"`  // Strict + Loose/2, the arithmetic benchWeightedScore uses
	Percent int     `json:"percent"` // 100*Passed/Total, rounded
}

func (s *benchCatScore) add(r BenchResult) {
	// A question the profile budget never asked is not a category weakness. It
	// has to stay out of the DENOMINATOR as well as the numerator: counting it in
	// Total alone reads as a miss, so a truncated profile would report a model as
	// bad at whichever categories it ran out of time in — the exact opposite of a
	// strengths-and-weaknesses map, and worst on the slow workers that get
	// truncated most.
	if r.Skipped {
		return
	}
	s.Total++
	switch {
	case r.Pass:
		s.Strict++
		s.Passed++
	case r.Loose:
		s.Loose++
		s.Passed += 0.5
	}
}

func (s *benchCatScore) finish() {
	if s.Total > 0 {
		s.Percent = int(math.Round(100 * s.Passed / float64(s.Total)))
	}
}

// benchCatTier is one tier's contribution to one category — the detail behind a
// category row, and the only place the difficulty axis is still visible. A
// worker at 60% on maths reads very differently when the misses are all tier 8
// than when they are spread from tier 4 up.
type benchCatTier struct {
	Tier    int            `json:"tier"`
	Think   benchCatScore  `json:"think"`
	NoThink *benchCatScore `json:"nothink,omitempty"`
}

// benchCatBreakdown is one category, in both thinking modes.
//
// NoThink is a POINTER and is nil when the run that produced this profile did
// not keep its no-think per-question results. Zero would be indistinguishable
// from a worker that got everything wrong with thinking off, which is a real and
// common outcome — the gap this whole view exists to show is a measured 93/41 on
// one live worker — so "not stored" has to be a different value, not a small one.
type benchCatBreakdown struct {
	Category string         `json:"category"`
	Think    benchCatScore  `json:"think"`
	NoThink  *benchCatScore `json:"nothink,omitempty"`
	Tiers    []benchCatTier `json:"tiers"`
}

// benchCategoryBreakdown groups a stored profiling run by category.
//
// think is the mixed run's per-question results (BenchResults) — the score the
// router's own traffic experiences, hard tiers answered with a scratchpad.
// nothink is the same questions answered with thinking suppressed, which is what
// a client forcing requirements.thinking="off" actually talks to. Both matter
// and they are not close: the point of the two-score benchmark is that a worker
// can be excellent reasoning and useless without it, and a per-category view is
// where that becomes actionable — a worker may hold its maths thinking-off and
// lose all of its coding.
//
// nothink is matched to think BY INDEX, the same alignment the rest of the
// benchmark uses (both slices are built by walking benchmarkQuestions in order —
// see runQualityBenchmark and needsNoThinkBackfill). A length mismatch means the
// two runs graded different sets, so the no-think half is dropped entirely
// rather than zipped up wrongly: a per-category number silently built from
// mismatched pairs would look exactly as plausible as a correct one.
func benchCategoryBreakdown(think, nothink []BenchResult) []benchCatBreakdown {
	if len(nothink) != len(think) {
		nothink = nil
	}
	type acc struct {
		think   benchCatScore
		nothink benchCatScore
		tiers   map[int]*benchCatTier
	}
	cats := map[string]*acc{}
	order := []string{}
	for i, r := range think {
		cat := benchCategoryOf(r.Tier, r.Prompt)
		a := cats[cat]
		if a == nil {
			a = &acc{tiers: map[int]*benchCatTier{}}
			cats[cat] = a
			order = append(order, cat)
		}
		a.think.add(r)
		t := a.tiers[r.Tier]
		if t == nil {
			t = &benchCatTier{Tier: r.Tier}
			a.tiers[r.Tier] = t
		}
		t.Think.add(r)
		if nothink == nil {
			continue
		}
		a.nothink.add(nothink[i])
		if t.NoThink == nil {
			t.NoThink = &benchCatScore{}
		}
		t.NoThink.add(nothink[i])
	}
	// Canonical order first, then anything this file does not know about — a new
	// category added below without being listed still renders rather than
	// vanishing.
	sort.SliceStable(order, func(i, j int) bool {
		return benchCategoryRank(order[i]) < benchCategoryRank(order[j])
	})
	out := make([]benchCatBreakdown, 0, len(order))
	for _, cat := range order {
		a := cats[cat]
		a.think.finish()
		row := benchCatBreakdown{Category: cat, Think: a.think}
		if nothink != nil {
			a.nothink.finish()
			nt := a.nothink
			row.NoThink = &nt
		}
		for _, t := range a.tiers {
			t.Think.finish()
			if t.NoThink != nil {
				t.NoThink.finish()
			}
			row.Tiers = append(row.Tiers, *t)
		}
		sort.Slice(row.Tiers, func(i, j int) bool { return row.Tiers[i].Tier < row.Tiers[j].Tier })
		out = append(out, row)
	}
	return out
}

func benchCategoryRank(cat string) int {
	for i, c := range benchCategoryOrder {
		if c == cat {
			return i
		}
	}
	return len(benchCategoryOrder)
}

// benchCategorySummary renders the breakdown as one compact line for a LIST
// view: "coding 47%→11%  maths 90%→85%  reasoning 78%→31%".
//
// It exists so `ask -l` can show where a worker's quality comes from without a
// round trip per backend. The full breakdown lives on
// GET /backends/{id}/benchmark, which is per-worker by nature; asking a list
// command to fetch it N times to render one line each is the wrong trade, and
// putting the whole structure on /backends instead would bloat a payload the
// dashboard polls every ten seconds.
//
// The arrow is the thinking-on → thinking-off pair, which is the number worth
// seeing at a glance: a category with a small gap is a genuine ability, a large
// one is a scratchpad doing the work. When the no-think run's per-question
// results were not stored the arrow is omitted rather than shown as →0 — a
// worker really can score zero with thinking off, so "not measured" and
// "measured as nothing" must not render alike.
//
// Computed ONCE when a profile is applied and carried on the row, not derived
// per request: the inputs only change when the worker is re-profiled.
func benchCategorySummary(think, nothink []BenchResult) string {
	cats := benchCategoryBreakdown(think, nothink)
	if len(cats) == 0 {
		return ""
	}
	var b strings.Builder
	for _, c := range cats {
		if b.Len() > 0 {
			b.WriteString("  ")
		}
		fmt.Fprintf(&b, "%s %d%%", c.Category, c.Think.Percent)
		if c.NoThink != nil {
			fmt.Fprintf(&b, "→%d%%", c.NoThink.Percent)
		}
	}
	return b.String()
}
