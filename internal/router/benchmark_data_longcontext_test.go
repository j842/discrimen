package router

import (
	"fmt"
	"hash/fnv"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// The tests below all work by PARSING the rendered prompt back out, rather than
// by consulting longCtxDoc's fields. That is the point of them: a generator that
// computes an answer and a renderer that prints a different document would agree
// with each other perfectly and ship a bank where nine questions have keys no
// model could ever produce. Reading the answer off the page is the only check
// that catches it.

var longCtxRecordRe = regexp.MustCompile(`(?m)^\[(\d{4})\] unit=(\S+) keeper=(\S+) t=(\d{5}) status=(\S+) delta=([+-]\d+)$`)

type longCtxParsed struct {
	Index  int
	Unit   string
	Keeper string
	T      int
	Status string
	Delta  int
}

// longCtxParse reads every record back out of a rendered prompt.
func longCtxParse(t *testing.T, prompt string) []longCtxParsed {
	t.Helper()
	var out []longCtxParsed
	for _, m := range longCtxRecordRe.FindAllStringSubmatch(prompt, -1) {
		idx, err := strconv.Atoi(m[1])
		if err != nil {
			t.Fatalf("record index %q: %v", m[1], err)
		}
		tt, err := strconv.Atoi(m[4])
		if err != nil {
			t.Fatalf("record t %q: %v", m[4], err)
		}
		d, err := strconv.Atoi(m[6])
		if err != nil {
			t.Fatalf("record delta %q: %v", m[6], err)
		}
		out = append(out, longCtxParsed{Index: idx, Unit: m[2], Keeper: m[3], T: tt, Status: m[5], Delta: d})
	}
	if len(out) == 0 {
		t.Fatal("no records parsed out of the prompt — the render format and this test have diverged")
	}
	return out
}

// longCtxRosterFrom reads the unit -> keeper table out of the prompt's header,
// so the contradiction check uses the roster the MODEL is shown rather than the
// package-level one.
func longCtxRosterFrom(t *testing.T, prompt string) map[string]string {
	t.Helper()
	re := regexp.MustCompile(`(?m)^  (RX-\d+)  keeper=(\S+)$`)
	out := map[string]string{}
	for _, m := range re.FindAllStringSubmatch(prompt, -1) {
		out[m[1]] = m[2]
	}
	if len(out) != len(longCtxRoster) {
		t.Fatalf("parsed %d roster entries, want %d", len(out), len(longCtxRoster))
	}
	return out
}

// A profile is only comparable to another profile measured on the SAME
// questions: autoTargetQuality reads every score as one absolute 0-100 bar, a
// graded answer is permacached under benchQuestionQID (which is the prompt, the
// answer, the match mode and the grader version), and benchQuestionHash orders
// questions within a tier by the content of the prompt. A generator that produced
// different bytes on two runs would therefore change the question set — and the
// order it is asked in — silently, with every number still looking plausible, and
// would re-generate nine 48K prompts on every profile because nothing cached
// could ever match. These hashes are the thing that says it did not.
//
// If this test fails after a deliberate edit to longCtxBuild, pasting in the new
// hashes is the right fix — the edit gives those three questions new qids, so
// they are re-asked on their own and the rest of the bank is untouched. What it
// is NOT is free: each re-ask is a 48K prefill per worker per mode. Check that
// the edit was worth that before accepting it.
var longCtxGolden = []struct {
	Tier   int
	Match  string
	Expect string
	Hash   uint64 // FNV-64a of the prompt
}{
	{6, "numeric", "51", 0x35e407cc69ed493c},
	{6, "mcq", "C", 0xe1898b006a56e955},
	{6, "numeric", "93", 0x97a798e18d3f75b2},
	{9, "numeric", "64", 0x5333cb3c89df863d},
	{9, "mcq", "J", 0x62e81d170f6a3cf7},
	{9, "numeric", "302", 0xbb95e135574aaaef},
	{10, "numeric", "63", 0xa7886892c0bd0de0},
	{10, "mcq", "F", 0x3a7010fc8b574516},
	{10, "numeric", "968", 0x31e4f4e86c169de6},
}

func longCtxHash(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}

func TestLongContextDeterministic(t *testing.T) {
	a, b := longCtxQuestions(), longCtxQuestions()
	if len(a) != len(b) || len(a) != len(longCtxGolden) {
		t.Fatalf("built %d and %d questions, golden table has %d", len(a), len(b), len(longCtxGolden))
	}
	for i := range a {
		if a[i].Prompt != b[i].Prompt {
			t.Errorf("question %d: two calls produced different prompts (%d vs %d bytes)",
				i, len(a[i].Prompt), len(b[i].Prompt))
		}
		if a[i].Expect != b[i].Expect {
			t.Errorf("question %d: two calls produced different answers %q vs %q", i, a[i].Expect, b[i].Expect)
		}
		g := longCtxGolden[i]
		if a[i].Tier != g.Tier || a[i].Match != g.Match || a[i].Expect != g.Expect {
			t.Errorf("question %d is {tier %d, %s, %q}, golden says {tier %d, %s, %q}",
				i, a[i].Tier, a[i].Match, a[i].Expect, g.Tier, g.Match, g.Expect)
		}
		if h := longCtxHash(a[i].Prompt); h != g.Hash {
			t.Errorf("question %d prompt hash is %#016x, golden says %#016x — the question set changed; "+
				"bump benchmarkVersion in the same commit or the fleet is graded on two different banks",
				i, h, g.Hash)
		}
	}
	// Byte-identical across a fresh generator too, not just across two calls that
	// happen to share package state.
	for _, rung := range longCtxRungs {
		x, y := longCtxBuild(rung.Seed, rung.Chars), longCtxBuild(rung.Seed, rung.Chars)
		if x.Body != y.Body {
			t.Errorf("rung %dK: two builds produced different bodies", rung.Tokens/1000)
		}
	}
}

// Every answer must be a fact about the document, re-derivable by reading it.
// This is the check that a question is ANSWERABLE at all: an Expect the log does
// not support fails every worker, gets classified "no model can pass this", and
// ships as an item nobody can answer.
func TestLongContextAnswersAreInTheDocument(t *testing.T) {
	for i, q := range longCtxQuestions() {
		recs := longCtxParse(t, q.Prompt)
		switch {
		case strings.Contains(q.Prompt, "Add up the delta values"):
			sum, n := 0, 0
			for _, r := range recs {
				if r.Status == "fault" {
					sum += r.Delta
					n++
				}
			}
			if n != longCtxFaults {
				t.Errorf("question %d: %d fault records in the log, the question implies %d", i, n, longCtxFaults)
			}
			if got := strconv.Itoa(sum); got != q.Expect {
				t.Errorf("question %d: the log's fault deltas sum to %s, Expect says %s", i, got, q.Expect)
			}
		case strings.Contains(q.Prompt, "THIRD record in that order"):
			var esc []longCtxParsed
			for _, r := range recs {
				if r.Status == "escalated" {
					esc = append(esc, r)
				}
			}
			if len(esc) != longCtxEscalations {
				t.Fatalf("question %d: %d escalated records, want %d", i, len(esc), longCtxEscalations)
			}
			sort.Slice(esc, func(a, b int) bool { return esc[a].T < esc[b].T })
			for j := 1; j < len(esc); j++ {
				if esc[j].T == esc[j-1].T {
					t.Errorf("question %d: two escalated records share t=%d, so \"the third\" is ambiguous", i, esc[j].T)
				}
			}
			seen := map[string]bool{}
			for _, e := range esc {
				if seen[e.Keeper] {
					t.Errorf("question %d: keeper %s files two escalated records, so the answer is ambiguous", i, e.Keeper)
				}
				seen[e.Keeper] = true
			}
			// The key letter has to name the third-earliest keeper in the option
			// list the prompt actually printed.
			want := longCtxOptionFor(t, q.Prompt, q.Expect)
			if want != esc[2].Keeper {
				t.Errorf("question %d: option %s is %q, but the third-earliest escalated keeper is %q",
					i, q.Expect, want, esc[2].Keeper)
			}
			if q.Expect == "I" {
				t.Errorf("question %d keys on option I; benchLetterRe cannot read a bare \"I\" as a pick", i)
			}
		case strings.Contains(q.Prompt, "disagrees with the roster"):
			roster := longCtxRosterFrom(t, q.Prompt)
			var bad []int
			for _, r := range recs {
				if k, ok := roster[r.Unit]; ok && k != r.Keeper {
					bad = append(bad, r.Index)
				}
			}
			if len(bad) != 1 {
				t.Fatalf("question %d: %d records contradict the roster, the question promises exactly one: %v", i, len(bad), bad)
			}
			if got := strconv.Itoa(bad[0]); got != q.Expect {
				t.Errorf("question %d: record %s contradicts the roster, Expect says %s", i, got, q.Expect)
			}
		default:
			t.Errorf("question %d matched no known long-context family — a new family needs a case here "+
				"or it ships ungrounded: %s", i, benchSnippet(q.Prompt))
		}
	}
}

// longCtxOptionFor returns the option text the prompt printed against a letter.
func longCtxOptionFor(t *testing.T, prompt, letter string) string {
	t.Helper()
	re := regexp.MustCompile(`(?m)^([A-J])\) (\S+)$`)
	for _, m := range re.FindAllStringSubmatch(prompt, -1) {
		if m[1] == letter {
			return m[2]
		}
	}
	t.Fatalf("no option %q in the prompt", letter)
	return ""
}

// THE ANSWER MUST REQUIRE THE LONG INPUT. A question a model can answer by
// ignoring the context and guessing measures nothing, and one it can answer from
// the first screenful duplicates contextprobe.go's needle test.
//
// Measured two ways, because "long" has two failure modes. DEPTH: how far into
// the log the last record the answer depends on sits — if that is 30%, a model
// that stops reading at half way still answers. TRUNCATION: what the same
// derivation yields over only the first half of the records — it must be a
// DIFFERENT answer, or the second half of the window was decoration.
func TestLongContextAnswersRequireTheWholeLog(t *testing.T) {
	const minDepth = 0.50
	for i, q := range longCtxQuestions() {
		recs := longCtxParse(t, q.Prompt)
		n := len(recs)
		half := recs[:n/2]
		var depth int
		var truncated string
		switch {
		case strings.Contains(q.Prompt, "Add up the delta values"):
			sum := 0
			for _, r := range recs {
				if r.Status == "fault" {
					depth = r.Index
				}
			}
			for _, r := range half {
				if r.Status == "fault" {
					sum += r.Delta
				}
			}
			truncated = strconv.Itoa(sum)
		case strings.Contains(q.Prompt, "THIRD record in that order"):
			// You cannot know WHICH three escalated records are earliest until you
			// have seen all nine, so the depth is the last one of them.
			for _, r := range recs {
				if r.Status == "escalated" {
					depth = r.Index
				}
			}
			var seen []longCtxParsed
			for _, r := range half {
				if r.Status == "escalated" {
					seen = append(seen, r)
				}
			}
			sort.Slice(seen, func(a, b int) bool { return seen[a].T < seen[b].T })
			truncated = "(too few escalated records to have a third)"
			if len(seen) >= 3 {
				truncated = seen[2].Keeper
			}
		case strings.Contains(q.Prompt, "disagrees with the roster"):
			roster := longCtxRosterFrom(t, q.Prompt)
			for _, r := range recs {
				if k, ok := roster[r.Unit]; ok && k != r.Keeper {
					depth = r.Index
				}
			}
			truncated = "(no contradiction in the first half)"
			for _, r := range half {
				if k, ok := roster[r.Unit]; ok && k != r.Keeper {
					truncated = strconv.Itoa(r.Index)
				}
			}
		}
		if got := float64(depth) / float64(n); got < minDepth {
			t.Errorf("question %d: its answer is settled by record %d of %d (%.0f%% in), so a model that "+
				"stopped reading half way would still answer it", i, depth, n, 100*got)
		}
		// The truncated derivation is compared against the ANSWER KEY, so this also
		// catches a question whose two halves happen to give the same total.
		if truncated == q.Expect {
			t.Errorf("question %d: reading only the first %d of %d records yields the same answer (%s) — "+
				"the rest of the window is decoration", i, n/2, n, q.Expect)
		}
		// For the mcq family the key is a LETTER and the derivation above yields a
		// keeper name, so compare through the option table too.
		if q.Match == "mcq" && truncated == longCtxOptionFor(t, q.Prompt, q.Expect) {
			t.Errorf("question %d: the first half of the log already names the third-earliest keeper (%s)",
				i, truncated)
		}
	}
}

// The rungs have to be far enough apart, and actually as long as they claim, or
// the ladder cannot distinguish a model that degrades at 16K from one that holds
// to 48K — which is the only reason there are three of them.
func TestLongContextRungLengths(t *testing.T) {
	qs := longCtxQuestions()
	if len(qs) != len(longCtxRungs)*3 {
		t.Fatalf("%d questions for %d rungs", len(qs), len(longCtxRungs))
	}
	for i, q := range qs {
		rung := longCtxRungs[i/3]
		est := benchPromptTokenEstimate(q.Prompt)
		// ±10%: the record count is chosen from a rendered length, and stamping the
		// special statuses afterwards moves it by a few characters.
		if lo, hi := rung.Tokens*9/10, rung.Tokens*11/10; est < lo || est > hi {
			t.Errorf("question %d is ~%d tokens, its rung claims ~%d (allowed %d–%d)",
				i, est, rung.Tokens, lo, hi)
		}
		if q.Tier != rung.Tier {
			t.Errorf("question %d is tier %d, its rung says %d", i, q.Tier, rung.Tier)
		}
	}
	for i := 1; i < len(longCtxRungs); i++ {
		if longCtxRungs[i].Tokens <= longCtxRungs[i-1].Tokens*2 {
			t.Errorf("rung %d (%d tokens) is less than double rung %d (%d) — two rungs this close "+
				"will read the same on any real worker",
				i, longCtxRungs[i].Tokens, i-1, longCtxRungs[i-1].Tokens)
		}
		if longCtxRungs[i].Tier <= longCtxRungs[i-1].Tier {
			t.Errorf("rung %d is longer than rung %d but not a higher tier (%d vs %d)",
				i, i-1, longCtxRungs[i].Tier, longCtxRungs[i-1].Tier)
		}
	}
}

// The long-context questions have to survive the machinery the rest of the bank
// goes through: the production grader, the category map, and the ordering hash.
func TestLongContextFitsTheBank(t *testing.T) {
	qs := longCtxQuestions()
	for i, q := range qs {
		if !checkAnswer(q, q.Expect) {
			t.Errorf("question %d does not grade its own answer %q", i, q.Expect)
		}
		if !checkAnswer(q, "After working through the log, the answer is "+q.Expect+".") {
			t.Errorf("question %d does not grade a declared answer of %q", i, q.Expect)
		}
		if q.Tier >= benchInsightTier {
			t.Errorf("question %d is tier %d: benchWeightedScore would put it in the insight or coding "+
				"bucket and dilute the weight that exists to measure those", i, q.Tier)
		}
		if _, ok := benchTierCategory[q.Tier]; !ok {
			t.Errorf("question %d uses tier %d, which benchTierCategory does not map — it would "+
				"categorise as unknown", i, q.Tier)
		}
		// benchCategoryOf checks the LiveBench markers first and benchCodeRe
		// second. A synthetic log full of systems vocabulary is an easy way to get
		// a long-context reasoning measurement filed under "coding".
		if m := benchCategoryOfLive(q.Prompt); m != "" {
			t.Errorf("question %d matches a LiveBench format marker and would be categorised %q", i, m)
		}
		if benchCodeRe.MatchString(q.Prompt) {
			t.Errorf("question %d matches benchCodeRe (%q) and would be filed as coding",
				i, benchCodeRe.FindString(q.Prompt))
		}
		if got := benchCategoryOf(q.Tier, q.Prompt); got != benchCatReasoning {
			t.Errorf("question %d categorises as %q, want %q", i, got, benchCatReasoning)
		}
	}
	// Distinct content hashes, or benchStratifiedOrder's within-tier sort is
	// unstable between them and two profiles of one worker ask in different orders.
	seen := map[uint64]int{}
	for i, q := range qs {
		h := benchQuestionHash(q)
		if j, dup := seen[h]; dup {
			t.Errorf("questions %d and %d share a content hash", j, i)
		}
		seen[h] = i
	}
	// And they must actually be IN the bank, appended by init().
	inBank := 0
	for _, q := range benchmarkQuestions {
		if strings.HasPrefix(q.Prompt, longCtxPreamble) {
			inBank++
		}
	}
	if inBank != len(qs) {
		t.Errorf("%d long-context questions in benchmarkQuestions, generated %d", inBank, len(qs))
	}
}

// A hand check that the generated prose is varied. A log of a thousand identical
// lines is a compression exercise: a model can skim it, which would make the
// rungs measure something much easier than they claim to.
func TestLongContextNotesAreVaried(t *testing.T) {
	d := longCtxBuild(longCtxRungs[2].Seed, longCtxRungs[2].Chars)
	notes := map[string]bool{}
	for _, r := range d.Records {
		notes[r.Note] = true
	}
	if ratio := float64(len(notes)) / float64(len(d.Records)); ratio < 0.75 {
		t.Errorf("only %d distinct notes across %d records (%.0f%%) — the log is repetitive enough to skim",
			len(notes), len(d.Records), 100*ratio)
	}
	if testing.Verbose() {
		fmt.Printf("%d distinct notes over %d records\n", len(notes), len(d.Records))
	}
}
