package router

import (
	"context"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// timeoutErr is a net.Error reporting a timeout, like an http client/context deadline.
type timeoutErr struct{}

func (timeoutErr) Error() string   { return "i/o timeout" }
func (timeoutErr) Timeout() bool   { return true }
func (timeoutErr) Temporary() bool { return false }

// isTimeout decides speed-fail (the answer deadline fired) vs. transient error (retry):
// only a deadline/net timeout is "too slow"; a transport error or non-2xx HTTP status is not.
func TestIsTimeout(t *testing.T) {
	speedFails := []error{
		context.DeadlineExceeded,
		fmt.Errorf("Post ...: %w", context.DeadlineExceeded), // wrapped, as net/http returns it
		timeoutErr{}, // net.Error with Timeout()==true
	}
	for i, err := range speedFails {
		if !isTimeout(err) {
			t.Errorf("speed-fail case %d: isTimeout(%v) = false, want true", i, err)
		}
	}
	retryable := []error{
		nil,
		errors.New("connection refused"),
		fmt.Errorf("completion returned 500: %s", "internal error"), // HTTP status error must retry, not count as slow
		fmt.Errorf("completion returned 400: %s", "bad request"),
	}
	for i, err := range retryable {
		if isTimeout(err) {
			t.Errorf("retryable case %d: isTimeout(%v) = true, want false", i, err)
		}
	}
}

func TestCheckAnswer(t *testing.T) {
	cases := []struct {
		q    benchmarkQ
		ans  string
		want bool
	}{
		{benchmarkQ{Match: "numeric", Expect: "391"}, "17 times 23 is 391.", true},
		{benchmarkQ{Match: "numeric", Expect: "391"}, "17×23 = 392", false},
		{benchmarkQ{Match: "numeric", Expect: "1024"}, "2^10 = 1,024", true}, // commas stripped
		{benchmarkQ{Match: "numeric", Expect: "84"}, "average speed is 84 km/h", true},
		{benchmarkQ{Match: "mcq", Expect: "C"}, "The answer is C.", true},
		{benchmarkQ{Match: "mcq", Expect: "C"}, "B", false},
		{benchmarkQ{Match: "mcq", Expect: "B"}, "(B)", true},
		{benchmarkQ{Match: "mcq", Expect: "A"}, "Is it B? No — A kilogram of steel is heavier. Answer: A", true}, // anchored declaration ("Answer: A"), not the article "A kilogram"
		{benchmarkQ{Match: "contains", Expect: "Paris"}, "The capital is Paris.", true},
		{benchmarkQ{Match: "contains", Expect: "Paris"}, "It's Lyon, I think.", false},
		{benchmarkQ{Match: "contains", Expect: "Newton"}, "Sir Isaac NEWTON discovered it.", true}, // case-insensitive
		{benchmarkQ{Match: "contains", Expect: "H2O"}, "The formula is H₂O.", true},                // unicode subscript normalized
		{benchmarkQ{Match: "numeric", Expect: "2"}, "x² means x squared", true},                    // unicode superscript normalized
		{benchmarkQ{Match: "contains", Expect: "H2O"}, `$\text{H}_2\text{O}$`, true},               // LaTeX scaffolding stripped
		{benchmarkQ{Match: "contains", Expect: "1/7"}, `the probability is \frac{1}{7}`, true},     // \frac{a}{b} -> a/b
		{benchmarkQ{Match: "numeric", Expect: "6"}, "Six", true},                                   // spelled-out number word
		{benchmarkQ{Match: "numeric", Expect: "6"}, "A hexagon has six sides.", true},
		{benchmarkQ{Match: "numeric", Expect: "1"}, "There are none left.", false}, // "none" must not read as "one"
	}
	for i, c := range cases {
		if got := checkAnswer(c.q, c.ans); got != c.want {
			t.Errorf("case %d: checkAnswer(%s,%q)=%v want %v", i, c.q.Match, c.ans, got, c.want)
		}
	}
}

// The echo-trap questions (Mary / Johnny / Everest) carry their expected token in the
// prompt itself, so they grade with "final-contains": a wrong answer that merely
// restates the premise must FAIL, while the token asserted as the answer must PASS.
func TestCheckAnswerFinalContains(t *testing.T) {
	cases := []struct {
		expect string
		ans    string
		want   bool
	}{
		// verified false-passes under plain "contains" — premise echoes must fail
		{"Mary", "Mary's father's fifth daughter must be Lulu.", false},
		{"Johnny", "Johnny's mother's fourth child was named July.", false},
		{"Everest", "Before Mount Everest was discovered, K2 was the tallest mountain.", false},
		{"Everest", "K2", false},
		{"Mary", "Mary’s father has five daughters, so the fifth is Lulu.", false}, // curly possessive is still possessive
		{"Mary", "It is Lulu, not Mary.", false},                                   // negated token is not an answer
		// correct answers in natural phrasing — must pass
		{"Mary", "The answer is Mary.", true},
		{"Mary", "Mary", true},
		{"Mary", "It's Mary herself.", true},
		{"Mary", "Mary, of course.", true}, // terse leading clause is the stated answer
		{"Mary", "Mary. She is the fifth daughter.", true},
		{"Johnny", "The fourth child was Johnny", true},
		{"Everest", "Mount Everest — it just hadn't been discovered yet.", true},
		{"Everest", "It was still Mount Everest.", true},
	}
	for i, c := range cases {
		q := benchmarkQ{Match: "final-contains", Expect: c.expect}
		if got := checkAnswer(q, c.ans); got != c.want {
			t.Errorf("case %d: final-contains %q on %q = %v, want %v", i, c.expect, c.ans, got, c.want)
		}
	}
}

// A declared pick followed by a plain word ("The answer is C because…") must anchor —
// the last-letter fallback used to grade the trailing "B" — while the lowercase
// article in "the answer is a syntax error" must still anchor nothing.
func TestCheckAnswerMCQDeclaredBoundary(t *testing.T) {
	cases := []struct {
		expect string
		ans    string
		want   bool
	}{
		{"C", "The answer is C because B would be wrong.", true},  // was a false fail: fallback graded the last letter B
		{"B", "The answer is C because B would be wrong.", false}, // …and the same string must no longer pass for B
		{"C", "The answer is a syntax error, so C.", true},        // article "a" must not anchor; the trailing C decides
		{"A", "The answer is a syntax error, so C.", false},
		{"C", "the answer is c", true}, // lowercase still anchors at EOL/punctuation
		{"A", "the answer is a", true},
		{"D", "The answer is D since B overflows.", true},
	}
	for i, c := range cases {
		q := benchmarkQ{Match: "mcq", Expect: c.expect}
		if got := checkAnswer(q, c.ans); got != c.want {
			t.Errorf("case %d: mcq %q on %q = %v, want %v", i, c.expect, c.ans, got, c.want)
		}
	}
}

// The tier-9 questions carry ten options, so the grader must read picks past D. Every
// letter A-J grades through the declared, leading and bare-letter tiers — except "I",
// which is the pronoun far more often than it is a pick: it anchors only when punctuated
// ("Answer: I.", "I)") and a bare trailing "I" must NOT be read as a choice, or every
// reply opening "I need to check…" would grade as picking option I.
func TestCheckAnswerMCQTenOptions(t *testing.T) {
	cases := []struct {
		expect string
		ans    string
		want   bool
	}{
		{"E", "The answer is E.", true},
		{"F", "Answer: F", true},
		{"G", "G) 0.74", true},
		{"H", "…so the packing factor is 0.78 → H", true},
		{"J", "The answer is J (iodide).", true},
		{"J", "Iodide lies lowest in the spectrochemical series, so J.", true},
		{"E", "The answer is J (iodide).", false},
		// "I" only anchors when punctuated…
		{"I", "Answer: I.", true},
		{"I", "I) uncoupling protein 1", true},
		// …and never as the bare pronoun, however the reply is shaped.
		{"I", "I think the answer is C.", false},
		{"C", "I think the answer is C.", true},
		{"I", "I need to work out which complex pumps protons. I", false},
		// A ten-option list enumerated in full still grades by its conclusion.
		{"C", "A) C1 — no.\nB) Cs — no.\nC) C2v — yes, SF4 is see-saw.", true},
		{"A", "A) C1 — no.\nB) Cs — no.\nC) C2v — yes, SF4 is see-saw.", false},
	}
	for i, c := range cases {
		q := benchmarkQ{Match: "mcq", Expect: c.expect}
		if got := checkAnswer(q, c.ans); got != c.want {
			t.Errorf("case %d: mcq %q on %q = %v, want %v", i, c.expect, c.ans, got, c.want)
		}
	}
}

// The expert tiers (9 and 10) are only as precise as their option lists: ten choices put
// the guess floor at 10%, and no question may key on "I", which the grader deliberately
// cannot read as a bare pick (see TestCheckAnswerMCQTenOptions).
func TestExpertTierQuestionShape(t *testing.T) {
	seen := map[int]int{}
	for _, q := range benchmarkQuestions {
		if q.Tier < 9 {
			continue
		}
		seen[q.Tier]++
		if q.Match != "mcq" {
			continue
		}
		if q.Expect == "I" {
			t.Errorf("tier-%d question keys on option I, which cannot be graded as a bare pick: %q", q.Tier, benchSnippet(q.Prompt))
		}
		for _, opt := range []string{"A)", "B)", "C)", "D)", "E)", "F)", "G)", "H)", "I)", "J)"} {
			if !strings.Contains(q.Prompt, opt) {
				t.Errorf("tier-%d mcq is missing option %s (ten options keep the guess floor at 10%%): %q", q.Tier, opt, benchSnippet(q.Prompt))
			}
		}
	}
	for _, tier := range []int{9, 10} {
		if seen[tier] == 0 {
			t.Fatalf("no tier-%d questions found — the expert tiers are what spread the strong models", tier)
		}
	}
}

// Every expert-tier question must grade its own expected answer as CORRECT through the real
// matcher, in the shapes a thinking model actually emits — a declared pick, a bare letter,
// a value with units, a value restated after working. A question whose answer can't be
// recognised scores every worker a fail and silently drags the whole fleet's quality down.
func TestExpertTierAnswersGrade(t *testing.T) {
	for _, q := range benchmarkQuestions {
		if q.Tier < 9 {
			continue
		}
		shapes := []string{
			q.Expect,
			"The answer is " + q.Expect + ".",
		}
		switch q.Match {
		case "mcq":
			shapes = append(shapes, "Working through each option, the correct one is "+q.Expect)
		case "contains":
			shapes = append(shapes, "Applying the rules in order gives "+q.Expect+".")
		default:
			shapes = append(shapes, "Substituting into the formula gives "+q.Expect)
		}
		for _, ans := range shapes {
			if !checkAnswer(q, ans) {
				t.Errorf("tier-%d question does not grade its own answer %q on reply %q: %s", q.Tier, q.Expect, ans, benchSnippet(q.Prompt))
			}
		}
	}
}

// An answer that ENUMERATES the options must not be graded by its first line: the
// leading-letter rule stands aside and the concluding option (or a declared pick)
// decides — restoring what the old last-letter grader got right.
func TestCheckAnswerMCQEnumeration(t *testing.T) {
	const enum = "A) it prints 10 — no. B) it loops forever — no. C) it fails with an error — yes."
	cases := []struct {
		expect string
		ans    string
		want   bool
	}{
		{"C", enum, true},  // was a false fail: the leading-letter rule graded A
		{"A", enum, false}, // …and the same string must no longer pass for A
		{"C", "A) no. B) no. The answer is C.", true},            // a declared pick still beats the enumeration
		{"B", "A. wrong, octal needs 0o.\nB. correct.", true},    // newline-separated enumeration
		{"A", "A. because the leading zero forces octal.", true}, // a single anchor is a lead, not an enumeration
		{"B", "b) it prints 9", true},
	}
	for i, c := range cases {
		q := benchmarkQ{Match: "mcq", Expect: c.expect}
		if got := checkAnswer(q, c.ans); got != c.want {
			t.Errorf("case %d: mcq %q on %q = %v, want %v", i, c.expect, c.ans, got, c.want)
		}
	}
}

// Negated and retracted tokens in final-contains: "it wasn't Everest" and
// "Everest, but actually it was K2." reject the token rather than asserting it,
// while the verified natural-phrasing PASS set must keep passing.
func TestCheckAnswerFinalContainsNegation(t *testing.T) {
	cases := []struct {
		expect string
		ans    string
		want   bool
	}{
		// contracted/adverbial negations near the token must fail, not just "not"
		{"Everest", "The tallest was K2; it wasn't Everest.", false},
		{"Everest", "The tallest was K2; it wasn’t Everest.", false}, // curly apostrophe
		{"Everest", "The tallest was K2; it isn't Everest.", false},
		{"Everest", "K2 was the tallest; it was never Everest.", false},
		// a terse lead the next clause walks back is not a stated answer
		{"Everest", "Everest, but actually it was K2.", false},
		{"Everest", "Everest. No, it was K2.", false},
		// …but negation/retraction elsewhere must not reject a real answer
		{"Everest", "It wasn't K2; it was Everest.", true},
		{"Everest", "It was not K2 but Everest.", true},      // "not" two words back across "but" asserts the token
		{"Mary", "The answer is 'Mary'.", true},              // closing quote is not a possessive
		{"Everest", "K2, before Everest's discovery", false}, // possessive still rejected
		// the verified PASS set from the grading overhaul stays green
		{"Mary", "It's Mary.", true},
		{"Mary", "Mary", true},
		{"Mary", "The answer is Mary", true},
		{"Mary", "Mary is the fifth daughter", true},
		{"Everest", "Everest was.", true},
		{"Mary", "Mary, of course.", true},
		{"Everest", "Mount Everest — it just hadn't been discovered yet.", true},
	}
	for i, c := range cases {
		q := benchmarkQ{Match: "final-contains", Expect: c.expect}
		if got := checkAnswer(q, c.ans); got != c.want {
			t.Errorf("case %d: final-contains %q on %q = %v, want %v", i, c.expect, c.ans, got, c.want)
		}
	}
}

// MCQ letter extraction: anchored declarations and leading letters beat the
// last-standalone-letter fallback, and a bare lowercase "a"/"d" (the article, "I'd")
// never counts as a pick — it used to swallow the real answer.
func TestCheckAnswerMCQ(t *testing.T) {
	cases := []struct {
		expect string
		ans    string
		want   bool
	}{
		{"C", "C. The leading zero makes 09 invalid octal, causing a syntax error.", true}, // article "a" must not override the leading C
		{"B", "the answer is B", true},
		{"A", "A. because the leading zero forces octal.", true},
		{"D", "I'd say D", true},
		{"A", "It causes a crash", false}, // article "a" is not a pick: no candidate letter at all
		{"C", `\boxed{C}`, true},
		{"C", `The answer is \boxed{C}.`, true},
		{"A", "Answer: A — a kilogram of steel weighs the same as a kilogram of feathers.", true}, // trailing articles must not override the declaration
		{"C", "The answer is C because the leading zero is octal.", true},
		{"B", "A) prints 10 — wrong. B) prints 9 — right. The answer is B.", true},
		{"C", "c", true}, // a bare lowercase letter is still a pick
		{"C", "C\nThe leading zero makes it octal.", true},
		{"B", "b) it prints 9", true},
		{"C", "It fails with an error, so C.", true},
	}
	for i, c := range cases {
		q := benchmarkQ{Match: "mcq", Expect: c.expect}
		if got := checkAnswer(q, c.ans); got != c.want {
			t.Errorf("case %d: mcq %q on %q = %v, want %v", i, c.expect, c.ans, got, c.want)
		}
	}
}

// Numeric grades the LAST number, so a self-correction grades by the final value, and
// reads compound number words ("twenty-two") as one number.
func TestCheckAnswerNumericLast(t *testing.T) {
	const selfCorrect = "total distance 200 m, 200/100 = 2... wait, actually the answer is 1"
	cases := []struct {
		expect string
		ans    string
		want   bool
	}{
		{"2", selfCorrect, false}, // the discarded intermediate must not rescue it
		{"1", selfCorrect, true},  // the corrected final value grades
		{"22", "twenty-two", true},
		{"22", "The total is twenty two.", true},
		{"20", "twenty", true},
		{"132", "12*8 = 96, 18*2 = 36, 96+36 = 132", true},
		{"4", "5/2 is 2, and 2 * 2.0 = 4.0, so it prints 4", true},
		{"4", "the output is 4.0", true}, // value compare on the last number
		{"0", `\boxed{0}`, true},
		{"16", "16 cows", true},
	}
	for i, c := range cases {
		q := benchmarkQ{Match: "numeric", Expect: c.expect}
		if got := checkAnswer(q, c.ans); got != c.want {
			t.Errorf("case %d: numeric %q on %q = %v, want %v", i, c.expect, c.ans, got, c.want)
		}
	}
}

// v35: the three-bucket weighted score. The cases that matter are the caps — what a
// worker reads when it can do the general tiers and neither hard band, then one, then
// both — because that spread is the whole reason the weighting exists, and the empty-
// bucket redistribution, which is what keeps a score measured on a set WITHOUT coding
// questions comparable with one measured on the full set.
func TestBenchWeightedScoreBuckets(t *testing.T) {
	// tiers is a helper: tally[t] = {pass, count} for tier t.
	build := func(tally map[int][2]int) (pass, count []int, maxTier int) {
		for tr := range tally {
			if tr > maxTier {
				maxTier = tr
			}
		}
		pass, count = make([]int, maxTier+1), make([]int, maxTier+1)
		for tr, pc := range tally {
			pass[tr], count[tr] = pc[0], pc[1]
		}
		return
	}
	cases := []struct {
		name  string
		tally map[int][2]int
		want  int
	}{
		{"sweeps general, no insight, no coding", map[int][2]int{5: {10, 10}, 11: {0, 5}, 12: {0, 28}}, 60},
		{"general + coding, no insight", map[int][2]int{5: {10, 10}, 11: {0, 5}, 12: {28, 28}}, 80},
		{"general + insight, no coding", map[int][2]int{5: {10, 10}, 11: {5, 5}, 12: {0, 28}}, 80},
		{"everything", map[int][2]int{5: {10, 10}, 11: {5, 5}, 12: {28, 28}}, 100},
		{"nothing", map[int][2]int{5: {0, 10}, 11: {0, 5}, 12: {0, 28}}, 0},
		{"half of each", map[int][2]int{5: {5, 10}, 11: {2, 4}, 12: {14, 28}}, 50},
		// A bucket with no questions must not silently cost its weight: without
		// redistribution these would read 60 and 80 rather than 75 and 100.
		{"no coding questions at all: base+insight rescale to 75/25", map[int][2]int{5: {10, 10}, 11: {0, 5}}, 75},
		{"no coding questions, all passed", map[int][2]int{5: {10, 10}, 11: {5, 5}}, 100},
		{"only general tiers present", map[int][2]int{5: {10, 10}}, 100},
		// Tier 12 and anything above it share the coding bucket, so a future tier 13
		// joins coding rather than silently reopening the insight bucket.
		{"tier 13 counts as coding", map[int][2]int{5: {10, 10}, 11: {5, 5}, 13: {0, 4}}, 80},
	}
	for _, c := range cases {
		pass, count, maxTier := build(c.tally)
		ok, got := benchWeightedScore(pass, count, maxTier)
		if !ok {
			t.Errorf("%s: reported unmeasurable, want score %d", c.name, c.want)
			continue
		}
		if got != c.want {
			t.Errorf("%s: score = %d, want %d", c.name, got, c.want)
		}
	}
	if ok, _ := benchWeightedScore([]int{0}, []int{0}, 0); ok {
		t.Error("an empty question set must report unmeasurable, not a zero score")
	}
}

// Tier 12 is the coding tier. It is graded thinking-on (>= benchHardTier) and gets the
// frontier answer deadline (>= benchFrontierTier), and it must stay TRACE-shaped: the 47
// multiple-choice candidates written alongside these were all cut, because in every one
// the correct option was the longest and an 8B with no thinking mode scored 79% on them
// by picking the elaborate one. A trace question cannot be gamed that way — its answer is
// a fact about the language. This guards the set against drifting back to MCQ.
func TestCodingTierShape(t *testing.T) {
	n := 0
	for _, q := range benchmarkQuestions {
		if q.Tier < benchCodingTier {
			continue
		}
		n++
		if q.Match == "mcq" || q.Match == "mcq-repeat" {
			t.Errorf("tier-%d question is multiple choice, which measures option length here: %q", q.Tier, benchSnippet(q.Prompt))
		}
		if q.Expect == "" {
			t.Errorf("tier-%d question has an empty Expect: %q", q.Tier, benchSnippet(q.Prompt))
		}
		if strings.HasPrefix(q.Expect, "-") {
			t.Errorf("tier-%d question has a negative Expect (%q); grading it needs the signed benchNumberRe and a test proving it: %q", q.Tier, q.Expect, benchSnippet(q.Prompt))
		}
	}
	if n == 0 {
		t.Fatal("no tier-12 questions found — the coding bucket would silently redistribute its weight away")
	}
	if benchCodingTier <= benchInsightTier {
		t.Fatalf("benchCodingTier (%d) must sit above benchInsightTier (%d), or coding and insight share a bucket", benchCodingTier, benchInsightTier)
	}
	if benchCodingTier < benchFrontierTier {
		t.Errorf("benchCodingTier (%d) is below benchFrontierTier (%d), so coding questions would get the short answer deadline", benchCodingTier, benchFrontierTier)
	}
}

// v35: the sign belongs to the number. Before this, benchNumberRe and both of
// benchNumDeclaredRe's groups matched digits only, so a negative Expect could never
// be matched and — the silent direction, which is why this test exists — a positive
// Expect graded a negative answer as CORRECT. The last two cases are the regression
// guard: an unsigned subtraction must still read its operands unsigned, or every
// arithmetic chain in tiers 4–8 would start grading by a negated operand.
func TestCheckAnswerNumericSigned(t *testing.T) {
	cases := []struct {
		expect string
		ans    string
		want   bool
	}{
		{"-2", "-2", true},
		{"-2", "The answer is -2", true},
		{"-2", "(-5 // 2) is -3 and (-5 % 2) is 1, so the total is -2", true},
		{"-3", "-3.0", true}, // value compare, not string compare
		{"2", "-2", false},   // the sign error must NOT grade as correct
		{"-2", "2", false},   // and not in the other direction either
		{"7", "10 - 3 = 7", true},
		{"7", "The answer is 7, since 10 - 3 = 7.", true},
	}
	for i, c := range cases {
		q := benchmarkQ{Match: "numeric", Expect: c.expect}
		if got := checkAnswer(q, c.ans); got != c.want {
			t.Errorf("case %d: numeric %q on %q = %v, want %v", i, c.expect, c.ans, got, c.want)
		}
	}
}

// Numeric's declared/leading tiers: an answer-then-breakdown reply must grade by its
// asserted value, not by whichever component of the breakdown comes last — while
// declared self-corrections and intermediate-showing chains keep grading by their
// final claim.
func TestCheckAnswerNumericDeclared(t *testing.T) {
	const cows = "He has 16 cows: the 8 that survived plus the 8 he bought."
	const batBall = "The ball costs 5 cents and the bat costs 105 cents."
	const list = "1. He starts with 15 cows.\n2. All but 8 die, leaving 8.\n3. He buys 8 more, giving 16."
	cases := []struct {
		expect string
		ans    string
		want   bool
	}{
		{"16", cows, true},      // was a false fail: last-number graded the trailing 8
		{"8", cows, false},      // …and the breakdown's 8 must no longer pass
		{"5", batBall, true},    // was a false fail: last-number graded the bat's 105
		{"105", batBall, false}, // …and the bat's price must no longer pass
		// the LAST declaration wins, so a self-correction grades by its final claim
		{"1", "The answer is 2... no wait, the answer is 1.", true},
		{"2", "The answer is 2... no wait, the answer is 1.", false},
		{"16", "Answer: 16", true},
		{"120", "The result is 120.", true},
		{"22", "total = 22", true},
		// a leading "N." asserts N even when the breakdown follows
		{"16", "16. He keeps 8 and buys 8.", true},
		{"8", "16. He keeps 8 and buys 8.", false},
		// …but a numbered list's lead is a marker, not the answer: the final value grades
		{"16", list, true},
		{"15", list, false},
		// a lead value re-used in the working below it is an operand, not a conclusion
		{"1", "The total distance is 200 m. 200/100 = 2. Hmm, with the rear it is 1.", true},
	}
	for i, c := range cases {
		q := benchmarkQ{Match: "numeric", Expect: c.expect}
		if got := checkAnswer(q, c.ans); got != c.want {
			t.Errorf("case %d: numeric %q on %q = %v, want %v", i, c.expect, c.ans, got, c.want)
		}
	}
}

// The added thinking-on tier-4 discriminators must grade cleanly under realistic
// answers: a model graded thinking-on emits a terse final answer (often after a
// reasoned line), so both forms must PASS, while a fooled answer must FAIL — otherwise
// a matcher false-negative would silently under-rate good models, the very failure mode
// this benchmark exists to avoid.
func TestCheckAnswerAddedTier4(t *testing.T) {
	cases := []struct {
		q    benchmarkQ
		ans  string
		want bool
	}{
		// (3 == 3 == 3): left-associative, prints 0
		{benchmarkQ{Match: "numeric", Expect: "0"}, "0", true},
		{benchmarkQ{Match: "numeric", Expect: "0"}, "(3==3) is 1, then 1==3 is false, so it prints 0", true},
		// 0.1 + 0.2 == 0.3 in C → 0
		{benchmarkQ{Match: "numeric", Expect: "0"}, "0.1+0.2 is 0.30000000000000004, not 0.3, so it prints 0", true},
		// 2 ** 3 ** 2 is right-associative → 512 (a left-assoc model says 64)
		{benchmarkQ{Match: "numeric", Expect: "512"}, "512", true},
		{benchmarkQ{Match: "numeric", Expect: "512"}, "** is right-associative, so 3**2=9 then 2**9=512", true},
		{benchmarkQ{Match: "numeric", Expect: "512"}, "(2**3)**2 = 64", false}, // fooled by left-assoc
		// 2#101 + 1 = 6
		{benchmarkQ{Match: "numeric", Expect: "6"}, "2#101 is binary 101 = 5, plus 1 is 6", true},
		// carbon-14: 6 electrons = atomic number, not mass 14 (MCQ so a verbose answer can't trip the numeric matcher)
		{benchmarkQ{Match: "mcq", Expect: "C"}, "Carbon's atomic number is 6, so 6 electrons → C", true},
		{benchmarkQ{Match: "mcq", Expect: "C"}, "A", false}, // fooled by the mass number 14
		// element 11 = Sodium = B
		{benchmarkQ{Match: "mcq", Expect: "B"}, "The answer is B (Sodium).", true},
		{benchmarkQ{Match: "mcq", Expect: "B"}, "A", false}, // Neon, wrong
		// 100 days from Wednesday → Friday
		{benchmarkQ{Match: "contains", Expect: "Friday"}, "100 mod 7 = 2, so Wednesday + 2 = Friday", true},
		{benchmarkQ{Match: "contains", Expect: "Friday"}, "Saturday", false},
		// GSM-Symbolic irrelevant-clause traps: the red boxes / blue floor are noise
		{benchmarkQ{Match: "numeric", Expect: "37"}, "5 boxes × 8 = 40 apples, minus 3 = 37", true},
		{benchmarkQ{Match: "numeric", Expect: "37"}, "40 - 3 - 16 = 21", false}, // fooled into subtracting the 2 red boxes
		{benchmarkQ{Match: "numeric", Expect: "8"}, "12 ÷ 3 = 4 sets, 4 × 2 = 8 dollars", true},
	}
	for i, c := range cases {
		if got := checkAnswer(c.q, c.ans); got != c.want {
			t.Errorf("case %d: checkAnswer(%s,%q)=%v want %v", i, c.q.Match, c.ans, got, c.want)
		}
	}
}

// The restored hard tiers (5–8) and the new frontier questions must grade cleanly under a
// realistically reasoned correct answer, and a typically-fooled answer must fail — the same
// guard as TestCheckAnswerAddedTier4, for the questions brought back from the pre-v16 set.
func TestCheckAnswerRestoredHard(t *testing.T) {
	cases := []struct {
		q    benchmarkQ
		ans  string
		want bool
	}{
		{benchmarkQ{Match: "numeric", Expect: "40"}, "time = 2 + 1 = 3 hours, 120/3 = 40 mph", true}, // harmonic-mean avg speed
		{benchmarkQ{Match: "numeric", Expect: "40"}, "(30 + 60) / 2 = 45", false},                    // fooled into the arithmetic mean
		{benchmarkQ{Match: "numeric", Expect: "10"}, "(220 - 200) / 2 = 10 cents", true},             // notebook/pen bat-and-ball trap
		{benchmarkQ{Match: "numeric", Expect: "10"}, "the pen costs 20 cents", false},                // the lure
		{benchmarkQ{Match: "numeric", Expect: "98"}, "3-multiples 63 + 5-multiples 50 - overlap 15 = 98", true},
		{benchmarkQ{Match: "numeric", Expect: "50.4"}, "28 L x $1.80 = $50.40", true}, // decimal value-compare
		{benchmarkQ{Match: "numeric", Expect: "7.5"}, "97.5 - 90 = 7.5 degrees", true},
		{benchmarkQ{Match: "numeric", Expect: "2207"}, "47^2 - 2 = 2207", true},
		{benchmarkQ{Match: "numeric", Expect: "376"}, "2^100 mod 1000 is 376", true},
		{benchmarkQ{Match: "numeric", Expect: "3"}, "exponent 7^7 = 3 mod 4, so 7^3 ends in 3", true},
		{benchmarkQ{Match: "contains", Expect: "1/7"}, "P = (1/8)/(7/8) = 1/7", true},
		{benchmarkQ{Match: "mcq", Expect: "B"}, "'my father's son' is himself, so the portrait is his son -> B", true},
		{benchmarkQ{Match: "mcq", Expect: "C"}, "the marble fell out when inverted, so it's on the table -> C", true},
		{benchmarkQ{Match: "contains", Expect: "Iowa"}, "The answer is Iowa.", true},
	}
	for i, c := range cases {
		if got := checkAnswer(c.q, c.ans); got != c.want {
			t.Errorf("case %d: checkAnswer(%s,%q)=%v want %v", i, c.q.Match, c.ans, got, c.want)
		}
	}
}

func TestAppendUnique(t *testing.T) {
	s := appendUnique([]string{"chat", "json"}, "json")
	if len(s) != 2 {
		t.Errorf("dup not deduped: %v", s)
	}
	if s = appendUnique(s, "vision"); len(s) != 3 || s[2] != "vision" {
		t.Errorf("append failed: %v", s)
	}
}

func TestWorkerProfileRoundTrip(t *testing.T) {
	// applyProfileIfGen must overwrite declared seeds with measured values
	// (gen 0 skips the generation check).
	reg := newTestRegistry()
	readyBackend(reg, "w", 0, 0, 0) // declared nothing (quality/tps/conc = 0)
	reg.applyProfileIfGen("w", 0, &WorkerProfile{
		Model: "qwen-x", Quality: 8, ContextK: 128, MaxConcurrency: 6,
		BaselineTPS: 95, Features: []string{"chat", "tools"}, Thinking: true,
	})
	b := reg.get("w")
	if b.Quality != 8 || b.ContextK != 128 || b.MaxConcurrency != 6 || b.BaselineTPS != 95 || !b.Thinking {
		t.Fatalf("profile not applied: %+v", b.BackendRegistration)
	}
	if !hasFeature(b, "tools") {
		t.Fatalf("features not applied: %v", b.Features)
	}
}

// The version history above benchmarkVersion is written as "vNN:" comment
// blocks, and benchmarkVersion is what invalidates every cached profile in
// worker_profiles. v33 and v34 both landed without the constant moving, so
// workers scored under the old flat-percentage rules stayed in the table marked
// current while autoTargetQuality read every score as one absolute 0-100 scale.
// Nothing caught it, because nothing was looking.
//
// This test looks: the constant must be at least as high as the highest version
// the file claims to have implemented. It reads the source rather than any
// in-memory value, because the failure mode is precisely that the comment and
// the constant disagree.
func TestBenchmarkVersionCoversScoringChanges(t *testing.T) {
	marker := regexp.MustCompile(`//\s*v(\d+):`)
	highest, where := 0, ""
	for _, name := range []string{"benchmark.go", "benchmark_data.go"} {
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		for _, m := range marker.FindAllStringSubmatch(string(src), -1) {
			n, err := strconv.Atoi(m[1])
			if err != nil {
				continue
			}
			if n > highest {
				highest, where = n, name
			}
		}
	}
	if highest == 0 {
		t.Fatal("found no vNN: version-history markers — has the comment style changed?")
	}
	if benchmarkVersion < highest {
		t.Errorf("benchmarkVersion is %d but %s documents changes up to v%d.\n"+
			"A scoring or question-set change that does not bump the constant leaves "+
			"profiles measured under the old rules in worker_profiles marked current, "+
			"on a quality scale that no longer means the same thing.",
			benchmarkVersion, where, highest)
	}
}
