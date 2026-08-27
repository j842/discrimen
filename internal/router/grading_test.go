package router

import (
	"strings"
	"testing"
	"time"
)

// Adversarial grading suite.
//
// Every historical grading bug in this package has had the same shape: the
// grader guesses where the answer is in free-form prose, and the guess is a bet
// on how models phrase things. The bets that lost so far —
//
//	exact-list split on commas and read element 0, so any prose lead-in broke it
//	coding_completion answers were run standalone, measuring instruction-following
//	an empty Expect made checkAnswer return true for ANY text
//	a comma inside the lead-in changed the element count and failed a right answer
//	bolding the option labels earned half credit for the wrong pick
//
// — were all invisible in production, because a wrong grade looks exactly like a
// wrong answer. So the protection has to be a corpus of REALISTIC replies rather
// than more heuristics, and it has to run in both directions: a correct answer
// dressed awkwardly must PASS, and a wrong answer dressed confidently must FAIL.
//
// The governing rule for everything below: THE GRADER MUST NEVER RESOLVE
// AMBIGUITY IN THE DIRECTION OF A PASS. A reply it cannot confidently read is a
// miss. Marking a correct answer wrong costs one question; marking a wrong
// answer right inflates a score that gates routing.

// amcRepeatPrompt is the shape LiveBench's AMC items ship with. The worked
// example matters to grading: every one of these prompts names F, F is never a
// valid option, and a model that quotes the instruction after answering ends its
// reply with a repeated run belonging to the PROMPT rather than to itself.
const amcRepeatPrompt = "What is the value of x?\n\nA) 1\nB) 2\nC) 3\nD) 4\nE) 5\n\n" +
	"Give your answer as the letter, duplicated five times in a single string. " +
	"For example, if the answer is F, then write FFFFF."

// gradeCase is one realistic (question, reply) pair and the grade it must get.
type gradeCase struct {
	name   string
	q      benchmarkQ
	answer string
	pass   bool // what checkAnswer must return
}

// ── correct answers that must PASS regardless of dressing ──────────────────

var mustPass = []gradeCase{
	{
		name:   "exact-list, comma inside the lead-in",
		q:      benchmarkQ{Match: "exact-list", Expect: "1, fisherman, coffee, musician"},
		answer: "So, the answer is 1, fisherman, coffee, musician",
		pass:   true,
	},
	{
		name:   "exact-list, 'Therefore,' lead-in and a full stop",
		q:      benchmarkQ{Match: "exact-list", Expect: "1, fisherman, coffee, musician"},
		answer: "Therefore, the final ordering is 1, fisherman, coffee, musician.",
		pass:   true,
	},
	{
		name:   "exact-list, correct line followed by a pleasantry containing commas",
		q:      benchmarkQ{Match: "exact-list", Expect: "a, b, c"},
		answer: "a, b, c\nLet me know if you want the reasoning, the clues, or the grid.",
		pass:   true,
	},
	{
		name:   "exact-list, working above the committed answer",
		q:      benchmarkQ{Match: "exact-list", Expect: "red, blue, green"},
		answer: "Trying blue, red, green — no, clue 3 forbids it.\nFinal answer: red, blue, green",
		pass:   true,
	},
	{
		name:   "numeric, answer stated then explained",
		q:      benchmarkQ{Match: "numeric", Expect: "16"},
		answer: "He has 16 cows: the 8 that survived plus the 8 he bought.",
		pass:   true,
	},
	{
		name:   "numeric, self-correction — the LAST value is right",
		q:      benchmarkQ{Match: "numeric", Expect: "1"},
		answer: "200/100 = 2 … wait, that double-counts. The answer is 1.",
		pass:   true,
	},
	{
		name:   "numeric, spelled out",
		q:      benchmarkQ{Match: "numeric", Expect: "22"},
		answer: "twenty-two",
		pass:   true,
	},
	{
		name:   "numeric, currency and decimals",
		q:      benchmarkQ{Match: "numeric", Expect: "50.4"},
		answer: "$50.40",
		pass:   true,
	},
	{
		name:   "mcq-repeat, the requested five-letter form",
		q:      benchmarkQ{Match: "mcq-repeat", Expect: "C"},
		answer: "CCCCC",
		pass:   true,
	},
	{
		name:   "mcq-repeat, correct pick then the model quotes the instruction's F example",
		q:      benchmarkQ{Match: "mcq-repeat", Expect: "C", Prompt: amcRepeatPrompt},
		answer: "CCCCC\n\n(The instruction said: if the answer were F, write FFFFF.)",
		pass:   true,
	},
	{
		name:   "mcq-repeat, declared answer without the repetition",
		q:      benchmarkQ{Match: "mcq-repeat", Expect: "D"},
		answer: "I ruled out A early. The answer is D.",
		pass:   true,
	},
	{
		name:   "mcq-repeat, correct pick then the prompt's F example is quoted inline",
		q:      benchmarkQ{Match: "mcq-repeat", Expect: "C", Prompt: amcRepeatPrompt},
		answer: "The answer is C, so I write CCCCC. Had it been F I would have written FFFFF.",
		pass:   true,
	},
	{
		name:   "mcq, letter with trailing punctuation",
		q:      benchmarkQ{Match: "mcq", Expect: "B"},
		answer: "The answer is B.",
		pass:   true,
	},
}

// ── wrong answers that must FAIL however confidently dressed ───────────────

var mustFail = []gradeCase{
	{
		name:   "exact-list single element, model ABANDONS the expected value",
		q:      benchmarkQ{Match: "exact-list", Expect: "asparagus"},
		answer: "Let me start by considering asparagus\nBut clue 3 rules that out, so the answer must be carrot.",
		pass:   false,
	},
	{
		name:   "exact-list single element, numeric, self-corrected AWAY from the expected",
		q:      benchmarkQ{Match: "exact-list", Expect: "3"},
		answer: "Counting the houses gives 3\nActually no — recounting, it is 5, so the answer is 5.",
		pass:   false,
	},
	{
		name:   "exact-list, right elements in the WRONG order",
		q:      benchmarkQ{Match: "exact-list", Expect: "a, b, c"},
		answer: "The answer is b, a, c",
		pass:   false,
	},
	{
		name:   "contains, expected token appears while being RULED OUT",
		q:      benchmarkQ{Match: "contains", Expect: "Friday"},
		answer: "Counting forward: Thursday, Friday, Saturday. So it is Saturday.",
		pass:   false,
	},
	{
		name:   "contains, expected token negated",
		q:      benchmarkQ{Match: "contains", Expect: "UnboundLocalError"},
		answer: "It raises a NameError, definitely not UnboundLocalError.",
		pass:   false,
	},
	{
		name:   "numeric, magnitude word must not collapse to its multiplier",
		q:      benchmarkQ{Match: "numeric", Expect: "100"},
		answer: "two hundred",
		pass:   false,
	},
}

// TestGradingCorrectAnswersPass is the false-negative direction: a right answer
// must not be failed for how it was phrased. Each of these is a shape a real
// model produces.
func TestGradingCorrectAnswersPass(t *testing.T) {
	for _, c := range mustPass {
		t.Run(c.name, func(t *testing.T) {
			if !checkAnswer(c.q, c.answer) {
				t.Errorf("correct answer graded WRONG\n  expect: %q\n  reply:  %q", c.q.Expect, c.answer)
			}
		})
	}
}

// TestGradingWrongAnswersFail is the false-positive direction, and the one that
// matters more: a wrong answer accepted inflates a score that gates routing.
func TestGradingWrongAnswersFail(t *testing.T) {
	for _, c := range mustFail {
		t.Run(c.name, func(t *testing.T) {
			if checkAnswer(c.q, c.answer) {
				t.Errorf("wrong answer graded CORRECT\n  expect: %q\n  reply:  %q", c.q.Expect, c.answer)
			}
		})
	}
}

// An empty Expect must fail in EVERY mode. checkAnswer previously returned true
// for any text in the contains/final-contains/code-exec/unknown branches, so one
// malformed question would have graded every answer correct and lifted the whole
// fleet's score.
func TestGradingEmptyExpectAlwaysFails(t *testing.T) {
	for _, mode := range []string{"numeric", "mcq", "mcq-repeat", "exact-list",
		"contains", "final-contains", "code-exec", "", "unrecognised-mode"} {
		if checkAnswer(benchmarkQ{Match: mode, Expect: ""}, "total nonsense") {
			t.Errorf("mode %q with an empty Expect accepted an arbitrary answer", mode)
		}
	}
}

// Loose grading is half credit for a right answer in the wrong FORMAT. It must
// never credit formatting alone — a model that bolds its option labels or its
// step headers is not thereby half-right.
func TestLooseGradingDoesNotCreditFormatting(t *testing.T) {
	cases := []gradeCase{
		{
			name:   "mcq: bolds every option label, picks the wrong one",
			q:      benchmarkQ{Match: "mcq", Expect: "B"},
			answer: "Let's check each:\n**A)** forward - no.\n**B)** backward - no.\n**C)** sideways - no.\nSo the answer is D.",
		},
		{
			name:   "numeric: bolds step headers, answers wrongly",
			q:      benchmarkQ{Match: "numeric", Expect: "3"},
			answer: "**Step 1**: count rows.\n**Step 2**: count columns.\n**Step 3**: multiply.\nSo the final total is 7.",
		},
		{
			name:   "numeric: sprays candidates in bold then picks a wrong one",
			q:      benchmarkQ{Match: "numeric", Expect: "42"},
			answer: "It could be **40**, **41**, **42**, **43** or **44**. I'll say 44.",
		},
		{
			name:   "numeric: expected value appears only as a variable binding",
			q:      benchmarkQ{Match: "numeric", Expect: "42"},
			answer: "Let n = 42. Then the total is 36.",
		},
		{
			name:   "mcq: considers and REJECTS the correct option",
			q:      benchmarkQ{Match: "mcq", Expect: "B"},
			answer: "One might think it goes backward, towards the rear of the car, but it stays put. I choose D.",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if checkAnswerLoose(c.q, c.answer) {
				t.Errorf("half credit awarded for formatting alone\n  expect: %q\n  reply:  %q", c.q.Expect, c.answer)
			}
		})
	}
}

// Loose grading must still do its job: a right answer that ignored the output
// format keeps half credit, or the mode is pointless.
func TestLooseGradingStillCreditsKnowledge(t *testing.T) {
	cases := []gradeCase{
		{
			name:   "numeric: right value emphasised as the conclusion",
			q:      benchmarkQ{Match: "numeric", Expect: "114"},
			answer: "Adding the two groups and subtracting the overlap gives **114**.",
		},
		{
			name:   "mcq: right option picked in an emphasised span",
			q:      benchmarkQ{Match: "mcq", Expect: "B"},
			answer: "Working through the physics, the answer is **B) backward, towards the rear**.",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if checkAnswer(c.q, c.answer) {
				t.Skip("already a strict pass; loose is not exercised")
			}
			if !checkAnswerLoose(c.q, c.answer) {
				t.Errorf("knowledge in the wrong format lost its half credit\n  expect: %q\n  reply:  %q",
					c.q.Expect, c.answer)
			}
		})
	}
}

// A truncated profile must sample each tier REPRESENTATIVELY, not just its first
// entries. This is the bug that made slowness pay: within tier 12 the 28
// answerable coding questions were indexed before the 40 unpassable headroom
// items, so a worker cut short by the profile budget scored 28/30 on the coding
// bucket where a worker that finished scored 28/68 — up to +10 points of quality
// for being too slow to finish, which inverts the ranking routing is built on.
func TestTruncatedProfileSamplesTiersRepresentatively(t *testing.T) {
	// One tier holding 30 answerable questions followed by 70 unpassable ones,
	// which is the shape that produced the bug.
	var qs []benchmarkQ
	for i := 0; i < 30; i++ {
		qs = append(qs, benchmarkQ{Tier: 12, Prompt: "answerable-" + string(rune('a'+i%26)) + string(rune('0'+i/26)), Expect: "x"})
	}
	for i := 0; i < 70; i++ {
		qs = append(qs, benchmarkQ{Tier: 12, Prompt: "impossible-" + string(rune('a'+i%26)) + string(rune('0'+i/26)), Expect: "y"})
	}
	order := benchStratifiedOrder(qs)
	if len(order) != len(qs) {
		t.Fatalf("order dropped questions: %d of %d", len(order), len(qs))
	}
	// In the first 30 dispatched, the answerable share must be near the 30%
	// population rate — not the 100% that index order produced.
	answerable := 0
	for _, i := range order[:30] {
		if qs[i].Expect == "x" {
			answerable++
		}
	}
	if answerable > 18 {
		t.Errorf("first 30 dispatched contain %d/30 answerable questions against a 30%% population rate; "+
			"a truncated profile still over-samples the front of the tier", answerable)
	}
	// Deterministic: two profiles of the same worker must ask the same questions
	// in the same order, or scores are not comparable across runs.
	again := benchStratifiedOrder(qs)
	for i := range order {
		if order[i] != again[i] {
			t.Fatalf("dispatch order is not stable across calls at position %d", i)
		}
	}
}

// Calibration and production must build the SAME request for the same question.
// They had drifted — production appended " Give the number only." and
// calibration did not — so 47 generated questions had their difficulty measured
// on a prompt production never sends.
func TestPromptConstructionIsShared(t *testing.T) {
	q := benchmarkQ{Tier: benchHardTier, Match: "numeric", Prompt: "How many cows?", Expect: "16"}
	prompt, budget := benchRequestFor(q, true, 0)
	if !strings.Contains(prompt, "Give the number only") {
		t.Errorf("numeric prompt lost its format instruction: %q", prompt)
	}
	if budget != benchThinkMaxTokens {
		t.Errorf("hard-tier budget = %d, want %d", budget, benchThinkMaxTokens)
	}
	// The budget is a property of the QUESTION, not the mode: the two quality
	// scores must differ only in thinking, or QualityNoThink measures the budget.
	_, offBudget := benchRequestFor(q, false, 0)
	if offBudget != budget {
		t.Errorf("no-think budget %d != thinking budget %d — the two scores would differ in two variables",
			offBudget, budget)
	}
	// An easy-tier question keeps the short budget in both modes.
	easy := benchmarkQ{Tier: 1, Match: "contains", Prompt: "Hello?", Expect: "hi"}
	if _, b := benchRequestFor(easy, true, 0); b != benchMaxTokens {
		t.Errorf("easy-tier budget = %d, want %d", b, benchMaxTokens)
	}
}

// The answer ceiling must never exceed the worker's own context window: a server
// handed a max_tokens it cannot honour either errors — turning a gradeable
// question into a transport failure — or silently clamps, in which case the
// ceiling was never real. Several workers here have a 32K window, which is the
// whole hard-tier ceiling.
func TestAnswerCeilingFitsTheContextWindow(t *testing.T) {
	q := benchmarkQ{Tier: benchHardTier, Match: "contains", Prompt: "short question", Expect: "x"}
	// 32K window: ceiling must come down to leave room for prompt + slack.
	_, budget := benchRequestFor(q, true, 32*1024)
	if budget >= 32*1024 {
		t.Errorf("ceiling %d does not fit a 32K window", budget)
	}
	if budget <= 0 {
		t.Errorf("ceiling %d is unusable", budget)
	}
	// A large window leaves the nominal ceiling intact.
	if _, big := benchRequestFor(q, true, 256*1024); big != benchThinkMaxTokens {
		t.Errorf("ceiling %d with a 256K window, want the nominal %d", big, benchThinkMaxTokens)
	}
	// Unknown window (0) must not clamp to zero — that would ask for no answer.
	if _, unknown := benchRequestFor(q, true, 0); unknown != benchThinkMaxTokens {
		t.Errorf("ceiling %d with an unknown window, want the nominal %d", unknown, benchThinkMaxTokens)
	}
	// A window too small for the prompt still asks for a usable answer rather
	// than a negative or zero ceiling.
	long := benchmarkQ{Tier: benchHardTier, Match: "contains", Expect: "x",
		Prompt: strings.Repeat("word ", 2000)}
	if _, tiny := benchRequestFor(long, true, 2048); tiny < benchMinAnswerTokens {
		t.Errorf("ceiling %d for an oversized prompt, want at least %d", tiny, benchMinAnswerTokens)
	}
}

// The benchmark must write the thinking gate in the spelling each endpoint was
// MEASURED to honour, exactly as production does. It used to hardcode
// chat_template_kwargs — a vLLM/llama.cpp extension — so on every relay row and
// every strict provider NEITHER pass switched anything: both ran in the
// template's default mode, and Quality and QualityNoThink came back as the same
// number measured twice in one mode. That is the two-score design failing at its
// root.
func TestBenchmarkUsesTheMeasuredThinkingDialect(t *testing.T) {
	effort := &Backend{}
	effort.ThinkingDialect = thinkingDialectEffort
	on := map[string]any{}
	applyBenchThinking(on, effort, true)
	if _, ok := on["chat_template_kwargs"]; ok {
		t.Error("a reasoning_effort endpoint was sent chat_template_kwargs")
	}
	if on["reasoning_effort"] == nil || on["reasoning_effort"] == "none" {
		t.Errorf("thinking-on wrote reasoning_effort=%v", on["reasoning_effort"])
	}
	off := map[string]any{}
	applyBenchThinking(off, effort, false)
	if off["reasoning_effort"] != "none" {
		t.Errorf("thinking-off wrote reasoning_effort=%v, want none", off["reasoning_effort"])
	}

	// The kwargs dialect, and an unprobed worker, both get the spelling the
	// fleet has always spoken.
	kwargs := &Backend{}
	kwargs.ThinkingDialect = thinkingDialectKwargs
	unprobed := &Backend{}
	for _, b := range []*Backend{kwargs, unprobed} {
		p := map[string]any{}
		applyBenchThinking(p, b, true)
		kw, ok := p["chat_template_kwargs"].(map[string]bool)
		if !ok || !kw["enable_thinking"] {
			t.Errorf("dialect %q did not get enable_thinking: %v", b.ThinkingDialect, p)
		}
		if _, has := p["reasoning_effort"]; has {
			t.Errorf("dialect %q was sent reasoning_effort", b.ThinkingDialect)
		}
	}

	// An endpoint measured to honour NEITHER gate is sent neither. Writing one
	// buys a field a strict endpoint can reject, in exchange for nothing.
	none := &Backend{}
	none.ThinkingDialect = thinkingDialectNone
	p := map[string]any{}
	applyBenchThinking(p, none, true)
	if len(p) != 0 {
		t.Errorf("a no-gate endpoint was sent %v", p)
	}
}

// A worker with no thinking mode must have its results recorded as NO-THINK
// evidence. They used to be filed under Thinking=true — the evidence of a worker
// that cannot think, filed as evidence about thinking — so routing queried the
// no-think bucket, found nothing, and reported it unmeasured in the only mode it
// has.
func TestMixedPassRecordsEachResultInItsOwnMode(t *testing.T) {
	results := []BenchResult{
		{Tier: 1, Prompt: "easy", Expect: "a", Pass: true},             // asked thinking-OFF
		{Tier: benchHardTier, Prompt: "hard", Expect: "b", Pass: true}, // asked thinking-ON
		{Tier: 12, Prompt: "harder", Expect: "c", Pass: false},         // asked thinking-ON
		{Tier: 2, Prompt: "skipped", Expect: "d", Skipped: true},       // never asked
	}
	rows := observationsFromMixed("w", results, time.Unix(0, 0))
	if len(rows) != 3 {
		t.Fatalf("got %d observations, want 3 (the skipped one contributes nothing)", len(rows))
	}
	byPrompt := map[string]bool{}
	for i, r := range results {
		if r.Skipped {
			continue
		}
		_ = i
		byPrompt[r.Prompt] = r.Tier >= benchHardTier
	}
	for _, o := range rows {
		want := byPrompt[promptOfQID(results, o.QID)]
		if o.Thinking != want {
			t.Errorf("observation recorded Thinking=%v, want %v — the mixed pass asks easy "+
				"tiers thinking-off and hard tiers thinking-on", o.Thinking, want)
		}
	}
}

func promptOfQID(results []BenchResult, qid string) string {
	for _, r := range results {
		if benchQuestionQID(r.Prompt, r.Expect) == qid {
			return r.Prompt
		}
	}
	return ""
}
