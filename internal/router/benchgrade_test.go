package router

// v43 grading and profiling corrections.
//
// Every case below was REPRODUCED against the real graders before it was fixed,
// and every one of them was silent in production: a mis-graded question is
// indistinguishable from a wrong answer, and a profiling pass that hangs or
// divides by the wrong number still returns a plausible score.
//
// The governing rule from grading_test.go applies throughout — the grader must
// never resolve ambiguity in the direction of a pass — with one documented
// exception, argued at the batBall cases in TestNumericGraderReadsPastAnd.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// ── DEFECT 1: the numeric grader broke on the word "and" ───────────────────

// "and" was a clause break, so benchLeadNumber cut the leading clause at it and
// COMMITTED to whatever single number was in front — which is the operand of a
// sum far more often than it is an answer. The tier-1 control question was
// graded wrong by it.
func TestNumericGraderReadsPastAnd(t *testing.T) {
	cases := []struct {
		name   string
		expect string
		ans    string
		want   bool
	}{
		// The four reproduced misgrades. All four graded FALSE before v43.
		{"tier-1 control", "13", "The sum of 8 and 5 is 13.", true},
		{"tier-2 floor", "8", "There are 7 sons and 1 sister, so 8.", true},
		{"bat-and-ball, bat named first", "5", "The bat costs 105 cents and the ball costs 5 cents.", true},
		{"and before the conclusion", "25", "20 and 5 more makes 25.", true},

		// The motivating case for the old rule still passes, in BOTH orderings.
		{"bat-and-ball, ball named first", "5", "The ball costs 5 cents and the bat costs 105 cents.", true},

		// An operand is not an answer: the number in front of the "and" is only
		// offered when its clause STATES something (benchAssertVerbRe). "The sum
		// of 8" and a bare "20" state nothing, so neither is a candidate.
		{"leading operand of a sum", "8", "The sum of 8 and 5 is 13.", false},
		{"bare leading operand", "20", "20 and 5 more makes 25.", false},
		{"trailing operand", "5", "The sum of 8 and 5 is 13.", false},

		// The answer-then-breakdown rule is untouched: a lead ended by real clause
		// punctuation is still a COMMITMENT, so the breakdown below it cannot
		// rescue a wrong expectation.
		{"breakdown component", "8", "He has 16 cows: the 8 that survived plus the 8 he bought.", false},
		{"breakdown after a leading N.", "8", "16. He keeps 8 and buys 8.", false},
		{"leading N. still asserts", "16", "16. He keeps 8 and buys 8.", true},

		// A declaration still beats everything, including an "and" in front of it.
		{"declared after an and", "-2", "(-5 // 2) is -3 and (-5 % 2) is 1, so the total is -2", true},
		{"declared self-correction", "1", "The total distance is 200 m. 200/100 = 2. Hmm, with the rear it is 1.", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			q := benchmarkQ{Match: "numeric", Expect: c.expect}
			if got := checkAnswer(q, c.ans); got != c.want {
				t.Errorf("numeric %q on %q = %v, want %v", c.expect, c.ans, got, c.want)
			}
		})
	}
}

// The tier-1 control is the question that proves the grader works at all, so its
// own answer has to grade through the shapes a model actually writes. It was
// failing the most natural one of them.
func TestTierOneControlGradesItsOwnAnswer(t *testing.T) {
	q := benchmarkQ{Tier: 1, Match: "numeric", Prompt: "What is 8 + 5?", Expect: "13"}
	for _, ans := range []string{
		"13",
		"13.",
		"The sum of 8 and 5 is 13.",
		"8 + 5 = 13",
		"8 and 5 make 13.",
		"Adding 8 and 5 gives 13.",
	} {
		if !checkAnswer(q, ans) {
			t.Errorf("the tier-1 control fails its own answer on %q", ans)
		}
	}
}

// ── DEFECT 2: the mcq last-standalone-letter fallback passed a wrong pick ──

// The old fallback graded the last standalone letter anywhere in the reply, so a
// reply that named the letters it was RULING OUT was graded on whichever it
// mentioned last. Tiers 9-10 carry ten options and invite exactly that shape,
// and this is the false-positive direction: it lets a weak model draw hard
// traffic.
func TestMCQFallbackNeedsAnActualPick(t *testing.T) {
	mustFail := []struct {
		name   string
		expect string
		ans    string
	}{
		{"rules out the expected option by name", "C", "The answer must be option D, since vitamin C is irrelevant here"},
		{"picks E, mentions B as a decoy", "B", "I pick E.\nNote: option B was a decoy"},
		{"picks J, calls G tempting", "G", "J — though G is tempting"},
		{"declines to choose", "B", "It could be a or b"},
	}
	for _, c := range mustFail {
		t.Run(c.name, func(t *testing.T) {
			if checkAnswer(benchmarkQ{Match: "mcq", Expect: c.expect}, c.ans) {
				t.Errorf("a pick the model never made graded CORRECT\n  expect: %q\n  reply:  %q", c.expect, c.ans)
			}
		})
	}

	// …while every shape that IS a pick keeps grading. Restricting the fallback
	// in the wrong direction would under-rate every model that answers in prose.
	mustPass := []struct {
		name   string
		expect string
		ans    string
	}{
		{"reply ends on the letter", "C", "It fails with an error, so C."},
		{"contraction then a letter", "D", "I'd say D"},
		{"arrow then a letter", "H", "…so the packing factor is 0.78 → H"},
		{"article earlier, letter last", "C", "The answer is a syntax error, so C."},
		{"emphasised letter", "C", "**C**"},
		{"only one candidate anywhere", "C", "Working through each option, the correct one is C"},
		{"enumeration concludes on its pick", "C", "A) it prints 10 — no. B) it loops forever — no. C) it fails with an error — yes."},
		{"newline enumeration", "B", "A. wrong, octal needs 0o.\nB. correct."},
		{"ten-option enumeration", "C", "A) C1 — no.\nB) Cs — no.\nC) C2v — yes, SF4 is see-saw."},
	}
	for _, c := range mustPass {
		t.Run(c.name, func(t *testing.T) {
			if !checkAnswer(benchmarkQ{Match: "mcq", Expect: c.expect}, c.ans) {
				t.Errorf("a real pick graded WRONG\n  expect: %q\n  reply:  %q", c.expect, c.ans)
			}
		})
	}
}

// ── DEFECT 3: contains/final-contains rejected an EXPLAINED right answer ───

// These 16 questions are the benchmark's sanity band — both tier-1 controls,
// three of the four tier-2 floor questions and all three tier-3 echo traps — and
// reading only the reply's final clause failed every one of them the moment the
// model added a sentence of explanation after its answer.
func TestContainsGradesTheLastAssertionNotTheLastClause(t *testing.T) {
	mustPass := []struct {
		name   string
		mode   string
		expect string
		ans    string
	}{
		{"bare answer", "contains", "Paris", "Paris."},
		{"answer in a sentence", "contains", "Paris", "The capital of France is Paris."},
		{"answer then a following sentence", "contains", "Paris", "The answer is Paris. It is located on the Seine."},
		{"answer then a labelled explanation", "contains", "Paris", "Answer: Paris\nExplanation: it has been the capital since 987."},
		{"answer then a bullet", "contains", "Paris", "The capital is Paris\n- Since 987 AD"},
		{"answer then a because-clause", "final-contains", "Mary", "The answer is Mary, because the puzzle names four daughters and then the speaker."},
		{"answer then several explanations", "contains", "Iowa", "The answer is Iowa.\n- It begins with I and o.\n- Every other state starts with one vowel."},
	}
	for _, c := range mustPass {
		t.Run(c.name, func(t *testing.T) {
			if !checkAnswer(benchmarkQ{Match: c.mode, Expect: c.expect}, c.ans) {
				t.Errorf("a correct answer with an explanation after it graded WRONG\n  expect: %q\n  reply:  %q",
					c.expect, c.ans)
			}
		})
	}

	// The walk-back must not turn into "the token appears somewhere". These are
	// the shapes the mode exists to reject, and each one stops the walk for a
	// different reason.
	mustFail := []struct {
		name   string
		mode   string
		expect string
		ans    string
	}{
		{"token inside an enumeration, conclusion elsewhere", "contains", "Friday",
			"Counting forward: Thursday, Friday, Saturday. So it is Saturday."},
		{"token in a subordinate premise echo", "final-contains", "Everest",
			"Before Mount Everest was discovered, K2 was the tallest mountain."},
		{"possessive premise echo", "final-contains", "Mary", "Mary's father's fifth daughter must be Lulu."},
		{"terse lead walked back", "final-contains", "Everest", "Everest, but actually it was K2."},
		{"terse lead contradicted", "final-contains", "Everest", "Everest. No, it was K2."},
		{"answer stated then retracted", "final-contains", "Everest",
			"The answer is Everest. Wait — K2 was taller at the time, so K2."},
		{"token negated", "contains", "UnboundLocalError", "It raises a NameError, definitely not UnboundLocalError."},
		{"competing claim after the token", "final-contains", "Everest", "Everest is often assumed. K2 was the tallest mountain."},
	}
	for _, c := range mustFail {
		t.Run(c.name, func(t *testing.T) {
			if checkAnswer(benchmarkQ{Match: c.mode, Expect: c.expect}, c.ans) {
				t.Errorf("a premise echo / wrong answer graded CORRECT\n  expect: %q\n  reply:  %q",
					c.expect, c.ans)
			}
		})
	}
}

// Every "contains" and "final-contains" question in the bank must grade its own
// expected answer through the shapes a model actually emits — including the one
// that broke: the answer, then a sentence explaining it. A question that cannot
// recognise its own answer scores every worker a fail on the tiers that are
// supposed to be the floor, and drags the whole fleet's quality down with it.
func TestSanityBandQuestionsGradeTheirOwnAnswers(t *testing.T) {
	n := 0
	for _, q := range benchmarkQuestions {
		if q.Match != "contains" && q.Match != "final-contains" {
			continue
		}
		n++
		for _, ans := range []string{
			q.Expect,
			"The answer is " + q.Expect + ".",
			// The shape that broke: the answer, then a sentence about it.
			"The answer is " + q.Expect + ". It is the one the rules give.",
			"Answer: " + q.Expect + "\nExplanation: it follows from the above.",
			q.Expect + "\n- because the rules say so",
			"The answer is " + q.Expect + ". Let me explain why below.",
		} {
			if !checkAnswer(q, ans) {
				t.Errorf("tier-%d %s question does not grade its own answer %q on reply %q: %s",
					q.Tier, q.Match, q.Expect, ans, benchSnippet(q.Prompt))
			}
		}
	}
	if n == 0 {
		t.Fatal("no contains/final-contains questions found — the sanity band is what proves the grader works")
	}
}

// ── DEFECT 4/5/6: the no-think pass's arithmetic ───────────────────────────

// benchQuizServer answers the given questions, optionally failing some of them
// with a 500 when thinking is disabled — the shape of a worker whose no-think
// path is broken or rate-limited. `delay` is how long an answer takes, which
// matters to the caller that checks recorded latency: a local httptest server
// answers in under a millisecond, so a real measurement and a dropped one both
// round to zero. It reports how many requests it saw.
func benchQuizServer(t *testing.T, qs []benchmarkQ, errWhenNoThink map[string]bool, delay time.Duration) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	answers := map[string]benchmarkQ{}
	for _, q := range qs {
		answers[q.Prompt] = q
	}
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		hits.Add(1)
		time.Sleep(delay)
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
		if !think && errWhenNoThink[prompt] {
			http.Error(w, "internal error", http.StatusInternalServerError)
			return
		}
		answer := "unknown question"
		if q, ok := answers[prompt]; ok {
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
	return srv, &hits
}

// An ERRORED question is one we could not GRADE, so it stays out of the
// denominator — a zero over a one is arithmetically identical to a wrong answer.
// v41 applied that to the mixed pass and not to its twin here, and the two scores
// are then compared against each other on one absolute scale: the asymmetry
// manufactured the exact signal the two-score design exists to detect.
func TestNoThinkScoreExcludesErroredQuestions(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	benchmarkQuestions = []benchmarkQ{
		{Tier: 1, Prompt: "easy-1", Expect: "1", Match: "numeric"},
		{Tier: 1, Prompt: "easy-2", Expect: "2", Match: "numeric"},
	}
	broken := map[string]bool{}
	for i := 0; i < 10; i++ {
		p := "hard-" + string(rune('a'+i))
		benchmarkQuestions = append(benchmarkQuestions,
			benchmarkQ{Tier: 5, Prompt: p, Expect: "42", Match: "numeric"})
		if i < 4 { // four of the ten are unreachable with thinking off
			broken[p] = true
		}
	}
	srv, _ := benchQuizServer(t, benchmarkQuestions, broken, 0)
	defer srv.Close()

	r := &Router{benchClient: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL, Model: "m"}}
	mixed, ok, _, _, details := r.runQualityBenchmark(b, 12)
	if !ok || mixed != 100 {
		t.Fatalf("mixed pass: score=%d ok=%v, want 100", mixed, ok)
	}
	nt, ntOK, breakdown, _ := r.runNoThinkQualityBenchmark(b, 12, details)
	if !ntOK {
		t.Fatal("no-think pass reported unmeasurable; 4 errors of 12 asked is not an outage")
	}
	// Six of six answerable hard questions right, both easy tiers carried over
	// right: the worker got everything it was asked. Counting the four errors in
	// the denominator scores it 67 and reads as a no-think collapse.
	if nt != 100 {
		t.Errorf("no-think score=%d (%s), want 100 — an unreachable question is not a wrong answer",
			nt, breakdown)
	}
}

// The give-up guard has to divide by what the pass actually ASKED — historically
// it divided by the questions the pass INTENDED to ask, which under the old
// wall-clock budget let an all-errored truncated run persist a near-zero score
// with ok=true. The budget is gone, but a cached mixed pass recorded under it
// can still carry skips, so the pass can still ask fewer questions than the
// hard-tier set holds.
func TestNoThinkGiveUpGuardCountsWhatItAsked(t *testing.T) {
	savedQs := benchmarkQuestions
	defer func() { benchmarkQuestions = savedQs }()
	benchmarkQuestions = nil
	broken := map[string]bool{}
	for i := 0; i < 6; i++ {
		p := "hard-" + string(rune('a'+i))
		benchmarkQuestions = append(benchmarkQuestions,
			benchmarkQ{Tier: 5, Prompt: p, Expect: "42", Match: "numeric"})
		broken[p] = true // every question this pass can reach will error
	}
	srv, _ := benchQuizServer(t, benchmarkQuestions, broken, 0)
	defer srv.Close()

	r := &Router{benchClient: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL, Model: "m"}}
	// A cached mixed pass shaped like an old budget-truncated run: it answered
	// the first two hard questions thinking-ON and never reached the rest.
	mixed := make([]BenchResult, len(benchmarkQuestions))
	for i, q := range benchmarkQuestions {
		if i < 2 {
			mixed[i] = BenchResult{Tier: q.Tier, Prompt: q.Prompt, Expect: q.Expect, Pass: true, LatencyMS: 5}
		} else {
			mixed[i] = BenchResult{Tier: q.Tier, Prompt: q.Prompt, Expect: q.Expect, Skipped: true}
		}
	}
	score, ok, _, _ := r.runNoThinkQualityBenchmark(b, 4, mixed)
	if ok {
		t.Errorf("a pass that errored on everything it dispatched returned score=%d with ok=true; "+
			"ok=true is what persists the profile", score)
	}
}

// The easy tiers are carried over from the mixed pass rather than re-asked, and
// "carried over verbatim" has to include the latency and the answer text.
// profile.go appends these rows AFTER the mixed rows into the outcome matrix,
// where record() supersedes on (QID, Backend, Thinking, Source) — and an easy
// tier is Thinking=false in both passes, so a zeroed row overwrote the measured
// one and destroyed the latency prediction for easy no-think traffic.
func TestCarriedOverEasyTiersKeepLatencyAndAnswer(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	benchmarkQuestions = []benchmarkQ{
		{Tier: 1, Prompt: "easy-1", Expect: "1", Match: "numeric"},
		{Tier: 1, Prompt: "easy-2", Expect: "2", Match: "numeric"},
		{Tier: 5, Prompt: "hard-1", Expect: "42", Match: "numeric"},
	}
	srv, _ := benchQuizServer(t, benchmarkQuestions, nil, 3*time.Millisecond)
	defer srv.Close()

	r := &Router{benchClient: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL, Model: "m"}}
	_, ok, _, _, details := r.runQualityBenchmark(b, 3)
	if !ok {
		t.Fatal("mixed pass did not score")
	}
	_, ntOK, _, ntResults := r.runNoThinkQualityBenchmark(b, 3, details)
	if !ntOK {
		t.Fatal("no-think pass did not score")
	}
	mixedByPrompt := map[string]BenchResult{}
	for _, d := range details {
		mixedByPrompt[d.Prompt] = d
	}
	for _, got := range ntResults {
		if got.Tier >= benchHardTier {
			continue // re-asked, not carried
		}
		want := mixedByPrompt[got.Prompt]
		if got.LatencyMS != want.LatencyMS {
			t.Errorf("carried-over %s: LatencyMS=%d, want the mixed pass's %d — a zero here "+
				"supersedes the measured row in the outcome matrix", got.Prompt, got.LatencyMS, want.LatencyMS)
		}
		if got.Got != want.Got {
			t.Errorf("carried-over %s: Got=%q, want the mixed pass's %q", got.Prompt, got.Got, want.Got)
		}
		if got.LatencyMS == 0 {
			t.Errorf("carried-over %s recorded no latency at all", got.Prompt)
		}
	}
}

// ── DEFECT 7: the busy-retry loop had no per-question cap ──────────────────

// The busy path `continue`d, so it never reached the attempt cap, and its only
// exit was a SHARED streak that any other question's success resets. On a
// rate-limited endpoint that answers some requests and 429s the rest, the
// unlucky ones retry at 1/sec forever: wg.Wait() never returns, profileBackend
// never returns, and the caller's `defer r.profiling.Delete(id)` never runs — so
// the worker shows "profiling" for the life of the process.
func TestBusyRetriesAreBoundedPerQuestion(t *testing.T) {
	savedWait, savedRetries := benchBusyMaxWait, benchBusyMaxRetries
	defer func() { benchBusyMaxWait, benchBusyMaxRetries = savedWait, savedRetries }()
	benchBusyMaxWait, benchBusyMaxRetries = 10*time.Millisecond, 3

	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		http.Error(w, "no capacity", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	r := &Router{benchClient: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL, Model: "m"}}
	// A tracker whose streak is kept alive by "other questions" succeeding, which
	// is the case the shared breaker cannot catch.
	busy := &benchBusyTracker{}
	done := make(chan benchOutcome, 1)
	go func() {
		go func() {
			for i := 0; i < 200; i++ {
				busy.ok()
				time.Sleep(time.Millisecond)
			}
		}()
		done <- r.benchOne(b, benchmarkQ{Tier: 1, Prompt: "q", Expect: "1", Match: "numeric"}, false, busy)
	}()
	select {
	case res := <-done:
		if !res.errd {
			t.Errorf("a question that never got a slot was recorded as errd=%v pass=%v — it must be "+
				"an error, not a wrong answer (got %q)", res.errd, res.pass, res.got)
		}
		if res.pass || res.slow {
			t.Errorf("busy exhaustion recorded as pass=%v slow=%v", res.pass, res.slow)
		}
	case <-time.After(30 * time.Second):
		t.Fatalf("benchOne never returned after %d busy responses: the retry loop has no "+
			"per-question bound, and this hangs the whole profile", hits.Load())
	}
}

// ── DEFECT 8: the numeric suffix contradicted the question's own format ────

// benchRequestFor appends " Give the number only." so a verbose reply stays
// parseable — but 21 of the LiveBench spatial items already say "put your answer
// in **bold** as a single integer". The worker was told to do two contradictory
// things in one prompt, so whatever it did violated one of them.
func TestNumericSuffixSkipsPromptsThatStateTheirOwnFormat(t *testing.T) {
	states := []string{
		"How many pieces are there after the cuts?  Think step by step, and then put your answer in **bold** as a single integer (for example, **0**). If you don't know, guess.",
		"How many blocks?  Give your answer as a single integer.",
		"How many? Answer with just the number, in the following format: ***X***.",
		"Count the vertices. Format your answer as a bare integer.",
		"How many cows? Give the number only.",
		"What does it print? Give only the output.",
	}
	for _, p := range states {
		q := benchmarkQ{Tier: 4, Match: "numeric", Prompt: p, Expect: "4"}
		got, _ := benchRequestFor(q, true, 0)
		if strings.Contains(got, "Give the number only.") && !strings.Contains(p, "Give the number only.") {
			t.Errorf("a prompt that states its own answer format was given a contradicting one:\n  %q", got)
		}
	}
	// …and a bare numeric question still gets it, or the suffix stops doing the
	// job it exists for.
	bare := benchmarkQ{Tier: 1, Match: "numeric", Prompt: "What is 8 + 5?", Expect: "13"}
	got, _ := benchRequestFor(bare, true, 0)
	if !strings.Contains(got, "Give the number only.") {
		t.Errorf("a bare numeric prompt lost its format instruction: %q", got)
	}
	// Non-numeric questions never got the suffix and still must not.
	mcq := benchmarkQ{Tier: 9, Match: "mcq", Prompt: "Which one?\nA) x\nB) y", Expect: "B"}
	if got, _ := benchRequestFor(mcq, true, 0); strings.Contains(got, "Give the number only.") {
		t.Errorf("an mcq question was told to give a number: %q", got)
	}
}

// The corpus guard, which is the one that cannot rot: no question in the bank may
// carry both its own answer-format instruction and the appended one. Detected on
// the prompt text rather than a list of ids, because `bench emit` regenerates the
// bank.
func TestNoBankQuestionGetsTwoAnswerFormats(t *testing.T) {
	own := []string{"put your answer in", "in the following format", "as a single integer", "in **bold**"}
	carrying, contradicted := 0, 0
	for _, q := range benchmarkQuestions {
		if q.Match != "numeric" {
			continue
		}
		lower := strings.ToLower(q.Prompt)
		stated := false
		for _, m := range own {
			if strings.Contains(lower, m) {
				stated = true
				break
			}
		}
		if !stated {
			continue
		}
		carrying++
		prompt, _ := benchRequestFor(q, true, 0)
		if strings.Contains(prompt, "Give the number only.") {
			contradicted++
			if contradicted == 1 {
				t.Errorf("this question states its own answer format and was given a contradicting one; "+
					"whatever the model does violates an instruction: %s", benchSnippet(q.Prompt))
			}
		}
	}
	if carrying == 0 {
		t.Skip("no bank question states its own answer format — nothing to contradict")
	}
	if contradicted > 0 {
		t.Errorf("%d of %d numeric questions that state their own answer format were given a second one",
			contradicted, carrying)
	}
}
