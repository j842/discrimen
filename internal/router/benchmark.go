package router

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"net"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// benchmarkVersion is bumped whenever the question set or profiling changes, so a
// cached profile measured against an older version is re-profiled, not reused.
// v24: a question the worker can't answer within benchAnswerDeadline is scored a
// speed fail (counted wrong, not retried) instead of a retried transport error.
// v25: length-truncation now counts as a FAILURE (was excluded from the
// denominator), aligning the cold-start score with the runtime inadequacy signal
// (responseInadequate treats finish_reason "length" as inadequate). Hard tiers are
// graded thinking-on with a 16k-token cap, so a truncation there means the model
// burned its whole budget without concluding — a real failure at that difficulty.
// v26: benchAnswerDeadline cut 5m→2m, so a model too slow to answer within 2 min is
// scored a speed fail at that difficulty — spreading slow reasoning models down the
// quality range and shortening profiling. Re-profiles since the scoring changed.
// v27: thinkingProbe now recognises the "reasoning" field (vLLM ≥0.23 renamed
// reasoning_content → reasoning), so vLLM workers no longer false-negative as
// thinking-incapable and get routed around in favour of slow CPU fallbacks.
// v28: all response extraction unified on extract.go (reasoning-aware speed/
// chat probes, benchmark grading falls back to the reasoning field, "<nil>"
// content bug fixed) and capability probes retry transients — prior profiles
// may carry false negatives from any of those, so re-measure.
// v29: added tier 9, twelve GPQA-Diamond-style expert-knowledge questions (ten-option
// MCQ + gated numerics), because tiers 5–8 are almost all maths and that class compresses
// thinking-on — a reasoning-tuned 4B lands within a few points of a 32B on competition
// arithmetic, while knowledge-bound science spreads the same models about twice as wide
// (see benchmark_data.go). The mcq grader was widened A-D → A-J to read the ten-option
// picks. Every worker re-profiles: the denominator grew and the top of the range moved.
// v30: added tier 10 after a Qwen 27B scored 100% on tier 9 — recall saturated from below,
// so the ceiling tier now uses what can't be recalled: rules defined in the prompt, priors
// that are actively wrong, and reasoning traces to audit. Deliberately compact, because
// BBEH-length problems truncate every worker alike and measure endurance, not reasoning.
// v31: the 27B aced v30 as well (97/97). Cut the expert-recall tier outright — measured at
// 17 points of spread, the worst ratio in the file — promoted the unrecallable tier to 9,
// and made 10 a SimpleBench-style world-model trap tier, the family the fleet's own per-tier
// numbers rank highest (tier 6: 62 points). Question count is unchanged at 97, so profiling
// takes no longer than v30 despite the new ceiling.
// v32: moved the two trap questions out of tier 2 (the train/tunnel one and the
// Mary's-father one) down to tier 3, so both are graded thinking-ON. A trap graded
// thinking-off measures one-shot reflex rather than capability, and tier 2's charter is
// "the weakest ~1B fails, every competent model passes" — so the fleet's STRONGEST model
// failing both was the tell that the labels, not the model, were wrong. A DeepSeek-V4-Flash
// 284B answered "200" (the distance) and "Lulu" thinking-off, and "2" and "Mary" thinking-on,
// which cost it two points while it went 91/91 across every thinking-on tier including the
// tier-10 ceiling. The train question was also the one place where the tier disagreed with
// the router's own reasoning gate in the direction that COSTS a worker — auditThinkingGate
// scores it r=0.69, well over the 0.35 threshold, so production serves it thinking-on while
// the tier forced it off, grading an answer that never occurs in production. Mary's father
// now sits beside Johnny's mother, the identical trap, which has always been tier 3.
// benchHardTier stays 3 and the audit stays an INDEPENDENT cross-check: the fix here is the
// labels, not the gate.
// v33: the frontier tiers (>= benchFrontierTier) got a longer answer deadline, and tier 11
// was added. See benchAnswerDeadlineFrontier below.
// v34: quality became a WEIGHTED two-bucket score instead of a flat percentage. See
// runQualityBenchmark.
//
// v33 and v34 both landed WITHOUT bumping this constant, which is the failure this constant
// exists to prevent: a profile measured under the flat-percentage rules stayed in
// worker_profiles marked current, and autoTargetQuality reads every cached score as one
// absolute 0-100 scale, so pre-v33 and post-v34 workers were being compared on scales that
// no longer meant the same thing. Bumped here to 34 to invalidate them and re-measure.
// TestBenchmarkVersionCoversScoringChanges now fails CI if this happens again.
// v35: tier 12 (programming / coding-agent fitness) added, and the weighted score grew a
// THIRD bucket for it. Also: the numeric grader now keeps the sign (benchNumberRe), which
// silently mis-graded any negative answer in both directions — see the var block.
// v36: the tier-10 lift question is revised — it was under-specified, and it was costing
// the fleet far more than the one point it was worth. Nothing in it stated that the express
// lift beats a man running up 30 flights, and "H) it depends on the lift's speed" was then
// offered as an option, so H was defensible as written. Three of the four benchmarked
// workers spent their ENTIRE budget on that single item rather than answering it: a
// Qwen3.8-27B burned all 16384 thinking tokens without emitting one content token (the
// trunc=1 in its breakdown), and a 7.9B and a 26B both hit the frontier deadline on it as
// speed-fails. Reading the 27B's trace — which the tier's own CAVEAT asks for before
// blaming the model — shows it never simulating the scenario at all: it oscillates between
// C, D, H and I doing puzzle-author psychology on the option list ("If rigorous eval: I
// (40%) — If puzzle author casual: H (30%) or D (25%)"). Raising the ceiling is not the fix
// either; the same worker given 32k concludes in 12k tokens and answers I.
//
// The fix is that the item now leaves NO quantity unstated, which is what separates the
// tier's clean items from this one. Measured on the 27B, pinning one side was not enough:
// with the lift alone pinned ("reaches the roof terrace in under a minute") it stopped
// truncating but still spent 6907 thinking tokens and answered I, because Dan's climb was
// still a judgement call it refused to make. With both sides pinned (Dan's stairs "a little
// over five minutes") it answers C in 399 thinking tokens, and all four benchmarked workers
// now pass it in seconds. H is also replaced by an age decoy in the style of the Rosa
// question, keeping ten options and keeping a wrong answer available to a model that does
// fall for the red herring.
//
// KNOWN TRADE-OFF: with 4/4 workers passing it, this item no longer spreads the fleet — it
// is now a guard like the ice-cube question rather than a discriminator. That is still worth
// more than it was, because an item that eats three workers' whole budgets measures
// endurance rather than reasoning, which this tier is explicitly built not to do. If tier 10
// needs its twelfth discriminating item back, the replacement should be written compact and
// fully specified from the start, and its spread measured across the fleet before it lands.
// v37: loose half-credit grading. A reply that fails strict extraction but visibly
// ASSERTS the expected answer — emphasised ("**114**"), as its final stated equation
// ("… = 42"), an emphasised option pick ("**B) backward…**"), or the expected option's
// own text — scores HALF a point, tallied separately as "loose". Motivated by
// Ornith-1.5-35B-A3B (2026-08-24): 8 of its 12 misses were semantically correct answers
// that ignored "Give the number only" / "Answer with just the letter", so its q=89
// measured formatting, not knowledge. Strict grading is unchanged — nobody's full
// credit moved — loose only ADDS credit, and the tally keeps the formatting cost
// visible per worker.
const benchmarkVersion = 37

// benchmarkQ is one graded question in the cold-start quality benchmark. The
// question set lives in benchmark_data.go.
type benchmarkQ struct {
	Tier   int    // difficulty band 1 (control) … 11 (budget-bounded insight) … 12 (programming); also sets grading mode — tiers >= benchHardTier are graded thinking-on (see benchmark_data.go)
	Prompt string // user prompt sent to the worker
	Expect string // expected answer token (see Match)
	Match  string // "contains" | "numeric" | "mcq" | "final-contains" (see checkAnswer)
}

// BenchResult is one graded question's full outcome, stored in the worker's
// profile so the exact questions + the model's actual answers from the most
// recent run can be inspected later (GET /backends/{id}/benchmark, `ask -qa`).
type BenchResult struct {
	Tier      int    `json:"tier"`
	Prompt    string `json:"prompt"`
	Expect    string `json:"expect"`
	Got       string `json:"got"` // the model's answer (tail-truncated)
	Pass      bool   `json:"pass"`
	Truncated bool   `json:"truncated,omitempty"`
	Errored   bool   `json:"errored,omitempty"`
	Slow      bool   `json:"slow,omitempty"`  // exceeded benchAnswerDeadline — too slow to be usable, scored a fail (not a transport error)
	Loose     bool   `json:"loose,omitempty"` // failed strict extraction but asserts the expected answer — half credit (see checkAnswerLoose)
}

var (
	// v35: the sign is part of the number. Without `-?` a question whose Expect is
	// negative could never match — every extraction tier dropped the minus before
	// numericMatches compared — and, worse, the reverse graded as a PASS: Expect "2"
	// against an answer of "-2" read as 2. Nothing in tiers 1–11 has a negative answer,
	// so the fault was dormant and would have surfaced as a model getting an easy
	// question wrong rather than as a grader fault. The sign is only taken when it is
	// adjacent to the digits, so a spaced subtraction ("10 - 3 = 7") still reads its
	// operands unsigned.
	benchNumberRe = regexp.MustCompile(`-?[0-9]+(?:\.[0-9]+)?`)
	// mcq pick extraction, tried in priority order by checkAnswer: an explicit
	// declaration ("the answer is B", "Answer: C"), a letter leading the answer
	// ("C. because…", "(B)"), then the last standalone letter. The fallback class
	// admits uppercase plus lowercase b/c only: a bare lowercase "a" or "d" is
	// almost always the article or the "I'd" contraction, which is how a plain
	// (?i)\b[a-d]\b misread prose answers ("…causing a syntax error" → "a").
	//
	// A declared UPPERCASE pick anchors on a plain word boundary ("The answer is C
	// because…"); a lowercase one still needs trailing punctuation/EOL so that
	// "the answer is a syntax error" can never anchor on the article "a".
	//
	// The range is A-J, not A-D, because the tier-9 questions carry TEN options
	// (MMLU-Pro's design: ten choices drop the random-guess floor from 25% to 10%
	// and cut prompt-sensitivity variance, which matters here because every question
	// is scored pass/fail and guess-luck is pure noise in the percentage). "I" is the
	// one letter that can't join the permissive tiers: bare "I" is the pronoun and
	// opens a huge share of reasoning prose ("I need to check…"), so it is excluded
	// from the last-standalone-letter class entirely and, when declared, must carry
	// trailing punctuation/EOL — the same guard the lowercase letters get, so that
	// "the answer is I think C" can never anchor on the pronoun. A model that picks
	// option I still grades correctly via "Answer: I." or "I)"; one that answers a
	// bare "I" grades as a miss, which is why no tier-9 question uses I as its key.
	benchDeclaredRe = regexp.MustCompile(`(?i)\banswer\s*(?:is)?\s*:?\s*\(?(?:(?-i:([A-HJ]))\b|([a-hj])(?:\s*[).:,!—–-]|\s*$)|([i])(?:\s*[).:,!—–-]|\s*$))`)
	benchLeadingRe  = regexp.MustCompile(`(?i)^\s*\(?([a-j])(?:[).:,\n]|\s*$)`)
	benchLetterRe   = regexp.MustCompile(`\b[A-HJbc]\b`)
	// benchEnumRe finds option anchors — an A-J letter opening the string, a line, or
	// a clause, written "X)" / "X." — so checkAnswer can tell an answer that
	// ENUMERATES the options ("A) … no. B) … no. C) … yes.") from one that leads
	// with its pick (see benchEnumerates). Widening to A-J lets "i.e."/"e.g." read as
	// two option anchors and flag a non-enumerating answer as enumerating; the only
	// consequence is that the leading-letter rule stands aside and the last
	// standalone letter decides, which for those replies is the same pick.
	benchEnumRe = regexp.MustCompile(`(?i)(?:^|[.!?;:\n—–])\s*\(?([a-j])[).]`)

	// Numeric mirrors mcq's tiers: an explicitly declared value ("the answer is 1",
	// "total: 16", "5 is the answer") is graded first — the LAST declaration, so a
	// self-correction ("…= 2 … wait, actually the answer is 1") grades by its final
	// claim — then a leading-clause assertion (benchLeadNumber), then the last number.
	// v35: both capture groups take an optional sign, for the same reason as
	// benchNumberRe — "the answer is -2" must grade as -2, not 2.
	benchNumDeclaredRe = regexp.MustCompile(`\b(?:answers?\s*(?:is|are|was)?\s*[:=]?\s*|(?:totals?|results?)\s*(?:is|are|was|:|=)\s*)\(?(-?[0-9]+(?:\.[0-9]+)?)\b|\b(-?[0-9]+(?:\.[0-9]+)?)\s+is\s+the\s+answer\b`)
	// benchThousandsRe collapses thousands separators ("1,024" → "1024") so a comma
	// inside a number is never read as a clause break by benchLeadNumber.
	benchThousandsRe = regexp.MustCompile(`[0-9]+(?:,[0-9]{3})+`)
	// benchLeadBreakRe ends the leading clause of a numeric answer: clause
	// punctuation, parens, or the conjunction "and" ("…costs 5 cents and the bat
	// costs 105 cents"). A '.'/',' between digits is part of a number and is
	// skipped by benchLeadNumber.
	benchLeadBreakRe = regexp.MustCompile(`[.!?;:,\n—–()]|\s-\s|\band\b`)
	// benchNumListRe matches a line-leading list marker ("1. ", "2) "); two or more
	// mean the answer is a numbered list, whose lead number is a marker, not the answer.
	benchNumListRe = regexp.MustCompile(`(?m)^\s*[0-9]{1,2}[.)]\s`)

	// v37 loose grading (checkAnswerLoose). benchEmphasisRe captures a markdown
	// bold/italic span — a model that ignores "give the number only" still marks its
	// answer up ("**114**", "**Step 6 is wrong.**", "**B) backward…**"), so an
	// emphasised span is where an answer is ASSERTED, unlike arbitrary prose. Spans are
	// single-line and short; backticks are code, not emphasis, and are not read.
	benchEmphasisRe = regexp.MustCompile(`\*{1,2}([^*\n]{1,120}?)\*{1,2}`)
	// benchLastEqRe reads a stated equation result ("… = 42", "… ≡ 77 (mod 100)"); the
	// LAST one in the reply is the model's final claim, mirroring the last-number rule.
	// Only the value directly after =/≡ counts, so "1 + 2 + 12" operands never match.
	benchLastEqRe = regexp.MustCompile(`[=≡]\s*\(?(-?[0-9]+(?:\.[0-9]+)?)`)
)

// Each question is graded in the MODE THE ROUTER WOULD SERVE IT IN: the easy tiers
// (below benchHardTier) thinking-off, the hard tiers thinking-on. This measures every
// worker the way production actually uses it — so a reasoning-first model is rated on
// its reasoning, with the scratchpad it gets in production, instead of on a thinking-off
// reflex it never runs at that difficulty (which is what used to rank it below a flatter
// model). The trade-off: the hard tiers must stay hard ENOUGH that a scratchpad alone
// doesn't let a weak model catch a strong one, or the fleet re-saturates at q≈9–10 and
// the quality range difficulty-routing needs collapses (see benchmark_data.go).
const (
	benchMaxTokens      = 1024  // token ceiling per thinking-off answer (short trap/recall replies)
	benchThinkMaxTokens = 16384 // token ceiling per thinking-on answer — hard reasoning needs headroom (8192 truncated some tier-7/8 questions)

	// benchHardTier is the difficulty at and above which a question is graded
	// THINKING-ON (and below which, thinking-off) — the same boundary the router's
	// production reasoning gate is expected to cross, so the benchmark serves each
	// prompt the way production would. auditThinkingGate cross-checks that the gate's
	// learned decision agrees with this hand-labelled boundary.
	benchHardTier = 3

	// A profiling request is retried up to benchMaxAttempts times before it counts
	// as errored: a dropped request under concurrent load is not a wrong
	// answer. Backoff grows per attempt (benchRetryBackoff × attempt). A request that
	// exceeds benchAnswerDeadline is the exception — it is NOT retried (see below).
	benchMaxAttempts  = 3
	benchRetryBackoff = time.Second

	// benchAnswerDeadline is the hard per-question usability bound. A worker that needs
	// longer than this to answer ANY question is too slow to be usable at that
	// difficulty, so the question is scored a FAIL (a speed fail, counted like a wrong
	// answer) and is NOT retried — a retry would only burn another deadline for the same
	// verdict. This is a benchmark criterion, independent of the live-proxy
	// BACKEND_TIMEOUT_SECONDS: benchCompletion uses a no-timeout client so this deadline
	// alone decides whether the answer arrived in time.
	benchAnswerDeadline = 2 * time.Minute // cut from 5m (v26): too-slow answers fail, spreading slow reasoning models

	// v33: the frontier tiers (>= benchFrontierTier) get a longer deadline. The 2-minute
	// bound scored a 284B MoE (18 tok/s thinking-on) BELOW a saturated 27B on nothing but
	// tier-10 speed-fails — its only misses in the whole set were "(too slow)", every
	// answer it finished was right. At these tiers the question is whether the model can
	// reason at all, and production already prices patience: the router ranks candidates
	// by expected completion time per request, so a slow-but-right worker loses the race
	// for quick prompts without its QUALITY being falsified first. Tiers 3–8 keep the
	// 2-minute bound — that spread of slow mid-tier reasoners (v26) is real and wanted.
	benchFrontierTier           = 9
	benchAnswerDeadlineFrontier = 6 * time.Minute

	// v34/v35: weighted scoring split (see runQualityBenchmark). Three buckets now, each
	// internally count-independent so questions can still be added freely within one:
	//
	//	base    tiers 1 .. benchInsightTier-1   benchBaseWeight    (general reasoning)
	//	insight tier  benchInsightTier          benchInsightWeight (budget-bounded insight)
	//	coding  tiers benchCodingTier and up    benchCodingWeight  (programming)
	//
	// v34 used two buckets with the insight one open-ended (`t >= benchInsightTier`).
	// Appending tier 12 to that would have put 28 coding questions in the same 30-point
	// bucket as tier 11's 5, cutting tier 11's contribution from 30 points to 4.5 and
	// undoing exactly what v34 was for — the top 30 points existed to separate models
	// that tiers 1–10 cannot. Coding therefore gets its own weight rather than sharing.
	//
	// The split is 60/20/20 rather than 70/30 plus an extra: a worker that sweeps the
	// general tiers but cannot read code now caps at 80, and one that cannot do either
	// hard band caps at 60.
	benchInsightTier   = 11
	benchCodingTier    = 12
	benchBaseWeight    = 60
	benchInsightWeight = 20
	benchCodingWeight  = 20
)

// benchOutcome is the graded result of one benchmark question against one worker.
type benchOutcome struct {
	tier  int
	pass  bool
	errd  bool
	slow  bool
	trunc bool
	loose bool
	got   string
}

// benchOne asks the worker one benchmark question in the given thinking mode and
// grades the answer. Extracted from runQualityBenchmark's loop so the no-think
// scoring pass (runNoThinkQualityBenchmark) asks questions EXACTLY the way the
// main benchmark does — a second copy of the request/grading logic would drift.
func (r *Router) benchOne(b *Backend, q benchmarkQ, think bool) benchOutcome {
	res := benchOutcome{tier: q.Tier}
	maxTokens := benchMaxTokens
	prompt := q.Prompt
	// Only NUMERIC grading is fooled by a verbose reasoned reply (an intermediate or a
	// trailing restatement gets read as the answer), so make the worker answer with
	// just the number — unless the prompt already says so. mcq (letter) and contains
	// (substring) grade fine through a long reply, so they need no extra instruction.
	if q.Match == "numeric" && !strings.Contains(q.Prompt, "number only") && !strings.Contains(q.Prompt, "only the output") {
		prompt += " Give the number only."
	}
	if think {
		maxTokens = benchThinkMaxTokens
	} else {
		prompt += " /no_think" // belt-and-suspenders with the kwarg (matches chatProbe)
	}
	// Greedy decoding (temperature 0): a graded benchmark must be
	// deterministic, so the same (model, question) returns the same answer on
	// every run and two identical models on different hosts score identically.
	// With default-temperature sampling the quality score wobbles ±1–2 between
	// re-profiles purely from the random draw — exactly the noise that made a
	// model's two instances disagree. vLLM and llama.cpp both treat
	// temperature 0 as argmax. (Heterogeneous hardware can still differ on a
	// rare near-tie in the logits, but the sampling noise — the dominant
	// source — is gone.)
	payload := map[string]any{
		"model":                probeModel(b),
		"stream":               false,
		"max_tokens":           maxTokens,
		"temperature":          0,
		"chat_template_kwargs": map[string]bool{"enable_thinking": think},
		"messages":             []map[string]string{{"role": "user", "content": prompt}},
	}
	// Two failure modes are graded differently. A request that exceeds the
	// per-question usability deadline (benchAnswerDeadline) is a SPEED fail —
	// too slow to be usable is a quality fail for this question, not a transport
	// hiccup, so it is recorded immediately and NOT retried (a retry would just
	// burn another deadline for the same verdict). Any other failure is treated
	// as transient (a dropped request under concurrent profiling load) and
	// retried with backoff to let the congestion clear; only one that still fails
	// every attempt is recorded as errored (worker likely down).
	var raw map[string]any
	var err error
	var slow bool
	deadline := benchAnswerDeadline
	if q.Tier >= benchFrontierTier {
		deadline = benchAnswerDeadlineFrontier // v33: frontier tiers measure capability, not patience
	}
	for attempt := 1; attempt <= benchMaxAttempts; attempt++ {
		ctx, cancel := context.WithTimeout(context.Background(), deadline)
		raw, err = r.benchCompletion(ctx, b, payload)
		cancel()
		if err == nil {
			break
		}
		if isTimeout(err) {
			slow = true
			break // too slow → a usability fail for this question, never retried
		}
		if attempt < benchMaxAttempts {
			time.Sleep(benchRetryBackoff * time.Duration(attempt))
		}
	}
	switch {
	case slow:
		res.slow = true // answered too slowly to be usable — counts as a wrong answer, not an outage
		res.got = "(too slow)"
	case err != nil:
		res.errd = true // every attempt failed (≠ a wrong answer) — worker likely down
		res.got = "(no response)"
	default:
		content, finish := completionContent(raw)
		res.trunc = finish == "length"
		res.got = answerTail(content)
		if checkAnswer(q, content) {
			res.pass = true
		} else if checkAnswerLoose(q, content) {
			res.loose = true // right answer, ignored the requested format — half credit
		}
	}
	return res
}

// runQualityBenchmark grades the worker against benchmarkQuestions and scores it as the
// PERCENTAGE of questions answered correctly (0–100): every question counts the same
// regardless of tier, so a model holds a high score only by staying correct through the
// hard tiers too, and the score is independent of how many questions the set has — so
// questions can be added without rescaling anything downstream. The frontier tiers (7–8)
// are hard enough that a perfect 100 stays out of reach, so the realistic ceiling sits
// just below it.
// A question the worker can't answer within the per-question usability deadline
// (benchAnswerDeadline) is scored a FAIL — too slow to be usable — and is not retried.
// Other transient request failures ARE retried; a length-truncated answer (the model
// ran out of token budget without concluding) counts as a FAILURE — the same signal the
// runtime adapter treats as inadequate (responseInadequate), so the cold-start score and
// the live inadequacy signal agree — while a request that still errors after every retry,
// producing nothing, also counts as a failure. (A truncated answer that still parsed to
// the correct result is rare but counts as a pass.) The truncated COUNT is still reported
// in the breakdown so a context/budget problem stays visible. This is the measured
// replacement for the old self-declared LLM_WORKER_QUALITY.
func (r *Router) runQualityBenchmark(b *Backend, concurrency int) (score int, ok bool, breakdown string, failed []string, details []BenchResult) {
	if len(benchmarkQuestions) == 0 {
		return 50, true, "", nil, nil
	}
	if concurrency < 1 {
		concurrency = 1
	}
	// Grade questions concurrently (bounded by the worker's measured capacity) so a
	// thinking-on benchmark finishes in minutes rather than ~20.
	//
	// Grade in the mode the router serves this difficulty in WHEN IT CHOOSES: hard
	// tiers thinking-on (with the reasoning headroom), easy tiers thinking-off with
	// a tight cap (see the const block and benchmark_data.go). This mixed-mode
	// score is what router-decided traffic experiences; a request whose client
	// forces thinking OFF is scored separately — see runNoThinkQualityBenchmark.
	results := make([]benchOutcome, len(benchmarkQuestions))
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup
	for i := range benchmarkQuestions {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()
			q := benchmarkQuestions[i]
			results[i] = r.benchOne(b, q, q.Tier >= benchHardTier)
		}(i)
	}
	wg.Wait()

	maxTier := 1
	for _, q := range benchmarkQuestions {
		if q.Tier > maxTier {
			maxTier = q.Tier
		}
	}
	pass := make([]int, maxTier+1)
	loose := make([]int, maxTier+1)
	count := make([]int, maxTier+1)
	errored, slowFailed, truncated, looseTotal := 0, 0, 0, 0
	for i, res := range results {
		q := benchmarkQuestions[i]
		count[res.tier]++
		switch {
		case res.errd:
			errored++
		case res.slow:
			slowFailed++ // too slow to be usable — a normal failure (stays in the denominator), not an outage
		case res.pass:
			pass[res.tier]++
		case res.loose:
			loose[res.tier]++ // semantically right, wrong format — half credit
			looseTotal++
		}
		// A length-truncated answer is a normal failure (it stays in the denominator
		// and, unless it still parsed to the right result, didn't pass), matching the
		// runtime inadequacy signal. The count is reported below so the budget/context
		// problem stays visible even though it's no longer excluded from the score.
		if res.trunc {
			truncated++
		}
		details = append(details, BenchResult{
			Tier: q.Tier, Prompt: q.Prompt, Expect: q.Expect,
			Got: res.got, Pass: res.pass, Truncated: res.trunc, Errored: res.errd, Slow: res.slow,
			Loose: res.loose,
		})
		if !res.pass {
			reason := " (wrong)"
			if res.errd {
				reason = " (error)"
			} else if res.slow {
				reason = " (too slow)"
			} else if res.trunc {
				reason = " (truncated)"
			} else if res.loose {
				reason = " (loose: right answer, ignored format — half credit)"
			}
			failed = append(failed, fmt.Sprintf("t%d %s%s", res.tier, benchSnippet(q.Prompt), reason))
		}
	}
	// If many requests failed to even respond, the worker went unreachable mid-run
	// — it isn't "bad", we just can't grade it. Signal not-ok so the caller
	// discards this run instead of persisting an under-rating.
	if errored*2 > len(benchmarkQuestions) {
		return 0, false, "", nil, nil
	}
	// v34/v35: quality is a WEIGHTED three-bucket score — base 60 / insight 20 / coding 20
	// (see the const block for the tier boundaries and why coding is not folded in with
	// insight). Under the old flat percentage the 5 insight questions were ~5% of the
	// score, so a model that swept tiers 1–10 but couldn't touch tier 11 still read ~96 —
	// indistinguishable from one that could. Weighted, a full sweep of the general tiers
	// with neither hard band caps at 60, and the top 40 points are earned only where the
	// top models actually differ. Each bucket is internally count-independent (questions
	// can still be added freely within a bucket without rescaling), and an empty bucket's
	// weight is redistributed proportionally so the score stays comparable across sets.
	// Every question stays in its bucket's denominator: a length-truncated answer (ran out
	// of token budget without concluding), a question that exceeded the usability deadline
	// (too slow), and a request that still errors after every retry all yielded no usable
	// answer and count as a failure, the same as a wrong answer — matching what the runtime
	// adapter penalizes.
	ok2, score2 := benchWeightedScore(pass, loose, count, maxTier)
	if !ok2 {
		return 0, false, "", nil, nil
	}
	score = score2
	// Per-tier breakdown (+ any truncations) so the spread is visible and a
	// truncation problem shows up explicitly rather than as fake low quality.
	// Per-tier counts are STRICT passes; the loose tally is global, like trunc.
	var bd strings.Builder
	for t := 1; t <= maxTier; t++ {
		if count[t] > 0 {
			fmt.Fprintf(&bd, "t%d=%d/%d ", t, pass[t], count[t])
		}
	}
	if looseTotal > 0 {
		fmt.Fprintf(&bd, "loose=%d ", looseTotal)
	}
	if truncated > 0 {
		fmt.Fprintf(&bd, "trunc=%d ", truncated)
	}
	if slowFailed > 0 {
		fmt.Fprintf(&bd, "slow=%d ", slowFailed)
	}
	if errored > 0 {
		fmt.Fprintf(&bd, "err=%d ", errored)
	}
	breakdown = strings.TrimSpace(bd.String())
	log.Printf("benchmark %s: q=%d%% (errored=%d, slow=%d, conc=%d) %s", b.ID, score, errored, slowFailed, concurrency, breakdown)
	return score, true, breakdown, failed, details
}

// runNoThinkQualityBenchmark scores the worker as it answers with thinking
// DISABLED — the mode a client's requirements.thinking="off" (or
// reasoning_effort:"none", or a direct-verdict auto classification) serves in.
//
// The headline mixed-mode score grades hard tiers thinking-on, so it describes a
// worker the no-think client never talks to: measured 2026-08-24, a 35B MoE that
// benchmarked q>=84 in its thinking mode wrote deterministic garbage SQL with
// thinking suppressed, and the router kept handing it hard no-think requests
// against the thinking-mode score. Two scores end that: selection compares the
// target against the score for the mode the request will actually be served in
// (see qualityFor).
//
// Cost control: the easy tiers (below benchHardTier) were ALREADY graded
// thinking-off in the mixed run, so only the hard tiers are re-asked; their
// outcomes are merged with the mixed run's easy-tier results and scored with the
// same weighted arithmetic. `mixed` is the detail list runQualityBenchmark
// returned, aligned index-for-index with benchmarkQuestions.
func (r *Router) runNoThinkQualityBenchmark(b *Backend, concurrency int, mixed []BenchResult) (score int, ok bool, breakdown string) {
	if len(benchmarkQuestions) == 0 || len(mixed) != len(benchmarkQuestions) {
		return 0, false, ""
	}
	if concurrency < 1 {
		concurrency = 1
	}
	var hardIdx []int
	for i, q := range benchmarkQuestions {
		if q.Tier >= benchHardTier {
			hardIdx = append(hardIdx, i)
		}
	}
	rerun := make([]benchOutcome, len(hardIdx))
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup
	for j, i := range hardIdx {
		wg.Add(1)
		sem <- struct{}{}
		go func(j, i int) {
			defer wg.Done()
			defer func() { <-sem }()
			rerun[j] = r.benchOne(b, benchmarkQuestions[i], false)
		}(j, i)
	}
	wg.Wait()

	maxTier := 1
	for _, q := range benchmarkQuestions {
		if q.Tier > maxTier {
			maxTier = q.Tier
		}
	}
	pass := make([]int, maxTier+1)
	loose := make([]int, maxTier+1)
	count := make([]int, maxTier+1)
	errored, slowFailed, truncated := 0, 0, 0
	outcomes := make([]benchOutcome, len(benchmarkQuestions))
	for i, m := range mixed {
		// Easy tiers ran thinking-off in the mixed pass; carry them over verbatim.
		outcomes[i] = benchOutcome{tier: m.Tier, pass: m.Pass, loose: m.Loose,
			errd: m.Errored, slow: m.Slow, trunc: m.Truncated}
	}
	for j, i := range hardIdx {
		outcomes[i] = rerun[j]
	}
	for _, res := range outcomes {
		count[res.tier]++
		switch {
		case res.errd:
			errored++
		case res.slow:
			slowFailed++
		case res.pass:
			pass[res.tier]++
		case res.loose:
			loose[res.tier]++
		}
		if res.trunc {
			truncated++
		}
	}
	// Same give-up rule as the mixed run, judged over the questions this pass
	// actually asked: a worker that went unreachable mid-run is unmeasurable,
	// not bad.
	if errored*2 > len(hardIdx) {
		return 0, false, ""
	}
	ok2, score2 := benchWeightedScore(pass, loose, count, maxTier)
	if !ok2 {
		return 0, false, ""
	}
	var bd strings.Builder
	for t := 1; t <= maxTier; t++ {
		if count[t] > 0 {
			fmt.Fprintf(&bd, "t%d=%d/%d ", t, pass[t], count[t])
		}
	}
	if truncated > 0 {
		fmt.Fprintf(&bd, "trunc=%d ", truncated)
	}
	if slowFailed > 0 {
		fmt.Fprintf(&bd, "slow=%d ", slowFailed)
	}
	if errored > 0 {
		fmt.Fprintf(&bd, "err=%d ", errored)
	}
	log.Printf("benchmark %s (no-think): q=%d%% (errored=%d, slow=%d, conc=%d) %s",
		b.ID, score2, errored, slowFailed, concurrency, strings.TrimSpace(bd.String()))
	return score2, true, strings.TrimSpace(bd.String())
}

// needsNoThinkBackfill reports whether a cached profile is missing its no-think
// quality score AND carries enough stored state to backfill it without a full
// re-benchmark: the mixed run's per-question results, aligned with the CURRENT
// question set (the caller has already gated on BenchVersion, so a length match
// means the same questions). A non-thinking worker needs nothing — its mixed
// score already is its no-think score, and qualityFor reads Quality for it
// (exact) until its profile is rewritten anyway. A THINKING worker awaiting
// backfill ranks as unmeasured on no-think requests, which is the pressure to
// run this promptly rather than a reason to widen the fallback.
func needsNoThinkBackfill(p *WorkerProfile) bool {
	return p.Thinking && p.QualityNoThink == 0 && len(p.BenchResults) == len(benchmarkQuestions)
}

// benchWeightedScore turns the per-tier tallies into the 0-100 quality score, split
// across the three weighted buckets described in the const block. Extracted from
// runQualityBenchmark so the arithmetic can be tested directly: a second copy written
// for the test would drift from the one that actually scores workers, which is the
// failure mode the preview endpoint's shared-plan design exists to avoid.
//
// An empty bucket's weight is redistributed across the buckets that DO have questions,
// in proportion to their nominal weights, so the score stays a comparable 0-100 whatever
// the set carries. This generalises v34's `if insC == 0 { baseWeight = 100 }`: a set with
// no coding questions scores 75/25 base/insight rather than leaving 20 points unreachable,
// which would make every worker measured on it look 20 points worse than the same worker
// measured on the full set — and autoTargetQuality reads all of them on one absolute scale.
//
// Reports false when the set has no questions at all, which the caller treats as an
// unmeasurable worker rather than as a zero.
//
// v37: a loose pass (right answer, ignored the requested format — checkAnswerLoose)
// contributes HALF a point to its bucket, so knowledge is credited but a compliant
// worker still outranks a verbose one on the same answers.
func benchWeightedScore(pass, loose, count []int, maxTier int) (bool, int) {
	buckets := []struct {
		weight int
		pass   float64
		count  int
	}{
		{weight: benchBaseWeight},
		{weight: benchInsightWeight},
		{weight: benchCodingWeight},
	}
	for t := 1; t <= maxTier && t < len(count); t++ {
		i := 0
		switch {
		case t >= benchCodingTier:
			i = 2
		case t >= benchInsightTier:
			i = 1
		}
		buckets[i].pass += float64(pass[t])
		if loose != nil && t < len(loose) {
			buckets[i].pass += 0.5 * float64(loose[t])
		}
		buckets[i].count += count[t]
	}
	liveWeight := 0
	for _, b := range buckets {
		if b.count > 0 {
			liveWeight += b.weight
		}
	}
	if liveWeight == 0 {
		return false, 0
	}
	weighted := 0.0
	for _, b := range buckets {
		if b.count > 0 {
			share := 100 * float64(b.weight) / float64(liveWeight)
			weighted += share * b.pass / float64(b.count)
		}
	}
	return true, int(math.Round(weighted))
}

// isTimeout reports whether err is the request deadline firing — the benchAnswerDeadline
// context expiring, or any net-level timeout — as opposed to a transport error (connection
// refused/reset) or a non-2xx HTTP status. Only a deadline counts as a "too slow" speed
// fail; everything else is a transient error that gets retried.
func isTimeout(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var ne net.Error
	return errors.As(err, &ne) && ne.Timeout()
}

// answerTail trims the model's answer to a stored tail (the final answer sits at
// the end), rune-safe, so the per-question record stays small.
func answerTail(s string) string {
	s = strings.TrimSpace(s)
	const max = 500
	if len(s) <= max {
		return s
	}
	cut := len(s) - max
	for cut < len(s) && !utf8.RuneStart(s[cut]) {
		cut++
	}
	return "…" + s[cut:]
}

// benchSnippet shortens a question prompt to a one-line label for the failed list.
func benchSnippet(prompt string) string {
	if i := strings.IndexByte(prompt, '\n'); i >= 0 {
		prompt = prompt[:i]
	}
	prompt = strings.TrimSpace(prompt)
	if len(prompt) > 64 {
		prompt = strings.TrimSpace(prompt[:64]) + "…"
	}
	return prompt
}

// completionContent pulls the assistant text and finish_reason from a raw
// (non-streamed) completion response. Falls back to the reasoning field when
// content is empty: a thinking model that never emits its closing </think>
// leaves the ENTIRE output — final answer included — in reasoning with content
// null (both dialects), and grading "" would score a correct answer as wrong.
func completionContent(raw map[string]any) (content, finishReason string) {
	c, reasoning, finishReason := completionText(raw)
	return preferContent(c, reasoning), finishReason
}

// benchFracRe rewrites a LaTeX fraction \frac{a}{b} (also \dfrac/\tfrac) to a/b, so a
// fractional answer still matches once the surrounding braces are stripped.
var benchFracRe = regexp.MustCompile(`\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}`)

// benchBoxedRe unwraps \boxed{…} to its space-padded content before mathMarkup strips
// the braces, which would otherwise glue the markup onto the answer ("\boxed{C}" →
// "boxedC", invisible to the word-boundary letter matchers).
var benchBoxedRe = regexp.MustCompile(`\\boxed\s*\{([^{}]*)\}`)

// mathMarkup strips LaTeX / markdown scaffolding so a formatted answer matches the plain
// expected text — e.g. "$\text{H}_2\text{O}$" → "H2O", "$\mathbf{2207}$" → "2207".
var mathMarkup = strings.NewReplacer(
	`\text`, "", `\mathrm`, "", `\mathbf`, "", `\left`, "", `\right`, "",
	`\,`, "", `\!`, "", `\;`, "", `\:`, "",
	`$`, "", `{`, "", `}`, "", `_`, "", `^`, "", `\`, "",
)

// numberWords maps spelled-out numbers to digits so a worker that answers "six" instead
// of "6" grades correct on a numeric question; numberWordRe matches them on word
// boundaries so e.g. "none" isn't read as "one".
var numberWords = map[string]string{
	"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
	"six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
	"twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
	"seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
	"forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
	"ninety": "90", "hundred": "100", "thousand": "1000",
}
var numberWordRe = regexp.MustCompile(`(?i)\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b`)

// benchCompoundRe matches a hyphen/space compound number word (twenty-one … ninety-nine);
// these must be rewritten before the single number words, or "twenty-two" reads as the
// two numbers 20 and 2 — and the numeric matcher's last-number rule would grade the 2.
var benchCompoundRe = regexp.MustCompile(`(?i)\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[-\s](?:one|two|three|four|five|six|seven|eight|nine)\b`)

// compoundNumber rewrites one lower-cased compound match ("twenty-two", "forty five")
// to its digit form.
func compoundNumber(w string) string {
	parts := strings.Fields(strings.ReplaceAll(w, "-", " "))
	if len(parts) != 2 {
		return w
	}
	tens, _ := strconv.Atoi(numberWords[parts[0]])
	unit, _ := strconv.Atoi(numberWords[parts[1]])
	return strconv.Itoa(tens + unit)
}

// unicodeDigits maps sub/superscript digits to ASCII so a correctly-formatted
// answer (e.g. "H₂O", "x²") isn't marked wrong by the plain-text matchers.
var unicodeDigits = strings.NewReplacer(
	"₀", "0", "₁", "1", "₂", "2", "₃", "3", "₄", "4", "₅", "5", "₆", "6", "₇", "7", "₈", "8", "₉", "9",
	"⁰", "0", "¹", "1", "²", "2", "³", "3", "⁴", "4", "⁵", "5", "⁶", "6", "⁷", "7", "⁸", "8", "⁹", "9",
)

// normalizeBenchAnswer strips LaTeX scaffolding and unicode digit forms so a correctly
// formatted answer ("$\text{H}_2\text{O}$", "x²", "\frac{1}{7}", "\boxed{C}") still
// matches the plain expected text. Shared by strict and loose grading so both read the
// same text.
func normalizeBenchAnswer(answer string) string {
	a := strings.TrimSpace(answer)
	a = benchFracRe.ReplaceAllString(a, "${1}/${2}")
	a = benchBoxedRe.ReplaceAllString(a, " ${1} ")
	a = mathMarkup.Replace(a)
	a = unicodeDigits.Replace(a)
	return a
}

// benchDigits lowers a normalised answer to its numeric reading: spelled-out and
// compound numbers become digits, thousands separators collapse. Shared by the strict
// numeric tiers and loose grading.
func benchDigits(a string) string {
	digits := benchCompoundRe.ReplaceAllStringFunc(strings.ToLower(a), compoundNumber)
	digits = numberWordRe.ReplaceAllStringFunc(digits, func(w string) string { return numberWords[w] })
	digits = benchThousandsRe.ReplaceAllStringFunc(digits, func(w string) string { return strings.ReplaceAll(w, ",", "") })
	return digits
}

// checkAnswer reports whether answer satisfies q's expected result.
func checkAnswer(q benchmarkQ, answer string) bool {
	a := normalizeBenchAnswer(answer)
	exp := strings.TrimSpace(q.Expect)
	switch q.Match {
	case "numeric":
		// Tiered like mcq: a declared value, then a leading-clause assertion, then
		// the LAST number. The last-number fallback exists because a model that
		// shows intermediates or self-corrects ("200/100 = 2 … wait, the answer is
		// 1") states its final value last, and an intermediate must not rescue a
		// wrong conclusion — but an answer-then-breakdown reply ("He has 16 cows:
		// the 8 that survived plus the 8 he bought.") states its value FIRST, which
		// the earlier tiers catch. Compare by numeric VALUE, not exact string, so
		// different decimals still match — "$50.40" vs "50.4", "114.00" vs "114" —
		// and accept spelled-out numbers ("six" → 6, "twenty-two" → 22) so a worker
		// that writes the word isn't marked wrong.
		digits := benchDigits(a)
		if n, ok := lastSubmatch(benchNumDeclaredRe, digits); ok {
			return numericMatches(n, exp)
		}
		if n, ok := benchLeadNumber(digits); ok {
			return numericMatches(n, exp)
		}
		ns := benchNumberRe.FindAllString(digits, -1)
		return len(ns) > 0 && numericMatches(ns[len(ns)-1], exp)
	case "mcq":
		return matchesMCQ(a, exp)
	case "mcq-repeat":
		// LiveBench's AMC items end with "duplicate that letter five times in a
		// single string" — the answer arrives as "DDDDD". The plain mcq branch
		// cannot read that: benchLetterRe matches STANDALONE letters and "DDDDD"
		// is one token, so every AMC question would grade as a miss. Read the last
		// run of repeated letters first, then fall back to the ordinary rules for
		// a model that ignored the instruction and simply answered "D".
		if pick, ok := lastRepeatedLetter(a); ok {
			return strings.EqualFold(pick, exp)
		}
		return matchesMCQ(a, exp)
	case "exact-list":
		// LiveBench's zebra_puzzle and olympiad answers are ordered comma-separated
		// lists ("1, filmmaking, police-officer, journalist"). Every element must
		// match in order — these are permutation answers, so a partial overlap is
		// simply wrong — but the separator spacing, letter case and any surrounding
		// prose are the model's business, not the grader's. The list is read from
		// the END of the reply: a reasoning model enumerates candidate orderings
		// while it works and only commits to one last, exactly as the mcq and
		// numeric branches already assume.
		return matchesAnswerList(a, exp)
	case "final-contains":
		// For questions that plant the expected token in the prompt itself (the
		// echo traps): substring-anywhere would pass a wrong answer that merely
		// restates the premise ("Mary's father's fifth daughter is Lulu"), so the
		// token only counts where an answer is asserted (see containsFinalAnswer).
		return containsFinalAnswer(a, exp)
	default: // "contains" — case-insensitive substring
		return strings.Contains(strings.ToLower(a), strings.ToLower(exp))
	}
}

// checkAnswerLoose is the half-credit second chance, tried only after checkAnswer
// fails: is the expected answer SEMANTICALLY asserted even though the reply ignored
// the requested format ("Give the number only", "Answer with just the letter")? A
// loose pass scores half a point (benchWeightedScore) and is tallied separately, so a
// verbose worker's knowledge still counts while the formatting cost stays visible and
// still separates it from a compliant worker.
//
// Deliberately conservative: it looks only where a model ASSERTS its result — an
// emphasised span ("**114**", "**Step 6 is wrong.**"), the final stated equation
// ("… = 42"), an emphasised option pick ("**B) backward…**"), or the expected option's
// own text — never anywhere-in-the-text, which would hand half credit to a value
// merely mentioned while reasoning. Only the numeric and mcq modes have a loose
// reading: "contains"/"final-contains"/"exact-list" already accept any format, so
// failing them means the answer is absent, not misformatted.
func checkAnswerLoose(q benchmarkQ, answer string) bool {
	a := normalizeBenchAnswer(answer)
	exp := strings.TrimSpace(q.Expect)
	switch q.Match {
	case "numeric":
		digits := benchDigits(a)
		for _, m := range benchEmphasisRe.FindAllStringSubmatch(digits, -1) {
			for _, n := range benchNumberRe.FindAllString(m[1], -1) {
				if numericMatches(n, exp) {
					return true
				}
			}
		}
		if ms := benchLastEqRe.FindAllStringSubmatch(digits, -1); len(ms) > 0 {
			return numericMatches(ms[len(ms)-1][1], exp)
		}
	case "mcq", "mcq-repeat":
		for _, m := range benchEmphasisRe.FindAllStringSubmatch(a, -1) {
			s := strings.TrimSpace(m[1])
			if s == "" || !strings.EqualFold(s[:1], exp) {
				continue
			}
			// "**B**" or a span opening "B)"/"B."/"B:"/"B," is a pick; "**Braking**"
			// is prose that merely starts with the letter.
			if len(s) == 1 || s[1] == ')' || s[1] == '.' || s[1] == ':' || s[1] == ',' {
				return true
			}
		}
		if opt := benchOptionText(q.Prompt, exp); opt != "" {
			return benchLooseTextMatch(a, opt)
		}
	}
	return false
}

// benchOptionText returns the text of option letter in an MCQ prompt ("backward,
// towards the rear of the car" for "B"), or "" when the option can't be found or is
// too short to be distinctive — a single-word option (usually a bare number) would
// loose-match prose that merely mentions the value.
func benchOptionText(prompt, letter string) string {
	if len(letter) != 1 {
		return ""
	}
	marker := "\n" + strings.ToUpper(letter) + ") "
	i := strings.Index(prompt, marker)
	if i < 0 {
		return ""
	}
	opt := prompt[i+len(marker):]
	if j := strings.IndexByte(opt, '\n'); j >= 0 {
		opt = opt[:j]
	}
	opt = strings.TrimSpace(opt)
	if len(strings.Fields(opt)) < 2 || len(opt) < 8 {
		return ""
	}
	return opt
}

// benchLooseTextMatch reports whether option's words appear in order, near-
// consecutively, in the answer. Per-word comparison is prefix-tolerant (both ≥4
// chars) so "towards" matches "toward"; up to two interposed words are allowed so a
// light interjection ("backward — that is, towards the rear") still reads. Word-level
// rather than substring so punctuation and emphasis never break a match.
func benchLooseTextMatch(answer, option string) bool {
	words := func(s string) []string {
		var b strings.Builder
		for _, r := range strings.ToLower(s) {
			if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
				b.WriteRune(r)
			} else {
				b.WriteByte(' ')
			}
		}
		return strings.Fields(b.String())
	}
	wordEq := func(x, y string) bool {
		if x == y {
			return true
		}
		return len(x) >= 4 && len(y) >= 4 && (strings.HasPrefix(x, y) || strings.HasPrefix(y, x))
	}
	opt, ans := words(option), words(answer)
	if len(opt) == 0 {
		return false
	}
	for start := range ans {
		if !wordEq(ans[start], opt[0]) {
			continue
		}
		i, gaps := 1, 0
		for j := start + 1; j < len(ans) && i < len(opt) && gaps <= 2; j++ {
			if wordEq(ans[j], opt[i]) {
				i++
				gaps = 0
			} else {
				gaps++
			}
		}
		if i == len(opt) {
			return true
		}
	}
	return false
}

// lastSubmatch returns the non-empty capture group of the LAST match of re in s.
// Both declared-answer regexes grade by their final match, so a self-correction's
// closing claim beats anything stated while reasoning.
// matchesMCQ is the letter-answer grader, unchanged in behaviour and split out
// only so mcq-repeat can fall back to it. An explicitly declared pick wins — the
// LAST one, since verbose models restate the options ("A) …") while reasoning
// and only commit at the end — then a letter leading the answer, then the last
// standalone letter (see the benchDeclaredRe var block for why bare lowercase
// a/d never count). An answer that ENUMERATES the options starts with an option
// label rather than its pick, so the leading-letter rule stands aside there.
func matchesMCQ(a, exp string) bool {
	if pick, ok := lastSubmatch(benchDeclaredRe, a); ok {
		return strings.EqualFold(pick, exp)
	}
	if !benchEnumerates(a) {
		if m := benchLeadingRe.FindStringSubmatch(a); m != nil {
			return strings.EqualFold(m[1], exp)
		}
	}
	ms := benchLetterRe.FindAllString(a, -1)
	return len(ms) > 0 && strings.EqualFold(ms[len(ms)-1], exp)
}

// lastRepeatedLetter finds the final standalone run of three or more identical
// option letters — the shape LiveBench's AMC prompts ask for ("FFFFF"). Three
// rather than five so a model that miscounts the repetition still grades on the
// letter it picked.
//
// Written as a scan rather than a regex because Go's RE2 has no backreferences,
// so `([A-J])\1{2,}` doesn't compile. UPPERCASE ONLY, for the same reason
// benchLetterRe won't read a bare lowercase letter as a pick: a lowercase run
// is far likelier to be prose than an answer, and the mcq fallback still catches
// a model that answered in some other form. Last rather than first, because a
// reasoning model echoes the instruction's own example ("if the answer is F,
// then write FFFFF") before committing to its own pick.
func lastRepeatedLetter(s string) (string, bool) {
	found, ok := "", false
	for i := 0; i < len(s); {
		c := s[i]
		if c < 'A' || c > 'J' {
			i++
			continue
		}
		j := i
		for j < len(s) && s[j] == c {
			j++
		}
		// Standalone only: "DDDDD" is a pick, the "DDD" inside a longer token is not.
		boundedLeft := i == 0 || !isBenchWordByte(s[i-1])
		boundedRight := j == len(s) || !isBenchWordByte(s[j])
		if j-i >= 3 && boundedLeft && boundedRight {
			found, ok = string(c), true
		}
		i = j
	}
	return found, ok
}

func isBenchWordByte(b byte) bool {
	return b == '_' || isASCIIDigit(b) || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

// benchTagRe strips XML-ish wrappers. LiveBench's zebra_puzzle prompts ask for
// "<solution>a, b, c</solution>", which would otherwise split into a first
// element of "<solution>a" and a last of "c</solution>" and never match.
var benchTagRe = regexp.MustCompile(`</?[a-zA-Z][^>]*>`)

// matchesAnswerList grades an ordered comma-separated answer. It scans the reply
// line by line from the bottom and passes on the first line whose comma-separated
// elements equal the expected ones — so a worker that shows its working and then
// states the answer passes, while one whose final answer is wrong is not rescued
// by a correct intermediate it wrote earlier. Elements are compared case- and
// space-insensitively; a trailing full stop or a "The answer is:" lead-in on the
// answer line is tolerated, since neither is part of the answer.
func matchesAnswerList(answer, expect string) bool {
	want := splitAnswerList(expect)
	if len(want) == 0 {
		return false
	}
	lines := strings.Split(benchTagRe.ReplaceAllString(answer, " "), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := lines[i]
		// Drop a leading label ("Answer:", "The final answer is:") so the list is
		// compared, not the sentence around it.
		if idx := strings.LastIndex(line, ":"); idx >= 0 && idx < len(line)-1 {
			line = line[idx+1:]
		}
		got := splitAnswerList(line)
		if len(got) != len(want) {
			continue
		}
		same := true
		for j := range got {
			if got[j] != want[j] {
				same = false
				break
			}
		}
		if same {
			return true
		}
	}
	return false
}

// splitAnswerList normalises one comma-separated list into comparable elements.
func splitAnswerList(s string) []string {
	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, ".")
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.ToLower(strings.TrimSpace(p))
		// Strip the markdown a model wraps answers in ("**filmmaking**").
		p = strings.Trim(p, "*_`\"'")
		if p == "" {
			return nil // an empty element means this isn't the answer list
		}
		out = append(out, p)
	}
	return out
}

func lastSubmatch(re *regexp.Regexp, s string) (string, bool) {
	ms := re.FindAllStringSubmatch(s, -1)
	if len(ms) == 0 {
		return "", false
	}
	for _, g := range ms[len(ms)-1][1:] {
		if g != "" {
			return g, true
		}
	}
	return "", false
}

// benchEnumerates reports whether an mcq answer ENUMERATES the options — two or
// more distinct A-J letters each opening the string, a line, or a clause as "X)" /
// "X." ("A) … no. B) … no. C) … yes."). Such an answer leads with an option label,
// not its pick, so checkAnswer skips the leading-letter rule and lets a declared
// pick or the last standalone letter (the concluding option) decide.
func benchEnumerates(a string) bool {
	distinct := map[string]bool{}
	for _, m := range benchEnumRe.FindAllStringSubmatch(a, -1) {
		distinct[strings.ToUpper(m[1])] = true
	}
	return len(distinct) >= 2
}

// numericMatches reports whether candidate n states the expected value — exact
// string or parsed-value comparison, so "4.0" matches "4" and "50.40" matches "50.4".
func numericMatches(n, exp string) bool {
	if n == exp {
		return true
	}
	want, werr := strconv.ParseFloat(exp, 64)
	got, gerr := strconv.ParseFloat(n, 64)
	return werr == nil && gerr == nil && math.Abs(got-want) < 1e-9
}

// benchLeadNumber extracts the single number asserted by the leading clause of an
// answer-then-breakdown reply: "He has 16 cows: the 8 that survived plus the 8 he
// bought." asserts 16, "The ball costs 5 cents and the bat costs 105 cents."
// asserts 5. It declines when the lead holds zero or several numbers (an
// arithmetic chain like "12*8 = 96, …" is working, not an assertion), when the
// lead value is re-used later (then it's an operand mid-computation — "total
// distance 200 m, 200/100 = …" — and the final value must grade), or when the
// answer is a numbered list (a line-leading "1." is a marker, not the answer).
func benchLeadNumber(s string) (string, bool) {
	if len(benchNumListRe.FindAllString(s, -1)) >= 2 {
		return "", false
	}
	cut := len(s)
	for _, loc := range benchLeadBreakRe.FindAllStringIndex(s, -1) {
		// A separator flanked by digits is part of a calculation, not a clause break: a
		// decimal/thousands "." or "," inside a number, or " - " read as subtraction —
		// "97.5 - 90 = 7.5" asserts its result, not the leading operand 97.5.
		if sep := s[loc[0]:loc[1]]; sep == "." || sep == "," || sep == " - " {
			if loc[0] > 0 && isASCIIDigit(s[loc[0]-1]) && loc[1] < len(s) && isASCIIDigit(s[loc[1]]) {
				continue
			}
		}
		cut = loc[0]
		break
	}
	lead := benchNumberRe.FindAllString(s[:cut], -1)
	if len(lead) != 1 {
		return "", false
	}
	for _, later := range benchNumberRe.FindAllString(s[cut:], -1) {
		if numericMatches(later, lead[0]) {
			return "", false // lead value re-used in the working below it
		}
	}
	return lead[0], true
}

func isASCIIDigit(b byte) bool { return b >= '0' && b <= '9' }

// benchClauseRe ends a clause: sentence enders, newlines, commas, semicolons, colons
// and dashes (a hyphen only when it stands alone, so hyphenated words survive).
var benchClauseRe = regexp.MustCompile(`[.!?;:,\n—–]|\s-\s`)

// containsFinalAnswer implements "final-contains": tok must stand as its own word
// where the answer is asserted — in the final clause, or in a terse (≤3-word) leading
// clause followed by a declarative break ("Mary, of course." / "Mount Everest — it
// just hadn't been discovered yet."). These questions plant tok in the prompt, and a
// wrong answer restating the premise carries it only possessive-attached ("Mary's
// father…") or inside a longer subordinate clause ("Before Mount Everest was
// discovered, K2 …") — never as the finally-asserted answer.
func containsFinalAnswer(s, tok string) bool {
	clauses := benchClauseRe.Split(s, -1)
	for i := len(clauses) - 1; i >= 0; i-- {
		c := strings.TrimSpace(clauses[i])
		if c == "" {
			continue // empty tail from trailing punctuation
		}
		if tokenAsAnswer(c, tok) {
			return true
		}
		break // only the final clause states the answer
	}
	// A terse leading clause is a stated answer; a long one is a restated premise.
	// A leading question ("Everest? No — K2.") asserts nothing, and neither does a
	// lead the next clause walks back ("Everest, but actually it was K2.").
	if loc := benchClauseRe.FindStringIndex(s); loc != nil && s[loc[0]:loc[1]] != "?" {
		lead := strings.TrimSpace(s[:loc[0]])
		if len(strings.Fields(lead)) <= 3 && tokenAsAnswer(lead, tok) && !retractsLead(s[loc[1]:]) {
			return true
		}
	}
	return false
}

// benchRetractRe marks a clause that contradicts what came before it.
var benchRetractRe = regexp.MustCompile(`(?i)\b(?:but|actually|however|wait)\b`)

// retractsLead reports whether the clause following a terse lead walks the lead
// back ("Everest, but actually it was K2." / "Everest. No, it was K2.") rather
// than carrying on from it ("Mary, of course."). A bare "no" only counts when it
// is the whole clause, so "Everest, no doubt." isn't read as a retraction.
func retractsLead(rest string) bool {
	for _, c := range benchClauseRe.Split(rest, -1) {
		c = strings.TrimSpace(c)
		if c == "" {
			continue
		}
		return benchRetractRe.MatchString(c) || strings.EqualFold(c, "no")
	}
	return false
}

// tokenAsAnswer reports whether tok (a single word) occurs in s as its own word: not
// run into a longer word, not possessive ("Mary's" restates the prompt subject
// rather than answering), and not negated ("not Mary", "it wasn't Everest" reject
// the token rather than asserting it).
func tokenAsAnswer(s, tok string) bool {
	re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(tok) + `\b`)
	for _, m := range re.FindAllStringIndex(s, -1) {
		if isPossessive(s[m[1]:]) {
			continue
		}
		if negatedNear(strings.Fields(s[:m[0]])) {
			continue
		}
		return true
	}
	return false
}

// isPossessive reports whether the text after a token match starts with a
// possessive "'s" (straight or curly). A bare trailing apostrophe is a closing
// quote ("the answer is 'Mary'."), not a possessive.
func isPossessive(rest string) bool {
	rest = strings.ToLower(rest)
	return strings.HasPrefix(rest, "'s") || strings.HasPrefix(rest, "’s")
}

// negatedNear reports whether either of the last two words before the token is a
// negative form — adjacent ("not Mary") or one word removed ("wasn't actually
// Everest"). The window stays at two words because three reaches back across a
// corrective "but": "not K2 but Everest" asserts the token.
func negatedNear(pre []string) bool {
	for i := len(pre) - 1; i >= 0 && i >= len(pre)-2; i-- {
		if negativeWord(pre[i]) {
			return true
		}
	}
	return false
}

// negativeWord matches the common negative forms: "not", "never", and any
// n't-contraction ("wasn't", "isn't", "doesn't"), straight or curly apostrophe.
func negativeWord(w string) bool {
	w = strings.ToLower(strings.Trim(w, `"'“”‘’()[]*_`))
	return w == "not" || w == "never" ||
		strings.HasSuffix(w, "n't") || strings.HasSuffix(w, "n’t")
}

// auditThinkingGate runs the router's own reasoning gate over the benchmark
// prompts and logs where its learned enable-thinking decision disagrees with the
// tier's expectation (tiers >= benchHardTier "should think" in production — the same
// boundary the benchmark now grades each question by). The benchmark set is
// effectively a labelled difficulty corpus, so a disagreement flags the gate's
// threshold/seeds for tuning — it does NOT feed any quality score. The gate is
// model-independent, so this logs once per router lifetime; it's best-effort, and
// if the embeddings worker is down it stays unaudited so a later profile retries.
func (r *Router) auditThinkingGate() {
	if r.classifier == nil {
		return
	}
	if !r.gateAudited.CompareAndSwap(false, true) {
		return // another profile already audited (or is auditing) the gate
	}
	var mismatches []string
	classified := 0
	for _, q := range benchmarkQuestions {
		cl, ok := r.classifier.classify(&ChatRequest{
			Messages: []Message{{Role: "user", Content: q.Prompt}},
		})
		if !ok {
			continue
		}
		classified++
		gate := r.classifier.wantThinking(cl.reasoning)
		expect := q.Tier >= benchHardTier
		if gate != expect {
			mismatches = append(mismatches, fmt.Sprintf("t%d r=%.2f gate=%s tier-expects=%s — %s",
				q.Tier, cl.reasoning, thinkLabel(gate), thinkLabel(expect), benchSnippet(q.Prompt)))
		}
	}
	if classified == 0 {
		r.gateAudited.Store(false) // embeddings unavailable — let a later profile retry
		return
	}
	if len(mismatches) == 0 {
		log.Printf("thinking-gate audit: all %d benchmark prompts agree with tier expectation", classified)
		return
	}
	log.Printf("thinking-gate audit: %d/%d prompts where the reasoning gate disagrees with tier (tune ReasoningThreshold/seeds):", len(mismatches), classified)
	for _, m := range mismatches {
		log.Printf("  gate-mismatch %s", m)
	}
}

// thinkLabel renders an enable-thinking decision for logs.
func thinkLabel(think bool) string {
	if think {
		return "think"
	}
	return "no_think"
}
