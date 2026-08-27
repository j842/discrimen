package router

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"net"
	"net/http"
	"regexp"
	"sort"
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
// v33: the frontier tiers got a longer answer deadline, and tier 11 was added.
// v38: that split is gone — every question now gets the long deadline, because the short
// one was measured censoring 75% of the strongest home worker's misses. See
// benchAnswerDeadline below.
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
// v43: eight grading and profiling corrections, all reproduced against the real
// graders. Every one of them moved scores, and — as always here — a mis-graded
// question is indistinguishable from a wrong answer, so none of them showed up
// as anything but a plausible number:
//
//	the numeric grader broke on the word "and". "and" was a clause break, so
//	"The sum of 8 and 5 is 13." (the TIER-1 CONTROL question) had a leading
//	clause of "The sum of 8", whose single number the grader then COMMITTED to.
//	13 graded wrong; so did "There are 7 sons and 1 sister, so 8." and "20 and 5
//	more makes 25.". 137 of 392 questions are numeric, and any model that
//	explains its working was understated. See benchLeadBreakRe.
//
//	the mcq last-standalone-letter fallback passed a WRONG pick: "The answer
//	must be option D, since vitamin C is irrelevant here" graded as C. Tiers
//	9-10 carry TEN options and invite exactly that shape of reply, so this let a
//	weak model draw hard traffic. See benchBareLetterPick.
//
//	"contains"/"final-contains" read only the FINAL clause, so a correct answer
//	that was then EXPLAINED failed: "The answer is Paris. It is located on the
//	Seine." graded wrong. Those 16 questions are the sanity band — both tier-1
//	controls, three of the four tier-2 floor questions and all three tier-3 echo
//	traps. See containsFinalAnswer.
//
//	the no-think pass counted ERRORED questions in its denominator, which v41
//	fixed in the mixed pass and not in its twin. Measured: same worker, same 10
//	tier-5 questions, 5 errored — mixed 100, no-think 50, fabricating a
//	thinking-vs-no-think collapse that never happened.
//
//	the no-think give-up guard divided by the questions the pass INTENDED to
//	ask rather than the ones it dispatched, so a budget-stopped pass that
//	errored on all 48 of ~300 returned a score near 0 with ok=true — and ok=true
//	is what persists a profile.
//
//	the carried-over easy tiers dropped their latency and their text, and
//	profile.go appends those rows AFTER the mixed rows with the same matrix key,
//	so the zeroed row superseded the good one and destroyed the latency
//	prediction for easy no-think traffic.
//
//	the busy-retry loop had no per-question cap: `continue` skipped the attempt
//	check and the only exit was a SHARED streak that any other question's
//	success reset. A rate-limited endpoint could hang wg.Wait() forever, and
//	with it profileBackend and the caller's `defer r.profiling.Delete(id)` — so
//	the worker showed "profiling" for the life of the process and could never be
//	re-certified. See benchBusyMaxWait.
//
//	benchRequestFor appended " Give the number only." to 21 LiveBench spatial
//	items whose own text already says "put your answer in **bold** as a single
//	integer". The model was told to do two contradictory things, so whatever it
//	did violated one of them — a confound on the numeric grader, and the only
//	place instruction-following enters the score at all (the bank has no
//	instruction-following category). See benchStatesAnswerFormat.
//
// v42: the generated question set is re-tiered to match the emit rules — the 40
// unpassable headroom items move off tier 12 (benchCodingTier), which they were
// filling with maths, so the 20-point coding bucket is now 28 coding questions
// rather than 28 real ones diluted by 40 impossible non-code ones. Max
// achievable quality goes from ~88 back to 100. Also: the no-think pass now asks
// only the questions the mixed pass actually asked, so the two scores are
// computed over the same exam before being compared on one absolute scale.
// v41: three grading and profiling corrections, each of which changed scores.
// An ERRORED question no longer counts in the denominator — it used to be a zero
// over a one, arithmetically identical to a wrong answer, so an unreachable
// sandbox scored a perfect worker at 75 with ok=true. The thinking gate is now
// written in the dialect each endpoint was MEASURED to honour rather than a
// hardcoded chat_template_kwargs, which on every relay row and strict provider
// meant neither pass switched anything and both scores were the same mode
// measured twice. And the mixed pass records each result in the mode it was
// actually asked in, instead of publishing its thinking-off easy tiers as
// thinking evidence.
// v40: the token ceiling stops being a binding constraint. Routing cares whether
// a worker answers correctly in reasonable TIME, and time is already bounded by
// benchAnswerDeadline; a second ceiling in tokens measured nothing on top of it.
// At 16384 the two were nearly the same constraint — that is ~6 minutes at ~45
// tok/s, where most of this fleet sits — so the cap did no independent work
// while still failing questions on the workers fast enough to reach it
// (llm-6000pro recorded 29 truncations). Now 32768, clamped per worker to what
// its context window can actually hold.
// v39: GRADING correctness. Four confirmed defects, each of which changed scores
// silently — a mis-graded question is indistinguishable from a wrong answer:
//
//	exact-list (31% of the set) failed a CORRECT answer whose lead-in contained a
//	comma, because the element count was gated before anything was compared.
//	"So," and "Therefore," are among the most common things a model writes.
//
//	exact-list with a single-element answer PASSED a wrong one: almost any prose
//	line parses as a one-element list, so the scan walked up into abandoned
//	working and graded a value the model had explicitly retracted.
//
//	"contains" was a bare substring, so a reply that mentioned the expected token
//	while ruling it out ("Thursday, Friday, Saturday. So it is Saturday.") passed.
//
//	loose half-credit rewarded FORMATTING: bolding the option labels matched the
//	expected letter whichever option was chosen, and bolding step headers matched
//	on the step number. It was gameable outright — "**40**, **41**, **42**, **43**
//	or **44**. I'll say 44." collected half credit for an expected 42.
//
// Also v39: the token budget is a property of the QUESTION rather than of the
// thinking mode, so Quality and QualityNoThink differ only in the thing they are
// named for (they were 16384 and 1024 on the same hard questions); calibration
// and production now build the request through one shared function; and the
// headroom bands move off tier 12 (benchCodingTier), which they were filling
// with unpassable maths.
// v38: the per-question answer deadline goes 2m -> 6m and the per-tier split is removed.
// Every cached profile is invalidated because a profile is only comparable to another
// measured under the same clock: at 2 minutes, 75% of llm-6000pro's recorded misses were
// answers it was cut off part-way through, and its score was that much a measure of
// throughput. Also: coding_completion answers are now graded as prefix+answer (they are
// FRAGMENTS continuing a partial solution shown in the prompt), which had scored the two
// strongest workers 0% on that task while the weakest scored highest. See
// benchAnswerDeadline and poolCode.Prefix.
const benchmarkVersion = 43

// benchmarkQ is one graded question in the cold-start quality benchmark. The
// question set lives in benchmark_data.go.
type benchmarkQ struct {
	Tier   int    // difficulty band 1 (control) … 11 (budget-bounded insight) … 12 (programming); also sets grading mode — tiers >= benchHardTier are graded thinking-on (see benchmark_data.go)
	Prompt string // user prompt sent to the worker
	Expect string // expected answer token (see Match)
	Match  string // "contains" | "numeric" | "mcq" | "final-contains" | "code-exec" (see checkAnswer)
	// Code is set only when Match == benchMatchCodeExec. Such a question has no
	// Expect: its ground truth is its test cases, and grading means running the
	// answer in the sandbox sidecar (codeexec.go) rather than comparing strings.
	Code *benchCode
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
	// LatencyMS is how long this answer took, for the outcome matrix.
	LatencyMS int64 `json:"latency_ms,omitempty"`
	// Skipped means the profile budget stopped before this question was asked.
	// Deliberately phrased as "skipped" rather than "ran": false is the zero
	// value, so a result recorded before this field existed reads as HAVING run,
	// which is what those profiles did.
	Skipped bool `json:"skipped,omitempty"`
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
	// benchFinalLetterRe matches a standalone option letter that ENDS the reply —
	// only trailing punctuation, emphasis or whitespace may follow it ("…so C.",
	// "…→ H", "**B**"). A pick a model writes last is a pick; a letter it wrote
	// while ruling an option out is not. Trailing '*' is allowed so an emphasised
	// answer reads, since emphasis is where a model that ignored "just the
	// letter" puts its choice (see benchEmphasisRe).
	benchFinalLetterRe = regexp.MustCompile(`\b([A-HJbc])[\s.,;:!?)\]*_"'’—–-]*$`)
	// benchLetterOrRe matches an undecided disjunction of two option letters ("a
	// or b", "C or D"). A model offering two letters has picked neither, and the
	// bare last-letter rule read the one after "or" as its answer.
	benchLetterOrRe = regexp.MustCompile(`(?i)\b([a-j])\s+or\s+\(?([a-j])\b`)
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
	// punctuation or parens. A '.'/',' between digits is part of a number and is
	// skipped by benchLeadNumber.
	//
	// v43: `\band\b` was in this set and is not any more. A conjunction joins two
	// coordinate clauses; it does not end an assertion, and treating it as one
	// made benchLeadNumber COMMIT to the first number of any reply that used the
	// word. The TIER-1 CONTROL question was graded wrong by it — "The sum of 8
	// and 5 is 13." led with "The sum of 8" — as were "There are 7 sons and 1
	// sister, so 8." and "20 and 5 more makes 25.". 137 of 392 questions are
	// numeric, so this understated every model that explains its working.
	// Coordinate replies are read by benchCoordinateLeadNumber instead, which
	// offers a candidate rather than a commitment.
	benchLeadBreakRe = regexp.MustCompile(`[.!?;:,\n—–()]|\s-\s`)
	// benchCoordAndRe finds the conjunction that joins two coordinate clauses.
	benchCoordAndRe = regexp.MustCompile(`\band\b`)
	// benchAssertVerbRe marks a clause that STATES a value ("the ball costs 5
	// cents", "there are 7 sons") rather than naming one in passing ("the sum of
	// 8", "20"). Without it the first coordinate of "The sum of 8 and 5 is 13."
	// would be offered as an answer, and an operand is not an answer.
	benchAssertVerbRe = regexp.MustCompile(`(?i)\b(?:is|are|was|were|has|have|had|costs?|makes?|gives?|leaves?|equals?|remains?|weighs?|holds?|contains?)\b|[=:]`)
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
	benchMaxTokens = 1024 // token ceiling for an easy-tier answer (short trap/recall replies)
	// benchThinkMaxTokens is the ceiling for a HARD question, and it is set to a
	// value the answer deadline reaches first on most of this fleet — i.e. it is
	// deliberately not the binding constraint.
	//
	// v40: 16384 -> 32768. A token cap measures nothing routing cares about; the
	// product question is "will this worker answer correctly in reasonable TIME",
	// and time is already bounded by benchAnswerDeadline. Two ceilings meant a
	// question could fail for hitting either, and at 16384 the two were nearly
	// the same constraint — 16384 tokens is ~6 minutes at ~45 tok/s, which is
	// where most of the fleet sits, so the cap was doing no independent work
	// while still producing truncation failures on the workers fast enough to
	// reach it (llm-6000pro recorded 29). Raised until the deadline binds for
	// anything slower than ~91 tok/s, which is every local worker.
	//
	// A submission still generating at 32768 tokens is looping rather than
	// reasoning, so truncation there remains a genuine failure.
	benchThinkMaxTokens = 32768

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

	// A BUSY worker is not a wrong answer and not an outage — it is a worker with
	// no free slot right now, which is the normal state of a fleet being profiled
	// while it also serves production. Failing the question would grade the queue
	// rather than the model, so a busy response waits benchBusyWait and tries
	// again.
	//
	// benchBusyMaxStreak is the circuit breaker on that patience: a worker that
	// has been busy this many times CONSECUTIVELY (across the whole run, reset by
	// any successful answer) is not congested, it is gone — a relay row whose
	// upstream dropped it, or an endpoint that 404s everything. Profiling is
	// abandoned rather than spinning forever, and the run is reported incomplete
	// so it can be resumed deliberately instead of silently scoring a zero.
	benchBusyWait      = time.Second
	benchBusyMaxStreak = 100

	// benchAnswerDeadline is the hard per-question bound. A worker that needs
	// longer is scored a FAIL for that question and is NOT retried — a retry
	// would only burn another deadline for the same verdict. This is a benchmark
	// criterion, independent of the live-proxy BACKEND_TIMEOUT_SECONDS:
	// benchCompletion uses a no-timeout client so this deadline alone decides
	// whether the answer arrived in time.
	//
	// v38: 2m -> 6m, and the per-tier split is gone. v33 had already given tiers
	// >= 9 six minutes after the 2-minute bound scored a 284B MoE below a
	// saturated 27B on nothing but speed-fails. That reasoning was right and the
	// scope was too narrow — MEASURED on the LiveBench pool, the 2-minute bound
	// was censoring:
	//
	//	work:llm-5090-ornith-35b-a3b  244 tok/s    0% of its failures
	//	llm-6000pro-qwen38-27b         44 tok/s   75% (all 12 sampled finished
	//	                                               inside 5 min, median 192s)
	//	llm-naples-ornith15-397B       12 tok/s   88%
	//
	// Three quarters of the strongest home worker's recorded misses were answers
	// it was cut off mid-way through, not answers it got wrong. A bound that
	// falls hardest on the slowest hardware measures tokens per second, and the
	// pool it grades — LiveBench maths, olympiad and code — needs a long
	// scratchpad by construction.
	//
	// Slowness is NOT thereby unpriced: the router ranks candidates by expected
	// completion time per request, so a slow-but-right worker loses the race for
	// interactive prompts without its QUALITY being falsified first. That is the
	// right split of concerns — speed belongs in the ranking, capability in the
	// score — and it is what lets a slow worker serve as overflow instead of
	// being scored into uselessness.
	//
	// A worker too slow to finish inside SIX minutes is still scored a fail, and
	// llm-cpu-gemma-26B-silver (6.9 min on a math_comp question that finished,
	// three of four unfinished at eight) is one. That is a true statement about
	// what it can serve. What it must not do is vote on how hard the QUESTION is
	// — see benchIsObserver.
	benchAnswerDeadline = 6 * time.Minute

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

// benchBusyMaxWait bounds how long ONE question may spend being told the worker
// is busy, and benchBusyMaxRetries bounds how many times it may be told.
//
// The shared streak (benchBusyMaxStreak) is NOT enough on its own, and believing
// it was is how a profile could hang forever. busy.ok() clears the streak
// whenever ANY OTHER concurrent question succeeds, so on a rate-limited endpoint
// that answers some requests and 429s the rest, the streak never reaches its
// ceiling while the unlucky questions retry at 1/sec indefinitely. The busy path
// `continue`s, so it never reaches the attempt cap either, and the profile budget
// gates DISPATCH only — wg.Wait() is unbounded. runQualityBenchmark then never
// returns, profileBackend never returns, and the caller's
// `defer r.profiling.Delete(id)` never runs: that worker shows "profiling"
// forever and can never be re-certified or re-profiled for the life of the
// process. A per-question budget is what makes the wait terminate.
//
// Vars rather than consts so the test that proves termination can shrink them;
// nothing in the router writes them.
var (
	benchBusyMaxWait    = 10 * time.Minute
	benchBusyMaxRetries = 600
)

// benchOutcome is the graded result of one benchmark question against one worker.
type benchOutcome struct {
	// latencyMS is how long the answer took. Recorded per question because the
	// outcome matrix predicts "how long does this KIND of prompt take this
	// worker", which a fleet-wide throughput average cannot answer.
	latencyMS int64
	// skipped means the question was never asked, because the profile budget ran
	// out before it was dispatched. It is NOT a miss and must not be scored as
	// one; it exists so the results slice can stay index-aligned with
	// benchmarkQuestions, which runNoThinkQualityBenchmark consumes positionally.
	skipped bool
	tier    int
	pass    bool
	errd    bool
	slow    bool
	trunc   bool
	loose   bool
	got     string
}

// benchOne asks the worker one benchmark question in the given thinking mode and
// grades the answer. Extracted from runQualityBenchmark's loop so the no-think
// scoring pass (runNoThinkQualityBenchmark) asks questions EXACTLY the way the
// main benchmark does — a second copy of the request/grading logic would drift.
func (r *Router) benchOne(b *Backend, q benchmarkQ, think bool, busy *benchBusyTracker) benchOutcome {
	res := benchOutcome{tier: q.Tier}
	prompt, maxTokens := benchRequestFor(q, think, usableContextTokens(b))
	// A prompt that does not fit the worker's window is a MISS, not an outage.
	//
	// The request would otherwise go out anyway, the worker would reject it for
	// length, and the generic retry path would record res.errd — which leaves the
	// question out of the denominator entirely. A 32K worker asked a 48K question
	// would come back neither right nor wrong but UNMEASURED, reported exactly
	// like a worker the profile budget never got to.
	//
	// That is backwards for the one thing these questions exist to measure. "This
	// model cannot reason over 48K of context" is a real, useful weakness and the
	// reason the long-context set was added; recording it as "we could not tell"
	// hides precisely the finding. Judged before dispatch because the verdict does
	// not need the worker's opinion — it follows from the advertised window — and
	// asking anyway would spend a real generation to learn something already known.
	//
	// Only when the window is actually KNOWN. usableContextTokens returns 0 for a
	// worker still being profiled, and guessing a miss from a missing number would
	// fail questions for a worker that can answer them.
	if ctx := usableContextTokens(b); ctx > 0 && benchPromptTokenEstimate(prompt)+benchMinAnswerTokens > ctx {
		res.got = fmt.Sprintf("(prompt needs ~%d tokens, worker window is %d)",
			benchPromptTokenEstimate(prompt)+benchMinAnswerTokens, ctx)
		return res // pass=false, errd=false: counted, and counted as a miss
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
		"model":       probeModel(b),
		"stream":      false,
		"max_tokens":  maxTokens,
		"temperature": 0,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
	}
	// The thinking gate in the spelling this endpoint was MEASURED to honour, not
	// a hardcoded one. Production does exactly this (patchForwardedBody), and the
	// benchmark did not: it always wrote chat_template_kwargs, which is a vLLM
	// and llama.cpp extension. Every relay row speaks reasoning_effort, as does
	// any strict provider — so on those workers NEITHER pass switched anything,
	// both ran in whatever mode the template defaults to, and Quality and
	// QualityNoThink came back as the same number measured twice in one mode.
	// That is the two-score design failing at its root, and it reproduces the
	// exact regression the design exists to prevent.
	applyBenchThinking(payload, b, think)
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
	abandoned := false
	busyRetries, busyStarved := 0, false
	// Timed across the whole call including retries, because that is the wall
	// clock a caller would have experienced. A busy-retry is time the request
	// really spent waiting.
	started := time.Now()
	for attempt := 1; ; attempt++ {
		if busy != nil && busy.abandoned() {
			abandoned = true
			break
		}
		ctx, cancel := context.WithTimeout(context.Background(), deadline)
		raw, err = r.benchCompletion(ctx, b, payload)
		cancel()
		if err == nil {
			if busy != nil {
				busy.ok()
			}
			break
		}
		if isTimeout(err) {
			slow = true
			break // too slow → a usability fail for this question, never retried
		}
		// A busy worker is a queue, not an answer: wait a beat and ask again,
		// rather than grading the congestion. Two things bound the patience — the
		// shared streak breaker (this worker is gone) and this question's OWN
		// budget (this question is starving while others get through). The second
		// is what keeps the run finite on a rate-limited endpoint, where the first
		// never fires because every success resets it.
		if benchBusyStatus(err) {
			if busy != nil && busy.busy() {
				abandoned = true
				break
			}
			busyRetries++
			wait := benchBusyWait
			if left := benchBusyMaxWait - time.Since(started); left < wait {
				wait = left
			}
			if busyRetries >= benchBusyMaxRetries || wait <= 0 {
				abandoned, busyStarved = true, true
				break
			}
			time.Sleep(wait)
			continue
		}
		if attempt >= benchMaxAttempts {
			break
		}
		time.Sleep(benchRetryBackoff * time.Duration(attempt))
	}
	res.latencyMS = time.Since(started).Milliseconds()
	switch {
	case abandoned:
		// Not a wrong answer and not this question's fault: the worker had no slot
		// for it, either for the whole run (the shared streak) or for this
		// question's whole budget (benchBusyMaxWait). Marked ERRORED, not failed,
		// so it stays out of the denominator and the score is reported incomplete
		// rather than as a zero the worker did not earn.
		res.errd = true
		res.got = "(abandoned — worker busy)"
		if busyStarved {
			res.got = fmt.Sprintf("(abandoned — busy for %d attempts over %s)",
				busyRetries, time.Since(started).Round(time.Second))
		}
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
		if q.Match == benchMatchCodeExec {
			// Graded by RUNNING it. A sandbox that cannot be reached leaves the
			// question errored rather than failed — see gradeCode.
			ctx, cancel := context.WithTimeout(context.Background(), codeExecHTTPGrace)
			pass, err := r.gradeCode(ctx, content, q)
			cancel()
			switch {
			case err != nil:
				res.errd = true
				res.got = "(sandbox: " + err.Error() + ")"
			case pass:
				res.pass = true
			}
			return res
		}
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
	busy := &benchBusyTracker{}
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup
	// Bounded, and dispatched tier-first-then-round-robin so that stopping early
	// leaves a balanced sample rather than every easy question and no hard one.
	//
	// The bound exists because the deadline and the question count multiply: at
	// six minutes a question, a worker with one slot and 277 questions is a
	// 27-hour profile, and benchmarkVersion bumps re-profile the whole fleet at
	// once. That worker would be saturated for a day while still serving traffic.
	ran := make([]bool, len(benchmarkQuestions))
	started := time.Now()
	dispatched := 0
	for _, i := range benchStratifiedOrder(benchmarkQuestions) {
		// The floor wins over the budget: a score from a handful of questions is
		// worse than a slow profile, so the budget can only stop dispatch once
		// enough questions are already in flight to score on.
		if dispatched >= benchMinProfileQuestions && time.Since(started) > benchProfileBudget {
			break
		}
		ran[i] = true
		dispatched++
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()
			prog := r.progressFor(b.ID)
			prog.begin()
			defer func() { prog.end(); prog.step() }()
			q := benchmarkQuestions[i]
			results[i] = r.benchOne(b, q, q.Tier >= benchHardTier, busy)
		}(i)
	}
	wg.Wait()
	if dispatched < len(benchmarkQuestions) {
		log.Printf("quality benchmark %s: stopped after %d/%d questions (%s budget) — scoring the sample",
			b.ID, dispatched, len(benchmarkQuestions), benchProfileBudget)
	}

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
		// A question that was never dispatched is not a miss — excluded from every
		// tally, but still emitted so the slice stays index-aligned with
		// benchmarkQuestions. runNoThinkQualityBenchmark indexes into it and
		// refuses to run at all if the lengths disagree, so dropping the entry
		// would silently disable no-think scoring on any truncated profile.
		if !ran[i] {
			details = append(details, BenchResult{
				Tier: q.Tier, Prompt: q.Prompt, Expect: q.Expect, Skipped: true,
			})
			continue
		}
		// An errored question stays OUT of the denominator. It is one we could not
		// GRADE — sandbox unreachable, transport failed, worker abandoned — and
		// counting it left a zero in the numerator and a one in the denominator,
		// which is arithmetically identical to a wrong answer. codeexec.go
		// promises the opposite in prose ("an outage is not a wrong answer") and
		// the promise was false the moment the result reached this tally.
		// Measured: sandbox down, worker answering everything else correctly,
		// score=75 with ok=true — and ok=true is what persists the profile and
		// routes on it. Contrast res.slow, which IS counted, deliberately: too
		// slow to be usable is a real verdict about the worker.
		if !res.errd {
			count[res.tier]++
		}
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
			Loose: res.loose, LatencyMS: res.latencyMS,
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
	// Judged over the questions this pass actually ASKED. Against the full set the
	// guard could never trip on a truncated run — a budget-limited pass that asked
	// 48 of 392 and errored on all 48 scored 0 with ok=true, and the caller
	// persisted it. Slow workers are both the ones that truncate and the ones this
	// guard most protects.
	if dispatched > 0 && errored*2 > dispatched {
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
// The fourth return is the per-question record for THIS pass. Without it the
// no-think outcomes were tallied into per-tier counters and then discarded, so
// the stored profile could say a worker scored 41 with thinking off but not
// which questions it lost — and a tier does not map to one category, so the
// category breakdown could only ever show the thinking-on half. Building it
// costs one append per question in a loop that already visits every outcome.
func (r *Router) runNoThinkQualityBenchmark(b *Backend, concurrency int, mixed []BenchResult) (score int, ok bool, breakdown string, details []BenchResult) {
	if len(benchmarkQuestions) == 0 || len(mixed) != len(benchmarkQuestions) {
		return 0, false, "", nil
	}
	if concurrency < 1 {
		concurrency = 1
	}
	// Only the hard questions the MIXED pass actually asked. The two scores are
	// compared against each other and against the whole fleet on one absolute
	// 0-100 scale, so they have to be computed over the same exam — and they were
	// not: this pass gets a fresh budget, and no-think answers are short, so it
	// routinely completed while the mixed pass truncated. Measured on one worker:
	// 137 questions in the mixed pass against 380 here, meaning 255 hard questions
	// were graded no-think that Quality never saw.
	//
	// A question the mixed pass skipped is therefore skipped here too, which also
	// keeps the pair meaningful per question: the matrix records both modes for
	// the same item or neither.
	var hardIdx []int
	for i, q := range benchmarkQuestions {
		if q.Tier >= benchHardTier && !mixed[i].Skipped {
			hardIdx = append(hardIdx, i)
		}
	}
	r.progressFor(b.ID).enter(phaseQualityNT, len(hardIdx))
	rerun := make([]benchOutcome, len(hardIdx))
	busy := &benchBusyTracker{}
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup
	// Bounded on the same terms as the mixed pass — this is a second full pass
	// over the hard tiers, so leaving it unbounded would undo the bound there.
	started := time.Now()
	dispatched := 0
	for _, j := range benchStratifiedOrderOf(hardIdx) {
		if dispatched >= benchMinProfileQuestions && time.Since(started) > benchProfileBudget {
			rerun[j].skipped = true
			continue
		}
		dispatched++
		i := hardIdx[j]
		wg.Add(1)
		sem <- struct{}{}
		go func(j, i int) {
			defer wg.Done()
			defer func() { <-sem }()
			prog := r.progressFor(b.ID)
			prog.begin()
			defer func() { prog.end(); prog.step() }()
			rerun[j] = r.benchOne(b, benchmarkQuestions[i], false, busy)
		}(j, i)
	}
	wg.Wait()
	if dispatched < len(hardIdx) {
		log.Printf("no-think benchmark %s: stopped after %d/%d questions (%s budget)",
			b.ID, dispatched, len(hardIdx), benchProfileBudget)
	}

	maxTier := 1
	for _, q := range benchmarkQuestions {
		if q.Tier > maxTier {
			maxTier = q.Tier
		}
	}
	pass := make([]int, maxTier+1)
	loose := make([]int, maxTier+1)
	count := make([]int, maxTier+1)
	// asked is the give-up guard's denominator: every question this pass has an
	// answer (or an error) for. Counted here rather than taken from `dispatched`
	// so it covers the same population as `errored` — see the guard below.
	errored, slowFailed, truncated, asked := 0, 0, 0, 0
	outcomes := make([]benchOutcome, len(benchmarkQuestions))
	for i, m := range mixed {
		// Easy tiers ran thinking-off in the mixed pass; carry them over verbatim,
		// INCLUDING whether they were asked at all, how long they took and what
		// the worker actually said.
		//
		// latencyMS and got used to be dropped here, and "verbatim" was a lie about
		// six fields out of eight. The detail row below then wrote LatencyMS: 0,
		// and profile.go appends these rows AFTER the mixed rows into the outcome
		// matrix, where record() supersedes on (QID, Backend, Thinking, Source).
		// An easy tier is Thinking=false in BOTH passes, so the zeroed row
		// overwrote the measured one and the matrix's latency prediction for easy
		// no-think traffic — the traffic these rows exist to predict — was
		// destroyed. Dropping `got` cost the stored answer for the same rows.
		outcomes[i] = benchOutcome{tier: m.Tier, pass: m.Pass, loose: m.Loose,
			errd: m.Errored, slow: m.Slow, trunc: m.Truncated, skipped: m.Skipped,
			latencyMS: m.LatencyMS, got: m.Got}
	}
	for j, i := range hardIdx {
		outcomes[i] = rerun[j]
	}
	for i, res := range outcomes {
		q := benchmarkQuestions[i]
		// Same rule as the mixed pass: never asked is not wrong.
		if res.skipped {
			details = append(details, BenchResult{
				Tier: q.Tier, Prompt: q.Prompt, Expect: q.Expect, Skipped: true,
			})
			continue
		}
		asked++
		// Same rule as the mixed pass, which is the only reason to state it twice:
		// an ERRORED question stays out of the denominator. It is one we could not
		// GRADE, and counting it leaves a zero over a one — arithmetically
		// identical to a wrong answer. v41 fixed this in runQualityBenchmark and
		// not in its twin here, and the two scores are compared against each other
		// on one absolute scale, so the asymmetry manufactured exactly the signal
		// the two-score design exists to detect: measured on one worker, the same
		// 10 tier-5 questions with 5 errored scored mixed=100 and no-think=50 — a
		// thinking-vs-no-think collapse that never happened.
		if !res.errd {
			count[res.tier]++
		}
		details = append(details, BenchResult{
			Tier: q.Tier, Prompt: q.Prompt, Expect: q.Expect,
			Got: res.got, Pass: res.pass, Truncated: res.trunc, Errored: res.errd, Slow: res.slow,
			Loose: res.loose, LatencyMS: res.latencyMS,
		})
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
	// actually ASKED: a worker that went unreachable mid-run is unmeasurable, not
	// bad.
	//
	// It used to divide by len(hardIdx) — the questions the pass INTENDED to ask —
	// while the comment claimed otherwise, and against that denominator the guard
	// could not trip on a truncated run: a budget-stopped pass that dispatched 48
	// of ~300 hard questions and errored on all 48 asks whether 96 > 300, decides
	// the worker is fine, and returns a score near 0 with ok=true. ok=true is what
	// persists the profile and routes on it. The mixed pass has always divided by
	// what it dispatched; this is the same arithmetic, counted over the same
	// population `errored` is (the hard questions dispatched here PLUS the easy
	// tiers carried over from the mixed pass, since both are in the merged score).
	if asked > 0 && errored*2 > asked {
		return 0, false, "", nil
	}
	ok2, score2 := benchWeightedScore(pass, loose, count, maxTier)
	if !ok2 {
		return 0, false, "", nil
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
	return score2, true, strings.TrimSpace(bd.String()), details
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
	// Magnitudes multiply; they are not values. Mapping bare "hundred" to 100
	// turned "two hundred" into "2 100", whose LAST number is 100 — so a reply of
	// two hundred graded correct against an expected 100. Resolved before the
	// word map runs, and after the compound pass so "twenty-two hundred" arrives
	// here as "22 hundred".
	digits = benchMagnitudeRe.ReplaceAllStringFunc(digits, benchMagnitude)
	digits = numberWordRe.ReplaceAllStringFunc(digits, func(w string) string { return numberWords[w] })
	digits = benchThousandsRe.ReplaceAllStringFunc(digits, func(w string) string { return strings.ReplaceAll(w, ",", "") })
	return digits
}

// checkAnswer reports whether answer satisfies q's expected result.
func checkAnswer(q benchmarkQ, answer string) bool {
	a := normalizeBenchAnswer(answer)
	exp := strings.TrimSpace(q.Expect)
	// An empty expectation cannot be satisfied by anything, and the check belongs
	// HERE rather than in each branch: the modes that fell through to the
	// substring default — "contains", "final-contains", "code-exec" and any
	// unrecognised string — all returned true for arbitrary text, so one
	// malformed question graded every answer correct and lifted the entire
	// fleet's score. A per-mode guard is exactly what left those four behind.
	if exp == "" {
		return false
	}
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
		// A conjunction joins two coordinate assertions, and the reply commits to
		// neither: "The ball costs 5 cents and the bat costs 105 cents." states
		// both, and so does the same sentence with its clauses swapped. So the
		// first coordinate is tried and, when it misses, the last number still
		// decides — where the lead rule above SHORT-CIRCUITS, which is what
		// graded the tier-1 control ("The sum of 8 and 5 is 13.") wrong.
		if n, ok := benchCoordinateLeadNumber(digits); ok && numericMatches(n, exp) {
			return true
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
		// Every one of these prompts carries "if the answer is F, write FFFFF" as
		// its worked example, and F is never a valid AMC option. A model that
		// quotes the instruction after answering ("CCCCC — had it been F I'd have
		// written FFFFF") therefore ends with a repeated run that is the PROMPT's
		// letter, not its own, and reading the last run blindly graded it on F.
		runs := repeatedLetters(a)
		if ex, ok := lastRepeatedLetter(q.Prompt); ok && !strings.EqualFold(ex, exp) {
			kept := runs[:0]
			for _, r := range runs {
				if !strings.EqualFold(r, ex) {
					kept = append(kept, r)
				}
			}
			runs = kept
		}
		if len(runs) > 0 {
			return strings.EqualFold(runs[len(runs)-1], exp)
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
	default:
		// "contains" — the expected token, where the reply ASSERTS it. Formerly a
		// bare substring, which passed a reply that mentioned the token while
		// ruling it out ("Thursday, Friday, Saturday. So it is Saturday." for
		// Friday) or explicitly negated it ("definitely not UnboundLocalError").
		// Both are wrong answers reading as right, which is the direction that
		// inflates a score. This is the same rule "final-contains" already used;
		// the two modes now differ only in name, kept because the question data
		// distinguishes them.
		return containsFinalAnswer(a, exp)
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
// reading: "contains"/"final-contains"/"exact-list" accept any format, so failing
// them means the answer is absent, not misformatted.
//
// v43: that last sentence used to be false and is now true. "contains" and
// "final-contains" read only the reply's FINAL CLAUSE, so an answer that was
// stated and then explained ("The answer is Paris. It is located on the Seine.")
// failed strict grading with nothing to catch it here — a formatting failure
// dressed as ignorance, across the questions that are supposed to be the floor.
// The fix belongs in strict grading rather than in half credit, because a right
// answer with an explanation after it deserves the whole point; see
// containsFinalAnswer.
func checkAnswerLoose(q benchmarkQ, answer string) bool {
	a := normalizeBenchAnswer(answer)
	exp := strings.TrimSpace(q.Expect)
	if exp == "" {
		return false
	}
	switch q.Match {
	case "numeric":
		return looseNumeric(benchDigits(a), exp)
	case "mcq", "mcq-repeat":
		return looseMCQ(a, exp, q.Prompt)
	}
	return false
}

// looseNumeric decides half credit for a numeric answer in the wrong format.
//
// The organising idea is POSITION: a reply's later assertions supersede its
// earlier ones. Emphasis says the model considered a value; a declaration
// ("the total is 7") says it chose one; whichever comes LAST is what it meant.
// Searching the whole reply for the expected value — the previous behaviour —
// credited a model for any value it happened to mention, which made the mode
// gameable outright ("**40**, **41**, **42**, **43** or **44**. I'll say 44.").
func looseNumeric(digits, exp string) bool {
	// Where the reply last asserted the expected value in an emphasised span,
	// and where it last declared any value. Both are byte offsets, or -1.
	lastExpEmphasis, lastBare, bareAt := -1, "", -1
	for _, loc := range benchEmphasisRe.FindAllStringSubmatchIndex(digits, -1) {
		span := strings.TrimSpace(digits[loc[2]:loc[3]])
		// A span that is a BARE number is an asserted value; one with words around
		// it ("Step 3", "30 feet", "Total cupcakes:") is a label.
		if t := strings.Trim(span, ".,;:!?()[]"); benchBareNumberRe.MatchString(t) {
			lastBare, bareAt = t, loc[0]
		}
		for _, n := range benchNumberRe.FindAllString(span, -1) {
			if numericMatches(n, exp) {
				lastExpEmphasis = loc[0]
			}
		}
	}
	declared, declaredAt := "", -1
	if locs := benchNumDeclaredRe.FindAllStringSubmatchIndex(digits, -1); len(locs) > 0 {
		l := locs[len(locs)-1]
		for g := 1; g <= 2; g++ {
			if l[2*g] >= 0 {
				declared, declaredAt = digits[l[2*g]:l[2*g+1]], l[0]
			}
		}
	}
	// A declaration vetoes only what comes BEFORE it. "Total = 76 … = **114**"
	// is bookkeeping followed by a result, not a commitment to 76.
	if declared != "" && declaredAt > lastExpEmphasis && !numericMatches(declared, exp) {
		return false
	}
	// Among asserted bare values, only the last one counts.
	if lastBare != "" && bareAt > lastExpEmphasis {
		return numericMatches(lastBare, exp)
	}
	if lastExpEmphasis >= 0 {
		return true
	}
	// No emphasis carried the value: fall back to the final stated equation.
	if ms := benchLastEqRe.FindAllStringSubmatch(digits, -1); len(ms) > 0 {
		return numericMatches(ms[len(ms)-1][1], exp)
	}
	return false
}

// looseMCQ decides half credit for a multiple-choice answer in the wrong format.
//
// Only the LAST emphasised span counts. Scanning every span credited formatting
// rather than knowledge: a model that walks the options in bold ("**A)** no.
// **B)** no. **C)** no.") contains an emphasised span for the right letter
// whichever option it finally picks.
func looseMCQ(a, exp, prompt string) bool {
	spans := benchEmphasisRe.FindAllStringSubmatch(a, -1)
	if len(spans) > 0 {
		last := strings.TrimSpace(spans[len(spans)-1][1])
		if last != "" && strings.EqualFold(last[:1], exp) {
			// "**B**" or a span opening "B)"/"B."/"B:"/"B," is a pick;
			// "**Braking**" is prose that merely starts with the letter.
			if len(last) == 1 || last[1] == ')' || last[1] == '.' || last[1] == ':' || last[1] == ',' {
				return true
			}
		}
	}
	// The option's own words, for a reply that answered in prose. Vetoed by an
	// UNAMBIGUOUS declaration of a different option — "one might think it goes
	// backward … I choose D" states the right option's text while choosing
	// another. Ambiguity does not veto, because a bare "A" is also the article.
	if pick, ok := soleDeclaredMCQPick(a); ok && !strings.EqualFold(pick, exp) {
		return false
	}
	if opt := benchOptionText(prompt, exp); opt != "" {
		return benchLooseTextMatch(a, opt)
	}
	return false
}

// soleDeclaredMCQPick reports the option the reply committed to, but only when
// exactly ONE option reads as declared.
//
// Ambiguity must not veto: matchesMCQ accepts a standalone letter, and "A" is
// also the English article — "A helium balloon moves the other way" reads as a
// declaration of option A. Requiring uniqueness keeps that from cancelling a
// correct emphasised pick elsewhere in the same reply.
func soleDeclaredMCQPick(a string) (string, bool) {
	found := ""
	for _, letter := range []string{"A", "B", "C", "D", "E"} {
		if !matchesMCQ(a, letter) {
			continue
		}
		if found != "" {
			return "", false // more than one reads as declared — not a commitment
		}
		found = letter
	}
	return found, found != ""
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
// matchesMCQ is the letter-answer grader, split out so mcq-repeat can fall back
// to it. An explicitly declared pick wins — the LAST one, since verbose models
// restate the options ("A) …") while reasoning and only commit at the end — then
// a letter leading the answer, then benchBareLetterPick (see the benchDeclaredRe
// var block for why bare lowercase a/d never count). An answer that ENUMERATES
// the options starts with an option label rather than its pick, so the
// leading-letter rule stands aside there.
func matchesMCQ(a, exp string) bool {
	if pick, ok := lastSubmatch(benchDeclaredRe, a); ok {
		return strings.EqualFold(pick, exp)
	}
	enumerates := benchEnumerates(a)
	if !enumerates {
		if m := benchLeadingRe.FindStringSubmatch(a); m != nil {
			return strings.EqualFold(m[1], exp)
		}
	}
	pick, ok := benchBareLetterPick(a, enumerates)
	return ok && strings.EqualFold(pick, exp)
}

// benchBareLetterPick reads a pick out of a reply that neither declared one nor
// led with one. It is the last resort, and it used to be "the last standalone
// letter anywhere in the reply" — which is a bet that the last letter a model
// types is the one it chose, and the bet loses on the exact shape tiers 9-10
// invite. All four of these graded as a PASS for a pick the model never made:
//
//	expect C   "The answer must be option D, since vitamin C is irrelevant here"
//	expect B   "I pick E.\nNote: option B was a decoy"
//	expect G   "J — though G is tempting"
//	expect B   "It could be a or b"
//
// That is the false-positive direction — it lets a weak model draw hard traffic
// — and with TEN options a reply that names several letters while eliminating
// them is the norm rather than the exception. So the fallback now wants a reason
// to believe the letter is a choice:
//
//	the reply ENDS on it ("…so C.", "I'd say D", "…0.78 → H", "**C**"), or
//	the reply enumerates the options, in which case its concluding option
//	anchor ("… C) it fails with an error — yes.") is the pick, or
//	the reply contains exactly ONE candidate letter, so there is nothing else
//	it could have meant ("Working through each option, the correct one is C").
//
// A letter disjunction vetoes all three: "a or b" is a model declining to pick.
func benchBareLetterPick(a string, enumerates bool) (string, bool) {
	if benchLetterOrRe.MatchString(a) {
		return "", false
	}
	if m := benchFinalLetterRe.FindStringSubmatch(a); m != nil {
		return m[1], true
	}
	if enumerates {
		// The option labels are the reply's own structure, so its LAST anchor is
		// the option it worked its way to — better evidence than a stray letter
		// in the prose, which is what the old rule graded.
		if ms := benchEnumRe.FindAllStringSubmatch(a, -1); len(ms) > 0 {
			return ms[len(ms)-1][1], true
		}
	}
	seen := map[string]bool{}
	last := ""
	for _, m := range benchLetterRe.FindAllString(a, -1) {
		seen[strings.ToUpper(m)] = true
		last = m
	}
	if len(seen) == 1 {
		return last, true
	}
	return "", false
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
	runs := repeatedLetters(s)
	if len(runs) == 0 {
		return "", false
	}
	return runs[len(runs)-1], true
}

// repeatedLetters returns every standalone run of one repeated option letter in
// s, in order of appearance.
//
// lastRepeatedLetter answers "what did the model finish with"; the full list is
// what filtering the PROMPT's own worked example needs — every mcq-repeat prompt
// carries "if the answer is F, write FFFFF", and a model that quotes it after
// answering would otherwise be graded on F.
func repeatedLetters(s string) []string {
	var out []string
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
			out = append(out, string(c))
		}
		i = j
	}
	return out
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

	// A SINGLE-element expectation cannot use the scan below. Almost any prose
	// line parses as a one-element list, so "the last line that parses" is every
	// line, and walking upward finds the value inside abandoned working —
	// "considering asparagus … but clue 3 rules that out, so the answer is
	// carrot" graded as asparagus. For n=1 only the model's final line counts.
	if len(want) == 1 {
		for i := len(lines) - 1; i >= 0; i-- {
			if strings.TrimSpace(lines[i]) == "" {
				continue
			}
			return listLineMatches(lines[i], want)
		}
		return false
	}

	// Scan upward for the last CANDIDATE line and grade only that one.
	//
	// A candidate is a line that parses as a list AND shares at least one element
	// with the expected answer. Both halves are load-bearing:
	//
	//	requiring a parse alone lets a trailing pleasantry consume the scan —
	//	"Let me know if you want the reasoning, the clues, or the grid." is three
	//	comma-separated parts, so a correct answer on the line above never gets read
	//
	//	scanning for a MATCH instead lets an abandoned intermediate rescue a wrong
	//	conclusion — a reasoning model enumerates orderings while it works, and the
	//	discarded one would be graded in place of the one it committed to
	//
	// Sharing vocabulary separates them, because these are permutation answers:
	// a candidate ordering is built from the same items, while a pleasantry is
	// built from different words entirely.
	for i := len(lines) - 1; i >= 0; i-- {
		if !listLineIsCandidate(lines[i], want) {
			continue
		}
		return listLineMatches(lines[i], want)
	}
	return false
}

// listLineIsCandidate reports whether a line is plausibly the model's answer
// list rather than surrounding prose: it parses into at least as many elements
// as the answer has, and at least one of them is an item from the answer.
func listLineIsCandidate(line string, want []string) bool {
	if idx := strings.LastIndex(line, ":"); idx >= 0 && idx < len(line)-1 {
		line = line[idx+1:]
	}
	got := splitAnswerList(line)
	if len(got) < len(want) {
		return false
	}
	inWant := make(map[string]bool, len(want))
	for _, w := range want {
		inWant[w] = true
	}
	for _, g := range got {
		if inWant[g] {
			return true
		}
	}
	return false
}

// listLineMatches reports whether one line states the wanted list.
//
// The elements are read from the TAIL of the line, not the head. A lead-in is
// prose and may contain commas of its own — "So, the answer is 1, fisherman,
// coffee, musician" splits into five parts for a four-element answer — and
// gating on the total count (the previous behaviour) failed every correct answer
// whose lead-in contained a comma. "So," and "Therefore," are among the most
// common things a model writes, so this was rejecting right answers across 31%
// of the question set.
func listLineMatches(line string, want []string) bool {
	// Drop a leading label ("Answer:", "The final answer is:") so the list is
	// compared, not the sentence around it.
	if idx := strings.LastIndex(line, ":"); idx >= 0 && idx < len(line)-1 {
		line = line[idx+1:]
	}
	got := splitAnswerList(line)
	if len(got) < len(want) {
		return false
	}
	got = got[len(got)-len(want):] // tail-align: drop whatever the lead-in contributed
	for j := range got {
		// The first surviving element may still carry prose ("the answer is 1"),
		// and only it may — every later element must match exactly, because these
		// are permutation answers where a partial overlap is simply wrong.
		if j == 0 && listHeadMatches(got[0], want[0]) {
			continue
		}
		if got[j] != want[j] {
			return false
		}
	}
	return true
}

// listHeadMatches compares the first element of an answer list, tolerating a
// prose lead-in. The suffix has to sit on a word boundary: without that check
// "the answer is 21" would match a wanted "1", turning a wrong permutation into
// a pass.
func listHeadMatches(got, want string) bool {
	if got == want {
		return true
	}
	if !strings.HasSuffix(got, want) {
		return false
	}
	pre := got[:len(got)-len(want)]
	if pre == "" {
		return true
	}
	switch pre[len(pre)-1] {
	case ' ', ':', '*', '_', '`', '"', '\'', '=', '-':
		return true
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
// bought." asserts 16. It declines when the lead holds zero or several numbers (an
// arithmetic chain like "12*8 = 96, …" is working, not an assertion), when the
// lead value is re-used later (then it's an operand mid-computation — "total
// distance 200 m, 200/100 = …" — and the final value must grade), or when the
// answer is a numbered list (a line-leading "1." is a marker, not the answer).
//
// What it returns is a COMMITMENT: the caller grades on it and does not look
// further, because the rest of such a reply is a breakdown of the value asserted
// here and its numbers are components, not conclusions. That is only sound
// because the lead ends at real clause punctuation — see benchCoordinateLeadNumber
// for the conjunction case, which is not a commitment.
func benchLeadNumber(s string) (string, bool) {
	if len(benchNumListRe.FindAllString(s, -1)) >= 2 {
		return "", false
	}
	cut := benchLeadCut(s)
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

// benchLeadCut returns where the leading clause of s ends.
func benchLeadCut(s string) int {
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
	return cut
}

// benchCoordinateLeadNumber reads the value stated by the FIRST of two coordinate
// clauses joined by "and": "The ball costs 5 cents and the bat costs 105 cents."
// states 5 as well as 105.
//
// Unlike benchLeadNumber this is a CANDIDATE, not a commitment — the caller falls
// through to the last-number rule when it does not match. It has to be, because
// the two orderings of that sentence are structurally identical: "The bat costs
// 105 cents and the ball costs 5 cents." is the same reply with its clauses
// swapped, and no positional rule can pick the ball's price out of both. The old
// code treated "and" as a clause break and committed to whatever came first,
// which graded the reversed (equally natural) ordering WRONG — along with every
// reply that reached its conclusion after an "and": "The sum of 8 and 5 is 13."
// (the tier-1 control), "There are 7 sons and 1 sister, so 8.", "20 and 5 more
// makes 25.".
//
// KNOWN TRADE-OFF: offering both values means a reply that VOLUNTEERS a second
// quantity can be graded on it — "The ball costs 5 cents and the bat costs 105
// cents." now also grades a question expecting 105. That is a real widening, and
// it is the direction this file normally refuses. It is accepted here because it
// is unavoidable (see the symmetry argument above), because the model did state
// the value in a full assertion rather than leaving it lying in the working, and
// because the alternative was a systematic false NEGATIVE across 137 numeric
// questions including a tier-1 control. benchAssertVerbRe keeps the widening to
// clauses that actually state something, so an operand ("The sum of 8") is not
// offered.
func benchCoordinateLeadNumber(s string) (string, bool) {
	if len(benchNumListRe.FindAllString(s, -1)) >= 2 {
		return "", false // a numbered list's lead is a marker, not the answer
	}
	cut := benchLeadCut(s)
	loc := benchCoordAndRe.FindStringIndex(s[:cut])
	if loc == nil {
		return "", false
	}
	first := s[:loc[0]]
	if !benchAssertVerbRe.MatchString(first) {
		return "", false
	}
	ns := benchNumberRe.FindAllString(first, -1)
	if len(ns) != 1 {
		return "", false
	}
	for _, later := range benchNumberRe.FindAllString(s[loc[1]:], -1) {
		if numericMatches(later, ns[0]) {
			return "", false // re-used below: an operand, not a conclusion
		}
	}
	return ns[0], true
}

func isASCIIDigit(b byte) bool { return b >= '0' && b <= '9' }

// benchClauseRe ends a clause: sentence enders, newlines, commas, semicolons, colons
// and dashes (a hyphen only when it stands alone, so hyphenated words survive).
var benchClauseRe = regexp.MustCompile(`[.!?;:,\n—–]|\s-\s`)

// containsFinalAnswer implements "contains" and "final-contains": tok must stand
// as its own word where the answer is asserted — in the reply's last ASSERTION,
// or in a terse (≤3-word) leading clause followed by a declarative break ("Mary,
// of course." / "Mount Everest — it just hadn't been discovered yet."). These
// questions plant tok in the prompt, and a wrong answer restating the premise
// carries it only possessive-attached ("Mary's father…") or inside a longer
// subordinate clause ("Before Mount Everest was discovered, K2 …") — never as
// the finally-asserted answer.
//
// v43: a DECLARED answer counts wherever it sits, not only in the final clause.
// Reading the final clause alone failed a CORRECT answer the moment the model
// explained it, and these 16 questions are the benchmark's sanity band — both
// tier-1 controls, three of the four tier-2 floor questions and all three tier-3
// echo traps. Reproduced, expecting "Paris":
//
//	"The answer is Paris. It is located on the Seine."      graded WRONG
//	"Answer: Paris\nExplanation: it has been the capital…"  graded WRONG
//	"The capital is Paris\n- Since 987 AD"                  graded WRONG
//
// and, expecting "Mary", "The answer is Mary, because the puzzle names four
// daughters and then the speaker." All four state the answer and then say
// something about it. See benchDeclaresAnswer for why a DECLARATION is the thing
// to look for rather than "the last clause that mentions the token": the echo
// traps mention it too, and the whole point of these questions is that a
// mention is not an answer.
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
	// Last: an answer DECLARED anywhere and then explained.
	return benchDeclaresAnswer(s, tok)
}

// benchRetractRe marks a clause that contradicts what came before it.
var benchRetractRe = regexp.MustCompile(`(?i)\b(?:but|actually|however|wait)\b`)

// benchRetractLeadRe marks a clause that OPENS by walking back what came before
// it ("but actually it was K2", "No, it was K2"). Anchored at the start, because
// the marker has to be what the clause opens with: "It was not K2 but Everest"
// contains "but" and ASSERTS the token.
var benchRetractLeadRe = regexp.MustCompile(`(?i)^(?:but|actually|however|wait|no|nope|correction)\b`)

// benchLineRe splits a reply into lines. A declaration is tested against lines
// as well as clauses because ':' is itself a clause separator — "Answer: Paris"
// is two clauses, neither of which holds both the marker and the token.
var benchLineRe = regexp.MustCompile(`\n`)

// benchSegment is one clause or line of a reply, with the offset just past it so
// what FOLLOWS a declaration can be checked for a retraction.
type benchSegment struct {
	text string
	end  int
}

// benchDeclaresAnswer reports whether the reply DECLARES tok as its answer
// anywhere in the text — a clause or line ending on "<copula|:|=> tok" ("The
// answer is Paris", "Answer: Paris", "The capital is Paris") — and does not walk
// it back afterwards.
//
// A declaration rather than a mention, and that distinction is the whole design
// of this mode: these questions plant the expected token in the PROMPT, so a
// wrong answer restating the premise contains it too. "Before Mount Everest was
// discovered, K2 was the tallest mountain." puts the token in FRONT of the
// copula rather than after it, and "Thursday, Friday, Saturday. So it is
// Saturday." puts it in an enumeration with no copula at all — so neither
// declares anything, while every shape a right answer takes does. Scanning for
// the token instead would pass all of them, which is exactly what the plain
// substring "contains" did before v39.
func benchDeclaresAnswer(s, tok string) bool {
	re := regexp.MustCompile(`(?i)(?:\bis|\bare|\bwas|\bwere|\bbe|:|=|->|→)\s*["'“‘*_(]*` +
		regexp.QuoteMeta(tok) + `\b["'”’*_)\]]*\s*[.!?]?\s*$`)
	for _, seg := range benchDeclSegments(s) {
		t := strings.TrimSpace(seg.text)
		// tokenAsAnswer as well as the shape, so a possessive or a negated token
		// ("it wasn't Everest") cannot read as a declaration.
		if !re.MatchString(t) || !tokenAsAnswer(t, tok) {
			continue
		}
		if benchRetractedAfter(s[seg.end:]) {
			continue // "The answer is Everest. Wait — K2 was taller at the time."
		}
		return true
	}
	return false
}

// benchDeclSegments returns every clause and every line of s, each with the
// offset just past it.
func benchDeclSegments(s string) []benchSegment {
	var out []benchSegment
	for _, re := range []*regexp.Regexp{benchClauseRe, benchLineRe} {
		start := 0
		for _, loc := range re.FindAllStringIndex(s, -1) {
			out = append(out, benchSegment{text: s[start:loc[0]], end: loc[0]})
			start = loc[1]
		}
		out = append(out, benchSegment{text: s[start:], end: len(s)})
	}
	return out
}

// benchRetractedAfter reports whether anything following a declaration walks it
// back. EVERY following clause is checked rather than just the next one, because
// a self-correction can arrive a sentence later ("The answer is Paris. Hmm,
// wait, actually it's Lyon.") and a declaration the model abandoned is not an
// answer. Conservative by design: it costs a right answer whose explanation
// happens to open on "but", where the other direction costs a wrong answer full
// credit.
func benchRetractedAfter(rest string) bool {
	for _, c := range benchClauseRe.Split(rest, -1) {
		if c = strings.TrimSpace(c); c == "" {
			continue
		}
		if benchRetractLeadRe.MatchString(c) {
			return true
		}
	}
	return false
}

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

// benchBusyStatus reports whether a completion error means "no capacity right
// now" rather than "wrong" or "broken".
//
// 404 is in the list, which looks wrong and is not: a relayed row whose upstream
// is saturated gets health-pruned and re-added on the next refresh, so a pin
// lands on a row that momentarily does not exist. That is congestion wearing a
// 404, and treating it as a permanent failure is what silently recorded 100+
// questions as wrong answers during the first calibration run.
func benchBusyStatus(err error) bool {
	switch completionStatusCode(err) {
	case http.StatusNotFound, http.StatusRequestTimeout, http.StatusTooManyRequests,
		http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	}
	return false
}

// benchBusyTracker counts CONSECUTIVE busy responses across a profiling run.
// Shared by every concurrent question, so the streak measures the worker rather
// than any one question.
type benchBusyTracker struct {
	mu     sync.Mutex
	streak int
	gaveUp bool
}

// busy records a busy response and reports whether the run should be abandoned.
func (t *benchBusyTracker) busy() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.streak++
	if t.streak >= benchBusyMaxStreak {
		t.gaveUp = true
	}
	return t.gaveUp
}

// ok clears the streak: any answer at all proves the worker is reachable.
func (t *benchBusyTracker) ok() {
	t.mu.Lock()
	t.streak = 0
	t.mu.Unlock()
}

func (t *benchBusyTracker) abandoned() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.gaveUp
}

// benchProfileBudget bounds one quality-benchmark pass. It is a wall-clock
// bound, not a question bound, because the thing that varies between workers by
// two orders of magnitude is time per question, not count.
//
// benchMinProfileQuestions is the floor the budget cannot cut into: below it the
// sample is too small to place a worker on a 0-100 scale, and a confidently
// wrong quality score is worse than a slow profile. A worker slow enough to need
// longer than the budget for even this many still gets scored — it just takes as
// long as it takes.
//
// Vars rather than consts so a test can shrink them: what a budget-stopped pass
// does with its results is exactly where the no-think give-up guard was wrong,
// and reproducing that against the real 90 minutes is not a test anyone runs.
// Nothing in the router writes them; timing_log_test.go asserts their values.
var (
	benchProfileBudget       = 90 * time.Minute
	benchMinProfileQuestions = 48
)

// benchStratifiedOrder returns question indices ordered so that any PREFIX of
// the result is spread across tiers, by taking one question from each tier in
// turn. Index order would walk the tiers in blocks, so a truncated run would
// score a worker entirely on whichever tiers happen to come first — and since
// the set is authored easy-to-hard, that means a run cut short would report a
// weak worker as strong.
func benchStratifiedOrder(qs []benchmarkQ) []int {
	byTier := map[int][]int{}
	tiers := []int{}
	for i, q := range qs {
		if _, seen := byTier[q.Tier]; !seen {
			tiers = append(tiers, q.Tier)
		}
		byTier[q.Tier] = append(byTier[q.Tier], i)
	}
	sort.Ints(tiers)
	// Shuffled WITHIN each tier as well as across them, deterministically by
	// content hash. Index order put the hand-authored questions before the
	// generated ones in every tier, so a truncated profile sampled a biased slice
	// of each — and where a tier mixes answerable and unpassable items, that bias
	// had a direction: a worker too slow to finish collected the answerable ones
	// and skipped the rest, scoring HIGHER for being slower. Hashing keeps the
	// order stable across runs (two profiles of the same worker ask the same
	// questions in the same order) while making any prefix representative.
	for _, t := range tiers {
		idx := byTier[t]
		sort.SliceStable(idx, func(a, b int) bool {
			return benchQuestionHash(qs[idx[a]]) < benchQuestionHash(qs[idx[b]])
		})
	}
	out := make([]int, 0, len(qs))
	for round := 0; len(out) < len(qs); round++ {
		progressed := false
		for _, t := range tiers {
			if round < len(byTier[t]) {
				out = append(out, byTier[t][round])
				progressed = true
			}
		}
		if !progressed {
			break // defensive: cannot happen while len(out) < len(qs)
		}
	}
	return out
}

// benchStratifiedOrderOf is benchStratifiedOrder for a list of INDICES into
// benchmarkQuestions, returning positions within that list. Used by the no-think
// pass, which works over the hard-tier subset rather than the whole set.
func benchStratifiedOrderOf(idx []int) []int {
	qs := make([]benchmarkQ, len(idx))
	for j, i := range idx {
		qs[j] = benchmarkQuestions[i]
	}
	return benchStratifiedOrder(qs)
}

// benchBareNumberRe matches a span that is a number and nothing else — the only
// shape loose numeric grading accepts. "114" qualifies; "Step 3" does not.
var benchBareNumberRe = regexp.MustCompile(`^-?\d+(?:\.\d+)?$`)

// benchMagnitudeRe matches a count followed by a magnitude word, in either
// digits ("22 hundred") or words ("two hundred").
var benchMagnitudeRe = regexp.MustCompile(`(?i)\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(hundred|thousand|million)\b`)

// benchMagnitude folds "<count> <magnitude>" into its product. A bare magnitude
// with no count in front of it is left alone for the word map to handle, which
// keeps "a hundred" reading as 100.
func benchMagnitude(m string) string {
	parts := benchMagnitudeRe.FindStringSubmatch(m)
	if len(parts) != 3 {
		return m
	}
	count, err := strconv.Atoi(parts[1])
	if err != nil {
		n, ok := numberWords[strings.ToLower(parts[1])]
		if !ok {
			return m
		}
		if count, err = strconv.Atoi(n); err != nil {
			return m
		}
	}
	scale := map[string]int{"hundred": 100, "thousand": 1000, "million": 1000000}[strings.ToLower(parts[2])]
	if scale == 0 {
		return m
	}
	return strconv.Itoa(count * scale)
}

// benchRequestFor builds the prompt and token budget for one graded question.
//
// SHARED with `bench calibrate` (benchGradeOne) on purpose. The two had drifted:
// this path appended " Give the number only." and calibration did not, so 47 of
// the generated questions had their difficulty measured on a prompt production
// never sends — and that suffix exists precisely because numeric grading is the
// mode a verbose reply fools. A question calibrated under different conditions
// than it is graded under is calibrated for nothing, which the calibration file
// asserts in a comment while doing the opposite.
func benchRequestFor(q benchmarkQ, think bool, ctxTokens int) (prompt string, maxTokens int) {
	prompt = q.Prompt
	// Only NUMERIC grading is fooled by a verbose reasoned reply (an intermediate
	// or a trailing restatement gets read as the answer), so make the worker
	// answer with just the number — unless the prompt already states an answer
	// format of its own. mcq (letter) and contains (substring) grade fine through
	// a long reply.
	if q.Match == "numeric" && !benchStatesAnswerFormat(q.Prompt) {
		prompt += " Give the number only."
	}
	// The budget is a property of the QUESTION, not of the thinking mode. Tying
	// it to the mode made the two quality scores differ in two variables at once:
	// Quality was measured with 16384 tokens and QualityNoThink with 1024 on the
	// same hard-tier questions — LiveBench maths, olympiad and AMC, where the
	// working does not fit in 1024 and truncation is scored a fail. So
	// QualityNoThink largely measured "can you fit the working into a sixteenth
	// of the room" rather than "how good are you without a scratchpad", and every
	// no-think routing decision was made against a systematically depressed
	// number. The two scores now differ ONLY in the thing they are named for.
	maxTokens = benchMaxTokens
	if q.Tier >= benchHardTier {
		maxTokens = benchThinkMaxTokens
	}
	// Never ask for more room than the worker has. A 32K ceiling exceeds the
	// whole context window of several workers here (llm-cpu-gemma is 32K
	// including the prompt), and a server handed a max_tokens it cannot honour
	// either errors — turning a gradeable question into a transport failure — or
	// silently clamps, which is fine but means the ceiling was never real. The
	// deadline still binds long before this does on a slow worker.
	if ctxTokens > 0 {
		room := ctxTokens - benchPromptTokenEstimate(prompt) - benchAnswerReserve
		if room < benchMinAnswerTokens {
			room = benchMinAnswerTokens
		}
		if room < maxTokens {
			maxTokens = room
		}
	}
	if !think {
		prompt += " /no_think" // belt-and-suspenders with the kwarg (matches chatProbe)
	}
	return prompt, maxTokens
}

// benchStatesAnswerFormat reports whether a prompt already tells the worker how
// to shape its answer, in which case benchRequestFor must not append one of its
// own.
//
// v43: 21 of the LiveBench spatial items say, verbatim, "put your answer in
// **bold** as a single integer (for example, **0**)" — and the benchmark then
// appended " Give the number only." to all of them. The model was told to do two
// contradictory things in one prompt, so whatever it did violated one of them.
// That is a confound on the very grader the suffix exists to help, and since the
// bank has no instruction-following category (maths 199, reasoning 147, coding
// 38, general 8), the only place instruction-following showed up at all was as
// noise inside the numeric score.
//
// Detected from the prompt TEXT rather than from a list of question ids, because
// `bench emit` regenerates the bank and an id list would rot silently. The test
// is deliberately loose: skipping the suffix on a question that did not need it
// costs a little parse robustness, while adding it to one that specifies its own
// format corrupts the measurement, so when in doubt, skip.
func benchStatesAnswerFormat(prompt string) bool {
	p := strings.ToLower(prompt)
	for _, marker := range []string{
		"number only", "only the output", // the shapes this file's own questions use
		"put your answer", "put your final answer", "in **bold**", "in bold",
		"following format", "single integer", "single number", "single word",
		"answer with", "answer as", "answer in the form", "give your answer",
		"give the answer", "give only", "respond with", "reply with",
		"output only", "state only", "format your answer", "express your answer",
		"for example, **",
	} {
		if strings.Contains(p, marker) {
			return true
		}
	}
	return false
}

// benchQuestionHash is a stable content hash used to order questions within a
// tier. Content-based rather than positional so inserting a question does not
// reshuffle the rest, and stable across processes so a re-profile of the same
// worker asks the same questions in the same order.
func benchQuestionHash(q benchmarkQ) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(q.Prompt))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(q.Expect))
	return h.Sum64()
}

const (
	// benchAnswerReserve is slack between the prompt and the answer ceiling, for
	// the chat template's own tokens and for the difference between a character
	// estimate and the worker's real tokenizer.
	benchAnswerReserve = 512
	// benchMinAnswerTokens is the floor the context clamp will not go below. A
	// worker whose window cannot hold even this much answer for this prompt is
	// going to fail the question either way, and the failure should read as a
	// wrong answer rather than as a request the server rejects.
	benchMinAnswerTokens = 256
)

// benchPromptTokenEstimate approximates a prompt's token count. ~4 characters
// per token, the same approximation used everywhere else in the router that has
// no tokenizer; it only has to be good enough to keep the answer ceiling inside
// the context window, and benchAnswerReserve absorbs the error.
func benchPromptTokenEstimate(prompt string) int { return len(prompt)/4 + 1 }

// applyBenchThinking writes the thinking gate into a benchmark payload using the
// dialect measured for this worker.
//
// Deliberately mirrors patchForwardedBody's switch rather than sharing it: that
// operates on a caller's raw JSON body and has to merge with whatever the client
// already sent, while this builds a payload the router owns outright. Sharing
// would mean marshalling a map to bytes and back on every graded question. What
// must not drift is the DECISION — which spelling for which dialect — so the
// cases are kept in the same order with the same reasoning.
func applyBenchThinking(payload map[string]any, b *Backend, think bool) {
	switch b.ThinkingDialect {
	case thinkingDialectEffort:
		if think {
			payload["reasoning_effort"] = "medium"
		} else {
			payload["reasoning_effort"] = "none"
		}
	case thinkingDialectNone:
		// Measured: neither gate does anything on this endpoint. Writing one buys
		// a field a strict endpoint can reject, in exchange for nothing — and the
		// two scores will legitimately agree, which is the truth about it.
	default:
		// Unknown — never probed, or a profile cached before the dialect was —
		// so use the spelling the fleet has always spoken.
		payload["chat_template_kwargs"] = map[string]bool{"enable_thinking": think}
	}
}
