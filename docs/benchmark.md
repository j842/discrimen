# The quality benchmark

This is the full question set discrimen uses to rate a worker's quality, and its
answer key. Both are published deliberately. A quality score nobody can inspect
is a number you have to take on faith, and this one decides where requests go.

## What it is for

When a worker joins the fleet the router knows nothing about it except what its
configuration claims. Cold-start profiling replaces that claim with a
measurement: the router asks the worker all 130 questions below, grades the
answers, and stores a score from 0 to 100 on the worker's profile. Difficulty
routing then maps a request's estimated difficulty onto that measured scale, so
a hard prompt goes to a worker that has actually demonstrated it can handle hard
prompts rather than to one that declared itself capable.

That is also why the questions are chosen to **spread** the fleet rather than to
be uniformly hard. If every worker scores the same, the score stops
distinguishing them and routing collapses to pure speed. Tiers 9, 10 and 11 — 29
of the 130 questions — exist only because the tier below each of them saturated,
and two further tiers were built, measured and deleted along the way. Tier 12 is
there for a different reason: it is the only tier that asks whether a worker can
read code rather than solve a puzzle.

## Versioning

`benchmarkVersion` (in `internal/router/benchmark.go`) is currently **36**. It is
part of the profile cache key: a stored profile is reused only if its
`BenchVersion` matches the running binary's. Changing a question, adding a tier,
or altering how answers are graded means bumping that constant, which invalidates
every cached profile in the fleet and re-measures every worker against the new
set.

That coupling is the reason the question set is committed to the repository
rather than fetched at install time. If the questions could change without the
version changing, a worker profiled on Tuesday and one profiled on Thursday
would be graded on different sets and then compared on the same 0-100 scale,
silently, with every number still looking plausible.

> Source of truth: `internal/router/benchmark_data.go` (the questions) and
> `internal/router/benchmark.go` (the grading). This document is a derived
> snapshot. If the two disagree, the Go files win.

### Version history

Every entry below is a change that made an older score mean something different,
which is the only kind of change that belongs in this list.

| Version | Change |
|---------|--------|
| v24 | A question the worker cannot answer within `benchAnswerDeadline` is scored a speed fail, counted wrong and not retried, instead of a retried transport error. |
| v25 | Length truncation counts as a failure. It used to be excluded from the denominator; now it matches `responseInadequate`, which treats a length finish as inadequate at runtime. |
| v26 | `benchAnswerDeadline` cut from 5 minutes to 2, so a model too slow to answer inside 2 minutes fails at that difficulty. This is what spread the slow reasoning models down the quality range. |
| v27 | `thinkingProbe` recognises the `reasoning` field. vLLM 0.23 renamed `reasoning_content` to `reasoning`, and until this landed vLLM workers false-negatived as thinking-incapable and were routed around in favour of CPU fallbacks. |
| v28 | All response extraction unified on `extract.go`: reasoning-aware speed and chat probes, benchmark grading falls back to the reasoning field, the `"<nil>"` content bug fixed, and capability probes retry transients. |
| v29 | Tier 9 added: twelve GPQA-Diamond-style expert-knowledge questions. The mcq grader widened from A-D to A-J to read ten-option picks. |
| v30 | Tier 10 added, after a 27B scored 100% on tier 9. |
| v31 | The expert-recall tier cut outright at 17 points of measured spread, the unrecallable tier promoted to 9, and tier 10 rebuilt as SimpleBench-style world-model traps. Question count unchanged at 97. |
| v32 | The train/tunnel and Mary's-father traps moved from tier 2 to tier 3, so both are graded thinking-on. A trap graded thinking-off measures one-shot reflex, not capability. |
| v33 | Tier 11 added, and the frontier tiers (at or above `benchFrontierTier`) given the longer 6-minute deadline. |
| v34 | Quality became a weighted two-bucket score instead of a flat percentage. |
| v35 | Tier 12 added and the weighted score grew a third bucket for it. The numeric grader now keeps the sign (`benchNumberRe`). |
| v36 | Question 89 (tier 10, the lift) revised. It never stated that the lift beats a man running up 30 flights while offering "it depends on the lift's speed" as an option, so that option was defensible and three of four workers spent their whole budget on the one item instead of answering it. Both journey times are now stated, and the escape hatch is replaced by an age decoy. |

v33 and v34 both landed **without** bumping the constant, which is exactly the
failure the constant exists to prevent. Profiles measured under the flat
percentage stayed in `worker_profiles` marked current, and `autoTargetQuality`
reads every cached score as one absolute 0-100 scale, so pre-v33 and post-v34
workers were being compared on scales that no longer meant the same thing.
`TestBenchmarkVersionCoversScoringChanges` fails CI if it happens again: it
finds the highest `vNN:` marker in `benchmark.go` and `benchmark_data.go` and
requires `benchmarkVersion` to be at least that.

## Grading

**Size.** 130 questions across 12 tiers.

| Tier | Questions | Mode | Deadline |
|------|-----------|------|----------|
| 1 — controls | 4 | thinking-off | 2 min |
| 2 — floor | 4 | thinking-off | 2 min |
| 3 — mid | 7 | thinking-on | 2 min |
| 4 — upper-mid | 25 | thinking-on | 2 min |
| 5 — hard | 7 | thinking-on | 2 min |
| 6 — traps | 8 | thinking-on | 2 min |
| 7 — harder | 5 | thinking-on | 2 min |
| 8 — frontier | 13 | thinking-on | 2 min |
| 9 — unrecallable | 12 | thinking-on | 6 min |
| 10 — ceiling | 12 | thinking-on | 6 min |
| 11 — insight | 5 | thinking-on | 6 min |
| 12 — programming | 28 | thinking-on | 6 min |

**Thinking mode.** Every question is graded in the mode the router would serve
it in. `benchHardTier` is **3**: tiers below it are graded thinking-off, tiers at
or above it thinking-on. This measures a worker the way production uses it, so a
reasoning-first model is rated on its reasoning with the scratchpad it actually
gets, rather than on a thinking-off reflex it never runs at that difficulty.

The mode is set two ways at once, because workers disagree about which one they
honour: `chat_template_kwargs: {"enable_thinking": …}` in the request body, and,
for thinking-off, ` /no_think` appended to the prompt text.

**Token ceilings.** Thinking-off answers get **1024** tokens (`benchMaxTokens`);
they are short trap and recall replies. Thinking-on answers get **16384**
(`benchThinkMaxTokens`) — 8192 truncated some tier-7 and tier-8 questions.

**Temperature.** 0, for every question. A graded benchmark has to be
deterministic: the same model must return the same answer on every re-profile,
and two identical models on different hosts must score identically. With
default-temperature sampling the score wobbles by 1 to 2 points between
re-profiles purely from the random draw. vLLM and llama.cpp both treat
temperature 0 as argmax. Heterogeneous hardware can still differ on a rare
near-tie in the logits, but the dominant noise source is gone.

**Prompt rewriting.** Questions graded `numeric` get ` Give the number only.`
appended unless the prompt already says `number only` or `only the output`. Only
numeric grading is fooled by a verbose reply, where an intermediate value or a
trailing restatement can be read as the answer; `mcq` and `contains` grade fine
through a long reply and are sent unchanged.

**Weighted scoring.** The score is a three-bucket weighted percentage, not a flat
one:

- **Base**, tiers 1 to 10 (below `benchInsightTier` = **11**), shares
  `benchBaseWeight` = **60** points pro rata. With the current set that is 97
  questions, so each is worth about 0.62 points.
- **Insight**, tier 11, shares `benchInsightWeight` = **20** points pro rata.
  With the current set that is 5 questions, so each is worth 4 points.
- **Coding**, tier 12 and above (`benchCodingTier` = **12**), shares
  `benchCodingWeight` = **20** points pro rata. With the current set that is 28
  questions, so each is worth about 0.71 points.

So `score = round(60 × base/97 + 20 × insight/5 + 20 × coding/28)`.

A model that sweeps tiers 1 to 10 and can touch neither hard band caps at **60**;
one that can also read code but not find the insight shortcut reads **80**. Under
the flat percentage this file used before v34, the 5 insight questions were about
5% of the score and a model that could not solve them read about 96 —
indistinguishable from one that could. The top 40 points are now earned only
where the top models actually differ.

Coding gets its own bucket rather than joining insight, which is what appending
tier 12 to the old `t >= benchInsightTier` rule would have done. That would have
put 28 coding questions in the same 30-point bucket as tier 11's 5 and cut tier
11's contribution from 30 points to 4.5 — undoing exactly what the weighting was
introduced for.

Each bucket is internally count-independent, so questions can be added within a
bucket without rescaling anything downstream. An empty bucket's weight is
redistributed across the buckets that do have questions, in proportion to their
nominal weights, so a score measured on a set without coding questions stays
comparable with one measured on the full set — `autoTargetQuality` reads all of
them on a single absolute 0 to 100 scale. See `benchWeightedScore`.

**Failures.** Everything that produces no usable answer counts as wrong and
stays in the denominator:

- **Wrong answer.** Counted wrong, obviously.
- **Truncated.** `finish_reason: "length"` — the model ran out of token budget
  without concluding. Counted wrong. This matches the runtime signal:
  `responseInadequate` treats a length finish as inadequate, so the cold-start
  score and the live inadequacy signal agree. On a thinking-on tier with a 16k
  budget, a truncation means the model burned its entire budget without reaching
  a conclusion, which is a real failure at that difficulty. A truncated answer
  that still parsed to the right result is rare, but counts as a pass. The
  truncation count is reported separately in the breakdown so a context or budget
  problem stays visible rather than showing up as fake low quality.
- **Too slow.** A question the worker cannot answer within its deadline is a
  usability failure, counted wrong and **not retried** — a retry would only burn
  another deadline for the same verdict. The bound is `benchAnswerDeadline` = **2
  minutes** for tiers 1 to 8, and `benchAnswerDeadlineFrontier` = **6 minutes**
  for tiers at or above `benchFrontierTier` = **9**. The frontier tiers get the
  longer bound because at that difficulty the question is whether the model can
  reason at all, and production already prices patience separately: the router
  ranks candidates by expected completion time, so a slow-but-right worker loses
  the race for quick prompts without having its quality falsified first. Tiers 3
  to 8 keep the tight bound on purpose, because that spread of slow mid-tier
  reasoners is real and wanted.

  The test is `q.Tier >= benchFrontierTier`, so **tier 12 inherits the 6-minute
  deadline** even though nothing about it was considered when the bound was
  chosen. That rationale was written in v33 for long-form frontier reasoning,
  where a 284B MoE running at 18 tok/s was scoring below a saturated 27B on
  nothing but tier-10 speed fails. Tier 12 is 28 short code traces whose answers
  are single numbers or single words. Six minutes each is far more patience than
  those questions need, and the effect is that a worker slow enough to be
  unusable on code tracing is not caught by the deadline on this tier. If a
  narrower bound is wanted for tier 12, it needs its own constant rather than a
  change to `benchFrontierTier`, which would also move tiers 9 to 11.
- **Errored.** Any other request failure is treated as transient — a dropped
  request under concurrent profiling load — and retried up to
  `benchMaxAttempts` = **3** times with a backoff of 1 second times the attempt
  number. Only a request that fails every attempt is recorded as errored, and it
  counts as a failure.

**Discarded runs.** If more than half the questions error, the worker went
unreachable mid-run. It is not bad, it just cannot be graded, so the run is
discarded rather than persisted as an under-rating.

**Concurrency.** Questions are graded concurrently, bounded by the worker's
measured concurrency, so a full thinking-on benchmark finishes in minutes rather
than about twenty.

**Records.** Each question's prompt, expected answer, the model's actual answer
(tail-truncated to 500 bytes, since the final answer sits at the end), and the
pass/fail/truncated/errored/slow flags are stored on the worker's profile and
served from `GET /backends/{id}/benchmark`. The per-tier breakdown is logged as
`t1=4/4 t2=4/4 …`, with `trunc=`, `slow=` and `err=` counts appended when
non-zero.

**Gate audit.** `auditThinkingGate` runs the router's own learned reasoning gate
over these prompts once per router lifetime and logs where its decision disagrees
with the hand-labelled `benchHardTier` boundary. It is an independent
cross-check on the gate's threshold and seeds. It feeds no score.

## Match modes

Before any matching, the answer is normalised: `\frac{a}{b}` (and `\dfrac`,
`\tfrac`) becomes `a/b`, `\boxed{X}` is unwrapped, LaTeX and markdown scaffolding
is stripped (`\text`, `\mathrm`, `\mathbf`, `\left`, `\right`, spacing macros,
`$`, braces, `_`, `^`, backslashes), and unicode sub/superscript digits are
folded to ASCII. So `$\text{H}_2\text{O}$` matches `H2O`, `x²` matches `x2`, and
`\boxed{C}` matches `C`.

`checkAnswer` then dispatches on the question's mode.

**`numeric`** — tried in three stages, in order:

1. An explicitly declared value: "the answer is 1", "total: 16", "5 is the
   answer". The **last** declaration wins, so a self-correction ("…= 2 … wait,
   actually the answer is 1") grades by its final claim.
2. A leading-clause assertion. An answer-then-breakdown reply ("He has 16 cows:
   the 8 that survived plus the 8 he bought.") states its value first. This stage
   declines when the leading clause holds zero or several numbers (an arithmetic
   chain is working, not an assertion), when the lead value reappears in the
   working below it (then it was an operand, not a conclusion), or when the reply
   is a numbered list (a line-leading "1." is a marker).
3. The **last** number in the reply. A model that shows intermediates or
   self-corrects states its final value last, and an intermediate must never
   rescue a wrong conclusion.

Comparison is by numeric **value**, not string, so `50.40` matches `50.4` and
`114.00` matches `114`. Spelled-out numbers are accepted (`six` → 6,
`twenty-two` → 22; compounds are rewritten before the single words, or
"twenty-two" would read as 20 and 2 and the last-number rule would grade the 2).
Thousands separators are collapsed first, so a comma inside `1,024` is never read
as a clause break.

**The sign is part of the number (v35).** `benchNumberRe` is `-?[0-9]+(?:\.[0-9]+)?`,
and both capture groups in `benchNumDeclaredRe` take an optional sign too. Before
v35 there was no `-?` anywhere, and the extraction stages dropped the minus before
`numericMatches` ever compared, which broke grading in both directions:

- A question whose expected answer was negative **could never pass**. Every
  extraction stage handed `numericMatches` an unsigned number, so an expected
  `-2` had nothing that could match it.
- Worse, the reverse graded as a **false pass**. An expected `2` against a model
  answer of `-2` read as `2` and was marked correct, so a model that got the
  sign wrong scored as if it had got the question right.

Nothing in tiers 1 to 11 has a negative answer, so the fault was dormant. It
surfaced while the answer keys for tier 12 were being verified by executing every
program, which is the only reason it was found at all: a grader fault of this
shape presents as a model getting an easy question wrong, not as a bug in the
grader. Tier 12 question 130 still carries a `+ 10` in its expression, added by
its author to keep the expected answer positive before the fix landed.

The sign is taken only when it is adjacent to the digits, so a spaced subtraction
("10 - 3 = 7") still reads its operands unsigned.

**`mcq`** — a letter answer, tried in three stages:

1. An explicitly declared pick ("the answer is B", "Answer: C"). The **last**
   one, since verbose models restate the options while reasoning and only commit
   at the end.
2. A letter leading the answer ("C. because…", "(B)") — unless the reply
   *enumerates* the options ("A) … no. B) … no. C) … yes."), in which case the
   leading letter is an option label rather than a pick and this stage stands
   aside.
3. The **last** standalone letter. This is the fallback that matters most,
   because it is where a reasoning model's concluding option ends up.

The range is A–J, not A–D, because tiers 9 and 10 carry ten options. Ten choices
drop the random-guess floor from 25% to 10% and cut prompt-sensitivity variance,
which matters when every question is scored pass/fail and guess luck is pure
noise in the percentage.

Two letters are handled specially. The last-standalone-letter class admits
uppercase plus lowercase `b` and `c` only: a bare lowercase "a" or "d" is almost
always the article or the "I'd" contraction, and a plain `(?i)\b[a-d]\b` misread
prose answers ("…causing a syntax error" → "a"). And bare "I" is the pronoun,
which opens a large share of reasoning prose ("I need to check…"), so it is
excluded from that class entirely and, when declared, must carry trailing
punctuation or end of line. A model that picks option I still grades correctly
via "Answer: I." or "I)"; one that answers a bare "I" grades as a miss, which is
why no question in this file uses I as its key.

**`mcq-repeat`** — for questions that ask the model to repeat its chosen letter
("DDDDD"). The plain `mcq` matchers look for *standalone* letters and "DDDDD" is
one token, so those questions would all grade as misses. This mode reads the last
run of three or more identical uppercase option letters, then falls back to the
ordinary `mcq` rules for a model that ignored the instruction and answered "D".
Three rather than five, so a model that miscounts the repetition still grades on
the letter it picked; last rather than first, because a reasoning model echoes
the instruction's own example before committing to its own pick. Not used by any
question in this file — it exists for LiveBench's AMC items.

**`exact-list`** — an ordered comma-separated answer ("1, filmmaking,
police-officer, journalist"). Every element must match in order, because these
are permutation answers and a partial overlap is simply wrong. Separator spacing,
letter case and any surrounding prose are the model's business, not the grader's;
XML-ish wrappers such as `<solution>…</solution>` are stripped. The reply is
scanned line by line **from the bottom**, so a worker that shows its working and
then states the answer passes, while one whose final answer is wrong is not
rescued by a correct intermediate written earlier. Not used by any question in
this file — it exists for LiveBench's zebra puzzle and olympiad items.

**`final-contains`** — for the echo traps, where the expected token appears in
the prompt itself. Plain substring matching would pass a wrong answer that merely
restates the premise ("Mary's father's fifth daughter is Lulu"). The token counts
only where an answer is actually asserted: in the final clause, or in a terse
(three words or fewer) leading clause followed by a declarative break ("Mary, of
course."). A possessive form ("Mary's") does not count, a negated one ("not
Mary") does not count, and a terse lead the next clause walks back ("Everest, but
actually it was K2.") does not count.

**`contains`** — case-insensitive substring. The default, used where the expected
answer is a distinctive word or token that cannot plausibly appear by accident.

Across the 130 questions: 90 `numeric`, 24 `mcq`, 13 `contains`, 3
`final-contains`. Tier 12 is deliberately all `numeric` and `contains` — see
that tier's comment block for why multiple choice was cut from it.

## What actually spreads the fleet

Two whole tiers have been built, measured and deleted getting to the current
twelve. The per-tier numbers off the live fleet (Qwen3.6-27B / Granite-8B /
Gemma-26B-A4B) overturn the obvious intuitions and are worth recording:

| Tier | 27B | 8B | 26B-A4B | Spread |
|------|-----|-----|---------|--------|
| 6 — traps | 100% | 38% | 100% | 62 points |
| 9 — unrecallable | 100% | 67% | 42% | 58 points |
| 8 — frontier maths | 100% | 69% | 77% | 31 points |
| expert recall (built, measured, cut) | 100% | 92% | 83% | 17 points |

Harder maths does not work: reported AIME'25 across the Qwen3 family runs 4B 65.6
→ 8B 67.3 → 14B 70.4 → 30B-A3B 70.9 → 32B 72.9. Seven points from a 4B to a 32B,
because a scratchpad is most of what those problems need.

Harder knowledge does not work either. A graduate-science tier in the
GPQA-Diamond style looked right on paper, since a reasoning budget cannot supply
a fact the model never learned. It scored 17 points of spread and a flat 100% for
the 27B, and was deleted. Single-fact recall saturates from below far faster than
you expect.

What works is anything the model can neither retrieve nor grind out: rules
invented in the prompt (tier 9), world-model traps where the obvious computation
is a decoy (tiers 6 and 10), and problems whose brute-force path exceeds the
token budget while the closed form fits in a thousand tokens (tier 11). So the
standing rule for adding a tier: do not pick a limit that is the model's memory,
and do not pick one that is its token budget. Pick one that is its world model.

Tier 12 is the exception to that rule, and it is there for a different purpose.
It measures 51 points of spread across four live workers (39.6 / 68.8 / 87.5 /
97.9 percent), and its Spearman rho against the existing quality score is
**+1.00**, so it ranks the fleet the same way tiers 1 to 11 already do. It is
therefore not buying new discrimination. What it buys is a programming-specific
measurement in a set that is otherwise entirely puzzles, so a worker's score says
something about whether it can be handed a codebase rather than a riddle.

All hand-authored questions here are original rather than lifted from the
benchmarks they are modelled on. GPQA in particular is gated behind an agreement
not to reproduce its items in plain text, precisely to keep them out of training
corpora, so copying would both breach that and self-contaminate this file. The
same reasoning applies to the SimpleBench and MisguidedAttention material the
trap tiers are built in the spirit of.

---

# The questions

Questions are numbered 1 to 130 in the order they appear in
`benchmark_data.go`. Multi-line prompts are shown in code blocks so the exact
text sent to the worker is reproduced; the router sends each as a single string.

## Tier 1 — controls (4 questions, thinking-off)

*Every working model passes. A miss here means the worker is broken, not weak.*

**1.** What is 8 + 5?

**2.** What is the capital of France? Answer with one word.

**3.** How many days are in a week?

**4.** What is the chemical symbol for water? Answer with the formula only.

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 1 | **13** | numeric | Sanity floor. |
| 2 | **Paris** | contains | |
| 3 | **7** | numeric | |
| 4 | **H2O** | contains | The normaliser folds `$\text{H}_2\text{O}$` and `H₂O` onto this. |

## Tier 2 — floor (4 questions, thinking-off)

*The weakest model in the fleet (about 1B) fails these; every competent model
passes. Nothing here is a trap: a trap graded thinking-off measures one-shot
reflex rather than capability, which is why the two that used to sit here now sit
in tier 3.*

**5.** A man has 7 sons, and each of his sons has exactly one sister. How many children does the man have in total? Give the number only.

**6.** Who discovered the neutron? Surname only.

**7.** What is the collective noun for a group of crows? Answer with one word.

**8.** A hen and a half lays an egg and a half in a day and a half. How many eggs does one hen lay in one day? Give your answer as a fraction.

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 5 | **8** | numeric | The 7 sons all share **one** sister, not seven each. 7 + 1 = 8. |
| 6 | **Chadwick** | contains | James Chadwick, 1932. |
| 7 | **murder** | contains | |
| 8 | **2/3** | contains | Rate = eggs / (hens × days) = 1.5 / (1.5 × 1.5) = two thirds of an egg per hen per day. |

## Tier 3 — mid (7 questions, thinking-on)

*Careful-reading traps a small model (about 4B) slips on even with a scratchpad,
while a strong model reads them correctly. Plain arithmetic loses its bite
thinking-on, so this tier is all traps. Three use `final-contains` because the
expected token appears in the prompt itself.*

**9.** Johnny's mother had four children. The first was named April, the second was named May, and the third was named June. What was the name of the fourth child? Answer with one word.

**10.** Mary's father has five daughters named Lala, Lele, Lili, and Lolo, and one more. What is the name of the fifth daughter? Answer with one word.

**11.** A bat and a ball cost 1 dollar and 10 cents in total. The bat costs 1 dollar more than the ball. How many cents does the ball cost? Give the number only.

**12.** A farmer has 15 cows. All but 8 of them die. He then buys 8 more cows. How many cows does he have now? Give the number only.

**13.** A clock takes 6 seconds to strike 4 o'clock (it chimes 4 times). How many seconds pass between two consecutive chimes? Give the number only.

**14.** A train 100 metres long travels at 100 metres per second through a tunnel that is 100 metres long. How many seconds pass from when the front of the train enters the tunnel until the rear of the train exits it? Give the number only.

**15.** Before Mount Everest was discovered, what was the tallest mountain above sea level on Earth? Answer with one word.

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 9 | **Johnny** | final-contains | April/May/June primes you for "July", but the fourth child is the subject of the sentence. |
| 10 | **Mary** | final-contains | The -la/-le/-li/-lo pattern lures you to "Lulu"; it is *Mary's* father. |
| 11 | **5** | numeric | ball + (ball + 100c) = 110c, so 5c, not the reflexive 10c. |
| 12 | **16** | numeric | "All but 8 die" leaves 8, plus 8 bought = 16. Not 15 − 8. |
| 13 | **2** | numeric | Four chimes leave three gaps: 6 / 3 = 2, not 6 / 4. |
| 14 | **2** | numeric | The rear clears the tunnel after the train covers tunnel + train = 200 m at 100 m/s. The trap answer is 1. |
| 15 | **Everest** | final-contains | Being undiscovered does not change a mountain's height. |

## Tier 4 — upper-mid (25 questions, thinking-on)

*The largest tier, and mixed on purpose. The multi-step arithmetic items were
thinking-off discriminators and are now easy thinking-on points; the spread in
this tier comes from the compiler, shell and chemistry gotchas, which hinge on
knowledge a reasoning budget cannot supply, and from the GSM-Symbolic-style
irrelevant-clause traps, where a true-but-useless extra clause lures a model into
over-computing.*

**16.** How many cubic metres of soil are inside a hole that is 2 metres deep, 1 metre wide, and 1 metre long? Give the number only.

**17.** A book has 250 pages. Tom reads 40% of it on Monday and another 30 pages on Tuesday. How many pages are left? Give the number only.

**18.** What is 15% of 80, plus 20% of 50? Give the number only.

**19.** A worker earns 12 dollars per hour for the first 8 hours of a shift and 18 dollars per hour for every hour after that. How much does she earn for a 10-hour shift? Give the number only.

**20.**
```
What does this C++ program print? Give only the output.
#include <iostream>
int main(){ std::cout << 5 / 2 * 2.0; }
```

**21.**
```
What does this C++ program print? Give only the output.
#include <iostream>
int main(){ int a = -1; unsigned b = 1; std::cout << (a < b); }
```

**22.**
```
What does this bash script print? Give only the output.
x=$(printf 'hello\n\n\n')
echo "${#x}"
```

**23.**
```
What happens when this bash script runs?
month=09
echo $((month + 1))
A) it prints 10
B) it prints 9
C) it fails with an error
Answer with just the letter.
```

**24.**
```
What does this C++ program print? Give only the output.
#include <iostream>
#include <vector>
int main(){ std::vector<int> v = {1, 2, 3}; std::cout << (v.size() > -1); }
```

**25.**
```
What does this C++ program print? Give only the output.
#include <iostream>
int main(){ std::cout << (3 == 3 == 3); }
```

**26.**
```
What does this C program print? Give only the output.
#include <stdio.h>
int main(){ printf("%d", 0.1 + 0.2 == 0.3); }
```

**27.** In Python 3, what does the expression 2 ** 3 ** 2 evaluate to? Give the number only.

**28.**
```
What does this bash script print? Give only the output.
x=$((2#101 + 1))
echo "$x"
```

**29.**
```
How many electrons are in a neutral atom of carbon-14?
A) 14
B) 8
C) 6
D) 12
Answer with just the letter.
```

**30.**
```
Which chemical element has atomic number 11?
A) Neon
B) Sodium
C) Nitrogen
D) Lithium
Answer with just the letter.
```

**31.** If today is Wednesday, what day of the week will it be 100 days from now? Answer with the day name.

**32.** Liam has 5 boxes, each holding 8 apples. He gives away 3 apples, and notices that 2 of the boxes are painted red. How many apples does Liam have now? Give the number only.

**33.** A shop sells pens at 3 for 2 dollars. Tom buys 12 pens. The shop floor is tiled in blue. How many dollars does Tom pay? Give the number only.

**34.** A baker makes 3 boxes of 12 cupcakes and 5 boxes of 8 cupcakes. She sells exactly three quarters of all the cupcakes at $2 each. What is her total revenue in dollars?

**35.** A store buys widgets at $8 each and marks them up by 50%. During a sale it takes 20% off the marked price. If it sells 30 widgets at the sale price, what is the total revenue in dollars?

**36.** Two trains start 300 km apart on the same track and head toward each other, one at 60 km/h and the other at 90 km/h. How many minutes until they meet?

**37.**
```
Consider this Python function:
def f(n):
    r = 0
    for i in range(1, n + 1):
        if i % 3 == 0 or i % 5 == 0:
            r += i
    return r
What does f(20) return?
```

**38.** How many distinct arrangements (orderings) are there of the letters in the word BANANA?

**39.** A car uses 8 litres of fuel per 100 km. On a 350 km trip with fuel at $1.80 per litre, what is the total fuel cost in dollars?

**40.** What is the smaller angle, in degrees, between the hour and minute hands of a clock at exactly 3:15?

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 16 | **0** | numeric | A hole holds no soil. It is empty by definition. |
| 17 | **120** | numeric | 40% of 250 = 100, plus 30 = 130 read; 250 − 130 = 120 left. |
| 18 | **22** | numeric | 12 + 10. |
| 19 | **132** | numeric | 8 h at $12 = $96, plus 2 h at $18 = $36. |
| 20 | **4** | numeric | `5 / 2` is **integer** division = 2, then `2 * 2.0` = 4.0. Not 5. |
| 21 | **0** | numeric | `a` is converted to **unsigned** for the comparison, becoming a huge value, so `a < b` is false. |
| 22 | **5** | numeric | Command substitution strips trailing newlines, so `x` is "hello". |
| 23 | **C** | mcq | Arithmetic context reads a leading-zero literal as **octal**, and 9 is not a valid octal digit. |
| 24 | **0** | numeric | `size()` returns `size_t`; `-1` converts to `SIZE_MAX`, so `3 > SIZE_MAX` is false. An all-fail marker thinking-off; the strong models crack it thinking-on. |
| 25 | **0** | numeric | `==` is left-associative: `(3 == 3)` is `1`, then `1 == 3` is false. |
| 26 | **0** | numeric | In IEEE-754 double, `0.1 + 0.2` is not `0.3`. |
| 27 | **512** | numeric | `**` is right-associative: `2 ** (3 ** 2)` = `2 ** 9`. Not `(2 ** 3) ** 2` = 64. |
| 28 | **6** | numeric | `2#101` is binary 101 = 5, plus 1. |
| 29 | **C** | mcq | A neutral atom has electrons equal to the **atomic number**, 6. The 14 is the mass number. |
| 30 | **B** | mcq | Sodium. |
| 31 | **Friday** | contains | 100 mod 7 = 2, and Wednesday plus 2 days is Friday. |
| 32 | **37** | numeric | 5 × 8 = 40, less 3 given away. The two red boxes are an irrelevant clause. |
| 33 | **8** | numeric | 12 / 3 = 4 sets at $2. The tiled floor is irrelevant. |
| 34 | **114** | numeric | 36 + 40 = 76 cupcakes; three quarters is 57; at $2 each. |
| 35 | **288** | numeric | $8 marked up to $12, less 20% is $9.60, times 30. |
| 36 | **120** | numeric | Closing speed 150 km/h; 300 / 150 = 2 hours. Answer wanted in minutes. |
| 37 | **98** | numeric | 63 (multiples of 3) + 50 (multiples of 5) − 15 (the multiple of 15 counted twice). |
| 38 | **60** | numeric | 6! / (3! 2! 1!) for three As, two Ns, one B. |
| 39 | **50.4** | numeric | 28 litres at $1.80. The matcher compares by value, so a reply of "$50.40" also passes. |
| 40 | **7.5** | numeric | At 3:15 the hour hand has moved to 97.5°, not 90°, so the hands are not aligned. |

## Tier 5 — hard (7 questions, thinking-on)

*Number theory and multi-step word problems.*

**41.** What is the sum of all three-digit positive integers that are divisible by 7?

**42.** A snail is at the bottom of a 30-foot well. Each day it climbs up 3 feet, and each night it slips back 2 feet. On which day does it first reach the top and get out?

**43.** How many trailing zeros does 100! (100 factorial) have?

**44.** There is exactly one three-digit number that equals 11 times the sum of its own digits. What is that number?

**45.** How many integers from 1 to 1000 inclusive are divisible by 3 or by 5 but not by 15?

**46.** You mix 50 litres of a 30% acid solution with 30 litres of a 70% acid solution. What is the acid concentration of the mixture, as a percentage?

**47.** What is the remainder when 13 raised to the 99th power is divided by 100? Give the number only.

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 41 | **70336** | numeric | 105 to 994, 128 terms: 128 × (105 + 994) / 2. |
| 42 | **28** | numeric | Net progress is 1 ft/day, but it climbs 3 ft *during* the day, so it clears 30 ft on day 28 before slipping. Not day 30. |
| 43 | **24** | numeric | Count factors of 5: ⌊100/5⌋ + ⌊100/25⌋ = 20 + 4. |
| 44 | **198** | numeric | 198 = 11 × 18, and 1 + 9 + 8 = 18. |
| 45 | **401** | numeric | 333 divisible by 3, 200 by 5, 66 by 15; the union is 467, less the 66 divisible by 15. |
| 46 | **45** | numeric | (50 × 0.30 + 30 × 0.70) / 80 = 36 / 80. |
| 47 | **77** | numeric | 13²⁰ ≡ 1 mod 100, and 99 mod 20 = 19, so 13¹⁹ ≡ 77. |

## Tier 6 — traps (8 questions, thinking-on)

*Misleading classics a model pattern-matches wrong even with a scratchpad: the
bat-and-ball and kilogram-of-steel family. The highest measured spread of any
tier in the file, and the cheapest to run.*

**48.** A notebook and a pen cost 220 cents in total. The notebook costs 200 cents more than the pen. How many cents does the pen cost?

**49.**
```
Sally has 3 brothers. Each of her brothers has 2 sisters. How many sisters does Sally have?
A) 0
B) 1
C) 2
D) 3
Answer with just the letter.
```

**50.**
```
A farmer is at a river with a wolf, a goat, and a cabbage, and a boat that carries the farmer plus one item. He only needs the goat on the far side and does not care what happens to the wolf or cabbage. What is the minimum number of river crossings?
A) 1
B) 3
C) 5
D) 7
Answer with just the letter.
```

**51.**
```
Which is heavier?
A) one kilogram of steel
B) one feather
C) they weigh exactly the same
Answer with just the letter.
```

**52.**
```
A cat that is already dead is sealed in a box with a vial of poison that has a 50% chance of breaking. One hour later, before the box is opened, the cat is:
A) alive
B) dead
C) in a superposition of alive and dead
Answer with just the letter.
```

**53.**
```
You are standing in London facing due west. Is Edinburgh to your left or your right?
A) left
B) right
Answer with just the letter.
```

**54.** Start with the number 10. Add 5. Then subtract 3. Then double the result. Then subtract 4. What is the final number?

**55.** Which is the only U.S. state whose name begins with two vowels? Answer with one word.

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 48 | **10** | numeric | pen + (pen + 200c) = 220c. The reflexive answer is 20. |
| 49 | **B** | mcq | Each brother has 2 sisters, so there are two girls: Sally and one other. Sally has **1** sister. |
| 50 | **A** | mcq | He needs only the goat across and does not care about the wolf or cabbage, so one crossing does it. The famous puzzle's 7 is the decoy. |
| 51 | **A** | mcq | A kilogram against a single feather. The classic pits a kilogram against a pound, and a model that recognises the shape answers "the same". |
| 52 | **B** | mcq | The cat is *already dead*. There is nothing to superpose. |
| 53 | **B** | mcq | Edinburgh is north of London; facing west, north is on your right. |
| 54 | **20** | numeric | 15, 12, 24, 20. |
| 55 | **Iowa** | contains | |

## Tier 7 — harder (5 questions, thinking-on)

*Combinatorics, sequences and digit problems.*

**56.** How many integers from 1 to 10000 inclusive are perfect squares but not perfect cubes?

**57.** How many positive integers less than 1000 have digits that sum to exactly 5?

**58.** A sequence is defined by a(1) = 3 and a(n+1) = a(n)^2 - 2. What is the value of a(4)? Give the number only.

**59.** How many diagonals does a regular dodecagon (12-sided polygon) have? Give the number only.

**60.** What is the units (last) digit of 3 raised to the power 2026? Give the number only.

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 56 | **96** | numeric | 100 perfect squares up to 10000, less the 4 that are also cubes: 1, 64, 729, 4096. |
| 57 | **21** | numeric | Non-negative digit triples summing to 5: C(7,2) = 21, and no digit can exceed 9 so nothing is excluded. |
| 58 | **2207** | numeric | 3 → 7 → 47 → 2207. |
| 59 | **54** | numeric | n(n−3)/2 = 12 × 9 / 2. |
| 60 | **9** | numeric | The units digit of 3ⁿ cycles 3, 9, 7, 1; 2026 mod 4 = 2. |

## Tier 8 — frontier (13 questions, thinking-on)

*Modular arithmetic, probability and spatial reasoning, where even a strong model
slips. This is the tier that keeps a perfect 100 out of reach on the arithmetic
side.*

**61.**
```
A marble is placed in an empty glass. The glass is turned upside down and set on a table, then picked up and put in a microwave.
Where is the marble now?
A) in the glass
B) in the microwave
C) on the table
Answer with just the letter.
```

**62.** You drive 60 miles at 30 mph, then 60 miles at 60 mph. What is your average speed for the whole 120-mile trip, in mph? Give the number only.

**63.** Three fair coins are flipped. Given that at least one comes up heads, what is the probability that all three are heads? Answer as a fraction.

**64.** A colony of bacteria triples every hour, and its jar is completely full after 12 hours. After how many hours was the jar exactly one-third full?

**65.** What are the last two digits of 7 raised to the power 2026? Give the number only.

**66.** What are the last two digits of the sum 1! + 2! + 3! + ... + 100! ? Give the number only.

**67.** What is the sum of the first 10 prime numbers?

**68.** How many positive divisors does 2025 have?

**69.** How many different ways can you make exactly 25 cents using only pennies, nickels, and dimes?

**70.** A 3x3x3 cube is painted on all six outer faces, then cut into 27 unit cubes. How many unit cubes have exactly two painted faces?

**71.** What is the remainder when 2 raised to the 100th power is divided by 1000? Give the number only.

**72.**
```
A man looking at a portrait says: "Brothers and sisters I have none, but this man's father is my father's son." Whose portrait is it?
A) his own
B) his son
C) his father
D) his nephew
Answer with just the letter.
```

**73.** What is the last digit of 7 raised to the power 7^7 (that is, 7 to the power (7 to the power 7))? Give the number only.

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 61 | **C** | mcq | Inverting the glass drops the marble onto the table; only the empty glass goes to the microwave. |
| 62 | **40** | numeric | Harmonic, not arithmetic: 2 h + 1 h = 3 h for 120 miles. The trap answer is 45. |
| 63 | **1/7** | contains | (1/8) / (7/8). Conditioning on "at least one head" removes only the all-tails outcome. |
| 64 | **11** | numeric | Tripling means one-third full is exactly one hour before full, not four hours. |
| 65 | **49** | numeric | The last two digits of 7ⁿ cycle with period 4; 2026 mod 4 = 2, so 7². |
| 66 | **13** | numeric | Every factorial from 10! upwards ends in 00, so only 1! to 9! contribute. |
| 67 | **129** | numeric | 2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29. |
| 68 | **15** | numeric | 2025 = 3⁴ × 5², so (4+1)(2+1). |
| 69 | **12** | numeric | Twelve combinations of pennies, nickels and dimes total 25c. |
| 70 | **12** | numeric | Exactly two painted faces means an edge cube, and a cube has 12 edges. |
| 71 | **376** | numeric | 2¹⁰⁰ ≡ 0 mod 8 and ≡ 1 mod 125; the Chinese remainder theorem gives 376. |
| 72 | **B** | mcq | "My father's son" is the speaker, since he has no siblings, so the portrait's father is the speaker. |
| 73 | **3** | numeric | The last digit of 7ⁿ has period 4, and 7⁷ mod 4 = 3, so 7³ = 343. |

## Tier 9 — unrecallable (12 questions, thinking-on)

*Nothing here exists in training data to retrieve. Three levers, borrowed from
BBEH but kept deliberately compact: rules defined in the prompt, so the model
must execute an unfamiliar procedure; priors that are actively wrong, so
pattern-matching is punished rather than merely unhelpful; and other people's
reasoning traces to audit, which is a different skill from producing one.*

*BBEH's length is deliberately not copied. Its tasks average roughly seven times
the output length of BBH, and its own results report models scoring below random
because they could not finish within their effective output length and started
degenerating. Against a 16k budget that failure mode truncates every worker alike
and measures endurance, not reasoning. Measured spread on this tier: 12/12 for a
Qwen3.6-27B, 8/12 for a Granite-8B, 5/12 for a Gemma-26B-A4B.*

**74.** Define the operator a # b as follows: if a is even, a # b = a/2 + b; if a is odd, a # b = 3a - b. The operator is LEFT-associative. Compute 8 # 3 # 5 # 2. Give the number only.

**75.** Define the operator a @ b = 2a - b. Unusually, this operator is RIGHT-associative, so x @ y @ z means x @ (y @ z). Compute 5 @ 3 @ 4 @ 1. Give the number only.

**76.** Define an operation on the set {0,1,2,3} by a <> b = (a + 2b) mod 4. Compute ((3 <> 1) <> (2 <> 3)) <> 3. Give the number only.

**77.** In a certain numeral system the digits are written as usual, but the place value of a digit is the FACTORIAL of its position counted from the right, starting at 1! for the rightmost digit. So a string of digits d...d3 d2 d1 has the value (d1 x 1!) + (d2 x 2!) + (d3 x 3!) + ... What is the decimal value of the string 3211 in this system? Give the number only.

**78.**
```
In a fictional language, the plural of a noun is formed by applying the FIRST of these rules that matches:
1. If the word ends in a vowel, add -ku.
2. Otherwise, if the word ends in 'n', replace that final 'n' with -mi.
3. Otherwise, if the word contains the letter 'a' more than once, double the final consonant and add -a.
4. Otherwise, add -et.
What is the plural of 'tirek'? Give the word only.
```

**79.**
```
Consider an alphabet identical to English except that the letters M and T have swapped places in the ordering: T now sorts where M normally does, and M now sorts where T normally does. Every other letter keeps its usual position. Under this ordering, which of these words comes FIRST alphabetically?
A) match
B) tiger
C) mango
D) table
E) minor
F) tulip
G) medal
H) thumb
I) mercy
J) trace
Answer with just the letter.
```

**80.**
```
In how many distinct ways can 4 people be seated around a circular table, if seatings that differ only by a rotation are considered the same AND seatings that differ only by a reflection are also considered the same?
A) 3
B) 4
C) 6
D) 8
E) 12
F) 16
G) 24
H) 2
I) 48
J) 1
Answer with just the letter.
```

**81.** A rope ladder hangs over the side of a ship that is floating freely at anchor. Its rungs are 30 cm apart. At low tide exactly 10 rungs are above the water. The tide then rises at 15 cm per hour. How many rungs are above the water 4 hours later? Give the number only.

**82.**
```
A student works out the remainder when 3^100 is divided by 7:
Step 1: 3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5, 3^6=1 (all mod 7).
Step 2: So the powers of 3 repeat with period 6.
Step 3: 100 = 6 x 16 + 4.
Step 4: So 3^100 is congruent to 3^4, which is 5 mod 7.
Step 5: The remainder is 5.
Which is the FIRST step that contains an error? Give the step number only.
```

**83.**
```
A student calculates the pH of a 0.10 mol/L solution of a weak acid with Ka = 1.0 x 10^-5:
Step 1: Set up x^2/(0.10 - x) = 1.0 x 10^-5.
Step 2: Assume x is much smaller than 0.10, giving x^2/0.10 = 1.0 x 10^-5.
Step 3: So x^2 = 1.0 x 10^-6.
Step 4: So x = 1.0 x 10^-3.
Step 5: pH = -log(1.0 x 10^-3) = 3.
Step 6: But the acid is weak, so the pH must be above 7; the answer is 7.5.
Which is the FIRST step that contains an error? Give the step number only.
```

**84.**
```
Five runners - Ana, Ben, Cara, Dev and Eli - finished a race in some order with no ties.
- Ben finished ahead of exactly two runners.
- Cara finished immediately after Dev.
- Ana finished ahead of Ben, but Ana was not first.
- Eli did not finish last.
In what position did Dev finish? Give the position as a number, where 1 means first.
```

**85.** Let X be the number of trailing zeros of 25! (25 factorial). Let Y be the remainder when 2^40 is divided by 9. What is X multiplied by Y? Give the number only.

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 74 | **10** | numeric | 8 # 3 = 7, 7 # 5 = 16, 16 # 2 = 10. Associativity is stated explicitly because it is the whole difficulty. |
| 75 | **11** | numeric | Right-associative: 4 @ 1 = 7, 3 @ 7 = −1, 5 @ (−1) = 11. Assuming the usual left-to-right gives a clean, confident 19. |
| 76 | **3** | numeric | (3<>1) = 1, (2<>3) = 0, (1<>0) = 1, then 1<>3 = 3. |
| 77 | **87** | numeric | 1×1! + 1×2! + 2×3! + 3×4! = 1 + 2 + 12 + 72. |
| 78 | **tireket** | contains | The rules must be checked in order. "tirek" ends in a consonant that is not n, and has no repeated 'a', so rule 4 applies. Rule 3's letter-count condition is what a skimming model drops. |
| 79 | **D** | mcq | With M and T swapped, every T-word now sorts before every M-word. Among the T-words, "table" leads on its second letter. Every option starts with M or T, so the swap cannot be sidestepped. |
| 80 | **A** | mcq | (4−1)! = 6 counts rotations only; identifying reflections as well halves it to 3. The famous answer is the half of the question the model does not read. |
| 81 | **10** | numeric | A freely floating ship rises with the tide, so the rung count never changes. The arithmetic the prompt invites (60 cm risen / 30 cm per rung = 2 submerged) is the wrong move. |
| 82 | **4** | numeric | Steps 1 to 3 are correct. 3⁴ ≡ 4 mod 7, not 5, so step 4 is the first error — and step 5 follows consistently from it, so a model that only checks the conclusion names the wrong step. |
| 83 | **6** | numeric | The chemistry through step 5 is right. Step 6 invents a rule that a weak acid's solution must be basic. |
| 84 | **4** | numeric | Ben is third and Ana second, so Dev and Cara take fourth and fifth, and Eli (not last) takes first. |
| 85 | **42** | numeric | 25! has 6 trailing zeros, and 2⁴⁰ ≡ 7 mod 9, so 6 × 7. Two independent sub-problems fused, so an error in either half destroys the result and partial credit is impossible. |

## Tier 10 — ceiling (12 questions, thinking-on)

*SimpleBench-style world-model traps: fresh everyday scenarios in which an
arithmetic or pattern-matched answer is a decoy and the real answer turns on
physical, temporal or social common sense. Unlike tier 9, a reasoning budget does
not help — the model either simulates the situation or it does not. Frontier
models reach roughly 76-82% on SimpleBench against a human baseline of about 84%,
so the family still has headroom at the very top.*

*Question 93 is deliberately a **non-trap**: a tray of ice cubes in a working
freezer is still twelve ice cubes. Without it, a model could score well on this
tier by learning "answer zero" or "answer the counterintuitive option", which
would measure nothing.*

*This is the most subjective style in the file. Every item is written so the
intended answer is forced by an explicit clause ("stirs it thoroughly", "the
freezer runs normally", "the bridge comfortably holds all three"). If a strong
model misses one, read its reasoning before assuming the model is wrong — the
question may be, and that is a real failure mode here in a way it never was for
arithmetic.*

*Question 89 is the worked example of that, and of how strict "forced by an
explicit clause" has to be. It used to leave the lift's speed unstated while
offering "it depends on the lift's speed" as an option, so that option was
defensible; three of the four benchmarked workers then spent their entire budget
on it rather than answering, one truncating at 16384 thinking tokens and two
blowing the frontier deadline. Pinning only the lift was not enough either — the
27B stopped truncating but still answered "it cannot be determined" after 6907
thinking tokens, because the stairs were still a judgement call. An item here is
forced only when no quantity is left to judgement, which is what the clean items
do: both beanbags are on the ground, and the candles' burn time and lighting
times are all given.*

**86.**
```
Every minute for five minutes, Ravi drops two sugar cubes into a mug of freshly boiled tea and stirs it thoroughly each time. At the end of the five minutes, how many whole, undissolved sugar cubes are in the mug?
A) 10
B) 8
C) 5
D) 2
E) 0
F) 4
G) 6
H) 12
I) 1
J) 20
Answer with just the letter.
```

**87.**
```
A street performer standing in the middle of an empty, flat car park throws a red beanbag 2 metres straight up, then immediately throws a blue beanbag 4 metres straight up. He then walks away and does not touch either beanbag again. Fifteen minutes later, which beanbag is higher above the ground?
A) the red one
B) the blue one
C) the red one, by 2 metres
D) the blue one, by 2 metres
E) the blue one, by 4 metres
F) they are at the same height
G) the red one, by 4 metres
H) it depends on their masses
I) it depends on air resistance
J) there is not enough information
Answer with just the letter.
```

**88.**
```
You reach a fork in a corridor where two attendants are standing. Both attendants always tell the truth. A large illuminated sign above the left-hand passage reads "EXIT - THIS WAY". What is the smallest number of questions you must ask the attendants in order to leave?
A) 1
B) 2
C) 3
D) 0
E) 4
F) 5
G) 6
H) 7
I) 8
J) it is impossible to know
Answer with just the letter.
```

**89.**
```
Three colleagues leave the ground floor of a 30-storey office tower at the same moment, all heading for the roof terrace. Priya takes the express lift, which travels non-stop and reaches the roof terrace in under a minute. Dan takes the stairs, running two at a time, which takes him a little over five minutes. Marcus rides the same express lift as Priya, carrying a full tray of coffees, and is 94 years old. Who reaches the roof terrace last?
A) Priya
B) Marcus
C) Dan
D) Priya and Marcus, together
E) Dan and Marcus, together
F) all three arrive together
G) Priya and Dan, together
H) Marcus, because of his age
I) it cannot be determined
J) nobody reaches the roof terrace
Answer with just the letter.
```

**90.**
```
At the start of every hour, for four hours, Nadia places three fresh birthday candles on a cake and lights them. Each candle burns for about twenty minutes before going out. At the end of the four hours, how many candles on the cake are still alight?
A) 12
B) 9
C) 6
D) 3
E) 2
F) 1
G) 0
H) 4
I) 15
J) 20
Answer with just the letter.
```

**91.**
```
Rosa must choose someone to carry a tray of full wine glasses across a crowded room without spilling any. Tom is 22, a champion sprinter, and has drunk four pints of beer. Wei is 61, walks slowly and carefully, and has drunk only water. Ben is 30, very strong, and has drunk three glasses of wine. Who is most likely to succeed?
A) Wei
B) Tom
C) Ben
D) Tom, because he is the fastest
E) Ben, because he is the strongest
F) all three are equally likely
G) either Tom or Ben
H) it cannot be determined
I) none of them could manage it
J) whichever of them is tallest
Answer with just the letter.
```

**92.**
```
Three hikers need to cross a bridge at night. The bridge comfortably holds all three at once, and a floodlight is permanently mounted on it, so no torch is needed. Walking at their own paces they would take 1, 2 and 5 minutes respectively, and they can walk side by side. What is the minimum total time for all three to get across?
A) 1
B) 2
C) 3
D) 8
E) 5
F) 6
G) 7
H) 9
I) 10
J) 17
Answer with just the letter.
```

**93.**
```
Kofi puts a tray of twelve ice cubes into his freezer at 9am. The freezer runs normally all day and nobody opens it. At 3pm he opens the freezer. How many ice cubes are in the tray?
A) 0
B) 1
C) 2
D) 3
E) 6
F) 9
G) 11
H) 12
I) 13
J) 24
Answer with just the letter.
```

**94.**
```
A shop assistant counts 40 apples into a crate at 8am. During the day she adds 15 more to the crate and sells 22 from it. Separately, a display basket by the till has held 6 apples all day, untouched. At 6pm a driver takes the entire crate away to another branch. How many apples are in the shop at 7pm?
A) 33
B) 39
C) 0
D) 6
E) 22
F) 40
G) 15
H) 27
I) 55
J) 11
Answer with just the letter.
```

**95.**
```
A child sits in a moving car holding the string of a helium balloon that floats freely inside the cabin. All the windows are shut. The driver brakes sharply. Which way does the balloon move relative to the inside of the car?
A) forward, the same way the passengers lurch
B) backward, towards the rear of the car
C) it does not move relative to the car
D) straight up
E) straight down
F) to the left
G) to the right
H) it depends on the car's speed
I) it depends on the balloon's size
J) forward first, then backward
Answer with just the letter.
```

**96.**
```
A flight leaves Auckland at 11pm on Monday and lands in Sydney four hours later. At that time of year Sydney's local time is two hours behind Auckland's. What is the local day and time in Sydney when the flight lands?
A) 1am Monday
B) 3am Tuesday
C) 1am Tuesday
D) 5am Tuesday
E) 9pm Monday
F) 11pm Monday
G) 3am Monday
H) 1pm Tuesday
I) 5am Monday
J) 11am Tuesday
Answer with just the letter.
```

**97.**
```
Amara's friend cancels their dinner plans by text an hour beforehand, for the third time this month. Amara replies: "Fantastic. I absolutely love it when people do this to me." How does Amara most likely feel?
A) delighted
B) grateful
C) indifferent
D) relieved
E) confused
F) excited
G) proud
H) amused
I) surprised
J) annoyed
Answer with just the letter.
```

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 86 | **E** | mcq | Sugar dissolves in freshly boiled tea, and every addition is stirred thoroughly. The 2 × 5 = 10 is the decoy. |
| 87 | **F** | mcq | Both beanbags were thrown fifteen minutes ago and both landed. The throw heights are irrelevant. |
| 88 | **D** | mcq | Both attendants tell the truth and a sign already states which way the exit is, so no question is needed. The setup is the classic two-guards riddle with its difficulty removed. |
| 89 | **C** | mcq | Five minutes of stairs lose to a lift that reaches the roof in under a minute. Marcus's age and the coffee tray are distractions; he is in the lift, so he arrives when Priya does. Revised in v36: with either journey time left unstated the question was arguable, and it was eating whole reasoning budgets. It is now a guard rather than a discriminator — every worker passes it. |
| 90 | **G** | mcq | Each set burns out after about twenty minutes, and the last set was lit an hour before the end. None is still alight. |
| 91 | **A** | mcq | Wei is the only one who has not been drinking. Speed and strength are the decoys. |
| 92 | **E** | mcq | The bridge holds all three, it is floodlit, and they can walk side by side, so the time is the slowest walker's. The famous torch-relay answer of 8 does not apply. |
| 93 | **H** | mcq | The tier's guard. A working freezer keeps ice frozen: twelve in, twelve out. |
| 94 | **D** | mcq | The crate leaves the shop at 6pm, so only the untouched display basket remains. All the crate arithmetic is a decoy. |
| 95 | **B** | mcq | Under braking the denser cabin air surges forward, so the buoyant balloon moves backwards, opposite to the passengers. |
| 96 | **C** | mcq | 11pm Monday plus four hours is 3am Tuesday in Auckland; Sydney is two hours behind, so 1am Tuesday. The day still rolls over. |
| 97 | **J** | mcq | The reply is sarcastic. Reading it literally is the trap. |

## Tier 11 — budget-bounded insight (5 questions, thinking-on)

*The insight bucket: 20 of the 100 points, so each of these five questions is
worth 4 points, more than six times what a base-bucket question is worth.
Enumeration problems with a hidden closed-form shortcut. A model that sees the
shortcut answers in about 1000 tokens; a model that grinds the list out dies at
the 16k cap with no answer at all, which the grader scores as a failure. This is
the first tier that separated a 27B from a 284B.*

*Four of the five are digit or base enumeration, which is one trick family. It
was chosen because it was the only family of seven tried that spread that pair.
Perturbed classics (ignorant-host Monty Hall, January-restricted birthday
paradox), deeper state tracking, compositional fusion, constraint puzzles and
fresh world-model items all produced zero spread — 2026 models have absorbed that
playbook. Fresh epistemic sum/product puzzles turned out to have no unique
solution outside the memorised classic. When a future model learns the bijection
reflex, this tier saturates like the ten before it and needs the next lever.*

*All items are original, and the answers were brute-force verified by script
rather than by hand.*

**98.** Consider the positive integers that contain no digit 9 in ordinary base-10 notation, listed in increasing order: 1, 2, ..., 8, 10, 11, ... What is the 1000th number in this list? Give the number only.

**99.** Consider positive integers whose base-7 representation contains no digit 3. List them in increasing order: 1, 2, 4, 5, 6, 11(base 7)=8, ... What is the 100th such integer, expressed in ordinary base 10? Give the number only.

**100.** Consider the positive integers whose base-8 (octal) representation contains no digit 5, listed in increasing order. What is the 300th such integer, expressed in ordinary base 10? Give the number only.

**101.** Consider the positive integers whose base-7 representation contains no digit 3, listed in increasing order. What is the SUM of the first 40 such integers (the sum expressed in ordinary base 10)? Give the number only.

**102.** How many positive integers less than 1,000,000 have digits whose PRODUCT is exactly 96? Give the number only.

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 98 | **1331** | numeric | The no-9 integers are order-isomorphic to counting in base 9, so the answer is 1000 written in base 9. This is the most famous variant and is kept as the tier's guard: a 27B found the bijection here and burned out on the other three. |
| 99 | **138** | numeric | Same bijection with a banned interior digit: 100 in base 6 is 244, mapped digit-wise onto {0,1,2,4,5,6} it reads 255 in base 7, which is 138. |
| 100 | **455** | numeric | 300 in base 7 is 606, mapped onto the no-5 octal digits it reads 707 octal, which is 455. |
| 101 | **1121** | numeric | The sum of the first 40 terms of the sequence in question 99. The bijection produces each term directly; scanning for them does not fit the budget. |
| 102 | **1462** | numeric | 96 = 2⁵ × 3, so the digits are drawn from {2,3,4,6,8} with 1s padding to six places. The one item in this tier without the base bijection. |

## Tier 12 — programming (28 questions, thinking-on)

*The coding bucket: 20 of the 100 points, shared pro rata, so each question is
worth about 0.71 points. The only tier that asks whether a worker can read code
exactly rather than solve a puzzle. Every question is a **trace**: a short program
with one counter-intuitive interaction, answered by its exact output. The answer
is a fact about the language, so the grader compares it exactly and nothing has
to be executed at profiling time. This gap does not close by itself, because
`benchgen.go` deliberately excludes LiveBench's `coding` and `agentic_coding`
categories for needing execution.*

*Every item is abstracted from a real bug: a commit from two months of
production work across a Go router, an agent platform, a deploy tool and its bash
templates, a Python portal and a Kotlin app, mined for the shape of the trap
rather than the code. Nothing here names a real host, repository or service. The
recurring class across all of it, and the one this tier keeps hitting: absent or
unknown is a distinct third state from negative.*

*Calibration. 95 candidates were graded against 4 live workers spanning q59 to
q94 and cut by item analysis (D = top-half pass rate minus bottom-half, the same
statistic `benchgen_emit.go` uses). 28 survived with D > 0. Every answer key was
verified by executing the program, which is how the sign fault in `benchNumberRe`
was found. Each row below carries its measured p (pass rate) and D.*

*Noise, stated honestly: 6 of 33 re-answered cells, 18%, flipped verdict at
temperature 0. With two workers per half, one flip moves D by 0.50, so the
p=0.50 D=+1.00 items are the trustworthy ones and the rest are one flip from
D=0. Rows whose verdict was directly observed to flip say so.*

*Multiple choice was tried here and cut. 47 MCQ items were authored alongside
these and all 47 were dropped: in every one the correct option was the longest,
because the answer had been written with its full justification and the
distractors as one-liners. An 8B with no thinking mode scored 79% on them against
45% on the traces, and the q94 worker scored 100%, giving 21 points of spread
against the traces' 51. They were measuring option length. If MCQ is revisited
here, length-match every option first and re-calibrate.*

**103.**
```
This Go program prints one number. What is it?

package main

import "fmt"

func f() int {
	n := 0
	for i := 0; i < 3; i++ {
		defer func() { n++ }()
	}
	return n
}

func main() { fmt.Println(f()) }

Give the number only.
```

**104.**
```
What does this bash script print?

f() { local x=$(false); echo "$?"; }
f

Give the number only.
```

**105.**
```
This Go program prints one number. What is it?

package main

import "fmt"

type MyErr struct{}

func (e *MyErr) Error() string { return "boom" }

func mayFail(ok bool) error {
	var p *MyErr
	if !ok {
		p = &MyErr{}
	}
	return p
}

func main() {
	n := 0
	if mayFail(true) != nil {
		n++
	}
	fmt.Println(n)
}

Give the number only.
```

**106.**
```
How many lines does this bash script print in total?

set -e
f() { false; echo reached; }
if f; then echo yes; else echo no; fi
echo end

Give the number only.
```

**107.**
```
This Go program prints one number. What is it?

package main

import "fmt"

func main() {
	var x int32 = 300
	fmt.Println(int8(x))
}

Give the number only.
```

**108.**
```
What does this bash script print?

set -o pipefail
seq 1 200000 | head -1 > /dev/null
echo $?

Give the number only.
```

**109.**
```
This Go program prints one number. What is it?

package main

import "fmt"

func main() {
	a := make([]int, 3, 5)
	a[0], a[1], a[2] = 1, 2, 3
	_ = append(a[:2], 9)
	fmt.Println(a[2])
}

Give the number only.
```

**110.**
```
In Python 3, what happens when this runs? Answer in one short phrase.

class A:
    def __eq__(self, other):
        return True

print(len({A(), A()}))
```

**111.**
```
What does this bash script print?

n=$(printf 'x\ny\n' | grep -c 'ZZZ' || echo 0)
echo "${#n}"

Give the number only.
```

**112.**
```
In Python 3, this class body raises NameError. Which LINE raises it? Count the "class C:" line as line 1.

class C:
    xs = [1, 2, 3]
    ys = [x * 2 for x in xs]
    ws = [y for y in range(3) if y in xs]

Give the line number only.
```

**113.**
```
In Python 3, running this raises an exception. Name the exception type exactly (one word).

def f():
    try:
        1 / 0
    except Exception as e:
        pass
    return e

f()
```

**114.**
```
What does this bash script print?

printf 'a\t\tc\n' | while IFS=$'\t' read -r x y z; do echo "${#x}${#y}${#z}"; done

Give only the output.
```

**115.**
```
In Python, this program prints one number. What is it?

n = float('nan')
values = [n]
count = 0
if n in values: count += 1
if float('nan') in values: count += 1
print(count)

Give the number only.
```

**116.**
```
This Go program prints one number. What is it?

package main

import "fmt"

func main() {
	s := "\u65e5\u672c"
	fmt.Println(len(s) + len([]rune(s)))
}

Give the number only.
```

**117.**
```
This bash script prints one number. What is it?

v=""
n=0
if [ -n $v ]; then n=$((n+1)); fi
if [ -n "$v" ]; then n=$((n+1)); fi
echo "$n"

Give the number only.
```

**118.**
```
This Go program prints one number. What is it?

package main

import ("fmt"; "strings")

func main() {
	fmt.Println(len(strings.TrimLeft("filename.tar", "fil")))
}

Give the number only.
```

**119.**
```
In Python 3, this raises an exception. Quote the exception MESSAGE exactly (not the type).

def g():
    try:
        yield 1
    finally:
        yield 2

it = g()
next(it)
it.close()
```

**120.**
```
This Go program prints one number. What is it?

package main

import "fmt"

func main() {
	s := "h\u00e9llo"
	n := 0
	for range s {
		n++
	}
	fmt.Println(len(s) + n)
}

Give the number only.
```

**121.**
```
In Python, what does this print?

import os
print(len(os.path.join("/var/data/uploads", "/etc/passwd")))

Give the number only.
```

**122.**
```
In Python, this program prints one number. What is it?

grid = [[]] * 3
grid[0].append(1)
print(sum(len(row) for row in grid))

Give the number only.
```

**123.**
```
In Python 3, what does this print?

print(len({1: 'a', True: 'b', 1.0: 'c'}))

Give the number only.
```

**124.**
```
In bash, directory src/ contains one file f, and an empty directory dst/ already exists. After running:

cp -a src dst

what is the full path of file f's copy? Give only the path.
```

**125.**
```
In Python, what does this print?

names = ["part2", "part10", "part1"]
print(sorted(names).index("part2"))

Give the number only.
```

**126.**
```
What does this bash script print?

n=0
printf 'a\nb\nc\n' | while read -r l; do n=$((n+1)); done
echo "$n"

Give the number only.
```

**127.**
```
In Python 3, what does this print?

print(int(1 in [1] == True))

Give the number only.
```

**128.**
```
This Go program prints one number. What is it?

package main

import "fmt"

func main() {
	s := []int{1, 2, 3, 4}
	t := s[1:3]
	_ = append(t, 99)
	fmt.Println(s[3])
}

Give the number only.
```

**129.**
```
What does this bash command print?

printf '%d\n' 010

Give the number only.
```

**130.**
```
In Python 3, what does this print?

print((-5 // 2) + (-5 % 2) + 10)

Give the number only.
```

| # | Answer | Match | Why |
|---|--------|-------|-----|
| 103 | **0** | numeric | A `defer` runs after the return value has been copied, so incrementing a local that is not a named result changes nothing the caller sees. p=0.50, D=+1.00. |
| 104 | **0** | numeric | `local x=$(false)` declares and assigns on one line, so `$?` is the exit status of `local`, not of the command substitution. The failure is hidden. p=0.50, D=+1.00. |
| 105 | **1** | numeric | A nil `*MyErr` stored in an `error` interface is non-nil: the interface carries a type. `mayFail(true)` returns a typed nil, the check fires, and the happy path reports failure. p=0.50, D=+1.00. |
| 106 | **3** | numeric | `errexit` is suspended inside a function used as an `if` condition, so `f` runs past the failing `false` and prints `reached`. Three lines: `reached`, `yes`, `end`. p=0.50, D=+1.00. |
| 107 | **44** | numeric | A narrowing conversion wraps rather than clamping, and Go issues no warning. 300 mod 256 is 44. p=0.50, D=+1.00. |
| 108 | **141** | numeric | `head -1` exits after one line, so `seq` takes SIGPIPE. `pipefail` reports the pipeline as 128 + 13. p=0.50, D=+1.00. |
| 109 | **9** | numeric | `a` has length 3 and capacity 5, so appending to `a[:2]` writes into the shared backing array at index 2 and overwrites the 3 that was there. p=0.50, D=+1.00. |
| 110 | **unhashable** | contains | Defining `__eq__` without `__hash__` sets `__hash__` to `None`, so the instances cannot go in a set at all. Graded `contains` on the word in the TypeError, not on a count. p=0.50, D=+1.00. Verdict observed to flip at temperature 0. |
| 111 | **3** | numeric | `grep -c` prints `0` and exits 1 when nothing matches, so the `|| echo 0` fallback fires as well and `n` is the two-line string `0\n0`, which is 3 characters. p=0.50, D=+1.00. |
| 112 | **4** | numeric | A comprehension's first iterable is evaluated in the enclosing scope, so line 3 sees `xs`. The body and the conditions get their own scope, which cannot see class scope, so the `if y in xs` on line 4 raises. p=0.25, D=+0.50. |
| 113 | **UnboundLocalError** | contains | The `except ... as e` name is deleted when the block ends. Inside a function that makes `e` an unbound local, so the exception is `UnboundLocalError`, not `NameError`. p=0.25, D=+0.50. |
| 114 | **110** | numeric | `IFS=$'\t'` is still IFS whitespace, so the two consecutive tabs collapse into one separator. The legitimately empty middle field vanishes and every later field shifts left: `x`="a", `y`="c", `z`="". p=0.25, D=+0.50. |
| 115 | **1** | numeric | `in` tests identity before equality, so the NaN already in the list matches itself, while a freshly built NaN is a different object and matches nothing. p=0.75, D=+0.50. |
| 116 | **8** | numeric | `len` on a string counts bytes and `[]rune` counts characters. Two CJK characters are 6 bytes and 2 runes. p=0.75, D=+0.50. |
| 117 | **1** | numeric | Unquoted, the empty `$v` disappears by word splitting, leaving `[ -n ]`, a one-argument test that is true because the argument `-n` is non-empty. Quoted, it is correctly false. p=0.75, D=+0.50. |
| 118 | **9** | numeric | `TrimLeft` takes a cutset, not a prefix, so it strips every leading `f`, `i` or `l` and removes `fil`, leaving the 9-character `ename.tar`. Go's mirror of the `str.strip` trap. p=0.75, D=+0.50. Verdict observed to flip at temperature 0. |
| 119 | **ignored GeneratorExit** | contains | Yielding from a `finally` during `close()` refuses the `GeneratorExit`, which Python reports as `RuntimeError: generator ignored GeneratorExit`. Cleanup that yields is not cleanup. p=0.75, D=+0.50. |
| 120 | **11** | numeric | `len` counts bytes and `range` counts runes. `h\u00e9llo` is 6 bytes and 5 runes. Any offset arithmetic that mixes the two corrupts non-ASCII input. p=0.75, D=+0.50. |
| 121 | **11** | numeric | `os.path.join` discards everything before an absolute component, so the result is `/etc/passwd`, 11 characters. A base directory prefix is not a sandbox. p=0.75, D=+0.50. |
| 122 | **3** | numeric | `[[]] * 3` copies the reference three times, not the list, so all three rows are the same object and appending once makes every row length 1. p=0.75, D=+0.50. |
| 123 | **1** | numeric | `True`, `1` and `1.0` hash equal and compare equal, so all three are one key and the later values overwrite the earlier ones. p=0.75, D=+0.50. |
| 124 | **dst/src/f** | contains | `cp -a SRC DST` copies INTO `DST` when `DST` already exists, so the file lands at `dst/src/f` rather than `dst/f`. Graded `contains`. p=0.75, D=+0.50. |
| 125 | **2** | numeric | Sorting is lexicographic, so `part1`, `part10`, `part2` and the index of `part2` is 2. The bug only appears once there are ten or more, which is why it ships. p=0.75, D=+0.50. |
| 126 | **0** | numeric | The right-hand side of a pipeline runs in a subshell, so the increments are lost and the outer `n` is still 0. p=0.75, D=+0.50. |
| 127 | **0** | numeric | Comparison operators chain: `1 in [1] == True` means `(1 in [1]) and ([1] == True)`. The second half is false, so the whole thing is false and `int(...)` is 0. p=0.75, D=+0.50. Verdict observed to flip at temperature 0. |
| 128 | **99** | numeric | `t := s[1:3]` shares the backing array, so appending to `t` writes at index 3 of `s` and replaces the 4 with 99. p=0.75, D=+0.50. |
| 129 | **8** | numeric | `printf %d` reads a leading zero as octal, so `010` is 8. A zero-padded counter or date field silently changes value. p=0.75, D=+0.50. Verdict observed to flip at temperature 0. |
| 130 | **8** | numeric | Python floors toward negative infinity and its modulo takes the divisor's sign, unlike C, Go and Java: `-5 // 2` is -3 and `-5 % 2` is 1. The `+ 10` is the author keeping the expected answer positive; see the note on the sign fix under `numeric` above. p=0.75, D=+0.50. |

---

## Contamination

Publishing this file exposes it to scrapers, and eventually the questions and
their answers will appear in some model's training data. A model that has
memorised the answer key scores high without demonstrating anything. That is a
real cost, and it is accepted knowingly: a quality score whose questions are
secret cannot be audited, argued with, or reproduced, and this score decides
routing.

The mitigation is refresh, not secrecy. The router carries a three-phase
generator (`bench fetch`, `bench calibrate`, `bench emit`) that pulls fresh
candidate questions from LiveBench, grades them against the live fleet with the
exact production grader, and selects the ones that actually discriminate. Each
refresh is a commit that bumps `benchmarkVersion` alongside the new questions,
which invalidates every cached profile and re-measures the whole fleet against
questions the models have not seen. LiveBench replaces about a sixth of its own
questions monthly, with a full turnover roughly every six months, for exactly
this reason.

Two honest limits on that mitigation.

First, it only covers the sourced arithmetic tiers. LiveBench cannot supply the
trap, unrecallable and world-model tiers (6, 9, 10 and 11), because their whole
value is being absent from any training corpus. Nor can it supply tier 12:
`benchgen.go` excludes LiveBench's `coding` and `agentic_coding` categories
because grading them needs execution, which the router does not do at profiling
time. That is 65 of the 130 questions the generator can never refresh, and they
are the tiers with the best measured spread, so they are the ones publication
damages most. Refreshing them means authoring new ones by hand.

Second, saturation arrives on its own schedule regardless of contamination. Tier
9 was added because a 27B scored 100% on the tier before it; tier 10 was added
because the same model then aced tier 9; tier 11 was added because it aced tier
10 as well, and two other tiers were built, measured and deleted along the way.
Contamination accelerates that treadmill. It did not start it.
