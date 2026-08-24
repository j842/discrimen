package router

// Automatic prompt classification + latency-aware ordering.
//
// A training-free embedding-centroid classifier scores each prompt on two axes
// from a single embedding:
//
//   - difficulty (simple ↔ hard)  → drives model-tier selection (Phase 1)
//   - reasoning  (direct ↔ reasoning) → drives whether to enable the model's
//     thinking mode (Phase 3a "adaptive reasoning")
//
// Labelled seed prompts are embedded once (through the registered embeddings
// worker) to form four centroids; an incoming prompt is scored by its cosine
// margin between each pair. Both axes are derived from one embedding and cached
// together. Everything is best-effort: if the embeddings worker is unreachable
// or classification fails, callers fall through to their non-classified
// behaviour — no request is ever blocked or degraded by this path.
//
// This file also holds the Phase 2 latency-aware ordering helpers.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

// ── Tunables ────────────────────────────────────────────────────────────────

const (
	defaultDifficultyBands = "0.40:2,0.70:7,1.0:9"
	// bootstrapCooldown bounds how often we retry centroid bootstrap when the
	// embeddings worker is down, so a misconfigured cluster isn't hammered.
	bootstrapCooldown = 30 * time.Second
	// embedFailCooldown bounds what a reachable-but-wedged embeddings worker can
	// cost once the classifier is ready: after a live-embed failure, classify
	// skips embedding (ok=false → default routing) until the cooldown elapses,
	// instead of paying the embed timeout up to twice per request. A successful
	// embed clears it.
	embedFailCooldown = 20 * time.Second
)

// latencyEstTokens is the nominal generated-token count for a request WITHOUT a
// thinking phase. Its absolute value used to barely matter (it scaled every
// backend equally), but once prefill entered the estimate the decode and prefill
// terms trade off against each other, so the figure has to be roughly right.
var latencyEstTokens = 256

// latencyEstThinkTokens is the nominal generated-token count for a request the
// router has enabled thinking on. Thinking turns measured on this fleet ran
// 1450–2000 tokens against ~250 for a direct answer, so estimating every job at
// latencyEstTokens understated decode cost ~6x on exactly the requests where
// decode dominates the wall clock.
var latencyEstThinkTokens = 1500

// uncappedNominalSlots is the concurrency assumed for a backend that registered
// max_concurrency=0 (unbounded), so its queue occupancy can still be estimated.
var uncappedNominalSlots = 4

// deadlineSafetyFactor is the fraction of the caller's remaining budget a job's
// estimate must fit inside to count as placeable. Below 1 because the estimate is
// approximate and a job finishing at exactly the deadline is still a failure.
var deadlineSafetyFactor = 0.8

// jobCost is the shape of the request being placed. expectedLatency needs it
// because both phases scale with the request: prefill with the prompt (tool
// schemas and an agent system prompt are thousands of tokens) and decode with the
// expected output (a thinking turn is ~6x a direct answer). Ranking on a fixed
// nominal job made a 4k-token thinking request look like a 256-token chat turn on
// every worker, which is how an agent turn ended up on a CPU worker that needed
// >120s for it while an idle GPU was ranked below it.
type jobCost struct {
	promptTokens int
	outputTokens int
	// incumbent is the worker that served the previous turn of this conversation,
	// if any (see session.go). It rides in the job rather than in a separate
	// argument so every existing caller of expectedLatency — the ranker AND the
	// deadline filter — accounts for the discount consistently; a filter that
	// priced the incumbent as a cold worker could drop the very candidate the
	// ranker was about to prefer.
	incumbent string
}

// nominalJob is the request-independent fallback for callers with no request in
// hand (the pinned/debug paths and tests). It reproduces the old behaviour.
func nominalJob() jobCost {
	return jobCost{outputTokens: latencyEstTokens}
}

// withIncumbent returns the job priced for a conversation whose previous turn was
// served by backendID.
func (j jobCost) withIncumbent(backendID string) jobCost {
	j.incumbent = backendID
	return j
}

// costForRequest derives the job shape from the request and the thinking decision
// the router has already resolved for it.
func costForRequest(req *ChatRequest, thinking bool) jobCost {
	out := latencyEstTokens
	if thinking {
		out = latencyEstThinkTokens
	}
	// A client budget BELOW the nominal figure is a real ceiling and shortens the
	// job. Above it, the nominal estimate is the better predictor: max_tokens is
	// usually a generous cap rather than a forecast (the agent turn that timed out
	// declared 8192 and would have produced ~1700).
	if req != nil && req.MaxTokens > 0 && req.MaxTokens < out {
		out = req.MaxTokens
	}
	return jobCost{promptTokens: promptTokensFor(req), outputTokens: out}
}

// contextAnswerReserve is the output room the HARD context filter charges when
// the client declared a bigger ceiling than that. It is not a cap on the answer
// (patchForwardedBody forwards the client's real budget, trimmed only to fit the
// worker) — only on how much room a worker must prove it has spare before it is
// allowed to be a candidate.
var contextAnswerReserve = 8192

// contextReserveTokens is how much OUTPUT the hard context filter charges a
// request. max_tokens is a generous ceiling rather than a forecast — costForRequest
// already makes exactly this call for the latency estimate — and charging it as a
// hard requirement is what a coding harness trips over: pi declares 131072 by
// default, which added a flat 128K to every prompt's context estimate, filtered
// every worker under 128K out of even a one-word request, and then let
// autoTargetQuality derive its tier from the two survivors. Measured 2026-08-09:
// the prompt "say hi" routed to the 284B at 22 tok/s with q>=98, and to a 26B at
// 64 tok/s with q>=71 once the declared ceiling was removed. Past ~128K of real
// context it stopped routing at all — no worker could satisfy prompt + ceiling —
// so the session 503'd long before the client's own compaction would have fired.
//
// Charge the smaller of the client's ceiling and a nominal answer. A worker that
// then can't fit the full ceiling gets it trimmed in patchForwardedBody rather
// than being excluded here.
func contextReserveTokens(req *ChatRequest) int {
	if req == nil {
		return contextAnswerReserve
	}
	if req.MaxTokens > 0 && req.MaxTokens < contextAnswerReserve {
		return req.MaxTokens
	}
	return contextAnswerReserve
}

// promptTokensFor estimates the prefill size of a request, tool schemas included.
// Uses the same conservative chars/3 divisor as estimateContextK — JSON-heavy
// tool definitions tokenize nearer 3 chars/token than 4.
func promptTokensFor(req *ChatRequest) int {
	if req == nil {
		return 0
	}
	chars := 0
	for _, m := range req.Messages {
		chars += estimateContentChars(m.Content)
	}
	if n := len(req.Tools); n > 0 && string(req.Tools) != "null" {
		chars += n
	}
	return chars / 3
}

// ── Seed prompts ────────────────────────────────────────────────────────────
//
// The difficulty pair (simple/hard) anchors the model-tier axis; the reasoning
// pair (direct/reasoning) anchors the thinking-mode axis. They are deliberately
// distinct: the direct seeds include long-but-shallow tasks (which are not
// "simple" but need no reasoning), and the reasoning seeds include short-but-
// tricky ones (which look easy but need step-by-step thought). All are embedded
// once at first use; the centroids adapt to whatever embedding model is served.

var simpleSeeds = []string{
	"hello",
	"what time is it?",
	"translate \"good morning\" into French",
	"what is the capital of France?",
	"summarise this paragraph in one sentence",
	"classify the sentiment of this review as positive or negative",
	"extract the email address from this text",
	"what's 2 + 2?",
	"convert 10 kilometres to miles",
	"give me a synonym for \"happy\"",
	"fix the spelling in this sentence",
	"what day of the week is 2026-01-01?",
	"list three fruits",
	"tag this message with a category",
	"reformat this date as YYYY-MM-DD",
	"is this sentence a question? answer yes or no",
	"capitalise the first letter of each word",
	"what does the abbreviation ASAP stand for?",
	"rewrite this sentence more politely",
	"count the words in this paragraph",
}

var hardSeeds = []string{
	"prove that there are infinitely many prime numbers",
	"debug this stack trace and explain the root cause",
	"design a horizontally scalable rate limiter for a distributed system",
	"derive the time complexity of this recursive algorithm step by step",
	"refactor this module to remove the circular dependency without breaking the public API",
	"analyse the trade-offs between optimistic and pessimistic concurrency control for this workload",
	"write a proof of correctness for this concurrent queue implementation",
	"plan a zero-downtime migration from a monolith to microservices",
	"explain why this deadlock occurs and propose three different fixes with trade-offs",
	"optimise this SQL query that does a full table scan over 100 million rows",
	"reason through the edge cases of this date-parsing function across time zones",
	"compare the convergence guarantees of Adam and SGD with momentum on non-convex objectives",
	"design a consensus protocol that tolerates Byzantine faults and justify each message round",
	"work through this multi-step word problem and show every intermediate calculation",
	"identify the security vulnerability in this authentication flow and explain the exploit",
	"architect a fault-tolerant event pipeline with exactly-once delivery semantics",
	"explain the memory-ordering bug in this lock-free code and how to fix it",
	"evaluate whether this change is backwards compatible and enumerate every breaking case",
	"write and explain a dynamic-programming solution to this constrained optimisation problem",
	"diagnose why throughput collapses under load and propose a remediation plan",
}

// reasoningSeeds need genuine multi-step thought, independent of length.
var reasoningSeeds = []string{
	"if all bloops are razzies and some razzies are lazzies, are some bloops definitely lazzies?",
	"a bat and a ball cost $1.10 together and the bat costs $1.00 more than the ball; how much is the ball?",
	"work through this step by step and show your reasoning before answering",
	"prove that the square root of 2 is irrational",
	"debug why this function returns the wrong value on the second call",
	"what is the root cause of this intermittent failure?",
	"derive the recurrence relation and solve it for a closed form",
	"plan the order of operations to migrate this database with no downtime",
	"compare these two designs and justify which is better under high load",
	"explain why this lock-free code has a data race and how to fix it",
	"which is heavier, a kilogram of steel or a kilogram of feathers, and why?",
	"solve this puzzle and explain each deduction you make",
	"reason about the edge cases before you give an answer",
	"what conclusion follows logically from these premises?",
	"trace the execution and find exactly where the invariant breaks",
	"given these constraints, find an assignment that satisfies all of them",
	"is this argument valid, or does it contain a logical fallacy?",
	"calculate the result step by step and then double-check your answer",
}

// directSeeds want a direct answer with no reasoning — including long-but-
// shallow generation tasks, which are NOT simple but still need no thinking.
var directSeeds = []string{
	"hello, how are you today?",
	"what is the capital of Japan?",
	"translate \"thank you very much\" into Spanish",
	"summarise this article in three bullet points",
	"write a 500-word blog post about the history of coffee",
	"list ten common houseplants",
	"rephrase this paragraph in a friendlier tone",
	"what does the acronym NASA stand for?",
	"give me a simple recipe for pancakes",
	"convert 25 degrees Celsius to Fahrenheit",
	"extract all the phone numbers from this text",
	"classify this email as spam or not spam",
	"what year did the Berlin Wall fall?",
	"fix the grammar in this sentence",
	"write a short product description for these headphones",
	"format this list of items as a markdown table",
	"name the three primary colours",
	"spell-check this paragraph and return the corrected text",
}

// ── Classifier ──────────────────────────────────────────────────────────────

// difficultyBand maps an upper bound on the difficulty score [0,1] to a target
// minimum backend quality. Bands are held sorted ascending by upTo.
type difficultyBand struct {
	upTo    float64
	quality int
}

// classification holds both axis scores for a prompt, each in [0,1].
type classification struct {
	difficulty float64 // 0 = trivial, 1 = hard → model tier
	reasoning  float64 // 0 = direct, 1 = needs reasoning → thinking mode
}

// centroids are the four normalised seed-group centroids.
type centroids struct {
	simple, hard, reasoning, direct []float64
}

type difficultyClassifier struct {
	// embed turns texts into vectors. Injected so tests can fake it; in
	// production it points at Router.embedTexts. Must be concurrency-safe.
	embed func(ctx context.Context, texts []string) ([][]float64, error)

	bands              []difficultyBand
	temp               float64 // softness of the margin → score map
	reasoningThreshold float64 // reasoning score ≥ this → enable thinking
	maxChars           int     // cap on classified prompt text length
	// deadlineNanos is the per-classification embedding budget. Atomic because
	// certification writes it from the health loop's goroutine (see
	// observeEmbedLatency) while classify reads it on request goroutines.
	deadlineNanos atomic.Int64

	mu            sync.Mutex
	ready         bool
	cents         centroids
	lastAttempt   time.Time // last bootstrap attempt (for cooldown)
	lastEmbedFail time.Time // last live-embed failure (for cooldown)

	cache *difficultyCache
}

func newDifficultyClassifier(cfg *Config, embed func(context.Context, []string) ([][]float64, error)) *difficultyClassifier {
	// Empty bands → automatic, fleet-derived tiers (see autoTargetQuality). An
	// explicit ROUTER_DIFFICULTY_QUALITY_BANDS is an optional override.
	bands := parseDifficultyBands(cfg.DifficultyBands)
	temp := cfg.DifficultyTemp
	if temp <= 0 {
		temp = 0.10
	}
	threshold := cfg.ReasoningThreshold
	if threshold <= 0 || threshold >= 1 {
		threshold = 0.35
	}
	timeout := cfg.DifficultyTimeout
	if timeout <= 0 {
		timeout = 2 * time.Second
	}
	maxChars := cfg.DifficultyMaxChars
	if maxChars <= 0 {
		maxChars = 4000
	}
	c := &difficultyClassifier{
		embed:              embed,
		bands:              bands,
		temp:               temp,
		reasoningThreshold: threshold,
		maxChars:           maxChars,
		cache:              newDifficultyCache(cfg.DifficultyCacheSize),
	}
	c.deadlineNanos.Store(int64(timeout))
	return c
}

// classifierDeadlineFactor scales the embeddings worker's MEASURED probe latency
// into the per-classification deadline. Eight, because the probe embeds one
// short sentence on an otherwise idle worker while a real classification embeds
// up to difficultyMaxChars of prompt with live traffic alongside it: the factor
// has to cover a much larger input AND a much busier moment.
//
// The floor is difficultyTimeoutFallback, so a quick worker can never make the
// deadline TIGHTER than the two seconds the router shipped with — the point of
// measuring is to accommodate a slow box, not to squeeze a fast one.
//
// The ceiling is embedFailCooldown, which is not a new number but the one the
// router already uses to bound what a wedged embeddings worker may cost: past
// it, a single classification would wait longer than the router is willing to
// skip classifying altogether, and waiting has stopped being the cheaper option.
const classifierDeadlineFactor = 8

// deadline is the per-classification embedding budget.
func (c *difficultyClassifier) deadline() time.Duration {
	return time.Duration(c.deadlineNanos.Load())
}

// observeEmbedLatency sets the classification deadline from a MEASURED
// embeddings round trip and returns what it settled on. A fixed two seconds
// silently disabled classification on a slow box — every classify() timed out,
// routing quietly fell back to quality/speed, and /health still reported the
// embeddings worker present, so nothing said so.
func (c *difficultyClassifier) observeEmbedLatency(measured time.Duration) time.Duration {
	d := measured * classifierDeadlineFactor
	if d < difficultyTimeoutFallback {
		d = difficultyTimeoutFallback
	}
	if d > embedFailCooldown {
		d = embedFailCooldown
	}
	c.deadlineNanos.Store(int64(d))
	return d
}

// classify scores a chat request on both axes from a single embedding, cached
// by prompt text. ok=false when classification is unavailable (caller falls
// back to non-classified behaviour).
func (c *difficultyClassifier) classify(req *ChatRequest) (classification, bool) {
	text := classifyInput(req, c.maxChars)
	if text == "" {
		return classification{}, false
	}
	if cl, ok := c.cache.get(text); ok {
		return cl, true
	}
	if !c.ensureReady() {
		return classification{}, false
	}
	// After a live-embed failure, fall back unclassified until the cooldown
	// elapses — a wedged-but-reachable worker must cost at most one embed
	// timeout per window, not two per request (difficulty, then reasoning).
	c.mu.Lock()
	if !c.lastEmbedFail.IsZero() && time.Since(c.lastEmbedFail) < embedFailCooldown {
		c.mu.Unlock()
		return classification{}, false
	}
	c.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), c.deadline())
	defer cancel()
	vecs, err := c.embed(ctx, []string{text})
	if err != nil || len(vecs) == 0 || len(vecs[0]) == 0 {
		c.mu.Lock()
		c.lastEmbedFail = time.Now()
		c.mu.Unlock()
		return classification{}, false
	}
	c.mu.Lock()
	c.lastEmbedFail = time.Time{} // a successful embed clears the cooldown
	cents := c.cents
	if len(vecs[0]) != len(cents.simple) {
		// A changed vector dimension means the embeddings model was swapped after
		// bootstrap: the centroids and every cached score are from the old space,
		// and dot over a truncated prefix would mid-tier everything. Invalidate
		// and re-bootstrap (under ensureReady's cooldown) instead of scoring.
		c.ready = false
		c.mu.Unlock()
		c.cache.reset()
		return classification{}, false
	}
	c.mu.Unlock()

	v := normalize(vecs[0])
	cl := classification{
		difficulty: clamp01(0.5 + 0.5*math.Tanh((dot(v, cents.hard)-dot(v, cents.simple))/c.temp)),
		reasoning:  clamp01(0.5 + 0.5*math.Tanh((dot(v, cents.reasoning)-dot(v, cents.direct))/c.temp)),
	}
	c.cache.put(text, cl)
	return cl, true
}

// targetQuality classifies a request and returns the target minimum backend
// quality, the raw difficulty score, and ok=false when unavailable.
func (c *difficultyClassifier) targetQuality(req *ChatRequest) (int, float64, bool) {
	cl, ok := c.classify(req)
	if !ok {
		return 0, 0, false
	}
	return c.bandQuality(cl.difficulty), cl.difficulty, true
}

// targetForFleet maps a difficulty score to a target quality. With an explicit
// bands override configured it uses that; otherwise it derives the target from
// the candidate fleet's own quality range — zero-config and self-adapting.
func (c *difficultyClassifier) targetForFleet(candidates []*Backend, score float64, thinkOff bool) int {
	if len(c.bands) > 0 {
		return c.bandQuality(score)
	}
	return autoTargetQuality(candidates, score, thinkOff)
}

// benchmarkQualityScale is the ceiling of runQualityBenchmark's score: quality
// is the percentage of the versioned question set answered correctly, so the
// scale is absolute and comparable across workers and across time.
const benchmarkQualityScale = 100

// autoTargetQuality maps a difficulty score onto the benchmark's own absolute
// 0–100 scale: score 0 → no quality bar, score 1 → the (unreachable) perfect
// score, clamped to the best quality actually registered so an out-of-reach
// bar degrades to "the smartest there is" rather than an empty set. Still no
// hand-set thresholds — the bar is a property of the QUESTION.
//
// It used to be linear over the fleet's own [qmin, qmax], which made "smart
// enough" relative to whoever happened to be registered: measured 2026-08-13,
// registering a quality-93 CPU worker (17 tok/s) stretched the range and
// re-tiered the same d=0.65 question from q>=79 (a 1.8s GPU answer) to q>=87
// (26.4s on the CPU worker) — the speed ranking never saw more than one
// candidate. Anchoring the bar absolutely keeps every worker above it in the
// race, and rankByDifficulty's expected-completion ordering does the rest.
func autoTargetQuality(candidates []*Backend, score float64, thinkOff bool) int {
	if len(candidates) == 0 {
		return 0
	}
	// The clamp reads the score for the mode the request will be SERVED in: a
	// no-think request clamps against the best no-think quality, so the bar a
	// thinking-mode score set can't strand it above every worker's real
	// no-think ability.
	qmax := qualityFor(candidates[0], thinkOff)
	for _, b := range candidates {
		if q := qualityFor(b, thinkOff); q > qmax {
			qmax = q
		}
	}
	target := int(math.Round(clamp01(score) * benchmarkQualityScale))
	if target > qmax {
		return qmax
	}
	return target
}

// wantThinking maps a reasoning score to an enable_thinking decision. A low
// threshold keeps thinking on for more prompts (conservative — only clearly
// direct prompts get it disabled), since wrongly disabling thinking is the
// quality-risky direction.
func (c *difficultyClassifier) wantThinking(reasoning float64) bool {
	return reasoning >= c.reasoningThreshold
}

// ensureReady lazily bootstraps the centroids. Only one goroutine attempts the
// bootstrap at a time (others fall through to default routing); a cooldown
// avoids re-hitting a down embeddings worker on every request.
func (c *difficultyClassifier) ensureReady() bool {
	c.mu.Lock()
	if c.ready {
		c.mu.Unlock()
		return true
	}
	if !c.lastAttempt.IsZero() && time.Since(c.lastAttempt) < bootstrapCooldown {
		c.mu.Unlock()
		return false
	}
	c.lastAttempt = time.Now()
	c.mu.Unlock()

	cents, err := c.bootstrap()
	if err != nil {
		log.Printf("prompt classifier bootstrap failed (will retry after %s): %v", bootstrapCooldown, err)
		return false
	}
	c.mu.Lock()
	c.cents, c.ready = cents, true
	c.mu.Unlock()
	log.Printf("prompt classifier ready (difficulty + reasoning centroids from %d seeds)",
		len(simpleSeeds)+len(hardSeeds)+len(reasoningSeeds)+len(directSeeds))
	return true
}

func (c *difficultyClassifier) bootstrap() (centroids, error) {
	groups := [][]string{simpleSeeds, hardSeeds, reasoningSeeds, directSeeds}
	all := []string{}
	for _, g := range groups {
		all = append(all, g...)
	}
	ctx, cancel := context.WithTimeout(context.Background(), c.deadline()*2+5*time.Second)
	defer cancel()
	vecs, err := c.embed(ctx, all)
	if err != nil {
		return centroids{}, err
	}
	if len(vecs) != len(all) {
		return centroids{}, fmt.Errorf("embedder returned %d vectors for %d seeds", len(vecs), len(all))
	}
	off := 0
	take := func(n int) []float64 {
		cen := centroid(vecs[off : off+n])
		off += n
		return cen
	}
	cents := centroids{
		simple:    take(len(simpleSeeds)),
		hard:      take(len(hardSeeds)),
		reasoning: take(len(reasoningSeeds)),
		direct:    take(len(directSeeds)),
	}
	if cents.simple == nil || cents.hard == nil || cents.reasoning == nil || cents.direct == nil {
		return centroids{}, errors.New("empty centroid from seed embeddings")
	}
	return cents, nil
}

// bandQuality maps a difficulty score to the configured target quality.
func (c *difficultyClassifier) bandQuality(score float64) int {
	for _, band := range c.bands {
		if score <= band.upTo {
			return band.quality
		}
	}
	if len(c.bands) > 0 {
		return c.bands[len(c.bands)-1].quality
	}
	return 0
}

// ── Candidate ordering ──────────────────────────────────────────────────────

// rankByDifficulty orders candidates so the cheapest backend that clears the
// target quality comes first, breaking ties within a quality tier by expected
// completion time under current load. Backends below the bar trail (closest
// first) as graceful fallback. The returned slice feeds pickAndAcquire, which
// still spills past any momentarily-saturated front-runner.
func rankByDifficulty(candidates []*Backend, target int, job jobCost, thinkOff bool) []*Backend {
	sort.SliceStable(candidates, func(i, j int) bool {
		a, b := candidates[i], candidates[j]
		// Quality is read for the mode this request will be SERVED in — see
		// qualityFor. A reasoning MoE ranks by its no-think score on a no-think
		// request, not the thinking-mode score it benchmarked.
		aq, bq := qualityFor(a, thinkOff), qualityFor(b, thinkOff)
		// 1. Prefer a backend with a free slot (matches rankBackends; spilling is
		//    still handled downstream by pickAndAcquire).
		if af, bf := isFull(a), isFull(b); af != bf {
			return !af
		}
		// 2. Prefer backends that clear the quality bar.
		am, bm := aq >= target, bq >= target
		if am != bm {
			return am
		}
		// 3. Among those below the bar, the highest quality (closest) first.
		if !am && aq != bq {
			return aq > bq
		}
		// 3.5 Among those that CLEAR the bar, prefer the free ones (PLAN.md, P4).
		//     Deliberately scoped to the above-bar group: below the bar the router
		//     has already missed the quality it wanted, and buying a worse answer
		//     to save money there is not a trade anyone asked for. Above it every
		//     survivor is good enough by construction, so cost is free to decide.
		//
		//     This is only the ORDER. Refusing to spend is the acquire step's job:
		//     qualityFloorPreference holds the request on the free set for the
		//     existing grace before it will touch a paid endpoint at all.
		if am {
			if af, bf := isFreeBackend(a), isFreeBackend(b); af != bf {
				return af
			}
		}
		// 4. Otherwise pick whichever will FINISH FIRST — expected completion time
		//    for THIS job from live prefill/decode rates + queue occupancy. For
		//    backends that clear the bar this replaces "cheapest tier": slow workers
		//    lose on latency on their own (no speed cutoff to tune), and a busy fast
		//    worker sheds load to idle ones automatically.
		if la, lb := expectedLatency(a, job), expectedLatency(b, job); la != lb {
			return la < lb
		}
		// 5. Exact tie → keep the conversation where it is. Only reachable when the
		//    completion-time estimate can't separate them at all (e.g. neither has a
		//    measured prefill rate yet), where staying is free and switching costs a
		//    cold prefix.
		if ai, bi := sessionIncumbent(a, job), sessionIncumbent(b, job); ai != bi {
			return ai
		}
		return a.ID < b.ID
	})
	return candidates
}

// sessionIncumbent reports whether b served the previous turn of this job's
// conversation.
func sessionIncumbent(b *Backend, job jobCost) bool {
	return job.incumbent != "" && b.ID == job.incumbent
}

// deadlineFilter drops candidates that cannot plausibly finish the job inside the
// caller's remaining budget. Callers abandon a request at their own timeout (the
// clabtree agent at 120s), so placing a job on a worker that needs several times
// that spends the whole budget and fails anyway — the 503s in the request log are
// all of this shape.
//
// Deliberately never returns an empty set: if no worker fits, every candidate is
// kept and the ranker still puts the soonest-to-finish first. The estimate is a
// heuristic and must not become a new source of refusals on its own.
func deadlineFilter(candidates []*Backend, job jobCost, budget time.Duration) ([]*Backend, bool) {
	if budget <= 0 || len(candidates) == 0 {
		return candidates, false
	}
	limit := budget.Seconds() * deadlineSafetyFactor
	fit := filterCandidates(candidates, func(b *Backend) bool {
		return expectedLatency(b, job) <= limit
	})
	if len(fit) == 0 || len(fit) == len(candidates) {
		return candidates, false
	}
	return fit, true
}

// isFull reports whether a capped backend is at its concurrency limit.
func isFull(b *Backend) bool {
	return b.MaxConcurrency > 0 && occupancy(b) >= b.MaxConcurrency
}

// occupancy is how many requests are in flight against a backend's real
// hardware, which is not always what this router dispatched.
//
// For a local worker the two are the same and this is ActiveRequests. For a
// RELAY row the upstream router is dispatching to those GPUs as well — that is
// the point of a relay, one queue for both fleets — so its own count is the
// fleet truth, refreshed on each poll (see setRelayLoad).
//
// The two are combined with max rather than sum, and the difference matters:
// this router's in-flight requests are occupying upstream slots, so they are
// ALREADY inside the number the upstream reported, and adding them would count
// each one twice. Max instead takes the upstream's figure as the floor and lets
// a burst dispatched since the last poll — which the upstream has not told us
// about yet — raise it.
func occupancy(b *Backend) int {
	if b.RemoteActive > b.ActiveRequests {
		return b.RemoteActive
	}
	return b.ActiveRequests
}

// liveTPS prefers the live EWMA throughput, falling back to the certified then
// the statically-declared baseline. This is what makes ObservedTPS actually
// influence routing (Phase 2).
func liveTPS(b *Backend) float64 {
	if b.ObservedTPS > 0 {
		return b.ObservedTPS
	}
	if b.Certification.TokensPerSec > 0 {
		return b.Certification.TokensPerSec
	}
	return b.BaselineTPS
}

// expectedLatency estimates seconds to finish a specific job on a backend, from
// its prefill rate, decode throughput and current queue occupancy. Lower is
// better. Both work terms scale with the job, so a long prompt or a thinking turn
// ranks a slow worker correctly instead of looking identical to a short chat turn.
func expectedLatency(b *Backend, job jobCost) float64 {
	tps := liveTPS(b)
	if tps <= 0 {
		tps = 1
	}
	gen := float64(job.outputTokens) / tps       // decode time for this output
	first := prefillSeconds(b, job.promptTokens) // time to first token for this prompt
	// Session affinity: the worker that served the previous turn of this
	// conversation should still hold its prefix, so most of this prompt's prefill
	// is already paid for. Discounting the PREFILL term (rather than adding a flat
	// stay bonus) is what gives stickiness the right shape for free — it grows with
	// the conversation and vanishes for a short one-off. See session.go.
	if job.incumbent != "" && b.ID == job.incumbent {
		first *= 1 - sessionPrefillDiscount
	}
	slots := b.MaxConcurrency
	if slots <= 0 {
		slots = uncappedNominalSlots
	}
	if slots <= 0 {
		slots = 1
	}
	wait := 0.0
	if over := float64(occupancy(b) + 1 - slots); over > 0 {
		wait = math.Ceil(over / float64(slots))
	}
	// One request's service time is first-token latency plus decode; queueing past
	// the slot count multiplies it.
	return (first + gen) * (1 + wait)
}

// prefillSeconds estimates time-to-first-token for THIS request's prompt.
//
// Prefill rate varies far more across workers than decode does — measured on one
// ~4k-token prompt: 0.67s on the GPU worker vs 37.2s on a CPU worker, a 55x gap —
// so scaling a measured tok/s by the actual prompt beats reusing a raw TTFT
// average. The raw EWMA is also not comparable across workers, because each one's
// average reflects whatever prompt sizes it happens to receive. Falls back to the
// TTFT EWMA (previous behaviour) until a prefill rate has been measured.
//
// A relay row adds its round trip on top, and this is the only place it is
// added. Every latency term on such a row describes the far ENDPOINT with the
// link between the two routers excluded — imported that way (relayProfile) and
// stripped back out of this router's own samples (Registry.observe) — precisely
// so that the link can be one term here rather than something folded into a
// rate, which cannot carry it, or into a TTFT that the rate then shadows.
// Zero on every other row, which is the truth for a worker this router reaches
// directly.
func prefillSeconds(b *Backend, promptTokens int) float64 {
	wire := b.RelayRTTMillis / 1000
	if rate := b.ObservedPrefillTPS; rate > 0 && promptTokens > 0 {
		return float64(promptTokens)/rate + wire
	}
	return liveTTFT(b) + wire
}

// liveTTFT returns the backend's first-token latency in seconds — the live EWMA
// if we have one, else the value measured at certification. Zero when unknown.
func liveTTFT(b *Backend) float64 {
	if b.ObservedTTFTMillis > 0 {
		return b.ObservedTTFTMillis / 1000.0
	}
	if b.Certification.TTFTMillis > 0 {
		return float64(b.Certification.TTFTMillis) / 1000.0
	}
	return 0
}

// ── Prompt text extraction ──────────────────────────────────────────────────

// classifyInput picks the text to classify for a request: an explicit
// classify_text hint when the client sent one (the genuine user message, before
// any agent-injected runtime context), else the last user turn extracted from
// the messages. Both paths are capped at maxChars. Keeping selection here (not
// in classify) makes it unit-testable without an embeddings worker.
func classifyInput(req *ChatRequest, maxChars int) string {
	if t := strings.TrimSpace(req.ClassifyText); t != "" {
		return cutChars(t, maxChars)
	}
	return classifyText(req.Messages, maxChars)
}

// classifyText picks the salient text to classify: the last user turn, falling
// back to the system prompt, truncated to a rune boundary at maxChars.
func classifyText(messages []Message, maxChars int) string {
	var system, lastUser string
	for _, m := range messages {
		switch m.Role {
		case "system", "developer":
			if s := contentText(m.Content); s != "" {
				system = s
			}
		case "user":
			if s := contentText(m.Content); s != "" {
				lastUser = s
			}
		}
	}
	text := strings.TrimSpace(lastUser)
	if text == "" {
		text = strings.TrimSpace(system)
	}
	return cutChars(text, maxChars)
}

// contentText flattens OpenAI message content (a string, or an array of typed
// parts) into its text, ignoring images and other non-text parts.
func contentText(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var b strings.Builder
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if t, _ := m["type"].(string); t != "text" {
				continue
			}
			if s, _ := m["text"].(string); s != "" {
				if b.Len() > 0 {
					b.WriteByte(' ')
				}
				b.WriteString(s)
			}
		}
		return b.String()
	}
	return ""
}

// cutChars truncates s to at most max bytes without splitting a UTF-8 rune.
func cutChars(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut]
}

// ── Vector math ─────────────────────────────────────────────────────────────

func normalize(v []float64) []float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return v
	}
	inv := 1.0 / math.Sqrt(sum)
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

func dot(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s float64
	for i := 0; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}

// centroid averages a group of vectors (each L2-normalised first) and returns
// the normalised mean direction.
func centroid(vecs [][]float64) []float64 {
	if len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil
	}
	dim := len(vecs[0])
	acc := make([]float64, dim)
	n := 0
	for _, v := range vecs {
		if len(v) != dim {
			continue
		}
		nv := normalize(v)
		for i := range nv {
			acc[i] += nv[i]
		}
		n++
	}
	if n == 0 {
		return nil
	}
	for i := range acc {
		acc[i] /= float64(n)
	}
	return normalize(acc)
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// ── Score cache (bounded, FIFO eviction) ────────────────────────────────────

type difficultyCache struct {
	mu    sync.Mutex
	max   int
	m     map[uint64]classification
	order []uint64
}

func newDifficultyCache(max int) *difficultyCache {
	if max < 1 {
		max = 1
	}
	return &difficultyCache{max: max, m: make(map[uint64]classification, max)}
}

func cacheKey(text string) uint64 {
	h := fnv.New64a()
	_, _ = io.WriteString(h, text)
	return h.Sum64()
}

func (c *difficultyCache) get(text string) (classification, bool) {
	k := cacheKey(text)
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.m[k]
	return v, ok
}

func (c *difficultyCache) put(text string, v classification) {
	k := cacheKey(text)
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, exists := c.m[k]; exists {
		c.m[k] = v
		return
	}
	if len(c.order) >= c.max {
		delete(c.m, c.order[0])
		c.order = c.order[1:]
	}
	c.m[k] = v
	c.order = append(c.order, k)
}

// reset drops every cached score. Scores are only meaningful within the
// centroid space they were computed in, so a re-bootstrap must clear them.
func (c *difficultyCache) reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.m = make(map[uint64]classification, c.max)
	c.order = nil
}

// ── Thinking-mode resolution ────────────────────────────────────────────────

// normalizeThinking canonicalizes a requirements.thinking value to "on", "off"
// or "auto". Legacy synonyms from older clients are folded in
// ("required"/"true"/"optional" → on, "disabled"/"false"/"none" → off);
// anything else — including absent — means auto (the router decides).
func normalizeThinking(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "on", "required", "true", "optional":
		return "on"
	case "off", "disabled", "false", "none":
		return "off"
	default:
		return "auto"
	}
}

// thinkingResolution is the single resolved enable_thinking decision for one
// request. It is derived ONCE (resolveThinking) from the client's explicit
// signals and — on a normal route — the auto reasoning classifier, then drives
// BOTH the candidate filter in selectBackends and the single-pass body patch in
// patchForwardedBody. Computing it once is what guarantees the worker we pick
// and the enable_thinking we forward can never disagree (fix #1).
type thinkingResolution struct {
	patch     bool   // write chat_template_kwargs.enable_thinking into the forwarded body
	enable    bool   // value to write when patch is true
	effort    string // chat_template_kwargs.reasoning_effort to write beside it ("" ⇒ write none)
	hardThink bool   // hard-require a thinking-capable worker (explicit "on"; may 503)
	softThink bool   // prefer a thinking-capable worker if any survive (auto reasoning)
	// noThink records that this request WILL be served with thinking disabled —
	// explicit "off", reasoning_effort "none", a kwargs escape hatch pinning it
	// off, or a direct verdict from the auto classifier. Selection then compares
	// quality targets against each worker's no-think benchmark score instead of
	// the mixed-mode one (see qualityFor): a reasoning MoE can be a different,
	// far worse model with thinking suppressed, and routing it hard no-think
	// requests on its thinking-mode score is how a q=84 worker came to write
	// deterministic garbage SQL (2026-08-24). False when the mode is unknown
	// (no classifier and nothing explicit), where the mixed score is the best
	// available estimate — the pre-two-score behaviour.
	noThink bool
	// dialect is the spelling the CHOSEN worker was measured to honour (a
	// thinkingDialect* constant). Empty ⇒ unknown, use chat_template_kwargs, the
	// dialect the fleet has always spoken. Stamped after selection rather than
	// resolved with the rest: the decision is the same whoever serves the
	// request, but how to say it is a property of the endpoint, and two workers
	// in one fleet can disagree.
	dialect string
}

// forBackend stamps the chosen worker's measured thinking dialect onto the
// resolution, ready for patchForwardedBody. Called at each patch site rather
// than once, because an inline escalation re-patches for a different worker.
func (tr thinkingResolution) forBackend(b *Backend) thinkingResolution {
	if b != nil {
		tr.dialect = b.ThinkingDialect
	}
	return tr
}

// normalizeEffort canonicalizes an OpenAI-standard reasoning_effort value.
// Returns "" for absent (⇒ auto, the router decides) and "none" for every
// spelling of off; anything else is passed through verbatim, because the
// meaningful set is the WORKER's chat template's, not ours — DeepSeek V4
// branches on "high" and "max", other templates on other words, and a level this
// router has never heard of must still reach the template that understands it.
func normalizeEffort(v string) string {
	s := strings.ToLower(strings.TrimSpace(v))
	switch s {
	case "":
		return ""
	case "none", "off", "disabled", "false", "no":
		return "none"
	}
	return s
}

// resolveThinking computes the single thinkingResolution for a request, in
// strict precedence order:
//
//  1. An explicit OpenAI-standard reasoning_effort is the harness knob: "none"
//     patches enable_thinking=false, any other level patches enable_thinking=true
//     AND carries the level itself through to the template. Like "on" below it
//     hard-requires a thinking-capable worker — a harness that asked for high
//     effort is not served by silently landing on a model that cannot think.
//  2. Else an explicit requirements.thinking ("on"/"off", incl. legacy synonyms):
//     it patches enable_thinking on EVERY route (incl. pinned/debug). "on"
//     additionally hard-requires a thinking-capable worker; "off" never filters
//     (any worker can serve a non-thinking request).
//  3. Else an explicit chat_template_kwargs.enable_thinking in the body is the
//     low-level escape hatch — never overwritten (no patch), but it still
//     hard-filters when true (mirrors thinkingFromRequest feeding the filter).
//  4. Else, on an auto-routed request ("route"/"model" prefix, not pinned/debug)
//     with auto-thinking enabled and a usable classification, the reasoning axis
//     decides: a "think" verdict patches enable_thinking=true and SOFT-prefers a
//     thinking-capable worker; a direct verdict patches enable_thinking=false.
//
// Rule 4 covers the "model" route deliberately: naming a model is a statement
// about WHICH worker, not about whether to think, so a harness that pins a model
// and says nothing about effort still gets the automatic decision every other
// client gets. Only the header pin and the debug endpoint opt out.
//
// cl is the classification computed once in selectBackends (nil on the
// pinned/debug path, where auto-thinking is intentionally skipped). The actual
// write still respects a present enable_thinking in the body — patchForwardedBody
// never overwrites the kwargs escape hatch — so rule 2's intent on a body that
// already pins the kwarg is a no-op, exactly as before.
func (r *Router) resolveThinking(chatReq *ChatRequest, route string, cl *classification) thinkingResolution {
	if eff := normalizeEffort(chatReq.ReasoningEffort); eff != "" {
		if eff == "none" {
			return thinkingResolution{patch: true, enable: false, noThink: true}
		}
		return thinkingResolution{patch: true, enable: true, effort: eff, hardThink: true}
	}
	pref := "auto"
	if chatReq.Requirements != nil {
		pref = normalizeThinking(chatReq.Requirements.Thinking)
	}
	switch pref {
	case "on":
		return thinkingResolution{patch: true, enable: true, hardThink: true}
	case "off":
		return thinkingResolution{patch: true, enable: false, noThink: true}
	}
	// No explicit requirements.thinking — the kwargs escape hatch is next.
	if clientSetKwargThinking(chatReq) {
		kwarg := normalizeThinking(thinkingFromRequest(chatReq))
		return thinkingResolution{hardThink: kwarg == "on", noThink: kwarg == "off"}
	}
	// Auto reasoning. Only on normal routes, with auto-thinking enabled and a
	// usable classification (nil on pinned/debug, or when the classifier was
	// unavailable). Best-effort: anything missing leaves the body untouched.
	if cl == nil || r.classifier == nil || r.cfg == nil || !r.cfg.AutoThinking {
		return thinkingResolution{}
	}
	if !strings.HasPrefix(route, "route") && !strings.HasPrefix(route, "model") { // skip "pinned" and "debug"
		return thinkingResolution{}
	}
	if r.classifier.wantThinking(cl.reasoning) {
		return thinkingResolution{patch: true, enable: true, softThink: true}
	}
	return thinkingResolution{patch: true, enable: false, noThink: true}
}

// clientSetKwargThinking reports whether the client pinned a thinking gate at
// the chat_template_kwargs level — the low-level escape hatch auto-thinking
// must not override. Either spelling counts (see thinkingKwargKeys).
// (requirements.thinking and reasoning_effort are resolved in resolveThinking.)
func clientSetKwargThinking(req *ChatRequest) bool {
	if req.ChatTemplateKwargs != nil {
		for _, key := range thinkingKwargKeys {
			if _, ok := req.ChatTemplateKwargs[key]; ok {
				return true
			}
		}
	}
	return false
}

// routerOnlyFields are the request fields the ROUTER reads and nothing
// downstream should ever see: routing hints the northbound API accepts as
// additions to the OpenAI shape (see ChatRequest). They are stripped from every
// forwarded body.
var routerOnlyFields = []string{"requirements", "classify_text", "deadline_ms"}

// mayCarryRouterFields reports whether the body is worth parsing to strip
// router-only fields. It is a pre-filter for the "nothing to patch" fast exit,
// not the test itself: a false positive costs one parse and the authoritative
// key lookup in patchForwardedBody then finds nothing. A multi-MB vision body
// is scanned at memchr speed, which is the point — the exit exists so such a
// body is not unmarshalled and re-marshalled for no reason.
func mayCarryRouterFields(body []byte) bool {
	for _, field := range routerOnlyFields {
		if bytes.Contains(body, []byte(`"`+field+`"`)) {
			return true
		}
	}
	return false
}

// patchForwardedBody rewrites the chat body for the worker in a SINGLE pass: it
// fills in max_tokens when the caller decided the client set no budget
// (defaultMaxTokens > 0 — the caller resolves max_tokens vs
// max_completion_tokens via effectiveMaxTokens, so pass 0 to leave a client
// budget untouched), clamps a client budget the chosen worker cannot fit
// (ceiling > 0), and writes the chat-template thinking gate per the resolved
// decision — one unmarshal, one marshal — so a multi-MB vision body is copied
// once rather than re-parsed per patch site. An explicit thinking gate already
// in the body is never overwritten (the kwargs escape hatch wins).
// servedModel, when non-empty, replaces the request's "model" with the id the
// chosen worker actually advertises: clients name router-side spellings
// (aliases like "qwen3.8", registration labels), which llama.cpp ignores but
// vLLM rejects with a 404 — the worker only accepts its own --served-model-name.
// It also removes the router's OWN fields (routerOnlyFields), which no endpoint
// has ever needed and a strict one rejects outright.
// Best-effort: any JSON error returns the body unchanged.
func patchForwardedBody(body []byte, defaultMaxTokens, ceiling int, tr thinkingResolution, servedModel string) []byte {
	needMaxTokens := defaultMaxTokens > 0
	if !needMaxTokens && ceiling <= 0 && !tr.patch && servedModel == "" && !mayCarryRouterFields(body) {
		return body
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return body
	}
	changed := false

	// The router's own fields never go south. They were documented as harmless
	// and were, for as long as every endpoint was a local vLLM or llama.cpp that
	// ignores keys it doesn't know; a metered provider validates its input and
	// fails the whole request instead.
	for _, field := range routerOnlyFields {
		if _, present := raw[field]; present {
			delete(raw, field)
			changed = true
		}
	}

	if servedModel != "" {
		if sm, err := json.Marshal(servedModel); err == nil && string(raw["model"]) != string(sm) {
			raw["model"] = sm
			changed = true
		}
	}

	if needMaxTokens {
		// Overwrite an unusable value too: "max_tokens": null or 0 previously
		// slipped through as "client set it", forwarding an unbounded request
		// while the router budgeted with the default.
		existing, present := raw["max_tokens"]
		if !present || string(existing) == "null" || string(existing) == "0" {
			if mt, err := json.Marshal(defaultMaxTokens); err == nil {
				raw["max_tokens"] = mt
				changed = true
			}
		}
	}

	// A client's max_tokens is a ceiling, not a forecast, so the hard context
	// filter no longer demands a worker able to emit all of it (see
	// contextReserveTokens). That makes this clamp load-bearing: trim the budget
	// to what the worker we actually picked can still fit. llama.cpp would
	// silently truncate n_predict to the free context, but vLLM rejects the
	// request outright, and a 400 on a harness's generous cap is exactly the
	// failure the reserve change is meant to remove.
	if ceiling > 0 {
		for _, field := range []string{"max_tokens", "max_completion_tokens"} {
			existing, present := raw[field]
			if !present {
				continue
			}
			var have int
			if err := json.Unmarshal(existing, &have); err != nil || have <= ceiling {
				continue // absent, null, or already inside the worker's context
			}
			if mt, err := json.Marshal(ceiling); err == nil {
				raw[field] = mt
				changed = true
			}
		}
	}

	// Write the thinking gate in the spelling the chosen endpoint was MEASURED to
	// honour (see thinkingProbe). The kwargs form is a vLLM and llama.cpp
	// extension: on a provider that speaks the standard it does nothing at best
	// and fails the request at worst.
	if tr.patch {
		switch tr.dialect {
		case thinkingDialectEffort:
			if ej, ok := thinkingEffortValue(raw["reasoning_effort"], tr.enable, tr.effort); ok {
				raw["reasoning_effort"] = ej
				changed = true
			}
		case thinkingDialectNone:
			// Measured: neither gate does anything here. Writing one anyway buys a
			// field a strict endpoint can reject, in exchange for nothing.
		default:
			// Unknown — never probed, or a profile cached before the dialect was —
			// so use the spelling the fleet has always spoken.
			if kj, ok := mergeThinkingKwargs(raw["chat_template_kwargs"], tr.enable, tr.effort); ok {
				raw["chat_template_kwargs"] = kj
				changed = true
			}
		}
	}

	if !changed {
		return body
	}
	patched, err := json.Marshal(raw)
	if err != nil {
		return body
	}
	return patched
}

// thinkingEffortValue renders the resolved thinking decision as the
// OpenAI-standard top-level reasoning_effort, for an endpoint measured to honour
// that spelling. existing is the raw value already in the body (nil when
// absent); ok=false means leave it alone — a level the client set is the escape
// hatch and outranks anything the router resolved, exactly as with the kwargs
// gate.
//
// Off is written as "none", which is the standard's own spelling and the one
// normalizeEffort already folds every synonym onto. Thinking ON with no level
// asked for is written as "medium": the probe proved the field does something
// here, and medium is the standard's neutral level — an auto-thinking verdict
// should buy reasoning, not the most expensive tier available, for a prompt the
// classifier merely leaned on.
func thinkingEffortValue(existing json.RawMessage, enable bool, effort string) (json.RawMessage, bool) {
	if len(existing) > 0 && string(existing) != "null" {
		return nil, false
	}
	level := "none"
	if enable {
		if level = effort; level == "" {
			level = "medium"
		}
	}
	ej, err := json.Marshal(level)
	if err != nil {
		return nil, false
	}
	return ej, true
}

// stripBodyFields removes top-level fields from a JSON body. An empty field set
// returns the body untouched WITHOUT parsing it, which is the common case and
// the reason this is a second pass rather than another argument to
// patchForwardedBody: the multi-MB vision body that patch is careful to copy
// once is not copied twice here unless an endpoint has actually rejected
// something. Best-effort, same as patchForwardedBody: any JSON error returns the
// body unchanged.
func stripBodyFields(body []byte, fields []string) []byte {
	if len(fields) == 0 {
		return body
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return body
	}
	changed := false
	for _, field := range fields {
		if _, present := raw[field]; present {
			delete(raw, field)
			changed = true
		}
	}
	if !changed {
		return body
	}
	out, err := json.Marshal(raw)
	if err != nil {
		return body
	}
	return out
}

// mergeThinkingKwargs returns the chat_template_kwargs object with the thinking
// gate set to enable — and, when thinking is on and a level was requested,
// reasoning_effort set to it — preserving any sibling kwargs. existing is the
// raw chat_template_kwargs value from the body (nil when absent). ok=false means
// leave the body's kwargs untouched: a malformed object, or — the escape hatch —
// a thinking gate already present in either spelling.
//
// The gate is written as enable_thinking because that is the spelling both
// families read: Qwen templates take it directly, and DeepSeek V4 falls back to
// it whenever the client left `thinking` undefined.
func mergeThinkingKwargs(existing json.RawMessage, enable bool, effort string) (json.RawMessage, bool) {
	kwargs := map[string]any{}
	if len(existing) > 0 {
		if err := json.Unmarshal(existing, &kwargs); err != nil {
			return nil, false
		}
		if kwargs == nil { // "chat_template_kwargs": null unmarshals to a nil map
			kwargs = map[string]any{}
		}
		for _, key := range thinkingKwargKeys {
			if _, set := kwargs[key]; set {
				return nil, false // respect an explicit value at the kwargs level
			}
		}
	}
	kwargs["enable_thinking"] = enable
	// The level only means anything to a template that reads it, and only when
	// thinking is on at all. A level the client placed here directly outranks the
	// one we resolved, same escape-hatch rule as the gate itself.
	if enable && effort != "" {
		if _, set := kwargs["reasoning_effort"]; !set {
			kwargs["reasoning_effort"] = effort
		}
	}
	kj, err := json.Marshal(kwargs)
	if err != nil {
		return nil, false
	}
	return kj, true
}

// ── Embeddings transport ────────────────────────────────────────────────────

// embedTexts embeds texts via the best registered embeddings worker. A missing
// or unreachable worker surfaces as an error, which the difficulty classifier
// treats as "no classification" and falls back to default routing.
func (r *Router) embedTexts(ctx context.Context, texts []string) ([][]float64, error) {
	backends, err := r.selectBackendsByFeature("embeddings")
	if err != nil {
		return nil, err
	}
	backend := backends[0]
	payload := map[string]any{"model": probeModel(backend), "input": texts, "encoding_format": "float"}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamPathURL(backend, "/v1/embeddings"), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if backend.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+backend.APIKey)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("embeddings backend %s returned %d: %s", backend.ID, resp.StatusCode, truncate(string(data), 200))
	}
	var parsed struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return nil, err
	}
	out := make([][]float64, len(texts))
	for _, d := range parsed.Data {
		if d.Index >= 0 && d.Index < len(out) {
			out[d.Index] = d.Embedding
		}
	}
	for i := range out {
		if len(out[i]) == 0 {
			return nil, fmt.Errorf("embeddings response missing vector for index %d", i)
		}
	}
	return out, nil
}

// ── Config helpers ──────────────────────────────────────────────────────────

func parseDifficultyBands(raw string) []difficultyBand {
	out := []difficultyBand{}
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		kv := strings.SplitN(part, ":", 2)
		if len(kv) != 2 {
			continue
		}
		upTo, err1 := strconv.ParseFloat(strings.TrimSpace(kv[0]), 64)
		q, err2 := strconv.Atoi(strings.TrimSpace(kv[1]))
		if err1 != nil || err2 != nil {
			continue
		}
		out = append(out, difficultyBand{upTo: upTo, quality: q})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].upTo < out[j].upTo })
	return out
}

func envBool(key string, fallback bool) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func envFloat(key string, fallback float64) float64 {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return fallback
}
