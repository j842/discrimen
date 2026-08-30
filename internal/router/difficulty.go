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
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

// ── Tunables ────────────────────────────────────────────────────────────────

const (
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

// uncappedNominalSlots is the concurrency assumed for a backend whose capacity
// is UNKNOWN — a beacon the capacity ramp has not reached yet, or a relay row
// whose upstream reported none. It is a stand-in for a measurement, not a
// measurement, so it is deliberately NOT applied to a row that has told us it has
// no ceiling at all: see queueSlots for the difference and why it matters.
var uncappedNominalSlots = 4

// fallbackPrefillTPS is the prompt tokens per second assumed for a worker that
// nothing has measured at length: no live prefill EWMA, and no context ladder to
// read a rate off either. Units are PROMPT tokens per second — prefill work, not
// decode; on one box the two differ by more than two orders of magnitude.
//
// A constant rate is a poor estimate of any particular worker and a far better
// one than the constant TIME it replaced. 200 tok/s is wrong for everything, but
// it is wrong by a bounded factor, and it prices 15k tokens of prefill three
// thousand times above five tokens of it — which "liveTTFT, whatever the prompt"
// did not (see prefillSeconds).
//
// The value sits just above the CPU end of every prefill rate this fleet has
// actually measured: 23 tok/s on llm-naples-deepseek-284B-q4 (4116 prompt tokens
// in 178 real seconds), ~108 on the CPU worker in the 55x comparison below,
// 125-143 on the 284B CPU+GPU box — against 643 on llm-a750-Granite4.1-8B, 4,665
// on llm-6000pro, ~5,970 on the GPU in that same comparison. Erring toward the
// slow end is the safe direction for a latency estimate, the same call speedProbe
// and prefillProbe both make: an unmeasured worker loses long prompts to one
// measured fast, which is the right answer more often than not, and deadlineFilter
// never returns an empty set, so no worker can be priced out of existence by it.
var fallbackPrefillTPS = 200.0

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
	// promptChars is the text promptTokens was estimated FROM, kept so an
	// absolute fit decision can re-size it at a ratio of its own choosing.
	// promptTokens is priced at the nominal divisor because ranking is
	// comparative and a shared bias cancels; budgetCeiling is not comparative,
	// and reusing the ranking number there granted an output budget the worker
	// had no room for. See budgetCeiling.
	promptChars int
	// incumbent is the worker that served the previous turn of this conversation,
	// if any (see session.go). It rides in the job rather than in a separate
	// argument so every existing caller of expectedLatency — the ranker AND the
	// deadline filter — accounts for the discount consistently; a filter that
	// priced the incumbent as a cold worker could drop the very candidate the
	// ranker was about to prefer.
	incumbent string
	// mode is the thinking decision already resolved for this request. It selects
	// which of the backend's two measured profiles to price against — the same
	// split selection uses to choose between a worker's two quality scores.
	mode thinkingMode
	// maxTokens is the caller's own ceiling (0 = none), kept separately from
	// outputTokens so it can still cap a per-backend MEASUREMENT, not just the
	// nominal estimate. Without it a backend measured at 4000 tokens would be
	// priced for 4000 even when the caller allowed 200.
	maxTokens int
}

// expectedGenTokens is how many tokens THIS backend is expected to emit for this
// job, which is the largest term in how long the request takes.
//
// Prefers what the backend has actually been measured doing in this thinking
// mode over the fleet-wide nominal figure. The nominal one is a single constant
// applied to every worker, so it necessarily misprices any model that is more or
// less verbose than average — and a reasoning model with thinking on is far more
// verbose than average, which is exactly when decode dominates the wall clock.
func expectedGenTokens(b *Backend, job jobCost) float64 {
	est := float64(job.outputTokens)
	if measured := observedGenFor(b, job.mode); measured > 0 {
		est = measured
	}
	// The caller's ceiling still binds: it is a hard cap on the reply, so a
	// backend measured above it cannot spend more than the caller allowed.
	if job.maxTokens > 0 && float64(job.maxTokens) < est {
		est = float64(job.maxTokens)
	}
	if est < 1 {
		est = 1
	}
	return est
}

// observedGenFor returns the measured generated length for one thinking mode, or
// 0 when that mode has not been seen on this backend. Unknown deliberately does
// NOT fall back to the pooled figure: there is no pooled generated-length EWMA,
// because averaging a 200-token direct answer with a 12000-token reasoning trace
// produces a number that describes neither.
func observedGenFor(b *Backend, mode thinkingMode) float64 {
	switch mode {
	case thinkingOn:
		return b.ObservedGenThink
	case thinkingOff:
		return b.ObservedGenNoThink
	}
	return 0
}

// liveTPSFor is liveTPS restricted to one thinking mode, falling back to the
// pooled figure for a mode this backend has not served yet.
func liveTPSFor(b *Backend, mode thinkingMode) float64 {
	switch mode {
	case thinkingOn:
		if b.ObservedTPSThink > 0 {
			return b.ObservedTPSThink
		}
	case thinkingOff:
		if b.ObservedTPSNoThink > 0 {
			return b.ObservedTPSNoThink
		}
	}
	return liveTPS(b)
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
	// The nominal divisor, not a measured one, and deliberately: this job prices
	// every candidate's prefill, and ranking is comparative — a bias shared by all
	// of them cancels. The measured ratio is for the hard filter, where the number
	// is compared against a fixed window rather than against other workers, and an
	// error there refuses a request instead of mildly reordering two.
	job := jobCost{promptTokens: promptTokensFor(req, defaultCharsPerToken),
		promptChars: promptCharsFor(req), outputTokens: out, mode: thinkingOff}
	if thinking {
		job.mode = thinkingOn
	}
	if req != nil && req.MaxTokens > 0 {
		job.maxTokens = req.MaxTokens
	}
	return job
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

// promptCharsFor is the text a request will prefill, tool schemas included.
func promptCharsFor(req *ChatRequest) int {
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
	return chars
}

// promptTokensFor estimates the prefill size of a request at a given
// chars-per-token — measured for the model where the router has measured one,
// and defaultCharsPerToken where it has not. See tokens.go.
func promptTokensFor(req *ChatRequest, charsPerToken float64) int {
	return tokensForChars(promptCharsFor(req), charsPerToken)
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

// classification holds both axis scores for a prompt, each in [0,1].
type classification struct {
	difficulty float64 // 0 = trivial, 1 = hard → model tier
	reasoning  float64 // 0 = direct, 1 = needs reasoning → thinking mode
	// vec is the prompt's normalised embedding, kept rather than discarded so
	// the outcome matrix can find similar profiled questions without a second
	// embedding call. It rides in the classification (and therefore in the
	// cache) because it is derived from exactly the same text: computing it
	// separately would double the embedding cost of every request and could
	// disagree with the difficulty score it sits beside.
	vec []float64
}

// centroids are the four normalised seed-group centroids.
type centroids struct {
	simple, hard, reasoning, direct []float64
}

type difficultyClassifier struct {
	// embed turns texts into vectors. Injected so tests can fake it; in
	// production it points at Router.embedTexts. Must be concurrency-safe.
	embed func(ctx context.Context, texts []string) ([][]float64, error)

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
		vec:        v,
	}
	c.cache.put(text, cl)
	return cl, true
}

// benchmarkQualityScale is the ceiling of runQualityBenchmark's score: quality
// is the percentage of the versioned question set answered correctly, so the
// scale is absolute and comparable across workers and across time.
const benchmarkQualityScale = 100

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

// ── Candidate ordering ──────────────────────────────────────────────────────

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
	tps := liveTPSFor(b, job.mode)
	if tps <= 0 {
		tps = 1
	}
	gen := expectedGenTokens(b, job) / tps       // decode time for this output
	first := prefillSeconds(b, job.promptTokens) // time to first token for this prompt
	// Session affinity: the worker that served the previous turn of this
	// conversation should still hold its prefix, so most of this prompt's prefill
	// is already paid for. Discounting the PREFILL term (rather than adding a flat
	// stay bonus) is what gives stickiness the right shape for free — it grows with
	// the conversation and vanishes for a short one-off. See session.go.
	if job.incumbent != "" && b.ID == job.incumbent {
		first *= 1 - sessionPrefillDiscount
	}
	// One request's service time is first-token latency plus decode; what the
	// worker is already doing stretches it.
	return (first + gen) * loadPenalty(b)
}

// loadPenalty is how much longer this request takes because of what the backend
// is ALREADY running. One multiplier over the whole service time, never below 1.
//
// It used to be a pure step — `over := occupancy + 1 - slots`, then ceil()'d —
// which is wrong at both ends. Below saturation `over` is ≤ 0, so on an 8-slot
// worker SEVEN busy slots predicted identically to idle; then the eighth doubled
// the estimate in one jump. Real continuous batching does neither: a request that
// fits in the batch does not queue, but it does share the GPU with everything
// else in it, and per-request throughput falls smoothly as the batch grows.
//
// Two continuous terms, and the second is only the first one's overflow:
//
//	share  batch^(1-α), from the worker's own MEASURED throughput ramp — the
//	       curve capacityProbe collects at n = 1, 2, 4, 8 and used to discard
//	       (see concurrencyAlpha). α = 1 is perfect batching and no penalty at
//	       all, which is what an unmeasured worker gets: an invented slowdown is
//	       worse than none.
//	queue  the requests that do not fit, spread over the slots that will free up:
//	       (n-slots)/slots rather than ceil() of it. One request past the last
//	       slot waits for the FIRST of `slots` in-flight generations to finish,
//	       not for a whole one. On a 1-slot worker — the commonest row in the
//	       fleet, and the case the ceil() was right for — the two forms are
//	       arithmetically identical, so nothing there changes.
func loadPenalty(b *Backend) float64 {
	slots := queueSlots(b)
	if slots <= 0 {
		return 1 // no ceiling declared and none to assume — see queueSlots
	}
	n := float64(occupancy(b) + 1) // this request's own place in the line
	batch := math.Min(n, float64(slots))
	share := math.Pow(batch, 1-concurrencyAlphaFor(b))
	queue := 0.0
	if over := n - float64(slots); over > 0 {
		queue = over / float64(slots)
	}
	return share * (1 + queue)
}

// queueSlots is how many requests a backend runs at once for the purpose of
// pricing a queue — or 0 for one that has told us it has no ceiling worth
// modelling, which is not the same thing as a ceiling of zero.
//
// MaxConcurrency == 0 used to mean "assume uncappedNominalSlots" everywhere, and
// that conflates two different rows. On a beacon the capacity ramp has not
// reached yet, or a relay row whose upstream published no capacity, four is a
// stand-in for a number nobody has measured, and pricing some queue beats
// pricing none.
//
// On a hosted API row it is a stand-in for nothing. The operator entered the row
// and left the ceiling blank because the provider has no per-customer
// concurrency limit worth modelling, and charging it a queue penalty off THIS
// router's dispatch count doubled a paid endpoint's estimate at four concurrent
// requests — a number that says nothing whatever about a provider fronting
// thousands. The effect was to push traffic off a genuinely fast paid endpoint
// under mild local load, which is the opposite of what it was bought for.
//
// A local row is excluded from that even when an operator typed it: a vLLM
// somebody entered by hand is still one box with one real ceiling. "manual" says
// who wrote the row down, not what is behind it.
func queueSlots(b *Backend) int {
	if b.MaxConcurrency > 0 {
		return b.MaxConcurrency
	}
	if isManualRow(b) && !isLocalProvider(b.Provider) {
		return 0
	}
	if uncappedNominalSlots > 0 {
		return uncappedNominalSlots
	}
	return 1
}

// prefillSeconds estimates time-to-first-token for THIS request's prompt.
//
// Prefill rate varies far more across workers than decode does — measured on one
// ~4k-token prompt: 0.67s on the GPU worker vs 37.2s on a CPU worker, a 55x gap —
// so scaling a measured tok/s by the actual prompt beats reusing a raw TTFT
// average. The raw EWMA is also not comparable across workers, because each one's
// average reflects whatever prompt sizes it happens to receive.
//
// EVERY branch now scales with promptTokens, including the one for a worker
// nothing has measured. That branch used to return liveTTFT alone, described as a
// fallback "until a prefill rate has been measured", and that description
// understated it: it was not degraded accuracy on the dominant term of a
// long-context request, it was total blindness to it. Measured against the live
// fleet 2026-08-27 through /v1/route-preview, a 45-character prompt and a
// 61,000-character one produced BYTE-IDENTICAL expected_seconds on every worker
// (16.553 / 52.051 / 144.455 / 205.287) bar the single one carrying a measured
// rate. Six of seven had ObservedPrefillTPS == 0, and not by accident: it is only
// ever set from a completed FULL profile or from live STREAMING samples over
// minPrefillTokens, so a worker serving non-streaming traffic, or one that has
// only finished the quick half of profiling, never acquires one at all.
//
// The rates are tried in order of how much each knows about THIS worker:
//
//  1. ObservedPrefillTPS — the live EWMA, seeded from the prefill probe.
//  2. The context ladder's own per-rung rate, measured on this worker at 4K
//     tokens and up, which is exactly the range at issue (contextProbePrefillTPS).
//  3. fallbackPrefillTPS, a fleet constant, when nothing has measured it at
//     length at all.
//
// Only (3) adds liveTTFT, and the asymmetry is the point. The two measured rates
// are prompt tokens ÷ first-token latency, so fixed per-request overhead is
// already inside them and adding it again would bill for it twice. The constant is
// a pure rate and needs an overhead floor — and on a worker in case (3) that is
// precisely what liveTTFT is: observe() feeds the TTFT EWMA only from
// non-thinking requests and feeds the prefill EWMA from those SAME requests once
// the prompt clears minPrefillTokens, so a worker holding a TTFT average but no
// rate has been measured on nothing longer than 256 tokens. Failing that the
// figure is the certification one, taken on the speed probe's ~30-token prompt.
// Either way it is overhead, not prefill work.
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
	if promptTokens <= 0 {
		// No request in hand (the pinned/debug paths, and nominalJob). There is no
		// prompt to scale by, so the flat average is all there is.
		return liveTTFT(b) + wire
	}
	if rate := b.ObservedPrefillTPS; rate > 0 {
		return float64(promptTokens)/rate + wire
	}
	if rate := contextProbePrefillTPS(b, promptTokens); rate > 0 {
		return float64(promptTokens)/rate + wire
	}
	return liveTTFT(b) + float64(promptTokens)/fallbackPrefillTPS + wire
}

// contextProbePrefillTPS reads a provisional prefill rate off the context ladder,
// for a worker whose own EWMA is still empty.
//
// It costs nothing to have: the ladder plants a needle in a haystack of KNOWN
// length and times the answer, so each rung already carries tokens ÷ TTFT for
// this worker (see contextprobe.go), and it runs at the end of every full profile
// and is persisted with it. That makes it a measured prefill curve at 4K, 8K, 16K
// and up — the sizes the flat-TTFT fallback was most wrong about, and the ones
// the founding goal cares about most.
//
// The rung NEAREST this request is used rather than one summary figure, because
// prefill rate falls with length (attention is not linear in it) and a 4K reading
// over-rates a 128K prompt. Nearest means nearest by RATIO, since the ladder
// doubles: for a 60K prompt, 64K is one rung away and 4K is four, though by
// absolute difference 4K would look the closer of the two on any worker that
// climbed past 120K.
//
// It reads LOW, deliberately. A rung's timing is a whole non-streamed call, so
// the ≤64 generated tokens sit inside the number — a real share of the call at
// 4K, noise at 64K. Erring pessimistic is the same call every other latency
// measurement here makes.
func contextProbePrefillTPS(b *Backend, promptTokens int) float64 {
	if b == nil || b.ContextProbe == nil || promptTokens <= 0 {
		return 0
	}
	best, nearest := 0.0, math.Inf(1)
	for _, lv := range b.ContextProbe.Levels {
		if lv.Errored || lv.PrefillTPS <= 0 || lv.Tokens <= 0 {
			continue
		}
		if d := math.Abs(math.Log(float64(lv.Tokens) / float64(promptTokens))); d < nearest {
			best, nearest = lv.PrefillTPS, d
		}
	}
	return best
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
