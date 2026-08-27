package router

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"strings"
	"sync"
	"time"
)

// Background answer judging (LLM-as-judge). A sampled fraction of served answers
// is graded in the background by another worker in the fleet; a "bad" verdict
// raises that difficulty bin's tier bias in the online adapter, and BOTH verdicts
// are filed in the outcome matrix.
//
// This is the quality signal the adapter otherwise lacks: responseInadequate only
// catches truncated/empty replies, so a fast-but-dumb model that returns a
// complete-but-wrong answer reads as success and is silently trusted. Grading a
// sample against a second opinion closes that loop — and under completion-time
// routing, raising a bin's floor is exactly what kicks a cheap-fast model out of
// the prompts it keeps getting wrong.
//
// It is also the ONLY evidence that will ever cover the traffic actually being
// served. The question bank is LiveBench maths and code; production is agent
// loops and chat. Every row the matrix holds about a real prompt got there
// through recordJudgedOutcome below, which is why "which answers are eligible to
// be graded" is a routing question and not a housekeeping one — see judgeGrader.

const (
	judgeMaxConcurrent = 2    // cap background judge calls in flight
	judgeMaxTokens     = 200  // the verdict is one word; keep it cheap
	judgeMaxChars      = 2000 // cap question/answer length fed to the judge
)

// judgeMaxCallTokens is the most one judge call can cost: both texts are capped
// at judgeMaxChars and the verdict at judgeMaxTokens. Three characters per token
// is the pessimistic end of the range across the fleet's tokenisers, which is
// the right end for a spending bound.
const judgeMaxCallTokens = 2*judgeMaxChars/3 + judgeMaxTokens

// judgePaidTokenBudget is how much the judge may spend on a PAID grader per
// judgePaidBudgetWindow, once no free model in the fleet is good enough to
// grade with.
//
// Denominated in TOKENS, not money. A price is per-endpoint and in whatever
// currency the operator is billed in, so a money cap is a number only the
// operator could set — and PLAN.md's first design principle is that the router
// does not hand them numbers to set. Tokens are the unit it already counts:
// per-key budgets are in tokens for the same reason.
//
// The size is derived rather than picked. One call costs at most
// judgeMaxCallTokens (~1.5k), so this is ~78 graded answers an hour, and 78 is
// the number the ADAPTER needs: adaptMaxBias/adaptLRUp is eight BAD verdicts to
// drive one difficulty bin to its ceiling, and there are adaptBins of them, so
// ~80 verdicts is one full sweep of the curve. An hour buys the adapter
// everything it can learn; past that the judge is repeating itself, and
// repeating itself on someone's invoice.
const (
	judgePaidTokenBudget  = 120_000
	judgePaidBudgetWindow = time.Hour
)

// judgeBudget is the rolling allowance above. Its zero value is usable, so a
// hand-built Router (every test) grades without constructing anything.
//
// allow() does not RESERVE, it only reports: the charge lands afterwards, from
// what the endpoint says it used. A burst can therefore overshoot by the calls
// already in flight — at most judgeMaxConcurrent of them — which is the same
// trade the per-key budget makes. A budget is a spending bound, not an invoice.
type judgeBudget struct {
	mu      sync.Mutex
	spent   int
	resetAt time.Time
	logged  bool // at most one "budget spent" line per window
}

func (b *judgeBudget) allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if now := time.Now(); now.After(b.resetAt) {
		b.spent, b.resetAt, b.logged = 0, now.Add(judgePaidBudgetWindow), false
	}
	if b.spent < judgePaidTokenBudget {
		return true
	}
	if !b.logged {
		b.logged = true
		log.Printf("judge: paid grading budget spent (%d tokens this %s) — sampled grading pauses until the window rolls over",
			b.spent, judgePaidBudgetWindow)
	}
	return false
}

func (b *judgeBudget) charge(tokens int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.spent += tokens
}

// maybeJudge samples a served answer and, in the background, has another worker
// grade it, feeding the verdict into the tier adapter and the outcome matrix.
// Non-blocking; a no-op unless judging is enabled and some worker other than the
// one that served the request is available to look at it.
// thinking and latencyMS describe the request being graded, and exist so the
// verdict can be recorded in the outcome matrix against the right (worker, mode)
// pair. Without the mode a judged result would be filed against whichever mode
// the matrix happened to be queried for, which is the one thing goal 3 forbids.
func (r *Router) maybeJudge(messages []Message, stream bool, served *Backend, route, output string, thinking bool, latencyMS int64, vec []float64) {
	// The adapter is NO LONGER required. Judging serves two consumers now: the
	// tier adapter, which needs a numeric difficulty score and therefore only
	// learns from tier-routed requests, and the outcome matrix, which needs
	// neither. Requiring the adapter here is what kept the matrix from ever
	// seeing a judged answer on the path that routes most traffic.
	if served == nil || r.judgeSem == nil || r.cfg == nil || r.cfg.JudgeSampleRate <= 0 {
		return
	}
	n := uint64(math.Round(1 / r.cfg.JudgeSampleRate))
	if n < 1 {
		n = 1
	}
	if r.judgeCount.Add(1)%n != 0 {
		return // not in this sample
	}
	grader, paid := r.judgeGrader(served, vec)
	// The question here is "is there anyone else who can look at this", NOT "was
	// the served worker beaten by someone". Those were the same question under
	// the quality scalar and have not been since the outcome matrix became the
	// routing policy: the test that used to sit here, `best.Quality <= served
	// .Quality`, is unsatisfiable for the fleet's own best worker and so excluded
	// it from judging permanently. judgeGrader carries what that cost.
	if grader == nil || grader.ID == served.ID {
		return // nobody else in the fleet can offer a second opinion on this answer
	}
	if paid && !r.judgePaid.allow() {
		return // no free model can grade this and the paid allowance is spent
	}
	question := lastUserText(messages)
	answer := extractAnswer([]byte(output), stream)
	if strings.TrimSpace(question) == "" || strings.TrimSpace(answer) == "" {
		return
	}
	select {
	case r.judgeSem <- struct{}{}:
	default:
		return // too many judges already in flight; skip this sample
	}
	go func() {
		defer func() { <-r.judgeSem }()
		bad, ok := r.askJudge(grader, question, answer)
		if !ok {
			return
		}
		// The tier adapter only understands a difficulty score, which exists only
		// on a tier route. On a matrix route the verdict still lands in the matrix
		// below — that is the whole point of grading it.
		if bad {
			if score, ok := parseRouteScore(route); ok && r.adapter != nil {
				r.adapter.observe(score, true)
				log.Printf("judge: %s answer for d=%.2f graded BAD by %s → raised tier bias", served.ID, score, grader.ID)
			} else {
				log.Printf("judge: %s answer graded BAD by %s (route %q)", served.ID, grader.ID, route)
			}
		}
		// BOTH verdicts are recorded. The tier adapter only ever cared about bad
		// ones — it is a correction signal — but the matrix is an estimate of a
		// hit rate, and throwing away every success would make every judged
		// worker look uniformly terrible on real traffic.
		r.recordJudgedOutcome(served.ID, question, thinking, !bad, latencyMS)
	}()
}

// askJudge asks the judge backend whether answer adequately addresses question.
func (r *Router) askJudge(judge *Backend, question, answer string) (bad, ok bool) {
	payload := map[string]any{
		"model":                probeModel(judge),
		"stream":               false,
		"max_tokens":           judgeMaxTokens,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
		"messages": []map[string]string{
			{"role": "system", "content": "You grade another assistant's answer. Reply with exactly one word: GOOD if it correctly and usefully addresses the question, or BAD if it is wrong, irrelevant, incomplete, or unhelpful. /no_think"},
			{"role": "user", "content": fmt.Sprintf("Question:\n%s\n\nAnswer:\n%s\n\nVerdict (GOOD or BAD):", truncate(question, judgeMaxChars), truncate(answer, judgeMaxChars))},
		},
	}
	raw, err := r.rawCompletion(judge, payload)
	if err != nil {
		log.Printf("judge call to %s failed: %v", judge.ID, err)
		return false, false
	}
	// Charge before reading the verdict: a reply we cannot parse still cost money.
	if !isFreeBackend(judge) {
		r.judgePaid.charge(judgeCallTokens(raw))
	}
	content, reasoning, _ := completionText(raw)
	return parseJudgeVerdict(preferContent(content, reasoning))
}

// judgeCallTokens is what one judge call actually cost, read off the endpoint's
// own usage block. An endpoint that reports none is charged judgeMaxCallTokens
// — the worst case the call could have been — because a budget that reads
// "didn't say" as "free" is not a budget.
func judgeCallTokens(raw map[string]any) int {
	u, ok := raw["usage"].(map[string]any)
	if !ok {
		return judgeMaxCallTokens
	}
	if total, _ := u["total_tokens"].(float64); total > 0 {
		return int(total)
	}
	in, _ := u["prompt_tokens"].(float64)
	out, _ := u["completion_tokens"].(float64)
	if in+out > 0 {
		return int(in + out)
	}
	return judgeMaxCallTokens
}

// parseJudgeVerdict reads a GOOD/BAD verdict. ok is false when the reply is
// ambiguous (contains both words or neither) so it's ignored rather than guessed.
func parseJudgeVerdict(content string) (bad, ok bool) {
	s := strings.ToUpper(content)
	if i := strings.LastIndex(s, "</THINK>"); i >= 0 {
		s = s[i+len("</THINK>"):] // ignore any leaked reasoning block
	}
	hasBad := strings.Contains(s, "BAD")
	hasGood := strings.Contains(s, "GOOD")
	if hasBad == hasGood {
		return false, false
	}
	return hasBad, true
}

// judgeGrader picks the model that grades this answer: the strongest eligible
// chat worker OTHER than the one being graded, free preferred over paid. paid
// reports which, so the caller knows whether to charge the allowance.
//
// IT DOES NOT REQUIRE THE GRADER TO OUTRANK THE ANSWERER. That is the substance
// of this function, not an omission, and it is worth being explicit about why.
//
// The old bar was `Quality >`, applied here and again in maybeJudge. It is
// unsatisfiable for the fleet's own best worker — by definition nothing outranks
// it — so judgeGrader handed maybeJudge the worker itself and maybeJudge dropped
// the sample. Since recordJudgedOutcome is the only route by which production
// traffic reaches the outcome matrix, the best worker accumulated ZERO judged
// observations, permanently, in BOTH thinking modes. It then had no prediction
// for a real prompt, fell into the matrix's `unmeasured` band, and ranked behind
// any cheap worker with a known rate at or above outcomeCorrectFloor. A blind
// spot on exactly one worker, chosen by a number routing stopped consulting the
// day the matrix became the policy.
//
// Dropping the bar is right on the merits too. Grading is VERIFICATION;
// answering is GENERATION; the first is the easier task. A model that could not
// have produced the right answer can still see that a fluent, confident, wrong
// one does not address the question — which is precisely the failure
// responseInadequate cannot catch and the judge exists to catch. "Better than
// the answerer" was never the requirement; it was a proxy for "credible",
// borrowed from a scalar that no longer ranks anything.
//
// And it makes paid grading RARER, not commoner — which matters, because this is
// the one place cost leaks into a path nobody asked to pay for: the judge grades
// a sampled fraction of ordinary traffic, forever. Had the old rule simply been
// relaxed to let the best worker be graded, a fleet whose best worker is free
// and local would have sent every sample of its BUSIEST worker's traffic to a
// metered grader, since nothing free outranks it. Here a paid grader is reached
// only when the served worker is the sole free chat worker registered.
//
// eligible() already guarantees Healthy && Certification.Ready && !isExpired;
// only the embeddings-only skip and the never-itself rule are needed here.
func (r *Router) judgeGrader(served *Backend, vec []float64) (grader *Backend, paid bool) {
	if served == nil || r.registry == nil {
		return nil, false
	}
	var bestFree, bestPaid *Backend
	for _, b := range r.registry.eligible() {
		// Never the worker being graded. A model asked to mark its own answer has
		// already committed to it once and agrees with itself by construction, so
		// the verdict carries no information at any price.
		if b.ID == served.ID || isEmbeddingsOnly(b) {
			continue
		}
		if isFreeBackend(b) {
			if r.betterGrader(b, bestFree, vec) {
				bestFree = b
			}
		} else if r.betterGrader(b, bestPaid, vec) {
			bestPaid = b
		}
	}
	if bestFree != nil {
		return bestFree, false
	}
	if bestPaid != nil {
		return bestPaid, true
	}
	return nil, false
}

// betterGrader reports whether candidate should displace best.
func (r *Router) betterGrader(candidate, best *Backend, vec []float64) bool {
	if best == nil {
		return true
	}
	cs, bs := r.graderStrength(candidate, vec), r.graderStrength(best, vec)
	if cs != bs {
		return cs > bs
	}
	// eligible() walks a map, so a tie has to break on something stable. Left to
	// iteration order, the fleet's grader — and therefore what the matrix learns
	// from a run of identical requests — would vary from one sample to the next.
	return candidate.ID < best.ID
}

// graderStrength ranks a candidate grader on the evidence the router actually
// holds, as one comparable number.
//
// A worker the outcome matrix has MEASURED scores its hit rate, in [0,1]. One it
// has not scores its registered Quality mapped into [-1,0), so every measured
// worker outranks every unmeasured one and the retired scalar only ever breaks
// ties among the unmeasured — a fleet before its first profile has run, which is
// the one situation where it is still the best number available. Routing does
// not consult it and neither should this, except there.
//
// The NO-THINK rate, whatever mode produced the answer being graded: askJudge
// sends enable_thinking:false, so it is the no-think model that does the
// grading. Founding goal 3 treats the two modes as separate models, and reading
// a mixed-mode score here would rank a grader on a model nobody is about to
// call. qualityFor carries the same rule for the scalar fallback.
//
// Per-prompt when the router has the prompt's embedding, overall otherwise.
//
// "How would this worker do on a prompt LIKE THIS ONE" is the better question
// than "how does it do in general", and the matrix can answer it — a worker that
// is mediocre overall but strong on this topic is the right grader for this
// answer. The vector is the classification's, computed once for routing and
// passed down rather than re-derived: re-embedding here would either double the
// embedding cost of every sampled request or plant a vector with no observations
// behind it, which neighboursOf would then count towards outcomeNeighbours while
// supplying no evidence to anyone.
//
// It falls back to the overall rate when the request was not classified (no
// embeddings worker) or the matrix has no neighbours for this prompt — which is
// exactly the "no neighbours to consult" case summary() documents itself for —
// and to the retired scalar only before any worker has been profiled. The scalar
// is shifted below zero so that ANY measured evidence outranks it: an unprofiled
// worker should never beat one the router has actually observed.
func (r *Router) graderStrength(b *Backend, vec []float64) float64 {
	if r.outcomes != nil {
		if len(vec) > 0 {
			if p := r.outcomes.predict(vec, b.ID, false); p.known() {
				return p.Correct
			}
		}
		if rate, n := r.outcomes.summary(b.ID, false); n > 0 {
			return rate
		}
	}
	return float64(qualityFor(b, true))/100 - 1
}

// lastUserText returns the text of the most recent user message.
func lastUserText(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return contentText(messages[i].Content)
		}
	}
	return ""
}

// extractAnswer pulls the assistant's answer text (content only, not reasoning)
// from a buffered completion response, streamed or not.
func extractAnswer(body []byte, stream bool) string {
	if !stream {
		var resp struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if json.Unmarshal(body, &resp) == nil && len(resp.Choices) > 0 {
			return resp.Choices[0].Message.Content
		}
		return ""
	}
	var sb strings.Builder
	for _, line := range bytes.Split(body, []byte("\n")) {
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		payload := line[6:]
		if bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		var ev struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal(payload, &ev) == nil && len(ev.Choices) > 0 {
			sb.WriteString(ev.Choices[0].Delta.Content)
		}
	}
	return sb.String()
}

// classificationVec is the prompt embedding a classification carries, or nil.
//
// A helper rather than an inline `if plan.cl != nil` at the call site because
// "missing" has to stay distinguishable from "present but empty": both mean the
// judge falls back to the overall rate, and neither should ever panic on a
// request the embeddings worker never saw.
func classificationVec(cl *classification) []float64 {
	if cl == nil {
		return nil
	}
	return cl.vec
}
