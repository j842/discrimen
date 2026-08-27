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

// Background answer judging (LLM-as-judge). A sampled fraction of answers served
// by a cheaper-than-best backend are graded by the best model in the background;
// a "bad" verdict raises that difficulty bin's tier bias in the online adapter.
//
// This is the quality signal the adapter otherwise lacks: responseInadequate only
// catches truncated/empty replies, so a fast-but-dumb model that returns a
// complete-but-wrong answer reads as success and is silently trusted. Grading a
// sample against the best model closes that loop — and under completion-time
// routing, raising a bin's floor is exactly what kicks a cheap-fast model out of
// the prompts it keeps getting wrong.

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

// maybeJudge samples a cheaper-than-best answer and, in the background, grades it
// with the best model, feeding the verdict into the tier adapter. Non-blocking;
// a no-op unless judging is enabled and a better model than the one that served
// the request exists.
// thinking and latencyMS describe the request being graded, and exist so the
// verdict can be recorded in the outcome matrix against the right (worker, mode)
// pair. Without the mode a judged result would be filed against whichever mode
// the matrix happened to be queried for, which is the one thing goal 3 forbids.
func (r *Router) maybeJudge(messages []Message, stream bool, served *Backend, score float64, output string, thinking bool, latencyMS int64) {
	if r.adapter == nil || r.judgeSem == nil || r.cfg.JudgeSampleRate <= 0 {
		return
	}
	n := uint64(math.Round(1 / r.cfg.JudgeSampleRate))
	if n < 1 {
		n = 1
	}
	if r.judgeCount.Add(1)%n != 0 {
		return // not in this sample
	}
	best, paid := r.judgeGrader(served)
	if best == nil || best.ID == served.ID || best.Quality <= served.Quality {
		return // served by (or as good as) the best model — nothing better to grade with
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
		bad, ok := r.askJudge(best, question, answer)
		if !ok {
			return
		}
		if bad {
			r.adapter.observe(score, true)
			log.Printf("judge: %s answer for d=%.2f graded BAD by %s → raised tier bias", served.ID, score, best.ID)
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

// judgeGrader picks the model that grades this answer: the best FREE chat
// backend, falling back to the best paid one only when no free model outranks
// the worker that served the answer. paid reports which, so the caller knows
// whether to charge the allowance.
//
// This is the one place in P4 where cost leaks into a path nobody asked to pay
// for. The judge grades a sampled fraction of ordinary traffic, forever, and
// the moment the best model in the fleet is a metered one that becomes a
// standing spend on every cheap answer the router serves. The free grader is
// preferred even when a paid model is better, because a verdict from a model
// that merely outranks the one being graded is the signal the adapter wants —
// the best possible grader is a nicety, not the requirement.
//
// eligible() already guarantees Healthy && Certification.Ready && !isExpired;
// only the embeddings-only skip is still needed here.
func (r *Router) judgeGrader(served *Backend) (grader *Backend, paid bool) {
	var bestFree, bestPaid *Backend
	for _, b := range r.registry.eligible() {
		if isEmbeddingsOnly(b) {
			continue
		}
		if isFreeBackend(b) {
			if bestFree == nil || b.Quality > bestFree.Quality {
				bestFree = b
			}
		} else if bestPaid == nil || b.Quality > bestPaid.Quality {
			bestPaid = b
		}
	}
	// "Good enough to grade with" is the same bar maybeJudge already applies to
	// the grader it is handed — better than the worker being graded. It is
	// re-tested here so that "no free model is good enough" can fall through to a
	// paid one rather than abandoning the sample.
	if bestFree != nil && served != nil && bestFree.ID != served.ID && bestFree.Quality > served.Quality {
		return bestFree, false
	}
	if bestPaid != nil {
		return bestPaid, true
	}
	return bestFree, false
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
