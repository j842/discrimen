package router

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"strings"
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

// maybeJudge samples a cheaper-than-best answer and, in the background, grades it
// with the best model, feeding the verdict into the tier adapter. Non-blocking;
// a no-op unless judging is enabled and a better model than the one that served
// the request exists.
func (r *Router) maybeJudge(messages []Message, stream bool, served *Backend, score float64, output string) {
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
	best := r.bestChatBackend()
	if best == nil || best.ID == served.ID || best.Quality <= served.Quality {
		return // served by (or as good as) the best model — nothing better to grade with
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
		if bad, ok := r.askJudge(best, question, answer); ok && bad {
			r.adapter.observe(score, true)
			log.Printf("judge: %s answer for d=%.2f graded BAD by %s → raised tier bias", served.ID, score, best.ID)
		}
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
	content, err := r.simpleCompletion(judge, payload)
	if err != nil {
		log.Printf("judge call to %s failed: %v", judge.ID, err)
		return false, false
	}
	return parseJudgeVerdict(content)
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

// bestChatBackend returns the highest-quality healthy chat backend, or nil.
func (r *Router) bestChatBackend() *Backend {
	var best *Backend
	for _, b := range r.registry.eligible() {
		// eligible() already guarantees Healthy && Certification.Ready && !isExpired;
		// only the embeddings-only skip is still needed here.
		if isEmbeddingsOnly(b) {
			continue
		}
		if best == nil || b.Quality > best.Quality {
			best = b
		}
	}
	return best
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
