package router

// Shared assistant-text extraction across the two OpenAI-compatible dialects
// the router fronts: llama.cpp puts a model's reasoning block in
// "reasoning_content" while vLLM ≥0.23 renamed it to "reasoning" (and either
// may leave "content" JSON-null when the reasoning parser consumed the whole
// output, e.g. an unterminated <think> block). Every place that interprets a
// completion body MUST go through these helpers so the next field rename
// breaks one function, not five — the 2026-07-06 incident was thinkingProbe
// alone missing the rename, mis-certifying the GPU worker as non-thinking and
// routing chat to a CPU fallback.

import "strings"

// messageText pulls the assistant text out of a non-streamed completion
// message object: the content plus whichever reasoning field the dialect
// uses. Null-safe — a JSON null or absent field yields "", never "<nil>".
func messageText(msg map[string]any) (content, reasoning string) {
	if msg == nil {
		return "", ""
	}
	content, _ = msg["content"].(string)
	if rc, _ := msg["reasoning_content"].(string); rc != "" {
		reasoning = rc
	} else if rc, _ := msg["reasoning"].(string); rc != "" {
		reasoning = rc
	}
	return content, reasoning
}

// completionText extracts the first choice's content, reasoning, and
// finish_reason from a raw non-streamed chat completion response.
func completionText(raw map[string]any) (content, reasoning, finishReason string) {
	choices, _ := raw["choices"].([]any)
	if len(choices) == 0 {
		return "", "", ""
	}
	ch, _ := choices[0].(map[string]any)
	finishReason, _ = ch["finish_reason"].(string)
	msg, _ := ch["message"].(map[string]any)
	content, reasoning = messageText(msg)
	return content, reasoning, finishReason
}

// preferContent returns the content when it carries text, else the reasoning.
// The fallback matters for thinking models that never close their reasoning
// block: the parser then leaves content empty and the whole answer — including
// the conclusion — sits in the reasoning field. Callers that must NOT see
// reasoning (e.g. strict JSON parsing) should use the fields directly.
func preferContent(content, reasoning string) string {
	if strings.TrimSpace(content) != "" {
		return content
	}
	return reasoning
}

// sseDelta is the delta payload of one streamed chat chunk across both
// dialects.
type sseDelta struct {
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content"`
	Reasoning        string `json:"reasoning"`
}

func (d sseDelta) reasoningText() string {
	if d.ReasoningContent != "" {
		return d.ReasoningContent
	}
	return d.Reasoning
}
