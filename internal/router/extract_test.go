package router

import (
	"strings"
	"testing"
)

// The two dialects in play: llama.cpp uses "reasoning_content", vLLM ≥0.23
// uses "reasoning", and either leaves "content" JSON-null when the reasoning
// parser consumed the whole output.
func TestCompletionTextDialects(t *testing.T) {
	cases := []struct {
		name          string
		raw           map[string]any
		wantContent   string
		wantReasoning string
		wantFinish    string
	}{
		{
			name: "llama.cpp reasoning_content",
			raw: map[string]any{"choices": []any{map[string]any{
				"finish_reason": "stop",
				"message":       map[string]any{"content": "answer", "reasoning_content": "thinking"},
			}}},
			wantContent: "answer", wantReasoning: "thinking", wantFinish: "stop",
		},
		{
			name: "vLLM 0.23 reasoning",
			raw: map[string]any{"choices": []any{map[string]any{
				"finish_reason": "stop",
				"message":       map[string]any{"content": "answer", "reasoning": "thinking"},
			}}},
			wantContent: "answer", wantReasoning: "thinking", wantFinish: "stop",
		},
		{
			name: "null content, all output in reasoning (unclosed think block)",
			raw: map[string]any{"choices": []any{map[string]any{
				"finish_reason": "stop",
				"message":       map[string]any{"content": nil, "reasoning": "…the answer is 60 km/h"},
			}}},
			wantContent: "", wantReasoning: "…the answer is 60 km/h", wantFinish: "stop",
		},
		{
			name:        "no choices",
			raw:         map[string]any{"choices": []any{}},
			wantContent: "", wantReasoning: "", wantFinish: "",
		},
	}
	for _, tc := range cases {
		content, reasoning, finish := completionText(tc.raw)
		if content != tc.wantContent || reasoning != tc.wantReasoning || finish != tc.wantFinish {
			t.Errorf("%s: got (%q, %q, %q), want (%q, %q, %q)",
				tc.name, content, reasoning, finish, tc.wantContent, tc.wantReasoning, tc.wantFinish)
		}
	}
}

// Null content must never render as the string "<nil>" — that non-empty
// garbage previously passed the chat probe's emptiness check.
func TestMessageTextNullSafe(t *testing.T) {
	content, reasoning := messageText(map[string]any{"content": nil})
	if content != "" || reasoning != "" {
		t.Errorf("null content: got (%q, %q), want empty", content, reasoning)
	}
	if got := preferContent(content, reasoning); got != "" {
		t.Errorf("preferContent on null message = %q, want empty", got)
	}
}

func TestPreferContentFallsBackToReasoning(t *testing.T) {
	if got := preferContent("  ", "the real answer"); got != "the real answer" {
		t.Errorf("preferContent = %q, want reasoning fallback", got)
	}
	if got := preferContent("content wins", "reasoning"); got != "content wins" {
		t.Errorf("preferContent = %q, want content", got)
	}
}

func TestBoundedCapture(t *testing.T) {
	c := &boundedCapture{headCap: 10, tailCap: 10}
	// Fits entirely in the head: exact.
	c.Write([]byte("hello"))
	if got := c.String(); got != "hello" {
		t.Fatalf("small capture = %q", got)
	}
	// Overflow: head keeps the start, tail keeps the end, marker in between.
	c.Write([]byte(strings.Repeat("x", 100)))
	c.Write([]byte("THE-END"))
	out := c.String()
	if !strings.HasPrefix(out, "helloxxxxx") {
		t.Errorf("capture lost the head: %q", out)
	}
	if !strings.HasSuffix(out, "THE-END") {
		t.Errorf("capture lost the tail (the final answer!): %q", out)
	}
	if !strings.Contains(out, "truncated") {
		t.Errorf("capture missing omission marker: %q", out)
	}
	if c.truncated() <= 0 {
		t.Errorf("truncated() = %d, want > 0", c.truncated())
	}
}

func TestSSEStats(t *testing.T) {
	s := &sseStats{}
	// Feed in split writes to exercise the partial-line buffer, mixing dialects.
	chunks := []string{
		": heartbeat\n\n",
		`data: {"choices":[{"delta":{"content":"Hi"}}]}` + "\n",
		`data: {"choices":[{"delta":{"reasoning_content":"think"}}]}` + "\n",
		`data: {"choices":[{"delta":{"reason`, `ing":"more"}}]}` + "\n",
		`data: {"choices":[{"delta":{"content":""}}]}` + "\n", // empty delta — not a token
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0}]},"finish_reason":null}]}` + "\n",
		`data: {"choices":[{"delta":{},"finish_reason":"length"}]}` + "\n",
		"data: [DONE]\n",
	}
	for _, ch := range chunks {
		s.Write([]byte(ch))
	}
	if s.tokens != 3 {
		t.Errorf("tokens = %d, want 3 (content + both reasoning dialects)", s.tokens)
	}
	if !s.sawToolCall {
		t.Error("sawToolCall = false, want true")
	}
	if !s.sawFinishLength {
		t.Error("sawFinishLength = false, want true")
	}
	if !s.inadequate() {
		t.Error("inadequate() = false, want true (finish_reason length)")
	}

	empty := &sseStats{}
	empty.Write([]byte(": ping\n\n"))
	if !empty.inadequate() {
		t.Error("heartbeats-only stream should be inadequate")
	}
	toolOnly := &sseStats{}
	toolOnly.Write([]byte(`data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}` + "\n"))
	if toolOnly.inadequate() {
		t.Error("tool-call-only stream should be adequate")
	}
}

// TestSSEStatsGenTokens: spec-decode workers pack a whole accepted MTP step
// into ONE delta (measured on Qwen3.8-27B 2026-08-15: 52 tokens in 25 deltas),
// so counting deltas halved their live decode EWMA. genTokens must prefer an
// exact usage count when the client requested one, and otherwise floor the
// estimate with chars/4.8 so multi-token deltas can't under-read.
func TestSSEStatsGenTokens(t *testing.T) {
	// Single-token deltas, no usage: delta count is exact and must win over the
	// char estimate ("Hi" etc. are short — estimate stays below the count).
	single := &sseStats{}
	for _, tok := range []string{"Hi", " to", " you"} {
		single.Write([]byte(`data: {"choices":[{"delta":{"content":"` + tok + `"}}]}` + "\n"))
	}
	if got := single.genTokens(); got != 3 {
		t.Errorf("single-token deltas: genTokens = %d, want 3", got)
	}

	// Multi-token deltas (spec decode), no usage: 2 deltas carrying ~12 tokens
	// of prose; the char estimate must overrule the delta count.
	mtp := &sseStats{}
	mtp.Write([]byte(`data: {"choices":[{"delta":{"content":"Reliable local routing keeps traffic"}}]}` + "\n"))
	mtp.Write([]byte(`data: {"choices":[{"delta":{"content":" on the fastest healthy worker available."}}]}` + "\n"))
	if got := mtp.genTokens(); got <= 2 {
		t.Errorf("multi-token deltas: genTokens = %d, want char-based estimate > 2", got)
	}

	// A usage chunk (cumulative, final) is exact and wins over both.
	withUsage := &sseStats{}
	withUsage.Write([]byte(`data: {"choices":[{"delta":{"content":"Reliable local routing keeps traffic flowing"}}]}` + "\n"))
	withUsage.Write([]byte(`data: {"choices":[],"usage":{"prompt_tokens":30,"completion_tokens":52}}` + "\n"))
	if got := withUsage.genTokens(); got != 52 {
		t.Errorf("usage chunk: genTokens = %d, want 52", got)
	}
}

// vLLM serializes "tool_calls":[] on EVERY buffered message; an empty response
// on that dialect must still be detected as inadequate.
func TestResponseInadequateVLLMEmptyToolCalls(t *testing.T) {
	body := []byte(`{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":null,"tool_calls":[]}}]}`)
	if !responseInadequate(body, false) {
		t.Error("empty vLLM response with tool_calls:[] not flagged inadequate")
	}
	withCall := []byte(`{"choices":[{"finish_reason":"tool_calls","message":{"content":null,"tool_calls":[{"id":"x","function":{"name":"f","arguments":"{}"}}]}}]}`)
	if responseInadequate(withCall, false) {
		t.Error("real tool call flagged inadequate")
	}
	reasoningOnly := []byte(`{"choices":[{"finish_reason":"stop","message":{"content":null,"reasoning":"worked it out","tool_calls":[]}}]}`)
	if responseInadequate(reasoningOnly, false) {
		t.Error("reasoning-only response flagged inadequate")
	}
}

func TestEffectiveMaxTokens(t *testing.T) {
	if got := effectiveMaxTokens(&ChatRequest{MaxTokens: 100}); got != 100 {
		t.Errorf("max_tokens only = %d, want 100", got)
	}
	if got := effectiveMaxTokens(&ChatRequest{MaxTokens: 100, MaxCompletionTokens: 200}); got != 200 {
		t.Errorf("max_completion_tokens should win = %d, want 200", got)
	}
	if got := effectiveMaxTokens(&ChatRequest{}); got != 0 {
		t.Errorf("unset = %d, want 0", got)
	}
}

func TestParseChatRequestBudgets(t *testing.T) {
	// max_completion_tokens flows into the effective budget.
	req, err := parseAndValidateChatRequest([]byte(`{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":9000}`), 4096)
	if err != nil {
		t.Fatal(err)
	}
	if req.MaxTokens != 9000 || !req.ClientSetMaxTokens {
		t.Errorf("max_completion_tokens: MaxTokens=%d ClientSet=%v, want 9000/true", req.MaxTokens, req.ClientSetMaxTokens)
	}
	// null max_tokens = unset → default, and marked not-client-set.
	req, err = parseAndValidateChatRequest([]byte(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":null}`), 4096)
	if err != nil {
		t.Fatal(err)
	}
	if req.MaxTokens != 4096 || req.ClientSetMaxTokens {
		t.Errorf("null max_tokens: MaxTokens=%d ClientSet=%v, want 4096/false", req.MaxTokens, req.ClientSetMaxTokens)
	}
}

// null/0 max_tokens in the body must be overwritten when injecting the default
// (the router previously budgeted with the default while forwarding null).
func TestPatchForwardedBodyNullMaxTokens(t *testing.T) {
	out := patchForwardedBody([]byte(`{"max_tokens":null}`), 4096, 0, thinkingResolution{}, "")
	if !strings.Contains(string(out), `"max_tokens":4096`) {
		t.Errorf("null max_tokens not replaced: %s", out)
	}
	out = patchForwardedBody([]byte(`{"max_tokens":0}`), 4096, 0, thinkingResolution{}, "")
	if !strings.Contains(string(out), `"max_tokens":4096`) {
		t.Errorf("zero max_tokens not replaced: %s", out)
	}
	// A real client budget is left alone (caller passes 0 in that case).
	out = patchForwardedBody([]byte(`{"max_tokens":7}`), 0, 0, thinkingResolution{}, "")
	if !strings.Contains(string(out), `"max_tokens":7`) {
		t.Errorf("client max_tokens clobbered: %s", out)
	}
}
