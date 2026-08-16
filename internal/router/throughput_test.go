package router

import "testing"

func TestCountSSETokensCountsReasoning(t *testing.T) {
	// A turn that reasons at length then gives a short answer: every chunk
	// (reasoning or content) must count, so throughput reflects real generation
	// speed and a thinking model isn't mistaken for a slow one (which would poison
	// the completion-time ranking).
	data := []byte(
		`data: {"choices":[{"delta":{"reasoning_content":"let"}}]}` + "\n" +
			`data: {"choices":[{"delta":{"reasoning_content":"me"}}]}` + "\n" +
			`data: {"choices":[{"delta":{"reasoning_content":"think"}}]}` + "\n" +
			`data: {"choices":[{"delta":{"content":"Hi"}}]}` + "\n" +
			`data: {"choices":[{"delta":{"content":""}}]}` + "\n" +
			`data: [DONE]` + "\n")
	if got := countSSETokens(data); got != 4 {
		t.Errorf("countSSETokens = %d, want 4 (3 reasoning + 1 content; empty + [DONE] ignored)", got)
	}
}
