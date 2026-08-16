package router

import (
	"strings"
	"testing"
)

func TestParseJudgeVerdict(t *testing.T) {
	cases := []struct {
		in      string
		wantBad bool
		wantOK  bool
	}{
		{"GOOD", false, true},
		{"BAD", true, true},
		{"good", false, true},
		{"  bad\n", true, true},
		{"<think>hmm, mostly right</think> GOOD", false, true},
		{"The answer is BAD — it's wrong.", true, true},
		{"meh", false, false},                     // neither word
		{"GOOD parts but also BAD", false, false}, // both → ambiguous, ignored
	}
	for _, c := range cases {
		bad, ok := parseJudgeVerdict(c.in)
		if bad != c.wantBad || ok != c.wantOK {
			t.Errorf("parseJudgeVerdict(%q) = (%v,%v), want (%v,%v)", c.in, bad, ok, c.wantBad, c.wantOK)
		}
	}
}

func TestExtractAnswer(t *testing.T) {
	nonStream := []byte(`{"choices":[{"message":{"content":"The capital is Paris."}}]}`)
	if got := extractAnswer(nonStream, false); got != "The capital is Paris." {
		t.Errorf("non-stream: got %q", got)
	}
	// Streamed: content deltas concatenated, reasoning_content ignored.
	stream := []byte(
		`data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}` + "\n" +
			`data: {"choices":[{"delta":{"content":"4"}}]}` + "\n" +
			`data: {"choices":[{"delta":{"content":"2"}}]}` + "\n" +
			`data: [DONE]` + "\n")
	if got := extractAnswer(stream, true); got != "42" {
		t.Errorf("stream: got %q want 42", got)
	}
}

func TestLastUserText(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "reply"},
		{Role: "user", Content: "second"},
	}
	if got := lastUserText(msgs); got != "second" {
		t.Errorf("lastUserText = %q want second", got)
	}
	mm := []Message{{Role: "user", Content: []any{
		map[string]any{"type": "text", "text": "describe this"},
		map[string]any{"type": "image_url"},
	}}}
	if got := strings.TrimSpace(lastUserText(mm)); got != "describe this" {
		t.Errorf("multimodal lastUserText = %q want 'describe this'", got)
	}
}
