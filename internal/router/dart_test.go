package router

import (
	"strings"
	"testing"
)

// The single most dangerous mistake in this file: sampling the drafts greedily.
// At temperature 0 both drafts are the same string by construction, so the gate
// fires on every prompt, serves an unchecked answer, and every metric looks
// excellent. This pins the constant that prevents it.
func TestDraftsMustBeSampled(t *testing.T) {
	if draftTemperature <= 0 {
		t.Fatalf("draftTemperature is %v — at zero the drafts are identical by "+
			"construction, so agreement measures nothing while appearing perfect",
			draftTemperature)
	}
	if draftCount < 2 {
		t.Fatalf("draftCount is %d; agreement needs at least two samples", draftCount)
	}
}

// Multiple choice must be excluded: with a handful of options two independent
// drafts agree by chance often enough to destroy the signal.
func TestLooksMultipleChoice(t *testing.T) {
	mcq := []string{
		"What is the value of x?\n\nA) 1\nB) 2\nC) 3\nD) 4",
		"Pick one:\n(A) north\n(B) south\n(C) east",
		"Question text\nA. alpha\nB. beta\nC. gamma",
	}
	for _, p := range mcq {
		if !looksMultipleChoice(p) {
			t.Errorf("multiple choice not detected:\n%s", p)
		}
	}
	notMCQ := []string{
		"What is 17 times 23?",
		"Summarise this email and tell me if I need to reply.",
		"Write a function that reverses a linked list.",
		// A single lettered line is a list item, not an option set.
		"Steps:\nA) start the server\nthen check the logs",
		// Prose mentioning options must not trip it.
		"Explain why option A is better than the alternatives.",
	}
	for _, p := range notMCQ {
		if looksMultipleChoice(p) {
			t.Errorf("false positive on non-MCQ prompt: %q", p)
		}
	}
}

// Two samples of the same answer differ in formatting, not content. Exact match
// on a normalised form is the cheap path and must not be defeated by casing,
// markdown or a full stop.
func TestNormaliseDraft(t *testing.T) {
	same := []struct{ a, b string }{
		{"42", "42."},
		{"The answer is 42", "the answer is 42"},
		{"**42**", "42"},
		{" 42 \n", "42"},
		{"the  answer   is 42", "the answer is 42"},
	}
	for _, c := range same {
		if normaliseDraft(c.a) != normaliseDraft(c.b) {
			t.Errorf("%q and %q should normalise alike, got %q and %q",
				c.a, c.b, normaliseDraft(c.a), normaliseDraft(c.b))
		}
	}
	if normaliseDraft("42") == normaliseDraft("43") {
		t.Error("different answers normalised to the same string")
	}
}

// The gate must decline rather than run in every case where its evidence would
// be meaningless or its answer unrecallable.
func mcqRequest() *ChatRequest {
	return &ChatRequest{Messages: []Message{{Role: "user",
		Content: "Pick:\nA) one\nB) two\nC) three"}}}
}

func TestDraftGateDeclines(t *testing.T) {
	r := &Router{cfg: &Config{DraftGating: true}}
	b := testBackend("w", 100)
	msg := []Message{{Role: "user", Content: "What is 17 times 23?"}}

	cases := []struct {
		name string
		req  *ChatRequest
		tr   thinkingResolution
	}{
		{"streamed request cannot be recalled", &ChatRequest{Stream: true, Messages: msg},
			thinkingResolution{softThink: true}},
		{"caller demanded thinking", &ChatRequest{Messages: msg},
			thinkingResolution{hardThink: true}},
		{"caller demanded no thinking", &ChatRequest{Messages: msg},
			thinkingResolution{noThink: true}},
		{"router did not choose the mode", &ChatRequest{Messages: msg},
			thinkingResolution{}},
		{"multiple choice", mcqRequest(), thinkingResolution{softThink: true}},
		{"no user text", &ChatRequest{Messages: nil}, thinkingResolution{softThink: true}},
	}
	for _, c := range cases {
		if v := r.draftGate(t.Context(), b, c.req, c.tr); v.Ran {
			t.Errorf("%s: gate ran when it should have declined", c.name)
		}
	}
	// ...and declines entirely when disabled, which is the default.
	off := &Router{cfg: &Config{DraftGating: false}}
	if v := off.draftGate(t.Context(), b, &ChatRequest{Messages: msg},
		thinkingResolution{softThink: true}); v.Ran {
		t.Error("gate ran while disabled")
	}
}

// An agreed draft must come back in the ordinary chat-completion shape, or a
// gated request would look different to every client.
func TestDraftAnswerBody(t *testing.T) {
	body := string(draftAnswerBody("qwen", "391"))
	for _, want := range []string{`"object":"chat.completion"`, `"model":"qwen"`,
		`"role":"assistant"`, `"content":"391"`, `"finish_reason":"stop"`} {
		if !strings.Contains(body, want) {
			t.Errorf("draft response missing %s:\n%s", want, body)
		}
	}
}

// A failed draft is not a disagreement — it is no measurement, and must not be
// reported as evidence that the prompt was hard.
func TestEmptyDraftIsNotDisagreement(t *testing.T) {
	r := &Router{}
	if sim, agreed := r.draftsAgree(t.Context(), []string{"only one"}); agreed || sim != 0 {
		t.Error("a single draft cannot agree with anything")
	}
	// Identical drafts agree without needing an embedder at all, which is what
	// keeps the cheap path cheap.
	if sim, agreed := r.draftsAgree(t.Context(), []string{"42", "42."}); !agreed || sim != 1 {
		t.Errorf("identical answers did not agree (sim=%.2f)", sim)
	}
	// Different drafts with no embedder available must fail closed — erring
	// towards thinking costs latency, erring the other way serves an unchecked
	// answer.
	if _, agreed := r.draftsAgree(t.Context(), []string{"42", "17"}); agreed {
		t.Error("different drafts agreed with no embedder to compare them")
	}
}
