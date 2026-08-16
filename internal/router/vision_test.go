package router

import (
	"errors"
	"fmt"
	"testing"
)

// rcErr fabricates an error in the exact shape rawCompletion produces for a
// non-2xx response ("completion returned %d: %s", main.go).
func rcErr(status int, body string) error {
	return fmt.Errorf("completion returned %d: %s", status, body)
}

func TestCompletionStatusCode(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want int
	}{
		{"nil error", nil, 0},
		{"plain 429", rcErr(429, "engine overloaded"), 429},
		{"plain 400", rcErr(400, "image input not supported"), 400},
		{"empty body", rcErr(503, ""), 503},
		// The status must come from the anchored prefix, never from digits that
		// happen to sit inside the error body (the original bug's failure mode).
		{"status-like digits in body", rcErr(503, "upstream returned 400 Bad Request"), 503},
		{"transport error", errors.New(`Post "http://10.0.0.7:8080/v1/chat/completions": connection refused`), 0},
		{"json decode error", errors.New("invalid character '<' looking for beginning of value"), 0},
		{"similar prefix without code", errors.New("completion returned no choices"), 0},
		{"prefix not at start", errors.New(`proxy: completion returned 404: not found`), 0},
	}
	for _, c := range cases {
		if got := completionStatusCode(c.err); got != c.want {
			t.Errorf("%s: completionStatusCode(%v)=%d, want %d", c.name, c.err, got, c.want)
		}
	}
}

func TestIsClientReject(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		// Transient 4xx — load/timing conditions, not verdicts on image support.
		{"408 request timeout", rcErr(408, "request timed out"), false},
		{"425 too early", rcErr(425, "too early"), false},
		{"429 overloaded", rcErr(429, "engine overloaded, try again"), false},
		// Definitive 4xx — the worker rejected the image payload itself.
		{"400 bad request", rcErr(400, "this model does not support image input"), true},
		{"404 not found", rcErr(404, "no such route"), true},
		{"422 unprocessable", rcErr(422, "image_url content not accepted"), true},
		// 5xx and non-HTTP errors are never client rejections.
		{"503 unavailable", rcErr(503, "model loading"), false},
		{"500 internal", rcErr(500, "worker crashed"), false},
		// "returned 4" inside the body must not classify a 5xx as a rejection
		// (the loose substring match this replaced would have).
		{"503 with returned-4xx body", rcErr(503, `proxy said: upstream returned 429 Too Many Requests`), false},
		{"transport error", errors.New(`Post "http://10.0.0.7:8080/v1/chat/completions": connection refused`), false},
		{"unparseable completion error", errors.New("completion returned no choices"), false},
		{"nil error", nil, false},
	}
	for _, c := range cases {
		if got := isClientReject(c.err); got != c.want {
			t.Errorf("%s: isClientReject(%v)=%v, want %v", c.name, c.err, got, c.want)
		}
	}
}

// llamaCppNoVisionBody is the verbatim response a text-only llama.cpp server gives
// for image content — captured from llm-6000pro-deepseek-284B-q8. It arrives as a
// 500, which isClientReject treats as transient, so without isVisionUnsupported the
// probe exhausts its retries and reports inconclusive.
const llamaCppNoVisionBody = `{"error":{"code":500,"message":"image input is not supported - hint: if this is unexpected, you may need to provide the mmproj","type":"server_error"}}`

func TestIsVisionUnsupported(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		// The case this was written for: a permanent verdict wearing a 500.
		{"llama.cpp 500 verbatim", rcErr(500, llamaCppNoVisionBody), true},
		{"vLLM 400 phrasing", rcErr(400, "this model does not support image input"), true},
		{"mmproj hint alone", rcErr(500, "you may need to provide the mmproj"), true},
		{"no multimodal support", rcErr(501, "no multimodal support in this build"), true},
		{"case insensitive", rcErr(500, "IMAGE INPUT IS NOT SUPPORTED"), true},

		// Must stay transient — these are faults, not capability statements. A
		// worker that crashed or is still loading has to be retried, not written
		// off as text-only forever.
		{"generic 500", rcErr(500, "worker crashed"), false},
		{"503 model loading", rcErr(503, "model loading"), false},
		{"429 overloaded", rcErr(429, "engine overloaded"), false},
		{"transport error", errors.New(`Post "http://10.0.0.7:8080/v1/chat/completions": connection refused`), false},
		{"nil error", nil, false},
	}
	for _, c := range cases {
		if got := isVisionUnsupported(c.err); got != c.want {
			t.Errorf("%s: isVisionUnsupported(%v)=%v, want %v", c.name, c.err, got, c.want)
		}
	}
}

// The two predicates must stay independent: isVisionUnsupported is additive and
// must not have loosened isClientReject's treatment of 5xx as retryable.
func TestVisionRejectPredicatesAreIndependent(t *testing.T) {
	err := rcErr(500, llamaCppNoVisionBody)
	if isClientReject(err) {
		t.Error("isClientReject must still treat a 500 as transient, even this one")
	}
	if !isVisionUnsupported(err) {
		t.Error("isVisionUnsupported must catch what isClientReject deliberately does not")
	}
}
