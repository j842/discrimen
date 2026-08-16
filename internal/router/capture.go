package router

// Streaming capture for request logging + online quality signals.
//
// The proxy used to buffer the ENTIRE raw SSE stream per request (every
// heartbeat and JSON envelope — 10-30x the token text) just to log a 16KB
// prefix of it, which both cost unbounded memory on long streams and stored
// exactly the wrong 16KB: the early heartbeats instead of the final answer.
// boundedCapture keeps the head AND tail of the stream within the log budget,
// and sseStats computes the signals that previously required the full body
// (token count, tool-call presence, finish_reason:length) incrementally as
// bytes flow through, so they stay exact even when the capture truncates.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// ── Mid-stream failure ──────────────────────────────────────────────────────

// writeSSEError reports a failure that happened AFTER the SSE preamble went
// out. By then the status code is spent: the client holds a 200 and an open
// event-stream, so truncating the body is all the router could otherwise do,
// and a truncated stream is indistinguishable from a short answer. The client
// sees a plausible reply that stops early and has no way to know it was robbed.
//
// So emit the failure as an OpenAI-shaped error event and close the stream
// properly with [DONE]. Both SDKs and hand-rolled readers surface a `data:`
// frame carrying "error" as an error rather than content.
//
// Nothing is written for a stream that ended cleanly, so a well-behaved worker's
// bytes still reach the client exactly as they arrived. Best-effort by
// construction — the connection may already be gone, and there is nothing left
// to report the failure to report a failure with.
func writeSSEError(w http.ResponseWriter, message string) {
	payload, err := json.Marshal(errorEnvelope{Error: errorBody{
		Message: message,
		Type:    errorTypeForStatus(http.StatusBadGateway),
	}})
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w, "data: %s\n\n", payload)
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	if fl, ok := w.(http.Flusher); ok {
		fl.Flush()
	}
}

// isEventStream reports whether the response the router forwarded is an SSE
// stream. It is the test for "the client is mid-stream": a worker that answered
// a stream:true request with a JSON error body never opened one, and injecting
// an SSE frame into that would corrupt the very error it is trying to relay.
func isEventStream(h http.Header) bool {
	return strings.Contains(strings.ToLower(h.Get("Content-Type")), "text/event-stream")
}

// boundedCapture is an io.Writer that retains the first headCap and last
// tailCap bytes written, discarding the middle. Bytes() joins them with an
// omission marker sized to fit within the log store's per-row body cap.
type boundedCapture struct {
	headCap, tailCap int
	head, tail       []byte
	total            int64
}

func newBoundedCapture(maxBody int) *boundedCapture {
	if maxBody <= 0 {
		maxBody = 16384
	}
	// Halve the budget between head and tail, leaving headroom for the omission
	// marker so the log store's insert-time clip (which keeps the head) never
	// re-truncates away the tail we deliberately kept.
	half := maxBody/2 - 48
	if half < 1024 {
		half = 1024
	}
	return &boundedCapture{headCap: half, tailCap: half}
}

func (c *boundedCapture) Write(p []byte) (int, error) {
	n := len(p)
	c.total += int64(n)
	if room := c.headCap - len(c.head); room > 0 {
		take := room
		if take > len(p) {
			take = len(p)
		}
		c.head = append(c.head, p[:take]...)
		p = p[take:]
	}
	if len(p) > 0 {
		c.tail = append(c.tail, p...)
		if over := len(c.tail) - c.tailCap; over > 0 {
			c.tail = append(c.tail[:0], c.tail[over:]...)
		}
	}
	return n, nil
}

// truncated returns how many bytes of the stream are NOT represented in Bytes().
func (c *boundedCapture) truncated() int64 {
	return c.total - int64(len(c.head)) - int64(len(c.tail))
}

func (c *boundedCapture) Bytes() []byte {
	if len(c.tail) == 0 {
		return c.head
	}
	gap := c.truncated()
	if gap <= 0 {
		return append(append([]byte(nil), c.head...), c.tail...)
	}
	marker := fmt.Sprintf("\n…[capture truncated: %d bytes omitted]…\n", gap)
	out := make([]byte, 0, len(c.head)+len(marker)+len(c.tail))
	out = append(out, c.head...)
	out = append(out, marker...)
	out = append(out, c.tail...)
	return out
}

func (c *boundedCapture) String() string { return string(c.Bytes()) }

// sseStats is an io.Writer that scans an SSE stream line-by-line as it flows
// through, computing the adapter/throughput signals over the FULL stream —
// countSSETokens over a truncated capture would undercount long generations.
type sseStats struct {
	partial         []byte
	tokens          int
	chars           int
	usageTokens     int
	sawToolCall     bool
	sawFinishLength bool
}

// genTokens is the generated-token count the throughput EWMA should use.
// Preference order:
//  1. usage.completion_tokens, when the client requested include_usage — exact.
//  2. max(delta count, chars/4.8). One-delta-per-token engines make the delta
//     count exact and it wins (code runs ~3.8 chars/token, so the estimate
//     stays below it). Spec-decode workers pack ~2.5 tokens into each delta —
//     measured on Qwen3.8-27B MTP 2026-08-15, 52 tokens in 25 deltas — which
//     halved their live EWMA; there the char estimate wins and bounds the
//     error at roughly ±25% (prose ~5.05 chars/token) instead of −60%.
func (s *sseStats) genTokens() int {
	if s.usageTokens > 0 {
		return s.usageTokens
	}
	if est := int(float64(s.chars) / 4.8); est > s.tokens {
		return est
	}
	return s.tokens
}

func (s *sseStats) Write(p []byte) (int, error) {
	s.partial = append(s.partial, p...)
	for {
		i := bytes.IndexByte(s.partial, '\n')
		if i < 0 {
			break
		}
		s.scanLine(s.partial[:i])
		s.partial = s.partial[i+1:]
	}
	// Safety valve: a stream that never sends a newline (not SSE) must not
	// accumulate unboundedly — matches newLargeScanner's 1MB max line.
	if len(s.partial) > 1<<20 {
		s.partial = nil
	}
	return len(p), nil
}

func (s *sseStats) scanLine(line []byte) {
	line = bytes.TrimSuffix(line, []byte("\r"))
	if bytes.Contains(line, []byte(`"finish_reason":"length"`)) ||
		bytes.Contains(line, []byte(`"finish_reason": "length"`)) {
		s.sawFinishLength = true
	}
	// A key match ("tool_calls":[) can't false-positive on assistant text: inside
	// a JSON string value the quotes would be \"-escaped.
	if bytes.Contains(line, []byte(`"tool_calls":[`)) ||
		bytes.Contains(line, []byte(`"tool_calls": [`)) {
		s.sawToolCall = true
	}
	if !bytes.HasPrefix(line, []byte("data: ")) {
		return
	}
	payload := line[6:]
	if bytes.Equal(payload, []byte("[DONE]")) {
		return
	}
	// Cumulative in both dialects, so last-seen wins (vLLM sends usage once on
	// the final chunk; llama.cpp's per-chunk usage counts up).
	if bytes.Contains(payload, []byte(`"usage"`)) {
		if n := usageCompletionTokens(payload); n > 0 {
			s.usageTokens = n
		}
	}
	// Same token heuristic as countSSETokens: one non-empty content OR reasoning
	// delta = one token (both dialects' reasoning field names). genTokens()
	// corrects this for spec-decode workers, whose deltas carry several tokens.
	counted := false
	for _, key := range []string{`"content":"`, `"reasoning_content":"`, `"reasoning":"`} {
		if n := deltaTextLen(payload, key); n > 0 {
			s.chars += n
			counted = true
		}
	}
	if counted {
		s.tokens++
	}
}

// deltaTextLen returns the byte length of the JSON string value that follows
// key in payload, or 0 if the key is absent or its value empty. Escape
// sequences count as their encoded length (\n is 2) — genTokens' divisor
// absorbs that slack; this is an estimate's input, not a decoder.
func deltaTextLen(payload []byte, key string) int {
	idx := bytes.Index(payload, []byte(key))
	if idx < 0 {
		return 0
	}
	n := 0
	for i := idx + len(key); i < len(payload); i++ {
		switch payload[i] {
		case '\\':
			i++
			n += 2
		case '"':
			return n
		default:
			n++
		}
	}
	return n
}

// usageCompletionTokens extracts usage's completion_tokens from an SSE chunk,
// 0 if absent or unparsable.
func usageCompletionTokens(payload []byte) int {
	var chunk struct {
		Usage *struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(payload, &chunk); err != nil || chunk.Usage == nil {
		return 0
	}
	return chunk.Usage.CompletionTokens
}

// inadequate mirrors responseInadequate for a streamed exchange, computed from
// the full-stream stats rather than a (possibly truncated) capture.
func (s *sseStats) inadequate() bool {
	if s.sawFinishLength {
		return true
	}
	return !s.sawToolCall && s.tokens == 0
}
