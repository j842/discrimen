package router

import (
	"encoding/json"
	"io"
	"strings"
	"testing"
)

// runGuard streams src through the guard in small reads, so a chunk boundary
// falls mid-line — the case that breaks naive line filters.
func runGuard(t *testing.T, src string, chunk int) (string, *toolCallGuard) {
	t.Helper()
	backend := &Backend{BackendRegistration: BackendRegistration{ID: "llm-test", Model: "qwen"}}
	g := newToolCallGuard(&choppyReader{data: []byte(src), chunk: chunk}, backend)
	out, err := io.ReadAll(g)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	return string(out), g
}

// choppyReader hands out at most chunk bytes per Read.
type choppyReader struct {
	data  []byte
	chunk int
	pos   int
}

func (c *choppyReader) Read(p []byte) (int, error) {
	if c.pos >= len(c.data) {
		return 0, io.EOF
	}
	n := len(c.data) - c.pos
	if c.chunk > 0 && n > c.chunk {
		n = c.chunk
	}
	if n > len(p) {
		n = len(p)
	}
	copy(p, c.data[c.pos:c.pos+n])
	c.pos += n
	return n, nil
}

// toolCallNames replays a sanitized stream the way a client does, returning the
// assembled name per emitted slot index.
func toolCallNames(t *testing.T, stream string) map[int]string {
	t.Helper()
	names := map[int]string{}
	for _, line := range strings.Split(stream, "\n") {
		if !strings.HasPrefix(line, "data: ") || strings.TrimSpace(line[6:]) == "[DONE]" {
			continue
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(line[6:]), &chunk); err != nil {
			t.Fatalf("client could not parse %q: %v", line, err)
		}
		choices, _ := chunk["choices"].([]any)
		for _, c := range choices {
			choice, _ := c.(map[string]any)
			delta, _ := choice["delta"].(map[string]any)
			entries, _ := delta["tool_calls"].([]any)
			for _, e := range entries {
				entry, _ := e.(map[string]any)
				idx := indexOf(entry["index"])
				if _, seen := names[idx]; !seen {
					names[idx] = ""
				}
				name, _ := functionFields(entry)
				names[idx] += name
			}
		}
	}
	return names
}

// The captured llm-5090 failure: three slots, the middle one opened with a null
// name and never mentioned again.
const phantomStream = `data: {"choices":[{"delta":{"tool_calls":[{"id":"a","type":"function","index":0,"function":{"name":"bash","arguments":""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":null,"arguments":"{\"command\":\"ls\"}"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"id":"b","type":"function","index":1,"function":{"name":null,"arguments":""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"id":"c","type":"function","index":2,"function":{"name":"read","arguments":""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":2,"function":{"name":null,"arguments":"{\"path\":\"x\"}"}}]}}]}
data: [DONE]
`

func TestGuardDropsPhantomAndRenumbers(t *testing.T) {
	for _, chunk := range []int{0, 1, 7, 64} { // 1 = every line split repeatedly
		out, g := runGuard(t, phantomStream, chunk)
		if g.drops != 1 {
			t.Errorf("chunk=%d: drops = %d, want 1", chunk, g.drops)
		}
		names := toolCallNames(t, out)
		// The survivors must be contiguous: a hole at index 1 would leave the
		// client with an empty-named slot, the defect this guard removes.
		want := map[int]string{0: "bash", 1: "read"}
		if len(names) != len(want) {
			t.Fatalf("chunk=%d: emitted slots %v, want %v", chunk, names, want)
		}
		for i, n := range want {
			if names[i] != n {
				t.Errorf("chunk=%d: slot %d = %q, want %q", chunk, i, names[i], n)
			}
		}
		// Arguments must follow their renumbered slot, not the original index.
		if !strings.Contains(out, `{\"path\":\"x\"}`) {
			t.Errorf("chunk=%d: lost arguments of the renumbered slot", chunk)
		}
	}
}

func TestGuardLeavesCleanStreamByteIdentical(t *testing.T) {
	clean := `data: {"choices":[{"delta":{"content":"hi"}}]}
data: {"choices":[{"delta":{"tool_calls":[{"id":"a","type":"function","index":0,"function":{"name":"bash","arguments":""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":null,"arguments":"{}"}}]}}]}
data: {"choices":[],"usage":{"prompt_tokens":1540,"total_tokens":1707,"completion_tokens":167}}
data: [DONE]
`
	out, g := runGuard(t, clean, 5)
	if g.drops != 0 {
		t.Errorf("drops = %d, want 0", g.drops)
	}
	if out != clean {
		t.Errorf("clean stream was rewritten:\n got %q\nwant %q", out, clean)
	}
}

// Large integers must survive re-encoding: decoding into float64 would turn
// created=1785837601 into 1.785837601e+09 and corrupt the chunk.
func TestGuardPreservesIntegersWhenRewriting(t *testing.T) {
	src := `data: {"created":1785837601,"usage":{"total_tokens":1707},"choices":[{"delta":{"tool_calls":[{"id":"b","type":"function","index":0,"function":{"name":null,"arguments":""}}]}}]}
data: [DONE]
`
	out, g := runGuard(t, src, 0)
	if g.drops != 1 {
		t.Fatalf("drops = %d, want 1", g.drops)
	}
	if strings.Contains(out, "e+09") || !strings.Contains(out, "1785837601") {
		t.Errorf("integer mangled by re-encoding: %s", out)
	}
	if !strings.Contains(out, "1707") {
		t.Errorf("usage mangled by re-encoding: %s", out)
	}
	// The array emptied out, so the key must go: an empty tool_calls array still
	// reads as "a tool call happened".
	if strings.Contains(out, "tool_calls") {
		t.Errorf("empty tool_calls array left in place: %s", out)
	}
}

func TestGuardPassesThroughMalformedAndNonSSE(t *testing.T) {
	// Unparseable JSON on a tool_calls line, and a stream with no trailing
	// newline, must both survive untouched rather than be swallowed.
	src := `data: {"choices":[{"delta":{"tool_calls":[ BROKEN
data: {"choices":[{"delta":{"content":"tail no newline"}}]}`
	out, g := runGuard(t, src, 3)
	if g.drops != 0 {
		t.Errorf("drops = %d, want 0", g.drops)
	}
	if out != src {
		t.Errorf("passthrough altered:\n got %q\nwant %q", out, src)
	}
}

// A slot rescued by a later name must be kept, not stay dropped.
func TestGuardKeepsSlotNamedInALaterDelta(t *testing.T) {
	src := `data: {"choices":[{"delta":{"tool_calls":[{"id":"a","type":"function","index":0,"function":{"name":null,"arguments":""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"bash","arguments":"{}"}}]}}]}
data: [DONE]
`
	out, g := runGuard(t, src, 0)
	if g.drops != 1 {
		t.Errorf("drops = %d, want 1 (the empty opener)", g.drops)
	}
	if names := toolCallNames(t, out); names[0] != "bash" {
		t.Errorf("rescued slot = %q, want \"bash\"", names[0])
	}
}
