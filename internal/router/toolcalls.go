package router

// Guard against nameless tool-call slots in a streamed response.
//
// vLLM v0.23.0 with --tool-call-parser qwen3_xml (llm-5090, Qwen3.6-27B) opens a
// tool-call slot it then never fills when the model emits several tool calls in
// one turn. The slot arrives as a single delta and is never mentioned again:
//
//	{"id":"chatcmpl-tool-b644…","type":"function","index":1,
//	 "function":{"name":null,"arguments":""}}
//
// Measured at 4/10 responses on a 3-call turn (2026-08-04). Clients faithfully
// replay it: an agent harness tries to invoke the empty name, gets "Tool  not
// found", and feeds that back. The model then sees a tool call it supposedly
// made with no name, abandons structured tool calls altogether, and starts
// writing shell commands as markdown prose — repeating them until the stream
// never terminates (610KB of tokens, no [DONE], in the case that prompted this).
// One malformed delta poisons the whole conversation, so the router drops it
// rather than letting every client rediscover the same bug.
//
// The rule is deliberately conservative: a slot survives iff it is ever given a
// name. Legitimate slots always carry their name in their opening delta and
// arguments in the continuations, so nothing well-formed is affected. A slot
// whose arguments somehow preceded its name would be dropped — not observed
// from any worker, and such a call is unusable anyway.
//
// Upstream is unfixed as of vLLM v0.26.0: the qwen3_xml streaming parser has a
// family of state-machine bugs (vllm-project/vllm#43713, still open, same
// model). Remove this guard once a worker's parser is proven clean.

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"strconv"
)

// maxSSELine bounds the buffer for a stream that never sends a newline, matching
// the 1MB safety valve in sseStats.
const maxSSELine = 1 << 20

// toolCallGuard filters an SSE stream in flight, dropping tool-call deltas for
// slots that never receive a function name and renumbering the survivors so the
// emitted indices stay contiguous — a gap at index 1 would leave a hole in the
// client's tool_calls array, reproducing the very defect being removed.
//
// It wraps the upstream body as an io.Reader rather than the client as a Writer
// so copyStreaming keeps its flush cadence, and the capture/stats sink sees
// exactly the bytes the client received. Complete lines pass through untouched
// unless they carry tool calls; only those are parsed and re-encoded.
type toolCallGuard struct {
	src     io.Reader
	backend *Backend

	scratch []byte // upstream read buffer
	out     []byte // sanitized bytes awaiting Read
	partial []byte // trailing line fragment, not yet terminated
	eof     bool

	remap map[int]int // upstream slot index -> emitted index (allocated on first name)
	next  int         // next emitted index
	drops int         // nameless deltas dropped, for the summary log
}

func newToolCallGuard(src io.Reader, backend *Backend) *toolCallGuard {
	return &toolCallGuard{
		src:     src,
		backend: backend,
		scratch: make([]byte, 32*1024),
		remap:   map[int]int{},
	}
}

func (g *toolCallGuard) Read(p []byte) (int, error) {
	for len(g.out) == 0 {
		if g.eof {
			if len(g.partial) > 0 { // unterminated tail: forward verbatim
				g.out, g.partial = g.partial, nil
				break
			}
			return 0, io.EOF
		}
		n, err := g.src.Read(g.scratch)
		if n > 0 {
			g.feed(g.scratch[:n])
		}
		if err != nil {
			if err != io.EOF {
				return 0, err
			}
			g.eof = true
		}
	}
	n := copy(p, g.out)
	g.out = g.out[n:]
	return n, nil
}

// feed splits incoming bytes into complete lines, sanitizing each and holding
// any trailing fragment until its newline arrives.
func (g *toolCallGuard) feed(chunk []byte) {
	g.partial = append(g.partial, chunk...)
	for {
		i := bytes.IndexByte(g.partial, '\n')
		if i < 0 {
			break
		}
		line := g.partial[:i+1] // keep the terminator: SSE framing must survive
		g.out = append(g.out, g.sanitize(line)...)
		g.partial = g.partial[i+1:]
	}
	if len(g.partial) > maxSSELine { // not SSE — stop buffering, pass it on
		g.out = append(g.out, g.partial...)
		g.partial = nil
	}
}

// sanitize rewrites one line if it carries tool-call deltas, else returns it
// unchanged. Any parse failure returns the line untouched: the guard must never
// be the reason a valid stream breaks.
func (g *toolCallGuard) sanitize(line []byte) []byte {
	// A key match can't false-positive on assistant text: inside a JSON string
	// value the quotes would be \"-escaped (same test as sseStats.scanLine).
	if !bytes.Contains(line, []byte(`"tool_calls":[`)) && !bytes.Contains(line, []byte(`"tool_calls": [`)) {
		return line
	}
	body := bytes.TrimSuffix(bytes.TrimSuffix(line, []byte("\n")), []byte("\r"))
	if !bytes.HasPrefix(body, []byte("data: ")) {
		return line
	}
	payload := body[6:]
	if bytes.Equal(payload, []byte("[DONE]")) {
		return line
	}

	// UseNumber keeps created/usage integers verbatim — decoding them as float64
	// would re-encode 1785837601 as 1.785837601e+09 and corrupt the chunk.
	dec := json.NewDecoder(bytes.NewReader(payload))
	dec.UseNumber()
	var chunk map[string]any
	if err := dec.Decode(&chunk); err != nil {
		return line
	}
	choices, ok := chunk["choices"].([]any)
	if !ok {
		return line
	}

	changed := false
	for _, c := range choices {
		choice, ok := c.(map[string]any)
		if !ok {
			continue
		}
		delta, ok := choice["delta"].(map[string]any)
		if !ok {
			continue
		}
		entries, ok := delta["tool_calls"].([]any)
		if !ok {
			continue
		}
		kept, mutated := g.filterEntries(entries)
		if !mutated {
			continue
		}
		changed = true
		if len(kept) == 0 {
			delete(delta, "tool_calls") // an empty array reads as "a call happened"
		} else {
			delta["tool_calls"] = kept
		}
	}
	if !changed {
		return line
	}

	encoded, err := json.Marshal(chunk)
	if err != nil {
		return line
	}
	out := append([]byte("data: "), encoded...)
	return append(out, '\n')
}

// filterEntries drops nameless slots and renumbers the survivors. It reports
// whether anything changed so untouched lines keep their original bytes.
func (g *toolCallGuard) filterEntries(entries []any) ([]any, bool) {
	kept := make([]any, 0, len(entries))
	mutated := false
	for _, e := range entries {
		entry, ok := e.(map[string]any)
		if !ok {
			kept = append(kept, e)
			continue
		}
		idx := indexOf(entry["index"])
		name, args := functionFields(entry)

		out, live := g.remap[idx]
		switch {
		case name != "":
			if !live { // opening delta of a real slot
				out = g.next
				g.next++
				g.remap[idx] = out
			}
		case live: // continuation carrying arguments for a named slot
		default:
			// Never named: the phantom. Dropping its arguments too (if any ever
			// arrive) is deliberate — a call with no function name is unusable.
			g.drops++
			mutated = true
			if g.drops == 1 {
				log.Printf("tool-call guard: dropped nameless tool_call slot backend=%s model=%s index=%d args=%q — upstream streaming parser bug",
					g.backendID(), g.backendModel(), idx, truncate(args, 80))
			}
			continue
		}

		if out != idx {
			entry["index"] = json.Number(strconv.Itoa(out))
			mutated = true
		}
		kept = append(kept, entry)
	}
	return kept, mutated
}

// report logs a summary once the stream ends. Called even on error paths so a
// truncated stream still accounts for what it dropped.
func (g *toolCallGuard) report() {
	if g.drops == 0 {
		return
	}
	log.Printf("tool-call guard: dropped %d nameless tool_call delta(s) backend=%s model=%s — see vllm-project/vllm#43713",
		g.drops, g.backendID(), g.backendModel())
}

func (g *toolCallGuard) backendID() string {
	if g.backend == nil {
		return ""
	}
	return g.backend.ID
}

func (g *toolCallGuard) backendModel() string {
	if g.backend == nil {
		return ""
	}
	return g.backend.Model
}

// functionFields pulls name and arguments out of a tool-call delta, treating a
// JSON null (what the buggy parser emits) the same as an absent field.
func functionFields(entry map[string]any) (name, args string) {
	fn, ok := entry["function"].(map[string]any)
	if !ok {
		return "", ""
	}
	name, _ = fn["name"].(string)
	args, _ = fn["arguments"].(string)
	return name, args
}

// indexOf reads a delta's slot index, defaulting to 0 when absent — the OpenAI
// streaming format omits it for single-call responses.
func indexOf(v any) int {
	switch n := v.(type) {
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return 0
		}
		return int(i)
	case float64:
		return int(n)
	}
	return 0
}
