package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// A worker that spent its whole budget inside the thinking block: finish_reason
// "length", a full reasoning trace, and not one word of answer. This is what a
// caller was handed before the rescue existed.
const burnout = `{"choices":[{"message":{"role":"assistant","content":"",` +
	`"reasoning_content":"Let me work through this. The capital of France is Paris. Now let me double-check by considering"` +
	`,"tool_calls":[]},"finish_reason":"length"}],"usage":{"completion_tokens":4096,"total_tokens":4200}}`

const conclusion = `{"choices":[{"message":{"role":"assistant","content":"Paris"},"finish_reason":"stop"}],` +
	`"usage":{"completion_tokens":3,"total_tokens":120}}`

// scriptedWorker answers each request with the next body in the script, holding
// the last one once the script runs out. The rescue is a SECOND call to the SAME
// worker, so a canned single-body worker cannot express it.
func scriptedWorker(t *testing.T, hits *atomic.Int64, bodies ...string) (*httptest.Server, func(int) []byte) {
	t.Helper()
	var seen [][]byte
	var mu atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		n := int(mu.Add(1)) - 1
		body := make([]byte, req.ContentLength)
		_, _ = req.Body.Read(body)
		seen = append(seen, body)
		if hits != nil {
			hits.Add(1)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if n >= len(bodies) {
			n = len(bodies) - 1
		}
		_, _ = w.Write([]byte(bodies[n]))
	}))
	t.Cleanup(srv.Close)
	return srv, func(i int) []byte {
		if i >= len(seen) {
			return nil
		}
		return seen[i]
	}
}

func rescueRouter(t *testing.T, worker *httptest.Server) *Router {
	t.Helper()
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{
		ID: "solo", URL: worker.URL, Model: "default", Quality: 80,
		BaselineTPS: 100, MaxConcurrency: 2, TTLSeconds: 3600, Features: []string{"chat"},
	})
	reg.finishCertification("solo", true, map[string]Check{}, 100, 10, "")
	dir := t.TempDir()
	logs, err := openLogStore(dir+"/logs.sqlite", 16384, "")
	if err != nil {
		t.Fatalf("open log store: %v", err)
	}
	t.Cleanup(func() { logs.Close() })
	return &Router{
		cfg:      &Config{DefaultMaxTokens: 4096, RescueTruncated: true},
		registry: reg,
		client:   &http.Client{}, streamClient: &http.Client{},
		logs: logs, sessions: newSessionTracker(0, 0),
	}
}

func TestRescueAsksTheSameWorkerForItsConclusion(t *testing.T) {
	var hits atomic.Int64
	worker, sent := scriptedWorker(t, &hits, burnout, conclusion)
	r := rescueRouter(t, worker)

	rec := runChat(t, r, `{"model":"default","stream":false,"messages":[{"role":"user","content":"capital of France?"}]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if hits.Load() != 2 {
		t.Fatalf("want the original turn plus one rescue on the same worker, got %d calls", hits.Load())
	}
	if got := rec.Header().Get("X-LLM-Rescue"); got != "length" {
		t.Fatalf("rescue not reported to the caller: %q", got)
	}

	var resp struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unreadable response: %v (%s)", err, rec.Body.String())
	}
	if resp.Choices[0].Message.Content != "Paris" {
		t.Fatalf("conclusion not spliced in: %q", resp.Choices[0].Message.Content)
	}
	// "length" was true of the first pass and is not true of what the caller now
	// holds; a harness that branches on it would treat a complete answer as cut off.
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop — the answer is complete now", resp.Choices[0].FinishReason)
	}
	// The trace is the evidence for the answer and a client may render it.
	if !strings.Contains(resp.Choices[0].Message.ReasoningContent, "capital of France") {
		t.Error("the working notes were dropped from the rescued response")
	}
	// Both generations happened on the caller's behalf and both are billed.
	if resp.Usage.CompletionTokens != 4099 || resp.Usage.TotalTokens != 4203 {
		t.Errorf("usage = %d/%d, want 4099/4203 (the burnout plus the rescue)",
			resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
	}

	// The follow-up must replay the conversation, append the notes, ask for the
	// conclusion, and turn thinking OFF — a rescue that thinks again burns the same
	// budget the same way.
	var second map[string]any
	if err := json.Unmarshal(sent(1), &second); err != nil {
		t.Fatalf("unreadable rescue request: %v (%s)", err, sent(1))
	}
	msgs, _ := second["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("rescue replayed %d turns, want the original plus notes plus instruction", len(msgs))
	}
	if notes, _ := msgs[1].(map[string]any); !strings.Contains(notes["content"].(string), "double-check") {
		t.Error("the working notes were not replayed to the worker")
	}
	if ask, _ := msgs[2].(map[string]any); !strings.Contains(ask["content"].(string), "state your final answer now") {
		t.Error("the rescue did not ask for the conclusion")
	}
	kwargs, _ := second["chat_template_kwargs"].(map[string]any)
	if kwargs == nil || kwargs["enable_thinking"] != false {
		t.Errorf("rescue must run with thinking off, got chat_template_kwargs=%v", kwargs)
	}
	if mt, _ := second["max_tokens"].(float64); int(mt) != rescueMaxTokens {
		t.Errorf("rescue max_tokens = %v, want the small %d — it only has to state the answer", second["max_tokens"], rescueMaxTokens)
	}
	if _, streamed := second["stream"].(bool); streamed && second["stream"] == true {
		t.Error("the rescue turn must not stream")
	}
}

// Everything the rescue must keep its hands off. Each of these is a response a
// caller asked for, and a second generation against it is cost with no answer.
func TestRescueLeavesEverythingElseAlone(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
	}{{
		// The caller's own max_tokens cut a real answer short. That ceiling is the
		// client's to set, and re-asking would bill them for a shorter answer they
		// did not want.
		"truncation that produced text",
		`{"choices":[{"message":{"content":"the capital is"},"finish_reason":"length"}]}`,
	}, {
		"a complete answer",
		`{"choices":[{"message":{"content":"Paris"},"finish_reason":"stop"}]}`,
	}, {
		// Not a failed turn, whatever the finish reason says.
		"a tool call",
		`{"choices":[{"message":{"content":"","reasoning_content":"thinking","tool_calls":[{"id":"1"}]},"finish_reason":"length"}]}`,
	}, {
		// Nothing to conclude FROM. This is an empty answer, which is escalation's
		// case: another worker may do better, but this one has said nothing to build on.
		"empty with no working notes",
		`{"choices":[{"message":{"content":"","tool_calls":[]},"finish_reason":"length"}]}`,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			if notes, ok := rescuableNotes([]byte(tc.body)); ok {
				t.Errorf("rescue fired on %s (notes %q)", tc.name, notes)
			}
			var hits atomic.Int64
			worker, _ := scriptedWorker(t, &hits, tc.body, conclusion)
			r := rescueRouter(t, worker)
			rec := runChat(t, r, `{"model":"default","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
			if hits.Load() != 1 {
				t.Errorf("worker called %d times, want 1 — no rescue was warranted", hits.Load())
			}
			if rec.Header().Get("X-LLM-Rescue") != "" {
				t.Error("a rescue was reported that did not happen")
			}
		})
	}
}

// A rescue that comes back with nothing leaves the caller exactly what the worker
// said, and never loops asking again.
func TestRescueFallsBackToTheTruncation(t *testing.T) {
	for _, name := range []string{"empty conclusion", "second burnout"} {
		t.Run(name, func(t *testing.T) {
			second := `{"choices":[{"message":{"content":""},"finish_reason":"stop"}]}`
			if name == "second burnout" {
				second = burnout
			}
			var hits atomic.Int64
			worker, _ := scriptedWorker(t, &hits, burnout, second)
			r := rescueRouter(t, worker)

			rec := runChat(t, r, `{"model":"default","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
			if hits.Load() != 2 {
				t.Fatalf("exactly one rescue attempt allowed, got %d calls", hits.Load())
			}
			if rec.Header().Get("X-LLM-Rescue") != "" {
				t.Error("a failed rescue must not be reported as one")
			}
			if !strings.Contains(rec.Body.String(), `"finish_reason":"length"`) {
				t.Errorf("the original truncation should be returned untouched: %s", rec.Body.String())
			}
		})
	}
}

// A caller that declared a deadline has already spent it on the generation that
// burned out. Spending more of it on a rescue turns a bad answer into no answer.
func TestRescueRespectsADeclaredDeadline(t *testing.T) {
	var hits atomic.Int64
	worker, _ := scriptedWorker(t, &hits, burnout, conclusion)
	r := rescueRouter(t, worker)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"default","stream":false,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-LLM-Deadline-MS", "1")
	rec := httptest.NewRecorder()
	r.handleChatCompletions(rec, req)

	if hits.Load() != 1 {
		t.Fatalf("worker called %d times; there was no budget left to rescue in", hits.Load())
	}
	if !strings.Contains(rec.Body.String(), `"finish_reason":"length"`) {
		t.Errorf("the truncation should be returned as it stands: %s", rec.Body.String())
	}
}

func TestRescueDisabled(t *testing.T) {
	var hits atomic.Int64
	worker, _ := scriptedWorker(t, &hits, burnout, conclusion)
	r := rescueRouter(t, worker)
	r.cfg.RescueTruncated = false

	runChat(t, r, `{"model":"default","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	if hits.Load() != 1 {
		t.Fatalf("rescue is off; the worker must be called once (%d times)", hits.Load())
	}
}

// The notes can be tens of thousands of tokens — the worker just spent its entire
// budget producing them — so replaying the whole trace would rebuild the prompt
// that caused the problem. The TAIL is kept: a model that reasoned its way toward
// an answer was closest to it at the end.
func TestRescueNotesAreCappedAtTheirTail(t *testing.T) {
	long := strings.Repeat("thinking. ", rescueNotesChars) + "THE END"
	got := tailRunes(long, rescueNotesChars)
	if len(got) > rescueNotesChars+8 {
		t.Errorf("notes not capped: %d bytes", len(got))
	}
	if !strings.HasSuffix(got, "THE END") {
		t.Error("the tail of the trace is the part worth replaying, and it was cut")
	}
	if !strings.HasPrefix(got, "…") {
		t.Error("a truncated trace should say so, or the model reads it as the whole thing")
	}
	// Multi-byte runes must survive the cut, or the replayed turn is invalid UTF-8.
	if cut := tailRunes(strings.Repeat("→", 4000), 999); !utf8ValidString(cut) {
		t.Error("cut mid-rune")
	}
}

func utf8ValidString(s string) bool {
	for _, r := range s {
		if r == '�' {
			return false
		}
	}
	return true
}
