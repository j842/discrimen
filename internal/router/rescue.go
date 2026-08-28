package router

// Length rescue — recover the answer a worker spent its whole budget failing to
// start.
//
// A hybrid reasoning model can pour every token it has into the thinking block
// and stop at the cap before writing a word of the answer. What comes back is a
// 200 with finish_reason "length", a full reasoning trace, and content "". To
// the caller that is a failed turn; to the router it is not even an empty
// response, because classifyResponse counts the reasoning trace as output — so
// escalation never fired on it and the caller was simply handed the nothing.
//
// The repair is one cheap follow-up turn to the SAME worker: replay the
// conversation, append the working notes as an incomplete assistant turn, and
// ask for the conclusion with thinking OFF and a small budget. The prefix is
// already in that worker's KV cache, so the second pass is mostly decode.
//
// This is the rescue the ornith and gemma sidecars run in front of their own
// workers, lifted to the router so every worker gets it — including the paid and
// relayed ones, which cannot have a sidecar in front of them. The router can also
// do it BETTER than a sidecar can: it knows each worker's measured thinking
// dialect, so it writes the off-switch in the spelling that endpoint was proven
// to honour, where a sidecar has to hard-code its one worker's dialect.
//
// Boundaries, and why each one:
//
//   - EMPTY CONTENT ONLY. A truncation that produced real text is the CALLER's
//     max_tokens doing its job: the client asked for at most that many tokens and
//     got them. Replacing a cut-off answer with a shorter summary would be a
//     second generation the caller did not ask for, billed to them, on every
//     request from any client that deliberately sets a small ceiling. It passes
//     through untouched, exactly as it does in the sidecars.
//   - THERE MUST BE WORKING NOTES. Without a reasoning trace there is nothing to
//     conclude FROM, and the response is an empty answer rather than a truncated
//     one — which is escalation's case, not this one.
//   - NO TOOL CALLS. A response carrying a tool call is not a failed turn, whatever
//     the finish reason says.
//   - SAME WORKER, ONE HOP. Escalation moves an empty answer to a better model;
//     that is the wrong move here, because a bigger model runs into the same token
//     ceiling and bills twice for it. The fix is not a better model, it is asking
//     this one to stop thinking.
//   - NON-STREAMED ONLY. Once SSE bytes are on the wire they cannot be recalled.
//     Same boundary, and the same reason, as escalation.

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	// rescueMaxTokens bounds the follow-up. The answer already exists in the
	// working notes; this turn only has to state it. A large cap here would let a
	// model that just burned 32K tokens thinking do it again.
	rescueMaxTokens = 2048
	// rescueNotesChars caps how much of the reasoning trace is replayed. The tail
	// is the part that matters — a model that reasoned its way to an answer and ran
	// out of room was closest to it at the end — and the whole trace can be tens of
	// thousands of tokens on a worker that just spent its entire budget producing
	// it, which would put the rescue turn's prompt near the ceiling that caused the
	// problem.
	rescueNotesChars = 24000
	// rescueInstruction is the sidecars' wording, unchanged because it is the
	// version verified against live workers. " /no_think" is the fleet's
	// belt-and-braces convention for a thinking-off turn; models without the switch
	// ignore the token.
	rescueInstruction = "Time is up. Based on your working above, state your final answer now. Answer only. /no_think"
)

// truncatedResponse is the shape the rescue needs out of a completion: enough to
// decide whether this is a thinking-budget burnout, and the notes to build the
// follow-up from.
type truncatedResponse struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content          string            `json:"content"`
			ReasoningContent string            `json:"reasoning_content"`
			Reasoning        string            `json:"reasoning"`
			ToolCalls        []json.RawMessage `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
}

// rescuableNotes returns the working notes to conclude from, and whether this
// response is a thinking-budget burnout at all. Everything it rejects is either
// a good answer or somebody else's repair.
func rescuableNotes(body []byte) (string, bool) {
	var resp truncatedResponse
	if err := json.Unmarshal(body, &resp); err != nil || len(resp.Choices) != 1 {
		return "", false
	}
	c := resp.Choices[0]
	if c.FinishReason != "length" {
		return "", false
	}
	if len(c.Message.ToolCalls) > 0 {
		return "", false
	}
	if strings.TrimSpace(c.Message.Content) != "" {
		return "", false
	}
	notes := c.Message.ReasoningContent
	if strings.TrimSpace(notes) == "" {
		notes = c.Message.Reasoning
	}
	if strings.TrimSpace(notes) == "" {
		return "", false
	}
	return notes, true
}

// rescueLength asks the worker that just ran out of budget to state its
// conclusion. Returns ok=false — leaving the original response exactly as it
// came back — whenever this is not a burnout, there is no budget left for a
// second turn, or the follow-up produced nothing usable. It never returns
// something worse than what it was given.
func (r *Router) rescueLength(req *http.Request, d *dispatch, orig bufferedResult) (bufferedResult, bool) {
	if r.cfg == nil || !r.cfg.RescueTruncated {
		return orig, false
	}
	notes, ok := rescuableNotes(orig.body)
	if !ok {
		return orig, false
	}
	backend := *d.backend
	// The caller declared how long it would wait and one failed generation has
	// already spent most of it. Priced the same way every other mover prices its
	// move — a rescue that lands after the client has given up is a generation
	// billed to nobody's benefit.
	if d.budget > 0 {
		remaining := (d.budget - time.Since(d.start)).Seconds() * deadlineSafetyFactor
		if expectedLatency(backend, rescueJob(d, backend)) > remaining {
			log.Printf("rescue: only %.1fs of the caller's budget is left after %s — returning the truncation",
				remaining, backend.ID)
			return orig, false
		}
	}
	body, err := r.rescueBody(d, backend, notes)
	if err != nil {
		return orig, false
	}
	start := time.Now()
	// No retry ladder. It exists for a worker that is loading or saturated, and
	// this one demonstrably just answered — sleeping on it would only pin the slot
	// the rescue is already holding.
	res := r.requestBufferedWithDelays(req, backend, body, nil, d.remainingBudget())
	if !res.ok() {
		log.Printf("rescue: %s could not be asked for its conclusion (%s) — returning the truncation",
			backend.ID, describeFailure(res))
		return orig, false
	}
	answer, tokens := rescueAnswer(res.body)
	if strings.TrimSpace(answer) == "" {
		log.Printf("rescue: %s produced no conclusion — returning the truncation", backend.ID)
		return orig, false
	}
	patched, err := spliceRescue(orig.body, answer, tokens)
	if err != nil {
		return orig, false
	}
	log.Printf("rescue: %s spent its budget thinking; its conclusion took %.1fs and %d tokens: %.60q",
		backend.ID, time.Since(start).Seconds(), tokens, answer)
	orig.body = patched
	return orig, true
}

// rescueJob prices the follow-up turn. It is the original job with two things
// changed: the output is the rescue's small ceiling rather than the caller's,
// and the worker is its own incumbent — the prefix really is in that worker's KV
// cache, because it finished prefilling it moments ago, so charging the rescue a
// cold prefill would price it out of every budget a long prompt could have.
func rescueJob(d *dispatch, b *Backend) jobCost {
	job := d.job
	job.incumbent = b.ID
	job.mode = thinkingOff
	job.outputTokens = rescueMaxTokens
	job.maxTokens = rescueMaxTokens
	return job
}

// rescueBody builds the follow-up turn. It is assembled fresh rather than edited
// out of the request that just failed, because that body carries the thinking
// gate the first pass resolved — and patchForwardedBody will not overwrite a gate
// that is already present (the kwargs escape hatch outranks the router). Patching
// the original in place would therefore leave thinking ON for the one turn whose
// entire purpose is to have it off.
func (r *Router) rescueBody(d *dispatch, backend *Backend, notes string) ([]byte, error) {
	var client map[string]json.RawMessage
	if err := json.Unmarshal(d.body, &client); err != nil {
		return nil, err
	}
	messages, ok := client["messages"]
	if !ok {
		return nil, fmt.Errorf("no messages to replay")
	}
	var turns []json.RawMessage
	if err := json.Unmarshal(messages, &turns); err != nil || len(turns) == 0 {
		return nil, fmt.Errorf("no messages to replay")
	}
	for _, turn := range []map[string]string{
		{"role": "assistant", "content": "(working notes, incomplete)\n" + tailRunes(notes, rescueNotesChars)},
		{"role": "user", "content": rescueInstruction},
	} {
		raw, err := json.Marshal(turn)
		if err != nil {
			return nil, err
		}
		turns = append(turns, raw)
	}
	replayed, err := json.Marshal(turns)
	if err != nil {
		return nil, err
	}
	payload := map[string]json.RawMessage{"messages": replayed}
	// Carry through the fields that shape the answer's FORM, so a deterministic
	// request gets a deterministic rescue. Nothing that shapes its length or its
	// thinking: those are what this turn is overriding. tools are dropped with
	// them — the model already had its chance to call one and did not, and a turn
	// that asks for a conclusion in prose must not come back as a half-formed call.
	for _, field := range []string{"model", "temperature", "top_p", "seed"} {
		if v, ok := client[field]; ok {
			payload[field] = v
		}
	}
	payload["stream"] = json.RawMessage("false")
	payload["max_tokens"] = json.RawMessage(fmt.Sprintf("%d", rescueMaxTokens))
	base, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	// Thinking off, in the spelling this endpoint was measured to honour.
	off := thinkingResolution{patch: true, enable: false, noThink: true}.forBackend(backend)
	return patchForwardedBody(base, 0, 0, off, backend.ServedID), nil
}

// rescueAnswer pulls the conclusion and its cost out of the follow-up.
func rescueAnswer(body []byte) (string, int) {
	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &resp); err != nil || len(resp.Choices) == 0 {
		return "", 0
	}
	return resp.Choices[0].Message.Content, resp.Usage.CompletionTokens
}

// spliceRescue writes the conclusion into the original response, keeping
// everything else about it — id, model, the reasoning trace the notes came from,
// every field a client might key on. The finish reason becomes "stop" because
// that is now true: the answer is complete. The rescue's tokens are ADDED to the
// usage rather than replacing it, so the caller is billed for what was actually
// generated on their behalf.
func spliceRescue(body []byte, answer string, tokens int) ([]byte, error) {
	var resp map[string]any
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	choices, _ := resp["choices"].([]any)
	if len(choices) == 0 {
		return nil, fmt.Errorf("no choices to splice into")
	}
	choice, _ := choices[0].(map[string]any)
	if choice == nil {
		return nil, fmt.Errorf("unreadable choice")
	}
	msg, _ := choice["message"].(map[string]any)
	if msg == nil {
		return nil, fmt.Errorf("unreadable message")
	}
	msg["content"] = answer
	choice["finish_reason"] = "stop"
	if usage, _ := resp["usage"].(map[string]any); usage != nil && tokens > 0 {
		for _, field := range []string{"completion_tokens", "total_tokens"} {
			if n, ok := usage[field].(float64); ok {
				usage[field] = n + float64(tokens)
			}
		}
	}
	return json.Marshal(resp)
}

// tailRunes returns the last max bytes of s, cut on a rune boundary so the
// replayed notes are never invalid UTF-8.
func tailRunes(s string, max int) string {
	if len(s) <= max {
		return s
	}
	cut := len(s) - max
	for cut < len(s) && !utf8.RuneStart(s[cut]) {
		cut++
	}
	return "…" + s[cut:]
}
