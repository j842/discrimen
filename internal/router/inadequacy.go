package router

// How the router decides a 2xx response is not actually an answer.
//
// Split out of the online tier adapter when that was removed. The adapter learnt
// a per-difficulty score bias from these verdicts; nothing does any more, but the
// verdicts themselves were never the adapter's — they are what escalation and the
// expert panel test to decide whether a worker returned anything at all.

import (
	"bytes"
	"encoding/json"
	"strings"
)

// inadequacy grades a 2xx body. The two failures are not interchangeable:
//
//   - EMPTY is the worker's failure — no content, no reasoning, no tool calls —
//     and is what inline escalation repairs.
//   - TRUNCATED is the CALLER's ceiling: finish_reason "length" means max_tokens
//     stopped it, and a bigger model hits the same wall at twice the cost, so it
//     is NOT escalated.
type inadequacy int

const (
	responseOK inadequacy = iota
	responseEmpty
	responseTruncated
)

// classifyResponse grades a 2xx response body. Emptiness is checked FIRST: a
// reply that is both empty and length-capped is a worker that burned its whole
// budget without answering, which is the escalatable failure, not a caller whose
// ceiling was too tight.
func classifyResponse(body []byte, streamed bool) inadequacy {
	truncated := bytes.Contains(body, []byte(`"finish_reason":"length"`)) ||
		bytes.Contains(body, []byte(`"finish_reason": "length"`))
	if streamed {
		// A key match ("tool_calls":[) can't false-positive on assistant text:
		// inside a JSON string value the quotes would be \"-escaped.
		hasCall := bytes.Contains(body, []byte(`"tool_calls":[`)) || bytes.Contains(body, []byte(`"tool_calls": [`))
		if !hasCall && countSSETokens(body) == 0 {
			return responseEmpty
		}
		if truncated {
			return responseTruncated
		}
		return responseOK
	}
	var r struct {
		Choices []struct {
			Message struct {
				// content is null (not "") when a reasoning parser consumed the whole
				// output; json leaves the zero value, which is what we want. A raw
				// substring test on "tool_calls" is wrong here: vLLM serializes
				// "tool_calls":[] on EVERY message, which made empty responses
				// undetectable on that dialect.
				Content          string            `json:"content"`
				ReasoningContent string            `json:"reasoning_content"`
				Reasoning        string            `json:"reasoning"`
				ToolCalls        []json.RawMessage `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	// A body that does not parse into a usable completion is EMPTY, not OK. The
	// guard used to be `Unmarshal == nil && len(Choices) > 0`, so everything that
	// failed either test fell through to responseOK — measured: zero bytes,
	// truncated JSON, an HTML error page, `{"choices":[]}` and `{"error":{…}}`
	// all classified as a good answer. Escalation therefore never fired on them,
	// and writeBuffered scored them a SUCCESS, resetting the consecutive-failure
	// run so the breaker could never trip on this shape.
	if err := json.Unmarshal(body, &r); err != nil || len(r.Choices) == 0 {
		return responseEmpty
	}
	msg := r.Choices[0].Message
	if len(msg.ToolCalls) == 0 && strings.TrimSpace(msg.Content+msg.ReasoningContent+msg.Reasoning) == "" {
		return responseEmpty
	}
	if truncated {
		return responseTruncated
	}
	return responseOK
}
