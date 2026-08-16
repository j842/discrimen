package router

// Online tier adapter — self-correcting layer over auto-tier routing.
//
// It learns, per difficulty-score region, a small UPWARD bias on the score when
// the chosen tier's responses come back inadequate — empty or truncated
// (finish_reason "length"), the same signals the clabtree agent itself retries
// on — and decays that bias back to baseline when responses are clean. The
// router self-labels from the responses it already proxies, so no caller
// cooperation is needed.
//
// Two deliberate safety properties:
//
//   - One-directional. The bias is clamped to [0, maxBias], so the adapter can
//     only route a region UP from the auto-tier baseline, never below it. Worst
//     case is over-conservative (more cost); it can never push quality below the
//     baseline. (Routing a region cheaper than baseline is the quality-risky
//     direction — that's left to an offline-trained classifier with real
//     preference labels, not learned on the fly from passive signals.)
//   - Fleet-agnostic. The bias is in SCORE space, not quality points, so adding
//     or removing workers reshapes the auto-tier mapping underneath while the
//     learned correction still applies — exactly like the auto-tiers themselves.

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type tierAdapter struct {
	mu      sync.Mutex
	bias    []float64 // per-bin upward score bias, each in [0, maxBias]
	bins    int
	maxBias float64
	lrUp    float64 // increment per inadequate response
	lrDown  float64 // decay per clean response
	path    string
	dirty   bool
}

func newTierAdapter(cfg *Config, path string) *tierAdapter {
	bins := cfg.AdaptBins
	if bins < 2 {
		bins = 10
	}
	a := &tierAdapter{
		bias:    make([]float64, bins),
		bins:    bins,
		maxBias: clampFloat(cfg.AdaptMaxBias, 0.01, 1.0, 0.30),
		lrUp:    clampFloat(cfg.AdaptLRUp, 0.0001, 1.0, 0.04),
		lrDown:  clampFloat(cfg.AdaptLRDown, 0.0001, 1.0, 0.01),
		path:    path,
	}
	a.load()
	return a
}

func (a *tierAdapter) binOf(score float64) int {
	b := int(clamp01(score) * float64(a.bins))
	if b >= a.bins {
		b = a.bins - 1
	}
	if b < 0 {
		b = 0
	}
	return b
}

// adjust returns the effective difficulty score after the learned upward bias.
// Safe on a nil receiver (returns the score unchanged), so callers need no guard.
func (a *tierAdapter) adjust(score float64) float64 {
	if a == nil {
		return score
	}
	a.mu.Lock()
	b := a.bias[a.binOf(score)]
	a.mu.Unlock()
	return clamp01(score + b)
}

// observe feeds back an outcome for a request auto-routed at the given RAW
// difficulty score (the same space adjust bins by). needHigher=true (inadequate
// response) nudges the region's bias up; a clean response decays it.
func (a *tierAdapter) observe(score float64, needHigher bool) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	i := a.binOf(score)
	if needHigher {
		a.bias[i] = math.Min(a.maxBias, a.bias[i]+a.lrUp)
	} else {
		a.bias[i] = math.Max(0, a.bias[i]-a.lrDown)
	}
	a.dirty = true
}

// snapshot returns a copy of the current per-bin biases (for debug/observability).
func (a *tierAdapter) snapshot() []float64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]float64(nil), a.bias...)
}

// ── Outcome signals (parsed from the proxied response) ──────────────────────

// parseRouteScore extracts the raw difficulty score from a "route:d=0.82,…" hint.
func parseRouteScore(route string) (float64, bool) {
	const prefix = "route:d="
	if !strings.HasPrefix(route, prefix) {
		return 0, false
	}
	rest := route[len(prefix):]
	if i := strings.IndexByte(rest, ','); i >= 0 {
		rest = rest[:i]
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(rest), 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// inadequacy is WHY a 2xx response is unusable. The two causes are split apart
// because they call for different remedies:
//
//   - responseEmpty — the worker produced nothing at all. Another worker can
//     plausibly do better, so this is worth re-dispatching inline (escalate.go).
//   - responseTruncated — the answer ran into the caller's own token ceiling.
//     Re-running it on a better model just truncates again at twice the cost, so
//     it is NOT escalated; only the adapter learns from it.
//
// Both still count as inadequate for the tier adapter, which learns about the
// difficulty region rather than repairing the request.
type inadequacy int

const (
	responseOK inadequacy = iota
	responseEmpty
	responseTruncated
)

// responseInadequate reports whether a 2xx response looks inadequate: empty (no
// content, no reasoning, no tool calls) or truncated (finish_reason "length").
// These match the failure signals the clabtree agent already retries on. Works
// on both a buffered JSON body and a captured SSE stream.
func responseInadequate(body []byte, streamed bool) bool {
	return classifyResponse(body, streamed) != responseOK
}

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
	if json.Unmarshal(body, &r) == nil && len(r.Choices) > 0 {
		msg := r.Choices[0].Message
		if len(msg.ToolCalls) == 0 && strings.TrimSpace(msg.Content+msg.ReasoningContent+msg.Reasoning) == "" {
			return responseEmpty
		}
	}
	if truncated {
		return responseTruncated
	}
	return responseOK
}

// ── Persistence (best-effort JSON file beside the log DB) ───────────────────

func (a *tierAdapter) load() {
	data, err := os.ReadFile(a.path)
	if err != nil {
		return
	}
	var saved []float64
	if json.Unmarshal(data, &saved) != nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for i := 0; i < len(a.bias) && i < len(saved); i++ {
		a.bias[i] = clampFloat(saved[i], 0, a.maxBias, 0)
	}
}

func (a *tierAdapter) save() {
	a.mu.Lock()
	if !a.dirty {
		a.mu.Unlock()
		return
	}
	data, _ := json.Marshal(a.bias)
	a.dirty = false
	a.mu.Unlock()
	_ = os.WriteFile(a.path, data, 0o600)
}

func (a *tierAdapter) persistLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		a.save()
	}
}

// clampFloat returns v clamped to [lo, hi], substituting fallback when v<=0.
func clampFloat(v, lo, hi, fallback float64) float64 {
	if v <= 0 {
		v = fallback
	}
	if v < lo {
		v = lo
	}
	if v > hi {
		v = hi
	}
	return v
}
