package router

// Prompt sizing — how many tokens is this text?
//
// The router has to answer that before it has picked a worker, which is the one
// moment it cannot ask a tokenizer: the tokenizer belongs to the model, and
// choosing the model is what the answer is FOR. So it estimates from character
// count, and the divisor it used was a constant 3.0 — deliberately below the ~4
// of English prose, because JSON-heavy payloads (tool schemas, structured args)
// pack nearer 3 characters per token and an under-estimate routes a prompt to a
// worker that then rejects it.
//
// A constant is the wrong shape for it. Everything else about a worker is
// MEASURED — quality, speed, capacity, context window, thinking dialect — and
// this was the last thing still guessed at, while the evidence to settle it went
// past on every single response: the endpoint reports `usage.prompt_tokens`, its
// own tokenizer's count of the text we just sent. The router logged that number
// and learned nothing from it.
//
// So: divide by a MEASURED chars-per-token, per model, from the router's own
// traffic. On the fleet this was written for, real traffic runs about 3.5 for
// code and JSON and about 4 for prose; assuming 3.0 inflated a 200K-token prompt
// to a 233K estimate, and a coding harness sat far enough above that to be
// refused by every worker in the fleet for a prompt every one of them could hold.
//
// Three properties make this safe to learn from live traffic:
//
//   - It moves DOWN fast and UP slowly. Down is toward caution (more estimated
//     tokens for the same text), so denser content than expected tightens the
//     estimate almost at once, while a run of unusually sparse prose can only
//     loosen it gradually.
//   - It is CLAMPED at both ends, so no amount of pathological input can push the
//     estimate somewhere absurd. The floor is below any real tokenizer's ratio and
//     the ceiling above it.
//   - It only samples LARGE prompts. A chat template's fixed overhead — role
//     markers, system scaffolding, a tools block rendered into text — is tokens
//     the endpoint counts and the router never saw, which on a short prompt is
//     most of the difference and would teach a ratio far denser than the text.
//     Above calibrationMinTokens it is noise.
//
// Held in memory, not persisted. A restarted router estimates conservatively
// again for a few requests, which costs nothing now that an over-estimate no
// longer refuses the request outright — it routes to the widest window and lets
// the engine rule (see contextOverflow in planRoute), and THAT path calibrates
// too, from the endpoint's own count in either its usage block or its refusal.

import (
	"math"
	"regexp"
	"strconv"
	"sync"
)

const (
	// defaultCharsPerToken is the estimate before a model has been measured, and
	// the value the router used unconditionally before it measured anything. It
	// stays deliberately conservative: an unmeasured model should over-estimate.
	defaultCharsPerToken = 3.0
	// The clamps. Below the floor is denser than any real tokenizer on text (it
	// is roughly base64's ratio); above the ceiling is sparser than plain English.
	minCharsPerToken = 2.0
	maxCharsPerToken = 5.0
	// calibrationMinTokens is the prompt size below which a sample says more about
	// the chat template than about the text.
	calibrationMinTokens = 2000
	// The asymmetric weights. Toward caution in one step, away from it in ten.
	calibrationDenserWeight  = 0.5
	calibrationSparserWeight = 0.1
)

// tokenRatios is the measured chars-per-token for each model the router has
// served. Keyed by model rather than by worker: the tokenizer is a property of
// the model, so two workers serving it share one measurement and each one's
// traffic informs the other's estimates.
type tokenRatios struct {
	mu      sync.RWMutex
	byModel map[string]float64
}

func newTokenRatios() *tokenRatios {
	return &tokenRatios{byModel: map[string]float64{}}
}

// forModel is the divisor to use for this model, and the default for one never
// measured. Nil-safe: a router assembled without a table (and every test that
// builds a hardFilter by hand) gets the constant the router always used.
func (t *tokenRatios) forModel(model string) float64 {
	if t == nil {
		return defaultCharsPerToken
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	if cpt, ok := t.byModel[model]; ok {
		return cpt
	}
	return defaultCharsPerToken
}

// observe records one endpoint-reported prompt size against the characters that
// produced it. chars is what the router would have divided; promptTokens is what
// the model's own tokenizer made of the same text.
func (t *tokenRatios) observe(model string, chars, promptTokens int) {
	if t == nil || model == "" || promptTokens < calibrationMinTokens || chars <= 0 {
		return
	}
	sample := float64(chars) / float64(promptTokens)
	if math.IsNaN(sample) || math.IsInf(sample, 0) {
		return
	}
	sample = math.Max(minCharsPerToken, math.Min(maxCharsPerToken, sample))
	t.mu.Lock()
	defer t.mu.Unlock()
	current, seen := t.byModel[model]
	if !seen {
		t.byModel[model] = sample
		return
	}
	// Denser than we thought means the estimate is running low, which is the
	// direction that gets a prompt rejected by an endpoint; correct it fast.
	weight := calibrationSparserWeight
	if sample < current {
		weight = calibrationDenserWeight
	}
	t.byModel[model] = current + weight*(sample-current)
}

// snapshot copies the table, for the dashboard and for tests.
func (t *tokenRatios) snapshot() map[string]float64 {
	if t == nil {
		return nil
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make(map[string]float64, len(t.byModel))
	for k, v := range t.byModel {
		out[k] = v
	}
	return out
}

// tokensForChars and charsForTokens are inverses, and must stay so: the context
// probe builds a haystack of charsForTokens(n) characters and labels the rung n
// tokens, and the hard filter then admits a prompt it sizes at n tokens on the
// strength of that rung. If the two disagreed, the ladder would be proving
// retrieval at one length and licensing prompts of another — which is exactly
// what it was doing, the probe filling at 4 characters per token while the filter
// sized at 3, so a rung labelled 128K was built from 512K characters that the
// filter would have called 170K.
func tokensForChars(chars int, charsPerToken float64) int {
	if charsPerToken <= 0 {
		charsPerToken = defaultCharsPerToken
	}
	return int(float64(chars) / charsPerToken)
}

func charsForTokens(tokens int, charsPerToken float64) int {
	if charsPerToken <= 0 {
		charsPerToken = defaultCharsPerToken
	}
	return int(float64(tokens) * charsPerToken)
}

// contextLimitPromptTokens reads the prompt size out of an endpoint's refusal.
//
// A 400 for an over-long prompt is the most valuable calibration sample there is
// — it is the endpoint's own tokenizer disagreeing with the router's estimate, in
// the one region where the disagreement costs the caller their request — and both
// of the engines this fleet runs put the number in the message:
//
//	vLLM:       "This model's maximum context length is 262144 tokens. However,
//	             you requested 270000 tokens (265000 in the messages, 5000 in the
//	             completion)."
//	llama.cpp:  "the request exceeds the available context size... tokens: 265000"
//
// Best-effort by construction: an engine that words it differently, or a future
// version that renumbers it, simply yields no sample and the router carries on
// estimating. It must never guess — a wrong number here would poison the ratio
// for every subsequent request against that model — so it takes only the
// explicitly labelled "in the messages" form, and ignores the request total,
// which includes the completion budget the router did not send as text.
var promptTokensInError = regexp.MustCompile(`(\d+)\s+in the messages`)

func contextLimitPromptTokens(body []byte) int {
	m := promptTokensInError.FindSubmatch(body)
	if m == nil {
		return 0
	}
	n, err := strconv.Atoi(string(m[1]))
	if err != nil || n <= 0 {
		return 0
	}
	return n
}
