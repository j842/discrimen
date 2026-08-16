package router

// Providers — what a registry row now says about where it runs, what it costs,
// and, more importantly, WHO OWNS THE VALUES ON IT.
//
// The router grew up on a fleet of local workers that all declare the same
// thing: an id, a URL, an api key, and nothing else worth trusting. Everything
// that matters — quality, speed, capacity, context, capabilities — is measured,
// and the measurement overwrites whatever the worker said. That rule is the
// whole design and it stays.
//
// It stops being the whole rule once a row can be entered by hand. An operator
// typing a metered endpoint into the admin API is not guessing: they are
// reading the provider's own documentation, and the number they type is often
// one the router CANNOT measure without spending money to be told what it was
// already told. Concurrency is the sharp case. A rate-limited endpoint answers
// a burst of concurrent probes with 429s, the ramp reads that as the saturation
// knee, and the verdict is cached per (id, model) and never re-measured — so one
// bad minute permanently under-rates the endpoint.
//
// Hence the split, which is the central invariant of this file:
//
//	MANUAL rows are operator-owned. A probe refines a field the operator left
//	blank and NEVER overwrites one they filled in.
//
//	BEACON rows behave exactly as they always have. Everything a worker declares
//	is a seed the measurement replaces. Every worker deployed today is a beacon
//	row and none of them change.
//
// The precedence itself is not new. profileBackend already lets llama.cpp's
// published total_slots outrank the ramp, on the same reasoning: where a value
// is PUBLISHED rather than inferred, the publication wins. A declared
// max_concurrency is that same fact arriving through a person instead of
// /props.

import "strings"

const (
	// providerLocal is where a registration that names no provider runs. Every
	// worker deployed before this field existed sends none, and a local worker
	// costs nothing per token, so the default has to be the free one — see the
	// compatibility contract in PLAN.md.
	providerLocal = "local"

	// sourceBeacon is a row that registered ITSELF through /backends/register and
	// re-posts a keepalive; sourceManual is one an operator entered through the
	// admin API. Anything that arrives on the push endpoint is a beacon by
	// construction, whatever its payload claims.
	sourceBeacon = "beacon"
	sourceManual = "manual"
)

// normalizeProviderFields settles the P2 fields on a registration. Called from
// normalizeRegistration, so it runs on every path a row can arrive by: the push
// endpoint, the admin API, and the persisted rows reloaded at startup.
//
// An empty source normalises to beacon rather than to manual, which is what
// makes the upgrade safe: every row already in backend_registrations predates
// the field, and every one of them is a worker that registered itself.
func normalizeProviderFields(reg *BackendRegistration) {
	reg.Provider = strings.ToLower(strings.TrimSpace(reg.Provider))
	if reg.Provider == "" {
		reg.Provider = providerLocal
	}
	// Only the exact spelling counts as manual. A typo must not silently grant a
	// row operator ownership of its own values.
	if strings.ToLower(strings.TrimSpace(reg.Source)) == sourceManual {
		reg.Source = sourceManual
	} else {
		reg.Source = sourceBeacon
	}
	// A negative price is not a discount.
	if reg.InputPricePerMtok < 0 {
		reg.InputPricePerMtok = 0
	}
	if reg.OutputPricePerMtok < 0 {
		reg.OutputPricePerMtok = 0
	}
}

// isManualRow reports whether a row is operator-owned.
func isManualRow(b *Backend) bool { return b != nil && b.Source == sourceManual }

// operatorDeclared is the set of values an OPERATOR entered by hand on this row,
// or the zero registration when the row is a beacon (which owns nothing).
//
// It reads lastReg rather than the live Backend on purpose: applyProfileIfGen
// writes its measurements straight over the embedded registration, so after one
// profile the live row no longer remembers what was declared. lastReg is the
// registration content AS RECEIVED and is only replaced by another registration.
//
// Callers hold the registry lock.
func operatorDeclared(b *Backend) BackendRegistration {
	if !isManualRow(b) || b.lastReg == nil {
		return BackendRegistration{}
	}
	return *b.lastReg
}

// operatorMaxConcurrency is the concurrency ceiling an operator declared on a
// manual row, or 0 for a beacon row and for a manual row that declared none.
//
// It outranks the capacity ramp in both directions, which is the difference
// between this and the llama.cpp total_slots rule next to it in profileBackend.
// total_slots can only ever LOWER the ramp's answer, because the failure it
// guards against is a serialising worker whose aggregate throughput
// over-reports. Here the failure is the opposite one — a metered endpoint
// answering the ramp with 429s and being written off at a fraction of its real
// capacity — so the declared number has to be able to raise it too.
func (r *Registry) operatorMaxConcurrency(id string) int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return operatorDeclared(r.backends[id]).MaxConcurrency
}

// declaredRegistration returns a row's registration AS RECEIVED — what the
// operator typed, before the profiler wrote its measurements over the fields
// they left blank. It is what an edit has to be applied to: patching the live
// row instead would fold every measured value back in as a declaration, and
// after one edit the probe could never refine those fields again.
func (r *Registry) declaredRegistration(id string) (BackendRegistration, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	b := r.backends[id]
	if b == nil || b.lastReg == nil {
		return BackendRegistration{}, false
	}
	reg := *b.lastReg
	reg.Features = append([]string(nil), b.lastReg.Features...)
	return reg, true
}

// manualRows returns the operator-owned rows, id-sorted, with api keys scrubbed.
// The admin provider API lists these; beacon rows appear in /backends and are
// managed by the worker that registered them.
func (r *Registry) manualRows() []*Backend {
	out := []*Backend{}
	for _, b := range publicBackends(r.snapshot()) {
		if isManualRow(b) {
			out = append(out, b)
		}
	}
	return out
}

// ── Price ───────────────────────────────────────────────────────────────────
//
// Everything cost does to routing is below, and it is deliberately two
// functions. Price is a declared fact about an endpoint, in the same category
// as the `uncensored` tag, and NOT a routing knob: there is no cost weight to
// tune, no price threshold to set and no budget mode. Free versus paid is a
// binary derived from a declared price of zero, which is what makes the rule
// need no tuning — every local worker declares zero, so "prefer the free ones"
// is already right on the fleet this router grew up on.

// isFreeBackend reports whether an endpoint costs nothing per token.
func isFreeBackend(b *Backend) bool {
	return b == nil || (b.InputPricePerMtok <= 0 && b.OutputPricePerMtok <= 0)
}

// tokenCost is what promptTokens in and outputTokens out cost at this row's
// DECLARED prices, in whatever currency the operator is billed in. Zero for
// every local worker, which is the truth and not a placeholder — see
// isFreeBackend for why a zero here is load-bearing.
func tokenCost(b *Backend, promptTokens, outputTokens int) float64 {
	if b == nil {
		return 0
	}
	return float64(promptTokens)/1e6*b.InputPricePerMtok + float64(outputTokens)/1e6*b.OutputPricePerMtok
}

// sameEndpointModel reports whether two rows describe the same servable thing.
//
// The router treats a row as ONE servable thing, so a catalogue endpoint that
// serves hundreds of models needs one row per model rather than one row with a
// list. (endpoint, model) is therefore the real identity of a row, and the id is
// only its name. Two rows that agree on both are the same row entered twice.
//
// URLs compare with the trailing slash and any /v1 suffix removed, because
// https://api.example.com/v1 and https://api.example.com/ reach the same
// endpoint — upstreamPathURL already collapses that difference when it dials.
func sameEndpointModel(a, b *Backend) bool {
	return backendRootURL(a) == backendRootURL(b) && a.Model == b.Model
}
