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
	// sourceRelay is a row DERIVED from another discrimen's fleet (see relay.go).
	// It owns none of its values — every one is imported from upstream on each
	// refresh — which is why it is neither of the two above: a beacon's values are
	// a seed the measurement replaces, a manual row's are the operator's to keep,
	// and a relay row's are somebody else's measurement, replaced wholesale each
	// time it is re-read.
	sourceRelay = "relay"
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
	// Only the exact spellings count. A typo must not silently grant a row
	// operator ownership of its own values, nor mark it as somebody else's fleet;
	// anything unrecognised falls back to beacon, which owns nothing and is what
	// every registration predating these fields is.
	switch strings.ToLower(strings.TrimSpace(reg.Source)) {
	case sourceManual:
		reg.Source = sourceManual
	case sourceRelay:
		reg.Source = sourceRelay
	default:
		reg.Source = sourceBeacon
	}
	if reg.Source != sourceRelay {
		reg.Relay = ""
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

// isRelayRow reports whether a row was derived from another router's fleet.
//
// The distinction matters wherever this router would otherwise treat the row's
// URL as an endpoint: it is a router, so it must not be probed, benchmarked or
// asked what weights it loaded, and the thinking gate spoken to it is the
// client-facing one rather than any endpoint's dialect. See relay.go.
func isRelayRow(b *Backend) bool { return b != nil && b.Source == sourceRelay }

// isLocalProvider reports whether a provider name means "runs here, and nobody
// bills for it". The empty string counts, because that is what
// normalizeProviderFields settles it to — a caller that has not normalised yet
// must not get a different answer from one that has, and every path that reaches
// the price code can be reached before normalisation by a test.
func isLocalProvider(provider string) bool {
	p := strings.ToLower(strings.TrimSpace(provider))
	return p == "" || p == providerLocal
}

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

// beaconMaxConcurrency is the ceiling a SELF-registering worker declared, or 0.
//
// It behaves like llama.cpp's total_slots rather than like an operator's
// declaration: it can only ever LOWER the ramp, never raise it. A beacon row is
// the worker describing itself, so letting it inflate its own capacity would let
// a misconfigured start.sh claim whatever it liked; but a worker does know its
// own engine's hard limit, and the ramp cannot always see it.
//
// SGLang is the case that needs this. It QUEUES past --max-running-requests
// instead of refusing, so eight concurrent callers all get answers and
// measureConcurrent reads eight slots' worth of aggregate throughput off a
// worker running six. Measured on llm-6000pro-qwen38-flash-next 2026-08-30:
// declared 6, ramp 8, "8 concurrent (aggregate tok/s 1:108 2:179 4:262 8:331)".
// The over-count is not free — it is what the queue-depth term in
// expectedLatency prices against.
func (r *Registry) beaconMaxConcurrency(id string) int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	b := r.backends[id]
	if b == nil || b.lastReg == nil || isManualRow(b) {
		return 0
	}
	return b.lastReg.MaxConcurrency
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

// alreadySeeded reports the price and context fields a row's STORED declaration
// has already settled, so an edit that does not mention them does not hand them
// back to the seeder.
//
// It exists because statedness cannot be read off a stored row — an operator's
// explicit 0 and a field they left blank both persist as 0, and recording the
// difference would need a field on BackendRegistration. It does not have to be
// recorded, because seeding is deterministic in (model, provider): every field
// the table could fill for that pair was filled the first time the row was
// written. So while the pair is unchanged, a field still at zero is either one
// the operator declared zero or one the table publishes nothing for, and
// re-seeding it is a no-op at best and the bug at worst. (The exception is a
// snapshot refreshed between the two writes, which could now publish a number it
// did not before. That is a number the operator can still type, and it is not
// worth being unable to say "free" for.)
//
// Point the row at a different model or provider and the pair is new, so nothing
// has been settled for it and the row goes back to being seedable — which is the
// one case re-seeding on an edit was ever useful for.
func alreadySeeded(before, after BackendRegistration) priceStated {
	if !strings.EqualFold(strings.TrimSpace(before.Model), strings.TrimSpace(after.Model)) {
		return priceStated{}
	}
	if !strings.EqualFold(strings.TrimSpace(before.Provider), strings.TrimSpace(after.Provider)) {
		return priceStated{}
	}
	return priceStated{Input: true, Output: true, Context: true}
}

// restore puts a row back exactly as it was — measurements, certification and
// all. It has one caller: an EDIT whose persistence failed, where remove() would
// be wrong. A create that cannot be written has nothing on disk to come back
// from, so removing it is the truth; an edit that cannot be written leaves the
// PRE-EDIT row on disk, so removing it takes a provider out of routing and
// /v1/models until a restart resurrects it. Same shape as saveGroup.
//
// upsert would not do here: it takes a registration, so the row would come back
// "probing" with every measured value cleared and its certification restarted —
// a different row from the one the failed edit was supposed to leave untouched.
func (r *Registry) restore(b *Backend) {
	if b == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	cp := cloneBackend(b)
	r.backends[cp.ID] = cp
	// Put the slot channel back on the pre-edit ceiling too, or an edit that
	// changed max_concurrency would leave the failed value enforced.
	if cp.MaxConcurrency > 0 {
		r.syncSlotsLocked(cp.ID, cp.MaxConcurrency)
	} else {
		delete(r.slots, cp.ID)
		delete(r.slotCap, cp.ID)
	}
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
// Everything cost does to routing is below, and it is deliberately three small
// functions. Price is a declared fact about an endpoint, in the same category
// as the `uncensored` tag, and NOT a routing knob: there is no cost weight to
// tune, no price threshold to set and no budget mode. Free versus paid is a
// binary derived from a DECLARED price of zero, which is what makes the rule
// need no tuning — every local worker declares zero, so "prefer the free ones"
// is already right on the fleet this router grew up on.
//
// The weight the word "declared" carries is the whole of isPriceUnknown: a row
// that simply has no price is not the same claim as one that says it is free,
// and only the second may be treated as free.

// isFreeBackend reports whether an endpoint costs nothing per token.
//
// Zero prices only mean FREE where a zero is a DECLARATION, which is the word
// the paragraph above is careful to use. On the fleet this router grew up on it
// always is one: a local worker declares zero because nobody bills for the GPU
// in the next room, and a beacon declares zero because /backends/register is
// frozen and carries no price at all — so every worker deployed today is free,
// and stays free, with nothing set.
//
// A row an operator entered by hand for SOMEONE ELSE'S endpoint is the case
// where a zero is not necessarily a declaration. It is also what the row holds
// when nobody typed a number and the model is absent from the embedded table,
// and reading THAT as free is backwards in every direction at once: the endpoint
// sorts to the head of the free band, the free-first grace holds requests for it,
// the judge picks it as the free grader and grades against it forever, and the
// paid-spill log line that would have told the operator never fires.
//
// The two are distinguishable without storing anything. Every manual row is
// seeded on the way in and re-checked on every edit (see alreadySeeded), so a
// model the table DOES publish could only be sitting at zero because someone
// overrode it — that is a declaration, and it is how a free tier on a metered
// endpoint stays sayable. A model it publishes nothing for was never seedable,
// so its zero says nothing, and unknown has to fail towards paid: guessing wrong
// that way costs a little latency, and guessing wrong the other way costs money.
func isFreeBackend(b *Backend) bool {
	if b == nil {
		return true
	}
	if b.InputPricePerMtok > 0 || b.OutputPricePerMtok > 0 {
		return false
	}
	return !isPriceUnknown(b)
}

// isPriceUnknown reports a row that costs something nobody has said what: an
// operator-entered row on a provider that is not local, still at zero, for a
// model the embedded table publishes no price for. See isFreeBackend.
//
// The table is only consulted for that last group. A local worker and a beacon
// answer false on the fields already in hand, so the fleet never touches it.
func isPriceUnknown(b *Backend) bool {
	if b == nil || !isManualRow(b) || isLocalProvider(b.Provider) {
		return false
	}
	if b.InputPricePerMtok > 0 || b.OutputPricePerMtok > 0 {
		return false
	}
	return !publishesPrice(b.Model, b.Provider)
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
