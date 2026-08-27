package router

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"path"
	"regexp"
	"strings"
	"sync"
	"time"
)

// WorkerProfile is everything the router measures about a worker at cold start,
// so the worker itself declares almost nothing (just id/url/api_key). It's
// persisted per (id, model) and reused on warm restarts — see the /llm skill and
// the README.
type WorkerProfile struct {
	Model string `json:"model"`
	// ModelHash is WHICH MODEL this profile measured, recorded so the results
	// outlive the worker: a decommissioned box leaves its evidence behind, and
	// the same weights deployed elsewhere inherit it. See modelHash.
	//
	// A profile CACHED before this field existed carries "", and it too gets no
	// clause in backfillCachedProfile — for a different reason from CapacityCurve,
	// and a stronger one. There is nothing to re-measure: the identity of a run
	// that already happened is not a property of the endpoint today, and asking a
	// live worker for its fingerprint would attribute an old profile's answers to
	// whatever is serving that id NOW. The empty value is handled where it is read
	// instead — backfillOutcomesFromProfiles falls back to unfingerprintedModelHash
	// (see outcomes_bank.go), which is the non-sharing direction: an undated,
	// unfingerprinted profile keeps its own per-worker identity and can never be
	// credited to the wrong model.
	ModelHash string `json:"model_hash,omitempty"`
	Quality   int    `json:"quality"`
	// QualityNoThink is the same weighted benchmark scored with thinking
	// DISABLED on every question — the model a requirements.thinking="off"
	// client actually talks to. The headline Quality grades hard tiers
	// thinking-on, and the two can diverge sharply on MoE reasoners (measured
	// 2026-08-24: a 35B A3B at q=84 thinking wrote deterministic garbage SQL
	// no-think). Zero means NOT MEASURED. On a thinking worker that zero is
	// what selection reads for no-think requests — unmeasured ranks below
	// every measured worker rather than inheriting the mixed score (which
	// once let a still-profiling worker outrank the whole measured fleet); a
	// non-thinking worker falls back to Quality, which for it is exact. See
	// qualityFor.
	QualityNoThink       int    `json:"quality_nothink,omitempty"`
	QualityNoThinkDetail string `json:"quality_nothink_detail,omitempty"`
	ContextK             int    `json:"context_k"`
	// ContextProbe is the measured usable window (contextprobe.go). Persisted
	// with the profile so a warm restart keeps it, but NOT keyed by
	// benchmarkVersion: it is a capability measurement like speed, not part of
	// the quality score, so changing the ladder does not re-profile the fleet.
	ContextProbe   *ContextProbe `json:"context_probe,omitempty"`
	MaxConcurrency int           `json:"max_concurrency"`
	// CapacityCurve is what the ramp measured on the way to MaxConcurrency, one
	// entry per concurrency level it accepted. The ramp collected it all along and
	// threw it away, keeping only the integer it inferred: see concurrencyAlpha for
	// the question the integer cannot answer and the curve can.
	//
	// A profile CACHED before this field existed carries none, and nothing
	// backfills it. That is now a DECIDED omission rather than an outstanding one:
	// backfillCachedProfile — whose own comment says every new WorkerProfile field
	// needs a clause there or it inherits the prefill probe's silent staleness
	// (shipped, cached over, never measured) — states the exception in full. Re-
	// deriving a curve means re-running the concurrency ramp, which fires up to
	// CapacityProbeMax simultaneous generations at a worker already serving live
	// traffic; that is the most disruptive probe there is, and nothing like the
	// one-GET cost that makes the context re-read unconditional there.
	//
	// The staleness is bounded rather than silent: an absent curve prices as
	// α = 1, the neutral default, so an already-profiled fleet keeps exactly the
	// load model it had before the curve existed — never priced WRONGLY, only
	// un-refined — and each worker acquires one at its next cold profile.
	CapacityCurve []CapacityLevel `json:"capacity_curve,omitempty"`
	BaselineTPS   float64         `json:"baseline_tps"`
	TTFTMillis    int64           `json:"ttft_millis"`
	// PrefillTPS is prompt tokens per second, measured on a prompt of known length.
	// Unlike TTFTMillis it scales with the request, which is what routing needs: the
	// spread across the fleet is far wider for prefill than for decode (0.67s vs 37.2s
	// on the same ~4k prompt), so a flat TTFT average misprices long prompts badly.
	PrefillTPS float64 `json:"prefill_tps,omitempty"`
	// SpeedVersion is the version of the DECODE measurement BaselineTPS was taken
	// with. Separate from BenchVersion because the two have different
	// re-measurement costs: correcting how decode is counted is one 64-token
	// generation per worker, while a BenchVersion bump invalidates every cached
	// profile in the fleet at once and parks each worker at a provisional quality
	// until its re-profile finishes. A speed-probe correction should not cost that.
	//
	// What that bump costs has DROPPED sharply, and it is worth being precise about
	// why, because the old figure (hours per worker, the whole fleet at once) is
	// still the one people carry. The re-profile re-runs the pass over every
	// question, but a bench version is not part of a question's identity: benchOne
	// looks each one up in the permacache by qid — prompt, expected answer, match
	// mode and GRADER version — against this model's hash, and a hit skips the
	// generation entirely (see identity.go and cachedVerdict). So a bump that adds
	// questions or changes the scoring re-asks only what is genuinely new; the cheap
	// probes at the front of profileBackend, the capacity ramp and the context
	// ladder, are what actually get re-run. It is the questions that made a bump
	// expensive, and they no longer are.
	SpeedVersion int      `json:"speed_version,omitempty"`
	Features     []string `json:"features"`
	Thinking     bool     `json:"thinking"`
	// ThinkingDialect is the spelling of the thinking gate this endpoint was
	// MEASURED to honour (one of the thinkingDialect* constants).
	//
	// Empty means UNKNOWN, not unsupported: a profile cached before the dialect
	// was probed has the zero value, and so does a worker whose probe never got a
	// verdict. Unknown falls back to chat_template_kwargs, the spelling the whole
	// fleet has always spoken — a zero value that meant "cannot think" would
	// silently switch thinking off across the fleet on the first warm restart.
	ThinkingDialect string `json:"thinking_dialect,omitempty"`
	BenchVersion    int    `json:"bench_version"`            // question-set version this quality was measured against
	QualityDetail   string `json:"quality_detail,omitempty"` // per-tier + truncation breakdown
	// CategorySummary is the one-line "coding 47%→11%  maths 90%→85%" form,
	// rendered once here rather than per request. A tier is a difficulty band and
	// cuts across every subject, so the per-tier line above cannot answer "how is
	// it at coding?" — this can. See benchCategorySummary.
	CategorySummary string        `json:"category_summary,omitempty"`
	Failed          []string      `json:"failed,omitempty"`        // labels of benchmark questions the worker missed
	BenchResults    []BenchResult `json:"bench_results,omitempty"` // full per-question Q&A from the most recent run
	// BenchResultsNoThink is the same, for the no-think pass: aligned index-for-
	// index with BenchResults (and so with benchmarkQuestions), or empty.
	//
	// It exists because QualityNoThinkDetail — the per-tier line "t1=4/4 t2=3/4 …"
	// — cannot be split by anything except tier. A per-CATEGORY no-think score
	// needs to know WHICH questions were missed, not how many, and no tier maps
	// to one category: tier 4 mixes compiler gotchas with arithmetic, and the
	// generated half lands maths and reasoning items in the same tiers. Without
	// the per-question record the breakdown can show one mode only, which is the
	// mode a no-think client never talks to.
	//
	// EMPTY IS NOT ZERO, and every consumer has to keep that straight: a worker
	// really can score 0 with thinking off (measured 2026-08-24, a 35B A3B at
	// q=84 thinking wrote deterministic garbage no-think), so a missing run and a
	// terrible one must not render the same. serveBenchmark omits the no-think
	// half of the category breakdown rather than sending zeroes.
	//
	// POPULATED, from one of two places, and full-length either way. On a THINKING
	// worker it is the second pass's own record: runNoThinkQualityBenchmark re-asks
	// only the hard tiers, then rebuilds a full-length slice by carrying the mixed
	// pass's easy-tier rows across verbatim — those were already asked thinking-off,
	// so they ARE no-think results. On a worker with no thinking mode there is no
	// second pass at all, and BenchResults is copied here as-is for the same reason:
	// every answer it gave was a no-think answer.
	//
	// The alignment is what makes it safe for the category breakdown to zip this
	// against BenchResults and read both modes of one question off the same index.
	BenchResultsNoThink []BenchResult    `json:"bench_results_nothink,omitempty"`
	Checks              map[string]Check `json:"checks,omitempty"`
	MeasuredAt          time.Time        `json:"measured_at"`
	ProfileMillis       int64            `json:"profile_ms,omitempty"` // wall time of the full cold-start profile (capacity ramp + quality benchmark)
	// What the run that produced this profile consumed, and what that cost at the
	// endpoint's declared prices. Profiling a paid model spends real money — the
	// set is 130 questions, 122 of them graded thinking-on with a 16k ceiling, so a
	// cold profile lands near 250-400k output tokens — and a number nobody can see
	// is a number nobody can budget for (PLAN.md, "Known costs").
	//
	// A ZERO token count means NOT MEASURED, not free: every profile cached before
	// these fields existed has one, and there is no way to re-derive what a run
	// that already happened cost short of paying for another. Free is tokens > 0
	// with ProfileCost == 0, which is every local worker. Read them as a pair.
	//
	// Scoped exactly like ProfileMillis, to profileBackend — which re-runs the
	// quick probes itself, so the two describe the same span.
	ProfilePromptTokens int     `json:"profile_prompt_tokens,omitempty"`
	ProfileOutputTokens int     `json:"profile_output_tokens,omitempty"`
	ProfileCost         float64 `json:"profile_cost,omitempty"` // in whatever currency the operator is billed in
	// Incomplete marks a profile where one or more capability probes never got a
	// verdict (transient errors exhausted their retries). The worker stays
	// routable on these values, but the profile must NOT be persisted — a cached
	// "not detected" from a probe blip would misroute traffic until the next
	// benchmarkVersion bump (the 2026-07-06 thinking incident's stickiness half).
	Incomplete bool `json:"-"`
}

// provisionalQuality is the conservative tier a worker gets between the quick
// profile and the background quality benchmark — low enough (on the 0–100 quality
// percentage scale) that an unproven worker only draws easy traffic until it earns more.
const provisionalQuality = 30

// CapacityLevel is one rung of the capacity ramp: the aggregate throughput a
// worker sustained with N requests running at once.
//
// The ramp measured every rung and used each only to compare against the last
// one (`agg < prev*1.15`), then dropped it. What survived was the integer — how
// many at once — which cannot answer the question routing actually asks, namely
// how much slower EACH of them is while the worker is doing that.
type CapacityLevel struct {
	N   int     `json:"n"`   // concurrent requests in flight
	TPS float64 `json:"tps"` // aggregate tokens/sec summed across all N
}

// concurrencyAlpha fits the exponent α in agg(n) ≈ agg(1)·n^α to a measured
// capacity ramp. It is the whole reason the curve is now kept.
//
// Per-request throughput at batch n is agg(n)/n = agg(1)·n^(α-1), so one
// request's service time inside a batch of n is n^(1-α) times what it costs
// alone — a continuous, monotonic load term, which is what expectedLatency needs
// and what a step function could not give it (see loadPenalty). α = 1 is perfect
// batching, α = 0 a worker that serialises.
//
// Least squares in log-log space, forced through (1, agg(1)) because that point
// is the definition of the axis rather than an observation on it. Clamped to
// [0,1]: above 1 is super-linear scaling, which is sampling noise and not a
// property of any batching engine, and below 0 is throughput that FELL, which
// the ramp reads as a plateau and stops climbing at anyway.
//
// Returns 1 — no penalty at all — for a curve too short to fit. An unmeasured
// slowdown has to cost nothing, or every worker profiled before this existed
// silently acquires an invented one.
func concurrencyAlpha(curve []CapacityLevel) float64 {
	base := 0.0
	for _, lv := range curve {
		if lv.N == 1 && lv.TPS > 0 {
			base = lv.TPS
		}
	}
	if base <= 0 {
		return 1
	}
	var sxx, sxy float64
	for _, lv := range curve {
		if lv.N <= 1 || lv.TPS <= 0 {
			continue
		}
		x := math.Log(float64(lv.N))
		y := math.Log(lv.TPS / base)
		sxx += x * x
		sxy += x * y
	}
	if sxx <= 0 {
		return 1
	}
	return clamp01(sxy / sxx)
}

// minMeasuredAlpha keeps a measured exponent distinguishable from an unmeasured
// one on a plain float field.
//
// concurrencyAlpha legitimately returns 0 — a worker whose aggregate throughput
// does not grow with batch size at all, which is what a single-stream llama.cpp
// row measures — and that is the OPPOSITE of the unmeasured default of 1. On a
// float field with no presence bit the two would be the same number, and every
// fully-serial worker would silently be priced as a perfect batcher: exactly the
// wrong direction, since that worker is the one whose queue actually hurts.
//
// A measured zero is therefore stored as this floor instead. The cost is
// arithmetically nil — loadPenalty raises batch to 1-alpha, so 0.01 differs from
// 0 by batch^0.01, under 3% at a batch of 16 — and it buys an unambiguous zero
// value. The alternative, a *float64, would alias across cloneBackend.
const minMeasuredAlpha = 0.01

// setConcurrencyAlpha records what a capacity ramp measured about how a worker
// degrades under load. A curve with nothing to fit records nothing, leaving the
// worker at the neutral default — see concurrencyAlphaFor.
//
// Callers hold the registry write lock and own b.
func setConcurrencyAlpha(b *Backend, curve []CapacityLevel) {
	if b == nil || len(curve) < 2 {
		return
	}
	alpha := concurrencyAlpha(curve)
	if alpha < minMeasuredAlpha {
		alpha = minMeasuredAlpha
	}
	b.ConcurrencyAlpha = alpha
}

// concurrencyAlphaFor is the exponent routing prices a backend's load with. The
// default of 1 (perfect batching, no penalty) covers a worker whose ramp predates
// the curve, one still on its provisional profile, and a relay row whose upstream
// sends a capacity but no curve behind it — all cases where the router has not
// measured the worker under load and should not invent a penalty for it.
func concurrencyAlphaFor(b *Backend) float64 {
	if b == nil || b.ConcurrencyAlpha <= 0 {
		return 1
	}
	return b.ConcurrencyAlpha
}

// A capacity-probe level is only believed to be the ceiling after this many
// consecutive failures, spaced by the delay below. Cheap insurance: the measured
// value is cached per (id, model) and never re-measured on its own, so a single
// false negative here permanently under-rates a worker.
//
// The delay is a package var for the same reason slotMaxWait and qualityFloorWait
// are — a test that has to spend nine real seconds proving a retry ladder works
// is a test nobody runs.
const capacityProbeAttempts = 3

var capacityProbeRetryDelay = 3 * time.Second

// profileMeter accumulates the tokens one profiling run consumes at one
// endpoint, so the money it spent can be recorded on the profile it produced.
//
// It is registered per backend id for the span of profileBackend rather than
// threaded through the dozen probe functions under it. doCompletion is the
// single funnel every non-streamed probe and every benchmark question already
// goes through, so metering there picks up the probes that exist today and the
// ones added later, with no signature to keep in sync — and the benchmark is
// ~99% of the bill.
//
// The cost of that choice is attribution: anything else the router sends to
// this endpoint during the run — a background judge call, say — lands here too.
// That is a fraction of a percent of a 200-300k token profile, and it errs
// HIGH, which is the safe direction for a number an operator will reconcile
// against an invoice.
type profileMeter struct {
	mu     sync.Mutex
	prompt int
	output int
}

func (m *profileMeter) add(prompt, output int) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.prompt += prompt
	m.output += output
}

func (m *profileMeter) totals() (prompt, output int) {
	if m == nil {
		return 0, 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.prompt, m.output
}

// meterProfile opens a metering span for one endpoint and returns the meter and
// its closer. Overlapping profiles of the same id cannot happen — certifyBackend
// holds a single atomic guard for the whole span — so one meter per id is enough.
func (r *Router) meterProfile(id string) (*profileMeter, func()) {
	m := &profileMeter{}
	r.profileMeters.Store(id, m)
	return m, func() { r.profileMeters.Delete(id) }
}

// profileSpend reads what a run has consumed so far and prices it. Prices come
// from the LIVE row rather than the clone the profile was started with: an
// operator may have corrected them while the benchmark ran, and the later number
// is the one they will be billed at.
func (r *Router) profileSpend(b *Backend, meter *profileMeter) (prompt, output int, cost float64) {
	prompt, output = meter.totals()
	priced := b
	if live := r.registry.get(b.ID); live != nil {
		priced = live
	}
	return prompt, output, tokenCost(priced, prompt, output)
}

// abortedProfile is a profiling run discarded before it produced anything —
// carrying what it spent on the way.
//
// The tokens are gone either way; the point of carrying them is that a discarded
// run must not look free. Only the SUCCESS path writes ProfilePromptTokens and
// friends onto a profile, so an abort used to lose the number entirely: a
// metered endpoint that fails the benchmark under its own rate limiting can run
// the full 130-question set, bill for it, be discarded, and repeat — with
// ProfileCost recorded as zero and nothing at all in front of the operator.
type abortedProfile struct {
	reason string
	prompt int
	output int
	cost   float64
}

func (e *abortedProfile) Error() string {
	spend := fmt.Sprintf("%.0fk prompt + %.0fk completion tokens",
		float64(e.prompt)/1000, float64(e.output)/1000)
	if e.cost > 0 {
		spend += fmt.Sprintf(", %.4g at declared prices", e.cost)
	}
	return fmt.Sprintf("%s; discarding partial measurements (spent %s)", e.reason, spend)
}

// abort discards a profiling run, recording what it consumed before it was.
func (r *Router) abort(b *Backend, meter *profileMeter, reason string) error {
	prompt, output, cost := r.profileSpend(b, meter)
	return &abortedProfile{reason: reason, prompt: prompt, output: output, cost: cost}
}

// meterProfileTokens folds a measured token count into the profiling run in
// flight against this endpoint. A no-op — one failed map load — when the
// endpoint is not being profiled, which is every live request.
func (r *Router) meterProfileTokens(id string, prompt, output int) {
	if m, ok := r.profileMeters.Load(id); ok {
		m.(*profileMeter).add(prompt, output)
	}
}

// meterProfileUsage is meterProfileTokens reading an OpenAI usage block. The
// endpoint's OWN count, not an estimate: it is the number the invoice is
// computed from, and measure-don't-trust cuts this way too.
func (r *Router) meterProfileUsage(id string, raw map[string]any) {
	u, ok := raw["usage"].(map[string]any)
	if !ok {
		return
	}
	in, _ := u["prompt_tokens"].(float64)
	out, _ := u["completion_tokens"].(float64)
	r.meterProfileTokens(id, int(in), int(out))
}

// fmtProfileDuration renders a profile's wall time as "6m 48s" (minutes + seconds).
func fmtProfileDuration(ms int64) string {
	d := (time.Duration(ms) * time.Millisecond).Round(time.Second)
	return fmt.Sprintf("%dm %02ds", int(d/time.Minute), int((d%time.Minute)/time.Second))
}

// profileQuick runs the FAST half of profiling — capability discovery, speed, and
// context — so a brand-new worker becomes routable in seconds. Quality and
// capacity stay provisional (the declared seed or a conservative default) until
// profileBackend measures them in the background.
//
// The two stop being provisional at different MOMENTS, and the MaxConcurrency=1
// written here is why that matters. Capacity is published the instant the ramp
// settles it, part-way through profileBackend and long before the run finishes
// (see publishCapacity) — because this placeholder prices the worker as serial,
// and leaving it in place for the length of a quality benchmark spilled live
// traffic off a worker the profile was itself driving four ways. Quality really
// does wait for the end of the run, which is the only point at which there is a
// score to commit.
func (r *Router) profileQuick(b *Backend, model string) (*WorkerProfile, error) {
	checks := map[string]Check{}
	if err := r.chatProbe(b); err != nil {
		return nil, fmt.Errorf("chat probe failed: %w", err)
	}
	checks["chat"] = Check{OK: true}
	features := []string{"chat"}

	incomplete := false
	// json and tools are the two capabilities settled by an error-returning probe,
	// and they were settled by two byte-identical eleven-line blocks differing only
	// in the probe called and the name written down. Identical handling wants to be
	// one piece of code: the three-way verdict (supported / definitively refused /
	// never got an answer) is the same for both, and the third arm — the one that
	// must set Incomplete so the profile is not CACHED as a measurement it never
	// took — is precisely the arm nobody notices is missing from a copy.
	for _, capability := range []struct {
		name  string
		probe func() error
	}{
		{"json", func() error { return r.jsonProbe(b) }},
		{"tools", func() error { return r.toolProbe(b) }},
	} {
		err, inconclusive := r.capabilityProbe(capability.probe)
		switch {
		case err == nil:
			features = append(features, capability.name)
			checks[capability.name] = Check{OK: true}
		case inconclusive:
			incomplete = true
			checks[capability.name] = Check{Message: "inconclusive (probe errored, will re-probe): " + err.Error()}
		default:
			checks[capability.name] = Check{Message: err.Error()}
		}
	}
	thinking, dialect, thinkInconclusive := r.thinkingProbe(b)
	switch {
	case thinkInconclusive:
		incomplete = true
		checks["thinking"] = Check{OK: false, Message: "inconclusive (probe errored, will re-probe)"}
	case thinking:
		checks["thinking"] = Check{OK: true, Message: "supported via " + dialect}
	default:
		checks["thinking"] = Check{OK: false, Message: "not detected"}
	}

	// Vision is detected empirically — send an image, see if the worker reads it —
	// rather than trusted from a declaration (see visionProbe).
	vision, visionInconclusive := r.visionProbe(b)
	if visionInconclusive {
		incomplete = true
		checks["vision"] = Check{OK: false, Message: "inconclusive (probe errored, will re-probe)"}
	} else {
		checks["vision"] = Check{OK: vision, Message: mapBool(vision, "detected", "not detected")}
	}
	if vision {
		features = appendUnique(features, "vision")
	}

	// Preserve declared semantic tags the router can't fully settle by probing.
	for _, tag := range b.Features {
		switch tag {
		case "uncensored": // pure policy — unprobeable, so trust the declaration
			features = appendUnique(features, tag)
		case "vision": // probed above; keep a declared tag as a robustness backstop
			features = appendUnique(features, tag)
			if !vision {
				log.Printf("worker %s declares vision but the probe couldn't confirm it — keeping the declared tag; verify the model is multimodal", b.ID)
			}
		}
	}

	tps, ttft, err := r.speedProbe(b)
	if err != nil {
		return nil, fmt.Errorf("speed probe failed: %w", err)
	}
	checks["speed"] = Check{OK: true, Message: fmt.Sprintf("%.1f tok/s, ttft %dms", tps, ttft)}

	// Only a MANUAL row's declared quality is honoured. On a manual row the
	// number was typed by an operator reading a provider's documentation, and it
	// is authoritative by design (see providers.go). On a beacon row it is
	// whatever the worker said about itself — the exact self-declared quality the
	// measured benchmark replaced, and the README's registration contract
	// promises is not accepted ("No quality, no speed, no context window, no
	// concurrency").
	//
	// The window this closes is not small: a full benchmark is 8-30 minutes per
	// worker and has run to five hours on a slow one. A beacon claiming 100 would
	// have drawn the fleet's hardest prompts for all of it, on its own say-so.
	// Nothing deployed sends the field, so this changes no live behaviour — it
	// removes the way in.
	quality := 0
	if isManualRow(b) {
		quality = b.Quality
	}
	if quality <= 0 {
		quality = provisionalQuality
	}
	return &WorkerProfile{
		// Recorded here, at the end of the quick half, because queryModelInfo has
		// run by now and the fingerprint is complete. It is what lets these results
		// be found again after the worker is gone.
		ModelHash: modelHash(b),
		Model:     model, Quality: quality, ContextK: r.queryContext(b), MaxConcurrency: 1,
		BaselineTPS: tps, TTFTMillis: ttft, SpeedVersion: speedProbeVersion,
		Features: features, Thinking: thinking, ThinkingDialect: dialect,
		Checks: checks, MeasuredAt: time.Now(), Incomplete: incomplete,
	}, nil
}

// profileBackend is the FULL cold-start measurement: profileQuick plus the slow
// capacity ramp and quality benchmark. Run in the background after the worker is
// provisionally ready; it refines the live values and is persisted for warm
// restarts. Returns an error only if the worker can't serve chat at all.
func (r *Router) profileBackend(b *Backend, model string) (*WorkerProfile, error) {
	start := time.Now()
	meter, done := r.meterProfile(b.ID)
	defer done()
	p, err := r.profileQuick(b, model)
	if err != nil {
		return nil, err
	}
	// Carry the MEASURED context window onto the local Backend before anything
	// downstream reads it.
	//
	// b is a clone (Registry.get returns cloneBackend), and a beacon worker
	// declares no context at all — the payload is {id, url, model}. profileQuick
	// discovers the real window and writes it to the PROFILE, which reaches the
	// registry later via applyProfileIfGen; the clone in hand stays at zero for
	// the whole of this function. Two things silently depended on it:
	//
	//	runContextProbe saw AdvertisedTokens 0, returned without making a single
	//	HTTP call, and the `> 0` guard meant nothing was ever stored — so no
	//	beacon worker has ever had a measured usable context.
	//
	//	benchOne's usableContextTokens(b) was 0, so the answer-ceiling clamp was
	//	skipped and every hard question asked for the full 32768 tokens
	//	regardless of the window — on an 8K worker that truncates, and truncation
	//	is scored a failure. The same worker profiled WARM (after the health loop
	//	reconciles the registry) gets the clamp and passes. Same worker, opposite
	//	verdicts, decided by router lifecycle timing.
	if p.ContextK > 0 {
		b.ContextK = p.ContextK
	}
	prog := r.progressFor(b.ID)
	prog.enter(phaseCapacity, 0)
	capN, capCurve, capOK := r.capacityProbe(b)
	// Abort BEFORE the benchmark when the capacity probe already failed: a hung
	// worker otherwise burns 28 questions × attempts × the request timeout
	// (many hours) holding the profiling guard, only for the results to be
	// discarded here anyway. The worker keeps its provisional profile and the
	// caller schedules a retry.
	if !capOK {
		return nil, r.abort(b, meter, "worker unreachable during capacity probe")
	}
	capN = r.resolveCapacity(b, capN)
	// PUBLISH the measurement now, not at the end of the run.
	//
	// It used to sit in this local until after the quality benchmark, which is
	// 25+ minutes on a typical worker and has run to five hours on a slow one. For
	// all of it the router priced the worker off profileQuick's provisional
	// MaxConcurrency=1: expectedLatency's queue term reads that field, isFull reads
	// it, and the `ask -l` slots column reads it — so live traffic was spilled off
	// the worker as though it were serial, while THIS function drove the same
	// worker at up to 4-way concurrency for the whole benchmark.
	//
	// Cleanly separable from the atomic quality commit at the end. capOK is
	// validated immediately above, so what goes out here is a COMPLETED
	// measurement, not half of one. An abort further down discards the QUALITY the
	// run was after and leaves the worker on its provisional profile by design; it
	// does not un-measure the capacity, and should not — the ramp succeeded, and
	// the retry would only measure the same thing again.
	//
	// The applyProfileIfGen at the end writes this same number, and syncSlotsLocked
	// no-ops on an unchanged cap, so the commit stays a commit rather than becoming
	// a conflict that rebuilds the slot channel under live traffic.
	//
	// A false return is the SAME abort as the capacity probe failing above, arrived
	// at from the other direction: the row has been deleted, or re-registered with
	// new content, since this run took the profiling guard. applyProfileIfGen at the
	// end applies the identical generation check and drops the whole profile —
	// "background profile %s finished for a stale registration generation —
	// discarded" — so everything from here on is measurement that has already been
	// decided to be unusable. Carrying on spends the quality benchmark to learn it:
	// 25+ minutes typically, five hours at worst, and on a metered endpoint a real
	// invoice, for a result that is thrown away on arrival. The new registration is
	// running its own certification meanwhile, so nothing is lost by stopping.
	//
	// This return value was previously ignored, which is what made the early abort
	// above inconsistent: the two conditions cost the same and only one of them
	// stopped.
	if !r.registry.publishCapacity(b.ID, b.profileGen, capN, capCurve) {
		return nil, r.abort(b, meter, "worker deleted or re-registered during the capacity probe")
	}
	// Prefill rate is measured here rather than left to the live EWMA, which only
	// samples non-thinking requests and so never fills in for a thinking-heavy worker.
	//
	// A failure is not fatal, and what it costs is now smaller than it was. It used
	// to drop the worker onto a FLAT TTFT average that ignored prompt length
	// altogether — total blindness to the dominant term of a long-context request,
	// not merely a coarse estimate of it. prefillSeconds no longer has such a
	// branch: an unmeasured worker falls through to the context ladder's own
	// per-rung rate if the ladder ran, and failing that to fallbackPrefillTPS, a
	// fleet constant that is wrong for every worker but wrong by a bounded factor
	// and still scales with the prompt. So this probe now buys accuracy rather than
	// the difference between an estimate and none.
	prog.enter(phasePrefill, 0)
	if rate, err := r.prefillProbe(b); err != nil {
		log.Printf("prefill probe failed for %s: %v — routing will price its prefill from the context ladder or the fleet constant", b.ID, err)
	} else {
		p.PrefillTPS = rate
		p.Checks["prefill"] = Check{OK: true, Message: fmt.Sprintf("%.0f tok/s on a %d-token prompt", rate, prefillProbeTokens)}
	}
	// Deliberately BEFORE benchConc: the quality benchmark grades at this concurrency,
	// so an over-reported capacity makes half the questions queue behind a full
	// generation and blow benchAnswerDeadline. On llm-naples-deepseek-284B-q4 (probe
	// said 2, --parallel 1) that scored 46 questions "(too slow)" with not one wrong
	// answer, dragging a capable model down to quality 53.
	benchConc := capN
	if benchConc > 4 {
		benchConc = 4 // cap concurrency so profiling leaves headroom for live traffic
	}
	prog.enter(phaseQuality, len(benchmarkQuestions))
	quality, qOK, qBreakdown, qFailed, qResults := r.runQualityBenchmark(b, benchConc)
	r.auditThinkingGate() // log where the reasoning gate disagrees with tier (model-independent, once)
	// If the worker went unreachable mid-benchmark, the quality number is
	// garbage — abort rather than persist an under-rating. Note what it cost on
	// the way out: runQualityBenchmark only gives up AFTER wg.Wait(), so up to
	// half the 130 questions were generated and billed before we got here.
	if !qOK {
		return nil, r.abort(b, meter, "worker unreachable during quality benchmark")
	}
	p.MaxConcurrency = capN
	p.CapacityCurve = capCurve
	p.Checks["capacity"] = Check{OK: true, Message: capacityMessage(capN, capCurve)}
	p.Quality = quality
	p.BenchVersion = benchmarkVersion
	p.QualityDetail = qBreakdown
	p.CategorySummary = benchCategorySummary(qResults, nil)
	p.Failed = qFailed
	p.BenchResults = qResults
	qMsg := fmt.Sprintf("%d%%", quality)
	if qBreakdown != "" {
		qMsg += " " + qBreakdown // per-tier + truncation breakdown, visible via /backends/{id}
	}
	p.Checks["quality"] = Check{OK: true, Message: qMsg}
	// Second score: the worker with thinking disabled, for requests a client
	// forces no-think (see WorkerProfile.QualityNoThink). Only thinking-capable
	// workers diverge — for the rest the mixed run already asked every question
	// no-think, so the score is the same number and costs nothing to reuse.
	if p.Thinking {
		prog.enter(phaseQualityNT, 0)
		ntScore, ntOK, ntBreakdown, ntResults := r.runNoThinkQualityBenchmark(b, benchConc, qResults)
		if ntOK {
			p.QualityNoThink = ntScore
			p.QualityNoThinkDetail = ntBreakdown
			// Stored so the category breakdown can split the no-think half too:
			// a tier does not map to one category, so the per-tier line alone
			// cannot answer "how is it at coding with thinking off?".
			p.BenchResultsNoThink = ntResults
			// Recompute now that both runs are in hand, so the summary carries the
			// thinking-on→off arrow rather than the thinking-only half.
			p.CategorySummary = benchCategorySummary(qResults, ntResults)
			p.Checks["quality_nothink"] = Check{OK: true,
				Message: fmt.Sprintf("%d%% %s", ntScore, ntBreakdown)}
		}
		// A failed no-think pass leaves the field zero → the worker ranks
		// below every measured worker on no-think requests until the next
		// certification retries the pass (unmeasured must not inherit the
		// mixed score — see qualityFor); not worth aborting a profile the
		// main benchmark already earned.
	} else {
		// A worker with no thinking mode answers everything thinking-off, so its
		// two scores are the same number. Its RESULTS are likewise no-think
		// results, and they have to be recorded as such: without this,
		// BenchResultsNoThink stayed empty and every observation was filed under
		// Thinking=true — the evidence of a worker that cannot think, filed as
		// evidence about thinking. Routing then queried the no-think bucket, found
		// nothing, and reported the worker unmeasured in the only mode it has.
		p.QualityNoThink = quality
		p.QualityNoThinkDetail = qBreakdown
		p.BenchResultsNoThink = qResults
	}
	// Feed the outcome matrix. This is what routing will query — per question,
	// not per worker — so it is recorded from the SAME results the score above
	// was computed from, rather than re-derived later from a stored profile that
	// might have been written by a different grader.
	//
	// Best effort: a matrix write that fails costs prediction quality, and
	// failing the profile over it would throw away a run that has just cost
	// hours. The rows are re-created by the next profile.
	if r.outcomes != nil {
		at := time.Now().UTC()
		// The MIXED pass asks easy tiers thinking-off and hard tiers thinking-on
		// (benchOne gates on benchHardTier), so publishing all of it as
		// Thinking=true files a dozen no-think answers as thinking evidence. Each
		// result is recorded in the mode it was actually asked in.
		rows := observationsFromMixed(p.ModelHash, b.ID, p.BenchResults, at)
		rows = append(rows, observationsFrom(p.ModelHash, b.ID, p.BenchResultsNoThink, false, at)...)
		if err := r.outcomes.record(context.Background(), rows); err != nil {
			log.Printf("outcome matrix: recording %s failed: %v", b.ID, err)
		} else if err := r.ensureBankVectors(context.Background()); err != nil {
			// Without vectors the rows are unreachable — every prediction falls
			// back — so this is worth a line even though it is not fatal.
			log.Printf("outcome matrix: embedding the bank failed, predictions will fall back: %v", err)
		}
	}

	// Usable context, measured rather than believed. Last because it is the one
	// probe whose cost scales with the worker's own claim — a model advertising
	// 256K spends real minutes in prefill proving it — and because a failure here
	// must not cost the quality score already earned above.
	prog.enter(phaseContext, 0)
	if probe := r.runContextProbe(b, false); probe.AdvertisedTokens > 0 {
		p.ContextProbe = &probe
		// OK from the ladder rather than from the fact that it ran: a probe that
		// errored, and a worker that could not retrieve at the smallest size tested,
		// are both failures and used to render with a tick. See contextProbeOK.
		p.Checks["context_usable"] = Check{OK: contextProbeOK(&probe), Message: contextProbeMessage(&probe)}
	}
	p.MeasuredAt = time.Now()
	p.ProfileMillis = time.Since(start).Milliseconds()
	// What the run cost, recorded on the profile it produced so it survives the
	// restart and reaches the API.
	p.ProfilePromptTokens, p.ProfileOutputTokens, p.ProfileCost = r.profileSpend(b, meter)
	p.Checks["cost"] = Check{OK: true, Message: profileCostMessage(p)}
	return p, nil
}

// profileCostMessage renders a run's spend for the per-probe check list, which
// is what puts the number in front of an operator on /backends — cached
// profiles included, since the checks are persisted with them.
func profileCostMessage(p *WorkerProfile) string {
	tokens := fmt.Sprintf("%.0fk prompt + %.0fk completion tokens",
		float64(p.ProfilePromptTokens)/1000, float64(p.ProfileOutputTokens)/1000)
	if p.ProfileCost <= 0 {
		return tokens + "; free at declared prices"
	}
	return fmt.Sprintf("%s; %.4g at declared prices", tokens, p.ProfileCost)
}

func appendUnique(s []string, v string) []string {
	for _, x := range s {
		if x == v {
			return s
		}
	}
	return append(s, v)
}

// chatProbe verifies the worker serves chat completions at all.
func (r *Router) chatProbe(b *Backend) error {
	content, err := r.simpleCompletion(b, map[string]any{
		"model": probeModel(b), "stream": false, "max_tokens": 16,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
		"messages":             []map[string]string{{"role": "user", "content": "Reply with the single word: ready /no_think"}},
	})
	if err != nil {
		return err
	}
	if strings.TrimSpace(content) == "" {
		return errors.New("empty chat response")
	}
	return nil
}

// thinkingDialect names the spelling of the "think about this" gate an endpoint
// honours. Measured rather than assumed: enable_thinking inside
// chat_template_kwargs is a vLLM and llama.cpp extension, not part of the OpenAI
// API, so a provider either ignores it — thinking silently never happens, and
// the router believes it asked — or rejects the unknown field and fails the
// whole request, while reading reasoning_effort perfectly well.
const (
	thinkingDialectKwargs = "chat_template_kwargs" // enable_thinking nested in chat_template_kwargs
	thinkingDialectEffort = "reasoning_effort"     // the OpenAI-standard top-level field
	thinkingDialectNone   = "none"                 // neither gate produced a reasoning block
)

// thinkingProbeQuestion is genuinely multi-step on purpose — a trivial one lets
// an adaptive reasoning model skip its thinking block and false-negative the
// probe.
const thinkingProbeQuestion = "A train goes 60 km in 1.5 hours, then 90 km in 1 hour. What is its average speed for the whole trip in km/h? Show your reasoning step by step."

// thinkingProbe reports whether the worker emits a reasoning block when asked
// to, and WHICH spelling of the ask it honours. inconclusive=true means the
// probe never got a verdict (transient errors exhausted the retries) — the
// caller must not persist that as a durable "not detected" (see the Incomplete
// handling in profileQuick).
//
// The dialects are tried in fleet order: chat_template_kwargs first, because
// every local vLLM and llama.cpp worker speaks it and answers on the first
// request, then the standard reasoning_effort. Only a POSITIVE result settles
// it, so an endpoint that accepts the kwargs object and quietly discards it
// still gets its standard field tried before the router writes it off.
func (r *Router) thinkingProbe(b *Backend) (thinking bool, dialect string, inconclusive bool) {
	gates := []struct {
		dialect string
		field   string
		value   any
	}{
		{thinkingDialectKwargs, "chat_template_kwargs", map[string]bool{"enable_thinking": true}},
		// "high" rather than a milder level: the probe asks whether the field does
		// anything at all here, and the strongest ask is the least likely to be
		// answered without a reasoning block.
		{thinkingDialectEffort, "reasoning_effort", "high"},
	}
	unknown := false
	for _, g := range gates {
		payload := map[string]any{
			"model": probeModel(b), "stream": false, "max_tokens": 1024,
			"messages": []map[string]string{{"role": "user", "content": thinkingProbeQuestion}},
			g.field:    g.value,
		}
		thought, settled := r.thinkingProbeOnce(b, payload)
		if !settled {
			unknown = true
			continue
		}
		if thought {
			return true, g.dialect, false
		}
	}
	if unknown {
		return false, "", true
	}
	return false, thinkingDialectNone, false
}

// The transient-retry discipline every capability probe shares: a probe that
// errors is retried this many times, spaced by this delay, and only a SUCCESS or
// a definitive 4xx rejection settles it. The distinction is the point — a
// capability that never got a verdict marks the profile Incomplete rather than
// being cached as absent (the 2026-07-06 thinking incident's stickiness half) —
// and the numbers were written out separately at each of the three ladders that
// implement it, which is how two of them could drift apart without anyone
// noticing they were meant to agree.
//
// The delay is what makes the ladders slow to test, so it is a package var for
// the same reason capacityProbeRetryDelay and slotMaxWait are: a test that has to
// spend six real seconds proving a retry works is a test nobody runs.
const capabilityProbeAttempts = 3

var capabilityProbeRetryDelay = 2 * time.Second

// thinkingProbeOnce runs one dialect's probe under the same transient-retry
// discipline as capabilityProbe. settled=false means no verdict at all: every
// attempt hit a transient error.
func (r *Router) thinkingProbeOnce(b *Backend, payload map[string]any) (thinking, settled bool) {
	for attempt := 0; attempt < capabilityProbeAttempts; attempt++ {
		if attempt > 0 {
			time.Sleep(capabilityProbeRetryDelay)
		}
		raw, err := r.rawCompletion(b, payload)
		if err != nil {
			// A definitive 4xx settles THIS dialect — a strict endpoint refusing a
			// field it does not recognise — and says nothing about the other one.
			if isClientReject(err) {
				return false, true
			}
			continue // transient (load shedding, 5xx, transport) → retry
		}
		content, reasoning, _ := completionText(raw)
		return strings.TrimSpace(reasoning) != "" || strings.Contains(content, "<think>"), true
	}
	return false, false
}

// capabilityProbe runs an error-returning probe (json/tools) with the same
// transient-retry discipline as thinkingProbe/visionProbe: up to 3 attempts, a
// definitive 4xx reject or success short-circuits. inconclusive=true means
// retries were exhausted on transient errors only — the capability is UNKNOWN,
// not absent, and the profile must not be cached as if it were measured.
func (r *Router) capabilityProbe(probe func() error) (err error, inconclusive bool) {
	for attempt := 0; attempt < capabilityProbeAttempts; attempt++ {
		if attempt > 0 {
			time.Sleep(capabilityProbeRetryDelay)
		}
		err = probe()
		if err == nil || isClientReject(err) {
			return err, false
		}
	}
	return err, true
}

// capacityProbe estimates how many concurrent requests the worker handles before
// aggregate throughput stops scaling (the saturation knee) or it starts failing.
//
// It also returns the CURVE it climbed — one CapacityLevel per rung it accepted.
// That was always measured and always discarded, and it is the only thing in the
// router that describes how much a request slows down while sharing a worker
// rather than merely whether it fits; see concurrencyAlpha.
//
// The curve holds only the ACCEPTED rungs, not the plateau the ramp stopped on.
// A plateau point sits outside the concurrency the worker will ever be dispatched
// at, so fitting through it would flatten the exponent inside the range that IS
// used: a worker that scales perfectly to 2 and then flattens at 4 would come out
// at α≈0.63 and be charged a slowdown at a batch of 2 it does not have.
func (r *Router) capacityProbe(b *Backend) (capacity int, curve []CapacityLevel, ok bool) {
	maxN := r.cfg.CapacityProbeMax
	if maxN < 1 {
		maxN = 16
	}
	// A declared ceiling on a manual row bounds the ramp as well as settling it.
	// capacityProbeMax is one global number for a whole fleet; this is the
	// operator saying what THIS endpoint will take, and firing 32 concurrent
	// probes at a provider they have told us tops out at 4 buys nothing but rate
	// limiting — on a metered endpoint, paid for.
	if declared := r.registry.operatorMaxConcurrency(b.ID); declared > 0 && declared < maxN {
		maxN = declared
	}
	best, prev := 1, 0.0
	for _, n := range []int{1, 2, 4, 8, 16, 32, 64} {
		if n > maxN {
			break
		}
		// Retry a failed level before believing it. measureConcurrent fails the
		// whole level if ANY of its n requests errors, so a single transient blip
		// — a worker still capturing CUDA graphs seconds after it first answers,
		// a momentary 503 — otherwise becomes a permanent verdict: the ramp
		// breaks, capacity sticks at the last good level, and the value is cached
		// per (id, model) so it never re-measures. That is exactly how
		// llm-6000pro (--max-num-seqs 6, ~4x aggregate throughput at n=6) ended
		// up pinned to serial dispatch, funnelling fleet traffic onto slower
		// backends until an operator noticed.
		var agg float64
		var good bool
		for attempt := 0; attempt < capacityProbeAttempts; attempt++ {
			if attempt > 0 {
				time.Sleep(capacityProbeRetryDelay)
				log.Printf("capacity probe: retrying n=%d for %s (attempt %d/%d)", n, b.ID, attempt+1, capacityProbeAttempts)
			}
			if agg, good = r.measureConcurrent(b, n); good {
				break
			}
		}
		if !good {
			if n == 1 {
				return 1, nil, false // can't serve even one request → worker unreachable
			}
			log.Printf("capacity probe: %s failed n=%d after %d attempts; capacity=%d", b.ID, n, capacityProbeAttempts, best)
			break // repeatable errors at a higher level → that's the capacity ceiling
		}
		// Retry a PLATEAU verdict before believing it, for exactly the reason a
		// failed level is retried above: one noisy sample otherwise becomes a
		// permanent verdict. The knee compares two short throughput samples, and
		// on a speculative-decoding worker the acceptance rate alone swings the
		// per-sample rate by 2-3x (measured 35-100% on near-identical requests
		// on an MTP engine, 2026-08-18: its n=2 sample missed the 1.15x knee once
		// and a 2-lane worker was cached at serial dispatch). Capacity is a
		// capability claim — "CAN it run n concurrently" — so the level keeps its
		// best sample: noise can fake a plateau, it cannot fake genuine scaling.
		for attempt := 1; n > 1 && agg < prev*1.15 && attempt < capacityProbeAttempts; attempt++ {
			time.Sleep(capacityProbeRetryDelay)
			log.Printf("capacity probe: n=%d for %s plateaued (%.0f vs %.0f needed) — re-sampling (attempt %d/%d)",
				n, b.ID, agg, prev*1.15, attempt+1, capacityProbeAttempts)
			if again, ok := r.measureConcurrent(b, n); ok && again > agg {
				agg = again
			}
		}
		if n > 1 && agg < prev*1.15 {
			break // plateau held across every re-sample → saturation knee
		}
		best, prev = n, agg
		curve = append(curve, CapacityLevel{N: n, TPS: agg})
	}
	return best, curve, true
}

// publishCapacity applies a measured concurrency ceiling — and the throughput
// curve behind it — to a LIVE backend the moment the ramp settles them, instead
// of waiting for the profile the ramp is only the first step of.
//
// The guards are applyProfileIfGen's, for its reasons: a result from a stale
// registration generation is dropped (the worker re-registered or was deleted
// mid-measurement), and an operator's declared ceiling on a manual row is never
// overwritten by a probe. The slot channel is synced to the EFFECTIVE cap rather
// than the measured one, or that declared ceiling would be advisory only —
// again the same rule, and the same reason.
//
// Returns false when the row has gone or re-registered underneath the run.
func (r *Registry) publishCapacity(id string, gen int64, n int, curve []CapacityLevel) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil || (gen != 0 && b.profileGen != gen) {
		return false
	}
	if n > 0 && operatorDeclared(b).MaxConcurrency == 0 {
		b.MaxConcurrency = n
	}
	r.syncSlotsLocked(id, b.MaxConcurrency)
	setConcurrencyAlpha(b, curve)
	return true
}

// capacityMessage renders the ramp for the per-probe check list. The curve goes
// in beside the verdict because the verdict alone cannot say what it cost:
// "4 concurrent" says the worker takes four, and "1:900 2:1500 4:1900" says each
// of those four runs at 475 tok/s against 900 alone, so a request that lands in a
// full batch takes about twice as long as one that lands on an idle worker. That
// is the difference between a worker to send a burst to and one to spread away
// from, and nothing else in /backends shows it.
func capacityMessage(n int, curve []CapacityLevel) string {
	msg := fmt.Sprintf("%d concurrent", n)
	if len(curve) < 2 {
		return msg
	}
	parts := make([]string, 0, len(curve))
	for _, lv := range curve {
		parts = append(parts, fmt.Sprintf("%d:%.0f", lv.N, lv.TPS))
	}
	return fmt.Sprintf("%s (aggregate tok/s %s; scaling exponent %.2f)",
		msg, strings.Join(parts, " "), concurrencyAlpha(curve))
}

// resolveCapacity settles a worker's concurrency from the ramp's inference and
// the two PUBLICATIONS that outrank it. Both say the same thing — where a real
// number exists, an inference from aggregate throughput does not get to
// contradict it — but they guard opposite failures, so they move the answer in
// opposite directions.
//
// llama.cpp's total_slots can only ever LOWER it. The ramp over-reports on a
// worker that serialises: at n=2 a single-slot worker runs the two requests
// back-to-back, and if the samples are noisy the second level still clears the
// 1.15x knee test. --parallel is a hard limit regardless of what throughput
// appeared to show, but the ramp may still find a lower PRACTICAL knee than the
// configured slot count, so it caps rather than replaces.
//
// An operator's declared ceiling on a manual row replaces it outright, and can
// raise it. The failure there is the mirror image: a rate-limited endpoint
// answers a burst with 429s, measureConcurrent fails the whole level, and the
// verdict is cached per (id, model) and never re-measured on its own — so one
// throttled minute would cost that endpoint its capacity permanently.
func (r *Router) resolveCapacity(b *Backend, ramp int) int {
	if slots := r.querySlots(b); slots > 0 && ramp > slots {
		log.Printf("capacity probe: %s ramp suggested %d but the worker reports %d slot(s) — using %d",
			b.ID, ramp, slots, slots)
		ramp = slots
	}
	if declared := r.registry.operatorMaxConcurrency(b.ID); declared > 0 && declared != ramp {
		log.Printf("capacity probe: %s ramp measured %d but the operator declared %d — using the declared value",
			b.ID, ramp, declared)
		ramp = declared
	}
	return ramp
}

// measureConcurrent fires n identical short completions at once and returns the
// aggregate throughput (approx tokens/sec). ok is false if any request failed.
func (r *Router) measureConcurrent(b *Backend, n int) (float64, bool) {
	// The sample window has to be long enough to average out per-request noise:
	// at 64 tokens a fast worker finishes in ~0.3-0.4s and a speculative
	// decoder's acceptance-rate swing dominates the reading (see the plateau
	// retry in capacityProbe). Ten sentences under a 256-token cap runs the
	// window ~4x longer for a few thousand extra tokens per profile — noise in
	// the aggregate, next to the benchmark's ~99% share of the bill. The prompt
	// asks for the length; raising max_tokens alone would change nothing, the
	// model stops where the prompt stops it.
	payload := map[string]any{
		"model": probeModel(b), "stream": false, "max_tokens": 256,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
		"messages":             []map[string]string{{"role": "user", "content": "Write ten sentences about the ocean. /no_think"}},
	}
	type res struct {
		toks int
		err  error
	}
	ch := make(chan res, n)
	start := time.Now()
	for i := 0; i < n; i++ {
		go func() {
			content, err := r.simpleCompletion(b, payload)
			ch <- res{len(strings.Fields(content)), err}
		}()
	}
	total := 0
	for i := 0; i < n; i++ {
		x := <-ch
		if x.err != nil {
			return 0, false
		}
		total += x.toks
	}
	elapsed := time.Since(start).Seconds()
	if elapsed <= 0 {
		return 0, false
	}
	return float64(total) / elapsed, true
}

// ModelMeta is what a worker's runtime reports about the weights it actually
// loaded, as distinct from the model *name*, which is only ever a string an
// operator chose. The two disagree routinely: an unaliased llama.cpp worker
// advertises its gguf path, one started with --alias advertises that alias, and
// neither has to describe what is in the file (llm-6000pro-deepseek-284B-q8
// serves MXFP4 weights). Zero fields mean the runtime doesn't publish that
// datum — vLLM exposes no parameter count — so consumers must degrade rather
// than render zeroes. Embedded in Backend, so these serialise flat into
// /backends alongside "model".
type ModelMeta struct {
	ModelPath      string `json:"model_path,omitempty"`       // weights file the worker loaded
	ModelParams    int64  `json:"model_params,omitempty"`     // parameter count read off the weights
	ModelQuant     string `json:"model_quant,omitempty"`      // ftype, e.g. "Q4_K - Medium", "MXFP4 MoE"
	ModelSizeBytes int64  `json:"model_size_bytes,omitempty"` // size of the loaded weights
	ModelCtxTrain  int    `json:"model_ctx_train,omitempty"`  // context length the model was trained at
	// Engine is the serving runtime, measured rather than declared: llama.cpp
	// answers /props and vLLM does not. It is part of the model hash because the
	// two disagree on samplers and kernels, so the same weights can grade
	// differently under each — the one sharing case worth refusing.
	Engine string `json:"engine,omitempty"`
	// The id the worker's own /v1/models advertises, "default" included — the
	// one spelling the runtime is guaranteed to accept in a request's "model"
	// field. llama.cpp ignores that field, but vLLM 404s anything else, so
	// forwarded bodies are rewritten to this (see patchForwardedBody).
	ServedID string `json:"served_model_id,omitempty"`
}

// probeModel is the model name a probe puts in its request. Every probe used to
// send the literal "default": harmless against a single-model vLLM or llama.cpp
// worker, which either publishes that id or ignores the field, and fatal against
// an endpoint that validates model names — a strict provider failed
// certification for reasons that had nothing to do with its capabilities.
//
// Precedence follows how much the router actually knows about the endpoint:
//
//  1. ServedID, what the endpoint's own /v1/models advertises, and therefore the
//     one spelling its validator is guaranteed to accept.
//  2. The registration's declared model, which is all there is on the very first
//     probe against a brand-new registration — queryModelInfo has not run yet,
//     or the endpoint publishes no catalogue to read.
//  3. "default", the historical literal, when even that is empty.
func probeModel(b *Backend) string {
	if b == nil {
		return "default"
	}
	if b.ServedID != "" {
		return b.ServedID
	}
	if b.Model != "" {
		return b.Model
	}
	return "default"
}

// shardSuffix matches the multi-part GGUF shard counter ("-00001-of-00005")
// that a sharded model's first file carries in its name.
var shardSuffix = regexp.MustCompile(`-\d{5}-of-\d{5}$`)

// niceModelName is the human-readable name of what a backend serves, for
// display surfaces only (response headers, dashboards) — the profile
// fingerprint stays Backend.Model untouched, or renaming would re-profile the
// fleet. The advertised model id is used unless it is just the worker id
// echoed back (an --alias masking the real name), in which case the weights
// path recovers it; either way a file path reduces to its basename with the
// .gguf extension and any shard counter stripped, so
// "/models/UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf"
// displays as "DeepSeek-V4-Flash-0731-UD-Q4_K_XL".
func niceModelName(b *Backend) string {
	name := b.Model
	if (name == "" || name == "default" || name == b.ID) && b.ModelPath != "" {
		name = b.ModelPath
	}
	name = path.Base(name)
	name = strings.TrimSuffix(name, ".gguf")
	name = shardSuffix.ReplaceAllString(name, "")
	if name == "" || name == "." || name == "/" {
		return b.Model
	}
	return name
}

// queryModelInfo asks a worker what it is serving: the advertised model id (the
// profile cache fingerprint) and whatever hard metadata the runtime publishes
// about the weights behind that name. llama.cpp nests the metadata under the
// same /v1/models entry the id comes from, so the id costs nothing extra;
// /props adds the weights path, which is the only way to recover a real model
// name from a worker whose --alias masks it.
//
// Which entry that is comes from modelEntryFor, not from data[0] — see there.
func (r *Router) queryModelInfo(b *Backend) (string, ModelMeta) {
	var meta ModelMeta
	id := ""
	if raw, err := r.backendGET(b, "/v1/models"); err == nil {
		if m, ok := modelEntryFor(raw, b.Model); ok {
			if v, _ := m["id"].(string); v != "" {
				meta.ServedID = v
				if v != "default" {
					id = v
				}
			}
			if mm, ok := m["meta"].(map[string]any); ok {
				meta.ModelParams = int64(jsonNum(mm, "n_params"))
				meta.ModelSizeBytes = int64(jsonNum(mm, "size"))
				meta.ModelCtxTrain = int(jsonNum(mm, "n_ctx_train"))
				meta.ModelQuant, _ = mm["ftype"].(string)
			}
		}
	}
	// llama.cpp only (vLLM has no /props). Recent builds put the weights path at
	// the top level, older ones under default_generation_settings.model.
	// Answering /props at all is the llama.cpp signal — vLLM and every OpenAI-
	// shaped provider 404 it. That makes the engine a MEASUREMENT like everything
	// else in this struct rather than something a worker declares about itself.
	meta.Engine = engineOpenAI
	if raw, err := r.backendGET(b, "/props"); err == nil {
		meta.Engine = engineLlamaCPP
		if p, _ := raw["model_path"].(string); p != "" {
			meta.ModelPath = p
		} else if dg, ok := raw["default_generation_settings"].(map[string]any); ok {
			meta.ModelPath, _ = dg["model"].(string)
		}
	}
	switch {
	case id != "":
	case b.Model != "" && b.Model != "default":
		id = b.Model
	default:
		id = b.ID
	}
	return id, meta
}

// modelEntryFor picks the entry of an OpenAI-shaped /v1/models response that
// describes what this backend serves.
//
// It used to be data[0], unconditionally. That is right for the single-model
// worker the router grew up with, and arbitrary for a catalogue endpoint serving
// hundreds — and the id it yields is stamped onto every forwarded request (see
// patchForwardedBody), so an arbitrary pick means every client request is
// rewritten to a model nobody asked for.
//
// So: the declared model wins when the endpoint's list actually contains it,
// because that is the row the registration is about. Otherwise only a
// single-entry list is unambiguous. A catalogue with no declared match yields
// nothing, which leaves the client's own model name to travel through untouched
// — the correct answer for an endpoint that serves whatever it is asked for.
func modelEntryFor(raw map[string]any, declared string) (map[string]any, bool) {
	data, ok := raw["data"].([]any)
	if !ok || len(data) == 0 {
		return nil, false
	}
	if declared != "" {
		for _, item := range data {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if id, _ := m["id"].(string); id == declared {
				return m, true
			}
		}
	}
	if len(data) != 1 {
		return nil, false
	}
	m, ok := data[0].(map[string]any)
	return m, ok
}

// jsonNum reads a numeric field from a decoded JSON object (every JSON number
// decodes to float64), returning 0 when it is absent or another type.
func jsonNum(m map[string]any, key string) float64 {
	v, _ := m[key].(float64)
	return v
}

// queryContext discovers the worker's context window (vLLM max_model_len /
// llama.cpp n_ctx), falling back to any declared value, in 1024-token units.
func (r *Router) queryContext(b *Backend) int {
	if k, ok := r.queryContextMeasured(b); ok {
		return k
	}
	return b.ContextK
}

// queryContextMeasured is queryContext without the fallback, reporting whether the
// worker actually answered.
//
// The fallback is right for a cold profile — no measurement, so keep whatever the
// registry holds — but wrong for refreshing a CACHED profile, where the two values
// come from different places and silently disagreeing is the whole bug being fixed.
// A failed probe there must leave the cached number alone, not overwrite it with the
// registry's, so the caller needs to tell "measured 256k" from "could not measure".
func (r *Router) queryContextMeasured(b *Backend) (int, bool) {
	if raw, err := r.backendGET(b, "/v1/models"); err == nil {
		// Same entry selection as queryModelInfo: on a catalogue endpoint,
		// data[0].max_model_len is some other model's context window.
		if m, ok := modelEntryFor(raw, b.Model); ok {
			if v := jsonNum(m, "max_model_len"); v > 0 {
				return int(v) / 1024, true
			}
		}
	}
	if raw, err := r.backendGET(b, "/props"); err == nil {
		if v, _ := raw["n_ctx"].(float64); v > 0 {
			return int(v) / 1024, true
		}
		if dg, ok := raw["default_generation_settings"].(map[string]any); ok {
			if v, _ := dg["n_ctx"].(float64); v > 0 {
				return int(v) / 1024, true
			}
		}
	}
	return 0, false
}

// querySlots reads the worker's OWN concurrent-request ceiling when it publishes one.
// llama.cpp's /props reports total_slots, which is exactly --parallel: a hard limit on
// how many sequences it will decode at once, regardless of what a throughput ramp
// appears to show. Used to cap capacityProbe, never to raise it — the ramp can still
// find a LOWER practical knee than the configured slot count.
//
// Returns 0 when the worker publishes nothing usable (vLLM does not expose
// max_num_seqs on any endpoint), which leaves the ramp's verdict untouched.
func (r *Router) querySlots(b *Backend) int {
	raw, err := r.backendGET(b, "/props")
	if err != nil {
		return 0
	}
	if v, _ := raw["total_slots"].(float64); v > 0 {
		return int(v)
	}
	return 0
}

// backendGET does an authenticated GET against a worker and decodes JSON.
func (r *Router) backendGET(b *Backend, path string) (map[string]any, error) {
	req, err := http.NewRequest("GET", strings.TrimRight(b.URL, "/")+path, nil)
	if err != nil {
		return nil, err
	}
	if b.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.APIKey)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: HTTP %d", path, resp.StatusCode)
	}
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// applyProfileIfGen copies measured values onto the live backend — a beacon's
// declared values are only ever a seed; the measured profile wins. It applies
// only when the backend still exists at the registration generation the profile
// was measured against: results from a stale generation (the worker re-registered
// with new content, or was deleted, mid-measurement) are dropped, and the
// caller must not persist them either. gen 0 skips the check (tests, callers
// that hold no generation).
//
// A MANUAL row is the exception, and the central invariant of P2: it is
// operator-owned, so a probe fills in what the operator left blank and never
// overwrites what they entered. See providers.go for why the two are not the
// same kind of claim.
func (r *Registry) applyProfileIfGen(id string, gen int64, p *WorkerProfile) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil || (gen != 0 && b.profileGen != gen) {
		return false
	}
	// Zero for every field on a beacon row, so the guards below collapse to the
	// behaviour the whole deployed fleet already has.
	declared := operatorDeclared(b)
	if p.Model != "" && declared.Model == "" {
		b.Model = p.Model
	}
	if p.Quality > 0 && declared.Quality == 0 {
		b.Quality = p.Quality
	}
	if p.ContextProbe != nil {
		b.ContextProbe = p.ContextProbe
	}
	if p.ContextK > 0 && declared.ContextK == 0 {
		b.ContextK = p.ContextK
	}
	if p.MaxConcurrency > 0 && declared.MaxConcurrency == 0 {
		b.MaxConcurrency = p.MaxConcurrency
	}
	if p.BaselineTPS > 0 && declared.BaselineTPS == 0 {
		b.BaselineTPS = p.BaselineTPS
	}
	// Seed the prefill EWMA from the probe, but never overwrite a live one — same rule
	// as ObservedTPS. Without this seed a worker that mostly serves thinking traffic
	// never gets a prefill rate at all (observe() skips those samples), and every long
	// prompt on it is priced from the next-best source prefillSeconds can find: the
	// context ladder's per-rung rate, or failing that the fleet-wide
	// fallbackPrefillTPS. Both scale with the prompt, so the cost of missing this
	// seed is a less accurate rate rather than the flat, length-blind TTFT average it
	// used to be — but this is still the only figure measured on THIS worker at
	// length, and it is the one to prefer.
	if p.PrefillTPS > 0 && b.ObservedPrefillTPS == 0 {
		b.ObservedPrefillTPS = p.PrefillTPS
	}
	// Features are exempt from the operator-owned rule, and deliberately so:
	// capabilities are the one thing the router settles by sending a request and
	// reading the answer, and profileQuick already carries the declared semantic
	// tags it cannot probe (uncensored, vision) across into the measured set. So
	// nothing an operator declares is lost here — it is merged, not overwritten.
	if len(p.Features) > 0 {
		b.Features = append([]string(nil), p.Features...)
	}
	b.Thinking = p.Thinking
	b.ThinkingDialect = p.ThinkingDialect
	b.QualityDetail = p.QualityDetail
	b.CategorySummary = p.CategorySummary
	// No operator-declared analog exists for the no-think score: it is purely
	// measured, so the profile is always authoritative. Zero (old profile, or a
	// failed no-think pass) means qualityFor falls back to Quality.
	b.QualityNoThink = p.QualityNoThink
	b.QualityNoThinkDetail = p.QualityNoThinkDetail
	b.Failed = append([]string(nil), p.Failed...)
	// Measured capacity becomes the slot cap — the bound that makes requests
	// queue at the router and spill across the fleet instead of piling onto a
	// saturated worker. Only a FULL profile carries a measured value
	// (BenchVersion is set); profileQuick's MaxConcurrency=1 is a provisional
	// placeholder that must not throttle a fresh worker to serial dispatch.
	//
	// This is NO LONGER the first time a cold run's capacity reaches the slot
	// channel, and the gate below should not be read as though it were. A live
	// cold profile publishes the ramp's answer the moment it settles, through
	// publishCapacity, which syncs the slots there and then; by the time the
	// profile arrives here the cap is usually already correct and syncSlotsLocked
	// no-ops on it. What still comes through this path is every case with no
	// mid-run publish behind it: a WARM restart applying a cached profile, a relay
	// import, and the tests.
	//
	// Sync the EFFECTIVE cap rather than the profile's own: on a manual row the
	// guard above kept the operator's number, and the slot channel has to hold
	// that one or the declared ceiling would be advisory only.
	if p.BenchVersion > 0 {
		r.syncSlotsLocked(id, b.MaxConcurrency)
	}
	// The load curve rides with the capacity it was measured beside, so a WARM
	// restart from a cached profile prices load the way the run that measured it
	// did — otherwise the exponent would exist only for the minutes between a cold
	// ramp and the next restart. Unconditional: a profile with no curve (cached
	// before this existed, imported from a relay, provisional) publishes nothing
	// and the worker keeps the neutral default. See setConcurrencyAlpha.
	setConcurrencyAlpha(b, p.CapacityCurve)
	return true
}

// SaveWorkerProfile/LoadWorkerProfile/DeleteWorkerProfile persist measured
// profiles so a warm restart (same id + model) skips re-profiling.
func (s *LogStore) SaveWorkerProfile(ctx context.Context, id string, p *WorkerProfile) error {
	data, err := json.Marshal(p)
	if err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, `INSERT INTO worker_profiles (id, model, updated_at, profile_json)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET model=excluded.model, updated_at=excluded.updated_at, profile_json=excluded.profile_json`,
		id, p.Model, time.Now().UTC().Format(time.RFC3339Nano), string(data))
	return err
}

func (s *LogStore) LoadWorkerProfile(ctx context.Context, id, model string) (*WorkerProfile, bool) {
	var raw string
	if err := s.db.QueryRowContext(ctx, `SELECT profile_json FROM worker_profiles WHERE id=? AND model=?`, id, model).Scan(&raw); err != nil {
		return nil, false
	}
	var p WorkerProfile
	if json.Unmarshal([]byte(raw), &p) != nil {
		return nil, false
	}
	return &p, true
}

// LoadWorkerProfileByID loads the most recent persisted profile for a worker id
// regardless of model (the table is keyed by id) — for the benchmark-inspection
// endpoint, which only has the id.
func (s *LogStore) LoadWorkerProfileByID(ctx context.Context, id string) (*WorkerProfile, bool) {
	var raw string
	if err := s.db.QueryRowContext(ctx, `SELECT profile_json FROM worker_profiles WHERE id=?`, id).Scan(&raw); err != nil {
		return nil, false
	}
	var p WorkerProfile
	if json.Unmarshal([]byte(raw), &p) != nil {
		return nil, false
	}
	return &p, true
}

func (s *LogStore) DeleteWorkerProfile(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM worker_profiles WHERE id=?`, id)
	return err
}
