package router

// Tests for the latency model — what a request is predicted to COST, and the
// measurements that feed it. Four defects, each with the failure that motivated
// it recorded beside the assertion:
//
//	prompt length did not move the estimate at all on a worker with no measured
//	prefill rate (six of seven live workers);
//	the capacity ramp's answer was withheld from routing for the whole quality
//	benchmark;
//	the throughput curve the ramp measured was discarded;
//	an endpoint that declared no concurrency limit was modelled as a 4-slot box.

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// ── Defect 1: prefill has to scale with the prompt, measured or not ──────────

// The estimate has to move when the PROMPT moves, on a worker that has no
// prefill rate of its own — which was six of the seven live workers.
//
// Measured 2026-08-27 through /v1/route-preview: a 45-character prompt and a
// 61,000-character one (~15k tokens) produced BYTE-IDENTICAL expected_seconds on
// every worker (16.553 / 52.051 / 144.455 / 205.287), the single worker carrying
// a measured rate excepted. prefillSeconds fell back to the flat TTFT average,
// which does not read promptTokens at all, so the dominant term of a
// long-context request was not approximate — it was absent.
func TestUnmeasuredPrefillStillScalesWithThePrompt(t *testing.T) {
	// The shape of those live rows: a TTFT average accumulated from short
	// traffic, and no prefill rate, because ObservedPrefillTPS is only ever set
	// by a completed full profile or by a streamed request whose prompt cleared
	// minPrefillTokens.
	b := &Backend{
		BackendRegistration: BackendRegistration{ID: "unmeasured", MaxConcurrency: 4},
		ObservedTPS:         50, ObservedTTFTMillis: 978,
	}
	if b.ObservedPrefillTPS != 0 {
		t.Fatal("the premise of this test is a worker with NO measured prefill rate")
	}

	short, long := prefillSeconds(b, 5), prefillSeconds(b, 15_000)
	if long <= short*10 {
		t.Fatalf("15k tokens priced at %.3fs against %.3fs for five: prompt length does not move the estimate",
			long, short)
	}
	// 15k tokens at the fleet constant, on top of the flat average as an overhead
	// floor — see prefillSeconds for why only this branch adds it.
	if want := 0.978 + 15_000/fallbackPrefillTPS; long < want*0.99 || long > want*1.01 {
		t.Errorf("15k prompt priced at %.1fs, want ~%.1fs", long, want)
	}
	// A five-token prompt must still cost about what it always did, or this
	// bought long-context accuracy by mispricing every chat turn in the fleet.
	if short < 0.9*0.978 || short > 1.1*0.978 {
		t.Errorf("a five-token prompt now costs %.3fs, want ~0.978s (the flat average, unchanged)", short)
	}

	// And it has to survive into the number routing actually ranks on.
	sj := jobCost{promptTokens: 5, outputTokens: 256}
	lj := jobCost{promptTokens: 15_000, outputTokens: 256}
	if a, z := expectedLatency(b, sj), expectedLatency(b, lj); z <= a*10 {
		t.Fatalf("expectedLatency ignores the prompt: %.3fs for five tokens, %.3fs for 15k", a, z)
	}
}

// A worker with no prefill EWMA has usually still climbed the context ladder,
// which timed a needle in haystacks of KNOWN length — a measured prefill curve
// at exactly the sizes the flat average was worst at, already persisted with the
// profile. Preferring it to the fleet constant is the difference between pricing
// a 60k prompt off this worker's own 64k rung and off a number chosen for
// everybody.
func TestContextLadderSuppliesAProvisionalPrefillRate(t *testing.T) {
	b := &Backend{
		BackendRegistration: BackendRegistration{ID: "laddered", MaxConcurrency: 1},
		ObservedTPS:         50, ObservedTTFTMillis: 400,
		ContextProbe: &ContextProbe{
			UsableTokens: 65536, AdvertisedTokens: 131072,
			Levels: []ContextProbeLevel{
				{Tokens: 4096, Passed: 3, Total: 3, PrefillTPS: 2000},
				{Tokens: 65536, Passed: 3, Total: 3, PrefillTPS: 400},
			},
		},
	}
	// The rung NEAREST the request, not the first and not the best: prefill rate
	// falls with length, so the 4k reading would over-rate a 60k prompt 5x.
	if got, want := prefillSeconds(b, 60_000), 60_000/400.0; got < want*0.99 || got > want*1.01 {
		t.Errorf("60k prompt priced at %.1fs, want ~%.1fs (the 64k rung's own 400 tok/s)", got, want)
	}
	if got, want := prefillSeconds(b, 4_000), 4_000/2000.0; got < want*0.99 || got > want*1.01 {
		t.Errorf("4k prompt priced at %.1fs, want ~%.1fs (the 4k rung's own 2000 tok/s)", got, want)
	}
	// A measured rung already contains the fixed per-request overhead (it is
	// tokens ÷ TTFT), so liveTTFT must not be charged on top of it as well.
	if got := prefillSeconds(b, 60_000); got > 60_000/400.0+0.01 {
		t.Errorf("the ladder's rate was charged the TTFT overhead twice: %.3fs", got)
	}
	// An errored rung measured nothing, and must not be read as a rate.
	b.ContextProbe.Levels = []ContextProbeLevel{{Tokens: 4096, Errored: true, PrefillTPS: 9999}}
	if got, want := prefillSeconds(b, 4_000), 0.4+4_000/fallbackPrefillTPS; got < want*0.99 || got > want*1.01 {
		t.Errorf("an errored rung was believed: %.1fs, want the fleet fallback ~%.1fs", got, want)
	}
}

// ── Defect 3: the queue term, and the curve that shapes it ──────────────────

// The queue term was a step function: `over = occupancy + 1 - slots` is ≤ 0
// below saturation, so on an eight-slot worker SEVEN busy slots predicted
// identically to idle, and the eighth doubled the estimate in one jump. Real
// continuous batching does neither — it degrades roughly monotonically with
// batch size — and the capacity ramp had already measured that curve and thrown
// it away.
func TestLoadPenaltyDegradesContinuously(t *testing.T) {
	// A worker measured at eight slots whose aggregate throughput scales
	// sub-linearly, the shape every batching engine has: 900 tok/s alone, 3400
	// across eight, so each of the eight is worth under half of the one.
	const id = "batched"
	alpha := concurrencyAlpha([]CapacityLevel{{N: 1, TPS: 900}, {N: 2, TPS: 1500}, {N: 4, TPS: 2300}, {N: 8, TPS: 3400}})
	if alpha <= 0 || alpha >= 1 {
		t.Fatalf("fixture wrong: sub-linear scaling should fit 0 < alpha < 1, got %v", alpha)
	}

	at := func(active int) *Backend {
		return &Backend{
			BackendRegistration: BackendRegistration{ID: id, MaxConcurrency: 8},
			ObservedTPS:         50, ObservedPrefillTPS: 4000, ActiveRequests: active,
			ConcurrencyAlpha: alpha,
		}
	}
	job := jobCost{promptTokens: 1000, outputTokens: 256}

	idle, seven := expectedLatency(at(0), job), expectedLatency(at(7), job)
	if seven <= idle {
		t.Fatalf("seven of eight slots busy priced identically to idle (%.3fs vs %.3fs)", seven, idle)
	}

	// Monotonic across the boundary, with no discontinuity at it: one request
	// past the last slot waits for the FIRST of eight generations to finish, not
	// for a whole one, so it cannot cost double.
	prev := 0.0
	for active := 0; active <= 12; active++ {
		got := expectedLatency(at(active), job)
		if got <= prev {
			t.Fatalf("estimate did not grow at %d in flight: %.4fs after %.4fs", active, got, prev)
		}
		if prev > 0 && got > prev*1.5 {
			t.Errorf("discontinuity at %d in flight: %.4fs jumped from %.4fs (%.2fx)", active, got, prev, got/prev)
		}
		prev = got
	}

	// An UNMEASURED worker gets no invented slowdown, only the smoothed queue: a
	// curve nobody measured is not evidence, and every profile written before the
	// curve existed would otherwise acquire one.
	plain := &Backend{
		BackendRegistration: BackendRegistration{ID: "no-curve", MaxConcurrency: 8},
		ObservedTPS:         50, ObservedPrefillTPS: 4000,
	}
	if got := loadPenalty(plain); got != 1 {
		t.Errorf("a worker with no measured curve was charged %.3fx while idle", got)
	}
	plain.ActiveRequests = 8
	if got := loadPenalty(plain); got >= 2 {
		t.Errorf("crossing the last of eight slots still doubles the estimate: %.3fx", got)
	}

	// A one-slot worker is the case the old ceil() was right for, and the
	// commonest row in the fleet: the two forms have to agree exactly there.
	serial := &Backend{BackendRegistration: BackendRegistration{ID: "serial", MaxConcurrency: 1}, ActiveRequests: 2}
	if got := loadPenalty(serial); got != 3 {
		t.Errorf("a 1-slot worker with two in flight prices at %.3fx, want 3x (unchanged)", got)
	}
}

// concurrencyAlpha has to read a perfectly-scaling worker as free and a
// serialising one as fully charged, and refuse to invent either from noise.
func TestConcurrencyAlphaFit(t *testing.T) {
	cases := []struct {
		name  string
		curve []CapacityLevel
		want  float64
	}{
		{"perfect scaling", []CapacityLevel{{N: 1, TPS: 100}, {N: 2, TPS: 200}, {N: 4, TPS: 400}}, 1},
		{"serialises", []CapacityLevel{{N: 1, TPS: 100}, {N: 2, TPS: 100}, {N: 4, TPS: 100}}, 0},
		{"square root", []CapacityLevel{{N: 1, TPS: 100}, {N: 4, TPS: 200}, {N: 16, TPS: 400}}, 0.5},
		{"super-linear noise is clamped", []CapacityLevel{{N: 1, TPS: 100}, {N: 2, TPS: 400}}, 1},
		{"throughput fell", []CapacityLevel{{N: 1, TPS: 100}, {N: 2, TPS: 50}}, 0},
	}
	for _, c := range cases {
		if got := concurrencyAlpha(c.curve); math.Abs(got-c.want) > 0.01 {
			t.Errorf("%s: alpha = %.3f, want %.3f", c.name, got, c.want)
		}
	}
	// Nothing to fit reads as "no penalty", never as a fitted zero.
	for _, curve := range [][]CapacityLevel{nil, {{N: 1, TPS: 100}}, {{N: 2, TPS: 100}}, {{N: 1}, {N: 2, TPS: 100}}} {
		if got := concurrencyAlpha(curve); got != 1 {
			t.Errorf("curve %v fitted alpha %.3f, want the neutral 1", curve, got)
		}
	}
}

// ── Defect 4: unknown capacity is not the same as unbounded ─────────────────

// A hosted endpoint that declared no concurrency limit was modelled as a 4-slot
// worker, so four requests THIS router happened to have in flight doubled its
// estimate — a number that says nothing whatever about a provider fronting
// thousands of them. The effect was to push traffic off a genuinely fast paid
// endpoint under mild local load.
func TestUncappedProviderRowIsNotChargedAQueue(t *testing.T) {
	job := jobCost{promptTokens: 500, outputTokens: 256}
	provider := func(active int) *Backend {
		return &Backend{
			BackendRegistration: BackendRegistration{
				ID: "openai", Source: sourceManual, Provider: "openai", MaxConcurrency: 0,
			},
			ObservedTPS: 120, ObservedPrefillTPS: 8000, ActiveRequests: active,
		}
	}
	idle, busy := expectedLatency(provider(0), job), expectedLatency(provider(8), job)
	if busy != idle {
		t.Errorf("a provider that declared no limit was charged %.2fx for the router's own eight requests (%.3fs vs %.3fs)",
			busy/idle, busy, idle)
	}

	// "Capacity unknown" is a different row and keeps the nominal assumption: a
	// beacon the ramp has not reached yet does have a ceiling, we just do not
	// know it, and pricing some queue beats pricing none.
	beacon := func(active int) *Backend {
		return &Backend{
			BackendRegistration: BackendRegistration{ID: "beacon", Source: sourceBeacon, Provider: providerLocal},
			ObservedTPS:         120, ObservedPrefillTPS: 8000, ActiveRequests: active,
		}
	}
	if a, z := expectedLatency(beacon(0), job), expectedLatency(beacon(8), job); z <= a {
		t.Errorf("an unmeasured beacon with eight in flight priced identically to idle (%.3fs vs %.3fs)", z, a)
	}

	// And a LOCAL vLLM an operator happened to type in by hand is still one box
	// with one real ceiling — "manual" says who wrote the row down, not what is
	// behind it.
	local := func(active int) *Backend {
		return &Backend{
			BackendRegistration: BackendRegistration{ID: "hand-entered", Source: sourceManual, Provider: providerLocal},
			ObservedTPS:         120, ObservedPrefillTPS: 8000, ActiveRequests: active,
		}
	}
	if a, z := expectedLatency(local(0), job), expectedLatency(local(8), job); z <= a {
		t.Errorf("a hand-entered LOCAL worker was treated as unbounded (%.3fs idle vs %.3fs with eight in flight)", z, a)
	}
}

// ── Defect 2: capacity has to be published when it is measured ──────────────

// The capacity ramp's answer used to sit in a local variable until after the
// quality benchmark — 25+ minutes on a typical worker, and this repo's own
// comment records profiles running to five hours. For all of it the router
// priced the worker off profileQuick's provisional MaxConcurrency=1, so
// expectedLatency's queue term, isFull and the `ask -l` slots column all read
// "serial" and live traffic was spilled off a worker that the SAME profile was
// simultaneously driving at 4-way concurrency.
//
// Run against a fake worker that holds its benchmark answers until the test has
// looked, exactly as the progress tracker's test does: the assertion is about
// what the registry says WHILE the benchmark is in flight, which no amount of
// checking the finished profile can establish.
func TestCapacityIsPublishedBeforeTheBenchmarkFinishes(t *testing.T) {
	withCapacityProbeRetryDelay(t, time.Millisecond)
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	benchmarkQuestions = []benchmarkQ{{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Match: "numeric"}}

	reg := newTestRegistry()
	release := make(chan struct{})
	observed := make(chan int, 1)
	var once sync.Once
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.Method == http.MethodGet {
			http.NotFound(w, req) // no catalogue and no /props: context and slots stay unmeasured
			return
		}
		body, _ := io.ReadAll(req.Body)
		if bytes.Contains(body, []byte(`"tools"`)) || bytes.Contains(body, []byte(`"response_format"`)) {
			http.Error(w, `{"error":{"message":"unsupported"}}`, http.StatusBadRequest)
			return
		}
		if bytes.Contains(body, []byte("What is 2+2?")) {
			once.Do(func() { observed <- reg.get("w").MaxConcurrency })
			<-release
		}
		if bytes.Contains(body, []byte(`"stream":true`)) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok ok ok ok\"}}]}\n\n")
			fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":4}}\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		fmt.Fprint(w, `{"choices":[{"message":{"content":"4"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1024,"completion_tokens":8}}`)
	}))
	defer srv.Close()

	reg.upsert(BackendRegistration{ID: "w", URL: srv.URL, Model: "m", TTLSeconds: 3600})
	// The state certifyBackend leaves a fresh worker in before it hands the
	// background profile the guard: routable, and provisionally serial.
	reg.applyProfileIfGen("w", 0, &WorkerProfile{Model: "m", MaxConcurrency: 1})
	if got := reg.get("w").MaxConcurrency; got != 1 {
		t.Fatalf("setup: provisional concurrency = %d, want the placeholder 1", got)
	}
	r := &Router{
		cfg:         &Config{CapacityProbeMax: 4},
		registry:    reg,
		client:      &http.Client{Timeout: 10 * time.Second},
		benchClient: &http.Client{},
	}

	type result struct {
		prof *WorkerProfile
		err  error
	}
	done := make(chan result, 1)
	go func() {
		prof, err := r.profileBackend(reg.get("w"), "m")
		done <- result{prof, err}
	}()

	var mid int
	select {
	case mid = <-observed:
	case <-time.After(30 * time.Second):
		close(release)
		t.Fatal("the quality benchmark never reached the worker")
	}
	if mid <= 1 {
		t.Errorf("the worker was still priced at %d concurrent DURING its own benchmark — the measured capacity was withheld until the run finished",
			mid)
	}
	// The slot channel is the half that actually gates dispatch: a MaxConcurrency
	// nothing enforces would leave the ranker and the acquire step disagreeing.
	reg.mu.RLock()
	slotCap := reg.slotCap["w"]
	reg.mu.RUnlock()
	if slotCap != mid {
		t.Errorf("slot channel holds %d token(s) against a published capacity of %d", slotCap, mid)
	}

	close(release)
	res := <-done
	if res.err != nil {
		t.Fatalf("profileBackend: %v", res.err)
	}
	// The early publication and the atomic commit at the end must agree, or the
	// second is a conflict rather than the no-op it is meant to be.
	if res.prof.MaxConcurrency != mid {
		t.Errorf("profile committed %d concurrent after publishing %d mid-run",
			res.prof.MaxConcurrency, mid)
	}
	// And the curve the ramp climbed has to survive onto the profile, or it is
	// measured and discarded all over again.
	if len(res.prof.CapacityCurve) < 2 {
		t.Fatalf("the ramp's throughput curve was not kept: %v", res.prof.CapacityCurve)
	}
	if res.prof.CapacityCurve[0].N != 1 || res.prof.CapacityCurve[0].TPS <= 0 {
		t.Errorf("the curve has no n=1 baseline to scale against: %v", res.prof.CapacityCurve)
	}
	if msg := res.prof.Checks["capacity"].Message; !bytes.Contains([]byte(msg), []byte("scaling exponent")) {
		t.Errorf("the capacity check does not report the curve behind the number: %q", msg)
	}
}

// A profile applied on a WARM restart has to re-publish its curve, or the
// exponent exists only for the minutes between a cold ramp and the next restart
// — the same silent-staleness shape as the prefill probe that shipped and then
// sat idle on five of seven cached workers.
func TestWarmRestartRepublishesTheCapacityCurve(t *testing.T) {
	reg := newTestRegistry()
	register(t, reg, "warm", 0)
	reg.applyProfileIfGen("warm", 0, &WorkerProfile{
		Model: "m", BenchVersion: benchmarkVersion, MaxConcurrency: 4,
		CapacityCurve: []CapacityLevel{{N: 1, TPS: 100}, {N: 2, TPS: 100}, {N: 4, TPS: 100}},
	})
	// A flat curve is a MEASURED zero — a worker that gets nothing from batching.
	// It is stored as minMeasuredAlpha so it stays distinguishable from the
	// unmeasured default of 1, which is its exact opposite.
	if got := concurrencyAlphaFor(reg.get("warm")); got != minMeasuredAlpha {
		t.Errorf("a cached profile's serialising curve was not re-published: alpha = %.3f, want %.3f", got, minMeasuredAlpha)
	}
	// A profile with no curve — cached before this existed, or imported from a
	// relay — must leave the worker on the neutral default rather than zero.
	reg2 := newTestRegistry()
	register(t, reg2, "curveless", 0)
	reg2.applyProfileIfGen("curveless", 0, &WorkerProfile{Model: "m", BenchVersion: benchmarkVersion, MaxConcurrency: 4})
	if got := concurrencyAlphaFor(reg2.get("curveless")); got != 1 {
		t.Errorf("a curveless profile published alpha %.3f, want the neutral 1", got)
	}
}

// publishCapacity is the early half of the fix, and it inherits
// applyProfileIfGen's two guards for applyProfileIfGen's reasons.
func TestPublishCapacityRespectsOperatorAndGeneration(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(manualReg(t, BackendRegistration{
		ID: "provider", URL: "http://provider", Model: "m", MaxConcurrency: 4, TTLSeconds: 3600,
	}))
	gen := reg.get("provider").profileGen

	// A probe never overwrites what an operator entered.
	if !reg.publishCapacity("provider", gen, 1, nil) {
		t.Fatal("publishCapacity refused a live row at its own generation")
	}
	if got := reg.get("provider").MaxConcurrency; got != 4 {
		t.Errorf("the ramp overwrote the operator's declared ceiling: %d, want 4", got)
	}
	// The slot channel tracks the EFFECTIVE cap, or the declared ceiling is
	// advisory only.
	reg.mu.RLock()
	slotCap := reg.slotCap["provider"]
	reg.mu.RUnlock()
	if slotCap != 4 {
		t.Errorf("slot channel holds %d token(s), want the declared 4", slotCap)
	}

	// A result from a stale registration generation describes a worker that no
	// longer exists at this id.
	if reg.publishCapacity("provider", gen+1, 8, nil) {
		t.Error("a stale generation's capacity was applied")
	}
	if reg.publishCapacity("gone", 0, 8, nil) {
		t.Error("capacity was applied to a row that is not registered")
	}

	// On a beacon the measurement is the answer, and re-publishing the same
	// number is a no-op rather than a rebuilt slot channel.
	reg.upsert(BackendRegistration{ID: "beacon", URL: "http://beacon", Model: "m", TTLSeconds: 3600})
	reg.publishCapacity("beacon", 0, 6, nil)
	if got := reg.get("beacon").MaxConcurrency; got != 6 {
		t.Errorf("measured capacity = %d, want 6", got)
	}
	reg.mu.RLock()
	first := reg.slots["beacon"]
	reg.mu.RUnlock()
	reg.publishCapacity("beacon", 0, 6, nil)
	reg.mu.RLock()
	second := reg.slots["beacon"]
	reg.mu.RUnlock()
	if first != second {
		t.Error("re-publishing an unchanged capacity rebuilt the slot channel, dropping every in-flight token")
	}
}
