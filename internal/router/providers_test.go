package router

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// manualReg builds an operator-entered row. Only the fields a test cares about
// are set; everything else defaults through normalizeRegistration, which is the
// path the admin API uses too.
func manualReg(t *testing.T, reg BackendRegistration) BackendRegistration {
	t.Helper()
	reg.Source = sourceManual
	if reg.URL == "" {
		reg.URL = "http://" + reg.ID
	}
	if err := normalizeRegistration(&reg); err != nil {
		t.Fatalf("normalizeRegistration: %v", err)
	}
	return reg
}

// TestRegistrationDefaultsAreLocalBeaconFree is the compatibility contract in
// one assertion: the payload every deployed worker sends carries no provider,
// no source and no price, and it has to keep meaning a free local worker that
// owns none of its own declared values.
func TestRegistrationDefaultsAreLocalBeaconFree(t *testing.T) {
	// Byte-for-byte what a beacon sidecar posts today.
	body := `{"id":"llm-a750","url":"http://a750:8080","model":"gemma4","features":["chat"],"api_key":"k"}`
	var reg BackendRegistration
	if err := json.Unmarshal([]byte(body), &reg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if err := normalizeRegistration(&reg); err != nil {
		t.Fatalf("normalizeRegistration: %v", err)
	}
	if reg.Provider != providerLocal {
		t.Errorf("provider = %q, want %q", reg.Provider, providerLocal)
	}
	if reg.Source != sourceBeacon {
		t.Errorf("source = %q, want %q", reg.Source, sourceBeacon)
	}
	// Zero cost is not cosmetic: it is what P4's "prefer the free ones" rule
	// reads, so a default that drifted would start spending money on local
	// traffic.
	if reg.InputPricePerMtok != 0 || reg.OutputPricePerMtok != 0 {
		t.Errorf("a worker that declared no price must cost nothing, got %v/%v",
			reg.InputPricePerMtok, reg.OutputPricePerMtok)
	}
	// The frozen defaults have to survive the new fields landing next to them.
	if reg.TTLSeconds != 90 {
		t.Errorf("ttl = %d, want the frozen default 90", reg.TTLSeconds)
	}
	if reg.HealthPath != "/health" {
		t.Errorf("health path = %q, want the frozen default /health", reg.HealthPath)
	}
}

func TestNormalizeProviderFields(t *testing.T) {
	cases := []struct {
		name             string
		in               BackendRegistration
		provider, source string
		inPrice          float64
	}{
		{"empty defaults", BackendRegistration{}, providerLocal, sourceBeacon, 0},
		{"case folded", BackendRegistration{Provider: " OpenAI ", Source: "MANUAL"}, "openai", sourceManual, 0},
		{"typo is not manual", BackendRegistration{Source: "manul"}, providerLocal, sourceBeacon, 0},
		{"negative price clamped", BackendRegistration{InputPricePerMtok: -3}, providerLocal, sourceBeacon, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reg := tc.in
			normalizeProviderFields(&reg)
			if reg.Provider != tc.provider || reg.Source != tc.source || reg.InputPricePerMtok != tc.inPrice {
				t.Errorf("got provider=%q source=%q in=%v, want %q/%q/%v",
					reg.Provider, reg.Source, reg.InputPricePerMtok, tc.provider, tc.source, tc.inPrice)
			}
		})
	}
}

// TestManualRowIsOperatorOwned is the central invariant of P2. A probe may fill
// in what the operator left blank and must never overwrite what they entered;
// the same profile applied to a beacon row must overwrite everything, because
// that is what the whole deployed fleet already relies on.
func TestManualRowIsOperatorOwned(t *testing.T) {
	// One profile, deliberately disagreeing with every declared value.
	measured := func() *WorkerProfile {
		return &WorkerProfile{
			Model: "measured-model", Quality: 91, ContextK: 256,
			MaxConcurrency: 1, BaselineTPS: 5, BenchVersion: benchmarkVersion,
		}
	}

	t.Run("manual keeps what the operator declared", func(t *testing.T) {
		reg := newTestRegistry()
		b, _ := reg.upsert(manualReg(t, BackendRegistration{
			ID: "openai-gpt-4o", Model: "gpt-4o", Quality: 80,
			ContextK: 128, MaxConcurrency: 8, BaselineTPS: 120, TTLSeconds: 3600,
		}))
		if !reg.applyProfileIfGen("openai-gpt-4o", b.profileGen, measured()) {
			t.Fatal("profile should have applied")
		}
		got := reg.get("openai-gpt-4o")
		for _, c := range []struct {
			field string
			got   any
			want  any
		}{
			{"model", got.Model, "gpt-4o"},
			{"quality", got.Quality, 80},
			{"context_k", got.ContextK, 128},
			{"max_concurrency", got.MaxConcurrency, 8},
			{"baseline_tps", got.BaselineTPS, 120.0},
		} {
			if c.got != c.want {
				t.Errorf("%s = %v, want the declared %v", c.field, c.got, c.want)
			}
		}
		// The declared ceiling has to reach the slot channel, or it is advisory
		// only and the router still queues eight requests onto one slot.
		if reg.slotCap["openai-gpt-4o"] != 8 {
			t.Errorf("slot cap = %d, want the declared 8", reg.slotCap["openai-gpt-4o"])
		}
	})

	t.Run("manual still accepts what the operator left blank", func(t *testing.T) {
		reg := newTestRegistry()
		b, _ := reg.upsert(manualReg(t, BackendRegistration{
			ID: "openrouter-x", Model: "vendor/x", MaxConcurrency: 4, TTLSeconds: 3600,
		}))
		reg.applyProfileIfGen("openrouter-x", b.profileGen, measured())
		got := reg.get("openrouter-x")
		if got.Quality != 91 || got.ContextK != 256 || got.BaselineTPS != 5 {
			t.Errorf("probe must refine the blanks: q=%d ctx=%d tps=%v", got.Quality, got.ContextK, got.BaselineTPS)
		}
		if got.MaxConcurrency != 4 {
			t.Errorf("max_concurrency = %d, want the declared 4", got.MaxConcurrency)
		}
	})

	t.Run("beacon rows are unchanged", func(t *testing.T) {
		reg := newTestRegistry()
		b, _ := reg.upsert(BackendRegistration{
			ID: "llm-a750", URL: "http://a750", Model: "gemma4", Quality: 80,
			ContextK: 128, MaxConcurrency: 8, BaselineTPS: 120, TTLSeconds: 3600,
		})
		reg.applyProfileIfGen("llm-a750", b.profileGen, measured())
		got := reg.get("llm-a750")
		if got.Model != "measured-model" || got.Quality != 91 || got.ContextK != 256 ||
			got.MaxConcurrency != 1 || got.BaselineTPS != 5 {
			t.Fatalf("a beacon's declarations are a seed the measurement replaces, got %+v", got.BackendRegistration)
		}
		if reg.slotCap["llm-a750"] != 1 {
			t.Errorf("slot cap = %d, want the measured 1", reg.slotCap["llm-a750"])
		}
	})
}

// TestDeclaredRegistrationSurvivesProfiling: an edit is applied to what the
// operator typed, not to what the profiler measured over the top of it.
// Otherwise the first PATCH promotes every measurement to a declaration and the
// probe can never refine those fields again.
func TestDeclaredRegistrationSurvivesProfiling(t *testing.T) {
	reg := newTestRegistry()
	b, _ := reg.upsert(manualReg(t, BackendRegistration{
		ID: "p", Model: "gpt-4o", MaxConcurrency: 8, TTLSeconds: 3600,
	}))
	reg.applyProfileIfGen("p", b.profileGen, &WorkerProfile{Quality: 91, ContextK: 256, BenchVersion: benchmarkVersion})

	if live := reg.get("p"); live.Quality != 91 {
		t.Fatalf("setup: measured quality should be live, got %d", live.Quality)
	}
	declared, ok := reg.declaredRegistration("p")
	if !ok {
		t.Fatal("declared registration missing")
	}
	if declared.Quality != 0 || declared.ContextK != 0 {
		t.Errorf("measurements leaked into the declared registration: %+v", declared)
	}
	if declared.MaxConcurrency != 8 {
		t.Errorf("declared max_concurrency = %d, want 8", declared.MaxConcurrency)
	}
}

// ── The declared ceiling against the ramp ───────────────────────────────────

// throttledWorker is a fake endpoint that serves a completion at up to limit
// concurrent requests and answers 429 above it — a rate-limited provider during
// the capacity ramp. It counts how far the ramp actually pushed it.
func throttledWorker(t *testing.T, limit int) (*httptest.Server, *int64) {
	t.Helper()
	var mu sync.Mutex
	inFlight, peak := 0, int64(0)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		mu.Lock()
		inFlight++
		if int64(inFlight) > atomic.LoadInt64(&peak) {
			atomic.StoreInt64(&peak, int64(inFlight))
		}
		over := inFlight > limit
		mu.Unlock()
		defer func() {
			mu.Lock()
			inFlight--
			mu.Unlock()
		}()
		if over {
			http.Error(w, `{"error":"rate limited"}`, http.StatusTooManyRequests)
			return
		}
		// Hold the request briefly so concurrent probes really do overlap.
		time.Sleep(20 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"the ocean is wide and deep"}}]}`))
	}))
	t.Cleanup(srv.Close)
	return srv, &peak
}

// withCapacityProbeRetryDelay shrinks the ramp's retry spacing so a test that
// deliberately provokes the retry ladder costs milliseconds rather than nine
// real seconds. Same shape as withQualityFloorWait.
func withCapacityProbeRetryDelay(t *testing.T, d time.Duration) {
	t.Helper()
	prev := capacityProbeRetryDelay
	capacityProbeRetryDelay = d
	t.Cleanup(func() { capacityProbeRetryDelay = prev })
}

// TestDeclaredConcurrencyOutranksRamp: a metered endpoint that 429s a burst must
// not be written off at serial dispatch. The declared value is the answer, and
// it also bounds how hard the ramp pushes — capacityProbeMax is one number for
// a whole fleet, this is the operator saying what THIS endpoint will take.
func TestDeclaredConcurrencyOutranksRamp(t *testing.T) {
	withCapacityProbeRetryDelay(t, time.Millisecond)
	srv, peak := throttledWorker(t, 1) // only ever serves one at a time
	reg := newTestRegistry()
	reg.upsert(manualReg(t, BackendRegistration{
		ID: "provider", URL: srv.URL, Model: "gpt-4o", MaxConcurrency: 4, TTLSeconds: 3600,
	}))
	r := &Router{
		cfg:      &Config{CapacityProbeMax: 16},
		registry: reg,
		client:   &http.Client{Timeout: 5 * time.Second},
	}
	b := reg.get("provider")

	ramp, ok := r.capacityProbe(b)
	if !ok {
		t.Fatal("capacity probe reported the worker unreachable")
	}
	// The ramp must never have fired more than the declared ceiling at the
	// endpoint: on a metered provider every extra probe is money and a rate limit.
	if p := atomic.LoadInt64(peak); p > 4 {
		t.Errorf("ramp fired %d concurrent probes at an endpoint declared at 4", p)
	}
	// And the throttling really did under-measure it, or there is nothing to fix.
	if ramp >= 4 {
		t.Fatalf("setup proves nothing: the ramp measured %d unaided", ramp)
	}
	if got := r.resolveCapacity(b, ramp); got != 4 {
		t.Fatalf("resolved capacity = %d, want the declared 4 — a 429 burst permanently under-rated the endpoint", got)
	}
}

// TestPublishedSlotsStillCapTheRamp: the llama.cpp rule this sits next to is
// unchanged. total_slots may only LOWER the ramp's answer, because there the
// failure being guarded against is a serialising worker over-reporting.
func TestPublishedSlotsStillCapTheRamp(t *testing.T) {
	b, done := fakeWorker(t, `{"data":[{"id":"m"}]}`, `{"total_slots":2}`)
	defer done()
	r := &Router{registry: newTestRegistry(), client: &http.Client{Timeout: 2 * time.Second}}

	if got := r.resolveCapacity(b, 8); got != 2 {
		t.Errorf("ramp 8 against 2 published slots = %d, want 2", got)
	}
	if got := r.resolveCapacity(b, 1); got != 1 {
		t.Errorf("published slots must not RAISE the ramp: got %d, want 1", got)
	}
}

// TestOperatorMaxConcurrencyIsManualOnly: the override is operator ownership,
// not a general "declared beats measured" rule. A beacon that declares a cap is
// still measured, exactly as the deployed fleet is today.
func TestOperatorMaxConcurrencyIsManualOnly(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "beacon", URL: "http://b", Model: "m", MaxConcurrency: 8, TTLSeconds: 3600})
	reg.upsert(manualReg(t, BackendRegistration{ID: "manual", Model: "m2", MaxConcurrency: 8, TTLSeconds: 3600}))

	if got := reg.operatorMaxConcurrency("beacon"); got != 0 {
		t.Errorf("beacon row reported an operator ceiling of %d", got)
	}
	if got := reg.operatorMaxConcurrency("manual"); got != 8 {
		t.Errorf("manual row reported %d, want 8", got)
	}
	if got := reg.operatorMaxConcurrency("gone"); got != 0 {
		t.Errorf("unknown id reported %d", got)
	}
}

// ── The push endpoint stays the beacon endpoint ─────────────────────────────

// newTestLogStore opens a throwaway store on a temp directory. Real SQLite, so
// the migrations under test are the ones that run.
func newTestLogStore(t *testing.T) *LogStore {
	t.Helper()
	logs, err := openLogStore(filepath.Join(t.TempDir(), "logs.sqlite"), 16384, "test-secret")
	if err != nil {
		t.Fatalf("openLogStore: %v", err)
	}
	t.Cleanup(func() { logs.Close() })
	return logs
}

func registerRouter(t *testing.T) *Router {
	t.Helper()
	return &Router{
		cfg:      &Config{},
		registry: newTestRegistry(),
		logs:     newTestLogStore(t),
		// handleRegisterBackend kicks a background certification, which dials.
		// Give it a client with a short timeout so it fails fast instead of
		// dereferencing nil.
		client: &http.Client{Timeout: time.Second},
	}
}

// TestPushRegistrationCannotClaimManual: manual is what stops a probe
// overwriting a row's declared values. A caller holding the worker token must
// not be able to grant that to itself by putting "source":"manual" in the
// payload.
func TestPushRegistrationCannotClaimManual(t *testing.T) {
	r := registerRouter(t)
	rec := httptest.NewRecorder()
	r.handleRegisterBackend(rec, post("/backends/register",
		`{"id":"sneaky","url":"http://sneaky","model":"m","source":"manual","input_price_per_mtok":99}`, ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("register = %d: %s", rec.Code, rec.Body.String())
	}
	b := r.registry.get("sneaky")
	if b.Source != sourceBeacon {
		t.Errorf("source = %q, want %q — the push endpoint only makes beacons", b.Source, sourceBeacon)
	}
	if reg := r.registry.get("sneaky"); reg.MaxConcurrency != 0 {
		t.Errorf("unexpected concurrency %d", reg.MaxConcurrency)
	}
}

// TestPushRegistrationCannotTakeOverManualRow: an id collision with an
// operator's row is refused rather than silently converting it to a beacon,
// which would discard the declared values manual rows exist to protect.
func TestPushRegistrationCannotTakeOverManualRow(t *testing.T) {
	r := registerRouter(t)
	r.registry.upsert(manualReg(t, BackendRegistration{
		ID: "openai-gpt-4o", Model: "gpt-4o", MaxConcurrency: 8, TTLSeconds: 3600,
	}))
	rec := httptest.NewRecorder()
	r.handleRegisterBackend(rec, post("/backends/register",
		`{"id":"openai-gpt-4o","url":"http://impostor","model":"m"}`, ""))
	if rec.Code != http.StatusConflict {
		t.Fatalf("register over a manual row = %d, want 409: %s", rec.Code, rec.Body.String())
	}
	if got := r.registry.get("openai-gpt-4o"); got.URL == "http://impostor" || got.MaxConcurrency != 8 {
		t.Fatalf("the operator's row was overwritten: %+v", got.BackendRegistration)
	}
	errorEnvelopeOf(t, rec)
}

// TestPushRegistrationUnchanged: the frozen endpoint still upserts, still keeps
// the 90-second default TTL, and still answers the same body — the new fields
// change nothing a deployed worker can observe.
func TestPushRegistrationUnchanged(t *testing.T) {
	r := registerRouter(t)
	rec := httptest.NewRecorder()
	r.handleRegisterBackend(rec, post("/workers/register",
		`{"id":"llm-a750","url":"http://a750:8080","model":"gemma4","features":["chat"],"api_key":"k"}`, ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("register = %d: %s", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("response is not JSON: %v", err)
	}
	if body["status"] != "registered" || body["id"] != "llm-a750" {
		t.Errorf("response body changed: %v", body)
	}
	b := r.registry.get("llm-a750")
	if b.TTLSeconds != 90 || b.HealthPath != "/health" || b.APIKey != "k" {
		t.Errorf("registration semantics changed: %+v", b.BackendRegistration)
	}
	// An identical re-post is still a keepalive, not a change — the new fields
	// are normalised on both sides, so registrationsEqual still recognises it.
	before := r.registry.get("llm-a750").profileGen
	rec = httptest.NewRecorder()
	r.handleRegisterBackend(rec, post("/workers/register",
		`{"id":"llm-a750","url":"http://a750:8080","model":"gemma4","features":["chat"],"api_key":"k"}`, ""))
	if after := r.registry.get("llm-a750").profileGen; after != before {
		t.Error("keepalive bumped the profile generation — the new fields broke registrationsEqual")
	}
}

// TestPersistedPreP2RegistrationLoadsAsLocalBeacon: the rows already sitting in
// backend_registrations were written before provider/source existed. They come
// back with both empty, and the startup path has to settle them the same way the
// push endpoint does — local, beacon, free — or an in-place upgrade would hand
// operator ownership to every worker in the fleet.
func TestPersistedPreP2RegistrationLoadsAsLocalBeacon(t *testing.T) {
	logs := newTestLogStore(t)
	// Exactly the JSON the old code wrote: no provider, no source, no prices.
	old := `{"id":"llm-a750","url":"http://a750:8080","model":"gemma4","quality":0,` +
		`"thinking":false,"context_k":0,"baseline_tps":0,"features":["chat"],` +
		`"health_path":"/health","ttl_seconds":90,"max_concurrency":0,"api_key":"plaintext-legacy"}`
	if _, err := logs.db.ExecContext(t.Context(),
		`INSERT INTO backend_registrations (id, updated_at, registration_json) VALUES (?, ?, ?)`,
		"llm-a750", time.Now().UTC().Format(time.RFC3339Nano), old); err != nil {
		t.Fatalf("seed old row: %v", err)
	}

	loaded, err := logs.LoadBackendRegistrations(t.Context())
	if err != nil || len(loaded) != 1 {
		t.Fatalf("load = %d rows, err=%v", len(loaded), err)
	}
	reg := loaded[0]
	if reg.APIKey != "plaintext-legacy" {
		t.Errorf("pre-encryption api key mangled: %q", reg.APIKey)
	}
	if err := normalizeRegistration(&reg); err != nil {
		t.Fatalf("normalizeRegistration: %v", err)
	}
	if reg.Provider != providerLocal || reg.Source != sourceBeacon {
		t.Fatalf("an upgraded row must be a local beacon, got provider=%q source=%q", reg.Provider, reg.Source)
	}
	if reg.InputPricePerMtok != 0 || reg.OutputPricePerMtok != 0 {
		t.Errorf("an upgraded row must cost nothing: %v/%v", reg.InputPricePerMtok, reg.OutputPricePerMtok)
	}
	// And it owns nothing, so the profiler still overwrites its declared values.
	regis := newTestRegistry()
	b, _ := regis.upsert(reg)
	regis.applyProfileIfGen("llm-a750", b.profileGen, &WorkerProfile{Quality: 88, MaxConcurrency: 3, BenchVersion: benchmarkVersion})
	if got := regis.get("llm-a750"); got.Quality != 88 || got.MaxConcurrency != 3 {
		t.Errorf("upgraded beacon row stopped accepting measurements: %+v", got.BackendRegistration)
	}
}

// TestSameEndpointModel: a row is one (endpoint, model) pair, and the trailing
// /v1 an operator may or may not paste is not part of the endpoint's identity.
func TestSameEndpointModel(t *testing.T) {
	mk := func(url, model string) *Backend {
		return &Backend{BackendRegistration: BackendRegistration{URL: url, Model: model}}
	}
	cases := []struct {
		a, b *Backend
		want bool
	}{
		{mk("https://api.example.com/v1", "gpt-4o"), mk("https://api.example.com/", "gpt-4o"), true},
		{mk("https://api.example.com/v1", "gpt-4o"), mk("https://api.example.com/v1", "gpt-4o-mini"), false},
		{mk("https://api.example.com/v1", "gpt-4o"), mk("https://other.example.com/v1", "gpt-4o"), false},
	}
	for _, c := range cases {
		if got := sameEndpointModel(c.a, c.b); got != c.want {
			t.Errorf("sameEndpointModel(%s/%s, %s/%s) = %v, want %v",
				c.a.URL, c.a.Model, c.b.URL, c.b.Model, got, c.want)
		}
	}
}

// ── P4: cost in the ranker ──────────────────────────────────────────────────

// pricedBackend is mkBackend with declared prices — the same worker, metered.
func pricedBackend(id string, quality, tps, maxConc, active int, in, out float64) *Backend {
	b := mkBackend(id, quality, tps, maxConc, active)
	b.InputPricePerMtok, b.OutputPricePerMtok = in, out
	return b
}

// registerPriced registers a ready worker carrying declared prices.
func registerPriced(t *testing.T, reg *Registry, id string, quality, maxConcurrency int, in, out float64) *Backend {
	t.Helper()
	reg.upsert(BackendRegistration{
		ID: id, URL: "http://" + id, Model: "default", Quality: quality,
		MaxConcurrency: maxConcurrency, TTLSeconds: 3600, Features: []string{"chat"},
		InputPricePerMtok: in, OutputPricePerMtok: out,
	})
	reg.finishCertification(id, true, map[string]Check{}, 50, 10, "")
	return reg.get(id)
}

func TestIsFreeBackendAndTokenCost(t *testing.T) {
	cases := []struct {
		name     string
		in, out  float64
		free     bool
		perMtoks float64 // cost of one million prompt + one million completion tokens
	}{
		{"a local worker declares nothing", 0, 0, true, 0},
		{"input priced only", 3, 0, false, 3},
		{"output priced only", 0, 15, false, 15},
		{"both priced", 3, 15, false, 18},
	}
	for _, c := range cases {
		b := &Backend{BackendRegistration: BackendRegistration{InputPricePerMtok: c.in, OutputPricePerMtok: c.out}}
		if got := isFreeBackend(b); got != c.free {
			t.Errorf("%s: isFreeBackend = %v, want %v", c.name, got, c.free)
		}
		if got := tokenCost(b, 1_000_000, 1_000_000); math.Abs(got-c.perMtoks) > 1e-9 {
			t.Errorf("%s: tokenCost = %g, want %g", c.name, got, c.perMtoks)
		}
	}
	if !isFreeBackend(nil) {
		t.Error("a nil backend must not read as paid")
	}
	if got := tokenCost(nil, 1000, 1000); got != 0 {
		t.Errorf("tokenCost(nil) = %g", got)
	}
}

// TestIsFreeBackendNeedsADeclaredZero: zero prices only mean FREE where a zero
// is a declaration. On a row an operator entered for someone else's endpoint it
// is also what "nobody typed a number and the table publishes none" looks like,
// and reading that as free puts a metered endpoint at the head of the free band,
// holds requests for it through the free-first grace, hands it to the judge as
// the free grader and never logs the paid spill.
//
// Which of the two a zero is can be read off the row, without storing anything:
// every manual row is seeded on the way in, so a model the table DOES publish
// could only be sitting at zero because someone overrode it.
func TestIsFreeBackendNeedsADeclaredZero(t *testing.T) {
	if len(prices().exact) == 0 {
		t.Skip("prices.json is the empty snapshot — nothing is published, so every zero reads as unknown")
	}
	// A model the shipped table prices, and one nothing publishes.
	const published, unpublished = "gpt-4o", "some-orgs-private-finetune-v3"
	if _, ok := lookupPrice(published, "openai"); !ok {
		t.Skipf("the embedded snapshot no longer publishes %s", published)
	}
	if _, ok := lookupPrice(unpublished, "whoever"); ok {
		t.Fatalf("%q was meant to be unpublished", unpublished)
	}
	row := func(source, provider, model string, in, out float64) *Backend {
		return &Backend{BackendRegistration: BackendRegistration{
			ID: "row", URL: "http://row", Source: source, Provider: provider, Model: model,
			InputPricePerMtok: in, OutputPricePerMtok: out,
		}}
	}
	cases := []struct {
		name string
		b    *Backend
		free bool
	}{
		// The fleet. Most of it is local, all of it declares nothing, and none of
		// it may ever read as unpriced.
		{"a local manual row", row(sourceManual, providerLocal, published, 0, 0), true},
		{"a local beacon", row(sourceBeacon, providerLocal, published, 0, 0), true},
		// /backends/register cannot express a vouched-for price, and every worker
		// deployed on it posts none — so a beacon's zero stays a declaration
		// whatever provider it names.
		{"a beacon that names a provider", row(sourceBeacon, "openai", published, 0, 0), true},
		{"an unnormalised row", row("", "", published, 0, 0), true},
		// The operator overrode a price the table publishes: a deliberate zero.
		{"a declared zero on a priced model", row(sourceManual, "openai", published, 0, 0), true},
		// Nobody typed a number and nothing could have seeded one.
		{"an unpriced metered row", row(sourceManual, "whoever", unpublished, 0, 0), false},
		{"any declared price at all", row(sourceManual, "whoever", unpublished, 0, 15), false},
	}
	for _, c := range cases {
		if got := isFreeBackend(c.b); got != c.free {
			t.Errorf("%s: isFreeBackend = %v, want %v", c.name, got, c.free)
		}
	}
}

// TestRankByDifficultyPrefersFreeAboveTheBar is PLAN.md's P4 rule as a ranking:
// among the workers that clear the quality bar the free one leads, even when a
// paid one would finish sooner. Below the bar cost says nothing — the router has
// already missed the quality it wanted, and buying a worse answer to save money
// is not a trade anyone asked for.
func TestRankByDifficultyPrefersFreeAboveTheBar(t *testing.T) {
	free := mkBackend("free", 8, 30, 4, 0)
	paidFast := pricedBackend("paid-fast", 9, 200, 4, 0, 3, 15)
	if got := rankByDifficulty([]*Backend{paidFast, free}, 7, nominalJob(), false); got[0].ID != "free" {
		t.Errorf("both clear q>=7: want the free worker first, got %s", got[0].ID)
	}
	// Only the paid worker clears the bar → it leads. Cost never buys worse.
	if got := rankByDifficulty([]*Backend{free, paidFast}, 9, nominalJob(), false); got[0].ID != "paid-fast" {
		t.Errorf("only paid clears q>=9: want paid-fast, got %s", got[0].ID)
	}
	// Below the bar, closest quality still wins.
	freeWeak := mkBackend("free-weak", 3, 200, 4, 0)
	paidMid := pricedBackend("paid-mid", 6, 30, 4, 0, 3, 15)
	if got := rankByDifficulty([]*Backend{freeWeak, paidMid}, 9, nominalJob(), false); got[0].ID != "paid-mid" {
		t.Errorf("both below q>=9: want the closest (paid-mid), got %s", got[0].ID)
	}
	// A saturated free worker still loses the head of the list: holding the
	// request for it is the acquire step's job, not the ranker's.
	fullFree := mkBackend("free", 8, 30, 1, 1)
	if got := rankByDifficulty([]*Backend{fullFree, paidFast}, 7, nominalJob(), false); got[0].ID != "paid-fast" {
		t.Errorf("free worker full: want paid-fast at the head, got %s", got[0].ID)
	}
}

// TestRankBackendsCostIsOnlyATieBreak covers the FALLBACK ranker (no classifier,
// so no quality bar): cost separates workers this ranker already considers
// interchangeable, and nothing more.
func TestRankBackendsCostIsOnlyATieBreak(t *testing.T) {
	// Identical quality and speed. The ids are chosen so that WITHOUT the cost
	// rule the final a.ID < b.ID tiebreak would put the paid one first.
	free := mkBackend("zzz-free", 7, 50, 4, 0)
	paid := pricedBackend("aaa-paid", 7, 50, 4, 0, 3, 15)
	if got := rankBackends([]*Backend{paid, free}, nominalJob(), false); got[0].ID != "zzz-free" {
		t.Errorf("equal score: want the free worker first, got %s", got[0].ID)
	}
	better := pricedBackend("aaa-paid-better", 10, 50, 4, 0, 3, 15)
	if got := rankBackends([]*Backend{free, better}, nominalJob(), false); got[0].ID != "aaa-paid-better" {
		t.Errorf("a clearly better paid worker must still lead, got %s", got[0].ID)
	}
}

// TestQualityFloorPreferenceTiers pins which bounded preference applies, since
// that single choice is where free-first and the quality floor are reconciled.
func TestQualityFloorPreferenceTiers(t *testing.T) {
	free8 := mkBackend("free8", 8, 50, 2, 0)
	free3 := mkBackend("free3", 3, 50, 2, 0)
	paid9 := pricedBackend("paid9", 9, 50, 2, 0, 3, 15)
	paid2 := pricedBackend("paid2", 2, 50, 2, 0, 3, 15)
	cases := []struct {
		name       string
		candidates []*Backend
		target     int
		why        string
	}{
		{"free and paid both clear the bar", []*Backend{free8, paid9}, 7, "free-first"},
		{"no classification, so no bar to clear", []*Backend{free8, paid9}, 0, "free-first"},
		{"every above-bar worker is paid", []*Backend{free3, paid9}, 7, "quality-floor"},
		{"no free worker at all", []*Backend{paid2, paid9}, 7, "quality-floor"},
		{"everything is free and above the bar", []*Backend{free8, free3}, 3, ""},
		{"everything is free, no bar", []*Backend{free8, free3}, 0, ""},
		{"nothing clears the bar", []*Backend{free3}, 9, ""},
	}
	for _, c := range cases {
		pref := qualityFloorPreference(c.candidates, c.target, false)
		if pref.why != c.why {
			t.Errorf("%s: preference = %q, want %q", c.name, pref.why, c.why)
		}
		if c.why == "" && pref.keep != nil {
			t.Errorf("%s: a no-op preference must be the zero value", c.name)
		}
	}
}

// TestFreeFirstServesFreeWhileItHasASlot: the ranked head is a paid endpoint
// (the free worker is slower), and acquisition still lands on the free one.
func TestFreeFirstServesFreeWhileItHasASlot(t *testing.T) {
	withQualityFloorWait(t, 5*time.Second) // large; must NOT be consumed
	reg := newTestRegistry()
	paid := registerPriced(t, reg, "paid", 9, 1, 3, 15)
	free := registerPriced(t, reg, "free", 8, 1, 0, 0)
	r := &Router{registry: reg}

	start := time.Now()
	got, slot, missed, err := r.pickAndAcquireWithFloor(context.Background(), []*Backend{paid, free}, 7, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ID != "free" {
		t.Fatalf("a free worker had a slot: want free, got %s", got.ID)
	}
	if missed {
		t.Error("serving the preferred free worker must not report a miss")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("took %s — the free worker was idle, there was nothing to wait for", elapsed)
	}
	reg.releaseSlot(slot)
}

// TestFreeFirstSpillsToPaidOnlyPastTheGrace: money is spent only after every
// free candidate has been saturated for the whole existing grace.
func TestFreeFirstSpillsToPaidOnlyPastTheGrace(t *testing.T) {
	withQualityFloorWait(t, 80*time.Millisecond)
	reg := newTestRegistry()
	paid := registerPriced(t, reg, "paid", 9, 1, 3, 15)
	free := registerPriced(t, reg, "free", 8, 1, 0, 0)
	r := &Router{registry: reg}

	if _, ok := reg.tryAcquireSlot("free"); !ok {
		t.Fatal("could not saturate the free worker")
	}
	start := time.Now()
	got, slot, missed, err := r.pickAndAcquireWithFloor(context.Background(), []*Backend{paid, free}, 7, false)
	if err != nil {
		t.Fatalf("the spill must still serve, got err=%v", err)
	}
	if got.ID != "paid" {
		t.Fatalf("free stayed full past the grace: want the paid spill, got %s", got.ID)
	}
	if !missed {
		t.Error("spending money after the grace must report a missed preference")
	}
	if elapsed := time.Since(start); elapsed < qualityFloorWait {
		t.Errorf("spent money after %s, before the %s grace elapsed", elapsed, qualityFloorWait)
	}
	reg.releaseSlot(slot)
}

// TestFreeFirstTakesTheFreeSlotThatFreesWithinTheGrace: the mirror case — the
// free worker frees up in time, so nothing is spent.
func TestFreeFirstTakesTheFreeSlotThatFreesWithinTheGrace(t *testing.T) {
	withQualityFloorWait(t, 2*time.Second)
	reg := newTestRegistry()
	paid := registerPriced(t, reg, "paid", 9, 1, 3, 15)
	free := registerPriced(t, reg, "free", 8, 1, 0, 0)
	r := &Router{registry: reg}

	held, ok := reg.tryAcquireSlot("free")
	if !ok {
		t.Fatal("could not saturate the free worker")
	}
	go func() {
		time.Sleep(50 * time.Millisecond)
		reg.releaseSlot(held)
	}()
	got, slot, missed, err := r.pickAndAcquireWithFloor(context.Background(), []*Backend{paid, free}, 7, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ID != "free" || missed {
		t.Fatalf("a free slot freed within the grace: got %s (missed=%v), want free", got.ID, missed)
	}
	reg.releaseSlot(slot)
}

// ── P4: what a profiling run cost ───────────────────────────────────────────

// profilingWorker is a fake endpoint that answers everything a cold profile
// asks it, reporting a FIXED usage block per completion so the accounting can
// be checked against a known total.
//
// Tools and JSON mode are refused with a 400 — a definitive reject, which is
// what a text-only endpoint does, and which keeps the capability probes from
// spending their transient-retry ladders (and this test's wall clock) on them.
func profilingWorker(t *testing.T, promptTokens, completionTokens int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.Method == http.MethodGet {
			http.NotFound(w, req) // no catalogue, no /props: context and slots stay unmeasured
			return
		}
		body, _ := io.ReadAll(req.Body)
		if bytes.Contains(body, []byte(`"tools"`)) || bytes.Contains(body, []byte(`"response_format"`)) {
			http.Error(w, `{"error":{"message":"unsupported"}}`, http.StatusBadRequest)
			return
		}
		usage := fmt.Sprintf(`"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}`,
			promptTokens, completionTokens, promptTokens+completionTokens)
		if bytes.Contains(body, []byte(`"stream":true`)) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok ok ok ok\"}}]}\n\n")
			fmt.Fprintf(w, "data: {\"choices\":[],%s}\n\n", usage)
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		fmt.Fprintf(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],%s}`, usage)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// profileCostRouter is a router wired for a real cold profile against srv, with
// the capacity ramp capped so the test doesn't measure a fake worker's ceiling.
func profileCostRouter(t *testing.T, srv *httptest.Server, in, out float64) (*Router, *Backend) {
	t.Helper()
	reg := newTestRegistry()
	r := &Router{
		cfg:         &Config{CapacityProbeMax: 2},
		registry:    reg,
		client:      &http.Client{Timeout: 10 * time.Second},
		benchClient: &http.Client{},
	}
	reg.upsert(manualReg(t, BackendRegistration{
		ID: "metered", URL: srv.URL, Model: "m", TTLSeconds: 3600,
		InputPricePerMtok: in, OutputPricePerMtok: out,
	}))
	return r, reg.get("metered")
}

// TestProfileRecordsWhatItSpent: a cold profile of a metered endpoint has to
// leave behind what it consumed and what that cost, or the operator finds out
// from their invoice (PLAN.md, "Known costs").
func TestProfileRecordsWhatItSpent(t *testing.T) {
	srv := profilingWorker(t, 1000, 40)
	r, b := profileCostRouter(t, srv, 3, 15)

	prof, err := r.profileBackend(b, "m")
	if err != nil {
		t.Fatalf("profileBackend: %v", err)
	}
	if prof.ProfilePromptTokens <= 0 || prof.ProfileOutputTokens <= 0 {
		t.Fatalf("nothing was metered: %d prompt / %d completion",
			prof.ProfilePromptTokens, prof.ProfileOutputTokens)
	}
	// The benchmark is the bulk of a run, so the totals have to be at least the
	// question count — anything less means whole probes went unmetered.
	if prof.ProfileOutputTokens < 40*len(benchmarkQuestions) {
		t.Errorf("completion tokens %d are below the %d questions' worth alone",
			prof.ProfileOutputTokens, len(benchmarkQuestions))
	}
	want := float64(prof.ProfilePromptTokens)/1e6*3 + float64(prof.ProfileOutputTokens)/1e6*15
	if math.Abs(prof.ProfileCost-want) > 1e-9 {
		t.Errorf("ProfileCost = %g, want %g at the row's declared prices", prof.ProfileCost, want)
	}
	if msg := prof.Checks["cost"].Message; !strings.Contains(msg, "at declared prices") {
		t.Errorf("the run's cost is not in the check list: %q", msg)
	}
	// The metering span closes with the run: nothing may keep counting afterwards.
	if _, still := r.profileMeters.Load("metered"); still {
		t.Error("the profile meter outlived the profiling run")
	}
}

// TestProfileOfAFreeWorkerCostsNothingButIsStillMeasured is the other half of
// "zero means not measured": a local worker consumes tokens and costs nothing,
// which must not read the same as a profile taken before the accounting existed.
func TestProfileOfAFreeWorkerCostsNothingButIsStillMeasured(t *testing.T) {
	srv := profilingWorker(t, 1000, 40)
	r, b := profileCostRouter(t, srv, 0, 0)

	prof, err := r.profileBackend(b, "m")
	if err != nil {
		t.Fatalf("profileBackend: %v", err)
	}
	if prof.ProfilePromptTokens <= 0 || prof.ProfileOutputTokens <= 0 {
		t.Fatal("a free worker's profile still has to record what it consumed")
	}
	if prof.ProfileCost != 0 {
		t.Errorf("a free worker's run cost %g", prof.ProfileCost)
	}
	if msg := prof.Checks["cost"].Message; !strings.Contains(msg, "free at declared prices") {
		t.Errorf("cost check = %q, want it to say the run was free", msg)
	}

	// A profile cached before P4 carries neither, and that reads as "not
	// measured" rather than "free" — the pair is what carries the distinction.
	old := &WorkerProfile{Model: "m", Quality: 71}
	if old.ProfilePromptTokens != 0 || old.ProfileOutputTokens != 0 || old.ProfileCost != 0 {
		t.Fatal("an old profile must decode to zeroes")
	}
}

// TestProfileMeterIgnoresTrafficOutsideARun: the meter is scoped to one run
// against one endpoint, so ordinary traffic can never inflate a stored cost.
func TestProfileMeterIgnoresTrafficOutsideARun(t *testing.T) {
	srv := profilingWorker(t, 100, 50)
	r, b := profileCostRouter(t, srv, 3, 15)
	payload := map[string]any{"model": "m", "messages": []map[string]string{{"role": "user", "content": "hi"}}}

	if _, err := r.rawCompletion(b, payload); err != nil {
		t.Fatalf("completion before the span: %v", err)
	}
	meter, done := r.meterProfile(b.ID)
	if _, err := r.rawCompletion(b, payload); err != nil {
		t.Fatalf("completion inside the span: %v", err)
	}
	done()
	if _, err := r.rawCompletion(b, payload); err != nil {
		t.Fatalf("completion after the span: %v", err)
	}
	prompt, output := meter.totals()
	if prompt != 100 || output != 50 {
		t.Errorf("metered %d/%d tokens, want exactly the one call inside the span (100/50)", prompt, output)
	}
	// A run against a DIFFERENT endpoint is a different meter.
	r.meterProfileTokens("someone-else", 999, 999)
	if prompt, output = meter.totals(); prompt != 100 || output != 50 {
		t.Errorf("another endpoint's tokens landed on this meter: %d/%d", prompt, output)
	}
}

func TestSlugifyAndProviderRowID(t *testing.T) {
	cases := []struct{ provider, model, want string }{
		{"openai", "gpt-4o", "openai-gpt-4o"},
		{"", "gpt-4o", "gpt-4o"},
		{"local", "gemma4", "gemma4"},
		{"OpenRouter", "meta-llama/Llama-3.3-70B-Instruct", "openrouter-meta-llama-llama-3-3-70b-instruct"},
		// Already prefixed with the provider — don't say it twice.
		{"mistral", "mistral-large-latest", "mistral-large-latest"},
	}
	for _, c := range cases {
		if got := providerRowID(c.provider, c.model); got != c.want {
			t.Errorf("providerRowID(%q, %q) = %q, want %q", c.provider, c.model, got, c.want)
		}
	}
	if got := slugify("  --Weird__Name!!  "); !strings.EqualFold(got, "weird-name") {
		t.Errorf("slugify = %q", got)
	}
}

// TestPlateauVerdictIsRetried: one noisy throughput sample must not cache a
// concurrent worker at serial dispatch. This is the mirror image of the
// failed-level retry ladder (and of TestDeclaredConcurrencyOutranksRamp's 429
// case): here the worker serves n=2 perfectly well, but its FIRST n=2 sample
// comes back slow — a speculative decoder's acceptance swing, a graph capture,
// scheduler jitter — and under the old single-sample knee test that one reading
// was a permanent verdict (measured live on an MTP engine, 2026-08-18: a
// 2-lane worker cached at capacity 1). The ramp must re-sample a plateau and
// keep the level's best reading.
func TestPlateauVerdictIsRetried(t *testing.T) {
	withCapacityProbeRetryDelay(t, time.Millisecond)
	var served int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Request 1 is the n=1 baseline (fast). Requests 2-3 are the first n=2
		// sample: slow enough that the level's aggregate lands under the
		// 1.15x knee. Requests 4+ (the re-sample) run at full speed.
		i := atomic.AddInt64(&served, 1)
		if i == 2 || i == 3 {
			time.Sleep(300 * time.Millisecond)
		} else {
			time.Sleep(20 * time.Millisecond)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"the ocean is wide and deep"}}]}`))
	}))
	t.Cleanup(srv.Close)

	reg := newTestRegistry()
	reg.upsert(manualReg(t, BackendRegistration{
		ID: "noisy", URL: srv.URL, Model: "m", TTLSeconds: 3600,
	}))
	r := &Router{
		// Cap the ramp at 2 so the test settles the level under dispute and
		// nothing above it.
		cfg:      &Config{CapacityProbeMax: 2},
		registry: reg,
		client:   &http.Client{Timeout: 5 * time.Second},
	}

	ramp, ok := r.capacityProbe(reg.get("noisy"))
	if !ok {
		t.Fatal("capacity probe reported the worker unreachable")
	}
	if ramp != 2 {
		t.Fatalf("capacity = %d, want 2 — a single slow n=2 sample became a permanent serial-dispatch verdict", ramp)
	}
	// The slow first sample must actually have been drawn, or the re-sample
	// path was never exercised: 1 (n=1) + 2 (slow n=2) + 2 (re-sample) = 5.
	if n := atomic.LoadInt64(&served); n < 5 {
		t.Fatalf("only %d requests served — the plateau was never provoked or never re-sampled", n)
	}
}
