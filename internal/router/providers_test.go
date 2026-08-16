package router

import (
	"encoding/json"
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
	if isMetered(&Backend{BackendRegistration: reg}) {
		t.Error("a local worker must not read as metered")
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
