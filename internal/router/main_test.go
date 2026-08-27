package router

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func newTestRegistry() *Registry {
	return &Registry{
		backends: map[string]*Backend{},
		slots:    map[string]chan struct{}{},
		slotCap:  map[string]int{},
	}
}

func register(t *testing.T, reg *Registry, id string, maxConcurrency int) *Backend {
	t.Helper()
	b, _ := reg.upsert(BackendRegistration{
		ID:             id,
		URL:            "http://" + id,
		Model:          "default",
		MaxConcurrency: maxConcurrency,
	})
	return b
}

// TestPickAndAcquireSpill is the core of fix #2: when the best-ranked backend
// has no free slot, the request must spill to the next candidate instead of
// blocking on the first.
func TestPickAndAcquireSpill(t *testing.T) {
	reg := newTestRegistry()
	register(t, reg, "a", 1)
	register(t, reg, "b", 1)
	r := &Router{registry: reg}
	a, b := reg.get("a"), reg.get("b")

	// Both free: pickAndAcquire takes the first (best-ranked) candidate.
	got, slot, err := r.pickAndAcquire(context.Background(), []*Backend{a, b})
	if err != nil || got.ID != "a" {
		t.Fatalf("both free: got=%v err=%v, want a", got, err)
	}
	reg.releaseSlot(slot)

	// Saturate a; the next pick must spill to b.
	if _, ok := reg.tryAcquireSlot("a"); !ok {
		t.Fatal("could not saturate a")
	}
	got, slot, err = r.pickAndAcquire(context.Background(), []*Backend{a, b})
	if err != nil || got.ID != "b" {
		t.Fatalf("a full: got=%v err=%v, want spill to b", got, err)
	}
	reg.releaseSlot(slot)
}

func TestPickAndAcquireUnbounded(t *testing.T) {
	reg := newTestRegistry()
	register(t, reg, "u", 0) // no declared cap
	r := &Router{registry: reg}
	u := reg.get("u")
	// An uncapped backend is always immediately available with a nil slot.
	for i := 0; i < 5; i++ {
		got, slot, err := r.pickAndAcquire(context.Background(), []*Backend{u})
		if err != nil || got.ID != "u" || slot != nil {
			t.Fatalf("iter %d: got=%v slot=%v err=%v", i, got, slot, err)
		}
	}
}

func TestPickAndAcquireContextCancel(t *testing.T) {
	reg := newTestRegistry()
	register(t, reg, "a", 1)
	r := &Router{registry: reg}
	a := reg.get("a")
	if _, ok := reg.tryAcquireSlot("a"); !ok { // saturate
		t.Fatal("could not saturate a")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := r.pickAndAcquire(ctx, []*Backend{a}); err == nil {
		t.Fatal("expected error when all candidates full and ctx cancelled")
	}
}

// registerQ registers a ready backend with an explicit Quality and concurrency
// cap, returning its clone — used by the quality-floor tests.
func registerQ(t *testing.T, reg *Registry, id string, quality, maxConcurrency int) *Backend {
	t.Helper()
	reg.upsert(BackendRegistration{
		ID:             id,
		URL:            "http://" + id,
		Model:          "default",
		Quality:        quality,
		MaxConcurrency: maxConcurrency,
		TTLSeconds:     3600,
		Features:       []string{"chat"},
	})
	reg.finishCertification(id, true, map[string]Check{}, 50, 10, "")
	return reg.get(id)
}

// withQualityFloorWait temporarily overrides the package-var grace so the floor
// tests run fast, restoring it on cleanup.
func withQualityFloorWait(t *testing.T, d time.Duration) {
	t.Helper()
	prev := qualityFloorWait
	qualityFloorWait = d
	t.Cleanup(func() { qualityFloorWait = prev })
}

// TestCircuitBreaker covers fix #4: repeated proxy failures eject a backend and
// mark it for re-certification; a success resets the counter.
func TestCircuitBreaker(t *testing.T) {
	reg := newTestRegistry()
	register(t, reg, "a", 1)
	reg.finishCertification("a", true, map[string]Check{}, 50, 10, "")
	if !reg.get("a").Healthy {
		t.Fatal("backend should be healthy after successful certification")
	}

	// Two failures then a success must not trip the breaker.
	reg.noteProxyResult("a", false)
	reg.noteProxyResult("a", false)
	reg.noteProxyResult("a", true)
	if !reg.get("a").Healthy {
		t.Fatal("breaker tripped too early (success should reset the counter)")
	}

	// proxyFailureThreshold consecutive failures must eject it.
	for i := 0; i < proxyFailureThreshold; i++ {
		reg.noteProxyResult("a", false)
	}
	b := reg.get("a")
	if b.Healthy || b.Certification.Ready {
		t.Fatalf("breaker did not trip: healthy=%v ready=%v", b.Healthy, b.Certification.Ready)
	}
	if !reg.dueForRecertify("a") {
		t.Fatal("ejected backend should be due for re-certification")
	}
}

// TestFinishCertificationPreservesLiveTPS covers the minor fix: a re-cert must
// SEED ObservedTPS from the probe only when there's no live EWMA yet; an existing
// runtime-learned value is more current than a one-shot profile and must survive.
func TestFinishCertificationPreservesLiveTPS(t *testing.T) {
	reg := newTestRegistry()
	register(t, reg, "a", 1)

	// First certification seeds ObservedTPS from the probe (it starts at 0).
	reg.finishCertification("a", true, map[string]Check{}, 50, 10, "")
	if got := reg.get("a").ObservedTPS; got != 50 {
		t.Fatalf("initial cert should seed ObservedTPS=50, got %v", got)
	}

	// Runtime learns a higher live throughput via the streamed-request EWMA.
	reg.observe("a", 20*time.Millisecond, time.Second, 120, 0, false) // 120 tok/s sample
	live := reg.get("a").ObservedTPS
	if live <= 50 {
		t.Fatalf("observe should have moved the EWMA above the seed, got %v", live)
	}

	// A re-cert (e.g. circuit-breaker recovery) with a stale one-shot probe must
	// NOT clobber the live EWMA back to the profiled baseline.
	reg.finishCertification("a", true, map[string]Check{}, 50, 10, "")
	if got := reg.get("a").ObservedTPS; got != live {
		t.Fatalf("re-cert clobbered the live EWMA: got %v, want preserved %v", got, live)
	}
}

// TestRecertBackoff covers fix #3: a failing backend is not eligible for
// immediate re-certification, and the gap grows with each failure.
func TestRecertBackoff(t *testing.T) {
	reg := newTestRegistry()
	register(t, reg, "a", 1)
	reg.finishCertification("a", false, map[string]Check{}, 0, 0, "boom")
	if reg.dueForRecertify("a") {
		t.Fatal("a just-failed backend should be in backoff, not due immediately")
	}

	cases := []struct {
		failures int
		want     time.Duration
	}{
		{1, 30 * time.Second},
		{2, 60 * time.Second},
		{3, 120 * time.Second},
		{20, 10 * time.Minute}, // capped
	}
	for _, c := range cases {
		if got := recertBackoff(c.failures); got != c.want {
			t.Errorf("recertBackoff(%d)=%s, want %s", c.failures, got, c.want)
		}
	}
}

func TestSecretBoxRoundTrip(t *testing.T) {
	for _, mk := range []struct {
		name string
		make func(t *testing.T) *secretBox
	}{
		{"env-secret", func(t *testing.T) *secretBox {
			box, err := newSecretBox("hunter2", "")
			if err != nil {
				t.Fatalf("newSecretBox: %v", err)
			}
			return box
		}},
		{"key-file", func(t *testing.T) *secretBox {
			box, err := newSecretBox("", filepath.Join(t.TempDir(), "persist.key"))
			if err != nil {
				t.Fatalf("newSecretBox: %v", err)
			}
			return box
		}},
	} {
		t.Run(mk.name, func(t *testing.T) {
			box := mk.make(t)
			const secret = "sk-abc123-very-secret"
			sealed, err := box.seal(secret)
			if err != nil {
				t.Fatalf("seal: %v", err)
			}
			if !strings.HasPrefix(sealed, encPrefix) {
				t.Fatalf("sealed value missing marker: %q", sealed)
			}
			if strings.Contains(sealed, secret) {
				t.Fatal("plaintext secret leaked into sealed value")
			}
			out, err := box.open(sealed)
			if err != nil || out != secret {
				t.Fatalf("open: got %q err=%v, want %q", out, err, secret)
			}
			// Empty stays empty; legacy plaintext passes through unchanged.
			if v, _ := box.seal(""); v != "" {
				t.Fatalf("seal(\"\")=%q, want empty", v)
			}
			if v, _ := box.open("plaintext-legacy"); v != "plaintext-legacy" {
				t.Fatalf("open(plaintext)=%q, want unchanged", v)
			}
		})
	}
}

func TestSecretBoxWrongKeyFails(t *testing.T) {
	a, _ := newSecretBox("key-a", "")
	b, _ := newSecretBox("key-b", "")
	sealed, _ := a.seal("secret")
	if _, err := b.open(sealed); err == nil {
		t.Fatal("decrypting with the wrong key must fail")
	}
}

func TestKeyFileStable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "persist.key")
	box1, err := newSecretBox("", path)
	if err != nil {
		t.Fatalf("first newSecretBox: %v", err)
	}
	sealed, _ := box1.seal("secret")
	// A second box loading the same key file must decrypt the first's output.
	box2, err := newSecretBox("", path)
	if err != nil {
		t.Fatalf("second newSecretBox: %v", err)
	}
	if out, err := box2.open(sealed); err != nil || out != "secret" {
		t.Fatalf("reloaded key file failed to decrypt: got %q err=%v", out, err)
	}
}

func TestClipLog(t *testing.T) {
	if got := clipLog("short", 16); got != "short" {
		t.Fatalf("short string was altered: %q", got)
	}
	long := strings.Repeat("a", 100)
	got := clipLog(long, 10)
	if !strings.HasPrefix(got, strings.Repeat("a", 10)) || !strings.Contains(got, "truncated 90 bytes") {
		t.Fatalf("unexpected clip: %q", got)
	}
	// Truncation must not split a multi-byte rune (would corrupt the field).
	multi := strings.Repeat("é", 20) // 2 bytes each
	clipped := clipLog(multi, 5)
	for _, r := range clipped {
		if r == '�' {
			t.Fatal("clip split a UTF-8 rune")
		}
	}
}

func authReq(authHeader string) *http.Request {
	req := httptest.NewRequest("GET", "/", nil)
	if authHeader != "" {
		req.Header.Set("Authorization", authHeader)
	}
	return req
}

func TestConstantTimeAuth(t *testing.T) {
	tokens := []string{"client-one", "client-two"}
	// Matching token authorizes.
	if !authorizedAsClient(authReq("Bearer client-two"), tokens) {
		t.Fatal("valid client token rejected")
	}
	// Wrong token rejected.
	if authorizedAsClient(authReq("Bearer nope"), tokens) {
		t.Fatal("invalid client token accepted")
	}
	// Empty config disables the check.
	if !authorizedAsClient(authReq(""), nil) {
		t.Fatal("empty token list should disable client auth")
	}
	// Worker token.
	if !authorizedAsWorker(authReq("Bearer w"), "w") {
		t.Fatal("valid worker token rejected")
	}
	if authorizedAsWorker(authReq("Bearer x"), "w") {
		t.Fatal("invalid worker token accepted")
	}
	if !authorizedAsWorker(authReq(""), "") {
		t.Fatal("empty worker token should disable worker auth")
	}
}

// TestUpsertKeepalivePreservesState: an unchanged re-registration (the ~60s
// keepalive) must not reset certification/health — resetting it knocked ready
// workers out of rotation and fought the background profiler. Changed content
// must do a full reset and bump the profile generation.
func TestUpsertKeepalivePreservesState(t *testing.T) {
	reg := newTestRegistry()
	mk := func(conc int) BackendRegistration {
		return BackendRegistration{
			ID: "w", URL: "http://w", Model: "default",
			MaxConcurrency: conc, TTLSeconds: 3600, Features: []string{"chat"},
		}
	}
	b1, changed := reg.upsert(mk(0))
	if !changed {
		t.Fatal("first registration must report changed")
	}
	gen1 := b1.profileGen
	reg.finishCertification("w", true, map[string]Check{}, 50, 10, "")

	b2, changed := reg.upsert(mk(0))
	if changed {
		t.Fatal("identical re-registration must be a keepalive, not a change")
	}
	if !b2.Certification.Ready || b2.Status != "ready" || !b2.Healthy {
		t.Fatalf("keepalive reset state: ready=%v status=%q healthy=%v",
			b2.Certification.Ready, b2.Status, b2.Healthy)
	}
	if b2.profileGen != gen1 {
		t.Fatal("keepalive must not bump the profile generation")
	}
	if len(reg.eligible()) != 1 {
		t.Fatal("backend must stay eligible across a keepalive")
	}

	b3, changed := reg.upsert(mk(4))
	if !changed || b3.Certification.Ready || b3.Status != "probing" {
		t.Fatalf("changed registration must reset to probing: changed=%v ready=%v status=%q",
			changed, b3.Certification.Ready, b3.Status)
	}
	if b3.profileGen == gen1 {
		t.Fatal("changed registration must bump the profile generation")
	}
}

// TestApplyProfileGenerationGuard: profile results measured against a stale
// registration generation (worker re-registered or deleted mid-profile) must
// not be applied.
func TestApplyProfileGenerationGuard(t *testing.T) {
	reg := newTestRegistry()
	b, _ := reg.upsert(BackendRegistration{ID: "w", URL: "http://w", Model: "default", TTLSeconds: 3600})
	gen := b.profileGen

	if !reg.applyProfileIfGen("w", gen, &WorkerProfile{Quality: 7}) {
		t.Fatal("matching generation must apply")
	}
	if reg.get("w").Quality != 7 {
		t.Fatal("profile not applied")
	}

	// Re-register with changed content → new generation → stale profile dropped.
	reg.upsert(BackendRegistration{ID: "w", URL: "http://w2", Model: "default", TTLSeconds: 3600})
	if reg.applyProfileIfGen("w", gen, &WorkerProfile{Quality: 9}) {
		t.Fatal("stale generation must not apply")
	}
	if reg.get("w").Quality == 9 {
		t.Fatal("stale profile leaked into the registry")
	}

	reg.remove("w")
	if reg.applyProfileIfGen("w", gen, &WorkerProfile{Quality: 9}) {
		t.Fatal("deleted backend must not accept a profile")
	}
}

// TestApplyProfileSyncsSlots: a FULL profile (BenchVersion set) must wire its
// measured capacity into the slot system; the quick profile's provisional
// MaxConcurrency=1 placeholder must not.
func TestApplyProfileSyncsSlots(t *testing.T) {
	reg := newTestRegistry()
	b, _ := reg.upsert(BackendRegistration{ID: "w", URL: "http://w", Model: "default", TTLSeconds: 3600})

	reg.applyProfileIfGen("w", b.profileGen, &WorkerProfile{Quality: 3, MaxConcurrency: 1})
	if _, ok := reg.slots["w"]; ok {
		t.Fatal("quick (provisional) profile must not create a slot channel")
	}

	reg.applyProfileIfGen("w", b.profileGen, &WorkerProfile{Quality: 8, MaxConcurrency: 2, BenchVersion: benchmarkVersion})
	if reg.slotCap["w"] != 2 {
		t.Fatalf("measured capacity not wired into slots: cap=%d", reg.slotCap["w"])
	}
	// Both slots acquirable, third attempt full — the spill machinery engages.
	if _, ok := reg.tryAcquireSlot("w"); !ok {
		t.Fatal("slot 1 should acquire")
	}
	if _, ok := reg.tryAcquireSlot("w"); !ok {
		t.Fatal("slot 2 should acquire")
	}
	if _, ok := reg.tryAcquireSlot("w"); ok {
		t.Fatal("third acquire should report full")
	}
}

// TestDueForRecertifyStuckProbing: a backend stranded in "probing" (its
// certification bailed on the in-flight-profile guard) must become due for
// re-certification once the staleness window passes.
func TestDueForRecertifyStuckProbing(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "w", URL: "http://w", Model: "default", TTLSeconds: 3600})
	if reg.dueForRecertify("w") {
		t.Fatal("fresh probing backend must not be due yet")
	}
	reg.mu.Lock()
	reg.backends["w"].Certification.StartedAt = time.Now().Add(-3 * time.Minute)
	reg.mu.Unlock()
	if !reg.dueForRecertify("w") {
		t.Fatal("stale probing backend must be due for rescue")
	}
}

// ── Measurement fixes: speed scale, thinking-safe TTFT, length-weighted decode ──

// TestSpeedScoreCommensurableWithQuality covers the stale-scale bug: speedScore
// was bucketed 1-10 from when Quality was also 1-10. Quality is now a 0-100
// benchmark percentage, which had reduced speed to ~3% of backendScore — in the
// ranker used precisely when the embeddings worker is down.
func TestSpeedScoreCommensurableWithQuality(t *testing.T) {
	slow := &Backend{BackendRegistration: BackendRegistration{BaselineTPS: 30}}
	fast := &Backend{BackendRegistration: BackendRegistration{BaselineTPS: 150}}
	if speedScore(fast) != 100 {
		t.Fatalf("a full-rate worker should score 100, got %d", speedScore(fast))
	}
	// The speed spread must be big enough to matter next to a quality spread.
	if gap := speedScore(fast) - speedScore(slow); gap < 50 {
		t.Fatalf("speed spread too small to influence backendScore: %d", gap)
	}
	// Quality still dominates (weighted 3x), so a much better model outranks a
	// much faster one — the fallback ranker stays quality-weighted.
	better := &Backend{BackendRegistration: BackendRegistration{Quality: 100, BaselineTPS: 30}}
	quicker := &Backend{BackendRegistration: BackendRegistration{Quality: 60, BaselineTPS: 150}}
	if backendScore(better, false) <= backendScore(quicker, false) {
		t.Fatalf("quality should still dominate: %d vs %d", backendScore(better, false), backendScore(quicker, false))
	}
	// Uncapped/unknown speed must not be scored as if it were fast.
	if speedScore(&Backend{}) != 0 {
		t.Fatal("unknown baseline_tps should score 0")
	}
}

// TestObserveSkipsThinkingTTFT covers the cross-worker comparability bug: vLLM
// buffers reasoning, so a thinking turn's whole think phase lands inside TTFT
// (measured 12.45s of a 13.15s turn) while llama.cpp streams it (0.7s). Folding
// both into one EWMA made the faster prefill engine look ~30x slower.
func TestObserveSkipsThinkingTTFT(t *testing.T) {
	reg := newTestRegistry()
	register(t, reg, "a", 1)

	// A thinking turn must contribute decode throughput but NOT TTFT/prefill.
	reg.observe("a", 12*time.Second, 2*time.Second, 600, 4000, true)
	b := reg.get("a")
	if b.ObservedTTFTMillis != 0 {
		t.Fatalf("thinking turn polluted the TTFT EWMA: %v", b.ObservedTTFTMillis)
	}
	if b.ObservedPrefillTPS != 0 {
		t.Fatalf("thinking turn polluted the prefill EWMA: %v", b.ObservedPrefillTPS)
	}
	if b.ObservedTPS == 0 {
		t.Fatal("thinking turn should still contribute decode throughput")
	}

	// A non-thinking turn with a real prompt yields a prefill rate.
	reg.observe("a", 500*time.Millisecond, 2*time.Second, 600, 2000, false)
	b = reg.get("a")
	if b.ObservedTTFTMillis == 0 {
		t.Fatal("non-thinking turn should fold TTFT")
	}
	if got := b.ObservedPrefillTPS; got < 3900 || got > 4100 {
		t.Fatalf("prefill rate = %v, want ~4000 tok/s (2000 tokens / 0.5s)", got)
	}

	// A tiny prompt says nothing about prefill throughput — TTFT there is mostly
	// fixed request overhead, so it must not move the rate.
	before := reg.get("a").ObservedPrefillTPS
	reg.observe("a", 300*time.Millisecond, time.Second, 100, 10, false)
	if reg.get("a").ObservedPrefillTPS != before {
		t.Fatal("a sub-minPrefillTokens prompt must not move the prefill EWMA")
	}
}

// TestObserveWeightsDecodeByLength covers the CPU-throughput overestimate: a
// stream of short replies pushed one worker's ObservedTPS to 51 tok/s when it
// sustained 17 over 1700 tokens, because every sample moved the EWMA equally.
func TestObserveWeightsDecodeByLength(t *testing.T) {
	short := newTestRegistry()
	register(t, short, "a", 1)
	long := newTestRegistry()
	register(t, long, "a", 1)

	// Seed both EWMAs to 20 tok/s through the real path (get() returns a clone,
	// so it can't be used to set state), then feed each one 200 tok/s sample —
	// differing only in how much generation it observed.
	short.finishCertification("a", true, map[string]Check{}, 20, 10, "")
	long.finishCertification("a", true, map[string]Check{}, 20, 10, "")
	if got := short.get("a").ObservedTPS; got != 20 {
		t.Fatalf("seed failed: ObservedTPS=%v, want 20", got)
	}
	short.observe("a", 0, time.Second/10, 20, 0, false) // 200 tok/s over 20 tokens
	long.observe("a", 0, 5*time.Second, 1000, 0, false) // 200 tok/s over 1000 tokens

	movedShort := short.get("a").ObservedTPS - 20
	movedLong := long.get("a").ObservedTPS - 20
	if movedShort <= 0 || movedLong <= 0 {
		t.Fatalf("both samples should raise the EWMA: %v / %v", movedShort, movedLong)
	}
	if movedLong <= movedShort*3 {
		t.Fatalf("a long sample must dominate a short one: moved %v vs %v", movedLong, movedShort)
	}
}

// ── Northbound API surface ──────────────────────────────────────────────────

// errorEnvelopeOf reads the OpenAI error envelope a northbound error must carry
// and fails the test if it is not there. The old shape was a bare
// {"message":"…"}, which every client written against the standard reads as an
// empty error, so this checks the wrapper as well as the content type.
func errorEnvelopeOf(t *testing.T, rec *httptest.ResponseRecorder) errorBody {
	t.Helper()
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("error content type = %q, want application/json", ct)
	}
	var got struct {
		Error *errorBody `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("error body is not JSON (%v): %s", err, rec.Body.String())
	}
	if got.Error == nil {
		t.Fatalf("error body has no \"error\" object: %s", rec.Body.String())
	}
	if got.Error.Message == "" {
		t.Errorf("error carries no message: %s", rec.Body.String())
	}
	return *got.Error
}

// envelopeRouter is a router with one ready worker and client auth switched on,
// enough to drive every northbound error path.
func envelopeRouter(t *testing.T) *Router {
	t.Helper()
	reg := newTestRegistry()
	registerQ(t, reg, "w", 50, 1)
	return &Router{
		cfg:      &Config{ClientTokens: []string{"good"}, DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second},
		registry: reg,
	}
}

func post(path, body string, token string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return req
}

// authGet is post's read-side twin, for the endpoints a client GETs.
func authGet(path string, token string) *http.Request {
	req := httptest.NewRequest(http.MethodGet, path, nil)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return req
}

// TestNorthboundErrorEnvelope: every error the router returns is the OpenAI
// envelope with a JSON content type and a type matching its status. This is the
// contract clients are written against, and it used to hold on none of these
// paths — validationError marshalled to {"message":…} and the http.Error sites
// sent a JSON body labelled text/plain.
func TestNorthboundErrorEnvelope(t *testing.T) {
	r := envelopeRouter(t)
	chat := `{"model":"default","messages":[{"role":"user","content":"hi"}]}`

	cases := []struct {
		name   string
		serve  func(rec *httptest.ResponseRecorder)
		status int
		typ    string
	}{
		{"no client token", func(rec *httptest.ResponseRecorder) {
			r.handleChatCompletions(rec, post("/v1/chat/completions", chat, ""))
		}, http.StatusUnauthorized, "authentication_error"},
		{"wrong client token", func(rec *httptest.ResponseRecorder) {
			r.handleModels(rec, authReq("Bearer nope"))
		}, http.StatusUnauthorized, "authentication_error"},
		{"malformed body", func(rec *httptest.ResponseRecorder) {
			r.handleChatCompletions(rec, post("/v1/chat/completions", `{not json`, "good"))
		}, http.StatusBadRequest, "invalid_request_error"},
		{"unsupported role", func(rec *httptest.ResponseRecorder) {
			r.handleChatCompletions(rec, post("/v1/chat/completions",
				`{"messages":[{"role":"wizard","content":"hi"}]}`, "good"))
		}, http.StatusBadRequest, "invalid_request_error"},
		{"unknown model", func(rec *httptest.ResponseRecorder) {
			r.handleChatCompletions(rec, post("/v1/chat/completions",
				`{"model":"gpt-whatever","messages":[{"role":"user","content":"hi"}]}`, "good"))
		}, http.StatusNotFound, "not_found_error"},
		{"unknown model object", func(rec *httptest.ResponseRecorder) {
			req := httptest.NewRequest(http.MethodGet, "/v1/models/gpt-whatever", nil)
			req.Header.Set("Authorization", "Bearer good")
			r.handleModelByID(rec, req)
		}, http.StatusNotFound, "not_found_error"},
		{"unrouted path", func(rec *httptest.ResponseRecorder) {
			r.handleDashboard(rec, httptest.NewRequest(http.MethodGet, "/nope", nil))
		}, http.StatusNotFound, "not_found_error"},
		{"wrong method", func(rec *httptest.ResponseRecorder) {
			r.handleChatCompletions(rec, httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil))
		}, http.StatusMethodNotAllowed, "invalid_request_error"},
		{"nothing to serve", func(rec *httptest.ResponseRecorder) {
			bare := &Router{cfg: r.cfg, registry: newTestRegistry()}
			bare.handleChatCompletions(rec, post("/v1/chat/completions", chat, "good"))
		}, http.StatusServiceUnavailable, "service_unavailable"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			tc.serve(rec)
			if rec.Code != tc.status {
				t.Fatalf("status = %d, want %d: %s", rec.Code, tc.status, rec.Body.String())
			}
			body := errorEnvelopeOf(t, rec)
			if body.Type != tc.typ {
				t.Errorf("error type = %q, want %q", body.Type, tc.typ)
			}
		})
	}
}

// TestClientCredentialIsNotForwardedDownstream: a caller's own key must never
// leave this router.
//
// The proxy used to send the CLIENT's Authorization header verbatim to any
// backend that had declared no api_key of its own. That was defensible while a
// client token was one shared LAN secret held by services the operator ran. It
// is not defensible now that callers hold their own `sk-` keys: every backend
// that pushed itself to /backends/register — which is every beacon, and anything
// a stranger managed to register — was handed each caller's live credential on
// every request it served.
//
// All four upstream request paths are covered, because each built the header
// separately and only one of them being right is the same as none of them being
// right.
func TestClientCredentialIsNotForwardedDownstream(t *testing.T) {
	const callerSecret = "sk-the-callers-own-key"
	routes := []struct {
		name, path, body string
	}{
		{"buffered chat", "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}]}`},
		{"streamed chat", "/v1/chat/completions", `{"stream":true,"messages":[{"role":"user","content":"hi"}]}`},
		{"passthrough completions", "/v1/completions", `{"prompt":"hi"}`},
		{"passthrough embeddings", "/v1/embeddings", `{"input":"hi"}`},
	}
	cases := []struct {
		name, backendKey, want string
	}{
		{"a backend with its own key is sent that key", "sk-the-backends-own-key", "Bearer sk-the-backends-own-key"},
		{"a backend with none is sent no Authorization at all", "", ""},
	}
	for _, tc := range cases {
		for _, route := range routes {
			t.Run(tc.name+"/"+route.name, func(t *testing.T) {
				var seen atomic.Value
				seen.Store("")
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
					seen.Store(req.Header.Get("Authorization"))
					if strings.Contains(route.body, `"stream":true`) {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n")
						return
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],`+
						`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
				}))
				t.Cleanup(srv.Close)

				reg := newTestRegistry()
				reg.upsert(BackendRegistration{
					ID: "w", URL: srv.URL, Model: "gemma4", Quality: 50, APIKey: tc.backendKey,
					TTLSeconds: 3600, Features: []string{"chat", "embeddings"},
				})
				reg.finishCertification("w", true, map[string]Check{}, 50, 10, "")
				r := &Router{
					cfg:      &Config{DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second},
					registry: reg, logs: newTestLogStore(t),
					client: &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
				}
				issueKey(t, r, callerSecret, apiKey{Role: roleClient, Name: "a stranger"})

				rec := httptest.NewRecorder()
				r.routes().ServeHTTP(rec, post(route.path, route.body, callerSecret))
				if rec.Code != http.StatusOK {
					t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
				}
				got, _ := seen.Load().(string)
				if strings.Contains(got, callerSecret) {
					t.Fatalf("the backend was handed the caller's own credential: %q", got)
				}
				if got != tc.want {
					t.Errorf("backend saw Authorization %q, want %q", got, tc.want)
				}
			})
		}
	}
}

// TestRetryAfterOn503: a 503 with no Retry-After tells a client nothing about
// whether to back off for a second or a minute, so both 503 causes carry one —
// derived from the health interval (when the router's view of the fleet can
// change) and the slot queue (how long the caller just spent waiting).
func TestRetryAfterOn503(t *testing.T) {
	r := &Router{
		cfg:      &Config{DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second},
		registry: newTestRegistry(),
	}
	rec := httptest.NewRecorder()
	r.handleChatCompletions(rec, post("/v1/chat/completions",
		`{"messages":[{"role":"user","content":"hi"}]}`, ""))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "15" {
		t.Errorf("Retry-After = %q, want the health interval (15)", got)
	}

	// Saturation is bounded by the queue the caller just spent, capped so a
	// ten-minute slot wait is not turned into a ten-minute back-off.
	if got := r.retryAfterSaturated(); got != retryAfterCeiling {
		t.Errorf("saturated hint = %s, want the ceiling %s (slotMaxWait is %s)", got, retryAfterCeiling, slotMaxWait)
	}
	// Never below one health interval: a retry sooner than that cannot see a
	// changed fleet.
	slow := &Router{cfg: &Config{HealthInterval: 2 * time.Minute}}
	if got := slow.retryAfterSaturated(); got != 2*time.Minute {
		t.Errorf("saturated hint = %s, want the health interval floor 2m", got)
	}
	// And never zero, whatever it is derived from — "0" reads as retry now.
	rec = httptest.NewRecorder()
	writeUnavailable(rec, 0, "nothing")
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want 1", got)
	}
}

// TestModelByID covers GET /v1/models/{id}, which used to fall through to the
// dashboard handler and 404 in HTML. The object must be the same one the list
// publishes, and both published spellings must resolve to it.
func TestModelByID(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{
		ID: "w1", URL: "http://w1", Model: "/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
		TTLSeconds: 3600, Features: []string{"chat"},
	})
	reg.finishCertification("w1", true, map[string]Check{}, 50, 10, "")
	r := &Router{cfg: &Config{}, registry: reg}

	get := func(path string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		r.handleModelByID(rec, httptest.NewRequest(http.MethodGet, path, nil))
		return rec
	}
	// The menu id is the alias; the raw model id is published as "root". Both
	// have to resolve, and to the same object the list carries.
	for _, name := range []string{"gemma4", "/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", "default"} {
		rec := get("/v1/models/" + name)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET /v1/models/%s = %d: %s", name, rec.Code, rec.Body.String())
		}
		var got map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatalf("model object is not JSON: %v", err)
		}
		if got["object"] != "model" {
			t.Errorf("GET /v1/models/%s returned object=%v, want \"model\"", name, got["object"])
		}
		want := "gemma4"
		if name == "default" {
			want = "default"
		}
		if got["id"] != want {
			t.Errorf("GET /v1/models/%s returned id=%v, want %q", name, got["id"], want)
		}
	}
	if rec := get("/v1/models/nope"); rec.Code != http.StatusNotFound {
		t.Errorf("unknown model = %d, want 404", rec.Code)
	}
	// The original defect was routing, not handling: /v1/models was an exact
	// match, so /v1/models/{id} fell through to the dashboard handler and 404'd
	// in HTML. Dispatch through the real mux, or that can silently come back.
	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models/gemma4", nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"object":"model"`) {
		t.Errorf("mux did not route the single-model path: %d %s", rec.Code, rec.Body.String())
	}
	// And the list itself is still reachable at the exact pattern.
	rec = httptest.NewRecorder()
	r.routes().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"object":"list"`) {
		t.Errorf("adding the subtree pattern broke the list: %d %s", rec.Code, rec.Body.String())
	}
	// A bare trailing slash is still the list, not a model named "".
	if rec := get("/v1/models/"); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"object":"list"`) {
		t.Errorf("GET /v1/models/ = %d: %s", rec.Code, rec.Body.String())
	}

	// Same client-auth scope as /v1/models.
	locked := &Router{cfg: &Config{ClientTokens: []string{"good"}}, registry: reg}
	rec = httptest.NewRecorder()
	locked.handleModelByID(rec, httptest.NewRequest(http.MethodGet, "/v1/models/gemma4", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("unauthenticated single-model fetch = %d, want 401", rec.Code)
	}
}

// TestLegacyFunctionRoleAccepted: "function" is OpenAI's deprecated spelling of
// a tool result and it still accepts it. The validator rejecting it was stricter
// than the standard, and inconsistent with the session tracker, which has always
// treated that role as continuing a tool loop.
func TestLegacyFunctionRoleAccepted(t *testing.T) {
	body := []byte(`{"messages":[
		{"role":"user","content":"weather?"},
		{"role":"assistant","content":"","function_call":{"name":"weather"}},
		{"role":"function","name":"weather","content":"17C"}]}`)
	req, err := parseAndValidateChatRequest(body, 4096)
	if err != nil {
		t.Fatalf("legacy function role rejected: %v", err)
	}
	if len(req.Messages) != 3 {
		t.Fatalf("parsed %d messages, want 3", len(req.Messages))
	}
	// It carries `name`, not `tool_call_id`, so the tool_call_id requirement must
	// not be applied to it.
	if req.Messages[2].ToolCallID != "" {
		t.Error("a function message should not need a tool_call_id")
	}
	// The tracker already agreed it continues a tool loop; the validator now does.
	if !inToolLoop(req.Messages) {
		t.Error("a trailing function result should read as an open tool loop")
	}
	// Genuinely unknown roles are still refused.
	if _, err := parseAndValidateChatRequest([]byte(`{"messages":[{"role":"wizard","content":"x"}]}`), 4096); err == nil {
		t.Error("an unknown role must still be rejected")
	}
}

// streamThen serves an SSE response of n content deltas and then either
// terminates it properly or drops the connection mid-body, which is what a
// worker dying mid-generation looks like from the router's side.
func streamThen(t *testing.T, deltas int, clean bool) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		for i := 0; i < deltas; i++ {
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tokens tokens \"}}]}\n\n")
			if fl != nil {
				fl.Flush()
			}
		}
		if clean {
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		panic(http.ErrAbortHandler) // close without the terminating chunk
	}))
	t.Cleanup(srv.Close)
	return srv
}

func runStream(t *testing.T, srv *httptest.Server) (*httptest.ResponseRecorder, *RequestLog) {
	t.Helper()
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "w", URL: srv.URL, Model: "m", TTLSeconds: 3600})
	r := &Router{
		cfg: &Config{BackendIdleTimeout: 5 * time.Second}, registry: reg,
		client: &http.Client{}, streamClient: &http.Client{},
	}
	logEntry := &RequestLog{}
	rec := httptest.NewRecorder()
	req := post("/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"stream":true}`, "")
	backend := reg.get("w")
	slot := make(chan struct{}, 1)
	// plan.auto is false here on purpose: this test exercises the PUMP, and
	// failover is a property of the dial. A plan with no candidates would decline
	// anyway, but saying so explicitly keeps the two concerns separate.
	r.dispatchStreaming(rec, req, &dispatch{
		backend: &backend, slot: &slot,
		body: []byte(`{"stream":true}`), raw: []byte(`{"stream":true}`),
		plan: &routePlan{}, job: nominalJob(), log: logEntry, output: io.Discard,
	}, "route", io.Discard, &sseStats{}, time.Now(), false)
	return rec, logEntry
}

// TestStreamingFailureBecomesAnSSEErrorEvent: once the preamble is out the
// status code is spent, so a mid-stream failure used to reach the client as
// nothing but a short answer — indistinguishable from the model stopping early.
// It has to be reported inside the stream instead.
func TestStreamingFailureBecomesAnSSEErrorEvent(t *testing.T) {
	rec, logEntry := runStream(t, streamThen(t, 3, false))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d; the preamble was already committed", rec.Code)
	}
	if logEntry.Error == "" {
		t.Error("a truncated stream must still be recorded as an error in the log")
	}
	body := rec.Body.String()
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Errorf("stream was not terminated: %q", body)
	}
	// The error frame has to be the OpenAI envelope, or a client reads it as
	// content.
	var found bool
	for _, line := range strings.Split(body, "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var chunk struct {
			Error *errorBody `json:"error"`
		}
		if json.Unmarshal([]byte(payload), &chunk) == nil && chunk.Error != nil {
			found = true
			if chunk.Error.Message == "" || chunk.Error.Type != "server_error" {
				t.Errorf("error event = %+v, want a server_error with a message", *chunk.Error)
			}
		}
	}
	if !found {
		t.Errorf("no error event in the stream: %q", body)
	}
}

// A clean stream must reach the client byte for byte — the guarantee the
// tool-call guard tests assert on the same path. Nothing is appended to a
// stream that ended properly.
func TestCleanStreamIsUnchanged(t *testing.T) {
	rec, logEntry := runStream(t, streamThen(t, 3, true))
	if logEntry.Error != "" {
		t.Errorf("clean stream recorded an error: %s", logEntry.Error)
	}
	want := strings.Repeat("data: {\"choices\":[{\"delta\":{\"content\":\"tokens tokens \"}}]}\n\n", 3) + "data: [DONE]\n\n"
	if got := rec.Body.String(); got != want {
		t.Errorf("clean stream was rewritten:\n got %q\nwant %q", got, want)
	}
}
