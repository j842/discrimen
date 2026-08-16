package router

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

func testAdapter(t *testing.T) *tierAdapter {
	t.Helper()
	return newTierAdapter(&Config{
		AdaptMaxBias: 0.30, AdaptLRUp: 0.04, AdaptLRDown: 0.01, AdaptBins: 10,
	}, filepath.Join(t.TempDir(), "tier_adapter.json"))
}

func TestAdapterBinOf(t *testing.T) {
	a := testAdapter(t)
	cases := map[float64]int{0.0: 0, 0.05: 0, 0.5: 5, 0.99: 9, 1.0: 9, 1.5: 9, -0.2: 0}
	for score, want := range cases {
		if got := a.binOf(score); got != want {
			t.Errorf("binOf(%.2f)=%d, want %d", score, got, want)
		}
	}
}

func TestAdapterAdjustNilSafe(t *testing.T) {
	var a *tierAdapter // nil
	if got := a.adjust(0.42); got != 0.42 {
		t.Fatalf("nil adapter must return the score unchanged, got %v", got)
	}
	a.observe(0.42, true) // must not panic
}

func TestAdapterObserveUpwardOnly(t *testing.T) {
	a := testAdapter(t)
	// Sustained inadequate responses lift the region, bounded at maxBias.
	for i := 0; i < 50; i++ {
		a.observe(0.5, true)
	}
	if got := a.adjust(0.5); got <= 0.5 || got > 0.5+a.maxBias+1e-9 {
		t.Fatalf("adjust after failures = %v, want in (0.5, 0.5+maxBias]", got)
	}
	if b := a.snapshot()[a.binOf(0.5)]; b != a.maxBias {
		t.Fatalf("bias should saturate at maxBias %.2f, got %v", a.maxBias, b)
	}
	// Clean responses decay it back to baseline (never below — upward-only).
	for i := 0; i < 200; i++ {
		a.observe(0.5, false)
	}
	if got := a.adjust(0.5); got != 0.5 {
		t.Fatalf("adjust after recovery = %v, want exactly 0.5 (no downward bias)", got)
	}
}

func TestAdapterPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tier_adapter.json")
	cfg := &Config{AdaptMaxBias: 0.30, AdaptLRUp: 0.10, AdaptLRDown: 0.01, AdaptBins: 5}
	a := newTierAdapter(cfg, path)
	a.observe(0.5, true)
	a.observe(0.5, true)
	a.save()

	reloaded := newTierAdapter(cfg, path)
	if reloaded.snapshot()[reloaded.binOf(0.5)] <= 0 {
		t.Fatalf("reloaded adapter lost its learned bias: %v", reloaded.snapshot())
	}
}

func TestParseRouteScore(t *testing.T) {
	if s, ok := parseRouteScore("route:d=0.82,q>=9"); !ok || s != 0.82 {
		t.Fatalf("parse difficulty route: got %v %v, want 0.82", s, ok)
	}
	for _, r := range []string{"route", "pinned", "debug", "completions", "route:d=bad"} {
		if _, ok := parseRouteScore(r); ok {
			t.Fatalf("%q should not parse a score", r)
		}
	}
}

func TestResponseInadequate(t *testing.T) {
	cases := []struct {
		name     string
		body     string
		streamed bool
		want     bool
	}{
		{"truncated", `{"choices":[{"finish_reason":"length","message":{"content":"Dun"}}]}`, false, true},
		{"truncated spaced", `{"choices":[{"finish_reason": "length"}]}`, false, true},
		{"empty buffered", `{"choices":[{"message":{"content":""},"finish_reason":"stop"}]}`, false, true},
		{"good buffered", `{"choices":[{"message":{"content":"hello there"},"finish_reason":"stop"}]}`, false, false},
		{"tool call", `{"choices":[{"message":{"tool_calls":[{"id":"x"}]},"finish_reason":"tool_calls"}]}`, false, false},
		{"empty stream", "data: {\"choices\":[{\"delta\":{}}]}\n\ndata: [DONE]\n", true, true},
		{"good stream", "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n", true, false},
	}
	for _, tc := range cases {
		if got := responseInadequate([]byte(tc.body), tc.streamed); got != tc.want {
			t.Errorf("%s: responseInadequate=%v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestHandleRouteFeedback(t *testing.T) {
	cfg := &Config{AdaptMaxBias: 0.30, AdaptLRUp: 0.10, AdaptLRDown: 0.01, AdaptBins: 10}
	a := newTierAdapter(cfg, filepath.Join(t.TempDir(), "a.json"))
	r := &Router{cfg: cfg, adapter: a}

	body := `{"route":"route:d=0.50,q>=7","verdict":"inadequate"}`
	w := httptest.NewRecorder()
	r.handleRouteFeedback(w, httptest.NewRequest(http.MethodPost, "/v1/route-feedback", strings.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("status %d, want 200; body=%s", w.Code, w.Body.String())
	}
	if a.snapshot()[a.binOf(0.50)] <= 0 {
		t.Fatalf("an 'inadequate' verdict should lift the region's bias: %v", a.snapshot())
	}

	// A request with neither route nor score is a 400.
	wBad := httptest.NewRecorder()
	r.handleRouteFeedback(wBad, httptest.NewRequest(http.MethodPost, "/v1/route-feedback", strings.NewReader(`{"verdict":"inadequate"}`)))
	if wBad.Code != http.StatusBadRequest {
		t.Fatalf("missing route/score should be 400, got %d", wBad.Code)
	}

	// Adapter disabled → 200 ignored, no panic.
	wOff := httptest.NewRecorder()
	(&Router{cfg: cfg}).handleRouteFeedback(wOff, httptest.NewRequest(http.MethodPost, "/v1/route-feedback", strings.NewReader(body)))
	if wOff.Code != http.StatusOK {
		t.Fatalf("disabled adapter should still 200, got %d", wOff.Code)
	}
}

// TestAdapterShiftsRoutingUp is the end-to-end behaviour: a region the cheap
// tier keeps failing on is lifted to a higher tier, then decays back once it
// recovers — all without any config change.
func TestAdapterShiftsRoutingUp(t *testing.T) {
	reg := newTestRegistry()
	readyBackend(reg, "tiny", 2, 200, 2)
	readyBackend(reg, "big", 10, 140, 6)
	cfg := &Config{
		DefaultMaxTokens: 4096, AutoDifficulty: true,
		AdaptMaxBias: 0.30, AdaptLRUp: 0.04, AdaptLRDown: 0.01, AdaptBins: 10,
	}
	r := &Router{
		cfg:        cfg,
		registry:   reg,
		classifier: testClassifierAuto(fakeEmbed),
		adapter:    newTierAdapter(cfg, filepath.Join(t.TempDir(), "tier_adapter.json")),
	}

	// Baseline: an easy prompt rides the cheap tier.
	if cands, route, _, _, _ := r.selectBackends(userReq("say hello"), 0); cands[0].ID != "tiny" || !strings.HasPrefix(route, "route:d=") {
		t.Fatalf("baseline easy → %v (%q), want tiny", ids(cands), route)
	}
	// The router keeps seeing this region come back inadequate → it lifts it.
	for i := 0; i < 12; i++ {
		r.adapter.observe(0.0, true)
	}
	if cands, _, _, _, _ := r.selectBackends(userReq("say hello"), 0); cands[0].ID != "big" {
		t.Fatalf("after sustained failures easy region should lift to big, got %v", ids(cands))
	}
	// Clean responses decay it back toward baseline.
	for i := 0; i < 100; i++ {
		r.adapter.observe(0.0, false)
	}
	if cands, _, _, _, _ := r.selectBackends(userReq("say hello"), 0); cands[0].ID != "tiny" {
		t.Fatalf("after recovery easy region should return to tiny, got %v", ids(cands))
	}
}
