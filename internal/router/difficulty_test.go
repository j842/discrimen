package router

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fakeEmbed returns deterministic 4-D one-hot vectors so the four seed
// centroids become the basis axes [simple, hard, reasoning, direct]. A query is
// scored on two independent axes from keywords, which lets the tests exercise
// the difficulty↔reasoning orthogonality (long-but-shallow, short-but-tricky)
// without a real embedding model.
func fakeEmbed(_ context.Context, texts []string) ([][]float64, error) {
	total := len(simpleSeeds) + len(hardSeeds) + len(reasoningSeeds) + len(directSeeds)
	out := make([][]float64, len(texts))
	if len(texts) == total { // the bootstrap batch: one-hot per group
		i := 0
		fill := func(n int, v []float64) {
			for k := 0; k < n; k++ {
				out[i] = v
				i++
			}
		}
		fill(len(simpleSeeds), []float64{1, 0, 0, 0})
		fill(len(hardSeeds), []float64{0, 1, 0, 0})
		fill(len(reasoningSeeds), []float64{0, 0, 1, 0})
		fill(len(directSeeds), []float64{0, 0, 0, 1})
		return out, nil
	}
	for i, t := range texts {
		l := strings.ToLower(t)
		hard := containsAnySubstr(l, "prove", "design", "hard", "debug", "summarise", "summarize", "essay", "architect")
		reason := containsAnySubstr(l, "prove", "reason", "step", "puzzle", "debug", "why", "deduce", "logic")
		v := []float64{0, 0, 0, 0}
		if hard {
			v[1] = 1 // hard axis
		} else {
			v[0] = 1 // simple axis
		}
		if reason {
			v[2] = 1 // reasoning axis
		} else {
			v[3] = 1 // direct axis
		}
		out[i] = v
	}
	return out, nil
}

func containsAnySubstr(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func testClassifier(embed func(context.Context, []string) ([][]float64, error)) *difficultyClassifier {
	return newDifficultyClassifier(&Config{
		DifficultyBands:     defaultDifficultyBands,
		DifficultyTemp:      0.10,
		ReasoningThreshold:  0.35,
		DifficultyTimeout:   time.Second,
		DifficultyCacheSize: 64,
		DifficultyMaxChars:  4000,
	}, embed)
}

// testClassifierAuto builds a classifier with no explicit bands → automatic
// fleet-derived tiers (the production default).
func testClassifierAuto(embed func(context.Context, []string) ([][]float64, error)) *difficultyClassifier {
	return newDifficultyClassifier(&Config{
		DifficultyTemp:      0.10,
		ReasoningThreshold:  0.35,
		DifficultyTimeout:   time.Second,
		DifficultyCacheSize: 64,
		DifficultyMaxChars:  4000,
	}, embed)
}

func userReq(content string) *ChatRequest {
	return &ChatRequest{
		MaxTokens: 256,
		Messages:  []Message{{Role: "user", Content: content}},
	}
}

func TestParseDifficultyBands(t *testing.T) {
	bands := parseDifficultyBands("0.70:7, 0.40:2 ,1.0:9")
	if len(bands) != 3 {
		t.Fatalf("want 3 bands, got %d: %+v", len(bands), bands)
	}
	if bands[0].upTo != 0.40 || bands[0].quality != 2 || bands[2].upTo != 1.0 {
		t.Fatalf("bands not sorted/parsed correctly: %+v", bands)
	}
	if got := parseDifficultyBands("junk,0.5:x,,0.5:3"); len(got) != 1 || got[0].quality != 3 {
		t.Fatalf("garbage handling wrong: %+v", got)
	}
}

func TestBandQuality(t *testing.T) {
	c := testClassifier(fakeEmbed)
	cases := []struct {
		score float64
		want  int
	}{
		{0.0, 2}, {0.40, 2}, {0.41, 7}, {0.70, 7}, {0.71, 9}, {1.0, 9}, {1.5, 9},
	}
	for _, tc := range cases {
		if got := c.bandQuality(tc.score); got != tc.want {
			t.Errorf("bandQuality(%.2f)=%d, want %d", tc.score, got, tc.want)
		}
	}
}

func TestVectorMath(t *testing.T) {
	n := normalize([]float64{3, 4})
	if math.Abs(n[0]-0.6) > 1e-9 || math.Abs(n[1]-0.8) > 1e-9 {
		t.Fatalf("normalize wrong: %v", n)
	}
	if normalize([]float64{0, 0})[0] != 0 {
		t.Fatal("normalize of zero vector must not divide by zero")
	}
	if d := dot([]float64{1, 2, 3}, []float64{4, 5}); d != 1*4+2*5 {
		t.Fatalf("dot should use the shorter length: got %v", d)
	}
	cen := centroid([][]float64{{2, 0}, {0, 2}})
	if math.Abs(cen[0]-cen[1]) > 1e-9 || math.Abs(cen[0]-math.Sqrt2/2) > 1e-9 {
		t.Fatalf("centroid should be the normalised mean direction: %v", cen)
	}
	if clamp01(-1) != 0 || clamp01(2) != 1 || clamp01(0.5) != 0.5 {
		t.Fatal("clamp01 bounds wrong")
	}
}

func TestClassifierScoreAndCache(t *testing.T) {
	calls := 0
	embed := func(ctx context.Context, texts []string) ([][]float64, error) {
		calls++
		return fakeEmbed(ctx, texts)
	}
	c := testClassifier(embed)

	hardQ, hardScore, ok := c.targetQuality(userReq("prove the four colour theorem"))
	if !ok || hardScore <= 0.5 || hardQ != 9 {
		t.Fatalf("hard prompt: ok=%v score=%.3f q=%d (want ok, >0.5, 9)", ok, hardScore, hardQ)
	}
	easyQ, easyScore, ok := c.targetQuality(userReq("say hello"))
	if !ok || easyScore >= 0.5 || easyQ != 2 {
		t.Fatalf("easy prompt: ok=%v score=%.3f q=%d (want ok, <0.5, 2)", ok, easyScore, easyQ)
	}
	if calls != 3 {
		t.Fatalf("embedder called %d times, want 3 (1 bootstrap + 2 prompts)", calls)
	}
	// A seen prompt is served from cache (no new embed call); both axes cached.
	if _, _, ok := c.targetQuality(userReq("say hello")); !ok || calls != 3 {
		t.Fatalf("cache miss: ok=%v calls=%d (want ok, still 3)", ok, calls)
	}
}

func TestClassifierFallbackOnEmbedError(t *testing.T) {
	c := testClassifier(func(context.Context, []string) ([][]float64, error) {
		return nil, errors.New("embeddings down")
	})
	if _, _, ok := c.targetQuality(userReq("anything")); ok {
		t.Fatal("classification must report ok=false when the embedder fails")
	}
	if _, ok := c.classify(&ChatRequest{Messages: []Message{{Role: "user", Content: ""}}}); ok {
		t.Fatal("empty prompt must not classify")
	}
}

// TestClassifierEmbedFailureCooldown: once ready, a live-embed failure must
// trip a cooldown — subsequent classifies (including the same request's second
// reasoning pass) return ok=false WITHOUT calling the embedder, so a wedged
// worker costs one timeout per window instead of two per request.
func TestClassifierEmbedFailureCooldown(t *testing.T) {
	fail := false
	calls := 0
	embed := func(ctx context.Context, texts []string) ([][]float64, error) {
		calls++
		if fail {
			return nil, errors.New("embeddings wedged")
		}
		return fakeEmbed(ctx, texts)
	}
	c := testClassifier(embed)

	if _, ok := c.classify(userReq("warm bootstrap")); !ok {
		t.Fatal("bootstrap classify should succeed")
	}

	fail = true
	if _, ok := c.classify(userReq("first failing prompt")); ok {
		t.Fatal("failing embed must classify ok=false")
	}
	got := calls
	if _, ok := c.classify(userReq("second prompt during cooldown")); ok {
		t.Fatal("classify during the failure cooldown must be ok=false")
	}
	if calls != got {
		t.Fatalf("embedder invoked during cooldown: %d → %d calls", got, calls)
	}

	// Cooldown elapsed but the worker is still failing → exactly one retry,
	// then the cooldown re-trips.
	c.mu.Lock()
	c.lastEmbedFail = time.Now().Add(-embedFailCooldown)
	c.mu.Unlock()
	if _, ok := c.classify(userReq("retry while failing")); ok {
		t.Fatal("retry against a failing embedder must stay ok=false")
	}
	if calls != got+1 {
		t.Fatalf("want exactly one retry embed call, calls %d → %d", got, calls)
	}
	if _, ok := c.classify(userReq("cooldown re-tripped")); ok || calls != got+1 {
		t.Fatalf("cooldown must re-trip after a failed retry: calls=%d", calls)
	}

	// After the cooldown a successful embed resumes classification and clears
	// the cooldown entirely.
	fail = false
	c.mu.Lock()
	c.lastEmbedFail = time.Now().Add(-embedFailCooldown)
	c.mu.Unlock()
	if _, ok := c.classify(userReq("worker recovered")); !ok {
		t.Fatal("classification must resume once the embedder recovers")
	}
	c.mu.Lock()
	cleared := c.lastEmbedFail.IsZero()
	c.mu.Unlock()
	if !cleared {
		t.Fatal("a successful embed must clear the failure cooldown")
	}
	if _, ok := c.classify(userReq("steady state")); !ok {
		t.Fatal("post-recovery classifies must not hit the cooldown")
	}
}

// TestClassifierRebootstrapOnDimChange: swapping the embeddings worker to a
// model with a different vector dimension must not score prompts against the
// old centroid space (dot over a truncated prefix ≈ 0.5 for everything) — the
// classifier goes unavailable, drops its cache, and re-bootstraps.
func TestClassifierRebootstrapOnDimChange(t *testing.T) {
	wide := false
	embed := func(ctx context.Context, texts []string) ([][]float64, error) {
		vecs, err := fakeEmbed(ctx, texts)
		if err != nil || !wide {
			return vecs, err
		}
		for i, v := range vecs {
			vecs[i] = append(v, 0, 0, 0, 0) // same directions, dim 4 → 8
		}
		return vecs, nil
	}
	c := testClassifier(embed)

	if _, ok := c.classify(userReq("prove this hard theorem")); !ok {
		t.Fatal("bootstrap classify should succeed")
	}

	wide = true // the worker now serves a different embedding model
	if _, ok := c.classify(userReq("say hello there")); ok {
		t.Fatal("dimension mismatch must return ok=false, not a truncated-prefix score")
	}
	if _, ok := c.cache.get("prove this hard theorem"); ok {
		t.Fatal("cached scores from the old centroid space must be dropped")
	}
	// ready was reset; re-bootstrap waits out the bootstrap cooldown first.
	if _, ok := c.classify(userReq("still cooling down")); ok {
		t.Fatal("re-bootstrap must respect the bootstrap cooldown")
	}

	c.mu.Lock()
	c.lastAttempt = time.Now().Add(-bootstrapCooldown)
	c.mu.Unlock()
	cl, ok := c.classify(userReq("prove this hard theorem"))
	if !ok || cl.difficulty <= 0.5 {
		t.Fatalf("re-bootstrapped classification should work in the new space: ok=%v d=%.3f", ok, cl.difficulty)
	}
}

// TestReasoningClassification exercises the reasoning axis and the key property
// that it is orthogonal to difficulty.
func TestReasoningClassification(t *testing.T) {
	c := testClassifier(fakeEmbed)

	cl, ok := c.classify(userReq("prove this theorem step by step"))
	if !ok || cl.reasoning <= 0.5 || !c.wantThinking(cl.reasoning) {
		t.Fatalf("reasoning prompt: ok=%v reasoning=%.3f (want >0.5 + thinking)", ok, cl.reasoning)
	}
	cl, ok = c.classify(userReq("translate hello into French"))
	if !ok || cl.reasoning >= 0.5 || c.wantThinking(cl.reasoning) {
		t.Fatalf("direct prompt: ok=%v reasoning=%.3f (want <0.5 + no thinking)", ok, cl.reasoning)
	}
	// Long-but-shallow → high tier BUT no reasoning.
	cl, _ = c.classify(userReq("summarise this very long quarterly report in detail"))
	if c.bandQuality(cl.difficulty) < 7 || c.wantThinking(cl.reasoning) {
		t.Fatalf("long-shallow: want high tier + no thinking, got q=%d reasoning=%.3f", c.bandQuality(cl.difficulty), cl.reasoning)
	}
	// Short-but-tricky → low tier BUT needs reasoning.
	cl, _ = c.classify(userReq("solve this logic puzzle"))
	if c.bandQuality(cl.difficulty) > 2 || !c.wantThinking(cl.reasoning) {
		t.Fatalf("short-tricky: want low tier + thinking, got q=%d reasoning=%.3f", c.bandQuality(cl.difficulty), cl.reasoning)
	}
}

func TestClassifyTextExtraction(t *testing.T) {
	got := classifyText([]Message{
		{Role: "system", Content: "be terse"},
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "ok"},
		{Role: "user", Content: "second"},
	}, 4000)
	if got != "second" {
		t.Fatalf("want last user message, got %q", got)
	}
	parts := []any{
		map[string]any{"type": "text", "text": "describe"},
		map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:..."}},
		map[string]any{"type": "text", "text": "this"},
	}
	if got := classifyText([]Message{{Role: "user", Content: parts}}, 4000); got != "describe this" {
		t.Fatalf("multimodal flatten wrong: %q", got)
	}
	long := strings.Repeat("é", 50) // 2 bytes each
	if cut := classifyText([]Message{{Role: "user", Content: long}}, 5); !isValidUTF8Cut(cut) || len(cut) > 5 {
		t.Fatalf("truncation split a rune or exceeded cap: %q (len %d)", cut, len(cut))
	}
}

func TestClassifyInput(t *testing.T) {
	// The agent wraps the genuine question in a big runtime-context blob inside
	// the user turn. With classify_text set, selection ignores the blob and
	// scores the real question — same text the bare prompt would yield.
	wrapped := "Current date/time: ...\n\n<reference_context>\nmemories + summaries " +
		strings.Repeat("blah ", 400) + "\n</reference_context>\n\nThe user's message follows:\n" +
		"whats the weather tomorrow"
	req := &ChatRequest{
		Messages:     []Message{{Role: "system", Content: "be terse"}, {Role: "user", Content: wrapped}},
		ClassifyText: "whats the weather tomorrow",
	}
	if got := classifyInput(req, 4000); got != "whats the weather tomorrow" {
		t.Fatalf("classify_text hint not used: got %q", got)
	}

	// No hint ⇒ fall back to the last user turn (the wrapped blob), unchanged.
	if got := classifyInput(&ChatRequest{Messages: req.Messages}, 4000); got != wrapped {
		t.Fatalf("fallback should return the last user turn, got %d chars", len(got))
	}

	// Whitespace-only hint ⇒ treated as absent, falls back.
	if got := classifyInput(&ChatRequest{Messages: req.Messages, ClassifyText: "   \n"}, 4000); got != wrapped {
		t.Fatalf("blank hint should fall back to messages, got %q", got)
	}

	// Hint longer than the cap is truncated at a rune boundary.
	long := strings.Repeat("é", 50) // 2 bytes each
	if cut := classifyInput(&ChatRequest{ClassifyText: long}, 5); !isValidUTF8Cut(cut) || len(cut) > 5 {
		t.Fatalf("hint truncation split a rune or exceeded cap: %q (len %d)", cut, len(cut))
	}
}

func isValidUTF8Cut(s string) bool {
	for _, r := range s {
		if r == '�' {
			return false
		}
	}
	return true
}

func TestLiveTPSAndExpectedLatency(t *testing.T) {
	b := &Backend{}
	b.BaselineTPS = 10
	if liveTPS(b) != 10 {
		t.Fatal("should fall back to baseline")
	}
	b.Certification.TokensPerSec = 20
	if liveTPS(b) != 20 {
		t.Fatal("certified should beat baseline")
	}
	b.ObservedTPS = 30
	if liveTPS(b) != 30 {
		t.Fatal("observed should win")
	}

	fast := &Backend{ObservedTPS: 100, BackendRegistration: BackendRegistration{MaxConcurrency: 4}}
	slow := &Backend{ObservedTPS: 10, BackendRegistration: BackendRegistration{MaxConcurrency: 4}}
	if expectedLatency(fast, nominalJob()) >= expectedLatency(slow, nominalJob()) {
		t.Fatal("faster backend should have lower expected latency")
	}
	queued := &Backend{ObservedTPS: 100, ActiveRequests: 8, BackendRegistration: BackendRegistration{MaxConcurrency: 4}}
	if expectedLatency(queued, nominalJob()) <= expectedLatency(fast, nominalJob()) {
		t.Fatal("a queued backend should have higher expected latency than the same idle one")
	}
}

func mkBackend(id string, quality, tps, maxConc, active int) *Backend {
	return &Backend{
		BackendRegistration: BackendRegistration{
			ID: id, Model: "default", Quality: quality,
			BaselineTPS: float64(tps), MaxConcurrency: maxConc,
		},
		ActiveRequests: active,
		ObservedTPS:    float64(tps),
	}
}

func TestRankByDifficulty(t *testing.T) {
	tiny := mkBackend("tiny", 2, 60, 2, 0)
	mid := mkBackend("mid", 7, 25, 2, 0)
	gem := mkBackend("gem", 7, 6, 4, 0)
	big := mkBackend("big", 10, 140, 6, 0)

	// Easy: every backend clears q>=2, so the one that FINISHES SOONEST wins —
	// the fastest (big), not the cheapest. (Completion-time ranking.)
	if got := rankByDifficulty([]*Backend{tiny, mid, gem, big}, 2, nominalJob()); got[0].ID != "big" {
		t.Fatalf("easy idle: want fastest sufficient (big), got %s", got[0].ID)
	}
	// Hard: only big clears q>=9.
	if got := rankByDifficulty([]*Backend{tiny, mid, gem, big}, 9, nominalJob()); got[0].ID != "big" {
		t.Fatalf("hard: want big, got %s", got[0].ID)
	}
	// Among equal-quality (q7) candidates the faster one wins.
	if got := rankByDifficulty([]*Backend{gem, mid}, 7, nominalJob()); got[0].ID != "mid" {
		t.Fatalf("q7 tie: want faster (mid), got %s", got[0].ID)
	}
	// Spill: when the fast big is full, the fastest backend with a free slot wins.
	fullBig := mkBackend("big", 10, 140, 6, 6) // active==cap → full
	if got := rankByDifficulty([]*Backend{fullBig, mid, tiny}, 2, nominalJob()); got[0].ID != "tiny" {
		t.Fatalf("big full: want fastest free slot (tiny), got %s", got[0].ID)
	}
}

func readyBackend(reg *Registry, id string, quality, tps, maxConc int) {
	reg.upsert(BackendRegistration{
		ID: id, URL: "http://" + id, Model: "default", Quality: quality,
		BaselineTPS: float64(tps), MaxConcurrency: maxConc, TTLSeconds: 3600,
		Features: []string{"chat"},
	})
	reg.finishCertification(id, true, map[string]Check{}, float64(tps), 10, "")
}

// readyThinkingBackend registers a ready chat worker with an explicit thinking
// capability — used to prove the reasoning axis steers SELECTION, not just the
// body patch.
func readyThinkingBackend(reg *Registry, id string, quality, tps, maxConc int, thinking bool) {
	reg.upsert(BackendRegistration{
		ID: id, URL: "http://" + id, Model: "default", Quality: quality,
		BaselineTPS: float64(tps), MaxConcurrency: maxConc, TTLSeconds: 3600,
		Thinking: thinking, Features: []string{"chat"},
	})
	reg.finishCertification(id, true, map[string]Check{}, float64(tps), 10, "")
}

func TestSelectBackendsAutoDifficulty(t *testing.T) {
	reg := newTestRegistry()
	readyBackend(reg, "tiny", 2, 200, 2)
	readyBackend(reg, "big", 10, 140, 6)
	cfg := &Config{
		DefaultMaxTokens: 4096, AutoDifficulty: true,
		DifficultyBands: defaultDifficultyBands, DifficultyTemp: 0.10,
		DifficultyTimeout: time.Second, DifficultyCacheSize: 16, DifficultyMaxChars: 4000,
	}
	r := &Router{cfg: cfg, registry: reg, classifier: testClassifier(fakeEmbed)}

	cands, route, _, _, err := r.selectBackends(userReq("say hello"), 0)
	if err != nil || cands[0].ID != "tiny" {
		t.Fatalf("easy auto-route: got %v err=%v, want tiny first", ids(cands), err)
	}
	if !strings.HasPrefix(route, "route:d=") {
		t.Fatalf("difficulty route hint missing: %q", route)
	}
	if cands, _, _, _, _ := r.selectBackends(userReq("prove a hard theorem"), 0); cands[0].ID != "big" {
		t.Fatalf("hard auto-route: got %v, want big first", ids(cands))
	}

	// A capability hint (thinking) must NOT suppress difficulty routing — there are
	// no quality/speed overrides any more, so every request auto-tiers.
	req2 := userReq("say hello")
	req2.Requirements = &Requirements{Thinking: "off"}
	cands, route, _, _, _ = r.selectBackends(req2, 0)
	if cands[0].ID != "tiny" || !strings.HasPrefix(route, "route:d=") {
		t.Fatalf("orthogonal thinking hint must keep auto-difficulty: first=%s route=%q", cands[0].ID, route)
	}

	// AutoDifficulty disabled → default ranking regardless of prompt.
	r.cfg.AutoDifficulty = false
	if cands, route, _, _, _ := r.selectBackends(userReq("say hello"), 0); cands[0].ID != "big" || route != "route" {
		t.Fatalf("disabled auto-difficulty should use default ranking: first=%s route=%q", cands[0].ID, route)
	}
}

// TestAutoReasoningGatesSelection is the core of fix #1: a LOW-difficulty but
// HIGH-reasoning prompt (short-but-tricky) must route to a thinking-capable
// worker when one exists — NOT to the fastest/cheapest non-thinking worker that
// would win on completion-time alone. The reasoning axis steers SELECTION, so the
// enable_thinking we patch lands on a worker that can act on it. When no thinking
// worker survives the hard filters, auto-thinking falls back to the full set
// rather than 503ing (best-effort steering, never a demand).
func TestAutoReasoningGatesSelection(t *testing.T) {
	cfg := &Config{
		DefaultMaxTokens: 4096, AutoDifficulty: true, AutoThinking: true,
	}
	newRouter := func(reg *Registry) *Router {
		return &Router{cfg: cfg, registry: reg, classifier: testClassifierAuto(fakeEmbed)}
	}

	// "solve this logic puzzle": low difficulty (no hard keyword) but high
	// reasoning ("logic"/"puzzle") under fakeEmbed.
	trickyLowDiff := userReq("solve this logic puzzle")

	// A thinking-capable worker exists alongside a faster non-thinking one. The
	// fast worker would win the completion-time race outright; the auto-reasoning
	// soft preference must pick the thinking-capable one instead.
	reg := newTestRegistry()
	readyThinkingBackend(reg, "fast-nothink", 2, 200, 4, false)
	readyThinkingBackend(reg, "slow-think", 2, 20, 4, true)
	r := newRouter(reg)

	cands, route, _, _, err := r.selectBackends(trickyLowDiff, 0)
	if err != nil {
		t.Fatalf("auto-reasoning select errored: %v", err)
	}
	if cands[0].ID != "slow-think" {
		t.Fatalf("low-difficulty + high-reasoning must prefer the thinking-capable worker, got %v (route %q)", ids(cands), route)
	}
	// Every surviving candidate is thinking-capable (the non-thinking worker was
	// filtered out, not merely out-ranked).
	for _, b := range cands {
		if !b.Thinking {
			t.Fatalf("auto soft-think filter leaked a non-thinking worker: %v", ids(cands))
		}
	}
	if !strings.HasPrefix(route, "route:d=") {
		t.Fatalf("difficulty route hint missing: %q", route)
	}

	// Fallback: NO thinking-capable worker. Auto-thinking must NOT 503 — it falls
	// back to the full candidate set and still serves the request.
	regNoThink := newTestRegistry()
	readyThinkingBackend(regNoThink, "fast-nothink", 2, 200, 4, false)
	readyThinkingBackend(regNoThink, "mid-nothink", 5, 80, 4, false)
	cands, _, _, _, err = newRouter(regNoThink).selectBackends(trickyLowDiff, 0)
	if err != nil {
		t.Fatalf("auto-thinking must never 503 when no thinking worker exists: %v", err)
	}
	if len(cands) != 2 {
		t.Fatalf("auto fallback should keep all candidates, got %v", ids(cands))
	}

	// Contrast — an EXPLICIT thinking:"on" keeps the HARD filter and legitimately
	// 503s when no thinking-capable worker exists (a user demand, not steering).
	hardOn := userReq("solve this logic puzzle")
	hardOn.Requirements = &Requirements{Thinking: "on"}
	if _, _, _, _, err := newRouter(regNoThink).selectBackends(hardOn, 0); err == nil {
		t.Fatal("explicit thinking:\"on\" with no thinking worker must 503 (hard filter)")
	}
}

func TestAutoTargetQuality(t *testing.T) {
	// Realistic measured qualities (benchmark percentages), not tiers.
	fleet := []*Backend{mkBackend("a", 40, 0, 0, 0), mkBackend("b", 71, 0, 0, 0), mkBackend("c", 82, 0, 0, 0)}
	if q := autoTargetQuality(fleet, 0.0); q != 0 {
		t.Fatalf("score 0 → %d, want 0 (no bar)", q)
	}
	if q := autoTargetQuality(fleet, 0.65); q != 65 {
		t.Fatalf("score .65 → %d, want 65 (absolute scale)", q)
	}
	if q := autoTargetQuality(fleet, 1.0); q != 82 {
		t.Fatalf("score 1 → %d, want 82 (clamped to best available)", q)
	}
	if q := autoTargetQuality([]*Backend{mkBackend("x", 50, 0, 0, 0)}, 0.9); q != 50 {
		t.Fatalf("single worker → %d, want 50 (clamped)", q)
	}

	// The regression this mapping exists to prevent: registering a slower,
	// higher-quality worker must not raise the bar for a question that the
	// existing fleet was already clearing — the bar belongs to the question.
	// Under the old fleet-range mapping this went from 67 to 74 the moment
	// the 93 registered, silently excluding the 82.
	withGenius := append(append([]*Backend{}, fleet...), mkBackend("genius", 93, 0, 0, 0))
	if before, after := autoTargetQuality(fleet, 0.65), autoTargetQuality(withGenius, 0.65); before != after {
		t.Fatalf("adding a high-quality worker moved the bar: %d → %d", before, after)
	}
}

// TestAutoBandsSelection: with no configured bands the router derives tiers from
// the registered fleet — no ROUTER_DIFFICULTY_QUALITY_BANDS needed.
func TestAutoBandsSelection(t *testing.T) {
	reg := newTestRegistry()
	readyBackend(reg, "tiny", 2, 200, 2)
	readyBackend(reg, "big", 10, 140, 6)
	r := &Router{
		cfg:        &Config{DefaultMaxTokens: 4096, AutoDifficulty: true},
		registry:   reg,
		classifier: testClassifierAuto(fakeEmbed),
	}
	cands, route, _, _, err := r.selectBackends(userReq("say hello"), 0)
	if err != nil || cands[0].ID != "tiny" || !strings.HasPrefix(route, "route:d=") {
		t.Fatalf("easy auto-band: got %v route=%q err=%v, want tiny", ids(cands), route, err)
	}
	if cands, _, _, _, _ := r.selectBackends(userReq("prove a hard theorem"), 0); cands[0].ID != "big" {
		t.Fatalf("hard auto-band: got %v, want big", ids(cands))
	}
}

func TestNormalizeThinking(t *testing.T) {
	cases := map[string]string{
		"on": "on", "On": "on", "required": "on", "true": "on", "optional": "on",
		"off": "off", "OFF": "off", "disabled": "off", "false": "off", "none": "off",
		"": "auto", "auto": "auto", "banana": "auto", "  off  ": "off",
	}
	for in, want := range cases {
		if got := normalizeThinking(in); got != want {
			t.Fatalf("normalizeThinking(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestClientSetKwargThinking(t *testing.T) {
	if clientSetKwargThinking(userReq("hi")) {
		t.Fatal("no thinking signal should be false")
	}
	r1 := userReq("hi")
	r1.Requirements = &Requirements{Thinking: "off"}
	if clientSetKwargThinking(r1) {
		t.Fatal("requirements.thinking is resolved in resolveThinking, not here")
	}
	r2 := userReq("hi")
	r2.ChatTemplateKwargs = map[string]any{"enable_thinking": false}
	if !clientSetKwargThinking(r2) {
		t.Fatal("chat_template_kwargs.enable_thinking should count")
	}
	r3 := userReq("hi")
	r3.ChatTemplateKwargs = map[string]any{"preserve_thinking": true}
	if clientSetKwargThinking(r3) {
		t.Fatal("unrelated kwargs should not count as a thinking choice")
	}
}

// TestPatchForwardedBody covers the single-pass body patch: max_tokens and
// chat_template_kwargs.enable_thinking written together in one unmarshal/marshal.
func TestPatchForwardedBody(t *testing.T) {
	// max_tokens only (no thinking patch): filled in when absent, left when present.
	out := patchForwardedBody([]byte(`{"messages":[]}`), 4096, 0, thinkingResolution{}, "")
	if !strings.Contains(string(out), `"max_tokens":4096`) {
		t.Fatalf("max_tokens not filled in: %s", out)
	}
	out = patchForwardedBody([]byte(`{"max_tokens":7}`), 4096, 0, thinkingResolution{}, "")
	if strings.Contains(string(out), "4096") || !strings.Contains(string(out), `"max_tokens":7`) {
		t.Fatalf("present max_tokens must not be overwritten: %s", out)
	}
	// defaultMaxTokens<=0 → don't touch max_tokens.
	if out := patchForwardedBody([]byte(`{"messages":[]}`), 0, 0, thinkingResolution{}, ""); strings.Contains(string(out), "max_tokens") {
		t.Fatalf("max_tokens added with no default: %s", out)
	}

	// Both in one pass: max_tokens + enable_thinking, preserving sibling kwargs.
	out = patchForwardedBody([]byte(`{"chat_template_kwargs":{"preserve_thinking":true}}`), 256, 0, thinkingResolution{patch: true, enable: true}, "")
	if !strings.Contains(string(out), `"max_tokens":256`) ||
		!strings.Contains(string(out), `"preserve_thinking":true`) ||
		!strings.Contains(string(out), `"enable_thinking":true`) {
		t.Fatalf("single-pass max_tokens + enable_thinking merge failed: %s", out)
	}

	// Escape hatch: an explicit enable_thinking is never overwritten — but a
	// max_tokens fill-in in the same pass still applies.
	out = patchForwardedBody([]byte(`{"chat_template_kwargs":{"enable_thinking":true}}`), 256, 0, thinkingResolution{patch: true, enable: false}, "")
	if strings.Contains(string(out), `"enable_thinking":false`) {
		t.Fatalf("must not override an explicit enable_thinking: %s", out)
	}
	if !strings.Contains(string(out), `"max_tokens":256`) {
		t.Fatalf("max_tokens fill-in must still apply alongside the kwargs escape hatch: %s", out)
	}

	// patch=false → kwargs untouched entirely.
	if out := patchForwardedBody([]byte(`{"messages":[]}`), 0, 0, thinkingResolution{patch: false, enable: true}, ""); strings.Contains(string(out), "enable_thinking") {
		t.Fatalf("patch=false must not write enable_thinking: %s", out)
	}

	// Invalid JSON passes through unchanged (best-effort).
	if out := patchForwardedBody([]byte(`not json`), 256, 0, thinkingResolution{patch: true, enable: true}, ""); string(out) != "not json" {
		t.Fatal("invalid body should pass through unchanged")
	}

	// servedModel rewrites the client's spelling to the worker's advertised id
	// (vLLM 404s anything else), and is a no-op when they already agree or when
	// the worker's id is unknown.
	out = patchForwardedBody([]byte(`{"model":"qwen3.8","messages":[]}`), 0, 0, thinkingResolution{}, "default")
	if !strings.Contains(string(out), `"model":"default"`) {
		t.Fatalf("model not rewritten to the served id: %s", out)
	}
	in := []byte(`{"model":"default","messages":[]}`)
	if out := patchForwardedBody(in, 0, 0, thinkingResolution{}, "default"); string(out) != string(in) {
		t.Fatalf("matching model must pass through byte-identical: %s", out)
	}
	in = []byte(`{"model":"qwen3.8","messages":[]}`)
	if out := patchForwardedBody(in, 0, 0, thinkingResolution{}, ""); string(out) != string(in) {
		t.Fatalf("unknown served id must leave the model untouched: %s", out)
	}

	// null kwargs is replaced, not panicked on.
	out = patchForwardedBody([]byte(`{"chat_template_kwargs":null}`), 0, 0, thinkingResolution{patch: true, enable: false}, "")
	if !strings.Contains(string(out), `"enable_thinking":false`) {
		t.Fatalf("null kwargs should be replaced, not panic: %s", out)
	}
}

// TestPatchForwardedBodyStripsRouterFields: the three fields the router reads
// and no endpoint understands must never be forwarded — including on the path
// where nothing else needs patching, which used to return the body untouched.
func TestPatchForwardedBodyStripsRouterFields(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"requirements", `{"messages":[],"requirements":{"thinking":"on"}}`},
		{"classify_text", `{"messages":[],"classify_text":"what the user really asked"}`},
		{"deadline_ms", `{"messages":[],"deadline_ms":30000}`},
		{"all three", `{"messages":[],"requirements":{},"classify_text":"x","deadline_ms":1}`},
		// A null value still leaves the key on the wire, which is what a strict
		// endpoint rejects.
		{"null requirements", `{"messages":[],"requirements":null}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// The no-other-work path: every patch input is inert.
			out := string(patchForwardedBody([]byte(tc.body), 0, 0, thinkingResolution{}, ""))
			for _, field := range routerOnlyFields {
				if strings.Contains(out, `"`+field+`"`) {
					t.Errorf("%s survived the strip: %s", field, out)
				}
			}
			if !strings.Contains(out, `"messages"`) {
				t.Errorf("strip removed more than it should: %s", out)
			}
			// And alongside the other patches, in the same pass.
			out = string(patchForwardedBody([]byte(tc.body), 4096, 0, thinkingResolution{patch: true, enable: true}, "served"))
			for _, field := range routerOnlyFields {
				if strings.Contains(out, `"`+field+`"`) {
					t.Errorf("%s survived alongside the other patches: %s", field, out)
				}
			}
			if !strings.Contains(out, `"max_tokens":4096`) || !strings.Contains(out, `"model":"served"`) {
				t.Errorf("the other patches stopped applying: %s", out)
			}
		})
	}

	// A body carrying none of them is still returned byte-identical — the fast
	// exit must survive the strip. The prompt text quotes one of the field names
	// on purpose: the cheap pre-scan matches, the key lookup does not, and
	// nothing may be rewritten on the strength of the scan alone.
	clean := []byte(`{"model":"m","messages":[{"role":"user","content":"what does \"deadline_ms\" do?"}]}`)
	if out := patchForwardedBody(clean, 0, 0, thinkingResolution{}, ""); string(out) != string(clean) {
		t.Errorf("clean body was rewritten: %s", out)
	}
}

// resolveAndPatchThinking mirrors the production hot path for thinking only
// (selectBackends classifies once, then proxyToBackend resolves + single-pass
// patches): classify on a normal route, pass the classification (nil on
// pinned/debug — exactly as the handlers thread it), resolve, patch. max_tokens
// is exercised by TestPatchForwardedBody, so 0 here keeps it untouched.
func (r *Router) resolveAndPatchThinking(body []byte, chatReq *ChatRequest, route string) []byte {
	var cl *classification
	if r.classifier != nil && (strings.HasPrefix(route, "route") || strings.HasPrefix(route, "model")) {
		if c, ok := r.classifier.classify(chatReq); ok {
			cl = &c
		}
	}
	return patchForwardedBody(body, 0, 0, r.resolveThinking(chatReq, route, cl), "")
}

// TestApplyThinking exercises the resolved thinking decision end-to-end through
// the single-pass patch (resolveThinking + patchForwardedBody), via the same
// classify-then-resolve flow production uses.
func TestApplyThinking(t *testing.T) {
	r := &Router{cfg: &Config{AutoThinking: true}, classifier: testClassifier(fakeEmbed)}
	body := func(prompt string) []byte {
		b, _ := json.Marshal(map[string]any{
			"messages": []map[string]string{{"role": "user", "content": prompt}},
		})
		return b
	}

	out := r.resolveAndPatchThinking(body("prove this step by step"), userReq("prove this step by step"), "route")
	if !strings.Contains(string(out), `"enable_thinking":true`) {
		t.Fatalf("reasoning prompt should enable thinking: %s", out)
	}
	out = r.resolveAndPatchThinking(body("translate hello"), userReq("translate hello"), "route:d=0.10,q>=2")
	if !strings.Contains(string(out), `"enable_thinking":false`) {
		t.Fatalf("direct prompt should disable thinking: %s", out)
	}
	if out := r.resolveAndPatchThinking(body("prove this"), userReq("prove this"), "pinned"); strings.Contains(string(out), "enable_thinking") {
		t.Fatal("pinned route must not auto-inject thinking")
	}

	// requirements.thinking is the standard knob: translated on every route.
	reqOff := userReq("prove this")
	reqOff.Requirements = &Requirements{Thinking: "off"}
	if out := r.resolveAndPatchThinking(body("prove this"), reqOff, "pinned"); !strings.Contains(string(out), `"enable_thinking":false`) {
		t.Fatalf("requirements.thinking off must patch even on pinned routes: %s", out)
	}
	reqOn := userReq("translate hello")
	reqOn.Requirements = &Requirements{Thinking: "required"} // legacy synonym for on
	if out := r.resolveAndPatchThinking(body("translate hello"), reqOn, "route"); !strings.Contains(string(out), `"enable_thinking":true`) {
		t.Fatalf("requirements.thinking on must patch enable_thinking=true: %s", out)
	}

	// An explicit kwargs value is the low-level escape hatch and beats requirements.
	kwargsWin := userReq("hi")
	kwargsWin.Requirements = &Requirements{Thinking: "off"}
	in := `{"messages":[],"chat_template_kwargs":{"enable_thinking":true}}`
	if out := r.resolveAndPatchThinking([]byte(in), kwargsWin, "route"); string(out) != in {
		t.Fatalf("explicit kwargs must win over requirements.thinking: %s", out)
	}
	kwargsOnly := userReq("prove this")
	kwargsOnly.ChatTemplateKwargs = map[string]any{"enable_thinking": false}
	if out := r.resolveAndPatchThinking([]byte(in), kwargsOnly, "route"); string(out) != in {
		t.Fatal("explicit kwargs must suppress auto-injection")
	}

	off := &Router{cfg: &Config{AutoThinking: false}, classifier: testClassifier(fakeEmbed)}
	if out := off.resolveAndPatchThinking(body("prove this"), userReq("prove this"), "route"); strings.Contains(string(out), "enable_thinking") {
		t.Fatal("auto-thinking disabled must not inject")
	}
	reqOff2 := userReq("hi")
	reqOff2.Requirements = &Requirements{Thinking: "off"}
	if out := off.resolveAndPatchThinking(body("hi"), reqOff2, "route"); !strings.Contains(string(out), `"enable_thinking":false`) {
		t.Fatal("requirements.thinking must work even with auto-thinking disabled")
	}
}

func TestDifficultyCacheEviction(t *testing.T) {
	c := newDifficultyCache(2)
	c.put("a", classification{difficulty: 0.1})
	c.put("b", classification{difficulty: 0.2})
	c.put("c", classification{difficulty: 0.3}) // evicts "a" (FIFO)
	if _, ok := c.get("a"); ok {
		t.Fatal("oldest entry should have been evicted")
	}
	if v, ok := c.get("c"); !ok || v.difficulty != 0.3 {
		t.Fatalf("newest entry missing: v=%v ok=%v", v, ok)
	}
	c.put("b", classification{difficulty: 0.9})
	if v, _ := c.get("b"); v.difficulty != 0.9 {
		t.Fatal("update of existing key failed")
	}
	if _, ok := c.get("c"); !ok {
		t.Fatal("updating an existing key must not evict others")
	}
}

func ids(bs []*Backend) []string {
	out := make([]string, len(bs))
	for i, b := range bs {
		out[i] = b.ID
	}
	return out
}

func TestIsEmbeddingsOnly(t *testing.T) {
	mk := func(features ...string) *Backend {
		return &Backend{BackendRegistration: BackendRegistration{Features: features}}
	}
	if !isEmbeddingsOnly(mk("embeddings", "cpu")) {
		t.Fatal("embeddings-only worker not detected")
	}
	if isEmbeddingsOnly(mk("chat", "json")) {
		t.Fatal("chat worker misflagged as embeddings-only")
	}
	if isEmbeddingsOnly(mk("chat", "embeddings")) {
		t.Fatal("a chat+embeddings worker must not be flagged embeddings-only")
	}
}

func TestCertifyEmbeddingsWorker(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
		case "/v1/embeddings":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}]}`)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	reg := newTestRegistry()
	reg.upsert(BackendRegistration{
		ID: "emb", URL: srv.URL, Model: "default", Quality: 5,
		Features: []string{"embeddings", "cpu"}, TTLSeconds: 3600, HealthPath: "/health",
	})
	r := &Router{cfg: &Config{}, registry: reg, client: &http.Client{Timeout: 5 * time.Second}}
	r.certifyBackend("emb")

	b := reg.get("emb")
	if b == nil || !b.Certification.Ready || !b.Healthy {
		t.Fatalf("embeddings-only worker should certify ready without a chat probe: %+v", b)
	}
	if !b.Certification.Checks["embeddings"].OK {
		t.Fatalf("embeddings check should have passed: %+v", b.Certification.Checks)
	}
	if got, err := r.selectBackendsByFeature("embeddings"); err != nil || got[0].ID != "emb" {
		t.Fatalf("certified embeddings worker not selectable by feature: %v %v", ids(got), err)
	}
}

func TestCertifyEmbeddingsWorkerProbeFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	reg := newTestRegistry()
	reg.upsert(BackendRegistration{
		ID: "emb", URL: srv.URL, Model: "default",
		Features: []string{"embeddings"}, TTLSeconds: 3600, HealthPath: "/health",
	})
	r := &Router{cfg: &Config{}, registry: reg, client: &http.Client{Timeout: 5 * time.Second}}
	r.certifyBackend("emb")
	if reg.get("emb").Certification.Ready {
		t.Fatal("a worker whose embeddings probe fails must not certify ready")
	}
}

func TestSelectBackendsExcludesEmbeddingsWorker(t *testing.T) {
	reg := newTestRegistry()
	readyBackend(reg, "big", 10, 140, 6)
	reg.upsert(BackendRegistration{
		ID: "emb", URL: "http://emb", Model: "default", Quality: 5,
		Features: []string{"embeddings", "cpu"}, TTLSeconds: 3600,
	})
	reg.finishCertification("emb", true, map[string]Check{}, 0, 0, "")
	r := &Router{cfg: &Config{DefaultMaxTokens: 4096}, registry: reg}

	cands, _, _, _, err := r.selectBackends(userReq("hello"), 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(cands) != 1 || cands[0].ID != "big" {
		t.Fatalf("chat selection should exclude the embeddings worker, got %v", ids(cands))
	}
	if got, err := r.selectBackendsByFeature("embeddings"); err != nil || got[0].ID != "emb" {
		t.Fatalf("embeddings feature selection failed: %v %v", ids(got), err)
	}
}

// ── Request-aware cost model ─────────────────────────────────────────────────

func TestCostForRequest(t *testing.T) {
	plain := costForRequest(&ChatRequest{
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, false)
	if plain.outputTokens != latencyEstTokens {
		t.Fatalf("non-thinking output estimate = %d, want %d", plain.outputTokens, latencyEstTokens)
	}

	think := costForRequest(&ChatRequest{
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, true)
	if think.outputTokens <= plain.outputTokens {
		t.Fatalf("a thinking turn must be estimated longer than a direct one: %d vs %d",
			think.outputTokens, plain.outputTokens)
	}

	// A client budget BELOW the nominal figure is a real ceiling.
	capped := costForRequest(&ChatRequest{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 40,
	}, true)
	if capped.outputTokens != 40 {
		t.Fatalf("max_tokens=40 should cap the estimate, got %d", capped.outputTokens)
	}

	// A budget far ABOVE the nominal figure is a cap, not a forecast — the agent
	// turn that timed out declared 8192 and would have produced ~1700.
	generous := costForRequest(&ChatRequest{
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 8192,
	}, true)
	if generous.outputTokens != latencyEstThinkTokens {
		t.Fatalf("max_tokens=8192 should not inflate the estimate, got %d", generous.outputTokens)
	}

	// Tool schemas are prefill work and must be counted.
	withTools := costForRequest(&ChatRequest{
		Messages: []Message{{Role: "user", Content: strings.Repeat("x", 3000)}},
		Tools:    json.RawMessage(`[{"function":{"name":"a","parameters":{}}}]`),
	}, false)
	bare := costForRequest(&ChatRequest{
		Messages: []Message{{Role: "user", Content: strings.Repeat("x", 3000)}},
	}, false)
	if withTools.promptTokens <= bare.promptTokens {
		t.Fatalf("tool schemas must add prefill tokens: %d vs %d",
			withTools.promptTokens, bare.promptTokens)
	}
}

func TestPrefillTermSeparatesWorkers(t *testing.T) {
	// Same decode rate, wildly different prefill rate (the measured GPU-vs-CPU
	// gap on an identical prompt was 0.67s vs 37.2s).
	fastPrefill := &Backend{ObservedTPS: 50, ObservedPrefillTPS: 6000,
		BackendRegistration: BackendRegistration{MaxConcurrency: 4}}
	slowPrefill := &Backend{ObservedTPS: 50, ObservedPrefillTPS: 110,
		BackendRegistration: BackendRegistration{MaxConcurrency: 4}}

	short := jobCost{promptTokens: 20, outputTokens: 256}
	long := jobCost{promptTokens: 4000, outputTokens: 256}

	// On a tiny prompt the two are near-identical; on a long one the slow-prefill
	// worker must rank clearly worse.
	if a, b := expectedLatency(fastPrefill, short), expectedLatency(slowPrefill, short); b-a > 1 {
		t.Fatalf("short prompt should barely separate them: %.2f vs %.2f", a, b)
	}
	a, b := expectedLatency(fastPrefill, long), expectedLatency(slowPrefill, long)
	if b <= a*2 {
		t.Fatalf("long prompt should heavily penalise slow prefill: %.2f vs %.2f", a, b)
	}
}

// TestThinkingJobPrefersGPU is the regression for the routing inversion that
// timed out a scheduled agent turn: with a fixed 256-token estimate the CPU
// worker ranked FIRST and needed >120s for the job, while the idle GPU ranked
// fourth and needed 13s. Numbers are the ones measured off the live fleet.
func TestThinkingJobPrefersGPU(t *testing.T) {
	gpu := &Backend{
		BackendRegistration: BackendRegistration{ID: "llm-6000pro", Quality: 100, MaxConcurrency: 8},
		ObservedTPS:         90.6, ObservedPrefillTPS: 6000, ObservedTTFTMillis: 3366,
	}
	cpu := &Backend{
		BackendRegistration: BackendRegistration{ID: "llm-cpu-gemma", Quality: 89, MaxConcurrency: 4},
		ObservedTPS:         51.4, ObservedPrefillTPS: 110, ObservedTTFTMillis: 111,
	}

	// The failing request: ~4k tokens of system prompt + tool schemas, thinking on.
	job := jobCost{promptTokens: 4000, outputTokens: latencyEstThinkTokens}
	got := rankByDifficulty([]*Backend{cpu, gpu}, 87, job)
	if got[0].ID != "llm-6000pro" {
		t.Fatalf("thinking agent turn should go to the GPU, got %s (gpu=%.1fs cpu=%.1fs)",
			got[0].ID, expectedLatency(gpu, job), expectedLatency(cpu, job))
	}

	// A trivial short turn may still legitimately go to the cheaper worker — the
	// point is that the estimate now depends on the job, not that the GPU always wins.
	tiny := jobCost{promptTokens: 12, outputTokens: 32}
	if expectedLatency(cpu, tiny) >= expectedLatency(gpu, tiny) {
		t.Logf("note: GPU also wins the tiny job (gpu=%.3fs cpu=%.3fs)",
			expectedLatency(gpu, tiny), expectedLatency(cpu, tiny))
	}
}

func TestDeadlineFilter(t *testing.T) {
	fast := &Backend{BackendRegistration: BackendRegistration{ID: "fast", MaxConcurrency: 4},
		ObservedTPS: 100, ObservedPrefillTPS: 6000}
	slow := &Backend{BackendRegistration: BackendRegistration{ID: "slow", MaxConcurrency: 4},
		ObservedTPS: 5, ObservedPrefillTPS: 100}
	job := jobCost{promptTokens: 2000, outputTokens: 1500}

	// Generous budget: nothing is filtered.
	if got, applied := deadlineFilter([]*Backend{fast, slow}, job, time.Hour); applied || len(got) != 2 {
		t.Fatalf("generous budget should not filter, got %d applied=%v", len(got), applied)
	}
	// Tight budget only fast can meet: slow is dropped.
	got, applied := deadlineFilter([]*Backend{fast, slow}, job, 60*time.Second)
	if !applied || len(got) != 1 || got[0].ID != "fast" {
		t.Fatalf("tight budget should keep only fast, got %v applied=%v", ids(got), applied)
	}
	// Impossible budget: never return an empty set — the estimate is a heuristic
	// and must not become a new source of refusals.
	if got, applied := deadlineFilter([]*Backend{fast, slow}, job, time.Millisecond); len(got) != 2 || applied {
		t.Fatalf("impossible budget must keep every candidate, got %v applied=%v", ids(got), applied)
	}
	// Unknown budget is a no-op.
	if got, applied := deadlineFilter([]*Backend{fast, slow}, job, 0); len(got) != 2 || applied {
		t.Fatalf("zero budget must be a no-op, got %v applied=%v", ids(got), applied)
	}
}

// ── Standard-client surface: model + reasoning_effort ────────────────────────
//
// A coding harness (pi, hermes) is written against a plain OpenAI server: it
// names a model from /v1/models and sets reasoning_effort. These prove the
// router answers to that vocabulary, and — just as important — that the 20+
// clabtree guests, which send "model":"default" and no effort, still get the
// fully automatic behaviour they had before.

func namedBackend(reg *Registry, id, model string, quality, tps, contextK int, thinking bool) {
	reg.upsert(BackendRegistration{
		ID: id, URL: "http://" + id, Model: model, Quality: quality,
		BaselineTPS: float64(tps), MaxConcurrency: 4, TTLSeconds: 3600,
		ContextK: contextK, Thinking: thinking, Features: []string{"chat"},
	})
	reg.finishCertification(id, true, map[string]Check{}, float64(tps), 10, "")
}

func TestRequestedModelSentinels(t *testing.T) {
	for _, name := range []string{"", "default", "Default", " auto ", "router"} {
		if got := requestedModel(&ChatRequest{Model: name}); got != "" {
			t.Fatalf("model %q must mean auto, got %q", name, got)
		}
	}
	if got := requestedModel(&ChatRequest{Model: " llm-6000pro "}); got != "llm-6000pro" {
		t.Fatalf("a named model must survive trimming, got %q", got)
	}
}

func TestSelectBackendsHonoursNamedModel(t *testing.T) {
	reg := newTestRegistry()
	namedBackend(reg, "gemma-a", "gemma.gguf", 80, 64, 32, true)
	namedBackend(reg, "gemma-b", "gemma.gguf", 78, 30, 32, true)
	namedBackend(reg, "deepseek", "deepseek-284B", 99, 22, 256, true)
	r := &Router{cfg: &Config{DefaultMaxTokens: 4096}, registry: reg}

	// By model name: every worker serving it stays a candidate, so a named model
	// still load-balances instead of collapsing onto one worker.
	req := userReq("hello")
	req.Model = "gemma.gguf"
	cands, route, _, _, err := r.selectBackends(req, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(cands) != 2 {
		t.Fatalf("named model should keep both workers serving it, got %v", ids(cands))
	}
	for _, b := range cands {
		if b.Model != "gemma.gguf" {
			t.Fatalf("candidate %s serves %q, not the named model", b.ID, b.Model)
		}
	}
	// Reported as "model:…" so parseRouteScore ignores it and the online tier
	// adapter never learns from a tier the client picked.
	if strings.HasPrefix(route, "route") {
		t.Fatalf("a client-named model must not report as an auto route, got %q", route)
	}

	// By worker id: the other spelling /v1/models publishes (owned_by).
	req.Model = "deepseek"
	cands, _, _, _, err = r.selectBackends(req, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(cands) != 1 || cands[0].ID != "deepseek" {
		t.Fatalf("naming a worker id should select it, got %v", ids(cands))
	}

	// Unknown: a 404, not a 503 — a harness can act on "you don't have that".
	req.Model = "gpt-5"
	if _, _, _, _, err = r.selectBackends(req, 0); err == nil {
		t.Fatal("an unknown model must be an error")
	} else if !errors.As(err, &unknownModelError{}) {
		t.Fatalf("unknown model must surface as unknownModelError (⇒404), got %T: %v", err, err)
	}
}

func TestSelectBackendsGuestDefaultIsUnchanged(t *testing.T) {
	reg := newTestRegistry()
	namedBackend(reg, "gemma-a", "gemma.gguf", 80, 64, 32, true)
	namedBackend(reg, "deepseek", "deepseek-284B", 99, 22, 256, true)
	r := &Router{cfg: &Config{DefaultMaxTokens: 4096}, registry: reg}

	// What every clabtree guest sends.
	req := userReq("hello")
	req.Model = "default"
	cands, route, _, _, err := r.selectBackends(req, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(cands) != 2 {
		t.Fatalf(`"default" must not filter anything, got %v`, ids(cands))
	}
	if !strings.HasPrefix(route, "route") {
		t.Fatalf(`"default" must stay on the auto route, got %q`, route)
	}
}

// TestClientCeilingDoesNotFilterTheFleet is the pi regression: a harness
// declaring a generous max_tokens must not be treated as needing that much
// CONTEXT. Measured 2026-08-09, before the fix: "say hi" with max_tokens 131072
// filtered every worker under 128K out and routed to the 284B at 22 tok/s.
func TestClientCeilingDoesNotFilterTheFleet(t *testing.T) {
	reg := newTestRegistry()
	namedBackend(reg, "small-fast", "gemma.gguf", 80, 64, 16, true)
	namedBackend(reg, "big-slow", "deepseek-284B", 99, 22, 256, true)
	r := &Router{cfg: &Config{DefaultMaxTokens: 4096}, registry: reg}

	req := userReq("say hi")
	req.MaxTokens = 131072 // pi's default ceiling
	cands, _, _, _, err := r.selectBackends(req, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(cands) != 2 {
		t.Fatalf("a declared ceiling must not exclude small-context workers, got %v", ids(cands))
	}

	// A prompt that genuinely needs the context still filters on it.
	big := &ChatRequest{MaxTokens: 4096, Messages: []Message{{Role: "user", Content: strings.Repeat("x", 200*1024*3)}}}
	cands, _, _, _, err = r.selectBackends(big, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(cands) != 1 || cands[0].ID != "big-slow" {
		t.Fatalf("a genuinely large prompt must still hard-filter on context, got %v", ids(cands))
	}
}

func TestReasoningEffortReachesTheTemplate(t *testing.T) {
	r := &Router{cfg: &Config{}, registry: newTestRegistry()}
	body := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)

	// An explicit level: thinking on, the level carried through to the template,
	// and a thinking-capable worker hard-required.
	tr := r.resolveThinking(&ChatRequest{ReasoningEffort: "max"}, "route", nil)
	if !tr.patch || !tr.enable || tr.effort != "max" || !tr.hardThink {
		t.Fatalf("reasoning_effort=max resolved to %+v", tr)
	}
	got := string(patchForwardedBody(body, 0, 0, tr, ""))
	if !strings.Contains(got, `"enable_thinking":true`) || !strings.Contains(got, `"reasoning_effort":"max"`) {
		t.Fatalf("forwarded body missing gate or level: %s", got)
	}

	// "none" and its synonyms mean off, and send no level.
	for _, off := range []string{"none", "off", "disabled"} {
		tr = r.resolveThinking(&ChatRequest{ReasoningEffort: off}, "route", nil)
		if !tr.patch || tr.enable || tr.hardThink {
			t.Fatalf("reasoning_effort=%q resolved to %+v", off, tr)
		}
		got = string(patchForwardedBody(body, 0, 0, tr, ""))
		if !strings.Contains(got, `"enable_thinking":false`) || strings.Contains(got, "reasoning_effort") {
			t.Fatalf("off must send the gate and no level: %s", got)
		}
	}

	// Absent ⇒ auto: nothing explicit, so the classifier decides (nil here).
	if tr = r.resolveThinking(&ChatRequest{}, "route", nil); tr.patch || tr.hardThink {
		t.Fatalf("absent reasoning_effort must leave the decision to auto, got %+v", tr)
	}
}

// TestThinkingEscapeHatchBothSpellings: DeepSeek V4 reads `thinking` and only
// falls back to `enable_thinking`, so a client pinning the former owns the
// decision — the router must neither overwrite it nor filter on the opposite
// value. Verified against the live worker:
// {"thinking":false,"enable_thinking":true} renders thinking OFF.
func TestThinkingEscapeHatchBothSpellings(t *testing.T) {
	r := &Router{cfg: &Config{}, registry: newTestRegistry()}
	for _, key := range []string{"thinking", "enable_thinking"} {
		req := &ChatRequest{ChatTemplateKwargs: map[string]any{key: false}}
		if tr := r.resolveThinking(req, "route", nil); tr.patch || tr.hardThink {
			t.Fatalf("a client-pinned %s must not be overwritten or filtered on, got %+v", key, tr)
		}
		if got := thinkingFromRequest(req); got != "off" {
			t.Fatalf("%s:false should read as off, got %q", key, got)
		}
		body := []byte(`{"chat_template_kwargs":{"` + key + `":false}}`)
		if got := string(patchForwardedBody(body, 0, 0, thinkingResolution{patch: true, enable: true, effort: "high"}, "")); got != string(body) {
			t.Fatalf("the %s escape hatch must survive the patch untouched, got %s", key, got)
		}
	}
}

func TestBudgetCeilingClampsClientMaxTokens(t *testing.T) {
	body := []byte(`{"max_tokens":131072,"messages":[{"role":"user","content":"hi"}]}`)
	// 16K of context, no prompt to speak of ⇒ the client's 128K ceiling can't fit.
	ceiling := budgetCeiling(&Backend{BackendRegistration: BackendRegistration{ContextK: 16}}, jobCost{promptTokens: 1000})
	if ceiling <= 0 || ceiling >= 131072 {
		t.Fatalf("ceiling %d should be a real trim below the client's ask", ceiling)
	}
	got := string(patchForwardedBody(body, 0, ceiling, thinkingResolution{}, ""))
	if strings.Contains(got, "131072") {
		t.Fatalf("client ceiling should have been clamped to the worker's context: %s", got)
	}
	// A budget that already fits is left exactly as the client sent it.
	small := []byte(`{"max_tokens":512,"messages":[{"role":"user","content":"hi"}]}`)
	if got := string(patchForwardedBody(small, 0, ceiling, thinkingResolution{}, "")); got != string(small) {
		t.Fatalf("a fitting budget must not be rewritten, got %s", got)
	}
	// An unknown context means don't guess.
	if c := budgetCeiling(&Backend{}, jobCost{promptTokens: 10}); c != 0 {
		t.Fatalf("a worker with no declared context must not be clamped, got %d", c)
	}
}

// TestReadSSEStreamCountsTokens: the cold-start speed probe used to estimate
// tokens as len(text)/4, which is not a constant — measured on the 284B worker
// 2026-08-09, prose ran 5.05 chars/token and code 3.79, so the same worker's
// decode rate came out ~26% high on one and ~5% low on the other. Counting
// deltas is what the live EWMA has always done, so cold-start and live now
// measure the same quantity.
func TestReadSSEStreamCountsTokens(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Hello"}}]}`,
		`data: {"choices":[{"delta":{"content":", world of very long words"}}]}`,
		`data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}`,
		`data: {"choices":[{"delta":{"content":""}}]}`, // empty delta is not a token
		`data: {"choices":[{"delta":{"role":"assistant"}}]}`,
		`data: [DONE]`,
		"",
	}, "\n")
	firsts := 0
	content, reasoning, tokens, err := readSSEStream(strings.NewReader(stream), func() { firsts++ })
	if err != nil {
		t.Fatal(err)
	}
	if tokens != 3 {
		t.Fatalf("want 3 counted deltas, got %d", tokens)
	}
	if content != "Hello, world of very long words" || reasoning != "hmm" {
		t.Fatalf("stream reassembly changed: %q / %q", content, reasoning)
	}
	// The character estimate this replaced would have called those 3 tokens ~8.
	if est := len(content+reasoning) / 4; est == tokens {
		t.Fatalf("test is not exercising the difference: estimate %d == count %d", est, tokens)
	}
}

// A usage chunk (from stream_options.include_usage) is exact and must win over
// the delta count: MTP workers pack ~2.5 tokens per delta, which is why the
// speed probe requests usage at all (speedProbeVersion 3).
func TestReadSSEStreamPrefersUsage(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"choices":[{"delta":{"content":"Reliable local routing"}}]}`,
		`data: {"choices":[{"delta":{"content":" keeps traffic flowing smoothly."}}]}`,
		`data: {"choices":[],"usage":{"prompt_tokens":30,"completion_tokens":52}}`,
		`data: [DONE]`,
		"",
	}, "\n")
	_, _, tokens, err := readSSEStream(strings.NewReader(stream), func() {})
	if err != nil {
		t.Fatal(err)
	}
	if tokens != 52 {
		t.Fatalf("want usage's 52 tokens, got %d", tokens)
	}
}

// ── Measured thinking dialect ───────────────────────────────────────────────

// dialectWorker is an endpoint that reasons only when asked in `honours`, and
// refuses the other gate outright when `strict` — which is what a provider that
// validates its input does with chat_template_kwargs.
func dialectWorker(t *testing.T, honours string, strict bool) *Backend {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(req.Body).Decode(&body)
		_, hasKwargs := body["chat_template_kwargs"]
		_, hasEffort := body["reasoning_effort"]
		asked := map[string]bool{thinkingDialectKwargs: hasKwargs, thinkingDialectEffort: hasEffort}
		if strict && asked[thinkingDialectKwargs] && honours != thinkingDialectKwargs {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"message":"Unrecognized request argument supplied: chat_template_kwargs","type":"invalid_request_error"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if asked[honours] {
			_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"75 km/h","reasoning_content":"60+90 over 2.5h"}}]}`)
			return
		}
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"75 km/h"}}]}`)
	}))
	t.Cleanup(srv.Close)
	return &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL, Model: "m"}}
}

// The router used to declare the dialect: it always wrote
// chat_template_kwargs.enable_thinking, which is a vLLM and llama.cpp extension.
// A provider that speaks reasoning_effort ignores it, and a strict one rejects
// the unknown field — either way the router believes it asked for thinking and
// never got it.
func TestThinkingProbeMeasuresTheDialect(t *testing.T) {
	cases := []struct {
		name         string
		honours      string
		strict       bool
		wantThinking bool
		wantDialect  string
	}{
		{"local worker: the kwargs gate, tried first and settled on the spot",
			thinkingDialectKwargs, false, true, thinkingDialectKwargs},
		{"provider that quietly ignores the kwargs object still gets its own field tried",
			thinkingDialectEffort, false, true, thinkingDialectEffort},
		{"strict provider that refuses the kwargs object outright",
			thinkingDialectEffort, true, true, thinkingDialectEffort},
		{"a model that reasons for neither gate",
			"nothing", false, false, thinkingDialectNone},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := &Router{client: &http.Client{Timeout: 5 * time.Second}}
			thinking, dialect, inconclusive := r.thinkingProbe(dialectWorker(t, tc.honours, tc.strict))
			if inconclusive {
				t.Fatal("probe was inconclusive against a responsive worker")
			}
			if thinking != tc.wantThinking || dialect != tc.wantDialect {
				t.Errorf("probe = (thinking %v, dialect %q), want (%v, %q)",
					thinking, dialect, tc.wantThinking, tc.wantDialect)
			}
		})
	}
}

// TestPatchForwardedBodyHonoursThinkingDialect: the gate goes out in the
// spelling the CHOSEN endpoint was measured to honour, and in no other.
func TestPatchForwardedBodyHonoursThinkingDialect(t *testing.T) {
	bare := []byte(`{"messages":[]}`)
	cases := []struct {
		name    string
		body    string
		tr      thinkingResolution
		want    []string
		wantNot []string
	}{
		{"unmeasured falls back to the dialect the fleet has always spoken",
			`{"messages":[]}`, thinkingResolution{patch: true, enable: true},
			[]string{`"enable_thinking":true`}, []string{`"reasoning_effort"`}},
		{"measured kwargs is the same thing, said explicitly",
			`{"messages":[]}`, thinkingResolution{patch: true, enable: true, dialect: thinkingDialectKwargs},
			[]string{`"enable_thinking":true`}, []string{`"reasoning_effort"`}},
		{"measured reasoning_effort: on with no level asked for is the neutral level",
			`{"messages":[]}`, thinkingResolution{patch: true, enable: true, dialect: thinkingDialectEffort},
			[]string{`"reasoning_effort":"medium"`}, []string{`chat_template_kwargs`}},
		{"measured reasoning_effort: a resolved level travels verbatim",
			`{"messages":[]}`, thinkingResolution{patch: true, enable: true, effort: "high", dialect: thinkingDialectEffort},
			[]string{`"reasoning_effort":"high"`}, []string{`chat_template_kwargs`}},
		{"measured reasoning_effort: off is the standard's own spelling for off",
			`{"messages":[]}`, thinkingResolution{patch: true, enable: false, dialect: thinkingDialectEffort},
			[]string{`"reasoning_effort":"none"`}, []string{`chat_template_kwargs`}},
		{"the escape hatch holds in the new dialect too",
			`{"reasoning_effort":"LOW"}`, thinkingResolution{patch: true, enable: true, effort: "high", dialect: thinkingDialectEffort},
			[]string{`"reasoning_effort":"LOW"`}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := string(patchForwardedBody([]byte(tc.body), 0, 0, tc.tr, ""))
			for _, want := range tc.want {
				if !strings.Contains(got, want) {
					t.Errorf("missing %s: %s", want, got)
				}
			}
			for _, unwanted := range tc.wantNot {
				if strings.Contains(got, unwanted) {
					t.Errorf("wrote %s to an endpoint that does not read it: %s", unwanted, got)
				}
			}
		})
	}

	// Measured as honouring neither gate: send neither. A field a strict endpoint
	// can reject, bought for nothing, is exactly what this is here to stop.
	tr := thinkingResolution{patch: true, enable: false, dialect: thinkingDialectNone}
	if got := patchForwardedBody(bare, 0, 0, tr, ""); string(got) != string(bare) {
		t.Errorf("wrote a gate to an endpoint that honours none: %s", got)
	}

	// The dialect is stamped from the worker that will actually serve the
	// request, after selection.
	if got := tr.forBackend(&Backend{ThinkingDialect: thinkingDialectEffort}).dialect; got != thinkingDialectEffort {
		t.Errorf("forBackend = %q, want the worker's measured dialect", got)
	}
	if got := tr.forBackend(nil).dialect; got != thinkingDialectNone {
		t.Errorf("forBackend(nil) changed the resolution: %q", got)
	}
}

// ── Derived classifier deadline ─────────────────────────────────────────────

// A fixed two-second classification deadline silently disabled auto-routing on a
// slow box: every classify() timed out, routing fell back to plain quality and
// speed, and /health still reported the embeddings worker present — so nothing
// said so. The deadline is derived from that worker's measured latency instead.
func TestClassifierDeadlineFromMeasuredLatency(t *testing.T) {
	c := newDifficultyClassifier(&Config{DifficultyTimeout: difficultyTimeoutFallback}, fakeEmbed)
	if got := c.deadline(); got != difficultyTimeoutFallback {
		t.Fatalf("deadline before any measurement = %s, want the %s fallback", got, difficultyTimeoutFallback)
	}
	cases := []struct {
		name     string
		measured time.Duration
		want     time.Duration
	}{
		{"a quick worker cannot make the deadline tighter than what the router shipped with",
			10 * time.Millisecond, difficultyTimeoutFallback},
		{"a slow worker raises it by the probe-to-classification factor",
			500 * time.Millisecond, 500 * time.Millisecond * classifierDeadlineFactor},
		{"a pathological worker is capped at the window the router already accepts being wrong for",
			30 * time.Second, embedFailCooldown},
	}
	for _, tc := range cases {
		if got := c.observeEmbedLatency(tc.measured); got != tc.want {
			t.Errorf("%s: %s measured → %s, want %s", tc.name, tc.measured, got, tc.want)
		}
	}
}

// End to end: certifying the embeddings worker measures it, and what the
// deadline settled on is readable rather than folklore.
func TestEmbeddingsCertificationPublishesTheDeadline(t *testing.T) {
	const latency = 400 * time.Millisecond
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		time.Sleep(latency)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}]}`)
	}))
	defer srv.Close()

	reg := newTestRegistry()
	reg.upsert(BackendRegistration{
		ID: "emb", URL: srv.URL, Model: "default",
		Features: []string{"embeddings"}, TTLSeconds: 3600, HealthPath: "/health",
	})
	cfg := &Config{DifficultyTimeout: difficultyTimeoutFallback}
	r := &Router{cfg: cfg, registry: reg, client: &http.Client{Timeout: 5 * time.Second},
		classifier: newDifficultyClassifier(cfg, fakeEmbed)}
	r.certifyBackend("emb")

	got := r.classifier.deadline()
	if got <= difficultyTimeoutFallback {
		t.Fatalf("deadline = %s; a %s round trip must have raised it above the %s fallback",
			got, latency, difficultyTimeoutFallback)
	}
	// The certification check records the measurement an operator would otherwise
	// have to infer.
	if msg := reg.get("emb").Certification.Checks["embeddings"].Message; !strings.Contains(msg, "classifier deadline") {
		t.Errorf("embeddings check does not report what the deadline settled on: %q", msg)
	}
	// And /health publishes it, because it is a number nobody can predict.
	rec := httptest.NewRecorder()
	r.handleHealth(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
	var health map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &health); err != nil {
		t.Fatalf("health is not JSON: %v", err)
	}
	if health["classifier_deadline"] != got.String() {
		t.Errorf("/health reports classifier_deadline=%v, want %s", health["classifier_deadline"], got)
	}
}
