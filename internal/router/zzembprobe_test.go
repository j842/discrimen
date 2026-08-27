package router

// THROWAWAY probe file (embeddings-dependency review) — delete before finishing.

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// A backend that certifies cleanly, then starts misbehaving.
func TestEmbProbeTransportAfterCert(t *testing.T) {
	var mode atomic.Value
	mode.Store(`{"data":[{"index":0,"embedding":[0.1,0.2,0.3,0.4]}]}`)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, mode.Load().(string))
	}))
	defer srv.Close()

	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "emb", URL: srv.URL, Model: "default",
		Features: []string{"embeddings"}, TTLSeconds: 3600, HealthPath: "/health"})
	cfg := &Config{DifficultyTimeout: difficultyTimeoutFallback, AutoDifficulty: true}
	r := &Router{cfg: cfg, registry: reg, client: &http.Client{Timeout: 5 * time.Second}}
	r.classifier = newDifficultyClassifier(cfg, r.embedTexts)
	r.certifyBackend("emb")
	t.Logf("certified: ready=%v healthy=%v eligible=%d",
		reg.get("emb").Certification.Ready, reg.get("emb").Healthy, len(reg.eligible()))

	for _, tc := range []struct{ name, body string }{
		{"200, empty data array", `{"data":[]}`},
		{"200, [[]] empty vector", `{"data":[{"index":0,"embedding":[]}]}`},
		{"200, null embedding", `{"data":[{"index":0,"embedding":null}]}`},
		{"200, no data key at all", `{"object":"list"}`},
		{"200, HTML not JSON", `<html>502</html>`},
		{"200, wrong dimension (2 not 4)", `{"data":[{"index":0,"embedding":[0.5,0.5]}]}`},
		{"200, all-zero vector", `{"data":[{"index":0,"embedding":[0,0,0,0]}]}`},
	} {
		mode.Store(tc.body)
		vecs, err := r.embedTexts(context.Background(), []string{"one"})
		t.Logf("%-32s -> vecs=%v err=%v", tc.name, vecs, err)
	}
}

// What ORDER do candidates come out in when classification is unavailable?
func TestEmbProbeFallbackRanking(t *testing.T) {
	mk := func(id string, quality int, tps float64, free bool) *Backend {
		b := &Backend{BackendRegistration: BackendRegistration{
			ID: id, Model: id, Quality: quality,
			BaselineTPS: tps, MaxConcurrency: 4, TTLSeconds: 3600,
		}}
		b.QualityNoThink = quality
		b.Healthy = true
		b.Certification.Ready = true
		b.LastSeen = time.Now()
		if !free {
			b.InputPricePerMtok = 3.0
			b.OutputPricePerMtok = 15.0
		}
		return b
	}
	cands := []*Backend{
		mk("cheap-fast-small", 40, 200, true),
		mk("mid", 70, 90, true),
		mk("big-slow-smart", 95, 20, true),
		mk("paid-frontier", 98, 120, false),
	}
	got := rankBackends(append([]*Backend{}, cands...), nominalJob(), false)
	for i, b := range got {
		t.Logf("fallback rank %d: %-18s quality=%-3d tps=%-5v score=%d",
			i, b.ID, b.Quality, b.BaselineTPS, backendScore(b, false))
	}

	// And what classified routing would have done for an easy prompt (d=0.10):
	for _, d := range []float64{0.10, 0.50, 0.90} {
		target := autoTargetQuality(cands, d, false)
		ranked := rankByDifficulty(append([]*Backend{}, cands...), target, nominalJob(), false)
		names := []string{}
		for _, b := range ranked {
			names = append(names, b.ID)
		}
		t.Logf("classified d=%.2f -> target q>=%-3d order=%v", d, target, names)
	}
}

// Route strings: what lands in RequestLog.Route for classified vs fallback.
func TestEmbProbeRouteStrings(t *testing.T) {
	t.Logf("classified auto     : %q", "route:d=0.42,q>=42")
	t.Logf("classified, outcome : %q", "route:outcome:p=0.90,n=8")
	t.Logf("FALLBACK (no embed) : %q", routeKind(""))
	t.Logf("FALLBACK, named mod : %q", routeKind("some-model"))
	// parseRouteScore is what the adapter/judge read.
	for _, s := range []string{"route:d=0.42,q>=42", "route", "model", "route:outcome:p=0.90,n=8"} {
		d, ok := parseRouteScore(s)
		t.Logf("parseRouteScore(%-26q) = %v, %v", s, d, ok)
	}
}

// Does a stale (already-ready) classifier self-heal after a dimension change,
// and what does it cost?
func TestEmbProbeDimChangeRecovery(t *testing.T) {
	dim := int32(4)
	var calls, seeds atomic.Int32
	embed := func(ctx context.Context, texts []string) ([][]float64, error) {
		calls.Add(1)
		seeds.Add(int32(len(texts)))
		d := int(dim)
		out := make([][]float64, len(texts))
		for i := range texts {
			v := make([]float64, d)
			v[i%d] = 1
			out[i] = v
		}
		return out, nil
	}
	c := testClassifier(embed)
	if _, ok := c.classify(userReq("first")); !ok {
		t.Fatal("warmup")
	}
	calls.Store(0)
	seeds.Store(0)
	dim = 8
	for i := 0; i < 4; i++ {
		_, ok := c.classify(userReq(string(rune('a'+i)) + " brand new prompt"))
		t.Logf("after dim swap, request %d: classified=%v embedCalls=%d textsEmbedded=%d",
			i, ok, calls.Load(), seeds.Load())
	}
}

// How does /health read across the three embeddings states?
func TestEmbProbeHealthStates(t *testing.T) {
	states := []struct {
		name   string
		mutate func(reg *Registry)
	}{
		{"registered + healthy", func(reg *Registry) { reg.setHealth("emb", true, "") }},
		{"registered, health check FAILING", func(reg *Registry) { reg.setHealth("emb", false, "connection refused") }},
		{"registered, certification FAILED", func(reg *Registry) {
			reg.finishCertification("emb", false, map[string]Check{}, 0, 0, "embeddings probe returned no vector")
		}},
	}
	for _, st := range states {
		reg := newTestRegistry()
		reg.upsert(BackendRegistration{ID: "emb", URL: "http://127.0.0.1:1", Model: "d",
			Features: []string{"embeddings"}, TTLSeconds: 3600, HealthPath: "/health"})
		reg.finishCertification("emb", true, map[string]Check{}, 0, 0, "")
		st.mutate(reg)
		cfg := &Config{DifficultyTimeout: difficultyTimeoutFallback, AutoDifficulty: true}
		r := &Router{cfg: cfg, registry: reg, client: &http.Client{Timeout: time.Second}}
		r.classifier = newDifficultyClassifier(cfg, r.embedTexts)
		rec := httptest.NewRecorder()
		r.handleHealth(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
		_, embErr := r.embedTexts(context.Background(), []string{"x"})
		t.Logf("%-34s /health=%s  || embedTexts err=%v", st.name, rec.Body.String(), embErr)
	}
}
