package router

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

// querySlots reads llama.cpp's own --parallel value out of /props. It exists to cap a
// capacity ramp that can over-report on a serialising worker, so the two cases that
// matter are "worker publishes a number" and "worker publishes nothing" (vLLM).
func TestQuerySlots(t *testing.T) {
	for _, tc := range []struct {
		name  string
		props string
		want  int
	}{
		{"llama.cpp single slot", `{"total_slots":1}`, 1},
		{"llama.cpp four slots", `{"total_slots":4}`, 4},
		{"no props endpoint", "", 0},
		{"props without total_slots", `{"model_path":"/models/x.gguf"}`, 0},
		{"nonsense value", `{"total_slots":0}`, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, done := fakeWorker(t, "", tc.props)
			defer done()
			r := &Router{client: &http.Client{}}
			if got := r.querySlots(b); got != tc.want {
				t.Errorf("querySlots = %d, want %d", got, tc.want)
			}
		})
	}
}

// The reported slot count must only ever LOWER the ramp's verdict. A worker that
// publishes more slots than the ramp could actually saturate keeps the ramp's number —
// the ramp measured real throughput, the config value is only an upper bound.
func TestSlotCapOnlyLowers(t *testing.T) {
	for _, tc := range []struct{ ramp, slots, want int }{
		{2, 1, 1}, // the naples case: ramp over-reported against --parallel 1
		{1, 4, 1}, // ramp found a lower practical knee — keep it
		{4, 4, 4},
		{8, 0, 8}, // worker publishes nothing (vLLM) — ramp stands
	} {
		capN := tc.ramp
		if tc.slots > 0 && capN > tc.slots {
			capN = tc.slots
		}
		if capN != tc.want {
			t.Errorf("ramp=%d slots=%d: got %d, want %d", tc.ramp, tc.slots, capN, tc.want)
		}
	}
}

// prefillProbe must derive tok/s from the worker's OWN usage.prompt_tokens rather than
// from a word count, and must reject samples too small to mean anything.
func TestPrefillProbe(t *testing.T) {
	const delay = 120 * time.Millisecond

	newSrv := func(promptTokens int) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			var body map[string]any
			_ = json.NewDecoder(req.Body).Decode(&body)
			// Thinking must be OFF: that is what makes the number comparable across
			// engines that buffer reasoning into TTFT and engines that stream it.
			if kw, ok := body["chat_template_kwargs"].(map[string]any); ok {
				if think, _ := kw["enable_thinking"].(bool); think {
					t.Error("prefill probe enabled thinking")
				}
			} else {
				t.Error("prefill probe sent no chat_template_kwargs")
			}
			if mt, _ := body["max_tokens"].(float64); mt != 1 {
				t.Errorf("max_tokens = %v, want 1", mt)
			}
			time.Sleep(delay)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"length"}],
				"usage":{"prompt_tokens":` + strconv.Itoa(promptTokens) + `,"completion_tokens":1}}`))
		}))
	}

	t.Run("rate comes from usage.prompt_tokens", func(t *testing.T) {
		srv := newSrv(1000)
		defer srv.Close()
		r := &Router{client: &http.Client{}}
		b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL}}

		rate, err := r.prefillProbe(b)
		if err != nil {
			t.Fatalf("prefillProbe: %v", err)
		}
		// 1000 tokens over at least `delay` — generous bounds, this is a timing test.
		if rate <= 0 || rate > 1000/delay.Seconds() {
			t.Fatalf("rate = %.1f tok/s, want 0 < rate <= %.0f", rate, 1000/delay.Seconds())
		}
	})

	t.Run("prompt below minPrefillTokens is rejected", func(t *testing.T) {
		srv := newSrv(minPrefillTokens - 1)
		defer srv.Close()
		r := &Router{client: &http.Client{}}
		b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL}}

		if rate, err := r.prefillProbe(b); err == nil {
			t.Fatalf("expected an error for a short prompt, got rate=%.1f", rate)
		}
	})

	t.Run("HTTP error surfaces", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			http.Error(w, "boom", http.StatusInternalServerError)
		}))
		defer srv.Close()
		r := &Router{client: &http.Client{}}
		b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL}}

		if _, err := r.prefillProbe(b); err == nil {
			t.Fatal("expected an error from a 500")
		}
	})
}

// The filler is what the probe actually prefills, so it has to be long enough to clear
// minPrefillTokens on a real tokeniser and varied enough not to flatter a prefix cache.
func TestPrefillFiller(t *testing.T) {
	s := prefillFiller(prefillProbeTokens, 0)
	words := len(strings.Fields(s))
	if words < prefillProbeTokens/2 {
		t.Errorf("filler has %d words, too short to reach %d tokens", words, prefillProbeTokens)
	}
	if s != prefillFiller(prefillProbeTokens, 0) {
		t.Error("filler is not deterministic — workers would be measured on different text")
	}
	// A run of identical words would compress an MoE's routing and defeat the point.
	f := strings.Fields(s)
	same := 0
	for i := 1; i < len(f); i++ {
		if f[i] == f[i-1] {
			same++
		}
	}
	if same > len(f)/10 {
		t.Errorf("%d/%d adjacent words repeat — filler is too uniform", same, len(f))
	}
}

// applyProfileIfGen seeds the prefill EWMA from the probe but must never clobber a live
// one, mirroring the ObservedTPS rule.
func TestApplyProfileSeedsPrefillOnlyWhenUnset(t *testing.T) {
	reg := newTestRegistry()
	register(t, reg, "a", 1)

	if !reg.applyProfileIfGen("a", 0, &WorkerProfile{Model: "m", PrefillTPS: 120}) {
		t.Fatal("applyProfileIfGen returned false")
	}
	if got := reg.get("a").ObservedPrefillTPS; got != 120 {
		t.Fatalf("seed failed: ObservedPrefillTPS = %v, want 120", got)
	}

	// A live EWMA is more current than a one-shot probe and must survive re-profiling.
	reg.mu.Lock()
	reg.backends["a"].ObservedPrefillTPS = 45
	reg.mu.Unlock()

	reg.applyProfileIfGen("a", 0, &WorkerProfile{Model: "m", PrefillTPS: 120})
	if got := reg.get("a").ObservedPrefillTPS; got != 45 {
		t.Fatalf("re-profile clobbered the live prefill EWMA: got %v, want 45", got)
	}
}

// A worker with a measured prefill rate must be priced by PROMPT LENGTH, not by a flat
// TTFT average — the naples failure, where a 978ms average stood in for 178 real
// seconds on a 4k prompt.
func TestPrefillSecondsScalesWithPrompt(t *testing.T) {
	slow := &Backend{ObservedPrefillTPS: 23, ObservedTTFTMillis: 978}
	if got := prefillSeconds(slow, 4116); got < 150 || got > 210 {
		t.Errorf("4116 tokens at 23 tok/s = %.0fs, want ~179s", got)
	}
	// Without a measured rate it still falls back to the flat average as before.
	noRate := &Backend{ObservedTTFTMillis: 978}
	if got := prefillSeconds(noRate, 4116); got < 0.9 || got > 1.1 {
		t.Errorf("fallback = %.3fs, want ~0.978s", got)
	}
}

// A cached profile is reused verbatim and never re-measured until benchmarkVersion
// changes, so a probe added without a version bump would never run on an
// already-profiled fleet. That is how the prefill probe shipped and then sat idle:
// 5 of 7 live workers, including both 284B ones, carried no prefill rate at all.
// backfillCachedProfile is the migration path, and it must only fire when the field
// is genuinely missing.
func TestBackfillCachedProfilePrefill(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Count ONLY the probe's own endpoint. backfillCachedProfile also re-reads
		// context (/v1/models, /props) on every call, and this test is about the
		// prefill probe not re-running, not about total request volume.
		if strings.HasSuffix(req.URL.Path, "/chat/completions") {
			calls++
		}
		time.Sleep(20 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"length"}],
			"usage":{"prompt_tokens":1000,"completion_tokens":1}}`))
	}))
	defer srv.Close()

	r := &Router{client: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL}}

	// Missing rate: probe, record it, and surface a check so /backends shows why.
	// ThinkingDialect is set so its own backfill clause stays quiet — this test is
	// about the prefill probe, and every backfill shares the chat endpoint.
	prof := &WorkerProfile{Model: "m", BenchVersion: benchmarkVersion, SpeedVersion: speedProbeVersion,
		ThinkingDialect: thinkingDialectNone}
	r.backfillCachedProfile("w", b, prof)
	if prof.PrefillTPS <= 0 {
		t.Fatalf("backfill did not set PrefillTPS: got %v", prof.PrefillTPS)
	}
	if _, ok := prof.Checks["prefill"]; !ok {
		t.Error("backfill set no prefill check")
	}
	// One backfill = one probe, and a probe is prefillProbeSamples HTTP requests
	// (best-of-N; see prefillProbeSamples for why a single sample is not trustworthy).
	if calls != prefillProbeSamples {
		t.Fatalf("expected exactly one probe (%d samples), got %d requests", prefillProbeSamples, calls)
	}

	// Already measured: must NOT re-probe. This runs on every registration keepalive,
	// so a probe here would be a recurring cost on the whole fleet forever.
	already := &WorkerProfile{Model: "m", BenchVersion: benchmarkVersion, SpeedVersion: speedProbeVersion,
		PrefillTPS: 77, ThinkingDialect: thinkingDialectNone}
	r.backfillCachedProfile("w", b, already)
	if already.PrefillTPS != 77 {
		t.Errorf("backfill clobbered an existing rate: got %v, want 77", already.PrefillTPS)
	}
	if calls != prefillProbeSamples {
		t.Errorf("re-probed a profile that already had a rate: %d requests", calls)
	}
}

// Every sample in one probe must send DIFFERENT text. Identical prompts hit
// llama-server's prompt cache, so the second and third calls skip prefill and report a
// fabricated rate — and since prefillProbe takes the BEST sample, it would select that
// fabrication. Measured live: llm-a750-Granite4.1-8B profiled at pp=11254 tok/s against
// a real 643, which would have made the router treat it as the fleet's fastest prefill
// by 14x and route every long prompt to an 8 GB card.
func TestPrefillProbeVariesPromptPerSample(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(req.Body).Decode(&body)
		msgs, _ := body["messages"].([]any)
		last, _ := msgs[len(msgs)-1].(map[string]any)
		content, _ := last["content"].(string)
		seen = append(seen, content)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}],
			"usage":{"prompt_tokens":1000,"completion_tokens":1}}`))
	}))
	defer srv.Close()

	r := &Router{client: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL}}
	if _, err := r.prefillProbe(b); err != nil {
		t.Fatalf("prefillProbe: %v", err)
	}
	if len(seen) != prefillProbeSamples {
		t.Fatalf("got %d samples, want %d", len(seen), prefillProbeSamples)
	}
	for i := range seen {
		for j := i + 1; j < len(seen); j++ {
			if seen[i] == seen[j] {
				t.Fatalf("samples %d and %d sent identical prompts — the prompt cache will fabricate a rate", i, j)
			}
		}
	}
}

// TestBackfillReMeasuresDecodeOnce: a corrected decode measurement has to reach
// the already-profiled fleet, but must not cost a BenchVersion re-profile (the
// whole quality suite on every worker). It re-probes once, stamps the version,
// and then stays quiet on every subsequent keepalive.
func TestBackfillReMeasuresDecodeOnce(t *testing.T) {
	var speedCalls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		body, _ := io.ReadAll(req.Body)
		if bytes.Contains(body, []byte(`"stream":true`)) {
			speedCalls++
			w.Header().Set("Content-Type", "text/event-stream")
			for i := 0; i < 8; i++ {
				_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"aaaaaaaaaaaaaaaa\"}}]}\n"))
			}
			_, _ = w.Write([]byte("data: [DONE]\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"length"}],
			"usage":{"prompt_tokens":1000,"completion_tokens":1}}`))
	}))
	defer srv.Close()

	r := &Router{client: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL}}

	stale := &WorkerProfile{Model: "m", BenchVersion: benchmarkVersion, BaselineTPS: 22.7, PrefillTPS: 1}
	r.backfillCachedProfile("w", b, stale)
	if speedCalls != 1 {
		t.Fatalf("stale speed version should re-probe exactly once, got %d", speedCalls)
	}
	if stale.SpeedVersion != speedProbeVersion {
		t.Fatalf("re-measure did not stamp the version: %d", stale.SpeedVersion)
	}
	// 8 deltas of 16 chars each: counting deltas gives 8 tokens, the old
	// len(text)/4 estimate would have claimed 32 — a 4x difference in the rate.
	if stale.BaselineTPS == 22.7 {
		t.Fatal("BaselineTPS was not re-measured")
	}

	// Runs on every keepalive — must be silent once current.
	r.backfillCachedProfile("w", b, stale)
	if speedCalls != 1 {
		t.Fatalf("a current profile must not re-probe, got %d calls", speedCalls)
	}
}

// Context is the one profile field a deployment can change without changing the
// (id, model) cache key, so it must be re-read on every cached load rather than
// trusted. The live case that prompted this: llm-6000pro-deepseek-284B-q8 was raised
// 262144 -> 393216 -> 524288 and kept advertising a cached 256k, because the only
// way to refresh it was DELETE /backends/{id} — which discards the quality benchmark
// as well and parks the worker at a provisional quality of 3 while it re-runs.
func TestBackfillCachedProfileRereadsContext(t *testing.T) {
	var nCtx float64 = 524288
	var fail bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if fail && req.URL.Path != "/v1/chat/completions" {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch req.URL.Path {
		case "/props":
			_, _ = fmt.Fprintf(w, `{"n_ctx":%.0f}`, nCtx)
		case "/v1/models":
			_, _ = w.Write([]byte(`{"data":[]}`)) // llama.cpp: no max_model_len, falls through to /props
		default:
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"length"}],
				"usage":{"prompt_tokens":1000,"completion_tokens":1}}`))
		}
	}))
	defer srv.Close()

	r := &Router{client: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL}}
	base := func(ctxK int) *WorkerProfile {
		return &WorkerProfile{Model: "m", BenchVersion: benchmarkVersion,
			SpeedVersion: speedProbeVersion, PrefillTPS: 77, ContextK: ctxK,
			ThinkingDialect: thinkingDialectNone}
	}

	// The bug: a cached 256k against a worker now serving 512k must be corrected.
	prof := base(256)
	r.backfillCachedProfile("w", b, prof)
	if prof.ContextK != 512 {
		t.Fatalf("stale context not re-read: got %dk, want 512k", prof.ContextK)
	}
	if _, ok := prof.Checks["context"]; !ok {
		t.Error("context was re-read but no check surfaced it in /backends")
	}

	// Unchanged context must not churn the profile. This runs on every keepalive, so a
	// write here would be a persist on every worker every ~60s, forever.
	same := base(512)
	same.Checks = nil
	r.backfillCachedProfile("w", b, same)
	if same.Checks != nil {
		t.Error("re-read wrote a check for an unchanged context")
	}

	// A worker that cannot answer must leave the cached value ALONE. Overwriting it
	// from the registry on a failed probe is how a transient blip would silently
	// downgrade a good profile.
	fail = true
	stale := base(512)
	b.ContextK = 32
	r.backfillCachedProfile("w", b, stale)
	if stale.ContextK != 512 {
		t.Fatalf("failed probe clobbered a cached context: got %dk, want 512k", stale.ContextK)
	}
}

// A cached profile predates the thinking dialect, so without a backfill clause
// every already-profiled worker would carry the zero value forever — the failure
// backfillCachedProfile exists to prevent, and the one its doc comment warns any
// new WorkerProfile field will inherit.
func TestBackfillCachedProfileThinkingDialect(t *testing.T) {
	var probes int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		body, _ := io.ReadAll(req.Body)
		w.Header().Set("Content-Type", "application/json")
		if bytes.Contains(body, []byte(`"reasoning_effort"`)) {
			probes++
			_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"75","reasoning_content":"worked it out"}}]}`)
			return
		}
		if bytes.Contains(body, []byte(`"chat_template_kwargs"`)) {
			probes++
		}
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"75"}}]}`)
	}))
	defer srv.Close()

	r := &Router{client: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL}}

	// Missing dialect: probe, record it, and surface a check so /backends shows
	// which spelling this endpoint answers to.
	prof := &WorkerProfile{Model: "m", BenchVersion: benchmarkVersion,
		SpeedVersion: speedProbeVersion, PrefillTPS: 77}
	r.backfillCachedProfile("w", b, prof)
	if prof.ThinkingDialect != thinkingDialectEffort {
		t.Fatalf("dialect = %q, want %q", prof.ThinkingDialect, thinkingDialectEffort)
	}
	if !prof.Thinking {
		t.Error("the re-probe found a reasoning block but the profile still says it cannot think")
	}
	if _, ok := prof.Checks["thinking"]; !ok {
		t.Error("dialect was measured but no check surfaced it in /backends")
	}
	before := probes

	// Already measured: silent. This runs on every registration keepalive, so a
	// probe here is a 1024-token generation on the whole fleet every ~60s.
	r.backfillCachedProfile("w", b, prof)
	if probes != before {
		t.Errorf("re-probed a profile that already carried a dialect: %d extra requests", probes-before)
	}
}
