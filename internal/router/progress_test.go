package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// The progress indicator has to be verified against a REAL benchmark run, not
// against direct calls to enter/step. Every method here is nil-safe by design,
// which means a tracker that was never wired in — the wrong id, a nil lookup, a
// phase call that sits on a path the profile does not take — reports a
// permanently empty progress and no error anywhere. That is the exact shape of
// bug this codebase keeps producing: implemented, tested, and inert.
//
// So: run the actual quality benchmark against a fake worker that blocks until
// the test has looked, and assert the numbers moved.
func TestProfileProgressTracksARealBenchmark(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	benchmarkQuestions = []benchmarkQ{
		{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Match: "numeric"},
		{Tier: 1, Prompt: "What is 3+3?", Expect: "6", Match: "numeric"},
		{Tier: 1, Prompt: "What is 4+4?", Expect: "8", Match: "numeric"},
		{Tier: 1, Prompt: "What is 5+5?", Expect: "10", Match: "numeric"},
	}

	// The worker holds every answer until `release` closes, so the test observes
	// the mid-flight state deterministically instead of racing a fast fake.
	release := make(chan struct{})
	var inFlight atomic.Int64
	arrived := make(chan struct{}, len(benchmarkQuestions))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		inFlight.Add(1)
		arrived <- struct{}{}
		<-release
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message":       map[string]any{"role": "assistant", "content": "0"},
				"finish_reason": "stop",
			}},
		})
	}))
	defer srv.Close()

	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "w", URL: srv.URL, Model: "m", MaxConcurrency: 4, TTLSeconds: 3600})
	r := &Router{cfg: &Config{}, registry: reg, benchClient: &http.Client{}}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL, Model: "m"}}

	prog := newProfileProgress()
	r.profiling.Store("w", prog)

	// The phase is entered by profileBackend, which we are not running; enter it
	// here exactly as profileBackend does, so what is under test is whether the
	// benchmark itself reports into the tracker.
	prog.enter(phaseQuality, len(benchmarkQuestions))

	done := make(chan struct{})
	go func() { defer close(done); r.runQualityBenchmark(b, 2) }()

	// Two questions in flight (benchConc=2), none finished.
	<-arrived
	<-arrived
	waitFor(t, func() bool { return prog.snapshot().InFlight == 2 }, "two generations never showed as in flight")
	v := prog.snapshot()
	if v.Phase != phaseQuality {
		t.Errorf("phase=%q, want %q", v.Phase, phaseQuality)
	}
	if v.Total != len(benchmarkQuestions) {
		t.Errorf("total=%d, want %d", v.Total, len(benchmarkQuestions))
	}
	if v.Done != 0 {
		t.Errorf("done=%d before any answer returned, want 0", v.Done)
	}

	// The load the benchmark is putting on the worker must show up in
	// ActiveRequests, or `ask -l` renders s=0/4 while the GPU is saturated and
	// the router's own latency estimates price the worker as idle.
	if got := reg.activeCount("w"); got != 2 {
		t.Errorf("ActiveRequests=%d during profiling, want 2", got)
	}

	close(release)
	<-done

	final := prog.snapshot()
	if final.Done != len(benchmarkQuestions) {
		t.Errorf("done=%d after the run, want %d", final.Done, len(benchmarkQuestions))
	}
	if final.InFlight != 0 {
		t.Errorf("in_flight=%d after the run, want 0 — begin/end are unbalanced", final.InFlight)
	}
	if got := reg.activeCount("w"); got != 0 {
		t.Errorf("ActiveRequests=%d after profiling, want 0 — incActive is unbalanced", got)
	}
}

// The tracker must reach /backends, which is the only place ask can read it.
func TestHandleBackendsPublishesProfileProgress(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "w", URL: "http://a", Model: "m", MaxConcurrency: 1, TTLSeconds: 3600})
	r := &Router{cfg: &Config{}, registry: reg, logs: newTestLogStore(t)}
	issueKey(t, r, adminSecret, apiKey{Role: roleAdmin, Name: "admin"})
	get := func() []byte {
		t.Helper()
		req := httptest.NewRequest(http.MethodGet, "/backends", nil)
		req.Header.Set("Authorization", "Bearer "+adminSecret)
		rec := httptest.NewRecorder()
		r.handleBackends(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("GET /backends: %d %s", rec.Code, rec.Body.String())
		}
		return rec.Body.Bytes()
	}

	var before struct {
		Backends []struct {
			Profiling bool                 `json:"profiling"`
			Progress  *ProfileProgressView `json:"profile_progress"`
		} `json:"backends"`
	}
	if err := json.Unmarshal(get(), &before); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(before.Backends) != 1 {
		t.Fatalf("got %d backends, want 1", len(before.Backends))
	}
	if before.Backends[0].Progress != nil {
		t.Error("profile_progress present on a worker that is not profiling")
	}

	prog := newProfileProgress()
	prog.enter(phaseQuality, 98)
	for i := 0; i < 12; i++ {
		prog.step()
	}
	prog.begin()
	r.profiling.Store("w", prog)

	var after struct {
		Backends []struct {
			Profiling bool                 `json:"profiling"`
			Progress  *ProfileProgressView `json:"profile_progress"`
		} `json:"backends"`
	}
	if err := json.Unmarshal(get(), &after); err != nil {
		t.Fatalf("decode: %v", err)
	}
	got := after.Backends[0].Progress
	if !after.Backends[0].Profiling {
		t.Fatal("profiling flag not set")
	}
	if got == nil {
		t.Fatal("profile_progress absent while profiling — ask can only ever print the bare word")
	}
	if got.Phase != phaseQuality || got.Done != 12 || got.Total != 98 || got.InFlight != 1 {
		t.Errorf("published %+v, want phase=%s done=12 total=98 in_flight=1", got, phaseQuality)
	}
}

// An ETA is worse than no ETA when it is invented from two samples, so it stays
// absent until there is enough of the phase behind it to extrapolate from.
func TestProfileProgressETAWaitsForSamples(t *testing.T) {
	p := newProfileProgress()
	p.enter(phaseQuality, 100)
	p.phaseAt.Store(time.Now().Add(-time.Minute).UnixNano()) // a minute of real work behind it
	for i := 0; i < profileETAMinSamples-1; i++ {
		p.step()
	}
	if v := p.snapshot(); v.RemainingMS != 0 {
		t.Errorf("published an ETA from %d samples (%dms)", v.Done, v.RemainingMS)
	}
	p.step()
	if v := p.snapshot(); v.RemainingMS <= 0 {
		t.Errorf("no ETA at %d samples, want one", v.Done)
	}
	// A phase with no countable unit never gets one, however long it runs.
	p.enter(phaseCapacity, 0)
	p.phaseAt.Store(time.Now().Add(-time.Minute).UnixNano())
	if v := p.snapshot(); v.RemainingMS != 0 || v.Total != 0 {
		t.Errorf("uncountable phase published total=%d eta=%dms", v.Total, v.RemainingMS)
	}
}

// The ETA divides by the CURRENT phase's elapsed time. Charging quality with the
// minutes the capacity ramp took would inflate per-question cost and project an
// ETA far past the truth — and the phases before quality are the slow ones.
func TestProfileProgressETAIsPerPhase(t *testing.T) {
	p := newProfileProgress()
	p.startedAt = time.Now().Add(-30 * time.Minute) // a long capacity ramp behind us
	p.enter(phaseQuality, 100)
	p.phaseAt.Store(time.Now().Add(-10 * time.Second).UnixNano())
	for i := 0; i < 10; i++ {
		p.step()
	}
	v := p.snapshot()
	// 10s for 10 questions -> ~1s each -> ~90s left. If it used startedAt the
	// answer would be in the hours.
	if v.RemainingMS < 60_000 || v.RemainingMS > 120_000 {
		t.Errorf("remaining=%dms, want ~90s — the ETA is not using the phase clock", v.RemainingMS)
	}
	if v.ElapsedMS < 29*60*1000 {
		t.Errorf("elapsed=%dms, want the whole profile (~30m)", v.ElapsedMS)
	}
}
