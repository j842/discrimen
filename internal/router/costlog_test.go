package router

import (
	"bytes"
	"context"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// costLogBuf is a log sink that is safe to read while the request's background
// bookkeeping goroutine may still be writing to it.
type costLogBuf struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *costLogBuf) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *costLogBuf) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func captureRouterLog(t *testing.T) *costLogBuf {
	t.Helper()
	sink := &costLogBuf{}
	prevOut, prevFlags := log.Writer(), log.Flags()
	log.SetOutput(sink)
	log.SetFlags(0)
	t.Cleanup(func() { log.SetOutput(prevOut); log.SetFlags(prevFlags) })
	return sink
}

const costTestAnswer = `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":4,"completion_tokens":1}}`

func costTestWorker(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(costTestAnswer))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// costSpillRouter builds the fleet shape the cost line exists for: one LOCAL,
// FREE worker with a single slot, and one metered worker beside it. Both serve
// "default", so the router chooses and the plan is a "route".
func costSpillRouter(t *testing.T) *Router {
	t.Helper()
	free, paid := costTestWorker(t), costTestWorker(t)
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{
		ID: "free", URL: free.URL, Model: "default", Quality: 80,
		BaselineTPS: 100, MaxConcurrency: 1, TTLSeconds: 3600, Features: []string{"chat"},
	})
	reg.finishCertification("free", true, map[string]Check{}, 100, 10, "")
	reg.upsert(BackendRegistration{
		ID: "paid", URL: paid.URL, Model: "default", Quality: 80,
		BaselineTPS: 100, MaxConcurrency: 1, TTLSeconds: 3600, Features: []string{"chat"},
		InputPricePerMtok: 3, OutputPricePerMtok: 15,
	})
	reg.finishCertification("paid", true, map[string]Check{}, 100, 10, "")

	logs, err := openLogStore(t.TempDir()+"/logs.sqlite", 16384, "")
	if err != nil {
		t.Fatalf("open log store: %v", err)
	}
	t.Cleanup(func() { logs.Close() })
	return &Router{
		cfg:      &Config{DefaultMaxTokens: 4096},
		registry: reg,
		client:   &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
		logs:     logs,
		sessions: newSessionTracker(time.Hour, 16),
	}
}

// settle waits for the request's background bookkeeping goroutine, whose last
// act is the log insert. Without it the store is closed underneath that
// goroutine on cleanup and the test's own log capture races it.
func settle(t *testing.T, r *Router) {
	t.Helper()
	waitFor(t, func() bool {
		rows, err := r.logs.List(context.Background(), "", 10, 0)
		return err == nil && len(rows) > 0
	}, "the request was never logged")
}

// TestCostLineFiresWhenLocalFreeSpillsToPaid: money spent because no FREE worker
// freed a slot has to be reported, and the top rung of the ladder was
// "local-free" rather than "free-first".
//
// This is the ordinary fleet: local workers that cost nothing plus a metered
// endpoint. qualityFloorPreference then picks `local-free` as its first choice
// (free ∧ not relayed is a strict, non-empty subset), and the cost line was gated
// on pref.why == "free-first" alone — the rung that is only reached when there is
// no free LOCAL worker to prefer. So on the one fleet shape where a spill costs
// real money, the log said nothing, while the paragraph above the check said to
// read the facts off the worker actually landed on rather than off the
// preference.
func TestCostLineFiresWhenLocalFreeSpillsToPaid(t *testing.T) {
	withQualityFloorWait(t, 60*time.Millisecond)
	r := costSpillRouter(t)

	// The preference the request will be given, and the rung it starts on.
	pref := qualityFloorPreference(r.registry.eligible(), 0, false, 0)
	if pref.why != "local-free" {
		t.Fatalf("this test needs the local-free rung, got %q", pref.why)
	}

	// Hold the free worker's only slot so nothing free can be acquired.
	held, ok := r.registry.tryAcquireSlot("free")
	if !ok {
		t.Fatal("could not saturate the free worker")
	}
	defer r.registry.releaseSlot(held)

	sink := captureRouterLog(t)
	rec := runChat(t, r, `{"model":"default","stream":false,"messages":[{"role":"user","content":"hi"}]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("X-LLM-Backend-ID"); got != "paid" {
		t.Fatalf("the free worker was saturated, so the spill must serve on paid, got %q", got)
	}
	if got := sink.String(); !strings.Contains(got, "cost:") || !strings.Contains(got, "PAID backend=paid") {
		t.Fatalf("spending money past the grace was not reported; log was:\n%s", got)
	}
	settle(t, r)
}

// TestCostLineSilentWhenTheSpillIsStillFree: the mirror case. A missed
// preference that lands on another FREE worker has spent nothing, so it must not
// be reported as a purchase.
func TestCostLineSilentWhenTheSpillIsStillFree(t *testing.T) {
	withQualityFloorWait(t, 60*time.Millisecond)
	r := costSpillRouter(t)
	// Make the metered row free after the fact, so the fleet is two free workers
	// and the spill costs nothing.
	r.registry.mu.Lock()
	r.registry.backends["paid"].InputPricePerMtok = 0
	r.registry.backends["paid"].OutputPricePerMtok = 0
	r.registry.mu.Unlock()

	held, ok := r.registry.tryAcquireSlot("free")
	if !ok {
		t.Fatal("could not saturate the first worker")
	}
	defer r.registry.releaseSlot(held)

	sink := captureRouterLog(t)
	rec := runChat(t, r, `{"model":"default","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if strings.Contains(sink.String(), "cost:") {
		t.Fatalf("a spill onto a free worker must not be reported as spending; log was:\n%s", sink.String())
	}
	settle(t, r)
}
