package router

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// Every router-originated generation must count against ActiveRequests, because
// expectedLatency, relayOccupancy and the Concurrency log column all read that
// counter. The capacity ramp is the sharp case: it fires up to CapacityProbeMax
// concurrent generations at a worker that would otherwise read as idle, so the
// ranker prices it as free exactly while a probe is saturating it.
//
// Asserted at doCompletion rather than at one caller: the counter was first added
// around benchCompletion alone, which left the judge, the capability probes and
// the capacity ramp still invisible.
func TestEveryRouterOriginatedCompletionCountsAsActive(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "w", URL: "http://x", Model: "m", MaxConcurrency: 4, TTLSeconds: 3600})

	seen := make(chan int, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		seen <- reg.activeCount("w") // read from INSIDE the call: it must be counted now, not after
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "ok"}}},
		})
	}))
	defer srv.Close()

	r := &Router{cfg: &Config{}, registry: reg, client: srv.Client(), benchClient: srv.Client()}
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w", URL: srv.URL, Model: "m"}}
	payload := map[string]any{"model": "m", "messages": []map[string]string{{"role": "user", "content": "hi"}}}

	// The three entry points every background caller uses: the judge and vision go
	// through rawCompletion, the capability and capacity probes through
	// simpleCompletion, the benchmark through benchCompletion.
	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"rawCompletion", func() error { _, err := r.rawCompletion(b, payload); return err }},
		{"simpleCompletion", func() error { _, err := r.simpleCompletion(b, payload); return err }},
		{"benchCompletion", func() error { _, err := r.benchCompletion(context.Background(), b, payload); return err }},
	} {
		if err := tc.call(); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if got := <-seen; got != 1 {
			t.Errorf("%s: ActiveRequests during the call = %d, want 1", tc.name, got)
		}
		if got := reg.activeCount("w"); got != 0 {
			t.Errorf("%s: ActiveRequests after the call = %d, want 0 — the counter is unbalanced", tc.name, got)
		}
	}
}

// A missing slot channel means "uncapped", which is only true while the backend
// exists. remove() deletes the channel, so without an existence check a worker
// being decommissioned reads as INFINITE free capacity to every request already
// queued in pickAndAcquire — deregistering made it maximally attractive.
func TestRemovedBackendIsNotInfiniteCapacity(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "gone", URL: "http://x", Model: "m", MaxConcurrency: 1, TTLSeconds: 3600})
	if _, ok := reg.tryAcquireSlot("gone"); !ok {
		t.Fatal("a registered backend with a free slot should acquire")
	}
	reg.releaseSlot(nil)

	reg.remove("gone")
	if slot, ok := reg.tryAcquireSlot("gone"); ok {
		t.Errorf("acquired a slot on a removed backend (slot=%v) — it reads as uncapped", slot)
	}
	if reg.stillRoutable("gone") {
		t.Error("a removed backend is still reported routable")
	}
}

// Workers beacon with max_concurrency: 0 — they do not know their own capacity,
// which is why capacityProbe exists. A re-registration that declares nothing must
// not retract the cap the router measured, or a worker restarting with a changed
// model takes its whole waiting queue at once while its model is still loading.
func TestReregisteringDoesNotRetractMeasuredCapacity(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "w", URL: "http://x", Model: "m", MaxConcurrency: 2, TTLSeconds: 3600})

	// Both slots taken; the worker is genuinely full.
	a, ok1 := reg.tryAcquireSlot("w")
	bslot, ok2 := reg.tryAcquireSlot("w")
	if !ok1 || !ok2 {
		t.Fatal("could not take the two declared slots")
	}
	if _, ok := reg.tryAcquireSlot("w"); ok {
		t.Fatal("acquired a third slot on a 2-slot worker")
	}

	// A content change arrives declaring no cap, as every beacon does.
	reg.upsert(BackendRegistration{ID: "w", URL: "http://y", Model: "m", MaxConcurrency: 0, TTLSeconds: 3600})

	if slot, ok := reg.tryAcquireSlot("w"); ok {
		t.Errorf("admission became unbounded after a re-registration declaring no cap (slot=%v)", slot)
	}
	reg.releaseSlot(a)
	reg.releaseSlot(bslot)
	if _, ok := reg.tryAcquireSlot("w"); !ok {
		t.Error("slots never came back after release — the channel was replaced, not preserved")
	}
}

// Waiting is only right for a worker that is busy: a slot will free. A worker
// that has gone away will never hand one back, so a request pinned to it should
// fail on the first scan rather than sit for the full slotMaxWait.
func TestAcquireFailsFastWhenNoCandidateRemains(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "w", URL: "http://x", Model: "m", MaxConcurrency: 1, TTLSeconds: 3600})
	b := reg.get("w")
	if _, ok := reg.tryAcquireSlot("w"); !ok {
		t.Fatal("could not saturate w")
	}
	reg.remove("w")

	done := make(chan error, 1)
	go func() {
		_, _, err := (&Router{registry: reg}).pickAndAcquire(context.Background(), []*Backend{b})
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("acquired a backend that no longer exists")
		}
	case <-time.After(3 * time.Second):
		t.Fatalf("pickAndAcquire is still waiting for a backend that is gone (slotMaxWait is %s)", slotMaxWait)
	}
}
