package router

import (
	"fmt"
	"net/http"
	"sync"
	"testing"
)

// A busy worker is a queue, not a wrong answer. These pin the classification and
// the circuit breaker, because both failure modes are silent: treating busy as
// wrong records questions the model never saw as failures, and retrying forever
// hangs a profile on a worker that is simply gone.
func TestBenchBusyStatus(t *testing.T) {
	busy := []int{http.StatusNotFound, http.StatusRequestTimeout, http.StatusTooManyRequests,
		http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout}
	for _, code := range busy {
		if !benchBusyStatus(fmt.Errorf("completion returned %d: upstream", code)) {
			t.Errorf("status %d: want busy", code)
		}
	}
	// A definitive rejection is the model's answer about the request, not congestion.
	for _, code := range []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusInternalServerError} {
		if benchBusyStatus(fmt.Errorf("completion returned %d: nope", code)) {
			t.Errorf("status %d: want NOT busy", code)
		}
	}
	if benchBusyStatus(nil) {
		t.Error("nil error must not be busy")
	}
	if benchBusyStatus(fmt.Errorf("dial tcp: connection refused")) {
		t.Error("a transport error is not a busy STATUS — it takes the ordinary retry path")
	}
}

func TestBenchBusyTrackerBreaksTheStreak(t *testing.T) {
	tr := &benchBusyTracker{}
	for i := 1; i < benchBusyMaxStreak; i++ {
		if tr.busy() {
			t.Fatalf("gave up after %d busy responses, want %d", i, benchBusyMaxStreak)
		}
	}
	if !tr.busy() {
		t.Fatalf("still going after %d consecutive busy responses", benchBusyMaxStreak)
	}
	if !tr.abandoned() {
		t.Fatal("abandoned() disagrees with busy()")
	}
}

// The streak is CONSECUTIVE: one answer proves the worker is alive and the
// patience resets. Without this a long, healthy run would eventually accumulate
// enough scattered busies to abandon a perfectly good worker.
func TestBenchBusyTrackerResetsOnSuccess(t *testing.T) {
	tr := &benchBusyTracker{}
	for i := 0; i < benchBusyMaxStreak-1; i++ {
		tr.busy()
	}
	tr.ok()
	for i := 0; i < benchBusyMaxStreak-1; i++ {
		if tr.busy() {
			t.Fatal("streak was not cleared by a successful answer")
		}
	}
}

// Every question in a run shares one tracker, so it must be safe under the
// bounded concurrency runQualityBenchmark grades with.
func TestBenchBusyTrackerConcurrent(t *testing.T) {
	tr := &benchBusyTracker{}
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); tr.busy(); tr.ok() }()
	}
	wg.Wait()
	if tr.abandoned() {
		t.Error("interleaved busy/ok abandoned the run")
	}
}
