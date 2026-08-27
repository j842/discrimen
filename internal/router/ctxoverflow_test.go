package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// A prompt too long for the worker's window is a MISS, not an outage.
//
// This is what makes the long-context questions measure anything. An errored
// question leaves the denominator — deliberately, so a profile interrupted by an
// outage is not reported as incompetence — so a 32K worker asked a 48K question
// came back neither right nor wrong but UNMEASURED, indistinguishable from one
// the profile budget never reached. "Cannot reason over 48K of context" is the
// exact weakness the set exists to find, and that hid it.
func TestAnOversizedPromptScoresAsAMissNotAnError(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	long := strings.Repeat("field log line with some padding text ", 4000) // ~150k chars
	benchmarkQuestions = []benchmarkQ{
		{Tier: 1, Prompt: "What is 2+2?", Expect: "4", Match: "numeric"},
		{Tier: 9, Prompt: long + " How many entries are there?", Expect: "4000", Match: "numeric"},
	}

	var asked atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		asked.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]any{"role": "assistant", "content": "4"}, "finish_reason": "stop",
			}},
		})
	}))
	defer srv.Close()

	// A small-window worker. ContextK is what usableContextTokens reads when no
	// probe has measured better.
	b := &Backend{BackendRegistration: BackendRegistration{ID: "small", URL: srv.URL, Model: "m", ContextK: 8}}
	r := &Router{cfg: &Config{}, benchClient: &http.Client{}}

	_, ok, _, _, details := r.runQualityBenchmark(b, 2)
	if !ok {
		t.Fatal("the run did not score at all")
	}
	if len(details) != 2 {
		t.Fatalf("got %d results, want 2", len(details))
	}

	short, oversized := details[0], details[1]
	if !short.Pass {
		t.Errorf("the question that FITS was not answered: %+v", short)
	}
	if oversized.Errored {
		t.Error("the oversized question was recorded as an OUTAGE — it leaves the denominator and the weakness disappears")
	}
	if oversized.Skipped {
		t.Error("the oversized question was recorded as never asked")
	}
	if oversized.Pass {
		t.Error("the oversized question was scored as PASSED on a worker that cannot read it")
	}
	if !strings.Contains(oversized.Got, "window") {
		t.Errorf("the result does not say why it failed: %q", oversized.Got)
	}
	// And it must not have cost a generation: the verdict follows from the
	// advertised window, so asking would spend a real request to learn a known
	// answer — on a 48K prompt, an expensive one.
	if got := asked.Load(); got != 1 {
		t.Errorf("%d requests sent, want 1 — the oversized prompt was dispatched anyway", got)
	}
}

// A worker whose window is not yet known must NOT have questions failed against
// it. usableContextTokens returns 0 mid-profile, and guessing a miss from a
// missing number would fail questions a worker can answer.
func TestAnUnknownWindowDoesNotFailQuestions(t *testing.T) {
	saved := benchmarkQuestions
	defer func() { benchmarkQuestions = saved }()
	long := strings.Repeat("padding ", 20000)
	benchmarkQuestions = []benchmarkQ{{Tier: 9, Prompt: long + " Answer?", Expect: "7", Match: "numeric"}}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]any{"role": "assistant", "content": "7"}, "finish_reason": "stop",
			}},
		})
	}))
	defer srv.Close()

	b := &Backend{BackendRegistration: BackendRegistration{ID: "unknown", URL: srv.URL, Model: "m"}} // ContextK 0
	r := &Router{cfg: &Config{}, benchClient: &http.Client{}}

	if _, ok, _, _, details := r.runQualityBenchmark(b, 1); !ok || len(details) != 1 {
		t.Fatalf("run did not score: ok=%v details=%d", ok, len(details))
	} else if !details[0].Pass {
		t.Errorf("a worker with an unmeasured window was failed on length alone: %+v", details[0])
	}
}
