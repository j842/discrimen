package router

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// abortingWorker answers the quick probes — billing for every one of them — and
// then fails the capacity probe, which is the cheap end of how a background
// profile aborts after spending real money. (The expensive end is the quality
// benchmark, which only gives up after all 130 questions have been generated.)
func abortingWorker(t *testing.T, promptTokens, completionTokens int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.Method == http.MethodGet {
			http.NotFound(w, req)
			return
		}
		body, _ := io.ReadAll(req.Body)
		if bytes.Contains(body, []byte("two sentences about the ocean")) {
			http.Error(w, `{"error":{"message":"rate limited"}}`, http.StatusTooManyRequests)
			return
		}
		if bytes.Contains(body, []byte(`"tools"`)) || bytes.Contains(body, []byte(`"response_format"`)) {
			http.Error(w, `{"error":{"message":"unsupported"}}`, http.StatusBadRequest)
			return
		}
		usage := fmt.Sprintf(`"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}`,
			promptTokens, completionTokens, promptTokens+completionTokens)
		if bytes.Contains(body, []byte(`"stream":true`)) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok ok ok ok\"}}]}\n\n")
			fmt.Fprintf(w, "data: {\"choices\":[],%s}\n\n", usage)
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		fmt.Fprintf(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],%s}`, usage)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// A discarded run still spent the money. Only the success path writes the token
// counts onto a profile, so an abort used to lose the number completely: an
// endpoint that fails this way every time can run the whole benchmark, bill for
// it, be thrown away, and repeat, with ProfileCost recorded as zero.
func TestAbortedProfileRecordsWhatItSpent(t *testing.T) {
	old := capacityProbeRetryDelay
	capacityProbeRetryDelay = 0 // the ladder is what's under test, not the wait
	defer func() { capacityProbeRetryDelay = old }()

	srv := abortingWorker(t, 1000, 40)
	r, b := profileCostRouter(t, srv, 3, 15)

	prof, err := r.profileBackend(b, "m")
	if prof != nil || err == nil {
		t.Fatalf("the run should have been discarded: profile=%+v err=%v", prof, err)
	}
	var aborted *abortedProfile
	if !errors.As(err, &aborted) {
		t.Fatalf("the abort does not carry what it spent: %v", err)
	}
	if aborted.prompt <= 0 || aborted.output <= 0 {
		t.Fatalf("a run that reached the capacity probe cannot have consumed nothing: %d prompt / %d completion",
			aborted.prompt, aborted.output)
	}
	want := float64(aborted.prompt)/1e6*3 + float64(aborted.output)/1e6*15
	if math.Abs(aborted.cost-want) > 1e-9 {
		t.Errorf("cost = %g, want %g at the row's declared prices", aborted.cost, want)
	}
	// And it has to be legible where an operator will meet it — the log line and
	// the check the caller writes both render this string.
	if msg := err.Error(); !strings.Contains(msg, "spent") || !strings.Contains(msg, "at declared prices") {
		t.Errorf("the spend is not in the error an operator sees: %q", msg)
	}
	if !strings.Contains(err.Error(), "capacity probe") {
		t.Errorf("the abort does not say what failed: %q", err.Error())
	}
	if _, still := r.profileMeters.Load("metered"); still {
		t.Error("the profile meter outlived the aborted run")
	}
}

// The delay between attempts has to grow, for the same reason recertBackoff's
// does: a failure that repeats should cost less each time.
func TestProfileRetryBackoff(t *testing.T) {
	cases := []struct {
		aborts int
		want   time.Duration
	}{
		{0, profileRetryDelay}, // defensive: treated as the first attempt
		{1, profileRetryDelay},
		{2, 4 * time.Minute},
		{3, 8 * time.Minute},
		{9, time.Hour}, // capped
		{99, time.Hour},
	}
	for _, c := range cases {
		if got := profileRetryBackoff(c.aborts); got != c.want {
			t.Errorf("profileRetryBackoff(%d)=%s, want %s", c.aborts, got, c.want)
		}
	}
}

// The loop this bounds costs real money on every pass — quick profile, capacity
// ramp, 130-question benchmark, discard, wait, repeat — and used to run for ever
// on a fixed two-minute timer with nothing in front of the operator.
func TestProfileRetriesAreBoundedBackedOffAndVisible(t *testing.T) {
	reg := newTestRegistry()
	registerQ(t, reg, "metered", 30, 2)
	r := &Router{cfg: &Config{}, registry: reg}
	cause := &abortedProfile{
		reason: "worker unreachable during quality benchmark",
		prompt: 120_000, output: 300_000, cost: 4.86,
	}

	var delays []time.Duration
	for i := 0; i < profileRetryMaxAttempts; i++ {
		delays = append(delays, r.scheduleProfileRetry("metered", cause))
	}
	for i := 0; i < len(delays)-1; i++ {
		if delays[i] <= 0 {
			t.Fatalf("attempt %d was not retried at all (%s)", i+1, delays[i])
		}
		if i > 0 && delays[i] <= delays[i-1] {
			t.Errorf("attempt %d did not back off: %s then %s", i+1, delays[i-1], delays[i])
		}
	}
	if delays[0] != profileRetryDelay {
		t.Errorf("first retry after %s, want %s", delays[0], profileRetryDelay)
	}
	if last := delays[len(delays)-1]; last != 0 {
		t.Fatalf("the router kept paying for attempt %d (scheduled in %s) — repeated failure has to stop",
			profileRetryMaxAttempts, last)
	}

	// Giving up is not the same as going quiet: the row says so, with what the
	// attempts cost and how to ask for another.
	check := reg.get("metered").Certification.Checks["profile"]
	if check.OK {
		t.Error("an abandoned profile is not a passing check")
	}
	for _, want := range []string{"abandoned", "provisional", "at declared prices"} {
		if !strings.Contains(check.Message, want) {
			t.Errorf("the check does not mention %q: %q", want, check.Message)
		}
	}

	// A run that finally works clears the count, so a later blip gets the full
	// allowance again rather than one attempt.
	reg.clearProfileAborts("metered")
	if got := r.scheduleProfileRetry("metered", cause); got != profileRetryDelay {
		t.Errorf("after a successful profile the next abort waits %s, want %s", got, profileRetryDelay)
	}
}
