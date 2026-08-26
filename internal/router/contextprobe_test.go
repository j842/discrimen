package router

import (
	"strings"
	"testing"
)

// The haystack has to actually contain the needle, at roughly the requested
// depth, and be roughly the requested size. Each of those failing is silent: a
// missing needle makes every worker look broken, a mis-sized haystack makes the
// ladder measure a different length than it reports.
func TestContextNeedlePrompt(t *testing.T) {
	for _, size := range []int{4096, 16384, 65536} {
		for _, depth := range contextProbeDepths {
			prompt, want := contextNeedlePrompt(size, depth, 42)
			if !strings.Contains(prompt, want) {
				t.Fatalf("size=%d depth=%.1f: needle %q absent from the haystack", size, depth, want)
			}
			if strings.Count(prompt, want) != 1 {
				t.Errorf("size=%d depth=%.1f: needle appears %d times; a second copy makes the probe unfalsifiable",
					size, depth, strings.Count(prompt, want))
			}
			// ~4 chars/token, so allow a wide band — the ladder doubles, and a
			// 25% error can never confuse one rung with the next.
			est := len(prompt) / 4
			if est < size/2 || est > size*2 {
				t.Errorf("size=%d: haystack estimates %d tokens, too far off to be labelled %d", size, est, size)
			}
			// Position: the needle should sit near the requested depth.
			at := float64(strings.Index(prompt, want)) / float64(len(prompt))
			if at < depth-0.25 || at > depth+0.25 {
				t.Errorf("size=%d depth=%.1f: needle landed at %.2f", size, depth, at)
			}
		}
	}
}

// Digits in the filler would give a model that ignored the question something
// plausible to return, and containment-grading would count it.
func TestContextFillerHasNoDigits(t *testing.T) {
	for _, f := range contextFiller {
		if strings.ContainsAny(f, "0123456789") {
			t.Errorf("filler line contains digits, which could be mistaken for the needle: %q", f)
		}
	}
}

// Same seed must produce the same question, or a re-probe measures a different
// thing and a change in the result cannot be attributed to the worker.
func TestContextNeedleDeterministic(t *testing.T) {
	a, wa := contextNeedlePrompt(8192, 0.5, 7)
	b, wb := contextNeedlePrompt(8192, 0.5, 7)
	if a != b || wa != wb {
		t.Error("same seed produced a different probe")
	}
	if _, wc := contextNeedlePrompt(8192, 0.5, 8); wc == wa {
		t.Error("different seeds produced the same needle; the ladder would reuse one value throughout")
	}
}

// Grading measures RETRIEVAL, not instruction-following. A model that retrieves
// the fact but wraps it in a sentence has passed the thing being tested.
func TestContextNeedleFound(t *testing.T) {
	cases := []struct {
		got  string
		want bool
	}{
		{"481920", true},
		{"The access code is 481920.", true},
		{"481,920", true},   // thousands separator
		{" 481920\n", true}, // whitespace
		{"I could not find an access code.", false},
		{"481921", false}, // near miss is a miss
		{"", false},
	}
	for _, c := range cases {
		if got := contextNeedleFound(c.got, "481920"); got != c.want {
			t.Errorf("contextNeedleFound(%q) = %v, want %v", c.got, got, c.want)
		}
	}
}

// usableContextTokens decides what routing filters on, so its fallbacks matter
// more than the happy path: an unprobed worker must not be filtered out of
// everything, and a probe must never license exceeding the server's window.
func TestUsableContextTokens(t *testing.T) {
	adv := &Backend{BackendRegistration: BackendRegistration{ContextK: 256}}
	if got := usableContextTokens(adv); got != 256*1024 {
		t.Errorf("unprobed worker = %d, want the advertised %d — it would be filtered out of everything", got, 256*1024)
	}
	probed := &Backend{BackendRegistration: BackendRegistration{ContextK: 256}, ContextProbe: &ContextProbe{UsableTokens: 64 * 1024}}
	if got := usableContextTokens(probed); got != 64*1024 {
		t.Errorf("probed worker = %d, want the measured 64K", got)
	}
	// A probe cannot authorise more than the server was configured for.
	over := &Backend{BackendRegistration: BackendRegistration{ContextK: 32}, ContextProbe: &ContextProbe{UsableTokens: 256 * 1024}}
	if got := usableContextTokens(over); got != 32*1024 {
		t.Errorf("probe above advertised = %d, want the advertised 32K", got)
	}
	// Failed at the smallest rung: UsableTokens 0 means "not established", and
	// falling back to the claim is the conservative reading — the alternative
	// removes the worker from all routing on one probe.
	failed := &Backend{BackendRegistration: BackendRegistration{ContextK: 128}, ContextProbe: &ContextProbe{UsableTokens: 0}}
	if got := usableContextTokens(failed); got != 128*1024 {
		t.Errorf("failed probe = %d, want fallback to advertised", got)
	}
}

func TestContextProbeMessage(t *testing.T) {
	if got := contextProbeMessage(nil); got != "not probed" {
		t.Errorf("nil probe = %q", got)
	}
	gap := contextProbeMessage(&ContextProbe{AdvertisedTokens: 256 * 1024, UsableTokens: 32 * 1024})
	if !strings.Contains(gap, "ADVERTISED") {
		t.Errorf("a 32K-of-256K result should call out the gap, got %q", gap)
	}
	ok := contextProbeMessage(&ContextProbe{AdvertisedTokens: 32 * 1024, UsableTokens: 16 * 1024})
	if strings.Contains(ok, "ADVERTISED") {
		t.Errorf("16K of 32K is the last rung that fits and should not read as a discrepancy, got %q", ok)
	}
	trunc := contextProbeMessage(&ContextProbe{AdvertisedTokens: 256 * 1024, UsableTokens: 16 * 1024, Truncated: true})
	if !strings.Contains(trunc, "at least") {
		t.Errorf("a budget-truncated ladder is a lower bound and must say so, got %q", trunc)
	}
	none := contextProbeMessage(&ContextProbe{AdvertisedTokens: 128 * 1024,
		Levels: []ContextProbeLevel{{Tokens: 4096, Passed: 1, Total: 3}}})
	if !strings.Contains(none, "FAILED") {
		t.Errorf("failing the smallest rung is a finding and must be stated, got %q", none)
	}
}

func TestMedianInt64(t *testing.T) {
	if got := medianInt64(nil); got != 0 {
		t.Errorf("empty median = %d, want 0", got)
	}
	if got := medianInt64([]int64{5}); got != 5 {
		t.Errorf("single median = %d", got)
	}
	if got := medianInt64([]int64{9, 1, 5}); got != 5 {
		t.Errorf("median of 9,1,5 = %d, want 5 — the input must not need pre-sorting", got)
	}
}
