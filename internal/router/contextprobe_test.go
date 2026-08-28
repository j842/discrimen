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
			prompt, want := contextNeedlePrompt(size, depth, 42, defaultCharsPerToken)
			if !strings.Contains(prompt, want) {
				t.Fatalf("size=%d depth=%.1f: needle %q absent from the haystack", size, depth, want)
			}
			if strings.Count(prompt, want) != 1 {
				t.Errorf("size=%d depth=%.1f: needle appears %d times; a second copy makes the probe unfalsifiable",
					size, depth, strings.Count(prompt, want))
			}
			// Sized by the SAME divisor the hard filter uses, which is the whole
			// point of the shared helper — a rung must contain the amount of text
			// the filter would call `size` tokens. Wide band because the ladder
			// doubles: a 25% error can never confuse one rung with the next.
			est := tokensForChars(len(prompt), defaultCharsPerToken)
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
	a, wa := contextNeedlePrompt(8192, 0.5, 7, defaultCharsPerToken)
	b, wb := contextNeedlePrompt(8192, 0.5, 7, defaultCharsPerToken)
	if a != b || wa != wb {
		t.Error("same seed produced a different probe")
	}
	if _, wc := contextNeedlePrompt(8192, 0.5, 8, defaultCharsPerToken); wc == wa {
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
	// A probe with no recorded claim cannot be reasoned about, so the measured
	// figure stands rather than being waved through as "ran out of room".
	noClaim := &Backend{BackendRegistration: BackendRegistration{ContextK: 256}, ContextProbe: &ContextProbe{UsableTokens: 64 * 1024}}
	if got := usableContextTokens(noClaim); got != 64*1024 {
		t.Errorf("probe with no AdvertisedTokens = %d, want the measured 64K", got)
	}
}

// Every power-of-two window was halved. The rungs double from 4096 and the climb
// stops when the next one would not fit inside the claim, so a 256K worker proves
// 128K and is never asked about 256K — and routing read the rung it declined to
// attempt as a measured ceiling. This is the whole bug: a 96-quality local worker
// hard-filtered out of every prompt over 128K, a coding harness pushed onto a
// relayed worker across a VPN, and a 503 once the estimate passed the last claim
// standing.
func TestUsableContextTokensBelievesAClaimTheLadderNeverTested(t *testing.T) {
	for _, tc := range []struct {
		name     string
		contextK int
		usable   int
		want     int
		wantRoom bool
	}{
		{"256K claim, ladder tops out at 128K", 256, 128 * 1024, 256 * 1024, true},
		{"128K claim, ladder tops out at 64K", 128, 64 * 1024, 128 * 1024, true},
		{"32K claim, ladder tops out at 16K", 32, 16 * 1024, 32 * 1024, true},
		// Not a power of two: 32K fits under 36K but 64K does not, so this also
		// ran out of room and the claim stands.
		{"36K claim, ladder tops out at 32K", 36, 32 * 1024, 36 * 1024, true},
		// A REAL edge. 64K fits twice over inside a 256K claim, so the rung above
		// it was attempted and failed. That verdict is what routing must keep.
		{"256K claim, needle lost at 128K", 256, 64 * 1024, 64 * 1024, false},
		{"256K claim, needle lost at 32K", 256, 16 * 1024, 16 * 1024, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &ContextProbe{UsableTokens: tc.usable, AdvertisedTokens: tc.contextK * 1024}
			if got := ladderRanOutOfRoom(p); got != tc.wantRoom {
				t.Errorf("ladderRanOutOfRoom = %v, want %v", got, tc.wantRoom)
			}
			b := &Backend{BackendRegistration: BackendRegistration{ContextK: tc.contextK}, ContextProbe: p}
			if got := usableContextTokens(b); got != tc.want {
				t.Errorf("usableContextTokens = %d (%dK), want %d (%dK)", got, got/1024, tc.want, tc.want/1024)
			}
		})
	}
}

// The production failure, at the layer that produced it. This is the profile a
// live RTX 6000 Pro worker was carrying — a 256K claim, every rung to 128K
// passed, no failure anywhere — and the hard filter was rejecting it for any
// prompt over 128K. With only one 256K endpoint left in the fleet (an unprobed
// relay across a VPN), a coding harness at 190K tokens had every worker rejected
// and got a 503 the model menu had promised would be served.
func TestHardFilterAdmitsAWorkerAtTheWindowItAdvertises(t *testing.T) {
	flashNext := &Backend{
		BackendRegistration: BackendRegistration{
			ID: "llm-6000pro-qwen38-flash-next", ContextK: 256, Features: []string{"chat", "tools"},
		},
		ContextProbe: &ContextProbe{
			UsableTokens: 131072, AdvertisedTokens: 262144,
			Levels: []ContextProbeLevel{
				{Tokens: 4096, Passed: 3, Total: 3}, {Tokens: 8192, Passed: 3, Total: 3},
				{Tokens: 16384, Passed: 3, Total: 3}, {Tokens: 32768, Passed: 3, Total: 3},
				{Tokens: 65536, Passed: 3, Total: 3}, {Tokens: 131072, Passed: 3, Total: 3},
			},
		},
	}
	for _, neededK := range []int{100, 128, 160, 200, 250} {
		if reason := admitReason(flashNext, hardFilter{minContextK: neededK}); reason != "" {
			t.Errorf("%dK prompt rejected from a 256K worker: %s", neededK, reason)
		}
	}
	// The check still bites where the ladder actually found an edge. Same claim,
	// same worker — but 128K was ATTEMPTED here and the needle was lost, so 64K is
	// a measurement and routing must keep enforcing it.
	lost := &Backend{
		BackendRegistration: BackendRegistration{ID: "forgetful", ContextK: 256, Features: []string{"chat"}},
		ContextProbe: &ContextProbe{
			UsableTokens: 65536, AdvertisedTokens: 262144,
			Levels: []ContextProbeLevel{
				{Tokens: 65536, Passed: 3, Total: 3}, {Tokens: 131072, Passed: 1, Total: 3},
			},
		},
	}
	reason := admitReason(lost, hardFilter{minContextK: 100})
	if reason == "" {
		t.Fatal("a worker measured to lose facts at 128K must not take a 100K prompt")
	}
	if !strings.Contains(reason, "measured") || !strings.Contains(reason, "256K advertised") {
		t.Errorf("the rejection should say the measurement disagrees with the claim, got %q", reason)
	}
}

// The two ways a climb ends without reaching the top of the claim are not the
// same, and neither is "ran out of room": both leave a rung that DID fit
// unresolved, so the last proven length stays the operative bound.
func TestLadderRanOutOfRoomExcludesErroredAndBudgetedClimbs(t *testing.T) {
	// The ladder gave up on 128K after 64K passed — 128K+1K fits inside 256K, so
	// this is an abandoned rung, not an absent one.
	budgeted := &ContextProbe{AdvertisedTokens: 256 * 1024, UsableTokens: 64 * 1024, Truncated: true}
	if ladderRanOutOfRoom(budgeted) {
		t.Error("a climb stopped on its time budget left a fitting rung untried — that is not running out of room")
	}
	errored := &ContextProbe{
		AdvertisedTokens: 256 * 1024,
		UsableTokens:     64 * 1024,
		Levels: []ContextProbeLevel{
			{Tokens: 64 * 1024, Passed: 3, Total: 3},
			{Tokens: 128 * 1024, Total: 3, Errored: true},
		},
	}
	if ladderRanOutOfRoom(errored) {
		t.Error("a rung that errored had to fit to have run at all")
	}
	if got := usableContextTokens(&Backend{
		BackendRegistration: BackendRegistration{ContextK: 256}, ContextProbe: errored,
	}); got != 64*1024 {
		t.Errorf("errored climb = %d, want the last proven 64K", got)
	}
}

// The ladder's own break condition, replayed against the predicate: whatever
// runContextProbe would stop on for a given claim, ladderRanOutOfRoom must agree
// about. Rewriting one without the other is what let the message say "consistent
// with 256K advertised" beside a routing cap of 128K.
func TestLadderRanOutOfRoomMatchesTheClimbItDescribes(t *testing.T) {
	for _, advertised := range []int{8192, 32768, 36864, 131072, 200000, 262144} {
		lastProven, ranOut := 0, false
		for size := contextProbeStart; size <= contextProbeCeiling; size *= 2 {
			if size+contextProbeReserve > advertised {
				ranOut = true // exactly the break runContextProbe takes
				break
			}
			lastProven = size // every rung passes in this replay
		}
		if lastProven == 0 {
			continue // claim under the first rung; nothing to measure
		}
		p := &ContextProbe{UsableTokens: lastProven, AdvertisedTokens: advertised}
		if got := ladderRanOutOfRoom(p); got != ranOut {
			t.Errorf("advertised %d: proved %d, ladder ran out = %v but predicate says %v",
				advertised, lastProven, ranOut, got)
		}
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

// "The probe never got an answer" and "the worker answered wrongly" are opposite
// findings, and the ladder stops on either. Rendering them the same way is how
// this line came to name a worker as unable to hold 4K of context on the strength
// of a request that returned a 503 — a confident diagnosis of something it had
// learned nothing about, which is worse for the operator than silence.
func TestContextProbeMessageSeparatesAnErrorFromAFailure(t *testing.T) {
	errored := &ContextProbe{
		AdvertisedTokens: 128 * 1024,
		Levels:           []ContextProbeLevel{{Tokens: 4096, Total: 3, Errored: true}},
	}
	msg := contextProbeMessage(errored)
	if strings.Contains(msg, "FAILED") {
		t.Errorf("a probe that ERRORED at the first rung is reported as a retrieval failure: %q", msg)
	}
	if !strings.Contains(msg, "errored") {
		t.Errorf("the message does not say the probe errored: %q", msg)
	}
	if !strings.Contains(msg, "UNMEASURED") {
		t.Errorf("an errored ladder establishes nothing and must say so: %q", msg)
	}

	// A genuine miss at the same rung must still read as the finding it is.
	missed := &ContextProbe{
		AdvertisedTokens: 128 * 1024,
		Levels:           []ContextProbeLevel{{Tokens: 4096, Passed: 1, Total: 3}},
	}
	if got := contextProbeMessage(missed); !strings.Contains(got, "FAILED") {
		t.Errorf("failing the smallest rung is a real finding and must still be stated: %q", got)
	}

	// Errored PARTWAY UP is the same bug one rung along: 4K passed and 8K never
	// answered, so 4K is a lower bound exactly like a budget truncation — not the
	// measured ceiling the default branch would have called it.
	partial := &ContextProbe{
		AdvertisedTokens: 128 * 1024,
		UsableTokens:     4096,
		Levels: []ContextProbeLevel{
			{Tokens: 4096, Passed: 3, Total: 3},
			{Tokens: 8192, Total: 3, Errored: true},
		},
	}
	got := contextProbeMessage(partial)
	if !strings.Contains(got, "at least") {
		t.Errorf("a ladder abandoned on an error is a LOWER BOUND and must say so: %q", got)
	}
	if strings.Contains(got, "ADVERTISED — routing uses the measured figure") {
		t.Errorf("an unmeasured ceiling is reported as a measured advertised-vs-actual gap: %q", got)
	}
}

// The glyph is the part that gets read. A tick beside "retrieval FAILED" or
// beside a probe that never ran tells the operator the opposite of the message
// next to it.
func TestContextProbeOK(t *testing.T) {
	cases := []struct {
		name string
		p    *ContextProbe
		want bool
	}{
		{"not probed at all", nil, false},
		{"errored at the first rung",
			&ContextProbe{AdvertisedTokens: 128 * 1024,
				Levels: []ContextProbeLevel{{Tokens: 4096, Total: 3, Errored: true}}}, false},
		{"retrieval failed at the first rung",
			&ContextProbe{AdvertisedTokens: 128 * 1024,
				Levels: []ContextProbeLevel{{Tokens: 4096, Passed: 1, Total: 3}}}, false},
		{"window under the first rung, nothing to test",
			&ContextProbe{AdvertisedTokens: 2048}, true},
		{"measured, far under the claim — a finding, not a fault",
			&ContextProbe{AdvertisedTokens: 256 * 1024, UsableTokens: 32 * 1024,
				Levels: []ContextProbeLevel{{Tokens: 32768, Passed: 3, Total: 3}}}, true},
		{"measured, ladder truncated on its time budget",
			&ContextProbe{AdvertisedTokens: 256 * 1024, UsableTokens: 16 * 1024, Truncated: true,
				Levels: []ContextProbeLevel{{Tokens: 16384, Passed: 3, Total: 3}}}, true},
		{"errored after establishing a lower bound",
			&ContextProbe{AdvertisedTokens: 128 * 1024, UsableTokens: 4096,
				Levels: []ContextProbeLevel{
					{Tokens: 4096, Passed: 3, Total: 3},
					{Tokens: 8192, Total: 3, Errored: true},
				}}, false},
	}
	for _, c := range cases {
		if got := contextProbeOK(c.p); got != c.want {
			t.Errorf("%s: contextProbeOK = %v, want %v (message: %q)", c.name, got, c.want, contextProbeMessage(c.p))
		}
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
