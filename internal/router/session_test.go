package router

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func convo(turns ...Message) *ChatRequest {
	return &ChatRequest{MaxTokens: 256, Messages: turns}
}

func sys(s string) Message  { return Message{Role: "system", Content: s} }
func usr(s string) Message  { return Message{Role: "user", Content: s} }
func asst(s string) Message { return Message{Role: "assistant", Content: s} }
func toolCall() Message {
	return Message{Role: "assistant", ToolCalls: json.RawMessage(`[{"id":"c1","function":{"name":"ls"}}]`)}
}
func toolResult() Message {
	return Message{Role: "tool", ToolCallID: "c1", Content: "a.txt b.txt"}
}

// The whole mechanism rests on one property: the key a conversation produces on
// turn 1 is the key it still produces on turn 5. Hashing the conversation minus
// its last message (the obvious wrong shape) fails exactly here.
func TestSessionKeyStableAcrossTurns(t *testing.T) {
	turn1 := convo(sys("you are a helpful agent"), usr("list my files"))
	turn2 := convo(sys("you are a helpful agent"), usr("list my files"), asst("ok"), usr("now delete the temp ones"))
	turn3 := convo(sys("you are a helpful agent"), usr("list my files"), asst("ok"), usr("now delete the temp ones"), asst("done"), usr("thanks"))

	k1, ok1 := sessionKeyFor(turn1)
	k2, ok2 := sessionKeyFor(turn2)
	k3, ok3 := sessionKeyFor(turn3)
	if !ok1 || !ok2 || !ok3 {
		t.Fatalf("every turn must yield a key: %v %v %v", ok1, ok2, ok3)
	}
	if k1 != k2 || k2 != k3 {
		t.Fatalf("key drifted across turns: %d %d %d", k1, k2, k3)
	}
}

func TestSessionKeyDistinguishesConversations(t *testing.T) {
	a, _ := sessionKeyFor(convo(sys("agent"), usr("list my files")))
	b, _ := sessionKeyFor(convo(sys("agent"), usr("deploy to prod")))
	if a == b {
		t.Fatal("different opening questions must not share a session key")
	}
	// Same question, different system prompt = different assistant = different session.
	c, _ := sessionKeyFor(convo(sys("terse agent"), usr("list my files")))
	if a == c {
		t.Fatal("different system prompts must not share a session key")
	}
	// A later user turn must not leak into the identity (that's what makes it stable).
	d, _ := sessionKeyFor(convo(sys("agent"), usr("list my files"), asst("ok"), usr("something else entirely")))
	if a != d {
		t.Fatalf("later turns must not change the identity: %d vs %d", a, d)
	}
}

func TestSessionKeyNeedsAUserTurn(t *testing.T) {
	if _, ok := sessionKeyFor(convo(sys("agent"))); ok {
		t.Fatal("system-only request must have no session")
	}
	if _, ok := sessionKeyFor(&ChatRequest{}); ok {
		t.Fatal("empty request must have no session")
	}
}

func TestInToolLoop(t *testing.T) {
	cases := []struct {
		name string
		msgs []Message
		want bool
	}{
		{"plain user turn", []Message{usr("hi")}, false},
		{"after an answer", []Message{usr("hi"), asst("hello")}, false},
		{"tool result pending", []Message{usr("ls"), toolCall(), toolResult()}, true},
		{"tool call just emitted", []Message{usr("ls"), toolCall()}, true},
		{"loop closed", []Message{usr("ls"), toolCall(), toolResult(), asst("here they are")}, false},
		{"empty", nil, false},
	}
	for _, tc := range cases {
		if got := inToolLoop(tc.msgs); got != tc.want {
			t.Errorf("%s: inToolLoop=%v want %v", tc.name, got, tc.want)
		}
	}
}

func TestSessionTrackerLifecycle(t *testing.T) {
	tr := newSessionTracker(time.Hour, 64)
	if _, ok := tr.lookup(42); ok {
		t.Fatal("unknown key must miss")
	}
	tr.remember(42, "big")
	e, ok := tr.lookup(42)
	if !ok || e.backendID != "big" {
		t.Fatalf("remember/lookup wrong: %+v ok=%v", e, ok)
	}
	// A later turn moving to another worker replaces the incumbent rather than
	// accumulating beside it: the discount is a claim about where the prefix is
	// cached NOW, and the worker that served two turns ago no longer holds it.
	tr.remember(42, "small")
	if e, _ := tr.lookup(42); e.backendID != "small" {
		t.Fatalf("incumbent not replaced by the worker that served the latest turn: %+v", e)
	}
	// A zero key is the "no session" sentinel and must never be stored.
	tr.remember(0, "big")
	if _, ok := tr.lookup(0); ok {
		t.Fatal("key 0 must never be tracked")
	}
	// Nil receiver is the feature-disabled path; it must not panic.
	var off *sessionTracker
	off.remember(1, "big")
	if _, ok := off.lookup(1); ok {
		t.Fatal("nil tracker must always miss")
	}
}

func TestSessionTrackerExpiry(t *testing.T) {
	tr := newSessionTracker(time.Millisecond, 64)
	tr.remember(7, "big")
	time.Sleep(5 * time.Millisecond)
	if _, ok := tr.lookup(7); ok {
		t.Fatal("expired session must not be returned — its KV cache is long gone")
	}
}

// A burst of one-shot prompts must not evict a live agent session: eviction is
// oldest-USE first, not insertion order.
func TestSessionTrackerEvictsOldestNotNewest(t *testing.T) {
	tr := newSessionTracker(time.Hour, 32)
	tr.remember(1, "agent-worker")
	for i := uint64(100); i < 200; i++ {
		tr.remember(i, "one-shot")
		tr.remember(1, "agent-worker") // the long session keeps being used
	}
	if e, ok := tr.lookup(1); !ok || e.backendID != "agent-worker" {
		t.Fatalf("actively used session evicted by one-shot traffic: %+v ok=%v", e, ok)
	}
	if tr.size() > 32+32/4 {
		t.Fatalf("tracker exceeded its bound: %d", tr.size())
	}
}

// An incumbent that no longer survives this turn's hard filters is not an
// incumbent — otherwise the router reports "stay" while discounting a worker
// that isn't even a candidate.
func TestSessionResolveIgnoresAbsentIncumbent(t *testing.T) {
	tr := newSessionTracker(time.Hour, 64)
	req := convo(sys("agent"), usr("list my files"))
	key, _ := sessionKeyFor(req)
	tr.remember(key, "gone")

	live := []*Backend{{BackendRegistration: BackendRegistration{ID: "here"}}}
	sr := tr.resolve(req, live)
	if !sr.active() {
		t.Fatal("a keyed conversation should still be active")
	}
	if sr.incumbent != "" {
		t.Fatalf("incumbent %q is not a candidate and must be ignored", sr.incumbent)
	}

	live = append(live, &Backend{BackendRegistration: BackendRegistration{ID: "gone"}})
	if sr := tr.resolve(req, live); sr.incumbent != "gone" {
		t.Fatalf("incumbent should be claimed once it is a candidate again, got %q", sr.incumbent)
	}
}

func TestSessionOutcomeLabels(t *testing.T) {
	cases := []struct {
		sr     sessionRoute
		chosen string
		want   string
	}{
		{sessionRoute{}, "big", ""},
		{sessionRoute{key: 1}, "big", "new"},
		{sessionRoute{key: 1, incumbent: "big"}, "big", "stay"},
		{sessionRoute{key: 1, incumbent: "big"}, "tiny", "switch"},
		{sessionRoute{key: 1, incumbent: "big", toolLoop: true}, "big", "lock"},
		{sessionRoute{key: 1, incumbent: "big", toolLoop: true}, "tiny", "lock-broken"},
	}
	for _, tc := range cases {
		if got := tc.sr.outcome(tc.chosen); got != tc.want {
			t.Errorf("outcome(%+v, %q) = %q want %q", tc.sr, tc.chosen, got, tc.want)
		}
	}
}

// The stay bias is a PREFILL discount, so it has to scale with the prompt: a long
// agent turn strongly prefers the incumbent, a one-word follow-up barely does.
func TestSessionPrefillDiscountScalesWithPrompt(t *testing.T) {
	incumbent := &Backend{
		BackendRegistration: BackendRegistration{ID: "incumbent", BaselineTPS: 100, MaxConcurrency: 4},
		ObservedTPS:         100, ObservedPrefillTPS: 1000,
	}
	rival := &Backend{
		BackendRegistration: BackendRegistration{ID: "rival", BaselineTPS: 100, MaxConcurrency: 4},
		ObservedTPS:         100, ObservedPrefillTPS: 1000,
	}

	short := jobCost{promptTokens: 20, outputTokens: 256}
	long := jobCost{promptTokens: 8000, outputTokens: 256}

	shortGain := expectedLatency(incumbent, short) - expectedLatency(incumbent, short.withIncumbent("incumbent"))
	longGain := expectedLatency(incumbent, long) - expectedLatency(incumbent, long.withIncumbent("incumbent"))
	if longGain <= shortGain {
		t.Fatalf("discount must grow with the conversation: short=%.4fs long=%.4fs", shortGain, longGain)
	}

	// The rival is unaffected by someone else's affinity.
	if a, b := expectedLatency(rival, long), expectedLatency(rival, long.withIncumbent("incumbent")); a != b {
		t.Fatalf("non-incumbent latency changed: %.4f vs %.4f", a, b)
	}
}

// Stickiness is a preference inside the ranking, not an override of it: a
// genuinely much faster worker must still win.
func TestSessionStickinessLosesToAMuchFasterWorker(t *testing.T) {
	slowIncumbent := &Backend{
		BackendRegistration: BackendRegistration{ID: "cpu", Quality: 60, BaselineTPS: 8, MaxConcurrency: 4},
		ObservedTPS:         8, ObservedPrefillTPS: 40,
	}
	fastRival := &Backend{
		BackendRegistration: BackendRegistration{ID: "gpu", Quality: 60, BaselineTPS: 120, MaxConcurrency: 4},
		ObservedTPS:         120, ObservedPrefillTPS: 6000,
	}
	job := jobCost{promptTokens: 4000, outputTokens: 1500}.withIncumbent("cpu")
	got := rankByDifficulty([]*Backend{slowIncumbent, fastRival}, 50, job, false)
	if got[0].ID != "gpu" {
		t.Fatalf("affinity must not pin a conversation to a far slower worker: got %s (cpu=%.1fs gpu=%.1fs)",
			got[0].ID, expectedLatency(slowIncumbent, job), expectedLatency(fastRival, job))
	}

	// Between equals, though, the incumbent wins — that's the point.
	twin := &Backend{
		BackendRegistration: BackendRegistration{ID: "gpu2", Quality: 60, BaselineTPS: 120, MaxConcurrency: 4},
		ObservedTPS:         120, ObservedPrefillTPS: 6000,
	}
	stickyJob := jobCost{promptTokens: 4000, outputTokens: 1500}.withIncumbent("gpu2")
	if got := rankByDifficulty([]*Backend{fastRival, twin}, 50, stickyJob, false); got[0].ID != "gpu2" {
		t.Fatalf("between equals the incumbent should win, got %s", got[0].ID)
	}
}

// The tool-loop lock is expressed as a bounded acquisition preference, and it
// must outrank the quality floor while a loop is open.
func TestSessionLockPreference(t *testing.T) {
	if p := sessionLockPreference(""); p.keep != nil {
		t.Fatal("no incumbent ⇒ no preference")
	}
	p := sessionLockPreference("big")
	if p.why != "session-lock" || p.wait != sessionLockWait {
		t.Fatalf("unexpected preference: %+v", p)
	}
	if !p.keep(&Backend{BackendRegistration: BackendRegistration{ID: "big"}}) {
		t.Fatal("incumbent must be preferred")
	}
	if p.keep(&Backend{BackendRegistration: BackendRegistration{ID: "tiny"}}) {
		t.Fatal("non-incumbent must not be preferred")
	}
}

// A conversation must land on its incumbent even when a same-tier rival is
// nominally a shade faster, and the route must say so.
func TestSelectBackendsPrefersIncumbent(t *testing.T) {
	reg := newTestRegistry()
	// Two indistinguishable workers: without affinity the tie breaks on id, which
	// would pick alpha every turn.
	readyBackend(reg, "alpha", 60, 100, 4)
	readyBackend(reg, "beta", 60, 100, 4)
	cfg := &Config{
		DefaultMaxTokens: 4096, AutoDifficulty: true,
		DifficultyBands: defaultDifficultyBands, DifficultyTemp: 0.10,
		DifficultyTimeout: time.Second, DifficultyCacheSize: 16, DifficultyMaxChars: 4000,
	}
	r := &Router{cfg: cfg, registry: reg, classifier: testClassifier(fakeEmbed),
		sessions: newSessionTracker(time.Hour, 64)}

	req := convo(sys("agent"), usr("say hello"))
	plan, err := r.planRoute(req, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if plan.session.incumbent != "" {
		t.Fatalf("first turn should have no incumbent, got %q", plan.session.incumbent)
	}
	r.sessions.remember(plan.session.key, "beta")

	next := convo(sys("agent"), usr("say hello"), asst("hi"), usr("and again"))
	plan2, err := r.planRoute(next, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if plan2.session.incumbent != "beta" {
		t.Fatalf("second turn should see incumbent beta, got %q", plan2.session.incumbent)
	}
	if plan2.candidates[0].ID != "beta" {
		t.Fatalf("second turn should stay on beta, got %s", plan2.candidates[0].ID)
	}
	if got := plan2.session.outcome("beta"); got != "stay" {
		t.Fatalf("outcome should be stay, got %q", got)
	}
}

// The question people actually ask about affinity: if turn 2 is HARDER, does the
// conversation get upgraded or does stickiness pin it to turn 1's model?
//
// It upgrades. The quality bar is sort key 2 in rankByDifficulty and the prefill
// discount only reaches sort key 4, so the discount can reorder workers WITHIN a
// tier but can never hold a below-bar incumbent ahead of one that clears the bar.
func TestSessionAffinityYieldsToAHarderTurn(t *testing.T) {
	reg := newTestRegistry()
	// A fast-but-weak worker that wins the easy race, and a slower strong one.
	readyBackend(reg, "cheap", 40, 400, 4)
	readyBackend(reg, "good", 90, 60, 4)
	cfg := &Config{
		DefaultMaxTokens: 4096, AutoDifficulty: true,
		DifficultyTemp: 0.10, DifficultyTimeout: time.Second,
		DifficultyCacheSize: 16, DifficultyMaxChars: 4000,
	}
	r := &Router{cfg: cfg, registry: reg, classifier: testClassifierAuto(fakeEmbed),
		sessions: newSessionTracker(time.Hour, 64)}

	// Turn 1: easy. Everyone clears a near-zero bar, so the fastest wins.
	first := convo(sys("agent"), usr("say hello"))
	plan1, err := r.planRoute(first, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if plan1.candidates[0].ID != "cheap" {
		t.Fatalf("easy turn should take the fast worker, got %s (target q>=%d)", plan1.candidates[0].ID, plan1.target)
	}
	r.sessions.remember(plan1.session.key, "cheap")

	// Turn 2: hard, same conversation. The incumbent is now BELOW the bar.
	second := convo(sys("agent"), usr("say hello"), asst("hi"), usr("prove a hard theorem"))
	plan2, err := r.planRoute(second, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if plan2.session.incumbent != "cheap" {
		t.Fatalf("affinity should still recognise the conversation, got %q", plan2.session.incumbent)
	}
	if plan2.target <= 40 {
		t.Fatalf("a hard turn should raise the bar above the incumbent's quality, got q>=%d", plan2.target)
	}
	if plan2.candidates[0].ID != "good" {
		t.Fatalf("a harder turn must escalate past the incumbent: got %s (target q>=%d, cheap q=40)",
			plan2.candidates[0].ID, plan2.target)
	}
	if got := plan2.session.outcome("good"); got != "switch" {
		t.Fatalf("the move should be reported as a switch, got %q", got)
	}

	// Turn 3: easy again, and the incumbent is now "good". Both clear the easy bar,
	// so the completion-time race resumes and the 6.7x faster worker takes it back.
	// Affinity is a DISCOUNT, not a lease: it scales with the prefill it saves, and
	// a one-line follow-up has almost no prefill to save.
	r.sessions.remember(plan2.session.key, "good")
	third := convo(sys("agent"), usr("say hello"), asst("hi"), usr("prove a hard theorem"), asst("done"), usr("say hello again"))
	plan3, err := r.planRoute(third, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if plan3.candidates[0].ID != "cheap" {
		t.Fatalf("a short easy turn should go back to the much faster worker, got %s", plan3.candidates[0].ID)
	}

	// ...but give the same easy turn a LONG conversation to re-prefill and affinity
	// starts to pay for itself, because that is the cost it is actually modelling.
	bulk := strings.Repeat("previous conversation context. ", 4000)
	long := convo(sys("agent"), usr("say hello"), asst(bulk), usr("say hello again"))
	longPlan, err := r.planRoute(long, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if longPlan.session.incumbent != "good" {
		t.Fatalf("same conversation, so the incumbent should still be good, got %q", longPlan.session.incumbent)
	}
	withAffinity := expectedLatency(reg.get("good"), longPlan.job)
	withoutAffinity := expectedLatency(reg.get("good"), longPlan.job.withIncumbent(""))
	if withAffinity >= withoutAffinity {
		t.Fatalf("a long conversation should measurably favour the incumbent: %.2fs vs %.2fs",
			withAffinity, withoutAffinity)
	}
}

// The one exception: while a TOOL LOOP is open, continuity outranks the tier.
// Handing a tool result to a model that never emitted the matching tool call
// breaks the loop outright, which is worse than serving that turn a tier low.
func TestToolLoopHoldsTheTierLow(t *testing.T) {
	var cheapHits, goodHits atomic.Int64
	cheap := cannedWorker(t, realAnswer, &cheapHits)
	good := cannedWorker(t, realAnswer, &goodHits)

	reg := newTestRegistry()
	for _, w := range []struct {
		id      string
		url     string
		quality int
		tps     int
	}{{"cheap", cheap.URL, 40, 400}, {"good", good.URL, 90, 60}} {
		reg.upsert(BackendRegistration{
			ID: w.id, URL: w.url, Model: "default", Quality: w.quality,
			BaselineTPS: float64(w.tps), MaxConcurrency: 4, TTLSeconds: 3600, Features: []string{"chat", "tools"},
		})
		reg.finishCertification(w.id, true, map[string]Check{}, float64(w.tps), 10, "")
	}
	dir := t.TempDir()
	logs, err := openLogStore(dir+"/logs.sqlite", 16384, "")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { logs.Close() })
	cfg := &Config{
		DefaultMaxTokens: 4096, AutoDifficulty: true,
		DifficultyTemp: 0.10, DifficultyTimeout: time.Second,
		DifficultyCacheSize: 16, DifficultyMaxChars: 4000,
	}
	r := &Router{cfg: cfg, registry: reg, classifier: testClassifierAuto(fakeEmbed), logs: logs,
		client: &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
		sessions: newSessionTracker(time.Hour, 64)}

	// A hard prompt whose tool loop is open, with "cheap" holding the loop.
	body := `{"model":"default","stream":false,"tools":[{"type":"function","function":{"name":"ls"}}],` +
		`"messages":[{"role":"system","content":"agent"},{"role":"user","content":"prove a hard theorem"},` +
		`{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"ls"}}]},` +
		`{"role":"tool","tool_call_id":"c1","content":"a.txt"}]}`

	key, _ := sessionKeyFor(convo(sys("agent"), usr("prove a hard theorem")))
	r.sessions.remember(key, "cheap")

	rec := runChat(t, r, body)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("X-LLM-Backend-ID"); got != "cheap" {
		t.Fatalf("an open tool loop must stay on the worker that opened it, got %q", got)
	}
	if got := rec.Header().Get("X-LLM-Session"); got != "lock" {
		t.Fatalf("the hold should be reported as a lock, got %q", got)
	}
	if goodHits.Load() != 0 {
		t.Fatalf("the better worker should not have been used mid-loop (%d calls)", goodHits.Load())
	}
	// The request log is written by a background goroutine; wait for it so the
	// deferred store close doesn't race it.
	waitFor(t, func() bool {
		rows, err := logs.List(context.Background(), "", 10, 0)
		return err == nil && len(rows) > 0
	}, "request never reached the log store")
}
