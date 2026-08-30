package router

import (
	"math"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
)

func TestTokenRatiosLearnsFromReportedPromptSizes(t *testing.T) {
	r := newTokenRatios()
	if got := r.forModel("never-seen"); got != defaultCharsPerToken {
		t.Errorf("unmeasured model = %v, want the conservative default %v", got, defaultCharsPerToken)
	}
	// 700000 chars reported as 200000 tokens: 3.5, which is what code and JSON
	// actually measure and a third away from the 3.0 the router assumed.
	r.observe("qwen", 700000, 200000)
	if got := r.forModel("qwen"); math.Abs(got-3.5) > 0.001 {
		t.Errorf("first sample = %v, want it taken as-is at 3.5", got)
	}
	// A different model is a different tokenizer and must not be touched.
	if got := r.forModel("gemma"); got != defaultCharsPerToken {
		t.Errorf("gemma = %v, want the default — one model's traffic is not another's", got)
	}
}

// Down fast, up slow. Under-estimating is what gets a prompt refused by an
// endpoint, so evidence that the text is DENSER than assumed has to land almost
// at once, while evidence that it is sparser may only loosen the estimate
// gradually.
func TestTokenRatiosMovesTowardCautionFaster(t *testing.T) {
	denser := newTokenRatios()
	denser.observe("m", 400000, 100000) // 4.0
	denser.observe("m", 300000, 100000) // 3.0 — tighten
	tightened := 4.0 - denser.forModel("m")

	sparser := newTokenRatios()
	sparser.observe("m", 300000, 100000) // 3.0
	sparser.observe("m", 400000, 100000) // 4.0 — loosen
	loosened := sparser.forModel("m") - 3.0

	if tightened <= loosened {
		t.Errorf("tightened by %.3f and loosened by %.3f — caution must be the faster direction",
			tightened, loosened)
	}
}

func TestTokenRatiosIgnoresSamplesItCannotLearnFrom(t *testing.T) {
	r := newTokenRatios()
	// A short prompt is mostly chat template — role markers and scaffolding the
	// endpoint counts and the router never saw — so its ratio describes the
	// template, not the text.
	r.observe("m", 300, 100)
	if got := r.forModel("m"); got != defaultCharsPerToken {
		t.Errorf("a %d-token prompt taught a ratio of %v; it is below calibrationMinTokens", 100, got)
	}
	for _, bad := range [][2]int{{0, 50000}, {-1, 50000}, {500000, 0}} {
		r.observe("m", bad[0], bad[1])
	}
	if got := r.forModel("m"); got != defaultCharsPerToken {
		t.Errorf("a degenerate sample moved the ratio to %v", got)
	}
	// Clamped: base64 or a pathological payload cannot drag the divisor somewhere
	// no real tokenizer lives.
	dense := newTokenRatios()
	dense.observe("m", 50000, 50000) // 1.0
	if got := dense.forModel("m"); got < minCharsPerToken {
		t.Errorf("ratio %v fell through the floor %v", got, minCharsPerToken)
	}
	sparse := newTokenRatios()
	sparse.observe("m", 5000000, 50000) // 100.0
	if got := sparse.forModel("m"); got > maxCharsPerToken {
		t.Errorf("ratio %v rose above the ceiling %v", got, maxCharsPerToken)
	}
}

// The whole point of the shared helper: a rung the probe labels N tokens must
// contain the amount of text the hard filter would call N tokens. They disagreed
// — the probe filling at 4 chars/token and the filter sizing at 3 — so the ladder
// proved retrieval at one length and licensed prompts of another.
func TestProbeAndFilterAgreeOnWhatARungMeans(t *testing.T) {
	for _, cpt := range []float64{defaultCharsPerToken, 3.5, 4.0} {
		for _, rung := range []int{4096, 32768, 131072} {
			chars := charsForTokens(rung, cpt)
			if back := tokensForChars(chars, cpt); back != rung {
				t.Errorf("cpt=%.1f rung=%d: the filter sizes the probe's own haystack at %d", cpt, rung, back)
			}
			prompt, _ := contextNeedlePrompt(rung, 0.5, 1, cpt)
			sized := tokensForChars(len(prompt), cpt)
			if sized < rung*3/4 || sized > rung*5/4 {
				t.Errorf("cpt=%.1f: a rung labelled %d is text the filter sizes at %d", cpt, rung, sized)
			}
		}
	}
}

// The hard filter asks per candidate, because a token is not a fixed amount of
// text: the same prompt is a different number of tokens to two models, and the
// difference is bigger than the margin that decides whether it fits.
func TestHardFilterSizesThePromptPerModel(t *testing.T) {
	ratios := newTokenRatios()
	ratios.observe("sparse", 400000, 100000) // 4.0 chars/token
	ratios.observe("dense", 250000, 100000)  // 2.5 chars/token

	// 480,000 characters: 120K tokens to the sparse tokenizer, 192K to the dense
	// one. A 160K window holds one and not the other.
	f := hardFilter{promptChars: 480000, ratios: ratios}
	sparse := &Backend{BackendRegistration: BackendRegistration{ID: "s", Model: "sparse", ContextK: 160}}
	dense := &Backend{BackendRegistration: BackendRegistration{ID: "d", Model: "dense", ContextK: 160}}

	if reason := admitReason(sparse, f); reason != "" {
		t.Errorf("the sparse-tokenizing model holds this prompt in 120K, but: %s", reason)
	}
	if admitReason(dense, f) == "" {
		t.Error("the dense-tokenizing model needs 192K of a 160K window and must be rejected")
	}
	// An explicit client floor is a number the CALLER stated and is never
	// re-estimated against anyone's tokenizer.
	explicit := hardFilter{minContextK: 200, promptChars: 480000, ratios: ratios}
	if admitReason(sparse, explicit) == "" {
		t.Error("requirements.min_context_k of 200K must be honoured as given against a 160K worker")
	}
}

// The regression this whole change exists for, in the units it happened in.
func TestMeasuredRatioAdmitsThePromptTheConstantRefused(t *testing.T) {
	// 700,000 characters of code and JSON — 200K real tokens at the 3.5 the fleet
	// actually measures, which fits a 256K window with room to spare. At the flat
	// 3.0 the router assumed, it estimates 233K and still fits; at 250K real
	// tokens (875,000 chars) it estimates 292K and is refused by everything.
	w := &Backend{BackendRegistration: BackendRegistration{ID: "w", Model: "qwen", ContextK: 256}}
	guessed := hardFilter{promptChars: 875000, reserveTokens: 1024}
	if admitReason(w, guessed) == "" {
		t.Fatal("precondition: the flat 3.0 divisor should refuse this prompt")
	}
	ratios := newTokenRatios()
	ratios.observe("qwen", 700000, 200000)
	measured := hardFilter{promptChars: 875000, reserveTokens: 1024, ratios: ratios}
	if reason := admitReason(w, measured); reason != "" {
		t.Errorf("250K real tokens still refused by a 256K worker after measuring: %s", reason)
	}
}

// An endpoint's refusal for an over-long prompt is the most valuable sample there
// is — the tokenizer disagreeing with the estimate in the one region where the
// disagreement costs the caller their request.
func TestContextLimitPromptTokens(t *testing.T) {
	vllm := []byte(`{"error":{"message":"This model's maximum context length is 262144 tokens. ` +
		`However, you requested 270000 tokens (265000 in the messages, 5000 in the completion). ` +
		`Please reduce the length of the messages or completion.","type":"BadRequestError","code":400}}`)
	if got := contextLimitPromptTokens(vllm); got != 265000 {
		t.Errorf("vLLM refusal = %d, want the 265000 it counted in the messages", got)
	}
	// It must take the MESSAGES figure, never the request total — that includes a
	// completion budget the router never sent as text, and learning from it would
	// teach a denser ratio than the truth.
	if got := contextLimitPromptTokens(vllm); got == 270000 {
		t.Error("read the request total instead of the prompt")
	}
	for _, other := range []string{
		`{"error":{"message":"the request exceeds the available context size"}}`,
		`{"error":{"message":"context length exceeded"}}`,
		``,
		`not json at all`,
	} {
		if got := contextLimitPromptTokens([]byte(other)); got != 0 {
			t.Errorf("a message this parser does not understand yielded %d; it must yield nothing rather than guess", got)
		}
	}
}

// When context is the ONLY thing in the way, the estimate is a guess and the
// endpoint holds the truth — so the request goes to the widest window and the
// engine rules, instead of a 503 the router cannot actually justify.
func TestContextOverflowGoesToTheWidestWindow(t *testing.T) {
	var narrowHits, wideHits atomic.Int64
	narrow := cannedWorker(t, realAnswer, &narrowHits)
	wide := cannedWorker(t, realAnswer, &wideHits)

	reg := newTestRegistry()
	for _, w := range []struct {
		id       string
		url      string
		contextK int
	}{{"narrow", narrow.URL, 8}, {"wide", wide.URL, 32}} {
		reg.upsert(BackendRegistration{
			ID: w.id, URL: w.url, Model: w.id, Quality: 80, ContextK: w.contextK,
			BaselineTPS: 100, MaxConcurrency: 2, TTLSeconds: 3600, Features: []string{"chat"},
		})
		reg.finishCertification(w.id, true, map[string]Check{}, 100, 10, "")
	}
	dir := t.TempDir()
	logs, err := openLogStore(dir+"/logs.sqlite", 16384, "")
	if err != nil {
		t.Fatalf("open log store: %v", err)
	}
	t.Cleanup(func() { logs.Close() })
	r := &Router{
		cfg: &Config{DefaultMaxTokens: 512}, registry: reg, ratios: newTokenRatios(),
		client: &http.Client{}, streamClient: &http.Client{}, logs: logs,
		sessions: newSessionTracker(0, 0),
	}

	// Comfortably past even the 32K worker.
	huge := strings.Repeat("package main and some more filler text here. ", 4000)
	rec := runChat(t, r, `{"model":"default","stream":false,"messages":[{"role":"user","content":"`+huge+`"}]}`)

	if rec.Code == http.StatusServiceUnavailable {
		t.Fatalf("still a 503: %s", rec.Body.String())
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if wideHits.Load() != 1 || narrowHits.Load() != 0 {
		t.Errorf("overflow went to narrow=%d wide=%d; it must pick the widest window",
			narrowHits.Load(), wideHits.Load())
	}
	if got := rec.Header().Get("X-LLM-Context-Overflow"); got != "32" {
		t.Errorf("overflow not reported to the caller: %q", got)
	}
}

// Overflow is for CONTEXT only. A missing feature is a fact about the worker, not
// an estimate, and no amount of trying will change it — refusing is the honest
// answer and the caller would rather have the error.
func TestContextOverflowDoesNotOverrideOtherRequirements(t *testing.T) {
	noTools := &Backend{BackendRegistration: BackendRegistration{
		ID: "no-tools", Model: "m", ContextK: 8, Features: []string{"chat"},
	}}
	f := hardFilter{promptChars: 900000, needTools: true}
	if widestContext([]*Backend{noTools}, f) != nil {
		t.Error("a worker without tools was offered as an overflow target for a tools request")
	}
	withTools := &Backend{BackendRegistration: BackendRegistration{
		ID: "tools", Model: "m", ContextK: 8, Features: []string{"chat", "tools"},
	}}
	if got := widestContext([]*Backend{noTools, withTools}, f); got != withTools {
		t.Errorf("overflow target = %v, want the tools-capable worker", got)
	}
}

// The overflow gamble must hold for a NAMED model too — a group member, a relay
// row, or the caller's explicit pick. Refusing named models broke the gamble
// across a relay: the upstream router forwarded its own overflow pinned to the
// relay's model name, the downstream planner refused on its estimate, and the
// endpoint never got to rule — the caller saw a 503 naming a worker they never
// asked for instead of the engine's exact 400.
func TestContextOverflowForwardsNamedModel(t *testing.T) {
	served := &Backend{BackendRegistration: BackendRegistration{
		ID: "narrow-named", Model: "relay-model", ContextK: 8, Features: []string{"chat"},
	}}
	other := &Backend{BackendRegistration: BackendRegistration{
		ID: "wide-other", Model: "different-model", ContextK: 256, Features: []string{"chat"},
	}}
	f := hardFilter{promptChars: 900000, wantModel: "relay-model"}
	if got := widestContext([]*Backend{served, other}, f); got != served {
		t.Errorf("named-model overflow target = %v, want the worker serving that model", got)
	}
	// And when nothing serves the model at all, refusal stays the honest answer.
	f = hardFilter{promptChars: 900000, wantModel: "unserved-model"}
	if widestContext([]*Backend{served, other}, f) != nil {
		t.Error("a worker not serving the named model was offered as an overflow target")
	}
}
