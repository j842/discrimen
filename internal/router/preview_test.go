package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func previewRouter(t *testing.T) *Router {
	t.Helper()
	reg := newTestRegistry()
	readyBackend(reg, "tiny", 20, 200, 2)
	readyThinkingBackend(reg, "big", 90, 140, 6, true)
	// A worker that cannot serve a long prompt, so the preview has a rejection to
	// explain.
	reg.upsert(BackendRegistration{
		ID: "shortctx", URL: "http://shortctx", Model: "default", Quality: 80,
		BaselineTPS: 300, ContextK: 4, MaxConcurrency: 2, TTLSeconds: 3600,
		Features: []string{"chat"},
	})
	reg.finishCertification("shortctx", true, map[string]Check{}, 300, 10, "")

	cfg := &Config{
		DefaultMaxTokens: 4096, AutoDifficulty: true, AutoThinking: true,
		DifficultyTemp: 0.10, ReasoningThreshold: 0.35,
		DifficultyTimeout: time.Second, DifficultyCacheSize: 16, DifficultyMaxChars: 4000,
	}
	return &Router{cfg: cfg, registry: reg, classifier: testClassifierAuto(fakeEmbed),
		sessions: newSessionTracker(time.Hour, 16)}
}

// previewAdminRouter is previewRouter with a keys table holding an admin key,
// which is what the per-worker half of the preview is gated behind.
func previewAdminRouter(t *testing.T) *Router {
	t.Helper()
	r := previewRouter(t)
	r.logs = newTestLogStore(t)
	issueKey(t, r, adminSecret, apiKey{Role: roleAdmin, Name: "preview admin"})
	return r
}

// postPreview asks as an unauthenticated CLIENT — previewRouter requires no
// credential, so this is the ordinary caller's view.
func postPreview(t *testing.T, r *Router, body string) (*httptest.ResponseRecorder, previewResponse) {
	t.Helper()
	return postPreviewAs(t, r, body, "")
}

func postPreviewAs(t *testing.T, r *Router, body, token string) (*httptest.ResponseRecorder, previewResponse) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/route-preview", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	r.handleRoutePreview(rec, req)
	var pv previewResponse
	if rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), &pv); err != nil {
			t.Fatalf("decode preview: %v (%s)", err, rec.Body.String())
		}
	}
	return rec, pv
}

// The preview must report the decision the router would actually make — same
// classifier, same tier, same ranked head.
//
// Asked as an ADMIN, because the per-candidate detail this grades is admin-only
// (the estimates it checks are per worker); the decision half is graded for a
// client in TestRoutePreviewGivesAClientTheDecisionWithoutTheFleet.
func TestRoutePreviewMatchesTheRealDecision(t *testing.T) {
	r := previewAdminRouter(t)
	body := `{"model":"default","messages":[{"role":"user","content":"prove a hard theorem"}]}`

	rec, pv := postPreviewAs(t, r, body, adminSecret)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	chatReq, err := parseAndValidateChatRequest([]byte(body), r.cfg.DefaultMaxTokens)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := r.planRoute(chatReq, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if pv.Route != plan.route {
		t.Fatalf("preview route %q != real route %q", pv.Route, plan.route)
	}
	if pv.WouldServe != plan.candidates[0].ID {
		t.Fatalf("preview would_serve %q != real first candidate %q", pv.WouldServe, plan.candidates[0].ID)
	}
	if pv.TargetQuality != plan.target {
		t.Fatalf("preview target %d != real target %d", pv.TargetQuality, plan.target)
	}
	if !pv.Classified || pv.Difficulty == nil || pv.Reasoning == nil {
		t.Fatalf("hard prompt should be classified on both axes: %+v", pv)
	}
	if !pv.Thinking.Enabled || pv.Thinking.Source != "auto" {
		t.Fatalf("a reasoning prompt should have auto-thinking on: %+v", pv.Thinking)
	}
	if len(pv.Candidates) == 0 || pv.Candidates[0].ExpectedSeconds <= 0 {
		t.Fatalf("candidates should carry a completion-time estimate: %+v", pv.Candidates)
	}
}

// "Why did this go to the CPU worker" is the question the endpoint exists to
// answer, so a hard-filtered worker must come back named, with the reason — to
// an operator debugging the fleet, which is who that answer is for.
func TestRoutePreviewExplainsRejections(t *testing.T) {
	r := previewAdminRouter(t)
	long := strings.Repeat("context ", 12000) // ~12k tokens, over shortctx's 4K
	body, err := json.Marshal(map[string]any{
		"model":    "default",
		"messages": []map[string]string{{"role": "user", "content": long}},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, pv := postPreviewAs(t, r, string(body), adminSecret)

	var found bool
	for _, rj := range pv.Rejected {
		if rj.ID == "shortctx" {
			found = true
			if !strings.Contains(rj.Reason, "context") {
				t.Fatalf("wrong rejection reason for shortctx: %q", rj.Reason)
			}
		}
	}
	if !found {
		t.Fatalf("shortctx should be rejected on context, rejections=%+v", pv.Rejected)
	}
}

func TestRoutePreviewReportsThinkingSource(t *testing.T) {
	r := previewRouter(t)
	cases := []struct{ body, want string }{
		{`{"messages":[{"role":"user","content":"say hello"}],"reasoning_effort":"high"}`, "reasoning_effort"},
		{`{"messages":[{"role":"user","content":"say hello"}],"requirements":{"thinking":"on"}}`, "requirements"},
		{`{"messages":[{"role":"user","content":"say hello"}],"chat_template_kwargs":{"enable_thinking":true}}`, "chat_template_kwargs"},
		{`{"messages":[{"role":"user","content":"say hello"}]}`, "auto"},
	}
	for _, tc := range cases {
		_, pv := postPreview(t, r, tc.body)
		if pv.Thinking.Source != tc.want {
			t.Errorf("body %s: thinking source %q want %q", tc.body, pv.Thinking.Source, tc.want)
		}
	}
}

// A pin bypasses selection entirely, so previewing one would describe a decision
// the router never makes. Refuse rather than mislead.
func TestRoutePreviewRefusesPinnedRequests(t *testing.T) {
	r := previewRouter(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/route-preview",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("X-LLM-Backend-ID", "tiny")
	rec := httptest.NewRecorder()
	r.handleRoutePreview(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400 for a pinned preview, got %d: %s", rec.Code, rec.Body.String())
	}
}

// The preview must not move anything: no session turn recorded, no slot taken.
func TestRoutePreviewIsReadOnly(t *testing.T) {
	r := previewRouter(t)
	body := `{"messages":[{"role":"system","content":"agent"},{"role":"user","content":"say hello"}]}`
	postPreview(t, r, body)
	postPreview(t, r, body)
	if n := r.sessions.size(); n != 0 {
		t.Fatalf("preview recorded %d session turns; it must record none", n)
	}
	for _, b := range r.registry.snapshot() {
		if b.ActiveRequests != 0 {
			t.Fatalf("%s shows active_requests=%d after a preview", b.ID, b.ActiveRequests)
		}
	}
}

func TestRoutePreviewShowsSessionState(t *testing.T) {
	r := previewRouter(t)
	req := convo(sys("agent"), usr("say hello"))
	key, _ := sessionKeyFor(req)
	r.sessions.remember(key, "big")

	_, pv := postPreview(t, r,
		`{"messages":[{"role":"system","content":"agent"},{"role":"user","content":"say hello"},`+
			`{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"ls"}}]},`+
			`{"role":"tool","tool_call_id":"c1","content":"a.txt"}]}`)

	if pv.Session.Incumbent != "big" {
		t.Fatalf("preview should name the incumbent, got %q", pv.Session.Incumbent)
	}
	if !pv.Session.ToolLoop || !pv.Session.Locked {
		t.Fatalf("mid-tool-loop should be reported as locked: %+v", pv.Session)
	}
	if !strings.Contains(strings.Join(pv.Notes, " "), "tool loop open") {
		t.Fatalf("the lock should be explained in the notes: %v", pv.Notes)
	}
}

func TestRoutePreviewUnknownModelIs404(t *testing.T) {
	r := previewRouter(t)
	rec, _ := postPreview(t, r, `{"model":"nope-9000","messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("want 404 for an unserved model, got %d: %s", rec.Code, rec.Body.String())
	}
}

// The preview answers a question about the CALLER's request. It must not double
// as a fleet listing: candidates and rejections together name every registered
// worker, alive or not, with its quality and its load — the inventory that moving
// GET /backends to admin scope was meant to stop handing to any client token.
func TestRoutePreviewGivesAClientTheDecisionWithoutTheFleet(t *testing.T) {
	r := previewRouter(t) // no credential required: the ordinary client case
	long := strings.Repeat("context ", 12000)
	body, err := json.Marshal(map[string]any{
		"model":    "default",
		"messages": []map[string]string{{"role": "user", "content": long}},
	})
	if err != nil {
		t.Fatal(err)
	}

	rec, pv := postPreview(t, r, string(body))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if len(pv.Candidates) != 0 {
		t.Errorf("a client was handed %d workers with their quality and load: %+v", len(pv.Candidates), pv.Candidates)
	}
	if len(pv.Rejected) != 0 {
		t.Errorf("a client was handed the filtered workers and why: %+v", pv.Rejected)
	}
	// Nothing may leak through the raw JSON either — an empty slice serialised as
	// [] would still be an answer about the fleet, and a key id must not appear.
	raw := rec.Body.String()
	for _, id := range []string{"shortctx", "tiny", "big"} {
		if pv.WouldServe == id {
			continue // the one worker the caller's own request would land on
		}
		if strings.Contains(raw, id) {
			t.Errorf("worker %q is named in a client's preview: %s", id, raw)
		}
	}

	// What survives is the decision, which is what the endpoint is for.
	if pv.WouldServe == "" || pv.Route == "" {
		t.Errorf("the decision itself went missing: route=%q would_serve=%q", pv.Route, pv.WouldServe)
	}
	if !pv.Classified || pv.Difficulty == nil || pv.Reasoning == nil {
		t.Errorf("the classification must survive: %+v", pv)
	}
	if pv.Job.PromptTokens == 0 {
		t.Error("the job's own size must survive")
	}
	// The tier and the thinking mode too, on a prompt that earns both.
	_, hard := postPreview(t, r, `{"model":"default","messages":[{"role":"user","content":"prove a hard theorem"}]}`)
	if hard.TargetQuality == 0 || !hard.Thinking.Enabled || hard.Thinking.Source != "auto" {
		t.Errorf("a client lost the tier or the thinking mode: target=%d thinking=%+v", hard.TargetQuality, hard.Thinking)
	}

	// The same request from an admin still gets the full picture, which is what
	// makes the endpoint good for debugging.
	admin := previewAdminRouter(t)
	_, full := postPreviewAs(t, admin, string(body), adminSecret)
	if len(full.Candidates) == 0 || len(full.Rejected) == 0 {
		t.Fatalf("an admin lost the per-worker detail: %d candidates, %d rejections", len(full.Candidates), len(full.Rejected))
	}
	if full.WouldServe != pv.WouldServe || full.Route != pv.Route {
		t.Errorf("the two views disagree about the decision: %q/%q vs %q/%q",
			full.Route, full.WouldServe, pv.Route, pv.WouldServe)
	}
}

func TestRoutePreviewRejectsNonPost(t *testing.T) {
	r := previewRouter(t)
	rec := httptest.NewRecorder()
	r.handleRoutePreview(rec, httptest.NewRequest(http.MethodGet, "/v1/route-preview", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("want 405, got %d", rec.Code)
	}
}

// ── The per-key allow-list ──────────────────────────────────────────────────

const previewClientSecret = "sk-test-preview-client"

// restrictedPreviewRouter is a two-model fleet plus a client key allowed only
// the small one. "big" is faster and higher quality, so automatic routing
// prefers it and the allow-list is the only thing that can keep it out.
func restrictedPreviewRouter(t *testing.T, allowed ...string) *Router {
	t.Helper()
	reg := newTestRegistry()
	for _, spec := range []struct {
		id, model   string
		quality, ps int
	}{{"small", "small-model", 20, 200}, {"big", "big-model", 90, 400}} {
		reg.upsert(BackendRegistration{
			ID: spec.id, URL: "http://" + spec.id, Model: spec.model, Quality: spec.quality,
			BaselineTPS: float64(spec.ps), MaxConcurrency: 4, TTLSeconds: 3600,
			Features: []string{"chat"},
		})
		reg.finishCertification(spec.id, true, map[string]Check{}, float64(spec.ps), 10, "")
	}
	r := &Router{
		cfg: &Config{
			DefaultMaxTokens: 4096, AutoDifficulty: true, AutoThinking: true,
			DifficultyTemp: 0.10, ReasoningThreshold: 0.35,
			DifficultyTimeout: time.Second, DifficultyCacheSize: 16, DifficultyMaxChars: 4000,
		},
		registry:   reg,
		classifier: testClassifierAuto(fakeEmbed),
		sessions:   newSessionTracker(time.Hour, 16),
		logs:       newTestLogStore(t),
	}
	issueKey(t, r, previewClientSecret, apiKey{Role: roleClient, Name: "restricted", Models: allowed})
	return r
}

// A key restricted to one model must not be told that its request would land on
// a worker it may not be served by.
//
// The preview shared planRoute and stopped there, but the proxy path calls
// restrictToAllowList on the very next line — so the two disagreed for exactly
// the callers that have a restriction. allowsModel refuses nothing to a caller
// who named nothing, and "default" IS naming nothing (autoModelNames), so the
// preview ranked the whole fleet and handed back the id of the largest worker in
// it. Two failures at once: the answer was wrong, because the real request would
// have been narrowed to the allowed set, and would_serve is in the CLIENT half of
// the response, so it disclosed a worker id this key is not entitled to learn —
// the inventory that moving GET /backends to admin scope was meant to withhold.
func TestRoutePreviewAppliesTheKeysAllowList(t *testing.T) {
	r := restrictedPreviewRouter(t, "small-model")

	rec, pv := postPreviewAs(t, r,
		`{"model":"default","messages":[{"role":"user","content":"prove a hard theorem"}]}`,
		previewClientSecret)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if pv.WouldServe != "small" {
		t.Errorf("would_serve %q: a key allowed only small-model was routed to a worker it may not use", pv.WouldServe)
	}
	if strings.Contains(rec.Body.String(), "big") {
		t.Errorf("a worker outside the key's allow-list is named in its preview: %s", rec.Body.String())
	}

	// An unrestricted key still sees the whole fleet's answer, so the narrowing
	// above is the allow-list and not a change to the ranking.
	open := restrictedPreviewRouter(t)
	_, all := postPreviewAs(t, open,
		`{"model":"default","messages":[{"role":"user","content":"prove a hard theorem"}]}`,
		previewClientSecret)
	if all.WouldServe != "big" {
		t.Fatalf("without an allow-list the fleet's own choice should stand, got %q", all.WouldServe)
	}
}

// The same narrowing, in the case where it leaves nothing: the real request gets
// restrictToAllowList's 503, so the preview must report that rather than a 200
// describing a route the caller cannot have.
func TestRoutePreviewReportsAnEmptyAllowList(t *testing.T) {
	r := restrictedPreviewRouter(t, "a-model-nobody-serves")
	rec, _ := postPreviewAs(t, r,
		`{"model":"default","messages":[{"role":"user","content":"say hello"}]}`,
		previewClientSecret)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("want 503 when no allowed worker is a candidate, got %d: %s", rec.Code, rec.Body.String())
	}
}

// The other direction, and the more embarrassing one: the preview REFUSED a
// request /v1/chat/completions serves.
//
// enforceKeyLimits is allowsModel OR mayNameWorker, because an allow-list entry
// is matched against the caller's spelling literally — so a key issued for
// "small-model" naming the ID of the worker serving small-model is legitimate,
// and is the spelling /relay/fleet advertises. The preview tested only the first
// half and answered 403 to a request the router would have answered 200 to.
func TestRoutePreviewAcceptsAWorkerIDTheAllowListCovers(t *testing.T) {
	r := restrictedPreviewRouter(t, "small-model")
	rec, pv := postPreviewAs(t, r,
		`{"model":"small","messages":[{"role":"user","content":"say hello"}]}`,
		previewClientSecret)
	if rec.Code != http.StatusOK {
		t.Fatalf("naming the worker that serves an allowed model must be previewable, got %d: %s",
			rec.Code, rec.Body.String())
	}
	if pv.WouldServe != "small" {
		t.Errorf("would_serve %q, want small", pv.WouldServe)
	}
	// A model genuinely off the list is still refused, so the widening above did
	// not turn the allow-list into a formality.
	off, _ := postPreviewAs(t, r,
		`{"model":"big-model","messages":[{"role":"user","content":"say hello"}]}`,
		previewClientSecret)
	if off.Code != http.StatusForbidden {
		t.Fatalf("a model off the allow-list must still be 403, got %d: %s", off.Code, off.Body.String())
	}
}

// ── The quality bar, where there is one ─────────────────────────────────────

// above_bar must be the ranker's own test. rankByDifficulty and
// qualityFloorPreference both read qualityFor, which for a NO-THINK request on a
// thinking worker with no measured no-think score is 0 — never the mixed-mode
// number, because that score was earned with reasoning on (a 284B worker
// inheriting its mixed q=93 that way drew all of Atlas's planner traffic onto a
// 10 tok/s slot on 2026-08-25). The preview read b.Quality instead, so it
// reported the still-unmeasured worker as an above-bar front-runner on precisely
// the request the router scored 0 and ranked last.
func TestRoutePreviewAboveBarUsesTheRankedQuality(t *testing.T) {
	reg := newTestRegistry()
	readyBackend(reg, "plain", 55, 200, 4)                 // non-thinking: q=55 in either mode
	readyThinkingBackend(reg, "thinker", 95, 140, 4, true) // thinking: q=95, no-think score unmeasured
	r := &Router{
		cfg: &Config{
			DefaultMaxTokens: 4096, AutoDifficulty: true, AutoThinking: true,
			DifficultyTemp: 0.10, ReasoningThreshold: 0.35,
			DifficultyTimeout: time.Second, DifficultyCacheSize: 16, DifficultyMaxChars: 4000,
		},
		registry:   reg,
		classifier: testClassifierAuto(fakeEmbed),
		sessions:   newSessionTracker(time.Hour, 16),
		logs:       newTestLogStore(t),
	}
	issueKey(t, r, adminSecret, apiKey{Role: roleAdmin, Name: "preview admin"})

	_, pv := postPreviewAs(t, r,
		`{"model":"default","requirements":{"thinking":"off"},`+
			`"messages":[{"role":"user","content":"prove a hard theorem"}]}`, adminSecret)
	if pv.TargetQuality <= 0 {
		t.Skipf("no quality bar in play (target=%d); above_bar is correctly absent", pv.TargetQuality)
	}
	var found bool
	for _, c := range pv.Candidates {
		if c.ID != "thinker" {
			continue
		}
		found = true
		if c.Quality != 0 {
			t.Errorf("thinker previewed at q=%d on a no-think request; qualityFor reads 0 for an unmeasured no-think score", c.Quality)
		}
		if c.AboveBar == nil {
			t.Fatalf("a target of %d is a bar, so above_bar must be reported", pv.TargetQuality)
		}
		if *c.AboveBar {
			t.Errorf("thinker reported above a q>=%d bar on a no-think request the ranker scores it 0 for", pv.TargetQuality)
		}
	}
	if !found {
		t.Fatalf("thinker missing from the candidate list: %+v", pv.Candidates)
	}
}

// With no bar there is nothing for above_bar to report, and reporting a vacuous
// true for every worker on every request is how the field came to mean nothing.
// The matrix path always carries target 0, so this is the live shape.
func TestRoutePreviewOmitsAboveBarWithoutATarget(t *testing.T) {
	r := previewAdminRouter(t)
	body := `{"model":"default","messages":[{"role":"user","content":"prove a hard theorem"}]}`
	_, pv := postPreviewAs(t, r, body, adminSecret)
	if pv.TargetQuality != 0 {
		// previewRouter has no matrix, so the tier path runs and a bar exists.
		// Re-run through a router that does, which is the deployed shape.
		r.outcomes = newOutcomeMatrix(nil)
		_, pv = postPreviewAs(t, r, body, adminSecret)
	}
	if pv.TargetQuality != 0 {
		t.Fatalf("the outcome matrix path must carry no quality target, got %d", pv.TargetQuality)
	}
	if pv.AdapterBias != 0 {
		t.Errorf("adapter_bias=%v on a route the adapter was never consulted for", pv.AdapterBias)
	}
	if len(pv.Candidates) == 0 {
		t.Fatalf("admin preview lost its candidates")
	}
	for _, c := range pv.Candidates {
		if c.AboveBar != nil {
			t.Errorf("%s reports above_bar=%v with no bar to clear", c.ID, *c.AboveBar)
		}
		if c.Outcome == nil {
			t.Errorf("%s carries no outcome-matrix prediction, which is what ordered it", c.ID)
			continue
		}
		switch c.Outcome.Band {
		case "able", "unmeasured", "unable":
		default:
			t.Errorf("%s landed in band %q, which is not one of chooseByOutcome's three", c.ID, c.Outcome.Band)
		}
	}
	if !strings.Contains(strings.Join(pv.Notes, " "), "outcome matrix") {
		t.Errorf("the notes should say what decided this route: %v", pv.Notes)
	}
}
