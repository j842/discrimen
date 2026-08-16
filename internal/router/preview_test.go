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

func postPreview(t *testing.T, r *Router, body string) (*httptest.ResponseRecorder, previewResponse) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/route-preview", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
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
func TestRoutePreviewMatchesTheRealDecision(t *testing.T) {
	r := previewRouter(t)
	body := `{"model":"default","messages":[{"role":"user","content":"prove a hard theorem"}]}`

	rec, pv := postPreview(t, r, body)
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
// answer, so a hard-filtered worker must come back named, with the reason.
func TestRoutePreviewExplainsRejections(t *testing.T) {
	r := previewRouter(t)
	long := strings.Repeat("context ", 12000) // ~12k tokens, over shortctx's 4K
	body, err := json.Marshal(map[string]any{
		"model":    "default",
		"messages": []map[string]string{{"role": "user", "content": long}},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, pv := postPreview(t, r, string(body))

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

func TestRoutePreviewRejectsNonPost(t *testing.T) {
	r := previewRouter(t)
	rec := httptest.NewRecorder()
	r.handleRoutePreview(rec, httptest.NewRequest(http.MethodGet, "/v1/route-preview", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("want 405, got %d", rec.Code)
	}
}
