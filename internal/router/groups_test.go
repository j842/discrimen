package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// groupRouter is a fleet of two named workers behind one fake endpoint, with a
// client key, so a group can be resolved end to end through the real mux.
func groupRouter(t *testing.T) *Router {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20}}`))
	}))
	t.Cleanup(srv.Close)

	reg := newTestRegistry()
	for _, w := range []struct {
		id, model string
		quality   int
		features  []string
	}{
		{"llm-a750", "gemma4", 50, []string{"chat"}},
		{"llm-naples", "qwen3", 70, []string{"chat", "tools"}},
	} {
		reg.upsert(BackendRegistration{
			ID: w.id, URL: srv.URL, Model: w.model, Quality: w.quality,
			TTLSeconds: 3600, Features: w.features,
		})
		reg.finishCertification(w.id, true, map[string]Check{}, 50, 10, "")
	}
	r := &Router{
		cfg:      &Config{DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second},
		registry: reg, logs: newTestLogStore(t),
		client: &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
	}
	issueKey(t, r, adminSecret, apiKey{Role: roleAdmin, Name: "test admin"})
	issueKey(t, r, clientSecret, apiKey{Role: roleClient, Name: "test client"})
	return r
}

// putGroup stores a group directly, for tests about routing rather than CRUD.
func putGroup(t *testing.T, r *Router, name string, members ...string) {
	t.Helper()
	g := Group{Name: name, Members: members}
	if bad := normalizeGroup(&g); bad != nil {
		t.Fatalf("normalizeGroup(%q): %s", name, bad.Message)
	}
	r.groups.put(g)
}

func chatWithModel(t *testing.T, r *Router, model string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/v1/chat/completions",
		`{"model":"`+model+`","messages":[{"role":"user","content":"hi"}]}`, clientSecret))
	return rec
}

// ── Resolution ──────────────────────────────────────────────────────────────

// TestGroupResolvesInOrder: the list is an ordered preference, and a member
// that is not registered is skipped rather than being an error — an operator
// may list a model they have not stood up yet.
func TestGroupResolvesInOrder(t *testing.T) {
	r := groupRouter(t)
	cases := []struct {
		name    string
		members []string
		want    string // the worker that must serve
	}{
		{"first member wins", []string{"qwen3", "gemma4"}, "llm-naples"},
		{"order is the preference, not quality", []string{"gemma4", "qwen3"}, "llm-a750"},
		{"an absent member is skipped", []string{"not-deployed-yet", "gemma4"}, "llm-a750"},
		{"a worker id is a member spelling too", []string{"llm-naples"}, "llm-naples"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			putGroup(t, r, "coding", tc.members...)
			rec := chatWithModel(t, r, "coding")
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
			}
			if got := rec.Header().Get("X-LLM-Backend-ID"); got != tc.want {
				t.Errorf("served by %q, want %q", got, tc.want)
			}
			// The chosen member is observable, the way X-LLM-Route already is.
			if got := rec.Header().Get("X-LLM-Group"); got != tc.members[len(tc.members)-1] && got != tc.members[0] {
				t.Errorf("X-LLM-Group = %q, want the member that was chosen", got)
			}
			if rec.Header().Get("X-LLM-Group") == "fallback" {
				t.Error("a group with a live member must not report a fallback")
			}
		})
	}
}

// TestGroupRespectsHardFilters: "qualifies" means past the hard filters, so a
// member that cannot do what the request needs is skipped like an absent one.
func TestGroupRespectsHardFilters(t *testing.T) {
	r := groupRouter(t)
	putGroup(t, r, "coding", "gemma4", "qwen3") // gemma4 first, but it has no tools
	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/v1/chat/completions",
		`{"model":"coding","tools":[{"type":"function","function":{"name":"f"}}],`+
			`"messages":[{"role":"user","content":"hi"}]}`, clientSecret))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("X-LLM-Backend-ID"); got != "llm-naples" {
		t.Errorf("served by %q, want the tools-capable member llm-naples", got)
	}
	if got := rec.Header().Get("X-LLM-Group"); got != "qwen3" {
		t.Errorf("X-LLM-Group = %q, want qwen3", got)
	}
}

// TestGroupFallsBackRatherThanRefusing is the rule the whole design turns on: a
// group is a preference, never a constraint. Nothing it names is available, so
// the filter is dropped and the request routes automatically — an answer plus a
// header saying what happened, not a 404.
func TestGroupFallsBackRatherThanRefusing(t *testing.T) {
	r := groupRouter(t)
	putGroup(t, r, "research", "never-deployed", "also-missing")
	rec := chatWithModel(t, r, "research")
	if rec.Code != http.StatusOK {
		t.Fatalf("a group with no live member must still be served, got %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("X-LLM-Group"); got != "fallback" {
		t.Errorf("X-LLM-Group = %q, want fallback", got)
	}
	if got := rec.Header().Get("X-LLM-Backend-ID"); got == "" {
		t.Error("the fallback served nothing")
	}
	// The router chose, so the route reads "route:" — which is what lets the tier
	// adapter and the background judge learn from it (see routeKind).
	if route := rec.Header().Get("X-LLM-Route"); !strings.HasPrefix(route, "route") {
		t.Errorf("X-LLM-Route = %q, want the automatic route after a fallback", route)
	}
}

// TestGroupNameIsCaseInsensitive: the router owns the name, so a client should
// not have to guess its capitalisation.
func TestGroupNameIsCaseInsensitive(t *testing.T) {
	r := groupRouter(t)
	putGroup(t, r, "  Coding  ", "qwen3")
	if g, ok := r.groups.lookup("CODING"); !ok || g.Name != "coding" {
		t.Fatalf("lookup(CODING) = %+v, ok=%v", g, ok)
	}
	if rec := chatWithModel(t, r, "CoDiNg"); rec.Code != http.StatusOK ||
		rec.Header().Get("X-LLM-Group") != "qwen3" {
		t.Errorf("mixed-case group name = %d, group=%q", rec.Code, rec.Header().Get("X-LLM-Group"))
	}
}

// TestGroupOutranksAModelOfTheSameName pins the precedence: group name, then
// model id, alias, worker id. The admin API refuses to create a group that
// shadows a live model, so this is the late-registration case — a worker that
// arrived carrying a name an operator had already claimed.
func TestGroupOutranksAModelOfTheSameName(t *testing.T) {
	r := groupRouter(t)
	putGroup(t, r, "gemma4", "qwen3") // "gemma4" is also llm-a750's model id
	rec := chatWithModel(t, r, "gemma4")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("X-LLM-Backend-ID"); got != "llm-naples" {
		t.Errorf("served by %q — the group must win the name, not the model", got)
	}
	// The worker that lost the name is still reachable by its own id.
	if rec := chatWithModel(t, r, "llm-a750"); rec.Code != http.StatusOK ||
		rec.Header().Get("X-LLM-Backend-ID") != "llm-a750" {
		t.Errorf("the shadowed worker became unreachable: %d %s", rec.Code, rec.Header().Get("X-LLM-Backend-ID"))
	}
}

// TestGroupResolveUnit exercises the resolver against the hard filter directly,
// including the case where the group's own name is the only thing a candidate
// would match — which must never be how a member resolves.
func TestGroupResolveUnit(t *testing.T) {
	reg := newTestRegistry()
	registerQ(t, reg, "w", 50, 1)
	cands := []*Backend{reg.get("w")} // model "default", features none
	g := Group{Name: "coding", Members: []string{"missing", "w"}}
	if member, ok := g.resolve(cands, hardFilter{}); !ok || member != "w" {
		t.Errorf("resolve = (%q, %v), want (w, true)", member, ok)
	}
	if _, ok := (Group{Name: "coding", Members: []string{"coding"}}).resolve(cands, hardFilter{}); ok {
		t.Error("a group must not resolve through its own name")
	}
	// The hard filter is respected: nothing here can think.
	if _, ok := g.resolve(cands, hardFilter{hardThink: true}); ok {
		t.Error("resolved a member that fails the hard filter")
	}
	if _, ok := (Group{Name: "empty"}).resolve(cands, hardFilter{}); ok {
		t.Error("a memberless group resolved to something")
	}
}

// ── The model menu ──────────────────────────────────────────────────────────

func TestGroupsAppearInTheModelMenu(t *testing.T) {
	r := groupRouter(t)
	putGroup(t, r, "coding", "qwen3", "gemma4")

	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, authGet("/v1/models", clientSecret))
	if rec.Code != http.StatusOK {
		t.Fatalf("models = %d: %s", rec.Code, rec.Body.String())
	}
	var list struct {
		Data []map[string]any `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatalf("models is not JSON: %v", err)
	}
	var entry map[string]any
	for _, m := range list.Data {
		if m["id"] == "coding" {
			entry = m
		}
	}
	if entry == nil {
		t.Fatalf("the group is not on the menu: %v", list.Data)
	}
	if entry["owned_by"] != routerOwner {
		t.Errorf("owned_by = %v, want %q — a group is the router's, not a worker's", entry["owned_by"], routerOwner)
	}
	if entry["group"] != true {
		t.Errorf("the menu row does not say it is a group: %v", entry)
	}
	// Features are the intersection over the registered members: llm-a750 has no
	// tools, so the group cannot promise them.
	if feats, _ := entry["features"].([]any); len(feats) != 1 || feats[0] != "chat" {
		t.Errorf("features = %v, want the intersection over the members", entry["features"])
	}

	// And the single-model fetch resolves it, whatever the capitalisation.
	for _, name := range []string{"coding", "CODING"} {
		rec = httptest.NewRecorder()
		r.routes().ServeHTTP(rec, authGet("/v1/models/"+name, clientSecret))
		if rec.Code != http.StatusOK {
			t.Fatalf("GET /v1/models/%s = %d: %s", name, rec.Code, rec.Body.String())
		}
		var one map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &one); err != nil || one["id"] != "coding" {
			t.Errorf("GET /v1/models/%s returned %v (err=%v)", name, one, err)
		}
	}
}

// TestMenuAdvertisesContextLength: harnesses size their compression window from
// the menu, and without a published context_length they fall back to family
// guesses (Hermes: any "qwen" → 131072). Pooled ids claim the smallest measured
// window (the router may land on any member), default/expert claim the fleet
// max (they context-filter per request), and a worker that never reported stays
// out of both the min and the row.
func TestMenuAdvertisesContextLength(t *testing.T) {
	reg := newTestRegistry()
	for _, w := range []struct {
		id, model string
		ctxK      int
	}{
		{"w-small", "qwen3", 32},
		{"w-big", "qwen3", 256},
		{"w-silent", "gemma4", 0},
	} {
		reg.upsert(BackendRegistration{
			ID: w.id, URL: "http://x", Model: w.model, Quality: 50,
			TTLSeconds: 3600, Features: []string{"chat"}, ContextK: w.ctxK,
		})
		reg.finishCertification(w.id, true, map[string]Check{}, 50, 10, "")
	}
	r := &Router{
		cfg:      &Config{DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second},
		registry: reg, logs: newTestLogStore(t),
	}
	putGroup(t, r, "coding", "qwen3", "gemma4")

	rows := map[string]map[string]any{}
	for _, m := range r.modelCatalogue() {
		rows[m["id"].(string)] = m
	}
	if got := rows["qwen3"]["context_length"]; got != 32*1024 {
		t.Errorf("pooled qwen3 context_length = %v, want the smallest member (32768)", got)
	}
	if _, ok := rows["gemma4"]["context_length"]; ok {
		t.Errorf("gemma4 published a context it never reported: %v", rows["gemma4"])
	}
	for _, id := range []string{"default", "expert"} {
		if got := rows[id]["context_length"]; got != 256*1024 {
			t.Errorf("%s context_length = %v, want the fleet max (262144)", id, got)
		}
	}
	if got := rows["coding"]["context_length"]; got != 32*1024 {
		t.Errorf("group context_length = %v, want the min over reporting members (32768)", got)
	}
}

// TestGroupReplacesAShadowedMenuRow: routing sends the name to the group, so the
// menu has to agree rather than advertising the worker that lost it.
func TestGroupReplacesAShadowedMenuRow(t *testing.T) {
	r := groupRouter(t)
	putGroup(t, r, "gemma4", "qwen3")
	rows := 0
	for _, m := range r.modelCatalogue() {
		if m["id"] == "gemma4" {
			rows++
			if m["owned_by"] != routerOwner {
				t.Errorf("the menu still points gemma4 at a worker: %v", m)
			}
		}
	}
	if rows != 1 {
		t.Errorf("gemma4 appears %d times on the menu; a duplicate id is a menu nobody can use", rows)
	}
}

// ── /v1/route-preview ───────────────────────────────────────────────────────

// TestRoutePreviewShowsGroupResolution: the preview exists so a decision can be
// inspected without paying for the answer, and a group that silently fell back
// is exactly the decision you would want to see.
func TestRoutePreviewShowsGroupResolution(t *testing.T) {
	r := groupRouter(t)
	putGroup(t, r, "coding", "missing", "qwen3")
	putGroup(t, r, "research", "never-deployed")

	preview := func(model string) previewResponse {
		t.Helper()
		rec := httptest.NewRecorder()
		r.routes().ServeHTTP(rec, post("/v1/route-preview",
			`{"model":"`+model+`","messages":[{"role":"user","content":"hi"}]}`, clientSecret))
		if rec.Code != http.StatusOK {
			t.Fatalf("preview %q = %d: %s", model, rec.Code, rec.Body.String())
		}
		var out previewResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("preview is not JSON: %v", err)
		}
		return out
	}

	got := preview("coding")
	if got.Group == nil || got.Group.Name != "coding" || got.Group.Member != "qwen3" || got.Group.Fallback {
		t.Errorf("resolved group preview = %+v", got.Group)
	}
	if len(got.Group.Members) != 2 {
		t.Errorf("the preview does not show what the group contains: %+v", got.Group)
	}
	if got.WouldServe != "llm-naples" {
		t.Errorf("would_serve = %q, want llm-naples", got.WouldServe)
	}

	got = preview("research")
	if got.Group == nil || !got.Group.Fallback || got.Group.Member != "" {
		t.Errorf("fallback preview = %+v", got.Group)
	}
	if !strings.Contains(strings.Join(got.Notes, " "), "fell back") {
		t.Errorf("the fallback is not explained in the notes: %v", got.Notes)
	}

	// No group named → no group section at all.
	if got = preview("qwen3"); got.Group != nil {
		t.Errorf("a plain model request reported a group: %+v", got.Group)
	}
}

// ── Admin CRUD ──────────────────────────────────────────────────────────────

func decodeGroup(t *testing.T, rec *httptest.ResponseRecorder) Group {
	t.Helper()
	var body struct {
		Group Group `json:"group"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("response is not JSON (%v): %s", err, rec.Body.String())
	}
	return body.Group
}

func TestAdminGroupCRUD(t *testing.T) {
	r := groupRouter(t)

	// ── create ──
	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/groups",
		`{"name":"  Coding  ","members":["qwen3"," qwen3 ","","llm-a750"]}`))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d: %s", rec.Code, rec.Body.String())
	}
	g := decodeGroup(t, rec)
	if g.Name != "coding" {
		t.Errorf("name = %q, want it lowercased and trimmed", g.Name)
	}
	// Blanks and duplicates go; the operator's own order and spelling stay,
	// because the order IS the preference.
	if len(g.Members) != 2 || g.Members[0] != "qwen3" || g.Members[1] != "llm-a750" {
		t.Errorf("members = %v", g.Members)
	}
	if g.UpdatedAt.IsZero() {
		t.Error("no updated_at stamp")
	}

	// It is persisted, so it survives a restart.
	saved, err := r.logs.LoadGroups(t.Context())
	if err != nil || len(saved) != 1 || saved[0].Name != "coding" || len(saved[0].Members) != 2 {
		t.Fatalf("persisted groups = %+v, err=%v", saved, err)
	}

	// ── list ──
	rec = serveAdmin(r, adminReq(http.MethodGet, "/admin/groups", ""))
	var list struct {
		Groups []Group `json:"groups"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatalf("list is not JSON: %v", err)
	}
	if len(list.Groups) != 1 || list.Groups[0].Name != "coding" {
		t.Fatalf("list = %+v", list.Groups)
	}

	// ── read one ──
	rec = serveAdmin(r, adminReq(http.MethodGet, "/admin/groups/CODING", ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("get = %d: %s", rec.Code, rec.Body.String())
	}

	// ── update ──
	rec = serveAdmin(r, adminReq(http.MethodPatch, "/admin/groups/coding", `{"members":["llm-a750"]}`))
	if rec.Code != http.StatusOK {
		t.Fatalf("update = %d: %s", rec.Code, rec.Body.String())
	}
	if g = decodeGroup(t, rec); len(g.Members) != 1 || g.Members[0] != "llm-a750" {
		t.Errorf("updated members = %v", g.Members)
	}
	if live, _ := r.groups.lookup("coding"); len(live.Members) != 1 {
		t.Errorf("the update did not reach the resolver: %+v", live)
	}

	// ── delete ──
	rec = serveAdmin(r, adminReq(http.MethodDelete, "/admin/groups/coding", ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("delete = %d: %s", rec.Code, rec.Body.String())
	}
	if _, ok := r.groups.lookup("coding"); ok {
		t.Error("the group survived the delete")
	}
	if saved, _ = r.logs.LoadGroups(t.Context()); len(saved) != 0 {
		t.Errorf("the persisted group survived the delete: %+v", saved)
	}
	if rec = serveAdmin(r, adminReq(http.MethodDelete, "/admin/groups/coding", "")); rec.Code != http.StatusNotFound {
		t.Errorf("deleting it twice = %d, want 404", rec.Code)
	}
}

// TestAdminGroupListReportsMemberResolution: the listing has to say which
// members the fleet can currently serve, because the admin UI must show an
// unresolved member as the legitimate thing it is rather than as an error — and
// cannot work it out for itself, since an alias is a server-side reduction of a
// raw model id.
func TestAdminGroupListReportsMemberResolution(t *testing.T) {
	r := groupRouter(t)
	// A worker whose raw model id only matches a member through its ALIAS, which
	// is the case a page comparing strings for itself would get wrong.
	r.registry.upsert(BackendRegistration{
		ID: "llm-turin", URL: "http://llm-turin", Model: "Qwen3.8-27B-Instruct-Q4_K_M", TTLSeconds: 3600,
	})
	// qwen3 is llm-naples's model, llm-a750 is a worker id, qwen3.8 is llm-turin's
	// alias, and gpt-4o is a model nobody has stood up.
	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/groups",
		`{"name":"coding","members":["qwen3","llm-a750","qwen3.8","gpt-4o"]}`))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d: %s", rec.Code, rec.Body.String())
	}

	rec = serveAdmin(r, adminReq(http.MethodGet, "/admin/groups", ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("list = %d: %s", rec.Code, rec.Body.String())
	}
	var list struct {
		Groups   []Group                      `json:"groups"`
		Resolves map[string]map[string]string `json:"resolves"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatalf("list is not JSON: %v", err)
	}
	got := list.Resolves["coding"]
	if got == nil {
		t.Fatalf("no resolution reported for the group: %s", rec.Body.String())
	}
	if got["qwen3"] != "llm-naples" {
		t.Errorf("member by model id resolved to %q, want llm-naples", got["qwen3"])
	}
	if got["llm-a750"] != "llm-a750" {
		t.Errorf("member by worker id resolved to %q, want llm-a750", got["llm-a750"])
	}
	if got["qwen3.8"] != "llm-turin" {
		t.Errorf("member by alias resolved to %q, want llm-turin", got["qwen3.8"])
	}
	// A member nothing serves is reported as unresolved, NOT omitted: the UI
	// distinguishes "listed and unserved" from "not listed".
	if serving, present := got["gpt-4o"]; !present || serving != "" {
		t.Errorf("unregistered member reported as %q (present=%v), want an empty string", serving, present)
	}
	if len(got) != 4 {
		t.Errorf("resolution covers %d members, want one entry per member: %v", len(got), got)
	}
}

func TestAdminGroupValidation(t *testing.T) {
	r := groupRouter(t)
	serveAdmin(r, adminReq(http.MethodPost, "/admin/groups", `{"name":"coding","members":["qwen3"]}`))

	cases := []struct {
		name, method, path, body string
		status                   int
	}{
		{"no name", http.MethodPost, "/admin/groups", `{"members":["qwen3"]}`, http.StatusBadRequest},
		{"blank name", http.MethodPost, "/admin/groups", `{"name":"   ","members":["qwen3"]}`, http.StatusBadRequest},
		{"no members", http.MethodPost, "/admin/groups", `{"name":"empty"}`, http.StatusBadRequest},
		{"members are all blank", http.MethodPost, "/admin/groups", `{"name":"empty","members":["","  "]}`, http.StatusBadRequest},
		{"invalid json", http.MethodPost, "/admin/groups", `{`, http.StatusBadRequest},
		{"already exists", http.MethodPost, "/admin/groups", `{"name":"CODING","members":["qwen3"]}`, http.StatusConflict},
		// A name that is already servable would be taken over by the group, since
		// groups resolve first. Refuse the write rather than redirect traffic.
		{"shadows a model id", http.MethodPost, "/admin/groups", `{"name":"gemma4","members":["qwen3"]}`, http.StatusConflict},
		{"shadows a worker id", http.MethodPost, "/admin/groups", `{"name":"llm-naples","members":["qwen3"]}`, http.StatusConflict},
		{"shadows the auto route", http.MethodPost, "/admin/groups", `{"name":"default","members":["qwen3"]}`, http.StatusConflict},
		{"renaming in place", http.MethodPatch, "/admin/groups/coding", `{"name":"other"}`, http.StatusConflict},
		{"unknown group", http.MethodPatch, "/admin/groups/nope", `{"members":["qwen3"]}`, http.StatusNotFound},
		{"wrong method", http.MethodPut, "/admin/groups", `{}`, http.StatusMethodNotAllowed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := serveAdmin(r, adminReq(tc.method, tc.path, tc.body))
			if rec.Code != tc.status {
				t.Fatalf("status = %d, want %d: %s", rec.Code, tc.status, rec.Body.String())
			}
		})
	}
	// An ALIAS is a servable spelling too, and the one most likely to be typed —
	// it is what /v1/models publishes. Use a worker whose alias differs from its
	// model id so the case is distinct from the one above.
	r.registry.upsert(BackendRegistration{
		ID: "llm-a4000", URL: "http://a4000", Model: "Granite-4.1-8b-Q4_K_M",
		TTLSeconds: 3600, Features: []string{"chat"},
	})
	alias := backendAlias(r.registry.get("llm-a4000"))
	if alias == "" || strings.EqualFold(alias, "Granite-4.1-8b-Q4_K_M") {
		t.Fatalf("this case needs an alias distinct from the model id, got %q", alias)
	}
	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/groups",
		`{"name":"`+alias+`","members":["qwen3"]}`))
	if rec.Code != http.StatusConflict {
		t.Errorf("a group shadowing the alias %q = %d, want 409: %s", alias, rec.Code, rec.Body.String())
	}
}

// TestAdminGroupsRequireAdmin: groups decide where traffic goes, so writing one
// is an operator action behind the same single gate as every other.
func TestAdminGroupsRequireAdmin(t *testing.T) {
	r := groupRouter(t)
	putGroup(t, r, "coding", "qwen3")
	for _, tc := range []struct{ method, path, body string }{
		{http.MethodGet, "/admin/groups", ""},
		{http.MethodPost, "/admin/groups", `{"name":"x","members":["qwen3"]}`},
		{http.MethodGet, "/admin/groups/coding", ""},
		{http.MethodPatch, "/admin/groups/coding", `{"members":["qwen3"]}`},
		{http.MethodDelete, "/admin/groups/coding", ""},
	} {
		for _, cred := range []string{"", clientSecret} {
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			if cred != "" {
				req.Header.Set("Authorization", "Bearer "+cred)
			}
			r.routes().ServeHTTP(rec, req)
			if rec.Code != http.StatusUnauthorized {
				t.Errorf("%s %s with cred %q = %d, want 401", tc.method, tc.path, cred, rec.Code)
			}
		}
	}
}

func TestGroupPersistenceRoundTrip(t *testing.T) {
	logs := newTestLogStore(t)
	ctx := t.Context()
	if err := logs.SaveGroup(ctx, Group{Name: "coding", Members: []string{"a,b", "c"}, UpdatedAt: time.Now().UTC()}); err != nil {
		t.Fatalf("SaveGroup: %v", err)
	}
	// Same name again is an update, not a second row.
	if err := logs.SaveGroup(ctx, Group{Name: "coding", Members: []string{"z"}, UpdatedAt: time.Now().UTC()}); err != nil {
		t.Fatalf("SaveGroup update: %v", err)
	}
	got, err := logs.LoadGroups(ctx)
	if err != nil || len(got) != 1 {
		t.Fatalf("LoadGroups = %+v, err=%v", got, err)
	}
	if len(got[0].Members) != 1 || got[0].Members[0] != "z" {
		t.Errorf("members = %v, want the updated list", got[0].Members)
	}
	// A member with a comma in it round-trips, which is why members are JSON.
	if err := logs.SaveGroup(ctx, Group{Name: "odd", Members: []string{"a,b"}}); err != nil {
		t.Fatalf("SaveGroup comma: %v", err)
	}
	got, _ = logs.LoadGroups(ctx)
	for _, g := range got {
		if g.Name == "odd" && (len(g.Members) != 1 || g.Members[0] != "a,b") {
			t.Errorf("comma member did not round-trip: %v", g.Members)
		}
	}
	// And the startup load reaches the resolver.
	r := &Router{logs: logs}
	r.loadGroups(ctx)
	if g, ok := r.groups.lookup("coding"); !ok || len(g.Members) != 1 {
		t.Errorf("startup load = %+v, ok=%v", g, ok)
	}
}
