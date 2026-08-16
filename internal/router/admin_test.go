package router

import (
	"crypto/tls"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

// adminRouter is a router with a real log store (so persistence and the keys
// table are both exercised) holding one admin key, which is what the admin
// surface is gated behind.
//
// adminSecret is a FIXED test credential rather than a generated one, so
// adminReq can stay a plain function. Nothing is weakened by fixing it:
// CreateAPIKey stores the SHA-256 of whatever plaintext it is handed, so the
// hash-and-lookup path is exactly the production one.
const adminSecret = "sk-test-admin-key"

func adminRouter(t *testing.T) *Router {
	t.Helper()
	r := &Router{
		cfg:      &Config{},
		registry: newTestRegistry(),
		logs:     newTestLogStore(t),
		client:   &http.Client{Timeout: time.Second},
	}
	issueKey(t, r, adminSecret, apiKey{Role: roleAdmin, Name: "test admin"})
	return r
}

// issueKey stores a key with a known plaintext and refreshes the cached
// "is a credential required" flags, exactly as the create handler does.
func issueKey(t *testing.T, r *Router, plain string, key apiKey) apiKey {
	t.Helper()
	key.Enabled = true
	key.CreatedAt = time.Now().UTC()
	if key.Prefix == "" {
		key.Prefix = plain
	}
	stored, err := r.logs.CreateAPIKey(t.Context(), plain, key)
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}
	r.refreshAuthRequired(t.Context())
	return stored
}

// adminReq builds an authorised admin request.
func adminReq(method, path, body string) *http.Request {
	var req *http.Request
	if body == "" {
		req = httptest.NewRequest(method, path, nil)
	} else {
		req = httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Authorization", "Bearer "+adminSecret)
	return req
}

// serveAdmin dispatches through the real mux, so a route that is wired to the
// wrong pattern (the /v1/models mistake) cannot hide behind a direct call.
func serveAdmin(r *Router, req *http.Request) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, req)
	return rec
}

func decodeProvider(t *testing.T, rec *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var body struct {
		Provider map[string]any `json:"provider"`
		Seeded   []string       `json:"seeded_from_price_data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("response is not JSON (%v): %s", err, rec.Body.String())
	}
	if body.Provider == nil {
		t.Fatalf("response carries no provider: %s", rec.Body.String())
	}
	return body.Provider
}

func TestAdminProviderCRUD(t *testing.T) {
	r := adminRouter(t)

	// ── create ──
	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/providers",
		`{"url":"https://api.example.com/v1","model":"gpt-4o","provider":"openai","api_key":"sk-upstream","max_concurrency":8}`))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d: %s", rec.Code, rec.Body.String())
	}
	row := decodeProvider(t, rec)
	if row["id"] != "openai-gpt-4o" {
		t.Errorf("generated id = %v, want openai-gpt-4o", row["id"])
	}
	if row["source"] != sourceManual || row["provider"] != "openai" {
		t.Errorf("row is not an operator-owned openai row: %v", row)
	}
	// The upstream credential must never come back out, same rule as /backends.
	if v, ok := row["api_key"]; ok && v != "" {
		t.Errorf("create response leaked the upstream api key: %v", v)
	}
	if !strings.Contains(rec.Body.String(), "seeded_from_price_data") {
		t.Error("prices were not seeded, and nothing said so")
	}
	// A provider will not answer /health; /v1/models is the default that works.
	if row["health_path"] != providerHealthPath {
		t.Errorf("health path = %v, want %q", row["health_path"], providerHealthPath)
	}

	// The api key is still stored (encrypted) and reloadable.
	saved, err := r.logs.LoadBackendRegistrations(t.Context())
	if err != nil || len(saved) != 1 {
		t.Fatalf("persisted registrations = %d, err=%v", len(saved), err)
	}
	if saved[0].APIKey != "sk-upstream" || saved[0].Source != sourceManual {
		t.Errorf("persisted row wrong: %+v", saved[0])
	}

	// ── list ──
	rec = serveAdmin(r, adminReq(http.MethodGet, "/admin/providers", ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("list = %d: %s", rec.Code, rec.Body.String())
	}
	var list struct {
		Providers []map[string]any `json:"providers"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatalf("list is not JSON: %v", err)
	}
	if len(list.Providers) != 1 || list.Providers[0]["id"] != "openai-gpt-4o" {
		t.Fatalf("list = %v", list.Providers)
	}

	// ── update ──
	rec = serveAdmin(r, adminReq(http.MethodPatch, "/admin/providers/openai-gpt-4o",
		`{"max_concurrency":4,"input_price_per_mtok":0,"output_price_per_mtok":0}`))
	if rec.Code != http.StatusOK {
		t.Fatalf("update = %d: %s", rec.Code, rec.Body.String())
	}
	row = decodeProvider(t, rec)
	if row["max_concurrency"] != float64(4) {
		t.Errorf("max_concurrency = %v, want 4", row["max_concurrency"])
	}
	// An explicit zero declares the model free and must survive re-seeding —
	// that is the whole reason the write shape uses pointers.
	if _, present := row["input_price_per_mtok"]; present {
		t.Errorf("price was re-seeded over an explicit 0: %v", row["input_price_per_mtok"])
	}

	// ── delete ──
	rec = serveAdmin(r, adminReq(http.MethodDelete, "/admin/providers/openai-gpt-4o", ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("delete = %d: %s", rec.Code, rec.Body.String())
	}
	if r.registry.get("openai-gpt-4o") != nil {
		t.Error("row survived the delete")
	}
	saved, _ = r.logs.LoadBackendRegistrations(t.Context())
	if len(saved) != 0 {
		t.Errorf("persisted row survived the delete: %+v", saved)
	}
}

// TestAdminProviderRequiresAuth: the write surface is scoped, and every method
// on it is scoped. P2 gates it behind the client token; P3 tightens the same
// single gate to admin.
func TestAdminProviderRequiresAuth(t *testing.T) {
	r := adminRouter(t)
	r.registry.upsert(manualReg(t, BackendRegistration{ID: "p", Model: "m", TTLSeconds: 3600}))

	for _, tc := range []struct{ method, path, body string }{
		{http.MethodGet, "/admin/providers", ""},
		{http.MethodPost, "/admin/providers", `{"url":"http://x","model":"m2"}`},
		{http.MethodGet, "/admin/providers/p", ""},
		{http.MethodPatch, "/admin/providers/p", `{"quality":9}`},
		{http.MethodDelete, "/admin/providers/p", ""},
	} {
		t.Run(tc.method+" "+tc.path, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body))
			rec := serveAdmin(r, req)
			if rec.Code != http.StatusUnauthorized {
				t.Fatalf("unauthenticated = %d, want 401: %s", rec.Code, rec.Body.String())
			}
			if body := errorEnvelopeOf(t, rec); body.Type != "authentication_error" {
				t.Errorf("error type = %q", body.Type)
			}
		})
	}
}

// TestAdminProviderOneRowPerEndpointModel: the router treats a row as one
// servable thing, so a catalogue endpoint gets one row per model. Two rows
// agreeing on both would double that model's share of the ranker.
func TestAdminProviderOneRowPerEndpointModel(t *testing.T) {
	r := adminRouter(t)
	create := func(body string) *httptest.ResponseRecorder {
		return serveAdmin(r, adminReq(http.MethodPost, "/admin/providers", body))
	}
	if rec := create(`{"id":"a","url":"https://api.example.com/v1","model":"gpt-4o"}`); rec.Code != http.StatusCreated {
		t.Fatalf("first create = %d: %s", rec.Code, rec.Body.String())
	}
	// Same endpoint, same model, different id — and the URL spelled without the
	// /v1, which reaches the identical endpoint.
	rec := create(`{"id":"b","url":"https://api.example.com","model":"gpt-4o"}`)
	if rec.Code != http.StatusConflict {
		t.Fatalf("duplicate (endpoint, model) = %d, want 409: %s", rec.Code, rec.Body.String())
	}
	// A second MODEL on the same endpoint is exactly what a catalogue looks like.
	if rec := create(`{"id":"c","url":"https://api.example.com/v1","model":"gpt-4o-mini"}`); rec.Code != http.StatusCreated {
		t.Fatalf("second model on the same endpoint = %d, want 201: %s", rec.Code, rec.Body.String())
	}
	// And the id itself still has to be unique.
	if rec := create(`{"id":"a","url":"https://other.example.com","model":"whatever"}`); rec.Code != http.StatusConflict {
		t.Fatalf("duplicate id = %d, want 409", rec.Code)
	}
}

// TestAdminProviderRefusesBeaconRows: a worker's values belong to the worker.
// Accepting an edit here would look like it worked and evaporate on the next
// keepalive.
func TestAdminProviderRefusesBeaconRows(t *testing.T) {
	r := adminRouter(t)
	r.registry.upsert(BackendRegistration{ID: "llm-a750", URL: "http://a750", Model: "gemma4", TTLSeconds: 3600})

	for _, method := range []string{http.MethodGet, http.MethodPatch, http.MethodDelete} {
		rec := serveAdmin(r, adminReq(method, "/admin/providers/llm-a750", `{"quality":9}`))
		if rec.Code != http.StatusConflict {
			t.Errorf("%s on a beacon row = %d, want 409: %s", method, rec.Code, rec.Body.String())
		}
	}
	if r.registry.get("llm-a750") == nil {
		t.Fatal("the beacon row was deleted through the provider API")
	}
}

// TestAdminProviderUpdatePatchesDeclaredValues: an edit applies to what the
// operator typed. Patching the live row instead would fold in every measured
// value as a declaration, and the probe could never refine those fields again.
func TestAdminProviderUpdatePatchesDeclaredValues(t *testing.T) {
	r := adminRouter(t)
	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/providers",
		`{"id":"p","url":"https://api.example.com/v1","model":"some-unpriced-model","max_concurrency":8}`))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d: %s", rec.Code, rec.Body.String())
	}
	// The profiler measures a quality and a context the operator never declared.
	b := r.registry.get("p")
	r.registry.applyProfileIfGen("p", b.profileGen, &WorkerProfile{
		Quality: 77, ContextK: 200, BenchVersion: benchmarkVersion,
	})

	rec = serveAdmin(r, adminReq(http.MethodPatch, "/admin/providers/p", `{"max_concurrency":2}`))
	if rec.Code != http.StatusOK {
		t.Fatalf("update = %d: %s", rec.Code, rec.Body.String())
	}
	declared, _ := r.registry.declaredRegistration("p")
	if declared.Quality != 0 || declared.ContextK != 0 {
		t.Errorf("the edit promoted measurements to declarations: %+v", declared)
	}
	if declared.MaxConcurrency != 2 {
		t.Errorf("declared max_concurrency = %d, want 2", declared.MaxConcurrency)
	}
}

// TestAdminProviderValidation covers the refusals a form needs to be told about.
func TestAdminProviderValidation(t *testing.T) {
	r := adminRouter(t)
	cases := []struct {
		name, body string
		status     int
	}{
		{"no model", `{"url":"https://api.example.com"}`, http.StatusBadRequest},
		{"no url", `{"model":"gpt-4o"}`, http.StatusBadRequest},
		{"bad url", `{"url":"not a url","model":"gpt-4o"}`, http.StatusBadRequest},
		{"bad json", `{`, http.StatusBadRequest},
		{"quality out of range", `{"url":"https://x.example.com","model":"m","quality":900}`, http.StatusBadRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/providers", tc.body))
			if rec.Code != tc.status {
				t.Fatalf("status = %d, want %d: %s", rec.Code, tc.status, rec.Body.String())
			}
			errorEnvelopeOf(t, rec)
		})
	}
	if rec := serveAdmin(r, adminReq(http.MethodPatch, "/admin/providers/nope", `{}`)); rec.Code != http.StatusNotFound {
		t.Errorf("patching an unknown row = %d, want 404", rec.Code)
	}
}

// TestAdminProviderSeedsFromPriceData: an operator types a base URL and a model
// id, and the published price arrives with it. Skipped on a build with the
// empty snapshot, which is a legitimate one.
func TestAdminProviderSeedsFromPriceData(t *testing.T) {
	if len(prices().exact) == 0 {
		t.Skip("prices.json is the empty snapshot")
	}
	r := adminRouter(t)
	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/providers",
		`{"url":"https://api.openai.com/v1","model":"gpt-4o","provider":"openai"}`))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d: %s", rec.Code, rec.Body.String())
	}
	row := decodeProvider(t, rec)
	in, _ := row["input_price_per_mtok"].(float64)
	out, _ := row["output_price_per_mtok"].(float64)
	if in <= 0 || out <= 0 || out <= in {
		t.Errorf("seeded prices look wrong: in=%v out=%v", in, out)
	}
	if ctx, _ := row["context_k"].(float64); ctx <= 0 {
		t.Errorf("context window not seeded: %v", row["context_k"])
	}
}

// TestAdminProviderListCarriesPriceReference: seeding leaves no trace on the
// row it seeds, so the listing has to carry the table's own answer alongside it
// or the admin page cannot attribute a single number it is showing.
func TestAdminProviderListCarriesPriceReference(t *testing.T) {
	if len(prices().exact) == 0 {
		t.Skip("prices.json is the empty snapshot")
	}
	r := adminRouter(t)
	serveAdmin(r, adminReq(http.MethodPost, "/admin/providers",
		`{"url":"https://api.openai.com/v1","model":"gpt-4o","provider":"openai"}`))
	// Nothing publishes a price for a local llama.cpp build, which is the normal
	// case and must simply be absent rather than zero.
	serveAdmin(r, adminReq(http.MethodPost, "/admin/providers",
		`{"id":"homelab","url":"http://192.0.2.9:8080","model":"some-local-build-Q4_K_M"}`))

	rec := serveAdmin(r, adminReq(http.MethodGet, "/admin/providers", ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("list = %d: %s", rec.Code, rec.Body.String())
	}
	var body struct {
		Providers []map[string]any          `json:"providers"`
		Reference map[string]publishedPrice `json:"price_reference"`
		Source    map[string]string         `json:"price_source"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("list is not JSON: %v", err)
	}
	ref, ok := body.Reference["openai-gpt-4o"]
	if !ok {
		t.Fatalf("no published price reported for a priced row: %s", rec.Body.String())
	}
	if ref.InputPricePerMtok <= 0 || ref.OutputPricePerMtok <= 0 || ref.ContextK <= 0 {
		t.Errorf("published reference looks wrong: %+v", ref)
	}
	// The stored row and the reference have to AGREE, or the page's "this is
	// what LiteLLM publishes" marker would never appear on a row that was seeded
	// from exactly that entry.
	for _, row := range body.Providers {
		if row["id"] != "openai-gpt-4o" {
			continue
		}
		if in, _ := row["input_price_per_mtok"].(float64); in != ref.InputPricePerMtok {
			t.Errorf("seeded row price %v != published %v", in, ref.InputPricePerMtok)
		}
	}
	if _, published := body.Reference["homelab"]; published {
		t.Errorf("a local build was given a published price: %+v", body.Reference["homelab"])
	}
	if body.Source["source"] != priceSourceURL || body.Source["fetched_at"] == "" {
		t.Errorf("price source not attributed: %+v", body.Source)
	}
}

// ── Admin session ───────────────────────────────────────────────────────────

// passwordRouter is a router whose admin password is set, without an admin key,
// so the session path is the only way in.
//
// The stored hash is written at a low iteration count. verifyPassword reads the
// cost out of the hash, so the production verification path is exercised
// unchanged — this only avoids paying 600k rounds of PBKDF2 per test setup,
// which under -race is most of the suite's runtime. TestPasswordHashing covers
// the real cost.
func passwordRouter(t *testing.T, password string) *Router {
	t.Helper()
	r := &Router{cfg: &Config{}, registry: newTestRegistry(), logs: newTestLogStore(t)}
	hash, err := hashPasswordIter(password, 1000)
	if err != nil {
		t.Fatalf("hashPassword: %v", err)
	}
	if err := r.logs.SaveSetting(t.Context(), settingAdminPasswordHash, hash); err != nil {
		t.Fatalf("SaveSetting: %v", err)
	}
	return r
}

func TestAdminSessionLifecycle(t *testing.T) {
	const password = "a-long-enough-password"
	r := passwordRouter(t, password)

	// A wrong password gets nothing, in the OpenAI envelope.
	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/admin/login", `{"password":"nope"}`, ""))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong password = %d, want 401", rec.Code)
	}
	if len(rec.Result().Cookies()) != 0 {
		t.Error("a failed login set a cookie")
	}
	errorEnvelopeOf(t, rec)

	// The right one issues a session.
	rec = httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/admin/login", `{"password":"`+password+`"}`, ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("login = %d: %s", rec.Code, rec.Body.String())
	}
	cookies := rec.Result().Cookies()
	if len(cookies) != 1 {
		t.Fatalf("login set %d cookies", len(cookies))
	}
	session := cookies[0]
	if session.Name != adminCookie {
		t.Errorf("cookie name = %q", session.Name)
	}
	if !session.HttpOnly {
		t.Error("the session cookie is readable from JavaScript")
	}
	if session.SameSite != http.SameSiteLaxMode {
		t.Errorf("SameSite = %v, want Lax", session.SameSite)
	}
	if session.Secure {
		t.Error("Secure was set on a plain-HTTP request, which makes the cookie unusable on a LAN deployment")
	}
	if session.MaxAge <= 0 || time.Duration(session.MaxAge)*time.Second != adminSessionTTL {
		t.Errorf("MaxAge = %d, want the bounded %s", session.MaxAge, adminSessionTTL)
	}
	if session.Value == "" || strings.Contains(session.Value, password) {
		t.Errorf("session token is empty or carries the password: %q", session.Value)
	}

	// The cookie is admin everywhere the bearer key would be.
	withSession := func(method, path string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(method, path, nil)
		req.AddCookie(session)
		rec := httptest.NewRecorder()
		r.routes().ServeHTTP(rec, req)
		return rec
	}
	if rec := withSession(http.MethodGet, "/backends"); rec.Code != http.StatusOK {
		t.Fatalf("session did not authorise /backends: %d", rec.Code)
	}
	if rec := withSession(http.MethodGet, "/admin/session"); !strings.Contains(rec.Body.String(), `"admin":true`) {
		t.Errorf("/admin/session with a session = %s", rec.Body.String())
	}

	// Logout invalidates it SERVER-SIDE: a client that keeps the cookie is still
	// locked out, which a cookie-clearing logout alone would not achieve.
	req := httptest.NewRequest(http.MethodPost, "/admin/logout", nil)
	req.AddCookie(session)
	rec = httptest.NewRecorder()
	r.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("logout = %d", rec.Code)
	}
	if rec := withSession(http.MethodGet, "/backends"); rec.Code != http.StatusUnauthorized {
		t.Fatalf("a revoked session still works: %d", rec.Code)
	}
}

// TestAdminSessionSecureOverTLS: the cookie is Secure when — and only when — the
// request arrived over TLS. Unconditional would break the plain-HTTP LAN
// deployment outright; never would leak the session on a reverse-proxied one.
func TestAdminSessionSecureOverTLS(t *testing.T) {
	const password = "a-long-enough-password"
	r := passwordRouter(t, password)
	req := post("https://router.example.com/admin/login", `{"password":"`+password+`"}`, "")
	req.TLS = &tls.ConnectionState{}
	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("login = %d: %s", rec.Code, rec.Body.String())
	}
	if c := rec.Result().Cookies(); len(c) != 1 || !c[0].Secure {
		t.Fatalf("cookie over TLS is not Secure: %+v", c)
	}
}

func TestAdminPasswordChange(t *testing.T) {
	r := passwordRouter(t, "the-original-password")
	issueKey(t, r, adminSecret, apiKey{Role: roleAdmin, Name: "admin"})

	change := func(body string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		r.routes().ServeHTTP(rec, post("/admin/password", body, adminSecret))
		return rec
	}
	// The current password is required even holding an admin credential: a
	// session left open on a shared machine must not lock the operator out.
	if rec := change(`{"current_password":"wrong","new_password":"a-new-long-password"}`); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong current password = %d, want 401", rec.Code)
	}
	if rec := change(`{"current_password":"the-original-password","new_password":"short"}`); rec.Code != http.StatusBadRequest {
		t.Fatalf("short new password = %d, want 400", rec.Code)
	}
	if rec := change(`{"current_password":"the-original-password","new_password":"a-new-long-password"}`); rec.Code != http.StatusOK {
		t.Fatalf("change = %d: %s", rec.Code, rec.Body.String())
	}
	hash, _ := r.logs.LoadSetting(t.Context(), settingAdminPasswordHash)
	if !verifyPassword(hash, "a-new-long-password") || verifyPassword(hash, "the-original-password") {
		t.Fatal("the stored password was not replaced")
	}
	// And unauthenticated callers never get there.
	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/admin/password", `{"current_password":"a-new-long-password","new_password":"another-long-one"}`, ""))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated password change = %d, want 401", rec.Code)
	}
}

// ── Key CRUD ────────────────────────────────────────────────────────────────

func TestAdminKeyCRUD(t *testing.T) {
	r := adminRouter(t)

	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/keys",
		`{"name":"harness","role":"client","models":["gemma4"],"token_budget":5000}`))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create = %d: %s", rec.Code, rec.Body.String())
	}
	var created struct {
		Key    apiKey `json:"key"`
		Secret string `json:"secret"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &created); err != nil {
		t.Fatalf("create response is not JSON: %v", err)
	}
	if !strings.HasPrefix(created.Secret, keyPrefix) {
		t.Fatalf("no secret in the create response: %s", rec.Body.String())
	}
	if created.Key.Role != roleClient || created.Key.TokenBudget != 5000 || len(created.Key.Models) != 1 {
		t.Errorf("stored key wrong: %+v", created.Key)
	}

	// Shown ONCE: the list never carries it.
	rec = serveAdmin(r, adminReq(http.MethodGet, "/admin/keys", ""))
	if rec.Code != http.StatusOK {
		t.Fatalf("list = %d", rec.Code)
	}
	if strings.Contains(rec.Body.String(), created.Secret) {
		t.Fatal("the key list carries the plaintext secret")
	}
	if !strings.Contains(rec.Body.String(), created.Key.Prefix) {
		t.Error("the key list does not show the prefix, so a key cannot be identified")
	}

	// The role is immutable.
	id := strconv.FormatInt(created.Key.ID, 10)
	if rec := serveAdmin(r, adminReq(http.MethodPatch, "/admin/keys/"+id, `{"role":"admin"}`)); rec.Code != http.StatusConflict {
		t.Fatalf("re-roling a key = %d, want 409: %s", rec.Code, rec.Body.String())
	}

	// Disabling is a revoke, and it takes effect immediately.
	if _, ok := r.logs.LookupAPIKey(t.Context(), created.Secret); !ok {
		t.Fatal("the new key does not authenticate")
	}
	if rec := serveAdmin(r, adminReq(http.MethodPatch, "/admin/keys/"+id, `{"enabled":false}`)); rec.Code != http.StatusOK {
		t.Fatalf("disable = %d: %s", rec.Code, rec.Body.String())
	}
	if _, ok := r.logs.LookupAPIKey(t.Context(), created.Secret); ok {
		t.Fatal("a disabled key still authenticates")
	}

	if rec := serveAdmin(r, adminReq(http.MethodDelete, "/admin/keys/"+id, "")); rec.Code != http.StatusOK {
		t.Fatalf("delete = %d", rec.Code)
	}
	if rec := serveAdmin(r, adminReq(http.MethodDelete, "/admin/keys/"+id, "")); rec.Code != http.StatusNotFound {
		t.Fatalf("deleting twice = %d, want 404", rec.Code)
	}
}

func TestAdminKeyValidation(t *testing.T) {
	r := adminRouter(t)
	for _, tc := range []struct {
		name, body string
		status     int
	}{
		{"unknown role", `{"role":"superuser"}`, http.StatusBadRequest},
		{"negative budget", `{"role":"client","token_budget":-1}`, http.StatusBadRequest},
		{"bad json", `{`, http.StatusBadRequest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/keys", tc.body))
			if rec.Code != tc.status {
				t.Fatalf("status = %d, want %d: %s", rec.Code, tc.status, rec.Body.String())
			}
			errorEnvelopeOf(t, rec)
		})
	}
	// Role defaults to client, which is the one anybody wants most of the time.
	rec := serveAdmin(r, adminReq(http.MethodPost, "/admin/keys", `{"name":"unspecified"}`))
	if rec.Code != http.StatusCreated || !strings.Contains(rec.Body.String(), `"role":"client"`) {
		t.Fatalf("default role: %d %s", rec.Code, rec.Body.String())
	}
	if rec := serveAdmin(r, adminReq(http.MethodPatch, "/admin/keys/not-a-number", `{}`)); rec.Code != http.StatusNotFound {
		t.Errorf("non-numeric key id = %d, want 404", rec.Code)
	}
}
