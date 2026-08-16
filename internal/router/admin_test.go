package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
