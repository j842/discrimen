package router

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// ── Migrations against a database the OLD code created ──────────────────────

// oldSchema is the schema openLogStore built before P2 and P3, copied verbatim
// from LogStore.init as it stood. Written out in full rather than derived, so a
// future edit to init cannot quietly change what "the old database" means.
var oldSchema = []string{
	`CREATE TABLE IF NOT EXISTS request_logs (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		created_at TEXT NOT NULL,
		backend_id TEXT NOT NULL,
		backend_model TEXT NOT NULL,
		route TEXT NOT NULL,
		observed_tps REAL NOT NULL DEFAULT 0,
		certified_tps REAL NOT NULL DEFAULT 0,
		baseline_tps REAL NOT NULL DEFAULT 0,
		speed_score INTEGER NOT NULL DEFAULT 0,
		stream INTEGER NOT NULL,
		status_code INTEGER NOT NULL,
		duration_ms INTEGER NOT NULL,
		input TEXT NOT NULL,
		output TEXT NOT NULL,
		error TEXT NOT NULL
	)`,
	`CREATE INDEX IF NOT EXISTS idx_request_logs_backend_created ON request_logs (backend_id, created_at DESC)`,
	`CREATE INDEX IF NOT EXISTS idx_request_logs_created ON request_logs (created_at DESC)`,
	`CREATE TABLE IF NOT EXISTS backend_registrations (
		id TEXT PRIMARY KEY,
		updated_at TEXT NOT NULL,
		registration_json TEXT NOT NULL
	)`,
	`CREATE TABLE IF NOT EXISTS worker_profiles (
		id TEXT PRIMARY KEY,
		model TEXT NOT NULL,
		updated_at TEXT NOT NULL,
		profile_json TEXT NOT NULL
	)`,
}

// TestMigrationsApplyToAnOldDatabase builds a database with the OLD schema and
// real rows in it, then opens it with the current code.
//
// The point is the second half. Asserting that the migration SQL parses proves
// nothing; what an operator upgrading in place needs is that worker_profiles
// survives — it is the cached cold-start profile for the whole fleet, and
// re-measuring it costs hours of GPU time and, since P2, real money — and that
// the log rows written before key_id existed are still readable afterwards.
func TestMigrationsApplyToAnOldDatabase(t *testing.T) {
	path := filepath.Join(t.TempDir(), "logs.sqlite")
	ctx := context.Background()

	// ── build the old database ──
	old, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	for _, stmt := range oldSchema {
		if _, err := old.ExecContext(ctx, stmt); err != nil {
			t.Fatalf("old schema %q: %v", stmt, err)
		}
	}
	if _, err := old.ExecContext(ctx, `INSERT INTO request_logs
		(created_at, backend_id, backend_model, route, observed_tps, certified_tps, baseline_tps, speed_score, stream, status_code, duration_ms, input, output, error)
		VALUES (?, 'llm-a750', 'gemma4', 'route:d=0.42,q>=2', 12.5, 11, 10, 40, 0, 200, 1234, 'prompt', 'answer', '')`,
		time.Now().UTC().Format(time.RFC3339Nano)); err != nil {
		t.Fatalf("seed request_logs: %v", err)
	}
	if _, err := old.ExecContext(ctx, `INSERT INTO backend_registrations (id, updated_at, registration_json)
		VALUES ('llm-a750', ?, '{"id":"llm-a750","url":"http://a750:8080","model":"gemma4","ttl_seconds":90}')`,
		time.Now().UTC().Format(time.RFC3339Nano)); err != nil {
		t.Fatalf("seed backend_registrations: %v", err)
	}
	profile := fmt.Sprintf(`{"model":"gemma4","quality":71,"context_k":128,"max_concurrency":4,"bench_version":%d}`, benchmarkVersion)
	if _, err := old.ExecContext(ctx, `INSERT INTO worker_profiles (id, model, updated_at, profile_json) VALUES ('llm-a750', 'gemma4', ?, ?)`,
		time.Now().UTC().Format(time.RFC3339Nano), profile); err != nil {
		t.Fatalf("seed worker_profiles: %v", err)
	}
	if err := old.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// ── open it with the current code ──
	logs, err := openLogStore(path, 16384, "test-secret")
	if err != nil {
		t.Fatalf("migrating an old database failed: %v", err)
	}
	defer logs.Close()

	// The expensive thing survived, values intact.
	prof, ok := logs.LoadWorkerProfile(ctx, "llm-a750", "gemma4")
	if !ok {
		t.Fatal("worker_profiles did not survive the migration — the whole fleet would re-benchmark")
	}
	if prof.Quality != 71 || prof.ContextK != 128 || prof.MaxConcurrency != 4 {
		t.Errorf("cached profile altered by the migration: %+v", prof)
	}

	// Rows written before key_id existed still read, with the empty default.
	rows, err := logs.List(ctx, "", 10, 0)
	if err != nil {
		t.Fatalf("listing pre-migration logs: %v", err)
	}
	if len(rows) != 1 || rows[0].Input != "prompt" || rows[0].Output != "answer" {
		t.Fatalf("pre-migration log row lost: %+v", rows)
	}
	if rows[0].KeyID != "" {
		t.Errorf("pre-migration row invented a key id: %q", rows[0].KeyID)
	}

	// And a registration written before provider/source existed comes back and
	// settles to a free local beacon.
	regs, err := logs.LoadBackendRegistrations(ctx)
	if err != nil || len(regs) != 1 {
		t.Fatalf("registrations = %d, err=%v", len(regs), err)
	}
	reg := regs[0]
	if err := normalizeRegistration(&reg); err != nil {
		t.Fatalf("normalizeRegistration: %v", err)
	}
	if reg.Provider != providerLocal || reg.Source != sourceBeacon {
		t.Errorf("upgraded registration = provider %q source %q", reg.Provider, reg.Source)
	}

	// The new tables are usable, and a new row round-trips with its key id.
	if _, err := logs.CreateAPIKey(ctx, "sk-migrated", apiKey{
		Prefix: "sk-migrat", Role: roleClient, Enabled: true, CreatedAt: time.Now().UTC(),
	}); err != nil {
		t.Fatalf("api_keys unusable after migration: %v", err)
	}
	if err := logs.SaveSetting(ctx, settingAdminPasswordHash, "x"); err != nil {
		t.Fatalf("router_settings unusable after migration: %v", err)
	}
	if err := logs.Insert(ctx, RequestLog{
		CreatedAt: time.Now().UTC(), BackendID: "b", BackendModel: "m", Route: "route",
		StatusCode: 200, Input: "i", Output: "o", KeyID: "7",
	}); err != nil {
		t.Fatalf("insert with key_id: %v", err)
	}
	rows, _ = logs.List(ctx, "b", 10, 0)
	if len(rows) != 1 || rows[0].KeyID != "7" {
		t.Fatalf("key_id did not round-trip: %+v", rows)
	}

	// Re-opening is a no-op: the ALTERs are all swallowed as duplicates.
	logs2, err := openLogStore(path, 16384, "test-secret")
	if err != nil {
		t.Fatalf("second open (migrations are not idempotent): %v", err)
	}
	logs2.Close()
}

// ── Keys ────────────────────────────────────────────────────────────────────

func TestAPIKeyRoundTrip(t *testing.T) {
	logs := newTestLogStore(t)
	ctx := t.Context()

	plain, key, err := newAPIKey("harness", roleClient, []string{"gemma4", " gemma4 ", "qwen3"}, 1000)
	if err != nil {
		t.Fatalf("newAPIKey: %v", err)
	}
	if !strings.HasPrefix(plain, keyPrefix) {
		t.Errorf("key %q is not sk- prefixed", plain)
	}
	if len(plain) < 40 {
		t.Errorf("key is too short to carry %d bytes of entropy: %q", keyRandomBytes, plain)
	}
	if !strings.HasPrefix(plain, key.Prefix) || len(key.Prefix) >= len(plain) {
		t.Errorf("displayed prefix %q does not identify (and must not reveal) %q", key.Prefix, plain)
	}
	if len(key.Models) != 2 {
		t.Errorf("allow-list not de-duplicated: %v", key.Models)
	}

	stored, err := logs.CreateAPIKey(ctx, plain, key)
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}

	// Shown once: nothing the store can return carries the plaintext.
	listed, err := logs.ListAPIKeys(ctx)
	if err != nil || len(listed) != 1 {
		t.Fatalf("list = %d, err=%v", len(listed), err)
	}
	if strings.Contains(jsonOf(t, listed), plain) {
		t.Fatal("the plaintext key came back out of the keys table")
	}

	got, ok := logs.LookupAPIKey(ctx, plain)
	if !ok {
		t.Fatal("a valid key did not look up")
	}
	if got.ID != stored.ID || got.Role != roleClient || got.TokenBudget != 1000 {
		t.Errorf("looked up the wrong row: %+v", got)
	}
	if _, ok := logs.LookupAPIKey(ctx, plain+"x"); ok {
		t.Error("a wrong key looked up")
	}
	if _, ok := logs.LookupAPIKey(ctx, ""); ok {
		t.Error("an empty key looked up")
	}

	// Disabling is a revoke, and it takes effect on the next lookup.
	stored.Enabled = false
	if err := logs.UpdateAPIKey(ctx, stored.ID, stored); err != nil {
		t.Fatalf("UpdateAPIKey: %v", err)
	}
	if _, ok := logs.LookupAPIKey(ctx, plain); ok {
		t.Fatal("a disabled key still authenticates")
	}
}

func TestAPIKeyUsageAccounting(t *testing.T) {
	logs := newTestLogStore(t)
	ctx := t.Context()
	plain, key, _ := newAPIKey("k", roleClient, nil, 0)
	stored, err := logs.CreateAPIKey(ctx, plain, key)
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}
	if !stored.LastUsedAt.IsZero() {
		t.Errorf("a fresh key has a last-used time: %v", stored.LastUsedAt)
	}
	for _, n := range []int{100, 250} {
		if err := logs.RecordKeyUse(ctx, stored.ID, n); err != nil {
			t.Fatalf("RecordKeyUse: %v", err)
		}
	}
	got, _ := logs.LookupAPIKey(ctx, plain)
	if got.TokensUsed != 350 {
		t.Errorf("tokens_used = %d, want 350", got.TokensUsed)
	}
	if got.LastUsedAt.IsZero() {
		t.Error("last_used_at was not stamped")
	}
}

func TestHasEnabledKey(t *testing.T) {
	logs := newTestLogStore(t)
	ctx := t.Context()
	if ok, err := logs.HasEnabledKey(ctx, roleClient); ok || err != nil {
		t.Fatalf("empty table = %v, %v", ok, err)
	}
	plain, key, _ := newAPIKey("k", roleWorker, nil, 0)
	stored, _ := logs.CreateAPIKey(ctx, plain, key)
	if ok, _ := logs.HasEnabledKey(ctx, roleClient); ok {
		t.Error("a worker key satisfied a client-role check")
	}
	if ok, _ := logs.HasEnabledKey(ctx, roleClient, roleWorker); !ok {
		t.Error("a worker key did not satisfy a multi-role check")
	}
	stored.Enabled = false
	if err := logs.UpdateAPIKey(ctx, stored.ID, stored); err != nil {
		t.Fatal(err)
	}
	if ok, _ := logs.HasEnabledKey(ctx, roleWorker); ok {
		t.Error("a disabled key still counts as existing")
	}
}

// ── Password hashing ────────────────────────────────────────────────────────

func TestPasswordHashing(t *testing.T) {
	const pw = "correct horse battery staple"
	hash, err := hashPassword(pw)
	if err != nil {
		t.Fatalf("hashPassword: %v", err)
	}
	if strings.Contains(hash, pw) {
		t.Fatal("the plaintext password is in the stored hash")
	}
	if !strings.HasPrefix(hash, passwordScheme+"$") {
		t.Errorf("stored hash does not name its scheme: %q", hash)
	}
	if !verifyPassword(hash, pw) {
		t.Fatal("the correct password did not verify")
	}
	if verifyPassword(hash, pw+"!") {
		t.Fatal("a wrong password verified")
	}
	// Salted: the same password twice must not produce the same hash.
	again, _ := hashPassword(pw)
	if again == hash {
		t.Fatal("hashing is not salted")
	}
	if !verifyPassword(again, pw) {
		t.Fatal("the second hash did not verify")
	}
	// A hash written at a different cost still verifies, which is what lets the
	// iteration count be raised later without locking anyone out.
	cheap, err := hashPasswordIter(pw, 1000)
	if err != nil {
		t.Fatalf("hashPasswordIter: %v", err)
	}
	if !verifyPassword(cheap, pw) {
		t.Fatal("a hash at a lower iteration count did not verify")
	}
	// Anything unparseable is a refusal, not an acceptance: a corrupt row must
	// lock admin access out rather than open it.
	for _, bad := range []string{"", "$", "bcrypt$1$a$b", passwordScheme + "$notanumber$a$b", passwordScheme + "$0$a$b", passwordScheme + "$1000$!!!$b"} {
		if verifyPassword(bad, pw) {
			t.Errorf("a malformed stored hash verified: %q", bad)
		}
	}
	if err := validPassword("short"); err == nil {
		t.Error("a short password was accepted")
	}
	if err := validPassword(strings.Repeat("a", minPasswordLength)); err != nil {
		t.Errorf("a long enough password was refused: %v", err)
	}
}

// ── Per-key limits ──────────────────────────────────────────────────────────

func TestIdentityLimits(t *testing.T) {
	limited := &identity{Role: roleClient, Models: []string{"gemma4", "qwen3"}, TokenBudget: 100, TokensUsed: 40}
	cases := []struct {
		name  string
		model string
		want  bool
	}{
		{"allowed", "gemma4", true},
		{"allowed, other case", "Gemma4", true},
		{"denied", "gpt-4o", false},
		// No model named is the auto route: the caller chose nothing, so there is
		// nothing to refuse. Routing still only reaches the fleet's own models.
		{"auto route", "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := limited.allowsModel(tc.model); got != tc.want {
				t.Errorf("allowsModel(%q) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
	// No allow-list means every model; a nil identity is the unauthenticated
	// open-fleet case and is likewise unrestricted.
	if !(&identity{}).allowsModel("anything") || !(*identity)(nil).allowsModel("anything") {
		t.Error("an empty allow-list must not restrict anything")
	}
	if limited.overBudget() {
		t.Error("40 of 100 is not over budget")
	}
	limited.TokensUsed = 100
	if !limited.overBudget() {
		t.Error("100 of 100 is over budget")
	}
	if (&identity{TokensUsed: 1 << 40}).overBudget() {
		t.Error("a key with no budget can never be over it")
	}
}

func TestUsageTotalTokens(t *testing.T) {
	cases := []struct {
		name string
		body string
		want int
	}{
		{"buffered total", `{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`, 15},
		{"no total, summed", `{"usage":{"prompt_tokens":10,"completion_tokens":5}}`, 15},
		{"sse last wins", "data: {\"usage\":{\"total_tokens\":3}}\n\ndata: {\"usage\":{\"total_tokens\":42}}\n\ndata: [DONE]\n\n", 42},
		{"spaced", `{"usage":{"total_tokens": 7}}`, 7},
		{"absent", `{"choices":[{"message":{"content":"hi"}}]}`, 0},
		{"truncated capture", "{\"choices\":\n…[capture truncated: 900 bytes omitted]…\n\"usage\":{\"total_tokens\":88}}", 88},
		{"not a number", `{"usage":{"total_tokens":null}}`, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := usageTotalTokens([]byte(tc.body)); got != tc.want {
				t.Errorf("usageTotalTokens = %d, want %d", got, tc.want)
			}
		})
	}
}

// ── Scope ───────────────────────────────────────────────────────────────────

// scopedRouter is a fleet with one ready worker plus an admin key and a client
// key, which is enough to drive every scope decision P3 makes.
func scopedRouter(t *testing.T) *Router {
	t.Helper()
	reg := newTestRegistry()
	registerQ(t, reg, "w", 50, 1)
	r := &Router{
		cfg:      &Config{DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second},
		registry: reg,
		logs:     newTestLogStore(t),
		client:   &http.Client{Timeout: time.Second},
	}
	issueKey(t, r, adminSecret, apiKey{Role: roleAdmin, Name: "admin"})
	issueKey(t, r, clientSecret, apiKey{Role: roleClient, Name: "client"})
	issueKey(t, r, workerSecret, apiKey{Role: roleWorker, Name: "worker"})
	return r
}

const (
	clientSecret = "sk-test-client-key"
	workerSecret = "sk-test-worker-key"
)

// TestEndpointScopes is the table of what moved and what did not.
//
// /logs and /backends carry every stored prompt and the shape of the operator's
// infrastructure, so a client token — which since P3 may belong to a stranger —
// must not reach them. /v1/models has to stay client-scoped, because a client
// cannot use an OpenAI API without it.
func TestEndpointScopes(t *testing.T) {
	r := scopedRouter(t)
	cases := []struct {
		method, path string
		client       int // status for a valid CLIENT key
		admin        int // status for a valid ADMIN key
	}{
		{http.MethodGet, "/backends", http.StatusUnauthorized, http.StatusOK},
		{http.MethodGet, "/backends/w", http.StatusUnauthorized, http.StatusOK},
		{http.MethodGet, "/backends/w/benchmark", http.StatusUnauthorized, http.StatusNotFound},
		{http.MethodGet, "/logs", http.StatusUnauthorized, http.StatusOK},
		{http.MethodPost, "/debug/backends/w/certify", http.StatusUnauthorized, http.StatusAccepted},
		{http.MethodGet, "/admin/providers", http.StatusUnauthorized, http.StatusOK},
		{http.MethodGet, "/admin/keys", http.StatusUnauthorized, http.StatusOK},
		// Client scope, and it stays that way.
		{http.MethodGet, "/v1/models", http.StatusOK, http.StatusOK},
		{http.MethodGet, "/v1/models/default", http.StatusOK, http.StatusOK},
	}
	for _, tc := range cases {
		t.Run(tc.method+" "+tc.path, func(t *testing.T) {
			for _, who := range []struct {
				name, secret string
				want         int
			}{{"client", clientSecret, tc.client}, {"admin", adminSecret, tc.admin}} {
				req := httptest.NewRequest(tc.method, tc.path, nil)
				req.Header.Set("Authorization", "Bearer "+who.secret)
				rec := httptest.NewRecorder()
				r.routes().ServeHTTP(rec, req)
				if rec.Code != who.want {
					t.Errorf("%s: %s %s = %d, want %d: %s", who.name, tc.method, tc.path, rec.Code, who.want, rec.Body.String())
				}
			}
			// And nobody at all is always refused on a scoped path.
			if tc.client == http.StatusUnauthorized {
				rec := httptest.NewRecorder()
				r.routes().ServeHTTP(rec, httptest.NewRequest(tc.method, tc.path, nil))
				if rec.Code != http.StatusUnauthorized {
					t.Errorf("unauthenticated %s %s = %d, want 401", tc.method, tc.path, rec.Code)
				}
			}
		})
	}
}

// TestWorkerTokenIsNotAClientToken: a worker credential registers backends and
// has never been able to spend them. It must not become a client key by being
// the only thing configured.
func TestWorkerTokenIsNotAClientToken(t *testing.T) {
	r := scopedRouter(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer "+workerSecret)
	r.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("a worker key reached /v1/models: %d", rec.Code)
	}
	// And the reverse: a client key cannot register a backend.
	rec = httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/backends/register", `{"id":"x","url":"http://x","model":"m"}`, clientSecret))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("a client key registered a backend: %d", rec.Code)
	}
}

// TestGeneratedCredentialsAreEnforced: an empty ROUTER_CLIENT_TOKENS used to
// mean no client authentication at all. Once a bootstrap key exists it has to
// actually gate the surface, or the generated credential is decoration.
func TestGeneratedCredentialsAreEnforced(t *testing.T) {
	reg := newTestRegistry()
	registerQ(t, reg, "w", 50, 1)
	r := &Router{cfg: &Config{DefaultMaxTokens: 4096}, registry: reg, logs: newTestLogStore(t)}

	// Before bootstrap, with nothing configured, the fleet is open — the
	// historical trusted-LAN behaviour.
	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("with nothing configured /v1/models = %d, want 200", rec.Code)
	}

	if err := r.bootstrapCredentials(t.Context()); err != nil {
		t.Fatalf("bootstrapCredentials: %v", err)
	}
	rec = httptest.NewRecorder()
	r.routes().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("after bootstrap /v1/models = %d, want 401 — the generated key is not enforced", rec.Code)
	}
	rec = httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/backends/register", `{"id":"x","url":"http://x","model":"m"}`, ""))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("after bootstrap registration = %d, want 401", rec.Code)
	}
}

// TestBootstrapGeneratesOnceAndOnlyWhenMissing.
func TestBootstrapGeneratesOnceAndOnlyWhenMissing(t *testing.T) {
	logs := newTestLogStore(t)
	ctx := context.Background()

	t.Run("generates what is missing", func(t *testing.T) {
		r := &Router{cfg: &Config{}, registry: newTestRegistry(), logs: logs}
		if err := r.bootstrapCredentials(ctx); err != nil {
			t.Fatalf("bootstrap: %v", err)
		}
		keys, _ := logs.ListAPIKeys(ctx)
		roles := map[string]int{}
		for _, k := range keys {
			roles[k.Role]++
		}
		if roles[roleClient] != 1 || roles[roleWorker] != 1 {
			t.Fatalf("generated %v, want one client and one worker key", roles)
		}
		if hash, _ := logs.LoadSetting(ctx, settingAdminPasswordHash); hash == "" {
			t.Fatal("no admin password was generated")
		}
	})

	t.Run("second start generates nothing", func(t *testing.T) {
		before, _ := logs.LoadSetting(ctx, settingAdminPasswordHash)
		r := &Router{cfg: &Config{}, registry: newTestRegistry(), logs: logs}
		if err := r.bootstrapCredentials(ctx); err != nil {
			t.Fatalf("bootstrap: %v", err)
		}
		keys, _ := logs.ListAPIKeys(ctx)
		if len(keys) != 2 {
			t.Fatalf("a second start minted more keys: %d", len(keys))
		}
		after, _ := logs.LoadSetting(ctx, settingAdminPasswordHash)
		if after != before {
			t.Fatal("a second start re-hashed the admin password")
		}
	})

	t.Run("environment tokens suppress generation", func(t *testing.T) {
		fresh := newTestLogStore(t)
		r := &Router{
			cfg: &Config{
				ClientTokens:  []string{"env-client"},
				WorkerToken:   "env-worker",
				AdminPassword: "a-seeded-admin-password",
			},
			registry: newTestRegistry(), logs: fresh,
		}
		if err := r.bootstrapCredentials(ctx); err != nil {
			t.Fatalf("bootstrap: %v", err)
		}
		keys, _ := fresh.ListAPIKeys(ctx)
		if len(keys) != 0 {
			t.Fatalf("generated %d keys despite the environment supplying them", len(keys))
		}
		// The admin password IS seeded from the environment, because there is
		// nowhere else for it to live — but the database is canonical from then on.
		hash, _ := fresh.LoadSetting(ctx, settingAdminPasswordHash)
		if !verifyPassword(hash, "a-seeded-admin-password") {
			t.Fatal("ROUTER_ADMIN_PASSWORD did not seed the database")
		}

		// A later start with a DIFFERENT environment password must not overwrite it.
		r.cfg.AdminPassword = "a-completely-different-one"
		if err := r.bootstrapCredentials(ctx); err != nil {
			t.Fatalf("second bootstrap: %v", err)
		}
		hash, _ = fresh.LoadSetting(ctx, settingAdminPasswordHash)
		if !verifyPassword(hash, "a-seeded-admin-password") {
			t.Fatal("the environment overwrote the stored admin password — the database is meant to be canonical")
		}
	})
}

// jsonOf renders a value the way the API would, for "this must not appear
// anywhere in the response" assertions.
func jsonOf(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}
