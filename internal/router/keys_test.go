package router

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
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
	// P4 added cost accounting to the profile JSON. A profile measured before it
	// existed has to come back with zeroes — "not measured", which is why the
	// token counts and not the cost are what say whether the run was ever priced
	// (see WorkerProfile.ProfilePromptTokens).
	if prof.ProfilePromptTokens != 0 || prof.ProfileOutputTokens != 0 || prof.ProfileCost != 0 {
		t.Errorf("pre-P4 profile invented a cost: %d/%d tokens, cost %g",
			prof.ProfilePromptTokens, prof.ProfileOutputTokens, prof.ProfileCost)
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
	// P5's groups table is created by the same migration and has to round-trip on
	// a database that predates it.
	if err := logs.SaveGroup(ctx, Group{Name: "coding", Members: []string{"gemma4", "qwen3"}, UpdatedAt: time.Now().UTC()}); err != nil {
		t.Fatalf("router_groups unusable after migration: %v", err)
	}
	groups, err := logs.LoadGroups(ctx)
	if err != nil || len(groups) != 1 || groups[0].Name != "coding" || len(groups[0].Members) != 2 {
		t.Fatalf("group did not round-trip: %+v, err=%v", groups, err)
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
	again, _ := hashPasswordIter(pw, 1000)
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

// TestReopenedGates is the rule behind the 409, stated on its own: a key edit
// may never flip a gate from "credential required" to "credential not required",
// and it may never refuse an edit for any other reason.
func TestReopenedGates(t *testing.T) {
	const (
		clientGate = "the OpenAI API (/v1/chat/completions and friends)"
		workerGate = "worker registration (/backends/register)"
	)
	cases := []struct {
		name string
		cfg  *Config
		keys []apiKey
		id   int64
		want []string
	}{
		{
			name: "the bootstrap pair: each key is the last of its kind",
			keys: []apiKey{{ID: 1, Role: roleClient, Enabled: true}, {ID: 2, Role: roleWorker, Enabled: true}},
			id:   2, want: []string{workerGate},
		},
		{
			name: "an admin key holds BOTH gates open on its own",
			keys: []apiKey{{ID: 1, Role: roleAdmin, Enabled: true}},
			id:   1, want: []string{clientGate, workerGate},
		},
		{
			// An admin key satisfies the client gate, so the client key is spare.
			name: "an admin key covers a client key",
			keys: []apiKey{{ID: 1, Role: roleAdmin, Enabled: true}, {ID: 2, Role: roleClient, Enabled: true}},
			id:   2, want: nil,
		},
		{
			name: "a second key of the same role is a replacement",
			keys: []apiKey{{ID: 1, Role: roleWorker, Enabled: true}, {ID: 2, Role: roleWorker, Enabled: true}},
			id:   1, want: nil,
		},
		{
			name: "a disabled key is not a credential and does not hold a gate",
			keys: []apiKey{{ID: 1, Role: roleWorker, Enabled: true}, {ID: 2, Role: roleWorker}},
			id:   1, want: []string{workerGate},
		},
		{
			// Nothing is being reopened, so nothing is refused: an open deployment
			// must still be able to tidy up its keys table.
			name: "a gate that is already open is not defended",
			keys: []apiKey{{ID: 1, Role: roleClient}},
			id:   1, want: nil,
		},
		{
			name: "the environment requires a credential whatever the table says",
			cfg:  &Config{ClientTokens: []string{"env"}, WorkerToken: "env"},
			keys: []apiKey{{ID: 1, Role: roleAdmin, Enabled: true}},
			id:   1, want: nil,
		},
		{
			name: "deleting a key that is not there reopens nothing",
			keys: []apiKey{{ID: 1, Role: roleAdmin, Enabled: true}},
			id:   99, want: nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tc.cfg
			if cfg == nil {
				cfg = &Config{}
			}
			got := (&Router{cfg: cfg}).reopenedGates(tc.keys, tc.id)
			names := make([]string, 0, len(got))
			for _, g := range got {
				names = append(names, g.name)
			}
			if strings.Join(names, "|") != strings.Join(tc.want, "|") {
				t.Errorf("reopenedGates = %v, want %v", names, tc.want)
			}
		})
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
		// nowhere else for it to live.
		hash, _ := fresh.LoadSetting(ctx, settingAdminPasswordHash)
		if !verifyPassword(hash, "a-seeded-admin-password") {
			t.Fatal("ROUTER_ADMIN_PASSWORD did not seed the database")
		}

		// A start with the SAME password rewrites nothing: bootstrap compares before
		// it hashes, so the stored row is left byte-identical.
		if err := r.bootstrapCredentials(ctx); err != nil {
			t.Fatalf("second bootstrap: %v", err)
		}
		again, _ := fresh.LoadSetting(ctx, settingAdminPasswordHash)
		if again != hash {
			t.Error("an unchanged ROUTER_ADMIN_PASSWORD re-hashed the stored password on restart")
		}
	})
}

// TestAdminPasswordRecovery is the lockout fix.
//
// ROUTER_ADMIN_PASSWORD used to seed a database that had none and do nothing
// afterwards. Every operator surface is behind the admin gate — /backends,
// /logs, the keys tab, /debug/backends/* — and the generated password is printed
// exactly once, to a container log. An operator who missed that line had no way
// back into their own router short of editing SQLite inside the data volume.
//
// So: a SET variable is authoritative on every start, an UNSET one changes
// nothing, and the login endpoint has to agree with both.
func TestAdminPasswordRecovery(t *testing.T) {
	const (
		generatedEra = "the-password-nobody-wrote-down"
		uiChosen     = "the-one-set-in-the-ui"
		recovery     = "the-recovery-password"
	)
	logs := newTestLogStore(t)
	ctx := t.Context()
	newRouter := func(envPassword string) *Router {
		return &Router{
			cfg:      &Config{AdminPassword: envPassword},
			registry: newTestRegistry(),
			logs:     logs,
		}
	}
	login := func(t *testing.T, r *Router, password string) int {
		t.Helper()
		rec := httptest.NewRecorder()
		r.routes().ServeHTTP(rec, post("/admin/login", `{"password":`+jsonOf(t, password)+`}`, ""))
		return rec.Code
	}

	// A database with a password already in it, as an upgrading deployment has.
	hash, err := hashPasswordIter(generatedEra, 1000)
	if err != nil {
		t.Fatalf("hashPasswordIter: %v", err)
	}
	if err := logs.SaveSetting(ctx, settingAdminPasswordHash, hash); err != nil {
		t.Fatalf("SaveSetting: %v", err)
	}

	cases := []struct {
		name        string
		envPassword string
		wantIn      string // the password that must now work
		wantOut     string // one that must not
	}{
		// The recovery path itself: set the variable, restart, you are back in.
		{"a set variable takes over an existing password", recovery, recovery, generatedEra},
		// And it is idempotent — a second restart with the same variable is a no-op
		// that still leaves the operator able to log in.
		{"the same variable again is a no-op", recovery, recovery, generatedEra},
		// An UNSET variable must never wipe a password. This is the property the old
		// early return was protecting, and it is the one that has to survive.
		{"an unset variable leaves the stored password alone", "", recovery, generatedEra},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := newRouter(tc.envPassword)
			if err := r.bootstrapCredentials(ctx); err != nil {
				t.Fatalf("bootstrap: %v", err)
			}
			if code := login(t, r, tc.wantIn); code != http.StatusOK {
				t.Errorf("login with %q = %d, want 200", tc.wantIn, code)
			}
			if code := login(t, r, tc.wantOut); code != http.StatusUnauthorized {
				t.Errorf("login with the superseded %q = %d, want 401", tc.wantOut, code)
			}
		})
	}

	// A password rotated through the UI while the variable is unset is canonical,
	// and the next restart does not revert it.
	r := newRouter("")
	rec := httptest.NewRecorder()
	req := post("/admin/password", `{"current_password":`+jsonOf(t, recovery)+`,"new_password":`+jsonOf(t, uiChosen)+`}`, "")
	issueKey(t, r, adminSecret, apiKey{Role: roleAdmin, Name: "admin"})
	req.Header.Set("Authorization", "Bearer "+adminSecret)
	r.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("password change = %d: %s", rec.Code, rec.Body.String())
	}
	if err := newRouter("").bootstrapCredentials(ctx); err != nil {
		t.Fatalf("bootstrap after a UI rotation: %v", err)
	}
	if code := login(t, r, uiChosen); code != http.StatusOK {
		t.Errorf("a restart reverted a password set in the UI: login = %d, want 200", code)
	}
}

// TestBootstrapBannerIsGreppable: the setup instructions tell an operator to
// find a generated credential with `docker compose logs discrimen | grep -i
// bootstrap`. The banner is the only place it is ever printed, so a banner that
// does not carry that word makes the one documented recovery find nothing.
func TestBootstrapBannerIsGreppable(t *testing.T) {
	var buf bytes.Buffer
	prevOut, prevFlags := log.Writer(), log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	t.Cleanup(func() { log.SetOutput(prevOut); log.SetFlags(prevFlags) })

	r := &Router{cfg: &Config{}, registry: newTestRegistry(), logs: newTestLogStore(t)}
	if err := r.bootstrapCredentials(t.Context()); err != nil {
		t.Fatalf("bootstrap: %v", err)
	}
	out := buf.String()
	// Case-insensitively, exactly as the documented grep would match it.
	for _, want := range []string{"bootstrap client token", "bootstrap worker token", "bootstrap admin password"} {
		if !strings.Contains(strings.ToLower(out), want) {
			t.Errorf("the startup banner has no %q line for `grep -i bootstrap` to find:\n%s", want, out)
		}
	}
}

// TestEnvPasswordIsNotLogged: the reset line says that it happened, never what
// it was. The operator supplied this password, so they already have it, and
// printing it would put a live credential in the log on every start.
func TestEnvPasswordIsNotLogged(t *testing.T) {
	const secret = "an-environment-password"
	logs := newTestLogStore(t)
	ctx := t.Context()
	if hash, err := hashPasswordIter("something-else", 1000); err != nil {
		t.Fatal(err)
	} else if err := logs.SaveSetting(ctx, settingAdminPasswordHash, hash); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	prevOut, prevFlags := log.Writer(), log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	t.Cleanup(func() { log.SetOutput(prevOut); log.SetFlags(prevFlags) })

	r := &Router{cfg: &Config{AdminPassword: secret}, registry: newTestRegistry(), logs: logs}
	if err := r.bootstrapCredentials(ctx); err != nil {
		t.Fatalf("bootstrap: %v", err)
	}
	out := buf.String()
	if strings.Contains(out, secret) {
		t.Errorf("ROUTER_ADMIN_PASSWORD was printed to the log:\n%s", out)
	}
	if !strings.Contains(strings.ToUpper(out), "RESET") {
		t.Errorf("replacing a stored password said nothing about it:\n%s", out)
	}
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

// ── Per-key limits on a real request ────────────────────────────────────────

// keyLimitRouter is a fleet with one fake worker that answers a completion with
// a usage block, plus a client key carrying the given allow-list and budget.
func keyLimitRouter(t *testing.T, models []string, budget int64) (*Router, *Router) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":30,"completion_tokens":70,"total_tokens":100}}`))
	}))
	t.Cleanup(srv.Close)

	reg := newTestRegistry()
	reg.upsert(BackendRegistration{
		ID: "w", URL: srv.URL, Model: "gemma4", Quality: 50,
		TTLSeconds: 3600, Features: []string{"chat"},
	})
	reg.finishCertification("w", true, map[string]Check{}, 50, 10, "")

	r := &Router{
		cfg:      &Config{DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second},
		registry: reg, logs: newTestLogStore(t),
		client: &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
	}
	issueKey(t, r, clientSecret, apiKey{Role: roleClient, Name: "limited", Models: models, TokenBudget: budget})
	return r, r
}

// TestModelAllowListEnforced: a key with an allow-list may only name models on
// it. Naming nothing is the auto route and stays allowed — the router still only
// picks from its own fleet.
func TestModelAllowListEnforced(t *testing.T) {
	r, _ := keyLimitRouter(t, []string{"gemma4"}, 0)
	cases := []struct {
		name, model string
		status      int
	}{
		{"allowed model", `"gemma4"`, http.StatusOK},
		// Off the list, whether or not the fleet serves it. The check runs BEFORE
		// model resolution on purpose: a 403 for one name and a 404 for another
		// would let a restricted key enumerate the fleet a model at a time.
		{"denied model the fleet does not serve", `"qwen3"`, http.StatusForbidden},
		{"auto route", `"default"`, http.StatusOK},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			r.routes().ServeHTTP(rec, post("/v1/chat/completions",
				`{"model":`+tc.model+`,"messages":[{"role":"user","content":"hi"}]}`, clientSecret))
			if rec.Code != tc.status {
				t.Fatalf("status = %d, want %d: %s", rec.Code, tc.status, rec.Body.String())
			}
		})
	}
	// A model the FLEET serves but the key may not is a permission error, which
	// tells a client to ask for access rather than to change model.
	r2, _ := keyLimitRouter(t, []string{"qwen3"}, 0)
	rec := httptest.NewRecorder()
	r2.routes().ServeHTTP(rec, post("/v1/chat/completions",
		`{"model":"gemma4","messages":[{"role":"user","content":"hi"}]}`, clientSecret))
	if rec.Code != http.StatusForbidden {
		t.Fatalf("a fleet model the key may not use = %d, want 403: %s", rec.Code, rec.Body.String())
	}
	if body := errorEnvelopeOf(t, rec); body.Type != "permission_error" {
		t.Errorf("error type = %q, want permission_error", body.Type)
	}
	// And no worker was contacted for a refused request: the check runs before a
	// slot is taken.
	if rows, _ := r2.logs.List(t.Context(), "", 10, 0); len(rows) != 0 {
		t.Errorf("a refused request reached a worker: %+v", rows)
	}
}

// TestTokenBudgetEnforced: a request is charged what the endpoint reported, and
// a key past its budget gets a 429 in the OpenAI envelope.
func TestTokenBudgetEnforced(t *testing.T) {
	r, _ := keyLimitRouter(t, nil, 150)
	chat := func() *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		r.routes().ServeHTTP(rec, post("/v1/chat/completions",
			`{"messages":[{"role":"user","content":"hi"}]}`, clientSecret))
		return rec
	}
	if rec := chat(); rec.Code != http.StatusOK {
		t.Fatalf("first request = %d: %s", rec.Code, rec.Body.String())
	}
	// The charge lands in a goroutine after the response is committed, so wait
	// for it rather than racing it.
	used := waitForTokens(t, r, clientSecret, 100)
	if used != 100 {
		t.Fatalf("charged %d tokens, want the 100 the endpoint reported", used)
	}

	// Still under 150: the second request is served and takes it over.
	if rec := chat(); rec.Code != http.StatusOK {
		t.Fatalf("second request = %d: %s", rec.Code, rec.Body.String())
	}
	waitForTokens(t, r, clientSecret, 200)

	rec := chat()
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("over budget = %d, want 429: %s", rec.Code, rec.Body.String())
	}
	body := errorEnvelopeOf(t, rec)
	if body.Type != "rate_limit_error" {
		t.Errorf("error type = %q, want rate_limit_error", body.Type)
	}
	if !strings.Contains(body.Message, "budget") {
		t.Errorf("the refusal does not say why: %q", body.Message)
	}
}

// waitForTokens polls until a key's charged total reaches want, or the test
// times out. The charge is deliberately asynchronous (see recordKeyUse).
func waitForTokens(t *testing.T, r *Router, secret string, want int64) int64 {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		key, ok := r.logs.LookupAPIKey(context.Background(), secret)
		if ok && key.TokensUsed >= want {
			return key.TokensUsed
		}
		if time.Now().After(deadline) {
			t.Fatalf("tokens_used did not reach %d (got %d)", want, key.TokensUsed)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// ── The allow-list as an access control ─────────────────────────────────────

// allowListRouter builds a two-model fleet — one local worker and one standing
// in for a metered endpoint — plus a third worker that is registered but never
// certified, so a test has a live id that is not routable.
//
// The metered worker is the HIGHER quality of the two, so the unrestricted auto
// route ranks it first. That is what makes the assertions sharp: a restricted
// key landing on the local worker has been steered there, not merely lucky.
func allowListRouter(t *testing.T, models []string) (*Router, map[string]*atomic.Int64) {
	t.Helper()
	hits := map[string]*atomic.Int64{}
	reg := newTestRegistry()
	for _, w := range []struct {
		id, model string
		quality   int
		certify   bool
	}{
		{"local", "local-8b", 40, true},
		{"metered", "gpt-4o", 90, true},
		{"dormant", "some-model", 60, false},
	} {
		counter := &atomic.Int64{}
		hits[w.id] = counter
		srv := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, _ *http.Request) {
			counter.Add(1)
			rw.Header().Set("Content-Type", "application/json")
			_, _ = rw.Write([]byte(`{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],` +
				`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
		}))
		t.Cleanup(srv.Close)
		reg.upsert(BackendRegistration{
			ID: w.id, URL: srv.URL, Model: w.model, Quality: w.quality,
			TTLSeconds: 3600, Features: []string{"chat"},
		})
		if w.certify {
			reg.finishCertification(w.id, true, map[string]Check{}, 50, 10, "")
		}
	}
	r := &Router{
		cfg:      &Config{DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second},
		registry: reg, logs: newTestLogStore(t),
		client: &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
	}
	issueKey(t, r, clientSecret, apiKey{Role: roleClient, Name: "limited", Models: models})
	issueKey(t, r, adminSecret, apiKey{Role: roleAdmin, Name: "admin"})
	return r, hits
}

const chatBody = `"messages":[{"role":"user","content":"hi"}]`

// TestAllowListRestrictsTheAutoRoute: a key with an allow-list must not reach
// the whole fleet by declining to name a model.
//
// allowsModel refuses nothing to a caller who named nothing, and autoModelNames
// maps "", "default", "auto" and "router" onto the auto route — which ranks
// every registered worker. So a key issued for one local model could reach every
// metered endpoint by asking for "default", and the allow-list was a label
// rather than a control.
func TestAllowListRestrictsTheAutoRoute(t *testing.T) {
	cases := []struct {
		name, body string
	}{
		{"no model field at all", `{` + chatBody + `}`},
		{"model:default", `{"model":"default",` + chatBody + `}`},
		{"model:auto", `{"model":"auto",` + chatBody + `}`},
		{"model:router", `{"model":"router",` + chatBody + `}`},
		{"model: empty string", `{"model":"",` + chatBody + `}`},
		{"the allow-listed model by name", `{"model":"local-8b",` + chatBody + `}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, hits := allowListRouter(t, []string{"local-8b"})
			rec := httptest.NewRecorder()
			r.routes().ServeHTTP(rec, post("/v1/chat/completions", tc.body, clientSecret))
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
			}
			if got := rec.Header().Get("X-LLM-Backend-ID"); got != "local" {
				t.Errorf("served by %q, want the allow-listed worker", got)
			}
			if n := hits["metered"].Load(); n != 0 {
				t.Errorf("a key restricted to local-8b reached the metered endpoint %d time(s)", n)
			}
		})
	}

	// Control: the SAME request from an unrestricted key does reach the metered
	// worker. Without this the assertions above would also pass on a fleet that
	// could never route there in the first place.
	t.Run("an empty allow-list is unrestricted", func(t *testing.T) {
		r, hits := allowListRouter(t, nil)
		rec := httptest.NewRecorder()
		r.routes().ServeHTTP(rec, post("/v1/chat/completions", `{"model":"default",`+chatBody+`}`, clientSecret))
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
		}
		if hits["metered"].Load() == 0 {
			t.Error("the unrestricted auto route did not prefer the higher-quality worker; the test proves nothing")
		}
	})

	// Nothing on the list is routable: a 503, because the key IS allowed these
	// models and the reason none is a candidate may be that they are all busy.
	t.Run("nothing on the list is available", func(t *testing.T) {
		r, _ := allowListRouter(t, []string{"a-model-nobody-serves"})
		rec := httptest.NewRecorder()
		r.routes().ServeHTTP(rec, post("/v1/chat/completions", `{"model":"default",`+chatBody+`}`, clientSecret))
		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("status = %d, want 503: %s", rec.Code, rec.Body.String())
		}
		if body := errorEnvelopeOf(t, rec); !strings.Contains(body.Message, "a-model-nobody-serves") {
			t.Errorf("the refusal does not name the key's own allow-list: %q", body.Message)
		}
	})

	// And the same on /v1/completions, which reaches the fleet through
	// selectBackends rather than planRoute.
	t.Run("the legacy completions endpoint too", func(t *testing.T) {
		r, hits := allowListRouter(t, []string{"local-8b"})
		rec := httptest.NewRecorder()
		r.routes().ServeHTTP(rec, post("/v1/completions", `{"prompt":"hi"}`, clientSecret))
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
		}
		if n := hits["metered"].Load(); n != 0 {
			t.Errorf("/v1/completions reached the metered endpoint %d time(s)", n)
		}
	})
}

// TestGroupFallbackRespectsTheAllowList: a group whose members cannot serve the
// request falls back to automatic routing, and that fallback used to clear the
// model filter entirely — handing a restricted key the whole fleet through a
// group name it was allowed to say.
func TestGroupFallbackRespectsTheAllowList(t *testing.T) {
	r, hits := allowListRouter(t, []string{"local-8b", "coding"})
	r.groups.put(Group{Name: "coding", Members: []string{"a-worker-that-is-not-here"}})

	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/v1/chat/completions", `{"model":"coding",`+chatBody+`}`, clientSecret))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("X-LLM-Group"); got != "fallback" {
		t.Fatalf("X-LLM-Group = %q, want fallback — the test is not exercising the fallback", got)
	}
	if n := hits["metered"].Load(); n != 0 {
		t.Errorf("a group fallback reached the metered endpoint %d time(s)", n)
	}
	if got := rec.Header().Get("X-LLM-Backend-ID"); got != "local" {
		t.Errorf("the fallback served %q, want the allow-listed worker", got)
	}
}

// TestPinIsCheckedAndIsNotAnOracle covers both halves of the X-LLM-Backend-ID
// hole.
//
// The pin routes by worker id with nothing re-checking the allow-list, so a
// restricted key could name any worker in the fleet and be served by it. And the
// three refusals were distinguishable — 404 for an unknown id, 503 naming
// healthy/ready/expired for a known one — which made the header a
// fleet-enumeration oracle: walk a list of guesses and read back every
// registered id and whether it was alive, the exact thing moving /backends
// behind the admin gate was meant to prevent.
func TestPinIsCheckedAndIsNotAnOracle(t *testing.T) {
	cases := []struct {
		name, pin string
	}{
		{"a worker off the allow-list", "metered"},
		{"a registered but uncertified worker", "dormant"},
		{"an id that does not exist", "no-such-worker"},
	}
	var bodies []string
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, hits := allowListRouter(t, []string{"local-8b"})
			req := post("/v1/chat/completions", `{"model":"default",`+chatBody+`}`, clientSecret)
			req.Header.Set("X-LLM-Backend-ID", tc.pin)
			rec := httptest.NewRecorder()
			r.routes().ServeHTTP(rec, req)
			if rec.Code != http.StatusNotFound {
				t.Fatalf("pin %q = %d, want 404: %s", tc.pin, rec.Code, rec.Body.String())
			}
			for id, n := range hits {
				if n.Load() != 0 {
					t.Errorf("a refused pin still reached worker %q", id)
				}
			}
			// Compare with the id the CALLER supplied taken back out: echoing it is
			// not a disclosure, and anything else that differs is.
			bodies = append(bodies, strings.ReplaceAll(rec.Body.String(), tc.pin, "<pin>"))
		})
	}
	for i := 1; i < len(bodies); i++ {
		if bodies[i] != bodies[0] {
			t.Errorf("a client can tell two pin refusals apart, so the header enumerates the fleet:\n  %s\n  %s", bodies[0], bodies[i])
		}
	}

	// The pin still works for a worker the key may use.
	t.Run("an allowed worker is still pinnable", func(t *testing.T) {
		r, hits := allowListRouter(t, []string{"local-8b"})
		req := post("/v1/chat/completions", `{"model":"default",`+chatBody+`}`, clientSecret)
		req.Header.Set("X-LLM-Backend-ID", "local")
		rec := httptest.NewRecorder()
		r.routes().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("pinning an allowed worker = %d, want 200: %s", rec.Code, rec.Body.String())
		}
		if hits["local"].Load() != 1 {
			t.Error("the pinned worker was not the one that served")
		}
	})

	// An ADMIN is told which of the three it was: they can already read /backends,
	// so there is nothing left to withhold, and a debugging operator needs it.
	t.Run("an admin is told why", func(t *testing.T) {
		r, _ := allowListRouter(t, nil)
		seen := map[string]bool{}
		for _, pin := range []string{"dormant", "no-such-worker"} {
			req := post("/v1/chat/completions", `{"model":"default",`+chatBody+`}`, adminSecret)
			req.Header.Set("X-LLM-Backend-ID", pin)
			rec := httptest.NewRecorder()
			r.routes().ServeHTTP(rec, req)
			if rec.Code != http.StatusNotFound {
				t.Fatalf("admin pin %q = %d, want 404: %s", pin, rec.Code, rec.Body.String())
			}
			seen[errorEnvelopeOf(t, rec).Message] = true
		}
		if len(seen) != 2 {
			t.Errorf("an admin gets the same message for every refusal: %v", seen)
		}
	})
}

// ── Charging what was actually spent ────────────────────────────────────────

// TestStreamedCompletionsAreCharged: /v1/completions with stream:true and no
// stream_options.include_usage is the DEFAULT shape, it carries no usage block,
// and it used to be charged nothing at all — so a budgeted key could stream this
// endpoint in a loop while tokens_used stayed where it was.
func TestStreamedCompletionsAreCharged(t *testing.T) {
	const frames = 40
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		for i := 0; i < frames; i++ {
			_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"text\":\" tok\",\"finish_reason\":null}]}\n\n")
			if fl != nil {
				fl.Flush()
			}
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	reg := newTestRegistry()
	reg.upsert(BackendRegistration{ID: "w", URL: srv.URL, Model: "local-8b", Quality: 50, TTLSeconds: 3600, Features: []string{"chat"}})
	reg.finishCertification("w", true, map[string]Check{}, 50, 10, "")
	r := &Router{
		cfg:      &Config{DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second},
		registry: reg, logs: newTestLogStore(t),
		client: &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
	}
	issueKey(t, r, clientSecret, apiKey{Role: roleClient, Name: "budgeted", TokenBudget: frames})

	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/v1/completions", `{"prompt":"hi","stream":true}`, clientSecret))
	if rec.Code != http.StatusOK {
		t.Fatalf("stream = %d: %s", rec.Code, rec.Body.String())
	}
	// One SSE frame is one generated token in both dialects, so the estimate is
	// at least the frame count. Asynchronous by contract — see recordKeyUse.
	used := waitForTokens(t, r, clientSecret, frames)
	if used < frames {
		t.Fatalf("charged %d tokens for a %d-frame stream", used, frames)
	}
	// And the budget it just spent now stops the next one, which is the whole
	// point of charging it.
	rec = httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/v1/completions", `{"prompt":"hi","stream":true}`, clientSecret))
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("second stream = %d, want 429 — the budget never bit: %s", rec.Code, rec.Body.String())
	}
}

// TestChargingFallbacksByRoute: what each route charges when the endpoint
// reports no usage of its own.
func TestChargingFallbacksByRoute(t *testing.T) {
	cases := []struct {
		name, path, body string
		reply            string
		contentType      string
		wantAtLeast      int64
		wantAtMost       int64
	}{
		{
			// A buffered chat reply with no usage block used to charge the prompt
			// alone, so an answer of any length was free.
			name: "buffered chat with no usage block", path: "/v1/chat/completions",
			body:        `{"model":"local-8b","messages":[{"role":"user","content":"hi"}]}`,
			reply:       `{"choices":[{"message":{"content":"` + strings.Repeat("word ", 200) + `"},"finish_reason":"stop"}]}`,
			contentType: "application/json",
			wantAtLeast: 150, wantAtMost: 400,
		},
		{
			// An embeddings reply generates nothing — its whole cost is the input —
			// and its body is a float array that must not be priced as prose.
			name: "embeddings charges the input only", path: "/v1/embeddings",
			body:        `{"input":"hi","model":"local-8b"}`,
			reply:       `{"data":[{"embedding":[` + strings.TrimSuffix(strings.Repeat("0.0123456789,", 1000), ",") + `]}]}`,
			contentType: "application/json",
			wantAtLeast: 1, wantAtMost: 50,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", tc.contentType)
				_, _ = io.WriteString(w, tc.reply)
			}))
			t.Cleanup(srv.Close)
			reg := newTestRegistry()
			reg.upsert(BackendRegistration{
				ID: "w", URL: srv.URL, Model: "local-8b", Quality: 50,
				TTLSeconds: 3600, Features: []string{"chat", "embeddings"},
			})
			reg.finishCertification("w", true, map[string]Check{}, 50, 10, "")
			r := &Router{
				cfg:      &Config{DefaultMaxTokens: 4096, HealthInterval: 15 * time.Second, LogMaxBodyBytes: 16384},
				registry: reg, logs: newTestLogStore(t),
				client: &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
			}
			issueKey(t, r, clientSecret, apiKey{Role: roleClient, Name: "budgeted"})

			rec := httptest.NewRecorder()
			r.routes().ServeHTTP(rec, post(tc.path, tc.body, clientSecret))
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
			}
			used := waitForTokens(t, r, clientSecret, tc.wantAtLeast)
			if used > tc.wantAtMost {
				t.Errorf("charged %d tokens, want no more than %d", used, tc.wantAtMost)
			}
		})
	}
}

// TestRequestLogStampsKeyID: every log row records who made the request, so a
// stored prompt can be attributed to a caller after the fact.
func TestRequestLogStampsKeyID(t *testing.T) {
	r, _ := keyLimitRouter(t, nil, 0)
	key, _ := r.logs.LookupAPIKey(t.Context(), clientSecret)

	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/v1/chat/completions",
		`{"messages":[{"role":"user","content":"hi"}]}`, clientSecret))
	if rec.Code != http.StatusOK {
		t.Fatalf("chat = %d: %s", rec.Code, rec.Body.String())
	}

	want := strconv.FormatInt(key.ID, 10)
	deadline := time.Now().Add(2 * time.Second)
	for {
		rows, err := r.logs.List(context.Background(), "", 10, 0)
		if err != nil {
			t.Fatalf("list logs: %v", err)
		}
		if len(rows) > 0 {
			if rows[0].KeyID != want {
				t.Fatalf("log row key_id = %q, want %q", rows[0].KeyID, want)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("no log row was written")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// TestEnvTokenStampsEnv: a bootstrap environment token has no row, so it stamps
// the one label that is true of it rather than a made-up id.
func TestEnvTokenStampsEnv(t *testing.T) {
	cases := []struct {
		name string
		id   *identity
		want string
	}{
		{"api key", &identity{KeyID: 42}, "42"},
		{"environment token", &identity{Role: roleClient}, "env"},
		{"no credential required", &identity{Role: roleClient, Anonymous: true}, ""},
		{"nobody", nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.id.logKeyID(); got != tc.want {
				t.Errorf("logKeyID = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestSharedEnvTokenStillRegisters: nothing ever forbade setting the same string
// as both ROUTER_WORKER_TOKEN and ROUTER_CLIENT_TOKENS, and a deployment that
// did must keep registering. identify resolves the client list first, so the
// worker check has to run ahead of it.
func TestSharedEnvTokenStillRegisters(t *testing.T) {
	const shared = "one-token-for-everything"
	r := &Router{
		cfg:      &Config{WorkerToken: shared, ClientTokens: []string{shared}},
		registry: newTestRegistry(),
		logs:     newTestLogStore(t),
		client:   &http.Client{Timeout: time.Second},
	}
	rec := httptest.NewRecorder()
	r.routes().ServeHTTP(rec, post("/backends/register",
		`{"id":"llm-a750","url":"http://a750:8080","model":"gemma4"}`, shared))
	if rec.Code != http.StatusOK {
		t.Fatalf("register with a shared token = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	// And the same token still works on the client surface.
	rec = httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer "+shared)
	r.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("/v1/models with a shared token = %d, want 200", rec.Code)
	}
}
