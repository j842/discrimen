package router

// Virtual keys — who is calling, what they may name, and how much they may spend.
//
// The router grew up behind a LAN with two shared secrets in the environment:
// one worker token and a comma-separated list of client tokens. That is a
// perfectly good design for a fleet whose callers you administer, and it has
// exactly one failure mode, which arrives the moment a caller is a stranger:
// there is no such thing as revoking one of them, no record of which one made a
// request, and no way to say "this caller gets these two models and a million
// tokens a month".
//
// So: a keys table. Keys are `sk-` prefixed so they paste into anything written
// against OpenAI, hashed with SHA-256 at rest, and shown exactly once — the
// create response is the only moment the plaintext exists anywhere. The
// environment variables stay and keep working, because they are named in the
// compatibility contract and because a single-operator LAN deployment should not
// have to grow a key management story it does not want.
//
// The at-rest hash is a plain SHA-256, not a password hash, and that is
// deliberate rather than an oversight. A key is 32 bytes of crypto/rand — 256
// bits of entropy with no structure to guess at — so the offline attack a slow
// KDF defends against does not exist here. The admin PASSWORD is the opposite
// case (a human chose it), and it gets PBKDF2 accordingly.

import (
	"bytes"
	"context"
	"crypto/pbkdf2"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Roles. Deliberately three and not a permission matrix: the router has three
// surfaces (the OpenAI API, worker registration, and everything an operator
// does), and a role that does not map onto one of them would be a scope nobody
// can describe.
const (
	roleAdmin  = "admin"
	roleClient = "client"
	roleWorker = "worker"
)

// keyPrefix is what makes a discrimen key paste into a client written against
// OpenAI without an argument about the format.
const keyPrefix = "sk-"

// keyRandomBytes is the entropy behind a key. 32 bytes is what makes the plain
// SHA-256 at rest defensible — see the file comment.
const keyRandomBytes = 32

// base64URL encodes keys and session tokens: URL-safe and unpadded, so a key
// survives a query string, a shell, and a copy-paste without escaping.
var base64URL = base64.RawURLEncoding

// apiKey is one row of the keys table. The plaintext key is never in here: it
// exists once, in the response to the call that created it.
type apiKey struct {
	ID     int64  `json:"id"`
	Prefix string `json:"prefix"`
	Name   string `json:"name"`
	Role   string `json:"role"`
	// Enabled is a soft revoke. Preferred over deleting, because the key id is
	// stamped on every request log row and deleting the key orphans that history.
	Enabled    bool      `json:"enabled"`
	CreatedAt  time.Time `json:"created_at"`
	LastUsedAt time.Time `json:"last_used_at,omitempty"`
	// Models is an allow-list. Empty means any model the fleet serves.
	Models []string `json:"models,omitempty"`
	// TokenBudget is a lifetime ceiling on TokensUsed. 0 means unlimited.
	TokenBudget int64 `json:"token_budget,omitempty"`
	TokensUsed  int64 `json:"tokens_used"`
}

// identity is who made a request. Nil means no recognised credential was
// presented, which is not the same as "not allowed" — see requireClient.
type identity struct {
	Role string
	// KeyID is the api_keys row, or 0 for an environment token, which has no row
	// and no per-key limits.
	KeyID int64
	Name  string
	// Anonymous marks a caller the router let through because it required no
	// credential at all — the trusted-LAN case. Distinct from an environment
	// token in the request log, where "nothing was asked for" and "one of the
	// two shared secrets" are different facts about who was there.
	Anonymous   bool
	Models      []string
	TokenBudget int64
	TokensUsed  int64
}

// logKeyID is what gets stamped on a request log row. A decimal row id for a
// real key, "env" for one of the environment tokens (which have no row and
// cannot be told apart), empty when the router required no credential at all.
func (id *identity) logKeyID() string {
	switch {
	case id == nil, id.Anonymous:
		return ""
	case id.KeyID > 0:
		return strconv.FormatInt(id.KeyID, 10)
	default:
		return "env"
	}
}

// allowsModel reports whether this key may name the given model. An empty name
// is the auto route — the caller named nothing, so there is nothing to refuse —
// and an empty allow-list means every model.
//
// The list is matched against whatever spelling the client sent, which is the
// same string backendServesModel resolves: the model id, the worker id, or the
// published alias. An allow-list entry therefore means what an operator reading
// /v1/models would expect it to mean.
func (id *identity) allowsModel(name string) bool {
	if id == nil || len(id.Models) == 0 || name == "" {
		return true
	}
	for _, m := range id.Models {
		if strings.EqualFold(m, name) {
			return true
		}
	}
	return false
}

// overBudget reports whether this key has spent its lifetime token allowance.
func (id *identity) overBudget() bool {
	return id != nil && id.TokenBudget > 0 && id.TokensUsed >= id.TokenBudget
}

// newAPIKey mints a key and returns the plaintext alongside the row to store.
// The plaintext is returned rather than kept, so nothing but the caller of the
// create endpoint ever holds it.
func newAPIKey(name, role string, models []string, budget int64) (plain string, key apiKey, err error) {
	random, err := randomToken(keyRandomBytes)
	if err != nil {
		return "", apiKey{}, err
	}
	plain = keyPrefix + random
	return plain, apiKey{
		// Enough to recognise a key in a list without being enough to use it:
		// 8 base64 characters is 48 bits, leaving 208 unrevealed.
		Prefix:      plain[:len(keyPrefix)+8],
		Name:        strings.TrimSpace(name),
		Role:        role,
		Enabled:     true,
		CreatedAt:   time.Now().UTC(),
		Models:      normalizeModelList(models),
		TokenBudget: budget,
	}, nil
}

// normalizeModelList trims and de-duplicates an allow-list, preserving the
// operator's order so the stored value reads back the way it was typed.
func normalizeModelList(models []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, m := range models {
		m = strings.TrimSpace(m)
		if m == "" || seen[strings.ToLower(m)] {
			continue
		}
		seen[strings.ToLower(m)] = true
		out = append(out, m)
	}
	return out
}

func validRole(role string) bool {
	switch role {
	case roleAdmin, roleClient, roleWorker:
		return true
	}
	return false
}

// ── Storage ─────────────────────────────────────────────────────────────────

func (s *LogStore) CreateAPIKey(ctx context.Context, plain string, key apiKey) (apiKey, error) {
	res, err := s.db.ExecContext(ctx, `INSERT INTO api_keys
		(key_hash, prefix, name, role, enabled, created_at, last_used_at, models, token_budget, tokens_used)
		VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, 0)`,
		sha256Hex(plain), key.Prefix, key.Name, key.Role, boolInt(key.Enabled),
		key.CreatedAt.Format(time.RFC3339Nano), strings.Join(key.Models, ","), key.TokenBudget)
	if err != nil {
		return apiKey{}, err
	}
	key.ID, err = res.LastInsertId()
	return key, err
}

// LookupAPIKey finds an ENABLED key by the hash of its plaintext. Indexed, not
// scanned: a linear walk over the table would put the whole key list in front of
// every request and grow linearly with the number of callers.
//
// The stored hash is compared again in constant time even though the SELECT
// already matched on it. That is not redundant belt-and-braces about SQL — it is
// the one comparison in the path this code owns, and leaving it out would make
// the discipline in authorizedAsClient look like an accident rather than a rule.
func (s *LogStore) LookupAPIKey(ctx context.Context, plain string) (apiKey, bool) {
	if strings.TrimSpace(plain) == "" {
		return apiKey{}, false
	}
	want := sha256Hex(plain)
	var (
		key      apiKey
		gotHash  string
		enabled  int
		created  string
		lastUsed string
		models   string
	)
	err := s.db.QueryRowContext(ctx,
		`SELECT id, key_hash, prefix, name, role, enabled, created_at, last_used_at, models, token_budget, tokens_used
		 FROM api_keys WHERE key_hash = ?`, want).
		Scan(&key.ID, &gotHash, &key.Prefix, &key.Name, &key.Role, &enabled,
			&created, &lastUsed, &models, &key.TokenBudget, &key.TokensUsed)
	if err != nil || !constantTimeEqual(gotHash, want) || enabled == 0 {
		return apiKey{}, false
	}
	key.Enabled = true
	key.CreatedAt, _ = time.Parse(time.RFC3339Nano, created)
	key.LastUsedAt, _ = time.Parse(time.RFC3339Nano, lastUsed)
	key.Models = splitModelList(models)
	return key, true
}

func (s *LogStore) ListAPIKeys(ctx context.Context) ([]apiKey, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, prefix, name, role, enabled, created_at, last_used_at, models, token_budget, tokens_used
		 FROM api_keys ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []apiKey{}
	for rows.Next() {
		var (
			key                       apiKey
			enabled                   int
			created, lastUsed, models string
		)
		if err := rows.Scan(&key.ID, &key.Prefix, &key.Name, &key.Role, &enabled,
			&created, &lastUsed, &models, &key.TokenBudget, &key.TokensUsed); err != nil {
			return nil, err
		}
		key.Enabled = enabled != 0
		key.CreatedAt, _ = time.Parse(time.RFC3339Nano, created)
		key.LastUsedAt, _ = time.Parse(time.RFC3339Nano, lastUsed)
		key.Models = splitModelList(models)
		out = append(out, key)
	}
	return out, rows.Err()
}

// UpdateAPIKey changes the mutable half of a key: whether it is enabled, its
// name, its allow-list and its budget. The hash and the role are immutable —
// re-roling a key in place would silently change what an already-issued
// credential can reach.
func (s *LogStore) UpdateAPIKey(ctx context.Context, id int64, key apiKey) error {
	res, err := s.db.ExecContext(ctx,
		`UPDATE api_keys SET name = ?, enabled = ?, models = ?, token_budget = ? WHERE id = ?`,
		key.Name, boolInt(key.Enabled), strings.Join(key.Models, ","), key.TokenBudget, id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (s *LogStore) DeleteAPIKey(ctx context.Context, id int64) error {
	res, err := s.db.ExecContext(ctx, `DELETE FROM api_keys WHERE id = ?`, id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// HasEnabledKey reports whether any enabled key holds one of the given roles.
// It is what decides whether a credential is REQUIRED: an empty
// ROUTER_CLIENT_TOKENS used to mean "no client authentication at all", which is
// fine on a trusted LAN and wrong for a public image, so a generated bootstrap
// key has to switch the check on.
func (s *LogStore) HasEnabledKey(ctx context.Context, roles ...string) (bool, error) {
	if len(roles) == 0 {
		return false, nil
	}
	placeholders := strings.TrimSuffix(strings.Repeat("?,", len(roles)), ",")
	args := make([]any, len(roles))
	for i, role := range roles {
		args[i] = role
	}
	var n int
	err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM api_keys WHERE enabled = 1 AND role IN (`+placeholders+`)`, args...).Scan(&n)
	return n > 0, err
}

// RecordKeyUse charges tokens against a key and stamps its last use. One
// statement, because both happen on exactly the same occasion.
func (s *LogStore) RecordKeyUse(ctx context.Context, id int64, tokens int) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE api_keys SET tokens_used = tokens_used + ?, last_used_at = ? WHERE id = ?`,
		tokens, time.Now().UTC().Format(time.RFC3339Nano), id)
	return err
}

func splitModelList(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	return normalizeModelList(strings.Split(raw, ","))
}

// ── Settings ────────────────────────────────────────────────────────────────

// settingAdminPasswordHash is the one setting so far. router_settings exists
// rather than a dedicated column because the alternative — a single-row table
// that grows a column per setting — needs a migration for every one of them.
const settingAdminPasswordHash = "admin_password_hash"

func (s *LogStore) LoadSetting(ctx context.Context, key string) (string, error) {
	var value string
	err := s.db.QueryRowContext(ctx, `SELECT value FROM router_settings WHERE key = ?`, key).Scan(&value)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	return value, err
}

func (s *LogStore) SaveSetting(ctx context.Context, key, value string) error {
	_, err := s.db.ExecContext(ctx, `INSERT INTO router_settings (key, value, updated_at)
		VALUES (?, ?, ?)
		ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`,
		key, value, time.Now().UTC().Format(time.RFC3339Nano))
	return err
}

// ── Password hashing ────────────────────────────────────────────────────────
//
// PBKDF2-HMAC-SHA256, from the standard library's crypto/pbkdf2 (Go 1.24).
//
// bcrypt or argon2 would be the usual answer and both would be better hashes.
// Both also live in golang.org/x/crypto, and this module has exactly ONE direct
// dependency — modernc.org/sqlite, which is what lets the image be a static
// binary on alpine with no cgo. Adding a second for one password is a real cost
// against a marginal gain, and the gain is marginal because PBKDF2 with a
// current iteration count is a specified, reviewed KDF (RFC 8018) rather than
// something invented here. Argon2's advantage is memory-hardness against GPU
// attack, which needs the hash to leak first: this one lives in a SQLite file
// that also holds every request body the router has logged.
//
// The iteration count is stored WITH the hash, so raising it later is a
// one-line change that leaves existing passwords working.

const (
	passwordIterations = 600_000 // OWASP's 2023 figure for PBKDF2-HMAC-SHA256
	passwordSaltBytes  = 16
	passwordKeyBytes   = 32
	passwordScheme     = "pbkdf2-sha256"
)

// hashPassword produces "pbkdf2-sha256$<iterations>$<salt>$<key>".
func hashPassword(password string) (string, error) {
	return hashPasswordIter(password, passwordIterations)
}

func hashPasswordIter(password string, iterations int) (string, error) {
	salt, err := randomToken(passwordSaltBytes)
	if err != nil {
		return "", err
	}
	raw, err := base64URL.DecodeString(salt)
	if err != nil {
		return "", err
	}
	key, err := pbkdf2.Key(sha256.New, password, raw, iterations, passwordKeyBytes)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s$%d$%s$%s", passwordScheme, iterations, salt, base64URL.EncodeToString(key)), nil
}

// verifyPassword checks a password against a stored hash. False for anything it
// cannot parse, so a corrupt or truncated row locks admin access out rather than
// opening it.
func verifyPassword(stored, password string) bool {
	parts := strings.Split(stored, "$")
	if len(parts) != 4 || parts[0] != passwordScheme {
		return false
	}
	iterations, err := strconv.Atoi(parts[1])
	if err != nil || iterations < 1 {
		return false
	}
	salt, err := base64URL.DecodeString(parts[2])
	if err != nil {
		return false
	}
	key, err := pbkdf2.Key(sha256.New, password, salt, iterations, passwordKeyBytes)
	if err != nil {
		return false
	}
	return constantTimeEqual(base64URL.EncodeToString(key), parts[3])
}

// minPasswordLength is a floor, not a policy. There are no character-class rules
// here on purpose: they are known to push people toward shorter, more guessable
// passwords, and this is one account on one appliance.
const minPasswordLength = 10

func validPassword(password string) error {
	if len([]rune(password)) < minPasswordLength {
		return fmt.Errorf("password must be at least %d characters", minPasswordLength)
	}
	return nil
}

// ── Authentication ──────────────────────────────────────────────────────────

// bearerToken extracts the credential from an Authorization header.
func bearerToken(req *http.Request) string {
	auth := req.Header.Get("Authorization")
	if len(auth) > 7 && strings.EqualFold(auth[:7], "bearer ") {
		return strings.TrimSpace(auth[7:])
	}
	return ""
}

// identify resolves the credential a request presented, or nil for none.
//
// The environment tokens are checked first and without a database read, so the
// deployment that has always run on ROUTER_CLIENT_TOKENS pays nothing for a keys
// table it does not use.
func (r *Router) identify(req *http.Request) *identity {
	token := bearerToken(req)
	if token == "" {
		return nil
	}
	if r.cfg != nil {
		if len(r.cfg.ClientTokens) > 0 && authorizedAsClient(req, r.cfg.ClientTokens) {
			return &identity{Role: roleClient, Name: "ROUTER_CLIENT_TOKENS"}
		}
		if r.cfg.WorkerToken != "" && authorizedAsWorker(req, r.cfg.WorkerToken) {
			return &identity{Role: roleWorker, Name: "ROUTER_WORKER_TOKEN"}
		}
	}
	if r.logs == nil {
		return nil
	}
	key, ok := r.logs.LookupAPIKey(req.Context(), token)
	if !ok {
		return nil
	}
	return &identity{
		Role: key.Role, KeyID: key.ID, Name: key.Name,
		Models: key.Models, TokenBudget: key.TokenBudget, TokensUsed: key.TokensUsed,
	}
}

// requireClient resolves the caller on the OpenAI surface, answering 401 and
// returning ok=false when it cannot.
//
// An ADMIN key is accepted here too. The alternative is an operator who has to
// hold two credentials to test the thing they just configured, and there is no
// authority a client has that an admin does not.
func (r *Router) requireClient(w http.ResponseWriter, req *http.Request) (*identity, bool) {
	if id := r.identify(req); id != nil {
		switch id.Role {
		case roleClient, roleAdmin:
			return id, true
		}
		// A worker token is not a client token. It has always been able to
		// register backends and never to spend them.
		unauthorized(w)
		return nil, false
	}
	if !r.clientAuthRequired() {
		// No client credential is configured at all: the historical open-fleet
		// behaviour, kept for the trusted-LAN deployment. Bootstrap generates one
		// on first run precisely so a public image never lands here.
		return &identity{Role: roleClient, Anonymous: true, Name: "unauthenticated"}, true
	}
	unauthorized(w)
	return nil, false
}

// requireWorker gates the frozen registration endpoints. Same shape as
// requireClient, including the "nothing configured means open" rule that every
// deployed beacon currently relies on.
func (r *Router) requireWorker(w http.ResponseWriter, req *http.Request) bool {
	if r.workerAuthorized(req) {
		return true
	}
	unauthorized(w)
	return false
}

// workerAuthorized is requireWorker without the 401, for the one caller that
// accepts either a worker or an admin and has to try both.
func (r *Router) workerAuthorized(req *http.Request) bool {
	// ROUTER_WORKER_TOKEN is checked here directly, ahead of identify, because
	// identify resolves the client list first. Nothing ever forbade setting the
	// same string as both ROUTER_WORKER_TOKEN and ROUTER_CLIENT_TOKENS, and a
	// deployment that did would otherwise have its beacons identified as clients
	// and refused at the registration endpoint they have always used.
	if r.cfg != nil && r.cfg.WorkerToken != "" && authorizedAsWorker(req, r.cfg.WorkerToken) {
		return true
	}
	if id := r.identify(req); id != nil {
		switch id.Role {
		case roleWorker, roleAdmin:
			return true
		}
		return false
	}
	return !r.workerAuthRequired()
}

// clientAuthRequired / workerAuthRequired report whether a credential must be
// presented at all. Cached in an atomic rather than queried per request: it
// changes only when a key is created, enabled or deleted, and those paths
// refresh it (see refreshAuthRequired).
func (r *Router) clientAuthRequired() bool {
	return (r.cfg != nil && len(r.cfg.ClientTokens) > 0) || r.clientKeysExist.Load()
}

func (r *Router) workerAuthRequired() bool {
	return (r.cfg != nil && r.cfg.WorkerToken != "") || r.workerKeysExist.Load()
}

// refreshAuthRequired re-reads whether any enabled key of each kind exists.
// Called at startup and after every write to the keys table, so revoking the
// last client key reopens the fleet exactly as deleting the environment variable
// would — surprising, but the alternative is a router nobody can call.
func (r *Router) refreshAuthRequired(ctx context.Context) {
	if r.logs == nil {
		return
	}
	if ok, err := r.logs.HasEnabledKey(ctx, roleClient, roleAdmin); err == nil {
		r.clientKeysExist.Store(ok)
	} else {
		log.Printf("check for enabled client keys failed: %v", err)
	}
	if ok, err := r.logs.HasEnabledKey(ctx, roleWorker, roleAdmin); err == nil {
		r.workerKeysExist.Store(ok)
	} else {
		log.Printf("check for enabled worker keys failed: %v", err)
	}
}

// ── Per-key limits on the request path ──────────────────────────────────────

// enforceKeyLimits applies a key's allow-list and budget. It writes the refusal
// itself and returns false, so a handler is one `if` away from being correct.
func (r *Router) enforceKeyLimits(w http.ResponseWriter, id *identity, model string) bool {
	if !id.allowsModel(model) {
		writeJSON(w, http.StatusForbidden, validationError{
			Message: fmt.Sprintf("this key may not use model %q (allowed: %s)", model, strings.Join(id.Models, ", ")),
			Param:   "model",
		})
		return false
	}
	if id.overBudget() {
		// 429 rather than 402 or 403: a budget is a rate limit over a long window,
		// and rate_limit_error is the type an OpenAI client already knows how to
		// back off from.
		writeJSON(w, http.StatusTooManyRequests, validationError{
			Message: fmt.Sprintf("token budget exhausted (%d of %d used)", id.TokensUsed, id.TokenBudget),
		})
		return false
	}
	return true
}

// recordKeyUse charges an exchange against a key's budget and stamps its last
// use. Best-effort and asynchronous by contract: it runs after the response is
// committed, and a budget is a spending bound rather than an invoice, so losing
// one request's charge to a write error must not fail a request that already
// succeeded.
//
// Environment tokens have no row and no budget, so they cost nothing here.
func (r *Router) recordKeyUse(id *identity, tokens int) {
	if id == nil || id.KeyID == 0 || r.logs == nil {
		return
	}
	if tokens <= 0 && !r.keyUseDue(id.KeyID) {
		return // nothing to charge and the last-used stamp is still fresh
	}
	if tokens < 0 {
		tokens = 0
	}
	if err := r.logs.RecordKeyUse(context.Background(), id.KeyID, tokens); err != nil {
		log.Printf("charge %d tokens to key %d failed: %v", tokens, id.KeyID, err)
	}
}

// usageCaptureBytes bounds the reply capture kept purely for budgeting on the
// passthrough path. Small on purpose: boundedCapture keeps a head and a TAIL,
// usage lives at the tail in both the buffered and the SSE shape, and nothing
// else here reads the body.
const usageCaptureBytes = 4096

// usageTotalTokens reads the last token count an OpenAI-shaped reply reported.
//
// It scans rather than decoding, and that is not laziness. It runs over a
// BOUNDED capture whose middle has been dropped, and over SSE bodies that were
// never one JSON document to begin with; a decoder fails on both. Both shapes
// put usage at the end, which is the half the capture keeps.
//
// total_tokens is preferred because it is what the endpoint charged for.
// Failing that, prompt + completion, which is the same number spelled out.
// Returns 0 when the endpoint reported nothing, which is a signal to the caller
// to fall back to the routing estimate rather than to charge zero.
func usageTotalTokens(body []byte) int {
	if n := lastJSONInt(body, "total_tokens"); n > 0 {
		return n
	}
	return lastJSONInt(body, "prompt_tokens") + lastJSONInt(body, "completion_tokens")
}

// lastJSONInt reads the integer following the LAST occurrence of "key": in body.
// Last rather than first because llama.cpp's per-chunk usage counts up, so the
// final value is the total.
func lastJSONInt(body []byte, key string) int {
	needle := []byte(`"` + key + `"`)
	idx := bytes.LastIndex(body, needle)
	if idx < 0 {
		return 0
	}
	i := idx + len(needle)
	for i < len(body) && (body[i] == ' ' || body[i] == ':') {
		i++
	}
	start := i
	for i < len(body) && body[i] >= '0' && body[i] <= '9' {
		i++
	}
	if start == i {
		return 0
	}
	n, err := strconv.Atoi(string(body[start:i]))
	if err != nil {
		return 0
	}
	return n
}

// keyUseStampInterval throttles the last-used stamp for calls that consume no
// tokens (/v1/models, a preview). Without it every such call is a write on a
// database with one connection, which is a lot of contention to buy a timestamp
// nobody reads to the minute.
const keyUseStampInterval = time.Minute

func (r *Router) keyUseDue(id int64) bool {
	now := time.Now()
	prev, loaded := r.keyStamped.Load(id)
	if loaded {
		if at, ok := prev.(time.Time); ok && now.Sub(at) < keyUseStampInterval {
			return false
		}
	}
	r.keyStamped.Store(id, now)
	return true
}

// ── Bootstrap ───────────────────────────────────────────────────────────────

// bootstrapCredentials generates the credentials an operator did not supply, so
// a bare `docker run` is authenticated rather than open.
//
// Each is printed ONLY on the run that generates it. Printing on every start
// would put a live credential in the log of a machine whose logs are shipped
// somewhere, and would train an operator to ignore the banner.
func (r *Router) bootstrapCredentials(ctx context.Context) error {
	if r.logs == nil {
		return nil
	}
	if err := r.bootstrapKey(ctx, roleClient, len(r.cfg.ClientTokens) > 0,
		"ROUTER_CLIENT_TOKENS", "callers of the OpenAI API (/v1/chat/completions and friends)"); err != nil {
		return err
	}
	if err := r.bootstrapKey(ctx, roleWorker, r.cfg.WorkerToken != "",
		"ROUTER_WORKER_TOKEN", "workers registering themselves at /backends/register"); err != nil {
		return err
	}
	if err := r.bootstrapAdminPassword(ctx); err != nil {
		return err
	}
	r.refreshAuthRequired(ctx)
	return nil
}

func (r *Router) bootstrapKey(ctx context.Context, role string, envSet bool, envName, purpose string) error {
	if envSet {
		return nil // the operator supplied one; the database does not need to
	}
	exists, err := r.logs.HasEnabledKey(ctx, role)
	if err != nil {
		return err
	}
	if exists {
		return nil
	}
	plain, key, err := newAPIKey("bootstrap", role, nil, 0)
	if err != nil {
		return err
	}
	if _, err := r.logs.CreateAPIKey(ctx, plain, key); err != nil {
		return err
	}
	announceCredential(strings.ToUpper(role)+" TOKEN", plain, fmt.Sprintf(
		"For %s.\nGenerated because %s is empty and no %s key existed.\nStored (hashed) in the database — this is the only time it is printed.",
		purpose, envName, role))
	return nil
}

func (r *Router) bootstrapAdminPassword(ctx context.Context) error {
	stored, err := r.logs.LoadSetting(ctx, settingAdminPasswordHash)
	if err != nil {
		return err
	}
	if stored != "" {
		// The database is canonical. ROUTER_ADMIN_PASSWORD seeds a database that
		// has none and does nothing afterwards, so a password rotated in the UI is
		// not silently reverted by the next restart.
		return nil
	}
	password := strings.TrimSpace(r.cfg.AdminPassword)
	generated := password == ""
	if generated {
		if password, err = randomToken(18); err != nil {
			return err
		}
	}
	hash, err := hashPassword(password)
	if err != nil {
		return err
	}
	if err := r.logs.SaveSetting(ctx, settingAdminPasswordHash, hash); err != nil {
		return err
	}
	if generated {
		announceCredential("ADMIN PASSWORD", password,
			"For the admin UI and /admin/*.\nGenerated because ROUTER_ADMIN_PASSWORD is empty.\n"+
				"Stored (hashed) in the database — this is the only time it is printed.")
	} else {
		log.Printf("admin password seeded from ROUTER_ADMIN_PASSWORD; the database is canonical from now on")
	}
	return nil
}

// announceCredential prints a generated secret so an operator scrolling
// `docker logs` cannot miss it. Loud on purpose: this is the one line in the
// whole startup sequence that cannot be recovered if it scrolls past.
func announceCredential(title, value, note string) {
	const rule = "=================================================================="
	var b strings.Builder
	b.WriteString("\n" + rule + "\n")
	b.WriteString("  GENERATED " + title + " — COPY IT NOW, IT IS NOT SHOWN AGAIN\n")
	b.WriteString(rule + "\n\n      " + value + "\n\n")
	for _, line := range strings.Split(note, "\n") {
		b.WriteString("  " + line + "\n")
	}
	b.WriteString(rule)
	log.Print(b.String())
}
