package router

// The admin surface: the single gate everything operator-scoped goes through,
// the password session behind it, and CRUD over the provider rows and api keys.
//
// Two things it deliberately is not. It is not a second registration endpoint —
// /backends/register is frozen by the compatibility contract and untouched, and
// a row that arrives there is a beacon whatever it says. And it is not a routing
// surface: everything here governs WHICH endpoints exist and WHO may call them,
// never how a request is routed. That line is the first design principle in
// PLAN.md and this file is the largest new configuration surface in the project,
// so it is the one most likely to erode it.

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ── Provider rows ───────────────────────────────────────────────────────────

// providerSpec is the write shape for a manual row.
//
// Every field is a POINTER so that "absent" and "zero" are different things. It
// matters most for price: seedPrices fills any field that is zero, so without
// the distinction an operator could set a price but never clear one, and a free
// model behind a metered endpoint would be unrepresentable — every write would
// re-seed the published price straight back over the zero. See stated().
type providerSpec struct {
	ID                 *string   `json:"id"`
	URL                *string   `json:"url"`
	Model              *string   `json:"model"`
	Provider           *string   `json:"provider"`
	APIKey             *string   `json:"api_key"`
	Features           *[]string `json:"features"`
	HealthPath         *string   `json:"health_path"`
	TTLSeconds         *int      `json:"ttl_seconds"`
	MaxConcurrency     *int      `json:"max_concurrency"`
	ContextK           *int      `json:"context_k"`
	Quality            *int      `json:"quality"`
	BaselineTPS        *float64  `json:"baseline_tps"`
	InputPricePerMtok  *float64  `json:"input_price_per_mtok"`
	OutputPricePerMtok *float64  `json:"output_price_per_mtok"`
}

// applyTo writes the fields the caller actually sent onto a registration.
func (s providerSpec) applyTo(reg *BackendRegistration) {
	assign(&reg.ID, s.ID)
	assign(&reg.URL, s.URL)
	assign(&reg.Model, s.Model)
	assign(&reg.Provider, s.Provider)
	assign(&reg.APIKey, s.APIKey)
	assign(&reg.HealthPath, s.HealthPath)
	assign(&reg.TTLSeconds, s.TTLSeconds)
	assign(&reg.MaxConcurrency, s.MaxConcurrency)
	assign(&reg.ContextK, s.ContextK)
	assign(&reg.Quality, s.Quality)
	assign(&reg.BaselineTPS, s.BaselineTPS)
	assign(&reg.InputPricePerMtok, s.InputPricePerMtok)
	assign(&reg.OutputPricePerMtok, s.OutputPricePerMtok)
	if s.Features != nil {
		reg.Features = append([]string(nil), (*s.Features)...)
	}
}

// stated reports which price/context fields the caller named, so seeding leaves
// them alone. Sending 0 is a declaration ("this model is free", "measure the
// context yourself"), and it has to outlast the next write.
func (s providerSpec) stated() priceStated {
	return priceStated{
		Input:   s.InputPricePerMtok != nil,
		Output:  s.OutputPricePerMtok != nil,
		Context: s.ContextK != nil,
	}
}

func assign[T any](dst *T, src *T) {
	if src != nil {
		*dst = *src
	}
}

// providerHealthPath is what a manual row's health check uses when the operator
// names none. A metered endpoint almost never serves /health, and /v1/models is
// the one route every OpenAI-compatible provider does publish — it also proves
// the api key works, which /health would not. Beacon rows keep /health: that
// default is part of the frozen registration contract.
const providerHealthPath = "/v1/models"

func (r *Router) handleAdminProviders(w http.ResponseWriter, req *http.Request) {
	if !r.requireAdmin(w, req) {
		return
	}
	switch req.Method {
	case http.MethodGet:
		rows := r.registry.manualRows()
		writeJSON(w, http.StatusOK, map[string]any{
			"providers": rows,
			// What the embedded LiteLLM table publishes for these models, so the
			// admin page can attribute a price or a context window it is showing
			// instead of presenting an invented number as an operator's own. See
			// priceReference: seeding leaves no trace on the row itself.
			"price_reference": priceReference(rows),
			"price_source":    priceSourceInfo(),
		})
	case http.MethodPost:
		r.createProvider(w, req)
	default:
		methodNotAllowed(w)
	}
}

func (r *Router) handleAdminProviderByID(w http.ResponseWriter, req *http.Request) {
	if !r.requireAdmin(w, req) {
		return
	}
	id := strings.Trim(strings.TrimPrefix(req.URL.Path, "/admin/providers/"), "/")
	if id == "" {
		r.handleAdminProviders(w, req)
		return
	}
	existing := r.registry.get(id)
	if existing == nil {
		writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("provider %q not found", id)})
		return
	}
	// Beacon and relay rows are reachable here by id, and must not be editable
	// here. Neither one's values are the operator's: a beacon's belong to the
	// worker that posts them and the next keepalive would overwrite whatever was
	// typed, and a relay's belong to the upstream router and the next fleet
	// refresh would do the same. Say so rather than accept an edit that silently
	// evaporates within the minute.
	if !isManualRow(existing) {
		message := fmt.Sprintf("backend %q registered itself (source=%s); it is managed by its own beacon, not here", id, existing.Source)
		if isRelayRow(existing) {
			message = fmt.Sprintf("backend %q is derived from relay %q; edit the relay at /admin/relays/%s, or change it on the upstream router",
				id, existing.Relay, existing.Relay)
		}
		writeJSON(w, http.StatusConflict, validationError{Message: message})
		return
	}
	switch req.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, publicBackends([]*Backend{existing})[0])
	case http.MethodPatch, http.MethodPut:
		// The live row goes with the edit, so a persist that fails has something
		// to put back — see saveProvider.
		r.updateProvider(w, req, existing)
	case http.MethodDelete:
		r.deleteProvider(w, req, id)
	default:
		methodNotAllowed(w)
	}
}

func (r *Router) createProvider(w http.ResponseWriter, req *http.Request) {
	spec, err := decodeProviderSpec(w, req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: %s", err)
		return
	}
	var reg BackendRegistration
	spec.applyTo(&reg)
	reg.Source = sourceManual
	if strings.TrimSpace(reg.Model) == "" {
		writeJSON(w, http.StatusBadRequest, validationError{Message: "model is required", Param: "model"})
		return
	}
	if reg.HealthPath == "" {
		reg.HealthPath = providerHealthPath
	}
	if strings.TrimSpace(reg.ID) == "" {
		reg.ID = providerRowID(reg.Provider, reg.Model)
	}
	if err := normalizeRegistration(&reg); err != nil {
		writeJSON(w, http.StatusBadRequest, validationError{Message: err.Error()})
		return
	}
	if r.registry.get(reg.ID) != nil {
		writeJSON(w, http.StatusConflict, validationError{
			Message: fmt.Sprintf("backend %q already exists", reg.ID), Param: "id",
		})
		return
	}
	if clash := r.duplicateEndpointModel(&reg, ""); clash != "" {
		writeJSON(w, http.StatusConflict, validationError{
			Message: fmt.Sprintf("%s already serves model %q (a row is one (endpoint, model) pair)", clash, reg.Model),
			Param:   "model",
		})
		return
	}
	// Seeding runs AFTER normalisation so it sees the settled provider name.
	r.saveProvider(w, req, reg, nil, http.StatusCreated, seedPrices(&reg, spec.stated()))
}

func (r *Router) updateProvider(w http.ResponseWriter, req *http.Request, existing *Backend) {
	spec, err := decodeProviderSpec(w, req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: %s", err)
		return
	}
	id := existing.ID
	// Patch the operator's own DECLARED registration, not the live row. The live
	// row carries whatever the profiler measured into the fields the operator left
	// blank, and folding those back in would silently promote a measurement to a
	// declaration — after one edit the probe could never refine them again.
	declared, ok := r.registry.declaredRegistration(id)
	if !ok {
		writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("provider %q not found", id)})
		return
	}
	reg := declared
	spec.applyTo(&reg)
	reg.ID = id // the path owns the identity; renaming is a delete plus a create
	reg.Source = sourceManual
	if err := normalizeRegistration(&reg); err != nil {
		writeJSON(w, http.StatusBadRequest, validationError{Message: err.Error()})
		return
	}
	if clash := r.duplicateEndpointModel(&reg, id); clash != "" {
		writeJSON(w, http.StatusConflict, validationError{
			Message: fmt.Sprintf("%s already serves model %q (a row is one (endpoint, model) pair)", clash, reg.Model),
			Param:   "model",
		})
		return
	}
	// What the request named is only part of what has been settled. This edit has
	// no reason to mention a price — the dashboard sends only the fields that
	// changed, and a zero is absent from the row it filled its form from — so the
	// fields the last write already settled count as stated too, or an explicit 0
	// would be re-seeded away on the first unrelated edit. See alreadySeeded.
	stated := spec.stated().or(alreadySeeded(declared, reg))
	r.saveProvider(w, req, reg, existing, http.StatusOK, seedPrices(&reg, stated))
}

// saveProvider commits a manual row: registry, persistence, certification.
//
// prev is the row as it stood before this write, or nil on a create. It is what
// a failed persist is rolled back to — see restore for why the two cases cannot
// share one undo.
func (r *Router) saveProvider(w http.ResponseWriter, req *http.Request, reg BackendRegistration, prev *Backend, status int, seeded []string) {
	backend, changed := r.registry.upsert(reg)
	if err := r.logs.SaveBackendRegistration(req.Context(), reg); err != nil {
		// Unlike a beacon row, nothing will re-post this one: a failed write here
		// means the row disappears on the next restart, so it is an error the
		// operator has to see rather than a log line.
		//
		// An EDIT undoes differently. Its pre-edit row is still on disk and still
		// correct, so removing it would take a live provider out of routing and
		// /v1/models until a restart brought it back — a much larger failure than
		// the edit that was refused.
		if prev != nil {
			r.registry.restore(prev)
		} else {
			r.registry.remove(reg.ID)
		}
		writeJSON(w, http.StatusInternalServerError, validationError{Message: fmt.Sprintf("persist provider: %s", err)})
		return
	}
	if changed || !backend.Certification.Ready {
		go r.certifyBackend(backend.ID)
	}
	resp := map[string]any{"provider": publicBackends([]*Backend{backend})[0]}
	if len(seeded) > 0 {
		// Say what was invented rather than letting numbers appear on the
		// operator's row without explanation.
		resp["seeded_from_price_data"] = seeded
	}
	writeJSON(w, status, resp)
}

func (r *Router) deleteProvider(w http.ResponseWriter, req *http.Request, id string) {
	// No live-backend guard here, unlike DELETE /backends/{id}. That guard exists
	// because a beacon re-registers within 60 seconds, so deleting a healthy one
	// is almost always a mistake that undoes itself confusingly. Nothing
	// re-creates a manual row, so deleting one is exactly what it looks like.
	if !r.registry.remove(id) {
		writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("provider %q not found", id)})
		return
	}
	if err := r.logs.DeleteBackendRegistration(req.Context(), id); err != nil {
		log.Printf("delete persisted provider %q: %v", id, err)
	}
	if err := r.logs.DeleteWorkerProfile(req.Context(), id); err != nil {
		log.Printf("delete worker profile for %q: %v", id, err)
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "removed", "id": id})
}

func decodeProviderSpec(w http.ResponseWriter, req *http.Request) (providerSpec, error) {
	var spec providerSpec
	err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 1<<20)).Decode(&spec)
	return spec, err
}

// duplicateEndpointModel reports the id of another row already serving this
// (endpoint, model) pair, or "". A catalogue endpoint gets one row per model it
// serves, so two rows agreeing on both are the same servable thing entered
// twice — which would double its share of the ranker and halve its apparent
// concurrency.
func (r *Router) duplicateEndpointModel(reg *BackendRegistration, ignoreID string) string {
	candidate := &Backend{BackendRegistration: *reg}
	for _, b := range r.registry.snapshot() {
		if b.ID == ignoreID || b.ID == reg.ID {
			continue
		}
		if sameEndpointModel(candidate, b) {
			return fmt.Sprintf("backend %q", b.ID)
		}
	}
	return ""
}

// providerRowID names a row the operator did not name. provider-model reads
// straight out of a fleet listing and stays distinct across the many models one
// catalogue endpoint serves, which is what "one row per (endpoint, model)"
// requires of the id.
func providerRowID(provider, model string) string {
	id := slugify(model)
	if p := slugify(provider); p != "" && p != providerLocal && !strings.HasPrefix(id, p) {
		id = p + "-" + id
	}
	return id
}

// slugify reduces a name to lowercase alphanumerics and single hyphens.
func slugify(s string) string {
	var b strings.Builder
	lastHyphen := true // leading hyphens are suppressed
	for _, r := range strings.ToLower(strings.TrimSpace(s)) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			lastHyphen = false
		case !lastHyphen:
			b.WriteByte('-')
			lastHyphen = true
		}
	}
	return strings.Trim(b.String(), "-")
}

// ── Admin authentication ────────────────────────────────────────────────────
//
// One password, hashed in the database, held in a session cookie. No OIDC and no
// external identity provider: this is a single-operator appliance, and every
// identity system that could be added here would be a larger dependency than the
// router itself.
//
// The DATABASE is canonical while ROUTER_ADMIN_PASSWORD is unset, so rotating
// the password is an admin action rather than a redeploy and an operator who
// changes it in the UI does not find it silently reverted on the next restart.
// A variable that IS set overrides the stored password on every start, because
// that is the only way back in for an operator who missed the generated one —
// see bootstrapAdminPassword.

// adminSessionTTL bounds how long a login lasts. Long enough that an operator
// working through the fleet is not re-prompted mid-task; short enough that a
// forgotten browser tab is not a standing grant.
const adminSessionTTL = 12 * time.Hour

// adminCookie is the session cookie name.
const adminCookie = "discrimen_admin"

// adminSessions holds live sessions in memory rather than in the database, which
// is what gives logout real teeth: invalidation is a map delete that no replica
// can be stale about, and a router restart ends every session — a safe direction
// to fail. The cost is that sessions do not survive a restart, which for a
// single-operator appliance is a re-login, not an outage.
//
// The zero value works: the map is created on first issue, under the lock.
type adminSessions struct {
	mu        sync.Mutex
	tokens    map[string]time.Time
	lastSwept time.Time
}

func (s *adminSessions) issue() (string, error) {
	token, err := randomToken(32)
	if err != nil {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tokens == nil {
		s.tokens = map[string]time.Time{}
	}
	s.sweepLocked()
	s.tokens[token] = time.Now().Add(adminSessionTTL)
	return token, nil
}

func (s *adminSessions) valid(token string) bool {
	if token == "" {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	expiry, ok := s.tokens[token]
	if !ok {
		return false
	}
	if time.Now().After(expiry) {
		delete(s.tokens, token)
		return false
	}
	return true
}

func (s *adminSessions) revoke(token string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.tokens, token)
}

// sweepLocked drops expired tokens. Called on issue rather than on a timer: the
// map only grows when someone logs in, so that is the only moment it can need
// pruning. Rate-limited so a login storm doesn't walk the map per request.
func (s *adminSessions) sweepLocked() {
	now := time.Now()
	if now.Sub(s.lastSwept) < time.Minute {
		return
	}
	s.lastSwept = now
	for token, expiry := range s.tokens {
		if now.After(expiry) {
			delete(s.tokens, token)
		}
	}
}

// requireAdmin is the single admin gate. Every handler that can read a stored
// prompt, enumerate the fleet or change what the router talks to goes through
// it, so the scope cannot drift apart across handlers the way the client check
// did.
//
// Two credentials are accepted, for two genuinely different callers: the session
// cookie a browser holds after POST /admin/login, and a bearer api key with the
// admin role, which is what a script or a terminal has.
func (r *Router) requireAdmin(w http.ResponseWriter, req *http.Request) bool {
	if r.adminAuthenticated(req) {
		return true
	}
	unauthorized(w)
	return false
}

func (r *Router) adminAuthenticated(req *http.Request) bool {
	if c, err := req.Cookie(adminCookie); err == nil && r.adminAuth.valid(c.Value) {
		return true
	}
	if id := r.identify(req); id != nil && id.Role == roleAdmin {
		return true
	}
	return false
}

// handleAdminLogin exchanges the admin password for a session cookie.
func (r *Router) handleAdminLogin(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var body struct {
		Password string `json:"password"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 4096)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: %s", err)
		return
	}
	stored, err := r.logs.LoadSetting(req.Context(), settingAdminPasswordHash)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	if stored == "" || !verifyPassword(stored, body.Password) {
		// One message for "no password is set" and for "wrong password": the
		// difference is exactly the thing an unauthenticated caller must not learn.
		writeError(w, http.StatusUnauthorized, "invalid admin password")
		return
	}
	token, err := r.adminAuth.issue()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name:     adminCookie,
		Value:    token,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		// Secure only when the request actually arrived over TLS. Setting it
		// unconditionally would make the cookie unusable on the plain-HTTP LAN
		// deployment this router still supports, which means no admin access at
		// all rather than slightly weaker admin access.
		Secure:  req.TLS != nil,
		Expires: time.Now().Add(adminSessionTTL),
		MaxAge:  int(adminSessionTTL / time.Second),
	})
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "expires_in": int(adminSessionTTL.Seconds())})
}

// handleAdminLogout invalidates the session server-side and clears the cookie.
func (r *Router) handleAdminLogout(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	if c, err := req.Cookie(adminCookie); err == nil {
		r.adminAuth.revoke(c.Value)
	}
	http.SetCookie(w, &http.Cookie{
		Name: adminCookie, Value: "", Path: "/",
		HttpOnly: true, SameSite: http.SameSiteLaxMode, Secure: req.TLS != nil, MaxAge: -1,
	})
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

// handleAdminSession reports whether the caller is currently an admin, so the
// dashboard can render a login form or the fleet without guessing from a 401.
func (r *Router) handleAdminSession(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"admin": r.adminAuthenticated(req)})
}

// handleAdminPassword changes the admin password. Requires the current session,
// and requires the current password as well: a session cookie left open on a
// shared machine should not be enough to lock the operator out of their own
// router.
func (r *Router) handleAdminPassword(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	if !r.requireAdmin(w, req) {
		return
	}
	var body struct {
		Current string `json:"current_password"`
		New     string `json:"new_password"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 4096)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: %s", err)
		return
	}
	stored, err := r.logs.LoadSetting(req.Context(), settingAdminPasswordHash)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	if stored == "" || !verifyPassword(stored, body.Current) {
		writeError(w, http.StatusUnauthorized, "current password is incorrect")
		return
	}
	if err := validPassword(body.New); err != nil {
		writeJSON(w, http.StatusBadRequest, validationError{Message: err.Error(), Param: "new_password"})
		return
	}
	hash, err := hashPassword(body.New)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	if err := r.logs.SaveSetting(req.Context(), settingAdminPasswordHash, hash); err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

// randomToken returns n bytes of crypto/rand as an unpadded base64url string.
func randomToken(n int) (string, error) {
	buf := make([]byte, n)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64URL.EncodeToString(buf), nil
}

// sha256Hex is the at-rest form of a bearer credential.
func sha256Hex(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

// constantTimeEqual is the comparison every credential check uses. Same
// discipline as authorizedAsClient: no early return on the first differing byte.
func constantTimeEqual(a, b string) bool {
	return subtle.ConstantTimeCompare([]byte(a), []byte(b)) == 1
}

// ── API keys ────────────────────────────────────────────────────────────────

// keySpec is the write shape for a key. Pointers again, and for the same reason
// as providerSpec: PATCHing token_budget to 0 removes a budget, which is a
// different instruction from not mentioning it.
type keySpec struct {
	Name        *string   `json:"name"`
	Role        *string   `json:"role"`
	Enabled     *bool     `json:"enabled"`
	Models      *[]string `json:"models"`
	TokenBudget *int64    `json:"token_budget"`
	Relay       *bool     `json:"relay"`
}

func (r *Router) handleAdminKeys(w http.ResponseWriter, req *http.Request) {
	if !r.requireAdmin(w, req) {
		return
	}
	switch req.Method {
	case http.MethodGet:
		keys, err := r.logs.ListAPIKeys(req.Context())
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
			return
		}
		// Environment tokens ride along as read-only rows so this page answers
		// "who can call this router" completely — see envCredential.
		writeJSON(w, http.StatusOK, map[string]any{
			"keys": keys, "env_tokens": envCredentials(r.cfg)})
	case http.MethodPost:
		r.createKey(w, req)
	default:
		methodNotAllowed(w)
	}
}

func (r *Router) handleAdminKeyByID(w http.ResponseWriter, req *http.Request) {
	if !r.requireAdmin(w, req) {
		return
	}
	raw := strings.Trim(strings.TrimPrefix(req.URL.Path, "/admin/keys/"), "/")
	if raw == "" {
		r.handleAdminKeys(w, req)
		return
	}
	id, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("no key %q", raw)})
		return
	}
	switch req.Method {
	case http.MethodPatch, http.MethodPut:
		r.updateKey(w, req, id)
	case http.MethodDelete:
		r.deleteKey(w, req, id)
	default:
		methodNotAllowed(w)
	}
}

// createKey mints a key and returns the plaintext. This response is the ONLY
// place it ever exists — the table holds a SHA-256 and the displayed prefix, so
// there is nothing to re-read it from and no endpoint that could.
func (r *Router) createKey(w http.ResponseWriter, req *http.Request) {
	var spec keySpec
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 1<<20)).Decode(&spec); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: %s", err)
		return
	}
	role := roleClient
	if spec.Role != nil {
		role = strings.ToLower(strings.TrimSpace(*spec.Role))
	}
	if !validRole(role) {
		writeJSON(w, http.StatusBadRequest, validationError{
			Message: fmt.Sprintf("role must be one of %s, %s, %s", roleAdmin, roleClient, roleWorker),
			Param:   "role",
		})
		return
	}
	var (
		name   string
		models []string
		budget int64
		relay  bool
	)
	assign(&name, spec.Name)
	assign(&budget, spec.TokenBudget)
	assign(&relay, spec.Relay)
	if spec.Models != nil {
		models = *spec.Models
	}
	if budget < 0 {
		writeJSON(w, http.StatusBadRequest, validationError{Message: "token_budget must not be negative", Param: "token_budget"})
		return
	}
	if err := validateRelayFlag(relay, role); err != nil {
		writeJSON(w, http.StatusBadRequest, *err)
		return
	}
	plain, key, err := newAPIKey(name, role, models, budget)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	key.Relay = relay
	stored, err := r.logs.CreateAPIKey(req.Context(), plain, key)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	r.refreshAuthRequired(req.Context())
	log.Printf("issued %s key %d (%q, prefix %s)", stored.Role, stored.ID, stored.Name, stored.Prefix)
	writeJSON(w, http.StatusCreated, map[string]any{
		"key":     stored,
		"secret":  plain,
		"warning": "this is the only time the secret is shown",
	})
}

func (r *Router) updateKey(w http.ResponseWriter, req *http.Request, id int64) {
	var spec keySpec
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 1<<20)).Decode(&spec); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: %s", err)
		return
	}
	keys, err := r.logs.ListAPIKeys(req.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	var current *apiKey
	for i := range keys {
		if keys[i].ID == id {
			current = &keys[i]
			break
		}
	}
	if current == nil {
		writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("no key %d", id)})
		return
	}
	// Role is immutable. Re-roling in place would change what an already-issued
	// credential can reach without the holder or the operator seeing a new key.
	if spec.Role != nil && !strings.EqualFold(*spec.Role, current.Role) {
		writeJSON(w, http.StatusConflict, validationError{
			Message: "a key's role cannot be changed; revoke it and issue another", Param: "role",
		})
		return
	}
	// Disabling is a revoke, so it reopens a gate exactly as a delete does.
	if spec.Enabled != nil && !*spec.Enabled && current.Enabled && !r.refuseIfLastCredential(w, keys, id, "disabling") {
		return
	}
	assign(&current.Name, spec.Name)
	assign(&current.Enabled, spec.Enabled)
	assign(&current.TokenBudget, spec.TokenBudget)
	assign(&current.Relay, spec.Relay)
	if spec.Models != nil {
		current.Models = normalizeModelList(*spec.Models)
	}
	if current.TokenBudget < 0 {
		writeJSON(w, http.StatusBadRequest, validationError{Message: "token_budget must not be negative", Param: "token_budget"})
		return
	}
	if err := validateRelayFlag(current.Relay, current.Role); err != nil {
		writeJSON(w, http.StatusBadRequest, *err)
		return
	}
	if err := r.logs.UpdateAPIKey(req.Context(), id, *current); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("no key %d", id)})
			return
		}
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	r.refreshAuthRequired(req.Context())
	writeJSON(w, http.StatusOK, map[string]any{"key": current})
}

// refuseIfLastCredential answers 409 and returns false when losing this key
// would leave an authentication gate requiring nothing at all.
//
// 409 rather than 403: nothing is wrong with the caller's authority — an admin
// may delete any key — the request conflicts with the state of the collection,
// which is exactly what a Conflict is for. The message names the replacement
// that makes the delete legal, because a refusal an operator cannot act on is a
// refusal they will work around.
func (r *Router) refuseIfLastCredential(w http.ResponseWriter, keys []apiKey, id int64, action string) bool {
	gates := r.reopenedGates(keys, id)
	if len(gates) == 0 {
		return true
	}
	names := make([]string, 0, len(gates))
	envs := make([]string, 0, len(gates))
	for _, g := range gates {
		names = append(names, g.name)
		envs = append(envs, g.envName)
	}
	writeJSON(w, http.StatusConflict, validationError{
		Message: fmt.Sprintf(
			"key %d is the last enabled credential for %s; %s it would leave that open to anyone who can reach this port. "+
				"Issue a replacement key first (POST /admin/keys), or set %s and restart.",
			id, strings.Join(names, " and "), action, strings.Join(envs, " / ")),
	})
	return false
}

func (r *Router) deleteKey(w http.ResponseWriter, req *http.Request, id int64) {
	keys, err := r.logs.ListAPIKeys(req.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	if !r.refuseIfLastCredential(w, keys, id, "deleting") {
		return
	}
	if err := r.logs.DeleteAPIKey(req.Context(), id); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("no key %d", id)})
			return
		}
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	r.refreshAuthRequired(req.Context())
	log.Printf("revoked api key %d", id)
	writeJSON(w, http.StatusOK, map[string]any{"status": "removed", "id": id})
}
