package router

// Named groups: `coding`, `research`, an ordered list of members, and a
// resolver that runs ahead of backend selection.
//
// A group is a PREFERENCE, never a constraint. That is the whole design and
// everything below follows from it. Walk the members in order, take the first
// one that is registered, healthy and past the request's hard filters, and if
// none of them qualify drop the group filter entirely and route automatically.
// A group must never turn into a refusal: an operator who lists a model they
// have not stood up yet, or whose only member is mid-restart, gets an answer
// from somewhere rather than a 404 — with X-LLM-Group: fallback saying so.
//
// PRECEDENCE. A client selects a group through the standard `model` field, the
// same field that already accepts a model id, an alias and a worker id, so the
// four spellings have to be ordered. Groups win:
//
//	group name → model id → alias → worker id
//
// and the admin API refuses to create a group whose name is already servable as
// any of the other three. The two can still meet, in one direction: a worker
// registered LATER can arrive carrying a group's name. The group keeps the name
// then too, which is the safe direction — a name an operator deliberately
// created cannot be taken away from them by a deployment they may not control —
// and /v1/models publishes the group in that row so the menu says where the
// name actually goes. The worker stays reachable by its own id.
//
// Membership is by NAME and resolved per request. A member that is not
// currently registered is skipped rather than rejected, so the list is a
// statement of preference over time and not a foreign key.
//
// Groups also fix a display wrinkle. Two workers running different builds of
// the same family have different raw model ids that reduce to the same alias,
// which modelCatalogue then suppresses as ambiguous even though routing pools
// them correctly. A group over both restores the readable name.
//
// One thing worth knowing: a per-key model allow-list is checked against what
// the CLIENT asked for, so a key restricted to specific models needs the group
// name in its list, not the members'. That is the right way round — the group
// is what the client named.

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// routerOwner is the `owned_by` a /v1/models entry carries when the router owns
// it rather than a worker: the automatic "default" route, and every group.
//
// Still the old name after the rename, deliberately. This is a wire value that
// deployed clients already filter on to tell a router-owned menu entry from a
// worker's, and nothing is gained by breaking them to match the box.
const routerOwner = "llm-router"

// Group is a named, ordered preference list. Members are model ids, aliases or
// worker ids, stored as the operator typed them — those spellings are matched
// case-sensitively by backendServesModel, so normalising them would break the
// match. The NAME is lowercased, because it is a name the router itself owns
// and clients should not have to guess its capitalisation.
type Group struct {
	Name      string    `json:"name"`
	Members   []string  `json:"members"`
	UpdatedAt time.Time `json:"updated_at,omitempty"`
}

// groupKey is the lookup form of a group name.
func groupKey(name string) string { return strings.ToLower(strings.TrimSpace(name)) }

// resolve walks the members in order and returns the first that something in
// candidates can currently serve. ok=false is the fallback case: no member
// qualified, so the caller drops the filter and routes automatically.
//
// It resolves through admitReason rather than a test of its own, for the same
// reason /v1/route-preview shares planRoute: a resolver with its own idea of
// "qualifies" eventually picks a member selection then rejects, and the request
// falls back for a reason nobody can see. candidates is already the eligible
// set, so "registered and healthy" is covered before this is called.
func (g Group) resolve(candidates []*Backend, hf hardFilter) (string, bool) {
	for _, member := range g.Members {
		probe := hf
		probe.wantModel = member
		for _, b := range candidates {
			if admitReason(b, probe) == "" {
				return member, true
			}
		}
	}
	return "", false
}

// groupRoute is what group resolution decided about one request, threaded onto
// the route plan so the proxy and the preview both report the same thing.
type groupRoute struct {
	name     string // the group the client named; "" when they named none
	member   string // the member it resolved to
	fallback bool   // no member qualified, so the filter was dropped
}

// header is the X-LLM-Group value: which member served, or "fallback". Empty
// when no group was involved, and then no header is set at all.
func (g groupRoute) header() string {
	switch {
	case g.name == "":
		return ""
	case g.fallback:
		return "fallback"
	default:
		return g.member
	}
}

// ── Store ───────────────────────────────────────────────────────────────────

// groupStore holds the groups in memory so resolution is a map read rather than
// a database round trip per request. The database is canonical across restarts;
// this is the copy routing uses.
//
// The zero value works — the map is created on first write, under the lock —
// for the same reason adminSessions does: every test that builds a Router by
// hand would otherwise have to remember to construct one, and forgetting would
// be a nil dereference on the routing path rather than a compile error.
type groupStore struct {
	mu     sync.RWMutex
	byName map[string]Group
}

func (s *groupStore) lookup(name string) (Group, bool) {
	key := groupKey(name)
	if key == "" {
		return Group{}, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	g, ok := s.byName[key]
	return g, ok
}

func (s *groupStore) put(g Group) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.byName == nil {
		s.byName = map[string]Group{}
	}
	s.byName[groupKey(g.Name)] = g
}

func (s *groupStore) remove(name string) bool {
	key := groupKey(name)
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.byName[key]; !ok {
		return false
	}
	delete(s.byName, key)
	return true
}

// list returns the groups name-sorted, so the admin API and the model menu are
// stable orderings rather than Go's randomised map order.
func (s *groupStore) list() []Group {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Group, 0, len(s.byName))
	for _, g := range s.byName {
		out = append(out, g)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// ── Persistence ─────────────────────────────────────────────────────────────
//
// Members are stored as a JSON array rather than a joined string: a member is a
// model id, and model ids contain commas about as rarely as they contain
// slashes — which is to say, until one does.

func (s *LogStore) SaveGroup(ctx context.Context, g Group) error {
	members, err := json.Marshal(g.Members)
	if err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, `INSERT INTO router_groups (name, members, updated_at)
		VALUES (?, ?, ?)
		ON CONFLICT(name) DO UPDATE SET members=excluded.members, updated_at=excluded.updated_at`,
		g.Name, string(members), g.UpdatedAt.UTC().Format(time.RFC3339Nano))
	return err
}

func (s *LogStore) LoadGroups(ctx context.Context) ([]Group, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT name, members, updated_at FROM router_groups ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Group{}
	for rows.Next() {
		var g Group
		var members, updated string
		if err := rows.Scan(&g.Name, &members, &updated); err != nil {
			return nil, err
		}
		if err := json.Unmarshal([]byte(members), &g.Members); err != nil {
			return nil, fmt.Errorf("group %q has unreadable members: %w", g.Name, err)
		}
		g.UpdatedAt, _ = time.Parse(time.RFC3339Nano, updated)
		out = append(out, g)
	}
	return out, rows.Err()
}

func (s *LogStore) DeleteGroup(ctx context.Context, name string) error {
	res, err := s.db.ExecContext(ctx, `DELETE FROM router_groups WHERE name = ?`, groupKey(name))
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// loadGroups populates the in-memory store from the database at startup.
func (r *Router) loadGroups(ctx context.Context) {
	saved, err := r.logs.LoadGroups(ctx)
	if err != nil {
		log.Printf("load persisted groups failed: %v", err)
		return
	}
	for _, g := range saved {
		r.groups.put(g)
	}
	if len(saved) > 0 {
		log.Printf("loaded %d model group(s)", len(saved))
	}
}

// ── The model menu ──────────────────────────────────────────────────────────

// groupEntries renders the groups for /v1/models, owned by the router because
// that is what they are: no endpoint knows a group exists.
//
// The advertised features are the INTERSECTION over the members that are
// currently registered, because the resolver may pick any of them. A group with
// no registered member advertises the fleet's own features instead — it falls
// back to automatic routing, which can reach anything the fleet can.
func groupEntries(groups []Group, servable []*Backend, fleetFeatures []string) []map[string]any {
	out := make([]map[string]any, 0, len(groups))
	for _, g := range groups {
		features, found := fleetFeatures, false
		for _, member := range g.Members {
			for _, b := range servable {
				if !backendServesModel(b, member) {
					continue
				}
				if !found {
					features, found = append([]string(nil), b.Features...), true
				} else {
					features = intersectFeatures(features, b.Features)
				}
			}
		}
		out = append(out, map[string]any{
			"id":       g.Name,
			"object":   "model",
			"owned_by": routerOwner,
			"group":    true,
			"members":  append([]string(nil), g.Members...),
			"features": features,
		})
	}
	return out
}

// ── Admin CRUD ──────────────────────────────────────────────────────────────

// groupSpec is the write shape. Pointers for the same reason providerSpec uses
// them: "absent" and "empty" are different instructions, and an empty member
// list is one the validator has to be able to see and refuse.
type groupSpec struct {
	Name    *string   `json:"name"`
	Members *[]string `json:"members"`
}

func (r *Router) handleAdminGroups(w http.ResponseWriter, req *http.Request) {
	if !r.requireAdmin(w, req) {
		return
	}
	switch req.Method {
	case http.MethodGet:
		groups := r.groups.list()
		writeJSON(w, http.StatusOK, map[string]any{
			"groups":   groups,
			"resolves": r.memberResolution(groups),
		})
	case http.MethodPost:
		r.createGroup(w, req)
	default:
		methodNotAllowed(w)
	}
}

func (r *Router) handleAdminGroupByName(w http.ResponseWriter, req *http.Request) {
	if !r.requireAdmin(w, req) {
		return
	}
	name := strings.Trim(strings.TrimPrefix(req.URL.Path, "/admin/groups/"), "/")
	if name == "" {
		r.handleAdminGroups(w, req)
		return
	}
	existing, ok := r.groups.lookup(name)
	if !ok {
		writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("group %q not found", name)})
		return
	}
	switch req.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, existing)
	case http.MethodPatch, http.MethodPut:
		r.updateGroup(w, req, existing)
	case http.MethodDelete:
		r.deleteGroup(w, req, existing)
	default:
		methodNotAllowed(w)
	}
}

// memberResolution reports which members of each group the fleet can currently
// serve: group name → member → the id of a backend serving that name, or "" for
// none.
//
// An unresolved member is LEGITIMATE, and that is the whole reason this is
// reported rather than validated. Membership is by name and resolved per
// request, so a model an operator has not stood up yet, or one whose only worker
// is mid-restart, is a statement of preference over time and not a broken
// reference — createGroup deliberately does not reject it. The admin page has to
// be able to show the difference, and it cannot work it out for itself: an alias
// is a server-side reduction of a raw model id (see backendAlias), so a page
// comparing strings would report every aliased member as missing.
//
// It answers about REGISTRATION only. Group.resolve additionally applies the
// request's hard filters, which depend on a request that has not arrived yet; a
// resolved member here is one the fleet serves, not a promise about the next
// call. The backend id is returned rather than a boolean so the caller can pair
// it with that backend's live status.
func (r *Router) memberResolution(groups []Group) map[string]map[string]string {
	live := []*Backend{}
	for _, b := range r.registry.snapshot() {
		if !isExpired(b) {
			live = append(live, b)
		}
	}
	out := map[string]map[string]string{}
	for _, g := range groups {
		members := map[string]string{}
		for _, m := range g.Members {
			members[m] = ""
			for _, b := range live {
				if backendServesModel(b, m) {
					members[m] = b.ID
					break
				}
			}
		}
		out[g.Name] = members
	}
	return out
}

func (r *Router) createGroup(w http.ResponseWriter, req *http.Request) {
	spec, err := decodeGroupSpec(w, req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: %s", err)
		return
	}
	var g Group
	if spec.Name != nil {
		g.Name = *spec.Name
	}
	if spec.Members != nil {
		g.Members = *spec.Members
	}
	if bad := normalizeGroup(&g); bad != nil {
		writeJSON(w, http.StatusBadRequest, *bad)
		return
	}
	if _, exists := r.groups.lookup(g.Name); exists {
		writeJSON(w, http.StatusConflict, validationError{
			Message: fmt.Sprintf("group %q already exists", g.Name), Param: "name",
		})
		return
	}
	if shadowed := r.groupShadows(g.Name); shadowed != "" {
		writeJSON(w, http.StatusConflict, validationError{
			Message: fmt.Sprintf("%q is already servable as %s; a group resolves ahead of it and would take the name over", g.Name, shadowed),
			Param:   "name",
		})
		return
	}
	r.saveGroup(w, req, g, nil, http.StatusCreated)
}

func (r *Router) updateGroup(w http.ResponseWriter, req *http.Request, existing Group) {
	spec, err := decodeGroupSpec(w, req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: %s", err)
		return
	}
	// The path owns the identity; renaming is a delete plus a create, same rule
	// as a provider row. A rename in place would silently move every client that
	// names the old one onto automatic routing.
	if spec.Name != nil && groupKey(*spec.Name) != existing.Name {
		writeJSON(w, http.StatusConflict, validationError{
			Message: "a group cannot be renamed; delete it and create the new one", Param: "name",
		})
		return
	}
	updated := existing
	if spec.Members != nil {
		updated.Members = *spec.Members
	}
	if bad := normalizeGroup(&updated); bad != nil {
		writeJSON(w, http.StatusBadRequest, *bad)
		return
	}
	r.saveGroup(w, req, updated, &existing, http.StatusOK)
}

func (r *Router) deleteGroup(w http.ResponseWriter, req *http.Request, existing Group) {
	r.groups.remove(existing.Name)
	if err := r.logs.DeleteGroup(req.Context(), existing.Name); err != nil {
		log.Printf("delete persisted group %q: %v", existing.Name, err)
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "removed", "name": existing.Name})
}

// saveGroup commits a group to memory and to the database, undoing the memory
// write if the database one fails. Nothing re-posts a group the way a beacon
// re-posts its registration, so a silent persistence failure means it vanishes
// on the next restart — an error the operator has to see rather than a log line.
func (r *Router) saveGroup(w http.ResponseWriter, req *http.Request, g Group, prev *Group, status int) {
	g.UpdatedAt = time.Now().UTC()
	r.groups.put(g)
	if err := r.logs.SaveGroup(req.Context(), g); err != nil {
		if prev != nil {
			r.groups.put(*prev)
		} else {
			r.groups.remove(g.Name)
		}
		writeJSON(w, http.StatusInternalServerError, validationError{Message: fmt.Sprintf("persist group: %s", err)})
		return
	}
	writeJSON(w, status, map[string]any{"group": g})
}

func decodeGroupSpec(w http.ResponseWriter, req *http.Request) (groupSpec, error) {
	var spec groupSpec
	err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 1<<20)).Decode(&spec)
	return spec, err
}

// normalizeGroup settles a group's stored form and reports why it cannot be
// stored, or nil. Members keep the operator's own order and spelling — the
// order IS the preference — with blanks and duplicates dropped.
func normalizeGroup(g *Group) *validationError {
	g.Name = groupKey(g.Name)
	if g.Name == "" {
		return &validationError{Message: "name is required", Param: "name"}
	}
	g.Members = normalizeModelList(g.Members)
	if len(g.Members) == 0 {
		return &validationError{Message: "a group needs at least one member", Param: "members"}
	}
	return nil
}

// groupShadows reports what a proposed group name would take over, or "".
//
// Groups resolve ahead of model ids, aliases and worker ids, so a name that
// collides with one of those silently redirects every request that already used
// it. Refusing the write is the only point at which that is cheap to fix; by
// the time a client notices, the group is in someone's configuration.
//
// Case-insensitively, because the lookup is: a group named "gpt-4o" would
// otherwise still capture a request naming the model "GPT-4o".
func (r *Router) groupShadows(name string) string {
	key := groupKey(name)
	if autoModelNames[key] {
		return "the automatic route"
	}
	if shadowed := expertShadow(key); shadowed != "" {
		return shadowed
	}
	for _, b := range r.registry.snapshot() {
		if isExpired(b) {
			continue
		}
		for _, spelling := range []string{b.Model, b.ID, backendAlias(b)} {
			if spelling != "" && strings.EqualFold(spelling, key) {
				return fmt.Sprintf("backend %q (model %q)", b.ID, b.Model)
			}
		}
	}
	return ""
}
