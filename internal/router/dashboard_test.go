package router

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The dashboard markup moved out of main.go into an embedded file. Two things
// can silently break that: the Dockerfile copying only *.go (it did), and the
// template losing an action during an edit. A build failure catches a missing
// file, but only at deploy time — this catches both here.
func TestDashboardTemplateRenders(t *testing.T) {
	var buf bytes.Buffer
	err := dashboardTemplate.Execute(&buf, map[string]any{
		"Backends": []*Backend{},
		"Title":    "test",
	})
	if err != nil {
		t.Fatalf("dashboard template failed to execute: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "<!doctype html>") {
		t.Error("rendered dashboard is not HTML")
	}
	if len(out) < 1000 {
		t.Errorf("rendered dashboard is suspiciously short (%d bytes)", len(out))
	}
}

func TestDockerfileCopiesEmbeddedAssets(t *testing.T) {
	// The Dockerfile lives at the repo root, two levels up from this package.
	df, err := os.ReadFile(filepath.Join("..", "..", "Dockerfile"))
	if err != nil {
		t.Skip("no Dockerfile at the repo root")
	}
	// A COPY of the whole package directory carries dashboard.html with it;
	// a narrower COPY has to name the file. Either satisfies go:embed.
	if !bytes.Contains(df, []byte("dashboard.html")) && !bytes.Contains(df, []byte("COPY . .")) {
		t.Error("Dockerfile copies neither dashboard.html nor the whole tree — " +
			"go:embed will fail the image build while `go build` here still succeeds")
	}
}

// GET / is unauthenticated by design — the page is a static shell and the fleet
// table is populated client-side from an admin-gated /backends fetch. That
// invariant lives in a comment above handleDashboard and is enforced by nothing:
// adding "Backends": registry.snapshot() to the template data to "save a
// round-trip" would publish every worker's id, URL and model to anyone who
// loads the page. This test is the enforcement.
func TestUnauthenticatedDashboardDisclosesNoFleetDetail(t *testing.T) {
	registry := &Registry{backends: map[string]*Backend{}}
	registry.backends["secret-worker"] = &Backend{
		BackendRegistration: BackendRegistration{
			ID:    "secret-worker",
			URL:   "http://192.0.2.77:9999",
			Model: "internal-model-name",
		},
		Status: "ready",
	}
	router := &Router{cfg: &Config{}, registry: registry}
	// The same rule holds for everything P6 added a tab for. A group name, a
	// provider row and a key prefix are all configuration an unauthenticated
	// caller has no business reading, and all three are one convenient template
	// field away from being published here.
	router.groups.put(Group{Name: "secret-group", Members: []string{"secret-member"}})

	rec := httptest.NewRecorder()
	router.handleDashboard(rec, httptest.NewRequest(http.MethodGet, "/", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("dashboard returned %d", rec.Code)
	}
	body := rec.Body.String()
	for _, secret := range []string{
		"secret-worker", "192.0.2.77", "internal-model-name",
		"secret-group", "secret-member",
	} {
		if strings.Contains(body, secret) {
			t.Errorf("unauthenticated dashboard leaked %q — every table on this page must "+
				"be fetched client-side behind the admin gate, not server-rendered", secret)
		}
	}
}

// The page is five tabs over the admin API, and each one is a table nobody sees
// fail: a renamed element id breaks a view silently, at runtime, in a file no
// compiler reads. These are the anchors every view is wired to.
func TestDashboardShellCarriesEveryAdminView(t *testing.T) {
	body := renderDashboard(t)
	for _, anchor := range []string{
		// The tab bar and its six panels.
		`id="mtab-fleet"`, `id="view-fleet"`,
		`id="mtab-providers"`, `id="view-providers"`,
		`id="mtab-keys"`, `id="view-keys"`,
		`id="mtab-groups"`, `id="view-groups"`,
		`id="mtab-relays"`, `id="view-relays"`,
		`id="mtab-logs"`, `id="view-logs"`,
		// The tables each view fills client-side.
		`id="backends-body"`, `id="providers-body"`, `id="keys-body"`,
		`id="groups-body"`, `id="relays-body"`, `id="logs-body"`,
		// The password session: a login form, a visible way out, and a way to
		// change the password without a redeploy.
		`id="login-form"`, `id="login-password"`, `id="btn-logout"`, `id="btn-password"`,
	} {
		if !strings.Contains(body, anchor) {
			t.Errorf("the dashboard shell has no %s — a view is wired to an element that does not exist", anchor)
		}
	}
}

// Every endpoint the page calls has to be a route this mux actually serves.
// Getting it wrong is not a compile error; it is a tab that answers 404 in HTML
// through the catch-all dashboard handler, which is the exact failure the
// /v1/models pattern bug produced.
func TestDashboardCallsOnlyRoutesTheMuxServes(t *testing.T) {
	body := renderDashboard(t)
	calls := regexp.MustCompile(`(?:request|fetch)\('(/[^']*)'`).FindAllStringSubmatch(body, -1)
	if len(calls) < 10 {
		t.Fatalf("found %d endpoint calls in the dashboard — the extraction has stopped matching", len(calls))
	}
	mux := (&Router{cfg: &Config{}, registry: newTestRegistry()}).routes()
	seen := map[string]bool{}
	for _, call := range calls {
		path, _, _ := strings.Cut(call[1], "?")
		if seen[path] {
			continue
		}
		seen[path] = true
		_, pattern := mux.Handler(httptest.NewRequest(http.MethodGet, path, nil))
		// "/" is the dashboard's own catch-all: matching only that means nothing
		// more specific is registered, so the call falls through to this page.
		if pattern == "" || pattern == "/" {
			t.Errorf("the dashboard calls %q, which no route serves (matched pattern %q)", path, pattern)
		}
	}
	// And the reverse direction: a tab whose endpoint was dropped from the page.
	for _, want := range []string{
		"/admin/session", "/admin/login", "/admin/logout", "/admin/password",
		"/admin/providers", "/admin/keys", "/admin/groups", "/admin/relays",
		"/backends", "/logs",
	} {
		if !seen[want] && !seen[want+"/"] {
			t.Errorf("nothing on the dashboard calls %s", want)
		}
	}
}

// The page reads what a cold profile cost from the benchmark endpoint, and has
// to render "not measured" and "free" differently: zero tokens means the run was
// never metered, and a confident zero on it would be a lie about money.
func TestDashboardDistinguishesUnmeasuredProfileCostFromFree(t *testing.T) {
	body := renderDashboard(t)
	for _, needle := range []string{"profile_cost_measured", "not measured", "'free'"} {
		if !strings.Contains(body, needle) {
			t.Errorf("the profile cost cell does not mention %q", needle)
		}
	}
}

// The fleet table's per-category benchmark view. Two things here are contracts
// with the Go side that no compiler checks: the field names it reads out of
// GET /backends/{id}/benchmark, and — the one that matters — the difference
// between a no-think score that was never stored and one that is zero. A worker
// really can score 0 with thinking off, so a page that rendered both as "0%"
// would be inventing a measurement.
func TestDashboardRendersTheBenchmarkCategoryBreakdown(t *testing.T) {
	body := renderDashboard(t)
	for _, needle := range []string{
		"function openBenchmarkDialog", "function benchCategoryTable", "function benchGapCell",
		"'Benchmark'",            // the button that opens it, on every fleet row
		"nothink_results_stored", // the flag that says which kind of missing it is
		"quality_nothink_detail", // the per-tier line, shown when the split cannot be
		"'not stored'",           // …and what an absent per-category score renders as
		"by difficulty tier",     // the tier axis, kept and folded away
	} {
		if !strings.Contains(body, needle) {
			t.Errorf("the benchmark category view is missing %q", needle)
		}
	}
	// Both scores in the fleet table itself. The gap between them decides which
	// worker should get a no-think request, so it does not belong behind a click.
	if !strings.Contains(body, "quality_nothink") || !strings.Contains(body, "no-think not measured") {
		t.Error("the fleet table does not show the no-think quality beside the headline one")
	}
}

// A log row's bodies are rendered as a conversation rather than as raw JSON,
// and the pieces that has to cover are the ones a fleet actually produces:
// both reply shapes (a buffered completion and a captured event stream), the
// separation of reasoning from the answer, tool calls, and the router's own
// truncation markers.
func TestDashboardRendersLogBodiesAsATranscript(t *testing.T) {
	body := renderDashboard(t)
	for _, needle := range []string{
		"function renderChatRequest", "function renderCompletion", "function sseAnswer",
		"tx-system", "tx-user", "tx-assistant", "tx-tool", "tx-think",
		"reasoning_content", // the other dialect's spelling, or thinking renders as nothing
	} {
		if !strings.Contains(body, needle) {
			t.Errorf("the log transcript renderer is missing %q", needle)
		}
	}
	// The raw bytes stay reachable. This renderer makes judgements about what
	// matters and the stored body is the thing of record, so a formatting bug
	// must never be the reason an operator cannot see what was actually sent.
	if !strings.Contains(body, "function pane(") || !strings.Contains(body, "'Formatted'") {
		t.Error("the log detail has no raw/formatted toggle; a body the renderer cannot parse would be unreachable")
	}
}

// The log viewer strips the router's own truncation markers out of a body
// before parsing it and shows them as a notice instead. That makes the marker
// text a contract between Go and a regex in a browser, which no compiler checks
// and no reader would think to look for: change the wording on the Go side and
// the page silently starts feeding "…[capture truncated: 8291 bytes omitted]…"
// to JSON.parse, which fails, and every streamed reply falls back to raw.
func TestDashboardTruncationPatternMatchesWhatTheRouterWrites(t *testing.T) {
	// Lifted from the page verbatim, so the two cannot drift apart unnoticed.
	const pattern = `…\[(capture )?truncated[^\]]*\]…?`
	if !strings.Contains(renderDashboard(t), pattern) {
		t.Fatalf("the log viewer's truncation pattern is no longer %q — update this test with the new one", pattern)
	}
	re := regexp.MustCompile(pattern)

	// Marker one: the insert-time clip in clipLog.
	if got := clipLog(strings.Repeat("x", 200), 32); !re.MatchString(got) {
		t.Errorf("clipLog writes %q, which the log viewer will not recognise", got)
	}
	// Marker two: the head-and-tail join in boundedCapture.
	cap := newBoundedCapture(4096)
	if _, err := cap.Write(bytes.Repeat([]byte("y"), 64<<10)); err != nil {
		t.Fatalf("boundedCapture write: %v", err)
	}
	if got := string(cap.Bytes()); !re.MatchString(got) {
		t.Errorf("boundedCapture writes a marker the log viewer will not recognise: %q", truncate(got, 120))
	}
}

// Prompts and model output are the most attacker-influenced strings this page
// shows: they arrive from any client key, are stored verbatim, and are rendered
// to an admin. They must reach the DOM as text nodes, never spliced into an
// HTML string — one forgotten escape there is script in an admin session.
func TestDashboardBuildsLogBodiesAsTextNodes(t *testing.T) {
	body := renderDashboard(t)
	start := strings.Index(body, "function showLogDetail")
	if start < 0 {
		t.Fatal("showLogDetail is gone; this guard no longer guards anything")
	}
	end := strings.Index(body[start:], "\n    function closeLogDetail")
	if end < 0 {
		t.Fatal("cannot find the end of showLogDetail")
	}
	// innerHTML = '' is a clear, which is fine. Assigning anything else there
	// would be building markup out of a stored prompt.
	for _, line := range strings.Split(body[start:start+end], "\n") {
		if !strings.Contains(line, "innerHTML") {
			continue
		}
		if !strings.Contains(line, "innerHTML = ''") {
			t.Errorf("showLogDetail assigns markup rather than text: %s", strings.TrimSpace(line))
		}
	}
}

// The bearer-token prompt is gone: admin is a password session, the cookie is
// HttpOnly, and a page that also kept a copy of an admin key in sessionStorage
// would be handing a live credential to anything that ever ran script on this
// origin. removeItem stays — it clears what the previous version stored.
func TestDashboardHoldsNoBearerCredential(t *testing.T) {
	body := renderDashboard(t)
	for _, banned := range []string{"sessionStorage.getItem", "sessionStorage.setItem", "Bearer '"} {
		if strings.Contains(body, banned) {
			t.Errorf("the dashboard still handles a bearer credential (%q); admin is a session cookie", banned)
		}
	}
	if !strings.Contains(body, "credentials: 'same-origin'") {
		t.Error("the dashboard does not send the session cookie with its admin calls")
	}
}

func renderDashboard(t *testing.T) string {
	t.Helper()
	rec := httptest.NewRecorder()
	router := &Router{cfg: &Config{}, registry: newTestRegistry()}
	router.handleDashboard(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("dashboard returned %d", rec.Code)
	}
	return rec.Body.String()
}
