package router

import (
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

// The whole usage chart rests on one fact about the request log that nothing in
// the schema records: created_at is when a request STARTED, so its load belongs
// to the buckets that follow it. Read the other way round — as a completion
// time, which is the more common convention for a log table — every column on
// the chart shifts by one request duration. Nothing looks wrong when that
// happens; the graph is still smooth, still plausible, and still says the fleet
// was busy at a time it was idle. So it is pinned here.
func TestUsageSeriesTreatsCreatedAtAsTheStartOfTheRequest(t *testing.T) {
	logs := newTestLogStore(t)
	now := time.Date(2026, 3, 4, 15, 47, 13, 0, time.UTC)
	frame := newUsageSeries(now, usageWindow, usageBucket)

	// Twenty minutes is exactly four buckets, started on a bucket boundary.
	start := time.Unix(frame.Buckets[60].At, 0).UTC()
	insertUsageRow(t, logs, "10.0.0.1", start, 20*time.Minute)

	series := usageSeries(t, logs, now)
	for _, idx := range []int{60, 61, 62, 63} {
		if got := usageValue(t, series, idx, "10.0.0.1"); math.Abs(got-1) > 1e-6 {
			t.Errorf("bucket %d = %v slots, want 1: a request in flight for the whole bucket is one slot held", idx, got)
		}
	}
	if got := usageValue(t, series, 59, "10.0.0.1"); got != 0 {
		t.Errorf("bucket 59 = %v slots, want 0 — load is being drawn BEFORE created_at, "+
			"which is what reading it as a completion time looks like", got)
	}
	if got := usageValue(t, series, 64, "10.0.0.1"); got != 0 {
		t.Errorf("bucket 64 = %v slots, want 0: the request had finished", got)
	}
}

// Slots, not requests. A worker with max_concurrency 1 is fully occupied by one
// long generation and untouched by a burst of embedding calls, and a chart that
// counted requests would draw those the same way round the wrong way.
func TestUsageSeriesMeasuresConcurrencyNotRequestCount(t *testing.T) {
	logs := newTestLogStore(t)
	now := time.Date(2026, 3, 4, 15, 47, 13, 0, time.UTC)
	frame := newUsageSeries(now, usageWindow, usageBucket)
	slot := time.Unix(frame.Buckets[50].At, 0).UTC()

	// Two requests running side by side for half the bucket: 300 busy seconds in
	// a 300-second bucket, so a mean of one slot held.
	insertUsageRow(t, logs, "10.0.0.1", slot, 150*time.Second)
	insertUsageRow(t, logs, "10.0.0.1", slot, 150*time.Second)
	// Forty times as many requests from somewhere else, none of which occupied
	// anything for measurable time.
	for i := 0; i < 40; i++ {
		insertUsageRow(t, logs, "10.0.0.2", slot.Add(time.Duration(i)*time.Second), 0)
	}

	series := usageSeries(t, logs, now)
	if got := usageValue(t, series, 50, "10.0.0.1"); math.Abs(got-1) > 1e-6 {
		t.Errorf("two half-bucket requests = %v slots, want 1", got)
	}
	if got := usageValue(t, series, 50, "10.0.0.2"); got != 0 {
		t.Errorf("forty instant requests = %v slots, want 0", got)
	}
	if got := usageRequests(t, series, 50, "10.0.0.2"); got != 40 {
		t.Errorf("forty instant requests = %d in the request count, want 40 — they still happened", got)
	}
}

// A fresh column on an existing database. Every row already in the table has no
// address and never will: it was not recorded when they were written. Dropping
// them would make a router that has been serving for months read as idle for the
// first twelve hours after an upgrade.
func TestUsageSeriesDrawsRowsWithNoAddressAsUnknown(t *testing.T) {
	logs := newTestLogStore(t)
	now := time.Date(2026, 3, 4, 15, 47, 13, 0, time.UTC)
	frame := newUsageSeries(now, usageWindow, usageBucket)
	slot := time.Unix(frame.Buckets[100].At, 0).UTC()

	insertUsageRow(t, logs, "", slot, usageBucket)
	insertUsageRow(t, logs, "10.0.0.1", slot, usageBucket)

	series := usageSeries(t, logs, now)
	if got := usageValue(t, series, 100, usageUnknownClient); math.Abs(got-1) > 1e-6 {
		t.Errorf("addressless row = %v slots under %q, want 1", got, usageUnknownClient)
	}
	// And it stacks on top of the addresses rather than among them, so a band an
	// operator can act on never has an unattributable one moving its baseline.
	if last := series.Clients[len(series.Clients)-1].ClientIP; last != usageUnknownClient {
		t.Errorf("stacking order ends with %q, want %q last", last, usageUnknownClient)
	}
}

// Past the eighth address the palette has no ninth colour that survives a
// colour-blind check, so the tail folds into one band. It must fold, not vanish:
// the total height of a column is the fleet's load, and a chart that silently
// dropped the long tail would understate it.
func TestUsageSeriesFoldsTheTailOfClientsIntoOther(t *testing.T) {
	logs := newTestLogStore(t)
	now := time.Date(2026, 3, 4, 15, 47, 13, 0, time.UTC)
	frame := newUsageSeries(now, usageWindow, usageBucket)
	slot := time.Unix(frame.Buckets[10].At, 0).UTC()

	// Twelve addresses, each busier than the last, all in one bucket.
	for i := 1; i <= 12; i++ {
		insertUsageRow(t, logs, ipForIndex(i), slot, time.Duration(i)*10*time.Second)
	}
	series := usageSeries(t, logs, now)

	named, folded := 0, false
	for _, client := range series.Clients {
		switch client.ClientIP {
		case usageOtherClients:
			folded = true
		case usageUnknownClient:
			t.Errorf("an addressed row was filed under %q", usageUnknownClient)
		default:
			named++
		}
	}
	if named != usageTopClients {
		t.Errorf("%d addresses got their own band, want %d", named, usageTopClients)
	}
	if !folded {
		t.Fatalf("twelve addresses produced no %q band", usageOtherClients)
	}
	// The busiest are the ones that kept a band, and the total is preserved.
	if usageValue(t, series, 10, ipForIndex(12)) == 0 {
		t.Error("the busiest address was folded into the tail")
	}
	// 10+20+…+120 seconds of busy time in a 300-second bucket.
	var total float64
	for _, v := range series.Buckets[10].Slots {
		total += v
	}
	if want := 780.0 / 300.0; math.Abs(total-want) > 1e-3 {
		t.Errorf("column total = %v slots, want %v — folding lost load", total, want)
	}
}

// The empty case has to be a chart with nothing in it, not a chart with nothing
// to draw an axis from. A router that has served no requests in twelve hours is
// a thing an operator needs to be able to look at and believe.
func TestUsageSeriesOnAnEmptyTableIsStillAWholeFrame(t *testing.T) {
	logs := newTestLogStore(t)
	now := time.Date(2026, 3, 4, 15, 47, 13, 0, time.UTC)

	series := usageSeries(t, logs, now)
	if want := int(usageWindow / usageBucket); len(series.Buckets) != want {
		t.Fatalf("empty table produced %d buckets, want %d", len(series.Buckets), want)
	}
	if len(series.Clients) != 0 {
		t.Errorf("empty table produced %d bands", len(series.Clients))
	}
	if series.To-series.From != int64(usageWindow/time.Second) {
		t.Errorf("frame spans %ds, want %ds", series.To-series.From, int64(usageWindow/time.Second))
	}
	// Bucket starts are absolute multiples of the bucket, not offsets from the
	// moment of the poll — otherwise every column moves on every refresh.
	if series.From%series.BucketSeconds != 0 {
		t.Errorf("window starts at %d, which is not a multiple of the %ds bucket", series.From, series.BucketSeconds)
	}
	// And the frame ends AFTER now: the newest column is the one in progress.
	if series.To <= now.Unix() {
		t.Errorf("frame ends at %d, before now (%d) — the live bucket is missing", series.To, now.Unix())
	}
}

// Requests that started before the window still occupied a slot inside it.
func TestUsageSeriesClampsARequestThatStraddlesTheWindowEdge(t *testing.T) {
	logs := newTestLogStore(t)
	now := time.Date(2026, 3, 4, 15, 47, 13, 0, time.UTC)
	frame := newUsageSeries(now, usageWindow, usageBucket)

	// Starts five minutes before the window opens and runs for ten, so half of it
	// is on screen and fills the first bucket.
	insertUsageRow(t, logs, "10.0.0.1", time.Unix(frame.From, 0).UTC().Add(-usageBucket), 10*time.Minute)

	series := usageSeries(t, logs, now)
	if got := usageValue(t, series, 0, "10.0.0.1"); math.Abs(got-1) > 1e-6 {
		t.Errorf("first bucket = %v slots, want 1: the in-window half of the request", got)
	}
}

// X-Forwarded-For, because in production this router is behind Caddy and behind
// Cloudflare and RemoteAddr is the proxy — every caller in the fleet would stack
// into one band. Anything that is not an address is discarded rather than stored,
// which is what bounds what a forged header can put in the column.
func TestClientIPPrefersForwardedForAndKeepsOnlyAddresses(t *testing.T) {
	for _, tc := range []struct{ name, forwarded, remote, want string }{
		{"no header falls back to the peer", "", "192.0.2.7:51000", "192.0.2.7"},
		{"leftmost entry is the original client", "203.0.113.9, 70.41.3.18, 150.172.238.178", "192.0.2.7:51000", "203.0.113.9"},
		{"a single entry", "203.0.113.9", "192.0.2.7:51000", "203.0.113.9"},
		{"surrounding space", "  203.0.113.9 , 70.41.3.18", "192.0.2.7:51000", "203.0.113.9"},
		{"ipv6 keeps its colons", "", "[2001:db8::1]:51000", "2001:db8::1"},
		{"ipv6 forwarded", "2001:db8::2", "192.0.2.7:51000", "2001:db8::2"},
		{"a peer with no port", "", "192.0.2.7", "192.0.2.7"},
		{"junk in the header falls back", "not-an-address", "192.0.2.7:51000", "192.0.2.7"},
		{"an obfuscated entry falls back", "_hidden", "192.0.2.7:51000", "192.0.2.7"},
		{"a hostname is not an address", "proxy.internal", "192.0.2.7:51000", "192.0.2.7"},
		{"an empty leading entry falls back", ", 70.41.3.18", "192.0.2.7:51000", "192.0.2.7"},
		{"nothing readable at all", "", "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			req.RemoteAddr = tc.remote
			if tc.forwarded != "" {
				req.Header.Set("X-Forwarded-For", tc.forwarded)
			}
			if got := clientIP(req); got != tc.want {
				t.Errorf("clientIP = %q, want %q", got, tc.want)
			}
		})
	}
}

// A forged header must not be able to write an unbounded string into the column
// or a wall of text into the chart legend.
func TestClientIPRejectsAnOversizedForwardedHeader(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.RemoteAddr = "192.0.2.7:51000"
	req.Header.Set("X-Forwarded-For", strings.Repeat("A", 8192))
	if got := clientIP(req); got != "192.0.2.7" {
		t.Errorf("clientIP = %q (%d bytes), want the peer address", truncate(got, 40), len(got))
	}
}

// The cache is what keeps a ten-second poll off a single-connection database on
// a router with a lot of history. Two properties matter and neither is visible
// from the outside: a slow query has to buy a longer reprieve than a fast one,
// and no reprieve at all may outlive the window it was computed for.
func TestUsageCacheHoldsInProportionToWhatTheQueryCost(t *testing.T) {
	now := time.Date(2026, 3, 4, 15, 47, 13, 0, time.UTC)
	frame := newUsageSeries(now, usageWindow, usageBucket)

	var cache usageCache
	if cache.get(frame.To, now) != nil {
		t.Fatal("an empty cache answered")
	}

	// A quiet router: milliseconds, so the chart stays live at a ten-second poll.
	cache.put(frame, now, 20*time.Millisecond)
	if cache.get(frame.To, now.Add(usageCacheMin-time.Second)) == nil {
		t.Error("a cheap query bought no reprieve at all")
	}
	if cache.get(frame.To, now.Add(9*time.Second)) != nil {
		t.Error("a 20ms query is holding the chart for nine seconds; it should be live")
	}

	// A busy one: the chart visibly settles down rather than hammering SQLite.
	cache.put(frame, now, time.Second)
	if cache.get(frame.To, now.Add(15*time.Second)) == nil {
		t.Error("a one-second query is being re-run every poll; that is a fifth of the connection")
	}
	if cache.get(frame.To, now.Add(2*usageCacheMax)) != nil {
		t.Error("the cache has no ceiling")
	}

	// And a rolled-over window is never served from the previous one, however
	// recently it was computed. A chart pinned to a stale window is worse than a
	// slow one, because nothing on screen says it has stopped moving.
	cache.put(frame, now, time.Second)
	if cache.get(frame.To+frame.BucketSeconds, now) != nil {
		t.Error("the cache answered for a window it was not computed for")
	}
}

// The endpoint is admin scope, exactly like /logs. The set of addresses talking
// to a router is a map of who runs what and from where, and GET / is public.
func TestUsageEndpointIsAdminOnly(t *testing.T) {
	router := &Router{cfg: &Config{}, registry: newTestRegistry(), logs: newTestLogStore(t)}
	rec := httptest.NewRecorder()
	router.handleUsage(rec, httptest.NewRequest(http.MethodGet, "/admin/usage", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("unauthenticated GET /admin/usage returned %d, want 401", rec.Code)
	}
}

// The usage view is wired to elements and to an endpoint that no compiler reads.
// A renamed id is a blank panel at runtime; a call to a path the mux does not
// serve is answered by the catch-all dashboard handler in HTML.
func TestDashboardCarriesTheUsageView(t *testing.T) {
	body := renderDashboard(t)
	for _, anchor := range []string{
		`id="mtab-usage"`, `id="view-usage"`,
		`id="usage-chart"`, `id="usage-legend"`,
		"function loadUsage", "function drawUsage", "function seriesFill",
		"'/admin/usage'",
	} {
		if !strings.Contains(body, anchor) {
			t.Errorf("the dashboard has no %s — the usage view is wired to something that does not exist", anchor)
		}
	}
	mux := (&Router{cfg: &Config{}, registry: newTestRegistry()}).routes()
	if _, pattern := mux.Handler(httptest.NewRequest(http.MethodGet, "/admin/usage", nil)); pattern != "/admin/usage" {
		t.Errorf("/admin/usage matched pattern %q — the chart would be fetching the dashboard itself", pattern)
	}
}

// A band's colour has to follow the ADDRESS, not its position in the ranking.
// The ranking is over a sliding twelve hours and drifts without anyone touching
// anything; if the hue came from the rank, two clients would trade colours
// between two polls and a reader who had learned "the laptop is blue" would be
// quietly misled. The mechanism is a map that outlives one response — recompute
// it from each payload and the property is gone.
func TestDashboardAssignsChartColoursByAddressNotRank(t *testing.T) {
	body := renderDashboard(t)
	if !strings.Contains(body, "let usageFill = {}") {
		t.Error("seriesFill has no map outliving a single response; a rank-indexed palette repaints on every reorder")
	}
	// The eight validated dark-surface slots, in order. Changing one is a
	// deliberate act — they were checked as a set against this page's surface for
	// lightness band, chroma, contrast and colour-blind separation.
	for _, hex := range []string{
		"#3987e5", "#d95926", "#199e70", "#c98500",
		"#d55181", "#008300", "#9085e9", "#e66767",
	} {
		if !strings.Contains(body, hex) {
			t.Errorf("the categorical palette is missing %s", hex)
		}
	}
}

// ── helpers ─────────────────────────────────────────────────────────────────

func insertUsageRow(t *testing.T, logs *LogStore, ip string, start time.Time, took time.Duration) {
	t.Helper()
	if err := logs.Insert(t.Context(), RequestLog{
		CreatedAt:      start.UTC(),
		BackendID:      "worker",
		BackendModel:   "model",
		Route:          "auto",
		StatusCode:     http.StatusOK,
		DurationMillis: took.Milliseconds(),
		ClientIP:       ip,
	}); err != nil {
		t.Fatalf("insert usage row: %v", err)
	}
}

func usageSeries(t *testing.T, logs *LogStore, now time.Time) *UsageSeries {
	t.Helper()
	series, err := logs.UsageSeries(t.Context(), now, usageWindow, usageBucket, usageTopClients)
	if err != nil {
		t.Fatalf("UsageSeries: %v", err)
	}
	return series
}

func usageBand(t *testing.T, series *UsageSeries, client string) int {
	t.Helper()
	for i, c := range series.Clients {
		if c.ClientIP == client {
			return i
		}
	}
	return -1
}

func usageValue(t *testing.T, series *UsageSeries, bucket int, client string) float64 {
	t.Helper()
	band := usageBand(t, series, client)
	if band < 0 {
		return 0
	}
	return series.Buckets[bucket].Slots[band]
}

func usageRequests(t *testing.T, series *UsageSeries, bucket int, client string) int64 {
	t.Helper()
	band := usageBand(t, series, client)
	if band < 0 {
		return 0
	}
	return series.Buckets[bucket].Requests[band]
}

func ipForIndex(i int) string {
	return "10.0.0." + strconv.Itoa(i)
}
