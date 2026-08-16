package router

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
// table is populated client-side from a token-gated /backends fetch. That
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

	rec := httptest.NewRecorder()
	router.handleDashboard(rec, httptest.NewRequest(http.MethodGet, "/", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("dashboard returned %d", rec.Code)
	}
	body := rec.Body.String()
	for _, secret := range []string{
		"secret-worker", "192.0.2.77", "internal-model-name",
	} {
		if strings.Contains(body, secret) {
			t.Errorf("unauthenticated dashboard leaked %q — the fleet table must "+
				"be fetched client-side with a token, not server-rendered", secret)
		}
	}
}
