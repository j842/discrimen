package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// answeringWorker is a chat backend that answers from a lookup table, so a test
// can decide which worker gets which question right — the shape the oracle pass
// and the optimality metric are built to detect.
func answeringWorker(t *testing.T, answers map[string]string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(req.Body).Decode(&body)
		prompt := ""
		if n := len(body.Messages); n > 0 {
			prompt = body.Messages[n-1].Content
		}
		answer := answers[prompt]
		if answer == "" {
			answer = "I don't know"
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message":       map[string]any{"role": "assistant", "content": answer},
				"finish_reason": "stop",
			}},
		})
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestArenaEndToEnd runs the real harness against a real router over HTTP: the
// dataset loader, /v1/route-preview, /v1/chat/completions, the pinned oracle
// pass, the production grader and the metrics. Nothing here is stubbed except
// the workers themselves.
func TestArenaEndToEnd(t *testing.T) {
	const qEasy = "What is 2+2?"
	const qHard = "prove that there are infinitely many primes"

	// tiny gets the easy one right and the hard one wrong; big gets both.
	tiny := answeringWorker(t, map[string]string{qEasy: "the answer is 4"})
	big := answeringWorker(t, map[string]string{
		qEasy: "the answer is 4",
		qHard: "the answer is Euclid",
	})

	reg := newTestRegistry()
	for _, w := range []struct {
		id      string
		url     string
		quality int
		tps     int
	}{{"tiny", tiny.URL, 20, 200}, {"big", big.URL, 90, 140}} {
		reg.upsert(BackendRegistration{
			ID: w.id, URL: w.url, Model: "default", Quality: w.quality,
			BaselineTPS: float64(w.tps), MaxConcurrency: 4, TTLSeconds: 3600,
			Thinking: true, Features: []string{"chat"},
		})
		reg.finishCertification(w.id, true, map[string]Check{}, float64(w.tps), 10, "")
	}

	dir := t.TempDir()
	logs, err := openLogStore(filepath.Join(dir, "logs.sqlite"), 16384, "")
	if err != nil {
		t.Fatal(err)
	}
	defer logs.Close()

	cfg := &Config{
		DefaultMaxTokens: 4096, AutoDifficulty: true, AutoThinking: true,
		DifficultyTemp: 0.10, ReasoningThreshold: 0.35,
		DifficultyTimeout: time.Second, DifficultyCacheSize: 64, DifficultyMaxChars: 4000,
	}
	router := &Router{
		cfg: cfg, registry: reg, classifier: testClassifierAuto(fakeEmbed), logs: logs,
		client: &http.Client{Timeout: 5 * time.Second}, streamClient: &http.Client{},
		sessions: newSessionTracker(time.Hour, 64),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/backends", router.handleBackends)
	mux.HandleFunc("/v1/chat/completions", router.handleChatCompletions)
	mux.HandleFunc("/v1/route-preview", router.handleRoutePreview)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	dataset := filepath.Join(dir, "arena.jsonl")
	lines := []string{
		`{"id":"easy1","question":` + jsonStr(qEasy) + `,"answer":"4","domain":"maths","difficulty":"easy"}`,
		`{"id":"hard1","question":` + jsonStr(qHard) + `,"answer":"Euclid","domain":"maths","difficulty":"hard"}`,
	}
	if err := os.WriteFile(dataset, []byte(strings.Join(lines, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}

	out := filepath.Join(dir, "results.json")
	err = arenaRun(arenaConfig{
		router: srv.URL, dataset: dataset, out: out,
		concurrency: 2, maxTokens: 512, deadline: 10 * time.Second,
		oracle: true, robustness: true,
	})
	if err != nil {
		t.Fatalf("arena run: %v", err)
	}

	blob, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var res arenaResults
	if err := json.Unmarshal(blob, &res); err != nil {
		t.Fatal(err)
	}
	if len(res.Outcomes) != 2 || len(res.Workers) != 2 {
		t.Fatalf("want 2 outcomes across 2 workers, got %d/%d", len(res.Outcomes), len(res.Workers))
	}

	byID := map[string]arenaOutcome{}
	for _, o := range res.Outcomes {
		byID[o.Question.ID] = o
	}

	easy, hard := byID["easy1"], byID["hard1"]
	if !easy.Pass {
		t.Fatalf("easy question should be answered correctly: %+v", easy)
	}
	if !hard.Pass {
		t.Fatalf("hard question should route high enough to be answered: served by %s, answer %q",
			hard.BackendID, hard.Answer)
	}
	// The classifier should separate them, and the preview should have said so
	// before a single token was generated.
	if !easy.Classified || !hard.Classified {
		t.Fatalf("both questions should be classified: easy=%v hard=%v", easy.Classified, hard.Classified)
	}
	if hard.Difficulty <= easy.Difficulty {
		t.Fatalf("difficulty scores not separated: easy=%.2f hard=%.2f", easy.Difficulty, hard.Difficulty)
	}
	if easy.DecisionMillis <= 0 {
		t.Fatal("routing latency should be measured via the preview endpoint")
	}
	if !strings.HasPrefix(easy.Route, "route:d=") {
		t.Fatalf("route not recorded from the preview: %q", easy.Route)
	}

	// The oracle pass must have run every question on every worker, and must show
	// that tiny cannot answer the hard one.
	if len(hard.Oracle) != 2 {
		t.Fatalf("oracle should cover both workers, got %+v", hard.Oracle)
	}
	if hard.Oracle["tiny"].Pass {
		t.Fatal("tiny should NOT be able to answer the hard question")
	}
	if !hard.Oracle["big"].Pass {
		t.Fatal("big should be able to answer the hard question")
	}
	if !easy.Oracle["tiny"].Pass {
		t.Fatal("tiny should be able to answer the easy question")
	}

	if len(easy.Perturbed) != len(arenaPerturbations) {
		t.Fatalf("robustness should preview every perturbation, got %d", len(easy.Perturbed))
	}

	// These workers answer in microseconds, so every question is "optimally"
	// routed under the absolute slack — the metric must not read sub-second
	// scheduling jitter as overspend.
	var m arenaMetrics
	for i := range res.Outcomes {
		arenaAccumulate([]*arenaMetrics{&m}, &res.Outcomes[i])
	}
	if m.optimal != 2 || m.overspend != 0 {
		t.Fatalf("timing noise leaked into the optimality metric: optimal=%d overspend=%d", m.optimal, m.overspend)
	}

	// And the report must render from that file without touching the fleet again.
	if err := arenaReport(out); err != nil {
		t.Fatalf("arena report: %v", err)
	}
}

func jsonStr(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
