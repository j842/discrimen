package router

// bench calibrate grades the whole candidate pool against every ready backend
// and records which worker got which question right.
//
// This is ITEM ANALYSIS, not scoring. The output isn't "how good is this model"
// — it's "how well does this question separate the fleet", which is the only
// property benchmark_data.go actually needs. Routing acts on quality
// DIFFERENCES; a question every worker passes and a question no worker passes
// both contribute exactly zero separation while still costing a slot in the
// profiling budget. Finding and discarding them is what this phase is for, and
// it is the same judgement the file's comments record being made by hand across
// v29–v32 — just measured instead of guessed.
//
// It doubles as the correlation check worth running before trusting any of this:
// the per-backend totals here, set against the router's own q, say whether the
// hand-tuned set has overfitted to the fleet it was tuned on.
//
// COST. Measured on the current fleet, 97 questions takes 8–30 min per worker,
// so a few hundred candidates is hours. Backends are separate machines and run
// concurrently, so wall-clock is the slowest single worker, not the sum. The run
// checkpoints continuously and skips completed (backend, question) pairs on
// restart — assume it will be interrupted.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// calibration is the persisted result. Pass is keyed by question id so a pool
// refresh that keeps a question also keeps its measured difficulty.
type calibration struct {
	RanAt   string         `json:"ran_at"`
	Results []*calibResult `json:"results"`
}

type calibResult struct {
	Backend string `json:"backend"`
	// RouterQuality is the backend's q at calibration time — the hand-authored
	// benchmark's verdict, kept so emit can report the correlation between the
	// old instrument and the new one.
	RouterQuality int             `json:"router_quality"`
	Pass          map[string]bool `json:"pass"`
}

func (c *calibration) forBackend(id string) *calibResult {
	for _, r := range c.Results {
		if r.Backend == id {
			return r
		}
	}
	return nil
}

func benchCalibPath() string { return filepath.Join(benchDataDir(), "calibration.json") }

func benchLoadCalibration() *calibration {
	c := &calibration{}
	buf, err := os.ReadFile(benchCalibPath())
	if err == nil {
		_ = json.Unmarshal(buf, c)
	}
	return c
}

func benchSaveCalibration(c *calibration) error {
	c.RanAt = time.Now().UTC().Format(time.RFC3339)
	sort.Slice(c.Results, func(i, j int) bool { return c.Results[i].Backend < c.Results[j].Backend })
	buf, err := json.MarshalIndent(c, "", " ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(benchDataDir(), 0o755); err != nil {
		return err
	}
	return os.WriteFile(benchCalibPath(), append(buf, '\n'), 0o644)
}

// benchLimitPerTask keeps at most n questions per task, preserving pool order.
// A smoke run with a small n covers every task/match combination in a few
// minutes, which is the cheap way to find out that a grader is broken before
// rather than after a multi-hour fleet run. Results it produces are real and
// are picked up by the full run rather than repeated.
func benchLimitPerTask(qs []poolQuestion, n int) []poolQuestion {
	if n <= 0 {
		return qs
	}
	seen := map[string]int{}
	out := make([]poolQuestion, 0, len(qs))
	for _, q := range qs {
		if seen[q.Task] >= n {
			continue
		}
		seen[q.Task]++
		out = append(out, q)
	}
	return out
}

func benchCalibrate(routerURL, token string, concurrency, limit int) error {
	pool, err := benchLoadPool()
	if err != nil {
		return err
	}
	pool.Questions = benchLimitPerTask(pool.Questions, limit)
	backends, err := benchFetchBackends(routerURL, token)
	if err != nil {
		return err
	}
	if len(backends) == 0 {
		return fmt.Errorf("no ready chat backends to calibrate against")
	}
	if concurrency < 1 {
		concurrency = 1
	}

	calib := benchLoadCalibration()
	fmt.Fprintf(os.Stderr, "calibrating %d questions against %d backends (concurrency %d/backend)\n\n",
		len(pool.Questions), len(backends), concurrency)

	var mu sync.Mutex // guards calib + the checkpoint write
	var wg sync.WaitGroup
	// Backends are distinct machines, so run them in parallel: wall-clock is the
	// slowest single worker rather than the sum of all of them.
	for _, b := range backends {
		wg.Add(1)
		go func(b benchBackend) {
			defer wg.Done()
			mu.Lock()
			res := calib.forBackend(b.ID)
			if res == nil {
				res = &calibResult{Backend: b.ID, Pass: map[string]bool{}}
				calib.Results = append(calib.Results, res)
			}
			res.RouterQuality = b.Quality
			mu.Unlock()

			todo := make([]poolQuestion, 0, len(pool.Questions))
			mu.Lock()
			for _, q := range pool.Questions {
				if _, done := res.Pass[q.ID]; !done {
					todo = append(todo, q)
				}
			}
			mu.Unlock()
			if len(todo) == 0 {
				fmt.Fprintf(os.Stderr, "  %-30s already complete\n", b.ID)
				return
			}

			start := time.Now()
			sem := make(chan struct{}, concurrency)
			var qwg sync.WaitGroup
			var done int
			for _, q := range todo {
				qwg.Add(1)
				sem <- struct{}{}
				go func(q poolQuestion) {
					defer qwg.Done()
					defer func() { <-sem }()
					pass := benchGradeOne(routerURL, token, b.ID, q)
					mu.Lock()
					res.Pass[q.ID] = pass
					done++
					// Checkpoint often: this runs for hours and will be interrupted.
					if done%10 == 0 {
						_ = benchSaveCalibration(calib)
					}
					mu.Unlock()
				}(q)
			}
			qwg.Wait()
			mu.Lock()
			passed := 0
			for _, ok := range res.Pass {
				if ok {
					passed++
				}
			}
			_ = benchSaveCalibration(calib)
			mu.Unlock()
			fmt.Fprintf(os.Stderr, "  %-30s %3d/%3d  (router q=%d, %s)\n",
				b.ID, passed, len(res.Pass), b.Quality, time.Since(start).Round(time.Second))
		}(b)
	}
	wg.Wait()

	if err := benchSaveCalibration(calib); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "\nwrote %s\n", benchCalibPath())
	return nil
}

// benchGradeOne asks one backend one question and grades it with the PRODUCTION
// grader. Everything about the request mirrors runQualityBenchmark — thinking
// on, temperature 0, the same token ceiling and usability deadline — because a
// question calibrated under different conditions than it will be graded under
// is calibrated for nothing. The tiers this pool replaces are all at or above
// benchHardTier, so thinking is unconditionally on.
func benchGradeOne(routerURL, token, backendID string, q poolQuestion) bool {
	payload := map[string]any{
		"model":                "default",
		"stream":               false,
		"max_tokens":           benchThinkMaxTokens,
		"temperature":          0,
		"chat_template_kwargs": map[string]bool{"enable_thinking": true},
		"messages":             []map[string]string{{"role": "user", "content": q.Prompt}},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return false
	}
	for attempt := 1; attempt <= benchMaxAttempts; attempt++ {
		ctx, cancel := context.WithTimeout(context.Background(), benchAnswerDeadline)
		req, err := http.NewRequestWithContext(ctx, http.MethodPost,
			trimSlash(routerURL)+"/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			cancel()
			return false
		}
		req.Header.Set("Content-Type", "application/json")
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		// Pin to the backend under test; without this the router would auto-route
		// by difficulty and we would be calibrating the routing policy.
		req.Header.Set("X-LLM-Backend-ID", backendID)

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			cancel()
			if isTimeout(err) {
				return false // too slow to be usable is a fail, and is never retried
			}
			time.Sleep(benchRetryBackoff * time.Duration(attempt))
			continue
		}
		raw, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		cancel()
		if readErr != nil || resp.StatusCode != http.StatusOK {
			time.Sleep(benchRetryBackoff * time.Duration(attempt))
			continue
		}
		var decoded map[string]any
		if json.Unmarshal(raw, &decoded) != nil {
			return false
		}
		content, _ := completionContent(decoded)
		return checkAnswer(benchmarkQ{Prompt: q.Prompt, Expect: q.Expect, Match: q.Match}, content)
	}
	return false
}

type benchBackend struct {
	ID      string
	Quality int
}

// benchFetchBackends lists the ready chat backends worth calibrating against.
// /backends is ADMIN scope since P3, so the token here has to be an admin key
// rather than the client token these developer commands used to take.
// Embeddings-only workers and anything not ready are skipped — a backend still
// profiling would contribute noise, not difficulty signal.
func benchFetchBackends(routerURL, token string) ([]benchBackend, error) {
	req, err := http.NewRequest(http.MethodGet, trimSlash(routerURL)+"/backends", nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("GET /backends: HTTP %d: %s", resp.StatusCode, b)
	}
	var payload struct {
		Backends []struct {
			ID       string   `json:"id"`
			Status   string   `json:"status"`
			Quality  int      `json:"quality"`
			Features []string `json:"features"`
		} `json:"backends"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	var out []benchBackend
	for _, b := range payload.Backends {
		if b.Status != "ready" || !benchHasFeature(b.Features, "chat") {
			continue
		}
		out = append(out, benchBackend{ID: b.ID, Quality: b.Quality})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

func benchHasFeature(features []string, want string) bool {
	for _, f := range features {
		if f == want {
			return true
		}
	}
	return false
}

func trimSlash(s string) string {
	for len(s) > 0 && s[len(s)-1] == '/' {
		s = s[:len(s)-1]
	}
	return s
}
