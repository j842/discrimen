package router

// `discrimen arena …` — a router-level regression gate.
//
// The cold-start benchmark (benchmark.go) measures WORKERS. Nothing measured the
// ROUTER: whether the difficulty classifier actually sends each prompt to the
// cheapest worker that can answer it. That is the one claim the whole design
// rests on, and until now it was untested.
//
// This is RouterArena's evaluation shape (ICLR 2026, arXiv:2510.00202) run
// against a live fleet: a graded dataset spanning domains and difficulty levels,
// scored on accuracy, cost, ROUTING OPTIMALITY, robustness and routing latency.
// Optimality is the metric that matters and the one the paper reports every
// router failing: routers "are inefficient at recognizing when smaller, cheaper
// models are sufficient". Measuring it needs an oracle — every question against
// every worker — which is why -oracle is opt-in and expensive.
//
// Two deliberate departures from the published benchmark:
//
//   - COST IS SECONDS, not dollars. There is no per-token price on a self-hosted
//     fleet; what a request actually costs is the worker-time it occupies. Every
//     cost figure here is wall-clock seconds on the worker, and the ratios
//     (router cost ÷ oracle cost) are the comparable numbers.
//   - ROBUSTNESS MEASURES THE DECISION, not the answer. Perturbing a prompt and
//     re-asking would cost another full pass; asking /v1/route-preview whether the
//     perturbed prompt still routes to the same worker costs milliseconds and
//     tests the classifier directly, which is what a router-level gate is for.
//
// Grading uses the PRODUCTION checkAnswer, for the same reason benchgen does: a
// second copy of the grader would diverge silently and score a routing policy
// that never existed.
//
//	discrimen arena run -router URL -dataset FILE [-oracle] [-out FILE]
//	discrimen arena report -in FILE

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ── Dataset ─────────────────────────────────────────────────────────────────

// arenaQuestion is one graded item. The loader accepts the field spellings the
// common router datasets use rather than demanding one schema, because the
// point is to run against RouterArena's prepped splits without first writing a
// converter — and a converter is exactly the sort of thing that silently drops
// the difficulty labels the optimality metric needs.
type arenaQuestion struct {
	ID         string `json:"id"`
	Prompt     string `json:"prompt"`
	Expect     string `json:"expect"`
	Match      string `json:"match"`      // checkAnswer mode; inferred when absent
	Domain     string `json:"domain"`     // for the per-domain breakdown
	Difficulty string `json:"difficulty"` // easy | medium | hard, when the dataset says
}

// arenaRawQuestion is the tolerant on-disk shape. Every alternative spelling
// here was observed in a published router dataset or its HuggingFace export.
type arenaRawQuestion struct {
	ID         any    `json:"id"`
	UID        any    `json:"uid"`
	QuestionID any    `json:"question_id"`
	Prompt     string `json:"prompt"`
	Question   string `json:"question"`
	Input      string `json:"input"`
	Text       string `json:"text"`

	Expect string `json:"expect"`
	Answer string `json:"answer"`
	Target string `json:"target"`
	Label  string `json:"label"`
	Gold   string `json:"gold"`

	Match string `json:"match"`

	Domain   string `json:"domain"`
	Category string `json:"category"`
	Subject  string `json:"subject"`

	Difficulty string `json:"difficulty"`
	Level      string `json:"level"`

	Choices []string `json:"choices"`
	Options []string `json:"options"`
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if s := strings.TrimSpace(v); s != "" {
			return s
		}
	}
	return ""
}

func anyToString(vals ...any) string {
	for _, v := range vals {
		switch t := v.(type) {
		case string:
			if t != "" {
				return t
			}
		case float64:
			return strconv.FormatFloat(t, 'f', -1, 64)
		}
	}
	return ""
}

// arenaMatchFor infers the checkAnswer mode when the dataset doesn't declare
// one: a lone A–E is multiple choice, a bare number is numeric, anything else
// has to be found in the answer text.
func arenaMatchFor(expect string, choices []string) string {
	e := strings.TrimSpace(expect)
	if len(choices) > 0 && len(e) == 1 && e[0] >= 'A' && e[0] <= 'E' {
		return "mcq"
	}
	if len(e) == 1 && e[0] >= 'A' && e[0] <= 'E' {
		return "mcq"
	}
	if _, err := strconv.ParseFloat(strings.ReplaceAll(e, ",", ""), 64); err == nil {
		return "numeric"
	}
	return "final-contains"
}

// arenaLoadDataset reads JSONL (one object per line) or a JSON array. Items
// missing a prompt or an expected answer are skipped with a count, not silently
// dropped — a dataset that half-loads would quietly change every metric.
func arenaLoadDataset(path string) ([]arenaQuestion, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raws []arenaRawQuestion
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) > 0 && trimmed[0] == '[' {
		if err := json.Unmarshal(trimmed, &raws); err != nil {
			return nil, fmt.Errorf("parse JSON array: %w", err)
		}
	} else {
		for n, line := range bytes.Split(trimmed, []byte("\n")) {
			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}
			var raw arenaRawQuestion
			if err := json.Unmarshal(line, &raw); err != nil {
				return nil, fmt.Errorf("parse line %d: %w", n+1, err)
			}
			raws = append(raws, raw)
		}
	}

	out := make([]arenaQuestion, 0, len(raws))
	skipped := 0
	for i, raw := range raws {
		prompt := firstNonEmpty(raw.Prompt, raw.Question, raw.Input, raw.Text)
		expect := firstNonEmpty(raw.Expect, raw.Answer, raw.Target, raw.Label, raw.Gold)
		if prompt == "" || expect == "" {
			skipped++
			continue
		}
		choices := raw.Choices
		if len(choices) == 0 {
			choices = raw.Options
		}
		// Multiple-choice items are only gradable if the options reach the model.
		if len(choices) > 0 && !strings.Contains(prompt, choices[0]) {
			var b strings.Builder
			b.WriteString(prompt)
			b.WriteString("\n")
			for j, c := range choices {
				fmt.Fprintf(&b, "\n%c. %s", 'A'+j, c)
			}
			prompt = b.String()
		}
		id := anyToString(raw.ID, raw.UID, raw.QuestionID)
		if id == "" {
			id = fmt.Sprintf("q%04d", i)
		}
		match := strings.TrimSpace(raw.Match)
		if match == "" {
			match = arenaMatchFor(expect, choices)
		}
		out = append(out, arenaQuestion{
			ID:         id,
			Prompt:     prompt,
			Expect:     expect,
			Match:      match,
			Domain:     firstNonEmpty(raw.Domain, raw.Category, raw.Subject),
			Difficulty: strings.ToLower(firstNonEmpty(raw.Difficulty, raw.Level)),
		})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no usable questions in %s (%d items lacked a prompt or an expected answer)", path, skipped)
	}
	if skipped > 0 {
		fmt.Fprintf(os.Stderr, "warning: skipped %d of %d items with no prompt or no expected answer\n", skipped, len(raws))
	}
	return out, nil
}

// ── Results ─────────────────────────────────────────────────────────────────

// arenaOutcome is one question's full record: what the router decided, what it
// answered, and (with -oracle) what every worker would have done instead.
type arenaOutcome struct {
	Question arenaQuestion `json:"question"`

	// Routing decision, from /v1/route-preview — measured separately from the
	// answer so routing latency is the DECISION cost, not the generation cost.
	Route          string  `json:"route"`
	TargetQuality  int     `json:"target_quality"`
	Difficulty     float64 `json:"difficulty_score"`
	Reasoning      float64 `json:"reasoning_score"`
	ThinkingOn     bool    `json:"thinking_on"`
	DecisionMillis float64 `json:"decision_ms"`
	Classified     bool    `json:"classified"`

	// Served answer.
	BackendID string  `json:"backend_id"`
	Quality   int     `json:"backend_quality"`
	Seconds   float64 `json:"seconds"`
	Pass      bool    `json:"pass"`
	Truncated bool    `json:"truncated"`
	Errored   bool    `json:"errored"`
	Answer    string  `json:"answer"`

	// Oracle: per-worker verdicts for this question (-oracle only).
	Oracle map[string]arenaWorkerResult `json:"oracle,omitempty"`

	// Robustness: perturbed variants and where each one WOULD route (-robustness).
	Perturbed []arenaPerturbation `json:"perturbed,omitempty"`
}

type arenaWorkerResult struct {
	Pass    bool    `json:"pass"`
	Seconds float64 `json:"seconds"`
	Errored bool    `json:"errored"`
}

type arenaPerturbation struct {
	Kind       string  `json:"kind"`
	WouldServe string  `json:"would_serve"`
	Target     int     `json:"target_quality"`
	Difficulty float64 `json:"difficulty_score"`
}

type arenaResults struct {
	StartedAt string            `json:"started_at"`
	Router    string            `json:"router"`
	Dataset   string            `json:"dataset"`
	MaxTokens int               `json:"max_tokens"`
	Workers   []arenaWorkerInfo `json:"workers"`
	Outcomes  []arenaOutcome    `json:"outcomes"`
	Notes     []string          `json:"notes,omitempty"`
}

type arenaWorkerInfo struct {
	ID      string `json:"id"`
	Quality int    `json:"quality"`
}

// ── Command wiring ──────────────────────────────────────────────────────────

func arenaUsage() {
	fmt.Fprint(os.Stderr, `discrimen arena — measure the ROUTER (not the workers) against a graded dataset

  arena run -router URL -dataset FILE [flags]   Route + grade every question
  arena report -in FILE                         Print the metrics table

Flags for run:
  -out FILE          where to write results json (default arena-results.json)
  -token TOKEN       router admin key (default $ROUTER_ADMIN_KEY) — the run
                     lists the fleet via /backends, which is admin scope
  -limit N           only the first N questions (0 = all)
  -concurrency N     questions in flight (default 4)
  -max-tokens N      completion budget per answer (default 8192)
  -deadline SECONDS  per-answer wall-clock bound (default 180)
  -oracle            ALSO run every question on every worker, to measure routing
                     optimality. Costs questions x workers generations.
  -robustness        ALSO preview 3 perturbed variants per question, to measure
                     decision stability. Cheap (no generation).

The dataset is JSONL or a JSON array. Field names are matched leniently:
prompt|question|input|text, expect|answer|target|label|gold, domain|category|
subject, difficulty|level, plus optional choices|options and an explicit match
(contains|final-contains|numeric|mcq), which is otherwise inferred.

To run RouterArena's own split (github.com/RouteWorks/RouterArena):
  uv run python ./scripts/process_datasets/prep_datasets.py   # writes their splits
  discrimen arena run -router http://localhost:8585 -dataset sub_10.jsonl -oracle
`)
}

// runArenaCommand dispatches `discrimen arena …`. Returns false if argv isn't an
// arena invocation, so main() can carry on and start the server.
func runArenaCommand(argv []string) bool {
	if len(argv) < 2 || argv[1] != "arena" {
		return false
	}
	if len(argv) < 3 {
		arenaUsage()
		os.Exit(2)
	}
	args := argv[3:]
	var err error
	switch argv[2] {
	case "run":
		fs := flag.NewFlagSet("run", flag.ExitOnError)
		router := fs.String("router", "", "router base URL (required)")
		dataset := fs.String("dataset", "", "graded dataset, JSONL or JSON array (required)")
		out := fs.String("out", "arena-results.json", "where to write results")
		token := fs.String("token", os.Getenv("ROUTER_ADMIN_KEY"), "router admin key — the run lists the fleet via /backends, which is admin scope")
		limit := fs.Int("limit", 0, "only the first N questions (0 = all)")
		conc := fs.Int("concurrency", 4, "questions in flight")
		maxTokens := fs.Int("max-tokens", 8192, "completion budget per answer")
		deadline := fs.Int("deadline", 180, "per-answer wall-clock bound, seconds")
		oracle := fs.Bool("oracle", false, "also run every question on every worker (routing optimality)")
		robustness := fs.Bool("robustness", false, "also preview perturbed variants (decision stability)")
		_ = fs.Parse(args)
		if *router == "" || *dataset == "" {
			arenaUsage()
			os.Exit(2)
		}
		err = arenaRun(arenaConfig{
			router:      *router,
			token:       *token,
			dataset:     *dataset,
			out:         *out,
			limit:       *limit,
			concurrency: *conc,
			maxTokens:   *maxTokens,
			deadline:    time.Duration(*deadline) * time.Second,
			oracle:      *oracle,
			robustness:  *robustness,
		})
	case "report":
		fs := flag.NewFlagSet("report", flag.ExitOnError)
		in := fs.String("in", "arena-results.json", "results file from `arena run`")
		_ = fs.Parse(args)
		err = arenaReport(*in)
	default:
		arenaUsage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "arena %s: %v\n", argv[2], err)
		os.Exit(1)
	}
	return true
}

type arenaConfig struct {
	router      string
	token       string
	dataset     string
	out         string
	limit       int
	concurrency int
	maxTokens   int
	deadline    time.Duration
	oracle      bool
	robustness  bool
}

// ── Run ─────────────────────────────────────────────────────────────────────

func arenaRun(cfg arenaConfig) error {
	questions, err := arenaLoadDataset(cfg.dataset)
	if err != nil {
		return err
	}
	if cfg.limit > 0 && cfg.limit < len(questions) {
		questions = questions[:cfg.limit]
	}
	workers, err := benchFetchBackends(cfg.router, cfg.token)
	if err != nil {
		return fmt.Errorf("list workers: %w", err)
	}
	if len(workers) == 0 {
		return fmt.Errorf("no ready chat workers registered at %s", cfg.router)
	}
	if cfg.concurrency < 1 {
		cfg.concurrency = 1
	}

	res := arenaResults{
		StartedAt: time.Now().UTC().Format(time.RFC3339),
		Router:    cfg.router,
		Dataset:   cfg.dataset,
		MaxTokens: cfg.maxTokens,
		Outcomes:  make([]arenaOutcome, len(questions)),
	}
	for _, w := range workers {
		res.Workers = append(res.Workers, arenaWorkerInfo{ID: w.ID, Quality: w.Quality})
	}
	if !cfg.oracle {
		res.Notes = append(res.Notes, "routing optimality NOT measured — re-run with -oracle")
	}
	if !cfg.robustness {
		res.Notes = append(res.Notes, "robustness NOT measured — re-run with -robustness")
	}

	total := len(questions)
	perQ := 1
	if cfg.oracle {
		perQ += len(workers)
	}
	fmt.Fprintf(os.Stderr, "arena: %d questions x %d generations each across %d workers\n", total, perQ, len(workers))

	// Warm the router's difficulty classifier before measuring anything. It
	// bootstraps its centroids lazily on first use, and requests that arrive
	// DURING that bootstrap fall back to unclassified quality/speed routing — so
	// against a freshly restarted router the first few questions would be scored
	// against a routing policy that wasn't running. One throwaway preview closes
	// the window, and if the fleet still isn't classifying, the whole run would
	// measure degraded routing and the operator needs to know before it starts.
	if warm, err := arenaPostPreview(cfg, "warm up the difficulty classifier"); err != nil {
		return fmt.Errorf("warm-up preview failed (is %s reachable, and is the token right?): %w", cfg.router, err)
	} else if !warm.Classified {
		msg := "auto-routing is DEGRADED — prompts are not being classified (no embeddings worker?), " +
			"so this run measures plain quality/speed routing, not difficulty routing"
		fmt.Fprintf(os.Stderr, "WARNING: %s\n", msg)
		res.Notes = append(res.Notes, msg)
	}

	var done int64
	var mu sync.Mutex
	sem := make(chan struct{}, cfg.concurrency)
	var wg sync.WaitGroup
	for i := range questions {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			q := questions[i]
			out := arenaOutcome{Question: q}
			arenaPreview(cfg, q.Prompt, &out)
			arenaAnswer(cfg, q, &out)
			if cfg.robustness {
				out.Perturbed = arenaRobustness(cfg, q)
			}
			if cfg.oracle {
				out.Oracle = arenaOracle(cfg, q, workers)
			}

			mu.Lock()
			res.Outcomes[i] = out
			done++
			if done%10 == 0 || done == int64(total) {
				fmt.Fprintf(os.Stderr, "  %d/%d\n", done, total)
			}
			mu.Unlock()
		}(i)
	}
	wg.Wait()

	blob, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(cfg.out, blob, 0o644); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "\nwrote %s\n\n", cfg.out)
	return arenaPrint(&res)
}

// arenaPreview records the routing DECISION and what it cost to make, via
// /v1/route-preview — no generation, so this is routing latency in isolation.
func arenaPreview(cfg arenaConfig, prompt string, out *arenaOutcome) {
	start := time.Now()
	pv, err := arenaPostPreview(cfg, prompt)
	out.DecisionMillis = float64(time.Since(start).Microseconds()) / 1000
	if err != nil || pv == nil {
		return
	}
	out.Route = pv.Route
	out.TargetQuality = pv.TargetQuality
	out.Classified = pv.Classified
	out.ThinkingOn = pv.Thinking.Enabled
	if pv.Difficulty != nil {
		out.Difficulty = *pv.Difficulty
	}
	if pv.Reasoning != nil {
		out.Reasoning = *pv.Reasoning
	}
}

func arenaPostPreview(cfg arenaConfig, prompt string) (*previewResponse, error) {
	payload := map[string]any{
		"model":    "default",
		"messages": []map[string]string{{"role": "user", "content": prompt}},
	}
	raw, status, err := arenaPost(cfg, "/v1/route-preview", payload, 30*time.Second)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("route-preview: HTTP %d", status)
	}
	var pv previewResponse
	if err := json.Unmarshal(raw, &pv); err != nil {
		return nil, err
	}
	return &pv, nil
}

// arenaAnswer asks the router the question the way a real client would — model
// "default", no thinking hint, no requirements — and grades the answer.
func arenaAnswer(cfg arenaConfig, q arenaQuestion, out *arenaOutcome) {
	start := time.Now()
	raw, status, hdr, err := arenaChat(cfg, q.Prompt, "")
	out.Seconds = time.Since(start).Seconds()
	if err != nil || status != http.StatusOK {
		out.Errored = true
		return
	}
	out.BackendID = hdr.Get("X-Llm-Backend-Id")
	if out.Route == "" {
		out.Route = hdr.Get("X-Llm-Route")
	}
	var decoded map[string]any
	if json.Unmarshal(raw, &decoded) != nil {
		out.Errored = true
		return
	}
	content, finish := completionContent(decoded)
	out.Answer = answerTail(content)
	out.Truncated = finish == "length"
	out.Pass = checkAnswer(benchmarkQ{Prompt: q.Prompt, Expect: q.Expect, Match: q.Match}, content)
}

// arenaOracle runs the question on EVERY worker, pinned, so the report can name
// the cheapest worker that would have got it right. This is the expensive half
// of the harness and the only way optimality can be computed at all.
func arenaOracle(cfg arenaConfig, q arenaQuestion, workers []benchBackend) map[string]arenaWorkerResult {
	oracle := make(map[string]arenaWorkerResult, len(workers))
	for _, w := range workers {
		start := time.Now()
		raw, status, _, err := arenaChat(cfg, q.Prompt, w.ID)
		secs := time.Since(start).Seconds()
		if err != nil || status != http.StatusOK {
			oracle[w.ID] = arenaWorkerResult{Seconds: secs, Errored: true}
			continue
		}
		var decoded map[string]any
		if json.Unmarshal(raw, &decoded) != nil {
			oracle[w.ID] = arenaWorkerResult{Seconds: secs, Errored: true}
			continue
		}
		content, _ := completionContent(decoded)
		oracle[w.ID] = arenaWorkerResult{
			Pass:    checkAnswer(benchmarkQ{Prompt: q.Prompt, Expect: q.Expect, Match: q.Match}, content),
			Seconds: secs,
		}
	}
	return oracle
}

// arenaPerturbations are deterministic surface rewrites that must not change a
// prompt's difficulty. Deterministic on purpose: a robustness number produced by
// random noise isn't comparable between runs, and the whole use of this metric is
// comparing runs.
var arenaPerturbations = []struct {
	kind  string
	apply func(string) string
}{
	{"lowercase", strings.ToLower},
	{"polite", func(s string) string { return "Could you please help me with this? " + s }},
	{"typo", arenaTypo},
}

// arenaTypo transposes two adjacent characters near the middle of the prompt —
// the commonest real typo, and one a human reader wouldn't notice.
func arenaTypo(s string) string {
	r := []rune(s)
	if len(r) < 8 {
		return s
	}
	i := len(r) / 2
	r[i], r[i+1] = r[i+1], r[i]
	return string(r)
}

func arenaRobustness(cfg arenaConfig, q arenaQuestion) []arenaPerturbation {
	out := make([]arenaPerturbation, 0, len(arenaPerturbations))
	for _, p := range arenaPerturbations {
		pv, err := arenaPostPreview(cfg, p.apply(q.Prompt))
		if err != nil || pv == nil {
			continue
		}
		item := arenaPerturbation{Kind: p.kind, WouldServe: pv.WouldServe, Target: pv.TargetQuality}
		if pv.Difficulty != nil {
			item.Difficulty = *pv.Difficulty
		}
		out = append(out, item)
	}
	return out
}

// ── HTTP ────────────────────────────────────────────────────────────────────

func arenaChat(cfg arenaConfig, prompt, pinBackend string) ([]byte, int, http.Header, error) {
	payload := map[string]any{
		"model":       "default",
		"stream":      false,
		"temperature": 0,
		"max_tokens":  cfg.maxTokens,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, 0, nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), cfg.deadline)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		trimSlash(cfg.router)+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, 0, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.token != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.token)
	}
	if pinBackend != "" {
		req.Header.Set("X-LLM-Backend-ID", pinBackend)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, 0, nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	return raw, resp.StatusCode, resp.Header, err
}

func arenaPost(cfg arenaConfig, path string, payload any, timeout time.Duration) ([]byte, int, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, 0, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, trimSlash(cfg.router)+path, bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.token != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	return raw, resp.StatusCode, err
}

// ── Report ──────────────────────────────────────────────────────────────────

// A pick counts as OPTIMAL when it cost no more than the cheapest correct
// worker, within a tolerance that is both relative and absolute:
//
//   - the relative term absorbs run-to-run variance — two runs of the same
//     worker on a loaded fleet differ by well over a rounding error, and a metric
//     that called that a miss would report noise as overspend;
//   - the absolute term stops sub-second differences counting at all. Real
//     routing decisions on this fleet are separated by seconds (the GPU/CPU
//     prefill gap alone is ~36s on a 4k prompt); anything under a quarter of a
//     second is scheduling jitter, and without this floor a fleet of fast workers
//     scores 0% optimal purely on measurement noise.
const (
	arenaOptimalTolerance = 1.10
	arenaOptimalSlack     = 0.25
)

func arenaReport(path string) error {
	blob, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var res arenaResults
	if err := json.Unmarshal(blob, &res); err != nil {
		return err
	}
	return arenaPrint(&res)
}

type arenaMetrics struct {
	n, passed, errored, truncated int
	seconds                       float64
	decisionMs                    []float64

	// Oracle-derived (zero when -oracle wasn't run).
	haveOracle        bool
	oracleAnswered    int     // some worker got it right
	oracleSeconds     float64 // cost of the cheapest CORRECT worker, summed
	answerableSeconds float64 // what the ROUTER spent on those same questions
	optimal           int     // router picked a cheapest-correct worker
	overspend         int     // router was right, but a cheaper worker was also right
	undershoot        int     // router was wrong, and some worker was right
	unanswerable      int     // no worker got it right — not the router's fault

	// Robustness (zero when -robustness wasn't run).
	haveRobust  int
	stableServe int
	stableTier  int
}

func arenaPrint(res *arenaResults) error {
	m := arenaMetrics{}
	byDomain := map[string]*arenaMetrics{}
	byDifficulty := map[string]*arenaMetrics{}
	byBackend := map[string]int{}

	for i := range res.Outcomes {
		o := &res.Outcomes[i]
		if o.Question.Prompt == "" {
			continue // never ran (interrupted run)
		}
		buckets := []*arenaMetrics{&m}
		if d := o.Question.Domain; d != "" {
			if byDomain[d] == nil {
				byDomain[d] = &arenaMetrics{}
			}
			buckets = append(buckets, byDomain[d])
		}
		if d := o.Question.Difficulty; d != "" {
			if byDifficulty[d] == nil {
				byDifficulty[d] = &arenaMetrics{}
			}
			buckets = append(buckets, byDifficulty[d])
		}
		if o.BackendID != "" {
			byBackend[o.BackendID]++
		}
		arenaAccumulate(buckets, o)
	}

	if m.n == 0 {
		return fmt.Errorf("no completed questions in the results file")
	}

	fmt.Printf("RouterArena-style report — %s (%s)\n", res.Router, res.StartedAt)
	fmt.Printf("dataset: %s   questions: %d   workers: %d   max_tokens: %d\n\n", res.Dataset, m.n, len(res.Workers), res.MaxTokens)

	fmt.Println("ACCURACY & COST")
	fmt.Printf("  accuracy            %6.2f%%  (%d/%d)\n", pct(m.passed, m.n), m.passed, m.n)
	fmt.Printf("  mean cost           %6.2fs   per query (worker wall-clock)\n", m.seconds/float64(m.n))
	fmt.Printf("  total cost          %6.1fs\n", m.seconds)
	if m.truncated > 0 {
		fmt.Printf("  truncated           %6d    answers hit max_tokens=%d — raise -max-tokens if this is large\n", m.truncated, res.MaxTokens)
	}
	if m.errored > 0 {
		fmt.Printf("  errored             %6d    requests failed outright (counted as wrong)\n", m.errored)
	}
	p50, p95 := percentile(m.decisionMs, 0.50), percentile(m.decisionMs, 0.95)
	fmt.Printf("  routing latency     %6.1fms p50, %.1fms p95  (decision only, via /v1/route-preview)\n\n", p50, p95)

	if m.haveOracle {
		fmt.Println("ROUTING OPTIMALITY  (the metric every published router fails)")
		fmt.Printf("  oracle accuracy     %6.2f%%  (%d/%d answerable by SOME worker)\n", pct(m.oracleAnswered, m.n), m.oracleAnswered, m.n)
		fmt.Printf("  accuracy gap        %6.2f pts below oracle\n", pct(m.oracleAnswered, m.n)-pct(m.passed, m.n))
		fmt.Printf("  optimal selection   %6.2f%%  (%d/%d picked a cheapest-correct worker)\n", pct(m.optimal, m.oracleAnswered), m.optimal, m.oracleAnswered)
		if m.oracleAnswered > 0 {
			oracleMean := m.oracleSeconds / float64(m.oracleAnswered)
			fmt.Printf("  oracle cost         %6.2fs   per query (cheapest correct worker)\n", oracleMean)
			// Both means are over the SAME answerable subset, so the ratio compares
			// like with like: routerAnswerableSeconds is what the router spent on the
			// very questions the oracle priced.
			if oracleMean > 0 {
				fmt.Printf("  cost ratio          %6.2fx  router vs oracle (over the %d answerable questions)\n",
					(m.answerableSeconds/float64(m.oracleAnswered))/oracleMean, m.oracleAnswered)
			}
		}
		fmt.Printf("  overspend           %6d    right answer, cheaper worker would also have been right\n", m.overspend)
		fmt.Printf("  undershoot          %6d    wrong answer, some worker would have been right\n", m.undershoot)
		fmt.Printf("  unanswerable        %6d    no worker got it — not a routing failure\n\n", m.unanswerable)
	}

	if m.haveRobust > 0 {
		fmt.Println("ROBUSTNESS  (perturbed prompt → same routing decision)")
		fmt.Printf("  same worker         %6.2f%%  (%d/%d)\n", pct(m.stableServe, m.haveRobust), m.stableServe, m.haveRobust)
		fmt.Printf("  same quality tier   %6.2f%%  (%d/%d)\n\n", pct(m.stableTier, m.haveRobust), m.stableTier, m.haveRobust)
	}

	if len(byDifficulty) > 0 {
		fmt.Println("BY DIFFICULTY")
		arenaBreakdown(byDifficulty)
	}
	if len(byDomain) > 0 {
		fmt.Println("BY DOMAIN")
		arenaBreakdown(byDomain)
	}

	fmt.Println("WORKER SHARE")
	quality := map[string]int{}
	for _, w := range res.Workers {
		quality[w.ID] = w.Quality
	}
	ids := make([]string, 0, len(byBackend))
	for id := range byBackend {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return byBackend[ids[i]] > byBackend[ids[j]] })
	for _, id := range ids {
		fmt.Printf("  %-32s %5.1f%%  (%d, q=%d)\n", id, pct(byBackend[id], m.n), byBackend[id], quality[id])
	}
	for _, n := range res.Notes {
		fmt.Printf("\nnote: %s\n", n)
	}
	return nil
}

func arenaAccumulate(buckets []*arenaMetrics, o *arenaOutcome) {
	// Oracle verdict for this question, computed once and folded into every bucket.
	cheapestSeconds, answered := math.Inf(1), false
	for _, w := range o.Oracle {
		if !w.Pass || w.Errored {
			continue
		}
		answered = true
		if w.Seconds < cheapestSeconds {
			cheapestSeconds = w.Seconds
		}
	}

	stableServe, stableTier := 0, 0
	for _, p := range o.Perturbed {
		if p.WouldServe != "" && p.WouldServe == o.BackendID {
			stableServe++
		}
		if p.Target == o.TargetQuality {
			stableTier++
		}
	}

	for _, b := range buckets {
		b.n++
		b.seconds += o.Seconds
		b.decisionMs = append(b.decisionMs, o.DecisionMillis)
		if o.Pass {
			b.passed++
		}
		if o.Errored {
			b.errored++
		}
		if o.Truncated {
			b.truncated++
		}
		if len(o.Oracle) > 0 {
			b.haveOracle = true
			switch {
			case !answered:
				b.unanswerable++
			default:
				b.oracleAnswered++
				b.oracleSeconds += cheapestSeconds
				b.answerableSeconds += o.Seconds
				switch {
				case !o.Pass:
					b.undershoot++
				case o.Seconds <= cheapestSeconds*arenaOptimalTolerance+arenaOptimalSlack:
					b.optimal++
				default:
					b.overspend++
				}
			}
		}
		if len(o.Perturbed) > 0 {
			b.haveRobust += len(o.Perturbed)
			b.stableServe += stableServe
			b.stableTier += stableTier
		}
	}
}

func arenaBreakdown(groups map[string]*arenaMetrics) {
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		g := groups[k]
		line := fmt.Sprintf("  %-24s %5.1f%% acc  %6.2fs/q  (n=%d)", k, pct(g.passed, g.n), g.seconds/float64(g.n), g.n)
		if g.haveOracle && g.oracleAnswered > 0 {
			line += fmt.Sprintf("  optimal %5.1f%%", pct(g.optimal, g.oracleAnswered))
		}
		fmt.Println(line)
	}
	fmt.Println()
}

func pct(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b) * 100
}

func percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	i := int(math.Round(p * float64(len(s)-1)))
	if i < 0 {
		i = 0
	}
	if i >= len(s) {
		i = len(s) - 1
	}
	return s[i]
}
