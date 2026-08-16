package router

// benchgen builds the GENERATED half of the quality benchmark from LiveBench.
// It runs as a subcommand of the router binary — `discrimen bench …` — rather
// than as a separate tool, for one reason that outranks the tidiness of a
// tools/ directory: calibration must grade with the EXACT production grader.
// A second copy of checkAnswer would silently diverge, and the tier assignments
// it produced would be wrong in a way nothing downstream could detect. Sharing
// the package is the only way to guarantee they can't drift.
//
// WHY THIS EXISTS. benchmark_data.go's version history is three bumps of
// saturation-chasing: v29 built a GPQA-style expert tier and v31 cut it at 17
// points of spread; v30 added a ceiling tier because a 27B scored 100% on tier
// 9; v31 recorded that the same 27B then aced v30. Each bump invalidates every
// cached profile and costs a full fleet re-benchmark (8–30 min per worker,
// measured). That treadmill is a maintenance cost, not a measurement problem,
// and LiveBench already runs it as a public project — about a sixth of its
// questions replaced monthly, a full turnover roughly every six months,
// explicitly to resolve saturation and contamination.
//
// So the arithmetic tiers get SOURCED rather than authored. They are the
// largest block in the file and, by the file's own measurements, the weakest
// discriminators — frontier maths 31 points against traps 62 and unrecallable
// 58. The trap, unrecallable and world-model tiers stay hand-written: they are
// the fleet's best spread AND the part LiveBench cannot supply, since their
// whole value is being absent from any training corpus.
//
// THREE PHASES, separate because their costs differ by orders of magnitude:
//
//	discrimen bench fetch      network, seconds.  HF        -> benchdata/pool.json
//	discrimen bench calibrate  fleet, HOURS.      pool.json -> benchdata/calibration.json
//	discrimen bench emit       hermetic, instant. calib     -> benchmark_data_live.go
//
// Only fetch touches the network, only calibrate touches the fleet, so emit can
// be re-run freely while tuning the tier shape.
//
// WHY THE OUTPUT IS COMMITTED, and not fetched at install time. The question set
// is part of the profile cache key: LoadWorkerProfile keys on (id, model) and
// benchmarkVersion, and autoTargetQuality reads scores as one absolute 0–100
// scale. If the questions could change without the version changing, a
// worker profiled on Tuesday and one profiled on Thursday would be graded on
// different sets and then compared on the same 0–100 scale — silently, with
// every number still looking plausible. Fetching from install-pre.sh or the
// Dockerfile has exactly that failure mode (and adds an egress dependency to a
// box that may not have one). Instead the generated file is committed, and a
// refresh is a reviewed commit that bumps benchmarkVersion alongside it.

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
)

// TASK ALLOWLIST. LiveBench publishes seven categories, but most need a grader
// the router doesn't have and shouldn't grow: instruction_following ships no
// ground_truth at all (an IFEval instruction_id_list + kwargs, needing the
// constraint verifiers), data_analysis grades reformatted tables as JSON,
// coding and agentic coding need execution, and language's connections /
// plot_unscrambling award PARTIAL credit — which this file's flat
// percentage-of-questions-correct scale cannot express without changing what
// the number means.
//
// What's left grades against checkAnswer, using the two match types added for
// LiveBench's answer formats:
//
//	spatial       small integer, asked for in bold        -> numeric
//	zebra_puzzle  ordered list in <solution></solution>    -> exact-list
//	olympiad      ordered permutation of tag indices       -> exact-list
//	math_comp     AIME integer ("073") or AMC letter       -> numeric / mcq-repeat
//
// AMPS_Hard is deliberately excluded: its ground truth is LaTeX expressions, and
// comparing those properly needs symbolic normalisation, not string matching.
// If it is ever added, add the grader first and the questions second.
//
// math_comp is absent from the map below because it is the one task whose match
// type can't be decided from the task name: it mixes AMC, whose answer is an
// option letter the prompt asks for five times over ("DDDDD"), with AIME, whose
// answer is a zero-padded integer. benchMatchFor reads the ground truth instead.
// Treating the whole task as numeric grades every AMC item a miss.
var benchAllowedTasks = map[string]string{ // LiveBench task -> checkAnswer match type
	"spatial":      "numeric",
	"zebra_puzzle": "exact-list",
	"olympiad":     "exact-list",
}

// benchAMCAnswerRe is a bare option letter — the AMC half of math_comp.
var benchAMCAnswerRe = regexp.MustCompile(`^[A-E]$`)

// benchMatchFor picks the grader for one candidate, from the task where that's
// enough and from the answer's shape where it isn't.
func benchMatchFor(task, groundTruth string) (string, bool) {
	if task == "math_comp" {
		if benchAMCAnswerRe.MatchString(strings.TrimSpace(groundTruth)) {
			return "mcq-repeat", true
		}
		return "numeric", true
	}
	m, ok := benchAllowedTasks[task]
	return m, ok
}

// benchMaxPromptChars bounds how long a candidate prompt may be, and exists
// because of the fleet's smallest context window rather than anything about the
// questions. A thinking-on answer gets benchThinkMaxTokens (16384) of budget; on
// a 16K-context worker a long prompt eats into that, the reply truncates, and
// runQualityBenchmark scores truncation as a failure. That would be measuring
// context, not capability — and worse, an item that passes on the 256K workers
// while truncating on the 16K ones looks like a strong DISCRIMINATOR to emit,
// so the tier assignment would quietly encode context size. 6000 chars is about
// 1500 tokens, leaving ~14.5K of thinking room on the smallest worker. It drops
// the long tail of olympiad and nothing else.
const benchMaxPromptChars = 6000

// benchDatasets are the LiveBench HF datasets holding the allowlisted tasks.
var benchDatasets = []string{"math", "reasoning"}

// poolQuestion is one candidate. ID is LiveBench's own content hash, so it is
// stable across refreshes and is what calibration keys on — a question that
// survives a refresh keeps its measured difficulty rather than being re-measured.
type poolQuestion struct {
	ID          string `json:"id"`
	Category    string `json:"category"`
	Task        string `json:"task"`
	Match       string `json:"match"`
	Prompt      string `json:"prompt"`
	Expect      string `json:"expect"`
	ReleaseDate string `json:"release_date"`
	Retired     bool   `json:"retired"` // LiveBench has rotated it out of the live set
}

type benchPool struct {
	FetchedAt string         `json:"fetched_at"`
	Source    string         `json:"source"`
	Questions []poolQuestion `json:"questions"`
}

// benchDataDir resolves to app/benchdata regardless of the working directory, so
// the subcommand behaves the same from anywhere in the tree. It relies on the
// source path, which is correct precisely because these are developer commands
// run from a checkout — they are never invoked in the container.
func benchDataDir() string {
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		return "benchdata"
	}
	return filepath.Join(filepath.Dir(self), "benchdata")
}

func benchAppDir() string { return filepath.Dir(benchDataDir()) }

func benchUsage() {
	fmt.Fprint(os.Stderr, `discrimen bench — build the generated half of the quality benchmark

  bench fetch                     Pull candidate questions from HuggingFace
  bench calibrate -router URL     Grade the pool against every ready backend
  bench emit                      Select items and write benchmark_data_live.go

Quarterly refresh. Each one is a benchmarkVersion bump and therefore a full
fleet re-benchmark, so monthly is not worth it for a set that only turns over
every six months:

  go run . bench fetch
  go run . bench calibrate -router http://localhost:8585 -token "$ROUTER_ADMIN_KEY"
  go run . bench emit
  # then bump benchmarkVersion, drop the replaced tiers from benchmark_data.go,
  # and commit the generated file in the SAME commit as the version bump.
`)
}

// runBenchCommand dispatches `discrimen bench …`. Returns false if argv isn't a
// bench invocation, so main() can carry on and start the server.
func runBenchCommand(argv []string) bool {
	if len(argv) < 2 || argv[1] != "bench" {
		return false
	}
	if len(argv) < 3 {
		benchUsage()
		os.Exit(2)
	}
	args := argv[3:]
	var err error
	switch argv[2] {
	case "fetch":
		fs := flag.NewFlagSet("fetch", flag.ExitOnError)
		includeRetired := fs.Bool("include-retired", true,
			"keep questions LiveBench has rotated out — more contaminated, but roughly doubles the pool")
		_ = fs.Parse(args)
		err = benchFetch(*includeRetired)
	case "calibrate":
		fs := flag.NewFlagSet("calibrate", flag.ExitOnError)
		router := fs.String("router", "", "router base URL (required)")
		token := fs.String("token", os.Getenv("ROUTER_ADMIN_KEY"), "router admin key — calibration lists the fleet via /backends, which is admin scope")
		conc := fs.Int("concurrency", 2, "concurrent questions per backend")
		limit := fs.Int("limit", 0, "max questions per task (0 = all) — for a smoke test before the full run")
		_ = fs.Parse(args)
		if *router == "" {
			benchUsage()
			os.Exit(2)
		}
		err = benchCalibrate(*router, *token, *conc, *limit)
	case "emit":
		fs := flag.NewFlagSet("emit", flag.ExitOnError)
		_ = fs.Parse(args)
		err = benchEmit()
	default:
		benchUsage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "bench %s: %v\n", argv[2], err)
		os.Exit(1)
	}
	return true
}

// hfRows is the HuggingFace datasets-server /rows response. Using the JSON API
// rather than the parquet files keeps this dependency-free — the router module
// has no parquet reader and should not gain one for a developer command.
type hfRows struct {
	Rows []struct {
		Row map[string]any `json:"row"`
	} `json:"rows"`
	NumRowsTotal int `json:"num_rows_total"`
}

func benchFetch(includeRetired bool) error {
	client := &http.Client{Timeout: 60 * time.Second}
	var all []poolQuestion
	for _, ds := range benchDatasets {
		qs, err := benchFetchDataset(client, ds, includeRetired)
		if err != nil {
			return fmt.Errorf("%s: %w", ds, err)
		}
		fmt.Fprintf(os.Stderr, "  %-10s %3d candidates\n", ds, len(qs))
		all = append(all, qs...)
	}
	if len(all) == 0 {
		return fmt.Errorf("no questions matched the task allowlist — has LiveBench renamed its tasks?")
	}
	// Deterministic order, so a re-fetch that changes nothing produces no diff.
	sort.Slice(all, func(i, j int) bool { return all[i].ID < all[j].ID })

	byTask := map[string]int{}
	retired := 0
	for _, q := range all {
		byTask[q.Task]++
		if q.Retired {
			retired++
		}
	}
	fmt.Fprintf(os.Stderr, "\n%d candidates (%d live, %d retired)\n", len(all), len(all)-retired, retired)
	for _, t := range benchSortedKeys(byTask) {
		fmt.Fprintf(os.Stderr, "  %-14s %3d\n", t, byTask[t])
	}

	if err := os.MkdirAll(benchDataDir(), 0o755); err != nil {
		return err
	}
	path := filepath.Join(benchDataDir(), "pool.json")
	buf, err := json.MarshalIndent(benchPool{
		FetchedAt: time.Now().UTC().Format(time.RFC3339),
		Source:    "https://huggingface.co/livebench — LiveBench, ICLR 2025 (arxiv.org/abs/2406.19314)",
		Questions: all,
	}, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, append(buf, '\n'), 0o644); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "\nwrote %s\n", path)
	return nil
}

func benchFetchDataset(client *http.Client, ds string, includeRetired bool) ([]poolQuestion, error) {
	var out []poolQuestion
	const page = 100
	for offset := 0; ; offset += page {
		u := fmt.Sprintf("https://datasets-server.huggingface.co/rows?dataset=%s&config=default&split=test&offset=%d&length=%d",
			url.QueryEscape("livebench/"+ds), offset, page)
		resp, err := client.Get(u)
		if err != nil {
			return nil, err
		}
		var rows hfRows
		err = json.NewDecoder(resp.Body).Decode(&rows)
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
		if len(rows.Rows) == 0 {
			return out, nil
		}
		for _, r := range rows.Rows {
			q, ok := benchToPoolQuestion(r.Row, ds)
			if !ok || (q.Retired && !includeRetired) {
				continue
			}
			out = append(out, q)
		}
		if offset+page >= rows.NumRowsTotal {
			return out, nil
		}
	}
}

func benchToPoolQuestion(row map[string]any, ds string) (poolQuestion, bool) {
	task, _ := row["task"].(string)
	gt, _ := row["ground_truth"].(string)
	id, _ := row["question_id"].(string)
	if gt == "" || id == "" {
		return poolQuestion{}, false
	}
	match, allowed := benchMatchFor(task, gt)
	if !allowed {
		return poolQuestion{}, false
	}
	// Every allowlisted task is single-turn, so turns[0] is the whole prompt.
	// Anything else is a task shape this tool doesn't understand — skip rather
	// than silently grade half a question.
	turns, _ := row["turns"].([]any)
	if len(turns) != 1 {
		return poolQuestion{}, false
	}
	prompt, _ := turns[0].(string)
	if prompt == "" || len(prompt) > benchMaxPromptChars {
		return poolQuestion{}, false
	}
	removal, _ := row["livebench_removal_date"].(string)
	release, _ := row["livebench_release_date"].(string)
	return poolQuestion{
		ID: id, Category: ds, Task: task, Match: match,
		Prompt: prompt, Expect: gt, ReleaseDate: release,
		Retired: removal != "",
	}, true
}

func benchSortedKeys[V any](m map[string]V) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}

func benchLoadPool() (*benchPool, error) {
	buf, err := os.ReadFile(filepath.Join(benchDataDir(), "pool.json"))
	if err != nil {
		return nil, fmt.Errorf("%w (run `bench fetch` first)", err)
	}
	var p benchPool
	if err := json.Unmarshal(buf, &p); err != nil {
		return nil, err
	}
	return &p, nil
}
