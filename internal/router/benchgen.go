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
// WHY THE OUTPUT IS COMMITTED, and not fetched at install time. A quality score
// is a percentage over whatever questions were asked, and autoTargetQuality
// reads every worker's score as one absolute 0–100 scale — so two workers are
// only comparable if they sat the same exam. A bank fetched at install time
// gives one router's workers a different exam from the next router's, and a
// worker profiled on Tuesday a different one from the same worker's neighbour
// profiled on Thursday, silently, with every number still looking plausible.
// Fetching from install-pre.sh or the Dockerfile has exactly that failure mode
// (and adds an egress dependency to a box that may not have one). Instead the
// generated file is committed: what the fleet is graded on is a reviewed diff in
// the source tree.
//
// NOT because the question set is part of the profile cache key. It was written
// that way and it is no longer true. LoadWorkerProfile keys on (id, model) and
// benchmarkVersion, and benchmarkVersion has stopped tracking the questions —
// it marks a change to the profiling METHOD (see benchmark.go). What tracks the
// questions now is each question's own content hash: prompt, expected answer,
// match mode and grader version (identity.go). Editing a question gives it a new
// qid and it is asked afresh; leaving one alone means its answers are served
// from the permacache and never re-earned.
//
// So a refresh no longer has a version bump attached to it by necessity, and the
// hazard the bump was named for did not go away with the mechanism: it moved. A
// worker's SCORE is only recomputed when that worker re-profiles, and only a
// benchmarkVersion bump makes the whole fleet do that, so a refresh landed
// without one leaves the fleet holding scores from the old bank until each
// worker next re-profiles for its own reasons. See benchUsage for the recipe
// that says so.

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
	// LiveBench's coding tasks carry NO ground_truth: the answer is a program,
	// and the only way to grade it is to run it against the question's test
	// cases. "code-exec" is therefore not a checkAnswer mode at all — checkAnswer
	// is pure and does no I/O — it is a marker that routes the question to the
	// execution sidecar instead. See codeexec.go.
	"LCB_generation":    benchMatchCodeExec,
	"coding_completion": benchMatchCodeExec,
}

// benchMatchCodeExec marks a question graded by RUNNING the answer. Kept as a
// named constant because it is tested for in three places and a typo would
// silently route a coding question into the "contains" default, where every
// program would grade as wrong.
const benchMatchCodeExec = "code-exec"

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
var benchDatasets = []string{"math", "reasoning", "coding"}

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
	// Code is set only for benchMatchCodeExec questions and carries what the
	// sidecar needs to run the answer. Nil for every string-graded question, so
	// pool.json for the maths/reasoning tasks is unchanged.
	Code *poolCode `json:"code,omitempty"`
}

// poolCode is the execution payload for a coding question.
//
// PrivateB64 stays COMPRESSED AND UNDECODED here on purpose. LiveBench ships it
// base64(zlib(pickle)) — one row is 3.4 MB — and unpickling runs arbitrary code.
// Decoding it on the machine running `bench fetch` would be doing exactly what
// the sidecar exists to contain, so the blob is carried opaquely and handed to
// the sidecar's /decode-private, which unpickles inside the jail.
type poolCode struct {
	Class   string `json:"class"`   // e.g. "Solution" — LeetCode entry class
	Func    string `json:"func"`    // metadata.func_name, the method to call
	Starter string `json:"starter"` // starter_code, so a completion task has its stub
	// Prefix is the PARTIAL SOLUTION a coding_completion answer continues, and
	// is empty for every other task. Starter is not a substitute: it holds the
	// three-line LeetCode stub, while the prompt shows a function truncated
	// part-way through its body and asks for "only the missing portion of the
	// code ... directly appending your code after the partial code should
	// produce a correct completion solution".
	//
	// Grading the answer on its own therefore grades a fragment, which never
	// compiles. That is not a hypothetical: it scored the two strongest workers
	// 0% on this task while the weakest scored 16%, because obeying the
	// instruction guaranteed a fail and ignoring it (emitting the whole
	// function) was the only way to pass. The task measured disobedience.
	Prefix     string `json:"prefix,omitempty"`
	PublicJSON string `json:"public"` // plain JSON list of {input,output,testtype}
	// PrivateB64 is deliberately NOT serialised into pool.json. The blobs total
	// ~250 MB across 128 questions and pool.json is a committed source artefact;
	// carrying them inline made it a 244 MB file. Each blob is written beside the
	// pool under private/<id>.b64 (gitignored) and read back on demand — see
	// benchWritePrivate / benchLoadPrivate.
	PrivateB64 string `json:"-"`
	HasPrivate bool   `json:"has_private,omitempty"`
}

// benchPrivateDir holds the undecoded private-test blobs, one file per question.
func benchPrivateDir() string { return filepath.Join(benchDataDir(), "private") }

func benchPrivatePath(id string) string {
	// LiveBench ids are content hashes, so they are already filename-safe; the
	// replace is belt and braces against a future id format with a separator in it.
	return filepath.Join(benchPrivateDir(), strings.NewReplacer("/", "_", "\\", "_", "..", "_").Replace(id)+".b64")
}

// benchWritePrivate stores one question's undecoded blob. Failure is fatal to the
// fetch rather than silent: a coding question whose private tests went missing
// would still look gradable and would quietly grade on public cases alone, which
// a model can satisfy by reading them out of its own prompt.
func benchWritePrivate(id, blob string) error {
	if blob == "" {
		return nil
	}
	if err := os.MkdirAll(benchPrivateDir(), 0o755); err != nil {
		return err
	}
	return os.WriteFile(benchPrivatePath(id), []byte(blob), 0o644)
}

// benchLoadPrivate reads one back. Empty (not an error) when the question has
// none, so callers can treat "no private tests" and "public only" alike.
func benchLoadPrivate(id string) (string, error) {
	buf, err := os.ReadFile(benchPrivatePath(id))
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

// benchCodePayload pulls the execution payload out of a LiveBench coding row.
// Returns false when the row is missing anything the sidecar would need, which
// is better than shipping a question that can only ever grade as wrong.
func benchCodePayload(row map[string]any) (*poolCode, bool) {
	// original_json arrives as a nested OBJECT from the datasets server, not as
	// the JSON string its name suggests — and asserting .(string) on it silently
	// rejected all 128 coding rows. Re-marshal whatever shape it is and decode
	// once, so either form works.
	rawOJ, ok := row["original_json"]
	if !ok || rawOJ == nil {
		return nil, false
	}
	buf, err := json.Marshal(rawOJ)
	if err != nil {
		return nil, false
	}
	if s, isStr := rawOJ.(string); isStr {
		buf = []byte(s)
	}
	var oj struct {
		StarterCode string `json:"starter_code"`
		Metadata    string `json:"metadata"`
	}
	if json.Unmarshal(buf, &oj) != nil {
		return nil, false
	}
	// metadata is a JSON string nested inside the JSON row, not an object.
	//
	// The error is CHECKED, and that is the whole point of this block. It used to
	// be `_ = json.Unmarshal(...)`, so a metadata blob that would not parse left
	// Func at its zero value and the row was returned ok=true anyway. Func is the
	// method the sandbox calls: sandbox/runner.py's resolve_entry raises
	// "no entry function was given for a functional test" the moment it is empty,
	// which fails EVERY case for EVERY worker. Calibration then measures p=0,
	// benchEmit files it under the ceiling band as headroom nobody can reach, and
	// it ships as a question no model can answer — indistinguishable, from the
	// outside, from a genuinely hard one.
	//
	// An ABSENT func_name is a different thing and stays legal: LiveCodeBench's
	// stdin-style problems have no entry point to name, resolve_entry is never
	// called for them, and 38 of this pool's 78 LCB_generation rows are that
	// shape. benchSelfGrades is what tells the two apart, by reading the test
	// cases' testtype.
	var md struct {
		FuncName string `json:"func_name"`
	}
	if oj.Metadata != "" {
		if err := json.Unmarshal([]byte(oj.Metadata), &md); err != nil {
			return nil, false
		}
	}
	pub, _ := row["public_test_cases"].(string)
	priv, _ := row["private_test_cases"].(string)
	if pub == "" && priv == "" {
		return nil, false // nothing to grade against
	}
	return &poolCode{
		Class:      benchEntryClass(oj.StarterCode),
		Func:       md.FuncName,
		Starter:    oj.StarterCode,
		PublicJSON: pub,
		PrivateB64: priv,
	}, true
}

// benchFillPrefixes stamps the completion prefix onto every coding_completion
// question. Run over the pool rather than at row-parse time because the prefix
// is derived from the assembled prompt, and because it has to be backfillable
// onto a pool.json fetched before the field existed — re-fetching 128 questions
// and ~250 MB of test blobs to recover something already sitting in the prompt
// would be a lot of network for no new information.
func benchFillPrefixes(qs []poolQuestion) (filled, missing int) {
	for i := range qs {
		if qs[i].Task != "coding_completion" || qs[i].Code == nil {
			continue
		}
		p := benchCompletionPrefix(qs[i].Prompt)
		if p == "" {
			missing++
			continue
		}
		qs[i].Code.Prefix = p
		filled++
	}
	return filled, missing
}

// benchCompletionFence matches a fenced code block, capturing its body.
var benchCompletionFence = regexp.MustCompile("(?s)```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)```")

// benchCompletionPrefix pulls the partial solution out of a coding_completion
// prompt: the answer is graded as prefix+answer, so without this the answer is
// graded as a bare fragment. See poolCode.Prefix for why that inverted the task.
//
// Derived from the prompt rather than from a dataset column because LiveBench
// does not ship one — starter_code is the empty stub, and the truncated body
// exists only inside the prompt's fenced block. Every one of the 50 completion
// prompts carries exactly one such block, so "the last fence" is unambiguous;
// returning "" when it is not leaves the old behaviour rather than inventing a
// prefix, and benchCodePrefixFor reports the question so it can be dropped.
func benchCompletionPrefix(prompt string) string {
	m := benchCompletionFence.FindAllStringSubmatch(prompt, -1)
	if len(m) == 0 {
		return ""
	}
	return m[len(m)-1][1]
}

// benchCodePrefixFor is the single place that decides whether a question's
// answer needs a prefix, so calibration and production grading cannot disagree
// about it — a question calibrated under one grader and served under another is
// calibrated for nothing.
func benchCodePrefixFor(q poolQuestion) string {
	if q.Task != "coding_completion" {
		return ""
	}
	return benchCompletionPrefix(q.Prompt)
}

// benchEntryClass reads the entry class out of the starter code ("class Solution:"
// -> "Solution"). Empty for a task with no class wrapper, which the sidecar reads
// as "call the function at module scope".
func benchEntryClass(starter string) string {
	for _, line := range strings.Split(starter, "\n") {
		line = strings.TrimSpace(line)
		rest, ok := strings.CutPrefix(line, "class ")
		if !ok {
			continue
		}
		if i := strings.IndexAny(rest, "(:"); i > 0 {
			return strings.TrimSpace(rest[:i])
		}
	}
	return ""
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
  bench emit [-sandbox URL]       Select items and write benchmark_data_live.go

Quarterly refresh. Monthly is not worth it for a set that only turns over every
six months, and calibrate costs hours of fleet time:

  go run . bench fetch
  go run . bench calibrate -router http://localhost:8585 -token "$ROUTER_ADMIN_KEY"
  go run . bench emit
  # then drop the replaced tiers from benchmark_data.go and commit that in the
  # SAME commit as the generated file.

benchmarkVersion is NOT part of that recipe any more, and this text used to say
it was. It no longer means "the question set changed" — it marks a change to the
profiling METHOD (benchmark.go), and a question's identity is now its own content
hash (identity.go), so a refresh cannot make a cached answer be reused for a
question it was not given. Nothing needs invalidating for correctness.

What a bump still buys is one exam for everybody. A worker holding a current
cached profile is never re-benchmarked, so after a refresh it keeps the score it
earned on the old bank while a newly registered worker earns one on the new bank
— and both are read on the same absolute 0-100 scale. Bump if that mixture
matters to you; it is cheap now, because re-profiling pays generations only for
the questions this refresh actually changed and serves the rest from the
permacache. It still parks every worker at provisional quality while it runs.
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
		sandbox := fs.String("sandbox", os.Getenv("SANDBOX_URL"), "code-execution sidecar base URL — required to grade the coding tasks")
		skip := fs.String("skip", "", "comma-separated backend ids to exclude — for a row the registry still calls ready whose engine is gone")
		_ = fs.Parse(args)
		if *router == "" {
			benchUsage()
			os.Exit(2)
		}
		err = benchCalibrate(*router, *token, *sandbox, *conc, *limit, benchSplitCSV(*skip))
	case "emit":
		fs := flag.NewFlagSet("emit", flag.ExitOnError)
		// Only needed when the selection contains coding questions: their test
		// cases arrive base64(zlib(pickle)) and only the sidecar may decode them.
		sandbox := fs.String("sandbox", os.Getenv("SANDBOX_URL"), "code-execution sidecar base URL — required if any code-exec question is selected")
		_ = fs.Parse(args)
		err = benchEmit(*sandbox)
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
	// Blobs to their own files before the pool is written, so a pool.json on disk
	// always has its private tests beside it.
	for i := range all {
		if all[i].Code == nil || all[i].Code.PrivateB64 == "" {
			continue
		}
		if err := benchWritePrivate(all[i].ID, all[i].Code.PrivateB64); err != nil {
			return fmt.Errorf("private tests for %s: %w", all[i].ID, err)
		}
		all[i].Code.HasPrivate = true
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
	if id == "" {
		return poolQuestion{}, false
	}
	match, allowed := benchMatchFor(task, gt)
	if !allowed {
		return poolQuestion{}, false
	}
	// Every other task is graded by comparing a string, so no ground truth means
	// nothing to compare against. A code-exec question is the exception by
	// design: its ground truth IS its test cases, carried separately below.
	if gt == "" && match != benchMatchCodeExec {
		return poolQuestion{}, false
	}
	var code *poolCode
	if match == benchMatchCodeExec {
		var ok bool
		if code, ok = benchCodePayload(row); !ok {
			return poolQuestion{}, false
		}
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
		Retired: removal != "", Code: code,
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
	// Backfill on every load, so calibrate and emit see identical questions
	// whether pool.json predates the field or not. Cheap: one regexp over 50
	// prompts.
	//
	// A MISS IS FATAL, not a warning. It used to print one and carry on, which
	// meant the failure mode was: benchSelfGrades rejects a completion question
	// with no prefix, emit counts it under "ungradeable", and the run ends with a
	// cheerful summary line — so the whole coding_completion task could vanish
	// from the bank without anything in the pipeline stopping. That is not a
	// hypothetical shape: it is why the shipped bank has zero executable coding
	// questions. A prompt with no fenced partial solution means the task's format
	// changed under benchCompletionPrefix, which is a decision for a human (fix
	// the extractor, or drop the task from benchAllowedTasks) and not something
	// to absorb silently.
	if _, missing := benchFillPrefixes(p.Questions); missing > 0 {
		var ids []string
		for _, q := range p.Questions {
			if q.Task == "coding_completion" && (q.Code == nil || q.Code.Prefix == "") {
				ids = append(ids, q.ID[:12])
			}
			if len(ids) == 5 {
				break
			}
		}
		return nil, fmt.Errorf("%d coding_completion questions carry no fenced partial solution in their prompt "+
			"(e.g. %s) — their answers are FRAGMENTS and would be graded standalone, which scores a correct "+
			"completion zero. Fix benchCompletionPrefix for the new prompt format, or drop coding_completion "+
			"from benchAllowedTasks and re-fetch", missing, strings.Join(ids, ", "))
	}
	return &p, nil
}
