package router

// code-exec — grading an answer by RUNNING it.
//
// Every other benchmark question is graded by comparing strings, which
// checkAnswer does in-process with no I/O. A LiveBench coding question has no
// ground truth at all: the answer is a program, and the only thing that decides
// whether it is right is executing it against the question's test cases. That
// cannot happen here. The router holds the fleet's API keys, its registry and
// its log database; running model-generated Python in that address space would
// put all of it one `import os` away from an untrusted program.
//
// So the execution lives in a separate hardened container (sandbox/), reached
// over localhost, and this file is the whole of the router's side of it. The
// split is the point: the worst a hostile submission can do is ruin its own
// grade and, at the very worst, the sidecar — which holds nothing and is
// restarted by dropshell.
//
// WHY THE PRIVATE TESTS ARE DECODED THERE TOO. LiveBench ships private test
// cases as base64(zlib(pickle)), and unpickling executes arbitrary code by
// construction. Decoding them here would hand the router exactly the capability
// the sidecar exists to contain, so the blob crosses opaquely and comes back as
// plain JSON — see benchgen.go's poolCode.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// benchCase is one test case: call the entry point with Input and expect Output.
// Testtype is "functional" (compare the return value) or "stdin" (run the
// submission as a script and compare stdout).
type benchCase struct {
	Input    string `json:"input"`
	Output   string `json:"output"`
	Testtype string `json:"testtype"`
}

// benchCode is what a code-exec question carries instead of an Expect string.
// Nil on every string-graded question, so nothing about the existing questions
// changes shape.
type benchCode struct {
	Class  string      `json:"class"`            // LeetCode entry class, "" for a module-level function
	Func   string      `json:"func"`             // method to call; "" for a stdin-only question
	Prefix string      `json:"prefix,omitempty"` // partial solution a completion answer continues
	Tests  []benchCase `json:"tests"`
}

// codeGradeRequest / codeGradeResult mirror the sidecar's HTTP contract. Kept as
// explicit structs rather than map[string]any so a field rename on either side
// is a compile error here rather than a question that silently grades as wrong.
type codeGradeRequest struct {
	Language string `json:"language"`
	Code     string `json:"code"`
	// Prefix is prepended to the submission before it runs, after the sidecar has
	// stripped markdown fences — a completion task's answer is only valid as a
	// continuation of the partial solution it was shown. Empty for every other
	// task. The join happens in the sidecar rather than here because the fences
	// have to come off first, and only the sidecar knows where they were.
	Prefix    string      `json:"prefix,omitempty"`
	Entry     codeEntry   `json:"entry"`
	Tests     []benchCase `json:"tests"`
	TimeoutMS int         `json:"timeout_ms"`
	MemoryMB  int         `json:"memory_mb"`
}

type codeEntry struct {
	Class string `json:"class"`
	Func  string `json:"func"`
}

type codeGradeResult struct {
	Pass        bool `json:"pass"`
	CasesRun    int  `json:"cases_run"`
	CasesPassed int  `json:"cases_passed"`
	// Error is the ONE thing that went wrong with the RUN, as against with a
	// case: see main.py's _run_error. A submission that simply computed the wrong
	// answer leaves it empty and sets pass:false, so a non-empty Error always
	// means the run did not finish normally — and who is at fault for that is
	// what codeRunFault decides.
	Error string `json:"error"`
	// PrefixApplied is the sidecar confirming it understood Prefix. A POINTER
	// because the three states differ: true (applied), false (none was sent) and
	// ABSENT (a sidecar too old to know about prefixes at all). Only the last is
	// a problem, and a plain bool could not express it.
	PrefixApplied *bool `json:"prefix_applied"`
}

// codeVerdict is what one graded submission actually did, as opposed to whether
// it passed.
//
// The boolean alone throws away the measurement the sidecar already made: 9 of
// 10 cases and 0 of 10 are the same false, and they are not the same fact about
// a model. One is an edge case missed, the other is a program that does not
// work — the exact distinction a strengths-and-weaknesses map exists to draw.
// The sidecar has sent cases_passed all along; nothing read it.
type codeVerdict struct {
	Pass        bool
	CasesRun    int
	CasesPassed int
}

func codeVerdictOf(out codeGradeResult) codeVerdict {
	return codeVerdict{Pass: out.Pass, CasesRun: out.CasesRun, CasesPassed: out.CasesPassed}
}

// Detail is the one-line account of the run, for the report stored against the
// question ("(sandbox: …)" in benchOne's BenchResult.Got, stderr in
// calibration). Cheap to read in a failure list and the difference between
// re-reading a model's answer and not bothering.
func (v codeVerdict) Detail() string {
	switch {
	case v.CasesRun == 0:
		return "no test cases ran"
	case v.Pass:
		return fmt.Sprintf("passed %d/%d cases", v.CasesPassed, v.CasesRun)
	default:
		return fmt.Sprintf("%d/%d cases passed", v.CasesPassed, v.CasesRun)
	}
}

// codeSandboxFaults are the run-level failures the sidecar authored about
// ITSELF, or about the QUESTION — never about the submission. Matching them is
// how an infrastructure fault stops being recorded as a capability fact.
//
// WHY THIS TABLE LISTS THE SIDECAR'S FAULTS AND NOT THE SUBMISSION'S, which is
// the direction that looks more natural: the submission's side is not
// enumerable. A program that fails to import is reported by whatever exception
// Python raised — "SyntaxError: …", "NameError while loading the submission: …"
// — which is unbounded free text. The sidecar's own messages are fixed strings
// in supervisor.py, runner.py and main.py, so they can be listed, and anything
// not on the list is the program. That also fixes the direction of the residual
// risk the right way round: a message a future sidecar adds grades as a wrong
// answer, which is what it does today, rather than silently excusing every
// model that writes broken Python.
//
// Kept in sync by hand, like benchSandboxBodyLimit. TestCodeRunFaultMatchesTheSandbox
// pins the strings against the sidecar's source so a rewording there is a test
// failure rather than a fleet quietly re-tiering itself.
var codeSandboxFaults = []struct {
	marker string
	// permanent distinguishes a question no worker can EVER be graded on from a
	// sidecar having a bad minute. The first should be dropped from the pool, the
	// second retried — see errBenchUngradeable.
	permanent bool
}{
	// supervisor.py, Outcome.spawn_error. The first of these is why this table
	// exists: /scratch is a 256 MB tmpfs shared by four concurrent runs under an
	// 8 MB RLIMIT_FSIZE, so ONE submission writing large files fails mkdtemp for
	// a DIFFERENT run — and that run recorded a wrong answer, for a different
	// worker, on a question whose program was fine.
	{marker: "could not create a scratch directory"},
	{marker: "could not open the scratch directory to"},
	{marker: "could not start the sandbox process"},
	{marker: "the sandbox process did not die after SIGKILL"},
	// runner.py: the grader's own crash, a request the sidecar could not parse,
	// and a mode it has never heard of (a router newer than its sandbox).
	{marker: "grader crashed: "},
	{marker: "unreadable request: "},
	{marker: "unknown runner mode "},
	// jail.py aborting rather than executing a submission as root — a deployment
	// fault (`docker run --user root`, a runtime that ignores USER), and one that
	// would otherwise fail every code question on that host.
	{marker: "refusing to execute as root"},
	// main.py's fallbacks for a child that died with nothing to say. Every death
	// a SUBMISSION can cause is converted into a fatal record carrying a message
	// first — its own SIGALRM/SIGXCPU handler, MemoryError under RLIMIT_AS, the
	// load-error path — and a fatal record outranks all three of these in
	// _run_error. So a death with no message is the host or the container rather
	// than the program: the cgroup OOM killer picking a victim, the pid limit, a
	// kill from outside. Ambiguous, and resolved towards "we could not tell",
	// because being unable to tell is not evidence against the model.
	{marker: "the sandbox reported a failure with no detail"},
	{marker: "the sandbox process was killed by signal "},
	{marker: "the sandbox process exited with status "},
	// A fault in the QUESTION's own data rather than in the answer: the private
	// test cases decoded to something the grader cannot use. No worker can ever
	// be scored on it.
	{marker: "malformed test case ", permanent: true},
	{marker: "malformed expected output", permanent: true},
	// Nothing in the answer reached the entry point: the graded function is still
	// the partial solution's truncated stub, so what ran was the PREFIX, not the
	// model's text. Ungradeable rather than wrong — the sandbox measured the
	// question's own scaffolding and learned nothing about the worker.
	//
	// Not permanent. This is usually the answer's shape (a bare restatement the
	// assembler could not place), which the next worker may not repeat; only if
	// every worker trips it is the QUESTION at fault, and the pass-rate floor
	// already catches that.
	// Matched on the tail: the message interpolates "{class}.{func}" mid-string,
	// so only the part after it is a literal to pin against.
	{marker: "as the partial solution's truncated stub"},
}

// codeRunFault reports why a graded run cannot be scored, or nil when the
// sidecar's error is the submission's own doing and pass:false is the truth.
//
// The failures it catches are all real and all distinguishable, and every one of
// them used to arrive at the caller as a wrong answer: a jail that would not
// start, a scratch directory that could not be made, test data that will never
// parse. Scoring those as wrong answers is how a fleet re-tiers itself around a
// broken dependency — the model that happened to be profiled while the sidecar
// was unwell is the model that gets routed less traffic afterwards.
//
// What it deliberately does NOT catch is the submission failing on its own
// terms: "time limit exceeded", "cpu limit exceeded", "memory limit exceeded",
// the result-size cap, and any exception the program raised while loading. Those
// are the sidecar answering the question it was asked — this program does not
// work — and excusing them would inflate the coding score of every model that
// writes a loop that never terminates.
func codeRunFault(out codeGradeResult) error {
	msg := strings.TrimSpace(out.Error)
	if msg == "" {
		return nil
	}
	v := codeVerdictOf(out)
	for _, f := range codeSandboxFaults {
		if !strings.Contains(msg, f.marker) {
			continue
		}
		if f.permanent {
			return fmt.Errorf("%w: %s (%s)", errBenchUngradeable, msg, v.Detail())
		}
		return fmt.Errorf("sandbox could not run the submission: %s (%s)", msg, v.Detail())
	}
	return nil
}

// checkPrefixHonoured fails a grade whose prefix the sidecar ignored.
//
// Version skew here is silent and total: an older sidecar drops the field, runs
// the fragment alone, and returns a confident pass:false for every completion
// question and every worker. The task then looks hard rather than broken, which
// is the exact failure this whole path was rewritten to fix. Reporting an error
// routes it to the ungraded path instead, where an outage already lives — a
// question we could not grade must never count as one the model got wrong.
func checkPrefixHonoured(sentPrefix string, out codeGradeResult) error {
	if sentPrefix == "" || (out.PrefixApplied != nil && *out.PrefixApplied) {
		return nil
	}
	return fmt.Errorf("sandbox ignored the completion prefix — it predates prefix support "+
		"and would grade the answer as a bare fragment; redeploy discrimen-sandbox (cases_run=%d)", out.CasesRun)
}

// codeExecDefaults. The timeout is per RUN, not per case: a submission gets one
// budget for the whole set, which is what stops a question with 100 private
// cases costing 100x a question with one.
const (
	codeExecTimeoutMS = 20000
	codeExecMemoryMB  = 512
	codeExecHTTPGrace = 30 * time.Second // sidecar wall-clock + slack, never less than the run budget
)

// gradeCode is the boolean-only form of gradeCodeVerdict, kept for the benchOne
// call site in benchmark.go. Callers that want the case fraction — which is the
// only thing separating "missed one edge case" from "wrote nothing that runs" —
// should take the verdict instead.
func (r *Router) gradeCode(ctx context.Context, code string, q benchmarkQ) (bool, error) {
	v, err := r.gradeCodeVerdict(ctx, code, q)
	return v.Pass, err
}

// gradeCodeVerdict runs one submission against one question's test cases and
// reports what happened to it.
//
// A run we could not grade returns an ERROR rather than false, and the
// distinction is the whole point: false means "the model got it wrong" and
// lowers its quality score, while an error means "we could not tell", which
// benchOne records as errored — the same treatment a worker that failed to
// answer gets, and the same reason. Scoring an outage as a wrong answer is how a
// fleet quietly re-tiers itself around a broken dependency.
//
// Three things reach that error path, not one. A sidecar that could not be
// reached or replied with a non-200; a sidecar too old to honour the completion
// prefix (checkPrefixHonoured); and — the one that was missing — a 200 whose
// body says the RUN failed rather than the submission (codeRunFault). The third
// is the most damaging of the three, because it is per-question and intermittent
// rather than obvious: a spawn failure caused by a neighbouring run filling
// /scratch looks exactly like a model that cannot code.
func (r *Router) gradeCodeVerdict(ctx context.Context, code string, q benchmarkQ) (codeVerdict, error) {
	var none codeVerdict
	if r.cfg == nil || strings.TrimSpace(r.cfg.SandboxURL) == "" {
		return none, fmt.Errorf("code-exec question but no sandbox configured (SANDBOX_URL)")
	}
	if q.Code == nil || len(q.Code.Tests) == 0 {
		return none, fmt.Errorf("code-exec question carries no test cases")
	}
	body, err := json.Marshal(codeGradeRequest{
		Language: "python", Code: code, Prefix: q.Code.Prefix,
		Entry:     codeEntry{Class: q.Code.Class, Func: q.Code.Func},
		Tests:     q.Code.Tests,
		TimeoutMS: codeExecTimeoutMS, MemoryMB: codeExecMemoryMB,
	})
	if err != nil {
		return none, err
	}
	url := strings.TrimSuffix(strings.TrimSpace(r.cfg.SandboxURL), "/") + "/grade"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return none, err
	}
	req.Header.Set("Content-Type", "application/json")
	if tok := strings.TrimSpace(r.cfg.SandboxToken); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := r.benchClient.Do(req)
	if err != nil {
		return none, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return none, fmt.Errorf("sandbox returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var out codeGradeResult
	if err := json.Unmarshal(raw, &out); err != nil {
		return none, err
	}
	if err := checkPrefixHonoured(q.Code.Prefix, out); err != nil {
		return none, err
	}
	// A submission that crashed, hung or ran out of memory is a WRONG ANSWER, not
	// an outage — the sidecar answered, and its answer is "this program does not
	// work". A sidecar that could not START the program is the opposite, and used
	// to be indistinguishable from it: same 200, same pass:false, the only
	// difference being an error field nothing read. codeRunFault is what reads it.
	if err := codeRunFault(out); err != nil {
		return none, err
	}
	return codeVerdictOf(out), nil
}

// benchGradeCodeStandalone is the calibration-time twin of gradeCode: same
// sidecar, same contract, but reachable without a *Router.
//
// `discrimen bench calibrate` runs as a CLI against a REMOTE router — it has a
// URL and a token, not a Config — so it cannot call the method. The duplication
// is deliberately confined to the transport: both build the same
// codeGradeRequest and read the same codeGradeResult, so a change to the
// contract is a compile error in both places rather than a drift that would
// make calibrated difficulty and production grading disagree about the same
// program.
//
// It also has to decode the private tests, which the router does not: at
// calibration time the question is still a poolQuestion carrying an opaque
// base64 blob, and only the sidecar may unpickle it.
func benchGradeCodeStandalone(sandboxURL string, q poolQuestion, code string) (bool, error) {
	if strings.TrimSpace(sandboxURL) == "" {
		return false, fmt.Errorf("code-exec question %s but no -sandbox URL given", q.ID)
	}
	if q.Code == nil {
		return false, fmt.Errorf("code-exec question %s carries no payload", q.ID)
	}
	tests, err := benchCodeTests(sandboxURL, q)
	if err != nil {
		return false, err
	}
	if len(tests) == 0 {
		return false, fmt.Errorf("code-exec question %s has no test cases", q.ID)
	}
	body, err := json.Marshal(codeGradeRequest{
		Language: "python", Code: code, Prefix: benchCodePrefixFor(q),
		Entry:     codeEntry{Class: q.Code.Class, Func: q.Code.Func},
		Tests:     tests,
		TimeoutMS: codeExecTimeoutMS, MemoryMB: codeExecMemoryMB,
	})
	if err != nil {
		return false, err
	}
	// A body the sidecar will refuse is a question that can never be graded, not
	// a worker that got it wrong. Caught here so it reports as ungradeable and is
	// dropped from the pool, rather than arriving as a 413 that every worker
	// records as a failure — which would make the question look maximally hard
	// and promote it straight into the top tier.
	if len(body) > benchSandboxBodyLimit {
		return false, fmt.Errorf("%w: %s needs %d bytes of test data, over the %d byte sidecar limit",
			errBenchUngradeable, q.ID, len(body), benchSandboxBodyLimit)
	}
	var out codeGradeResult
	if err := benchSandboxPost(sandboxURL, "/grade", body, &out); err != nil {
		return false, err
	}
	if err := checkPrefixHonoured(benchCodePrefixFor(q), out); err != nil {
		return false, err
	}
	// The same read of the same field as gradeCodeVerdict, and it matters MORE
	// here: calibration is what assigns a question its tier, so a run of spawn
	// failures does not merely mis-score one worker, it bakes an infrastructure
	// blip into the difficulty of the question for every worker afterwards. A
	// question every worker "failed" calibrates as maximally hard and promotes
	// straight into the top tier.
	if err := codeRunFault(out); err != nil {
		return false, fmt.Errorf("%s: %w", q.ID, err)
	}
	return out.Pass, nil
}

// benchSandboxBodyLimit mirrors the sidecar's own request cap (main.py). Kept in
// sync by hand: the sidecar must keep enforcing it regardless of what a client
// believes, so this is a courtesy check, not the boundary.
const benchSandboxBodyLimit = 32 << 20

// errBenchUngradeable marks a question that no worker can ever be scored on —
// as distinct from a sidecar that happens to be down. The first must be dropped
// from the pool; the second must be retried, and recording it as a wrong answer
// is how a whole task silently calibrates to zero.
//
// Carried but not yet acted on: benchgen_calibrate.go's caller treats every
// error alike and reports benchGradeUngraded, which is the SAFE half of the
// distinction (the pair goes unrecorded and a later run retries) and not the
// whole of it — a question with unusable test data is retried forever instead of
// being dropped. Wrapping it here is what makes the drop an errors.Is away.
var errBenchUngradeable = errors.New("ungradeable question")

// benchCodeTests assembles a question's cases: the public ones, which are plain
// JSON, plus the private ones, which only the sidecar may decode.
//
// Public cases alone are not enough to grade with. They are printed in the
// question's own prompt, so a model that simply hardcodes them scores full marks
// — which measures prompt-reading, not programming. The private cases are the
// actual test.
func benchCodeTests(sandboxURL string, q poolQuestion) ([]benchCase, error) {
	var tests []benchCase
	if q.Code.PublicJSON != "" {
		if err := json.Unmarshal([]byte(q.Code.PublicJSON), &tests); err != nil {
			return nil, fmt.Errorf("public tests for %s: %w", q.ID, err)
		}
	}
	if !q.Code.HasPrivate {
		return tests, nil
	}
	blob, err := benchLoadPrivate(q.ID)
	if err != nil || blob == "" {
		// The blobs live beside pool.json and are gitignored, so a fresh clone has
		// the questions and not the tests. Say so rather than grading on public
		// cases alone and quietly reporting a softer number.
		return nil, fmt.Errorf("private tests for %s missing — re-run `discrimen bench fetch`", q.ID)
	}
	body, _ := json.Marshal(map[string]string{"blob": blob})
	var decoded struct {
		Tests []benchCase `json:"tests"`
	}
	if err := benchSandboxPost(sandboxURL, "/decode-private", body, &decoded); err != nil {
		return nil, err
	}
	return append(tests, decoded.Tests...), nil
}

// benchSandboxPost is one JSON round trip to the sidecar.
func benchSandboxPost(sandboxURL, path string, body []byte, out any) error {
	url := strings.TrimSuffix(strings.TrimSpace(sandboxURL), "/") + path
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if tok := strings.TrimSpace(os.Getenv("SANDBOX_TOKEN")); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	client := &http.Client{Timeout: codeExecHTTPGrace}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("sandbox %s returned %d: %s", path, resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	return json.Unmarshal(raw, out)
}
