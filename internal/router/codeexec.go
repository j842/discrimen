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
	Pass        bool   `json:"pass"`
	CasesRun    int    `json:"cases_run"`
	CasesPassed int    `json:"cases_passed"`
	Error       string `json:"error"`
}

// codeExecDefaults. The timeout is per RUN, not per case: a submission gets one
// budget for the whole set, which is what stops a question with 100 private
// cases costing 100x a question with one.
const (
	codeExecTimeoutMS = 20000
	codeExecMemoryMB  = 512
	codeExecHTTPGrace = 30 * time.Second // sidecar wall-clock + slack, never less than the run budget
)

// gradeCode runs one submission against one question's test cases and reports
// whether it passed ALL of them.
//
// A sidecar that is unreachable returns an error rather than false. The
// distinction matters: false means "the model got it wrong" and lowers its
// quality score, while an error means "we could not tell", which benchOne
// records as errored — the same treatment a worker that failed to answer gets,
// and the same reason. Scoring an outage as a wrong answer is how a fleet
// quietly re-tiers itself around a broken dependency.
func (r *Router) gradeCode(ctx context.Context, code string, q benchmarkQ) (bool, error) {
	if r.cfg == nil || strings.TrimSpace(r.cfg.SandboxURL) == "" {
		return false, fmt.Errorf("code-exec question but no sandbox configured (SANDBOX_URL)")
	}
	if q.Code == nil || len(q.Code.Tests) == 0 {
		return false, fmt.Errorf("code-exec question carries no test cases")
	}
	body, err := json.Marshal(codeGradeRequest{
		Language: "python", Code: code, Prefix: q.Code.Prefix,
		Entry:     codeEntry{Class: q.Code.Class, Func: q.Code.Func},
		Tests:     q.Code.Tests,
		TimeoutMS: codeExecTimeoutMS, MemoryMB: codeExecMemoryMB,
	})
	if err != nil {
		return false, err
	}
	url := strings.TrimSuffix(strings.TrimSpace(r.cfg.SandboxURL), "/") + "/grade"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return false, err
	}
	req.Header.Set("Content-Type", "application/json")
	if tok := strings.TrimSpace(r.cfg.SandboxToken); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := r.benchClient.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("sandbox returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var out codeGradeResult
	if err := json.Unmarshal(raw, &out); err != nil {
		return false, err
	}
	// A submission that crashed or timed out is a WRONG ANSWER, not an outage —
	// the sidecar answered, and its answer is "this program does not work". Only
	// a sidecar that could not be reached or could not reply reaches the error
	// path above.
	return out.Pass, nil
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
