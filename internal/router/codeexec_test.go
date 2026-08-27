package router

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// codeSidecar stands in for the hardened container: one canned /grade response,
// served over real HTTP so the whole of gradeCodeVerdict runs — marshalling,
// transport, status check, decode, and the two checks on the decoded body.
func codeSidecar(t *testing.T, body codeGradeResult) *Router {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(body)
	}))
	t.Cleanup(srv.Close)
	return &Router{cfg: &Config{SandboxURL: srv.URL}, benchClient: &http.Client{}}
}

func codeQuestion() benchmarkQ {
	return benchmarkQ{
		Tier: 12, Match: benchMatchCodeExec,
		Prompt: "### Instructions: You are an expert Python programmer.",
		Code: &benchCode{
			Func:  "solve",
			Tests: []benchCase{{Input: "1", Output: "1", Testtype: "functional"}},
		},
	}
}

// The distinction the whole file turns on: the sidecar returns 200 and
// pass:false for a program that computed the wrong answer AND for a program it
// could not start, and only the error field tells them apart. Nothing read that
// field, so a jail that would not start, a scratch directory that could not be
// made and test data that will never parse all arrived at benchOne as evidence
// that the model cannot code.
//
// The concrete one is the first case. /scratch is a 256 MB tmpfs shared by four
// concurrent runs under an 8 MB RLIMIT_FSIZE, so one submission writing large
// files fails mkdtemp for a different run — a different worker, on a question
// its program actually solved, recorded as a wrong answer.
func TestGradeCodeTellsAnOutageFromAWrongAnswer(t *testing.T) {
	cases := []struct {
		name     string
		sandbox  string // the sidecar's error field, verbatim from its source
		ungrade  bool   // true: we could not grade this. false: the model got it wrong.
		dropable bool   // and further: this QUESTION can never be graded by anyone
	}{
		// Infrastructure. None of these is a fact about the model.
		{name: "scratch directory could not be made", ungrade: true,
			sandbox: "could not create a scratch directory: [Errno 28] No space left on device: '/scratch/grade-xk3'"},
		{name: "scratch directory could not be opened to the run uid", ungrade: true,
			sandbox: "could not open the scratch directory to 65534: [Errno 1] Operation not permitted"},
		{name: "the process could not be spawned", ungrade: true,
			sandbox: "could not start the sandbox process: [Errno 11] Resource temporarily unavailable"},
		{name: "the process would not die", ungrade: true,
			sandbox: "the sandbox process did not die after SIGKILL"},
		{name: "the grader itself crashed", ungrade: true,
			sandbox: "grader crashed: KeyError: 'testtype'"},
		{name: "the sidecar could not parse our own request", ungrade: true,
			sandbox: "unreadable request: Expecting value: line 1 column 1 (char 0)"},
		{name: "router newer than its sandbox", ungrade: true,
			sandbox: "unknown runner mode 'grade2'"},
		{name: "deployed as root, so the jail refused to run", ungrade: true,
			sandbox: "could not drop privileges to 65534:65534; refusing to execute as root"},
		{name: "the child died with nothing to say", ungrade: true,
			sandbox: "the sandbox reported a failure with no detail"},
		{name: "killed from outside — the container OOM killer picking a victim", ungrade: true,
			sandbox: "the sandbox process was killed by signal 9"},
		{name: "died before finishing and left no account of why", ungrade: true,
			sandbox: "the sandbox process exited with status 137 before finishing"},
		// The question's own data. Ungradeable for every worker, forever.
		{name: "test data that will never parse", ungrade: true, dropable: true,
			sandbox: "malformed test case 3: expected output is not valid JSON"},

		// The submission failing on its own terms. These ARE wrong answers: the
		// sidecar answered the question it was asked, and its answer is "this
		// program does not work". Excusing them would hand a free pass to every
		// model that writes a loop that never terminates.
		{name: "the program never terminated", sandbox: "time limit exceeded"},
		{name: "the program burned its cpu budget", sandbox: "cpu limit exceeded"},
		{name: "the parent's wall clock caught it", sandbox: "timed out after 20000 ms"},
		{name: "the program allocated past its rlimit", sandbox: "memory limit exceeded"},
		{name: "the program does not parse", sandbox: "SyntaxError: invalid syntax (<submission>, line 3)"},
		{name: "the program raised while loading", sandbox: "NameError while loading the submission: name 'heapq' is not defined"},
		{name: "the program printed more than the cap", sandbox: "the submission produced more than 67108864 bytes of results"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := codeSidecar(t, codeGradeResult{Pass: false, CasesRun: 1, CasesPassed: 0, Error: c.sandbox})
			v, err := r.gradeCodeVerdict(t.Context(), "def solve(): pass", codeQuestion())
			switch {
			case c.ungrade && err == nil:
				t.Fatalf("%q graded as a WRONG ANSWER (pass=%v). An infrastructure fault recorded as a "+
					"capability fact is how a fleet re-tiers itself around a broken dependency.", c.sandbox, v.Pass)
			case !c.ungrade && err != nil:
				t.Fatalf("%q reported as ungradeable (%v) — the sidecar answered, and its answer is that "+
					"the program does not work. Excusing it inflates the coding score.", c.sandbox, err)
			}
			if err == nil {
				return
			}
			if got := errors.Is(err, errBenchUngradeable); got != c.dropable {
				t.Errorf("errors.Is(err, errBenchUngradeable) = %v, want %v — a question with unusable test "+
					"data must be distinguishable from a sidecar having a bad minute: %v", got, c.dropable, err)
			}
			// The message has to name the fault, or an operator reading a failure
			// list has no way to tell a sandbox outage from a bad model.
			if !strings.Contains(err.Error(), c.sandbox) {
				t.Errorf("error %q does not carry the sidecar's own account (%q)", err, c.sandbox)
			}
		})
	}
}

// A pass and a wrong answer both survive the fault check untouched — the guard
// must not become a second grader.
func TestGradeCodeLeavesACleanRunAlone(t *testing.T) {
	for _, pass := range []bool{true, false} {
		r := codeSidecar(t, codeGradeResult{Pass: pass, CasesRun: 10, CasesPassed: 10})
		v, err := r.gradeCodeVerdict(t.Context(), "def solve(): pass", codeQuestion())
		if err != nil {
			t.Fatalf("pass=%v: a run with no error field was reported ungradeable: %v", pass, err)
		}
		if v.Pass != pass {
			t.Errorf("verdict %v, sidecar said %v", v.Pass, pass)
		}
		// And the boolean wrapper the benchmark still calls agrees with it.
		got, err := r.gradeCode(t.Context(), "def solve(): pass", codeQuestion())
		if err != nil || got != pass {
			t.Errorf("gradeCode = (%v, %v), want (%v, nil) — the two forms must not drift", got, err, pass)
		}
	}
}

// cases_passed has been on the wire all along and nothing read it, so 9 of 10
// cases and 0 of 10 were the same false. They are not the same fact about a
// model: one is an edge case missed, the other is a program that does not work,
// and telling them apart is exactly what a strengths-and-weaknesses map is for.
func TestGradeCodeSurfacesTheCaseFraction(t *testing.T) {
	r := codeSidecar(t, codeGradeResult{Pass: false, CasesRun: 10, CasesPassed: 9})
	v, err := r.gradeCodeVerdict(t.Context(), "def solve(): pass", codeQuestion())
	if err != nil {
		t.Fatalf("gradeCodeVerdict: %v", err)
	}
	if v.CasesRun != 10 || v.CasesPassed != 9 {
		t.Fatalf("verdict = %+v, want 9 of 10 cases — the sidecar sent the fraction and it was dropped", v)
	}
	if got := v.Detail(); got != "9/10 cases passed" {
		t.Errorf("Detail() = %q, want %q", got, "9/10 cases passed")
	}
	if got := (codeVerdict{Pass: true, CasesRun: 10, CasesPassed: 10}).Detail(); got != "passed 10/10 cases" {
		t.Errorf("Detail() on a pass = %q", got)
	}
	if got := (codeVerdict{}).Detail(); got != "no test cases ran" {
		t.Errorf("Detail() with nothing run = %q — zero of zero must not read as a score", got)
	}
	// And it rides along on the ungradeable path too, which is the only channel
	// that reaches the caller without a change in benchmark.go.
	r = codeSidecar(t, codeGradeResult{CasesRun: 4, CasesPassed: 3,
		Error: "could not create a scratch directory: [Errno 28] No space left on device"})
	if _, err = r.gradeCodeVerdict(t.Context(), "def solve(): pass", codeQuestion()); err == nil {
		t.Fatal("a spawn failure graded as a wrong answer")
	} else if !strings.Contains(err.Error(), "3/4 cases passed") {
		t.Errorf("the outage report drops the case fraction: %v", err)
	}
}

// codeSandboxFaults matches PROSE, and the prose lives in another language in
// another directory. That is safe only for as long as the sidecar keeps
// spelling these failures the same way, which is exactly the kind of thing a
// rewrite changes without anybody noticing — and the failure mode is silent:
// every unmatched message goes back to being recorded as a wrong answer.
//
// Developer-side only. The router image ships without sandbox/, so a missing
// directory skips rather than fails.
func TestCodeRunFaultMatchesTheSandbox(t *testing.T) {
	dir := filepath.Join("..", "..", "sandbox")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no sandbox source to check against: %v", err)
	}
	var src strings.Builder
	for _, name := range []string{"main.py", "supervisor.py", "runner.py", "jail.py"} {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		src.Write(b)
	}
	all := src.String()
	for _, f := range codeSandboxFaults {
		if !strings.Contains(all, f.marker) {
			t.Errorf("no sidecar source emits %q any more — the message was reworded, and every run that "+
				"hits it is now recorded as a model getting the question wrong. Re-read main.py's "+
				"_run_error and update codeSandboxFaults.", f.marker)
		}
	}
}

// The other direction, and the one that keeps the table honest: a message the
// SUBMISSION caused must never be on it. Over-matching here is not a smaller
// version of the same bug, it is the opposite one — every model that writes an
// infinite loop would have the question dropped from its denominator instead of
// counted against it, and the coding score would climb for the worst reason.
func TestCodeRunFaultDoesNotExcuseTheSubmission(t *testing.T) {
	for _, msg := range []string{
		"time limit exceeded",
		"cpu limit exceeded",
		"timed out after 20000 ms",
		"memory limit exceeded",
		"the submission produced more than 67108864 bytes of results",
		"SyntaxError: unexpected EOF while parsing (<submission>, line 12)",
		"AttributeError while loading the submission: 'Solution' object has no attribute 'dp'",
		"TypeError: solve() missing 1 required positional argument: 'nums'",
	} {
		if err := codeRunFault(codeGradeResult{Error: msg}); err != nil {
			t.Errorf("%q was excused as an outage: %v", msg, err)
		}
	}
	if err := codeRunFault(codeGradeResult{}); err != nil {
		t.Errorf("an empty error field is a normal grade, not a fault: %v", err)
	}
}
