package router

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A question with unusable test data has to be told apart from a sidecar having
// a bad minute, and for the whole life of errBenchUngradeable it was not:
// benchGradeOne reported benchGradeUngraded for every error alike, so the pair
// went unrecorded, the resume path picked it up again, and the question was
// re-asked on every run, on every backend, forever — at the hard-tier token
// budget, which is the most expensive question in the pool to keep getting
// nowhere with.
//
// Both directions matter and the test asserts both. Over-classifying is the
// worse mistake of the two: a sidecar that was briefly down would have its
// questions written off permanently on the strength of one bad minute.
func TestCalibrationDropsAQuestionItCanNeverGrade(t *testing.T) {
	cases := []struct {
		name    string
		sandbox string // the sidecar's error field, verbatim from its source
		want    benchGradeKind
	}{
		{
			// main.py's _decide, lifted to the run-level error field by _assemble:
			// the row's own expected output is not the JSON the grader needs, so no
			// answer from any worker can ever be compared against it.
			name:    "the question's answer key will not parse",
			sandbox: "malformed expected output at case 3: could not decode expected output as JSON: Expecting value: line 1 column 1 (char 0)",
			want:    benchGradeUngradeable,
		},
		{
			// runner.py, parsing a case's input inside the jail.
			name:    "the question's test input will not parse",
			sandbox: "malformed test case 7: could not decode test input as JSON arguments: Extra data",
			want:    benchGradeUngradeable,
		},
		{
			// supervisor.py. A neighbouring run filled /scratch. Nothing about this
			// question, and trying again is exactly the right response.
			name:    "the sidecar had a bad minute",
			sandbox: "could not create a scratch directory: [Errno 28] No space left on device: '/scratch/grade-xk3'",
			want:    benchGradeUngraded,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			sandbox := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(codeGradeResult{Pass: false, CasesRun: 3, CasesPassed: 0, Error: c.sandbox})
			}))
			defer sandbox.Close()

			router := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					"choices": []map[string]any{{
						"message":       map[string]any{"role": "assistant", "content": "def solve(n):\n    return n + 1\n"},
						"finish_reason": "stop",
					}},
				})
			}))
			defer router.Close()

			q := poolQuestion{
				ID: "lcb-0001", Task: "LCB_generation", Match: benchMatchCodeExec,
				Prompt: "Write solve(n).",
				Code: &poolCode{
					Func:       "solve",
					PublicJSON: `[{"input":"1","output":"2","testtype":"functional"}]`,
				},
			}
			pass, kind := benchGradeOne(router.URL, "", "worker-0", sandbox.URL, q, &benchBusyTracker{})
			if pass {
				t.Error("a run the sidecar could not complete was reported as a PASS")
			}
			if kind != c.want {
				t.Fatalf("kind = %v, want %v.\n"+
					"ungradeable is recorded and never asked again; ungraded is left unrecorded and retried "+
					"on the next run. Reporting a permanently broken question as the second is an infinite "+
					"retry loop; reporting a transient outage as the first writes a good question off.",
					kind, c.want)
			}
		})
	}
}

// The other half of the fix: what calibration recorded, emit has to act on. A
// question no worker can ever be scored on fails for EVERY worker, so left in
// the pool it lands in the p==0 bucket and is selected as CEILING BAND HEADROOM
// — shipped as a question nobody can answer, indistinguishable from a genuinely
// hard one, costing a slot in every profiling run from then on.
func TestEmitDropsAQuestionCalibrationCouldNotGrade(t *testing.T) {
	const broken = "ceil-code-0"

	// The control first: with nothing flagged, this question IS selected. Without
	// this the test below would pass just as well against a fixture that never
	// selected it in the first place.
	pool, calib := benchSelectFixture()
	before, _, err := benchSelect(pool, calib)
	if err != nil {
		t.Fatalf("benchSelect: %v", err)
	}
	if !benchSelectedContains(before, broken) {
		t.Fatalf("%s is not selected even before it is flagged; the fixture no longer exercises this", broken)
	}

	// ONE backend's report is enough, and it is the last one — the point being
	// that this is not a vote. "The answer key is not JSON" is a fact about the
	// question, and only a backend whose submission actually ran gets far enough
	// to discover it, so the one that reports it may be the only one that could.
	pool, calib = benchSelectFixture()
	last := calib.Results[len(calib.Results)-1]
	last.Ungradeable = map[string]bool{broken: true}

	after, _, err := benchSelect(pool, calib)
	if err != nil {
		t.Fatalf("benchSelect: %v", err)
	}
	if benchSelectedContains(after, broken) {
		t.Errorf("%s was selected after calibration reported it ungradeable — it fails for every worker, "+
			"so it measures p=0 and ships in the ceiling band as headroom nobody can reach", broken)
	}
	if len(after) != len(before)-1 {
		t.Errorf("selection went from %d questions to %d; exactly one should have been dropped",
			len(before), len(after))
	}
	// Nothing else moved: a broken question must not take its neighbours with it.
	for _, s := range after {
		if !benchSelectedContains(before, s.q.ID) {
			t.Errorf("%s appeared only after the flag was set", s.q.ID)
		}
	}
}

func benchSelectedContains(selected []scoredQuestion, id string) bool {
	for _, s := range selected {
		if s.q.ID == id {
			return true
		}
	}
	return false
}

// An ungradeable pair is WRITTEN DOWN, unlike an ungraded one, and that is the
// whole of why it stops being retried: benchCalibrate's resume path keys on the
// question id being present in Pass. The false it stores there is never read as
// a fail — benchSelect drops the question outright — but it has to be there.
func TestUngradeableIsRecordedSoTheResumePathSkipsIt(t *testing.T) {
	res := &calibResult{Backend: "worker-0", Pass: map[string]bool{}}
	// What benchCalibrate does with a benchGradeUngradeable verdict.
	res.Pass["q-broken"] = false
	res.Ungradeable = map[string]bool{"q-broken": true}

	// The resume path, verbatim from benchCalibrate.
	todo := 0
	for _, id := range []string{"q-broken", "q-fresh"} {
		if _, done := res.Pass[id]; !done {
			todo++
		}
	}
	if todo != 1 {
		t.Fatalf("the resume path would re-ask %d of 2 questions, want 1 — a permanently ungradeable "+
			"question left absent from Pass is re-asked on every run, on every backend, forever", todo)
	}

	// And it survives the round trip through calibration.json, or the skip lasts
	// only until the process that discovered it exits.
	blob, err := json.Marshal(&calibration{Results: []*calibResult{res}})
	if err != nil {
		t.Fatal(err)
	}
	var back calibration
	if err := json.Unmarshal(blob, &back); err != nil {
		t.Fatal(err)
	}
	if got := back.forBackend("worker-0"); got == nil || !got.Ungradeable["q-broken"] {
		t.Fatalf("Ungradeable did not survive the checkpoint: %s", blob)
	}
}

// The sidecar's own request cap is mirrored here by hand, and a body over it is
// a question that can never be graded rather than a worker that got it wrong —
// so it takes the same permanent path as unusable test data. Pinned because the
// wrapping is what carries the distinction: a bare fmt.Errorf here would report
// as a transient outage and the question would be retried forever.
func TestAnOversizedRequestIsUngradeableRatherThanRetried(t *testing.T) {
	oversize, err := json.Marshal([]benchCase{{
		Input:    strings.Repeat("x", benchSandboxBodyLimit+(1<<20)),
		Output:   "1",
		Testtype: "functional",
	}})
	if err != nil {
		t.Fatal(err)
	}
	q := poolQuestion{
		ID: "lcb-huge", Task: "LCB_generation", Match: benchMatchCodeExec, Prompt: "Write solve(n).",
		Code: &poolCode{Func: "solve", PublicJSON: string(oversize)},
	}
	// The sidecar is deliberately a hole in the ground. The size check is meant
	// to fire before anything is sent, so if the request ever reaches this the
	// connection refusal is what the test sees — and a refused connection is a
	// transient outage, which is the wrong answer.
	_, err = benchGradeCodeStandalone("http://127.0.0.1:1", q, "def solve(n): return n")
	if err == nil {
		t.Fatal("an over-limit request was accepted")
	}
	if !errors.Is(err, errBenchUngradeable) {
		t.Errorf("an over-limit body reported as a retryable fault (%v) — every worker would record the "+
			"sidecar's 413 as a failure, the question would calibrate as maximally hard, and it would be "+
			"promoted straight into the top tier", err)
	}
}
