package router

import (
	"math"
	"testing"
)

// The replies below are the shapes a reasoning model actually produces on
// LiveBench's zebra_puzzle and olympiad items: a long scratchpad that
// enumerates candidate orderings, then a final line that commits to one. The
// grader must read the commitment and must not be rescued by a correct
// intermediate the model later abandoned.
func TestCheckAnswerExactList(t *testing.T) {
	q := benchmarkQ{Match: "exact-list", Expect: "1, filmmaking, police-officer, journalist"}

	cases := []struct {
		name   string
		answer string
		want   bool
	}{
		{"bare answer", "1, filmmaking, police-officer, journalist", true},
		{"labelled", "The answer is: 1, filmmaking, police-officer, journalist", true},
		{"spacing and case differ", "1,Filmmaking,  POLICE-OFFICER ,journalist", true},
		{"trailing full stop", "1, filmmaking, police-officer, journalist.", true},
		{"markdown emphasis", "**1**, *filmmaking*, police-officer, journalist", true},
		{
			name: "commits on the last line after working",
			answer: "Let me try 2, journalist, filmmaking, police-officer.\n" +
				"That violates the second constraint.\n" +
				"1, filmmaking, police-officer, journalist",
			want: true,
		},
		{
			// The whole point of reading from the end: a correct ordering written
			// while exploring must not pass an answer the model then abandoned.
			name: "abandoned intermediate does not rescue a wrong conclusion",
			answer: "Suppose 1, filmmaking, police-officer, journalist.\n" +
				"On reflection that's wrong.\n" +
				"Answer: 2, journalist, filmmaking, police-officer",
			want: false,
		},
		{"wrong order is wrong", "1, police-officer, filmmaking, journalist", false},
		{"missing element", "1, filmmaking, police-officer", false},
		{"extra element", "1, filmmaking, police-officer, journalist, teacher", false},
		{"empty", "", false},
		{"prose only", "I am not sure about this one.", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := checkAnswer(q, tc.answer); got != tc.want {
				t.Errorf("checkAnswer(%q) = %v, want %v", tc.answer, got, tc.want)
			}
		})
	}
}

// A numeric ground truth from LiveBench's math_comp is AIME-style zero-padded
// ("073"); the existing numeric branch compares by value, so it must not care.
func TestCheckAnswerZeroPaddedNumeric(t *testing.T) {
	q := benchmarkQ{Match: "numeric", Expect: "073"}
	for _, answer := range []string{"73", "073", "The answer is 73.", "so the final answer is \\boxed{73}"} {
		if !checkAnswer(q, answer) {
			t.Errorf("checkAnswer(%q) = false, want true", answer)
		}
	}
	if checkAnswer(q, "The answer is 74.") {
		t.Error("74 graded as a pass against 073")
	}
}

// LiveBench's zebra_puzzle prompts demand "<solution>a, b, c</solution>".
// Without stripping the wrapper the first and last elements never match and
// every zebra_puzzle item grades as a miss.
func TestCheckAnswerExactListSolutionTag(t *testing.T) {
	q := benchmarkQ{Match: "exact-list", Expect: "1, 2, romance, japanese"}
	for _, answer := range []string{
		"<solution>1, 2, romance, japanese</solution>",
		"Working through the constraints…\n<solution>1, 2, romance, japanese</solution>",
		"<solution>\n1, 2, romance, japanese\n</solution>",
	} {
		if !checkAnswer(q, answer) {
			t.Errorf("checkAnswer(%q) = false, want true", answer)
		}
	}
	if checkAnswer(q, "<solution>2, 1, romance, japanese</solution>") {
		t.Error("a wrong ordering inside the tag graded as a pass")
	}
}

// LiveBench's AMC items ask for the option letter five times over ("DDDDD"),
// which benchLetterRe cannot read because it only matches standalone letters.
func TestCheckAnswerMCQRepeat(t *testing.T) {
	q := benchmarkQ{Match: "mcq-repeat", Expect: "D"}
	pass := []string{
		"DDDDD",
		"The answer is D, so: DDDDD",
		"…therefore the answer is 5, which is choice D.\nDDDDD",
		"DDD", // a model that miscounts the repetition still committed to D
		"D",   // ignored the instruction; the mcq fallback handles it
		"Answer: D",
	}
	for _, a := range pass {
		if !checkAnswer(q, a) {
			t.Errorf("checkAnswer(%q) = false, want true", a)
		}
	}
	fail := []string{"EEEEE", "The answer is B", "AAAAA"}
	for _, a := range fail {
		if checkAnswer(q, a) {
			t.Errorf("checkAnswer(%q) = true, want false", a)
		}
	}
	// A model that echoes the prompt's own example ("if the answer is F, write
	// FFFFF") before committing must be read on its commitment, not the example.
	if !checkAnswer(q, "For example if the answer were F you would write FFFFF.\nMy answer: DDDDD") {
		t.Error("the echoed example beat the model's actual pick")
	}
}

// math_comp mixes AMC letters with AIME integers, so the grader has to come
// from the ground truth rather than the task name.
func TestBenchMatchForMathComp(t *testing.T) {
	cases := map[string]string{"D": "mcq-repeat", "A": "mcq-repeat", "073": "numeric", "236": "numeric"}
	for gt, want := range cases {
		if got, ok := benchMatchFor("math_comp", gt); !ok || got != want {
			t.Errorf("benchMatchFor(math_comp, %q) = %q, want %q", gt, got, want)
		}
	}
	if got, _ := benchMatchFor("zebra_puzzle", "a, b"); got != "exact-list" {
		t.Errorf("zebra_puzzle = %q, want exact-list", got)
	}
	if _, ok := benchMatchFor("AMPS_Hard", "x^2"); ok {
		t.Error("AMPS_Hard should not be allowlisted — its answers need symbolic comparison")
	}
}

// A prompt long enough to eat the thinking budget on a 16K-context worker gets
// excluded, or its truncation would read as a capability difference.
func TestBenchToPoolQuestionPromptCap(t *testing.T) {
	long := make([]byte, benchMaxPromptChars+1)
	for i := range long {
		long[i] = 'x'
	}
	row := map[string]any{
		"task": "olympiad", "question_id": "a", "ground_truth": "1,2,3",
		"turns": []any{string(long)},
	}
	if _, ok := benchToPoolQuestion(row, "math"); ok {
		t.Error("an over-long prompt should be excluded")
	}
	row["turns"] = []any{string(long[:benchMaxPromptChars])}
	if _, ok := benchToPoolQuestion(row, "math"); !ok {
		t.Error("a prompt exactly at the cap should be kept")
	}
}

func TestBenchSpearman(t *testing.T) {
	cases := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"identical ranking", []float64{1, 2, 3, 4}, []float64{10, 20, 30, 40}, 1},
		{"exactly inverted", []float64{1, 2, 3, 4}, []float64{40, 30, 20, 10}, -1},
		{"too short to rank", []float64{1}, []float64{1}, 0},
		{"mismatched lengths", []float64{1, 2}, []float64{1}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := benchSpearman(tc.a, tc.b); math.Abs(got-tc.want) > 1e-9 {
				t.Errorf("benchSpearman = %v, want %v", got, tc.want)
			}
		})
	}
	// One transposition in the middle of four: rho = 1 - 6*2/(4*15) = 0.8.
	if got := benchSpearman([]float64{1, 2, 3, 4}, []float64{10, 30, 20, 40}); math.Abs(got-0.8) > 1e-9 {
		t.Errorf("single transposition: got %v, want 0.8", got)
	}
}

// Ties must take the average rank, or a fleet where two workers score the same
// would bias the correlation the emit summary reports.
func TestBenchRanksAveragesTies(t *testing.T) {
	got := benchRanks([]float64{5, 5, 9})
	want := []float64{1.5, 1.5, 3}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("benchRanks = %v, want %v", got, want)
		}
	}
}

func TestBenchHalfRate(t *testing.T) {
	half := []benchRanked{
		{id: "a", pass: map[string]bool{"q1": true, "q2": false}},
		{id: "b", pass: map[string]bool{"q1": true, "q2": false}},
	}
	if got := benchHalfRate(half, "q1"); got != 1 {
		t.Errorf("q1 = %v, want 1", got)
	}
	if got := benchHalfRate(half, "q2"); got != 0 {
		t.Errorf("q2 = %v, want 0", got)
	}
	if got := benchHalfRate(nil, "q1"); got != 0 {
		t.Errorf("empty half = %v, want 0", got)
	}
}

// The allowlist is the only thing standing between the pool and a task whose
// ground truth checkAnswer cannot grade, so its filtering is worth pinning.
func TestBenchToPoolQuestionFiltering(t *testing.T) {
	ok := map[string]any{
		"task": "zebra_puzzle", "question_id": "abc", "ground_truth": "1, a, b",
		"turns": []any{"puzzle text"}, "livebench_removal_date": "",
		"livebench_release_date": "2024-11-25T00:00:00",
	}
	q, got := benchToPoolQuestion(ok, "reasoning")
	if !got {
		t.Fatal("a valid zebra_puzzle row was rejected")
	}
	if q.Match != "exact-list" || q.Retired {
		t.Errorf("got %+v, want match=exact-list retired=false", q)
	}

	reject := []struct {
		name string
		row  map[string]any
	}{
		{"task not allowlisted", map[string]any{"task": "AMPS_Hard", "question_id": "a", "ground_truth": "x", "turns": []any{"t"}}},
		{"no ground truth", map[string]any{"task": "spatial", "question_id": "a", "ground_truth": "", "turns": []any{"t"}}},
		{"multi-turn", map[string]any{"task": "spatial", "question_id": "a", "ground_truth": "3", "turns": []any{"t1", "t2"}}},
		{"empty prompt", map[string]any{"task": "spatial", "question_id": "a", "ground_truth": "3", "turns": []any{""}}},
	}
	for _, tc := range reject {
		t.Run(tc.name, func(t *testing.T) {
			if _, got := benchToPoolQuestion(tc.row, "reasoning"); got {
				t.Error("row should have been rejected")
			}
		})
	}

	retired := map[string]any{
		"task": "spatial", "question_id": "a", "ground_truth": "3",
		"turns": []any{"t"}, "livebench_removal_date": "2025-04-25T00:00:00",
	}
	if q, ok := benchToPoolQuestion(retired, "reasoning"); !ok || !q.Retired {
		t.Error("a rotated-out question should be kept and flagged retired")
	}
}
