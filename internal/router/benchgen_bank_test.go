package router

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// WHY THE COMMITTED BANK CANNOT BE BYTE-REGENERATED, AND WHAT IS CHECKED INSTEAD.
//
// The obvious test for a generated file is "run the generator and diff". It
// cannot be written here, and the reasons are structural rather than something
// to fix:
//
//   - THE INPUTS ARE NOT COMMITTED. .gitignore excludes
//     internal/router/benchdata/calibration.json* on purpose — it is a per-fleet
//     measurement, not a source artefact. A checkout has pool.json and no
//     calibration at all, so there is nothing for CI to regenerate FROM.
//   - RENDERING NEEDS A LIVE SANDBOX. A code-exec question's test cases arrive
//     base64(zlib(pickle)) and only the execution sidecar may decode them
//     (benchEmitCases -> /decode-private), so `bench emit` cannot run hermetically.
//   - THE OUTPUT EMBEDS time.Now(). benchRenderFile stamps "Regenerated
//     <date>", so the same inputs produce different bytes on different days.
//
// And it would fail anyway: re-running selection over the calibration snapshot
// that WAS on disk gives band counts 12/25/60/70/54/40 with 23 executable coding
// questions, against the committed file's 4/18/60/70/70/40 with none. The
// committed bank came from a calibration run nobody kept.
//
// So these tests assert the INVARIANTS the generator guarantees, over the file
// as committed. That is weaker than a diff and it is not nothing: it is what
// catches the hand edit that was actually there — Tier fields changed from 2 and
// 12 to 3 and 10 with the band header comments left saying 2 and 12, in a file
// whose first line reads DO NOT EDIT.
//
// (The headers read "measured fleet pass rate 0%–0%" for both reserve bands.
// That is odd for the FLOOR band, whose items are the ones every worker passed,
// and it is what benchRenderFile emits — benchBandFor returns zeroed rates for a
// band selected by reserve rather than by rate. Left alone deliberately: the
// committed file has to match what the generator writes, and improving the text
// is a generator change, not a data one.)

// benchLiveSource reads benchmark_data_live.go. The tier bands and the measured
// p/d live in COMMENTS, so the assertions below have to read the source rather
// than the compiled benchmarkQuestionsLive.
func benchLiveSource(t *testing.T) string {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(benchAppDir(), "benchmark_data_live.go"))
	if err != nil {
		t.Skipf("benchmark_data_live.go unreadable (%v) — these tests need a source checkout", err)
	}
	return string(buf)
}

var (
	benchLiveBandRe = regexp.MustCompile(`^\t// ---- Tier (\d+) — `)
	benchLiveStatRe = regexp.MustCompile(`p=([0-9.]+) d=(-?[0-9.]+)`)
	benchLiveItemRe = regexp.MustCompile(`^\t\{Tier: (\d+), Match: "([^"]+)"`)
)

// benchLiveItem is one question as the committed FILE records it, including the
// measurement `bench emit` wrote above it.
type benchLiveItem struct {
	Line   int
	Header int // tier from the band comment above it
	Tier   int // tier from the struct literal
	Match  string
	P      float64
	D      float64
	HasPD  bool
}

func benchLiveItems(t *testing.T) []benchLiveItem {
	t.Helper()
	lines := strings.Split(benchLiveSource(t), "\n")
	var out []benchLiveItem
	header := 0
	for i, l := range lines {
		if m := benchLiveBandRe.FindStringSubmatch(l); m != nil {
			header, _ = strconv.Atoi(m[1])
			continue
		}
		m := benchLiveItemRe.FindStringSubmatch(l)
		if m == nil {
			continue
		}
		it := benchLiveItem{Line: i + 1, Header: header, Match: m[2]}
		it.Tier, _ = strconv.Atoi(m[1])
		if i > 0 {
			if s := benchLiveStatRe.FindStringSubmatch(lines[i-1]); s != nil {
				it.P, _ = strconv.ParseFloat(s[1], 64)
				it.D, _ = strconv.ParseFloat(s[2], 64)
				it.HasPD = true
			}
		}
		out = append(out, it)
	}
	if len(out) == 0 {
		t.Fatal("parsed no questions out of benchmark_data_live.go — the emit format and this test have diverged")
	}
	return out
}

// THE HAND EDIT. benchRenderFile prints the band header and the Tier field from
// the same variable, so a generated file CANNOT have them disagree. When they
// do, someone edited the file.
//
// It says DO NOT EDIT because a tier is the one part of a question that is NOT
// in its content hash. benchQuestionQID covers the prompt, the expected answer,
// the match mode, the grader version and the code tests — not the Tier — so
// editing a tier changes which weight bucket the question scores in and which
// thinking mode it is graded in, while every cached verdict for it stays valid
// and no worker re-profiles. The score moves and nothing says so. (The old
// wording here blamed "the question set is part of the profile cache key",
// which stopped being the mechanism when identity.go landed; the hazard is real
// and is this one.)
func TestLiveBankBandHeadersMatchTierFields(t *testing.T) {
	for _, it := range benchLiveItems(t) {
		if it.Header == 0 {
			t.Errorf("line %d: question with no band header above it", it.Line)
			continue
		}
		if it.Header != it.Tier {
			t.Errorf("line %d: sits under a \"Tier %d\" header but says Tier: %d — benchRenderFile writes "+
				"both from one value, so this file was edited by hand", it.Line, it.Header, it.Tier)
		}
	}
}

// Every question must sit in the band its OWN measured pass rate puts it in.
// This is the closest thing to regenerating the file that a checkout can do: the
// selection rule is p-in-range, the p is recorded beside each question, and the
// two have to agree. It catches a stale file, a question moved between bands by
// hand, and a change to benchTierTargets that was never followed by an emit.
func TestLiveBankMatchesTheGeneratorsBandingRule(t *testing.T) {
	items := benchLiveItems(t)
	counts := map[int]int{}
	for _, it := range items {
		counts[it.Tier]++
		band, ok := benchBandForExact(it.Tier)
		if !ok {
			t.Errorf("line %d: tier %d is not a band benchTierTargets declares — `bench emit` could not "+
				"have produced it", it.Line, it.Tier)
			continue
		}
		if !it.HasPD {
			t.Errorf("line %d: no `p=… d=…` comment; every emitted question carries its measurement", it.Line)
			continue
		}
		switch band.Reserve {
		case reserveFloor:
			// The floor band is the all-pass reserve: p == 1 by definition.
			if it.P != 1 {
				t.Errorf("line %d: in the floor (all-pass) band with p=%.2f", it.Line, it.P)
			}
		case reserveCeiling:
			// The ceiling band is the all-fail reserve: p == 0 by definition.
			if it.P != 0 {
				t.Errorf("line %d: in the ceiling (all-fail) band with p=%.2f", it.Line, it.P)
			}
			// The shape guard, asserted against the committed file rather than only
			// against the generator: it was unreachable for the whole life of this
			// band (`band.Tier >= benchCodingTier` with a tier-10 band and
			// benchCodingTier 12), so a file emitted before the fix could carry
			// exactly what it was written to exclude.
			if it.Match == "mcq" || it.Match == "mcq-repeat" {
				t.Errorf("line %d: multiple-choice question in the ceiling band. p==0 on a five-option "+
					"item is close to no evidence — it happens by luck about a third of the time across "+
					"a five-worker fleet — so it is not headroom", it.Line)
			}
		default:
			if it.P < band.MinRate || it.P >= band.MaxRate {
				t.Errorf("line %d: tier %d draws from p in [%.2f,%.2f) and this question measured p=%.2f",
					it.Line, it.Tier, band.MinRate, band.MaxRate, it.P)
			}
			// D <= 0 means the question ranked the fleet backwards, which benchSelect
			// discards outright.
			if it.D <= 0 {
				t.Errorf("line %d: tier %d question with d=%.2f; a non-positive discriminator is dropped "+
					"by selection and cannot appear here", it.Line, it.Tier, it.D)
			}
		}
	}
	for tier, n := range counts {
		band, ok := benchBandForExact(tier)
		if !ok {
			continue
		}
		if n > band.Target {
			t.Errorf("tier %d holds %d questions, more than its target of %d — selection truncates at the "+
				"target, so the surplus was added by hand", tier, n, band.Target)
		}
	}
	if testing.Verbose() {
		fmt.Println("live bank band counts:", counts)
	}
}

// benchBandForExact is benchBandFor without its fall back to the first band,
// which would make an unknown tier look like a declared one.
func benchBandForExact(tier int) (benchTierTarget, bool) {
	for _, b := range benchTierTargets {
		if b.Tier == tier {
			return b, true
		}
	}
	return benchTierTargets[0], false
}

// Every question in the bank must be one the grader can score. A question the
// grader cannot read fails every worker, is measured as p=0, and ships as
// headroom nobody can reach — which is indistinguishable from a genuinely hard
// question and permanently costs a slot in the profiling budget.
func TestLiveBankQuestionsAreGradeable(t *testing.T) {
	// The modes checkAnswer DISPATCHES on. Anything else falls through to the
	// "contains" default, where a question would be graded by a rule its author
	// did not pick.
	supported := map[string]bool{
		"numeric": true, "mcq": true, "mcq-repeat": true,
		"exact-list": true, "contains": true, "final-contains": true,
		benchMatchCodeExec: true,
	}
	for i, q := range benchmarkQuestionsLive {
		if !supported[q.Match] {
			t.Errorf("question %d has match mode %q, which checkAnswer does not dispatch on", i, q.Match)
		}
		if q.Match == benchMatchCodeExec {
			if q.Code == nil || len(q.Code.Tests) == 0 {
				t.Errorf("question %d is code-exec with no test cases — its ground truth is missing", i)
			}
			continue
		}
		if strings.TrimSpace(q.Expect) == "" {
			t.Errorf("question %d has an empty Expect; checkAnswer rejects every answer against one, so "+
				"the question can never be passed", i)
		}
		if q.Tier < 1 || q.Tier > benchCodingTier {
			t.Errorf("question %d has tier %d, outside 1..%d", i, q.Tier, benchCodingTier)
		}
		if !checkAnswer(q, q.Expect) {
			t.Errorf("question %d does not grade its own expected answer %q", i, q.Expect)
		}
	}
}

// ─── DEFECT 2: the coding_completion contract ───────────────────────────────

// A coding_completion answer is a FRAGMENT — the prompt shows a function
// truncated part-way through its body and asks for "only the missing portion"
// — so it only runs when appended to that partial solution. Grading it alone
// grades something that never compiles, which scored the two strongest workers
// 0% on this task while the weakest scored highest: the task measured
// disobedience.
//
// pool.json shipped with no `prefix` on any of the 50 items, so benchSelfGrades
// dropped every one of them as ungradeable and the emitted bank had no
// executable coding questions at all. The extractor to recover it
// (benchCompletionPrefix) already existed and had simply never been run over the
// committed pool.
//
// This test is the one the repair has to pass: does the EMIT GATE accept them.
func TestPoolCodingCompletionPassesTheEmitGate(t *testing.T) {
	pool, err := benchLoadPool()
	if err != nil {
		t.Fatalf("benchLoadPool: %v", err)
	}
	n := 0
	for _, q := range pool.Questions {
		if q.Task != "coding_completion" {
			continue
		}
		n++
		if !benchSelfGrades(q) {
			t.Errorf("%s: rejected by benchSelfGrades — it would be silently dropped before scoring", q.ID[:12])
			continue
		}
		c := q.Code
		// VERBATIM. The sidecar joins with prefix.rstrip("\n") + "\n" + answer and
		// does not reindent, because the fragment's own leading whitespace is what
		// places it inside the function. A prefix that had been trimmed or
		// reindented would still look populated and would still fail every answer.
		if !strings.Contains(q.Prompt, c.Prefix) {
			t.Errorf("%s: prefix is not a verbatim slice of the prompt", q.ID[:12])
		}
		// The prefix must contain the def for Func. sandbox/runner.py measures the
		// answer's indentation against the prefix's last def line to tell a
		// continuation from a restatement; with no def in the prefix that
		// discrimination is gone and the column-0 dedent bug reopens.
		if c.Func == "" {
			t.Errorf("%s: no entry function; resolve_entry fails the whole request", q.ID[:12])
			continue
		}
		def := regexp.MustCompile(`(?m)^\s*def\s+` + regexp.QuoteMeta(c.Func) + `\s*\(`)
		if !def.MatchString(c.Prefix) {
			t.Errorf("%s: prefix does not contain `def %s(` — the sidecar cannot tell a continuation "+
				"from a restatement without it", q.ID[:12], c.Func)
		}
		if c.Class != "" && !strings.Contains(c.Prefix, "class "+c.Class) {
			t.Errorf("%s: entry class %q is not defined in the prefix", q.ID[:12], c.Class)
		}
		var cases []benchCase
		if c.PublicJSON != "" {
			if err := json.Unmarshal([]byte(c.PublicJSON), &cases); err != nil {
				t.Errorf("%s: public test cases will not parse: %v", q.ID[:12], err)
			}
		}
		if len(cases) == 0 && !c.HasPrivate {
			t.Errorf("%s: no test cases to run the answer against", q.ID[:12])
		}
	}
	if n == 0 {
		t.Fatal("no coding_completion questions in the pool — the task allowlist or the fetch changed")
	}
	if testing.Verbose() {
		fmt.Printf("%d coding_completion questions pass the emit gate\n", n)
	}

	// AND ON DISK, not only after benchLoadPool's backfill. The distinction is
	// the whole defect: benchFillPrefixes recovers the prefix from the prompt on
	// every load, so the in-memory pool has always looked correct while the
	// committed artefact — the reviewed, diffable half of this pipeline — carried
	// none of them. Anything reading pool.json without going through
	// benchLoadPool sees 50 questions that cannot be graded, and the field's own
	// documentation says what it is for.
	raw, err := os.ReadFile(filepath.Join(benchDataDir(), "pool.json"))
	if err != nil {
		t.Fatalf("pool.json: %v", err)
	}
	var onDisk benchPool
	if err := json.Unmarshal(raw, &onDisk); err != nil {
		t.Fatalf("pool.json: %v", err)
	}
	missing := 0
	for _, q := range onDisk.Questions {
		if q.Task == "coding_completion" && (q.Code == nil || q.Code.Prefix == "") {
			missing++
		}
	}
	if missing > 0 {
		t.Errorf("%d of %d coding_completion questions have no prefix in the COMMITTED pool.json; run the "+
			"extractor and commit the result, rather than relying on benchLoadPool to recover it each time",
			missing, n)
	}
}

// benchCodeEntryOK reads the entry-point rule off the sidecar rather than
// guessing at it, and the distinction is worth a test in both directions: 38 of
// this pool's 78 LCB_generation questions are stdin-style and legitimately name
// no function, so "Func must be set" would throw away half the task.
func TestCodeEntryPointFollowsTheTestShape(t *testing.T) {
	cases := []struct {
		name string
		code *poolCode
		want bool
	}{
		{"functional with a func", &poolCode{Func: "solve", PublicJSON: `[{"input":"1","output":"2","testtype":"functional"}]`}, true},
		{"functional without a func", &poolCode{PublicJSON: `[{"input":"1","output":"2","testtype":"functional"}]`}, false},
		{"stdin without a func", &poolCode{PublicJSON: `[{"input":"1","output":"2","testtype":"stdin"}]`}, true},
		{"mixed without a func", &poolCode{PublicJSON: `[{"input":"1","output":"2","testtype":"stdin"},{"input":"3","output":"4","testtype":"functional"}]`}, false},
		// runner.py defaults a missing testtype to "functional", so an entry point
		// is required; erring the other way ships a question that only grades wrong.
		{"no testtype without a func", &poolCode{PublicJSON: `[{"input":"1","output":"2"}]`}, false},
		{"unreadable cases without a func", &poolCode{PublicJSON: `{`, HasPrivate: true}, false},
	}
	for _, c := range cases {
		if got := benchCodeEntryOK(poolQuestion{Code: c.code}); got != c.want {
			t.Errorf("%s: benchCodeEntryOK = %v, want %v", c.name, got, c.want)
		}
	}
	if benchCodeEntryOK(poolQuestion{}) {
		t.Error("a question with no Code at all should not pass the entry-point check")
	}
	// And the real pool: every code question must pass, or the bank loses it.
	pool, err := benchLoadPool()
	if err != nil {
		t.Fatalf("benchLoadPool: %v", err)
	}
	for _, q := range pool.Questions {
		if q.Match == benchMatchCodeExec && !benchCodeEntryOK(q) {
			t.Errorf("%s (%s): fails the entry-point check", q.ID[:12], q.Task)
		}
	}
}

// benchCodePayload used to do `_ = json.Unmarshal(metadata)`, so a blob that
// would not parse left Func empty and the row was accepted anyway. That question
// then fails resolve_entry before a single case runs — every case, every worker
// — and ships as an item nobody can answer.
func TestBenchCodePayloadRejectsUnparseableMetadata(t *testing.T) {
	row := func(metadata string) map[string]any {
		return map[string]any{
			"original_json": map[string]any{
				"starter_code": "class Solution:\n    def solve(self):\n        ",
				"metadata":     metadata,
			},
			"public_test_cases": `[{"input":"1","output":"2","testtype":"functional"}]`,
		}
	}
	if c, ok := benchCodePayload(row(`{"func_name": "solve"}`)); !ok || c.Func != "solve" {
		t.Fatalf("well-formed metadata: ok=%v code=%+v", ok, c)
	}
	if _, ok := benchCodePayload(row(`{"func_name": `)); ok {
		t.Error("a metadata blob that will not parse was accepted; its Func is empty and every answer fails")
	}
	// An ABSENT metadata field is a different thing and stays legal — a stdin
	// question has no entry point to name.
	if _, ok := benchCodePayload(row("")); !ok {
		t.Error("absent metadata should still yield a payload; benchCodeEntryOK is what judges it")
	}
}

// benchLoadPool now returns an error rather than printing a warning when a
// completion prefix cannot be recovered. It printed one for the whole life of
// this bug, and the pipeline carried on and emitted a bank with the entire task
// missing.
func TestCompletionPrefixMissIsDetectable(t *testing.T) {
	qs := []poolQuestion{{
		Task: "coding_completion",
		Code: &poolCode{},
		// No fenced block, so nothing to recover.
		Prompt: "Finish the solution. class Solution: def solve(self): return",
	}}
	filled, missing := benchFillPrefixes(qs)
	if filled != 0 || missing != 1 {
		t.Fatalf("benchFillPrefixes = (%d filled, %d missing), want (0, 1)", filled, missing)
	}
	ok := []poolQuestion{{
		Task:   "coding_completion",
		Code:   &poolCode{},
		Prompt: "Finish it.\n```python\nclass Solution:\n    def solve(self):\n        x = 1\n```\n### Answer:",
	}}
	if filled, missing := benchFillPrefixes(ok); filled != 1 || missing != 0 {
		t.Fatalf("benchFillPrefixes on a fenced prompt = (%d, %d), want (1, 0)", filled, missing)
	}
	if got := ok[0].Code.Prefix; !strings.Contains(got, "def solve") {
		t.Errorf("recovered prefix %q does not carry the entry def", got)
	}
}

// ─── DEFECT 3: the ceiling band's shape guard ───────────────────────────────

// benchSelectFixture builds a pool and a calibration that between them exercise
// every band, so selection can be driven without a fleet.
//
// The fleet has to be plausible or nothing is selected at all: benchIsObserver
// throws out any backend scoring under 5% or over 95%, and a question needs
// benchMinObservers measured answers before its p means anything.
func benchSelectFixture() (*benchPool, *calibration) {
	pool := &benchPool{FetchedAt: "2026-01-01T00:00:00Z", Source: "test"}
	const backends = 4
	pass := make([]map[string]bool, backends)
	for i := range pass {
		pass[i] = map[string]bool{}
	}
	add := func(id, task, match, expect string, passers int, code *poolCode) {
		pool.Questions = append(pool.Questions, poolQuestion{
			ID: id, Task: task, Match: match, Expect: expect,
			Prompt: "synthetic question " + id, Code: code,
		})
		for i := 0; i < backends; i++ {
			pass[i][id] = i < passers
		}
	}
	// A spread of ordinary items so every backend lands strictly between 5% and
	// 95% and the top/bottom halves differ.
	for i := 0; i < 40; i++ {
		add(fmt.Sprintf("spread-%03d", i), "spatial", "numeric", strconv.Itoa(100+i), 1+i%3, nil)
	}
	// The ceiling candidates: nobody passes any of them.
	fnCode := &poolCode{Func: "solve", PublicJSON: `[{"input":"1","output":"2","testtype":"functional"}]`}
	for i := 0; i < 6; i++ {
		add(fmt.Sprintf("ceil-mcq-%d", i), "math_comp", "mcq-repeat", "C", 0, nil)
		add(fmt.Sprintf("ceil-num-%d", i), "spatial", "numeric", strconv.Itoa(900+i), 0, nil)
		add(fmt.Sprintf("ceil-code-%d", i), "LCB_generation", benchMatchCodeExec, "", 0, fnCode)
	}
	// The floor candidates: everybody passes.
	for i := 0; i < 6; i++ {
		add(fmt.Sprintf("floor-%d", i), "zebra_puzzle", "numeric", strconv.Itoa(700+i), backends, nil)
	}
	calib := &calibration{}
	for i := 0; i < backends; i++ {
		calib.Results = append(calib.Results, &calibResult{
			Backend: fmt.Sprintf("worker-%d", i), RouterQuality: 60 + i*10, Pass: pass[i],
		})
	}
	return pool, calib
}

// The guard read `if band.Tier >= benchCodingTier`. The ceiling band moved from
// tier 12 to tier 10 and benchCodingTier stayed 12, so the condition became
// 10 >= 12 and the guard was dead from that day until it was rewritten to key on
// the band's ROLE (band.Reserve == reserveCeiling) — which is why the committed
// tier-10 band went out unfiltered.
//
// It could not be fixed by lowering the constant: benchCodingTier is what
// benchWeightedScore uses to decide the 20-point coding bucket, so tier 10 would
// sweep 40 maths and olympiad questions into it and restore the exact fault that
// moved the band off tier 12 in the first place.
//
// This test DRIVES selection rather than reading the source, so it stays honest
// about the thing that actually went wrong: the condition compiled, read
// plausibly, and matched nothing.
func TestCeilingBandShapeGuardFires(t *testing.T) {
	pool, calib := benchSelectFixture()
	selected, _, err := benchSelect(pool, calib)
	if err != nil {
		t.Fatalf("benchSelect: %v", err)
	}
	var ceilingTier int
	for _, b := range benchTierTargets {
		if b.Reserve == reserveCeiling {
			ceilingTier = b.Tier
		}
	}
	if ceilingTier == 0 {
		t.Fatal("no ceiling band in benchTierTargets")
	}
	if ceilingTier >= benchCodingTier {
		t.Fatalf("the ceiling band is tier %d and benchCodingTier is %d; if the band ever moves back into "+
			"the coding bucket this test's premise changes and so does benchWeightedScore's",
			ceilingTier, benchCodingTier)
	}
	nMCQ, nCode, nCeiling := 0, 0, 0
	for _, s := range selected {
		if s.tier != ceilingTier {
			continue
		}
		nCeiling++
		switch {
		case s.q.Match == "mcq" || s.q.Match == "mcq-repeat":
			nMCQ++
		case s.q.Match == benchMatchCodeExec:
			nCode++
		}
	}
	if nCeiling == 0 {
		t.Fatal("the ceiling band selected nothing, so the guard was not exercised")
	}
	if nMCQ != 0 {
		t.Errorf("%d multiple-choice questions reached the ceiling band; the shape guard is unreachable again",
			nMCQ)
	}
	// The other direction, and the one that would do real damage: a code-exec
	// question has NO Expect by construction, so an empty-Expect rule applied to
	// it removes every coding question in the band — 59 of 116 candidates against
	// the real calibration data.
	if nCode == 0 {
		t.Error("no code-exec questions reached the ceiling band; the empty-Expect rule is being applied " +
			"to questions whose ground truth is their test cases, which drops every coding question")
	}
}

// Selection has to be a pure function of pool.json and the calibration, or the
// committed bank cannot be reasoned about at all: two emits from one calibration
// would produce two different question sets, and every worker profiled between
// them would be scored on a bank nobody can reconstruct — read afterwards on the
// same absolute 0-100 scale as every other worker's.
func TestBenchSelectIsDeterministic(t *testing.T) {
	pool, calib := benchSelectFixture()
	a, rhoA, err := benchSelect(pool, calib)
	if err != nil {
		t.Fatalf("benchSelect: %v", err)
	}
	b, rhoB, err := benchSelect(pool, calib)
	if err != nil {
		t.Fatalf("benchSelect (second call): %v", err)
	}
	if rhoA != rhoB {
		t.Errorf("Spearman differs between runs: %v vs %v", rhoA, rhoB)
	}
	if len(a) != len(b) {
		t.Fatalf("selected %d questions then %d", len(a), len(b))
	}
	for i := range a {
		if a[i].q.ID != b[i].q.ID || a[i].tier != b[i].tier || a[i].p != b[i].p || a[i].d != b[i].d {
			t.Fatalf("position %d differs between runs: %s tier %d p=%.2f d=%.2f vs %s tier %d p=%.2f d=%.2f",
				i, a[i].q.ID, a[i].tier, a[i].p, a[i].d, b[i].q.ID, b[i].tier, b[i].p, b[i].d)
		}
	}
	// No band may exceed its target, and every selected question must be one the
	// grader can score — the same two properties asserted above against the
	// committed file, here against a freshly generated selection.
	counts := map[int]int{}
	for _, s := range a {
		counts[s.tier]++
		if !benchSelfGrades(s.q) {
			t.Errorf("selection kept %s, which benchSelfGrades rejects", s.q.ID)
		}
	}
	for tier, n := range counts {
		band, ok := benchBandForExact(tier)
		if !ok {
			t.Errorf("selection produced tier %d, which benchTierTargets does not declare", tier)
			continue
		}
		if n > band.Target {
			t.Errorf("tier %d selected %d questions against a target of %d", tier, n, band.Target)
		}
	}
}
