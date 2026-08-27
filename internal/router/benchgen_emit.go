package router

// bench emit turns calibration results into benchmark_data_live.go.
//
// SELECTION IS BY MEASURED DISCRIMINATION, not by category, source difficulty
// or taste. Two statistics per candidate:
//
//	p  fleet pass rate — the fraction of backends that got it right
//	D  discrimination  — pass rate among the top half of the fleet minus the
//	                     pass rate among the bottom half
//
// p == 1 (everyone passes) and p == 0 (nobody does) contribute exactly nothing
// to [qmin,qmax] while still costing a slot in the profiling budget, so they are
// dropped outright. D <= 0 means the question ranks the fleet backwards — a weak
// model beats a strong one on it — which is either noise or a grader mismatch,
// and is worse than useless in a score that drives routing. Whatever survives is
// ranked by D and taken from the top.
//
// This is the same call the file's comments record being made by hand across
// v29–v32 ("measured at 17 points of spread, the worst ratio in the file — cut"),
// applied per question instead of per tier, and with numbers instead of intuition.
//
// TIERS ARE ASSIGNED FROM p, not carried over from LiveBench. A tier in this
// codebase means "difficulty band as this fleet experiences it", which is a
// property of the pairing rather than of the question — the same item is tier 4
// against a frontier fleet and tier 8 against a weak one. Deriving it from
// measurement is the only way the band keeps meaning what benchmark_data.go says.

import (
	"encoding/json"
	"fmt"
	"go/format"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// benchTierTarget is one band. A named type rather than the anonymous struct
// this used to be: the anonymous form had to be spelled out in full at every
// place a band was returned rather than ranged over — benchBandFor here and
// benchBandForExact in the tests — which is three verbatim copies of a five
// field list that a `go vet` cannot check against each other.
type benchTierTarget struct {
	Tier    int
	Target  int
	MinRate float64 // inclusive
	MaxRate float64 // exclusive; p==1 is handled by the reserve bands instead
	Reserve int     // reserveNone | reserveFloor | reserveCeiling
}

// benchTierTargets is how many questions to emit per tier, and the measured
// pass-rate band each tier draws from. The defaults mirror the shape of the
// hand-authored arithmetic tiers being replaced (4:25, 5:7, 7:5, 8:13 = 50), so
// the total question count — and therefore the profiling budget, which is the
// real constraint at 8–30 min per worker for 97 questions — stays put.
var benchTierTargets = []benchTierTarget{
	// Tier 3, not 2: benchHardTier is 3, so anything below it is graded
	// thinking-OFF at the short budget in production while calibration grades
	// every candidate thinking-ON at the long one. A floor item measured under
	// one set of conditions and served under the other is calibrated for nothing
	// — and these are AMC competition questions chosen because every worker
	// passed them, which at 1024 no-think tokens they would not.
	{Tier: 3, Target: 12, Reserve: reserveFloor},
	{Tier: 4, Target: 25, MinRate: 0.75, MaxRate: 1.00},
	{Tier: 5, Target: 60, MinRate: 0.50, MaxRate: 0.75},
	{Tier: 7, Target: 70, MinRate: 0.25, MaxRate: 0.50},
	{Tier: 8, Target: 70, MinRate: 0.00, MaxRate: 0.25},
	// Tier 10, not 12. Tier 12 is benchCodingTier, so the ceiling band was
	// filling the 20-point CODING bucket with maths, olympiad and spatial items
	// that no worker passes — 40 of its 68 questions, none of them code. Two
	// consequences, both bad: a worker that aced every real coding question
	// capped at 8.2 of 20 coding points, and because truncation samples a tier in
	// order, a profile cut short by the time budget got the 28 answerable coding
	// questions and few of the 40 impossible ones — worth up to +10 points of
	// quality for being too slow to finish.
	{Tier: 10, Target: 40, Reserve: reserveCeiling},
}

// reserveFloor / reserveCeiling mark the two HEADROOM bands, which exist for a
// fleet this calibration has not met.
//
// The ordinary bands drop p==1 and p==0 because a question every worker answers
// the same way carries no information ABOUT THIS FLEET. That is true and it is
// also why the bank needs replacing every time the fleet moves — the saturation
// treadmill this file was written to get off. A question today's best model
// fails is precisely the question that will rank tomorrow's, and one today's
// worst passes is what will rank a smaller/faster worker when one is deployed.
//
// So a bounded number of each is kept. They cost profiling budget and tell us
// nothing now; they are what lets a new worker be placed correctly WITHOUT a
// re-fetch and a re-calibration — hours of fleet time, and a bank refresh that
// every already-profiled worker would then be holding an old-bank score against
// until it next re-profiles.
//
// Selection cannot rank by discrimination — D is 0 for a constant item by
// definition — so it stratifies round-robin across source tasks and breaks ties
// by id, which keeps the choice deterministic and stops one task's quirks
// filling the whole band.
//
// NOTE the ceiling band deflates every current score, and that is the point: a
// fleet scoring 99/100 has no room to represent a better model. Headroom has to
// come from somewhere.
const (
	reserveNone = iota
	reserveFloor
	reserveCeiling
)

// benchRanked is one backend's calibration outcome, ordered by the score it got
// on this pool.
type benchRanked struct {
	id       string
	score    float64 // pass rate over MEASURED observations (censored ones excluded)
	routerQ  int
	pass     map[string]bool
	censored map[string]bool
	usable   int  // observations that produced an answer rather than a deadline
	total    int  // observations attempted
	observer bool // counts towards item statistics — see benchIsObserver
}

// benchMinObservers is how many MEASURED observations an item needs before its
// pass rate means anything. Below this the item is dropped rather than scored
// from one or two workers, because p is what assigns the tier and a tier drawn
// from two observations is a guess with a number on it.
const benchMinObservers = 3

// benchIsObserver decides whether a backend may vote on how hard a QUESTION is.
//
// Two ways to be disqualified, and neither is a judgement about the worker:
//
//   - Almost everything it attempted was CENSORED — it hit the deadline rather
//     than answering. Measured on llm-cpu-gemma-26B-silver, which needed 6.9
//     minutes for a math_comp question that finished and did not finish three
//     others inside eight. Its failures are a statement about tokens per second.
//   - Its measured pass rate is ~0 or ~1. A worker that answers everything the
//     same way separates nothing, which is the same reason all-pass and all-fail
//     ITEMS are dropped below. Standard item analysis excludes both.
//
// It keeps its production quality score either way; that score is SUPPOSED to
// count slowness as a fail, because a worker that cannot answer inside the
// deadline genuinely cannot serve the request. What it must not do is carry
// that verdict over into a claim about the question.
func benchIsObserver(f benchRanked) (bool, string) {
	if f.total == 0 {
		return false, "no observations"
	}
	if rate := float64(f.usable) / float64(f.total); rate < 0.25 {
		return false, fmt.Sprintf("%.0f%% of answers hit the deadline — measures throughput, not difficulty", (1-rate)*100)
	}
	if f.usable < benchMinObservers {
		return false, "too few measured answers"
	}
	if f.score < 0.05 || f.score > 0.95 {
		return false, fmt.Sprintf("pass rate %.0f%% separates nothing", f.score*100)
	}
	return true, ""
}

type scoredQuestion struct {
	q    poolQuestion
	tier int
	p    float64
	d    float64
}

func benchEmit(sandboxURL string) error {
	pool, err := benchLoadPool()
	if err != nil {
		return err
	}
	calib := benchLoadCalibration()
	selected, rho, err := benchSelect(pool, calib)
	if err != nil {
		return err
	}
	src, err := benchRenderFile(selected, pool, rho, sandboxURL)
	if err != nil {
		return err
	}
	path := filepath.Join(benchAppDir(), "benchmark_data_live.go")
	if err := os.WriteFile(path, src, 0o644); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "\nwrote %s (%d questions)\n", path, len(selected))
	// This used to print "bump benchmarkVersion ... the question set is part of
	// the profile cache key", and it was the loudest copy of a claim that stopped
	// being true when question identity became content-addressed (identity.go).
	// An instruction a tool prints is the one an operator follows, so it is the
	// one that most needed correcting.
	fmt.Fprint(os.Stderr, "\nNEXT: delete tiers 4/5/7/8 from benchmark_data.go and commit both files in the\n"+
		"SAME commit.\n\n"+
		"Do NOT bump benchmarkVersion to publish this. Each question is identified by its\n"+
		"own content hash, so the ones this run changed are asked afresh and the ones it\n"+
		"kept are served from the permacache — nothing needs invalidating. Bump it only if\n"+
		"you want the whole fleet re-scored on the new set now, since a worker with a\n"+
		"current cached profile otherwise keeps the score it earned on the old one.\n")
	return nil
}

// benchSelect is the whole of emit EXCEPT writing the file: rank the fleet, score
// every candidate, and fill each band. Split out from benchEmit so it can be
// driven by a test.
//
// That split is not tidiness. Rendering needs a live code-execution sidecar (a
// coding question's test cases arrive base64(zlib(pickle)) and only the sidecar
// may decode them) and stamps time.Now() into the file header, so the emit
// command as a whole cannot run in CI and cannot produce the same bytes twice.
// Selection has neither property — pool.json and the calibration file in,
// question list out — so it is the part that can be asserted deterministic. See
// benchgen_bank_test.go.
func benchSelect(pool *benchPool, calib *calibration) ([]scoredQuestion, float64, error) {
	if calib == nil || len(calib.Results) == 0 {
		return nil, 0, fmt.Errorf("no calibration data (run `bench calibrate` first)")
	}

	// Questions calibration found it could never grade, collected from EVERY
	// calibrated backend before any of them is excluded as an observer.
	//
	// One report is enough, and that is not a shortcut. "The answer key is not
	// JSON" is a fact about the QUESTION, so a second backend confirming it
	// learns nothing — and only a backend whose submission actually ran gets far
	// enough to find out, so the one that reported it may well be the only one
	// that could. Reading it off the observer list instead would lose exactly the
	// backends most likely to have been the reporter: a worker excluded for
	// answering nothing is excluded from voting on DIFFICULTY, which is a
	// judgement about it, and this is not a judgement about it.
	broken := map[string]bool{}
	for _, r := range calib.Results {
		for id, yes := range r.Ungradeable {
			if yes {
				broken[id] = true
			}
		}
	}

	// Rank backends by the score they got on THIS pool, not by the router's q:
	// the top/bottom split has to come from the instrument being calibrated, or
	// every question that disagrees with the old benchmark scores as noise.
	var fleet []benchRanked
	for _, r := range calib.Results {
		if len(r.Pass) == 0 {
			continue
		}
		// Scored over MEASURED answers only. Dividing by every attempt would rank
		// a slow worker by how often it beat the clock, and this ranking decides
		// the top/bottom halves that discrimination is measured against.
		passes, usable := 0, 0
		for id, ok := range r.Pass {
			if r.Censored[id] {
				continue
			}
			usable++
			if ok {
				passes++
			}
		}
		score := 0.0
		if usable > 0 {
			score = float64(passes) / float64(usable)
		}
		f := benchRanked{
			id: r.Backend, score: score, routerQ: r.RouterQuality,
			pass: r.Pass, censored: r.Censored, usable: usable, total: len(r.Pass),
		}
		f.observer, _ = benchIsObserver(f)
		fleet = append(fleet, f)
	}
	if len(fleet) < 2 {
		return nil, 0, fmt.Errorf("need at least 2 calibrated backends to measure discrimination, have %d", len(fleet))
	}
	sort.Slice(fleet, func(i, j int) bool { return fleet[i].score > fleet[j].score })

	fmt.Fprintln(os.Stderr, "fleet on this pool:")
	poolScores := make([]float64, 0, len(fleet))
	routerScores := make([]float64, 0, len(fleet))
	var observers []benchRanked
	for _, f := range fleet {
		note := ""
		if censored := f.total - f.usable; censored > 0 {
			note = fmt.Sprintf("  %d censored", censored)
		}
		if ok, why := benchIsObserver(f); !ok {
			fmt.Fprintf(os.Stderr, "  %-30s pool=%3.0f%%  router q=%-3d%s  EXCLUDED from item stats: %s\n",
				f.id, f.score*100, f.routerQ, note, why)
			continue
		}
		fmt.Fprintf(os.Stderr, "  %-30s pool=%3.0f%%  router q=%-3d%s\n", f.id, f.score*100, f.routerQ, note)
		observers = append(observers, f)
		poolScores = append(poolScores, f.score)
		routerScores = append(routerScores, float64(f.routerQ))
	}
	fleet = observers
	// Re-checked AFTER exclusion, not just before it. The guard above counts
	// calibrated backends; this counts the ones allowed to vote on difficulty,
	// and the two differ by however many were censored or degenerate. Without
	// this an over-eager exclusion leaves top and bottom empty, every D computes
	// as 0-0, every question is discarded as "negative discrimination", and emit
	// writes a valid-looking file with nothing in it.
	if len(fleet) < 2 {
		return nil, 0, fmt.Errorf("only %d backend(s) qualify as difficulty observers (need 2); "+
			"the rest were excluded as censored or non-separating — calibrate more workers, "+
			"or give the slow ones a deadline they can finish inside", len(fleet))
	}

	// The correlation check: do the two instruments RANK the fleet the same way?
	// A high value means the hand-authored set is tracking real capability and
	// this is a refresh; a low one means something is wrong with one of them and
	// committing the emitted file would bake it in.
	rho := benchSpearman(poolScores, routerScores)
	fmt.Fprintf(os.Stderr, "\nSpearman(pool, router q) = %+.2f  ", rho)
	switch {
	case rho >= 0.8:
		fmt.Fprintln(os.Stderr, "— the hand-authored set tracks capability; this is a refresh, not a rescue.")
	case rho >= 0.4:
		fmt.Fprintln(os.Stderr, "— partial agreement; read the disagreements before adopting.")
	default:
		fmt.Fprintln(os.Stderr, "— the instruments disagree. Investigate BEFORE committing this file.")
	}

	// An odd-sized fleet drops its median worker rather than assigning it to a
	// half arbitrarily.
	half := len(fleet) / 2
	top, bottom := fleet[:half], fleet[len(fleet)-half:]

	var scored, floor, ceiling []scoredQuestion
	var noData, tooEasy, tooHard, backwards, ungradeable int
	for _, q := range pool.Questions {
		// A question the grader cannot score is not a hard question, it is a
		// broken one — and it fails for EVERY worker, so unfiltered it lands in
		// the p==0 pool and masquerades as headroom. That is exactly how the
		// first headroom band shipped LiveBench list answers the numeric/mcq/
		// contains modes cannot match. benchmark_test.go asserts the same
		// property for tier >= 9; checking it here keeps a broken item out of
		// every band rather than failing the build after emit.
		if !benchSelfGrades(q) {
			ungradeable++
			continue
		}
		// The same verdict, reached the other way round. benchSelfGrades reads what
		// is visible in pool.json; this reads what calibration DISCOVERED by trying
		// to run the question — a private test case that decodes to something the
		// grader cannot use is invisible from here, because the blob is opaque
		// base64 until the sidecar unpickles it. Both drop the question outright
		// rather than letting it reach a band: it fails for every worker, so
		// unfiltered it lands in the p==0 pool and masquerades as headroom.
		if broken[q.ID] {
			ungradeable++
			continue
		}
		graded, passes := 0, 0
		for _, f := range fleet {
			ok, seen := f.pass[q.ID]
			// A censored observation is skipped, not counted as a miss. Counting it
			// makes a LONG question look like a HARD one, and since the fast half of
			// the fleet is also the strong half, it inflates discrimination on
			// exactly the items where the difference is throughput.
			if !seen || f.censored[q.ID] {
				continue
			}
			graded++
			if ok {
				passes++
			}
		}
		if graded < benchMinObservers {
			noData++ // an interrupted calibration is not evidence about the question
			continue
		}
		p := float64(passes) / float64(graded)
		switch p {
		case 1:
			tooEasy++
			floor = append(floor, scoredQuestion{q: q, p: p})
			continue
		case 0:
			tooHard++
			ceiling = append(ceiling, scoredQuestion{q: q, p: p})
			continue
		}
		d := benchHalfRate(top, q.ID) - benchHalfRate(bottom, q.ID)
		if d <= 0 {
			backwards++
			continue
		}
		scored = append(scored, scoredQuestion{q: q, p: p, d: d})
	}
	fmt.Fprintf(os.Stderr, "\n%d candidates: %d usable; dropped %d all-pass, %d all-fail, %d negative-discrimination, %d ungraded, %d ungradeable\n",
		len(pool.Questions), len(scored), tooEasy, tooHard, backwards, noData, ungradeable)

	var selected []scoredQuestion
	for _, band := range benchTierTargets {
		if band.Reserve != reserveNone {
			src := floor
			if band.Reserve == reserveCeiling {
				src = ceiling
			}
			// Shape rules for the CEILING band, and the condition is the band's
			// ROLE rather than its tier number for a reason worth writing down.
			//
			// It used to read `if band.Tier >= benchCodingTier`. When the ceiling
			// band moved from tier 12 to tier 10 (see benchTierTargets above)
			// benchCodingTier stayed 12, so the test became 10 >= 12 and the guard
			// was dead from that day until this line was rewritten — which is why
			// the committed tier-10 band went out unfiltered, and why
			// TestCeilingBandShapeGuardFires drives selection rather than reading
			// the source: the condition compiled, read plausibly, and matched
			// nothing. Lowering the constant would NOT have been the fix.
			// benchCodingTier is what benchWeightedScore uses to decide which
			// questions land in the 20-point coding bucket (and what
			// benchmark_test.go's TestCodingTierShape walks), so setting it to 10
			// would sweep all 40 ceiling questions into the coding bucket and
			// restore the exact fault that moved the band off tier 12. The guard
			// has to key on the thing it is actually about, which is "this is the
			// headroom band".
			//
			// A MULTIPLE-CHOICE ITEM CANNOT BE HEADROOM. This band is selected FOR
			// p == 0, and on a five-option question p == 0 across a five-worker
			// fleet happens by luck about a third of the time even for an item half
			// the fleet would normally get right — so "nobody passed it" is close
			// to no evidence at all when there is a guess floor under it. Free
			// response has no such floor and p == 0 means what it says. (The floor
			// band is the mirror image and needs no such rule: p == 1 on five
			// five-option questions is a 0.03% fluke.)
			//
			// Note what is NOT filtered, having been checked against the committed
			// band: long exact-list answers. The ordered permutations in there run
			// to 30 elements and look unpassable, but the measured pass rates say
			// otherwise — a 29-element item sits at p=0.20 — so excluding them
			// would be a guess overriding a measurement.
			if band.Reserve == reserveCeiling {
				kept := src[:0:0]
				for _, s := range src {
					if s.q.Match == "mcq" || s.q.Match == "mcq-repeat" {
						continue
					}
					// The Expect rules apply only to a question that HAS an Expect,
					// and skipping code-exec is not a nicety — it is most of the band.
					// A code question's ground truth is its test cases and its Expect
					// is empty by construction, so applying the empty-Expect rule to it
					// removes 59 of the 116 ceiling candidates measured against
					// benchdata's calibration: every coding question, which is the
					// fault this whole area keeps producing. What each rule is for:
					// an empty Expect makes checkAnswer grade any text correct, and a
					// negative one is what TestCodingTierShape forbids should this band
					// ever move back above benchCodingTier. Both are belt and braces —
					// benchSelfGrades already drops an empty Expect on a string-graded
					// question — and neither matches anything in the committed band.
					if s.q.Match == benchMatchCodeExec {
						kept = append(kept, s)
						continue
					}
					if s.q.Expect == "" || strings.HasPrefix(s.q.Expect, "-") {
						continue
					}
					kept = append(kept, s)
				}
				src = kept
			}
			pick := benchStratifyByTask(src, band.Target)
			for i := range pick {
				pick[i].tier = band.Tier
			}
			if len(pick) < band.Target {
				fmt.Fprintf(os.Stderr, "  WARNING tier %d (headroom): only %d available, wanted %d\n",
					band.Tier, len(pick), band.Target)
			}
			fmt.Fprintf(os.Stderr, "  tier %2d: %2d selected of %2d available (target %d, headroom)\n",
				band.Tier, len(pick), len(src), band.Target)
			selected = append(selected, pick...)
			continue
		}
		var pick []scoredQuestion
		for _, s := range scored {
			if s.p >= band.MinRate && s.p < band.MaxRate {
				s.tier = band.Tier
				pick = append(pick, s)
			}
		}
		// Best discriminators first; the question id breaks ties so a re-run with
		// unchanged inputs produces no diff.
		sort.Slice(pick, func(i, j int) bool {
			if pick[i].d != pick[j].d {
				return pick[i].d > pick[j].d
			}
			return pick[i].q.ID < pick[j].q.ID
		})
		available := len(pick)
		if available > band.Target {
			pick = pick[:band.Target]
		} else if available < band.Target {
			// Never under-fill silently: a short tier changes the score's shape, and
			// the operator needs to know before it becomes a version bump.
			fmt.Fprintf(os.Stderr, "  WARNING tier %d: only %d candidates in band [%.2f,%.2f), wanted %d\n",
				band.Tier, available, band.MinRate, band.MaxRate, band.Target)
		}
		fmt.Fprintf(os.Stderr, "  tier %d: %2d selected of %2d available (target %d)\n",
			band.Tier, len(pick), available, band.Target)
		selected = append(selected, pick...)
	}
	if len(selected) == 0 {
		return nil, 0, fmt.Errorf("nothing selected — is the calibration complete?")
	}
	return selected, rho, nil
}

// benchHalfRate is one half of the fleet's pass rate on one question, over the
// observations that were actually measured. A censored observation is excluded
// from the numerator AND the denominator: counting it as a miss would credit
// the fast half with discrimination it did not demonstrate.
func benchHalfRate(half []benchRanked, id string) float64 {
	n, seen := 0, 0
	for _, f := range half {
		if f.censored[id] {
			continue
		}
		if _, ok := f.pass[id]; !ok {
			continue
		}
		seen++
		if f.pass[id] {
			n++
		}
	}
	if seen == 0 {
		return 0
	}
	return float64(n) / float64(seen)
}

func benchRenderFile(selected []scoredQuestion, pool *benchPool, rho float64, sandboxURL string) ([]byte, error) {
	var b strings.Builder
	fmt.Fprintf(&b, `// Code generated by "discrimen bench emit"; DO NOT EDIT.
//
// Questions sourced from LiveBench (%s),
// a contamination-limited benchmark whose items are replaced on a rolling
// monthly schedule. Pool fetched %s.
//
// These replace the hand-authored arithmetic tiers, which by benchmark_data.go's
// own measurements were the weakest discriminators in the set (frontier maths 31
// points of spread against traps 62 and unrecallable 58) and the ones needing
// re-authoring every time a model got better. The trap, unrecallable and
// world-model tiers stay hand-written there: they are both the best spread and
// the part LiveBench cannot supply, since their value is being absent from any
// training corpus.
//
// Every question here was measured against the live fleet before selection.
// Items every backend passed, no backend passed, or that ranked the fleet
// backwards were discarded; what remains is ranked by discrimination (top-half
// pass rate minus bottom-half). Tier is the measured fleet pass-rate band, not a
// LiveBench label. Spearman correlation between this pool and the hand-authored
// benchmark's q at calibration time: %+.2f.
//
// To refresh: bench fetch && bench calibrate && bench emit, and commit the
// result. No benchmarkVersion bump goes with it — each question is identified by
// its own content hash (identity.go), so the ones a refresh changes are re-asked
// and the ones it keeps are served from the permacache. Bump only to force the
// already-profiled fleet onto the new set at once.
//
// Regenerated %s.

package router

func init() {
	// Package-level vars are fully initialised before any init() runs, so
	// appending here is safe regardless of file order.
	benchmarkQuestions = append(benchmarkQuestions, benchmarkQuestionsLive...)
}

var benchmarkQuestionsLive = []benchmarkQ{
`, pool.Source, pool.FetchedAt, rho, time.Now().UTC().Format("2006-01-02"))

	lastTier := 0
	for _, s := range selected {
		if s.tier != lastTier {
			fmt.Fprintf(&b, "\n\t// ---- Tier %d — measured fleet pass rate %.0f%%–%.0f%% ----\n",
				s.tier, benchBandFor(s.tier).MinRate*100, benchBandFor(s.tier).MaxRate*100)
			lastTier = s.tier
		}
		fmt.Fprintf(&b, "\t// %s %s  p=%.2f d=%.2f\n", s.q.Task, s.q.ID[:12], s.p, s.d)
		if s.q.Match == benchMatchCodeExec {
			// A coding question ships its TEST CASES instead of an Expect. They
			// have to be embedded rather than fetched at runtime for the same
			// reason the whole generated file is committed: a quality score is a
			// percentage over the exam that was sat, and two workers are only
			// comparable on one absolute scale if they sat the same one. Test
			// cases fetched at grading time are an exam that can differ between
			// two workers, or between one worker and itself a week later, with
			// nothing in the diff to show for it. Embedding them also puts them
			// in the qid — benchQuestionQID hashes Code.Tests — so a case list
			// that DOES change makes a new question rather than silently
			// re-grading an old one.
			cases, err := benchEmitCases(sandboxURL, s.q)
			if err != nil {
				return nil, fmt.Errorf("tests for %s: %w", s.q.ID, err)
			}
			// Prefix travels WITH the question. A completion answer is a fragment
			// that only runs when appended to the partial solution the prompt
			// showed, so a generated file that omitted it would grade bare
			// fragments in production — the original bug, restored silently, and
			// invisible to checkPrefixHonoured because that only fires when a
			// prefix was actually sent.
			fmt.Fprintf(&b, "\t{Tier: %d, Match: %q, Prompt: %q, Code: &benchCode{Class: %q, Func: %q, Prefix: %q, Tests: []benchCase{\n",
				s.tier, s.q.Match, s.q.Prompt, s.q.Code.Class, s.q.Code.Func, s.q.Code.Prefix)
			for _, c := range cases {
				fmt.Fprintf(&b, "\t\t{Input: %q, Output: %q, Testtype: %q},\n", c.Input, c.Output, c.Testtype)
			}
			fmt.Fprintf(&b, "\t}}},\n")
			continue
		}
		fmt.Fprintf(&b, "\t{Tier: %d, Match: %q, Expect: %q, Prompt: %q},\n",
			s.tier, s.q.Match, s.q.Expect, s.q.Prompt)
	}
	b.WriteString("}\n")
	return format.Source([]byte(b.String()))
}

func benchBandFor(tier int) benchTierTarget {
	for _, b := range benchTierTargets {
		if b.Tier == tier {
			return b
		}
	}
	return benchTierTargets[0]
}

// benchSpearman is the rank correlation between two equal-length series. With a
// fleet of five or six it is coarse, but it is the right statistic for the
// question being asked — do the two instruments RANK the fleet the same way —
// and coarse beats absent.
func benchSpearman(a, bs []float64) float64 {
	n := len(a)
	if n < 2 || n != len(bs) {
		return 0
	}
	ra, rb := benchRanks(a), benchRanks(bs)
	var d2 float64
	for i := range ra {
		d := ra[i] - rb[i]
		d2 += d * d
	}
	return 1 - (6*d2)/(float64(n)*(float64(n*n)-1))
}

// benchRanks returns average ranks, so ties don't bias the correlation.
func benchRanks(xs []float64) []float64 {
	type iv struct {
		i int
		v float64
	}
	s := make([]iv, len(xs))
	for i, v := range xs {
		s[i] = iv{i, v}
	}
	sort.Slice(s, func(i, j int) bool { return s[i].v < s[j].v })
	out := make([]float64, len(xs))
	for i := 0; i < len(s); {
		j := i
		for j+1 < len(s) && s[j+1].v == s[i].v {
			j++
		}
		avg := float64(i+j)/2 + 1
		for k := i; k <= j; k++ {
			out[s[k].i] = avg
		}
		i = j + 1
	}
	return out
}

// benchStratifyByTask takes up to n questions, dealt round-robin across their
// source tasks so one task's quirks cannot fill a headroom band on its own.
// Deterministic: tasks in name order, questions by id within each.
func benchStratifyByTask(src []scoredQuestion, n int) []scoredQuestion {
	if n <= 0 || len(src) == 0 {
		return nil
	}
	byTask := map[string][]scoredQuestion{}
	for _, s := range src {
		byTask[s.q.Task] = append(byTask[s.q.Task], s)
	}
	tasks := make([]string, 0, len(byTask))
	for k := range byTask {
		tasks = append(tasks, k)
		sort.Slice(byTask[k], func(i, j int) bool { return byTask[k][i].q.ID < byTask[k][j].q.ID })
	}
	sort.Strings(tasks)
	out := make([]scoredQuestion, 0, n)
	for round := 0; len(out) < n; round++ {
		progressed := false
		for _, task := range tasks {
			if round < len(byTask[task]) {
				out = append(out, byTask[task][round])
				progressed = true
				if len(out) == n {
					return out
				}
			}
		}
		if !progressed {
			break
		}
	}
	return out
}

// benchSelfGrades reports whether the production grader accepts the question's
// OWN expected answer, in the reply shapes a model actually produces. Mirrors
// benchmark_test.go's TestExpertTierAnswersGrade so a question that would fail
// the build is never selected in the first place.
func benchSelfGrades(q poolQuestion) bool {
	// A code-exec question has no Expect by construction — its ground truth is a
	// set of test cases — so the string-shape probes below cannot say anything
	// about it. Worse, they would PASS it vacuously: checkAnswer against an empty
	// Expect returns true for any text at all, including a wrong program. Check
	// what grading such a question actually requires instead.
	if q.Match == benchMatchCodeExec {
		if q.Code == nil || (q.Code.PublicJSON == "" && !q.Code.HasPrivate) {
			return false // nothing to run the answer against
		}
		// The sidecar needs an ENTRY POINT for a functional test case, and this
		// checked Prefix and PublicJSON and HasPrivate but never Func. A question
		// with an empty Func fails sandbox/runner.py's resolve_entry — "no entry
		// function was given for a functional test" — before a single case runs,
		// so it fails every worker in calibration, is classified as p=0 "nobody can
		// pass this", and ships in the ceiling band as an item nobody can answer.
		// Unlike the other faults in this function it is invisible from the pool
		// data alone: the question looks complete.
		if !benchCodeEntryOK(q) {
			return false
		}
		// A completion answer is a fragment and is only runnable when appended to
		// the partial solution. Emitting one without its prefix restores the bug
		// that scored the two strongest workers 0% on this task, so it is caught
		// here rather than in production.
		return q.Task != "coding_completion" || q.Code.Prefix != ""
	}
	// Every other mode compares against Expect, and an EMPTY one grades every
	// answer correct — a single malformed question would lift the whole fleet's
	// score. The shapes below cannot catch it, because they are built from Expect.
	if strings.TrimSpace(q.Expect) == "" {
		return false
	}
	bq := benchmarkQ{Prompt: q.Prompt, Expect: q.Expect, Match: q.Match}
	shapes := []string{q.Expect, "The answer is " + q.Expect + "."}
	switch q.Match {
	case "mcq":
		shapes = append(shapes, "Working through each option, the correct one is "+q.Expect)
	case "contains":
		shapes = append(shapes, "Applying the rules in order gives "+q.Expect+".")
	default:
		shapes = append(shapes, "Substituting into the formula gives "+q.Expect)
	}
	for _, a := range shapes {
		if !checkAnswer(bq, a) {
			return false
		}
	}
	return true
}

// benchCodeEntryOK reports whether a coding question names the entry point its
// own test cases will demand.
//
// The rule is the sidecar's, not this file's, and is read straight off
// sandbox/runner.py's run_grade: resolve_entry is called if ANY case has a
// testtype other than "stdin", and it raises the moment Func is empty. A
// stdin-only question needs no entry point at all and passing one would be
// meaningless — which is why "Func must be set" is not the rule and 38 of this
// pool's 78 LCB_generation rows (all stdin) would be wrongly rejected by it.
//
// A case with NO testtype counts as functional, matching runner.py's own
// `case.get("testtype") or "functional"` default. So does a question whose
// public cases cannot be parsed or are absent: the testtype is then unknowable
// here, and the sidecar's default would demand an entry point. Erring towards
// "needs one" costs at most a question; erring the other way ships one that can
// only ever grade as wrong.
func benchCodeEntryOK(q poolQuestion) bool {
	if q.Code == nil {
		return false
	}
	if q.Code.Func != "" {
		return true
	}
	var cases []benchCase
	if q.Code.PublicJSON == "" || json.Unmarshal([]byte(q.Code.PublicJSON), &cases) != nil || len(cases) == 0 {
		return false
	}
	for _, c := range cases {
		if !strings.EqualFold(strings.TrimSpace(c.Testtype), "stdin") {
			return false
		}
	}
	return true
}

// benchEmitCasesMax bounds how many test cases a coding question ships with.
//
// The private set is unbounded — one LiveBench question decodes to thousands —
// and every case is wall-clock in every future profiling run, on every worker,
// forever. A bounded sample answers the same question ("does this program
// work?") at a fixed cost. Public cases are always kept because they are what
// the prompt showed the model; the sample fills the rest from the private set,
// which is what stops a submission scoring by hardcoding the examples it was
// given.
const benchEmitCasesMax = 24

// benchEmitCases resolves one coding question's cases at emit time, decoding the
// private blob through the sandbox — the only place allowed to unpickle it.
func benchEmitCases(sandboxURL string, q poolQuestion) ([]benchCase, error) {
	all, err := benchCodeTests(sandboxURL, q)
	if err != nil {
		return nil, err
	}
	if len(all) == 0 {
		return nil, fmt.Errorf("no test cases")
	}
	if len(all) <= benchEmitCasesMax {
		return all, nil
	}
	// Keep the public cases, then take an evenly spaced stride through the rest
	// rather than the first N: LiveBench orders private cases by size, so a head
	// slice would be every tiny input and would miss the ones that catch an
	// O(n^2) solution.
	pub := 0
	if q.Code.PublicJSON != "" {
		var p []benchCase
		if json.Unmarshal([]byte(q.Code.PublicJSON), &p) == nil {
			pub = len(p)
		}
	}
	if pub > benchEmitCasesMax {
		return all[:benchEmitCasesMax], nil
	}
	out := append([]benchCase(nil), all[:pub]...)
	rest, want := all[pub:], benchEmitCasesMax-pub
	stride := float64(len(rest)) / float64(want)
	for i := 0; i < want; i++ {
		out = append(out, rest[int(float64(i)*stride)])
	}
	return out, nil
}
