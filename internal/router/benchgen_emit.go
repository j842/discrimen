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
	"fmt"
	"go/format"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// benchTierTargets is how many questions to emit per tier, and the measured
// pass-rate band each tier draws from. The defaults mirror the shape of the
// hand-authored arithmetic tiers being replaced (4:25, 5:7, 7:5, 8:13 = 50), so
// the total question count — and therefore the profiling budget, which is the
// real constraint at 8–30 min per worker for 97 questions — stays put.
var benchTierTargets = []struct {
	Tier    int
	Target  int
	MinRate float64 // inclusive
	MaxRate float64 // exclusive; p==1 is handled by the reserve bands instead
	Reserve int     // reserveNone | reserveFloor | reserveCeiling
}{
	{Tier: 2, Target: 12, Reserve: reserveFloor},
	{Tier: 4, Target: 25, MinRate: 0.75, MaxRate: 1.00},
	{Tier: 5, Target: 60, MinRate: 0.50, MaxRate: 0.75},
	{Tier: 7, Target: 70, MinRate: 0.25, MaxRate: 0.50},
	{Tier: 8, Target: 70, MinRate: 0.00, MaxRate: 0.25},
	{Tier: 12, Target: 40, Reserve: reserveCeiling},
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
// re-fetch, a re-calibration and a benchmarkVersion bump that re-profiles the
// whole fleet.
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
	id      string
	score   float64
	routerQ int
	pass    map[string]bool
}

type scoredQuestion struct {
	q    poolQuestion
	tier int
	p    float64
	d    float64
}

func benchEmit() error {
	pool, err := benchLoadPool()
	if err != nil {
		return err
	}
	calib := benchLoadCalibration()
	if len(calib.Results) == 0 {
		return fmt.Errorf("no calibration data (run `bench calibrate` first)")
	}

	// Rank backends by the score they got on THIS pool, not by the router's q:
	// the top/bottom split has to come from the instrument being calibrated, or
	// every question that disagrees with the old benchmark scores as noise.
	var fleet []benchRanked
	for _, r := range calib.Results {
		if len(r.Pass) == 0 {
			continue
		}
		n := 0
		for _, ok := range r.Pass {
			if ok {
				n++
			}
		}
		fleet = append(fleet, benchRanked{r.Backend, float64(n) / float64(len(r.Pass)), r.RouterQuality, r.Pass})
	}
	if len(fleet) < 2 {
		return fmt.Errorf("need at least 2 calibrated backends to measure discrimination, have %d", len(fleet))
	}
	sort.Slice(fleet, func(i, j int) bool { return fleet[i].score > fleet[j].score })

	fmt.Fprintln(os.Stderr, "fleet on this pool:")
	poolScores := make([]float64, 0, len(fleet))
	routerScores := make([]float64, 0, len(fleet))
	for _, f := range fleet {
		fmt.Fprintf(os.Stderr, "  %-30s pool=%3.0f%%  router q=%d\n", f.id, f.score*100, f.routerQ)
		poolScores = append(poolScores, f.score)
		routerScores = append(routerScores, float64(f.routerQ))
	}

	// The correlation check: do the two instruments RANK the fleet the same way?
	// A high value means the hand-authored set is tracking real capability and
	// this is a refresh; a low one means something is wrong with one of them and
	// bumping benchmarkVersion would bake it in.
	rho := benchSpearman(poolScores, routerScores)
	fmt.Fprintf(os.Stderr, "\nSpearman(pool, router q) = %+.2f  ", rho)
	switch {
	case rho >= 0.8:
		fmt.Fprintln(os.Stderr, "— the hand-authored set tracks capability; this is a refresh, not a rescue.")
	case rho >= 0.4:
		fmt.Fprintln(os.Stderr, "— partial agreement; read the disagreements before adopting.")
	default:
		fmt.Fprintln(os.Stderr, "— the instruments disagree. Investigate BEFORE bumping benchmarkVersion.")
	}

	// An odd-sized fleet drops its median worker rather than assigning it to a
	// half arbitrarily.
	half := len(fleet) / 2
	top, bottom := fleet[:half], fleet[len(fleet)-half:]

	var scored, floor, ceiling []scoredQuestion
	var noData, tooEasy, tooHard, backwards int
	for _, q := range pool.Questions {
		graded, passes := 0, 0
		for _, f := range fleet {
			ok, seen := f.pass[q.ID]
			if !seen {
				continue
			}
			graded++
			if ok {
				passes++
			}
		}
		if graded < len(fleet) {
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
	fmt.Fprintf(os.Stderr, "\n%d candidates: %d usable; dropped %d all-pass, %d all-fail, %d negative-discrimination, %d ungraded\n",
		len(pool.Questions), len(scored), tooEasy, tooHard, backwards, noData)

	var selected []scoredQuestion
	for _, band := range benchTierTargets {
		if band.Reserve != reserveNone {
			src := floor
			if band.Reserve == reserveCeiling {
				src = ceiling
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
		return fmt.Errorf("nothing selected — is the calibration complete?")
	}

	src, err := benchRenderFile(selected, pool, rho)
	if err != nil {
		return err
	}
	path := filepath.Join(benchAppDir(), "benchmark_data_live.go")
	if err := os.WriteFile(path, src, 0o644); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "\nwrote %s (%d questions)\n", path, len(selected))
	fmt.Fprint(os.Stderr, "\nNEXT: bump benchmarkVersion, delete tiers 4/5/7/8 from benchmark_data.go,\n"+
		"and commit both in the SAME commit — the question set is part of the profile cache key.\n")
	return nil
}

func benchHalfRate(half []benchRanked, id string) float64 {
	if len(half) == 0 {
		return 0
	}
	n := 0
	for _, f := range half {
		if f.pass[id] {
			n++
		}
	}
	return float64(n) / float64(len(half))
}

func benchRenderFile(selected []scoredQuestion, pool *benchPool, rho float64) ([]byte, error) {
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
// To refresh: bench fetch && bench calibrate && bench emit, then bump
// benchmarkVersion in the same commit — the question set is part of the profile
// cache key and must not change without it.
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
		fmt.Fprintf(&b, "\t{Tier: %d, Match: %q, Expect: %q, Prompt: %q},\n",
			s.tier, s.q.Match, s.q.Expect, s.q.Prompt)
	}
	b.WriteString("}\n")
	return format.Source([]byte(b.String()))
}

func benchBandFor(tier int) struct {
	Tier    int
	Target  int
	MinRate float64
	MaxRate float64
	Reserve int
} {
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
