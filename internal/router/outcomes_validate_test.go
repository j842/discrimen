package router

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"
)

// The statistics have to be right before any verdict built on them means
// anything, and AUROC with ties is the easiest of them to get subtly wrong.
func TestAUROC(t *testing.T) {
	cases := []struct {
		name   string
		scores []float64
		labels []float64
		want   float64
	}{
		{"perfect separation", []float64{0.9, 0.8, 0.2, 0.1}, []float64{1, 1, 0, 0}, 1.0},
		{"perfectly wrong", []float64{0.1, 0.2, 0.8, 0.9}, []float64{1, 1, 0, 0}, 0.0},
		{"all tied is chance", []float64{0.5, 0.5, 0.5, 0.5}, []float64{1, 1, 0, 0}, 0.5},
		{"one label only is undefined", []float64{0.9, 0.1}, []float64{1, 1}, 0.0},
		{"partial", []float64{0.9, 0.4, 0.6, 0.1}, []float64{1, 1, 0, 0}, 0.75},
	}
	for _, c := range cases {
		if got := auroc(c.scores, c.labels); math.Abs(got-c.want) > 1e-9 {
			t.Errorf("%s: auroc = %.4f, want %.4f", c.name, got, c.want)
		}
	}
}

func TestPearsonAndBrier(t *testing.T) {
	if r := pearson([]float64{1, 2, 3}, []float64{2, 4, 6}); math.Abs(r-1) > 1e-9 {
		t.Errorf("perfect positive correlation = %.4f", r)
	}
	if r := pearson([]float64{1, 2, 3}, []float64{6, 4, 2}); math.Abs(r+1) > 1e-9 {
		t.Errorf("perfect negative correlation = %.4f", r)
	}
	if r := pearson([]float64{1, 1, 1}, []float64{1, 2, 3}); r != 0 {
		t.Errorf("zero-variance input = %.4f, want 0", r)
	}
	if b := brier([]float64{1, 0}, []float64{1, 0}); b != 0 {
		t.Errorf("perfect brier = %.4f", b)
	}
	if b := brier([]float64{0, 1}, []float64{1, 0}); b != 1 {
		t.Errorf("worst brier = %.4f, want 1", b)
	}
}

// The whole point of the harness: a matrix that has learned TOPIC rather than
// difficulty must look good in-distribution and collapse under a domain
// holdout. If this test ever passes trivially, the harness has stopped
// detecting the failure it was built for.
func TestDomainHoldoutDetectsTopicConfound(t *testing.T) {
	m := newTestMatrix(t)
	ctx := context.Background()
	var recs []Observation
	// Two well-separated topics. Within each, the worker's outcome is CONSTANT —
	// so a neighbour from the same topic predicts perfectly, and a neighbour from
	// the other topic predicts backwards. That is the confound, constructed.
	for i := 0; i < 10; i++ {
		mq := "m" + string(rune('a'+i))
		cq := "c" + string(rune('a'+i))
		m.setVector(mq, []float64{1, float64(i) / 200, 0})
		m.setVector(cq, []float64{0, float64(i) / 200, 1})
		recs = append(recs, obs(mq, "w", true, true, 10, obsSourceBench))  // aces maths
		recs = append(recs, obs(cq, "w", true, false, 10, obsSourceBench)) // fails code
	}
	_ = m.record(ctx, recs)
	domainOf := func(qid string) string {
		if qid[0] == 'm' {
			return "maths"
		}
		return "coding"
	}
	rep := m.validate(domainOf)

	// In-distribution, neighbours come from the same topic and the prediction is
	// perfect — which is exactly the misleading number.
	if rep.InDistribution.Predicted == 0 {
		t.Fatal("in-distribution produced no predictions")
	}
	// Held out, a domain's questions have no same-topic neighbours left. The
	// predictor should either decline (Predicted == 0) or do badly — what it must
	// NOT do is look as good as in-distribution.
	if rep.HoldoutMean.Predicted > 0 && rep.HoldoutMean.AUROC >= rep.InDistribution.AUROC {
		t.Errorf("domain holdout AUROC %.3f >= in-distribution %.3f — the harness is not "+
			"detecting the topic confound it exists to catch",
			rep.HoldoutMean.AUROC, rep.InDistribution.AUROC)
	}
	if rep.Workers != 1 || rep.Questions != 20 {
		t.Errorf("report shape wrong: %d workers, %d questions", rep.Workers, rep.Questions)
	}
}

// Degenerate questions bound what any router can achieve, so the fraction has to
// be reported honestly — including that a single worker is never "unanimous".
func TestDegenerateFractions(t *testing.T) {
	obsMap := map[string][]Observation{
		"everyone-passes": {
			obs("q", "a", true, true, 1, obsSourceBench),
			obs("q", "b", true, true, 1, obsSourceBench),
		},
		"everyone-fails": {
			obs("q", "a", true, false, 1, obsSourceBench),
			obs("q", "b", true, false, 1, obsSourceBench),
		},
		"discriminates": {
			obs("q", "a", true, true, 1, obsSourceBench),
			obs("q", "b", true, false, 1, obsSourceBench),
		},
		"single-worker": {obs("q", "a", true, true, 1, obsSourceBench)},
	}
	allCorrect, allFail := degenerateFractions(obsMap)
	// Three questions have two workers; the single-worker one is excluded.
	if math.Abs(allCorrect-1.0/3) > 1e-9 {
		t.Errorf("all-correct fraction = %.3f, want 0.333", allCorrect)
	}
	if math.Abs(allFail-1.0/3) > 1e-9 {
		t.Errorf("all-fail fraction = %.3f, want 0.333", allFail)
	}
}

// The distance/agreement correlation is kNN's central assumption made
// measurable: near questions should behave alike (negative correlation between
// similarity and disagreement).
func TestDistanceAgreementCorrelation(t *testing.T) {
	vecs := map[string][]float64{}
	obsMap := map[string][]Observation{}
	// Two tight clusters; a worker is consistent within a cluster and opposite
	// across them, so similar questions agree and dissimilar ones do not.
	for i := 0; i < 6; i++ {
		a, b := "a"+string(rune('a'+i)), "b"+string(rune('a'+i))
		vecs[a] = normalize([]float64{1, float64(i) / 100})
		vecs[b] = normalize([]float64{float64(i) / 100, 1})
		obsMap[a] = []Observation{obs(a, "w", true, true, 1, obsSourceBench)}
		obsMap[b] = []Observation{obs(b, "w", true, false, 1, obsSourceBench)}
	}
	r := distanceAgreementCorrelation(vecs, obsMap)
	if r > -0.5 {
		t.Errorf("distance/agreement r = %+.2f; a clustered matrix should be strongly "+
			"negative — near questions behaving alike is what makes a neighbour evidence", r)
	}
}

// An empty or single-worker matrix must produce a report that says so rather
// than a number that looks like a measurement.
func TestValidateOnThinEvidence(t *testing.T) {
	empty := newTestMatrix(t)
	rep := empty.validate(nil)
	if rep.Questions != 0 || rep.Workers != 0 {
		t.Errorf("empty matrix reported %d questions / %d workers", rep.Questions, rep.Workers)
	}
	if v := rep.verdict(); !strings.Contains(v, "not enough evidence") {
		t.Errorf("empty matrix verdict does not say so: %q", v)
	}
}

// auroc must terminate on a NaN score. NaN == NaN is false, so the tie-scan
// could not advance and the outer loop pinned a core for the life of the
// process — reachable from one authenticated /admin/outcomes?validate=1.
func TestAUROCTerminatesOnNaN(t *testing.T) {
	done := make(chan float64, 1)
	go func() {
		done <- auroc([]float64{math.NaN(), 0.5, math.NaN(), 0.1}, []float64{1, 1, 0, 0})
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("auroc did not terminate on NaN input — one admin request pins a core forever")
	}
}
