package router

import "testing"

// The matrix branch of planRoute NEVER declines, and the tier branch below it is
// therefore unreachable.
//
// planRoute's comment claimed the opposite for months — "supersedes the tier path
// wherever it has evidence, and declines where it does not, so the two can run
// side by side". That one sentence is why the tier machinery looked alive, why
// the online adapter that fed it looked enabled, and why three separate audits
// each had to re-derive the truth from the code. It is pinned here so the claim
// cannot quietly become fiction again.
//
// The mechanism: chooseByOutcome RANKS, it does not filter. With no evidence at
// all it falls through to its own bank-rate fallback and still returns every
// candidate. Its only empty return needs an empty input, which planRoute has
// already excluded.
func TestTheMatrixNeverDeclines(t *testing.T) {
	empty := newOutcomeMatrix(nil) // no observations, no vectors — as blind as it gets
	cands := []*Backend{testBackend("a", 10), testBackend("b", 100), testBackend("c", 50)}

	for _, tc := range []struct {
		name string
		vec  []float64
	}{
		{"no evidence and a query vector", vec(1, 0)},
		{"no evidence and a zero vector", vec(0, 0)},
		{"no evidence and no vector at all", nil},
	} {
		got, reason, able := empty.chooseByOutcome(cands, tc.vec, true, jobCost{outputTokens: 200})
		if len(got) != len(cands) {
			t.Errorf("%s: returned %d of %d candidates — if it can narrow, the branch below "+
				"planRoute's matrix arm becomes reachable and the whole dead-tier analysis changes",
				tc.name, len(got), len(cands))
		}
		if reason == "" {
			t.Errorf("%s: no route reason", tc.name)
		}
		if able != 0 {
			t.Errorf("%s: able=%d with no evidence, want 0 — acquisition would protect a band "+
				"the matrix never measured", tc.name, able)
		}
	}

	// The one input that DOES yield nothing is the one planRoute cannot pass.
	if got, _, _ := empty.chooseByOutcome(nil, vec(1, 0), true, jobCost{}); got != nil {
		t.Error("an empty candidate list produced candidates")
	}
}
