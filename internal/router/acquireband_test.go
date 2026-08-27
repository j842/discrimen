package router

import "testing"

// Acquisition may reorder INSIDE the band the matrix judged interchangeable on
// correctness, and must not reorder across it.
//
// The matrix plan carries target == 0, which made `aboveBar` vacuously true for
// every candidate and collapsed the ladder to "cheapest local worker first"
// applied to the WHOLE ranked list — including the band the matrix had predicted
// wrong. A fleet of one local CPU worker and one strong paid endpoint served
// every hard prompt on the CPU worker for as long as it had a free slot, however
// confident the matrix was that it would get the answer wrong.
func TestAcquisitionDoesNotPreferAcrossTheCorrectnessBand(t *testing.T) {
	paid, local := bandFixture()

	// The matrix ranked the paid endpoint first and put the local worker in a
	// lower band: one candidate is interchangeable-on-correctness, not two.
	candidates := []*Backend{paid, local}

	pref := acquirePreferenceFor(candidates, 1)
	if pref.keep == nil {
		t.Fatal("no preference at all — the ladder found nothing to prefer")
	}
	kept := filterCandidates(candidates, pref.keep)
	if len(kept) != 1 || kept[0].ID != "paid-api" {
		t.Fatalf("preferred %v (why=%q), want just paid-api — the free/local ladder reached "+
			"below the correctness band", idsOf(kept), pref.why)
	}
	for i, weaker := range pref.weaker {
		for _, b := range filterCandidates(candidates, weaker) {
			if b.ID == "local-cpu" {
				t.Errorf("weaker rung %d admits local-cpu, which the matrix predicted wrong", i)
			}
		}
	}

	// With no band declared — the tier path, and the matrix's own fallback, both
	// of which have no correctness judgement to protect — the cost/locality ladder
	// applies across the whole list exactly as before.
	pref = acquirePreferenceFor(candidates, 0)
	kept = filterCandidates(candidates, pref.keep)
	if len(kept) != 1 || kept[0].ID != "local-cpu" {
		t.Fatalf("unbounded: preferred %v (why=%q), want local-cpu — the free/local ladder "+
			"should be unchanged where no band is protected", idsOf(kept), pref.why)
	}
}

// A band covering every candidate is not a constraint, and must not suppress the
// ladder — inside one band the local-vs-relay correction is still wanted, because
// a relayed queue depth is up to 15s stale while a local one is exact.
func TestAFullBandStillPrefersWithinItself(t *testing.T) {
	paid, local := bandFixture()
	candidates := []*Backend{paid, local}

	pref := acquirePreferenceFor(candidates, len(candidates))
	kept := filterCandidates(candidates, pref.keep)
	if len(kept) != 1 || kept[0].ID != "local-cpu" {
		t.Fatalf("all-in-one-band: preferred %v, want local-cpu", idsOf(kept))
	}
}

// bandFixture is a metered remote endpoint and a free local worker — the fleet
// shape the defect was found on. "Paid" is a price, not an API key: isFreeBackend
// reads InputPricePerMtok/OutputPricePerMtok.
func bandFixture() (paid, local *Backend) {
	paid = &Backend{BackendRegistration: BackendRegistration{ID: "paid-api", URL: "https://api", Model: "big"}}
	paid.InputPricePerMtok, paid.OutputPricePerMtok = 3, 15
	local = &Backend{BackendRegistration: BackendRegistration{ID: "local-cpu", URL: "http://local", Model: "small"}}
	return paid, local
}
