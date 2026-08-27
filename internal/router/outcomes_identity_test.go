package router

import (
	"context"
	"strings"
	"testing"
	"time"
)

// Observations are filed under the MODEL that answered, and every read path has
// to agree about that. This file is the guard on the ones that were found
// disagreeing, plus two properties of the neighbour scan that nothing pinned.

// bankObs builds a bench observation whose MODEL and WORKER are stated
// separately, which the shared obs() helper cannot express — it derives both
// from one id, so every fixture built with it has the two fields agreeing by
// construction and cannot detect a read path filtering on the wrong one.
func bankObs(qid, model, backend string, correct bool) Observation {
	return Observation{
		QID: qid, ModelHash: model, Backend: backend,
		Thinking: true, Correct: correct, LatencyMS: 100,
		Source: obsSourceBench, At: time.Unix(0, 0),
	}
}

// The validation harness has to identify an answerer the same way routing does.
//
// It did not: predictExcluding filtered observations on o.Backend while
// predictFrom filters on o.ModelHash. `llm-cpu-gemma-26B` runs on two hosts
// here, and a re-profile of the second supersedes the first on the questions it
// reached — so one model's rows carry two different backend ids, split between
// whichever deployment measured each question last. Filtered by box, the harness
// saw two answerers with half the evidence each, declined predictions routing
// would have made, and reported a coverage figure and a worker count that
// described a fleet the router does not have.
//
// The fixture is that situation, minimally: one model, one thinking mode, four
// near-identical questions, and the backend id alternating between the two
// hosts that measured them.
func TestValidationIdentifiesAnAnswererByModelNotByBox(t *testing.T) {
	m := newTestMatrix(t)
	const model = "m-shared-weights"
	var recs []Observation
	for i, qid := range []string{"q1", "q2", "q3", "q4"} {
		m.setVector(qid, vec(1, float64(i)/100))
		host := "host-a"
		if i%2 == 1 {
			host = "host-b" // the same weights, measured on the other box
		}
		// Not unanimous, or AUROC is undefined and the report says nothing.
		recs = append(recs, bankObs(qid, model, host, i < 3))
	}
	if err := m.record(context.Background(), recs); err != nil {
		t.Fatalf("record: %v", err)
	}

	rep := m.validate(nil)
	if rep.Workers != 1 {
		t.Errorf("validate reported %d workers for ONE model on two hosts — the harness is counting "+
			"boxes, and everything it derives per worker is split the same way", rep.Workers)
	}
	// Every question has three neighbours from the same model, which clears
	// outcomeMinObservations. Split by box it has one or two, and declines.
	if rep.InDistribution.Predicted != 4 {
		t.Errorf("in-distribution predicted %d of %d bench rows, want all 4 — evidence about one model "+
			"is being split across the workers that happened to measure it",
			rep.InDistribution.Predicted, rep.InDistribution.N)
	}
	if rep.InDistribution.N != 4 {
		t.Errorf("in-distribution N = %d, want 4", rep.InDistribution.N)
	}
	// The pair correlation asks "did the same answerer differ between these two
	// questions", which is the same identity question and had the same fault.
	if r := distanceAgreementCorrelation(map[string][]float64{
		"q1": normalize(vec(1, 0)), "q2": normalize(vec(1, 0.01)),
		"q3": normalize(vec(0, 1)), "q4": normalize(vec(0.01, 1)),
	}, map[string][]Observation{
		"q1": {bankObs("q1", model, "host-a", true)},
		"q2": {bankObs("q2", model, "host-b", true)},
		"q3": {bankObs("q3", model, "host-a", false)},
		"q4": {bankObs("q4", model, "host-b", true)},
	}); r == 0 {
		t.Error("distance/agreement r = 0 with four questions from one model — pairs are being matched " +
			"on the backend id, so a model measured on two hosts contributes none")
	}
}

// The permanent result cache is keyed by model, and so is the display view. Both
// were checked by hand against a fixture whose model and worker agree; this
// states the property that actually matters — a row measured on one box is
// reachable from another deployment of the same weights, and only its LATENCY is
// withheld.
func TestEveryReadPathFindsARowMeasuredOnAnotherBox(t *testing.T) {
	m := newTestMatrix(t)
	const model = "m-shared-weights"
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(1, 0.01))
	if err := m.record(context.Background(), []Observation{
		bankObs("q1", model, "host-a", true),
		bankObs("q2", model, "host-a", true),
	}); err != nil {
		t.Fatalf("record: %v", err)
	}

	if hit, sameWorker, ok := m.cachedVerdict("q1", model, "host-b", true); !ok {
		t.Error("cachedVerdict missed a row for the same model on another host — the generation it exists " +
			"to save is spent again")
	} else if sameWorker || !hit.Correct {
		t.Errorf("cachedVerdict said sameWorker=%v correct=%v; the verdict travels and the latency "+
			"must not", sameWorker, hit.Correct)
	}
	if rate, n := m.summary(model, true); n != 2 || rate != 1 {
		t.Errorf("summary = %.2f over %d rows, want 1.00 over 2", rate, n)
	}
	if got := m.bankRates(true)[model]; got.total != 2 || got.correct != 2 {
		t.Errorf("bankRates tally = %+v, want 2/2", got)
	}
	if s := m.summarise(model, true, nil); s.Total != 2 || s.Correct != 2 || s.Insufficient {
		t.Errorf("summarise = %+v, want 2/2 and not insufficient", s)
	}
	if p := m.predict(vec(1, 0.005), model, true); !p.known() || p.Correct != 1 {
		t.Errorf("predict = %+v, want a known 1.00", p)
	}
	if ids := m.backendsWithEvidence(); len(ids) != 1 || ids[0] != model {
		t.Errorf("backendsWithEvidence = %v, want the one model hash", ids)
	}
}

// summary() and bankRates() are one walk of the observation map now, so they
// cannot drift apart on what a bench row is. Stated as agreement rather than as
// a call count, because the point is the answer, not the implementation.
func TestSummaryAgreesWithTheBankRateItSharesAWalkWith(t *testing.T) {
	m := newTestMatrix(t)
	ctx := context.Background()
	if err := m.record(ctx, []Observation{
		obs("q1", "w", true, true, 100, obsSourceBench),
		obs("q2", "w", true, false, 100, obsSourceBench),
		obs("q3", "w", true, true, 100, obsSourceBench),
		// Neither of these is bank evidence: a judged verdict, and the other mode.
		obs("q4", "w", true, false, 100, obsSourceJudge),
		obs("q5", "w", false, false, 100, obsSourceBench),
	}); err != nil {
		t.Fatalf("record: %v", err)
	}
	rate, n := m.summary(mh("w"), true)
	tally := m.bankRates(true)[mh("w")]
	if n != tally.total || n != 3 {
		t.Errorf("summary counted %d rows, bankRates counted %d, want 3 for both", n, tally.total)
	}
	if rate != tally.rate() {
		t.Errorf("summary rate %.4f vs bankRates rate %.4f", rate, tally.rate())
	}
	// A model with nothing in this mode is "not measured" here and fleet-neutral
	// in the fallback ordering, and those are deliberately different numbers.
	if _, none := m.summary(mh("never-seen"), true); none != 0 {
		t.Errorf("summary of an unprofiled model returned %d observations", none)
	}
	if r := m.bankRates(true)[mh("never-seen")].rate(); r != 0.5 {
		t.Errorf("bankRates of an unprofiled model = %.2f, want the fleet-neutral 0.5", r)
	}
}

// The measured regression this file's counters exist for was a scan of the
// VECTOR map once per candidate — 13ms per routed request on a seven-worker
// fleet. fullScans could never have caught it: neighboursOf does not touch the
// observation map. One scan for the whole request, whatever the fleet size.
func TestChooseByOutcomeScansTheVectorMapOnce(t *testing.T) {
	noExploration(t)
	m := newTestMatrix(t)
	ctx := context.Background()
	m.setVector("q1", vec(1, 0))
	m.setVector("q2", vec(0.999, 0.02))
	var recs []Observation
	cands := make([]*Backend, 0, 7)
	for i := 0; i < 7; i++ {
		id := "w" + string(rune('a'+i))
		recs = append(recs, obs("q1", id, true, true, 100, obsSourceBench))
		recs = append(recs, obs("q2", id, true, true, 100, obsSourceBench))
		cands = append(cands, testBackend(id, float64(10*(i+1))))
	}
	if err := m.record(ctx, recs); err != nil {
		t.Fatalf("record: %v", err)
	}
	// A prompt the matrix DOES know, so the primary path runs — the one where a
	// per-candidate predict() would reinstate the fault.
	m.vecScans.Store(0)
	m.fullScans.Store(0)
	_, reason, _ := m.chooseByOutcome(cands, vec(1, 0.01), true, jobCost{outputTokens: 200, mode: thinkingOn})
	if !strings.Contains(reason, "outcome:p=") {
		t.Fatalf("expected the measured path, got reason %q", reason)
	}
	if n := m.vecScans.Load(); n != 1 {
		t.Errorf("%d scans of the vector map for %d candidates, want 1 — routing is rescanning every "+
			"vector per candidate again, which measured 13ms a request", n, len(cands))
	}
	if n := m.fullScans.Load(); n != 0 {
		t.Errorf("%d walks of the observation map on the measured path, want 0 — the overall tally is "+
			"the fallback's, and computing it here pays for a number this branch does not use", n)
	}
}

// Which neighbours survive the cut must not depend on map iteration order.
//
// With more than outcomeNeighbours questions at the same similarity — a bank of
// templated near-duplicates is exactly that — an unordered truncation kept a
// different twelve on each call, so the same prompt against unchanged evidence
// could predict differently from one request to the next. A route that is not
// reproducible cannot be explained to whoever is reading X-Llm-Route.
func TestNeighbourSelectionIsDeterministicAcrossTies(t *testing.T) {
	m := newTestMatrix(t)
	ctx := context.Background()
	// Every question sits at the SAME cosine from the query, so nothing but the
	// tie-break decides which outcomeNeighbours of them are kept. Half are
	// answered correctly and half are not, so a different cut is a different
	// predicted hit rate rather than the same number by luck.
	const n = outcomeNeighbours * 3
	var recs []Observation
	for i := 0; i < n; i++ {
		qid := "q" + string(rune('a'+i))
		m.setVector(qid, vec(1, 0))
		recs = append(recs, obs(qid, "w", true, i%2 == 0, 100, obsSourceBench))
	}
	if err := m.record(ctx, recs); err != nil {
		t.Fatalf("record: %v", err)
	}
	first := m.predict(vec(1, 0), mh("w"), true)
	for i := 0; i < 200; i++ {
		got := m.predict(vec(1, 0), mh("w"), true)
		if got.Correct != first.Correct || got.Observations != first.Observations {
			t.Fatalf("call %d predicted %.4f over %d observations, the first call predicted %.4f over "+
				"%d — the neighbour cut is following map iteration order",
				i, got.Correct, got.Observations, first.Correct, first.Observations)
		}
	}
	if first.Observations != outcomeNeighbours {
		t.Errorf("prediction rested on %d observations, want the %d nearest", first.Observations, outcomeNeighbours)
	}
}

// bankQIDByPrompt maps a stored result back to its question BY PROMPT, so two
// bank questions sharing a prompt would collapse onto one qid and one of them
// would silently never be recoverable from a stored profile. Cheap to assert,
// and the alternative is finding out from a worker whose backfilled profile is
// quietly short.
func TestBankPromptsAreUnique(t *testing.T) {
	seen := make(map[string]benchmarkQ, len(benchmarkQuestions))
	for _, q := range benchmarkQuestions {
		key := strings.TrimSpace(q.Prompt)
		if prev, dup := seen[key]; dup {
			t.Errorf("two bank questions share a prompt (%q vs %q match mode, tier %d vs %d): "+
				"bankQIDByPrompt keys on the prompt, so one of them cannot be backfilled",
				prev.Match, q.Match, prev.Tier, q.Tier)
			continue
		}
		seen[key] = q
	}
	if len(seen) != len(benchmarkQuestions) {
		t.Errorf("%d distinct prompts across %d bank questions", len(seen), len(benchmarkQuestions))
	}
}
