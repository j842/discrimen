package router

import "testing"

// The 2026-08-30 wedge, as arithmetic. A worker advertising 192K was handed a
// prompt whose real size was 195,302 tokens together with a 62,770-token output
// budget, and vLLM answered by truncating max_new_tokens and starting a prefill
// it could never finish — 0% GPU, every later request queued behind it. Both
// readings that produced that budget are fixed here: the window is the one
// routing enforces, and the prompt is sized at the densest ratio rather than the
// nominal one used for ranking.
func TestBudgetCeilingIsPessimisticOnBothInputs(t *testing.T) {
	// A probe that measured a real ceiling must bound the output budget, exactly
	// as it bounds admission. Advertised 128K, measured 16K.
	probed := &Backend{
		BackendRegistration: BackendRegistration{ContextK: 128},
		ContextProbe:        &ContextProbe{UsableTokens: 16 * 1024, AdvertisedTokens: 128 * 1024},
	}
	got := budgetCeiling(probed, jobCost{promptTokens: 8 * 1024})
	if want := 16*1024 - 8*1024 - contextClampMargin; got != want {
		t.Errorf("ceiling on a probed worker = %d, want %d — sized against the advertised window, "+
			"which is how an output budget outruns the context routing admits", got, want)
	}

	// Dense content: the nominal 3.0 divisor under-counts JSON (measured at 2.25
	// chars/token on 2026-08-30), so the ceiling must be computed from the chars,
	// not from the ranking estimate that came with them.
	const chars = 300000
	dense := &Backend{BackendRegistration: BackendRegistration{ContextK: 128}}
	nominal := tokensForChars(chars, defaultCharsPerToken)
	got = budgetCeiling(dense, jobCost{promptTokens: nominal, promptChars: chars})
	want := 128*1024 - tokensForChars(chars, minCharsPerToken) - contextClampMargin
	if want < contextClampFloor {
		want = contextClampFloor
	}
	if got != want {
		t.Errorf("ceiling on dense text = %d, want %d — the nominal divisor granted %d more tokens "+
			"than the worker had room for", got, want, got-want)
	}

	// Sparse text: the nominal estimate is already the pessimistic one, so the
	// densest reading must never be allowed to INFLATE the available room.
	sparse := jobCost{promptTokens: tokensForChars(chars, minCharsPerToken), promptChars: chars}
	if got, floor := budgetCeiling(dense, sparse), contextClampFloor; got < floor {
		t.Errorf("ceiling = %d, want at least the floor %d", got, floor)
	}

	// Without chars there is nothing to re-size from; the estimate stands.
	plain := &Backend{BackendRegistration: BackendRegistration{ContextK: 16}}
	if got, want := budgetCeiling(plain, jobCost{promptTokens: 1000}), 16*1024-1000-contextClampMargin; got != want {
		t.Errorf("ceiling with no promptChars = %d, want %d", got, want)
	}
}

// The measured chars-per-token ratio is biased toward over-estimating on purpose
// (calibration moves toward denser at 0.5 and away at 0.1). That bias is right
// until it empties the candidate set, because the next step is the overflow
// gamble that wedged a GPU. optimistic() is the second opinion.
func TestOptimisticSizingAdmitsAPromptTheMeasuredRatioRejects(t *testing.T) {
	ratios := newTokenRatios()
	// Prose measured at 4.91 chars/token against a model whose learned ratio is
	// 2.585 — the real 2026-08-30 numbers.
	ratios.byModel["prose-model"] = 2.585
	const chars = 400000
	f := hardFilter{promptChars: chars, ratios: ratios}
	b := &Backend{BackendRegistration: BackendRegistration{ContextK: 128, Model: "prose-model"}}

	measured := f.contextNeededK(b)
	optimistic := f.optimistic().contextNeededK(b)
	if optimistic >= measured {
		t.Fatalf("optimistic sizing = %dK, measured = %dK — the point is that it asks for less", optimistic, measured)
	}
	if reason := admitReason(b, f); reason == "" {
		t.Fatalf("measured sizing admitted %dK on a 128K worker; the test needs a prompt it rejects", measured)
	}
	if reason := admitReason(b, f.optimistic()); reason != "" {
		t.Errorf("optimistic sizing still rejected (%s) — a prompt that really is %d chars of prose "+
			"fits this worker, and rejecting it forces the overflow gamble", reason, chars)
	}
}

// The relaxation must not become a way around the other hard filters: it only
// ever re-asks the context question.
func TestOptimisticSizingDoesNotRelaxOtherFilters(t *testing.T) {
	f := hardFilter{promptChars: 1000, needTools: true, ratios: newTokenRatios()}
	noTools := &Backend{BackendRegistration: BackendRegistration{ContextK: 128, Features: []string{"chat"}}}
	if reason := admitReason(noTools, f.optimistic()); reason == "" {
		t.Error("optimistic sizing admitted a worker with no tools support")
	}
}

// A worker that misses one deadline under load is not broken; one that misses
// two in a row is. Pulling on the first is how an overload becomes an outage.
func TestServeCheckRequiresConsecutiveFailures(t *testing.T) {
	r := &Registry{backends: map[string]*Backend{"w": {}}}
	if pull := r.noteServeResult("w", false); pull {
		t.Error("pulled the worker on a single missed deadline")
	}
	if pull := r.noteServeResult("w", true); pull {
		t.Error("pulled the worker after it recovered")
	}
	// A success must reset the run, not merely pause it.
	if pull := r.noteServeResult("w", false); pull {
		t.Error("pulled on the first failure after a success — the counter did not reset")
	}
	if pull := r.noteServeResult("w", false); !pull {
		t.Errorf("did not pull after %d consecutive failures", serveCheckFailures)
	}
	if pull := r.noteServeResult("missing", false); pull {
		t.Error("reported a pull for a backend that is not registered")
	}
}
