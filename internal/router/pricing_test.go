package router

import (
	"strings"
	"testing"
)

// testPriceJSON is a hand-built snapshot in the shape `prices fetch` writes,
// carrying the collisions that matter: a bare key and a reseller's prefixed one
// at different prices, a model that exists only under its vendor prefix, and a
// reseller's listing of open weights anyone can also run themselves (the
// azure_ai row, copied from the shipped snapshot).
const testPriceJSON = `{
 "source": "test",
 "fetched_at": "2026-01-01T00:00:00Z",
 "models": {
  "gpt-4o": {"input_cost_per_token":0.0000025,"output_cost_per_token":0.00001,"max_input_tokens":128000,"litellm_provider":"openai"},
  "azure/gpt-4o": {"input_cost_per_token":0.00000275,"output_cost_per_token":0.000011,"max_input_tokens":128000,"litellm_provider":"azure"},
  "azure_ai/gpt-oss-120b": {"input_cost_per_token":0.00000015,"output_cost_per_token":0.0000006,"max_input_tokens":131072,"litellm_provider":"azure_ai"},
  "claude-sonnet-4-5": {"input_cost_per_token":0.000003,"output_cost_per_token":0.000015,"max_input_tokens":200000,"litellm_provider":"anthropic"},
  "xai/grok-4": {"input_cost_per_token":0.000003,"output_cost_per_token":0.000015,"max_input_tokens":256000,"litellm_provider":"xai"},
  "openrouter/qwen/qwen3-32b": {"input_cost_per_token":0.0000001,"output_cost_per_token":0.00000028,"max_input_tokens":40960,"litellm_provider":"openrouter"},
  "no-context": {"input_cost_per_token":0.000001,"output_cost_per_token":0.000002,"litellm_provider":"whoever"}
 }
}`

// TestPriceLookupTolerance: providers spell the same model several ways, and an
// operator should not have to know which one LiteLLM happened to file it under.
func TestPriceLookupTolerance(t *testing.T) {
	tbl := newPriceTable([]byte(testPriceJSON))
	cases := []struct {
		name             string
		model, provider  string
		wantIn           float64
		wantCtx          int
		wantMissing      bool
		wantProviderName string
	}{
		{name: "exact", model: "gpt-4o", provider: "openai", wantIn: 0.0000025, wantCtx: 128000, wantProviderName: "openai"},
		{name: "case folded", model: "GPT-4o", provider: "OpenAI", wantIn: 0.0000025, wantProviderName: "openai"},
		// The provider hint decides between two rows that differ only by
		// reseller, and it has to beat the bare key, not lose to it.
		{name: "provider prefix wins", model: "gpt-4o", provider: "azure", wantIn: 0.00000275, wantProviderName: "azure"},
		// Nothing is filed under "openai/gpt-4o", so this falls through to the
		// basename index — where the unprefixed key is the canonical answer.
		{name: "prefixed query", model: "openai/gpt-4o", wantIn: 0.0000025, wantProviderName: "openai"},
		{name: "release date stripped", model: "claude-sonnet-4-5-20250929", provider: "anthropic", wantIn: 0.000003},
		{name: "iso date stripped", model: "claude-sonnet-4-5-2025-09-29", provider: "anthropic", wantIn: 0.000003},
		{name: "vertex revision stripped", model: "claude-sonnet-4-5@001", provider: "anthropic", wantIn: 0.000003},
		{name: "openrouter variant suffix", model: "qwen/qwen3-32b:free", provider: "openrouter", wantIn: 0.0000001},
		{name: "vendor-prefixed only", model: "grok-4", provider: "xai", wantIn: 0.000003, wantCtx: 256000},
		{name: "unknown model", model: "nothing-like-this", provider: "openai", wantMissing: true},
		{name: "empty model", model: "", provider: "openai", wantMissing: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := tbl.lookup(tc.model, tc.provider)
			if ok == tc.wantMissing {
				t.Fatalf("lookup(%q, %q) ok=%v, wantMissing=%v", tc.model, tc.provider, ok, tc.wantMissing)
			}
			if tc.wantMissing {
				return
			}
			if got.InputCostPerToken != tc.wantIn {
				t.Errorf("input cost = %v, want %v (matched %q)", got.InputCostPerToken, tc.wantIn, got.Provider)
			}
			if tc.wantCtx != 0 && got.MaxInputTokens != tc.wantCtx {
				t.Errorf("context = %d, want %d", got.MaxInputTokens, tc.wantCtx)
			}
			if tc.wantProviderName != "" && got.Provider != tc.wantProviderName {
				t.Errorf("matched the %q row, want %q", got.Provider, tc.wantProviderName)
			}
		})
	}
}

// TestPriceTableTolerantOfBrokenSnapshot: seeding is an assist, never
// load-bearing. Every degenerate snapshot has to yield an empty table rather
// than an error, a panic, or a router that refuses to start.
func TestPriceTableTolerantOfBrokenSnapshot(t *testing.T) {
	for _, data := range []string{"", "   \n", "{}", `{"models":{}}`, "not json at all", `{"models":null}`} {
		t.Run(strings.TrimSpace(data), func(t *testing.T) {
			tbl := newPriceTable([]byte(data))
			if _, ok := tbl.lookup("gpt-4o", "openai"); ok {
				t.Error("an unusable snapshot must match nothing")
			}
			reg := BackendRegistration{Model: "gpt-4o", Provider: "openai"}
			if filled := seedPricesWith(tbl, &reg, priceStated{}); len(filled) != 0 {
				t.Errorf("seeded %v from an unusable snapshot", filled)
			}
		})
	}
}

// TestSeedPricesOnlyFillsBlanks: the same rule the rest of P2 runs on. A price
// the operator typed is theirs, including an explicit zero — a free model behind
// a metered endpoint has to be expressible.
func TestSeedPricesOnlyFillsBlanks(t *testing.T) {
	tbl := newPriceTable([]byte(testPriceJSON))

	t.Run("fills everything left blank", func(t *testing.T) {
		reg := BackendRegistration{Model: "gpt-4o", Provider: "openai"}
		filled := seedPricesWith(tbl, &reg, priceStated{})
		if reg.InputPricePerMtok != 2.5 || reg.OutputPricePerMtok != 10 {
			t.Errorf("per-Mtok conversion wrong: %v / %v", reg.InputPricePerMtok, reg.OutputPricePerMtok)
		}
		if reg.ContextK != 125 { // 128000 / 1024
			t.Errorf("context_k = %d, want 125", reg.ContextK)
		}
		if len(filled) != 3 {
			t.Errorf("reported %v, want all three fields", filled)
		}
	})

	t.Run("keeps what the operator entered", func(t *testing.T) {
		reg := BackendRegistration{
			Model: "gpt-4o", Provider: "openai",
			InputPricePerMtok: 1, OutputPricePerMtok: 2, ContextK: 64,
		}
		filled := seedPricesWith(tbl, &reg, priceStated{})
		if reg.InputPricePerMtok != 1 || reg.OutputPricePerMtok != 2 || reg.ContextK != 64 {
			t.Errorf("seeding overwrote declared values: %+v", reg)
		}
		if len(filled) != 0 {
			t.Errorf("reported seeding %v when nothing was blank", filled)
		}
	})

	t.Run("no match seeds nothing", func(t *testing.T) {
		reg := BackendRegistration{Model: "some-local-gguf", Provider: "whoever"}
		if filled := seedPricesWith(tbl, &reg, priceStated{}); len(filled) != 0 || reg.InputPricePerMtok != 0 {
			t.Errorf("an unpublished model must stay blank: %v %+v", filled, reg)
		}
	})

	// A local GPU is not a cloud reseller. The snapshot is keyed by model NAME, so
	// the basename index matches a worker serving gpt-oss-120b to whatever azure_ai
	// charges for weights of the same name — and the price it lands on is money
	// nobody owes, which then drops the worker out of the free-first band and
	// starts consuming the judge's paid allowance.
	//
	// The context window goes with it: a seeded value lands in lastReg, where
	// applyProfileIfGen reads it as an operator declaration it must not overwrite,
	// so a reseller's 128k would permanently outrank the 8k the local llama.cpp
	// actually serves and long prompts would route to a worker that truncates.
	t.Run("a local row is never seeded", func(t *testing.T) {
		reg := BackendRegistration{Model: "gpt-oss-120b", Provider: providerLocal}
		filled := seedPricesWith(tbl, &reg, priceStated{})
		if len(filled) != 0 || reg.InputPricePerMtok != 0 || reg.OutputPricePerMtok != 0 || reg.ContextK != 0 {
			t.Errorf("a local worker was priced from a reseller: filled=%v %+v", filled, reg)
		}
	})

	// The control the old fixture was missing. Its model matched nothing, so the
	// case above passed without the provider argument ever being read — this pins
	// that the id really does resolve, and that the local guard is what stops it.
	t.Run("the same model on a metered row still seeds", func(t *testing.T) {
		reg := BackendRegistration{Model: "gpt-oss-120b", Provider: "azure_ai"}
		filled := seedPricesWith(tbl, &reg, priceStated{})
		if len(filled) != 3 || reg.InputPricePerMtok != 0.15 {
			t.Fatalf("the fixture no longer resolves this model: filled=%v %+v", filled, reg)
		}
	})

	t.Run("a row with no published context keeps its own", func(t *testing.T) {
		reg := BackendRegistration{Model: "no-context", Provider: "whoever"}
		filled := seedPricesWith(tbl, &reg, priceStated{})
		if reg.ContextK != 0 {
			t.Errorf("context_k = %d, want 0 — nothing was published", reg.ContextK)
		}
		if len(filled) != 2 {
			t.Errorf("reported %v, want the two prices only", filled)
		}
	})
}

// TestEmbeddedSnapshotUsable guards the shipped file rather than the code: a
// refresh that lands a malformed or truncated prices.json would otherwise fail
// silently, since every lookup path degrades to "no match".
//
// An EMPTY snapshot is a legitimate build — a checkout with no network gets one,
// and the plumbing is required to work without it — so this skips rather than
// fails when there is nothing in the file.
func TestEmbeddedSnapshotUsable(t *testing.T) {
	tbl := prices()
	if len(tbl.exact) == 0 {
		t.Skip("prices.json is the empty snapshot — run `go run . prices fetch`")
	}
	if tbl.snapshot.Source == "" || tbl.snapshot.FetchedAt == "" {
		t.Error("snapshot has no source or fetch date — provenance is the point of committing it")
	}
	// A handful of ids an operator is likely to type first. Not an exhaustive
	// list: this is checking the file parsed and carries prices, not LiteLLM's
	// catalogue.
	for _, m := range []struct{ model, provider string }{
		{"gpt-4o", "openai"},
		{"claude-sonnet-4-5", "anthropic"},
		{"gemini-2.5-pro", "gemini"},
	} {
		e, ok := tbl.lookup(m.model, m.provider)
		if !ok {
			t.Errorf("%s/%s not found in the embedded snapshot", m.provider, m.model)
			continue
		}
		if e.InputCostPerToken <= 0 || e.OutputCostPerToken <= 0 {
			t.Errorf("%s has no price: %+v", m.model, e)
		}
		if e.Provider != m.provider {
			t.Errorf("%s matched the %q row, want %q", m.model, e.Provider, m.provider)
		}
	}
}
