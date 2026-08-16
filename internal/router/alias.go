package router

import (
	"regexp"
	"strings"
)

// modelAlias reduces a nice model name (niceModelName's output) to the short
// spelling a human would type — "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL" → "gemma4",
// "granite-4.1-8b-Q4_K_M" → "granite4.1", "llm-6000pro-deepseek-284B-q8" →
// "deepseek". The full ids stay accepted; the alias is an ADDITIONAL spelling,
// published as the /v1/models menu id when it is unambiguous, so a harness
// config reads gemma4 instead of a quant-encrusted file path. Family words and
// version numbers survive; everything that encodes packaging rather than
// identity (size, quant, precision, shard/date counters, deployment prefixes)
// is dropped. Empty means "no usable alias" — the caller falls back to the
// real id.
func modelAlias(name string) string {
	tokens := strings.FieldsFunc(strings.ToLower(name), func(r rune) bool {
		return r == '-' || r == '_' || r == '/' || r == ' '
	})
	kept := []string{}
	for _, tok := range tokens {
		switch {
		// Packaging vocabulary first: q4/fp8/int4 would otherwise read as a
		// family word with a version fused on.
		case aliasNoiseWords[tok], aliasSizePat.MatchString(tok), aliasQuantPat.MatchString(tok):
		case aliasVersionPat.MatchString(tok):
			kept = append(kept, tok) // bare version: the "4" in gemma-4, "v4", "4.1"
		case aliasWordVersionPat.MatchString(tok):
			kept = append(kept, tok) // fused family+version: qwen3.6, gemma3, llama4
		case strings.ContainsAny(tok, "0123456789"):
			// Any other digit-bearing token is packaging: 284b handled above, but
			// also 6000pro, 0731 date stamps, 00001 shard counters, exl2…
		default:
			kept = append(kept, tok)
		}
	}
	return strings.Join(kept, "")
}

var (
	// A version standing alone between separators. Capped at two integer digits
	// and starting non-zero, so date/build stamps (0731), shard counters
	// (00001) and the 0 in Q8_0 don't read as versions.
	aliasVersionPat = regexp.MustCompile(`^v?[1-9]\d?(\.\d+)*$`)
	// A family word with the version fused on: qwen3.6, gemma3. Digits anywhere
	// but the tail (6000pro, a4b, q4) stay droppable.
	aliasWordVersionPat = regexp.MustCompile(`^[a-z]+\d{1,2}(\.\d+)*$`)
	// Parameter counts: 27b, 8b, 1.5b, 284b, and MoE active counts like a4b.
	aliasSizePat = regexp.MustCompile(`^a?\d+(\.\d+)?[bm]$`)
	// Quant/precision vocabulary, including the fragments Q4_K_XL splits into.
	aliasQuantPat = regexp.MustCompile(`^(i?q\d[\w]*|fp\d+|bf16|f16|f32|int\d|awq|gptq|autoround|mlx|k|s|m|l|xl|xxl|xs)$`)
	// Words that describe packaging or the deployment, not the model.
	aliasNoiseWords = map[string]bool{
		"it": true, "instruct": true, "chat": true, "base": true, "qat": true,
		"ud": true, "gguf": true, "llm": true, "hf": true, "ggml": true,
		"of": true, // the connective in "-00001-of-00005" shard counters
	}
)

// backendAlias is the alias spelling for what a backend serves, or "" when the
// name reduces to nothing. Derived from the display name so an --alias-masked
// llama.cpp worker still aliases from its real weights path.
func backendAlias(b *Backend) string {
	a := modelAlias(niceModelName(b))
	if autoModelNames[a] || isExpertModel(a) {
		return "" // an alias must never shadow a name the router owns
	}
	return a
}
