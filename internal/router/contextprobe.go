package router

// contextprobe — measuring the context a worker can actually USE.
//
// ContextK is a CLAIM. It comes from vLLM's max_model_len or llama.cpp's n_ctx
// (see queryContext), and it is the window the server was configured with, not
// the window the model stays coherent in. A model advertising 256K that starts
// losing facts at 60K passes the router's hard context filter for a 100K prompt
// and then answers it badly — which surfaces as "the agent got confused on a
// long tool loop", not as an error anyone can grep for.
//
// ONE PROBE, BOTH ANSWERS. Each measurement is a haystack of length L with a
// needle planted at depth d, and asking for the needle back yields two things
// from the same request:
//
//	correctness  did it retrieve the fact — at this LENGTH and this POSITION
//	timing       TTFT at length L gives the prefill rate, which is nearly all
//	             of the latency on a long prompt
//
// So the accuracy map and the prefill curve come out of one grid rather than
// two test regimes. It is synthetic, which is what makes that possible: it
// generates at any length, cannot be contaminated by a training set, and grades
// by exact match with no model in the loop.
//
// A DOUBLING LADDER, STOPPING AT THE FIRST FAILURE. Cost is then proportional
// to the context a worker actually has rather than to the largest one we might
// test. That difference is not marginal: 128K tokens of prefill is ~27s on
// llm-6000pro's measured 4,665 tok/s and over twenty minutes on a CPU worker.
//
// DELIBERATELY NOT PART OF THE QUALITY SCORE. This is a capability probe like
// the speed probe, so it is not keyed by benchmarkVersion and tuning the ladder
// does not re-profile the fleet. It also means it can be added to a running
// deployment without invalidating anything.

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

// contextProbeLevel is one rung of the ladder: everything measured at one
// prompt length.
type ContextProbeLevel struct {
	Tokens     int     `json:"tokens"`      // approximate prompt size probed
	Passed     int     `json:"passed"`      // depths that returned the needle
	Total      int     `json:"total"`       // depths attempted
	TTFTMillis int64   `json:"ttft_ms"`     // median first-token latency at this length
	PrefillTPS float64 `json:"prefill_tps"` // Tokens / TTFT — the long-context timing curve
	Errored    bool    `json:"errored,omitempty"`
}

// ContextProbe is what one worker's ladder measured.
type ContextProbe struct {
	// UsableTokens is the largest length at which the worker returned the needle
	// from EVERY probed depth. Zero means it failed the first rung, which is a
	// real finding and not an error: it means the advertised window is not usable
	// even at the smallest size tested.
	UsableTokens int `json:"usable_tokens"`
	// AdvertisedTokens is what the server claimed, for comparison. The gap
	// between the two is the whole point of this probe.
	AdvertisedTokens int                 `json:"advertised_tokens"`
	Thinking         bool                `json:"thinking"`
	Levels           []ContextProbeLevel `json:"levels,omitempty"`
	MeasuredAt       string              `json:"measured_at,omitempty"`
	// Truncated records that the ladder stopped on its time budget rather than on
	// a failure, so UsableTokens is a LOWER BOUND. Without this a slow worker
	// would be indistinguishable from one that genuinely breaks at that length.
	Truncated bool `json:"truncated,omitempty"`
}

const (
	// contextProbeStart is the first rung. Below this the answer is not in doubt
	// for any worker the router would register, so starting lower buys nothing.
	contextProbeStart = 4096
	// contextProbeCeiling bounds the ladder regardless of what a worker claims.
	// Some servers advertise a million tokens; proving it costs minutes of
	// prefill per probe and tells us nothing about traffic that never gets there.
	contextProbeCeiling = 256 * 1024
	// contextProbeBudget bounds the whole ladder for one worker. Reaching it
	// stops the climb and marks the result Truncated — a lower bound, not a
	// failure.
	contextProbeBudget = 8 * time.Minute
	// contextProbeReserve is the room left for the answer inside the worker's
	// window, so a probe never fails merely because the prompt filled it.
	contextProbeReserve = 1024
	// contextProbeMaxTokens caps the reply. The answer is a short code; anything
	// longer is the model rambling, and a large cap on a long prompt is how a
	// probe turns into a multi-minute generation.
	contextProbeMaxTokens = 64
)

// contextProbeDepths are the needle positions, as a fraction of the haystack.
// Three rather than five: start, middle and end is enough to catch
// lost-in-the-middle (the failure that actually happens — models attend to the
// ends and lose the centre), and each extra depth multiplies the cost of every
// rung by the prefill time at that length.
var contextProbeDepths = []float64{0.1, 0.5, 0.9}

// runContextProbe climbs the ladder for one worker in one thinking mode.
//
// It returns what it measured even on an early failure: a worker that cannot
// retrieve at 4K has told us something worth recording, and returning an error
// there would leave the field empty and indistinguishable from "not probed".
func (r *Router) runContextProbe(b *Backend, thinking bool) ContextProbe {
	advertised := b.ContextK * 1024
	out := ContextProbe{
		AdvertisedTokens: advertised,
		Thinking:         thinking,
		MeasuredAt:       time.Now().UTC().Format(time.RFC3339),
	}
	if advertised < contextProbeStart {
		// Nothing to measure: the window is smaller than the smallest rung, so
		// the advertised figure is already the operative bound.
		return out
	}
	started := time.Now()
	for size := contextProbeStart; size <= contextProbeCeiling; size *= 2 {
		if size+contextProbeReserve > advertised {
			break // the next rung would not fit; the claim is unrefuted up to here
		}
		if time.Since(started) > contextProbeBudget {
			out.Truncated = true
			break
		}
		level := r.probeContextLevel(b, size, thinking)
		out.Levels = append(out.Levels, level)
		if level.Errored || level.Passed < level.Total {
			break // found the edge — climbing further measures nothing
		}
		out.UsableTokens = size
	}
	return out
}

// probeContextLevel runs every depth at one length and summarises them.
func (r *Router) probeContextLevel(b *Backend, size int, thinking bool) ContextProbeLevel {
	level := ContextProbeLevel{Tokens: size, Total: len(contextProbeDepths)}
	var ttfts []int64
	for i, depth := range contextProbeDepths {
		// Seeded per (size, depth) so a re-probe of the same rung asks the same
		// question — a worker that improves or regresses did so for a reason
		// other than being handed a different needle.
		prompt, want := contextNeedlePrompt(size, depth, int64(size)+int64(i))
		got, ttft, err := r.askContextNeedle(b, prompt, thinking)
		if err != nil {
			level.Errored = true
			return level
		}
		if ttft > 0 {
			ttfts = append(ttfts, ttft)
		}
		if contextNeedleFound(got, want) {
			level.Passed++
		}
	}
	if len(ttfts) > 0 {
		level.TTFTMillis = medianInt64(ttfts)
		if level.TTFTMillis > 0 {
			level.PrefillTPS = float64(size) / (float64(level.TTFTMillis) / 1000)
		}
	}
	return level
}

// contextNeedlePrompt builds a haystack of roughly size tokens with a needle at
// the given depth, and returns the prompt and the value to look for.
//
// The filler is varied rather than one repeated sentence: a long run of
// identical text is compressible in a way real context is not, and a model that
// skims it is not being tested on the thing we care about. It also carries no
// digits, so the needle's value cannot be confused with the surroundings.
func contextNeedlePrompt(size int, depth float64, seed int64) (prompt, want string) {
	rng := rand.New(rand.NewSource(seed))
	want = strconv.Itoa(100000 + rng.Intn(899999))
	needle := "The access code for the north archive is " + want + ". Remember it."

	// ~4 characters per token is the same approximation the router uses
	// everywhere else it has no tokenizer (see estimatePromptTokens). Being
	// approximate is fine: the ladder doubles, so a 20% error never confuses one
	// rung with the next.
	targetChars := size * 4
	var sb strings.Builder
	sb.Grow(targetChars + len(needle) + 256)
	insertAt := int(float64(targetChars) * depth)
	placed := false
	for sb.Len() < targetChars {
		if !placed && sb.Len() >= insertAt {
			sb.WriteString(needle)
			sb.WriteByte(' ')
			placed = true
		}
		sb.WriteString(contextFiller[rng.Intn(len(contextFiller))])
		sb.WriteByte(' ')
	}
	if !placed {
		sb.WriteString(needle)
	}
	return "Read the following archive notes.\n\n" + sb.String() +
		"\n\nWhat is the access code for the north archive? Reply with the number only.", want
}

// contextFiller is digit-free prose. Digits anywhere in the haystack would give
// a model that ignored the question something plausible to return, and the
// grader would count it.
var contextFiller = []string{
	"The archive keeps its ledgers in the east wing under a slate roof.",
	"Visitors are asked to sign the register before entering the reading room.",
	"Humidity in the lower vault is controlled by a pair of ageing dehumidifiers.",
	"The cataloguing scheme predates the current staff and nobody has replaced it.",
	"Correspondence is filed by sender rather than by subject, for historical reasons.",
	"A skylight above the stairwell leaks during heavy rain and is patched each spring.",
	"The reading room closes early on days when the heating is being serviced.",
	"Maps are stored flat in wide drawers along the northern wall.",
	"Duplicate prints are kept in the annexe until someone decides what to do with them.",
	"The index cards were typed by hand and many have faded to near illegibility.",
	"Photographic plates require gloves and are brought out only by appointment.",
	"Staff meetings are held in the room behind the conservation bench.",
	"An old bell system connects the vault to the front desk and no longer works.",
	"Donations arrive in unlabelled boxes more often than the policy allows.",
	"The binding workshop shares its ventilation with the tea room, to general complaint.",
}

// askContextNeedle sends one probe and returns the answer text and its TTFT.
func (r *Router) askContextNeedle(b *Backend, prompt string, thinking bool) (string, int64, error) {
	payload := map[string]any{
		"model":                b.Model,
		"stream":               false,
		"max_tokens":           contextProbeMaxTokens,
		"temperature":          0,
		"chat_template_kwargs": map[string]bool{"enable_thinking": thinking},
		"messages":             []map[string]string{{"role": "user", "content": prompt}},
	}
	// Prefill at 128K can take minutes on a slow worker, so this uses the
	// benchmark's deadline rather than a proxy timeout — the probe is measuring
	// exactly the case a short deadline would misreport as a failure.
	ctx, cancel := context.WithTimeout(context.Background(), benchAnswerDeadline)
	defer cancel()
	start := time.Now()
	raw, err := r.benchCompletion(ctx, b, payload)
	if err != nil {
		return "", 0, err
	}
	// Non-streamed, so this is total latency rather than true TTFT. At these
	// lengths the two are close — the reply is capped at a few dozen tokens while
	// the prompt is tens of thousands, so prefill dominates — and streaming a
	// 64-token answer to measure it more precisely would cost a second request.
	elapsed := time.Since(start).Milliseconds()
	body, _ := json.Marshal(raw)
	content, _ := completionContent(raw)
	if content == "" && len(body) > 0 {
		return "", elapsed, fmt.Errorf("empty answer from %s", b.ID)
	}
	return content, elapsed, nil
}

// contextNeedleFound reports whether the answer contains the planted code.
//
// Containment rather than equality: the instruction asks for the number alone,
// but a model that says "The access code is 481920." has retrieved the fact,
// and this probe measures RETRIEVAL, not instruction-following. Grading those
// as failures would conflate the two and understate long-context ability.
func contextNeedleFound(got, want string) bool {
	return strings.Contains(strings.ReplaceAll(got, ",", ""), want)
}

func medianInt64(xs []int64) int64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]int64(nil), xs...)
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
	return s[len(s)/2]
}

// usableContextTokens is the window ROUTING should believe: the measured figure
// when there is one, and the advertised one otherwise.
//
// Never larger than advertised — a probe cannot license exceeding the server's
// configured window — and never zero merely because no probe has run, since
// that would hard-filter an unprobed worker out of everything.
func usableContextTokens(b *Backend) int {
	advertised := b.ContextK * 1024
	if b.ContextProbe == nil || b.ContextProbe.UsableTokens <= 0 {
		return advertised
	}
	if b.ContextProbe.UsableTokens < advertised {
		return b.ContextProbe.UsableTokens
	}
	return advertised
}

// contextProbeMessage renders a ladder for the per-probe check list, which is
// what puts the advertised-versus-measured gap in front of an operator.
func contextProbeMessage(p *ContextProbe) string {
	if p == nil {
		return "not probed"
	}
	adv := p.AdvertisedTokens / 1024
	if p.UsableTokens == 0 {
		if len(p.Levels) == 0 {
			return fmt.Sprintf("%dK advertised, too small to probe", adv)
		}
		return fmt.Sprintf("%dK advertised, but retrieval FAILED at %dK — the smallest size tested",
			adv, p.Levels[0].Tokens/1024)
	}
	use := p.UsableTokens / 1024
	switch {
	case p.Truncated:
		return fmt.Sprintf("at least %dK usable of %dK advertised (ladder hit its time budget)", use, adv)
	case p.UsableTokens*2+contextProbeReserve > p.AdvertisedTokens:
		return fmt.Sprintf("%dK usable, consistent with %dK advertised", use, adv)
	default:
		return fmt.Sprintf("%dK usable of %dK ADVERTISED — routing uses the measured figure", use, adv)
	}
}
