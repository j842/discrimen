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
	// from EVERY probed depth.
	//
	// ZERO IS THREE DIFFERENT THINGS, and this field cannot tell them apart on its
	// own — read it with Levels, which can:
	//
	//	no Levels          the advertised window is under the first rung, so there
	//	                   was nothing to test and the claim stands unrefuted.
	//	Levels[0] passed
	//	fewer than Total   a real finding and not an error: the worker answered and
	//	                   could not retrieve at the smallest size tested.
	//	Levels[0].Errored  no finding at all: the request never came back, so
	//	                   nothing about this worker's window has been established.
	//
	// The middle and the last used to render identically, which had the check list
	// declaring a worker unable to hold 4K of context on the strength of a request
	// that returned a 503. See contextProbeMessage.
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
			// The doubling ladder can never reach the top of the window: the next
			// rung is always DOUBLE the last, and a claim is rarely double a rung.
			// For any power-of-two window the climb therefore stops at exactly
			// half, and the whole upper half of the worker's claim goes untested —
			// which is the gap ladderRanOutOfRoom exists to paper over.
			//
			// Test the top itself instead. It is the rung that matters: on
			// llm-6000pro-qwen38-flash-next this is 195,584 tokens, and the prompt
			// that wedged that worker on 2026-08-30 was 195,302. The one length
			// the ladder declined to attempt is where the worker actually broke.
			//
			// Safe to attempt: top + contextProbeMaxTokens is still inside the
			// window, so an endpoint that refuses oversized requests has nothing
			// to refuse, and one that would grind on a too-long prefill is never
			// handed one.
			if top := advertised - contextProbeReserve; top > out.UsableTokens &&
				top >= contextProbeStart && top <= contextProbeCeiling {
				if time.Since(started) > contextProbeBudget {
					out.Truncated = true
					break
				}
				level := r.probeContextLevel(b, top, thinking)
				out.Levels = append(out.Levels, level)
				if !level.Errored && level.Passed >= level.Total {
					out.UsableTokens = top
				}
			}
			break
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
		prompt, want := contextNeedlePrompt(size, depth, int64(size)+int64(i), r.ratios.forModel(b.Model))
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
func contextNeedlePrompt(size int, depth float64, seed int64, charsPerToken float64) (prompt, want string) {
	rng := rand.New(rand.NewSource(seed))
	want = strconv.Itoa(100000 + rng.Intn(899999))
	needle := "The access code for the north archive is " + want + ". Remember it."

	// The haystack is built with the SAME chars-per-token the hard filter sizes
	// prompts by, because the two numbers have to mean the same thing: this rung
	// licenses the filter to admit a prompt it calls `size` tokens, and it can only
	// do that honestly if it tested the amount of text the filter would call
	// `size` tokens. The comment here used to say 4 was "the same approximation the
	// router uses everywhere else" — it was not; the filter divided by 3, so a rung
	// labelled 128K was built from 512K characters the filter would have called
	// 170K, and the ladder was quietly proving a different length from the one it
	// reported. See tokensForChars.
	targetChars := charsForTokens(size, charsPerToken)
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

// ladderRanOutOfRoom reports that the climb ended because the NEXT rung would
// not have fit inside the advertised window — not because anything failed at
// the top. The claim above the last proven rung was never tested, and therefore
// never refuted.
//
// This is the ordinary outcome for a power-of-two window, and it used to halve
// every one of them. The rungs double from 4096, and runContextProbe breaks when
// `size+contextProbeReserve > advertised`; for a 256K claim the last rung that
// fits is 128K, because 256K+1K exceeds 256K. UsableTokens therefore settles at
// exactly half the window on every worker whose claim is a power of two — which
// is every large one — and routing read that as a measured ceiling. A 256K
// worker was hard-filtered out of anything over 128K on the strength of a rung
// the ladder declined to attempt.
//
// The arithmetic distinguishes it from a real edge on its own, with no need to
// record why the loop stopped (which also keeps every profile already cached in
// worker_profiles readable): a rung that FAILED, or one that ERRORED, had to fit
// to have run at all, so `last*2+reserve <= advertised` holds and this is false.
// A ladder abandoned on its time budget is the same — the room check passes
// before the budget check, so the rung it gave up on also fit. Only running out
// of room leaves the last proven rung more than half the claim.
//
// EXCEPT for the top rung, which runContextProbe now probes directly when the
// doubling runs out of room (see the room break there). That rung sits just under
// the claim rather than at or below half, so a genuine ceiling found up there
// still satisfies `last*2+reserve > advertised` and the arithmetic alone would
// wave it through as "never tested" — discarding the one measurement that most
// needed keeping. Direct evidence therefore comes first: a rung recorded above
// the proven mark ran, and did not pass.
//
// An ERRORED rung counts as refuting here, unlike in contextProbeMessage where
// the distinction is about what to tell an operator. Routing is deciding whether
// to send a prompt that long, and a worker whose answer at that length never
// arrived is not one to send it to.
func ladderRanOutOfRoom(p *ContextProbe) bool {
	if p == nil || p.UsableTokens <= 0 {
		return false
	}
	// Without a recorded claim there is no ladder to reason about — every
	// comparison against zero would read as "ran out of room" and hand back the
	// advertised window on the strength of nothing. A probe that predates the
	// field, or one stored malformed, keeps the old conservative reading.
	if p.AdvertisedTokens <= 0 {
		return false
	}
	for _, level := range p.Levels {
		if level.Tokens > p.UsableTokens {
			return false // attempted above the proven mark, and it did not pass
		}
	}
	return p.UsableTokens*2+contextProbeReserve > p.AdvertisedTokens
}

// usableContextTokens is the window ROUTING should believe: the measured figure
// when the ladder actually measured a ceiling, and the advertised one otherwise.
//
// Never larger than advertised — a probe cannot license exceeding the server's
// configured window — and never zero merely because no probe has run, since
// that would hard-filter an unprobed worker out of everything.
//
// A ladder that merely ran out of room refuted nothing, so it does not lower the
// window either. Believing it does not weaken the check this probe exists for: a
// model that loses the needle at 64K of a 256K claim still fails a rung, and that
// verdict is still what routing uses.
func usableContextTokens(b *Backend) int {
	advertised := b.ContextK * 1024
	if b.ContextProbe == nil || b.ContextProbe.UsableTokens <= 0 {
		return advertised
	}
	if ladderRanOutOfRoom(b.ContextProbe) {
		return advertised
	}
	if b.ContextProbe.UsableTokens < advertised {
		return b.ContextProbe.UsableTokens
	}
	return advertised
}

// erroredRung reports the length of the rung the ladder ABANDONED, and whether
// it stopped for that reason at all.
//
// runContextProbe breaks on `level.Errored || level.Passed < level.Total`, which
// merges two findings that are not the same kind of thing. A rung that came back
// WRONG is a measurement: the worker answered, and it could not retrieve the fact
// at that length. A rung that ERRORED is the absence of one: the request never
// produced an answer to grade — a 5xx, a transport failure, a prompt the endpoint
// rejected outright, or benchAnswerDeadline expiring on a worker whose prefill at
// that length is slower than the ladder is willing to wait for.
//
// Only the last rung can carry it, because either verdict ends the climb, so this
// looks at exactly one entry rather than scanning.
//
// Truncated cannot be set at the same time: the budget check breaks BEFORE
// probing, so the rung it gives up on is never appended.
func (p *ContextProbe) erroredRung() (tokens int, errored bool) {
	if p == nil || len(p.Levels) == 0 {
		return 0, false
	}
	last := p.Levels[len(p.Levels)-1]
	if !last.Errored {
		return 0, false
	}
	return last.Tokens, true
}

// contextProbeMessage renders a ladder for the per-probe check list, which is
// what puts the advertised-versus-measured gap in front of an operator.
//
// FOUR OUTCOMES, NOT TWO, and the distinctions are the whole value of the line:
//
//	not probed   no ladder ran at all (nil) — the check is absent from a worker
//	             still on its quick profile, and routing believes the claim.
//	too small    the advertised window is under the first rung, so there was
//	             nothing to test and the claim stands unrefuted.
//	measured     the worker answered at every depth up to some length and then
//	             either missed the needle or ran out of window. This is the only
//	             outcome where the figure is a CEILING.
//	unmeasured   the climb stopped on an error or on the time budget, so the
//	             figure is a LOWER BOUND and the ceiling is unknown.
//
// The last two used to render identically, which is how this line came to state a
// confident wrong diagnosis about a worker it had learned nothing about: a probe
// that 503'd on its first request reported "128K advertised, but retrieval FAILED
// at 4K — the smallest size tested", naming the worker as one that cannot hold
// four thousand tokens on the strength of an answer it never received. An
// operator reading that has been told the opposite of the truth, which is worse
// than being told nothing — the honest rendering of an errored ladder is that the
// window is still unknown.
func contextProbeMessage(p *ContextProbe) string {
	if p == nil {
		return "not probed"
	}
	adv := p.AdvertisedTokens / 1024
	errAt, errored := p.erroredRung()
	if p.UsableTokens == 0 {
		switch {
		case len(p.Levels) == 0:
			return fmt.Sprintf("%dK advertised, too small to probe", adv)
		case errored:
			return fmt.Sprintf("%dK advertised, still UNMEASURED — the probe errored at %dK, the smallest size tested",
				adv, errAt/1024)
		}
		return fmt.Sprintf("%dK advertised, but retrieval FAILED at %dK — the smallest size tested",
			adv, p.Levels[0].Tokens/1024)
	}
	use := p.UsableTokens / 1024
	switch {
	case errored:
		return fmt.Sprintf("at least %dK usable of %dK advertised (the probe errored at %dK, so the ceiling is unmeasured)",
			use, adv, errAt/1024)
	case p.Truncated:
		return fmt.Sprintf("at least %dK usable of %dK advertised (ladder hit its time budget)", use, adv)
	case ladderRanOutOfRoom(p):
		// The same predicate routing uses, deliberately: this message said
		// "consistent with" while usableContextTokens was capping the worker at the
		// number beside it. Reading the two together was the only way to see it.
		return fmt.Sprintf("%dK proven of %dK advertised, ladder ran out of rungs — routing uses the claim", use, adv)
	default:
		return fmt.Sprintf("%dK usable of %dK ADVERTISED — routing uses the measured figure", use, adv)
	}
}

// contextProbeOK is the pass/fail half of the same check — the glyph an operator
// scans before they read anything, and the one scripts/profile-worker.sh renders
// as ✓ or ✗.
//
// It was hard-wired to true, so a worker that failed retrieval at the smallest
// size tested, and one whose probe never got an answer at all, both showed a tick
// beside a message saying so. The tick is the part that gets read.
//
// A ladder that measured a window is a pass even when that window is far under
// the claim: nothing is broken, the gap is stated in the message, and routing is
// already using the smaller figure (see usableContextTokens). What fails is the
// worker that could not retrieve at the first rung — a real capability finding,
// reported exactly like "thinking: not detected" — and the probe that errored,
// where the honest verdict is that the check did not complete.
func contextProbeOK(p *ContextProbe) bool {
	if p == nil {
		return false
	}
	if _, errored := p.erroredRung(); errored {
		return false
	}
	if p.UsableTokens == 0 {
		// Nothing was tested (the window is under the first rung), so there is
		// nothing to have failed. A ladder that DID run and returned no usable
		// length has failed.
		return len(p.Levels) == 0
	}
	return true
}
