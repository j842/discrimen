package router

// Draft-agreement gating — deciding whether a prompt needs thinking by MEASURING
// rather than predicting.
//
// The reasoning classifier guesses, from an embedding, whether a prompt needs a
// scratchpad. It is the same instrument that was measured to be a topic detector
// on the difficulty axis, and there is no reason to expect it to be better on
// this one. Draft agreement replaces the guess with an experiment: sample two
// cheap no-think answers, and if they land in the same place, that agreement IS
// the evidence that the prompt was easy.
//
// The published signal is stronger than anything text-only. DART reports a
// point-biserial correlation of r = 0.56 between two-draft unanimity and
// always-think correctness, holding across two model families from 0.6B to 32B —
// better than every difficulty predictor in the literature, most of which sit
// near r = 0.15.
//
// WHY IT PAYS FOR ITSELF. When the drafts agree, one of them is the answer, so
// the cost is two short generations instead of one long thinking turn. When they
// disagree, the drafts are wasted and the thinking turn happens anyway. Easy
// prompts dominate real traffic, so the trade is favourable — DART measured 29.5s
// against 67.6s of wall clock on Qwen3-8B — but it is a trade, and on a fleet
// whose traffic is mostly hard it would be a loss.
//
// TWO THINGS THAT MAKE IT WRONG IF IGNORED:
//
//	The drafts must be SAMPLED. At temperature 0 both drafts are the same string
//	by construction, so they always agree and the gate measures nothing while
//	appearing to work perfectly. This is the failure mode most likely to ship
//	unnoticed, because every metric looks excellent.
//
//	It breaks on multiple choice. With a handful of options, two independent
//	drafts agree by chance often enough to destroy the signal — DART excludes MCQ
//	explicitly, and so does this.

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"sync"
	"time"
)

// draftVerdict is the outcome of one gating attempt.
type draftVerdict struct {
	// Agreed reports that the drafts landed in the same place, so Answer can be
	// served as-is and the thinking turn skipped.
	Agreed bool
	Answer string
	// Similarity is how alike the drafts were, for the route header. Exposed
	// because a gate that fires at 0.99 and one that fires at the threshold are
	// worth telling apart when tuning.
	Similarity float64
	Elapsed    time.Duration
	// PromptTokens and CompletionTokens are what the drafts COST, summed. The
	// caller is billed for them whether or not the gate fired: they were spent on
	// their request. Reporting zero would make a gated request look free and
	// silently under-charge every budgeted key.
	PromptTokens     int
	CompletionTokens int
	// Ran distinguishes "the gate declined to apply" from "the gate ran and the
	// drafts disagreed". Only the second is evidence about the prompt.
	Ran bool
}

const (
	// draftCount is how many drafts to sample. Two, following DART: the marginal
	// signal from a third is small and the cost is linear.
	draftCount = 2
	// draftTemperature is what makes the drafts INDEPENDENT. Greedy decoding
	// would return the same string twice and the gate would fire on everything.
	draftTemperature = 0.7
	// draftMaxTokens bounds a draft. A no-think answer to an easy question is
	// short; a draft running long is itself a sign the question is not easy, and
	// the truncation ends the attempt rather than paying for a full generation
	// twice over.
	draftMaxTokens = 512
	// draftDeadline bounds the whole gate. It sits in front of a request that has
	// not started yet, so its cost is pure added latency when it fails.
	draftDeadline = 20 * time.Second
	// draftAgreeSimilarity is the cosine above which two drafts count as the same
	// answer. High, because the question is "did these land in the same place",
	// not "are these on the same topic" — and embeddings say yes to the second far
	// too readily.
	draftAgreeSimilarity = 0.94
)

// draftGate decides whether this prompt can be answered without thinking, by
// trying it.
//
// Returns Ran=false when the gate does not apply, which is the common case: it
// only makes sense on a non-streamed request whose thinking mode the ROUTER
// chose, and never on multiple choice.
func (r *Router) draftGate(ctx context.Context, b *Backend, chatReq *ChatRequest, tr thinkingResolution) draftVerdict {
	if r.cfg == nil || !r.cfg.DraftGating || b == nil || chatReq == nil {
		return draftVerdict{}
	}
	// Only where the ROUTER inferred the mode. An explicit requirements.thinking,
	// a reasoning_effort or a kwargs override is the caller's instruction, not a
	// hypothesis to test — and note that explicit "off" and classifier-decided
	// "off" produce otherwise identical resolutions, which is why this keys on a
	// dedicated field rather than on noThink.
	//
	// It runs whichever way the classifier leaned. Agreement is evidence the
	// prompt is easy and thinking can be skipped; disagreement is evidence it is
	// not, INCLUDING when the classifier had already decided no-think — which is
	// the case where the guess would otherwise have shipped a bad answer with
	// nothing to catch it.
	if !tr.autoDecided {
		return draftVerdict{}
	}
	if chatReq.Stream {
		return draftVerdict{} // a stream cannot be recalled if the drafts disagree
	}
	prompt := lastUserText(chatReq.Messages)
	if strings.TrimSpace(prompt) == "" || looksMultipleChoice(prompt) {
		return draftVerdict{}
	}

	ctx, cancel := context.WithTimeout(ctx, draftDeadline)
	defer cancel()
	started := time.Now()

	drafts := make([]string, draftCount)
	usage := make([]usageCount, draftCount)
	var wg sync.WaitGroup
	for i := 0; i < draftCount; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			drafts[i], usage[i] = r.sampleDraft(ctx, b, chatReq)
		}(i)
	}
	wg.Wait()

	v := draftVerdict{Ran: true, Elapsed: time.Since(started)}
	for _, u := range usage {
		v.PromptTokens += u.prompt
		v.CompletionTokens += u.completion
	}
	for _, d := range drafts {
		if strings.TrimSpace(d) == "" {
			return v // a failed draft is not a disagreement; it is no measurement
		}
	}
	v.Similarity, v.Agreed = r.draftsAgree(ctx, drafts)
	if v.Agreed {
		v.Answer = drafts[0]
	}
	return v
}

// sampleDraft asks the worker for one cheap no-think answer.
func (r *Router) sampleDraft(ctx context.Context, b *Backend, chatReq *ChatRequest) (string, usageCount) {
	payload := map[string]any{
		"model":                probeModel(b),
		"stream":               false,
		"max_tokens":           draftMaxTokens,
		"temperature":          draftTemperature,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
		"messages":             chatReq.Messages,
	}
	raw, err := r.benchCompletion(ctx, b, payload)
	if err != nil {
		return "", usageCount{}
	}
	content, _ := completionContent(raw)
	return content, usageFrom(raw)
}

// draftsAgree reports whether the drafts landed in the same place.
//
// Exact match first, on a normalised form: for a short factual answer that is
// both the cheapest test and the most reliable one. Falling straight to
// embeddings would compare two identical strings through a lossy encoder and
// occasionally decide they differ.
//
// Otherwise, cosine over the drafts' embeddings — the same worker the classifier
// uses. Prose answers to the same easy question are worded differently and mean
// the same thing, which is exactly what an embedding is good at, and it needs no
// per-domain answer extractor.
func (r *Router) draftsAgree(ctx context.Context, drafts []string) (float64, bool) {
	if len(drafts) < 2 {
		return 0, false
	}
	first := normaliseDraft(drafts[0])
	same := true
	for _, d := range drafts[1:] {
		if normaliseDraft(d) != first {
			same = false
			break
		}
	}
	if same {
		return 1, true
	}
	// No embedder configured at all — fail closed, for the same reason a failed
	// embed does below. This path is reachable with a partially constructed
	// Router, and a panic here would take down a request the gate is only
	// supposed to make faster.
	if r == nil || r.cfg == nil {
		return 0, false
	}
	vecs, err := r.embedTexts(ctx, drafts)
	if err != nil || len(vecs) != len(drafts) {
		// No embedder: fall back to "they differ". Erring towards thinking costs
		// latency; erring the other way serves an answer nothing checked.
		return 0, false
	}
	base := normalize(vecs[0])
	worst := 1.0
	for _, v := range vecs[1:] {
		if sim := dot(base, normalize(v)); sim < worst {
			worst = sim
		}
	}
	return worst, worst >= draftAgreeSimilarity
}

// normaliseDraft strips the formatting two samples of the same answer differ in
// — case, whitespace, surrounding markdown and terminal punctuation — so an
// exact match means "the same answer" rather than "the same bytes".
func normaliseDraft(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.Trim(s, "*_`\"' \t\n.")
	return strings.Join(strings.Fields(s), " ")
}

// draftMCQRe spots the option list of a multiple-choice question: two or more
// lettered options on their own lines.
var draftMCQRe = regexp.MustCompile(`(?m)^\s*\(?[A-E][).:]\s+\S`)

// looksMultipleChoice reports whether the gate must decline.
//
// With a small option set two independent drafts agree by chance often enough to
// destroy the signal — a four-option question has a 25% floor on spurious
// agreement before the model knows anything. DART excludes multiple choice for
// this reason and reports the exclusion as necessary rather than cautious.
func looksMultipleChoice(prompt string) bool {
	return len(draftMCQRe.FindAllString(prompt, 3)) >= 2
}

// draftAnswerBody renders an agreed draft as a chat completion, so a gated
// request returns the same shape as any other.
func draftAnswerBody(model, answer string, promptTokens, completionTokens int) []byte {
	body, _ := json.Marshal(map[string]any{
		"object": "chat.completion",
		"model":  model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]string{"role": "assistant", "content": answer},
			"finish_reason": "stop",
		}},
		// What the DRAFTS cost, not what the served one did. Both were generated
		// for this caller, so both are theirs to pay for — and per-key budgeting
		// reads this block, so omitting it would make a gated request free.
		"usage": map[string]int{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		},
	})
	return body
}
