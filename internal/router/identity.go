package router

// Content-addressed identity for the two things a graded result is about: WHICH
// MODEL answered, and WHICH QUESTION, graded WHICH WAY.
//
// Both are hashes of the thing itself rather than of where it happened to be
// running or where it happened to sit in a list. That buys one property, and it
// is the whole reason this file exists:
//
//	NOTHING IS EVER INVALIDATED.
//
// Fix a grader and its questions get new qids. The old rows are not stale and
// not wrong — they are simply never looked up again, because nothing asks for
// that qid any more. Decommission a worker and its results survive, because they
// were filed under the model, not the machine. Redeploy the same weights on
// another host and the profile is already there.
//
// The alternative, and what this replaces for question results, is a single
// global benchmarkVersion. That is a cliff: bumping it to fix ONE grader discards
// every graded answer for every worker in both thinking modes, including the
// questions whose grading did not change by a byte. Bumping 42 -> 43 to fix the
// numeric grader would have thrown away ~5,600 gradings to re-derive an identical
// verdict for every mcq, exact-list and contains question in the bank — and one
// 397B worker alone had 194 minutes of measurement in that pile.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strconv"
	"strings"
)

// The serving runtimes the router can tell apart, measured from whether /props
// answers. Two values is all the fidelity the model hash needs: what matters is
// that llama.cpp and an OpenAI-shaped server are not assumed to grade alike.
const (
	engineLlamaCPP = "llama.cpp"
	engineOpenAI   = "openai"
)

// graderVersions is the version of the CODE that grades each match mode. Bump one
// when its grading behaviour changes; every question in that mode gets a new qid
// and is re-asked, and no other mode is touched.
//
// Bumped BY HAND, deliberately. Deriving these from the source (hashing the
// grader functions, say) churns on a comment edit and silently re-profiles the
// fleet for nothing — the same over-firing this file exists to stop.
//
// History:
//
//	numeric 2       — benchLeadBreakRe no longer treats "and" as a clause break;
//	                  a correct answer whose sentence contained "and" was graded
//	                  wrong, including the tier-1 control "8 and 5 is 13".
//	mcq/mcq-repeat 2 — the bare last-standalone-letter fallback passed a WRONG
//	                  pick ("must be option D, since vitamin C is irrelevant"
//	                  graded correct against C).
//	contains 2      — containsFinalAnswer read only the final clause, so a
//	final-contains 2   correct answer followed by an explanation failed.
//
// Keyed by the match mode as the BANK spells it (benchmarkQ.Match), which is a
// bare string there; benchMatchCodeExec is the only one with a named constant.
var graderVersions = map[string]int{
	"numeric":          2,
	"mcq":              2,
	"mcq-repeat":       2,
	"exact-list":       1,
	"contains":         2,
	"final-contains":   2,
	benchMatchCodeExec: 1,
}

// graderVersionFor is the grading-code version for a match mode. An unknown mode
// yields 0, which is a distinct identity rather than an error: a question graded
// by a mode this build does not recognise cannot match a cached result from one
// that does.
func graderVersionFor(match string) int { return graderVersions[match] }

// benchQuestionQID is the identity of a bank question AND of the way it is
// graded. Two questions with the same text but different match modes are
// different questions, because the answer that satisfies one may not satisfy the
// other; the same question after a grader fix is likewise a different question,
// because the verdict is no longer the same function of the answer.
func benchQuestionQID(q benchmarkQ) string {
	h := sha256.New()
	writeField(h, q.Prompt)
	writeField(h, q.Expect)
	writeField(h, q.Match)
	writeField(h, strconv.Itoa(graderVersionFor(q.Match)))
	// Code questions are graded by running their test cases, so the cases are
	// part of the question the same way Expect is for a string-matched one.
	if q.Code != nil {
		writeField(h, q.Code.Class)
		writeField(h, q.Code.Func)
		writeField(h, q.Code.Prefix)
		// Field order in a Go struct is fixed by the type, so this rendering is
		// stable for a given build — and a build that changes the shape of a test
		// case has changed how the question is graded anyway.
		cases, _ := json.Marshal(q.Code.Tests)
		writeField(h, string(cases))
	}
	return "q" + hex.EncodeToString(h.Sum(nil))[:24]
}

// modelHash identifies the MODEL a result is about, so results outlive the
// worker that produced them and are shared by every deployment of the same
// weights. `llm-cpu-gemma-26B` runs on two hosts here; without this they profile
// the same model twice and neither inherits the other's answers.
//
// Engine is in the hash and hardware is not, which is the line this draws:
// CORRECTNESS is a property of the weights, the engine and the thinking mode;
// LATENCY is all of that plus the box and its current load. Only correctness
// travels between hosts (see cachedVerdict, which drops a latency it did not
// measure here). Including the engine costs almost nothing — running one model on
// both llama.cpp and vLLM is rare — and removes the sharing most likely to be
// wrong, since the two disagree on samplers and kernels and the repo has already
// measured them differing on near-ties in the logits.
//
// A worker whose model the router could not fingerprint at all — no served id, no
// parameter count, no size, which is a bare provider row — falls back to its own
// id and declared model. That is deliberately the NON-sharing direction: two
// unidentifiable endpoints must not collide into one identity and inherit each
// other's answers, so they each get their own.
func modelHash(b *Backend) string {
	if b == nil {
		return ""
	}
	if b.ServedID == "" && b.ModelParams == 0 && b.ModelSizeBytes == 0 {
		return unfingerprintedModelHash(b.ID, b.Model)
	}
	h := sha256.New()
	writeField(h, b.ServedID)
	writeField(h, strconv.FormatInt(b.ModelParams, 10))
	writeField(h, b.ModelQuant)
	writeField(h, strconv.FormatInt(b.ModelSizeBytes, 10))
	writeField(h, strconv.Itoa(b.ModelCtxTrain))
	writeField(h, b.Engine)
	// NOT ModelPath. The same weights sit at different paths on different hosts,
	// and putting the path in would defeat the cross-host sharing that is the
	// point. Params + quant + size + trained context is a strong enough
	// fingerprint on its own.
	return "m" + hex.EncodeToString(h.Sum(nil))[:24]
}

// unfingerprintedModelHash is the identity of a model the router could not
// fingerprint: a bare provider row that reports no served id, parameter count or
// size, and a profile stored before the fingerprint was recorded.
//
// Deliberately per-WORKER, which is the non-sharing direction. Two models the
// router cannot tell apart must not collide into one identity and inherit each
// other's answers; giving each its own hash costs a re-profile and cannot be
// wrong, while merging them silently attributes one model's results to another.
func unfingerprintedModelHash(id, model string) string {
	h := sha256.New()
	writeField(h, "unfingerprinted")
	writeField(h, id)
	writeField(h, model)
	return "m" + hex.EncodeToString(h.Sum(nil))[:24]
}

// writeField appends a length-delimited field, so ("ab","c") and ("a","bc")
// cannot hash alike. A plain concatenation would let a prompt ending in a digit
// collide with the version that follows it.
func writeField(h interface{ Write([]byte) (int, error) }, s string) {
	_, _ = h.Write([]byte(strconv.Itoa(len(s))))
	_, _ = h.Write([]byte{':'})
	_, _ = h.Write([]byte(s))
}

// judgedQuestionQID is the identity of a PRODUCTION question, so repeated asks of
// the same thing accumulate evidence instead of scattering it.
//
// Same hash as a bank question and a different prefix, because the two are not
// interchangeable: a bank question is graded against a known answer, a production
// one by another model. There is no grader version here — the judge is not a
// match mode, and its verdicts are already weighted below bench evidence.
func judgedQuestionQID(question string) string {
	h := sha256.New()
	writeField(h, truncateForEmbed(question))
	return "j" + hex.EncodeToString(h.Sum(nil))[:24]
}

// bankQIDByPrompt maps a stored result back to the question it answered.
//
// A BenchResult records the prompt but not the match mode, so its qid cannot be
// recomputed from the row alone — it is resolved against the LIVE bank, which is
// also what makes the resolution correct: the qid it yields carries today's
// grader version, so a result graded by an older grader resolves to today's qid
// only if the grading has not changed. A prompt no longer in the bank yields
// nothing, which is right — that question was removed and its answers are
// unreachable by construction.
func bankQIDByPrompt() map[string]string {
	out := make(map[string]string, len(benchmarkQuestions))
	for _, q := range benchmarkQuestions {
		out[strings.TrimSpace(q.Prompt)] = benchQuestionQID(q)
	}
	return out
}
