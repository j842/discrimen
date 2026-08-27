package router

// Filling the outcome matrix — from the question bank, and from real traffic.
//
// The matrix is useless until it has rows, and there are exactly two sources.
// Profiling contributes graded answers to bank questions, which is dense and
// exact but drawn from LiveBench's distribution rather than this fleet's. The
// background judge contributes graded answers to REAL prompts, which is sparse
// and noisier but is the only evidence that will ever cover the traffic actually
// being served. Both land in the same table, tagged, so the second can be
// weighted down without being thrown away.

import (
	"context"
	"errors"
	"log"
	"time"
)

// benchQuestionQID is the stable identity of a bank question. Content-derived,
// so a question keeps its identity across a re-fetch and its observations
// survive; a positional id would silently re-point every stored row when a
// question is inserted.
func benchQuestionQID(prompt, expect string) string {
	return uint64Hex(benchQuestionHash(benchmarkQ{Prompt: prompt, Expect: expect}))
}

func uint64Hex(v uint64) string {
	const hexDigits = "0123456789abcdef"
	out := make([]byte, 16)
	for i := 15; i >= 0; i-- {
		out[i] = hexDigits[v&0xf]
		v >>= 4
	}
	return string(out)
}

// observationsFrom converts one profiling pass into matrix rows.
//
// A SKIPPED question contributes nothing — the profile budget never asked it, so
// there is no evidence either way, and recording it as a miss would make a slow
// worker look bad at whatever the budget ran out on. An ERRORED question is the
// same: a transport failure says nothing about the model.
//
// A question that was too SLOW does contribute, as a miss. That is a real
// statement about this worker on this kind of prompt, and it is exactly what the
// routing decision needs to know — the matrix predicts "will be correct AND will
// complete", and a worker that cannot finish has failed the second half.
func observationsFrom(backendID string, results []BenchResult, thinking bool, at time.Time) []Observation {
	out := make([]Observation, 0, len(results))
	for _, r := range results {
		if r.Skipped || r.Errored {
			continue
		}
		out = append(out, Observation{
			QID:       benchQuestionQID(r.Prompt, r.Expect),
			Backend:   backendID,
			Thinking:  thinking,
			Correct:   r.Pass,
			LatencyMS: r.LatencyMS,
			Source:    obsSourceBench,
			At:        at,
		})
	}
	return out
}

// ensureBankVectors embeds any bank question the matrix has no vector for.
//
// Lazy and incremental: embedding is a network call to the embeddings worker,
// and doing it at startup would make the router's boot depend on that worker
// being up. Questions are embedded once and the vectors kept in memory —
// re-derived on restart rather than persisted, because they are a pure function
// of the question text and the embedding model, and persisting them would mean
// detecting when the embedding model changed.
func (r *Router) ensureBankVectors(ctx context.Context) error {
	if r.outcomes == nil {
		return nil
	}
	var missing []benchmarkQ
	for _, q := range benchmarkQuestions {
		if !r.outcomes.hasVector(benchQuestionQID(q.Prompt, q.Expect)) {
			missing = append(missing, q)
		}
	}
	if len(missing) == 0 {
		return nil
	}
	// Batched: one call per chunk rather than per question, since the embeddings
	// worker charges per request far more than per token at this size.
	const chunk = 64
	for start := 0; start < len(missing); start += chunk {
		end := start + chunk
		if end > len(missing) {
			end = len(missing)
		}
		batch := missing[start:end]
		texts := make([]string, len(batch))
		for i, q := range batch {
			texts[i] = benchEmbedText(q)
		}
		vecs, err := r.embedTexts(ctx, texts)
		if err != nil {
			return err
		}
		if len(vecs) != len(batch) {
			return errEmbedCountMismatch
		}
		for i, q := range batch {
			r.outcomes.setVector(benchQuestionQID(q.Prompt, q.Expect), vecs[i])
		}
	}
	log.Printf("outcome matrix: embedded %d bank questions (%s)", len(missing), r.outcomes)
	return nil
}

// benchEmbedText is what gets embedded for a bank question.
//
// The PROMPT only, never the expected answer. An incoming request has no answer,
// so including one here would embed bank questions into a different region than
// the prompts they are meant to be compared against — the neighbours would be
// chosen by a signal the query cannot have.
//
// Truncated to the classifier's own cap so a long question and a long prompt are
// treated the same way, and because the embedding model has a 512-token window
// regardless: past that the tail is invisible on both sides.
func benchEmbedText(q benchmarkQ) string {
	p := q.Prompt
	if len(p) > benchEmbedMaxChars {
		p = p[:benchEmbedMaxChars]
	}
	return p
}

const benchEmbedMaxChars = 2000

var errEmbedCountMismatch = errors.New("embeddings worker returned a different number of vectors than requested")
