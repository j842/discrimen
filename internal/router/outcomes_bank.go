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
	"strings"
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

// recordJudgedOutcome files a graded PRODUCTION answer in the matrix.
//
// This is what makes the matrix converge on the traffic actually being served.
// A fixed question bank is drawn from someone else's distribution — LiveBench
// maths and code, where this fleet mostly sees agent tool loops and chat — so
// without this the matrix would predict confidently about questions nobody asks
// and fall back on everything real.
//
// The question is embedded and kept as a neighbour in its own right, so the NEXT
// similar prompt has evidence to draw on. That is the mechanism by which
// coverage improves on its own, and it is also why it needs a bound: production
// questions arrive forever.
func (r *Router) recordJudgedOutcome(backendID, question string, thinking, correct bool, latencyMS int64) {
	if r.outcomes == nil || strings.TrimSpace(question) == "" {
		return
	}
	qid := judgedQID(question)
	ctx, cancel := context.WithTimeout(context.Background(), judgedEmbedTimeout)
	defer cancel()
	if !r.outcomes.hasVector(qid) {
		vecs, err := r.embedTexts(ctx, []string{truncateForEmbed(question)})
		if err != nil || len(vecs) == 0 {
			// No vector means the row is unreachable by any query, so there is no
			// point storing it. Silent: the embeddings worker being briefly down
			// must not fill the log with one line per sampled request.
			return
		}
		r.outcomes.setVector(qid, vecs[0])
	}
	obs := []Observation{{
		QID: qid, Backend: backendID, Thinking: thinking, Correct: correct,
		LatencyMS: latencyMS, Source: obsSourceJudge, At: time.Now().UTC(),
	}}
	if err := r.outcomes.record(ctx, obs); err != nil {
		log.Printf("outcome matrix: recording a judged answer for %s failed: %v", backendID, err)
		return
	}
	r.outcomes.pruneJudged(ctx, maxJudgedQuestions)
}

// judgedQID identifies a production question by content, so repeated asks of the
// same thing accumulate evidence on one row instead of spreading across many.
func judgedQID(question string) string {
	return "j" + uint64Hex(benchQuestionHash(benchmarkQ{Prompt: truncateForEmbed(question)}))
}

func truncateForEmbed(s string) string {
	if len(s) > benchEmbedMaxChars {
		return s[:benchEmbedMaxChars]
	}
	return s
}

const (
	// judgedEmbedTimeout bounds the extra embedding call. Short: this runs on a
	// background sample, and a slow embeddings worker must not pile up goroutines.
	judgedEmbedTimeout = 10 * time.Second
	// maxJudgedQuestions bounds how many production questions the matrix keeps.
	// Production traffic is unbounded and the matrix is scanned linearly on every
	// routed request, so without a cap both memory and per-request latency grow
	// forever. Oldest go first, which is also the right policy on the merits: a
	// judged result from six months ago describes a fleet that has since changed.
	maxJudgedQuestions = 4000
)

// bankTopicOf labels a bank question for the strengths-and-weaknesses display.
//
// Reuses the existing category labelling rather than inventing a second one, so
// the summary and the older per-category breakdown cannot disagree about what
// counts as coding. Judged production questions have no label — they are not
// part of the bank — and are counted separately.
func (r *Router) bankTopicOf(qid string) string {
	r.bankTopicOnce.Do(func() {
		r.bankTopics = make(map[string]string, len(benchmarkQuestions))
		for _, q := range benchmarkQuestions {
			r.bankTopics[benchQuestionQID(q.Prompt, q.Expect)] = benchCategoryOf(q.Tier, q.Prompt)
		}
	})
	return r.bankTopics[qid]
}

// outcomeSummaryFor is the display view of one worker, in the mode routing would
// use for a request with no thinking preference.
func (r *Router) outcomeSummaryFor(backendID string, thinking bool) *OutcomeSummary {
	if r.outcomes == nil {
		return nil
	}
	s := r.outcomes.summarise(backendID, thinking, r.bankTopicOf)
	return &s
}

// ensureBankVectorsAsync fills the bank's embeddings in the background if they
// are missing, at most one attempt at a time.
//
// WHY THIS EXISTS AT ALL. The vectors are deliberately not persisted — they are
// a pure function of the question text and the embedding model, and storing them
// would mean detecting when that model changed. But the only thing deriving them
// was a cold-start profile, and a warm restart loads a cached profile and never
// profiles. So after any restart the matrix held observations with NO vectors,
// every neighbour lookup found nothing, and the entire prediction path was dead
// while reporting healthy — routing silently fell back to overall hit rate and
// speed for the process lifetime.
//
// Background rather than inline because filling needs the embeddings worker, and
// a request must not wait on it. Retried on failure rather than latched, because
// that worker is exactly the sort of thing that is down when the router starts.
func (r *Router) ensureBankVectorsAsync() {
	if r.outcomes == nil || r.bankVecsReady.Load() {
		return
	}
	if !r.bankVecsFilling.CompareAndSwap(false, true) {
		return // an attempt is already in flight
	}
	go func() {
		defer r.bankVecsFilling.Store(false)
		ctx, cancel := context.WithTimeout(context.Background(), bankVectorFillTimeout)
		defer cancel()
		if err := r.ensureBankVectors(ctx); err != nil {
			log.Printf("outcome matrix: bank embedding failed, predictions fall back until it succeeds: %v", err)
			return
		}
		r.bankVecsReady.Store(true)
	}()
}

// bankVectorFillTimeout bounds one filling attempt. Generous: it is a few
// hundred short texts against a local embedder, and a partial fill is worse than
// a slow one — half a bank means half the neighbours.
const bankVectorFillTimeout = 2 * time.Minute
