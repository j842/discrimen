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
	"encoding/json"
	"errors"
	"fmt"
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
		// Persisted, not just held: the question text is stored nowhere, so a
		// vector lost at restart cannot be re-derived and the observation below
		// becomes permanently unqueryable. See setJudgedVector.
		if err := r.outcomes.setJudgedVector(ctx, qid, vecs[0]); err != nil {
			// The vector is in memory either way, so this run still benefits. Worth
			// a line because the cost is silent and deferred: it is the NEXT restart
			// that loses the row.
			log.Printf("outcome matrix: persisting the vector for a judged answer failed, it will not survive a restart: %v", err)
		}
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

// observationsFromMixed converts the MIXED profiling pass, where the thinking
// mode is a property of each question rather than of the pass.
//
// benchOne asks easy tiers thinking-off and hard tiers thinking-on (it gates on
// benchHardTier), so recording the whole pass as Thinking=true files every
// easy-tier answer as evidence about a mode it was never asked in. The mode is
// therefore taken from the result, not from the caller — and it is read off the
// BenchResult rather than by indexing back into benchmarkQuestions, because
// observationsFrom drops skipped and errored entries and the two slices are not
// aligned.
func observationsFromMixed(backendID string, results []BenchResult, at time.Time) []Observation {
	out := make([]Observation, 0, len(results))
	for _, r := range results {
		if r.Skipped || r.Errored {
			continue
		}
		out = append(out, Observation{
			QID:       benchQuestionQID(r.Prompt, r.Expect),
			Backend:   backendID,
			Thinking:  r.Tier >= benchHardTier,
			Correct:   r.Pass,
			LatencyMS: r.LatencyMS,
			Source:    obsSourceBench,
			At:        at,
		})
	}
	return out
}

// backfillOutcomesFromProfiles rebuilds the matrix from profiles already on disk.
//
// WHY THIS EXISTS. The only writer of bench observations was profileBackend, and
// it runs only when a profile COMPLETES IN THIS PROCESS; load() reads the
// observations table and nothing else. But worker_profiles already holds the
// same evidence in the same shape — BenchResults and BenchResultsNoThink, which
// are exactly what observationsFromMixed and observationsFrom consume — for
// every worker ever profiled, under any binary. So an empty observations table
// meant no routing evidence at all, recoverable only by re-profiling the whole
// fleet: hours of GPU time, every worker at once, to reconstruct rows sitting in
// the next table over. Measured on the live fleet: /admin/outcomes reported "0
// questions, 392 vectors, 0 observations" and every routed request came back
// route:outcome:unknown,fallback-speed, while one worker held a complete
// 392-result profile contributing nothing.
//
// It is what makes a benchmarkVersion bump survivable rather than a fleet-wide
// outage of the routing evidence base, and what makes a profile written by an
// older binary visible at all.
//
// Idempotent, and safe beside the live writers:
//
//	A profile whose BenchVersion is not the current one is SKIPPED. Its answers
//	were graded against a different question set or a different grader, which is
//	the one thing that genuinely invalidates an observation — and record()'s key
//	does not include the version, so filing them would corrupt the current set
//	rather than sit beside it.
//
//	Rows are filed at the profile's MeasuredAt, and recordIfNewer refuses to
//	replace an observation already at least as fresh. So a backfill that races a
//	completing profile, or one that runs after the judge has been writing for a
//	week, cannot walk newer evidence backwards. Ties go to the incumbent, which
//	makes a repeated run a no-op.
//
// Intended to be called once at startup, after outcomeMatrix.load.
func (r *Router) backfillOutcomesFromProfiles(ctx context.Context) error {
	if r.outcomes == nil || r.logs == nil || r.logs.db == nil {
		return nil
	}
	// Read EVERYTHING first, then write. The log store runs on a single
	// connection (SetMaxOpenConns(1)), so an insert issued while this cursor is
	// still open would wait for a connection the cursor is holding.
	type stored struct {
		id   string
		prof *WorkerProfile
	}
	var profiles []stored
	rows, err := r.logs.db.QueryContext(ctx, `SELECT id, profile_json FROM worker_profiles`)
	if err != nil {
		return err
	}
	unreadable := 0
	for rows.Next() {
		var id, raw string
		if err := rows.Scan(&id, &raw); err != nil {
			rows.Close()
			return err
		}
		var p WorkerProfile
		if json.Unmarshal([]byte(raw), &p) != nil {
			// A profile this binary cannot parse is skipped rather than fatal: the
			// rest of the fleet's evidence is worth more than a clean failure.
			unreadable++
			continue
		}
		profiles = append(profiles, stored{id: id, prof: &p})
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return err
	}
	rows.Close()

	var recovered, stale, workers int
	for _, s := range profiles {
		if s.prof.BenchVersion != benchmarkVersion {
			stale++
			continue
		}
		// The profile's own timestamp, not now(): these observations describe what
		// happened when the profile ran, and dating them now would let a
		// reconstruction outrank a measurement taken since. A profile written
		// before MeasuredAt existed carries the zero time, which reads as the
		// oldest possible evidence — exactly what an undated profile is.
		at := s.prof.MeasuredAt.UTC()
		obs := observationsFromMixed(s.id, s.prof.BenchResults, at)
		obs = append(obs, observationsFrom(s.id, s.prof.BenchResultsNoThink, false, at)...)
		if len(obs) == 0 {
			continue
		}
		n, err := r.outcomes.recordIfNewer(ctx, obs)
		if err != nil {
			return fmt.Errorf("backfilling %s: %w", s.id, err)
		}
		if n > 0 {
			recovered += n
			workers++
		}
	}
	if recovered > 0 || stale > 0 || unreadable > 0 {
		log.Printf("outcome matrix: recovered %d observations from %d stored profiles (%d at an older bench version, %d unreadable) — %s",
			recovered, workers, stale, unreadable, r.outcomes)
	}
	return nil
}
