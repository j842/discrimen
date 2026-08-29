package router

// Filling the outcome matrix — from the question bank, and from real traffic.
//
// The matrix is useless until it has rows, and there are exactly two sources of
// them. Profiling contributes graded answers to bank questions, which is dense
// and exact but drawn from LiveBench's distribution rather than this fleet's.
// The background judge contributes graded answers to REAL prompts, which is
// sparse and noisier but is the only evidence that will ever cover the traffic
// actually being served. Both land in the same table, tagged, so the second can
// be weighted down without being thrown away.
//
// THREE routes in, from those two sources. A profile that finishes in this
// process writes its results directly (profile.go calls the observationsFrom
// pair below); the judge writes one row per sampled request; and at startup
// backfillOutcomesFromProfiles reconstructs the first kind from profiles already
// on disk, which is not a third source but the same evidence recovered from the
// table it was also written to. Rows are filed under the MODEL that answered —
// see identity.go — so none of the three is tied to the worker that carried it.
//
// Also here, because they are the bank's other half rather than the matrix's:
// the question vectors that make a row reachable at all, and the display views
// an operator reads on /backends.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"
)

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
func observationsFrom(hash, backendID string, results []BenchResult, thinking bool, at time.Time) []Observation {
	return observationsWith(hash, backendID, results, at, func(BenchResult) bool { return thinking })
}

// observationsWith is the conversion both callers share, with the one thing they
// disagree about — where the thinking mode comes from — passed in.
//
// The two were separate copies of the same fifteen lines, differing in that
// single expression. That is worth unifying precisely because of what the
// duplication cost once already: the skipped/errored rule and the
// question-resolution rule are the same rule for both passes, and a fix applied
// to one copy is a fix missing from the other for as long as nobody notices.
func observationsWith(hash, backendID string, results []BenchResult, at time.Time,
	thinkingOf func(BenchResult) bool) []Observation {
	byPrompt := bankQIDByPrompt()
	out := make([]Observation, 0, len(results))
	for _, r := range results {
		// Skipped: never asked, so no evidence either way. Errored: a transport
		// failure says nothing about the model. Unsupported: the prompt did not fit
		// THIS deployment's context window, which is a fact about the box the
		// weights are running on and not about the weights — filing it here would
		// let one small-window deployment poison every other deployment of the same
		// model, which would then read a genuine wrong answer and stop asking.
		if r.Skipped || r.Errored || r.Unsupported {
			continue
		}
		qid, ok := byPrompt[strings.TrimSpace(r.Prompt)]
		if !ok {
			continue // the question left the bank, or its grading changed
		}
		out = append(out, Observation{
			QID:       qid,
			ModelHash: hash,
			Backend:   backendID,
			Thinking:  thinkingOf(r),
			Correct:   r.Pass,
			Loose:     r.Loose,
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
// being up. Questions are embedded once and the vectors kept in memory,
// re-derived on restart rather than persisted — the question text is COMPILED
// INTO THE BINARY, so re-deriving costs one batched round trip and storing would
// only duplicate it.
//
// That reasoning does not extend to a judged production question, whose text is
// stored nowhere and cannot be re-derived at all, so those vectors ARE persisted
// (see setJudgedVector). Two policies for two cases, which is also why the
// matrix has to survive an embedder swap rather than assume one cannot happen:
// setVector drops every vector of a retired width the moment one of the new
// width arrives, in memory and on disk, and the first thing to arrive after a
// swap is the bank fill below.
func (r *Router) ensureBankVectors(ctx context.Context) error {
	if r.outcomes == nil {
		return nil
	}
	var missing []benchmarkQ
	for _, q := range benchmarkQuestions {
		if !r.outcomes.hasVector(benchQuestionQID(q)) {
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
			r.outcomes.setVector(benchQuestionQID(q), vecs[i])
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
//
// The truncation is truncateForEmbed's, and shared with it rather than repeated:
// a bank question and a judged production question have to be cut at the same
// point or the two halves of the matrix would sit in subtly different places.
func benchEmbedText(q benchmarkQ) string {
	return truncateForEmbed(q.Prompt)
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
func (r *Router) recordJudgedOutcome(served *Backend, question string, thinking, correct bool, latencyMS int64) {
	if r.outcomes == nil || served == nil || strings.TrimSpace(question) == "" {
		return
	}
	backendID := served.ID
	qid := judgedQuestionQID(question)
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
		// Filed under the MODEL, like every other observation. Keyed by worker id
		// it would be unreachable: routing looks a candidate up by its model hash,
		// so a judged row with none is stored, counted, and never consulted.
		QID: qid, ModelHash: modelHash(served), Backend: backendID,
		Thinking: thinking, Correct: correct,
		LatencyMS: latencyMS, Source: obsSourceJudge, At: time.Now().UTC(),
	}}
	if err := r.outcomes.record(ctx, obs); err != nil {
		log.Printf("outcome matrix: recording a judged answer for %s failed: %v", backendID, err)
		return
	}
	r.outcomes.pruneJudged(ctx, maxJudgedQuestions)
}

func truncateForEmbed(s string) string {
	if len(s) > benchEmbedMaxChars {
		return s[:benchEmbedMaxChars]
	}
	return s
}

const (
	// judgedEmbedTimeout bounds recordJudgedOutcome as a whole — the extra
	// embedding call and the writes after it share the one deadline. Short:
	// this runs on a background sample, and a slow embeddings worker must not
	// pile up goroutines.
	//
	// Sharing it is a known rough edge rather than a design: an embedding that
	// returns just inside the deadline leaves almost none of it for the insert,
	// so a verdict can be graded, embedded, and then dropped by a context that
	// expired between the two. The window is small (the writes are two statements
	// against a local SQLite file) and the failure is logged, so it is recorded
	// here rather than papered over with a second timeout nobody would tune.
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
			r.bankTopics[benchQuestionQID(q)] = benchCategoryOf(q.Tier, q.Prompt)
		}
	})
	return r.bankTopics[qid]
}

// outcomeSummaryFor is the display view shown against one worker, in the mode
// routing would use for a request with no thinking preference.
//
// It is the record of the worker's MODEL, looked up by modelHash(b) like every
// other read. So two workers serving the same weights show the same numbers —
// which is the honest rendering, since the evidence is about the weights — and
// a worker deployed today against a model already profiled elsewhere shows a
// full record immediately rather than an empty one.
func (r *Router) outcomeSummaryFor(b *Backend, thinking bool) *OutcomeSummary {
	if r.outcomes == nil {
		return nil
	}
	s := r.outcomes.summarise(modelHash(b), thinking, r.bankTopicOf)
	return &s
}

// ensureBankVectorsAsync fills the bank's embeddings in the background if they
// are missing, at most one attempt at a time.
//
// WHY THIS EXISTS AT ALL. Bank vectors are deliberately not persisted — the
// question text is compiled in, so a restart re-derives them in one batch (see
// ensureBankVectors). But the only thing deriving them was a cold-start profile,
// and a warm restart loads a cached profile and never profiles. So after any
// restart the matrix held observations with NO vectors,
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
func observationsFromMixed(hash, backendID string, results []BenchResult, at time.Time) []Observation {
	return observationsWith(hash, backendID, results, at,
		func(r BenchResult) bool { return r.Tier >= benchHardTier })
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
// WHAT IT IS FOR NOW, which is not what it was written for. It was written
// because a benchmarkVersion bump emptied the matrix and left the fleet routing
// blind for a full re-profile; content-addressed identity removed that entirely,
// since a bump no longer touches a single observation — qids carry the question
// and its grader, not a fleet-wide integer, so the rows survive the bump and the
// permacache answers most of the re-profile from them. See identity.go.
//
// What is left is every OTHER way the observations table ends up behind the
// profiles beside it: a table recreated by a migration, a profile written by a
// binary that predates the table, a database restored from a backup taken before
// the matrix existed, or a router whose last run never completed a profile. In
// all of them worker_profiles holds the evidence and the matrix does not, and
// this is the read that reunites them.
//
// Idempotent, and safe beside the live writers:
//
//	EVERY profile is read, whatever bench version wrote it. There is no version
//	gate any more and there does not need to be one: a result's validity lives in
//	its qid, which carries the question text, its match mode and the version of
//	the grader — so an answer graded by a superseded grader resolves to a qid
//	nothing looks up, and one graded by an unchanged grader is still good. The old
//	gate discarded both alike, which threw away every mcq result to fix the
//	numeric grader.
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

	var recovered, workers, failed int
	var firstErr error
	for _, s := range profiles {
		// Identity from the STORED profile, never from the live registry: the
		// worker may be long gone, and keeping its results is the point.
		hash := s.prof.ModelHash
		if hash == "" {
			// Written before the fingerprint was recorded, so the identity has to
			// come from somewhere else — and it has to be the SAME identity a live
			// query will compute, or the rows are filed at an address nothing ever
			// asks for.
			//
			// That is not hypothetical: this used to fall back to
			// unfingerprintedModelHash(id, model) unconditionally, and modelHash only
			// takes that path for a worker with no served id, no parameter count and
			// no size. Every real worker has at least a served id, so every legacy
			// profile was recovered into a hash no candidate could match. Measured on
			// the live fleet after a deploy: 497 observations present, every worker
			// reporting total=0. The backfill ran, reported success, and rescued
			// nothing.
			//
			// So: ask the registry what this worker hashes to TODAY. Only when it is
			// gone does the unfingerprinted form apply, and there it is honest — an
			// absent worker's fingerprint is genuinely unknown, and the row will be
			// matched only by another worker that is equally unfingerprintable.
			hash = unfingerprintedModelHash(s.id, s.prof.Model)
			if live := r.liveBackend(s.id); live != nil {
				hash = modelHash(live)
			}
		}
		// Filing evidence nothing can look up is the failure this whole mechanism
		// exists to prevent, so say when it happens rather than counting it as
		// recovered. A registered worker whose hash disagrees with what it was
		// profiled under means the fingerprint moved under it.
		if live := r.liveBackend(s.id); live != nil && modelHash(live) != hash {
			log.Printf("outcome matrix: %s was profiled under model %s but hashes to %s today — "+
				"its %d recovered observations will not be consulted until it re-profiles",
				s.id, hash, modelHash(live), len(s.prof.BenchResults)+len(s.prof.BenchResultsNoThink))
		}
		// The profile's own timestamp, not now(): these observations describe what
		// happened when the profile ran, and dating them now would let a
		// reconstruction outrank a measurement taken since. A profile written
		// before MeasuredAt existed carries the zero time, which reads as the
		// oldest possible evidence — exactly what an undated profile is.
		at := s.prof.MeasuredAt.UTC()
		obs := observationsFromMixed(hash, s.id, s.prof.BenchResults, at)
		obs = append(obs, observationsFrom(hash, s.id, s.prof.BenchResultsNoThink, false, at)...)
		if len(obs) == 0 {
			continue
		}
		n, err := r.outcomes.recordIfNewer(ctx, obs)
		if err != nil {
			// One worker's failure must not cost the rest of the fleet its evidence,
			// and this loop used to return here. What that cost is not theoretical:
			// a missing `loose` column made every persist fail, the first profile row
			// read aborted the whole walk, and because recordIfNewer files into the
			// map BEFORE it persists, the fleet went on routing with exactly one
			// worker's observations in memory and every other worker reading as
			// unmeasured. Carrying on would have left the matrix fully populated in
			// memory and only unpersisted — a degradation instead of an inversion.
			//
			// The error is still returned, after every profile has been tried, so a
			// broken table cannot pass as a clean start.
			failed++
			if firstErr == nil {
				firstErr = fmt.Errorf("backfilling %s: %w", s.id, err)
			}
			continue
		}
		if n > 0 {
			recovered += n
			workers++
		}
	}
	if recovered > 0 || unreadable > 0 || failed > 0 {
		log.Printf("outcome matrix: recovered %d observations from %d stored profiles (%d unreadable, %d failed) — %s",
			recovered, workers, unreadable, failed, r.outcomes)
	}
	return firstErr
}

// liveBackend is the registered worker with this id, or nil — including when
// there is no registry at all, which is a test driving the backfill directly.
func (r *Router) liveBackend(id string) *Backend {
	if r == nil || r.registry == nil {
		return nil
	}
	return r.registry.get(id)
}
