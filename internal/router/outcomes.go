package router

// The outcome matrix — what actually happened, per (question, worker, mode).
//
// This replaces the quality SCALAR and everything built on it: tiers, per-tier
// bucket weights, the fleet-relative pass rate that assigned them, and the
// comparison of a benchmark percentage against an embedding margin. None of
// those survived review. The scalar's core problem is that it compresses "what
// is this model good at" into one number and then asks a difficulty classifier
// to produce a comparable number for a prompt — two quantities that were never
// on the same scale, derived from a classifier measured to be a topic detector.
//
// Here, nothing is compressed. Every graded answer stays as a row, and a routing
// decision is a QUERY against those rows: for a prompt like this one, which
// workers got questions like it right, and which of those is fastest.
//
// THREE PROPERTIES THIS BUYS, all of which the scalar design lacked:
//
//	Adding a worker does not disturb anything. Under the old design the tier of
//	every question came from the FLEET pass rate, so a new worker re-tiered the
//	whole bank, which forced a benchmarkVersion bump and re-profiled everyone. A
//	new worker is a new set of rows here, and existing rows do not move.
//
//	Adding questions does not invalidate profiles. New questions are new rows.
//	Only a change to the GRADER invalidates, and since the raw answers are kept
//	that is a re-grade rather than a re-run.
//
//	Questions every worker passes, or every worker fails, are USEFUL. The old
//	selection dropped them for carrying no information about the fleet's spread.
//	Here "every model gets this kind of question right" is precisely what tells
//	the router to use the cheapest worker, and "none of them do" is what tells it
//	not to bother escalating.
//
// WHAT IS BEING PREDICTED, stated honestly. Similarity is computed over prompt
// embeddings, and embeddings capture TOPIC far better than difficulty — the same
// property that made the difficulty classifier a topic detector. So this does not
// predict "will worker w answer THIS question correctly". It predicts "what is
// worker w's hit rate on questions like this one", which is the quantity routing
// actually needs, and it will not separate a hard instance from an easy one
// within the same topic. The confidence figure below is what keeps that
// limitation visible instead of silently wrong.

import (
	"context"
	"database/sql"
	"fmt"
	"sort"
	"sync"
	"time"
)

// Observation is one graded answer: a worker, in one thinking mode, on one
// question. The unit of everything in this file.
type Observation struct {
	QID      string // stable content hash of the question — see benchQuestionHash
	Backend  string
	Thinking bool
	Correct  bool
	// LatencyMS and OutputTokens describe what the answer COST. Kept beside
	// correctness rather than in a separate table because the routing decision
	// needs both together — the fastest worker that will be right — and joining
	// them later would mean matching on (question, worker, mode) anyway.
	LatencyMS    int64
	OutputTokens int
	// Source distinguishes evidence graded against a known answer from evidence
	// graded by a model. Stored rather than merged so the two can be weighted
	// differently: exact-match grading is right or wrong, an LLM judge is
	// probabilistic and its mistakes correlate with the thing being measured.
	Source string
	At     time.Time
}

const (
	// obsSourceBench is a bank question graded against its expected answer.
	obsSourceBench = "bench"
	// obsSourceJudge is a PRODUCTION answer graded by the background judge. Real
	// traffic is the only thing that will ever cover what this fleet is actually
	// asked — a fixed question bank cannot — so these are what make the matrix
	// converge on the distribution being served rather than on LiveBench's.
	obsSourceJudge = "judge"
)

// obsJudgeWeight discounts a judged observation against a bench one. Not zero,
// because judged evidence is the only evidence about real traffic; not one,
// because a judge that shares a blind spot with the worker it is grading will
// agree with it, and that error is not independent.
const obsJudgeWeight = 0.5

// outcomeNeighbours is how many profiled questions a prediction draws on.
//
// Small enough that neighbours are genuinely similar, large enough that one
// unlucky question cannot decide a route. With a few hundred bank questions
// spread over maths, coding and reasoning, 12 lands inside a topic cluster
// without reaching across into a different one.
const outcomeNeighbours = 12

// outcomeMinSimilarity is the cosine below which a "neighbour" is not one.
// Prompts unlike anything profiled — and most real traffic is, since the bank is
// LiveBench maths and code while the traffic is agent loops and chat — must
// report low confidence rather than average over irrelevant questions.
const outcomeMinSimilarity = 0.55

// outcomeMinConfidence is the MEAN neighbour similarity a prediction needs
// before it may gate routing, and it is deliberately above the admission floor.
//
// It used to equal outcomeMinSimilarity, which made the gate vacuous: a
// neighbour is only admitted at all when its similarity clears that floor, and
// the mean of values each >= X is >= X. The check could never be false, so
// known() reduced to "at least two observations" while the file's comments
// rested their honesty on a confidence figure that gated nothing.
//
// Set so a prediction built from barely-admitted neighbours is treated as
// unknown, while one built from genuinely close ones is not.
const outcomeMinConfidence = 0.68

// outcomeMinObservations is the floor on how many nearby answers a prediction
// must rest on before it may gate routing. Below it the answer is "I do not
// know", which the caller handles explicitly.
//
// A COUNT rather than a weight, because the weight is a sum of cosine
// similarities: two neighbours at 0.999 sum to just under 2, so a "at least two
// observations" rule written as a weight threshold fails on exactly the case it
// is meant to admit. It is a floor on having any evidence at all, not a claim
// that two observations are sufficient — Confidence carries that.
const outcomeMinObservations = 2

// prediction is what the matrix says about one worker in one mode.
type prediction struct {
	// Correct is the similarity-weighted hit rate on nearby questions: an
	// estimate of how often this worker answers this KIND of prompt correctly.
	Correct float64
	// Confidence is how much the estimate should be believed — driven by how
	// similar the neighbours were and how much evidence backed them. A confident
	// 0.3 ("reliably bad at this") is a useful routing signal; an unconfident 0.9
	// is not.
	Confidence float64
	// Support is the total similarity weight behind Correct, and Observations is
	// how many answers contributed. Both are reported: the weight is what the
	// estimate is actually built from, the count is what "is there any evidence"
	// is judged on.
	Support      float64
	Observations int
	// MedianLatencyMS is what this worker took on those neighbours. Distinct from
	// the live throughput estimate: that answers "how fast is this worker now",
	// this answers "how long does this KIND of question take it", which is the
	// term the live estimate cannot supply.
	MedianLatencyMS int64
}

// known reports whether a prediction rests on enough nearby evidence to gate a
// routing decision.
func (p prediction) known() bool {
	return p.Observations >= outcomeMinObservations && p.Confidence >= outcomeMinConfidence
}

// outcomeMatrix holds the observations and the question vectors they refer to.
//
// Held in memory and rebuilt from SQLite at startup: it is queried on every
// auto-routed request, and a few hundred questions times a few dozen workers is
// small enough that a scan beats any index. The mutex is a plain RWMutex because
// reads dominate overwhelmingly — writes happen when a profile finishes or the
// judge grades a sample.
type outcomeMatrix struct {
	mu sync.RWMutex
	// vecs maps a question id to its normalised embedding.
	vecs map[string][]float64
	// obs maps a question id to every observation recorded against it.
	obs map[string][]Observation
	db  *sql.DB
}

func newOutcomeMatrix(db *sql.DB) *outcomeMatrix {
	return &outcomeMatrix{
		vecs: map[string][]float64{},
		obs:  map[string][]Observation{},
		db:   db,
	}
}

// setVector records a question's embedding. Idempotent: re-embedding the same
// question overwrites, which is what a change of embedding model requires.
func (m *outcomeMatrix) setVector(qid string, vec []float64) {
	if qid == "" || len(vec) == 0 {
		return
	}
	n := normalize(vec)
	m.mu.Lock()
	m.vecs[qid] = n
	m.mu.Unlock()
}

func (m *outcomeMatrix) hasVector(qid string) bool {
	m.mu.RLock()
	_, ok := m.vecs[qid]
	m.mu.RUnlock()
	return ok
}

// record adds observations and persists them.
//
// Replaces any earlier observation for the same (question, worker, mode) from
// the same source: a re-profile supersedes the previous one rather than
// accumulating alongside it, or a worker's history would drag its current
// estimate. Judged observations do NOT supersede bench ones, and vice versa —
// they are different evidence about different traffic.
func (m *outcomeMatrix) record(ctx context.Context, obs []Observation) error {
	if len(obs) == 0 {
		return nil
	}
	m.mu.Lock()
	for _, o := range obs {
		kept := m.obs[o.QID][:0]
		for _, prev := range m.obs[o.QID] {
			if prev.Backend == o.Backend && prev.Thinking == o.Thinking && prev.Source == o.Source {
				continue // superseded
			}
			kept = append(kept, prev)
		}
		m.obs[o.QID] = append(kept, o)
	}
	m.mu.Unlock()
	return m.persist(ctx, obs)
}

// predict estimates how a worker handles prompts like this one.
//
// vec is the prompt's embedding — the same vector the difficulty classifier
// already computes for every request, so this costs no extra embedding call.
func (m *outcomeMatrix) predict(vec []float64, backend string, thinking bool) prediction {
	if len(vec) == 0 {
		return prediction{}
	}
	q := normalize(vec)
	m.mu.RLock()
	defer m.mu.RUnlock()
	// dot() truncates to the shorter slice, so a vector from a DIFFERENT
	// embedding space still produces a plausible cosine. classify() guards
	// against this and predict did not: bank vectors are embedded once and never
	// re-derived on a model change, so swapping bge-small (384) for bge-base
	// (768) left the matrix in the old space while the classifier re-bootstrapped
	// into the new one. Measured: a cross-dimension pair scored 0.866, cleared
	// the admission floor, and produced a fully confident routing decision from
	// garbage.
	if len(m.vecs) > 0 && !m.dimensionMatches(len(q)) {
		return prediction{}
	}

	type near struct {
		qid string
		sim float64
	}
	var neighbours []near
	for qid, v := range m.vecs {
		if sim := dot(q, v); sim >= outcomeMinSimilarity {
			neighbours = append(neighbours, near{qid, sim})
		}
	}
	if len(neighbours) == 0 {
		return prediction{}
	}
	sort.Slice(neighbours, func(i, j int) bool { return neighbours[i].sim > neighbours[j].sim })
	if len(neighbours) > outcomeNeighbours {
		neighbours = neighbours[:outcomeNeighbours]
	}

	var weighted, total, simSum float64
	used := 0
	var latencies []int64
	for _, n := range neighbours {
		for _, o := range m.obs[n.qid] {
			if o.Backend != backend || o.Thinking != thinking {
				continue
			}
			w := n.sim
			if o.Source == obsSourceJudge {
				w *= obsJudgeWeight
			}
			total += w
			if o.Correct {
				weighted += w
			}
			if o.LatencyMS > 0 {
				latencies = append(latencies, o.LatencyMS)
			}
			simSum += n.sim
			used++
		}
	}
	if used == 0 {
		return prediction{}
	}
	// Confidence is the mean similarity of the neighbours that actually carried
	// evidence for THIS worker — not of the neighbours found. A prompt with a
	// dozen close neighbours, none of which this worker has answered, is not a
	// confident prediction about this worker; it is no prediction at all.
	return prediction{
		Correct:         weighted / total,
		Confidence:      simSum / float64(used),
		Support:         total,
		Observations:    used,
		MedianLatencyMS: medianInt64(latencies),
	}
}

// summary is the display-only headline for one worker: its overall hit rate
// across the whole bank. Explicitly NOT a routing input — routing queries
// neighbours — but a number an operator can eyeball, and the thing `ask -l` and
// the dashboard show in place of the retired quality score.
func (m *outcomeMatrix) summary(backend string, thinking bool) (rate float64, n int) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	correct := 0
	for _, list := range m.obs {
		for _, o := range list {
			if o.Backend != backend || o.Thinking != thinking || o.Source != obsSourceBench {
				continue
			}
			n++
			if o.Correct {
				correct++
			}
		}
	}
	if n == 0 {
		return 0, 0
	}
	return float64(correct) / float64(n), n
}

// forget drops every observation for a worker, for a delete or a re-profile
// under a changed grader.
func (m *outcomeMatrix) forget(ctx context.Context, backend string) error {
	m.mu.Lock()
	for qid, list := range m.obs {
		kept := list[:0]
		for _, o := range list {
			if o.Backend != backend {
				kept = append(kept, o)
			}
		}
		m.obs[qid] = kept
	}
	m.mu.Unlock()
	if m.db == nil {
		return nil
	}
	_, err := m.db.ExecContext(ctx, `DELETE FROM observations WHERE backend_id = ?`, backend)
	return err
}

// ── persistence ────────────────────────────────────────────────────────────

func (m *outcomeMatrix) persist(ctx context.Context, obs []Observation) error {
	if m.db == nil {
		return nil
	}
	tx, err := m.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	stmt, err := tx.PrepareContext(ctx, `INSERT OR REPLACE INTO observations
		(qid, backend_id, thinking, correct, latency_ms, output_tokens, source, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, o := range obs {
		if _, err := stmt.ExecContext(ctx, o.QID, o.Backend, boolInt(o.Thinking), boolInt(o.Correct),
			o.LatencyMS, o.OutputTokens, o.Source, o.At.UTC().Format(time.RFC3339Nano)); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// load rebuilds the in-memory matrix from SQLite at startup.
func (m *outcomeMatrix) load(ctx context.Context) error {
	if m.db == nil {
		return nil
	}
	rows, err := m.db.QueryContext(ctx, `SELECT qid, backend_id, thinking, correct, latency_ms,
		output_tokens, source, created_at FROM observations`)
	if err != nil {
		return err
	}
	defer rows.Close()
	loaded := map[string][]Observation{}
	for rows.Next() {
		var o Observation
		var thinking, correct int
		var at string
		if err := rows.Scan(&o.QID, &o.Backend, &thinking, &correct, &o.LatencyMS,
			&o.OutputTokens, &o.Source, &at); err != nil {
			return err
		}
		o.Thinking, o.Correct = thinking == 1, correct == 1
		o.At, _ = time.Parse(time.RFC3339Nano, at)
		loaded[o.QID] = append(loaded[o.QID], o)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	m.mu.Lock()
	m.obs = loaded
	m.mu.Unlock()
	return nil
}

// String renders the matrix's shape for the debug surface.
func (m *outcomeMatrix) String() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	n := 0
	for _, list := range m.obs {
		n += len(list)
	}
	return fmt.Sprintf("%d questions, %d vectors, %d observations", len(m.obs), len(m.vecs), n)
}

// backendsWithEvidence lists the workers the matrix knows anything about, for
// diagnostics.
func (m *outcomeMatrix) backendsWithEvidence() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	seen := map[string]bool{}
	for _, list := range m.obs {
		for _, o := range list {
			seen[o.Backend] = true
		}
	}
	out := make([]string, 0, len(seen))
	for id := range seen {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// ── choosing a worker ──────────────────────────────────────────────────────

// outcomeChoice is the matrix's verdict on one candidate for one request.
type outcomeChoice struct {
	Backend *Backend
	Pred    prediction
	// Seconds is the predicted wall clock, from expectedLatency refined by what
	// this worker actually took on similar questions. The live estimate knows the
	// worker's current throughput but not how long THIS KIND of prompt makes it
	// generate for; the matrix knows the second and not the first.
	Seconds float64
}

// outcomeCorrectFloor is the predicted hit rate a worker must clear to be
// considered able to answer. Deliberately not high: the estimate is a hit rate
// on SIMILAR questions, not a probability about this one, and demanding 0.9
// would empty the candidate set on any topic the fleet finds hard — leaving
// nothing to route to at exactly the moment routing matters.
const outcomeCorrectFloor = 0.5

// outcomeSpeedMargin is how much predicted accuracy the fallback will trade for
// speed. Applied only when the matrix has no usable prediction, where the
// instruction is "best overall, leaning speed": among workers within this much
// of the best overall hit rate, take the fastest. A margin rather than a pure
// speed sort, so the fallback cannot pick something markedly worse just because
// it is quick.
const outcomeSpeedMargin = 0.15

// chooseByOutcome ranks candidates for a prompt: workers predicted to answer it
// correctly, fastest first.
//
// Returns the ordered candidates and a short reason for the route header, so a
// decision can be read back afterwards — including, importantly, whether the
// matrix actually knew anything or fell back.
func (m *outcomeMatrix) chooseByOutcome(cands []*Backend, vec []float64, thinking bool, job jobCost) ([]*Backend, string) {
	if len(cands) == 0 {
		return nil, "no candidates"
	}
	choices := make([]outcomeChoice, 0, len(cands))
	known := 0
	for _, b := range cands {
		p := m.predict(vec, b.ID, thinking)
		if p.known() {
			known++
		}
		choices = append(choices, outcomeChoice{Backend: b, Pred: p, Seconds: outcomeSeconds(b, p, job)})
	}

	// The ordinary path: keep the workers the matrix expects to be right, and
	// among them take the fastest. This is the whole routing policy — there is no
	// quality target and nothing compared against one.
	if known > 0 {
		// RANK, never evict. Every candidate stays in the returned list; the matrix
		// decides only the ORDER.
		//
		// Filtering was the first shape of this and it was wrong twice over. A
		// worker with no prediction is not a bad worker, it is an unmeasured one,
		// so dropping it removed every newly registered worker from everything.
		// And this list is what failover and escalation move ALONG — shrinking it
		// leaves them nowhere to go at exactly the moment the chosen worker has
		// failed. Measured before the fix: one worker with two judged observations
		// reduced a three-worker fleet to one candidate.
		//
		// Three bands, best first:
		//
		//	predicted correct  by speed — the routing goal, stated directly
		//	unmeasured         by speed, behind anything known to be right
		//	predicted wrong    by predicted correctness, best of a bad lot
		//
		// The last band sorts on ACCURACY rather than speed because a request that
		// reaches it is one the fleet is not good at, and a fast wrong answer helps
		// nobody.
		var able, unmeasured, unable []outcomeChoice
		for _, c := range choices {
			switch {
			case !c.Pred.known():
				unmeasured = append(unmeasured, c)
			case c.Pred.Correct >= outcomeCorrectFloor:
				able = append(able, c)
			default:
				unable = append(unable, c)
			}
		}
		sortBySeconds(able)
		sortBySeconds(unmeasured)
		sort.SliceStable(unable, func(i, j int) bool { return unable[i].Pred.Correct > unable[j].Pred.Correct })
		ordered := make([]outcomeChoice, 0, len(choices))
		ordered = append(ordered, able...)
		ordered = append(ordered, unmeasured...)
		ordered = append(ordered, unable...)
		switch {
		case len(able) > 0:
			return backendsOf(ordered), fmt.Sprintf("outcome:p=%.2f,n=%d", able[0].Pred.Correct, able[0].Pred.Observations)
		case len(unmeasured) > 0:
			return backendsOf(ordered), "outcome:none-above-floor,unmeasured-first"
		default:
			return backendsOf(ordered), fmt.Sprintf("outcome:best-effort,p=%.2f", unable[0].Pred.Correct)
		}
	}

	// FALLBACK: nothing similar has been profiled, which is the common case for
	// traffic unlike the bank. Best overall, leaning speed — take the fastest
	// worker whose overall hit rate is within outcomeSpeedMargin of the best.
	best := 0.0
	rates := make(map[string]float64, len(cands))
	for _, b := range cands {
		rate, n := m.summary(b.ID, thinking)
		if n == 0 {
			// Never profiled in this mode. Treat as average rather than as zero,
			// which would exclude a newly registered worker from everything.
			rate = 0.5
		}
		rates[b.ID] = rate
		if rate > best {
			best = rate
		}
	}
	// Same rule as above: rank, do not narrow. Workers within the margin of the
	// best overall hit rate go first, fastest among them — "best overall, leaning
	// speed" — and everything else follows in accuracy order rather than being
	// dropped, so failover still has somewhere to go.
	var eligible, rest []outcomeChoice
	for _, c := range choices {
		if rates[c.Backend.ID] >= best-outcomeSpeedMargin {
			eligible = append(eligible, c)
		} else {
			rest = append(rest, c)
		}
	}
	sortBySeconds(eligible)
	sort.SliceStable(rest, func(i, j int) bool { return rates[rest[i].Backend.ID] > rates[rest[j].Backend.ID] })
	ordered := append(eligible, rest...)
	if len(ordered) == 0 {
		return nil, "no candidates"
	}
	return backendsOf(ordered), fmt.Sprintf("outcome:unknown,fallback-speed,q=%.2f", rates[ordered[0].Backend.ID])
}

// outcomeSeconds predicts wall clock for one worker, preferring what it actually
// took on similar questions over the generic estimate.
//
// The two know different things. expectedLatency knows the worker's live
// throughput and queue depth but assumes a generic answer length; the matrix
// knows how long this KIND of prompt made this worker generate for, but from
// measurements taken when the fleet was differently loaded. Using the matrix for
// the answer LENGTH and the live figures for the RATE would be better still, and
// is the obvious next refinement; for now the recorded median is used directly
// when the worker's live throughput is close to what it was, and the generic
// estimate otherwise.
func outcomeSeconds(b *Backend, p prediction, job jobCost) float64 {
	generic := expectedLatency(b, job)
	if p.MedianLatencyMS <= 0 {
		return generic
	}
	measured := float64(p.MedianLatencyMS) / 1000
	// Take the longer of the two. Both are estimates and the costs are
	// asymmetric: under-predicting puts a request on a worker that misses the
	// caller's deadline, over-predicting only loses a race it might have won.
	if measured > generic {
		return measured
	}
	return generic
}

func sortBySeconds(cs []outcomeChoice) {
	sort.SliceStable(cs, func(i, j int) bool { return cs[i].Seconds < cs[j].Seconds })
}

func backendsOf(cs []outcomeChoice) []*Backend {
	out := make([]*Backend, 0, len(cs))
	for _, c := range cs {
		out = append(out, c.Backend)
	}
	return out
}

// pruneJudged bounds how many production questions the matrix carries.
//
// Bank questions are never pruned: they are the fixed instrument, and dropping
// one would silently change what every worker's summary is computed over.
// Judged questions are unbounded in principle — production traffic never stops —
// and the matrix is scanned linearly per routed request, so both memory and
// request latency depend on this cap. Oldest first, which is also right on the
// merits: a verdict from months ago describes a fleet that has since changed.
func (m *outcomeMatrix) pruneJudged(ctx context.Context, max int) {
	if max <= 0 {
		return
	}
	type aged struct {
		qid string
		at  time.Time
	}
	m.mu.Lock()
	var judged []aged
	for qid, list := range m.obs {
		newest := time.Time{}
		onlyJudged := true
		for _, o := range list {
			if o.Source != obsSourceJudge {
				onlyJudged = false
				break
			}
			if o.At.After(newest) {
				newest = o.At
			}
		}
		// A question with ANY bench evidence is a bank question and stays.
		if onlyJudged && len(list) > 0 {
			judged = append(judged, aged{qid, newest})
		}
	}
	if len(judged) <= max {
		m.mu.Unlock()
		return
	}
	sort.Slice(judged, func(i, j int) bool { return judged[i].at.Before(judged[j].at) })
	drop := judged[:len(judged)-max]
	ids := make([]string, 0, len(drop))
	for _, d := range drop {
		delete(m.obs, d.qid)
		delete(m.vecs, d.qid)
		ids = append(ids, d.qid)
	}
	m.mu.Unlock()

	if m.db == nil {
		return
	}
	for _, qid := range ids {
		if _, err := m.db.ExecContext(ctx, `DELETE FROM observations WHERE qid = ? AND source = ?`,
			qid, obsSourceJudge); err != nil {
			return // best effort: the in-memory prune is what bounds the hot path
		}
	}
}

// TopicSummary is one cluster of the bank and how a worker did on it — the
// strengths-and-weaknesses map, computed from the same rows routing queries
// rather than from a separate per-category tally that could disagree with them.
type TopicSummary struct {
	Topic   string  `json:"topic"`
	Rate    float64 `json:"rate"`
	Total   int     `json:"total"`
	Correct int     `json:"correct"`
}

// OutcomeSummary is the display-only view of one worker in one mode.
//
// Explicitly NOT a routing input: routing queries neighbours of the actual
// prompt, and compressing that into a headline is exactly the mistake the
// quality scalar made. It exists so an operator has something to read, and so
// the fallback path (which has no neighbours to consult) has an overall ordering
// to work from.
type OutcomeSummary struct {
	Rate     float64        `json:"rate"`
	Total    int            `json:"total"`
	Correct  int            `json:"correct"`
	Judged   int            `json:"judged"`
	MedianMS int64          `json:"median_ms"`
	ByTopic  []TopicSummary `json:"by_topic,omitempty"`
	Thinking bool           `json:"thinking"`
	// Insufficient marks a worker with no bank evidence in this mode — a fresh
	// registration, or one profiled only in the other mode. Distinct from a rate
	// of zero, which means it answered and was wrong.
	Insufficient bool `json:"insufficient,omitempty"`
}

// summarise builds the display view, grouping bank questions by the topic label
// attached to them.
func (m *outcomeMatrix) summarise(backend string, thinking bool, topicOf func(qid string) string) OutcomeSummary {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := OutcomeSummary{Thinking: thinking}
	byTopic := map[string]*TopicSummary{}
	var latencies []int64
	for qid, list := range m.obs {
		for _, o := range list {
			if o.Backend != backend || o.Thinking != thinking {
				continue
			}
			if o.Source == obsSourceJudge {
				out.Judged++
				continue // real traffic has no topic label and is not part of the bank score
			}
			out.Total++
			if o.Correct {
				out.Correct++
			}
			if o.LatencyMS > 0 {
				latencies = append(latencies, o.LatencyMS)
			}
			topic := "other"
			if topicOf != nil {
				if t := topicOf(qid); t != "" {
					topic = t
				}
			}
			ts := byTopic[topic]
			if ts == nil {
				ts = &TopicSummary{Topic: topic}
				byTopic[topic] = ts
			}
			ts.Total++
			if o.Correct {
				ts.Correct++
			}
		}
	}
	if out.Total > 0 {
		out.Rate = float64(out.Correct) / float64(out.Total)
	}
	out.Insufficient = out.Total == 0
	out.MedianMS = medianInt64(latencies)
	for _, ts := range byTopic {
		if ts.Total > 0 {
			ts.Rate = float64(ts.Correct) / float64(ts.Total)
		}
		out.ByTopic = append(out.ByTopic, *ts)
	}
	sort.Slice(out.ByTopic, func(i, j int) bool { return out.ByTopic[i].Topic < out.ByTopic[j].Topic })
	return out
}

// BackendOutcomes is a worker's measured record in both thinking modes. The two
// are reported side by side and never merged: they are separate models, and a
// single headline would hide the case this fleet actually exhibits — a worker
// materially better in one mode than the other.
type BackendOutcomes struct {
	Thinking *OutcomeSummary `json:"thinking,omitempty"`
	NoThink  *OutcomeSummary `json:"nothink,omitempty"`
}

// dimensionMatches reports whether a query vector is in the same space as the
// stored ones. Sampled rather than checked exhaustively: every vector is written
// by the same embedder in one pass, so one is representative, and this runs on
// the routing path.
func (m *outcomeMatrix) dimensionMatches(n int) bool {
	for _, v := range m.vecs {
		return len(v) == n
	}
	return true
}
