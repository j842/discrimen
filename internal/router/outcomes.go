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
//	NOTHING invalidates a row, in fact — see identity.go. A change to the GRADER
//	gives its questions new qids, so those questions are asked again and every
//	other question in the bank still hits the permacache; the superseded rows are
//	not wrong, just never looked up. That is a re-run of the questions whose
//	grading changed, and nothing else. (It is not a re-GRADE: the row keeps the
//	verdict, not the answer text, so an old answer cannot be re-scored by the new
//	grader. The profile's BenchResult.Got holds a tail-truncated copy for the
//	operator to read, and nothing grades from it.)
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
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Observation is one graded answer: a worker, in one thinking mode, on one
// question. The unit of everything in this file.
type Observation struct {
	QID string // identity of the question AND its grading — see benchQuestionQID
	// ModelHash identifies the MODEL this is evidence about, and is what the row
	// is keyed on. Results therefore outlive the worker that produced them: a
	// decommissioned box takes nothing with it, and the same weights deployed
	// elsewhere inherit the whole profile instead of re-earning it.
	ModelHash string
	// Backend is provenance, not identity. It is what distinguishes a latency
	// this worker measured from one another deployment of the same model
	// measured on different hardware — see cachedVerdict.
	Backend  string
	Thinking bool
	Correct  bool
	// Loose records a HALF credit: the answer asserts the expected value but
	// failed strict extraction (see checkAnswerLoose). Stored because Correct
	// alone cannot carry it, and a cache hit that dropped it would re-score the
	// same answers lower on every re-profile — the same worker, the same replies,
	// a worse number, which is exactly the "two scores that no longer mean the
	// same thing" failure the grading rules exist to prevent.
	Loose bool
	// LatencyMS is what the answer COST. Kept beside correctness rather than in a
	// separate table because the routing decision needs both together — the
	// fastest worker that will be right — and joining them later would mean
	// matching on (question, model, mode) anyway.
	//
	// There was an OutputTokens field beside it, carried through the struct,
	// persist() and load() and read by nothing: no producer ever set it, so every
	// row held a zero that the comment here described as a cost signal. Removed
	// rather than wired up, because the term routing wants from an answer's
	// length is already derived from this one — outcomeLengthShape turns the
	// median latency into "questions like this run N times longer on this
	// worker", at the rate the profile was taken at, so a token count would be a
	// second and less direct route to the same number. The output_tokens COLUMN
	// stays where it is: it defaults to 0, dropping a column in SQLite is a table
	// rebuild, and nothing reads or writes it either way.
	LatencyMS int64
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
//
// It is applied in THREE places, and the first one on its own did nothing. As a
// weight on the hit rate it appears in both the numerator and the denominator of
// a weighted mean, so it CANCELS the moment every contributing observation is
// judged — which is the steady state for real traffic. Measured: one correct and
// one incorrect observation produced Correct=0.500, Confidence=1.000,
// Observations=2 and known()=true whether the source was all-judge or all-bench,
// so two GOOD verdicts from a single-token LLM-judge call carried the routing
// authority of two exact-match bench grades. The weight only bites where it does
// NOT cancel: on prediction.evidence(), which is what known() gates on, and on
// Support, which is what supportedCorrect penalises band admission by.
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
// A COUNT rather than a similarity weight, because the weight is a sum of
// cosines: two neighbours at 0.999 sum to just under 2, so an "at least two
// observations" rule written as a similarity threshold fails on exactly the case
// it is meant to admit. It is a floor on having any evidence at all, not a claim
// that two observations are sufficient — Confidence carries that.
//
// It counts prediction.evidence() rather than prediction.Observations, so the
// count is BENCH-WEIGHTED: two judged rows are worth one, and cannot on their
// own make a worker known. See obsJudgeWeight.
const outcomeMinObservations = 2

// outcomeEvidencePenalty is how far below its measured hit rate a prediction is
// judged when it rests on a single unit of evidence — the discount shrinking as
// the square root of the support behind it, which is how the uncertainty in a
// hit rate actually shrinks.
//
// 0.3 is set by the two cases that have to come out right. A worker that got
// both of two nearby questions right lands at 0.83 and is able, which it should
// be: two close bench answers are thin but real. A worker that got ONE of two
// right lands at 0.33 and is not, which is the whole point — that worker's raw
// rate is exactly the floor, so no amount of shrinking TOWARD the floor could
// ever exclude it.
const outcomeEvidencePenalty = 0.3

// prediction is what the matrix says about one MODEL in one mode. Two workers
// serving the same weights get the same prediction — that is the point of
// filing evidence under the model — and are separated afterwards on Seconds,
// which is the term that is genuinely per-box.
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
	// Judged is how many of those Observations came from the background judge
	// rather than from bench grading. It is carried so known() can gate on a
	// bench-weighted count — Observations alone cannot express "judged evidence
	// is weaker", and weighting the ESTIMATE turned out to be a no-op (see
	// obsJudgeWeight).
	//
	// Every producer sets it, and there is exactly one producer now: predictFrom.
	// The validation harness used to build a prediction of its own without
	// tracking the split, so its known() gated as if every neighbour were bench
	// evidence — the permissive direction, and a quiet reason its coverage figure
	// did not describe the gate routing actually applies. It shares predictFrom
	// now, so the two cannot differ again.
	Judged int
	// MedianLatencyMS is what this worker took on those neighbours. Distinct from
	// the live throughput estimate: that answers "how fast is this worker now",
	// this answers "how long does this KIND of question take it", which is the
	// term the live estimate cannot supply.
	MedianLatencyMS int64
}

// evidence is the bench-weighted observation count: a bench row counts one, a
// judged row counts obsJudgeWeight. Two judged verdicts therefore no longer
// satisfy outcomeMinObservations on their own — see obsJudgeWeight for the
// measurement that says they should not.
func (p prediction) evidence() float64 {
	return float64(p.Observations) - float64(p.Judged)*(1-obsJudgeWeight)
}

// known reports whether a prediction rests on enough nearby evidence to gate a
// routing decision.
func (p prediction) known() bool {
	return p.evidence() >= outcomeMinObservations && p.Confidence >= outcomeMinConfidence
}

// supportedCorrect is Correct discounted for how little evidence stands behind
// it — a lower bound rather than a point estimate. Band admission and band
// ordering both use it in place of the raw rate.
//
// outcomeMinObservations is a floor of TWO, so a worker that got one of two
// nearby questions right scored exactly 0.50, cleared outcomeCorrectFloor on the
// >= comparison, and — because the able band was then sorted purely on speed —
// was routed ahead of a worker with a dozen observations at 0.95. Reproduced: a
// worker at p=0.50 with two observations beat one at p=1.00 on a 3% speed edge.
//
// It discounts THINNESS, not evidence: the penalty falls off as the square root
// of the support, so a dozen observations at 0.95 is barely touched while two
// are moved a long way. Support is already judge-discounted, so judged evidence
// is penalised harder — the same statement obsJudgeWeight was always trying to
// make, in the one place where it does not cancel.
func (p prediction) supportedCorrect() float64 {
	if p.Support <= 0 {
		return 0
	}
	return p.Correct - outcomeEvidencePenalty/math.Sqrt(p.Support+1)
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
	// vecTableReady latches once observation_vectors has been created. The
	// schema list lives in main.go, which this file does not own, so the table is
	// created on first use instead; latching on SUCCESS only means a transient
	// failure retries rather than disabling judged-vector persistence for the
	// process lifetime.
	vecTableReady atomic.Bool
	// fullScans counts walks of the WHOLE observation map on the ROUTING path,
	// and vecScans counts walks of the whole VECTOR map. They exist for tests,
	// because this exact shape has now regressed twice: predict once rescanned
	// every vector per candidate (13ms per routed request on a 7-worker fleet,
	// all of it added latency and growing with the judged cache), and the
	// fallback below then did
	// the same thing by calling summary() per candidate. A comment asking the next
	// author to remember did not hold; a counter a test can assert on does.
	//
	// TWO counters because there are two maps and only one of them was pinned.
	// fullScans on its own cannot see the ORIGINAL regression: the per-candidate
	// scan that cost 13ms was in neighboursOf, over m.vecs, and it never touched
	// the observation map at all. A future predict()-per-candidate would have
	// reinstated the measured fault while the existing counter read 1.
	//
	// The DISPLAY and diagnostic walkers deliberately do not count — summarise,
	// String, validate and pruneJudged all walk the observation map and none of
	// them increments. They are not what the counter is protecting: a dashboard
	// poll or an admin request that scans the map is doing what it is for, and a
	// test that counted those would go red for a poll rather than for a routing
	// regression, which is how a guard stops being believed.
	fullScans atomic.Uint64
	vecScans  atomic.Uint64
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
//
// It also enforces the matrix's one invariant about vectors: they are all in the
// SAME embedding space. dot() truncates to the shorter slice, so a vector from a
// retired embedder still scores a plausible cosine against a query from the new
// one (measured at 0.866 — see neighboursOf), and dimensionMatchesLocked only
// SAMPLES, so a mixed map makes that check answer at random. The map could not
// become mixed while every vector was derived in one pass at startup; it can now
// that judged vectors are restored from disk while bank vectors are re-derived.
// So the first vector of a new dimension drops every vector of the old one, in
// memory and on disk — they describe a space nothing else is in any more.
func (m *outcomeMatrix) setVector(qid string, vec []float64) {
	if qid == "" || len(vec) == 0 {
		return
	}
	n := normalize(vec)
	m.mu.Lock()
	spaceChanged := !m.dimensionMatchesLocked(len(n))
	if spaceChanged {
		m.vecs = make(map[string][]float64, len(m.vecs))
	}
	m.vecs[qid] = n
	m.mu.Unlock()
	if spaceChanged {
		m.dropStoredVectors(len(n))
	}
}

// setJudgedVector records a PRODUCTION question's embedding and persists it.
//
// Bank vectors are deliberately not stored: they are a pure function of question
// text that is compiled into the binary, so a restart re-derives them. A judged
// question's text is stored NOWHERE — the observations table holds a qid, which
// is a 64-bit content hash, and no text. So load() used to restore every judged
// observation into m.obs while m.vecs came back empty and was refilled from the
// BANK only. neighboursOf scans m.vecs, so a qid with no vector can never be a
// neighbour: after any restart the judged half of the matrix survived as
// unqueryable ballast that still consumed the maxJudgedQuestions cap and still
// inflated the dashboard's Judged count. That is precisely the bug
// ensureBankVectorsAsync's own comment records finding and fixing for the bank
// half, left unfixed for the half this file's header calls the only evidence
// that will ever cover the traffic actually being served.
//
// The VECTOR is persisted rather than the question text. Re-embedding at startup
// would cost a round trip per judged question and would make the matrix's
// recovery depend on the embeddings worker being up — the exact dependency
// ensureBankVectorsAsync exists to avoid.
func (m *outcomeMatrix) setJudgedVector(ctx context.Context, qid string, vec []float64) error {
	if qid == "" || len(vec) == 0 {
		return nil
	}
	n := normalize(vec)
	m.setVector(qid, n)
	return m.persistVector(ctx, qid, n)
}

func (m *outcomeMatrix) hasVector(qid string) bool {
	m.mu.RLock()
	_, ok := m.vecs[qid]
	m.mu.RUnlock()
	return ok
}

// record adds observations and persists them.
//
// Replaces any earlier observation for the same (question, MODEL, mode) from the
// same source: a re-profile supersedes the previous one rather than accumulating
// alongside it, or a model's history would drag its current estimate. Judged
// observations do NOT supersede bench ones, and vice versa — they are different
// evidence about different traffic.
//
// The key is the model hash and not the worker, which is the same key the
// observations table declares as its PRIMARY KEY, and the two have to stay in
// step or the in-memory map and the file would disagree about what a re-profile
// replaced. It also means the same weights on two hosts hold ONE row per
// question: the second deployment supersedes the first rather than voting twice,
// and Backend records which of them measured the row that survived.
func (m *outcomeMatrix) record(ctx context.Context, obs []Observation) error {
	if len(obs) == 0 {
		return nil
	}
	m.mu.Lock()
	for _, o := range obs {
		kept := m.obs[o.QID][:0]
		for _, prev := range m.obs[o.QID] {
			if prev.ModelHash == o.ModelHash && prev.Thinking == o.Thinking && prev.Source == o.Source {
				continue // superseded
			}
			kept = append(kept, prev)
		}
		m.obs[o.QID] = append(kept, o)
	}
	m.mu.Unlock()
	return m.persist(ctx, obs)
}

// recordIfNewer is record for evidence being RECONSTRUCTED rather than measured:
// it files an observation only when nothing at least as fresh already covers the
// same (question, MODEL, mode, source), and reports how many it filed. Same key
// as record — see there.
//
// record supersedes unconditionally, which is right for a profile that has just
// finished — it is by definition the current truth. It is wrong for
// backfillOutcomesFromProfiles, which reads profiles that may be older than the
// rows already in the table: unconditional supersession would walk a fresh
// re-profile backwards onto whatever the stored profile happened to say. Ties go
// to the incumbent, which is what makes a repeated backfill a no-op.
func (m *outcomeMatrix) recordIfNewer(ctx context.Context, obs []Observation) (int, error) {
	if len(obs) == 0 {
		return 0, nil
	}
	filed := make([]Observation, 0, len(obs))
	m.mu.Lock()
	for _, o := range obs {
		superseded := false
		kept := m.obs[o.QID][:0]
		for _, prev := range m.obs[o.QID] {
			if prev.ModelHash == o.ModelHash && prev.Thinking == o.Thinking && prev.Source == o.Source {
				if !prev.At.Before(o.At) {
					superseded = true
					kept = append(kept, prev) // the incumbent is at least as fresh
				}
				continue
			}
			kept = append(kept, prev)
		}
		if superseded {
			m.obs[o.QID] = kept
			continue
		}
		m.obs[o.QID] = append(kept, o)
		filed = append(filed, o)
	}
	m.mu.Unlock()
	if len(filed) == 0 {
		return 0, nil
	}
	return len(filed), m.persist(ctx, filed)
}

// predict estimates how a worker handles prompts like this one.
//
// vec is the prompt's embedding — the same vector the difficulty classifier
// already computes for every request, so this costs no extra embedding call.
func (m *outcomeMatrix) predict(vec []float64, model string, thinking bool) prediction {
	nb, ok := m.neighboursOf(vec)
	if !ok {
		return prediction{}
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.predictFromLocked(nb, model, thinking)
}

// neighbour is one profiled question near the query, with its similarity.
type neighbour struct {
	qid string
	sim float64
}

// neighboursOf finds the questions nearest a prompt.
//
// Split from the per-worker scoring because it does not depend on the worker:
// the same neighbours answer "how did EVERY candidate do on questions like
// this". predict used to fold the two together, so a routing decision rescanned
// every vector once per candidate — measured at 13ms per request on a 7-worker
// fleet with a saturated judged cache, all of it added latency, and growing with
// the cache. Scanning once and intersecting per worker is the same answer for a
// seventh of the work.
func (m *outcomeMatrix) neighboursOf(vec []float64) ([]neighbour, bool) {
	if len(vec) == 0 {
		return nil, false
	}
	q := normalize(vec)
	m.vecScans.Add(1)
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
	if len(m.vecs) > 0 && !m.dimensionMatchesLocked(len(q)) {
		return nil, false
	}
	out := nearestNeighbours(m.vecs, q, nil)
	if len(out) == 0 {
		return nil, false
	}
	return out, true
}

// nearestNeighbours is the selection itself: every question whose vector clears
// the admission floor, keeping the closest outcomeNeighbours of them. `include`
// may be nil, which admits every question in the map.
//
// Free-standing, and over a plain map rather than the matrix, so the validation
// harness can run the SAME selection routing runs while restricting it to the
// questions its holdout allows. It kept a second copy of this loop until now,
// which is exactly how it drifted into scoring on a different field from the
// routing path and measuring a predictor nothing uses.
//
// q must already be normalised and every vector in vecs must be in the same
// space as it. neighboursOf is what enforces both on the routing path; the
// harness gets them for free, since it works from one snapshot of one map.
//
// TIES BREAK ON QID, and that is not cosmetic. Map iteration order is random, so
// with more than outcomeNeighbours questions at the same similarity — which is
// what a bank of near-duplicate templated questions looks like — an unordered
// truncation kept a DIFFERENT twelve on each call. The same prompt against
// unchanged evidence could then predict differently from one request to the
// next, and a route that cannot be reproduced cannot be explained to the
// operator reading X-Llm-Route.
func nearestNeighbours(vecs map[string][]float64, q []float64, include func(qid string) bool) []neighbour {
	var out []neighbour
	for qid, v := range vecs {
		if include != nil && !include(qid) {
			continue
		}
		if sim := dot(q, v); sim >= outcomeMinSimilarity {
			out = append(out, neighbour{qid, sim})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].sim != out[j].sim {
			return out[i].sim > out[j].sim
		}
		return out[i].qid < out[j].qid
	})
	if len(out) > outcomeNeighbours {
		out = out[:outcomeNeighbours]
	}
	return out
}

// predictFromLocked scores one MODEL against already-found neighbours. The
// caller holds the read lock.
func (m *outcomeMatrix) predictFromLocked(neighbours []neighbour, model string, thinking bool) prediction {
	return predictFrom(m.obs, neighbours, model, thinking)
}

// predictFrom is the scoring itself, over a plain observation map.
//
// Split out for the same reason as nearestNeighbours: the validation harness
// needs to score against a snapshot rather than against the live matrix, and
// while it kept its own copy of this loop the two disagreed about what
// identifies a worker. This one filters on the MODEL HASH, which is the field
// every write path sets and the only field the routing path looks up by; the
// harness's copy filtered on Backend, so a model deployed on two hosts had its
// evidence split in the report and nowhere else.
func predictFrom(obs map[string][]Observation, neighbours []neighbour, model string, thinking bool) prediction {
	var weighted, total, simSum float64
	used, judged := 0, 0
	var latencies []int64
	for _, n := range neighbours {
		for _, o := range obs[n.qid] {
			if o.ModelHash != model || o.Thinking != thinking {
				continue
			}
			w := n.sim
			// The similarity weight is discounted for a judged row and the row is
			// also COUNTED as judged, because only the second discount survives:
			// obsJudgeWeight cancels out of weighted/total whenever every
			// contributing row is judged, which is the steady state for real
			// traffic. See obsJudgeWeight.
			if o.Source == obsSourceJudge {
				w *= obsJudgeWeight
				judged++
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
	if used == 0 || total == 0 {
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
		Judged:          judged,
		MedianLatencyMS: medianInt64(latencies),
	}
}

// bankTally is one MODEL's overall bank record, for the fallback ordering.
type bankTally struct {
	correct int
	total   int
}

// rate is the model's overall hit rate on the bank, and ok reports whether
// anything was measured in this mode at all.
//
// TWO values rather than a rate with a stand-in for the empty case, and the
// difference is the whole reason this signature is what it is. It used to answer
// a "fleet-neutral" 0.5 for a model nothing had been profiled on — so that a new
// worker would not read as "reliably wrong" and be excluded from everything —
// and the fallback ordering below then compared that placeholder against real
// measurements inside outcomeSpeedMargin AS THOUGH IT WERE ONE. A model measured
// at 0.67 is 0.17 above the placeholder, which is wider than the margin, so a
// single measured worker demoted every unmeasured one in the fleet onto the far
// side of a correctness boundary and won every request without its speed being
// looked at. Measured on the live fleet: a CPU llama.cpp worker at 0.67 took
// commit-message prompts off a GPU that prefills 60x faster, 216 seconds against
// 11. "Not measured" is not a score, and a type that cannot express the
// difference will eventually be read as though there is none.
func (t bankTally) rate() (float64, bool) {
	if t.total == 0 {
		return 0, false
	}
	return float64(t.correct) / float64(t.total), true
}

// bankTallies is the one walk of the observation map that both overall-hit-rate
// callers are built from: the whole map, bench rows only, tallied per MODEL
// HASH. `only` restricts the tally to a single model; empty means every model.
//
// ONE walker rather than two, because there were two doing the identical thing
// and they are exactly the pair that caused the measured regression. The
// fallback called summary() per candidate — a full walk of 392 bank rows plus up
// to maxJudgedQuestions judged ones, taking the read lock each time, seven times
// on a seven-worker fleet — which is the same shape, on the same hot path, that
// neighboursOf was split out of predict to remove at 13ms a request. Splitting
// it was the fix; sharing one implementation is what stops the two drifting
// apart again, since a filter added to one of them can no longer miss the other.
//
// It stays a single pass whatever `only` says: filtering inside the walk cannot
// turn one scan into several, which is the property fullScans exists to hold.
func (m *outcomeMatrix) bankTallies(thinking bool, only string) map[string]bankTally {
	m.fullScans.Add(1)
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := map[string]bankTally{}
	for _, list := range m.obs {
		for _, o := range list {
			if o.Thinking != thinking || o.Source != obsSourceBench {
				continue
			}
			if only != "" && o.ModelHash != only {
				continue
			}
			t := out[o.ModelHash]
			t.total++
			if o.Correct {
				t.correct++
			}
			out[o.ModelHash] = t
		}
	}
	return out
}

// summary is the display-only headline for one MODEL: its overall hit rate
// across the whole bank, and how many answers that rests on.
//
// Explicitly NOT a routing input — routing queries neighbours. Its one live
// caller is graderStrength, which needs a single comparable number for a worker
// the matrix has no neighbours for, and which documents itself as falling back
// to exactly this. The per-worker view an operator reads on /backends is
// summarise(), not this: that one carries the topic breakdown, the judged count
// and the median, and a headline compressed out of it would be the same mistake
// the retired quality scalar made.
//
// Zero observations reports (0, 0), and the caller distinguishes "measured at
// zero" from "not measured" on n — the same distinction bankTally.rate makes
// with its second return value, drawn the same way for the same reason.
func (m *outcomeMatrix) summary(model string, thinking bool) (rate float64, n int) {
	t := m.bankTallies(thinking, model)[model]
	if t.total == 0 {
		return 0, 0
	}
	return float64(t.correct) / float64(t.total), t.total
}

// bankRates is summary() for EVERY model at once, in one pass — what the
// fallback ranks candidates on. See bankTallies.
func (m *outcomeMatrix) bankRates(thinking bool) map[string]bankTally {
	return m.bankTallies(thinking, "")
}

// benchObservationsFor is every bank row filed for one model, both modes, for
// publishing across a relay (see relayFleetFor). Bench only: a judged row is a
// grading of somebody's real prompt, and its vector — without which it is
// unreachable by any query anyway — would carry that prompt's embedding over the
// wire. The bank is a fixed, public question set, so these rows disclose nothing
// but how a model did on it.
func (m *outcomeMatrix) benchObservationsFor(hash string) []Observation {
	if hash == "" {
		return nil
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	var out []Observation
	for _, list := range m.obs {
		for _, o := range list {
			if o.Source == obsSourceBench && o.ModelHash == hash {
				out = append(out, o)
			}
		}
	}
	return out
}

// hasBenchEvidence reports whether anything has been graded on this model in
// either mode — "is this worker measured at all", which is what decides whether
// automatic routing will consider it. Early-returns, so the common answer (yes)
// costs a partial walk rather than a full one.
func (m *outcomeMatrix) hasBenchEvidence(hash string) bool {
	if hash == "" {
		return false
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, list := range m.obs {
		for _, o := range list {
			if o.Source == obsSourceBench && o.ModelHash == hash {
				return true
			}
		}
	}
	return false
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
	// output_tokens is left out on purpose and not by omission: nothing has ever
	// set it, the column defaults to 0, and naming it here would only restate the
	// default. See Observation.LatencyMS.
	stmt, err := tx.PrepareContext(ctx, `INSERT OR REPLACE INTO observations
		(qid, model_hash, backend_id, thinking, correct, loose, latency_ms, source, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, o := range obs {
		if _, err := stmt.ExecContext(ctx, o.QID, o.ModelHash, o.Backend, boolInt(o.Thinking), boolInt(o.Correct),
			boolInt(o.Loose), o.LatencyMS, o.Source, o.At.UTC().Format(time.RFC3339Nano)); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// observationVectorsDDL creates the table that makes judged observations
// survive a restart as something a query can reach. See setJudgedVector.
//
// It is created HERE, on first use, rather than in the migration list — that
// list lives in main.go, which this file does not own. IF NOT EXISTS makes it
// idempotent, and the store runs on a single connection, so there is no racing
// creator to lose to.
//
// dim is stored beside the blob rather than derived from its length so a
// truncated row is detectable: a vector in the wrong space that still decodes is
// exactly the failure neighboursOf's dimension check exists to catch.
const observationVectorsDDL = `CREATE TABLE IF NOT EXISTS observation_vectors (
	qid TEXT PRIMARY KEY,
	dim INTEGER NOT NULL,
	vec BLOB NOT NULL,
	created_at TEXT NOT NULL
)`

func (m *outcomeMatrix) ensureVectorTable(ctx context.Context) error {
	if m.db == nil || m.vecTableReady.Load() {
		return nil
	}
	if _, err := m.db.ExecContext(ctx, observationVectorsDDL); err != nil {
		return err
	}
	m.vecTableReady.Store(true)
	return nil
}

// encodeVector packs an embedding as little-endian float32.
//
// float32 rather than float64 because nothing here needs the precision: the
// vector is normalised and the only thing ever computed from it is a cosine
// compared against a 0.55 admission floor, where the two differ around the
// seventh decimal place. It halves what maxJudgedQuestions costs on disk — 4000
// judged questions at 384 dimensions is 6MB rather than 12.
func encodeVector(v []float64) []byte {
	out := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(float32(f)))
	}
	return out
}

func decodeVector(b []byte) []float64 {
	if len(b) == 0 || len(b)%4 != 0 {
		return nil
	}
	out := make([]float64, len(b)/4)
	for i := range out {
		out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:])))
	}
	return out
}

func (m *outcomeMatrix) persistVector(ctx context.Context, qid string, vec []float64) error {
	if m.db == nil || qid == "" || len(vec) == 0 {
		return nil
	}
	if err := m.ensureVectorTable(ctx); err != nil {
		return err
	}
	_, err := m.db.ExecContext(ctx, `INSERT OR REPLACE INTO observation_vectors (qid, dim, vec, created_at)
		VALUES (?, ?, ?, ?)`, qid, len(vec), encodeVector(vec), time.Now().UTC().Format(time.RFC3339Nano))
	return err
}

// dropStoredVectors deletes every persisted vector that is not in the current
// embedding space. Best effort and fire-and-forget: it runs from setVector,
// which has no context and no error path, and the in-memory purge it accompanies
// is what actually protects routing. Leaving a stale row would only cost the
// next restart the same purge again.
func (m *outcomeMatrix) dropStoredVectors(dim int) {
	if m.db == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), storedVectorPurgeTimeout)
	defer cancel()
	if err := m.ensureVectorTable(ctx); err != nil {
		return
	}
	if _, err := m.db.ExecContext(ctx, `DELETE FROM observation_vectors WHERE dim <> ?`, dim); err != nil {
		log.Printf("outcome matrix: dropping vectors from a retired embedding space failed: %v", err)
		return
	}
	log.Printf("outcome matrix: the embedder now returns %d dimensions — judged vectors from the old space were dropped", dim)
}

// storedVectorPurgeTimeout bounds that one statement. Short: it runs on the path
// that fills the bank, and the fill must not stall behind a busy database.
const storedVectorPurgeTimeout = 5 * time.Second

// loadVectors restores the persisted judged embeddings.
//
// Rows are grouped by dimension and only the LARGEST group is returned, because
// the matrix holds one embedding space at a time (see setVector) and a table
// spanning an embedder change holds two. Majority rather than first-seen so the
// answer does not depend on SQLite's row order: the smaller group is whatever
// was accumulating before the swap, and it is the half that is wrong.
func (m *outcomeMatrix) loadVectors(ctx context.Context) (map[string][]float64, error) {
	if m.db == nil {
		return nil, nil
	}
	if err := m.ensureVectorTable(ctx); err != nil {
		return nil, err
	}
	rows, err := m.db.QueryContext(ctx, `SELECT qid, dim, vec FROM observation_vectors`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	byDim := map[int]map[string][]float64{}
	for rows.Next() {
		var qid string
		var dim int
		var blob []byte
		if err := rows.Scan(&qid, &dim, &blob); err != nil {
			return nil, err
		}
		v := decodeVector(blob)
		// A blob whose length disagrees with its declared dimension is truncated
		// or corrupt, not merely stale: dropped rather than used, since dot()
		// would happily score it against a query of the right length.
		if len(v) == 0 || len(v) != dim {
			continue
		}
		if byDim[dim] == nil {
			byDim[dim] = map[string][]float64{}
		}
		byDim[dim][qid] = v
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	best, bestDim := map[string][]float64{}, 0
	for dim, set := range byDim {
		if len(set) > len(best) || (len(set) == len(best) && dim > bestDim) {
			best, bestDim = set, dim
		}
	}
	return best, nil
}

// load rebuilds the in-memory matrix from SQLite at startup.
func (m *outcomeMatrix) load(ctx context.Context) error {
	if m.db == nil {
		return nil
	}
	rows, err := m.db.QueryContext(ctx, `SELECT qid, model_hash, backend_id, thinking, correct, loose,
		latency_ms, source, created_at FROM observations`)
	if err != nil {
		return err
	}
	defer rows.Close()
	loaded := map[string][]Observation{}
	for rows.Next() {
		var o Observation
		var thinking, correct, loose int
		var at string
		if err := rows.Scan(&o.QID, &o.ModelHash, &o.Backend, &thinking, &correct, &loose,
			&o.LatencyMS, &o.Source, &at); err != nil {
			return err
		}
		o.Thinking, o.Correct, o.Loose = thinking == 1, correct == 1, loose == 1
		o.At, _ = time.Parse(time.RFC3339Nano, at)
		loaded[o.QID] = append(loaded[o.QID], o)
	}
	if err := rows.Err(); err != nil {
		return err
	}
	// The cursor must be closed before the next statement runs: the store is
	// opened with SetMaxOpenConns(1), so a query still holding the connection
	// deadlocks anything issued underneath it.
	rows.Close()

	// Judged vectors come back too, or every judged observation restored above
	// would be unreachable by any query — see setJudgedVector. Best effort: a
	// vector table that cannot be read costs prediction coverage, whereas failing
	// the load would take the observations down with it.
	vecs, err := m.loadVectors(ctx)
	if err != nil {
		log.Printf("outcome matrix: restoring judged vectors failed, they stay unqueryable until re-judged: %v", err)
		vecs = nil
	}
	// Vectors whose observations are gone — pruneJudged dropped the question, or
	// the observations table was recreated under it — are dropped rather than
	// restored: they would count against the dimension majority and be scanned on
	// every request for nothing. Removing a WORKER is not one of the cases: a
	// deleted worker keeps its results, because they are filed under the model.
	for qid := range vecs {
		if len(loaded[qid]) == 0 {
			delete(vecs, qid)
		}
	}

	m.mu.Lock()
	m.obs = loaded
	if len(vecs) > 0 {
		m.vecs = vecs
	}
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

// backendsWithEvidence lists the MODEL IDENTITIES the matrix knows anything
// about, for diagnostics — not worker ids, which is what it returned before
// results were filed under the model.
//
// They are opaque hashes, and deliberately so: the point of the identity is that
// it survives the worker, so there is no worker to name for a model whose only
// deployment has been decommissioned. What an operator gets from this is the
// COUNT and whether a freshly-deployed model's hash has appeared yet; mapping a
// hash back to a live worker is modelHash(b) on /backends.
func (m *outcomeMatrix) backendsWithEvidence() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	seen := map[string]bool{}
	for _, list := range m.obs {
		for _, o := range list {
			seen[o.ModelHash] = true
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
	// Seconds is the predicted wall clock: the live estimate, corrected by how
	// much longer questions like this one make this worker generate for. The live
	// estimate knows the worker's current rate and queue but not the answer
	// length; the matrix knows the length and nothing about now. See
	// outcomeSeconds.
	Seconds float64
}

// outcomeCorrectFloor is the predicted hit rate a worker must clear to be
// considered able to answer. Deliberately not high: the estimate is a hit rate
// on SIMILAR questions, not a probability about this one, and demanding 0.9
// would empty the candidate set on any topic the fleet finds hard — leaving
// nothing to route to at exactly the moment routing matters.
const outcomeCorrectFloor = 0.5

// outcomeSpeedMargin is how much predicted accuracy this file will trade for
// speed, and it is the constant that implements "leaning towards speed": among
// workers within this much of the best predicted correctness, take the fastest.
// A margin rather than a pure speed sort, so nothing markedly worse can be
// picked just because it is quick.
//
// It applies on BOTH paths now. It used to be applied only in the fallback,
// while the primary path sorted its able band purely on Seconds — so the policy
// the matrix chose with evidence was MORE speed-dominant than the one it fell
// back to with none, which is backwards. Reproduced: a worker at p=0.50 with two
// observations was routed ahead of one at p=1.00 for a 3% speed edge.
const outcomeSpeedMargin = 0.15

// AUTOMATIC ROUTING NEVER CHOOSES A WORKER NOTHING HAS MEASURED. Both rankings
// below put unmeasured workers last, behind even a worker measured to be wrong,
// and there is no mechanism anywhere that promotes one — measurement is what
// earns a worker traffic, and it is available on demand.
//
// There used to be one: outcomeExploreEvery, a counter that promoted the best
// unmeasured candidate to the front of one request in twenty, on the argument
// that a worker never tried never earns evidence and the matrix's first
// impression of a fleet therefore becomes permanent. That argument was written
// when profiling did not fill the matrix. It does now — every worker records its
// bench results as observations when it profiles, and a cold start profiles
// automatically — so the bootstrap it existed for happens before the worker takes
// a single request, and the exploration was left buying evidence at the price of
// misrouting live traffic to workers whose quality was unknown BECAUSE nobody had
// asked for it.
//
// What remains genuinely unmeasured is a relay row, whose evidence sits on the
// upstream router. Gambling one request in twenty on it is not the way to reach
// that evidence; importing it is.

// chooseByOutcome ranks candidates for a prompt: workers predicted to answer it
// correctly first, and among those of comparable predicted correctness, the
// fastest — "best overall considering quality and speed, leaning towards speed",
// with outcomeSpeedMargin fixing what comparable means.
//
// Returns the ordered candidates and a short reason for the route header, so a
// decision can be read back afterwards — including, importantly, whether the
// matrix actually knew anything or fell back.
// The third return value is how many LEADING candidates the matrix considers
// interchangeable on correctness — the `able` band. It exists because the
// ordering this function returns is a complete policy decision, and the
// acquisition step downstream used to re-sort the whole list by cost and
// locality, which silently promoted workers the matrix had predicted WRONG above
// ones it had predicted right. Acquisition may reorder within the prefix (where
// the matrix is genuinely indifferent) and must not reorder across it.
//
// Zero means "no band is authoritative" — the fallback, where nothing has been
// measured and there is no correctness judgement to protect.
func (m *outcomeMatrix) chooseByOutcome(cands []*Backend, vec []float64, thinking bool, job jobCost) ([]*Backend, string, int) {
	if len(cands) == 0 {
		return nil, "no candidates", 0
	}
	// ONE neighbour scan for the whole request, then a cheap intersect per
	// candidate — the neighbours are a property of the prompt, not of the worker.
	nb, haveNeighbours := m.neighboursOf(vec)
	choices := make([]outcomeChoice, 0, len(cands))
	known := 0
	m.mu.RLock()
	for _, b := range cands {
		var p prediction
		if haveNeighbours {
			p = m.predictFromLocked(nb, modelHash(b), thinking)
		}
		if p.known() {
			known++
		}
		choices = append(choices, outcomeChoice{Backend: b, Pred: p, Seconds: outcomeSeconds(b, p, job)})
	}
	m.mu.RUnlock()

	// The ordinary path: rank the workers the matrix expects to be right ahead of
	// the rest, and among the ones it cannot separate on correctness take the
	// fastest. This is the whole routing policy — there is no quality target and
	// nothing compared against one.
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
		//	predicted correct  by correctness margin, then speed — the routing goal
		//	predicted wrong    by predicted correctness, best of a bad lot
		//	unmeasured         by speed, behind both — never the router's choice
		//
		// The second band sorts on ACCURACY rather than speed because a request
		// that reaches it is one the fleet is not good at, and a fast wrong answer
		// helps nobody.
		//
		// The last band used to sit in the middle, ahead of the workers the matrix
		// predicted WRONG, on the reasoning that an unknown beats a known failure.
		// It is last now because automatic routing does not choose a worker nothing
		// has measured, at all: an unmeasured worker is one nobody has run the
		// profiler against, and the answer to that is to measure it — not to find
		// out at the cost of somebody's request. Being last is not being evicted;
		// failover, escalation and slot spill still reach it, and a fleet where
		// NOTHING is measured has one band and routes on speed as it always did.
		//
		// Admission to the first band is on supportedCorrect, not the raw rate: a
		// worker that got one of two nearby questions right sat exactly on the
		// floor, and the >= comparison let it in on evidence too thin to mean
		// anything. See supportedCorrect.
		var able, unmeasured, unable []outcomeChoice
		for _, c := range choices {
			switch {
			case !c.Pred.known():
				unmeasured = append(unmeasured, c)
			case c.Pred.supportedCorrect() > outcomeCorrectFloor:
				able = append(able, c)
			default:
				unable = append(unable, c)
			}
		}
		sortByCorrectnessThenSpeed(able)
		sort.SliceStable(unable, func(i, j int) bool { return unable[i].Pred.Correct > unable[j].Pred.Correct })
		sortBySeconds(unmeasured)
		ordered := make([]outcomeChoice, 0, len(choices))
		ordered = append(ordered, able...)
		ordered = append(ordered, unable...)
		ordered = append(ordered, unmeasured...)
		switch {
		case len(able) > 0:
			// Support is reported alongside the rate because the two are not
			// interchangeable and only one of them was visible: an operator reading
			// p=1.00,n=2 could not tell whether the prediction rested on one strong
			// neighbour or a dozen weak ones, and it is the second number that
			// decides whether the first one survives band admission.
			return backendsOf(ordered), fmt.Sprintf("outcome:p=%.2f,n=%d,sup=%.1f",
				able[0].Pred.Correct, able[0].Pred.Observations, able[0].Pred.Support), len(able)
		case len(unable) > 0:
			return backendsOf(ordered), fmt.Sprintf("outcome:best-effort,p=%.2f,sup=%.1f",
				unable[0].Pred.Correct, unable[0].Pred.Support), 0
		default:
			// Unreachable: this branch runs only when some candidate carried a
			// prediction, and such a candidate is in one of the two bands above. Said
			// as a total rather than by indexing a band that is empty by
			// construction — the previous shape read unable[0] here, which was safe
			// only because the case above it could not fall through to it.
			return backendsOf(ordered), "outcome:none-measured,unmeasured-last", 0
		}
	}

	// FALLBACK: nothing similar has been profiled, which is the common case for
	// traffic unlike the bank. Rank on each MODEL's overall record on the bank
	// instead — best overall, leaning speed.
	//
	// The SAME THREE BANDS as the measured path above, in the same order and for
	// the same reasons:
	//
	//	measured, and as good as the best   by speed
	//	measured, and worse than that       by hit rate
	//	not measured at all                 by speed, behind both
	//
	// The last band is the whole of this change. An unmeasured model used to be
	// handed a "fleet-neutral" 0.5 and compared against real bank rates inside
	// outcomeSpeedMargin as though that were a measurement of it (see
	// bankTally.rate), and both directions of that comparison were wrong. A single
	// model measured at 0.67 sits 0.17 above the placeholder, wider than the
	// margin, so ONE measured worker put every unmeasured worker in the fleet on
	// the far side of a correctness boundary and won every request without its
	// speed being looked at — 216 seconds on a CPU box against 11 on the GPU it
	// out-ranked. In the other direction the same placeholder outranked a model
	// honestly measured at 0.45, on no evidence whatever.
	//
	// Ranking them last needs no floor and no second constant: "measured worse
	// than the best" and "not measured" are different bands, so nothing has to
	// decide how a rate compares to the absence of one.
	//
	// ONE pass for every worker's overall rate, not one scan per candidate. See
	// bankRates: this branch used to call summary() in the loop, which is the same
	// per-candidate full scan that cost 13ms a request before neighboursOf was
	// split out of predict.
	tallies := m.bankRates(thinking)
	best := 0.0
	// Keyed by MODEL in the tally (that is what the evidence is about) and by
	// worker in `rates` (that is what the ordering below indexes). A worker absent
	// from `rates` is one nothing has measured in this mode, which is a state of
	// the evidence and not a value it can hold.
	rates := make(map[string]float64, len(cands))
	for _, b := range cands {
		rate, ok := tallies[modelHash(b)].rate()
		if !ok {
			continue
		}
		rates[b.ID] = rate
		if rate > best {
			best = rate
		}
	}
	// Rank, do not narrow — everything is returned in some order, so failover and
	// escalation still have somewhere to go, and a fleet with no bank evidence at
	// all is one band ordered by speed rather than a 503.
	var able, unmeasured, unable []outcomeChoice
	for _, c := range choices {
		rate, measured := rates[c.Backend.ID]
		switch {
		case !measured:
			unmeasured = append(unmeasured, c)
		case rate >= best-outcomeSpeedMargin:
			able = append(able, c)
		default:
			unable = append(unable, c)
		}
	}
	sortBySeconds(able)
	sort.SliceStable(unable, func(i, j int) bool { return rates[unable[i].Backend.ID] > rates[unable[j].Backend.ID] })
	sortBySeconds(unmeasured)
	ordered := make([]outcomeChoice, 0, len(choices))
	ordered = append(ordered, able...)
	ordered = append(ordered, unable...)
	ordered = append(ordered, unmeasured...)
	if len(ordered) == 0 {
		return nil, "no candidates", 0
	}
	// The fallback has measured nothing about THIS PROMPT, so it protects no
	// prefix: `able` here is a record on the bank as a whole, which says nothing
	// about the question in hand, and acquisition is free to reorder on cost and
	// locality across all of it. That is a different claim from the measured
	// path's, and returning a band length here would quietly promote it.
	//
	// q= is the leading worker's bank rate, and reads `none` when the request is
	// being served by a worker nothing has measured — the one number that says
	// which band won, from a log line.
	rate, measured := rates[ordered[0].Backend.ID]
	if !measured {
		return backendsOf(ordered), "outcome:unknown,fallback-speed,q=none", 0
	}
	return backendsOf(ordered), fmt.Sprintf("outcome:unknown,fallback-speed,q=%.2f", rate), 0
}

// outcomeSeconds predicts wall clock for one worker on this job.
//
// The two inputs know different things, and which one owns which term is the
// whole of it. expectedLatency knows the LIVE state — this worker's current
// decode rate, this prompt's prefill, the session prefill discount, and the
// queue it is sitting behind. The matrix knows the one thing the live figures
// cannot supply: how long questions LIKE this one make this worker generate for.
//
// This used to be max(live, recorded), defended as asymmetric cost — under-
// predicting misses a deadline, over-predicting only loses a race. That is fair
// for a tie-break and wrong as a rule. Whenever the recorded median dominated,
// the sort key became a CONSTANT per (worker, prompt-neighbourhood): it stopped
// moving with queue depth, with prompt length, and with the incumbent discount,
// so every live-load mechanism underneath it was dead for that comparison. The
// substituted number is also biased, because bench latencies are timed at each
// worker's OWN measured max batch size, which inflates a high-capacity worker's
// median relative to a low-capacity one's. Measured inversion: an idle 8-slot
// worker (median 42s, live estimate 6s) lost to a saturated 2-slot worker
// (median 20s, live estimate 18s), purely because 42 > 20.
//
// So the median is used for SHAPE only, and the live estimate keeps the rate and
// the queue. outcomeLengthShape reduces the median to "questions like this run N
// times longer than a generic answer on this worker"; the extra decode time that
// implies is ADDED to the live estimate rather than replacing it. Adding leaves
// the queue term always in force, which is the property that matters: a
// saturated worker can no longer rank ahead of an idle equivalent whatever its
// median says, because the two differ by a strictly positive queue multiplier
// applied to a term that is never discarded.
func outcomeSeconds(b *Backend, p prediction, job jobCost) float64 {
	generic := expectedLatency(b, job)
	if p.MedianLatencyMS <= 0 {
		return generic
	}
	shape := outcomeLengthShape(b, p, job)
	if shape <= 0 {
		return generic // no rate to divide by: the median says nothing usable
	}
	tps := liveTPSFor(b, job.mode)
	if tps <= 0 {
		tps = 1
	}
	// The decode term generic already contains, at today's rate. Scaling it by
	// (shape-1) is the extra generation this KIND of question implies; prefill is
	// untouched because answer length does not change it.
	decode := expectedGenTokens(b, job) / tps
	return generic + (shape-1)*decode
}

// outcomeLengthShape is how much longer questions like this one ran on this
// worker than a generic answer did, measured at the rates its profile was taken
// at so the rate cancels and only the LENGTH is left.
//
// The batching bias largely divides out with it: both the recorded median and
// the baseline it is divided by were measured on the same worker under the same
// profile, so a worker whose median is inflated by an eight-wide batch has an
// equally inflated denominator. Clamped anyway, because that cancellation is
// partial and because one pathological neighbour must not multiply a sort key by
// fifty.
func outcomeLengthShape(b *Backend, p prediction, job jobCost) float64 {
	rate := b.BaselineTPS
	if rate <= 0 {
		rate = liveTPS(b)
	}
	if rate <= 0 {
		return 0
	}
	nominal := float64(latencyEstTokens)
	if job.mode == thinkingOn {
		nominal = float64(latencyEstThinkTokens)
	}
	baseline := float64(b.Certification.TTFTMillis)/1000 + nominal/rate
	if baseline <= 0 {
		return 0
	}
	shape := (float64(p.MedianLatencyMS) / 1000) / baseline
	if shape < outcomeShapeMin {
		return outcomeShapeMin
	}
	if shape > outcomeShapeMax {
		return outcomeShapeMax
	}
	return shape
}

// The band the length factor is allowed to occupy. Wide enough to carry the real
// spread (a one-line answer against a full reasoning trace), narrow enough that a
// median distorted by batching, a cold cache or a single outlier neighbour cannot
// dominate the live estimate it is correcting.
const (
	outcomeShapeMin = 0.5
	outcomeShapeMax = 4.0
)

func sortBySeconds(cs []outcomeChoice) {
	sort.SliceStable(cs, func(i, j int) bool { return cs[i].Seconds < cs[j].Seconds })
}

// sortByCorrectnessThenSpeed orders the able band: "best overall considering
// quality and speed, leaning towards speed", stated as an ordering.
//
// Candidates within outcomeSpeedMargin of the BEST predicted correctness in the
// band are treated as equally likely to be right, and among those the fastest
// wins — that is the lean towards speed, and it is the same rule the fallback
// has always applied to overall hit rate. Anything further behind than the
// margin sorts after them by correctness, because at that point the difference
// is no longer a rounding error the speed term may overrule.
//
// The band used to be sorted purely on Seconds, which made accuracy weightless
// above the floor: a 3% speed edge beat a 2x difference in predicted
// correctness. Correctness is compared on supportedCorrect so that "better" also
// accounts for how much evidence each estimate rests on.
func sortByCorrectnessThenSpeed(cs []outcomeChoice) {
	if len(cs) < 2 {
		return
	}
	best := 0.0
	for _, c := range cs {
		if v := c.Pred.supportedCorrect(); v > best {
			best = v
		}
	}
	sort.SliceStable(cs, func(i, j int) bool {
		iNear := cs[i].Pred.supportedCorrect() >= best-outcomeSpeedMargin
		jNear := cs[j].Pred.supportedCorrect() >= best-outcomeSpeedMargin
		if iNear != jNear {
			return iNear
		}
		if iNear {
			return cs[i].Seconds < cs[j].Seconds
		}
		return cs[i].Pred.supportedCorrect() > cs[j].Pred.supportedCorrect()
	})
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
// one would silently change what every model's summary is computed over.
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
	haveVectors := m.ensureVectorTable(ctx) == nil
	for _, qid := range ids {
		if _, err := m.db.ExecContext(ctx, `DELETE FROM observations WHERE qid = ? AND source = ?`,
			qid, obsSourceJudge); err != nil {
			return // best effort: the in-memory prune is what bounds the hot path
		}
		// The stored vector goes with it. Without this the cap would bound the
		// observations while observation_vectors grew forever, and the next
		// restart would reload vectors for questions that no longer exist.
		if haveVectors {
			if _, err := m.db.ExecContext(ctx, `DELETE FROM observation_vectors WHERE qid = ?`, qid); err != nil {
				return
			}
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

// OutcomeSummary is the display-only view of one MODEL in one mode, which is
// what an operator sees against each worker serving it. Two workers running the
// same weights therefore show identical numbers, and that is not a bug to fix
// later: the evidence is about the weights, so attributing half of it to each
// box would be the fiction, not the shared figure.
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
func (m *outcomeMatrix) summarise(model string, thinking bool, topicOf func(qid string) string) OutcomeSummary {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := OutcomeSummary{Thinking: thinking}
	byTopic := map[string]*TopicSummary{}
	var latencies []int64
	for qid, list := range m.obs {
		for _, o := range list {
			if o.ModelHash != model || o.Thinking != thinking {
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

// dimensionMatchesLocked reports whether a query vector is in the same space as
// the stored ones. Sampled rather than checked exhaustively: setVector drops the
// whole map the moment a vector of a new width arrives, so the map holds ONE
// width at a time and any member of it is representative — and this runs on the
// routing path, where an exhaustive check would be a second full scan. The
// caller holds the lock; setVector holds the write one, since it acts on the
// answer.
func (m *outcomeMatrix) dimensionMatchesLocked(n int) bool {
	for _, v := range m.vecs {
		return len(v) == n
	}
	return true
}

// cachedVerdict is the permacache: has this MODEL already answered this exact
// question, graded this exact way?
//
// A hit means the generation does not have to happen. That is the expensive part
// of profiling by a wide margin — the capacity ramp and the speed probe are
// seconds, the question bank is hours — so this is what turns a re-profile from
// a fleet-wide outage into a few probes.
//
// Bench evidence only. A judged production answer was graded by another model on
// a prompt that merely resembles this one; it is fine as routing evidence and is
// not a substitute for asking the question.
//
// sameWorker reports whether the row was measured on the worker now asking. It is
// the caller's signal about the LATENCY, which unlike the verdict does not travel:
// correctness is a property of the weights, the engine and the mode, all of which
// the model hash pins, while latency is all of that plus the box and its load. A
// hit from another deployment of the same model carries a real verdict and a
// meaningless duration.
func (m *outcomeMatrix) cachedVerdict(qid, model, backendID string, thinking bool) (o Observation, sameWorker bool, ok bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, prev := range m.obs[qid] {
		if prev.ModelHash == model && prev.Thinking == thinking && prev.Source == obsSourceBench {
			return prev, prev.Backend == backendID && backendID != "", true
		}
	}
	return Observation{}, false, false
}
