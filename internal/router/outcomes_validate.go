package router

// Validating the outcome matrix — with a DOMAIN holdout, never a random split.
//
// This file exists because of one measurement. ADeLe ran the same ablation this
// router relies on — predict per-model correctness from a prompt embedding —
// across 15 models and 16,108 instances:
//
//	                    in-distribution   task-OOD   benchmark-OOD
//	text embeddings          0.805          0.740        0.480
//
// Benchmark-OOD is WORSE THAN CHANCE. In-distribution the same predictor looks
// perfectly healthy at 0.805, because it is reading which topics a model is bad
// at rather than which questions are hard. FORC reports the same shape from the
// other side: pooled AUROC 80.62, and per-dataset 50.00 on GSM8K, 50.00 on LSAT
// — the pooled figure was almost entirely "which benchmark did this come from".
//
// So a random train/test split cannot detect the failure mode this router is
// most exposed to. Splitting by DOMAIN can: hold out every question from one
// task, predict it using only the others, and see whether the prediction
// survives the move. The gap between the two numbers IS the finding.
//
// Two supporting diagnostics, both cheap and both worth knowing before trusting
// any routing decision:
//
//	Degenerate fraction — questions every worker passes or every worker fails.
//	They cannot discriminate between workers no matter how good the predictor is.
//	One published pool was 52.5% all-correct, which capped the achievable gain at
//	+1.2pp regardless of method.
//
//	Distance/agreement correlation — do embedding-near questions actually have
//	similar per-model outcomes? That is the assumption kNN rests on, and it is
//	directly measurable. A strong negative correlation is what "kNN will work
//	here" looks like; near zero means the neighbours are topical company rather
//	than evidence.
//
// Everything here identifies an answerer by its MODEL HASH, which is what the
// routing path predicts about — see predictExcluding for what identifying it by
// worker instead cost this harness.

import (
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
)

// validationScore is predictive performance over one set of held-out questions.
type validationScore struct {
	// AUROC over (predicted hit rate, actual correct) pairs. 0.5 is chance.
	// Reported rather than accuracy because the base rate varies by domain and an
	// accuracy figure would mostly report that.
	AUROC float64 `json:"auroc"`
	// Brier is the mean squared error of the probability estimate — the part
	// AUROC cannot see, since a perfectly ordered but badly scaled predictor
	// scores well on AUROC and still misroutes against a fixed threshold.
	Brier float64 `json:"brier"`
	N     int     `json:"n"`
	// Predicted is how many of the N had any usable prediction at all. A
	// predictor that declines most of the time can post a fine AUROC on the few
	// it answers while being useless in practice.
	Predicted int `json:"predicted"`
}

type domainScore struct {
	Domain string          `json:"domain"`
	Score  validationScore `json:"score"`
}

// ValidationReport is what `bench validate` prints and the admin surface serves.
type ValidationReport struct {
	InDistribution validationScore `json:"in_distribution"`
	// DomainHoldout is the honest number: each domain predicted using only the
	// others. If this is near 0.5 while InDistribution is high, the matrix is
	// recognising topics rather than predicting difficulty.
	DomainHoldout  []domainScore   `json:"domain_holdout"`
	HoldoutMean    validationScore `json:"holdout_mean"`
	DegenerateFrac float64         `json:"degenerate_fraction"`
	AllCorrectFrac float64         `json:"all_correct_fraction"`
	AllFailFrac    float64         `json:"all_fail_fraction"`
	// DistanceAgreementR is the correlation between embedding distance and
	// disagreement in per-model outcomes, over sampled question pairs. Negative
	// and large is good: near questions behave alike.
	DistanceAgreementR float64 `json:"distance_agreement_r"`
	// Questions is how many question VECTORS the matrix holds — the pool the
	// holdout draws from, since a question with no vector can never be a
	// neighbour and so can never be predicted from or about.
	Questions int `json:"questions"`
	// Workers is how many distinct MODELS have evidence, not how many boxes. The
	// json tag keeps its old name so an operator's saved command still works; the
	// number it carries changed when results moved from being about a worker to
	// being about the weights it serves.
	Workers int `json:"workers"`
}

// validate runs the whole diagnostic. domainOf labels each question; questions
// with no label are pooled under "other".
func (m *outcomeMatrix) validate(domainOf func(qid string) string) ValidationReport {
	m.mu.RLock()
	// Snapshot under the lock, then work on the copy: the holdout loop is O(n²)
	// in questions and must not hold up routing.
	vecs := make(map[string][]float64, len(m.vecs))
	for k, v := range m.vecs {
		vecs[k] = v
	}
	obs := make(map[string][]Observation, len(m.obs))
	for k, list := range m.obs {
		obs[k] = append([]Observation(nil), list...)
	}
	m.mu.RUnlock()

	rep := ValidationReport{Questions: len(vecs)}
	// Counted by MODEL HASH, which is what a "worker" is to everything downstream
	// of here — the field every write path sets and the only one predictFrom
	// looks up by. Counting distinct Backend values instead reported the number of
	// BOXES, which is a different and larger number whenever the same weights run
	// on two hosts, and it made the report claim more independent evidence than
	// the matrix holds.
	models := map[string]bool{}
	for _, list := range obs {
		for _, o := range list {
			models[o.ModelHash] = true
		}
	}
	rep.Workers = len(models)
	if len(vecs) == 0 || rep.Workers == 0 {
		return rep
	}

	byDomain := map[string][]string{}
	for qid := range vecs {
		d := "other"
		if domainOf != nil {
			if got := domainOf(qid); got != "" {
				d = got
			}
		}
		byDomain[d] = append(byDomain[d], qid)
	}

	// In-distribution: every question predicted with every question available,
	// itself excluded. This is the number a random split would report, and it is
	// here to be COMPARED against the holdout rather than believed on its own.
	rep.InDistribution = scoreOver(vecs, obs, keysOf(vecs), func(qid string) bool { return true })

	var holdouts []validationScore
	for domain, qids := range byDomain {
		if len(qids) < 4 {
			continue // too few to say anything
		}
		excluded := map[string]bool{}
		for _, q := range qids {
			excluded[q] = true
		}
		s := scoreOver(vecs, obs, qids, func(qid string) bool { return !excluded[qid] })
		rep.DomainHoldout = append(rep.DomainHoldout, domainScore{Domain: domain, Score: s})
		holdouts = append(holdouts, s)
	}
	sort.Slice(rep.DomainHoldout, func(i, j int) bool {
		return rep.DomainHoldout[i].Domain < rep.DomainHoldout[j].Domain
	})
	rep.HoldoutMean = meanScore(holdouts)
	rep.AllCorrectFrac, rep.AllFailFrac = degenerateFractions(obs)
	rep.DegenerateFrac = rep.AllCorrectFrac + rep.AllFailFrac
	rep.DistanceAgreementR = distanceAgreementCorrelation(vecs, obs)
	return rep
}

// scoreOver predicts each question in `targets` using only the questions
// `allowed` admits, and scores the predictions against what actually happened.
func scoreOver(vecs map[string][]float64, obs map[string][]Observation,
	targets []string, allowed func(qid string) bool) validationScore {
	var probs []float64
	var actual []float64
	for _, qid := range targets {
		v := vecs[qid]
		if len(v) == 0 {
			continue
		}
		for _, o := range obs[qid] {
			if o.Source != obsSourceBench {
				continue
			}
			// Predict this MODEL on this question WITHOUT using the question
			// itself, and without anything `allowed` excludes.
			p := predictExcluding(vecs, obs, v, o.ModelHash, o.Thinking, func(other string) bool {
				return other != qid && allowed(other)
			})
			if !p.known() {
				continue
			}
			probs = append(probs, p.Correct)
			if o.Correct {
				actual = append(actual, 1)
			} else {
				actual = append(actual, 0)
			}
		}
	}
	total := 0
	for _, qid := range targets {
		for _, o := range obs[qid] {
			if o.Source == obsSourceBench {
				total++
			}
		}
	}
	return validationScore{
		AUROC:     auroc(probs, actual),
		Brier:     brier(probs, actual),
		N:         total,
		Predicted: len(probs),
	}
}

// predictExcluding is predict() restricted to a subset of the question pool —
// the mechanism the holdout needs.
//
// It is now literally the routing path's own selection and scoring, over a
// snapshot instead of the live matrix: nearestNeighbours with the holdout's
// filter, then predictFrom. It USED to be a second copy of both loops, and the
// claim in this comment that it was "deliberately the same code path" was how a
// divergence went unnoticed — the copy filtered observations on o.Backend while
// routing filtered on o.ModelHash. Every model deployed on more than one host
// therefore had its evidence split in the report and nowhere else, so the
// holdout under-counted neighbours, declined predictions routing would have
// made, and reported a coverage figure lower than the real one. The measured
// thing was a predictor the router does not use.
//
// The snapshot's vectors are all in one space and already normalised, which is
// what lets it call nearestNeighbours directly; neighboursOf's dimension guard
// is for the live path, where a query arrives from an embedder that may have
// changed under the stored vectors.
func predictExcluding(vecs map[string][]float64, obs map[string][]Observation,
	q []float64, model string, thinking bool, include func(qid string) bool) prediction {
	return predictFrom(obs, nearestNeighbours(vecs, q, include), model, thinking)
}

// degenerateFractions is the share of questions every model passes, and the
// share every model fails. Neither can distinguish models, so together they
// bound how much any router can possibly gain over picking at random.
//
// No identity filter needed: record() supersedes on (question, model, mode), so
// the bench rows under one qid are already one per (model, mode) and counting
// them counts distinct answerers.
func degenerateFractions(obs map[string][]Observation) (allCorrect, allFail float64) {
	graded, correct, failed := 0, 0, 0
	for _, list := range obs {
		n, hits := 0, 0
		for _, o := range list {
			if o.Source != obsSourceBench {
				continue
			}
			n++
			if o.Correct {
				hits++
			}
		}
		if n < 2 {
			continue // one model cannot be unanimous
		}
		graded++
		switch hits {
		case n:
			correct++
		case 0:
			failed++
		}
	}
	if graded == 0 {
		return 0, 0
	}
	return float64(correct) / float64(graded), float64(failed) / float64(graded)
}

// distanceAgreementCorrelation measures kNN's central assumption directly: do
// questions that are close in embedding space produce similar per-model
// outcomes?
//
// For every pair of questions with shared models, correlate cosine SIMILARITY
// against outcome DISAGREEMENT (the fraction of shared models that answered
// them differently). A strong negative correlation means near questions behave
// alike, which is what makes a neighbour informative. Near zero means the
// neighbours are topical company rather than evidence — the matrix would then be
// predicting from questions that share a subject and nothing else.
//
// Paired on the MODEL HASH, like everything else that asks "is this the same
// answerer". On Backend it paired boxes, so a model on two hosts contributed no
// pairs at all for the questions its two profiles happened to split between
// them — the rows exist and describe one answerer, and the identity that says so
// is the hash.
func distanceAgreementCorrelation(vecs map[string][]float64, obs map[string][]Observation) float64 {
	qids := keysOf(vecs)
	sort.Strings(qids) // deterministic sampling
	// Bounded: this is O(n²) and runs on demand, not on the request path.
	const maxQuestions = 300
	if len(qids) > maxQuestions {
		stride := len(qids) / maxQuestions
		var sampled []string
		for i := 0; i < len(qids); i += stride {
			sampled = append(sampled, qids[i])
		}
		qids = sampled
	}
	var sims, disagreements []float64
	for i := 0; i < len(qids); i++ {
		for j := i + 1; j < len(qids); j++ {
			a, b := qids[i], qids[j]
			shared, differ := 0, 0
			for _, oa := range obs[a] {
				if oa.Source != obsSourceBench {
					continue
				}
				for _, ob := range obs[b] {
					if ob.Source != obsSourceBench || ob.ModelHash != oa.ModelHash || ob.Thinking != oa.Thinking {
						continue
					}
					shared++
					if oa.Correct != ob.Correct {
						differ++
					}
				}
			}
			if shared == 0 {
				continue
			}
			sims = append(sims, dot(vecs[a], vecs[b]))
			disagreements = append(disagreements, float64(differ)/float64(shared))
		}
	}
	return pearson(sims, disagreements)
}

// ── statistics ─────────────────────────────────────────────────────────────

// auroc is the probability a randomly chosen positive scores above a randomly
// chosen negative, computed by rank sum so ties are handled correctly.
func auroc(scores, labels []float64) float64 {
	if len(scores) != len(labels) || len(scores) == 0 {
		return 0
	}
	pos, neg := 0.0, 0.0
	for _, l := range labels {
		if l > 0.5 {
			pos++
		} else {
			neg++
		}
	}
	if pos == 0 || neg == 0 {
		return 0 // undefined: every observation had the same outcome
	}
	idx := make([]int, len(scores))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return scores[idx[a]] < scores[idx[b]] })
	ranks := make([]float64, len(scores))
	for i := 0; i < len(idx); {
		// j starts one past i, not at i. NaN == NaN is false, so a NaN score left
		// the inner loop unable to advance and the outer loop pinned a core for
		// the life of the process — reachable from one authenticated
		// /admin/outcomes?validate=1. The equivalent loop in benchgen_emit.go
		// already advances unconditionally.
		j := i + 1
		for j < len(idx) && scores[idx[j]] == scores[idx[i]] {
			j++
		}
		avg := float64(i+j+1) / 2 // average rank for the tied block, 1-based
		for k := i; k < j; k++ {
			ranks[idx[k]] = avg
		}
		i = j
	}
	sumPos := 0.0
	for i, l := range labels {
		if l > 0.5 {
			sumPos += ranks[i]
		}
	}
	return (sumPos - pos*(pos+1)/2) / (pos * neg)
}

func brier(scores, labels []float64) float64 {
	if len(scores) == 0 {
		return 0
	}
	sum := 0.0
	for i := range scores {
		d := scores[i] - labels[i]
		sum += d * d
	}
	return sum / float64(len(scores))
}

func pearson(xs, ys []float64) float64 {
	if len(xs) != len(ys) || len(xs) < 2 {
		return 0
	}
	var mx, my float64
	for i := range xs {
		mx += xs[i]
		my += ys[i]
	}
	mx /= float64(len(xs))
	my /= float64(len(ys))
	var num, dx, dy float64
	for i := range xs {
		a, b := xs[i]-mx, ys[i]-my
		num += a * b
		dx += a * a
		dy += b * b
	}
	if dx == 0 || dy == 0 {
		return 0
	}
	return num / math.Sqrt(dx*dy)
}

func meanScore(ss []validationScore) validationScore {
	if len(ss) == 0 {
		return validationScore{}
	}
	var out validationScore
	for _, s := range ss {
		out.AUROC += s.AUROC
		out.Brier += s.Brier
		out.N += s.N
		out.Predicted += s.Predicted
	}
	out.AUROC /= float64(len(ss))
	out.Brier /= float64(len(ss))
	return out
}

func keysOf(m map[string][]float64) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// String renders the report for a terminal, leading with the comparison that
// matters rather than the number that flatters.
func (v ValidationReport) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "outcome matrix: %d questions, %d workers\n\n", v.Questions, v.Workers)
	fmt.Fprintf(&b, "  in-distribution   AUROC %.3f  Brier %.3f  (%d/%d predicted)\n",
		v.InDistribution.AUROC, v.InDistribution.Brier, v.InDistribution.Predicted, v.InDistribution.N)
	fmt.Fprintf(&b, "  DOMAIN HOLDOUT    AUROC %.3f  Brier %.3f  (%d/%d predicted)  <- the honest number\n\n",
		v.HoldoutMean.AUROC, v.HoldoutMean.Brier, v.HoldoutMean.Predicted, v.HoldoutMean.N)
	for _, d := range v.DomainHoldout {
		fmt.Fprintf(&b, "    %-20s AUROC %.3f  (%d/%d)\n", d.Domain, d.Score.AUROC, d.Score.Predicted, d.Score.N)
	}
	fmt.Fprintf(&b, "\n  degenerate questions  %.0f%%  (%.0f%% all-correct, %.0f%% all-fail)\n",
		v.DegenerateFrac*100, v.AllCorrectFrac*100, v.AllFailFrac*100)
	fmt.Fprintf(&b, "  distance vs disagreement  r = %+.2f\n", v.DistanceAgreementR)
	fmt.Fprintln(&b, "\n"+v.verdict())
	return b.String()
}

// verdict states what the numbers mean, because the failure mode here is
// specifically that a healthy-looking in-distribution figure hides it.
func (v ValidationReport) verdict() string {
	switch {
	case v.HoldoutMean.Predicted == 0:
		return "  VERDICT: not enough evidence to validate — profile more workers or more questions."
	case v.HoldoutMean.AUROC < 0.55:
		return fmt.Sprintf("  VERDICT: at chance out of domain (%.3f). The matrix is recognising TOPICS, not\n"+
			"  predicting difficulty — the in-distribution %.3f is measuring which subjects each\n"+
			"  worker is bad at. Routing on it will not generalise beyond the bank.",
			v.HoldoutMean.AUROC, v.InDistribution.AUROC)
	case v.InDistribution.AUROC-v.HoldoutMean.AUROC > 0.15:
		return fmt.Sprintf("  VERDICT: generalises poorly. %.3f in-distribution against %.3f held out is the\n"+
			"  gap a random split would have hidden. Usable within the bank's domains; treat\n"+
			"  predictions for anything else as unknown.", v.InDistribution.AUROC, v.HoldoutMean.AUROC)
	default:
		return fmt.Sprintf("  VERDICT: holds up out of domain (%.3f). Predictions are about difficulty rather\n"+
			"  than topic.", v.HoldoutMean.AUROC)
	}
}

// handleOutcomes serves the matrix's state and, on request, the domain-holdout
// validation.
//
// The validation is opt-in via ?validate=1 because it is O(n²) in questions —
// fine to run on demand, wrong to compute for a dashboard poll.
//
// The "backends" key holds MODEL identities now rather than worker ids, since
// that is what the matrix files evidence under. The key name is left as it is
// because no dashboard reads this endpoint — it is a curl-level diagnostic — and
// renaming it would break whatever an operator has already scripted against it
// to buy a word. See backendsWithEvidence for what the hashes are good for.
func (r *Router) handleOutcomes(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	if !r.requireAdmin(w, req) {
		return
	}
	if r.outcomes == nil {
		writeJSON(w, http.StatusOK, map[string]any{"enabled": false})
		return
	}
	body := map[string]any{
		"enabled":  true,
		"summary":  r.outcomes.String(),
		"backends": r.outcomes.backendsWithEvidence(),
	}
	if req.URL.Query().Get("validate") == "1" {
		rep := r.outcomes.validate(r.bankTopicOf)
		body["validation"] = rep
		body["verdict"] = strings.TrimSpace(rep.verdict())
	}
	writeJSON(w, http.StatusOK, body)
}
