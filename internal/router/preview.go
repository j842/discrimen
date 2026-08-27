package router

// POST /v1/route-preview — what WOULD this request do, without doing it.
//
// Debugging a routing decision used to cost a generation: send the prompt, read
// X-Llm-Route off the response, and hope the fleet was in the same state it will
// be in next time. On a fleet whose slow workers take minutes per answer that is
// an expensive way to ask a question the router can answer in milliseconds.
//
// The preview runs the REAL selection pipeline (planRoute) — same classifier,
// same hard filters, same thinking resolution, same session affinity, same
// outcome-matrix ranking — and renders the result instead of dispatching it. It
// deliberately shares planRoute rather than reimplementing the explanation: a
// preview that derives its own answer eventually explains a route the router
// doesn't take, which is worse than no preview at all.
//
// SHARING planRoute IS NOT ENOUGH ON ITS OWN, and that is the lesson of the two
// defects this file has carried. Everything planRoute does not do — the per-key
// allow-list, which the proxy applies on the very next line, and the per-worker
// prediction, which the proxy never needs to render — has to be done here too,
// or the preview is wrong in exactly the cases nobody previews until something
// has already gone strange. A restricted key was told would_serve: a worker its
// real request could not reach; and the decision half described the
// difficulty-tier ranker with three fields the live path never sets. Both looked
// like working answers.
//
// It is strictly read-only. It does not acquire a slot, record a session turn,
// feed the adapter, or contact a worker. Two side effects it can have, both of
// them reads: populating the classifier's prompt→score cache, which is exactly
// what the next real request would have done anyway, and querying the outcome
// matrix, which is a read under its own lock and changes nothing in it.

import (
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"time"
)

type previewCandidate struct {
	ID    string `json:"id"`
	Model string `json:"model"`
	// Quality is the benchmark score THIS request would be ranked on, which is
	// not always the headline one: a no-think request reads the no-think score,
	// and a thinking worker whose no-think score is unmeasured reads zero rather
	// than inheriting its mixed-mode number (see qualityFor). Reporting
	// b.Quality here regardless was how the preview came to disagree with the
	// ranker it exists to explain — rankByDifficulty and qualityFloorPreference
	// both read qualityFor, so a still-profiling thinking worker previewed as a
	// q=93 above-bar front-runner on a no-think request that the router itself
	// scored 0 and put last.
	Quality int `json:"quality"`
	// AboveBar is present ONLY when there is a bar: a quality target of zero
	// makes the test vacuous, and reporting a vacuous true for every worker on
	// every request is worse than reporting nothing, because it reads as a
	// judgement that was made. The matrix path (the live one) always carries
	// target 0 — see previewResponse.TargetQuality — so in practice this field
	// appears only on the difficulty-tier fallback.
	AboveBar *bool `json:"above_bar,omitempty"`
	// Outcome is what the outcome matrix predicts about this worker for THIS
	// prompt, which on the live path is the whole basis of the ordering. Nil
	// when there is no matrix, no classification, or no embedding to query it
	// with — the same three conditions that make planRoute fall through to the
	// tier ranker.
	Outcome         *previewOutcome `json:"outcome,omitempty"`
	ExpectedSeconds float64         `json:"expected_seconds"`
	PrefillSeconds  float64         `json:"prefill_seconds"`
	DecodeSeconds   float64         `json:"decode_seconds"`
	ObservedTPS     float64         `json:"observed_tps,omitempty"`
	ContextK        int             `json:"context_k,omitempty"`
	ActiveRequests  int             `json:"active_requests"`
	MaxConcurrency  int             `json:"max_concurrency,omitempty"`
	Full            bool            `json:"full"`
	Thinking        bool            `json:"thinking"`
	Incumbent       bool            `json:"incumbent,omitempty"`
}

// previewOutcome is what the outcome matrix predicts about ONE worker for THIS
// prompt: the numbers that actually decided the order.
//
// It exists because the preview described a router that no longer runs. The
// decision half was three fields from the difficulty-tier ranker — target_quality,
// above_bar and adapter_bias — and the matrix path sets none of them. On every
// live request target_quality was 0, so above_bar was true for every candidate
// forever, and adapter_bias reported a correction from a table nothing consults
// (see adapter.go). The endpoint whose entire job is "explain this route" was
// explaining the route the router stopped taking, with three fields that never
// varied, which is a harder failure to notice than an absent field.
//
// The numbers are re-derived here rather than carried on the plan. routePlan is
// the PROXY path's shape, and threading a per-candidate prediction through it
// would make every real request build a struct only a preview reads. A preview
// is allowed to pay for its own explanation: this costs one neighbour scan per
// candidate, which was measured at ~13ms for a seven-worker fleet against a
// saturated judged cache — the measurement that split neighboursOf out of
// predict for the routing path, and an entirely acceptable price on a debugging
// endpoint that generates nothing. Querying the matrix does not mutate it, so
// the preview stays read-only.
type previewOutcome struct {
	// Band is which of chooseByOutcome's three groups this worker landed in,
	// and is the single most useful field here. "able" is "the matrix expects
	// this worker to get prompts like yours right"; "unmeasured" is "nothing
	// resembling your prompt has been graded on it"; "unable" is "prompts like
	// yours HAVE been graded on it and it got them wrong". The bands are ranked
	// in that order, and within the able band by correctness margin then speed.
	Band string `json:"band"`
	// Correct is the similarity-weighted hit rate on nearby graded questions,
	// in [0,1]. It is the number an operator reads and the number the route
	// header reports as p=; it is NOT the number band admission compared —
	// see Supported.
	Correct float64 `json:"correct"`
	// Supported is Correct discounted for how thin the evidence behind it is
	// (prediction.supportedCorrect). Admission to the able band tests THIS
	// against outcomeCorrectFloor, so it is reported beside the raw rate: the
	// two diverge exactly where a decision is most likely to surprise, because
	// the penalty falls off as the square root of the support. One of two
	// nearby questions right is p=0.50 supporting at 0.33 and is not able; both
	// of two right is p=1.00 supporting at 0.83 and is.
	Supported float64 `json:"supported_correct"`
	// Confidence is the mean similarity of the neighbours that carried evidence
	// for this worker. Below outcomeMinConfidence the prediction is not known at
	// all, whatever Correct says — a confident 0.3 is a useful routing signal
	// and an unconfident 0.9 is not.
	Confidence float64 `json:"confidence"`
	// Support is the total similarity weight behind Correct; Observations is how
	// many graded answers contributed. Both are reported because they answer
	// different questions, and only publishing one of them was a real gap: from
	// p=1.00,n=2 an operator cannot tell one strong neighbour from a dozen weak
	// ones, and it is the weight that decides whether the rate survives
	// admission.
	Support      float64 `json:"support"`
	Observations int     `json:"observations"`
	// Judged is how many of those Observations came from the background judge
	// rather than from bench grading; the remainder are bench rows. Known gates
	// on a bench-weighted count (obsJudgeWeight), so two judged verdicts do not
	// qualify a worker on their own.
	Judged int `json:"judged"`
	// Known is prediction.known() spelled out — enough bench-weighted evidence
	// AND enough similarity. A worker can carry observations and still not be
	// known, which is the difference between the unmeasured and unable bands and
	// is not derivable from the other fields without the two thresholds.
	Known bool `json:"known"`
	// MedianLatencyMS is what this worker took on those neighbours: how long
	// this KIND of question takes it, which the live throughput estimate cannot
	// supply. Zero when no neighbour carried a duration — including every hit
	// inherited from the same model on another host, where the verdict travels
	// and the timing does not.
	MedianLatencyMS int64 `json:"median_latency_ms,omitempty"`
}

type previewResponse struct {
	Route      string   `json:"route"`
	WouldServe string   `json:"would_serve"`
	Difficulty *float64 `json:"difficulty,omitempty"`
	Reasoning  *float64 `json:"reasoning,omitempty"`
	// TargetQuality is the difficulty-tier ranker's quality floor, and is 0 on
	// every route the outcome matrix decides — which is every classified route
	// on a deployed router, because the matrix is constructed unconditionally
	// and ranks rather than declines. It is kept, and kept unconditional, because
	// `discrimen arena` reads it off this struct; read a 0 as "no quality floor
	// was in play", not as "the floor was zero".
	TargetQuality int `json:"target_quality"`
	// AdapterBias is the upward score correction the online tier adapter would
	// contribute, and it is reported ONLY when a quality target exists — which
	// is the only branch of planRoute that ever calls adapter.adjust. On the
	// matrix path the adapter is not consulted at all, and publishing a non-zero
	// bias there described a correction that was never applied to anything. See
	// the reachability note at the top of adapter.go.
	AdapterBias float64         `json:"adapter_bias,omitempty"`
	Classified  bool            `json:"classified"`
	Thinking    previewThinking `json:"thinking"`
	Job         previewJob      `json:"job"`
	Session     previewSession  `json:"session"`
	Group       *previewGroup   `json:"group,omitempty"`
	Expert      *previewExpert  `json:"expert,omitempty"`
	// Able is how many of the leading Candidates the matrix judged
	// interchangeable on correctness. Acquisition may reorder within that prefix
	// — preferring a free local worker over a paid remote one — and must not
	// reorder across it, so this is the boundary that says which of the
	// candidate ordering below is a correctness judgement and which is a cost
	// preference. Zero means no prefix is authoritative: the tier path, the
	// matrix's own bank-rate fallback, and a request the matrix chose to explore
	// on all report 0. Admin-only, like the candidate list it indexes into.
	Able int `json:"able,omitempty"`
	// Candidates and Rejected are the ADMIN half — the fleet, worker by worker.
	// Omitted entirely for a client rather than sent empty, because an empty list
	// would read as "nothing qualified", which is a different answer. See
	// handleRoutePreview.
	Candidates []previewCandidate `json:"candidates,omitempty"`
	Rejected   []rejection        `json:"rejected,omitempty"`
	Notes      []string           `json:"notes,omitempty"`
}

// previewGroup is how a named group resolved. A group that silently fell back
// to automatic routing is exactly the thing this endpoint exists to make
// visible — the request still gets an answer, so nothing else about it looks
// wrong.
type previewGroup struct {
	Name     string   `json:"name"`
	Member   string   `json:"member,omitempty"`
	Fallback bool     `json:"fallback"`
	Members  []string `json:"members,omitempty"`
}

// previewExpert is how the ensemble route resolved: whether this request gets a
// panel, and how many models would be in it. The COUNT is client-visible and the
// membership is not, which is the same line the candidate list draws — a caller
// is entitled to know that its expensive route is about to cost N generations,
// and not to a list of the operator's workers.
type previewExpert struct {
	Members  int    `json:"members"`
	Fallback string `json:"fallback,omitempty"`
}

type previewThinking struct {
	Enabled bool   `json:"enabled"`
	Source  string `json:"source"` // auto | reasoning_effort | requirements | chat_template_kwargs | off
	Effort  string `json:"effort,omitempty"`
	Hard    bool   `json:"hard_filter"`
	Soft    bool   `json:"soft_preference"`
}

type previewJob struct {
	PromptTokens int `json:"prompt_tokens"`
	OutputTokens int `json:"output_tokens"`
	DeadlineMS   int `json:"deadline_ms,omitempty"`
}

// previewSession is the session-affinity state for this request. There is no
// `outcome` field: one was declared here and never assigned by anything, so with
// omitempty it was absent from every response ever sent — an author reading the
// struct would have believed the preview reported a session outcome, and an
// author reading the JSON would have concluded the session had none.
type previewSession struct {
	Key       string `json:"key,omitempty"`
	Incumbent string `json:"incumbent,omitempty"`
	ToolLoop  bool   `json:"tool_loop"`
	Locked    bool   `json:"locked"`
}

func (r *Router) handleRoutePreview(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	// CLIENT scope, with the fleet detail held back to admin.
	//
	// The endpoint exists so a caller can understand what its OWN request would
	// do, and that answer — the decision, the classification, the thinking mode,
	// whether a named group fell back — tells them nothing about the estate they
	// could not learn from the response headers of the request itself. The
	// candidate and rejection lists are a different thing entirely: every worker
	// id, alive or not, with its quality, its load and now the matrix's opinion of
	// it, to anyone holding any client token. That is the inventory moving GET /backends to
	// admin scope was meant to stop handing out, and X-LLM-Backend-ID was closed
	// for the same reason (see refusePin).
	ident, ok := r.requireClient(w, req)
	if !ok {
		return
	}
	// Asked through adminAuthenticated rather than off ident, so the browser
	// session cookie counts as well as an admin bearer key — the same two
	// credentials requireAdmin accepts, which is what makes this good for
	// debugging from the dashboard.
	full := r.adminAuthenticated(req)
	body, err := readRequestBody(w, req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "read request: %s", err)
		return
	}
	chatReq, err := parseAndValidateChatRequest(body, r.cfg.DefaultMaxTokens)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, validationError{Message: err.Error()})
		return
	}
	// The allow-list applies here too — previewing a model a key may not call
	// would answer a question about a worker the caller cannot reach. The budget
	// does not: a preview contacts nothing and generates nothing, so a key that
	// has spent its allowance can still ask where a request WOULD have gone.
	//
	// The test is enforceKeyLimits' allow-list half, spelled out rather than
	// delegated, precisely because the budget half must not come with it. Both
	// halves have to be here: allowsModel alone REFUSED A REQUEST THE ROUTER
	// WOULD HAVE SERVED. It compares the caller's spelling to the allow-list
	// literally, so a key issued for "shared-model" that names the id of a
	// worker serving shared-model — the spelling /relay/fleet advertises, and the
	// one an upstream router uses — got 403 from the preview and 200 from
	// /v1/chat/completions. mayNameWorker is what closes that gap on the proxy
	// path, and it grants nothing new: allowsBackend already lets the auto route
	// reach that same worker.
	model := requestedModel(chatReq)
	if !ident.allowsModel(model) && !r.mayNameWorker(ident, model) {
		writeJSON(w, http.StatusForbidden, validationError{
			Message: fmt.Sprintf("this key may not use model %q", model), Param: "model",
		})
		return
	}

	// A pinned request never reaches the selection pipeline, so previewing one
	// would show a decision the router would not make. Say so rather than lie.
	if pinID := req.Header.Get("X-LLM-Backend-ID"); pinID != "" {
		writeJSON(w, http.StatusBadRequest, validationError{
			Message: fmt.Sprintf("X-LLM-Backend-ID=%q pins the worker; there is no routing decision to preview", pinID),
		})
		return
	}

	budget := callerBudget(req, chatReq)
	plan, err := r.planRoute(chatReq, budget, true)
	if err != nil {
		var unknown unknownModelError
		if errors.As(err, &unknown) {
			writeJSON(w, http.StatusNotFound, validationError{Message: err.Error(), Param: "model"})
			return
		}
		writeUnavailable(w, r.retryAfterUnavailable(), err.Error())
		return
	}
	// The allow-list narrows the CANDIDATES, not only the model name, and the
	// proxy path does this on the very next line after planRoute. The preview did
	// not, and the gap was the whole decision: allowsModel refuses nothing to a
	// caller who named nothing, "default" and an absent model field are the auto
	// route, and the auto route ranks the entire fleet. So a key issued for one
	// small local worker previewed `{"model":"default"}` and was told
	// would_serve: the largest metered endpoint in the estate — a worker the real
	// request would never reach (restrictToAllowList narrows it, or answers 503),
	// and an id that key is not entitled to learn. would_serve is in the CLIENT
	// half of this response, which is exactly the inventory that moving
	// GET /backends to admin scope was meant to stop handing out.
	//
	// restrictToAllowList is called rather than reimplemented, for the same
	// reason the whole preview shares planRoute: it also owns the "no worker this
	// key may use is available" 503, which is the honest preview of a request
	// whose allow-list leaves nothing standing. It no-ops on an "expert"-prefixed
	// plan, so the panel filter below still has to run.
	kept, ok := r.restrictToAllowList(w, ident, plan.route, plan.candidates)
	if !ok {
		return
	}
	plan.candidates = kept
	// The ensemble's panel is narrowed by the key's allow-list on the proxy path
	// (see serveExpert), so it has to be narrowed here too, or a restricted key
	// previews a panel it would never get.
	if plan.expert.active() && ident != nil && len(ident.Models) > 0 {
		plan.candidates = filterCandidates(plan.candidates, ident.allowsBackend)
	}
	writeJSON(w, http.StatusOK, r.renderPreview(chatReq, plan, budget, full))
}

// renderPreview turns a plan into the preview body. full=false is the client
// view: the same decision, without the worker-by-worker inventory behind it.
func (r *Router) renderPreview(chatReq *ChatRequest, plan *routePlan, budget time.Duration, full bool) previewResponse {
	resp := previewResponse{
		Route:         plan.route,
		TargetQuality: plan.target,
		Thinking: previewThinking{
			Enabled: plan.tr.enable,
			Source:  thinkingSource(chatReq, plan),
			Effort:  plan.tr.effort,
			Hard:    plan.tr.hardThink,
			Soft:    plan.tr.softThink,
		},
		Job: previewJob{
			PromptTokens: plan.job.promptTokens,
			OutputTokens: plan.job.outputTokens,
			DeadlineMS:   int(budget.Milliseconds()),
		},
	}
	if full {
		resp.Rejected = plan.rejected
		if resp.Rejected == nil {
			resp.Rejected = []rejection{}
		}
	}
	if plan.cl != nil {
		d, rs := plan.cl.difficulty, plan.cl.reasoning
		resp.Difficulty, resp.Reasoning, resp.Classified = &d, &rs, true
		// Only where the adapter was actually consulted. planRoute calls
		// adapter.adjust in exactly one place — the branch that also sets
		// plan.target — so a target means the bias was applied to the score this
		// route was derived from, and no target means it was not. Reporting it
		// unconditionally attributed a correction to a decision that never saw it.
		if plan.target > 0 {
			resp.AdapterBias = round3(r.adapter.adjust(d) - d)
		} else if r.outcomes != nil && len(plan.cl.vec) > 0 {
			resp.Notes = append(resp.Notes,
				"ranked by the outcome matrix: predicted correctness on graded questions like this one, then speed. "+
					"target_quality is 0 and above_bar is absent because no quality bar is involved — the candidates carry "+
					"the prediction that ordered them instead")
		}
	} else {
		resp.Notes = append(resp.Notes,
			"prompt not classified — auto-tiering and auto-thinking are OFF for this request "+
				"(no embeddings worker, classifier not ready, or no user turn to classify)")
	}
	if plan.group.name != "" {
		// The group stays in the client view, members included. It is not fleet
		// inventory: it is the operator's definition of a name the CALLER used, it
		// lists what the group asks for rather than what is registered (a member
		// that was never deployed is still listed), and without it "fell back" is a
		// verdict with nothing behind it.
		g := previewGroup{Name: plan.group.name, Member: plan.group.member, Fallback: plan.group.fallback}
		if stored, ok := r.groups.lookup(plan.group.name); ok {
			g.Members = stored.Members
		}
		resp.Group = &g
		if plan.group.fallback {
			resp.Notes = append(resp.Notes, fmt.Sprintf(
				"group %q fell back: no member is registered, healthy and past the hard filters, so the group filter was dropped and this request routes automatically",
				plan.group.name))
		}
	}
	// The ensemble, where one was asked for. A panel of N is N generations plus a
	// synthesis, which is the one thing a caller previewing this route needs to
	// know before it sends it; a fallback is the other, because the request would
	// otherwise be answered normally with nothing looking wrong.
	if plan.expert.asked {
		resp.Expert = &previewExpert{Fallback: plan.expert.fallback}
		if plan.expert.fallback != "" {
			resp.Notes = append(resp.Notes, fmt.Sprintf(
				"expert: %s — a panel cannot answer this, because N models produce N incompatible tool calls and no merge of them "+
					"is honest, so it routes automatically instead", plan.expert.fallback))
		}
	}
	if plan.session.active() {
		resp.Session = previewSession{
			Key:       strconv.FormatUint(plan.session.key, 16),
			Incumbent: plan.session.incumbent,
			ToolLoop:  plan.session.toolLoop,
			Locked:    plan.session.locked(),
		}
	}

	// The decision itself, which every caller gets: the worker their request
	// would land on is the one X-LLM-Backend-ID would name if they sent it.
	//
	// An ensemble has no such worker — it lands on every model at once — so it
	// names none and reports the panel size instead. The workers it shows are the
	// panel rather than the whole eligible set, which is what would actually run.
	shown := plan.candidates
	if plan.expert.active() {
		shown = expertMembers(plan.candidates, plan.job)
		resp.Expert.Members = len(shown)
		resp.Notes = append(resp.Notes, fmt.Sprintf(
			"expert ensemble: this request would be put to %d model(s) and their answers synthesised by the highest-quality "+
				"worker that can fit them — %d generations plus one synthesis, all charged to this request", len(shown), len(shown)))
	} else if len(plan.candidates) > 0 {
		resp.WouldServe = plan.candidates[0].ID
	}
	if full {
		resp.Able = plan.able
		explain := r.outcomeExplainer(plan)
		for _, b := range shown {
			prefill := prefillSeconds(b, plan.job.promptTokens)
			incumbent := sessionIncumbent(b, plan.job)
			if incumbent {
				prefill *= 1 - sessionPrefillDiscount
			}
			decode := 0.0
			if tps := liveTPS(b); tps > 0 {
				decode = float64(plan.job.outputTokens) / tps
			}
			// Quality and above_bar both read qualityFor rather than b.Quality, so
			// the preview reports the number the ranker used. See previewCandidate.
			quality := qualityFor(b, plan.tr.noThink)
			var aboveBar *bool
			if plan.target > 0 {
				clears := quality >= plan.target
				aboveBar = &clears
			}
			resp.Candidates = append(resp.Candidates, previewCandidate{
				ID:              b.ID,
				Model:           b.Model,
				Quality:         quality,
				AboveBar:        aboveBar,
				Outcome:         explain(b),
				ExpectedSeconds: round3(expectedLatency(b, plan.job)),
				PrefillSeconds:  round3(prefill),
				DecodeSeconds:   round3(decode),
				ObservedTPS:     round3(b.ObservedTPS),
				ContextK:        b.ContextK,
				ActiveRequests:  b.ActiveRequests,
				MaxConcurrency:  b.MaxConcurrency,
				Full:            isFull(b),
				Thinking:        b.Thinking,
				Incumbent:       incumbent,
			})
		}
	} else {
		resp.Notes = append(resp.Notes,
			"per-worker candidate and rejection detail is admin-only; this is the decision for this request")
	}
	// Everything below is about acquiring ONE worker's slot, which an ensemble
	// does not do: it takes a slot per member, waits expertSlotWait for each, and
	// leaves behind whichever is still busy rather than spilling or queueing.
	if plan.expert.active() {
		resp.Notes = append(resp.Notes, fmt.Sprintf(
			"each member waits up to %s for its own worker's slot and is dropped from the panel if none frees — a busy worker "+
				"costs its model's answer, never the whole request", expertSlotWait))
		return resp
	}
	// The ranked head is only the FIRST choice: acquisition spills past a
	// saturated front-runner, and a bounded preference may hold the request
	// briefly for a preferred worker first. Say which preference is in play so the
	// preview isn't read as a promise. Read from qualityFloorPreference rather
	// than re-derived, for the same reason the whole preview shares planRoute.
	pref := qualityFloorPreference(plan.candidates, plan.target, plan.tr.noThink, plan.able)
	// What the cost/locality ladder is allowed to range over. On the tier path
	// that is the quality bar; on the matrix path there is no bar (target is 0)
	// and the bound is the able band instead — the leading run the matrix judged
	// interchangeable on correctness. Saying only "at q>=0", or saying nothing,
	// is how the note came to describe an unbounded ladder: without the band,
	// "prefers a free LOCAL worker" reads as a licence to prefer the cheapest
	// local worker in the whole ranked list, which is precisely the behaviour
	// the able parameter was added to stop.
	bound := ""
	switch {
	case plan.target > 0:
		bound = fmt.Sprintf(" at q>=%d", plan.target)
	case plan.able > 0 && plan.able < len(plan.candidates):
		bound = fmt.Sprintf(" from the %d worker(s) the outcome matrix predicts will get this right", plan.able)
	}
	switch {
	case plan.session.locked():
		resp.Notes = append(resp.Notes, fmt.Sprintf(
			"tool loop open — acquisition waits up to %s for incumbent %q before spilling (the session lock outranks the quality floor)",
			sessionLockWait, plan.session.incumbent))
	case pref.why == "local-free":
		resp.Notes = append(resp.Notes, fmt.Sprintf(
			"acquisition prefers a free LOCAL worker%s, stepping down to a free remote one and then to a paid endpoint as each is found busy (up to %s)",
			bound, qualityFloorWait))
	case pref.why == "free-first":
		resp.Notes = append(resp.Notes, fmt.Sprintf(
			"acquisition waits up to %s for a free worker%s before spilling to a paid endpoint",
			qualityFloorWait, bound))
	case pref.why == "quality-floor":
		resp.Notes = append(resp.Notes, fmt.Sprintf(
			"acquisition waits up to %s for a worker at q>=%d before serving below the floor",
			qualityFloorWait, plan.target))
	}
	if resp.WouldServe != "" && isFull(plan.candidates[0]) {
		resp.Notes = append(resp.Notes, fmt.Sprintf(
			"%q is at its concurrency limit right now — the request would spill to the next candidate", resp.WouldServe))
	}
	return resp
}

// outcomeExplainer returns a per-worker prediction renderer for this plan, or
// one that reports nothing when the matrix could not have decided this route.
//
// The three conditions are lifted from planRoute's own gate — a matrix, a
// classification, and an embedding to query it with — rather than sniffed off
// the route string, so the preview stops explaining the matrix at exactly the
// moment the router stops asking it. The thinking flag is the one
// chooseByOutcome is called with (!noThink), because a model with reasoning on
// and the same model with it off are two different models to the matrix and
// carry separate evidence.
func (r *Router) outcomeExplainer(plan *routePlan) func(*Backend) *previewOutcome {
	if r.outcomes == nil || plan.cl == nil || len(plan.cl.vec) == 0 {
		return func(*Backend) *previewOutcome { return nil }
	}
	thinking := !plan.tr.noThink
	return func(b *Backend) *previewOutcome {
		p := r.outcomes.predict(plan.cl.vec, modelHash(b), thinking)
		// The band test mirrors chooseByOutcome's partition exactly, including
		// the strict > against the floor: a worker sitting ON the floor is not
		// able, which is the whole reason supportedCorrect exists.
		band := "unmeasured"
		if p.known() {
			band = "unable"
			if p.supportedCorrect() > outcomeCorrectFloor {
				band = "able"
			}
		}
		return &previewOutcome{
			Band:            band,
			Correct:         round3(p.Correct),
			Supported:       round3(p.supportedCorrect()),
			Confidence:      round3(p.Confidence),
			Support:         round3(p.Support),
			Observations:    p.Observations,
			Judged:          p.Judged,
			Known:           p.known(),
			MedianLatencyMS: p.MedianLatencyMS,
		}
	}
}

// thinkingSource names which rule in resolveThinking decided the thinking mode,
// so a preview explains WHY thinking is on and not merely that it is.
func thinkingSource(req *ChatRequest, plan *routePlan) string {
	if normalizeEffort(req.ReasoningEffort) != "" {
		return "reasoning_effort"
	}
	if req.Requirements != nil {
		switch normalizeThinking(req.Requirements.Thinking) {
		case "on", "off":
			return "requirements"
		}
	}
	if clientSetKwargThinking(req) {
		return "chat_template_kwargs"
	}
	if plan.cl == nil {
		return "unclassified"
	}
	return "auto"
}

func round3(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	return math.Round(v*1000) / 1000
}
