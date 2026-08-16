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
// completion-time ranking — and renders the result instead of dispatching it. It
// deliberately shares planRoute rather than reimplementing the explanation: a
// preview that derives its own answer eventually explains a route the router
// doesn't take, which is worse than no preview at all.
//
// It is strictly read-only. It does not acquire a slot, record a session turn,
// feed the adapter, or contact a worker. The one side effect it can have is
// populating the classifier's prompt→score cache, which is exactly what the next
// real request would have done anyway.

import (
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"time"
)

type previewCandidate struct {
	ID              string  `json:"id"`
	Model           string  `json:"model"`
	Quality         int     `json:"quality"`
	AboveBar        bool    `json:"above_bar"`
	ExpectedSeconds float64 `json:"expected_seconds"`
	PrefillSeconds  float64 `json:"prefill_seconds"`
	DecodeSeconds   float64 `json:"decode_seconds"`
	ObservedTPS     float64 `json:"observed_tps,omitempty"`
	ContextK        int     `json:"context_k,omitempty"`
	ActiveRequests  int     `json:"active_requests"`
	MaxConcurrency  int     `json:"max_concurrency,omitempty"`
	Full            bool    `json:"full"`
	Thinking        bool    `json:"thinking"`
	Incumbent       bool    `json:"incumbent,omitempty"`
}

type previewResponse struct {
	Route         string          `json:"route"`
	WouldServe    string          `json:"would_serve"`
	TargetQuality int             `json:"target_quality"`
	Difficulty    *float64        `json:"difficulty,omitempty"`
	Reasoning     *float64        `json:"reasoning,omitempty"`
	AdapterBias   float64         `json:"adapter_bias"`
	Classified    bool            `json:"classified"`
	Thinking      previewThinking `json:"thinking"`
	Job           previewJob      `json:"job"`
	Session       previewSession  `json:"session"`
	Group         *previewGroup   `json:"group,omitempty"`
	Expert        *previewExpert  `json:"expert,omitempty"`
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

type previewSession struct {
	Key       string `json:"key,omitempty"`
	Incumbent string `json:"incumbent,omitempty"`
	ToolLoop  bool   `json:"tool_loop"`
	Locked    bool   `json:"locked"`
	Outcome   string `json:"outcome,omitempty"`
}

func (r *Router) handleRoutePreview(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	// CLIENT scope, with the fleet detail held back to admin.
	//
	// The endpoint exists so a caller can understand what its OWN request would
	// do, and that answer — the decision, the classification, the tier, the
	// thinking mode, whether a named group fell back — tells them nothing about
	// the estate they could not learn from the response headers of the request
	// itself. The candidate and rejection lists are a different thing entirely:
	// every worker id, alive or not, with its quality and its load, to anyone
	// holding any client token. That is the inventory moving GET /backends to
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
	// does not: a preview contacts nothing and generates nothing.
	if !ident.allowsModel(requestedModel(chatReq)) {
		writeJSON(w, http.StatusForbidden, validationError{
			Message: fmt.Sprintf("this key may not use model %q", requestedModel(chatReq)), Param: "model",
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
		resp.AdapterBias = round3(r.adapter.adjust(d) - d)
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
			resp.Candidates = append(resp.Candidates, previewCandidate{
				ID:              b.ID,
				Model:           b.Model,
				Quality:         b.Quality,
				AboveBar:        plan.target <= 0 || b.Quality >= plan.target,
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
	pref := qualityFloorPreference(plan.candidates, plan.target)
	switch {
	case plan.session.locked():
		resp.Notes = append(resp.Notes, fmt.Sprintf(
			"tool loop open — acquisition waits up to %s for incumbent %q before spilling (the session lock outranks the quality floor)",
			sessionLockWait, plan.session.incumbent))
	case pref.why == "free-first":
		bar := ""
		if plan.target > 0 {
			bar = fmt.Sprintf(" at q>=%d", plan.target)
		}
		resp.Notes = append(resp.Notes, fmt.Sprintf(
			"acquisition waits up to %s for a free worker%s before spilling to a paid endpoint",
			qualityFloorWait, bar))
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
