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
	Route         string             `json:"route"`
	WouldServe    string             `json:"would_serve"`
	TargetQuality int                `json:"target_quality"`
	Difficulty    *float64           `json:"difficulty,omitempty"`
	Reasoning     *float64           `json:"reasoning,omitempty"`
	AdapterBias   float64            `json:"adapter_bias"`
	Classified    bool               `json:"classified"`
	Thinking      previewThinking    `json:"thinking"`
	Job           previewJob         `json:"job"`
	Session       previewSession     `json:"session"`
	Candidates    []previewCandidate `json:"candidates"`
	Rejected      []rejection        `json:"rejected"`
	Notes         []string           `json:"notes,omitempty"`
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
	if !authorizedAsClient(req, r.cfg.ClientTokens) {
		unauthorized(w)
		return
	}
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
	writeJSON(w, http.StatusOK, r.renderPreview(chatReq, plan, budget))
}

func (r *Router) renderPreview(chatReq *ChatRequest, plan *routePlan, budget time.Duration) previewResponse {
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
		Rejected: plan.rejected,
	}
	if resp.Rejected == nil {
		resp.Rejected = []rejection{}
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
	if plan.session.active() {
		resp.Session = previewSession{
			Key:       strconv.FormatUint(plan.session.key, 16),
			Incumbent: plan.session.incumbent,
			ToolLoop:  plan.session.toolLoop,
			Locked:    plan.session.locked(),
		}
	}

	for _, b := range plan.candidates {
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
	if len(resp.Candidates) > 0 {
		resp.WouldServe = resp.Candidates[0].ID
	}
	// The ranked head is only the FIRST choice: acquisition spills past a
	// saturated front-runner, and a bounded preference may hold the request
	// briefly for a preferred worker first. Say which preference is in play so the
	// preview isn't read as a promise.
	switch {
	case plan.session.locked():
		resp.Notes = append(resp.Notes, fmt.Sprintf(
			"tool loop open — acquisition waits up to %s for incumbent %q before spilling (the session lock outranks the quality floor)",
			sessionLockWait, plan.session.incumbent))
	case plan.target > 0:
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
