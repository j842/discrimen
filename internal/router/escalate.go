package router

// Inline escalation — repair THIS request, not just the next one in its bin.
//
// The router already detects an inadequate answer (responseInadequate) and
// already reacts to it, but only by nudging the online tier adapter: the region
// gets routed higher NEXT time, while the caller that hit the problem is handed
// the empty response. That is the right shape for a slow-moving quality signal
// and the wrong shape for a request that is still in flight and still fixable.
//
// So: when a worker returns a 2xx with NOTHING in it, re-dispatch the same
// request to a strictly better worker before replying. Cheap-first with a verify
// step and escalation on failure is the pattern the cascade routers converged on
// (NadirClaw, Maestro); this is that loop with the verifier the router already
// had. The adapter still learns — an escalation is fed to it as "this bin needed
// a better model", precisely so the repair doesn't teach it the opposite.
//
// Four deliberate boundaries:
//
//   - NON-STREAMED ONLY. Once SSE bytes are on the wire they cannot be recalled,
//     and buffering a stream to inspect it would destroy the streaming latency
//     that is the point of streaming. dispatchStreaming is untouched.
//   - EMPTY ONLY, not truncated. A length-capped answer hit the CALLER's token
//     ceiling; a bigger model runs into the same wall and bills twice for it.
//     See classifyResponse.
//   - ROUTER-CHOSEN ROUTES ONLY. A client that named a model, pinned a worker or
//     hit /debug asked for that worker specifically. Silently answering from a
//     different model would be a worse failure than the empty reply.
//   - ONE HOP. If the better worker is also empty, the original response is
//     returned. Escalation may cost one extra generation; it may never loop.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// escalateSlotWait bounds how long an escalation waits for the better worker's
// slot. Short on purpose: the caller has already paid for one failed generation,
// so a long queue here turns a repairable request into a timeout. Past it, the
// original (empty) answer is returned rather than nothing.
var escalateSlotWait = 15 * time.Second

// dispatch is the mutable state of one buffered exchange. backend and slot are
// pointers because escalation may hand both to a different worker mid-request
// and proxyToBackend's deferred unwind has to see the swap.
type dispatch struct {
	backend   **Backend
	slot      *chan struct{}
	body      []byte // the body as patched for the CURRENT backend
	raw       []byte // the client's body before any worker-specific patching
	plan      *routePlan
	chatReq   *ChatRequest
	job       jobCost
	tr        thinkingResolution
	inject    int           // max_tokens to inject when the client set none (0 = leave alone)
	budget    time.Duration // what the caller declared it will wait (0 = unknown)
	start     time.Time     // when the request arrived, for budget accounting
	log       *RequestLog
	output    io.Writer
	escalated *bool
}

// bufferedResult is one completed exchange with a worker, before anything is
// written to the client. Separating the exchange from the write is what makes
// escalation possible at all: an inadequate answer can be replaced while nothing
// is yet committed to the wire.
type bufferedResult struct {
	statusCode int
	header     http.Header
	body       []byte
	netErr     error
}

func (res bufferedResult) ok() bool {
	return res.netErr == nil && res.statusCode >= 200 && res.statusCode < 300
}

// dispatchBuffered forwards a non-streaming request, retrying once without a
// field the endpoint refused to recognise, then escalating an empty answer to a
// better worker before replying.
func (r *Router) dispatchBuffered(w http.ResponseWriter, req *http.Request, d *dispatch) {
	backend := *d.backend
	res := r.stripAndRetry(req, backend, d, r.requestBuffered(req, backend, d.body))

	if res.ok() && classifyResponse(res.body, false) == responseEmpty {
		if better, betterRes, ok := r.escalate(req, d, res); ok {
			log.Printf("escalation: %s returned an empty answer for %s → re-served by %s (q %d→%d)",
				backend.ID, d.plan.route, better.ID, backend.Quality, better.Quality)
			backend = better
			res = betterRes
			*d.escalated = true
			w.Header().Set("X-LLM-Escalated", fmt.Sprintf("%s->%s", d.log.BackendID, better.ID))
			// The session header was written for the ORIGINAL pick; restate it for the
			// worker that actually answered, or it reports "stay" on a turn that moved.
			if s := d.plan.session.outcome(better.ID); s != "" {
				w.Header().Set("X-LLM-Session", s)
			}
			d.log.BackendID = better.ID
			d.log.BackendModel = better.Model
			d.log.ObservedTPS = better.ObservedTPS
			d.log.CertifiedTPS = better.Certification.TokensPerSec
			d.log.BaselineTPS = better.BaselineTPS
			d.log.SpeedScore = speedScore(better)
		}
	}

	r.writeBuffered(w, req, backend, d, res)
}

// escalate re-runs the request on the best worker strictly better than the one
// that came back empty. Returns ok=false — leaving the original answer in place —
// whenever there is no better worker, no slot for it in time, or its answer is no
// better. It never returns something worse than what it was given.
//
// On success the caller's slot and active-request accounting have ALREADY been
// moved to the returned worker.
func (r *Router) escalate(req *http.Request, d *dispatch, orig bufferedResult) (*Backend, bufferedResult, bool) {
	if r.cfg == nil || !r.cfg.EscalateInline {
		return nil, orig, false
	}
	// Only for tiers the ROUTER picked. parseRouteScore accepts exactly the
	// "route:d=" form, which is the same gate the adapter and the judge use — a
	// named model ("model:d="), a pin or /debug all fall out here.
	if _, auto := parseRouteScore(d.plan.route); !auto {
		return nil, orig, false
	}
	from := *d.backend
	better := betterCandidates(d.plan.candidates, from, d.job)
	if len(better) == 0 {
		return nil, orig, false
	}
	// A caller that declared how long it will wait has already spent part of that
	// budget on the failed generation. Only escalate onto a worker that can still
	// finish inside what's left — otherwise the repair turns a bad answer into no
	// answer, which is the trade deadlineFilter exists to refuse.
	if d.budget > 0 {
		remaining := (d.budget - time.Since(d.start)).Seconds() * deadlineSafetyFactor
		fits := filterCandidates(better, func(b *Backend) bool {
			return expectedLatency(b, d.job) <= remaining
		})
		if len(fits) == 0 {
			log.Printf("escalation: %s answered empty but only %.1fs of the caller's budget is left — keeping it",
				from.ID, remaining)
			return nil, orig, false
		}
		better = fits
	}

	// Take the better worker's slot before giving up the current one, so a
	// saturated fleet can't leave this request holding nothing. Bounded: past the
	// grace, keep the original answer rather than queue the caller indefinitely.
	ctx, cancel := context.WithTimeout(req.Context(), escalateSlotWait)
	defer cancel()
	target, slot, _, err := r.pickAndAcquirePreferred(ctx, better, acquirePreference{})
	if err != nil {
		log.Printf("escalation: no slot on a better worker within %s — keeping %s's empty answer", escalateSlotWait, from.ID)
		return nil, orig, false
	}

	// Re-patch from the CLIENT's body, not the already-patched one: the previous
	// worker's context ceiling may have clamped max_tokens, and inheriting that
	// clamp would hand the better worker a budget shaped by a worker it is
	// replacing. The served-model rewrite differs per worker too.
	body := patchForwardedBody(d.raw, d.inject, budgetCeiling(target, d.job), d.tr.forBackend(target), target.ServedID)
	body = stripBodyFields(body, target.RejectedFields)

	r.registry.incActive(target.ID, 1)
	res := r.requestBuffered(req, target, body)
	if !res.ok() || classifyResponse(res.body, false) == responseEmpty {
		// No better off. Give the slot back and leave everything as it was.
		r.registry.incActive(target.ID, -1)
		r.registry.releaseSlot(slot)
		return nil, orig, false
	}

	// Commit: hand the caller's slot/active accounting over to the new worker so
	// proxyToBackend's deferred unwind releases the right one.
	r.registry.incActive(from.ID, -1)
	r.registry.releaseSlot(*d.slot)
	*d.backend = target
	*d.slot = slot
	d.body = body
	return target, res, true
}

// betterCandidates returns the request's candidates that are strictly higher
// quality than the worker that failed, soonest-to-finish first. Strictly higher
// on purpose: re-running on an equal-quality worker is a coin flip dressed up as
// a fix, and the whole justification for spending a second generation is that the
// replacement is measurably better at this kind of prompt.
func betterCandidates(candidates []*Backend, from *Backend, job jobCost) []*Backend {
	out := filterCandidates(candidates, func(b *Backend) bool {
		return b.ID != from.ID && b.Quality > from.Quality
	})
	if len(out) == 0 {
		return nil
	}
	// The incumbent discount is meaningless here — we are deliberately leaving the
	// worker that holds the prefix — so rank on the undiscounted job.
	job.incumbent = ""
	return rankByDifficulty(out, 0, job)
}

// requestBuffered runs one buffered exchange against a worker, including the
// 5xx/429 retry ladder, and returns the result WITHOUT writing to the client.
func (r *Router) requestBuffered(req *http.Request, backend *Backend, body []byte) bufferedResult {
	upstreamURL := upstreamChatURL(backend)
	var last bufferedResult
	totalAttempts := len(proxyRetryDelays) + 1

	for attempt := 0; attempt < totalAttempts; attempt++ {
		if attempt > 0 {
			delay := proxyRetryDelays[attempt-1]
			log.Printf("backend=%s attempt %d/%d retrying after %s (prev status=%d err=%v)",
				backend.ID, attempt+1, totalAttempts, delay, last.statusCode, last.netErr)
			select {
			case <-time.After(delay):
			case <-req.Context().Done():
				return bufferedResult{netErr: req.Context().Err()}
			}
		}

		proxyReq, err := http.NewRequestWithContext(req.Context(), http.MethodPost, upstreamURL, bytes.NewReader(body))
		if err != nil {
			return bufferedResult{netErr: err}
		}
		proxyReq.Header.Set("Content-Type", "application/json")
		if backend.APIKey != "" {
			proxyReq.Header.Set("Authorization", "Bearer "+backend.APIKey)
		} else if auth := req.Header.Get("Authorization"); auth != "" {
			proxyReq.Header.Set("Authorization", auth)
		}

		resp, err := r.client.Do(proxyReq)
		if err != nil {
			r.registry.setError(backend.ID, err.Error())
			last = bufferedResult{netErr: err}
			continue
		}
		respBody, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			r.registry.setError(backend.ID, readErr.Error())
			last = bufferedResult{netErr: readErr}
			continue
		}
		last = bufferedResult{
			statusCode: resp.StatusCode,
			header:     resp.Header.Clone(),
			body:       respBody,
		}
		if resp.StatusCode < 500 && resp.StatusCode != http.StatusTooManyRequests {
			break // success or non-retryable client error
		}
		log.Printf("backend=%s attempt %d/%d returned %d (will %s)",
			backend.ID, attempt+1, totalAttempts, resp.StatusCode,
			func() string {
				if attempt+1 < totalAttempts {
					return "retry"
				}
				return "give up"
			}())
	}
	return last
}

// writeBuffered commits a completed exchange to the client and records the
// worker-health verdict.
func (r *Router) writeBuffered(w http.ResponseWriter, req *http.Request, backend *Backend, d *dispatch, res bufferedResult) {
	if res.netErr != nil {
		// Client hangups (context canceled) aren't backend failures — see the
		// matching guard in dispatchStreaming.
		if req.Context().Err() == nil {
			r.registry.noteProxyResult(backend.ID, false)
		}
		d.log.StatusCode = http.StatusBadGateway
		d.log.Error = res.netErr.Error()
		writeJSON(w, http.StatusBadGateway, validationError{Message: res.netErr.Error()})
		return
	}
	r.registry.noteProxyResult(backend.ID, res.statusCode < 500)
	copyHeaders(w.Header(), res.header)
	setRouteHeaders(w, backend, d.plan.route, d.log)
	w.WriteHeader(res.statusCode)
	d.log.StatusCode = res.statusCode
	if _, err := w.Write(res.body); err != nil {
		d.log.Error = err.Error()
		log.Printf("proxy write failed backend=%s: %v", backend.ID, err)
		return
	}
	d.output.Write(res.body)
	// A buffered (non-streamed) response can't separate first-token latency from
	// decode, so it isn't used as a throughput sample — the streamed path and the
	// cold-start speed probe feed the decode EWMA instead.
}

// ── Unknown-parameter backstop ──────────────────────────────────────────────
//
// The router works out what an endpoint speaks — the served model id, the
// thinking dialect — and strips the fields that are only ever ours, so a strict
// provider should not normally see something it cannot parse. This catches the
// ones it does. Rather than hand a client a 400 it can do nothing about, the
// field named in the rejection is removed, the request goes again, and the
// verdict is remembered against that backend so every later request omits the
// field up front and pays nothing for it.
//
// Bounded, deliberately:
//
//   - ONE retry. A second rejection is the endpoint's answer, not a puzzle to
//     keep solving at the caller's expense.
//   - NON-STREAMING only, for the reason recorded on proxyRetryDelays: bytes are
//     already on the wire and there is no rewinding them.
//   - Only a field the request actually carries, only when the error says
//     "unrecognised" in one of the spellings servers use, and never a field that
//     carries what the caller asked for (stripUnsafeFields). Where the field
//     cannot be identified, nothing is retried.
//   - In memory. A restart re-learns each field for the price of one rejected
//     request, which is cheaper than persisting a verdict the provider's next
//     deploy could invalidate.

// unknownFieldMarkers are the ways an OpenAI-compatible server says "I do not
// know this field": OpenAI's own wording, FastAPI/pydantic's (what a vLLM
// validation error reads like), and the JSON-schema phrasing gateways emit.
// The list is a gate, not a parser — its job is to keep a merely INVALID value
// ("temperature must be <= 2") from being mistaken for an unknown field and
// silently dropped.
var unknownFieldMarkers = []string{
	"unrecognized", "unrecognised", "unknown", "unsupported",
	"not supported", "unexpected", "extra field", "extra_forbidden",
	"not permitted", "not allowed", "additionalproperties", "no longer supported",
}

// stripUnsafeFields are never removed, whatever an endpoint says about them:
// without any one of them the request stops meaning what the caller asked for.
// The budget fields are the sharp ones — silently dropping max_tokens on a
// metered endpoint turns a 400 the client can see into a bill it cannot.
var stripUnsafeFields = map[string]bool{
	"messages": true, "model": true, "stream": true,
	"tools": true, "tool_choice": true,
	"max_tokens": true, "max_completion_tokens": true,
}

// stripAndRetry re-runs, once and without the offending field, a request the
// endpoint rejected for naming a field it does not recognise. It returns the
// original result untouched whenever there is nothing safe to do.
func (r *Router) stripAndRetry(req *http.Request, backend *Backend, d *dispatch, res bufferedResult) bufferedResult {
	if res.netErr != nil {
		return res
	}
	field, ok := rejectedField(res.statusCode, res.body, d.body)
	if !ok {
		return res
	}
	stripped := stripBodyFields(d.body, []string{field})
	if bytes.Equal(stripped, d.body) {
		return res
	}
	if r.registry.noteRejectedField(backend.ID, field) {
		log.Printf("backend=%s refused %q as an unrecognised request field — retrying without it, and omitting it from later requests (%s)",
			backend.ID, field, truncate(string(res.body), 160))
	}
	d.body = stripped
	retried := r.requestBuffered(req, backend, stripped)
	if retried.netErr != nil {
		// The retry failed to reach the worker at all. The original rejection at
		// least tells the caller what was wrong with the request; a transport
		// error tells them nothing.
		return res
	}
	return retried
}

// rejectedField reports which request field an endpoint refused as
// unrecognised, or ok=false when the router cannot tell — in which case nothing
// is retried.
//
// Conservative in three independent ways: the status has to be a validation
// reject, the message has to actually say "unrecognised" in one of its
// spellings, and the name has to match a top-level key the request really
// carries. That last test is what makes scanning free text safe — a message
// naming no field of ours identifies nothing, and one naming several is
// ambiguous rather than actionable.
func rejectedField(status int, errBody, reqBody []byte) (string, bool) {
	if status != http.StatusBadRequest && status != http.StatusUnprocessableEntity {
		return "", false
	}
	text := strings.ToLower(string(errBody))
	marked := false
	for _, m := range unknownFieldMarkers {
		if strings.Contains(text, m) {
			marked = true
			break
		}
	}
	if !marked {
		return "", false
	}
	var request map[string]json.RawMessage
	if err := json.Unmarshal(reqBody, &request); err != nil {
		return "", false
	}
	// The machine-readable answer first, where the endpoint bothered to give one.
	if p := rejectParam(errBody); strippableField(request, p) {
		return p, true
	}
	// Otherwise read the name out of the prose, believing only a name the request
	// actually carries.
	found := ""
	for field := range request {
		if !strippableField(request, field) || !strings.Contains(text, strings.ToLower(field)) {
			continue
		}
		if found != "" {
			return "", false // two candidates: picking one is a guess, not a diagnosis
		}
		found = field
	}
	return found, found != ""
}

// rejectParam pulls the offending field name out of the two machine-readable
// shapes servers use: OpenAI's error.param, and pydantic's detail[].loc, which
// is what a vLLM request-validation error looks like.
func rejectParam(errBody []byte) string {
	var shape struct {
		Error struct {
			Param string `json:"param"`
		} `json:"error"`
		Detail []struct {
			Loc []any `json:"loc"`
		} `json:"detail"`
	}
	if err := json.Unmarshal(errBody, &shape); err != nil {
		return ""
	}
	if p := strings.TrimSpace(shape.Error.Param); p != "" {
		return p
	}
	for _, d := range shape.Detail {
		// loc is a path from the request root, e.g. ["body","chat_template_kwargs"];
		// the leaf is the field.
		if n := len(d.Loc); n > 0 {
			if s, ok := d.Loc[n-1].(string); ok && s != "body" {
				return s
			}
		}
	}
	return ""
}

// strippableField reports whether field is one the request carries and the
// router is willing to drop.
func strippableField(request map[string]json.RawMessage, field string) bool {
	if field == "" || stripUnsafeFields[field] {
		return false
	}
	_, present := request[field]
	return present
}
