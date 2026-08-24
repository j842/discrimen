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
// Five deliberate boundaries:
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
//   - NEVER MID-TOOL-LOOP. See the guard in escalate: an open loop is the one
//     place where a second opinion cannot be asked for at all.
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
	// Never inside an open tool loop. Moving a mid-loop turn to another model hands
	// it a tool result whose matching tool call it never emitted (session.go), which
	// the receiving model usually refuses outright on the orphan tool_call_id — so
	// the caller is billed for a second generation, on a paid endpoint, and still
	// gets the empty answer back. Acquisition already spends sessionLockWait
	// defending this; escalating past it here would undo that in one hop.
	if d.plan.session.locked() {
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
	body = r.stripLearned(body, d.raw, target.ID)

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
	return rankByDifficulty(out, 0, job, false)
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
		// The backend's own credential and nothing else. proxyToBackend drops the
		// caller's header before it gets here, so the old "else relay the client's
		// Authorization" fallback was already dead — expressing the rule in one place
		// (setBackendCredential) rather than relying on that Del to neutralise a
		// second copy of it is the point.
		setBackendCredential(proxyReq, backend)
		r.stampRelayChain(proxyReq, req, backend)

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
// ones it does. Rather than hand a client a 400 caused by an addition of OURS,
// the injected field named in the rejection is removed, the request goes again,
// and — once that has been shown to work — the verdict is remembered against
// that backend so later requests omit the field up front and pay nothing for it.
//
// Bounded, deliberately:
//
//   - ONLY THE ROUTER'S OWN ADDITIONS (routerInjectedFields). This is the whole
//     design intent: a backstop for an endpoint refusing something the ROUTER
//     put in the body. It was never a negotiation over the caller's request.
//   - ONE retry. A second rejection is the endpoint's answer, not a puzzle to
//     keep solving at the caller's expense.
//   - NON-STREAMING only, for the reason recorded on proxyRetryDelays: bytes are
//     already on the wire and there is no rewinding them. Which is also why a
//     verdict learned here must be one that cannot change what an answer MEANS:
//     it is applied up front to every later request, streamed ones included,
//     where none of this logic runs and nothing can notice it went wrong.
//   - Only a field the request actually carries, and only when the error says
//     "unrecognised" in one of the spellings servers use. Where the field cannot
//     be identified, nothing is retried.
//   - LEARNED ONLY FROM A RETRY THAT WORKED. Recording the verdict before the
//     retry is validated blacklists a field on the strength of a guess, and the
//     blacklist outlives the request that made it.
//   - EXPIRING (rejectedFieldTTL) and in memory. A provider's next deploy can
//     start accepting a field it used to refuse, and nothing tells us; the cost
//     of finding out is one rejected-and-retried request per field per TTL.

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

// routerInjectedFields are the ONLY fields this backstop may drop: the ones
// patchForwardedBody adds on the router's own initiative. Everything else in the
// body came from the caller, and dropping any of it turns a 400 they can see
// into a silent change of meaning they cannot — a json_schema response_format
// removed on the way to a model that only does json_object gets the caller
// free-form prose with a 200, and every later request to that backend, streamed
// ones included, is quietly answered the same way.
//
// The list is short because the router adds little: the thinking gate, in
// whichever of the two spellings the endpoint was measured to honour. Its
// absence is a difference of degree — the model reasons or it doesn't — not a
// difference in what was asked for.
//
// max_tokens is injectable and deliberately NOT here. It is the sharp one:
// silently dropping a budget on a metered endpoint turns a 400 the client can
// see into a bill it cannot. A model, messages or tools field is not here
// either, and could not be — none of them is ever ours.
var routerInjectedFields = map[string]bool{
	"chat_template_kwargs": true,
	"reasoning_effort":     true,
}

// rejectedFieldTTL is how long a learned verdict stands before the field is sent
// again to see whether it is still refused. Not forever: providers redeploy, and
// nothing announces it, so a verdict learned once would otherwise strip a field
// from every request for the rest of the process's life — including, for a row
// an operator added by hand, across every keepalive, since only changed
// registration content clears the set. Re-testing costs one rejected-and-retried
// request per field per TTL, which is the same price a restart already pays.
var rejectedFieldTTL = 30 * time.Minute

// stripAndRetry re-runs, once and without the offending field, a request the
// endpoint rejected for naming a field the ROUTER injected and it does not
// recognise. It returns the original result untouched whenever there is nothing
// safe to do.
func (r *Router) stripAndRetry(req *http.Request, backend *Backend, d *dispatch, res bufferedResult) bufferedResult {
	if res.netErr != nil {
		return res
	}
	field, ok := rejectedField(res.statusCode, res.body, d.body, d.raw)
	if !ok {
		return res
	}
	stripped := stripBodyFields(d.body, []string{field})
	if bytes.Equal(stripped, d.body) {
		return res
	}
	retried := r.requestBuffered(req, backend, stripped)
	if retried.netErr != nil {
		// The retry failed to reach the worker at all. The original rejection at
		// least tells the caller what was wrong with the request; a transport
		// error tells them nothing.
		return res
	}
	if !retried.ok() {
		// Still refused. The field was not the problem, or not the only one — so
		// there is nothing here worth remembering, and the caller gets the
		// endpoint's own answer rather than a blacklist built on a wrong guess.
		return retried
	}
	// Only now is the verdict evidence rather than a hypothesis.
	d.body = stripped
	if r.registry.noteRejectedField(backend.ID, field) {
		log.Printf("backend=%s refused %q as an unrecognised request field and accepted the request without it — omitting it from later requests for %s (%s)",
			backend.ID, field, rejectedFieldTTL, truncate(string(res.body), 160))
	}
	return retried
}

// stripLearned removes the fields this endpoint has already been shown to refuse
// (rejectedField/stripAndRetry) from a body about to be forwarded to it.
//
// A field the CLIENT sent is never dropped, whatever was learned: the verdict
// was learned from a request where the router had injected that field, and the
// same name arriving from the caller is a different fact about a different
// request. Sending it and getting a 400 is a failure the caller can see and act
// on; dropping it is one they cannot.
//
// Free when nothing has been learned, which is every backend almost all of the
// time: one map read, no parse of what may be a multi-MB vision body.
func (r *Router) stripLearned(body, clientBody []byte, backendID string) []byte {
	learned := r.registry.rejectedFields(backendID)
	if len(learned) == 0 {
		return body
	}
	var client map[string]json.RawMessage
	if err := json.Unmarshal(clientBody, &client); err != nil {
		return body // can't tell what the caller sent, so drop nothing
	}
	drop := learned[:0] // learned is our own copy
	for _, field := range learned {
		if _, fromClient := client[field]; !fromClient {
			drop = append(drop, field)
		}
	}
	return stripBodyFields(body, drop)
}

// rejectedField reports which INJECTED request field an endpoint refused as
// unrecognised, or ok=false when the router cannot tell — in which case nothing
// is retried. reqBody is what was sent; clientBody is what the caller sent, and
// the difference between them is exactly the set this may name.
//
// Conservative in four independent ways: the status has to be a validation
// reject, the message has to actually say "unrecognised" in one of its
// spellings, the name has to match a top-level key the request really carries,
// and that key must be one the router put there. The last two are what make
// scanning free text safe — a message naming no field of ours identifies
// nothing, and one naming several is ambiguous rather than actionable.
func rejectedField(status int, errBody, reqBody, clientBody []byte) (string, bool) {
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
	var request, client map[string]json.RawMessage
	if err := json.Unmarshal(reqBody, &request); err != nil {
		return "", false
	}
	// A client body that won't parse means the router cannot tell its own
	// additions from the caller's fields, and the safe answer to that is to strip
	// nothing. (It parsed once already — parseAndValidateChatRequest — so this is
	// belt and braces.)
	if err := json.Unmarshal(clientBody, &client); err != nil {
		return "", false
	}
	// The machine-readable answer first, where the endpoint bothered to give one.
	if p := rejectParam(errBody); strippableField(request, client, p) {
		return p, true
	}
	// Otherwise read the name out of the prose, believing only a name the request
	// actually carries.
	found := ""
	for field := range request {
		if !strippableField(request, client, field) || !strings.Contains(text, strings.ToLower(field)) {
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

// strippableField reports whether field is one the ROUTER put in this request
// and is therefore willing to drop: an injectable field, carried by the body
// that was sent, and absent from the body the caller sent.
//
// The last test is what makes the first safe. chat_template_kwargs is the
// router's gate when the router wrote it and the client's escape hatch when the
// client did (see mergeThinkingKwargs), and the two are indistinguishable by the
// time the endpoint complains about one.
func strippableField(request, client map[string]json.RawMessage, field string) bool {
	if field == "" || !routerInjectedFields[field] {
		return false
	}
	if _, present := request[field]; !present {
		return false
	}
	_, fromClient := client[field]
	return !fromClient
}
