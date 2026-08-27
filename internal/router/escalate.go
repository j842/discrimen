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
	"strconv"
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

	// FIRST attempt with no delay ladder. The ladder exists for a worker that is
	// briefly loading or saturated, and sleeping is only the right answer when
	// there is nowhere else to go — while it sleeps, the caller's slot on the
	// FAILING worker stays held (releaseSlot is deferred in proxyToBackend), so
	// 17 seconds of ladder pins a slot on a broken worker while healthy ones sit
	// idle. Try somewhere else first; the ladder is the fallback, not the reflex.
	res := r.stripAndRetry(req, backend, d, r.requestBufferedWithDelays(req, backend, d.body, nil, d.remainingBudget()))

	// A 5xx with nothing written to the client is a ROUTING failure, not an
	// answer: the caller asked the router to pick a worker, and the one it picked
	// could not respond. Moving to the next candidate is the same decision the
	// router already made once, with better information. Nothing has been written
	// yet on this path, so it is invisible to the caller.
	tried := map[string]bool{backend.ID: true}
	if !res.ok() && retryableFailure(res) {
		if other, next, ok := r.failover(req, d, res, tried); ok {
			backend = other
			res = next
			w.Header().Set("X-LLM-Failover", fmt.Sprintf("%s->%s", d.log.BackendID, other.ID))
			if s := d.plan.session.outcome(other.ID); s != "" {
				w.Header().Set("X-LLM-Session", s)
			}
			d.log.BackendID = other.ID
			d.log.BackendModel = other.Model
			d.log.ObservedTPS = other.ObservedTPS
			d.log.CertifiedTPS = other.Certification.TokensPerSec
			d.log.BaselineTPS = other.BaselineTPS
			d.log.SpeedScore = speedScore(other)
		}
	}

	// Only now, with every alternative exhausted, is waiting the best available
	// move: the worker may simply be loading, and a slot held on it is no longer
	// denying the request a better home.
	if !res.ok() && retryableFailure(res) && len(nextCandidates(d.plan, tried)) == 0 {
		res = r.stripAndRetry(req, backend, d, r.requestBufferedWithDelays(req, backend, d.body, proxyRetryDelays, d.remainingBudget()))
	}

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
	// Only where the ROUTER picked the worker. Structural: this used to parse the
	// route string for a "route:d=" prefix, which quietly stopped matching the
	// moment a second auto branch emitted a different shape.
	if !d.plan.auto {
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
	if fits, ok := r.affordable(d, better, from); ok {
		better = fits
	} else {
		return nil, orig, false
	}
	// An escalation is only worth a second generation if the replacement is
	// BETTER, so a result that is merely non-empty does not commit.
	return r.redispatch(req, d, orig, better, func(res bufferedResult) bool {
		return res.ok() && classifyResponse(res.body, false) != responseEmpty
	}, nil)
}

// affordable narrows candidates to those that can still finish inside whatever
// remains of the caller's declared budget.
func (r *Router) affordable(d *dispatch, cands []*Backend, from *Backend) ([]*Backend, bool) {
	if d.budget <= 0 {
		return cands, true
	}
	remaining := (d.budget - time.Since(d.start)).Seconds() * deadlineSafetyFactor
	fits := filterCandidates(cands, func(b *Backend) bool {
		return expectedLatency(b, d.job) <= remaining
	})
	if len(fits) == 0 {
		log.Printf("redispatch: only %.1fs of the caller's budget is left after %s — not moving",
			remaining, from.ID)
		return nil, false
	}
	return fits, true
}

// redispatch is the MECHANICS of moving a request to another worker, with the
// policy left to the caller.
//
// Two callers with different policies share it: escalate() moves a request whose
// answer came back empty and only commits to something better, while failover()
// moves one whose worker returned a 5xx and commits to any worker that answers
// at all. The mechanics are identical and were fiddly enough to be worth having
// in exactly one place — the ordering below is load-bearing.
//
// On success the caller's slot and active-request accounting have ALREADY been
// moved to the returned worker.
func (r *Router) redispatch(req *http.Request, d *dispatch, orig bufferedResult,
	candidates []*Backend, accept func(bufferedResult) bool, tried map[string]bool) (*Backend, bufferedResult, bool) {
	from := *d.backend
	// Take the new worker's slot BEFORE giving up the current one, so a saturated
	// fleet cannot leave this request holding nothing. Bounded: past the grace,
	// keep what we have rather than queue the caller indefinitely.
	ctx, cancel := context.WithTimeout(req.Context(), escalateSlotWait)
	defer cancel()
	target, slot, _, err := r.pickAndAcquirePreferred(ctx, candidates, acquirePreference{})
	if err != nil {
		log.Printf("redispatch: no slot on an alternative within %s — keeping %s", escalateSlotWait, from.ID)
		// Nothing was contacted, so nothing has been ruled out. Retiring the whole
		// slate here is what stranded healthy workers behind a saturated one.
		return nil, orig, false
	}
	// Exactly one candidate has now had its chance.
	if tried != nil {
		tried[target.ID] = true
	}

	// Re-patch from the CLIENT's body, not the already-patched one: the previous
	// worker's context ceiling may have clamped max_tokens, and inheriting that
	// clamp would hand the new worker a budget shaped by a worker it is replacing.
	// The served-model rewrite differs per worker too.
	body := patchForwardedBody(d.raw, d.inject, budgetCeiling(target, d.job), d.tr.forBackend(target), target.ServedID)
	body = r.stripLearned(body, d.raw, target.ID)

	r.registry.incActive(target.ID, 1)
	// No delay ladder on the first attempt against a fresh worker: the point of
	// moving is that this one has not just failed.
	res := r.requestBufferedWithDelays(req, target, body, nil, d.remainingBudget())
	if !accept(res) {
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

// nextCandidates is the request's candidates minus the workers already tried,
// order preserved.
//
// Order is the whole value: plan.candidates arrives from the outcome matrix as
// "workers expected to answer this prompt correctly, fastest first", so the next
// one is already the right next one. Under the quality scalar this needed a
// predicate to find something BETTER; it no longer does, and betterCandidates'
// strict `Quality >` is now inconsistent with how routing actually ranks.
func nextCandidates(plan *routePlan, tried map[string]bool) []*Backend {
	if plan == nil {
		return nil
	}
	return filterCandidates(plan.candidates, func(b *Backend) bool { return !tried[b.ID] })
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
	return r.requestBufferedWithDelays(req, backend, body, proxyRetryDelays, 0)
}

// requestBufferedWithDelays is requestBuffered with the retry ladder supplied by
// the caller, so a first attempt against a FRESH worker can skip it entirely.
//
// The ladder exists for a worker that is briefly loading or briefly saturated,
// and sleeping is the right answer only when there is nowhere else to go. It
// costs more than it looks: the caller's slot stays held for the whole ladder
// (releaseSlot is deferred in proxyToBackend), so 17 seconds of sleep pins a slot
// on a worker that is failing while other workers sit idle. Failover tries the
// next candidate first with nil delays, and falls back to the ladder only once
// candidates are exhausted.
// budgetLeft is what remains of the CALLER's declared deadline, or 0 when they
// declared none. Passed explicitly because req.Context() has no deadline —
// net/http never sets one, there is no TimeoutHandler in the serving path, and
// the real budget arrives in a header — so the context check this replaced could
// never fire. With a 100ms declared deadline and a three-rung ladder it slept
// 1.2s; in production, 17s against any shorter budget.
func (r *Router) requestBufferedWithDelays(req *http.Request, backend *Backend, body []byte, delays []time.Duration, budgetLeft time.Duration) bufferedResult {
	deadline := time.Time{}
	if budgetLeft > 0 {
		deadline = time.Now().Add(budgetLeft)
	}
	upstreamURL := upstreamChatURL(backend)
	var last bufferedResult
	totalAttempts := len(delays) + 1

	for attempt := 0; attempt < totalAttempts; attempt++ {
		if attempt > 0 {
			delay := delays[attempt-1]
			// An upstream that said WHEN to come back knows better than the ladder.
			// This matters for relay rows and paid providers, where 503 means "busy,
			// try again in N" rather than "broken".
			if hint := retryAfterHint(last.header); hint > 0 && hint > delay {
				// Capped. A rate-limited provider answering "Retry-After: 60" would
				// otherwise make the router sleep a minute per rung, twice, holding
				// the caller's slot throughout — the hint is advice about the
				// upstream, not permission to abandon the caller.
				if hint > maxRetryAfterWait {
					hint = maxRetryAfterWait
				}
				delay = hint
			}
			// Never sleep past the caller's own deadline: waiting out a budget the
			// router just worked to honour converts a slow answer into no answer.
			if !deadline.IsZero() {
				left := time.Until(deadline)
				if left <= 0 {
					return last
				}
				if delay > left {
					delay = left
				}
			}
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
	// Health is judged on the ANSWER, not just the status line. A worker
	// returning a zero-byte or unparseable 200 was scored a success, which reset
	// its consecutive-failure run — so a worker two strikes from ejection could
	// launder itself clean by failing in this particular way, forever.
	empty := classifyResponse(res.body, false) == responseEmpty
	r.registry.noteProxyResult(backend.ID, res.statusCode < 500 && !empty)
	if empty && res.statusCode < 500 {
		r.registry.setError(backend.ID, fmt.Sprintf("HTTP %d with no usable answer (%d bytes)",
			res.statusCode, len(res.body)))
	}
	// A 5xx that reaches the client has survived failover, so record WHY on the
	// worker: /backends otherwise shows a failing worker with an empty LastError,
	// which reads as healthy. The body's first line is usually the whole story
	// ("Loading model", "CUDA out of memory").
	if res.statusCode >= 500 {
		r.registry.setError(backend.ID, upstreamErrorSnippet(res.statusCode, res.body))
	}
	copyHeaders(w.Header(), res.header)
	setRouteHeaders(w, backend, d.plan.route, d.log)
	r.backfillRetryAfter(w.Header(), res.statusCode)
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

// ── failover: a 5xx with nothing written is a routing failure ──────────────

// retryableFailure reports whether a buffered result is worth trying elsewhere.
//
// A 4xx is NOT: the request itself is wrong, and every worker will say the same.
// stripAndRetry owns the one 4xx worth retrying — an endpoint refusing a field
// it does not recognise — and it retries on the SAME worker, which is correct
// because the field, not the worker, is the problem.
func retryableFailure(res bufferedResult) bool {
	if res.netErr != nil {
		return true
	}
	return res.statusCode >= 500 || res.statusCode == http.StatusTooManyRequests
}

// maxFailoverHops bounds how many workers one request may be moved across.
// Two: enough to survive a worker that is down or loading, few enough that a
// fleet-wide outage fails fast rather than walking every worker in turn while
// the caller waits.
const maxFailoverHops = 2

// failover moves a request whose worker could not answer at all.
//
// The policy differs from escalate() in one way that matters: it accepts an
// EQUAL worker. Escalation exists because an answer was bad, so only a better
// worker justifies the second generation; here there was no answer, so any
// worker that responds is an improvement. plan.candidates is already ordered
// "expected to be correct, fastest first", so the next one is the right one.
// tried is shared with the caller and MUTATED: it is what tells dispatchBuffered
// whether every candidate has been exhausted, and therefore whether waiting on
// the ladder is the last remaining move or merely the laziest one.
func (r *Router) failover(req *http.Request, d *dispatch, res bufferedResult, tried map[string]bool) (*Backend, bufferedResult, bool) {
	// Only where the ROUTER chose the worker. A pin or a named model is an
	// instruction, and silently answering from somewhere else would break the
	// guarantee the caller asked for — they would rather have the error.
	if !d.plan.auto {
		return nil, res, false
	}
	// Never inside an open tool loop, for the same reason escalation is not: the
	// receiving model gets a tool result whose matching call it never emitted.
	if d.plan.session.locked() {
		return nil, res, false
	}
	current := res
	for hop := 0; hop < maxFailoverHops; hop++ {
		cands := nextCandidates(d.plan, tried)
		if len(cands) == 0 {
			break
		}
		if fits, ok := r.affordable(d, cands, *d.backend); ok {
			cands = fits
		} else {
			break
		}
		from := (*d.backend).ID
		// Accept anything that ANSWERS. An empty reply is escalate()'s business
		// and is handled after this returns; here the bar is simply "responded".
		// Mark only the worker redispatch ACTUALLY tried. Marking the whole slate
		// stranded healthy workers: redispatch attempts exactly one of the
		// candidates it is handed, so marking all of them retired the rest
		// unexamined. Measured with three workers (two down, one healthy): the
		// healthy one was never contacted and the client got a 503 while it sat
		// idle. It also made maxFailoverHops dead — the second hop always found
		// nothing left — and then dropped through to the 17-second ladder on the
		// original broken worker, the exact pathology failover exists to avoid.
		target, next, ok := r.redispatch(req, d, current, cands, func(rr bufferedResult) bool {
			return rr.ok()
		}, tried)
		if !ok {
			continue
		}
		// The worker that failed must still be judged for it. Only the worker that
		// ANSWERS reaches writeBuffered, so before failover existed a 5xx was
		// recorded there — and afterwards it was recorded nowhere. Measured: 20
		// consecutive 503s, all failed over, left proxyFailures=0, Healthy=true
		// and an empty LastError, so the breaker could never trip while one other
		// candidate was up and /backends showed a wedged worker as healthy.
		r.noteFailedAttempt(from, current)
		log.Printf("failover: %s returned %s for %s → re-served by %s (hop %d)",
			from, describeFailure(current), d.plan.route, target.ID, hop+1)
		return target, next, true
	}
	return nil, current, false
}

func describeFailure(res bufferedResult) string {
	if res.netErr != nil {
		return res.netErr.Error()
	}
	return fmt.Sprintf("HTTP %d", res.statusCode)
}

// retryAfterHint reads an upstream's own Retry-After, in seconds or as an HTTP
// date. Zero when absent or unparseable.
func retryAfterHint(h http.Header) time.Duration {
	if h == nil {
		return 0
	}
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil && secs >= 0 {
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

// backfillRetryAfter restores the invariant that a 503 from this router always
// tells the caller when to come back. An upstream 503 relayed verbatim often
// carries no hint, and a client that cannot tell "retry shortly" from "give up"
// retries immediately and makes the saturation worse.
func (r *Router) backfillRetryAfter(dst http.Header, status int) {
	if status != http.StatusServiceUnavailable || dst.Get("Retry-After") != "" {
		return
	}
	if after := r.retryAfterUnavailable(); after > 0 {
		dst.Set("Retry-After", strconv.Itoa(int(after.Seconds())))
	}
}

// upstreamErrorSnippet renders a failed upstream response for the worker's
// LastError, which is what an operator reads on /backends.
//
// Bounded and single-line: the body may be an HTML error page or a megabyte of
// JSON, and the useful part is almost always at the front — "Loading model",
// "CUDA out of memory", "model not found".
func upstreamErrorSnippet(status int, body []byte) string {
	msg := strings.TrimSpace(string(body))
	if len(msg) > upstreamSnippetMax {
		msg = msg[:upstreamSnippetMax] + "…"
	}
	msg = strings.Join(strings.Fields(msg), " ")
	if msg == "" {
		return fmt.Sprintf("HTTP %d", status)
	}
	return fmt.Sprintf("HTTP %d: %s", status, msg)
}

const upstreamSnippetMax = 200

// streamFailover moves a STREAMING request that failed at the dial.
//
// The constraint everyone remembers about streaming — that bytes cannot be
// recalled — is true from the first byte written TO THE CLIENT, not from the
// dial. Between the two there is a window in which a 503 is just a routing
// failure, indistinguishable to the caller from having picked the other worker
// first. This is that window.
//
// Returns moved=false and leaves the caller's response untouched in every case
// where it does not apply, so the ordinary path is unchanged.
// ctx is the caller's IDLE-WATCHDOG context, not req.Context(). The re-dial must
// use it or the replacement stream is unwatched: the watchdog cancels ctx, and a
// body dialled on req.Context() is not attached to it, so a silently hung
// replacement pins its slot until the CLIENT disconnects — which is precisely
// the failure the watchdog exists to prevent. Measured at a 200ms idle timeout:
// the control returned in 200ms, the failed-over stream was still blocked at 3s.
func (r *Router) streamFailover(ctx context.Context, req *http.Request, d *dispatch, resp *http.Response, dialErr error) (*Backend, *http.Response, bool) {
	if !d.plan.auto || d.plan.session.locked() {
		return nil, nil, false
	}
	// A dial error or a retryable status. Anything else — including every 4xx —
	// is the request's own problem and will fail identically everywhere.
	if dialErr == nil {
		if resp == nil || (resp.StatusCode < 500 && resp.StatusCode != http.StatusTooManyRequests) {
			return nil, nil, false
		}
	} else if req.Context().Err() != nil {
		return nil, nil, false // the CLIENT went away; not the worker's fault
	}

	tried := map[string]bool{(*d.backend).ID: true}
	for hop := 0; hop < maxFailoverHops; hop++ {
		cands := nextCandidates(d.plan, tried)
		if len(cands) == 0 {
			return nil, nil, false
		}
		fits, ok := r.affordable(d, cands, *d.backend)
		if !ok {
			return nil, nil, false
		}
		from := *d.backend
		slotCtx, cancel := context.WithTimeout(ctx, escalateSlotWait)
		target, slot, _, err := r.pickAndAcquirePreferred(slotCtx, fits, acquirePreference{})
		cancel()
		if err != nil {
			continue
		}
		// Only the worker actually contacted is retired, for the same reason as
		// the buffered path: marking the whole slate strands healthy workers.
		tried[target.ID] = true
		body := patchForwardedBody(d.raw, d.inject, budgetCeiling(target, d.job), d.tr.forBackend(target), target.ServedID)
		body = r.stripLearned(body, d.raw, target.ID)

		newResp, dialErr2 := r.dialStream(ctx, req, target, body)
		if dialErr2 != nil || newResp.StatusCode >= 500 || newResp.StatusCode == http.StatusTooManyRequests {
			if newResp != nil {
				newResp.Body.Close()
			}
			r.registry.releaseSlot(slot)
			continue
		}
		// The failed worker is still judged for its failure — see noteFailedAttempt.
		if dialErr != nil {
			r.noteFailedAttempt(from.ID, bufferedResult{netErr: dialErr})
		} else if resp != nil {
			r.noteFailedAttempt(from.ID, bufferedResult{statusCode: resp.StatusCode})
		}
		// Commit, in the same order redispatch uses: the new slot is already held,
		// so releasing the old one cannot leave this request holding nothing.
		r.registry.incActive(target.ID, 1)
		r.registry.incActive(from.ID, -1)
		r.registry.releaseSlot(*d.slot)
		*d.backend = target
		*d.slot = slot
		d.body = body
		log.Printf("failover(stream): %s failed at the dial for %s → re-dialled on %s (hop %d)",
			from.ID, d.plan.route, target.ID, hop+1)
		return target, newResp, true
	}
	return nil, nil, false
}

// dialStream opens one upstream streaming request without pumping it, so the
// status can be inspected before anything is committed to the client.
func (r *Router) dialStream(ctx context.Context, req *http.Request, backend *Backend, body []byte) (*http.Response, error) {
	// ctx, never req.Context(): the reply this opens is pumped under the idle
	// watchdog, and a body bound to the wrong context is not cancellable by it.
	proxyReq, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamChatURL(backend), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	proxyReq.Header.Set("Content-Type", "application/json")
	setBackendCredential(proxyReq, backend)
	r.stampRelayChain(proxyReq, req, backend)
	return r.streamClient.Do(proxyReq)
}

// noteFailedAttempt records a failure against the worker that produced it, for
// the circuit breaker and for LastError.
//
// Failover moved the request, so this worker never reaches writeBuffered and its
// failure would otherwise leave no trace at all — the breaker cannot trip and
// the dashboard shows it healthy with a blank error. A client hangup is excluded
// for the same reason it is on every other path: it says nothing about the
// worker.
func (r *Router) noteFailedAttempt(backendID string, res bufferedResult) {
	if res.netErr != nil {
		r.registry.setError(backendID, res.netErr.Error())
		r.registry.noteProxyResult(backendID, false)
		return
	}
	if res.statusCode >= 500 {
		r.registry.setError(backendID, upstreamErrorSnippet(res.statusCode, res.body))
	}
	r.registry.noteProxyResult(backendID, res.statusCode < 500)
}

// maxRetryAfterWait caps how long an upstream's own Retry-After may make this
// router sleep on one rung. Long enough to honour a real "busy, come back
// shortly"; short enough that a provider answering in minutes cannot hold a
// caller's slot for them.
const maxRetryAfterWait = 5 * time.Second

// remainingBudget is what is left of the caller's declared deadline, or 0 when
// they declared none. Zero means "unbounded" to every consumer, which matches
// the meaning of an absent X-LLM-Deadline-MS.
func (d *dispatch) remainingBudget() time.Duration {
	if d == nil || d.budget <= 0 {
		return 0
	}
	if left := d.budget - time.Since(d.start); left > 0 {
		return left
	}
	return time.Nanosecond // expired: let the caller see it rather than sleeping
}
