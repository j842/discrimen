package router

// The `expert` virtual model — one question, every model, one answer.
//
// `{"model":"expert"}` selects a mixture-of-agents route: the same prompt goes
// to every model the fleet can serve it with, and the highest-quality worker
// then reads the answers and writes the one the client gets. The client sees
// that final answer and nothing else — no panel, no members, no working.
//
// It is deliberately expensive: N generations plus a synthesis. Nothing routes
// here by accident; a client has to ask for it by name, the same way it asks
// for a model or a group.
//
// The shape of it:
//
//   - ONE MEMBER PER MODEL. "Every model", not every worker — three workers
//     serving the same weights are three copies of one opinion. The best-ranked
//     worker for each model is its member, chosen by the ranking that already
//     decides where an ordinary request goes.
//   - THE HARD FILTERS STILL APPLY. A worker that cannot fit the prompt, lacks a
//     required feature, or cannot think when the request demands it is not a
//     member. It is skipped silently: an ensemble that 503s because one model is
//     too small has learned nothing from the others it could have asked.
//   - NO TOOLS. N models emit N different tool calls and there is no honest way
//     to merge them into one; a synthesiser inventing one would be worse than
//     any single member's. A request carrying tools, or continuing a tool loop,
//     falls back to ordinary automatic routing and says so in X-LLM-Expert.
//   - SATURATION IS A SKIP, NOT A QUEUE. Each member takes its own slot with a
//     short bounded wait. One busy worker must not hold up the panel.
//   - FAILURE IS A SKIP TOO. Members that error, time out or answer nothing are
//     dropped. Two or more usable answers are synthesised; exactly one is
//     returned as it stands, because there is nothing to gather; none is a 503.
//   - THE WORKING IS STRIPPED. The synthesiser is fed content only, never a
//     member's reasoning trace, runs with thinking forced off, and has any
//     reasoning field removed from what it produces.
//
// What an ensemble must NOT do is teach the router anything, and it is kept out
// of both learners three times over. serveExpert owns the whole response and
// never goes through proxyToBackend, so the deferred goroutine that feeds the
// adapter and the judge is not on this path at all. An expert plan carries
// auto=false, which is the flag the JUDGE is gated on since it stopped sniffing
// the route string. And the route string is "expert", which parseRouteScore does
// not recognise, so the ADAPTER — which still needs a numeric difficulty score
// and therefore still parses it — has nothing to observe against. All three say
// the same thing: an ensemble is not a tier the router chose, and a bin biased by
// an ensemble's outcome would be biased by N models' work for a decision about
// one.
//
// The name is the router's, held the same way "default" and a group name are:
// no group may be created with it, and a worker that registers a model called
// `expert` does not capture it (see expertShadow and modelCatalogue).

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// expertModel is the reserved name, and routeExpert the X-LLM-Route value it
// serves under. Distinct constants because they are two different contracts:
// one is what a client may type, the other is what a client may parse.
const (
	expertModel = "expert"
	routeExpert = "expert"
)

// isExpertModel reports whether a client's model field selects the ensemble.
// Case-insensitively, like the auto-route sentinels and a group name: the router
// owns this name, so nobody should have to guess its capitalisation.
func isExpertModel(name string) bool {
	return strings.EqualFold(strings.TrimSpace(name), expertModel)
}

// expertSlotWait is how long ONE member waits for its own worker's slot before
// the panel gives up on it.
//
// Short on purpose, and no longer than any other grace in the router (level with
// sessionLockWait's 5s, against qualityFloorWait's 10s and escalateSlotWait's
// 15s), because it is the only one where waiting costs somebody else's answer:
// the panel is as slow as its slowest member, so a request held here delays every
// other member's result as well as its own. A worker that has not freed a slot in
// five seconds is mid-generation — tens of seconds on this fleet — so waiting
// longer means waiting out a whole other request, and the ensemble has N-1 other
// models to hear from in the meantime.
var expertSlotWait = 5 * time.Second

// expertMaxParallel caps how many members are in flight at once.
//
// Larger than any fleet this router has seen, so in practice it caps nothing:
// members sit on distinct workers and do not contend with each other. It exists
// for the shape P2 made possible — one metered endpoint with a row per model,
// where an unbounded fan-out is dozens of simultaneous paid generations against
// one provider's rate limit.
const expertMaxParallel = 8

// expertAnswerMaxChars caps how much of each member's answer is put in front of
// the synthesiser. Above the default completion budget (DEFAULT_MAX_TOKENS is
// 4096, ~16k characters at this fleet's tokenisers), so an ordinary answer is
// never cut; it is a bound on the pathological case, where several members each
// write to a large ceiling and the synthesis prompt no longer fits anything.
// When it does not fit anything, the ensemble hands back the best single answer
// rather than failing — see pickSynthesiser.
const expertAnswerMaxChars = 16000

// ── Route resolution ────────────────────────────────────────────────────────

// expertRoute is what ensemble resolution decided about one request, threaded
// onto the route plan so the proxy and the preview report the same thing. It is
// the zero value for every request that did not name the ensemble.
//
// Deliberately holds no members: who they are depends on the CALLER's
// allow-list, and who synthesises depends on answers that do not exist yet.
type expertRoute struct {
	asked    bool   // the client named the expert model
	fallback string // why it will not get an ensemble; "" ⇒ it will
}

// active reports an ensemble that is actually going to happen.
func (e expertRoute) active() bool { return e.asked && e.fallback == "" }

// header is the X-LLM-Expert value for a request that asked for an ensemble and
// did not get one. An ensemble that DID happen reports what it actually did
// instead (see expertOutcome.header).
func (e expertRoute) header() string {
	if !e.asked || e.fallback == "" {
		return ""
	}
	return "fallback=" + e.fallback
}

// expertFallback reports why a request naming the ensemble will be routed
// normally instead, or "" when the ensemble stands.
func expertFallback(req *ChatRequest) string {
	// /v1/completions reaches selection through the same planner with no messages
	// (see handleCompletions). A panel needs a conversation to put to it and a
	// synthesis needs answers in the shape the caller asked for, so the legacy
	// endpoints route normally rather than being served something else's shape.
	if len(req.Messages) == 0 {
		return "non-chat"
	}
	if len(req.Tools) > 0 && string(req.Tools) != "null" {
		return "tools"
	}
	// Mid-tool-loop: the same reason escalation refuses to move a turn (see
	// escalate.go). This turn's tool result belongs to the model that emitted the
	// matching tool call, and handing it to five others produces five refusals.
	if inToolLoop(req.Messages) {
		return "tool-loop"
	}
	return ""
}

// expertShadow reports what a proposed group name would take over when it is the
// ensemble's, mirroring groupShadows' other cases. A group resolving ahead of
// the ensemble would silently turn an N-model route into a one-model one.
func expertShadow(name string) string {
	if isExpertModel(name) {
		return "the expert ensemble route"
	}
	return ""
}

// expertEntry is the /v1/models row for the ensemble, owned by the router
// because that is what it is: no endpoint knows the name exists. It advertises
// the UNION of the fleet's features for the same reason "default" does — the
// panel is drawn from whatever can serve the request, so between them the
// members really can do whatever any worker can.
func expertEntry(fleetFeatures []string) map[string]any {
	return map[string]any{
		"id":       expertModel,
		"object":   "model",
		"owned_by": routerOwner,
		"expert":   true,
		"features": fleetFeatures,
	}
}

// ── The panel ───────────────────────────────────────────────────────────────

// expertMembers picks the panel: one worker per distinct model, best-ranked
// first.
//
// candidates has already been through the hard filter, so "can serve this
// request" is settled before this is called; what is left is which of several
// workers serving the same model should speak for it. That question already has
// an answer — the completion-time ranking every other route uses — so it is
// asked here rather than re-invented. target 0: an ensemble has no quality bar
// to clear, since it is asking everyone.
func expertMembers(candidates []*Backend, job jobCost) []*Backend {
	ranked := rankByDifficulty(append([]*Backend(nil), candidates...), 0, job, false)
	seen := map[string]bool{}
	members := make([]*Backend, 0, len(ranked))
	for _, b := range ranked {
		if isEmbeddingsOnly(b) || seen[b.Model] {
			continue
		}
		seen[b.Model] = true
		members = append(members, b)
	}
	return members
}

// pickSynthesiser is the highest MEASURED quality worker that can still fit the
// synthesis prompt, or nil when none can.
//
// Quality is the fleet's own benchmark result, not a declared number. It is NO
// LONGER the same source the background judge picks its grader from: judgeGrader
// now ranks on graderStrength, which reads the outcome matrix — the hit rate on
// prompts LIKE THIS ONE where the request carries an embedding, the worker's
// overall rate otherwise — and falls back to the benchmark scalar only for a
// fleet nothing has been observed on yet.
//
// The divergence is not an oversight, but it is not a settled decision either.
// The judge is choosing a grader for one specific answer and holds a vector for
// the prompt, computed for routing and passed down. Synthesis has N answers and
// one conversation, and no equivalent "which of these is the right question to
// ask about" — so the fleet-wide benchmark is still the only ordering over
// workers available here that is not a guess. Anyone wiring the matrix in should
// know they are answering a question this function has not been asked yet, not
// fixing an inconsistency.
//
// Ties are broken by the ordinary ranking, so two workers of equal measured
// quality are separated by which will finish first rather than by map order.
//
// A member may be the synthesiser; that is fine. It reads its own answer beside
// the others without knowing which was its, and the alternative — excluding it —
// would hand the synthesis to a worse model on a two-model fleet.
func pickSynthesiser(candidates []*Backend, neededContext int, job jobCost) *Backend {
	fits := filterCandidates(candidates, func(b *Backend) bool {
		// Same rule as admitReason: an undeclared context is not evidence of a
		// small one.
		return b.ContextK <= 0 || b.ContextK >= neededContext
	})
	if len(fits) == 0 {
		return nil
	}
	best := 0
	for _, b := range fits {
		if b.Quality > best {
			best = b.Quality
		}
	}
	tied := filterCandidates(fits, func(b *Backend) bool { return b.Quality == best })
	return rankByDifficulty(tied, 0, job, false)[0]
}

// ── Dispatch ────────────────────────────────────────────────────────────────

// expertAnswer is one member's usable reply: the worker, and the CONTENT it
// produced. The reasoning trace is deliberately not carried — see
// answerText for the one case where it is read at all.
type expertAnswer struct {
	backend *Backend
	text    string
}

// expertOutcome is what the ensemble did, for the response header and the log
// line. skipped counts members that were selected and produced nothing —
// saturated, failed, or empty. Workers the hard filter dropped are not counted:
// they were never members.
type expertOutcome struct {
	answered int
	skipped  int
	synth    string
}

// header is the X-LLM-Expert value: what happened, in the terms an operator
// reading a request log would ask about. synth=none is the single-answer case,
// where there was nothing to gather and no synthesis ran.
func (o expertOutcome) header() string {
	synth := o.synth
	if synth == "" {
		synth = "none"
	}
	return fmt.Sprintf("members=%d,skipped=%d,synth=%s", o.answered, o.skipped, synth)
}

// serveExpert runs the whole ensemble: fan out, gather, synthesise, reply. It
// owns the response from here — proxyToBackend is never involved, which is also
// why the accounting, logging and credential rules it enforces are restated
// below rather than inherited.
func (r *Router) serveExpert(w http.ResponseWriter, req *http.Request, ident *identity, plan *routePlan, body []byte, chatReq *ChatRequest) {
	start := time.Now()
	// The caller has been identified; their credential is of no further use to
	// this router and must not leave it. Same rule and the same reason as
	// proxyToBackend (see setBackendCredential), restated because an ensemble
	// reaches N endpoints without going through it.
	req.Header.Del("Authorization")

	logEntry := RequestLog{
		CreatedAt: start.UTC(),
		Route:     routeExpert,
		Stream:    chatReq.Stream,
		Input:     string(body),
		KeyID:     ident.logKeyID(),
		// The INBOUND request, read once here. An ensemble reaches N endpoints
		// without going through proxyToBackend, so there is no single outbound
		// request further down that this could have been taken off instead.
		ClientIP: clientIP(req),
	}
	tally := &usageTally{}
	defer func() {
		logEntry.DurationMillis = time.Since(start).Milliseconds()
		// Summed across the whole ensemble, which is what this row describes.
		// Thinking, concurrency and TTFT stay unset on purpose: an expert request
		// has no single backend to attribute them to, and their zero values read
		// as "not recorded" rather than as measurements. A timing model keyed on
		// (backend, thinking) therefore skips these rows instead of learning from
		// an average over N different workers.
		logEntry.PromptTokens = tally.prompt
		logEntry.CompletionTokens = tally.completion
		// Async, exactly as on the proxy path: a synchronous SQLite write against a
		// pool capped at one connection would extend every request by it.
		charged, caller, entry := tally.charged, ident, redactForRelay(logEntry, ident)
		go func() {
			r.recordKeyUse(caller, charged)
			if r.logs != nil {
				if err := r.logs.Insert(context.Background(), entry); err != nil {
					log.Printf("persist request log failed: %v", err)
				}
			}
		}()
	}()

	// The panel is the hard-filtered fleet narrowed to what this key may use. An
	// ensemble reaches every model at once, so the allow-list has to be applied
	// here for the same reason restrictToAllowList applies it to an auto route: a
	// key issued for one worker must not reach the whole fleet by naming a route.
	// A key restricted to one model degenerates to a panel of one, which is the
	// right answer rather than a refusal.
	panel := plan.candidates
	if ident != nil && len(ident.Models) > 0 {
		panel = filterCandidates(panel, ident.allowsBackend)
		if len(panel) == 0 {
			writeUnavailable(w, r.retryAfterUnavailable(), fmt.Sprintf(
				"no worker this key may use is available (allowed: %s)", strings.Join(ident.Models, ", ")))
			logEntry.StatusCode = http.StatusServiceUnavailable
			return
		}
	}

	members := expertMembers(panel, plan.job)
	answers := r.askPanel(req, plan, chatReq, members, body, tally)
	outcome := expertOutcome{answered: len(answers), skipped: len(members) - len(answers)}

	if len(answers) == 0 {
		retry := r.retryAfterUnavailable()
		if len(members) > 0 {
			// There were models to ask and none of them answered: a fleet that is busy
			// or failing rather than absent, so the saturation hint is the honest one.
			retry = r.retryAfterSaturated()
		}
		log.Printf("expert: none of %d member(s) produced an answer", len(members))
		w.Header().Set("X-LLM-Route", routeExpert)
		w.Header().Set("X-LLM-Expert", outcome.header())
		writeUnavailable(w, retry, "expert: no model in the fleet produced an answer")
		logEntry.StatusCode = http.StatusServiceUnavailable
		return
	}

	r.answerExpert(w, req, plan, chatReq, body, panel, answers, outcome, tally, &logEntry)
}

// askPanel fans the client's request out to every member and returns the usable
// answers, in panel order. Members that cannot be reached, refuse, fail or
// answer nothing are simply absent from the result.
func (r *Router) askPanel(req *http.Request, plan *routePlan, chatReq *ChatRequest, members []*Backend, body []byte, tally *usageTally) []expertAnswer {
	// The panel is buffered by necessity — nothing can be synthesised from a
	// stream still arriving — whatever the client asked for. Done once: it is the
	// same edit for every member.
	base := bufferedBody(body)
	// Same rule as proxyToBackend: fill in a budget only where the client set
	// none, never alongside one they did set.
	injectMaxTokens := 0
	if !chatReq.ClientSetMaxTokens {
		injectMaxTokens = chatReq.MaxTokens
	}

	out := make([]*expertAnswer, len(members))
	var mu sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, expertMaxParallel)
	for i, member := range members {
		wg.Add(1)
		go func(i int, b *Backend) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			answer, usage, ok := r.askMember(req, plan, b, body, base, injectMaxTokens)
			if !ok {
				return
			}
			mu.Lock()
			defer mu.Unlock()
			out[i] = &answer
			tally.add(b, usage, plan.job)
		}(i, member)
	}
	wg.Wait()

	answers := make([]expertAnswer, 0, len(members))
	for _, a := range out {
		if a != nil {
			answers = append(answers, *a)
		}
	}
	return answers
}

// askMember runs one member's generation. ok=false is every reason not to use
// it — no slot in time, a transport failure, a refusal, an empty reply — and
// none of them is an error the caller ever hears about.
func (r *Router) askMember(req *http.Request, plan *routePlan, b *Backend, client, base []byte, injectMaxTokens int) (expertAnswer, usageCount, bool) {
	slot, ok := r.acquireMemberSlot(req.Context(), b.ID)
	if !ok {
		log.Printf("expert: %s had no free slot within %s — asking the rest of the panel without it", b.ID, expertSlotWait)
		return expertAnswer{}, usageCount{}, false
	}
	defer r.registry.releaseSlot(slot)

	member := patchForwardedBody(base, injectMaxTokens, budgetCeiling(b, plan.job), plan.tr.forBackend(b), b.ServedID)
	member = r.stripLearned(member, client, b.ID)

	r.registry.incActive(b.ID, 1)
	defer r.registry.incActive(b.ID, -1)
	// Bounded by BACKEND_TIMEOUT_SECONDS (r.client) and by the caller's own
	// context, which is what a client hanging up cancels.
	res := r.requestBuffered(req, b, member)
	if res.netErr != nil {
		if req.Context().Err() == nil {
			r.registry.noteProxyResult(b.ID, false)
		}
		return expertAnswer{}, usageCount{}, false
	}
	r.registry.noteProxyResult(b.ID, res.statusCode < 500)
	if !res.ok() {
		log.Printf("expert: %s answered %d — dropped from the panel (%s)", b.ID, res.statusCode, truncate(string(res.body), 160))
		return expertAnswer{}, usageCount{}, false
	}
	var raw map[string]any
	if err := json.Unmarshal(res.body, &raw); err != nil {
		return expertAnswer{}, usageCount{}, false
	}
	text := answerText(raw)
	if strings.TrimSpace(text) == "" {
		return expertAnswer{}, usageCount{}, false
	}
	return expertAnswer{backend: b, text: text}, usageFrom(raw), true
}

// acquireMemberSlot takes THIS worker's slot, waiting at most expertSlotWait.
// Unlike pickAndAcquire it never spills to another worker: the panel already has
// one member per model, and spilling would ask a model twice while leaving
// another unasked.
func (r *Router) acquireMemberSlot(ctx context.Context, id string) (chan struct{}, bool) {
	if slot, ok := r.registry.tryAcquireSlot(id); ok {
		return slot, true
	}
	deadline := time.NewTimer(expertSlotWait)
	defer deadline.Stop()
	poll := time.NewTicker(slotPollInterval)
	defer poll.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil, false
		case <-deadline.C:
			return nil, false
		case <-poll.C:
		}
		if slot, ok := r.registry.tryAcquireSlot(id); ok {
			return slot, true
		}
	}
}

// answerText is the answer a member actually gave: its content, falling back to
// its reasoning ONLY when the content is empty.
//
// That fallback is not a way in for reasoning traces — it is the case where the
// trace is all there is, because the model never closed its thinking block and
// the endpoint's parser put the entire answer, conclusion included, in the
// reasoning field (see preferContent). Anything the model wrote as content wins
// outright, so a thinking model's working never reaches the synthesiser.
func answerText(raw map[string]any) string {
	content, reasoning, _ := completionText(raw)
	return preferContent(content, reasoning)
}

// ── Synthesis ───────────────────────────────────────────────────────────────

// expertSynthesisPrompt is the whole quality of this feature.
//
// Every line of it is defending against a specific failure of the naive
// version. Unattributed and unordered, because a synthesiser told which model
// said what grades the name instead of the answer, and one told nothing still
// treats the first entry as the incumbent. Explicitly permitted to be right when
// they are all wrong, because a synthesiser that believes its job is to select
// will select the least bad. Explicitly forbidden to average, because splitting
// the difference between two incompatible claims produces a third claim nobody
// made and nothing supports. And told to answer rather than to report, because
// the client asked a question and every word about the comparison is a word that
// answers something they did not ask.
const expertSynthesisPrompt = `Several candidate answers to the user's latest message are given below. They were written independently, they are unattributed, and their order carries no meaning. Judge them on their merits alone.

Write the single best answer to the user's latest message.

- Keep what the candidates get right, and use the strongest reasoning and the most accurate detail among them.
- Where they disagree, work out which is correct and follow only that. A claim is not more likely to be true because several candidates repeat it, and not less likely because only one makes it.
- Where they are all wrong, incomplete, or answer a different question, answer correctly yourself. You are not limited to what they contain.
- Drop anything you cannot support. Never split the difference between conflicting claims: a compromise between two incompatible answers is true of neither.

Output the final answer and nothing else. Do not mention these candidates, that you compared anything, or how you reached the answer. Do not preface the answer with your process, a restatement of the question, or a summary of what follows. Write as though answering the user for the first time, in the language, format and level of detail their request calls for.

Candidate answers:

`

// expertSynthesisSystem renders the synthesis instruction with the candidate
// answers under it.
//
// It goes in a SYSTEM message ahead of the client's own conversation rather than
// in a user turn after it, for two reasons: the client's messages reach the
// synthesiser exactly as they reached the members — full multi-turn context,
// nothing rewritten — and generation still begins directly after the user's last
// turn, so no chat template is asked to render two user messages in a row.
func expertSynthesisSystem(answers []expertAnswer) string {
	var sb strings.Builder
	sb.WriteString(expertSynthesisPrompt)
	for i, a := range answers {
		fmt.Fprintf(&sb, "--- Candidate %d ---\n%s\n\n", i+1, truncate(a.text, expertAnswerMaxChars))
	}
	return strings.TrimRight(sb.String(), "\n")
}

// answerExpert writes the client's reply: the synthesis where there was
// something to synthesise, and the one usable answer where there was not.
func (r *Router) answerExpert(w http.ResponseWriter, req *http.Request, plan *routePlan, chatReq *ChatRequest, body []byte, panel []*Backend, answers []expertAnswer, outcome expertOutcome, tally *usageTally, logEntry *RequestLog) {
	best := bestAnswer(answers)
	synth, messages := (*Backend)(nil), []Message(nil)
	if len(answers) > 1 {
		// Sized from the answers in hand rather than from an estimate of them: the
		// fan-out has already finished, so what the synthesis prompt costs is a fact
		// by now and does not need guessing.
		//
		// Drawn from the PANEL, so the allow-list that narrowed the members narrows
		// the synthesiser too — a key restricted to one model must not have another
		// one read its answers.
		messages = append([]Message{{Role: "system", Content: expertSynthesisSystem(answers)}}, chatReq.Messages...)
		synth = pickSynthesiser(panel, estimateContextK(messages, contextReserveTokens(chatReq)), plan.job)
		if synth == nil {
			log.Printf("expert: %d answers do not fit any worker's context — returning the best single answer from %s",
				len(answers), best.backend.ID)
		}
	}
	if synth == nil {
		// Nothing to gather (one answer) or nowhere to gather it. Either way the
		// client gets an answer rather than an apology.
		r.writeSingleAnswer(w, req, chatReq, best, outcome, tally, logEntry)
		return
	}

	outcome.synth = synth.ID
	logEntry.BackendID = synth.ID
	logEntry.BackendModel = synth.Model
	logEntry.ObservedTPS = synth.ObservedTPS
	logEntry.CertifiedTPS = synth.Certification.TokensPerSec
	logEntry.BaselineTPS = synth.BaselineTPS
	logEntry.SpeedScore = speedScore(synth)

	synthBody := r.synthesisBody(body, messages, synth, chatReq, plan)
	slot, ok := r.acquireMemberSlot(req.Context(), synth.ID)
	if !ok {
		log.Printf("expert: synthesiser %s had no free slot within %s — returning the best single answer", synth.ID, expertSlotWait)
		outcome.synth = ""
		r.writeSingleAnswer(w, req, chatReq, best, outcome, tally, logEntry)
		return
	}
	defer r.registry.releaseSlot(slot)
	r.registry.incActive(synth.ID, 1)
	defer r.registry.incActive(synth.ID, -1)

	if chatReq.Stream {
		r.streamSynthesis(w, req, plan, chatReq, synth, synthBody, best, outcome, tally, logEntry)
		return
	}
	res := r.requestBuffered(req, synth, synthBody)
	if res.netErr == nil {
		r.registry.noteProxyResult(synth.ID, res.statusCode < 500)
	}
	// The answers are already paid for and the synthesis is the one call that
	// cannot be repeated cheaply, so a synthesiser that refuses — or that returns
	// nothing at all, which is the same thing to a client — costs the synthesis
	// and not the answer. Empty is judged by the same classifyResponse that
	// decides an ordinary reply is unusable (see escalate.go); it is only testable
	// here because nothing has been written yet, which is exactly why the streamed
	// path below cannot do it.
	if !res.ok() || classifyResponse(res.body, false) == responseEmpty {
		log.Printf("expert: synthesis on %s produced nothing usable (status=%d err=%v) — returning the best single answer",
			synth.ID, res.statusCode, res.netErr)
		outcome.synth = ""
		if res.ok() {
			tally.add(synth, usageOf(res.body), plan.job)
		}
		r.writeSingleAnswer(w, req, chatReq, best, outcome, tally, logEntry)
		return
	}
	tally.add(synth, usageOf(res.body), plan.job)
	r.logExpert(outcome, tally)

	final := plainCompletion(res.body, tally.usageCount)
	setRouteHeaders(w, synth, routeExpert, logEntry)
	w.Header().Set("X-LLM-Expert", outcome.header())
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	logEntry.StatusCode = http.StatusOK
	if _, err := w.Write(final); err != nil {
		logEntry.Error = err.Error()
		return
	}
	logEntry.Output = r.logBody(final)
}

// bestAnswer is the answer to fall back on when there is no synthesis: the one
// from the highest measured quality worker. The fleet's own benchmark is the
// only ordering over answers the router has that is not a guess.
func bestAnswer(answers []expertAnswer) expertAnswer {
	best := answers[0]
	for _, a := range answers[1:] {
		if a.backend.Quality > best.backend.Quality {
			best = a
		}
	}
	return best
}

// writeSingleAnswer returns one member's answer as the client's own, in whichever
// shape they asked for. The buffered form is that member's real response body
// with its reasoning removed; the streamed form is rendered from its text,
// because a buffered reply cannot be replayed as the stream it never was.
func (r *Router) writeSingleAnswer(w http.ResponseWriter, req *http.Request, chatReq *ChatRequest, answer expertAnswer, outcome expertOutcome, tally *usageTally, logEntry *RequestLog) {
	r.logExpert(outcome, tally)
	logEntry.BackendID = answer.backend.ID
	logEntry.BackendModel = answer.backend.Model
	logEntry.ObservedTPS = answer.backend.ObservedTPS
	logEntry.CertifiedTPS = answer.backend.Certification.TokensPerSec
	logEntry.BaselineTPS = answer.backend.BaselineTPS
	logEntry.SpeedScore = speedScore(answer.backend)
	setRouteHeaders(w, answer.backend, routeExpert, logEntry)
	w.Header().Set("X-LLM-Expert", outcome.header())
	logEntry.StatusCode = http.StatusOK
	if chatReq.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		writeCompletionSSE(w, answer.text, answer.backend.Model, tally.usageCount)
		logEntry.Output = r.logBody([]byte(answer.text))
		return
	}
	final := completionOf(answer, tally.usageCount)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(final); err != nil {
		logEntry.Error = err.Error()
		return
	}
	logEntry.Output = r.logBody(final)
}

// logBody renders a reply for the request log the way the proxy path does: head
// and tail within the store's per-row budget, so a long answer keeps its end
// rather than only its beginning (see boundedCapture).
func (r *Router) logBody(body []byte) string {
	max := 0
	if r.cfg != nil {
		max = r.cfg.LogMaxBodyBytes
	}
	capture := newBoundedCapture(max)
	_, _ = capture.Write(body)
	return capture.String()
}

// logExpert is the one line an ensemble writes: what it asked, what answered,
// and — only when somebody is billing for it — what it cost. The cost is
// omitted rather than printed as zero on an all-local fleet, where a zero is the
// truth but a number nobody needs to read on every expert request.
func (r *Router) logExpert(outcome expertOutcome, tally *usageTally) {
	synth := outcome.synth
	if synth == "" {
		synth = "none (single answer)"
	}
	cost := ""
	if tally.paid {
		cost = fmt.Sprintf(", %.4g at declared prices", tally.cost)
	}
	log.Printf("expert: %d member(s) answered, %d skipped, synthesised by %s (%d tokens%s)",
		outcome.answered, outcome.skipped, synth, tally.charged, cost)
}

// synthesisBody is the client's own request with the messages replaced by the
// synthesis conversation.
//
// Built from their body rather than from scratch so their generation settings
// still govern the answer they actually receive: a response_format, a
// temperature, a stop sequence and a stream_options all belong to the final
// answer and would be silently dropped by a hand-built payload.
//
// Thinking is forced OFF, which is why the client's own thinking fields are
// removed before the gate is written: chat_template_kwargs and reasoning_effort
// are escape hatches patchForwardedBody will not overwrite, and this is the
// router's call, not the client's. The synthesiser is reading finished answers
// and choosing between them; there is nothing here to reason its way to, and a
// reasoning block would only be a second thing to strip out afterwards.
func (r *Router) synthesisBody(body []byte, messages []Message, synth *Backend, chatReq *ChatRequest, plan *routePlan) []byte {
	out := replaceBodyMessages(body, messages)
	out = stripBodyFields(out, []string{"chat_template_kwargs", "reasoning_effort"})
	// The synthesis prompt is the answers plus the conversation, so it is bigger
	// than the client's prompt; size the ceiling against it rather than against
	// the request the members were sent.
	job := plan.job
	job.promptTokens = estimateContextK(messages, 0) * 1024
	off := thinkingResolution{patch: true}
	out = patchForwardedBody(out, 0, budgetCeiling(synth, job), off.forBackend(synth), synth.ServedID)
	return r.stripLearned(out, body, synth.ID)
}

// ── Streaming the synthesis ─────────────────────────────────────────────────

// streamSynthesis relays the synthesiser's stream to the client, which is the
// whole reason a streaming client gets anything worth streaming: the fan-out
// before it had to be buffered, but the answer the client actually reads is
// generated live.
//
// It relays frame by frame rather than byte by byte because two things have to
// change on the way past: any reasoning delta is removed (the client asked for
// an answer, not a transcript), and a usage block is restated as the ensemble's
// total rather than the synthesiser's share of it. Frames needing neither are
// passed through unaltered.
func (r *Router) streamSynthesis(w http.ResponseWriter, req *http.Request, plan *routePlan, chatReq *ChatRequest, synth *Backend, body []byte, best expertAnswer, outcome expertOutcome, tally *usageTally, logEntry *RequestLog) {
	// Idle watchdog rather than a wall-clock cap, for the reason dispatchStreaming
	// records: a long generation is legitimate, a silent one is not.
	ctx, cancel := context.WithCancel(req.Context())
	defer cancel()
	proxyReq, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamChatURL(synth), bytes.NewReader(body))
	if err != nil {
		outcome.synth = ""
		r.writeSingleAnswer(w, req, chatReq, best, outcome, tally, logEntry)
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")
	setBackendCredential(proxyReq, synth)
	r.stampRelayChain(proxyReq, req, synth)

	idle := time.Duration(0)
	if r.cfg != nil {
		idle = r.cfg.BackendIdleTimeout
	}
	var watchdog *time.Timer
	if idle > 0 {
		watchdog = time.AfterFunc(idle, cancel)
		defer watchdog.Stop()
	}

	resp, err := r.streamClient.Do(proxyReq)
	if err != nil {
		if req.Context().Err() == nil {
			r.registry.setError(synth.ID, err.Error())
			r.registry.noteProxyResult(synth.ID, false)
		}
		// Nothing is on the wire yet, so this is still recoverable: hand back the
		// best answer the panel already produced.
		log.Printf("expert: synthesis stream from %s failed to start (%v) — returning the best single answer", synth.ID, err)
		outcome.synth = ""
		r.writeSingleAnswer(w, req, chatReq, best, outcome, tally, logEntry)
		return
	}
	defer resp.Body.Close()
	r.registry.noteProxyResult(synth.ID, resp.StatusCode < 500)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		log.Printf("expert: synthesiser %s refused the synthesis (%d) — returning the best single answer", synth.ID, resp.StatusCode)
		outcome.synth = ""
		r.writeSingleAnswer(w, req, chatReq, best, outcome, tally, logEntry)
		return
	}

	copyHeaders(w.Header(), resp.Header)
	setRouteHeaders(w, synth, routeExpert, logEntry)
	w.Header().Set("X-LLM-Expert", outcome.header())
	w.WriteHeader(http.StatusOK)
	logEntry.StatusCode = http.StatusOK
	if watchdog != nil {
		watchdog.Reset(idle)
	}

	flusher, _ := w.(http.Flusher)
	max := 0
	if r.cfg != nil {
		max = r.cfg.LogMaxBodyBytes
	}
	capture := newBoundedCapture(max)
	newline := []byte("\n")
	var synthUsage usageCount
	scanner := newLargeScanner(resp.Body)
	for scanner.Scan() {
		if watchdog != nil {
			watchdog.Reset(idle)
		}
		// The scanner strips the delimiter and reuses its buffer, so the newline is
		// written separately rather than appended to the token — appending would
		// write into the buffer the scanner is about to reuse.
		line, used := rewriteSSELine(scanner.Bytes(), tally.usageCount)
		if used != (usageCount{}) {
			synthUsage = used
		}
		_, _ = capture.Write(line)
		_, _ = capture.Write(newline)
		if _, err := w.Write(line); err != nil {
			logEntry.Error = err.Error()
			return
		}
		if _, err := w.Write(newline); err != nil {
			logEntry.Error = err.Error()
			return
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	if err := scanner.Err(); err != nil {
		if ctx.Err() != nil && req.Context().Err() == nil {
			err = fmt.Errorf("backend sent no bytes for %s (idle timeout): %w", idle, err)
		}
		logEntry.Error = err.Error()
		log.Printf("expert: synthesis stream from %s failed mid-flight: %v", synth.ID, err)
		// The status is spent and part of the answer is already on the wire, so the
		// failure can only be reported inside the stream (see writeSSEError).
		if req.Context().Err() == nil && isEventStream(w.Header()) {
			writeSSEError(w, fmt.Sprintf("expert: synthesis stream from backend %q failed: %s", synth.ID, err))
		}
	}
	tally.add(synth, synthUsage, plan.job)
	logEntry.Output = capture.String()
	r.logExpert(outcome, tally)
}

// rewriteSSELine returns the line to relay, and the synthesiser's own usage when
// this frame carried one.
//
// Most frames are returned untouched, byte for byte: the two markers are cheap
// to test for and almost never both present, so a token delta costs one
// bytes.Contains and no allocation. Only a frame that actually carries a
// reasoning delta or a usage block is decoded and re-encoded.
func rewriteSSELine(line []byte, members usageCount) ([]byte, usageCount) {
	if !bytes.HasPrefix(line, []byte("data: ")) {
		return line, usageCount{}
	}
	payload := line[len("data: "):]
	if bytes.Equal(payload, []byte("[DONE]")) {
		return line, usageCount{}
	}
	// A usage OBJECT, not the "usage":null vLLM puts on every chunk — that one
	// carries nothing to restate, and matching it would decode and re-encode every
	// token of every stream.
	hasReasoning := bytes.Contains(payload, []byte(`"reasoning"`)) || bytes.Contains(payload, []byte(`"reasoning_content"`))
	hasUsage := bytes.Contains(payload, []byte(`"usage":{`)) || bytes.Contains(payload, []byte(`"usage": {`))
	if !hasReasoning && !hasUsage {
		return line, usageCount{}
	}
	var chunk map[string]any
	if err := json.Unmarshal(payload, &chunk); err != nil {
		return line, usageCount{}
	}
	stripReasoningFields(chunk, "delta", false)
	used := usageFrom(chunk)
	if used != (usageCount{}) {
		// Cumulative in both dialects, so restating each usage frame as
		// members+this keeps the running total correct at every point in the stream
		// as well as at the end.
		chunk["usage"] = members.plus(used).object()
	}
	rewritten, err := json.Marshal(chunk)
	if err != nil {
		return line, used
	}
	return append([]byte("data: "), rewritten...), used
}

// writeCompletionSSE renders a finished answer as the two-frame stream a
// streaming client expects. It exists for the paths where the ensemble has an
// answer in hand but never opened a stream to produce it — a single member, or a
// synthesis that could not be run — and where handing a stream:true client a
// JSON body would break its SDK outright.
func writeCompletionSSE(w http.ResponseWriter, text, model string, usage usageCount) {
	created := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-expert-%d", created)
	frame := func(v map[string]any) {
		payload, err := json.Marshal(v)
		if err != nil {
			return
		}
		_, _ = fmt.Fprintf(w, "data: %s\n\n", payload)
	}
	frame(map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []any{map[string]any{
			"index": 0,
			"delta": map[string]any{"role": "assistant", "content": text},
		}},
	})
	final := map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"}},
	}
	if usage != (usageCount{}) {
		final["usage"] = usage.object()
	}
	frame(final)
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

// ── Response shaping ────────────────────────────────────────────────────────

// plainCompletion is a completion body with every reasoning field removed and
// the ensemble's own usage in place of the endpoint's.
//
// Removing the reasoning is not tidying: the client asked one question and is
// entitled to one answer, and a synthesiser's trace is a running commentary on
// answers it has been told not to mention. Where the content is empty and the
// trace is all there is, the trace becomes the content rather than being
// dropped — same rule, and the same reason, as answerText.
func plainCompletion(body []byte, usage usageCount) []byte {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return body
	}
	stripReasoningFields(raw, "message", true)
	if usage != (usageCount{}) {
		raw["usage"] = usage.object()
	}
	out, err := json.Marshal(raw)
	if err != nil {
		return body
	}
	return out
}

// completionOf renders one member's answer as a completion body the client can
// read, for the single-answer path.
func completionOf(answer expertAnswer, usage usageCount) []byte {
	created := time.Now().Unix()
	raw := map[string]any{
		"id":      fmt.Sprintf("chatcmpl-expert-%d", created),
		"object":  "chat.completion",
		"created": created,
		"model":   answer.backend.Model,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": answer.text},
			"finish_reason": "stop",
		}},
	}
	if usage != (usageCount{}) {
		raw["usage"] = usage.object()
	}
	out, err := json.Marshal(raw)
	if err != nil {
		return []byte(`{"error":{"message":"expert: could not render the answer"}}`)
	}
	return out
}

// stripReasoningFields removes both dialects' reasoning fields from every
// choice's message (buffered) or delta (streamed).
//
// promote is the difference between the two shapes, not a preference. A whole
// MESSAGE whose content is empty and whose reasoning holds everything is a
// model that never closed its thinking block, and dropping the field would drop
// the answer with it. A streamed DELTA is one token: it has no content by
// construction while the model is reasoning, so promoting would relay the entire
// trace token by token — which is the thing being prevented.
func stripReasoningFields(raw map[string]any, field string, promote bool) {
	choices, _ := raw["choices"].([]any)
	for _, c := range choices {
		choice, _ := c.(map[string]any)
		msg, _ := choice[field].(map[string]any)
		if msg == nil {
			continue
		}
		content, reasoning := messageText(msg)
		delete(msg, "reasoning")
		delete(msg, "reasoning_content")
		if promote && strings.TrimSpace(content) == "" && reasoning != "" {
			msg["content"] = reasoning
		}
	}
}

// bufferedBody forces a request body into the non-streamed shape. The panel
// cannot stream — nothing can be synthesised from an answer still arriving — so
// a stream:true request is fanned out buffered whatever the client asked for.
// stream_options goes with it: several endpoints reject it outright when stream
// is false.
func bufferedBody(body []byte) []byte {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return body
	}
	raw["stream"] = json.RawMessage("false")
	delete(raw, "stream_options")
	out, err := json.Marshal(raw)
	if err != nil {
		return body
	}
	return out
}

// replaceBodyMessages swaps the conversation in a request body, leaving every
// other field the client set exactly where it was.
func replaceBodyMessages(body []byte, messages []Message) []byte {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return body
	}
	encoded, err := json.Marshal(messages)
	if err != nil {
		return body
	}
	raw["messages"] = encoded
	out, err := json.Marshal(raw)
	if err != nil {
		return body
	}
	return out
}

// ── Usage ───────────────────────────────────────────────────────────────────

// usageCount is one endpoint's reported usage. A zero value means the endpoint
// reported nothing, which is a different fact from reporting zero and is treated
// as one.
type usageCount struct {
	prompt     int
	completion int
	total      int
}

func (u usageCount) plus(v usageCount) usageCount {
	return usageCount{u.prompt + v.prompt, u.completion + v.completion, u.total + v.total}
}

func (u usageCount) object() map[string]int {
	return map[string]int{"prompt_tokens": u.prompt, "completion_tokens": u.completion, "total_tokens": u.total}
}

// usageFrom reads a decoded response body's or SSE chunk's usage block.
// total_tokens is preferred because it is what the endpoint charged for; where
// it is absent the two halves are added, which is the same number spelled out.
func usageFrom(raw map[string]any) usageCount {
	u, _ := raw["usage"].(map[string]any)
	if u == nil {
		return usageCount{}
	}
	num := func(key string) int {
		v, _ := u[key].(float64)
		return int(v)
	}
	out := usageCount{prompt: num("prompt_tokens"), completion: num("completion_tokens"), total: num("total_tokens")}
	if out.total == 0 {
		out.total = out.prompt + out.completion
	}
	return out
}

// usageOf reads the usage block off an undecoded completion body.
func usageOf(body []byte) usageCount {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return usageCount{}
	}
	return usageFrom(raw)
}

// usageTally is what a whole ensemble spent: N members plus the synthesis.
//
// It keeps two numbers because they answer two questions. usageCount is what the
// endpoints REPORTED, and it is what the client is shown — an ensemble's usage
// is the sum of its parts, and reporting only the synthesiser's share would
// understate what the request cost by roughly the number of members. charged is
// what the key is billed, which cannot be the same number: an endpoint that
// reports nothing must not be free, so its share is estimated instead, and a
// spending bound built from estimates is still a bound while one built from
// silence is not.
type usageTally struct {
	usageCount
	charged int
	cost    float64
	paid    bool
}

// add folds one call into the tally, at the prices the row declares.
func (t *usageTally) add(b *Backend, used usageCount, job jobCost) {
	t.usageCount = t.usageCount.plus(used)
	charge := used
	if charge.total == 0 {
		// The endpoint said nothing about what it used. Charge what routing sized
		// the job at rather than zero — see the type comment.
		charge = usageCount{prompt: job.promptTokens, completion: job.outputTokens}
		charge.total = charge.prompt + charge.completion
	}
	t.charged += charge.total
	if !isFreeBackend(b) {
		t.paid = true
		t.cost += tokenCost(b, charge.prompt, charge.completion)
	}
}
