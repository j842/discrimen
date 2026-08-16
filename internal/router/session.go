package router

// Session-aware routing — conversation stickiness for multi-turn and agent traffic.
//
// Every turn used to be routed from scratch. That is right for a one-shot prompt
// and wrong for a tool loop: the worker that served turn N holds that
// conversation's prefix in its KV cache, so moving turn N+1 elsewhere throws the
// cache away and re-prefills the entire system prompt + tool schemas — exactly
// the prompt shape where prefill dominates the wall clock (0.67s vs 37.2s across
// this fleet on one ~4k prompt; see prefillSeconds). Worse, switching MID-loop
// hands a tool result to a model that never emitted the matching tool call, whose
// template may not even spell tool_call_id the same way.
//
// Two mechanisms, both expressed in terms the router already measures rather than
// as a new hand-set bias:
//
//   - Prefill discount. The incumbent's prefill for this job is charged at
//     (1 - sessionPrefillDiscount) of nominal, because the shared prefix
//     is already resident there. That IS the whole "stay bias", and it has the
//     right shape for free: it scales with the conversation, so a long agent turn
//     prefers the incumbent strongly and a one-word follow-up barely at all, and
//     it competes inside the existing completion-time ranking instead of
//     overriding it. A genuinely faster worker still wins.
//   - Tool-loop lock. While a tool loop is open the incumbent is preferred
//     outright for a bounded grace (sessionLockWait) before the
//     request spills. Continuity outranks tier here on purpose: half a tool loop
//     served by two different models is worse than all of it served by the
//     cheaper one.
//
// Deliberate limits. The router cannot see a worker's KV cache — that needs
// llm-d/Dynamo-style cache-event streams from the engine, not an OpenAI HTTP
// body — so the discount is a PROXY for cache locality, not a measurement of it.
// Everything here is advisory: it never hard-filters, never 503s, never holds a
// request on a worker that has left the candidate set, and never outranks an
// explicit client choice.

import (
	"hash/fnv"
	"io"
	"sort"
	"sync"
	"time"
)

// sessionSticky disables the whole mechanism when false (every turn routed
// independently, the pre-session behaviour).
var sessionSticky = true

// sessionTTL is how long after its last turn a conversation still counts as
// live. Past it the next turn re-routes from scratch — by then the worker's KV
// cache has almost certainly been evicted anyway, so the discount would be a lie.
var sessionTTL = 30 * time.Minute

// sessionMax bounds the tracker. One-shot traffic mints an entry per request, so
// this has to be a cap and not an assumption about how many conversations exist.
var sessionMax = 4096

// sessionPrefillDiscount is the fraction of this job's prefill assumed already
// cached on the worker that served the previous turn. Not 1.0: the new turn's own
// tokens are never cached, the engine may have evicted the prefix, and an
// over-confident discount would pin a conversation to a worker that has genuinely
// become the slower choice.
var sessionPrefillDiscount = 0.8

// sessionLockWait is the bounded grace a mid-tool-loop request waits for the
// incumbent's slot before spilling. Same shape and the same reasoning as
// qualityFloorWait: prefer briefly, then serve rather than refuse.
var sessionLockWait = 5 * time.Second

// sessionRoute is one request's session context, resolved once during selection
// and reused for ranking, acquisition and the response header.
type sessionRoute struct {
	key       uint64 // 0 ⇒ no session could be derived
	incumbent string // worker that served the previous turn ("" ⇒ none / first turn)
	toolLoop  bool   // the conversation is mid-tool-loop
}

// active reports whether this request has a session worth recording against.
func (s sessionRoute) active() bool { return s.key != 0 }

// locked reports whether the incumbent should be preferred outright rather than
// merely discounted: mid-tool-loop, with an incumbent to hold on to.
func (s sessionRoute) locked() bool { return s.toolLoop && s.incumbent != "" }

// outcome labels what the session did to this request, for X-LLM-Session.
func (s sessionRoute) outcome(chosenID string) string {
	switch {
	case !s.active():
		return ""
	case s.incumbent == "":
		return "new"
	case s.incumbent == chosenID && s.toolLoop:
		return "lock"
	case s.incumbent == chosenID:
		return "stay"
	case s.toolLoop:
		return "lock-broken"
	default:
		return "switch"
	}
}

// ── Conversation identity ───────────────────────────────────────────────────

// sessionHeadChars caps how much of each head message is hashed. Enough to make
// two conversations distinct without walking a megabyte of pasted file on every
// turn of a long agent session.
const sessionHeadChars = 4096

// sessionKeyFor identifies the conversation a request belongs to by hashing the
// HEAD of the message list — the system/developer prompt and the FIRST user turn.
//
// The head is the part that stays byte-identical as turns accumulate, which is
// the whole trick: hashing the conversation minus its last message (the obvious
// first attempt) yields a different key on every turn and matches nothing. Two
// conversations that genuinely start identically collide, and that is fine —
// same system prompt plus same opening question is the same task, and sharing a
// worker between them is what the prefix cache wants anyway.
//
// ok=false when there is no user turn to key on (a bare /v1/completions call, a
// system-only request), in which case the request simply has no session.
func sessionKeyFor(req *ChatRequest) (uint64, bool) {
	if req == nil || len(req.Messages) == 0 {
		return 0, false
	}
	h := fnv.New64a()
	sawUser := false
	for _, m := range req.Messages {
		switch m.Role {
		case "system", "developer":
			if !sawUser {
				writeSessionPart(h, m.Role, m.Content)
			}
		case "user":
			if sawUser {
				continue // only the FIRST user turn is part of the identity
			}
			sawUser = true
			writeSessionPart(h, m.Role, m.Content)
		}
	}
	if !sawUser {
		return 0, false
	}
	sum := h.Sum64()
	if sum == 0 {
		sum = 1 // 0 is the "no session" sentinel
	}
	return sum, true
}

// writeSessionPart folds one head message into the hash. The NUL separators stop
// a role/content boundary shift from colliding two different conversations.
func writeSessionPart(h io.Writer, role string, content any) {
	_, _ = io.WriteString(h, role)
	_, _ = io.WriteString(h, "\x00")
	_, _ = io.WriteString(h, cutChars(contentText(content), sessionHeadChars))
	_, _ = io.WriteString(h, "\x00")
}

// inToolLoop reports whether the request continues a tool call the previous
// completion started: the conversation ends with a tool result, or with an
// assistant turn that requested one. Either way the next completion belongs to
// the model that opened the loop.
func inToolLoop(msgs []Message) bool {
	if len(msgs) == 0 {
		return false
	}
	last := msgs[len(msgs)-1]
	switch last.Role {
	case "tool", "function":
		return true
	case "assistant":
		return len(last.ToolCalls) > 0 && string(last.ToolCalls) != "null"
	}
	return false
}

// ── Tracker ─────────────────────────────────────────────────────────────────

type sessionEntry struct {
	backendID string
	turns     int
	lastSeen  time.Time
}

// sessionTracker maps conversation key → the worker that last served it.
// In-memory and deliberately not persisted: after a router restart no worker's
// KV cache holds anything of ours either, so a stale affinity would be a
// discount with nothing behind it.
type sessionTracker struct {
	mu  sync.Mutex
	ttl time.Duration
	max int
	m   map[uint64]sessionEntry
}

func newSessionTracker(ttl time.Duration, max int) *sessionTracker {
	if ttl <= 0 {
		ttl = 30 * time.Minute
	}
	if max < 16 {
		max = 16
	}
	return &sessionTracker{ttl: ttl, max: max, m: make(map[uint64]sessionEntry)}
}

// lookup returns the worker that served this conversation's previous turn, or
// ok=false if there is none or it has gone stale. Safe on a nil receiver.
func (t *sessionTracker) lookup(key uint64) (sessionEntry, bool) {
	if t == nil || key == 0 {
		return sessionEntry{}, false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	e, ok := t.m[key]
	if !ok {
		return sessionEntry{}, false
	}
	if time.Since(e.lastSeen) > t.ttl {
		delete(t.m, key)
		return sessionEntry{}, false
	}
	return e, true
}

// remember records the worker that just served a turn of this conversation.
// Safe on a nil receiver.
func (t *sessionTracker) remember(key uint64, backendID string) {
	if t == nil || key == 0 || backendID == "" {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	e := t.m[key]
	e.backendID = backendID
	e.turns++
	e.lastSeen = time.Now()
	t.m[key] = e
	if len(t.m) > t.max {
		t.evictLocked()
	}
}

// evictLocked drops expired entries first and, if that wasn't enough, the oldest
// quarter by last use. Oldest-first (not insertion-order) matters: a long agent
// session must not be evicted by a burst of one-shot prompts that arrived after
// it started.
func (t *sessionTracker) evictLocked() {
	now := time.Now()
	for k, e := range t.m {
		if now.Sub(e.lastSeen) > t.ttl {
			delete(t.m, k)
		}
	}
	if len(t.m) <= t.max {
		return
	}
	type aged struct {
		key uint64
		at  time.Time
	}
	all := make([]aged, 0, len(t.m))
	for k, e := range t.m {
		all = append(all, aged{k, e.lastSeen})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].at.Before(all[j].at) })
	drop := len(t.m) - t.max + t.max/4
	for i := 0; i < drop && i < len(all); i++ {
		delete(t.m, all[i].key)
	}
}

// size reports the tracked conversation count (health/observability).
func (t *sessionTracker) size() int {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.m)
}

// resolve builds the session context for a request: its conversation key, the
// worker that served the previous turn (only when that worker is still a live
// candidate), and whether a tool loop is open. Returns the zero value — no
// stickiness at all — when the feature is off or no key could be derived.
func (t *sessionTracker) resolve(req *ChatRequest, candidates []*Backend) sessionRoute {
	if t == nil || !sessionSticky || req == nil {
		return sessionRoute{}
	}
	key, ok := sessionKeyFor(req)
	if !ok {
		return sessionRoute{}
	}
	sr := sessionRoute{key: key, toolLoop: inToolLoop(req.Messages)}
	e, ok := t.lookup(key)
	if !ok {
		return sr
	}
	// Only claim an incumbent that survived this request's own hard filters. A
	// worker that has died, dropped below the context this turn needs, or lost the
	// model the client named is not an incumbent — it's a stale row, and letting it
	// discount a candidate that isn't in the list would silently do nothing while
	// reporting "stay".
	for _, b := range candidates {
		if b.ID == e.backendID {
			sr.incumbent = e.backendID
			break
		}
	}
	return sr
}
