package router

import (
	"bufio"
	"bytes"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	_ "modernc.org/sqlite"
)

type Config struct {
	Port string
	// Bootstrap credentials. Both stay because the compatibility contract freezes
	// their names, and both are now only ONE of the ways in: since P3 the api_keys
	// table carries per-caller keys with roles, and an empty value here means the
	// database is canonical rather than that the surface is open (see
	// bootstrapCredentials).
	WorkerToken        string   // single token; auths POST /backends/register* + DELETE /backends/{id}
	ClientTokens       []string // any-of list; auths the /v1/* OpenAI surface
	DefaultMaxTokens   int
	HealthInterval     time.Duration
	BackendHTTPTimeout time.Duration // whole-exchange cap for BUFFERED requests + probes (streams use the idle timeout instead)
	BackendIdleTimeout time.Duration // streaming: abort when NO bytes arrive for this long (progress-based, no wall-clock cap)
	LogDBPath          string
	LogRetention       time.Duration
	LogMaxBodyBytes    int    // cap on stored input/output body length per request log
	PersistSecret      string // optional; encrypts persisted backend api keys at rest

	// Auto difficulty routing (Phase 1). When enabled, a chat request with no
	// explicit `requirements` is classified by prompt difficulty and routed to
	// the cheapest sufficient backend (see difficulty.go).
	AutoDifficulty      bool
	DifficultyBands     string        // "score:quality" CSV, ascending, e.g. "0.40:2,0.70:7,1.0:9"
	DifficultyTemp      float64       // softness of the simple↔hard margin → score map
	DifficultyTimeout   time.Duration // per-classification embedding deadline
	DifficultyCacheSize int           // bounded prompt→score cache size
	DifficultyMaxChars  int           // cap on classified prompt text length

	// Auto thinking (Phase 3a). When requirements.thinking is absent/"auto",
	// deduce whether the prompt needs reasoning and inject
	// chat_template_kwargs.enable_thinking accordingly.
	AutoThinking       bool
	ReasoningThreshold float64 // reasoning score ≥ this → enable thinking

	// JudgeSampleRate is the fraction of served answers graded in the background
	// by another worker; the verdict is recorded in the outcome matrix, which is
	// the only evidence the router ever gets about the traffic it actually serves
	// (the question bank is fixed and synthetic). 0 disables.
	//
	// It was "the fraction of CHEAPER-THAN-BEST answers", gated on the retired
	// Quality scalar — which meant the fleet's strongest worker was never graded
	// at all, so the matrix could never learn anything about the worker it was
	// most likely to pick.
	JudgeSampleRate float64

	// AdminPassword bootstraps the admin session on a database that has no
	// password set yet. The stored hash is canonical from then on, so changing
	// this variable after first run does nothing.
	AdminPassword string

	// ProfileWorkers measures quality/speed/capacity/capabilities/context at cold
	// start instead of trusting declared values (the worker declares ~nothing).
	// CapacityProbeMax caps the concurrency ramp used for the capacity test.
	SandboxURL       string // sandbox sidecar base URL; empty disables code-exec questions
	SandboxToken     string // optional bearer for the sidecar
	ProfileWorkers   bool
	CapacityProbeMax int

	// EscalateInline re-dispatches an EMPTY non-streamed answer to a better worker
	// before replying, instead of only teaching the adapter about the region (see
	// escalate.go).
	EscalateInline bool

	// RescueTruncated asks a worker that spent its whole token budget inside the
	// thinking block — finish_reason "length", no content, a full reasoning trace —
	// for its conclusion in one cheap follow-up turn, instead of handing the caller
	// the empty answer (see rescue.go).
	RescueTruncated bool
}

type BackendRegistration struct {
	ID          string   `json:"id"`
	URL         string   `json:"url"`
	Model       string   `json:"model"`
	Quality     int      `json:"quality"`
	Thinking    bool     `json:"thinking"`
	ContextK    int      `json:"context_k"`
	BaselineTPS float64  `json:"baseline_tps"`
	Features    []string `json:"features"`
	HealthPath  string   `json:"health_path"`
	TTLSeconds  int      `json:"ttl_seconds"`
	// MaxConcurrency is a declared concurrency ceiling. On a MANUAL row it
	// outranks the capacity ramp outright; on a beacon row it is the seed it has
	// always been, replaced by the measurement. See providers.go.
	MaxConcurrency int    `json:"max_concurrency"`
	APIKey         string `json:"api_key,omitempty"`

	// Provider names where this row runs ("local", "openai", "anthropic", …) and
	// Source records how it got here — "beacon" for a worker that registered
	// itself, "manual" for a row an operator entered. Both default on the way in
	// (see normalizeProviderFields): a registration that arrives without them is
	// a local beacon at zero cost, which is what every deployed worker sends.
	Provider string `json:"provider,omitempty"`
	Source   string `json:"source,omitempty"`
	// Relay names which configured relay a source=="relay" row was derived from
	// (see relay.go). Empty on every other kind of row, and cleared by
	// normalizeProviderFields on anything that is not a relay row — a beacon that
	// claimed to belong to a relay would be pruned by the next refresh.
	Relay string `json:"relay,omitempty"`
	// Price per MILLION tokens, in whatever currency the operator is billed in.
	// Zero means free, which is the truth for every local worker and therefore
	// the right default (see PLAN.md, P4).
	InputPricePerMtok  float64 `json:"input_price_per_mtok,omitempty"`
	OutputPricePerMtok float64 `json:"output_price_per_mtok,omitempty"`
}

type Backend struct {
	BackendRegistration
	// What the runtime says it actually loaded, refreshed on every certification
	// rather than cached with the profile — see queryModelInfo. Never persisted:
	// only the registration survives a restart, so this is always re-measured.
	ModelMeta
	LastSeen       time.Time `json:"last_seen"`
	LastHealthy    time.Time `json:"last_healthy,omitempty"`
	Status         string    `json:"status"`
	Profiling      bool      `json:"profiling,omitempty"` // a cold-start profile is in flight (quality/capacity still provisional)
	Healthy        bool      `json:"healthy"`
	ActiveRequests int       `json:"active_requests"`
	// ContextProbe is the MEASURED usable window (contextprobe.go). It lives here
	// rather than on BackendRegistration on purpose: a registration is what a
	// worker CLAIMS about itself, and the whole point of this field is that it is
	// not a claim. Keeping it out also keeps registrations stable — upsert treats
	// a registration content change as a re-registration, which resets
	// certification and rebuilds the slot channel.
	ContextProbe *ContextProbe `json:"context_probe,omitempty"`
	// ProfileProgress is live only while a cold-start profile is running, and is
	// attached for display rather than stored. Nil at every other moment.
	ProfileProgress *ProfileProgressView `json:"profile_progress,omitempty"`
	// Outcomes is the measured hit rate and per-topic breakdown, attached for
	// display only. Never persisted and never read by routing: routing queries
	// the matrix for neighbours of the actual prompt, and compressing that into
	// a headline is precisely the mistake the quality scalar made.
	Outcomes    *BackendOutcomes `json:"outcomes,omitempty"`
	ObservedTPS float64          `json:"observed_tps,omitempty"` // live EWMA of DECODE throughput (TTFT excluded), BOTH modes pooled
	// Per-thinking-mode measurements. A model with thinking on and the same model
	// with it off are treated as two models here rather than assumed equivalent:
	// decode rate ought to be mode-independent and the generated LENGTH certainly
	// is not, but "ought to" is the kind of assumption that produced a q=84 worker
	// writing deterministic garbage, so both are measured separately and the
	// pooled figures above are kept only as a fallback for a mode not yet seen.
	//
	// ObservedGen* is the one that changes routing most: it is how many tokens
	// this backend actually emits, which drives decode time and is the single
	// largest term in how long a request takes. A greeting and a hard maths
	// question differ by ~50x on the same worker, and thinking-on and -off differ
	// by about as much again.
	ObservedTPSThink   float64 `json:"observed_tps_think,omitempty"`
	ObservedTPSNoThink float64 `json:"observed_tps_nothink,omitempty"`
	ObservedGenThink   float64 `json:"observed_gen_think,omitempty"`
	ObservedGenNoThink float64 `json:"observed_gen_nothink,omitempty"`
	ObservedTTFTMillis float64 `json:"observed_ttft_ms,omitempty"`     // live EWMA of first-token latency (prefill + queue), ms
	ObservedPrefillTPS float64 `json:"observed_prefill_tps,omitempty"` // live EWMA of PREFILL throughput (prompt tokens / TTFT), tok/s
	// ConcurrencyAlpha is the measured batching exponent from the capacity ramp:
	// aggregate throughput scales as agg(1)*n^alpha, so 1 is perfect batching and 0
	// is none. Routing prices a queued request with it (see loadPenalty). Zero
	// means unmeasured, which loadPenalty reads as 1 — no penalty — so a worker
	// profiled before the curve existed keeps exactly the load model it had.
	ConcurrencyAlpha float64 `json:"concurrency_alpha,omitempty"`
	ThinkingDialect  string  `json:"thinking_dialect,omitempty"` // measured spelling of the thinking gate (see WorkerProfile.ThinkingDialect)
	// RemoteActive is how many requests the UPSTREAM says are in flight against a
	// relay row's model, refreshed on every relay poll. It exists because
	// ActiveRequests counts only what this router dispatched, and the point of a
	// relay is that the other router is dispatching to the same GPUs — pricing a
	// remote fleet by our own share of its queue would prefer it precisely when
	// it is busiest. See relayOccupancy.
	RemoteActive int `json:"remote_active,omitempty"`
	// RelayRTTMillis is the smoothed round trip to the upstream router. It is
	// folded into the certified TTFT at import (see relayProfile) so a WAN hop is
	// priced from the very first request rather than only once this router has
	// measured its own latency; the copy here is what puts the number in front of
	// an operator on /backends. Zero on every non-relay row.
	RelayRTTMillis float64 `json:"relay_rtt_ms,omitempty"`
	// RejectedFields are request fields this endpoint has refused as
	// unrecognised; they are omitted from everything sent to it until the verdict
	// ages out (see stripAndRetry and rejectedFieldTTL). Learned at runtime and
	// deliberately not persisted, so a provider that starts accepting a field is
	// forgiven on the next restart as well.
	RejectedFields []string `json:"rejected_fields,omitempty"`
	QualityDetail  string   `json:"quality_detail,omitempty"` // benchmark per-tier + truncation breakdown
	// CategorySummary is the compact per-subject line `ask -l` renders. Kept as a
	// pre-rendered string rather than a structure because /backends is polled
	// every ten seconds by the dashboard: one short line per worker is affordable,
	// the full breakdown is not, and it already has a home on
	// GET /backends/{id}/benchmark.
	CategorySummary string `json:"category_summary,omitempty"`
	// QualityNoThink is the benchmark scored with thinking DISABLED on every
	// question — what a requirements.thinking="off" client is actually served
	// by. Zero means not measured (pre-two-score profile, or provisional): a
	// thinking worker then ranks below every measured worker on no-think
	// requests, while a non-thinking worker falls back to Quality, which for
	// it is exact. See qualityFor and WorkerProfile.QualityNoThink.
	QualityNoThink int `json:"quality_nothink,omitempty"`
	// QualityNoThinkDetail is the per-tier breakdown behind QualityNoThink, the
	// no-think counterpart of QualityDetail — what `ask -ls` renders as the
	// second bench line.
	QualityNoThinkDetail string    `json:"quality_nothink_detail,omitempty"`
	Failed               []string  `json:"failed_benchmarks,omitempty"` // benchmark questions the worker missed
	LastError            string    `json:"last_error,omitempty"`
	Certification        CertState `json:"certification"`

	// Scheduling/eviction bookkeeping. In-memory only — unexported so it is
	// never serialized into the API or dashboard payloads.
	certFailures  int       // consecutive certification failures; drives re-cert backoff
	nextCertifyAt time.Time // earliest time the health loop may re-certify
	proxyFailures int       // consecutive proxy failures; drives circuit-breaking
	// lastReg is the registration content as last received, used to tell a pure
	// keepalive (identical re-registration) from a real configuration change.
	// profileGen identifies the registration generation a profile was measured
	// against, so a profile finishing after a re-register/delete can't apply or
	// persist stale results. The pointed-to registration is never mutated.
	// lastModelCheck is when the health loop last fingerprinted the served
	// model — keepalives with identical content no longer re-certify, so a
	// model swap behind an unchanged registration is caught here instead.
	lastReg        *BackendRegistration
	profileGen     int64
	lastModelCheck time.Time
	// rejectedAt is when each RejectedFields entry was learned, so a verdict can
	// age out and be re-tested instead of standing for the life of the process
	// (see rejectedFieldTTL).
	rejectedAt map[string]time.Time
	// profileAborts counts consecutive background profiles discarded for this
	// registration; it drives the retry backoff and the point at which the router
	// stops paying for another attempt (see scheduleProfileRetry).
	profileAborts int
}

// nextProfileGen issues registration generations. Globally unique (not per
// backend) so a delete + re-register can't reuse a generation an in-flight
// profile still holds.
var nextProfileGen atomic.Int64

type CertState struct {
	Status       string           `json:"status"`
	Ready        bool             `json:"ready"`
	StartedAt    time.Time        `json:"started_at,omitempty"`
	FinishedAt   time.Time        `json:"finished_at,omitempty"`
	TTFTMillis   int64            `json:"ttft_ms,omitempty"`
	TokensPerSec float64          `json:"tokens_per_sec,omitempty"`
	Checks       map[string]Check `json:"checks,omitempty"`
	LastError    string           `json:"last_error,omitempty"`
}

type Check struct {
	OK      bool   `json:"ok"`
	Message string `json:"message,omitempty"`
}

type Registry struct {
	mu       sync.RWMutex
	backends map[string]*Backend
	// Per-backend concurrency slot channels, indexed by backend ID. A backend
	// with max_concurrency=N has a channel of capacity N pre-filled with N
	// tokens; callers acquire one via tryAcquireSlot before dispatching and put
	// it back with releaseSlot. A REGISTERED backend with no declared cap has no
	// entry, in which case tryAcquireSlot returns a nil slot that is always
	// available and releaseSlot is a no-op (unbounded).
	//
	// An absent entry is not enough on its own to mean "uncapped", because
	// remove() and upsert()'s uncapped branch delete the entry too — so
	// tryAcquireSlot checks the backends map first and refuses a worker that has
	// gone away rather than reading it as infinite capacity. See tryAcquireSlot.
	slots   map[string]chan struct{}
	slotCap map[string]int
}

type ChatRequest struct {
	Model    string          `json:"model,omitempty"`
	Messages []Message       `json:"messages"`
	Tools    json.RawMessage `json:"tools,omitempty"`
	Stream   bool            `json:"stream,omitempty"`
	// MaxTokens after parseAndValidateChatRequest is the EFFECTIVE completion
	// budget used for routing bookkeeping (context estimation, band selection):
	// the client's max_completion_tokens or max_tokens, else the router default.
	// Whether the client actually set one is in ClientSetMaxTokens — the
	// forwarded body is only patched when they didn't (see patchForwardedBody).
	MaxTokens           int  `json:"max_tokens,omitempty"`
	MaxCompletionTokens int  `json:"max_completion_tokens,omitempty"` // newer OpenAI name; wins over max_tokens
	ClientSetMaxTokens  bool `json:"-"`

	// ReasoningEffort is the OpenAI-standard thinking control, and the one a
	// coding harness (pi, hermes) emits without being taught anything about this
	// router. Absent ⇒ auto: the reasoning classifier decides, which is what every
	// clabtree guest gets. "none" ⇒ thinking off. Any other level is forwarded to
	// the worker as chat_template_kwargs.reasoning_effort next to
	// enable_thinking:true, which is what the DeepSeek V4 template branches on.
	ReasoningEffort string `json:"reasoning_effort,omitempty"`

	Requirements       *Requirements  `json:"requirements,omitempty"`
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
	// ClassifyText, when set, is the text the classifier scores instead of the
	// last user turn — the client's GENUINE user message, before any
	// agent-injected runtime context (date, memories, conversation summaries)
	// that would otherwise dominate the embedding and mis-score difficulty/
	// reasoning. Optional and additive: absent ⇒ fall back to classifyText over
	// the messages. Router-only; rides through to the worker like requirements
	// and is harmlessly ignored there. See classifyInput.
	ClassifyText string `json:"classify_text,omitempty"`
	// DeadlineMillis, when set, is how long the CALLER will still wait for this
	// response. Purely advisory routing input: workers that can't finish the job
	// inside it are deprioritised (see deadlineFilter), which turns "spend the
	// caller's whole timeout on a worker that was never going to make it" into a
	// correct placement. Go's net/http gives the server no client-timeout signal,
	// so this has to be declared rather than inferred. Router-only; rides through
	// to the worker and is harmlessly ignored there.
	DeadlineMillis int `json:"deadline_ms,omitempty"`
}

type Message struct {
	Role       string          `json:"role"`
	Content    any             `json:"content,omitempty"`
	Name       string          `json:"name,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
}

// Requirements carries only the orthogonal CAPABILITY hints a request may need
// (context size, features, thinking). There are deliberately no quality/speed/
// preference levers: model-tier selection is always automatic (difficulty-based,
// see selectBackends). Any such field an old client still sends is ignored.
type Requirements struct {
	MinContextK      int      `json:"min_context_k,omitempty"`
	Thinking         string   `json:"thinking,omitempty"`
	RequiredFeatures []string `json:"required_features,omitempty"`
}

// validationError is one northbound error. It marshals through writeJSON into
// the OpenAI envelope:
//
//	{"error":{"message":"…","type":"invalid_request_error","param":null,"code":null}}
//
// It used to serialise to a bare {"message":"…"}, which every client written
// against the standard reads as an empty error — the SDKs look under
// error.message and find nothing there. The envelope is applied by writeJSON
// from the status code rather than by each of the ~20 construction sites, so
// `validationError{Message: err.Error()}` still says everything they need to.
type validationError struct {
	Message string
	// Param names the offending request field when the router knows it. Absent ⇒
	// JSON null, which is what the standard shows.
	Param string
}

// errorEnvelope is the wire form of validationError.
type errorEnvelope struct {
	Error errorBody `json:"error"`
}

type errorBody struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

// envelope renders the error for an HTTP status. param/code stay null rather
// than being omitted: clients index into them, and a missing key reads
// differently from an explicit null in several SDKs.
func (e validationError) envelope(status int) errorEnvelope {
	body := errorBody{Message: e.Message, Type: errorTypeForStatus(status)}
	if e.Param != "" {
		param := e.Param
		body.Param = &param
	}
	return errorEnvelope{Error: body}
}

// errorTypeForStatus is the OpenAI error `type` for a status code. The names are
// theirs, not ours: a client that switches on the type string is switching on
// the vocabulary its SDK was written against.
func errorTypeForStatus(status int) string {
	switch status {
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusServiceUnavailable:
		// Distinct from server_error on purpose: a 503 here means the fleet is
		// busy or absent, which is retryable, and it is the one error the router
		// pairs with a Retry-After (see writeUnavailable).
		return "service_unavailable"
	}
	if status >= 500 {
		return "server_error"
	}
	return "invalid_request_error"
}

// retryAfterCeiling caps the Retry-After hint at a minute.
// ROUTER_SLOT_MAX_WAIT_SECONDS defaults to ten minutes, and telling a client to
// sleep ten minutes is not a back-off hint, it is a hang-up: past a minute a
// caller is better off re-queueing — where it is served the moment a slot frees
// — than sitting idle while the fleet drains.
const retryAfterCeiling = 60 * time.Second

// writeUnavailable answers a 503 with a Retry-After the router can actually
// justify. Every northbound 503 goes through here, so none can be added without
// one: a 503 with no hint tells a client nothing about whether to come back in a
// second or a minute, and clients that guess get it wrong in both directions.
//
// retry is a duration; the header carries whole seconds (its integer form —
// an HTTP-date buys nothing when the interval is what matters), rounded UP so a
// sub-second hint never renders as "0", which reads as "retry immediately".
func writeUnavailable(w http.ResponseWriter, retry time.Duration, message string) {
	secs := int64(1)
	if retry > time.Second {
		secs = int64(math.Ceil(retry.Seconds()))
	}
	w.Header().Set("Retry-After", strconv.FormatInt(secs, 10))
	writeJSON(w, http.StatusServiceUnavailable, validationError{Message: message})
}

// retryAfterUnavailable is the hint for "the router had nothing to serve this
// request": no healthy worker, none past the hard filters, a pinned worker that
// is not ready. What the router knows about the fleet only changes when the
// health loop next sweeps it — that is when a recovering worker is noticed, a
// certification finishes, an expired registration is written off — so one health
// interval is the soonest a retry can get a different answer. Sooner is a wasted
// round trip; later leaves the caller idle after the fleet has recovered.
func (r *Router) retryAfterUnavailable() time.Duration {
	if r.cfg == nil || r.cfg.HealthInterval <= 0 {
		return time.Second
	}
	return r.cfg.HealthInterval
}

// retryAfterSaturated is the hint for a request that found a worker but never
// got a slot: every candidate stayed busy for the whole slotMaxWait queue. That
// is a fleet which is oversubscribed rather than absent, and the evidence is
// exactly how long the caller just spent queueing — so the queue is the hint,
// capped (retryAfterCeiling) and floored at one health interval, below which a
// retry cannot see a changed fleet anyway.
func (r *Router) retryAfterSaturated() time.Duration {
	d := slotMaxWait
	if d > retryAfterCeiling {
		d = retryAfterCeiling
	}
	if floor := r.retryAfterUnavailable(); d < floor {
		d = floor
	}
	return d
}

// writeError sends an error in the OpenAI envelope with a JSON content type.
// It exists for the paths that used http.Error, which stamps text/plain over a
// JSON body — a combination that makes a strict client refuse to parse the very
// message explaining what it did wrong.
func writeError(w http.ResponseWriter, status int, format string, args ...any) {
	writeJSON(w, status, validationError{Message: fmt.Sprintf(format, args...)})
}

// unauthorized is the single 401 answer, for both the client and worker scopes.
// Every scope check uses it, so the wording and the shape cannot drift apart
// across twenty handlers.
func unauthorized(w http.ResponseWriter) {
	writeError(w, http.StatusUnauthorized, "unauthorized: send a bearer token in the Authorization header")
}

// notFound is the 404 for a path this router does not serve.
func notFound(w http.ResponseWriter, req *http.Request) {
	writeError(w, http.StatusNotFound, "no route for %s %s", req.Method, req.URL.Path)
}

type RequestLog struct {
	ID             int64     `json:"id"`
	CreatedAt      time.Time `json:"created_at"`
	BackendID      string    `json:"backend_id"`
	BackendModel   string    `json:"backend_model"`
	Route          string    `json:"route"`
	ObservedTPS    float64   `json:"observed_tps,omitempty"`
	CertifiedTPS   float64   `json:"certified_tps,omitempty"`
	BaselineTPS    float64   `json:"baseline_tps,omitempty"`
	SpeedScore     int       `json:"speed_score,omitempty"`
	Stream         bool      `json:"stream"`
	StatusCode     int       `json:"status_code"`
	DurationMillis int64     `json:"duration_ms"`
	// The four fields below exist to make request_logs usable as TRAINING DATA
	// for predicting how long a request will take, not just as an audit trail.
	//
	// Timing and correctness are separate problems with wildly different data:
	// correctness needs a graded answer and comes only from the benchmark (a few
	// hundred rows, hours of fleet time), while timing is observable on every
	// request for free — thousands a day, and unlike any fixed question bank they
	// are drawn from the traffic actually being served. Recording these makes the
	// cheap, plentiful half learnable.
	//
	// PromptTokens/CompletionTokens come from the endpoint's own usage block,
	// which was already being parsed for per-key budgeting and then discarded.
	// The learnable quantity is CompletionTokens: tok/s is already measured, so
	// what the router cannot currently predict is how MANY tokens a given prompt
	// will draw out of a given model — 200 for a greeting, 12000 for hard maths.
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	// TTFTMillis is time to first token, and is only meaningful on a STREAMED
	// reply — a buffered one arrives whole, so its first byte is its last and the
	// number would just restate DurationMillis. Zero means "not measured", not
	// "instant". It matters most at long context, where prefill dominates and
	// almost all of the latency lands before the first token.
	TTFTMillis int64 `json:"ttft_ms,omitempty"`
	// Thinking is the resolved enable_thinking decision: 1 on, 0 off, -1 unknown
	// (a passthrough route, or a request whose mode was never decided). Recorded
	// because a model with thinking on and the same model with it off are, for
	// timing purposes, two different models — same tok/s, output lengths that
	// differ by more than an order of magnitude. Averaging the two regimes
	// produces a prediction that describes neither.
	Thinking thinkingMode `json:"thinking"`
	// Concurrency is how many requests this router had in flight on the chosen
	// backend when this one was dispatched, INCLUDING this one. Without it
	// DurationMillis conflates "this model is slow" with "the fleet was busy",
	// and a timing model trained on it would learn to avoid whichever worker
	// happened to be popular.
	//
	// Always read after the in-flight count is incremented, so a recorded value
	// is at least 1 — which makes zero mean "not recorded" with no separate
	// sentinel, and keeps the struct's zero value honest.
	Concurrency int    `json:"concurrency"`
	Input       string `json:"input"`
	Output      string `json:"output"`
	Error       string `json:"error,omitempty"`
	// KeyID identifies the caller: the api_keys row id as a decimal string, "env"
	// for one of the bootstrap environment tokens, empty when the router required
	// no credential (every row written before P3, and a trusted-LAN deployment
	// with no tokens configured). See identity.logKeyID.
	KeyID string `json:"key_id,omitempty"`
	// ClientIP is the address the request arrived from, and answers a question
	// KeyID cannot: one shared key is one credential and often several machines.
	// Empty on every row written before the column existed and on anything whose
	// origin could not be read as an address — the usage graph draws those as
	// "unknown" rather than dropping them, because a fresh column on an old
	// database is otherwise indistinguishable from an idle router. See clientIP
	// for how it is derived and, more to the point, what it must not be used for.
	ClientIP string `json:"client_ip,omitempty"`
}

type LogStore struct {
	db      *sql.DB
	maxBody int        // cap on stored input/output/error length per row
	box     *secretBox // encrypts persisted backend api keys at rest
}

func Main() {
	// `discrimen bench …` builds the generated half of the quality benchmark
	// (see benchgen.go). It lives in this binary rather than a tools/ directory
	// so calibration grades with the production checkAnswer — a second copy of
	// the grader would diverge silently and mis-tier every question it touched.
	if runBenchCommand(os.Args) {
		return
	}
	// `discrimen arena …` measures the ROUTER against a graded dataset (see
	// arena.go). Same reasoning as bench: it grades with the production
	// checkAnswer, so it lives in this binary rather than beside it.
	if runArenaCommand(os.Args) {
		return
	}
	// `discrimen prices fetch` refreshes the embedded LiteLLM price snapshot (see
	// pricing.go). Developer command, run from a checkout, never in the container.
	if runPricesCommand(os.Args) {
		return
	}
	// `discrimen snapshot DEST` writes a VACUUM INTO copy of the request log
	// database (see snapshot.go). Unlike the three above this one IS run in the
	// container: the deployment template's backup.sh calls it so restic captures
	// a consistent database rather than a live one read mid-write.
	if runSnapshotCommand(os.Args) {
		return
	}
	// Everything else in argv[1] is a typo, not a server invocation. Without
	// this the server starts and silently ignores the argument — which is how
	// `discrimen snapshot …` run against an image too old to know the subcommand
	// turns a backup step into a second router running until something kills it.
	// Leading dashes are left alone so the server can grow flags later.
	if len(os.Args) > 1 && !strings.HasPrefix(os.Args[1], "-") {
		fmt.Fprintf(os.Stderr, "discrimen: unknown command %q\nCommands: bench, arena, prices, snapshot\n", os.Args[1])
		os.Exit(2)
	}

	cfg := loadConfig()
	registry := &Registry{
		backends: make(map[string]*Backend),
		slots:    make(map[string]chan struct{}),
		slotCap:  make(map[string]int),
	}
	logs, err := openLogStore(cfg.LogDBPath, cfg.LogMaxBodyBytes, cfg.PersistSecret)
	if err != nil {
		log.Fatalf("open request log database: %v", err)
	}
	defer logs.Close()

	router := &Router{cfg: cfg, registry: registry, client: &http.Client{Timeout: cfg.BackendHTTPTimeout}, streamClient: &http.Client{}, benchClient: &http.Client{}, logs: logs}
	// The outcome matrix shares the request log's database — same file, same
	// single connection — so a profile's results and the requests that produced
	// them are backed up and restored together. Loading is best effort: an
	// unreadable matrix means predictions fall back to "unknown", which routes
	// on overall hit rate and speed, whereas refusing to start would take the
	// fleet down over a cache.
	router.outcomes = newOutcomeMatrix(logs.db)
	if err := router.outcomes.load(context.Background()); err != nil {
		log.Printf("outcome matrix: load failed, starting empty: %v", err)
	} else {
		log.Printf("outcome matrix: %s", router.outcomes)
	}
	// Reconstruct observations from profiles already on disk.
	//
	// Observations used to be written in exactly one place — the moment a profile
	// COMPLETED, in this process — so a router that restarted after a
	// benchmarkVersion bump ran with an empty matrix for the length of a full
	// fleet re-profile, hours during which every request took the fallback and
	// routed on speed alone with no correctness input. Measured on the live fleet
	// mid-deploy: "0 questions, 392 vectors, 0 observations", while a completed
	// 392-result profile representing 194 minutes of measurement sat unread in the
	// same database.
	//
	// The evidence was never missing, only unreachable: worker_profiles stores
	// BenchResults in the exact shape observationsFromMixed consumes. This turns
	// hours of blind routing into a startup read. Best effort, like load() — a
	// failed backfill costs prediction quality, and refusing to start would take
	// the fleet down over a cache.
	if err := router.backfillOutcomesFromProfiles(context.Background()); err != nil {
		log.Printf("outcome matrix: backfill from stored profiles failed: %v", err)
	} else {
		log.Printf("outcome matrix after backfill: %s", router.outcomes)
	}
	// Vectors are derived, not stored, so a restart starts with none. Kick the
	// fill off now rather than waiting for the first request to notice.
	router.ensureBankVectorsAsync()
	// Before anything is served: generate the credentials the operator did not
	// supply and print them once. An empty ROUTER_CLIENT_TOKENS used to mean no
	// client authentication at all, which is right on a trusted LAN and wrong for
	// a public image — .env.example already promises operators this behaviour.
	//
	// Failing here is fatal, and deliberately so: this step is also what switches
	// the credential checks ON (refreshAuthRequired runs at the end of it), so a
	// router that logged the error and carried on would serve with client AND
	// worker authentication off — an open fleet, the exact failure the bootstrap
	// exists to prevent. The usual cause is a data volume that is read-only or
	// full, so the message says which volume and what to do with it.
	if err := router.bootstrapCredentials(context.Background()); err != nil {
		log.Fatalf("bootstrap credentials: %v\n"+
			"  discrimen will not start without them: this step is what switches the credential checks on, "+
			"so serving anyway would leave the fleet open to anyone who can reach the port.\n"+
			"  Check that %s is on a writable volume with free space, then restart.", err, cfg.LogDBPath)
	}
	if cfg.AutoDifficulty || cfg.AutoThinking {
		router.classifier = newDifficultyClassifier(cfg, router.embedTexts)
		log.Printf("prompt auto-routing enabled (difficulty=%v bands=%q, thinking=%v threshold=%.2f)",
			cfg.AutoDifficulty, cfg.DifficultyBands, cfg.AutoThinking, cfg.ReasoningThreshold)
		go router.warnIfNoEmbeddings()
	}
	if cfg.JudgeSampleRate > 0 {
		router.judgeSem = make(chan struct{}, judgeMaxConcurrent)
		log.Printf("background answer judging enabled (sample rate=%.2f, %d concurrent)", cfg.JudgeSampleRate, judgeMaxConcurrent)
	}
	if sessionSticky {
		router.sessions = newSessionTracker(sessionTTL, sessionMax)
		log.Printf("session affinity enabled (ttl=%s max=%d prefill_discount=%.2f lock_wait=%s)",
			sessionTTL, sessionMax, sessionPrefillDiscount, sessionLockWait)
	}
	if cfg.EscalateInline {
		log.Printf("inline escalation enabled (empty non-streamed answers are re-dispatched to a better worker)")
	}
	if cfg.RescueTruncated {
		log.Printf("length rescue enabled (a worker that spends its whole budget thinking is asked for its conclusion)")
	}
	router.loadGroups(context.Background())
	router.loadRelays(context.Background())
	if saved, err := logs.LoadBackendRegistrations(context.Background()); err != nil {
		log.Printf("load persisted backend registrations failed: %v", err)
	} else {
		for _, reg := range saved {
			if err := normalizeRegistration(&reg); err != nil {
				log.Printf("skip persisted backend registration id=%q: %v", reg.ID, err)
				continue
			}
			backend, _ := registry.upsert(reg)
			go router.certifyBackend(backend.ID)
		}
		if len(saved) > 0 {
			log.Printf("loaded %d persisted backend registrations", len(saved))
		}
	}

	go router.healthLoop()
	go router.logRetentionLoop()
	// Relay rows are DERIVED, never persisted: the refresh below is what creates
	// them, and it runs its first pass immediately rather than on the first tick
	// so a restart does not leave the relayed half of the fleet missing.
	go router.relayRefreshLoop()

	server := &http.Server{
		Addr:              ":" + cfg.Port,
		Handler:           logRequests(router.routes()),
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("discrimen listening on :%s", cfg.Port)
	log.Fatal(server.ListenAndServe())
}

type Router struct {
	cfg          *Config
	registry     *Registry
	client       *http.Client
	streamClient *http.Client // no client-level timeout — streaming proxies are bounded by an IDLE watchdog (BackendIdleTimeout), not wall clock, so a legit multi-minute generation isn't killed mid-stream while a silent backend still frees its slot quickly
	benchClient  *http.Client // no client-level timeout — the cold-start benchmark bounds each request by benchAnswerDeadline via context instead
	logs         *LogStore
	classifier   *difficultyClassifier // nil unless auto difficulty routing is enabled
	// outcomes is the measured record of which worker answered which question
	// correctly, and how fast — the thing routing queries instead of comparing a
	// quality percentage against a difficulty score. Nil disables it, in which
	// case routing falls back to the older quality-target path. See outcomes.go.
	outcomes *outcomeMatrix
	// bankTopics maps a bank question id to its display category, built once and
	// reusing benchCategoryOf so the summary and the older per-category
	// breakdown cannot disagree about what counts as coding.
	bankTopics    map[string]string
	bankTopicOnce sync.Once
	// bankVecsReady/bankVecsFilling guard the background embedding of the question
	// bank. Not a sync.Once: the embeddings worker may be down at startup, and a
	// latched Once would leave the matrix dead for the process lifetime.
	bankVecsReady   atomic.Bool
	bankVecsFilling atomic.Bool
	judgeSem        chan struct{}   // bounds concurrent background judge calls; nil unless judging enabled
	judgeCount      atomic.Uint64   // sample counter for background answer judging
	judgePaid       judgeBudget     // rolling token allowance for grading with a PAID model (see judge.go)
	profiling       sync.Map        // worker ids with a cold-start profile in flight (de-dups overlapping profiles)
	gateAudited     atomic.Bool     // set once the thinking-gate-vs-tier audit has logged (model-independent)
	sessions        *sessionTracker // conversation → worker affinity; nil disables stickiness (see session.go)

	// profileMeters holds a *profileMeter per worker id for the span of that
	// worker's profiling run, so what the run consumed can be totalled onto the
	// profile it produces. Empty at every other moment (see profile.go).
	profileMeters sync.Map

	// adminAuth holds live admin password sessions. A value, not a pointer, so
	// the zero Router is usable — every test that builds one by hand would
	// otherwise have to remember to construct it, and forgetting would be a nil
	// dereference on the login path rather than a compile error.
	adminAuth adminSessions
	// groups holds the named preference lists a client can select through the
	// model field (see groups.go). A value for the same reason as adminAuth.
	groups groupStore
	// relays holds the upstream routers this one may route through, and the last
	// fleet each reported (see relay.go). A value for the same reason as groups.
	relays relayStore
	// usage caches the last render of the dashboard's usage chart. A value for
	// the same reason as groups. See usageCache.
	usage usageCache
	// relayRefresh serialises the reconcile pass, which the ticker and both admin
	// write paths can all start. See refreshRelays.
	relayRefresh sync.Mutex
	// selfID is this router's own identity, generated once and persisted, used to
	// recognise a relay chain that has already been here. Lazily resolved, so a
	// Router built by hand in a test needs no setup.
	selfID     atomic.Value
	selfIDOnce sync.Once
	// Whether a client / worker credential is REQUIRED, cached from the keys
	// table so the check is not a database read per request. Refreshed at startup
	// and after every write to that table (see refreshAuthRequired).
	clientKeysExist atomic.Bool
	workerKeysExist atomic.Bool
	// keyStamped throttles the last_used_at write for calls that spend no tokens
	// (see keyUseDue). key id → time.Time.
	keyStamped sync.Map
}

// Routing constants. These were environment variables. They are not any more.
//
// The test for whether a setting deserves to be a variable is whether it
// describes something only the operator can know. Hardware, network, ports,
// credentials, retention and how long a caller is willing to queue are facts
// about a site. Learning rates, classifier thresholds, latency estimates and
// tier bands are not: they are the router's own decisions, and a site that has
// to set them has been handed the problem the router exists to solve.
const (
	// difficultyTemp is the softness of the simple-vs-hard margin → score map.
	difficultyTemp = 0.10
	// difficultyCacheSize bounds the prompt → score cache.
	difficultyCacheSize = 4096
	// difficultyMaxChars caps how much of a prompt is classified. Past this the
	// centroid distance stops moving and the embedding call just costs more.
	difficultyMaxChars = 4000
	// reasoningThreshold is the reasoning score at or above which thinking is
	// enabled.
	reasoningThreshold = 0.35

	// judgeSampleRate is the fraction of SERVED answers graded in the background
	// by another worker in the fleet. This is the signal that makes a fast-but-dim
	// worker safe to route to, so it is on whenever auto-routing is.
	//
	// Every served answer is eligible, and the grader is chosen per PROMPT rather
	// than being the fleet's best model (see judgeGrader). It used to be "the
	// fraction of cheaper-than-best answers, graded by the best model", which was
	// gated on the retired Quality scalar and meant the strongest worker was never
	// graded at all — see Config.JudgeSampleRate.
	judgeSampleRate = 0.2

	// capacityProbeMax caps the concurrency ramp during profiling. A declared
	// max_concurrency on the registry row outranks it, which is better targeted
	// than a global ceiling.
	capacityProbeMax = 16

	// difficultyTimeoutFallback is the classifier deadline used before the
	// embeddings worker has been measured. Once it has, the deadline is derived
	// from its observed latency instead: a fixed two seconds silently disabled
	// classification on a slow box, and the health endpoint still reported the
	// worker present, so the failure was invisible.
	difficultyTimeoutFallback = 2 * time.Second
)

// defaultMaxTokensFallback is the budget a client that declares none gets.
//
// 16384, not 4096, and the difference is not a comfort margin. On a hybrid
// reasoning worker a thinking block cut off at the limit returns
// finish_reason=length with an EMPTY content field — no answer at all, not a
// short one. Measured 2026-08-20 on llm-6000pro-qwen38-27b-ninfer: 48 requests
// at 4096, 4.2% came back empty, every one a counting question whose chain ran
// past the cap. Those empties then look like a WORKER problem to the escalation
// path, which re-dispatches them to a better model, so a budget too small to
// think in gets diagnosed as inadequate quality.
//
// It was also chosen to MATCH benchThinkMaxTokens, which was 16384 at the time:
// at 4096 ordinary traffic was getting a quarter of the budget this router's own
// benchmark requires before it will judge a model competent. The two have since
// parted company — benchThinkMaxTokens went to 32768 so that the answer DEADLINE
// binds a graded question rather than a token ceiling (see benchmark.go, v40) —
// so ordinary traffic now gets half the benchmark's budget rather than all of it.
// That is a deliberate difference and not drift: a benchmark question may run to
// its deadline, while a client that declared no budget at all should not.
//
// The trade, from the same measurement: this does not fix a true runaway. It
// converts the chains that would have concluded between 4k and 16k into
// answers and makes the rest fail slower (~50s to ~3min of silence). On a slow
// worker an unbounded client's worst case is now governed by
// ROUTER_SLOT_MAX_WAIT_SECONDS rather than by this cap.
const defaultMaxTokensFallback = 16384

func loadConfig() *Config {
	// Clamp the health interval to >=1s: time.NewTicker(0) panics, so a
	// misconfigured HEALTH_INTERVAL_SECONDS=0 (or negative) must not reach it.
	healthSecs := envInt("HEALTH_INTERVAL_SECONDS", 15)
	if healthSecs < 1 {
		log.Printf("HEALTH_INTERVAL_SECONDS=%d is invalid; clamping to 1", healthSecs)
		healthSecs = 1
	}

	// One switch for the whole automatic layer: difficulty scoring, thinking
	// detection, online adaptation and background judging. Off turns discrimen
	// into a plain load balancer with no embeddings dependency, which is a
	// legitimate thing to want and the only reason to touch any of this.
	//
	// On by default. It used to be off, with only the deployment template
	// turning it on, so a bare `docker run` gave you the least interesting
	// version of the product.
	auto := envBool("ROUTER_AUTO_ROUTING", true)
	judge := 0.0
	if auto {
		judge = judgeSampleRate
	}

	return &Config{
		Port:               envOr("ROUTER_PORT", "8585"),
		WorkerToken:        os.Getenv("ROUTER_WORKER_TOKEN"),
		ClientTokens:       parseTokenList(os.Getenv("ROUTER_CLIENT_TOKENS")),
		AdminPassword:      os.Getenv("ROUTER_ADMIN_PASSWORD"),
		DefaultMaxTokens:   envInt("DEFAULT_MAX_TOKENS", defaultMaxTokensFallback),
		HealthInterval:     time.Duration(healthSecs) * time.Second,
		BackendHTTPTimeout: time.Duration(envInt("BACKEND_TIMEOUT_SECONDS", 600)) * time.Second,
		BackendIdleTimeout: time.Duration(envInt("BACKEND_IDLE_TIMEOUT_SECONDS", 120)) * time.Second,
		// Frozen path. The deployment template derives its Docker volume name
		// from the container name and this path lives inside it; moving either
		// orphans the volume, which takes worker_profiles with it and
		// re-benchmarks the whole fleet from cold.
		LogDBPath:       envOr("LOG_DB_PATH", "/data/llm-router/logs.sqlite"),
		LogRetention:    time.Duration(envInt("LOG_RETENTION_DAYS", 30)) * 24 * time.Hour,
		LogMaxBodyBytes: envInt("LOG_MAX_BODY_BYTES", 16384),
		PersistSecret:   os.Getenv("ROUTER_PERSIST_SECRET"),

		AutoDifficulty: auto,
		// Always fleet-derived. A hand-set band table is a claim about which
		// worker is good enough for which prompt, which is the measurement the
		// router already makes.
		DifficultyBands:     "",
		DifficultyTemp:      difficultyTemp,
		DifficultyTimeout:   difficultyTimeoutFallback,
		DifficultyCacheSize: difficultyCacheSize,
		DifficultyMaxChars:  difficultyMaxChars,

		AutoThinking:       auto,
		ReasoningThreshold: reasoningThreshold,

		JudgeSampleRate: judge,

		ProfileWorkers:   true,
		CapacityProbeMax: capacityProbeMax,

		// Empty disables code-exec grading entirely rather than failing those
		// questions: a deployment without the sidecar should score the questions
		// it can grade, not record every coding question as a wrong answer.
		SandboxURL:   strings.TrimSpace(os.Getenv("SANDBOX_URL")),
		SandboxToken: strings.TrimSpace(os.Getenv("SANDBOX_TOKEN")),

		EscalateInline:  true,
		RescueTruncated: true,
	}
}

// parseTokenList splits ROUTER_CLIENT_TOKENS on commas, trims, drops empties.
// Returns nil for an empty input — used by authorizedAsClient to mean
// "no client auth required".
func parseTokenList(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// routes wires the northbound and registry surfaces onto a mux. Separate from
// Main so the pattern set itself is testable: "/v1/models" is an exact match and
// "/v1/models/" a subtree, and getting that pair wrong is not a compile error —
// it is a single-model fetch quietly falling through to the dashboard handler
// and 404ing in HTML, which is exactly what it did.
func (r *Router) routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/", r.handleDashboard)
	mux.HandleFunc("/health", r.handleHealth)
	mux.HandleFunc("/v1/health", r.handleHealth)
	mux.HandleFunc("/backends", r.handleBackends)
	mux.HandleFunc("/backends/register", r.handleRegisterBackend)
	mux.HandleFunc("/backends/", r.handleBackendByID)
	mux.HandleFunc("/workers", r.handleBackends)
	mux.HandleFunc("/workers/register", r.handleRegisterBackend)
	mux.HandleFunc("/workers/", r.handleBackendByID)
	mux.HandleFunc("/v1/models", r.handleModels)
	mux.HandleFunc("/v1/models/", r.handleModelByID)
	mux.HandleFunc("/v1/chat/completions", r.handleChatCompletions)
	mux.HandleFunc("/chat/completions", r.handleChatCompletions)
	mux.HandleFunc("/v1/completions", r.handleCompletions)
	mux.HandleFunc("/completions", r.handleCompletions)
	mux.HandleFunc("/v1/embeddings", r.handleEmbeddings)
	mux.HandleFunc("/embeddings", r.handleEmbeddings)
	mux.HandleFunc("/logs", r.handleLogs)
	// The other view of the same table, kept beside it rather than down with the
	// admin block: /logs is unprefixed only because the compatibility contract
	// froze it there, and anything new is admin-scoped under /admin.
	mux.HandleFunc("/admin/usage", r.handleUsage)
	mux.HandleFunc("/admin/outcomes", r.handleOutcomes)
	mux.HandleFunc("/v1/route-preview", r.handleRoutePreview)
	mux.HandleFunc("/route-preview", r.handleRoutePreview)
	mux.HandleFunc("/debug/backends/", r.handleDebugBackends)
	// The admin write surface. Separate from /backends on purpose: that path is
	// frozen by the compatibility contract, and a row created here is
	// operator-owned in a way a pushed registration never is.
	mux.HandleFunc("/admin/providers", r.handleAdminProviders)
	mux.HandleFunc("/admin/providers/", r.handleAdminProviderByID)
	mux.HandleFunc("/admin/keys", r.handleAdminKeys)
	mux.HandleFunc("/admin/keys/", r.handleAdminKeyByID)
	mux.HandleFunc("/admin/groups", r.handleAdminGroups)
	mux.HandleFunc("/admin/groups/", r.handleAdminGroupByName)
	mux.HandleFunc("/admin/relays", r.handleAdminRelays)
	mux.HandleFunc("/admin/relays/", r.handleAdminRelayByName)
	// What a downstream router may see of this fleet. Client scope plus the relay
	// flag — see handleRelayFleet for why an ordinary client key is not enough.
	mux.HandleFunc("/relay/fleet", r.handleRelayFleet)
	// The password session. /admin/login and /admin/session are the only two
	// unauthenticated routes under /admin, and neither discloses anything: login
	// answers the same 401 for "wrong password" and "no password is set", and
	// session reports one boolean about the caller's own cookie.
	mux.HandleFunc("/admin/login", r.handleAdminLogin)
	mux.HandleFunc("/admin/logout", r.handleAdminLogout)
	mux.HandleFunc("/admin/session", r.handleAdminSession)
	mux.HandleFunc("/admin/password", r.handleAdminPassword)
	return mux
}

func (r *Router) handleHealth(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	backends := r.registry.snapshot()
	healthy := 0
	for _, b := range backends {
		if b.Healthy && !isExpired(b) {
			healthy++
		}
	}
	resp := map[string]any{
		"status":           "ok",
		"healthy_backends": healthy,
		"total_backends":   len(backends),
	}
	// Auto-routing requires an embeddings worker (the difficulty + thinking
	// classifiers embed through it); surface its absence instead of silently
	// degrading to plain quality/speed routing.
	if r.classifier != nil {
		if r.registry.hasBackendWithFeature("embeddings") {
			resp["auto_routing"] = "ok"
		} else {
			resp["auto_routing"] = "degraded: no embeddings worker registered — difficulty/thinking classification disabled"
		}
		// The classification deadline is derived from the embeddings worker's
		// measured latency (see observeEmbedLatency), so it is a number an operator
		// cannot predict and needs to be able to read. It stays at
		// difficultyTimeoutFallback until that worker has been certified once.
		resp["classifier_deadline"] = r.classifier.deadline().String()
	}
	// The relayed half of the fleet, when there is one. A benchmark-version
	// mismatch in particular is the one relay fault that leaves everything
	// apparently working, so it belongs somewhere a monitor reads (see
	// relayHealthLine).
	if line := r.relayHealthLine(); line != nil {
		resp["relays"] = line
	}
	writeJSON(w, http.StatusOK, resp)
}

// warnIfNoEmbeddings loudly and repeatedly warns when auto-routing is enabled
// but no embeddings worker is registered — the classifiers can't run without
// one, so routing silently falls back to plain quality/speed selection.
func (r *Router) warnIfNoEmbeddings() {
	time.Sleep(20 * time.Second) // grace for workers to register on startup
	for {
		if !r.registry.hasBackendWithFeature("embeddings") {
			log.Printf("WARNING: auto-routing enabled but NO embeddings worker registered — " +
				"difficulty + thinking classification disabled (routing falls back to quality/speed). " +
				"Deploy llm-embeddings pointed at this router.")
		}
		time.Sleep(5 * time.Minute)
	}
}

// handleBackends lists the fleet. ADMIN scope since P3: the payload is every
// worker's id, URL, model, measured quality, load and last error, which is a map
// of the operator's infrastructure. A client that needs to know what it can ask
// for reads /v1/models, which publishes the menu and not the estate.
func (r *Router) handleBackends(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	if !r.requireAdmin(w, req) {
		return
	}
	pub := publicBackends(r.registry.snapshot())
	for _, b := range pub {
		if v, ok := r.profiling.Load(b.ID); ok {
			b.Profiling = true // background cold-start profile still running → values provisional
			// ...and WHAT it is doing. A profile is the longest thing this router
			// runs, and publishing only a boolean left "is it stuck?" unanswerable
			// without reading container logs.
			if pp, ok := v.(*ProfileProgress); ok {
				b.ProfileProgress = pp.snapshot()
			}
		}
		// The measured hit rate and its per-topic breakdown, which is what
		// replaces the quality score on every display surface. Attached here
		// rather than stored on the Backend because it is derived from the matrix
		// on demand — storing it would create a second copy that could disagree
		// with the rows routing actually queries.
		if r.outcomes != nil {
			b.Outcomes = &BackendOutcomes{
				Thinking: r.outcomeSummaryFor(b, true),
				NoThink:  r.outcomeSummaryFor(b, false),
			}
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"backends": pub})
}

// serveBenchmark returns the stored per-question results (questions, expected
// answers, and the model's actual answers) from a worker's most recent profiling
// run: GET /backends/{id}/benchmark. Read-only, ADMIN scope since P3 — it is the
// grading set and one worker's answers to it, which is both fleet detail and the
// benchmark's own answer key.
//
// It also carries the per-CATEGORY breakdown (benchcategory.go), which is the
// same run grouped by what each question measures rather than by how hard it is.
// Derived here on request rather than stored on the profile: it is a pure
// function of results the profile already holds, so computing it costs a walk
// over ~400 records and cannot go stale, where a stored copy would survive a
// change to the categorisation and quietly contradict it.
//
// One endpoint rather than a second one beside it, because the dashboard already
// fetches this per backend for the profile-cost cell — a /categories route would
// double the round trips to show two halves of one run.
func (r *Router) serveBenchmark(w http.ResponseWriter, req *http.Request, id string) {
	if req.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	if !r.requireAdmin(w, req) {
		return
	}
	prof, ok := r.logs.LoadWorkerProfileByID(req.Context(), id)
	if !ok {
		writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("no stored profiling run for %q", id)})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":              id,
		"model":           prof.Model,
		"quality":         prof.Quality,
		"quality_nothink": prof.QualityNoThink,
		"thinking":        prof.Thinking,
		"bench_version":   prof.BenchVersion,
		"measured_at":     prof.MeasuredAt,
		"profile_ms":      prof.ProfileMillis,
		"profiled_in":     fmtProfileDuration(prof.ProfileMillis),
		// The two per-tier lines the benchmark logs. They stay even though
		// "categories" supersedes them for most reading: a tier line is what the
		// CLI prints and what the router's own log holds, so it is what an
		// operator will be comparing against.
		"quality_detail":         prof.QualityDetail,
		"quality_nothink_detail": prof.QualityNoThinkDetail,
		"categories":             benchCategoryBreakdown(prof.BenchResults, prof.BenchResultsNoThink),
		// Whether the no-think half of that breakdown exists at all. A consumer
		// must not read a missing per-category no-think score as a zero — see
		// WorkerProfile.BenchResultsNoThink — so say which it is rather than
		// leaving it to be inferred from an absent field.
		"nothink_results_stored": len(prof.BenchResultsNoThink) > 0 &&
			len(prof.BenchResultsNoThink) == len(prof.BenchResults),
		// What the run cost. Omitted entirely for a profile measured before the
		// accounting existed, so a UI renders nothing rather than a confident zero
		// (see WorkerProfile.ProfilePromptTokens).
		"profile_prompt_tokens": prof.ProfilePromptTokens,
		"profile_output_tokens": prof.ProfileOutputTokens,
		"profile_cost":          prof.ProfileCost,
		"profile_cost_measured": prof.ProfilePromptTokens > 0 || prof.ProfileOutputTokens > 0,
		"results":               prof.BenchResults,
	})
}

// handleBackendByID serves DELETE /backends/{id} (and /workers/{id}). A live
// backend with status="ready" is refused — only stale, expired, unhealthy or
// failed-certification entries can be cleared — unless ?force=true is passed,
// which clears it regardless (memory + persisted row). A real backend that's
// still running its keepalive will re-register on its next cycle.
func (r *Router) handleBackendByID(w http.ResponseWriter, req *http.Request) {
	id := strings.TrimPrefix(req.URL.Path, "/backends/")
	id = strings.TrimPrefix(id, "/workers/")
	id = strings.Trim(id, "/")
	if id == "" {
		notFound(w, req)
		return
	}
	// GET /backends/{id}/benchmark — stored per-question results from the most
	// recent profiling run (questions, expected answers, and the model's answers).
	if strings.HasSuffix(id, "/benchmark") {
		r.serveBenchmark(w, req, strings.TrimSuffix(id, "/benchmark"))
		return
	}
	switch req.Method {
	case http.MethodDelete:
		// DELETE accepts a worker credential (a worker de-registering itself on
		// shutdown) or admin (an operator clearing a stale entry). It used to
		// accept any CLIENT token, which since P3 is a stranger's credential and
		// has no business evicting workers. Live ready+healthy backends are still
		// refused below.
		if !r.workerAuthorized(req) && !r.adminAuthenticated(req) {
			unauthorized(w)
			return
		}
		b := r.registry.get(id)
		if b == nil {
			writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("backend %q not found", id)})
			return
		}
		force, _ := strconv.ParseBool(req.URL.Query().Get("force"))
		if strings.EqualFold(b.Status, "ready") && b.Healthy && !isExpired(b) {
			if !force {
				writeJSON(w, http.StatusConflict, validationError{Message: fmt.Sprintf("backend %q is ready — refusing to delete a live backend (pass force=true to override)", id)})
				return
			}
			log.Printf("force-deleting live backend %q (status=%s healthy=%t); it will re-register on its next keepalive", id, b.Status, b.Healthy)
		}
		removed := r.registry.remove(id)
		if !removed {
			writeJSON(w, http.StatusNotFound, validationError{Message: fmt.Sprintf("backend %q not found", id)})
			return
		}
		// Also drop the persisted row, or a restart's LoadBackendRegistrations
		// resurrects it. Best-effort: the in-memory removal already succeeded, so
		// don't fail the request on a persist error.
		if err := r.logs.DeleteBackendRegistration(req.Context(), id); err != nil {
			log.Printf("delete persisted registration for %q: %v", id, err)
		}
		if err := r.logs.DeleteWorkerProfile(req.Context(), id); err != nil {
			log.Printf("delete worker profile for %q: %v", id, err)
		}
		// The matrix rows deliberately DO NOT go with the profile.
		//
		// They are filed under the model, not the worker, so deleting a worker is
		// not evidence about the weights it was running. If it comes back with a
		// different model the new hash simply does not match the old rows; if it
		// comes back with the SAME model, those rows are still true and the
		// benchmark reuses them instead of spending hours re-deriving answers it
		// already has. Deleting a worker to force a hardware re-probe used to cost
		// a full re-grade; now it costs a speed, capacity and context probe.
		//
		// This is what "decommission a worker and keep the results" means, and it
		// is the reason the rows are keyed by model hash at all.
		writeJSON(w, http.StatusOK, map[string]any{"status": "removed", "id": id})
	case http.MethodGet:
		// Admin scope, same reasoning as the /backends list it is a row of.
		if !r.requireAdmin(w, req) {
			return
		}
		b := r.registry.get(id)
		if b == nil {
			notFound(w, req)
			return
		}
		// Never expose the worker's bearer key (the /backends list scrubs it the
		// same way — see publicBackends — and it's encrypted at rest).
		b.APIKey = ""
		writeJSON(w, http.StatusOK, b)
	default:
		methodNotAllowed(w)
	}
}

// handleLogs serves the request log. ADMIN scope since P3, and the single most
// important of the moves: the rows carry every stored prompt and response, so a
// client token here reads every other client's traffic. That was acceptable for
// a private fleet and stops being so the moment a token goes to someone else.
func (r *Router) handleLogs(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	if !r.requireAdmin(w, req) {
		return
	}
	limit := envBoundedInt(req.URL.Query().Get("limit"), 100, 1, 500)
	offset := envBoundedInt(req.URL.Query().Get("offset"), 0, 0, 1000000)
	backendID := strings.TrimSpace(req.URL.Query().Get("backend_id"))
	rows, err := r.logs.List(req.Context(), backendID, limit, offset)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"logs": rows})
}

// usageCache holds the last render of the usage chart, and exists because of one
// number: this query costs about 150ms over twenty thousand log rows and about
// 1.2 SECONDS over two hundred thousand, and the SQLite pool is capped at a
// single connection. A page polling every ten seconds would hand a fifth of that
// connection to a chart on a busy router — and every open browser tab would want
// its own copy.
//
// The lifetime is derived from what the query cost rather than fixed, because
// the two deployments want opposite things. A quiet router answers in
// milliseconds and should have a genuinely live chart; a router with a
// quarter-million rows in the window should not be re-deriving them on every
// tick, and would rather show a figure that is half a minute old. Twenty times
// the query's own cost holds the database's share at about five percent either
// way: ~3s at 150ms, so a ten-second poll is never served stale, and ~24s at
// 1.2s, where the chart visibly settles into updating less often.
//
// The cached frame is keyed on its end, so it also expires the instant the
// five-minute window rolls over. A cache that could pin the chart to a stale
// window would be far worse than a slow one.
type usageCache struct {
	mu     sync.Mutex
	series *UsageSeries
	until  time.Time
}

const (
	usageCacheRatio = 20
	usageCacheMin   = 2 * time.Second
	usageCacheMax   = time.Minute
)

func (c *usageCache) get(frameEnd int64, now time.Time) *UsageSeries {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.series == nil || c.series.To != frameEnd || now.After(c.until) {
		return nil
	}
	return c.series
}

func (c *usageCache) put(series *UsageSeries, now time.Time, cost time.Duration) {
	ttl := cost * usageCacheRatio
	if ttl < usageCacheMin {
		ttl = usageCacheMin
	}
	if ttl > usageCacheMax {
		ttl = usageCacheMax
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.series = series
	c.until = now.Add(ttl)
}

// handleUsage answers the dashboard's usage chart: the last twelve hours of
// request logs as mean concurrency per five-minute bucket, split by the address
// each request came from.
//
// Admin scope, for the same reason /logs is. This is the request log with the
// bodies taken out, and the set of addresses talking to a router is not less
// sensitive than the requests themselves — it is a map of who runs what and
// from where, and on a public deployment it is the reconnaissance half of the
// log without the volume that makes the log tedious to read.
//
// The window and the bucket are not query parameters. There is one chart, it
// answers one question, and an operator who wants a different window has the log
// table; a knob here would only mean two callers disagreeing about what the
// numbers mean.
func (r *Router) handleUsage(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	if !r.requireAdmin(w, req) {
		return
	}
	now := time.Now()
	frame := newUsageSeries(now, usageWindow, usageBucket)
	// No database configured at all. An empty frame rather than an error: the
	// chart is not broken, there is simply nothing recording.
	if r.logs == nil {
		writeJSON(w, http.StatusOK, frame)
		return
	}
	if cached := r.usage.get(frame.To, now); cached != nil {
		writeJSON(w, http.StatusOK, cached)
		return
	}
	started := time.Now()
	series, err := r.logs.UsageSeries(req.Context(), now, usageWindow, usageBucket, usageTopClients)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	r.usage.put(series, now, time.Since(started))
	writeJSON(w, http.StatusOK, series)
}

func (r *Router) handleRegisterBackend(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	if !r.requireWorker(w, req) {
		return
	}

	var reg BackendRegistration
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 1<<20)).Decode(&reg); err != nil {
		writeError(w, http.StatusBadRequest, "invalid json: %s", err)
		return
	}
	// A push registration is a beacon by construction, whatever the payload says.
	// Manual is the flag that makes a row operator-owned — probes stop overwriting
	// its declared values (see providers.go) — and it is not something a worker
	// holding the worker token gets to grant itself.
	reg.Source = sourceBeacon
	if err := normalizeRegistration(&reg); err != nil {
		writeError(w, http.StatusBadRequest, "%s", err)
		return
	}
	// Nor may it take over a row an operator entered by hand. The id collision is
	// almost certainly a mistake, and silently converting the operator's row into
	// a beacon would discard exactly the declared values manual rows exist to
	// protect. A relay's derived row is refused for the mirror-image reason: it
	// would be taken over here and pruned again by the next relay refresh, so the
	// two would fight over the id every fifteen seconds.
	switch existing := r.registry.get(reg.ID); {
	case isManualRow(existing):
		writeJSON(w, http.StatusConflict, validationError{
			Message: fmt.Sprintf("backend %q is an operator-managed provider row; register under a different id", reg.ID),
			Param:   "id",
		})
		return
	case isRelayRow(existing):
		writeJSON(w, http.StatusConflict, validationError{
			Message: fmt.Sprintf("backend %q is derived from relay %q; register under a different id", reg.ID, existing.Relay),
			Param:   "id",
		})
		return
	}

	backend, changed := r.registry.upsert(reg)
	if err := r.logs.SaveBackendRegistration(req.Context(), reg); err != nil {
		log.Printf("persist backend registration failed id=%s: %v", backend.ID, err)
	}
	// A keepalive (unchanged registration) of a ready backend is pure liveness —
	// re-certifying would knock it out of rotation (startCertification sets
	// "probing") for two worker round-trips every ~60s. Not-ready backends still
	// get a certification kick; the guard in certifyBackend de-dups overlaps.
	if changed || !backend.Certification.Ready {
		go r.certifyBackend(backend.ID)
	}

	writeJSON(w, http.StatusOK, map[string]any{"status": "registered", "id": backend.ID})
}

// handleModels stays CLIENT scope, and has to: a client cannot use an OpenAI
// API without being able to read the model list, and every SDK fetches it. It
// publishes the MENU — ids, aliases, features — and not the estate behind it.
func (r *Router) handleModels(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	if _, ok := r.requireClient(w, req); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": r.modelCatalogue()})
}

// handleModelByID serves GET /v1/models/{id} — the standard's single-model
// object, and the call an OpenAI SDK makes to check a model exists before using
// it. The path used to fall through to the dashboard handler, which answered
// 404 in HTML.
//
// The object is taken from modelCatalogue rather than rebuilt, so it is
// identical to the one /v1/models publishes for the same model; a client that
// reads a field from the list can read it here. Both published spellings are
// accepted — the menu id (the alias where there is one) and "root" (the raw
// model id) — which are exactly the ids this endpoint advertises.
func (r *Router) handleModelByID(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	if _, ok := r.requireClient(w, req); !ok {
		return
	}
	name := strings.TrimPrefix(req.URL.Path, "/v1/models/")
	if name == "" {
		r.handleModels(w, req) // a bare trailing slash is still the list
		return
	}
	// A group answers to its name however it is capitalised, because that is how
	// routing resolves it — the menu must not 404 a spelling chat/completions
	// would accept. The ensemble is the same case for the same reason.
	if g, ok := r.groups.lookup(name); ok {
		name = g.Name
	} else if isExpertModel(name) {
		name = expertModel
	}
	for _, m := range r.modelCatalogue() {
		if m["id"] == name || m["root"] == name {
			writeJSON(w, http.StatusOK, m)
			return
		}
	}
	writeJSON(w, http.StatusNotFound, validationError{
		Message: unknownModelError{name: name}.Error(),
		Param:   "model",
	})
}

// modelCatalogue builds the /v1/models menu: one entry per servable model, plus
// the "default" auto route at the head.
func (r *Router) modelCatalogue() []map[string]any {
	// This list is a MENU now, not a census: a client may name any id here and get
	// it (see requestedModel). So publish each model ONCE, however many workers
	// serve it — three identical gemma rows differing only by owned_by told a
	// harness nothing and read as three choices. Workers serving the same model
	// are pooled behind that one id and load-balanced by completion time.
	byModel := map[string]map[string]any{}
	order := []string{}
	fleetFeatures := []string{"chat"}
	fleetCtx := 0                        // largest measured window in the fleet
	aliasModels := map[string][]string{} // alias → distinct models claiming it
	servable := []*Backend{}             // what a group's members can resolve to
	for _, b := range r.registry.snapshot() {
		if isExpired(b) || isEmbeddingsOnly(b) {
			continue
		}
		servable = append(servable, b)
		fleetFeatures = append(fleetFeatures, b.Features...)
		// The window ROUTING enforces, not the one the server claims. These were
		// different numbers, and a coding harness believed this one: it read a 256K
		// context_length here, sized max_tokens as (256K - prompt), and grew its
		// conversation until every worker failed the hard context filter — a 503 the
		// menu had promised would be served. What is published and what is admitted
		// have to be the same figure.
		if usable := usableContextTokens(b); usable > fleetCtx {
			fleetCtx = usable
		}
		entry, seen := byModel[b.Model]
		if !seen {
			entry = map[string]any{
				"id":       b.Model,
				"object":   "model",
				"owned_by": b.ID,
				"features": append([]string(nil), b.Features...),
				"workers":  1,
			}
			if usable := usableContextTokens(b); usable > 0 {
				entry["context_length"] = usable
			}
			byModel[b.Model] = entry
			order = append(order, b.Model)
			if a := backendAlias(b); a != "" {
				aliasModels[a] = append(aliasModels[a], b.Model)
			}
			continue
		}
		entry["workers"] = entry["workers"].(int) + 1
		// A feature is only claimable for the pooled id if every worker behind it
		// has it — the router may send the request to any of them.
		entry["features"] = intersectFeatures(entry["features"].([]string), b.Features)
		// Same intersection logic for the window: the pooled id may land on any
		// worker, so advertise the smallest one routing would admit. A worker that
		// didn't report stays out of the min — its requests are context-filtered at
		// route time anyway (see hard filter), so the claim stays honest.
		usable := usableContextTokens(b)
		if cur, ok := entry["context_length"].(int); usable > 0 && (!ok || usable < cur) {
			entry["context_length"] = usable
		}
	}
	// The menu id is the human spelling when it is unambiguous: the alias
	// replaces the raw model id ("gemma4", not a quant-encrusted .gguf path),
	// with the raw id preserved under "root" — both spellings, plus the worker
	// id, are accepted by requestedModel/backendServesModel. Two distinct models
	// reducing to the same alias keep their raw ids: an ambiguous menu row would
	// silently pool different models.
	for a, ms := range aliasModels {
		if len(ms) != 1 || byModel[a] != nil {
			continue
		}
		byModel[ms[0]]["root"] = ms[0]
		byModel[ms[0]]["id"] = a
	}
	sort.Strings(order)
	// "default" is the auto route, and listing it makes the automatic behaviour a
	// visible choice rather than folklore: a harness can select it deliberately
	// instead of guessing which concrete model to name. It advertises the UNION
	// of the fleet's features — a tools request to default hard-filters to
	// tools-capable workers, so default really does serve whatever any worker
	// can — where a pooled concrete id advertises the intersection.
	fleet := normalizeFeatures(fleetFeatures)
	models := []map[string]any{{
		"id":       "default",
		"object":   "model",
		"owned_by": routerOwner,
		"features": fleet,
	}, expertEntry(fleet)}
	// default and expert advertise the fleet MAX, matching their union-of-features
	// stance: both routes context-filter workers per request, so a long prompt
	// really can be served as long as any worker holds it (expert just seats a
	// smaller panel).
	if fleetCtx > 0 {
		models[0]["context_length"] = fleetCtx
		models[1]["context_length"] = fleetCtx
	}
	for _, name := range order {
		// A worker that registered a model called "expert" does not get the name:
		// the ensemble resolves ahead of it, so the menu has to say where the name
		// actually goes rather than advertising a row nothing routes to. Same rule
		// as a shadowed group, and the worker stays reachable by its own id.
		if id, _ := byModel[name]["id"].(string); isExpertModel(id) {
			continue
		}
		models = append(models, byModel[name])
	}
	// Groups last, and they REPLACE a row of the same id rather than sitting
	// beside it. A group resolves ahead of model ids and aliases (see planRoute),
	// so where a worker has since registered under a group's name the menu has to
	// say where that name actually goes. The admin API refuses to create such a
	// group, so this only fires when the worker arrived second.
	for _, entry := range groupEntries(r.groups.list(), servable, fleet) {
		replaced := false
		for i, m := range models {
			if m["id"] == entry["id"] {
				models[i], replaced = entry, true
				break
			}
		}
		if !replaced {
			models = append(models, entry)
		}
	}
	return models
}

func (r *Router) handleDashboard(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	if req.URL.Path != "/" {
		notFound(w, req)
		return
	}
	// GET / requires no auth, so the server render must disclose NOTHING about the
	// fleet — no IDs, URLs, models, quality or load. The page is a static shell; the
	// backend table and per-backend log tabs are populated CLIENT-SIDE from a
	// admin-gated /backends fetch (see renderBackends in dashboardTemplate),
	// authenticated by the session cookie /admin/login sets.
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := dashboardTemplate.Execute(w, map[string]any{
		"GeneratedAt": time.Now().Format(time.RFC3339),
	}); err != nil {
		log.Printf("dashboard render failed: %v", err)
	}
}

func (r *Router) handleDebugBackends(w http.ResponseWriter, req *http.Request) {
	path := strings.TrimPrefix(req.URL.Path, "/debug/backends/")
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) != 2 {
		notFound(w, req)
		return
	}
	id, action := parts[0], parts[1]
	switch action {
	case "chat":
		r.handleDebugBackendChat(w, req, id)
	case "certify":
		r.handleDebugBackendCertify(w, req, id)
	default:
		notFound(w, req)
	}
}

func (r *Router) handleDebugBackendCertify(w http.ResponseWriter, req *http.Request, id string) {
	if req.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	// Admin scope since P3. It re-runs a cold-start profile, which on a metered
	// endpoint spends the operator's money and on a local one takes the worker
	// out of rotation for minutes.
	if !r.requireAdmin(w, req) {
		return
	}
	if r.registry.get(id) == nil {
		writeJSON(w, http.StatusNotFound, validationError{Message: "backend not found"})
		return
	}
	// An operator asking for this is saying they have fixed whatever was wrong, so
	// a worker whose background profile was abandoned for aborting too often gets
	// its full allowance back (see profileRetryMaxAttempts).
	r.registry.clearProfileAborts(id)
	go r.certifyBackend(id)
	writeJSON(w, http.StatusAccepted, map[string]any{"status": "probing", "id": id})
}

func (r *Router) handleDebugBackendChat(w http.ResponseWriter, req *http.Request, id string) {
	if req.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	// Admin scope since P3. It pins a worker by id, so it bypasses routing, the
	// per-key allow-list and every hard filter — a client-scoped escape hatch
	// around the controls this phase adds would make them decorative.
	if !r.requireAdmin(w, req) {
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
	backend := r.registry.get(id)
	if backend == nil {
		writeJSON(w, http.StatusNotFound, validationError{Message: "backend not found"})
		return
	}
	// No classification on the debug path (nil) — auto-thinking is skipped, as on
	// the pinned path; only explicit thinking signals apply. target=0 ⇒ no quality
	// floor (the debug path pins one backend anyway). The caller is an admin, so
	// there is no per-key budget to charge and the log row records who they were.
	r.proxyToBackend(w, req, r.identify(req), &routePlan{candidates: []*Backend{backend}, route: "debug"}, body, chatReq)
}

func (r *Router) handleChatCompletions(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	ident, ok := r.requireClient(w, req)
	if !ok {
		return
	}
	// Before the body is even read: a request that has already passed through
	// this router will not find a worker by going round again (see relay.go).
	if r.refuseRelayLoop(w, req) {
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
	// Per-key limits before anything is dispatched: a request that is going to be
	// refused should not first take a worker slot.
	if !r.enforceKeyLimits(w, ident, requestedModel(chatReq)) {
		return
	}

	var plan *routePlan

	if pinID := req.Header.Get("X-LLM-Backend-ID"); pinID != "" {
		// Pin to a specific backend by ID. No classification is computed here, so
		// auto-thinking is skipped (only explicit thinking signals apply), there is
		// no quality floor, and no session affinity — the caller has already made
		// every one of those decisions.
		//
		// What the caller has NOT made is the access-control decision.
		// enforceKeyLimits above only saw the body's model, and a pinned request
		// need not name one at all, so the allow-list is re-checked here against the
		// worker itself. Same reasoning that put /debug/backends/{id}/chat behind
		// admin: a way of naming a worker that skips the per-key controls makes them
		// decorative.
		backend := r.registry.get(pinID)
		switch {
		case backend == nil:
			r.refusePin(w, ident, pinID, "no backend is registered with that id")
			return
		case !ident.allowsBackend(backend):
			r.refusePin(w, ident, pinID, fmt.Sprintf("serves no model on this key's allow-list (%s)", strings.Join(ident.Models, ", ")))
			return
		case !backend.Healthy || !backend.Certification.Ready || isExpired(backend):
			r.refusePin(w, ident, pinID, fmt.Sprintf("not available (healthy=%v, ready=%v, expired=%v)",
				backend.Healthy, backend.Certification.Ready, isExpired(backend)))
			return
		}
		plan = &routePlan{candidates: []*Backend{backend}, route: "pinned"}
	} else {
		plan, err = r.planRoute(chatReq, callerBudget(req, chatReq), false)
		if err != nil {
			var unknown unknownModelError
			if errors.As(err, &unknown) {
				writeJSON(w, http.StatusNotFound, validationError{Message: err.Error(), Param: "model"})
				return
			}
			writeUnavailable(w, r.retryAfterUnavailable(), err.Error())
			return
		}
		kept, ok := r.restrictToAllowList(w, ident, plan.route, plan.candidates)
		if !ok {
			return
		}
		plan.candidates = kept
	}

	// The ensemble dispatches to every model at once and synthesises the replies,
	// so it cannot go through proxyToBackend, which is built around one worker
	// serving one request. It applies the allow-list to its own panel (see
	// expert.go) — restrictToAllowList above only narrows a "route"-prefixed plan.
	if plan.expert.active() {
		r.serveExpert(w, req, ident, plan, body, chatReq)
		return
	}
	r.proxyToBackend(w, req, ident, plan, body, chatReq)
}

// refusePin answers a pin the router will not serve.
//
// Every reason — no such id, a worker that is not ready, a worker off this key's
// allow-list — is the SAME 404 to a non-admin caller, and only an admin is told
// which. The three used to be distinguishable, and that turned X-LLM-Backend-ID
// into a fleet-enumeration oracle: any client key could walk a list of guesses
// and read back every registered worker id and whether it was alive, which is
// precisely what moving /backends behind the admin gate was meant to stop.
//
// 404 rather than the old 503 because a 404 is already the answer for a model
// this router will not serve the caller (see unknownModelError), and "the thing
// you named is not something you can have" is the same answer whether it was
// named in the model field or in this header. It costs a pinning client the
// Retry-After it used to get for a busy worker; a caller that wants the router
// to wait for a worker should be using the auto route, which does.
func (r *Router) refusePin(w http.ResponseWriter, ident *identity, pinID, reason string) {
	message := fmt.Sprintf("backend %q not found", pinID)
	if ident != nil && ident.Role == roleAdmin {
		message = fmt.Sprintf("backend %q: %s", pinID, reason)
	}
	writeJSON(w, http.StatusNotFound, validationError{Message: message})
}

// restrictToAllowList narrows a candidate set to the workers a key's allow-list
// names, for the routes where the ROUTER picked the model rather than the
// caller. Writes the refusal and returns ok=false when nothing survives.
//
// This is the other half of the allow-list, and without it the list was not an
// access control at all. allowsModel refuses nothing to a caller who named
// nothing, which is right for a name and useless as a gate: "default", "auto",
// "router" and an absent model field all mean the auto route (see
// autoModelNames), and the auto route ranks the WHOLE fleet — so a key issued
// for one local worker could reach every metered endpoint by asking for
// "default". A named group that no member could serve lands here too: group
// resolution clears the model filter on fallback precisely so a group is never a
// refusal, and that fallback was likewise unfiltered.
//
// The trigger is the route string rather than the model field because the route
// string is the router's own record of who chose: routeKind writes "route" when
// it chose and "model" when the client did, and the group fallback deliberately
// rewrites itself to "route" for exactly that reason.
func (r *Router) restrictToAllowList(w http.ResponseWriter, ident *identity, route string, candidates []*Backend) ([]*Backend, bool) {
	if ident == nil || len(ident.Models) == 0 || !strings.HasPrefix(route, "route") {
		return candidates, true
	}
	kept := filterCandidates(candidates, ident.allowsBackend)
	if len(kept) == 0 {
		// 503, not 403: the key is allowed these models, and the reason none is a
		// candidate right now may be that they are all busy, unhealthy or not yet
		// certified. Naming the allow-list back tells the caller nothing it did not
		// supply, so this leaks no more of the fleet than the key already knows.
		writeUnavailable(w, r.retryAfterUnavailable(), fmt.Sprintf(
			"no worker this key may use is available (allowed: %s)", strings.Join(ident.Models, ", ")))
		return nil, false
	}
	return kept, true
}

// completionRequest is the legacy /v1/completions shape. We don't rewrite
// the body — vLLM and llama.cpp both serve /v1/completions natively, so the
// router just selects a backend and passes the request through. The parsed
// fields exist only to drive backend selection and slot accounting.
type completionRequest struct {
	Model        string          `json:"model,omitempty"`
	Prompt       json.RawMessage `json:"prompt,omitempty"`
	MaxTokens    int             `json:"max_tokens,omitempty"`
	Stream       bool            `json:"stream,omitempty"`
	Requirements *Requirements   `json:"requirements,omitempty"`
}

func (r *Router) handleCompletions(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	ident, ok := r.requireClient(w, req)
	if !ok {
		return
	}
	if r.refuseRelayLoop(w, req) {
		return
	}
	body, err := readRequestBody(w, req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "read request: %s", err)
		return
	}

	var compReq completionRequest
	if err := json.Unmarshal(body, &compReq); err != nil {
		writeJSON(w, http.StatusBadRequest, validationError{Message: fmt.Sprintf("invalid json: %s", err)})
		return
	}

	// Re-use selectBackend by adapting to a minimal ChatRequest. No messages
	// (selectBackend tolerates that) but Model / MaxTokens / Stream /
	// Requirements all carry across.
	chatReq := &ChatRequest{
		Model:        compReq.Model,
		MaxTokens:    compReq.MaxTokens,
		Stream:       compReq.Stream,
		Requirements: compReq.Requirements,
	}
	if !r.enforceKeyLimits(w, ident, requestedModel(chatReq)) {
		return
	}
	candidates, route, _, err := r.selectBackends(chatReq, callerBudget(req, chatReq))
	if err != nil {
		writeUnavailable(w, r.retryAfterUnavailable(), err.Error())
		return
	}
	// Same gate as the chat path: an auto route here reaches the whole fleet too.
	candidates, ok = r.restrictToAllowList(w, ident, route, candidates)
	if !ok {
		return
	}
	r.proxyPassthrough(w, req, ident, candidates, body, "/v1/completions", routeCompletions)
}

func (r *Router) handleEmbeddings(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	ident, ok := r.requireClient(w, req)
	if !ok {
		return
	}
	if r.refuseRelayLoop(w, req) {
		return
	}
	body, err := readRequestBody(w, req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "read request: %s", err)
		return
	}
	// No model name to check against the allow-list here — /v1/embeddings routes
	// by feature — but a budget still applies.
	if !r.enforceKeyLimits(w, ident, "") {
		return
	}
	candidates, err := r.selectBackendsByFeature("embeddings")
	if err != nil {
		writeUnavailable(w, r.retryAfterUnavailable(), err.Error())
		return
	}
	r.proxyPassthrough(w, req, ident, candidates, body, "/v1/embeddings", routeEmbeddings)
}

// callerBudget reports how long the caller will still wait for a response, from
// the X-LLM-Deadline-MS header or the request's deadline_ms field (header wins).
// The request context's own deadline is checked first, but net/http never sets one
// for an inbound request, so in practice the budget has to be declared.
// Zero means unknown — every deadline-aware behaviour is then skipped.
func callerBudget(req *http.Request, chatReq *ChatRequest) time.Duration {
	if dl, ok := req.Context().Deadline(); ok {
		if d := time.Until(dl); d > 0 {
			return d
		}
	}
	ms := 0
	if h := strings.TrimSpace(req.Header.Get("X-LLM-Deadline-MS")); h != "" {
		if v, err := strconv.Atoi(h); err == nil {
			ms = v
		}
	}
	if ms <= 0 && chatReq != nil {
		ms = chatReq.DeadlineMillis
	}
	if ms <= 0 {
		return 0
	}
	return time.Duration(ms) * time.Millisecond
}

// selectBackendsByFeature returns eligible backends whose declared features
// include the given feature flag (e.g. "embeddings"), ranked best-first. Used
// by non-chat endpoints where the full chat-style selection (context, tools,
// vision) doesn't apply. The caller spills across the list when the best
// backend has no free slot (see pickAndAcquire).
func (r *Router) selectBackendsByFeature(feature string) ([]*Backend, error) {
	candidates := r.registry.eligible()
	if len(candidates) == 0 {
		return nil, errors.New("no healthy backends registered")
	}
	filtered := filterCandidates(candidates, func(b *Backend) bool {
		return hasFeature(b, feature)
	})
	if len(filtered) == 0 {
		return nil, fmt.Errorf("no healthy backends with feature %q", feature)
	}
	// Feature lookups (/v1/embeddings) carry no chat request to size, so rank on
	// the nominal job — unchanged from before jobCost existed.
	return rankBackends(filtered, nominalJob(), false), nil
}

// The two passthrough routes, named so the budgeting fallback can tell them
// apart without matching a string literal written somewhere else.
const (
	routeCompletions = "completions"
	routeEmbeddings  = "embeddings"
)

// setBackendCredential puts the BACKEND's own credential on a forwarded request,
// and nothing at all when it has none.
//
// It used to fall back to relaying the CLIENT's Authorization header verbatim.
// That was defensible while a client token was one shared LAN secret held by
// services the operator ran; it stopped being defensible the moment callers hold
// their own `sk-` keys. Every backend that declares no api_key — which includes
// anything that pushed itself to /backends/register, and therefore anything a
// stranger registered — was handed each caller's live credential on every
// request it served. Nothing needs it: a worker that wants a bearer token
// registers the one it wants (api_key in the registration payload), and a worker
// that registers none is running without authentication, so there is nothing for
// the header to satisfy. The router's own probes never had the fallback either,
// so a worker relying on it could not have passed certification.
func setBackendCredential(proxyReq *http.Request, backend *Backend) {
	if backend.APIKey != "" {
		proxyReq.Header.Set("Authorization", "Bearer "+backend.APIKey)
	}
}

// genCharsPerToken is the chars-per-token divisor the budgeting fallbacks use.
// It is the same 4.8 sseStats.genTokens applies to a streamed generation, so the
// buffered and streamed estimates agree — the two are separate literals, so
// change them together (capture.go).
const genCharsPerToken = 4.8

// estimateGenTokens turns a reply's size in bytes into a generated-token
// estimate. It over-counts by the JSON envelope — a couple of hundred bytes, so
// perhaps forty tokens on a one-word answer — which is the safe direction: it
// only ever runs for an endpoint that declined to say what it charged, and a
// spending bound that errs low is not a bound.
func estimateGenTokens(n int64) int {
	if n <= 0 {
		return 0
	}
	return int(float64(n) / genCharsPerToken)
}

// estimatePromptTokens sizes a passthrough request the router deliberately never
// parses. chars/3 is the divisor promptTokensFor uses on the chat path; applied
// to the whole raw body it also charges the JSON scaffolding, which is a
// rounding error beside a prompt.
func estimatePromptTokens(body []byte) int {
	return len(body) / 3
}

// replyMeter measures a passthrough reply as it is relayed, for the budgeting
// fallback that runs when the endpoint reported no usage block of its own.
//
// Two numbers, because there are two reply shapes and neither can be read off
// the other. A streamed reply is one `data:` frame per generated token in both
// dialects, so the frame count IS the token count; a buffered reply has no
// frames, and then its size is the only signal left. Measured on the wire rather
// than from the capture beside it: that capture keeps 4KB of a stream which may
// be megabytes.
type replyMeter struct {
	bytes   int64
	frames  int
	partial []byte // bytes after the last newline, held over for the next Write
	// first is when the first byte of the reply went past, for TTFT. Only
	// meaningful on a streamed reply: a buffered one is written in a single
	// Write at the very end, where first-byte and last-byte are the same instant
	// and the number would merely restate the total duration. ttftMillis reports
	// 0 rather than that, so a timing model can tell "not measured" from "fast".
	first time.Time
}

// ttftMillis is the time from start to the first byte of the reply, or 0 if
// nothing was ever written.
func (m *replyMeter) ttftMillis(start time.Time) int64 {
	if m == nil || m.first.IsZero() {
		return 0
	}
	return m.first.Sub(start).Milliseconds()
}

func (m *replyMeter) Write(p []byte) (int, error) {
	if m.first.IsZero() && len(p) > 0 {
		m.first = time.Now()
	}
	m.bytes += int64(len(p))
	m.partial = append(m.partial, p...)
	for {
		i := bytes.IndexByte(m.partial, '\n')
		if i < 0 {
			break
		}
		line := m.partial[:i]
		if bytes.HasPrefix(line, []byte("data: ")) && !bytes.Contains(line, []byte("[DONE]")) {
			m.frames++
		}
		m.partial = m.partial[i+1:]
	}
	// Safety valve: a reply that never sends a newline (not SSE, and large) must
	// not accumulate unboundedly. Same 1MB bound as sseStats.
	if len(m.partial) > 1<<20 {
		m.partial = nil
	}
	return len(p), nil
}

// genTokens is this reply's generated-token estimate: its frames when it
// streamed, its size otherwise.
func (m *replyMeter) genTokens() int {
	if m.frames > 0 {
		return m.frames
	}
	return estimateGenTokens(m.bytes)
}

// proxyPassthrough forwards a request body verbatim to openAIPath on the
// best available backend from candidates. Used by /v1/completions and
// /v1/embeddings, where the router does not interpret or rewrite the request
// body (in contrast to /v1/chat/completions, which uses proxyToBackend for
// retry + max_tokens patching). Honours backend slot accounting (spilling
// across candidates), circuit-breaker feedback, and request logging.
func (r *Router) proxyPassthrough(w http.ResponseWriter, req *http.Request, ident *identity, candidates []*Backend, body []byte, openAIPath, route string) {
	if len(candidates) == 0 {
		writeUnavailable(w, r.retryAfterUnavailable(), "no backend available")
		return
	}
	start := time.Now()

	backend, slot, slotErr := r.pickAndAcquire(req.Context(), candidates)
	if slotErr != nil {
		r.logSlotUnavailable(ident, clientIP(req), candidates[0], route, false, nil, start, slotErr)
		writeUnavailable(w, r.retryAfterSaturated(), fmt.Sprintf("no backend slot available: %s", slotErr))
		return
	}
	defer r.registry.releaseSlot(slot)

	logEntry := RequestLog{
		CreatedAt:    start.UTC(),
		BackendID:    backend.ID,
		BackendModel: backend.Model,
		Route:        route,
		ObservedTPS:  backend.ObservedTPS,
		CertifiedTPS: backend.Certification.TokensPerSec,
		BaselineTPS:  backend.BaselineTPS,
		SpeedScore:   speedScore(backend),
		KeyID:        ident.logKeyID(),
		ClientIP:     clientIP(req),
		// This route never resolves a thinking mode — it forwards the body
		// verbatim — so the field stays unknown rather than claiming "off".
		Thinking: thinkingUnknown,
	}
	// Small capture, kept only to read the endpoint's usage block for per-key
	// budgeting. Head-and-tail bounded, and usage sits at the tail in both the
	// buffered and the SSE shape.
	usage := newBoundedCapture(usageCaptureBytes)
	// The meter measures the WHOLE reply as it goes past, which the capture
	// cannot: it keeps 4KB of a stream that may be megabytes. Only read when the
	// endpoint reported no usage of its own.
	meter := &replyMeter{}
	defer func() {
		logEntry.DurationMillis = time.Since(start).Milliseconds()
		// Charge what the endpoint said it charged, and estimate when it said
		// nothing. stream_options.include_usage is off by default, so "nothing" is
		// the ordinary case for a streamed /v1/completions — and charging zero for
		// it meant a budgeted key could stream this endpoint in a loop without
		// tokens_used moving at all. The chat path has had this fallback all along.
		logEntry.PromptTokens = lastJSONInt(usage.Bytes(), "prompt_tokens")
		logEntry.CompletionTokens = lastJSONInt(usage.Bytes(), "completion_tokens")
		// ONLY on a genuinely streamed reply. A buffered one does not reach this
		// router until the upstream has finished generating, so its first chunk
		// arrives at the end and "time to first byte" would just restate
		// DurationMillis — a fabricated prefill measurement, and on /v1/embeddings
		// (which never streams) every row would carry one. Left at 0, meaning not
		// measured, which is what a timing model must see.
		if logEntry.Stream {
			logEntry.TTFTMillis = meter.ttftMillis(start)
		}
		charged := usageTotalTokens(usage.Bytes())
		if charged == 0 {
			charged = estimatePromptTokens(body)
			// An embeddings reply generates nothing — its whole cost is the input,
			// and its body is a float array that would price like an epic poem.
			if route != routeEmbeddings {
				charged += meter.genTokens()
			}
		}
		// Async, and BOTH writes: this defer runs before releaseSlot (LIFO), so a
		// synchronous SQLite write here — against a pool capped at one connection —
		// extends the busy window of a max_concurrency=1 worker after every request.
		// recordKeyUse was that write; the log insert next to it always knew better.
		caller, entry := ident, redactForRelay(logEntry, ident)
		go func() {
			r.recordKeyUse(caller, charged)
			if err := r.logs.Insert(context.Background(), entry); err != nil {
				log.Printf("persist request log failed: %v", err)
			}
		}()
	}()

	r.registry.incActive(backend.ID, 1)
	defer r.registry.incActive(backend.ID, -1)
	logEntry.Concurrency = r.registry.activeCount(backend.ID)

	proxyReq, err := http.NewRequestWithContext(req.Context(), http.MethodPost, upstreamPathURL(backend, openAIPath), bytes.NewReader(body))
	if err != nil {
		logEntry.StatusCode = http.StatusInternalServerError
		logEntry.Error = err.Error()
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")
	setBackendCredential(proxyReq, backend)
	r.stampRelayChain(proxyReq, req, backend)

	resp, err := r.client.Do(proxyReq)
	if err != nil {
		// Client hangups (context canceled) aren't backend failures — see the
		// matching guard in dispatchStreaming.
		if req.Context().Err() == nil {
			r.registry.setError(backend.ID, err.Error())
			r.registry.noteProxyResult(backend.ID, false)
		}
		logEntry.StatusCode = http.StatusBadGateway
		logEntry.Error = err.Error()
		writeJSON(w, http.StatusBadGateway, validationError{Message: err.Error()})
		return
	}
	defer resp.Body.Close()
	r.registry.noteProxyResult(backend.ID, resp.StatusCode < 500)

	w.Header().Set("X-LLM-Backend-ID", backend.ID)
	w.Header().Set("X-LLM-Backend-Model", backend.Model)
	w.Header().Set("X-LLM-Backend-Model-Name", niceModelName(backend))
	w.Header().Set("X-LLM-Backend-URL", backend.URL)
	w.Header().Set("X-LLM-Route", route)
	copyHeaders(w.Header(), resp.Header)
	r.backfillRetryAfter(w.Header(), resp.StatusCode) // see dispatchStreaming
	w.WriteHeader(resp.StatusCode)
	logEntry.StatusCode = resp.StatusCode
	// Taken from the reply the upstream actually sent rather than from what the
	// caller asked for: this path forwards the body verbatim and never parses it,
	// so the request's stream flag was never read here, and an endpoint may
	// decline to stream regardless. It gates the TTFT measurement above.
	logEntry.Stream = isEventStream(resp.Header)

	// Flush per chunk: this path serves stream:true /v1/completions too, and a
	// bare io.Copy buffers the whole generation before the client sees a byte.
	fl, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			_, _ = usage.Write(buf[:n])
			_, _ = meter.Write(buf[:n])
			if _, werr := w.Write(buf[:n]); werr != nil {
				logEntry.Error = werr.Error()
				return
			}
			if fl != nil {
				fl.Flush()
			}
		}
		if rerr != nil {
			if rerr != io.EOF {
				logEntry.Error = rerr.Error()
				// stream:true /v1/completions runs through here too, so the same
				// mid-stream rule applies: the status is spent, and a truncated
				// stream reads as a short answer (see writeSSEError).
				if req.Context().Err() == nil && isEventStream(w.Header()) {
					writeSSEError(w, fmt.Sprintf("upstream stream from backend %q failed: %s", backend.ID, rerr))
				}
			}
			return
		}
	}
}

// clientIP is the address a request came in from, for the usage graph and for
// the caller line on a log row.
//
// X-Forwarded-For is preferred over RemoteAddr whenever it parses, because this
// router is normally deployed behind Caddy and behind Cloudflare: RemoteAddr
// there is the proxy's own address, so every request in the fleet would stack
// into one band and the graph would be a solid block saying nothing. The
// LEFTMOST entry is the original client — each hop appends itself on the right.
//
// That header is trivially forged by anything that can reach this router
// directly, and it is taken at face value here with no list of trusted proxies.
// That is a deliberate trade, and it is only safe because of what this value
// decides: which coloured band a bar is drawn in, on a chart only an admin
// session can load. Nothing authenticates against it, nothing is rate-limited
// by it, nothing is refused because of it — the api_keys row does all of that
// (see identity, which is why this is NOT a field on it: a spoofable value
// living inside the struct called "identity" is an invitation to grant
// something on the strength of it). If that ever changes, this function is the
// wrong place to get the answer from: it would first need a configured set of
// trusted hops, because as it stands the caller chooses what the leftmost entry
// says.
//
// An entry that is not an IP address is discarded rather than stored, which is
// also what bounds what a forged header can put in the column and in the chart
// legend.
func clientIP(req *http.Request) string {
	if req == nil {
		return ""
	}
	forwarded, _, _ := strings.Cut(req.Header.Get("X-Forwarded-For"), ",")
	if ip := parseHost(forwarded); ip != "" {
		return ip
	}
	return parseHost(req.RemoteAddr)
}

// parseHost strips any port from an address and keeps it only if what is left is
// an IP. net.SplitHostPort rather than a cut at the last colon: a bare IPv6
// address is nothing but colons, and cutting one leaves a different address that
// still looks plausible. It fails on the bracketless bare form, which is exactly
// the case where there was no port to strip.
func parseHost(addr string) string {
	addr = strings.TrimSpace(addr)
	if host, _, err := net.SplitHostPort(addr); err == nil {
		addr = host
	}
	addr = strings.Trim(addr, "[]")
	if net.ParseIP(addr) == nil {
		return ""
	}
	return addr
}

// logSlotUnavailable records the 503 we return when no candidate backend had a
// free slot before the deadline. Attributed to the primary (best-ranked)
// candidate for context.
//
// It takes the caller for the same reason the success path does: a relayed
// request's prompt must not reach the log store on the failure path either, and
// a redaction that only covered the requests that worked would be no redaction.
//
// from is the caller's address, threaded in because this is the one place a
// request is logged without the *http.Request still being in scope.
func (r *Router) logSlotUnavailable(ident *identity, from string, primary *Backend, route string, stream bool, input []byte, start time.Time, cause error) {
	entry := redactForRelay(RequestLog{
		CreatedAt:      start.UTC(),
		BackendID:      primary.ID,
		BackendModel:   primary.Model,
		Route:          route,
		Stream:         stream,
		Input:          string(input),
		StatusCode:     http.StatusServiceUnavailable,
		Error:          cause.Error(),
		DurationMillis: time.Since(start).Milliseconds(),
		KeyID:          ident.logKeyID(),
		ClientIP:       from,
	}, ident)
	if err := r.logs.Insert(context.Background(), entry); err != nil {
		log.Printf("request log insert failed backend=%s: %v", primary.ID, err)
	}
}

// slotMaxWait caps how long the router will block a request waiting for a
// backend concurrency slot before returning 503. Callers' own timeouts
// (via request context) take precedence; this is the fallback bound. Tune
// via ROUTER_SLOT_MAX_WAIT_SECONDS: how long a caller queues before a 503 is a
// promise to clients, not a routing decision, so it stays an operator setting.
var slotMaxWait = envDuration("ROUTER_SLOT_MAX_WAIT_SECONDS", 10*time.Minute)

// qualityFloorWait is the BOUNDED grace a request auto-classified as needing
// quality >= target will wait for an above-target worker to free a slot before
// it spills BELOW the floor (fix #2). Routing too cheap is the quality-risky
// direction, so a HARD prompt shouldn't be instantly downgraded the instant the
// sufficient workers are momentarily full. It MUST be << slotMaxWait: it only
// briefly prefers the better tier, then falls through to the normal spill (up to
// slotMaxWait).
//
// Ten seconds, measured rather than picked: with a fast GPU worker at
// max_concurrency 1 backed by a slow CPU fallback, a shorter grace sent the
// second concurrent chat to the CPU for a multi-minute generation. Ten covers
// most in-flight GPU turns. This used to be an environment variable and was set
// to two different values in two files of the same deployment, which is the
// evidence that it is not a question an operator can answer.
var qualityFloorWait = 10 * time.Second

// proxyRetryDelays describes the backoff between 5xx/429 retries for
// non-streaming requests. Streaming requests are forwarded once and never
// retried (we've already started writing bytes to the client). Tune by
// shadowing in env if needed; the constants are intentional defaults.
var proxyRetryDelays = []time.Duration{
	2 * time.Second,
	5 * time.Second,
	10 * time.Second,
}

// slotPollInterval is how often pickAndAcquire re-scans all candidate backends
// for a freed slot while every candidate is momentarily saturated.
const slotPollInterval = 50 * time.Millisecond

// pickAndAcquire returns the highest-ranked candidate that has a free
// concurrency slot, acquiring that slot. Candidates are tried best-first; a
// backend with no declared cap is always immediately available (nil slot).
// When every candidate is momentarily full it re-scans every slotPollInterval
// until one frees, the caller's context is cancelled, or slotMaxWait elapses —
// so a burst spills across idle backends instead of queueing on a single one.
func (r *Router) pickAndAcquire(ctx context.Context, candidates []*Backend) (*Backend, chan struct{}, error) {
	if len(candidates) == 0 {
		return nil, nil, errors.New("no candidate backends")
	}
	// Fast path: take the best candidate with a slot free right now (covers
	// uncapped backends and any idle one — no timer/ticker allocation).
	if b, slot, ok := r.scanForSlot(candidates); ok {
		return b, slot, nil
	}
	// Every candidate is momentarily saturated; wait, re-scanning each tick
	// until one frees, the caller gives up, or slotMaxWait elapses.
	//
	// "Saturated" and "gone" are different waits. Waiting is only ever right for a
	// worker that is busy, because a slot will free; a worker that has gone
	// unhealthy, expired or deregistered will never hand one back, so once no
	// candidate is routable at all there is nothing left to wait for. Without this
	// a request pinned to a worker that just went down would sit for the full
	// slotMaxWait — ten minutes — before reporting a timeout, when the answer was
	// already known on the first scan.
	deadline := time.NewTimer(slotMaxWait)
	defer deadline.Stop()
	poll := time.NewTicker(slotPollInterval)
	defer poll.Stop()
	for {
		if !r.anyRoutable(candidates) {
			return nil, nil, fmt.Errorf("no routable backend remains among %d candidate(s)", len(candidates))
		}
		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		case <-deadline.C:
			return nil, nil, fmt.Errorf("timed out after %s waiting for a free slot across %d backend(s)", slotMaxWait, len(candidates))
		case <-poll.C:
		}
		if b, slot, ok := r.scanForSlot(candidates); ok {
			return b, slot, nil
		}
	}
}

// anyRoutable reports whether at least one candidate is still worth waiting for.
func (r *Router) anyRoutable(candidates []*Backend) bool {
	for _, b := range candidates {
		if r.registry.stillRoutable(b.ID) {
			return true
		}
	}
	return false
}

// scanForSlot tries each candidate best-first and returns the first that has a
// free concurrency slot (acquiring it), or ok=false if all are full.
//
// The candidate list was snapshotted when the request was planned, and
// pickAndAcquire re-scans it every slotPollInterval for up to slotMaxWait — ten
// minutes, during which a worker can go unhealthy, expire, or be taken down for
// maintenance. Re-checking routability here is what stops a queue of waiting
// requests from being handed to a worker that stopped being a candidate several
// minutes ago; the ranking is still the plan's, only liveness is re-read.
func (r *Router) scanForSlot(candidates []*Backend) (*Backend, chan struct{}, bool) {
	for _, b := range candidates {
		if !r.registry.stillRoutable(b.ID) {
			continue
		}
		if slot, ok := r.registry.tryAcquireSlot(b.ID); ok {
			return b, slot, true
		}
	}
	return nil, nil, false
}

// stillRoutable reports whether a backend the router already chose still exists.
//
// Deliberately just existence, not the full eligible() predicate. The planner
// applied health, readiness and expiry when it built the candidate list, and
// re-applying them here would let a worker that merely blipped unhealthy for one
// health tick drop a request that was already waiting patiently for it — the
// candidate list is a decision, not a subscription. What must be re-read is
// existence, because deregistration also deletes the slot channel, and an absent
// slot channel reads as "uncapped" (see tryAcquireSlot).
func (r *Registry) stillRoutable(id string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.backends[id]
	return ok
}

// acquirePreference is a BOUNDED first-choice set for slot acquisition: try to
// land inside `keep` for `wait`, then fall back to the whole ranked list. Two
// unrelated policies want exactly this shape, so it is expressed once:
//
//   - the auto-difficulty quality floor (fix #2) — prefer a worker at or above
//     the classified tier before serving cheaper, because routing too cheap is
//     the quality-risky direction;
//   - the session tool-loop lock (session.go) — prefer the worker that opened
//     the loop before handing its tool result to a different model.
//
// A zero preference (nil keep) means no preference at all.
type acquirePreference struct {
	keep func(*Backend) bool
	// weaker are progressively WIDER fallback sets, most-preferred first, tried
	// the instant nothing in `keep` has a slot free. Without them a busy first
	// choice fell straight through to the whole ranked list — so a loaded local
	// worker could spill BELOW the quality bar while an idle remote worker above
	// it sat unused. The cascade is what makes the ladder a ladder.
	weaker []func(*Backend) bool
	wait   time.Duration
	// why labels the preference for logging; it is also what tells the caller
	// which "missed" story to tell.
	why string
}

// The rung labels `why` carries. Constants rather than bare literals because the
// labels are READ in another function — proxyToBackend picks which "missed"
// story to tell from them — and a literal compared against a literal written
// somewhere else is exactly how the cost line came to notice only one of the two
// free-preferring rungs.
const (
	prefLocalFree    = "local-free"
	prefFreeFirst    = "free-first"
	prefQualityFloor = "quality-floor"
	prefSessionLock  = "session-lock"
)

// preferredFree reports whether this preference was holding out for a worker
// that costs nothing, so that landing elsewhere spent money the router was
// trying not to spend. Both free-preferring rungs count; prefQualityFloor does
// not, because there every above-bar worker was metered from the start and no
// amount of waiting would have found a free one.
func (p acquirePreference) preferredFree() bool {
	return p.why == prefLocalFree || p.why == prefFreeFirst
}

// qualityFor is the quality score to judge b by for a request served in the
// given thinking mode. A no-think request reads the no-think benchmark score —
// the model that client actually talks to.
//
// A THINKING worker whose no-think score is not measured yet reads ZERO, not
// its mixed-mode Quality: the mixed score was earned with reasoning on and
// says nothing about the model a no-think client gets. The old fallback made
// being unmeasured an advantage — a still-profiling 284B CPU worker inherited
// its mixed q=93 on no-think requests, outranked every honestly-measured
// worker and drew all of Atlas's planner traffic onto a 10 tok/s single slot
// (2026-08-25). Unmeasured now ranks below every measured worker. The tier
// target that incident also poisoned no longer exists, but the rule still
// matters: backendScore reads this, and so does the judge's grader choice.
//
// A NON-thinking worker keeps reading Quality: it serves the same model in
// either mode, so the mixed score IS its no-think score — exact, and also
// what covers a pre-two-score cached profile where the profile-time copy
// into QualityNoThink has not happened yet (see needsNoThinkBackfill).
func qualityFor(b *Backend, thinkOff bool) int {
	if thinkOff {
		if b.QualityNoThink > 0 {
			return b.QualityNoThink
		}
		if b.Thinking {
			return 0 // unmeasured: never inherit the mixed score
		}
	}
	return b.Quality
}

// acquirePreferenceFor is the bounded first choice for an ordinary request: the
// workers the outcome matrix judged interchangeable on correctness and, among
// those, the FREE ones.
//
// Cost rides INSIDE the quality floor rather than beside it. PLAN.md's rule is
// "among the workers that clear the quality bar, prefer the free ones, and
// spill to a paid endpoint only when nothing free clears the bar or every free
// candidate is saturated past the existing grace period" — which is one
// preference set and the grace that already exists, not a second mechanism. Two
// independent graces would also compose into a wait nobody budgeted for.
//
// The set is the first of these that is a non-empty, STRICT subset of the
// candidates:
//
//  1. free and at/above the tier — the ordinary case, and the one PLAN.md
//     describes;
//  2. at/above the tier — every above-bar worker is paid, so there is nothing
//     free to hold out for and the floor keeps its own grace unchanged.
//
// A subset that is the whole list is no preference at all, and returning one
// would report every slow acquisition as a missed floor. target <= 0 (an
// unclassified request, or the fallback ranker) makes the tier test vacuous and
// leaves free-first on its own, which is what "prefer the free ones" means when
// there is no bar to clear.
//
// able bounds the cost/locality ladder to the leading candidates the outcome
// matrix judged interchangeable on correctness. Zero means unbounded, which is
// the tier path and the matrix's own fallback — neither has a correctness
// judgement to protect.
//
// Bounding it is the whole point. The matrix plan carries target == 0, so
// `aboveBar` was vacuously true for EVERY candidate and the ladder collapsed to
// "cheapest local worker first" applied across the entire ranked list —
// including the band the matrix had predicted WRONG. A fleet of one local CPU
// worker and one strong paid endpoint served every hard prompt on the CPU
// worker, for as long as it had a free slot, however confident the matrix was
// that it would get the answer wrong. That is exactly the scoping the tier
// ranker called deliberate ("below the bar the router has already missed the
// quality it wanted, and buying a worse answer to save money there is not a
// trade anyone asked for"), dropped in the migration.
//
// Inside the band the ladder is still right, and for a reason unrelated to cost:
// a local queue depth is exact and live while a relayed one is up to 15s stale
// and blind to the upstream's own clients, so ranking on remote completion times
// alone over-picks remote. That corrects a biased estimator. It just must not
// reach across a correctness boundary to do it.
func acquirePreferenceFor(candidates []*Backend, able int) acquirePreference {
	// The quality-target arm is gone with the tier ranker that set it. It read
	// `target <= 0 || qualityFor(b, thinkOff) >= target` and target was always 0
	// on the matrix path, so it admitted everything; what actually bounds the
	// ladder is the matrix's own correctness band.
	aboveBar := func(b *Backend) bool { return true }
	if able > 0 && able < len(candidates) {
		ids := make(map[string]bool, able)
		for _, b := range candidates[:able] {
			ids[b.ID] = true
		}
		aboveBar = func(b *Backend) bool { return ids[b.ID] }
	}
	free := func(b *Backend) bool { return aboveBar(b) && isFreeBackend(b) }
	ladder := []struct {
		why  string
		keep func(*Backend) bool
	}{
		// Costs nothing, is ours, clears the bar. Preferring local is not about
		// the link being slow — prefillSeconds already prices the round trip and
		// it is milliseconds against a generation measured in seconds. It is that
		// the two OCCUPANCY numbers are not equally trustworthy: a local queue
		// depth is exact and live, while a relayed row's is a snapshot up to one
		// refresh interval old (15s) and blind to the upstream's own clients
		// competing for those same slots. Remote completion times are therefore
		// systematically optimistic, and ranking on them alone over-picks remote.
		// This corrects a biased estimator, not an operator's preference.
		{prefLocalFree, func(b *Backend) bool { return free(b) && !isRelayRow(b) }},
		// Costs nothing, anywhere. Reached when every local worker above the bar
		// is loaded — an idle remote worker beats waiting on a busy local one.
		{prefFreeFirst, free},
		// Clears the bar at any price. Reached when the only above-bar workers
		// left are metered.
		{prefQualityFloor, aboveBar},
	}
	// Keep the rungs that actually decide something: an empty set selects
	// nothing, and one equal to the whole candidate list is not a preference.
	tiers := ladder[:0:0]
	for _, rung := range ladder {
		if n := len(filterCandidates(candidates, rung.keep)); n > 0 && n < len(candidates) {
			tiers = append(tiers, rung)
		}
	}
	if len(tiers) == 0 {
		return acquirePreference{}
	}
	pref := acquirePreference{keep: tiers[0].keep, wait: qualityFloorWait, why: tiers[0].why}
	for _, rung := range tiers[1:] {
		pref.weaker = append(pref.weaker, rung.keep)
	}
	return pref
}

// sessionLockPreference prefers the worker that served the previous turn. It
// outranks the quality floor while a tool loop is open: continuity matters more
// there than tier, because half a tool loop served by two models is worse than
// all of it served by the cheaper one.
func sessionLockPreference(incumbent string) acquirePreference {
	if incumbent == "" {
		return acquirePreference{}
	}
	return acquirePreference{
		keep: func(b *Backend) bool { return b.ID == incumbent },
		wait: sessionLockWait,
		why:  prefSessionLock,
	}
}

// pickAndAcquirePreferred is pickAndAcquire plus a bounded first-choice set.
//
// It partitions candidates into the preferred set and the rest:
//   - no preference, or the preferred set is empty (nothing to wait for): behave
//     exactly like pickAndAcquire — no extra wait.
//   - preferred set non-empty: try to acquire inside it first, polling up to
//     pref.wait; if none frees in time, fall back to the full ranked list (normal
//     spill, up to slotMaxWait). The bool result reports whether the request
//     MISSED the preference, so the caller can make that observable.
//
// Best-effort: it never blocks longer than slotMaxWait / the caller's context and
// never fails a request the un-preferred path would have served.
func (r *Router) pickAndAcquirePreferred(ctx context.Context, candidates []*Backend, pref acquirePreference) (*Backend, chan struct{}, bool, error) {
	if pref.keep == nil || pref.wait <= 0 {
		b, slot, err := r.pickAndAcquire(ctx, candidates)
		return b, slot, false, err
	}
	preferred := filterCandidates(candidates, pref.keep)
	if len(preferred) == 0 {
		// Nothing to wait for. Serve the best available immediately, exactly as
		// without the preference (the ranked list already orders the fallback).
		b, slot, err := r.pickAndAcquire(ctx, candidates)
		return b, slot, false, err
	}
	// Fast path: walk the ladder — the first rung with a slot free right now
	// wins, so a busy top choice steps down one rung rather than off the end.
	cascade := func() (*Backend, chan struct{}, bool) {
		if b, slot, ok := r.scanForSlot(preferred); ok {
			return b, slot, true
		}
		for _, keep := range pref.weaker {
			if b, slot, ok := r.scanForSlot(filterCandidates(candidates, keep)); ok {
				return b, slot, true
			}
		}
		return nil, nil, false
	}
	if b, slot, ok := cascade(); ok {
		return b, slot, false, nil
	}
	// Every preferred worker is momentarily full. Wait a BOUNDED grace for one to
	// free before spilling — but bail early if the caller's context is cancelled
	// (don't outlive the request).
	grace := time.NewTimer(pref.wait)
	defer grace.Stop()
	poll := time.NewTicker(slotPollInterval)
	defer poll.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil, nil, false, ctx.Err()
		case <-grace.C:
			// Grace elapsed: fall back to the full ranked list (normal spill, up to
			// slotMaxWait). Re-check the preferred set one last time first so a slot
			// that frees exactly at the deadline is still honoured.
			if b, slot, ok := cascade(); ok {
				return b, slot, false, nil
			}
			b, slot, err := r.pickAndAcquire(ctx, candidates)
			return b, slot, err == nil, err
		case <-poll.C:
		}
		if b, slot, ok := cascade(); ok {
			return b, slot, false, nil
		}
	}
}

// proxyToBackend dispatches a chat request to the best available candidate in
// plan. The plan carries the auto-difficulty quality floor (target 0 ⇒ no floor):
// when non-zero, acquisition prefers to wait BRIEFLY (qualityFloorWait) for an
// above-target worker before spilling below it (fix #2). The pinned/debug callers
// pass a bare plan, so their behaviour is unchanged.
func (r *Router) proxyToBackend(w http.ResponseWriter, req *http.Request, ident *identity, plan *routePlan, body []byte, chatReq *ChatRequest) {
	candidates, route := plan.candidates, plan.route
	if len(candidates) == 0 {
		writeUnavailable(w, r.retryAfterUnavailable(), "no backend available")
		return
	}
	// The caller has been identified; their credential is of no further use to
	// this router and must not leave it. Dropping it here rather than only at each
	// point an upstream request is built makes the property structural: the whole
	// dispatch subtree below — streaming, buffered, the strip-and-retry, an inline
	// escalation — can only send what setBackendCredential gives it, whichever
	// file happens to build the request. See setBackendCredential for why the old
	// relay-the-client's-header fallback had to go.
	req.Header.Del("Authorization")
	// Patch the forwarded body in a SINGLE pass: fill in max_tokens when the
	// client omitted it (the backend needs an explicit limit) and write
	// chat_template_kwargs.enable_thinking per the resolved decision. One
	// unmarshal/marshal instead of two, so a multi-MB vision body is copied once.
	// The thinking decision uses the SAME resolution that gated selection (plan.cl
	// is the classification threaded from planRoute; nil on pinned/debug, where
	// only explicit thinking signals apply) so the worker and the enable_thinking
	// we forward can never disagree.
	tr := r.resolveThinking(chatReq, route, plan.cl)
	// Same job shape planRoute ranked with — session discount included — so the
	// live prefill/TTFT samples are attributed to the right prompt size and skipped
	// for thinking turns (see Registry.observe).
	thinking := tr.hardThink || tr.softThink
	job := costForRequest(chatReq, thinking).withIncumbent(plan.session.incumbent)
	// Inject the default budget ONLY when the client set none (max_tokens and
	// max_completion_tokens both absent/null/0) — never alongside a real client
	// budget, where a second conflicting limit is resolved differently per
	// backend dialect.
	injectMaxTokens := 0
	if !chatReq.ClientSetMaxTokens {
		injectMaxTokens = chatReq.MaxTokens
	}

	start := time.Now()

	// Acquire a concurrency slot before dispatching, spilling to the next-best
	// backend when the best one is momentarily full (see pickAndAcquirePreferred).
	// A backend with no declared cap is always immediately available. The request
	// stays queued here — not consuming a vLLM worker — until a slot frees, the
	// caller gives up, or slotMaxWait elapses.
	//
	// Which bounded preference applies: normally the quality floor (wait briefly
	// for an above-target worker before serving below the tier). While a TOOL LOOP
	// is open the session lock wins instead — handing a tool result to a model that
	// never emitted the matching tool call breaks the loop outright, which is worse
	// than serving that turn a tier low.
	pref := acquirePreferenceFor(candidates, plan.able)
	if plan.session.locked() {
		pref = sessionLockPreference(plan.session.incumbent)
	}
	backend, slot, missedPref, slotErr := r.pickAndAcquirePreferred(req.Context(), candidates, pref)
	if slotErr != nil {
		r.logSlotUnavailable(ident, clientIP(req), candidates[0], route, chatReq.Stream, body, start, slotErr)
		writeUnavailable(w, r.retryAfterSaturated(), fmt.Sprintf("no backend slot available: %s", slotErr))
		return
	}
	// The slot and the active-request count belong to whichever worker is serving
	// RIGHT NOW: inline escalation may hand both over to a better worker mid-request
	// (see escalate.go), so unwind through the variables rather than capturing
	// today's values in the defer.
	defer func() { r.registry.releaseSlot(slot) }()
	// Patch only now that the worker is known: the max_tokens clamp needs its
	// context. logSlotUnavailable above logs the unpatched body, which is what it
	// wants anyway — nothing was forwarded. Keep the client's original: an inline
	// escalation re-patches from THAT rather than inheriting a clamp shaped by the
	// worker it is replacing (see escalate.go).
	rawBody := body
	body = patchForwardedBody(body, injectMaxTokens, budgetCeiling(backend, job), tr.forBackend(backend), backend.ServedID)
	// Fields this endpoint has already refused as unrecognised never go out
	// again — see stripAndRetry. Read through the registry rather than off the
	// clone, so an aged-out verdict is re-tested rather than believed for ever.
	// Usually empty, and then free.
	body = r.stripLearned(body, rawBody, backend.ID)
	// Make a missed preference observable WITHOUT touching X-LLM-Route, which
	// clients and the arena both parse: a dedicated response header plus a log
	// line. The route header still records the target the prompt
	// was classified to need; this records that we couldn't honour it within the
	// grace and served elsewhere.
	if missedPref {
		if pref.why == prefSessionLock {
			log.Printf("session lock: %s tool loop moved off incumbent=%s to backend=%s after %s grace — no incumbent slot freed",
				route, plan.session.incumbent, backend.ID, sessionLockWait)
		} else {
			// Read the two facts off the worker we actually landed on rather than
			// off the preference: one missed floor can be a tier downgrade, a spill
			// onto a paid endpoint, or both, and a slot that frees during the
			// handover can make it neither.
			//
			// EITHER free-preferring rung counts. This tested "free-first" alone,
			// which is the rung reached only when there is no free LOCAL worker to
			// hold out for — so on the ordinary fleet, local workers that cost
			// nothing beside a metered endpoint, the top rung is "local-free" and
			// the one spill that actually spends money was the one nothing said a
			// word about.
			if pref.preferredFree() && !isFreeBackend(backend) {
				log.Printf("cost: %s served on PAID backend=%s (in %g / out %g per Mtok) after %s grace — no free worker above the bar freed a slot",
					route, backend.ID, backend.InputPricePerMtok, backend.OutputPricePerMtok, qualityFloorWait)
			}
		}
	}
	// TTFT base: measure first-token latency from the moment we hold a worker
	// slot, NOT from request arrival — pickAndAcquire can block up to slotMaxWait,
	// and folding that router-side queue wait into the worker's first-token
	// latency would pollute the ObservedTTFTMillis EWMA expectedLatency relies on
	// (fix #4). logEntry.DurationMillis below still measures total wall time from
	// start (queue included).
	ttftBase := time.Now()

	r.registry.incActive(backend.ID, 1)
	defer func() { r.registry.incActive(backend.ID, -1) }()

	logEntry := RequestLog{
		CreatedAt:    start.UTC(),
		BackendID:    backend.ID,
		BackendModel: backend.Model,
		Route:        route,
		ObservedTPS:  backend.ObservedTPS,
		CertifiedTPS: backend.Certification.TokensPerSec,
		BaselineTPS:  backend.BaselineTPS,
		SpeedScore:   speedScore(backend),
		Stream:       chatReq.Stream,
		Input:        string(body),
		KeyID:        ident.logKeyID(),
		ClientIP:     clientIP(req),
		Thinking:     thinkingLogValue(tr),
		// Read AFTER incActive above, so this request is included: "how loaded was
		// the worker while serving this" is what inflates the duration, and a count
		// taken before dispatch would read 0 for the only request in flight.
		Concurrency: r.registry.activeCount(backend.ID),
	}
	capture := newBoundedCapture(r.cfg.LogMaxBodyBytes)
	var stats *sseStats
	if chatReq.Stream {
		stats = &sseStats{}
	}
	// escalated records that an inline escalation happened, so X-Llm-Escalated can
	// report the repair even though the answer finally returned was fine.
	escalated := false
	defer func() {
		logEntry.DurationMillis = time.Since(start).Milliseconds()
		logEntry.Output = capture.String()
		// Session affinity: remember which worker served this conversation, so its
		// next turn prefers the same one (its prefix is cached there). Only on a
		// clean 2xx, and only for a route the ROUTER chose — a pinned request says
		// nothing about where the conversation belongs.
		if logEntry.Error == "" && logEntry.StatusCode >= 200 && logEntry.StatusCode < 300 && route != "pinned" && route != "debug" {
			r.sessions.remember(plan.session.key, backend.ID)
		}
		// The adapter/judge/log-insert bookkeeping runs in a goroutine: this defer
		// executes BEFORE releaseSlot (defer LIFO), and a synchronous SQLite insert
		// here would extend the busy window of a max_concurrency=1 worker after
		// every request.
		// Charge the key's budget from what the endpoint reported it used, falling
		// back to what routing estimated when it reported nothing (or when the
		// capture dropped the usage block). A budget is a spending bound, not an
		// invoice, so an approximation in the fallback is the right trade.
		// Prompt/completion counts as the ENDPOINT reported them, kept separately
		// from the single `charged` total that budgeting uses. Budgeting only needs
		// a sum and happily estimates one; a timing model needs the completion
		// count specifically, and needs to know when it is real rather than
		// inferred — so these stay 0 when the endpoint reported nothing.
		logEntry.PromptTokens = lastJSONInt(capture.Bytes(), "prompt_tokens")
		logEntry.CompletionTokens = lastJSONInt(capture.Bytes(), "completion_tokens")
		charged := usageTotalTokens(capture.Bytes())
		if charged == 0 {
			charged = job.promptTokens
			if stats != nil {
				charged += stats.genTokens()
			} else {
				// A BUFFERED reply has no per-token deltas to count, so the reply's own
				// size stands in for the answer inside it. Charging the prompt alone
				// made a generation of any length free on the one shape where every
				// OpenAI-compatible endpoint does report usage — so this only runs for
				// an endpoint that did not, and erring high there is the right way to
				// err for a spending bound.
				charged += estimateGenTokens(capture.total)
			}
		}
		entry := redactForRelay(logEntry, ident)
		served := backend
		caller := ident
		autoRoute := plan.auto
		go func() {
			r.recordKeyUse(caller, charged)
			// A failed or aborted transfer (client hung up mid-stream, worker died)
			// says nothing about answer quality, so it teaches nothing. A relayed
			// request was classified by the router that sent it, which is already
			// learning from this same outcome — see learnFromRelay.
			clean := entry.Error == "" && entry.StatusCode >= 200 && entry.StatusCode < 300
			// The judge is gated on whether THE ROUTER chose this worker, structurally
			// (plan.auto), not by sniffing the route string. It used to key on a
			// literal "route:d=" prefix, which meant it never ran on a matrix-routed
			// request: the matrix's own feedback loop was open, closing only when the
			// embeddings worker was down and the tier path took over.
			//
			// Judging parses the answer text back out of the capture, so it is
			// skipped when truncation removed part of it — a half answer grades as
			// garbage.
			if autoRoute && learnFromRelay(caller) && clean && capture.truncated() <= 0 {
				// plan.cl carries the embedding the classifier already computed for
				// routing; the judge reuses it to pick a grader that is strong on
				// THIS KIND of prompt rather than strong on average.
				r.maybeJudge(chatReq.Messages, chatReq.Stream, served, route, entry.Output,
					entry.Thinking == thinkingOn, entry.DurationMillis, classificationVec(plan.cl))
			}
			if err := r.logs.Insert(context.Background(), entry); err != nil {
				log.Printf("request log insert failed backend=%s: %v", served.ID, err)
			}
		}()
	}()

	if s := plan.session.outcome(backend.ID); s != "" {
		w.Header().Set("X-LLM-Session", s)
	}
	// Which member of the named group served, or "fallback" when none qualified
	// and the request routed automatically. Observable the same way X-LLM-Route
	// and X-LLM-Session are, and set here so streaming and buffered dispatch both
	// carry it. Deliberately NOT folded into X-LLM-Route, whose "route:d=" form
	// the tier adapter parses.
	if g := plan.group.header(); g != "" {
		w.Header().Set("X-LLM-Group", g)
	}
	// A request that asked for the ensemble and is being served here instead asked
	// for something this router will not fake (see expert.go). Saying so is the
	// difference between a client learning that tools and panels don't mix and a
	// client believing it got a panel.
	if x := plan.expert.header(); x != "" {
		w.Header().Set("X-LLM-Expert", x)
	}

	// Streaming requests can't be retried — once headers are committed we
	// can't rewind. Fall through to the single-shot path. ttftBase (slot
	// acquisition), not start (request arrival), is the TTFT/decode measurement
	// base so router-side queue wait doesn't pollute the latency EWMAs.
	// ONE dispatch for both paths. The streaming path needs the plan and the slot
	// to fail over before its first byte, which is exactly what the buffered path
	// already had — building it here rather than only for the buffered branch is
	// what lets both share redispatch's slot-swap.
	d := &dispatch{
		backend:   &backend,
		slot:      &slot,
		body:      body,
		raw:       rawBody,
		plan:      plan,
		chatReq:   chatReq,
		job:       job,
		tr:        tr,
		inject:    injectMaxTokens,
		budget:    callerBudget(req, chatReq),
		start:     start,
		log:       &logEntry,
		output:    capture,
		escalated: &escalated,
	}
	// ttftBase (slot acquisition), not start (request arrival), is the TTFT and
	// decode measurement base, so router-side queue wait does not pollute the
	// latency EWMAs.
	if chatReq.Stream {
		r.dispatchStreaming(w, req, d, route, capture, stats, ttftBase, thinking)
		return
	}
	r.dispatchBuffered(w, req, d)
}

// dispatchStreaming forwards a single SSE-style response from backend to
// client. Used for streaming requests where retrying after partial writes
// isn't possible.
// ttftBase is when the worker slot was acquired (NOT request arrival); the
// first chunk's arrival minus ttftBase is the worker's true first-token latency,
// excluding any router-side queue wait (see proxyToBackend / fix #4).
func (r *Router) dispatchStreaming(w http.ResponseWriter, req *http.Request, d *dispatch, route string, capture io.Writer, stats *sseStats, ttftBase time.Time, thinking bool) {
	backend, body, logEntry, job := *d.backend, d.body, d.log, d.job
	// Idle watchdog instead of a wall-clock cap: the old client-level
	// BACKEND_TIMEOUT_SECONDS bounded the WHOLE stream, killing legitimate
	// long generations mid-flow while still letting a silently hung backend
	// pin its concurrency slot for the full 10 minutes. Progress (any bytes,
	// heartbeats included) resets the timer; true silence cancels quickly.
	ctx, cancel := context.WithCancel(req.Context())
	defer cancel()
	upstreamURL := upstreamChatURL(backend)
	proxyReq, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(body))
	if err != nil {
		logEntry.StatusCode = http.StatusInternalServerError
		logEntry.Error = err.Error()
		writeJSON(w, http.StatusInternalServerError, validationError{Message: err.Error()})
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")
	setBackendCredential(proxyReq, backend)
	r.stampRelayChain(proxyReq, req, backend)

	idle := r.cfg.BackendIdleTimeout
	var watchdog *time.Timer
	if idle > 0 {
		watchdog = time.AfterFunc(idle, cancel)
		defer watchdog.Stop()
	}
	progress := func() {
		if watchdog != nil {
			watchdog.Reset(idle)
		}
	}

	resp, err := r.streamClient.Do(proxyReq)
	// PRE-FIRST-BYTE FAILOVER. "SSE bytes cannot be recalled" is true from the
	// first byte WRITTEN TO THE CLIENT, not from the dial — and nothing has been
	// written yet. A worker that answers the dial with a 503 has told us it
	// cannot serve this request, which is a routing failure exactly as it is on
	// the buffered path, and the client is none the wiser if we ask someone else.
	if other, newResp, moved := r.streamFailover(ctx, req, d, resp, err); moved {
		if resp != nil {
			resp.Body.Close()
		}
		backend, resp, err = other, newResp, nil
		// Keep the local mirror of what was actually sent. The URL is NOT
		// re-derived: streamFailover has already dialled the new worker, and a
		// second copy of the address nothing reads is a value computed and thrown
		// away.
		body = d.body
		w.Header().Set("X-LLM-Failover", fmt.Sprintf("%s->%s", logEntry.BackendID, backend.ID))
		logEntry.BackendID = backend.ID
		logEntry.BackendModel = backend.Model
		logEntry.ObservedTPS = backend.ObservedTPS
		logEntry.CertifiedTPS = backend.Certification.TokensPerSec
		logEntry.BaselineTPS = backend.BaselineTPS
		logEntry.SpeedScore = speedScore(backend)
		if s := d.plan.session.outcome(backend.ID); s != "" {
			w.Header().Set("X-LLM-Session", s)
		}
	}
	if err != nil {
		// A client hangup surfaces here as context-canceled (ctx inherits
		// req.Context()). That says nothing about the backend — with long prefills
		// an abort storm (user spamming stop, agent retry loops) would otherwise
		// trip the circuit breaker and eject a perfectly healthy worker. An idle
		// watchdog cancellation, by contrast, IS the backend's fault and counts.
		if req.Context().Err() == nil {
			r.registry.setError(backend.ID, err.Error())
			r.registry.noteProxyResult(backend.ID, false)
		}
		logEntry.StatusCode = http.StatusBadGateway
		logEntry.Error = err.Error()
		writeJSON(w, http.StatusBadGateway, validationError{Message: err.Error()})
		return
	}
	defer resp.Body.Close()
	r.registry.noteProxyResult(backend.ID, resp.StatusCode < 500)

	setRouteHeaders(w, backend, route, logEntry)
	// Forward upstream headers — without Content-Type: text/event-stream, Go
	// content-sniffs the first chunk to text/plain and strict SSE clients
	// (EventSource, OpenAI SDKs) refuse the stream.
	copyHeaders(w.Header(), resp.Header)
	// Same invariant as the buffered path: a 503 from this router always says
	// when to come back. writeUnavailable's comment claims every northbound 503
	// goes through it; two relayed ones do not, and this was one of them.
	r.backfillRetryAfter(w.Header(), resp.StatusCode)
	w.WriteHeader(resp.StatusCode)
	logEntry.StatusCode = resp.StatusCode
	progress() // headers arrived — the idle window now measures body silence
	sink := capture
	if stats != nil {
		sink = io.MultiWriter(capture, stats)
	}
	// Strip nameless tool-call slots before any client sees them: one malformed
	// delta poisons a whole agent conversation (see toolcalls.go).
	guard := newToolCallGuard(resp.Body, backend)
	firstByte, err := copyStreaming(w, guard, sink, progress)
	guard.report()
	if err != nil {
		if ctx.Err() != nil && req.Context().Err() == nil {
			err = fmt.Errorf("backend sent no bytes for %s (idle timeout): %w", idle, err)
		}
		logEntry.Error = err.Error()
		log.Printf("proxy copy failed backend=%s: %v", backend.ID, err)
		// The status code was committed with the preamble, so the failure can only
		// be reported inside the stream. Skip it when the CLIENT is the one that
		// went away (there is nobody to tell) or when what we forwarded was never a
		// stream in the first place — see writeSSEError.
		if req.Context().Err() == nil && isEventStream(w.Header()) {
			writeSSEError(w, fmt.Sprintf("upstream stream from backend %q failed: %s", backend.ID, err))
		}
		return
	}
	// Measure decode throughput over the post-first-token window so TTFT/prefill
	// don't deflate it (see observe). The first chunk's arrival minus ttftBase
	// (slot acquisition) is ≈ TTFT — router-side queue wait is excluded. Token
	// count comes from the full-stream stats, not the (possibly truncated) capture.
	if !firstByte.IsZero() && stats != nil {
		r.registry.observe(backend.ID, firstByte.Sub(ttftBase), time.Since(firstByte), stats.genTokens(), job.promptTokens, thinking)
		// Keep the per-request value as well as the EWMA it feeds. The EWMA answers
		// "how fast is this worker right now", which is what ranking needs; the row
		// answers "how long did THIS prompt take on this worker", which is what a
		// timing model has to be trained on. The average cannot be un-averaged
		// later, so a measurement discarded here is gone.
		if logEntry != nil {
			logEntry.TTFTMillis = firstByte.Sub(ttftBase).Milliseconds()
		}
	}
}

func setRouteHeaders(w http.ResponseWriter, backend *Backend, route string, logEntry *RequestLog) {
	w.Header().Set("X-LLM-Backend-ID", backend.ID)
	w.Header().Set("X-LLM-Backend-Model", backend.Model)
	w.Header().Set("X-LLM-Backend-Model-Name", niceModelName(backend))
	w.Header().Set("X-LLM-Backend-URL", backend.URL)
	w.Header().Set("X-LLM-Route", route)
	w.Header().Set("X-LLM-Observed-TPS", fmt.Sprintf("%.3f", logEntry.ObservedTPS))
	w.Header().Set("X-LLM-Certified-TPS", fmt.Sprintf("%.3f", logEntry.CertifiedTPS))
	w.Header().Set("X-LLM-Baseline-TPS", fmt.Sprintf("%.3f", logEntry.BaselineTPS))
	w.Header().Set("X-LLM-Speed-Score", strconv.Itoa(logEntry.SpeedScore))
}

// selectBackends is the narrow view of planRoute: the eligible backends that
// satisfy the request's hard context/feature requirements, ranked best-first,
// alongside the prompt's classification (nil when classification was unavailable
// or the request had no messages) so the caller can thread it into the body patch
// without re-classifying, and the auto-difficulty target QUALITY FLOOR (0 ⇒ no
// floor — the fallback/feature paths and any non-auto request). Model-tier
// selection is always automatic (difficulty-based) — there are no client
// quality/speed overrides; capability hints (thinking, required_features,
// min_context_k) only hard-filter, they never pick a tier.
//
// Its one production caller is /v1/completions, which forwards the body verbatim
// through proxyPassthrough and therefore takes the plain pickAndAcquire spill —
// the candidate list is walked best-first when the top backend has no free slot,
// so a burst spreads across the fleet, but there is no bounded grace and the
// target it returns is DISCARDED. The quality floor described above is applied on
// the chat path only, where proxyToBackend builds an acquirePreferenceFor from
// the whole plan. The chat path does not go through here at all: it
// calls planRoute directly, because it needs the rest of the plan.
//
// The request is classified ONCE here; both axes are used: difficulty drives the
// quality tier, and reasoning drives the thinking decision — which now also gates
// SELECTION (fix #1), not just the body patch. A high-reasoning prompt prefers a
// thinking-capable worker (soft for auto, hard for explicit "on"), so the
// enable_thinking we later forward lands on a worker that can act on it instead of
// being a no-op on a non-thinking model.
//
// budget is how long the caller is still willing to wait (0 = unknown); when
// known, workers that cannot finish the job inside it are filtered out first.
func (r *Router) selectBackends(req *ChatRequest, budget time.Duration) ([]*Backend, string, *classification, error) {
	plan, err := r.planRoute(req, budget, false)
	if err != nil {
		return nil, "", nil, err
	}
	return plan.candidates, plan.route, plan.cl, nil
}

// routePlan is everything selection decided about one request. selectBackends
// returns the subset the proxy path needs; /v1/route-preview renders the whole
// thing. They share ONE code path on purpose — a preview that re-derived the
// decision would eventually explain a route the router didn't take.
type routePlan struct {
	candidates []*Backend
	route      string
	// auto records that the ROUTER chose this worker, structurally rather than by
	// sniffing the route string. Three things hang off "did we choose this":
	// inline escalation, the online tier adapter and the background judge, and all
	// three used to test the route for a literal "route:d=" prefix. That worked
	// only while every auto route produced that exact shape — the moment the
	// outcome matrix started emitting "route:outcome:…" all three silently
	// switched off, taking the judge (and therefore the matrix's own feedback
	// loop) with them. A field cannot drift out of sync with a format string.
	auto bool
	cl   *classification
	// able is how many LEADING candidates the outcome matrix judged
	// interchangeable on correctness. Acquisition may reorder inside that prefix
	// and must not reorder across it — see acquirePreferenceFor. Zero means no
	// prefix is protected, which is the matrix's own fallback and the degraded
	// ranker.
	able    int
	job     jobCost
	tr      thinkingResolution
	session sessionRoute
	// group is what group resolution decided, and is the zero value when the
	// client named no group (see groups.go).
	group groupRoute
	// expert is what ensemble resolution decided, and is the zero value when the
	// client named no ensemble (see expert.go).
	expert expertRoute
	// rejected records why each eligible worker was hard-filtered out. Only
	// populated when explain is set — the proxy path allocates nothing for it.
	rejected []rejection
}

// widestContext returns the candidate with the largest window among those that
// pass every hard requirement EXCEPT context, or nil when context was not the
// only thing in the way.
//
// The filter is copied with its context fields cleared rather than re-derived,
// so "the same filter minus context" cannot drift from the filter itself.
func widestContext(candidates []*Backend, f hardFilter) *Backend {
	f.minContextK, f.promptChars, f.reserveTokens = 0, 0, 0
	var best *Backend
	for _, b := range candidates {
		if admitReason(b, f) != "" {
			continue
		}
		if best == nil || usableContextTokens(b) > usableContextTokens(best) {
			best = b
		}
	}
	return best
}

// rejection is one worker the hard filter dropped, and why.
type rejection struct {
	ID     string `json:"id"`
	Reason string `json:"reason"`
}

// hardFilter is the set of non-negotiable requirements a worker must meet to
// serve a request. Built once per request; applied through admitReason so
// selection and the preview can never disagree about who was eligible.
type hardFilter struct {
	wantModel string
	// minContextK is an explicit requirements.min_context_k. A number the CLIENT
	// stated, so it is used as given and never re-estimated.
	minContextK int
	// promptChars is the request's text, and reserveTokens the headroom kept for
	// the answer. They are kept as characters rather than as one token count
	// because the divisor is the MODEL's — see contextNeededK.
	promptChars      int
	reserveTokens    int
	ratios           *tokenRatios // measured chars-per-token; nil ⇒ the default
	needTools        bool
	requiredFeatures []string
	hardThink        bool
}

// contextNeededK sizes this request against ONE candidate, in K.
//
// Per candidate, because a token is not a fixed amount of text: two models given
// the same 700KB of Go and JSON disagree about how many tokens it is by more than
// the margin that decides whether it fits. The router sized every prompt at a
// flat 3 characters per token and compared the result against every worker, which
// on measured traffic nearer 3.5 inflated a 200K prompt by a third — enough to
// have every worker in a fleet reject a prompt every one of them could hold.
func (f hardFilter) contextNeededK(b *Backend) int {
	if f.minContextK > 0 {
		return f.minContextK
	}
	if f.promptChars <= 0 {
		return 0
	}
	model := ""
	if b != nil {
		model = b.Model
	}
	tokens := tokensForChars(f.promptChars, f.ratios.forModel(model)) + f.reserveTokens
	return int(math.Ceil(float64(tokens) / 1024.0))
}

// admitReason reports why b cannot serve the request, or "" if it can.
func admitReason(b *Backend, f hardFilter) string {
	if isEmbeddingsOnly(b) {
		return "embeddings-only worker (cannot serve chat)"
	}
	if f.wantModel != "" && !backendServesModel(b, f.wantModel) {
		return fmt.Sprintf("does not serve model %q", f.wantModel)
	}
	// Filter on the window retrieval was MEASURED to survive, not the one the
	// server advertises. A model claiming 256K that loses facts past 64K passes an
	// advertised-context filter for a 100K prompt and then answers it badly —
	// which shows up as "the agent got confused", never as an error. Falls back to
	// the claim when no probe has run, so an unprobed worker is not filtered out
	// of everything. See usableContextTokens.
	needed := f.contextNeededK(b)
	if usableK := usableContextTokens(b) / 1024; needed > 0 && usableK > 0 && usableK < needed {
		if b.ContextProbe != nil && b.ContextProbe.UsableTokens > 0 && usableK < b.ContextK {
			return fmt.Sprintf("context %dK < %dK required (measured; %dK advertised)",
				usableK, needed, b.ContextK)
		}
		return fmt.Sprintf("context %dK < %dK required", usableK, needed)
	}
	if f.needTools && !hasFeature(b, "tools") {
		return "no tools support"
	}
	for _, feature := range f.requiredFeatures {
		if !hasFeature(b, feature) {
			return fmt.Sprintf("missing required feature %q", feature)
		}
	}
	// An EXPLICIT thinking:"on" hard-filters to workers that can think — a user
	// demand that may legitimately leave nothing (a 503). "off"/"auto" never hard-
	// filter here: off is honoured by patching enable_thinking into the forwarded
	// body, which any worker can serve; an auto "think" is applied as a SOFT
	// preference, never a 503.
	if f.hardThink && !b.Thinking {
		return "cannot think (thinking explicitly required)"
	}
	return ""
}

// planRoute runs the full selection pipeline. explain=true additionally records
// the hard-filter rejections for /v1/route-preview.
func (r *Router) planRoute(req *ChatRequest, budget time.Duration, explain bool) (*routePlan, error) {
	candidates := r.registry.eligible()
	if len(candidates) == 0 {
		return nil, errors.New("no healthy LLM backends registered")
	}

	reqs := req.Requirements
	if reqs == nil {
		reqs = &Requirements{}
	}

	// An explicitly named model is the OpenAI-standard way for a client to choose,
	// and the only one a coding harness knows how to use. It constrains WHICH
	// workers are candidates; everything downstream (context/feature filters,
	// thinking, completion-time ranking, slot spilling) still runs, so naming a
	// model that three workers serve still load-balances across those three.
	// "default" — what every clabtree guest sends — and an absent model mean auto.
	wantModel := requestedModel(req)
	// The `expert` ensemble resolves ahead of everything, groups included: it is
	// a route rather than a model, so no worker can capture the name and no group
	// may be created with it (see expert.go). The filter below is cleared for the
	// same reason the group fallback clears it — the panel is drawn from the whole
	// fleet, not from a worker answering to "expert".
	expert := expertRoute{}
	if isExpertModel(wantModel) {
		expert.asked = true
		wantModel = ""
	}
	// A GROUP name resolves ahead of a model id, an alias and a worker id, which
	// is what fixes the precedence between the four spellings the model field
	// accepts (see groups.go). It is never an unknown model: a group always
	// routes, falling back to automatic selection when no member qualifies.
	group, isGroup := r.groups.lookup(wantModel)
	if !isGroup && wantModel != "" && len(filterCandidates(candidates, func(b *Backend) bool { return backendServesModel(b, wantModel) })) == 0 {
		return nil, unknownModelError{name: wantModel}
	}

	// Classify ONCE (best-effort; cached by prompt). Reused for both the quality
	// tier (difficulty) and the thinking decision (reasoning). A bare
	// /v1/completions call carries no messages, so classifyText is empty and the
	// classifier reports unavailable → cl stays nil and we fall through to the
	// non-classified paths below.
	var cl *classification
	if r.classifier != nil && len(req.Messages) > 0 {
		if c, ok := r.classifier.classify(req); ok {
			cl = &c
		}
	}

	// Resolve the SAME thinking decision the body patch will use, BEFORE the hard
	// filter, so selection and the enable_thinking patch can never disagree. The
	// auto path only fires on this normal "route" (selectBackends is never the
	// pinned/debug path).
	tr := r.resolveThinking(req, "route", cl)

	// The characters, not a token count: the divisor is the model's, and which
	// model is what the filter is about to decide. See hardFilter.contextNeededK.
	promptChars := messageChars(req.Messages)
	if len(req.Tools) > 0 && string(req.Tools) != "null" {
		promptChars += len(req.Tools)
	}
	needTools := len(req.Tools) > 0 && string(req.Tools) != "null"
	requiredFeatures := normalizeFeatures(reqs.RequiredFeatures)
	if requestNeedsVision(req.Messages) {
		requiredFeatures = append(requiredFeatures, "vision")
		requiredFeatures = normalizeFeatures(requiredFeatures)
	}

	hf := hardFilter{
		wantModel:        wantModel,
		minContextK:      reqs.MinContextK,
		promptChars:      promptChars,
		reserveTokens:    contextReserveTokens(req),
		ratios:           r.ratios,
		needTools:        needTools,
		requiredFeatures: requiredFeatures,
		hardThink:        tr.hardThink,
	}
	// Resolve the group now the hard filter exists — "qualifies" means "past the
	// hard filters", so the two cannot be settled in either order. A resolved
	// member replaces the group name in the filter; a fallback removes it
	// entirely, because a group is a preference and must never be a refusal.
	gr := groupRoute{}
	if isGroup {
		hf.wantModel = ""
		gr.name = group.Name
		if member, ok := group.resolve(candidates, hf); ok {
			gr.member, hf.wantModel = member, member
		} else {
			gr.fallback = true
			// The ROUTER is choosing now, so the route reads "route:" and the tier
			// adapter and judge learn from it — see routeKind.
			wantModel = ""
			if !explain {
				// The preview renders the fallback instead of logging it; logging on
				// a preview would make an inspection indistinguishable from traffic.
				log.Printf("group %q: no member is registered, healthy and past the hard filters (members=%v) — falling back to automatic routing",
					group.Name, group.Members)
			}
		}
	}
	var rejected []rejection
	filtered := filterCandidates(candidates, func(b *Backend) bool {
		reason := admitReason(b, hf)
		if reason != "" && explain {
			rejected = append(rejected, rejection{ID: b.ID, Reason: reason})
		}
		return reason == ""
	})
	overflow := false
	if len(filtered) == 0 {
		// Everything was dropped, and the reason matters. A missing feature or an
		// unserved model is a fact: no amount of trying will make a worker without
		// tools support grow them, and refusing is the honest answer.
		//
		// Context is not a fact, it is this router's ESTIMATE — and the endpoint
		// holds the truth. So when context is the only thing standing between the
		// request and a worker, send it to the widest window and let the tokenizer
		// that actually counts rule on it. Two outcomes, both better than a 503:
		// it fits, and the caller gets their answer plus a calibration sample; or
		// it does not, and the engine's own 400 says by how much, in exact tokens,
		// where the router could only have guessed. That refusal is also a sample —
		// see contextLimitPromptTokens — so the estimate that caused it is corrected
		// by the very request it turned away.
		if wantModel == "" {
			if widest := widestContext(candidates, hf); widest != nil {
				log.Printf("context overflow: no worker admits an estimated %dK prompt — sending it to %s (%dK) and letting the endpoint rule",
					hf.contextNeededK(widest), widest.ID, usableContextTokens(widest)/1024)
				filtered, overflow = []*Backend{widest}, true
			}
		}
	}
	if len(filtered) == 0 {
		if wantModel != "" {
			return nil, fmt.Errorf("no worker serving model %q satisfies hard context/feature requirements", wantModel)
		}
		return nil, errors.New("no backend satisfies hard context/feature requirements")
	}

	// The ensemble asks every MODEL, so its panel is the hard-filtered set as it
	// stands here — before the soft thinking preference narrows it below. A panel
	// missing every non-thinking model is not the fleet's opinion (see expert.go).
	panel := filtered

	// Auto-derived "think": prefer thinking-capable workers, but only if at least
	// one survives the other hard filters — otherwise fall back to the full set.
	// Auto-thinking must NEVER 503 a request on its own; it's best-effort steering,
	// not a demand (that's what the explicit hard filter above is for).
	if tr.softThink {
		if thinkers := filterCandidates(filtered, func(b *Backend) bool { return b.Thinking }); len(thinkers) > 0 {
			filtered = thinkers
		}
	}

	// The job's real cost — prompt size plus the output length implied by the
	// thinking decision resolved above — drives both the deadline filter and the
	// completion-time ranking.
	job := costForRequest(req, tr.hardThink || tr.softThink)

	// The ensemble is settled before anything below, because none of it applies:
	// it picks no single worker, so there is no tier to rank for, no incumbent to
	// stay with and no deadline filter that could mean anything across N workers.
	// A request it declines (tools, an open tool loop) carries on down the normal
	// path with the reason attached, which is what X-LLM-Expert then reports.
	if expert.asked {
		if expert.fallback = expertFallback(req); expert.fallback == "" {
			return &routePlan{
				candidates: panel,
				route:      routeExpert,
				cl:         cl,
				job:        job,
				tr:         tr,
				expert:     expert,
				rejected:   rejected,
			}, nil
		}
		if !explain {
			// Logged where it is ACTED on, not where it is previewed — an inspection
			// must not be indistinguishable from traffic (see the group fallback).
			log.Printf("expert: %s — this request cannot be answered by a panel, routing it automatically instead", expert.fallback)
		}
	}

	// Session affinity. Resolved AFTER the hard filter so an incumbent that no
	// longer qualifies for this turn (died, too little context, wrong model) is
	// simply not an incumbent. It discounts the incumbent's prefill inside the
	// existing ranking rather than overriding it — see session.go.
	sess := r.sessions.resolve(req, filtered)
	if sess.incumbent != "" {
		job = job.withIncumbent(sess.incumbent)
	}

	// Drop workers that can't finish inside the caller's declared budget.
	// Best-effort: never empties the candidate set (see deadlineFilter).
	if trimmed, applied := deadlineFilter(filtered, job, budget); applied {
		log.Printf("deadline filter: budget=%s job=%dp/%dc — %d of %d workers can finish",
			budget.Round(time.Second), job.promptTokens, job.outputTokens, len(trimmed), len(filtered))
		filtered = trimmed
	}

	// Classify once, use both axes: difficulty carries the vector the outcome
	// matrix ranks on, reasoning decides thinking mode. There are no
	// client-tunable quality or speed levers. Best-effort — if the classifier is
	// unavailable (no embeddings worker) or the request carries no messages (a
	// bare /v1/completions call), fall through to the degraded ranker below.
	// FIRST: the outcome matrix, which answers the routing question directly —
	// which of these workers got questions like this one right, and which of
	// those is fastest. No quality score, no difficulty target, nothing compared
	// against anything.
	//
	// IT NEVER DECLINES, and this comment used to say it did — "supersedes the
	// tier path wherever it has evidence, and declines where it does not, so the
	// two can run side by side". That sentence was wrong and expensive. It is the
	// reason the tier branch below LOOKS reachable, and three separate audits had
	// to re-derive from the code that it is not.
	//
	// chooseByOutcome RANKS; it does not filter. With no evidence it falls through
	// to its own bank-rate fallback and still returns every candidate in some
	// order. Its only empty return needs len(cands) == 0, which is already
	// excluded above. r.outcomes is assigned unconditionally at startup, and
	// classify() cannot report ok without having written a vector. So whenever
	// cl != nil this branch returns, and NOTHING BELOW IT RUNS.
	//
	// The tier ranker that used to sit below has been deleted on the strength of
	// exactly that argument. What remains below is the DEGRADED ranker, which is a
	// different thing: it runs when there is no classification to rank on at all.
	if cl != nil && r.outcomes != nil && len(cl.vec) > 0 {
		// Self-healing: cheap when already filled (one atomic load), and the only
		// thing that fills the bank after a warm restart.
		r.ensureBankVectorsAsync()
		if ordered, reason, able := r.outcomes.chooseByOutcome(filtered, cl.vec, !tr.noThink, job); len(ordered) > 0 {
			return &routePlan{
				candidates: ordered,
				// A client-named model is never "auto" however it was ranked: the
				// caller made the choice, so escalating or judging it would be
				// second-guessing an instruction.
				auto:     wantModel == "",
				route:    fmt.Sprintf("%s:%s", routeKind(wantModel), reason),
				cl:       cl,
				job:      job,
				tr:       tr,
				session:  sess,
				group:    gr,
				expert:   expert,
				rejected: rejected,
				able:     able,
			}, nil
		}
	}

	// Reached when the matrix could not run at all: no classifier, or a
	// classification that produced no vector — which in practice means the
	// embeddings worker is down. Rank by measured quality with speed and current
	// load breaking ties (backendScore), and hard-filter nothing, so the request
	// is always served by somebody.
	//
	// This is the ONLY fallback now. The difficulty-tier ranker that used to sit
	// between here and the matrix is gone: it scored a prompt to a quality target
	// and ranked the fleet against it, and nothing reached it — the matrix branch
	// above returns for every classified request, and an unclassified one has no
	// difficulty score to tier by.
	return &routePlan{
		candidates: rankBackends(filtered, job, tr.noThink),
		// The ROUTER still chose this worker — the classifier being unavailable
		// does not make it the caller's pick. Omitting this switched off
		// escalation, both failover paths and the judge in exactly the degraded
		// mode where they matter most, which is the same four-feature silent
		// disable dace4c9 was written to prevent.
		auto:     wantModel == "",
		route:    routeKind(wantModel),
		cl:       cl,
		job:      job,
		tr:       tr,
		session:  sess,
		group:    gr,
		expert:   expert,
		rejected: rejected,
	}, nil
}

// intersectFeatures keeps only the features present in both lists, preserving
// the order of the first.
func intersectFeatures(have []string, also []string) []string {
	out := []string{}
	for _, f := range have {
		for _, g := range also {
			if f == g {
				out = append(out, f)
				break
			}
		}
	}
	return out
}

// budgetCeiling is the largest completion budget the chosen worker can still fit
// after this request's prompt, or 0 when the worker never declared a context (⇒
// don't clamp). The margin absorbs the difference between our chars/3 estimate
// and the worker's real tokenizer: overshooting the clamp downward costs a few
// tokens of answer, overshooting upward is a 400 on a strict engine.
func budgetCeiling(b *Backend, job jobCost) int {
	if b == nil || b.ContextK <= 0 {
		return 0
	}
	room := b.ContextK*1024 - job.promptTokens - contextClampMargin
	if room < contextClampFloor {
		return contextClampFloor // let the worker itself reject a prompt this close to full
	}
	return room
}

const (
	contextClampMargin = 1024
	contextClampFloor  = 512
)

// autoModelNames are the model strings that mean "you choose" rather than naming
// a model. "default" is what every clabtree guest sends, so it must never be
// mistaken for a model id and 404'd.
var autoModelNames = map[string]bool{"": true, "default": true, "auto": true, "router": true}

// requestedModel returns the model the client explicitly asked for, or "" when
// it left the choice to the router.
func requestedModel(req *ChatRequest) string {
	name := strings.TrimSpace(req.Model)
	if autoModelNames[strings.ToLower(name)] {
		return ""
	}
	return name
}

// backendServesModel reports whether a backend answers to a client-supplied
// model name. All three spellings the fleet publishes are accepted: the model
// itself (what several workers share), the worker id (what /v1/models lists as
// "owned_by"), and the short alias (what /v1/models lists as "id" when it is
// unambiguous) — so a harness can name the family it wants, the exact worker,
// or the human spelling from the menu. The alias compares case-insensitively;
// the raw ids stay exact.
func backendServesModel(b *Backend, name string) bool {
	if b.Model == name || b.ID == name {
		return true
	}
	if a := backendAlias(b); a != "" && a == strings.ToLower(name) {
		return true
	}
	return false
}

// routeKind is the X-LLM-Route prefix: "model" when the client named one,
// "route" when the router chose.
func routeKind(wantModel string) string {
	if wantModel != "" {
		return "model"
	}
	return "route"
}

// unknownModelError is a named model no registered worker serves. It is a 404 —
// the OpenAI-standard answer, and the one a harness can act on — not the 503 an
// exhausted filter produces.
type unknownModelError struct{ name string }

func (e unknownModelError) Error() string {
	return fmt.Sprintf("model %q not found — GET /v1/models lists what this router serves", e.name)
}

// profileRetryDelay is how long after the FIRST aborted background profile
// (worker blipped mid-benchmark) the router retries it. Short, because until the
// retry succeeds the worker runs on a provisional quality that distorts the
// fleet's tier range. A worker that's actually down fails the health check in
// the retry and falls into the normal recert backoff instead.
const profileRetryDelay = 2 * time.Minute

// profileRetryMaxAttempts is how many aborted profiles in a row the router pays
// for before it stops trying.
//
// There has to be a limit, and it has to be small. An abort is not free: the
// benchmark set is ~400 questions, runQualityBenchmark only gives up after
// wg.Wait(), and a metered endpoint that sustains 429s under the benchmark's
// concurrency fails the same way every time — so an unbounded retry is an
// unbounded bill, spent on measurements that are discarded the moment they
// arrive. Past this the worker keeps its provisional profile and stays routable;
// re-registering it, or POST /debug/backends/{id}/certify, starts the count
// again, which is the operator saying they have fixed something.
const profileRetryMaxAttempts = 4

// profileRetryBackoff grows the gap between aborted background profiles: 2m, 4m,
// 8m, … capped at an hour. Same shape as recertBackoff, and for the same reason
// — a failure that repeats should cost less each time, not the same.
func profileRetryBackoff(aborts int) time.Duration {
	const ceiling = time.Hour
	if aborts < 1 {
		aborts = 1
	}
	d := profileRetryDelay
	for i := 1; i < aborts && d < ceiling; i++ {
		d *= 2
	}
	if d > ceiling {
		d = ceiling
	}
	return d
}

// scheduleProfileRetry reacts to a background profile that was discarded:
// records the attempt and what it spent where an operator will see it, and
// schedules the next attempt on a growing backoff. Returns the delay it
// scheduled, or 0 when it has given up on this registration.
func (r *Router) scheduleProfileRetry(id string, cause error) time.Duration {
	aborts := r.registry.noteProfileAbort(id)
	if aborts == 0 {
		// The row went away mid-profile (deleted, or replaced by a registration
		// whose own certification is already running). Nothing to retry, and
		// nothing left to hang the spend on but the log.
		log.Printf("background profile %s aborted after its registration went away: %v", id, cause)
		return 0
	}
	if aborts >= profileRetryMaxAttempts {
		log.Printf("background profile %s aborted %d times in a row: %v — GIVING UP. The worker keeps its provisional quality; re-register it or POST /debug/backends/%s/certify once the endpoint is healthy",
			id, aborts, cause, id)
		r.registry.setCheck(id, "profile", Check{Message: fmt.Sprintf(
			"abandoned after %d aborted attempts: %v — serving on a provisional quality until re-certified", aborts, cause)})
		return 0
	}
	delay := profileRetryBackoff(aborts)
	log.Printf("background profile %s aborted (attempt %d/%d): %v (keeping provisional; retrying in %s)",
		id, aborts, profileRetryMaxAttempts, cause, delay)
	r.registry.setCheck(id, "profile", Check{Message: fmt.Sprintf(
		"provisional — attempt %d/%d aborted: %v; retrying in %s", aborts, profileRetryMaxAttempts, cause, delay)})
	time.AfterFunc(delay, func() { r.certifyBackend(id) })
	return delay
}

// recertifyIfRegenerated re-certifies when a different registration generation
// landed while a certification/profile held the guard: that registration's own
// certifyBackend bailed on the guard, and the in-flight run may even have
// stamped its stale-generation verdict over the new state — so status alone
// can't detect it (a stale "ready" overwrites the new "probing"). The probing
// check is a belt for paths that never reached finishCertification. Called
// only after the guard is freed; the spawned certification re-captures the
// current generation, so this converges instead of looping.
func (r *Router) recertifyIfRegenerated(id string, gen int64) {
	if b := r.registry.get(id); b != nil && (b.profileGen != gen || b.Certification.Status == "probing") {
		go r.certifyBackend(id)
	}
}

func (r *Router) certifyBackend(id string) {
	// Single atomic guard for the whole certification + cold-profile span. The
	// old Load-then-Store pair left a multi-second window (two worker HTTP
	// round-trips wide) where two registrations triggered duplicate concurrent
	// capacity ramps + benchmarks against the same GPU.
	progress := newProfileProgress()
	if _, busy := r.profiling.LoadOrStore(id, progress); busy {
		return // a certification/profile is already in flight for this worker
	}
	var gen int64
	owned := true
	defer func() {
		if owned {
			r.profiling.Delete(id)
			r.recertifyIfRegenerated(id, gen)
		}
	}()
	backend := r.registry.get(id)
	if backend == nil {
		return
	}
	gen = backend.profileGen
	r.registry.startCertification(id)

	checks := map[string]Check{}
	if err := r.checkBackend(id); err != nil {
		checks["health"] = Check{OK: false, Message: err.Error()}
		r.registry.finishCertification(id, false, checks, 0, 0, err.Error())
		return
	}
	checks["health"] = Check{OK: true}

	// A relay row's URL is another router, not an endpoint. Nothing below applies
	// to it: there are no weights to fingerprint, no capabilities to probe (the
	// upstream probed them), and no benchmark to run (the upstream ran it, and
	// running it again would spend the upstream's GPUs to learn its own answer).
	// The relay refresh is this row's certification — see applyRelayEntry — so
	// there is nothing to do here but leave the imported values alone.
	//
	// This is reachable at all because healthLoop re-certifies a row that has gone
	// unhealthy, which for a relay means the WAN link or the upstream was down. It
	// comes back when the link does, and the next refresh re-certifies it.
	if isRelayRow(backend) {
		checks["relay"] = Check{OK: true, Message: fmt.Sprintf("derived from relay %q; profile imported, not measured", backend.Relay)}
		r.registry.finishCertification(id, true, checks, backend.BaselineTPS, backend.Certification.TTFTMillis, "")
		return
	}

	// Embeddings-only workers don't serve chat, so the chat-oriented speed/json/
	// tool probes don't apply. Certify them with an embeddings probe instead.
	if isEmbeddingsOnly(backend) {
		latency, err := r.embeddingsProbe(backend)
		if err != nil {
			checks["embeddings"] = Check{OK: false, Message: err.Error()}
			r.registry.finishCertification(id, false, checks, 0, 0, err.Error())
			return
		}
		// Measure-don't-trust applies to the classifier's own dependency too: the
		// per-classification deadline comes from this worker's real latency, not a
		// fixed two seconds that a slow box silently spends its whole budget on.
		deadline := r.noteEmbedLatency(latency)
		checks["embeddings"] = Check{OK: true, Message: fmt.Sprintf("%s round trip; classifier deadline %s",
			latency.Round(time.Millisecond), deadline)}
		r.registry.finishCertification(id, true, checks, 0, 0, "")
		return
	}

	// Fingerprint what the worker is really serving before either path below
	// needs it. The id keys the profile cache; the weights metadata is published
	// straight onto the registry rather than carried in the profile, so a warm
	// restart that short-circuits to a cached profile still reports current
	// weights instead of whatever was measured months ago.
	model, meta := r.queryModelInfo(backend)
	r.registry.setModelMeta(id, meta)
	// Re-read the backend now the served id is known: `backend` is a clone taken
	// before the fingerprint, and every probe below names probeModel(backend) in
	// its request. Without this they would all still be sending whatever the
	// registration declared, which is the spelling an endpoint that validates
	// model names is least likely to accept.
	if fresh := r.registry.get(id); fresh != nil {
		backend = fresh
	}

	// Measure-don't-trust: profile a chat worker (capabilities, context, speed,
	// capacity, quality) at cold start and cache it per (id, model); a warm
	// restart reuses the cached profile. The worker itself declares ~nothing.
	if r.cfg.ProfileWorkers {
		if prof, ok := r.logs.LoadWorkerProfile(context.Background(), id, model); ok && prof.BenchVersion == benchmarkVersion {
			r.backfillCachedProfile(id, backend, prof)
			r.registry.applyProfileIfGen(id, gen, prof)
			for k, v := range prof.Checks { // surface cached per-probe results (incl. quality breakdown) in /backends
				checks[k] = v
			}
			checks["profile"] = Check{OK: true, Message: fmt.Sprintf("cached: q=%d%%, %.0f tok/s, ctx %dk, conc %d (profiled in %s)", prof.Quality, prof.BaselineTPS, prof.ContextK, prof.MaxConcurrency, fmtProfileDuration(prof.ProfileMillis))}
			r.registry.finishCertification(id, true, checks, prof.BaselineTPS, prof.TTFTMillis, "")
			log.Printf("worker %s certified from cached profile (model=%s)", id, model)
			// A profile cached before the two-score benchmark lacks the no-think
			// quality. Backfill it in the BACKGROUND from the stored per-question
			// results — only the hard tiers re-run, thinking-off — so the worker
			// keeps serving on its certified values throughout. This is the
			// deliberate alternative to bumping benchmarkVersion, which would
			// re-profile every worker in the fleet and park each at a provisional
			// quality to learn one new number. (The permacache has since made a bump
			// much cheaper — a thinking-on question this model has already answered
			// is served from cache rather than re-asked, see benchOne — but it does
			// not make one free here: the no-think verdicts are exactly the ones no
			// model has ever been asked for, so they are the half that would still
			// have to be generated.) Too long in the guard for the certification
			// path (minutes, not the seconds the other backfills cost), hence the
			// ownership handoff, same as cold start.
			if needsNoThinkBackfill(prof) {
				owned = false // guard ownership moves to the background backfill
				go func() {
					defer r.recertifyIfRegenerated(id, gen)
					defer r.profiling.Delete(id)
					conc := prof.MaxConcurrency
					if conc > 4 {
						conc = 4 // same live-traffic headroom rule as the cold-start benchmark
					}
					if conc < 1 {
						conc = 1
					}
					score, ok, breakdown, ntResults := r.runNoThinkQualityBenchmark(backend, conc, prof.BenchResults)
					if !ok {
						// Worker likely blipped mid-run. Keep the single-score profile —
						// the worker ranks as unmeasured (below every measured worker) on
						// no-think requests, see qualityFor — and the next certification
						// (router restart, changed registration, recovery) retries.
						log.Printf("no-think quality backfill for %s failed — keeping the single-score profile until the next certification", id)
						return
					}
					prof.QualityNoThink = score
					prof.QualityNoThinkDetail = breakdown
					prof.BenchResultsNoThink = ntResults
					prof.CategorySummary = benchCategorySummary(prof.BenchResults, ntResults)
					if prof.Checks == nil {
						prof.Checks = map[string]Check{}
					}
					prof.Checks["quality_nothink"] = Check{OK: true, Message: fmt.Sprintf("%d%% %s (backfilled)", score, breakdown)}
					if !r.registry.applyProfileIfGen(id, gen, prof) {
						log.Printf("no-think backfill %s finished for a stale registration generation — discarded", id)
						return
					}
					if err := r.logs.SaveWorkerProfile(context.Background(), id, prof); err != nil {
						log.Printf("persist no-think backfill for %s failed: %v", id, err)
					}
					log.Printf("worker %s: backfilled no-think quality q=%d beside thinking-mode q=%d", id, score, prof.Quality)
				}()
			}
			return
		}
		// Cold start. Quick profile (capabilities + speed + context) makes the
		// worker routable in seconds; the slow quality + capacity measurement runs
		// in the background so a fresh deploy doesn't black out the fleet.
		quick, err := r.profileQuick(backend, model)
		if err != nil {
			checks["profile"] = Check{OK: false, Message: err.Error()}
			r.registry.finishCertification(id, false, checks, 0, 0, err.Error())
			return
		}
		r.registry.applyProfileIfGen(id, gen, quick)
		for k, v := range quick.Checks {
			checks[k] = v
		}
		checks["profile"] = Check{OK: true, Message: "provisional — measuring quality+capacity in background"}
		r.registry.finishCertification(id, true, checks, quick.BaselineTPS, quick.TTFTMillis, "")
		log.Printf("worker %s provisionally ready (q~%d, %.0f tok/s, ctx %dk); profiling quality+capacity in background (model=%s)",
			id, quick.Quality, quick.BaselineTPS, quick.ContextK, model)
		owned = false // guard ownership moves to the background goroutine
		go func() {
			// LIFO: the guard is released first, THEN any registration that
			// landed mid-profile (it bailed on the guard) gets its
			// certification re-kicked.
			defer r.recertifyIfRegenerated(id, gen)
			defer r.profiling.Delete(id)
			full, err := r.profileBackend(backend, model)
			if err != nil {
				// Worker likely blipped mid-profile. Keep the provisional profile
				// (the worker stays routable) and retry on a timer — nothing else
				// re-triggers a background profile for a ready backend — but on a
				// counter and a backoff, because a worker that fails this way every
				// time fails it expensively (see profileRetryMaxAttempts).
				r.scheduleProfileRetry(id, err)
				return
			}
			r.registry.clearProfileAborts(id)
			if !r.registry.applyProfileIfGen(id, gen, full) {
				// The worker re-registered with new content (or was deleted) while
				// we measured — these numbers describe the old generation. The new
				// registration runs its own certification; persisting here would
				// poison the profile cache (or resurrect a deleted row).
				log.Printf("background profile %s finished for a stale registration generation — discarded", id)
				return
			}
			if full.Incomplete {
				// One or more capability probes never got a verdict (transient
				// errors). Serve on these values but do NOT cache them — a persisted
				// "not detected" from a blip would misroute traffic on every warm
				// restart until the next benchmarkVersion bump.
				log.Printf("worker %s profile has inconclusive capability probes — not persisting; will re-probe on next certification", id)
			} else if err := r.logs.SaveWorkerProfile(context.Background(), id, full); err != nil {
				log.Printf("persist worker profile %s failed: %v", id, err)
			}
			log.Printf("worker %s profiled in %s: q=%d, %.0f tok/s (ttft %dms), ctx %dk, conc %d, features=%v, thinking=%v",
				id, fmtProfileDuration(full.ProfileMillis), full.Quality, full.BaselineTPS, full.TTFTMillis, full.ContextK, full.MaxConcurrency, full.Features, full.Thinking)
		}()
		return
	}

	// Legacy path (ROUTER_PROFILE_WORKERS=false): trust declared values, probe
	// only declared features.
	tps, ttft, err := r.speedProbe(backend)
	if err != nil {
		checks["speed"] = Check{OK: false, Message: err.Error()}
		r.registry.finishCertification(id, false, checks, 0, 0, err.Error())
		return
	}
	checks["speed"] = Check{OK: true, Message: fmt.Sprintf("%.1f tok/s", tps)}

	if hasFeature(backend, "json") {
		if err := r.jsonProbe(backend); err != nil {
			checks["json"] = Check{OK: false, Message: err.Error()}
			r.registry.finishCertification(id, false, checks, tps, ttft, err.Error())
			return
		}
		checks["json"] = Check{OK: true}
	}

	if hasFeature(backend, "tools") {
		if err := r.toolProbe(backend); err != nil {
			checks["tools"] = Check{OK: false, Message: err.Error()}
			r.registry.finishCertification(id, false, checks, tps, ttft, err.Error())
			return
		}
		checks["tools"] = Check{OK: true}
	}

	r.registry.finishCertification(id, true, checks, tps, ttft, "")
}

// speedProbe measures decode throughput with a short prose generation.
//
// Prose-only is deliberate, not an oversight. On a worker running speculative
// decoding the rate is workload-dependent — measured on llm-6000pro-deepseek-284B-q8
// 2026-08-09, 17.4 tok/s on prose against 27.0 on code, because draft acceptance and
// token density both differ — so there is no single true number to find. A fixed
// prompt keeps the figure COMPARABLE across workers and engines, which is all
// ranking needs, and prose is the slow end, so the estimate errs pessimistic: the
// safe direction for a latency estimate (same call the prefill probe makes).
func (r *Router) speedProbe(backend *Backend) (float64, int64, error) {
	payload := map[string]any{
		"model":      probeModel(backend),
		"stream":     true,
		"max_tokens": 64,
		"messages": []map[string]string{
			{"role": "system", "content": "You are a concise benchmark assistant. /no_think"},
			// The trailing tag defeats the worker's prompt cache the same way the
			// prefill filler's salt does. It matters less here — the prompt is ~15
			// tokens, so a cache hit saves almost no prefill — but a cached probe
			// still returns an optimistic TTFT, and TTFT seeds the live estimate.
			// Kept out of the instruction so it cannot change what is generated:
			// decode rate is measured from the tokens this reply produces.
			{"role": "user", "content": fmt.Sprintf(
				"Write exactly four short sentences about reliable local LLM routing. [run %08x]",
				nextProbeSalt())},
		},
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
		// Exact token count in the final chunk — delta-counting under-reads MTP
		// workers ~2.5x (see readSSEStream). Both fleet dialects support this.
		"stream_options": map[string]bool{"include_usage": true},
	}
	send := func(p map[string]any) (*http.Response, error) {
		body, _ := json.Marshal(p)
		req, err := http.NewRequest(http.MethodPost, upstreamChatURL(backend), bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		if backend.APIKey != "" {
			req.Header.Set("Authorization", "Bearer "+backend.APIKey)
		}
		return r.client.Do(req)
	}
	start := time.Now()
	resp, err := send(payload)
	if err != nil {
		return 0, 0, err
	}
	if resp.StatusCode >= 400 && resp.StatusCode < 500 {
		// A dialect that rejects stream_options must still get a baseline: retry
		// once without it and let the delta-count fallback carry the measurement.
		resp.Body.Close()
		delete(payload, "stream_options")
		start = time.Now()
		if resp, err = send(payload); err != nil {
			return 0, 0, err
		}
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return 0, 0, fmt.Errorf("speed probe returned %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	first := int64(0)
	content, reasoning, tokenCount, err := readSSEStream(resp.Body, func() {
		if first == 0 {
			first = time.Since(start).Milliseconds()
		}
	})
	if err != nil {
		return 0, 0, err
	}
	// Reasoning tokens are generated output too: a model that ignores
	// enable_thinking:false and spends the whole 64-token budget reasoning must
	// still measure as producing tokens, not fail as "empty" (which would leave
	// the worker uncertifiable — same failure class as the thinking-probe bug).
	out := content + reasoning
	if strings.TrimSpace(out) == "" {
		return 0, first, errors.New("empty speed probe response")
	}
	// Count the tokens the worker actually emitted rather than estimating them from
	// character length. len(out)/4 was wrong in BOTH directions and by a lot,
	// because chars-per-token is a property of the text, not a constant: measured
	// on the 284B worker 2026-08-09, prose ran 5.05 chars/token (so the estimate
	// inflated 17.4 tok/s into ~21.9) and code ran 3.79 (deflating 27.0 into
	// ~25.5). Since BaselineTPS feeds expectedLatency, that error went straight
	// into which worker gets picked — and it disagreed with the live EWMA, which
	// has always counted deltas.
	tokens := math.Max(1, float64(tokenCount))
	if tokenCount == 0 {
		tokens = math.Max(1, float64(len(out))/4.0) // non-delta dialect: fall back
	}
	// The one streamed probe, so it meters by hand. Completion tokens only: an SSE
	// stream carries no prompt count, and the probe's prompt is ~30 tokens against
	// a profile's several hundred thousand.
	r.meterProfileTokens(backend.ID, 0, int(tokens))
	// Decode throughput = tokens / (total − TTFT), so the certified speed reflects
	// generation rate rather than end-to-end latency — consistent with the live
	// EWMA fed by observe().
	decodeWindow := time.Since(start) - time.Duration(first)*time.Millisecond
	if decodeWindow <= 0 {
		decodeWindow = time.Since(start)
	}
	return tokens / decodeWindow.Seconds(), first, nil
}

// prefillProbeTokens is the prompt size used to measure prefill rate. Large enough
// that fixed per-request overhead is noise (minPrefillTokens is 256 for the same
// reason), small enough that the probe costs ~40s on the slowest CPU worker in the
// fleet rather than the ~5 minutes an 8k prompt would.
const prefillProbeTokens = 1024

// speedProbeVersion is bumped when the DECODE measurement itself changes, so
// cached profiles re-measure their BaselineTPS on load without paying for a full
// BenchVersion re-profile.
//
//	1 — original: token count estimated as len(text)/4
//	2 — count the deltas the worker actually emitted (see readSSEStream)
//	3 — prefer usage.completion_tokens via stream_options.include_usage:
//	    MTP workers pack ~2.5 tokens into each delta, so v2 certified the
//	    fleet's spec-decode workers at ~40% of their real decode rate
const speedProbeVersion = 3

// backfillCachedProfile fills in measurements that a cached profile predates.
//
// A profile is only re-measured when benchmarkVersion changes, so a probe added
// WITHOUT a version bump never runs on an already-profiled fleet. That is exactly what
// happened to the prefill probe: it shipped, every worker stayed certified from its
// cached profile, and 5 of 7 never acquired a prefill rate — including both 284B
// workers, the ones whose TTFT is most sensitive to prompt length and the reason the
// probe was written. Bumping benchmarkVersion is still the wrong instrument: it takes
// EVERY worker in the fleet back through a cold profile and parks each at a
// provisional quality, to recover a one-second probe on the few that lack it.
//
// It is no longer the cliff it was, and the reason is worth knowing before reading
// the rest of this: since the permacache, a re-profile does not re-earn the graded
// answers. Every question this model has already answered, graded this way, is
// served from cache (see benchOne and identity.go), so a bump costs the capability,
// speed, capacity and context probes rather than hours of grading. What it still
// costs is the fleet-wide disruption, which is why the clauses below exist.
//
// ANY future addition to WorkerProfile needs a clause here, or it inherits the same
// silent failure — shipped, cached over, never measured.
//
// There is a SECOND class of staleness with the same cure: a field that can change on
// the WORKER between restarts without changing the (id, model) cache key. Context is
// the known one — CTX_SIZE is a deployment choice, not a property of the model — and
// it is handled below. Anything else that a service.env edit can move belongs here
// too, provided it is cheap enough to re-measure unconditionally.
func (r *Router) backfillCachedProfile(id string, backend *Backend, prof *WorkerProfile) {
	if prof.SpeedVersion < speedProbeVersion {
		if tps, ttft, err := r.speedProbe(backend); err != nil {
			log.Printf("speed re-measure failed for %s: %v — keeping the cached %.1f tok/s", id, err, prof.BaselineTPS)
		} else {
			log.Printf("worker %s: decode re-measured %.1f -> %.1f tok/s (speed probe v%d)", id, prof.BaselineTPS, tps, speedProbeVersion)
			prof.BaselineTPS, prof.TTFTMillis, prof.SpeedVersion = tps, ttft, speedProbeVersion
			if prof.Checks == nil {
				prof.Checks = map[string]Check{}
			}
			prof.Checks["speed"] = Check{OK: true, Message: fmt.Sprintf("%.1f tok/s (re-measured)", tps)}
			if r.logs != nil {
				if err := r.logs.SaveWorkerProfile(context.Background(), id, prof); err != nil {
					log.Printf("persist re-measured decode rate for %s failed: %v", id, err)
				}
			}
			// No registry write here: the caller applies the profile and certifies
			// from it immediately after, same as the prefill backfill below.
		}
	}
	// Context, re-read unconditionally. It is the one profile field a deployment can
	// change without changing the cache key, so trusting the cache means advertising a
	// window the worker no longer has: llm-6000pro-deepseek-284B-q8 kept reporting 256k
	// for days after being raised to 384k, and the only cure was DELETE /backends/{id},
	// which throws away the quality benchmark too and parks the worker at a provisional
	// quality of 3 for ~14 min while it re-runs. Nobody pays that to correct one integer,
	// so in practice the fleet just stayed wrong.
	//
	// Re-measuring is free — one GET on vLLM, two on llama.cpp (/v1/models yields
	// nothing, then /props), the same cost class as the health check that ran a few
	// lines up — so there is nothing to trade off and no version gate to add; the
	// prefill probe below is gated because it is expensive, and this is not.
	//
	// This clause alone is NOT enough to make a context change land on restart, and
	// an earlier version of this comment claimed it was. It runs on the certification
	// path, and a restarted worker re-registers a byte-identical payload, which
	// handleRegisterBackend treats as a keepalive and does not certify. reconcileContext
	// covers that case from the health loop; this one still matters for every path that
	// DOES certify (a changed registration, a recovering backend, a model swap).
	if ctxK, ok := r.queryContextMeasured(backend); ok && ctxK != prof.ContextK {
		log.Printf("worker %s: context re-read %dk -> %dk on its cached profile", id, prof.ContextK, ctxK)
		prof.ContextK = ctxK
		if prof.Checks == nil {
			prof.Checks = map[string]Check{}
		}
		prof.Checks["context"] = Check{OK: true, Message: fmt.Sprintf("%dk (re-read)", ctxK)}
		if r.logs != nil {
			if err := r.logs.SaveWorkerProfile(context.Background(), id, prof); err != nil {
				log.Printf("persist re-read context for %s failed: %v", id, err)
			}
		}
	}

	// The thinking DIALECT postdates every cached profile, so without this clause
	// the whole fleet would carry the zero value forever — the exact failure this
	// function exists to prevent. Gated on absence rather than re-measured every
	// time, like prefill and unlike context: it costs a 1024-token generation, and
	// which gate an endpoint reads is a property of the endpoint's API rather than
	// of a deployment knob someone edits.
	if prof.ThinkingDialect == "" {
		if thinking, dialect, inconclusive := r.thinkingProbe(backend); inconclusive {
			log.Printf("thinking dialect backfill for %s was inconclusive — will re-probe next certification", id)
		} else {
			log.Printf("worker %s: thinking dialect measured as %q (thinking %v -> %v) on its cached profile",
				id, dialect, prof.Thinking, thinking)
			prof.Thinking, prof.ThinkingDialect = thinking, dialect
			if prof.Checks == nil {
				prof.Checks = map[string]Check{}
			}
			prof.Checks["thinking"] = Check{OK: thinking, Message: mapBool(thinking, "supported via "+dialect+" (re-probed)", "not detected (re-probed)")}
			if r.logs != nil {
				if err := r.logs.SaveWorkerProfile(context.Background(), id, prof); err != nil {
					log.Printf("persist thinking dialect for %s failed: %v", id, err)
				}
			}
		}
	}

	// ProfilePromptTokens/ProfileOutputTokens/ProfileCost get NO clause, and that
	// is the answer rather than an omission: they describe a run that has already
	// happened, and the only way to re-derive them is to pay for another one. A
	// profile cached before they existed keeps its zero, which reads as "not
	// measured" and never as "free" — see the field comment for why the token
	// count is what carries that distinction.
	//
	// CapacityCurve gets no clause either, for the same reason and with a stated
	// consequence. Re-deriving it means re-running the concurrency ramp, which
	// fires up to CapacityProbeMax simultaneous generations at a worker that is
	// serving live traffic — the most disruptive probe there is, and nothing like
	// the one-GET cost that makes the context re-read unconditional.
	//
	// The staleness is bounded rather than silent: an absent curve prices as
	// alpha = 1 (see concurrencyAlphaFor), the neutral default, so a worker cached
	// without one keeps exactly the load model it had before the curve existed. It
	// is never priced WRONGLY, only un-refined, and each worker acquires a curve at
	// its next cold profile. The benchmarkVersion bump that ships this re-profiles
	// the fleet anyway, so in practice no worker waits.
	if prof.PrefillTPS != 0 {
		return
	}
	rate, err := r.prefillProbe(backend)
	if err != nil {
		log.Printf("prefill backfill failed for %s: %v — routing will price its TTFT from the flat average", id, err)
		return
	}
	prof.PrefillTPS = rate
	if prof.Checks == nil {
		prof.Checks = map[string]Check{}
	}
	prof.Checks["prefill"] = Check{OK: true, Message: fmt.Sprintf("%.0f tok/s on a %d-token prompt (backfilled)", rate, prefillProbeTokens)}
	if r.logs != nil {
		if err := r.logs.SaveWorkerProfile(context.Background(), id, prof); err != nil {
			log.Printf("persist backfilled prefill rate for %s failed: %v", id, err)
		}
	}
	log.Printf("worker %s: backfilled prefill rate %.0f tok/s into its cached profile", id, rate)
}

// prefillProbeSamples is how many times prefillProbe measures before believing a
// number. One sample is not enough, and the failure is not hypothetical: on
// 2026-08-08 llm-a750-Granite4.1-8B was probed while a CPU worker on the SAME host
// (llm-cpu-gemma-26B-silver) was mid-benchmark, and recorded 22 tok/s. Re-measured on
// an idle host at the identical 1024-token prompt size: 643 tok/s — a 29x
// underestimate, cached permanently, on the fleet's only 8B GPU worker.
//
// Best-of-N rather than mean or median, because the error is ONE-SIDED: contention,
// scheduling and a cold cache can only ever make a prefill slower than the hardware is
// capable of. Nothing makes it spuriously fast. The fastest sample is therefore the
// least-corrupted estimate of what this worker does when it is the one being asked.
const prefillProbeSamples = 3

// probeSalt hands out a fresh salt for every probe run, so no two runs ever send
// the same filler to the same worker.
//
// The salt used to be the sample index, 0/1/2. That varies the text WITHIN a run,
// which is what best-of-N needs, and leaves it byte-identical BETWEEN runs — so
// sample 0 of a re-certification is the same 1024 tokens sample 0 of the first
// profile left sitting in the worker's cache. Measured on a live worker: 2725
// tok/s against a real 214, a 13x overestimate, then cached as that worker's
// prefill rate. Prefill is the primary term in expectedLatency, so that number
// decides who gets every long prompt, and deadlineFilter believes it too.
//
// This is the defence that does not depend on the endpoint being honest. The
// cached_tokens check in prefillProbeOnce is better when it fires, because it
// catches a cache hit the salt failed to prevent, but it can only fire on an
// endpoint that reports prompt_tokens_details.cached_tokens. Plenty do not, and
// on those a silent cache hit is indistinguishable from fast hardware.
//
// Seeded randomly rather than from a counter, because a restart would otherwise
// replay the same sequence into a cache that outlives the process.
var probeSalt atomic.Uint32

func init() {
	var b [4]byte
	if _, err := rand.Read(b[:]); err == nil {
		probeSalt.Store(uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3]))
	}
}

// nextProbeSalt reserves prefillProbeSamples consecutive salts, so the samples of
// one run cannot collide with the samples of the next.
func nextProbeSalt() uint32 { return probeSalt.Add(prefillProbeSamples) }

// prefillProbe measures how fast a worker turns prompt tokens into a first token, by
// sending a prompt of KNOWN length with a 1-token budget and timing the call. It takes
// the best of prefillProbeSamples runs, and returns the last error only if EVERY
// sample failed.
//
// The live EWMA cannot fill this gap on its own. observe() only records a prefill
// sample from NON-thinking requests — TTFT is not comparable across engines otherwise,
// because vLLM buffers reasoning into TTFT while llama.cpp streams it — so a worker the
// router mostly sends thinking traffic to never accumulates one, and prefillSeconds()
// falls back to a flat TTFT average that ignores prompt length entirely. Measured on
// llm-naples-deepseek-284B-q4: the router priced its time-to-first-token at 978ms while
// a 4116-token prompt really took 178s, a 140x underestimate, so it kept sending long
// prompts to the one worker in the fleet that could not serve them.
//
// Thinking is disabled here precisely so the number IS comparable across engines. The
// single generated token is included in the elapsed time, which slightly understates
// the rate (~6% on a fast GPU worker, well under 1% on a CPU one) — erring toward
// pessimism, the safe direction for a latency estimate.
func (r *Router) prefillProbe(backend *Backend) (float64, error) {
	best, lastErr := 0.0, error(nil)
	base := nextProbeSalt()
	for i := 0; i < prefillProbeSamples; i++ {
		// Distinct salt per sample AND per run: identical text hits the worker's
		// prompt cache and hands best-of-N a fabricated winner. See prefillFiller.
		rate, err := r.prefillProbeOnce(backend, base+uint32(i))
		if err != nil {
			lastErr = err
			continue
		}
		if rate > best {
			best = rate
		}
	}
	if best <= 0 {
		if lastErr == nil {
			lastErr = errors.New("no usable prefill sample")
		}
		return 0, lastErr
	}
	return best, nil
}

func (r *Router) prefillProbeOnce(backend *Backend, salt uint32) (float64, error) {
	payload := map[string]any{
		"model":                probeModel(backend),
		"stream":               false,
		"max_tokens":           1,
		"temperature":          0,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
		"messages": []map[string]string{
			{"role": "system", "content": "You are a concise benchmark assistant. /no_think"},
			{"role": "user", "content": prefillFiller(prefillProbeTokens, salt) + "\n\nReply with one word: ok"},
		},
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequest(http.MethodPost, upstreamChatURL(backend), bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if backend.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+backend.APIKey)
	}
	start := time.Now()
	resp, err := r.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return 0, fmt.Errorf("prefill probe returned %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	var raw map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return 0, err
	}
	elapsed := time.Since(start).Seconds()
	r.meterProfileUsage(backend.ID, raw) // dials the endpoint itself, so it meters itself
	// Trust the worker's OWN prompt_tokens rather than a word-count estimate:
	// tokenisers differ enough between model families to bias the rate systematically.
	promptTokens, cachedTokens := 0.0, 0.0
	if u, ok := raw["usage"].(map[string]any); ok {
		promptTokens, _ = u["prompt_tokens"].(float64)
		if d, ok := u["prompt_tokens_details"].(map[string]any); ok {
			cachedTokens, _ = d["cached_tokens"].(float64)
		}
	}
	if promptTokens < minPrefillTokens || elapsed <= 0 {
		return 0, fmt.Errorf("unusable sample: prompt_tokens=%.0f elapsed=%.3fs", promptTokens, elapsed)
	}
	// A sample the worker served from its prompt cache measures the cache, not the
	// hardware, and must be DISCARDED rather than merely diluted — prefillProbe takes
	// the best of its samples, so one cached sample wins outright.
	//
	// Varying the text is necessary but NOT sufficient, which is what the earlier fix
	// assumed. Measured 2026-08-16 against a 284B CPU+GPU worker: two probe prompts
	// built from different salts still came back with cached_tokens 619 of 623 prompt
	// tokens and completed in 0.28s, reporting ~2200 tok/s. The same worker on the same
	// day, given filler the cache could not match, measured 125-143 tok/s. The probe was
	// capable of returning either number for identical hardware, which is a spread wider
	// than any real hardware difference in the fleet.
	//
	// Reject rather than scale: a partially-cached sample's remaining prefill is not
	// necessarily proportional to its uncached tokens, so correcting the rate would be a
	// guess. If every sample is cached the probe reports an error and the caller keeps
	// its previous estimate, which is the safe direction — a missing prefill rate falls
	// back to the flat TTFT average, while a fabricated one actively misroutes.
	if cachedTokens > promptTokens*maxPrefillCachedFraction {
		return 0, fmt.Errorf("cached sample: %.0f of %.0f prompt tokens served from cache", cachedTokens, promptTokens)
	}
	return promptTokens / elapsed, nil
}

// maxPrefillCachedFraction is how much of a prefill probe's prompt may come from the
// worker's cache before the sample is discarded. Not zero: llama-server reports a small
// non-zero cached_tokens for the shared system-message prefix, which is a handful of
// tokens against a 1024-token prompt and does not meaningfully flatter the rate.
const maxPrefillCachedFraction = 0.10

// prefillFiller builds roughly n tokens of non-repeating filler (~1.3 tokens per word
// across common tokenisers). Repetitive text is a bad prefill sample: it compresses an
// MoE's expert-routing distribution and flatters any prefix cache the worker keeps.
//
// `salt` MUST differ between the samples of one probe. llama-server caches prompts by
// default, so sending identical text twice makes the second call skip prefill entirely
// and report a fabricated rate — and because prefillProbe takes the BEST of its
// samples, it would then select that fabrication every time. Measured consequence:
// llm-a750-Granite4.1-8B profiled at pp=11254 tok/s against a real 643, which would
// have made the router believe it was the fastest prefill in the fleet by 14x and send
// it every long prompt. (llama.cpp's own bench.sh carries the same warning about
// cache_prompt; /v1/chat/completions has no equivalent switch, so vary the text.)
//
// A given salt yields identical text, but salts are no longer reused: nextProbeSalt
// hands each run a fresh block, because a salt that repeats across runs is exactly
// the cache hit this function exists to avoid. The fleet stays comparable anyway —
// what has to match between workers is the number of prompt tokens, and the rate is
// computed from the worker's OWN reported prompt_tokens rather than from an assumed
// count, so a filler that tokenises a little differently costs nothing.
//
// A CLOSED VOCABULARY IS NOT ENOUGH, measured 2026-08-16. This function used to draw
// from 24 Greek letter-names, so a new salt reordered the words but every token had
// already been seen; llama-server matched almost the whole prompt anyway and reported
// cached_tokens 619 of 623. Shuffling a small vocabulary defeats an exact-prefix cache
// and not much else. The filler is therefore built from unique hex words, so no two
// samples — and no two probes of the same worker, ever — share a matchable span. The
// cached-token guard in prefillProbeOnce is the backstop for whatever this misses.
func prefillFiller(n int, salt uint32) string {
	var sb strings.Builder
	seed := uint32(2166136261) + salt*2654435761
	// ~2.4 tokens per 8-hex-digit word across common BPE tokenisers, against the ~1.3
	// of a dictionary word: high entropy costs more tokens per character, which is the
	// point. Sized so the prompt still lands near n tokens.
	for i := 0; i < int(float64(n)/2.4); i++ {
		seed = seed*1664525 + 1013904223
		if i > 0 {
			sb.WriteByte(' ')
		}
		// Two rounds per word so consecutive words don't share a linear-congruential
		// stride that a tokeniser could turn into a repeating pattern.
		hi := seed
		seed = seed*1664525 + 1013904223
		fmt.Fprintf(&sb, "%04x%04x", hi>>16, seed>>16)
	}
	return sb.String()
}

func (r *Router) jsonProbe(backend *Backend) error {
	payload := map[string]any{
		"model":           probeModel(backend),
		"stream":          false,
		"max_tokens":      80,
		"response_format": map[string]string{"type": "json_object"},
		"messages": []map[string]string{
			{"role": "system", "content": "Return only valid JSON. /no_think"},
			{"role": "user", "content": `Return {"router_ok":true,"score":7} and no other text.`},
		},
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
	}
	content, err := r.simpleCompletion(backend, payload)
	if err != nil {
		return err
	}
	var got struct {
		RouterOK bool `json:"router_ok"`
		Score    int  `json:"score"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(content)), &got); err != nil {
		return fmt.Errorf("invalid json response: %w; content=%q", err, truncate(content, 120))
	}
	if !got.RouterOK || got.Score != 7 {
		return fmt.Errorf("json response had wrong fields: %q", truncate(content, 120))
	}
	return nil
}

func (r *Router) toolProbe(backend *Backend) error {
	payload := map[string]any{
		"model":      probeModel(backend),
		"stream":     false,
		"max_tokens": 128,
		"tools": []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":        "router_probe_weather",
				"description": "Probe tool; returns weather for a city.",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]string{"type": "string"},
					},
					"required": []string{"city"},
				},
			},
		}},
		"messages": []map[string]string{
			{"role": "system", "content": "Use tools when needed. /no_think"},
			{"role": "user", "content": "Use the weather probe tool for Wellington."},
		},
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
	}
	raw, err := r.rawCompletion(backend, payload)
	if err != nil {
		return err
	}
	choices, _ := raw["choices"].([]any)
	if len(choices) == 0 {
		return errors.New("tool probe returned no choices")
	}
	choice, _ := choices[0].(map[string]any)
	msg, _ := choice["message"].(map[string]any)
	calls, _ := msg["tool_calls"].([]any)
	for _, item := range calls {
		call, _ := item.(map[string]any)
		fn, _ := call["function"].(map[string]any)
		if fn["name"] != "router_probe_weather" {
			continue
		}
		args := fmt.Sprint(fn["arguments"])
		if strings.Contains(strings.ToLower(args), "wellington") {
			return nil
		}
		return fmt.Errorf("tool arguments did not include Wellington: %s", truncate(args, 120))
	}
	return fmt.Errorf("missing expected tool call; response=%s", truncateJSON(raw, 300))
}

// embeddingsProbe verifies an embeddings worker returns a vector for a trivial
// input, and reports how long the round trip took. Used to certify
// embeddings-only backends, which can't serve the chat-oriented speed/json/tool
// probes — and the latency is what the classifier's own deadline is derived
// from (see observeEmbedLatency), because a classifier deadline shorter than the
// worker it depends on is a feature that switches itself off in silence.
func (r *Router) embeddingsProbe(backend *Backend) (time.Duration, error) {
	payload := map[string]any{"model": probeModel(backend), "input": "router embeddings certification probe"}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequest(http.MethodPost, upstreamPathURL(backend, "/v1/embeddings"), bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if backend.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+backend.APIKey)
	}
	start := time.Now()
	resp, err := r.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	elapsed := time.Since(start)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return elapsed, fmt.Errorf("embeddings probe returned %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	var parsed struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return elapsed, fmt.Errorf("invalid embeddings probe response: %w", err)
	}
	if len(parsed.Data) == 0 || len(parsed.Data[0].Embedding) == 0 {
		return elapsed, errors.New("embeddings probe returned no vector")
	}
	return elapsed, nil
}

// noteEmbedLatency folds a measured embeddings round trip into the classifier's
// deadline and reports what it settled on. A router with auto-routing off has no
// classifier and nothing to derive.
func (r *Router) noteEmbedLatency(measured time.Duration) time.Duration {
	if r.classifier == nil {
		return difficultyTimeoutFallback
	}
	was := r.classifier.deadline()
	now := r.classifier.observeEmbedLatency(measured)
	if now != was {
		log.Printf("classifier deadline %s -> %s (embeddings round trip measured at %s)",
			was, now, measured.Round(time.Millisecond))
	}
	return now
}

// isEmbeddingsOnly reports whether a backend serves embeddings but not chat, so
// it must be certified with an embeddings probe (not the chat probes) and kept
// out of chat routing.
func isEmbeddingsOnly(b *Backend) bool {
	return hasFeature(b, "embeddings") && !hasFeature(b, "chat")
}

func (r *Router) simpleCompletion(backend *Backend, payload map[string]any) (string, error) {
	raw, err := r.rawCompletion(backend, payload)
	if err != nil {
		return "", err
	}
	choices, _ := raw["choices"].([]any)
	if len(choices) == 0 {
		return "", errors.New("completion returned no choices")
	}
	content, reasoning, _ := completionText(raw)
	return preferContent(content, reasoning), nil
}

func (r *Router) rawCompletion(backend *Backend, payload map[string]any) (map[string]any, error) {
	return r.doCompletion(context.Background(), r.client, backend, payload)
}

// benchCompletion issues one cold-start benchmark request bounded by ctx (the
// per-question benchAnswerDeadline). It uses benchClient, which has no client-level
// timeout, so that deadline alone governs whether an answer arrived in time — the
// usability bound is a benchmark criterion, independent of the live-proxy
// BACKEND_TIMEOUT_SECONDS.
func (r *Router) benchCompletion(ctx context.Context, backend *Backend, payload map[string]any) (map[string]any, error) {
	return r.doCompletion(ctx, r.benchClient, backend, payload)
}

// doCompletion POSTs a non-streamed chat completion and returns the decoded body.
// The error string for a non-2xx response ("completion returned <code>: …") is parsed
// by vision.go (completionStatusCode/isClientReject) — keep that prefix stable.
//
// EVERY router-originated generation lands here — the quality benchmark, the
// capacity ramp, the chat/thinking/vision probes and the background judge all
// funnel through doCompletion, rawCompletion or simpleCompletion — so this is
// where they are counted against ActiveRequests.
//
// They were not counted, and the omission was not cosmetic. `ask -l` read s=0/8
// on a worker the benchmark was driving at four concurrent generations; worse,
// expectedLatency, relayOccupancy and the Concurrency request-log column all read
// that same counter, so the fleet's load was understated exactly while a profile
// saturated a GPU, and live traffic was priced as if the worker were idle and
// routed onto it. The capacity ramp is the sharpest case: it fires up to
// CapacityProbeMax concurrent generations (deployed at 16) at a worker that reads
// as completely idle.
//
// The counter belongs HERE rather than around one caller. It was first added
// around benchCompletion alone, which left the judge, the probes and the capacity
// ramp still invisible — the same bug, one call site further out.
//
// None of this takes a SLOT. Slots are the router's promise about how many
// requests a worker will be asked to serve at once, and this background work
// deliberately runs beside live traffic under its own concurrency cap; reserving
// slots would have profiling starve the traffic it exists to improve. Counting
// without reserving is the honest reading: the worker really is this busy, and
// the router really has not set that work aside.
func (r *Router) doCompletion(ctx context.Context, client *http.Client, backend *Backend, payload map[string]any) (map[string]any, error) {
	if r.registry != nil { // nil only in tests that drive a probe or the benchmark directly
		r.registry.incActive(backend.ID, 1)
		defer r.registry.incActive(backend.ID, -1)
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamChatURL(backend), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if backend.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+backend.APIKey)
	}
	// A router-originated call (an expert member, a judge grading) starts its own
	// chain: there is no inbound request behind it, so this router is hop one.
	r.stampRelayChain(req, nil, backend)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("completion returned %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	// Every non-streamed probe and every benchmark question funnels through here,
	// so this one line is the whole cost accounting for a profiling run. A no-op
	// unless that endpoint is being profiled right now (see profileMeter).
	r.meterProfileUsage(backend.ID, raw)
	return raw, nil
}

// modelRecheckInterval is how often the health loop re-fingerprints a ready
// backend's served model. Identical keepalives no longer trigger
// re-certification, so this is what catches a model swap behind an unchanged
// registration (the per-(id,model) profile cache then forces a re-profile).
const modelRecheckInterval = 10 * time.Minute

// healthCheckConcurrency bounds the per-tick parallel health checks so a large
// fleet can't spawn an unbounded burst of HTTP GETs at once.
const healthCheckConcurrency = 16

func (r *Router) healthLoop() {
	ticker := time.NewTicker(r.cfg.HealthInterval)
	defer ticker.Stop()
	for range ticker.C {
		// Fan the per-backend checks out into goroutines: each checkBackend is a
		// blocking HTTP GET with a 5s timeout, so doing them serially let a few
		// unreachable backends drag a single tick past the health interval. The
		// registry methods are all mutex-guarded, so concurrent checks are safe.
		// Concurrency is bounded by a semaphore in case the fleet is large.
		backends := r.registry.snapshot()
		var wg sync.WaitGroup
		sem := make(chan struct{}, healthCheckConcurrency)
		for _, backend := range backends {
			id := backend.ID
			wg.Add(1)
			sem <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() { <-sem }()
				if err := r.checkBackend(id); err == nil {
					// Re-certify failed backends after their backoff, and rescue
					// ones stuck in "probing" (see dueForRecertify); the guard in
					// certifyBackend de-dups overlaps.
					if r.registry.dueForRecertify(id) {
						go r.certifyBackend(id)
					} else if r.registry.modelCheckDue(id, modelRecheckInterval) {
						go r.recheckModel(id)
					}
				}
			}()
		}
		wg.Wait()
	}
}

// recheckModel re-certifies a ready backend whose served model no longer
// matches its profile — e.g. an operator swapped the model without touching
// the registration env.
//
// A relay row is exempt. Its URL answers /v1/models with the upstream's whole
// menu, so the fingerprint would compare this row's one model against a list
// and re-certify it on every check; the drift this guards against is instead
// caught upstream, and reaches here through the next fleet refresh.
func (r *Router) recheckModel(id string) {
	b := r.registry.get(id)
	if b == nil || !b.Certification.Ready || isEmbeddingsOnly(b) || isRelayRow(b) {
		return
	}
	model, meta := r.queryModelInfo(b)
	r.registry.setModelMeta(id, meta)
	if model != "" && b.Model != "" && model != b.Model {
		log.Printf("worker %s now serves model %q (profiled as %q) — re-certifying", id, model, b.Model)
		r.certifyBackend(id)
		return
	}
	// An --alias survives a weights swap, so a stable id proves nothing on its
	// own — compare the weights themselves. Warn rather than re-certify: a full
	// re-benchmark costs minutes of GPU, and this is a fingerprint mismatch, not
	// a measurement.
	if changed := describeWeightsChange(b.ModelMeta, meta); changed != "" {
		log.Printf("worker %s still advertises %q but its weights changed (%s) — cached quality may no longer describe it; delete the backend to force a re-profile", id, model, changed)
	}
	// Same class of drift, cheaper cure: a CTX_SIZE edit moves the context window
	// without moving the model, so neither check above fires. See reconcileContext.
	r.reconcileContext(id, b)
}

// reconcileContext re-reads a ready worker's context window and lands a change
// without re-certifying it.
//
// backfillCachedProfile already re-reads context, but only on the CERTIFICATION
// path, and its own comment overstated when that path runs. A keepalive does not
// certify: handleRegisterBackend deliberately skips a ready backend whose
// registration is unchanged, because certifying parks it in "probing" for two
// round trips every ~60s. A worker restarted with a new CTX_SIZE re-registers a
// byte-identical payload — same id, url, api_key — so nothing is "changed",
// nothing certifies, and the re-read never runs.
//
// Measured: llm-arcb60-Gemma-4-26B-A4B went from 32k to 128k per slot and the
// router kept hard-filtering it at 32k across the restart. The only cure was
// DELETE /backends/{id}, which is exactly the manual step (and the discarded
// quality benchmark) the backfill was written to remove.
//
// So the re-read also runs here, on the health loop's periodic metadata check,
// which is already paying for a round trip and already runs on ready backends
// that nothing else will re-certify. Cost is one GET on vLLM, two on llama.cpp;
// no benchmark, no capacity ramp, and the worker never leaves rotation.
func (r *Router) reconcileContext(id string, b *Backend) {
	ctxK, ok := r.queryContextMeasured(b)
	if !ok {
		return // could not measure — leave the standing value alone, never guess
	}
	was, changed := r.registry.setMeasuredContext(id, ctxK)
	if !changed {
		return
	}
	log.Printf("worker %s: context re-read %dk -> %dk (deployment change; no re-profile needed)", id, was, ctxK)
	// Persist it too, or a router restart certifies from the cached profile and
	// resurrects the window the worker no longer has.
	if r.logs == nil {
		return
	}
	prof, found := r.logs.LoadWorkerProfile(context.Background(), id, b.Model)
	if !found || prof.ContextK == ctxK {
		return
	}
	prof.ContextK = ctxK
	if prof.Checks == nil {
		prof.Checks = map[string]Check{}
	}
	prof.Checks["context"] = Check{OK: true, Message: fmt.Sprintf("%dk (re-read)", ctxK)}
	if err := r.logs.SaveWorkerProfile(context.Background(), id, prof); err != nil {
		log.Printf("persist re-read context for %s failed: %v", id, err)
	}
}

// setMeasuredContext writes a re-measured context window onto a live row and
// reports the previous value and whether it moved.
//
// A MANUAL row's declared context is left alone, the same invariant
// applyProfileIfGen enforces for every other measured field: a probe fills in
// what an operator left blank and never overwrites what they entered.
func (r *Registry) setMeasuredContext(id string, ctxK int) (int, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil || ctxK <= 0 || b.ContextK == ctxK {
		return 0, false
	}
	if operatorDeclared(b).ContextK != 0 {
		return b.ContextK, false
	}
	was := b.ContextK
	b.ContextK = ctxK
	return was, true
}

// describeWeightsChange reports how a worker's loaded weights differ from what
// was last recorded, or "" if they match. Only fields the new probe actually
// read are compared, so a /props blip never reads as a model swap.
func describeWeightsChange(was, now ModelMeta) string {
	var diffs []string
	if now.ModelParams > 0 && was.ModelParams > 0 && now.ModelParams != was.ModelParams {
		diffs = append(diffs, fmt.Sprintf("params %d→%d", was.ModelParams, now.ModelParams))
	}
	if now.ModelQuant != "" && was.ModelQuant != "" && now.ModelQuant != was.ModelQuant {
		diffs = append(diffs, fmt.Sprintf("quant %q→%q", was.ModelQuant, now.ModelQuant))
	}
	if now.ModelPath != "" && was.ModelPath != "" && now.ModelPath != was.ModelPath {
		diffs = append(diffs, fmt.Sprintf("path %q→%q", was.ModelPath, now.ModelPath))
	}
	return strings.Join(diffs, ", ")
}

// modelCheckDue reports whether a ready backend's periodic model fingerprint
// is due, and stamps the check time so concurrent ticks don't pile on.
func (r *Registry) modelCheckDue(id string, interval time.Duration) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil || !b.Certification.Ready {
		return false
	}
	if time.Since(b.lastModelCheck) < interval {
		return false
	}
	b.lastModelCheck = time.Now()
	return true
}

func (r *Router) checkBackend(id string) error {
	backend := r.registry.get(id)
	if backend == nil {
		return errors.New("backend not found")
	}

	healthURL := backendRootURL(backend) + backend.HealthPath
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		r.registry.setHealth(id, false, err.Error())
		return err
	}
	if backend.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+backend.APIKey)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		if isExpired(backend) {
			r.registry.setHealth(id, false, "registration expired")
			return errors.New("registration expired")
		}
		r.registry.setHealth(id, false, err.Error())
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		err := fmt.Errorf("health returned %d", resp.StatusCode)
		r.registry.setHealth(id, false, err.Error())
		return err
	}
	// Backend is healthy — refresh LastSeen to keep registration alive.
	// This means TTL only expires for backends that are truly unreachable.
	r.registry.refreshLastSeen(id)
	r.registry.setHealth(id, true, "")
	return nil
}

func normalizeRegistration(reg *BackendRegistration) error {
	reg.ID = strings.TrimSpace(reg.ID)
	if reg.ID == "" {
		reg.ID = strings.TrimSpace(reg.Model)
	}
	if reg.ID == "" {
		return errors.New("id or model is required")
	}
	if _, err := url.ParseRequestURI(reg.URL); err != nil {
		return fmt.Errorf("valid url is required: %w", err)
	}
	if reg.Model == "" {
		reg.Model = reg.ID
	}
	if reg.Quality < 0 || reg.Quality > 100 {
		return errors.New("quality must be 0..100")
	}
	if reg.HealthPath == "" {
		reg.HealthPath = "/health"
	}
	if !strings.HasPrefix(reg.HealthPath, "/") {
		reg.HealthPath = "/" + reg.HealthPath
	}
	if reg.TTLSeconds <= 0 {
		reg.TTLSeconds = 90
	}
	if reg.MaxConcurrency < 0 {
		reg.MaxConcurrency = 0
	}
	// An upper bound as well as a lower one. syncSlotsLocked fills the slot channel
	// one token at a time while holding the registry's WRITE lock, and the element
	// type is empty struct{} — so there is no memory pressure to stop a large cap,
	// only time: ~12.6ns a token, i.e. 1.3s at 1e8 and 13s at 1e9. For that whole
	// time every eligible(), snapshot(), get(), tryAcquireSlot() and health-loop
	// write blocks, and the router stops routing entirely.
	//
	// The number does not have to be malicious to be wrong: a units mistake, a
	// version-skewed relay reporting a fleet-wide total, or a corrupted provider row
	// all land here, and anything holding the worker token can send one.
	// maxDeclarableConcurrency is far above any real worker while keeping the fill
	// imperceptible.
	if reg.MaxConcurrency > maxDeclarableConcurrency {
		return fmt.Errorf("max_concurrency %d exceeds the %d maximum", reg.MaxConcurrency, maxDeclarableConcurrency)
	}
	reg.Features = normalizeFeatures(reg.Features)
	// Defaults a registration that predates P2 must land on: local, beacon, free.
	normalizeProviderFields(reg)
	return nil
}

func parseAndValidateChatRequest(body []byte, defaultMaxTokens int) (*ChatRequest, error) {
	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("invalid json: %w", err)
	}
	if len(req.Messages) == 0 {
		return nil, errors.New("messages must not be empty")
	}
	// Single source of truth for the completion budget: max_completion_tokens
	// wins over max_tokens (OpenAI's newer name), null/0/absent count as unset.
	// Routing bookkeeping (context estimation) and the forwarded-body patch must
	// agree on this — they previously disagreed on max_completion_tokens and on
	// "max_tokens": null, injecting a conflicting router default alongside the
	// client's real budget.
	eff := effectiveMaxTokens(&req)
	req.ClientSetMaxTokens = eff > 0
	if eff <= 0 {
		eff = defaultMaxTokens
	}
	req.MaxTokens = eff
	for i, msg := range req.Messages {
		switch msg.Role {
		// "function" is the deprecated pre-tools spelling of a tool result, and
		// OpenAI still accepts it. Rejecting it was stricter than the standard the
		// northbound API claims to implement, and inconsistent with the router's
		// own session tracker, which has always recognised it as continuing a tool
		// loop (see inToolLoop). It carries `name` rather than `tool_call_id`, so
		// the tool_call_id check below deliberately does not apply to it.
		case "system", "developer", "user", "assistant", "tool", "function":
		default:
			return nil, fmt.Errorf("messages[%d].role %q is not supported", i, msg.Role)
		}
		if msg.Role == "tool" && msg.ToolCallID == "" {
			return nil, fmt.Errorf("messages[%d] is a tool response without tool_call_id", i)
		}
		if len(msg.ToolCalls) > 0 && string(msg.ToolCalls) != "null" {
			var calls []struct {
				ID string `json:"id"`
			}
			if err := json.Unmarshal(msg.ToolCalls, &calls); err == nil {
				// Uniqueness is per assistant message, not across the whole
				// history — OpenAI semantics; clients that synthesize counter
				// ids (call_0, call_1, reset each turn) are valid.
				seenToolIDs := map[string]bool{}
				for _, call := range calls {
					if call.ID == "" {
						continue
					}
					if seenToolIDs[call.ID] {
						return nil, fmt.Errorf("messages[%d]: duplicate tool_call id %q", i, call.ID)
					}
					seenToolIDs[call.ID] = true
				}
			}
		}
	}
	if len(req.Tools) > 0 && string(req.Tools) != "null" && !json.Valid(req.Tools) {
		return nil, errors.New("tools must be valid json")
	}
	return &req, nil
}

// effectiveMaxTokens returns the completion budget the CLIENT set: the newer
// max_completion_tokens wins over max_tokens; 0 means the client set neither
// (absent, null, or explicit 0 all count as unset).
func effectiveMaxTokens(req *ChatRequest) int {
	if req.MaxCompletionTokens > 0 {
		return req.MaxCompletionTokens
	}
	if req.MaxTokens > 0 {
		return req.MaxTokens
	}
	return 0
}

// thinkingKwargKeys are the chat-template gates a client can pin directly, in
// the precedence the templates themselves use. DeepSeek V4 reads `thinking` and
// falls back to Qwen's `enable_thinking` only when `thinking` is undefined:
//
//	{%- if not thinking is defined -%}
//	  {%- if enable_thinking is defined -%}{%- set thinking = enable_thinking -%}
//
// so a router that looked only at enable_thinking would believe it had control
// of a request whose `thinking` was already pinned the other way, and would
// hard-filter on a value the template then ignored. Verified against the live
// worker: {"thinking":false,"enable_thinking":true} renders thinking OFF.
var thinkingKwargKeys = []string{"thinking", "enable_thinking"}

// thinkingFromRequest maps an explicit chat-template thinking gate (the
// low-level escape hatch) to the equivalent requirements.thinking value, so a
// kwargs choice hard-filters the same way as the standard knob. Absent → ""
// (auto: no filter).
func thinkingFromRequest(req *ChatRequest) string {
	if req.ChatTemplateKwargs == nil {
		return ""
	}
	for _, key := range thinkingKwargKeys {
		if enabled, ok := req.ChatTemplateKwargs[key].(bool); ok {
			if enabled {
				return "on"
			}
			return "off"
		}
	}
	return ""
}

// messageChars is the text the context estimate is derived from. Split out from
// estimateContextK so the same character count can be divided by a MEASURED
// chars-per-token for one model and a different one for another — the hard filter
// asks per candidate, because the tokenizer belongs to the model (see tokens.go).
func messageChars(messages []Message) int {
	chars := 0
	for _, msg := range messages {
		chars += estimateContentChars(msg.Content)
		// tool_calls and tool_call_id also consume tokens
		if len(msg.ToolCalls) > 0 {
			chars += len(msg.ToolCalls)
		}
	}
	return chars
}

// estimateContextK sizes a request in K of context at a given chars-per-token.
// Pass defaultCharsPerToken where no model is in view yet.
func estimateContextK(messages []Message, maxTokens int, charsPerToken float64) int {
	estimatedTokens := tokensForChars(messageChars(messages), charsPerToken) + maxTokens
	return int(math.Ceil(float64(estimatedTokens) / 1024.0))
}

func estimateContentChars(content any) int {
	switch v := content.(type) {
	case string:
		return len(v)
	case nil:
		return 0
	case []any:
		total := 0
		for _, item := range v {
			total += estimateContentChars(item)
		}
		return total
	case map[string]any:
		if t, _ := v["type"].(string); t == "image_url" {
			// Multimodal backends encode images separately; counting a data: URL's
			// base64 bytes as text context can reject valid vision requests.
			return 2048
		}
		total := 0
		for _, item := range v {
			total += estimateContentChars(item)
		}
		return total
	default:
		b, _ := json.Marshal(v)
		return len(b)
	}
}

func requestNeedsVision(messages []Message) bool {
	for _, msg := range messages {
		if contentNeedsVision(msg.Content) {
			return true
		}
	}
	return false
}

func contentNeedsVision(content any) bool {
	switch v := content.(type) {
	case []any:
		for _, item := range v {
			if contentNeedsVision(item) {
				return true
			}
		}
	case map[string]any:
		if t, _ := v["type"].(string); t == "image_url" {
			return true
		}
		for _, item := range v {
			if contentNeedsVision(item) {
				return true
			}
		}
	}
	return false
}

func filterCandidates(candidates []*Backend, keep func(*Backend) bool) []*Backend {
	out := make([]*Backend, 0, len(candidates))
	for _, b := range candidates {
		if keep(b) {
			out = append(out, b)
		}
	}
	return out
}

// backendScore is the fallback ranking score (used only when auto-tiering is
// unavailable): quality-weighted, with speed as a secondary pull.
//
//	quality*3 + speed*1  (prefer better models)
func backendScore(b *Backend, thinkOff bool) int {
	return qualityFor(b, thinkOff)*3 + speedScore(b)
}

// rankBackends sorts candidates best-first and returns the slice. The first
// element is the single best choice; the remaining order lets pickAndAcquire
// spill to the next-best backend when the best one has no free slot. This is the
// the DEGRADED ranker, used when there is no classification to rank on — no
// embeddings worker, or a request with no messages. The live path ranks by the
// outcome matrix (chooseByOutcome).
func rankBackends(candidates []*Backend, job jobCost, thinkOff bool) []*Backend {
	sort.SliceStable(candidates, func(i, j int) bool {
		a, b := candidates[i], candidates[j]
		// 1. Strongly prefer backends with free slots over full ones
		aFull := a.MaxConcurrency > 0 && a.ActiveRequests >= a.MaxConcurrency
		bFull := b.MaxConcurrency > 0 && b.ActiveRequests >= b.MaxConcurrency
		if aFull != bFull {
			return !aFull // prefer the one that's NOT full
		}
		// 2. Combined quality+speed score (quality-weighted)
		sa, sb := backendScore(a, thinkOff), backendScore(b, thinkOff)
		if sa != sb {
			return sa > sb
		}
		// 2.25 Equal score → take the free one. There is no quality bar on this
		//      path (the classifier is unavailable), so cost only gets to separate
		//      workers this ranker already considers interchangeable. Holding a
		//      request off a paid endpoint entirely is the acquire step's job, not
		//      this one — see acquirePreferenceFor.
		if af, bf := isFreeBackend(a), isFreeBackend(b); af != bf {
			return af
		}
		// 2.5 Equal score → prefer the one expected to finish sooner under current
		//     load (live prefill/decode rates + queue occupancy). Only breaks exact
		//     score ties, so it never reorders distinct tiers — but it lets the live
		//     ObservedTPS and active-request count influence routing.
		if la, lb := expectedLatency(a, job), expectedLatency(b, job); la != lb {
			return la < lb
		}
		// 2.75 Still tied → keep the conversation on the worker that served its
		//      previous turn (see session.go).
		if ai, bi := sessionIncumbent(a, job), sessionIncumbent(b, job); ai != bi {
			return ai
		}
		// 3. Context window as tiebreaker
		if a.ContextK != b.ContextK {
			return a.ContextK > b.ContextK
		}
		return a.ID < b.ID
	})
	return candidates
}

// speedScore returns a stable score derived from the backend's baseline_tps,
// scaled to the SAME 0-100 range as Quality so the two are commensurable in
// backendScore.
//
// "Stable" means profile-time, not registration-time: on a beacon row
// BaselineTPS is whatever the speed probe MEASURED (applyProfileIfGen overwrites
// the declared value, and a cached profile re-measures it when speedProbeVersion
// moves); only an operator-entered manual row keeps the number a human typed.
// Either way it does not move with load — that is ObservedTPS, the live EWMA,
// which ranking reads separately through expectedLatency.
//
// It was bucketed 1-10 back when quality was also 1-10. Quality is now a 0-100
// benchmark percentage, which left speed contributing ~3% of backendScore — and
// backendScore is the fallback ranker used when the embeddings worker is down,
// i.e. exactly when difficulty routing is unavailable and speed matters most.
func speedScore(b *Backend) int {
	if b.BaselineTPS <= 0 {
		return 0
	}
	score := int(math.Round(b.BaselineTPS / speedScoreFullTPS * 100))
	if score > 100 {
		score = 100
	}
	return score
}

// speedScoreFullTPS is the decode rate that scores a full 100. Set near the
// fastest worker class in the fleet so slower workers spread out below it.
var speedScoreFullTPS = 150.0

func hasFeature(b *Backend, feature string) bool {
	feature = strings.ToLower(feature)
	for _, f := range b.Features {
		if f == feature {
			return true
		}
	}
	return false
}

func normalizeFeatures(features []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, feature := range features {
		feature = strings.ToLower(strings.TrimSpace(feature))
		if feature == "" || seen[feature] {
			continue
		}
		seen[feature] = true
		out = append(out, feature)
	}
	return out
}

// upsert registers or refreshes a backend. A re-registration with UNCHANGED
// content (the ~60s keepalive) only refreshes liveness: resetting state on
// every keepalive used to knock ready workers out of rotation for two worker
// round-trips per minute and fight the background profiler (stranding fresh
// workers in "probing" for the whole benchmark). Changed content means a
// genuinely new deployment: full reset and a new profile generation. The
// second return value reports whether a full (re)registration happened.
func (r *Registry) upsert(reg BackendRegistration) (*Backend, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := time.Now()
	existing := r.backends[reg.ID]
	if existing == nil {
		existing = &Backend{}
		r.backends[reg.ID] = existing
	} else if existing.lastReg != nil && registrationsEqual(*existing.lastReg, reg) {
		existing.LastSeen = now
		return cloneBackend(existing), false
	}
	regCopy := reg
	existing.lastReg = &regCopy
	existing.profileGen = nextProfileGen.Add(1)
	existing.BackendRegistration = reg
	existing.LastSeen = now
	existing.Status = "probing"
	existing.Healthy = false
	existing.LastError = ""
	existing.Certification = CertState{
		Status:    "probing",
		StartedAt: now,
		Checks:    map[string]Check{},
	}
	// A fresh (re-)registration clears prior failure/backoff bookkeeping so an
	// operator re-registering a backend gets a prompt certification attempt. The
	// learned unknown-field set goes with it: changed registration content means a
	// different deployment, and what the old one refused says nothing about this
	// one (see stripAndRetry).
	existing.certFailures = 0
	existing.nextCertifyAt = time.Time{}
	existing.proxyFailures = 0
	existing.profileAborts = 0
	existing.RejectedFields = nil
	existing.rejectedAt = nil
	// Sync the per-backend slot channel with the registered max_concurrency.
	// On first registration we lazily create the channel; on re-registration
	// with a different cap we replace it. Any in-flight requests still hold
	// a reference to the old channel and release into it harmlessly when
	// they finish — the orphaned tokens are GC'd with the old channel.
	//
	// A registration that declares NO cap does not retract a cap the router
	// measured for itself. Workers beacon with max_concurrency: 0 — they do not
	// know their own capacity, which is the whole reason capacityProbe exists — so
	// deleting the channel here handed a re-registering worker UNBOUNDED admission
	// while the ranker still believed its measured cap. A worker restarting with a
	// changed model would take the entire queue waiting on it at once, mid-reload.
	// Only an explicit operator-declared zero should clear a measured cap, and that
	// arrives through the provider path, not a beacon.
	// r.slotCap, not existing.MaxConcurrency: `existing.BackendRegistration = reg`
	// above has already overwritten the latter with the incoming zero. slotCap is
	// the channel's own record of its capacity and survives the copy.
	switch {
	case reg.MaxConcurrency > 0:
		r.syncSlotsLocked(reg.ID, reg.MaxConcurrency)
	case r.slotCap[reg.ID] > 0:
		// Keep the existing channel as-is. Recreating it here would hand out a
		// fresh full set of tokens while in-flight requests still hold tokens from
		// the old one, briefly over-admitting by exactly the in-flight count.
	default:
		delete(r.slots, reg.ID)
		delete(r.slotCap, reg.ID)
	}
	return cloneBackend(existing), true
}

// registrationsEqual compares registration content (JSON form, so slice fields
// compare by value). Used to recognise keepalives.
func registrationsEqual(a, b BackendRegistration) bool {
	aj, errA := json.Marshal(a)
	bj, errB := json.Marshal(b)
	return errA == nil && errB == nil && bytes.Equal(aj, bj)
}

// maxDeclarableConcurrency bounds any capacity the router will build a slot
// channel for. Two orders of magnitude above the largest real worker, and small
// enough that filling the channel under the write lock is imperceptible.
const maxDeclarableConcurrency = 4096

// syncSlotsLocked (re)creates the slot channel when the concurrency cap
// changes. Callers hold r.mu. In-flight requests keep references to the old
// channel and release into it harmlessly (see releaseSlot).
//
// The cap is clamped here as well as at registration because the relay and
// provider paths reach this function without passing through
// normalizeRegistration — a peer router's /relay/fleet row and an operator-edited
// provider both arrive with a number this process did not choose. The fill loop
// runs under the write lock, so an absurd cap is an availability bug, not just a
// silly one.
func (r *Registry) syncSlotsLocked(id string, cap int) {
	if cap > maxDeclarableConcurrency {
		log.Printf("registry: %s declared max_concurrency %d, clamping to %d", id, cap, maxDeclarableConcurrency)
		cap = maxDeclarableConcurrency
	}
	if cap <= 0 || r.slotCap[id] == cap {
		return
	}
	ch := make(chan struct{}, cap)
	for i := 0; i < cap; i++ {
		ch <- struct{}{}
	}
	r.slots[id] = ch
	r.slotCap[id] = cap
}

// remove deletes a backend and its slot channel. Returns true if a record
// was actually removed. Safe to call concurrently with acquire/release: any
// in-flight requests holding the old slot channel will harmlessly push back
// into it on completion (the channel just lingers until GC'd).
func (r *Registry) remove(id string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.backends[id]; !ok {
		return false
	}
	delete(r.backends, id)
	delete(r.slots, id)
	delete(r.slotCap, id)
	return true
}

// tryAcquireSlot attempts a non-blocking slot acquisition for the named
// backend. Returns (slot, true) on success — including the unbounded case,
// where the slot is nil and releaseSlot is a no-op. Returns (nil, false) when
// the backend has a declared cap and is currently full, or when it is no longer
// registered. Blocking/spilling across backends is handled by pickAndAcquire.
//
// A MISSING slot channel means "uncapped", and that is only the truth while the
// backend still exists. Both remove() and upsert()'s uncapped branch delete the
// entry, so an absent-backend check has to come first: without it, deregistering
// a worker — or a worker re-registering with a changed model — made it look like
// INFINITE free capacity to every request already queued in pickAndAcquire, which
// polls a candidate list snapshotted at request start for up to slotMaxWait. Eight
// requests waiting behind a full single-slot GPU would all dispatch at once into
// the worker that had just dropped out, get 503s while its model reloaded, and
// trip its circuit breaker. Decommissioning a worker made it maximally attractive.
func (r *Registry) tryAcquireSlot(id string) (chan struct{}, bool) {
	r.mu.RLock()
	ch := r.slots[id]
	_, registered := r.backends[id]
	r.mu.RUnlock()
	if !registered {
		return nil, false
	}
	if ch == nil {
		return nil, true
	}
	select {
	case <-ch:
		return ch, true
	default:
		return nil, false
	}
}

// releaseSlot restores the slot to the channel it was acquired from. Safe
// to call with a nil channel (no-op). If the channel has been replaced via
// re-registration with a smaller cap, the orphan push is silently dropped.
func (r *Registry) releaseSlot(ch chan struct{}) {
	if ch == nil {
		return
	}
	select {
	case ch <- struct{}{}:
	default:
		// Old channel that's already at capacity — drop the token, it will
		// be GC'd along with the channel.
	}
}

func (r *Registry) snapshot() []*Backend {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Backend, 0, len(r.backends))
	for _, b := range r.backends {
		out = append(out, cloneBackend(b))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// hasBackendWithFeature reports whether any non-expired backend advertises the
// given feature (e.g. "embeddings"). Used to surface the embeddings dependency
// that auto-routing requires.
func (r *Registry) hasBackendWithFeature(feature string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, b := range r.backends {
		if !isExpired(b) && hasFeature(b, feature) {
			return true
		}
	}
	return false
}

func (r *Registry) eligible() []*Backend {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := []*Backend{}
	for _, b := range r.backends {
		if b.Healthy && b.Certification.Ready && !isExpired(b) {
			out = append(out, cloneBackend(b))
		}
	}
	return out
}

func (r *Registry) get(id string) *Backend {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if b := r.backends[id]; b != nil {
		return cloneBackend(b)
	}
	return nil
}

func (r *Registry) refreshLastSeen(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if b := r.backends[id]; b != nil {
		b.LastSeen = time.Now()
	}
}

// setModelMeta publishes what a worker's runtime reports about its loaded
// weights. Fields the probe couldn't read are left at their previous value: a
// /props that 401s, or a vLLM worker that publishes no parameter count, must
// not blank metadata an earlier certification did manage to read.
func (r *Registry) setModelMeta(id string, m ModelMeta) {
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil {
		return
	}
	if m.ModelPath != "" {
		b.ModelPath = m.ModelPath
	}
	if m.ModelParams > 0 {
		b.ModelParams = m.ModelParams
	}
	if m.ModelQuant != "" {
		b.ModelQuant = m.ModelQuant
	}
	if m.ModelSizeBytes > 0 {
		b.ModelSizeBytes = m.ModelSizeBytes
	}
	if m.ModelCtxTrain > 0 {
		b.ModelCtxTrain = m.ModelCtxTrain
	}
	if m.Engine != "" {
		b.Engine = m.Engine
	}
	if m.ServedID != "" {
		b.ServedID = m.ServedID
	}
}

// setHealth records a health-probe verdict. A SUCCESSFUL probe deliberately does
// not clear LastError: /health only says the process is listening, while
// LastError records what generation actually did, and wiping it every interval
// erased exactly the diagnostic the proxy path had just written. Measured: "HTTP
// 503: CUDA out of memory" became "" after one probe, so a worker that answers
// /health and fails every request showed a blank error field — which reads as
// healthy, and is the symptom upstreamErrorSnippet was added to remove.
func (r *Registry) setHealth(id string, healthy bool, lastError string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if b := r.backends[id]; b != nil {
		b.Healthy = healthy
		// Only a FAILED probe writes the error. A successful one leaves whatever
		// the proxy path recorded, because the two measure different things: the
		// probe says the process is listening, LastError says what a generation
		// did.
		if !healthy {
			b.LastError = lastError
		}
		if healthy {
			b.LastHealthy = time.Now()
			if b.Certification.Ready {
				b.Status = "ready"
			}
		} else {
			b.Status = "unhealthy"
		}
	}
}

func (r *Registry) startCertification(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if b := r.backends[id]; b != nil {
		b.Status = "probing"
		b.Certification = CertState{
			Status:    "probing",
			StartedAt: time.Now(),
			Checks:    map[string]Check{},
		}
	}
}

// setRelayLoad publishes what an upstream router reported about one of its
// models: how many requests it currently has in flight, how many it can hold,
// and how far away it is. Called on every relay refresh (see relay.go).
//
// The capacity also syncs the local slot channel, so this router's own dispatch
// is bounded by the real upstream pool rather than by the last measurement that
// happened to land. A zero capacity is ignored rather than treated as "no
// limit": an upstream that reported nothing has told us nothing.
func (r *Registry) setRelayLoad(id string, active, capacity int, rttMillis float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil {
		return
	}
	if active < 0 {
		active = 0
	}
	b.RemoteActive = active
	b.RelayRTTMillis = rttMillis
	if capacity > 0 {
		b.MaxConcurrency = capacity
		r.syncSlotsLocked(id, capacity)
	}
}

func (r *Registry) finishCertification(id string, ready bool, checks map[string]Check, tps float64, ttft int64, lastError string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if b := r.backends[id]; b != nil {
		b.Certification = CertState{
			Status:       mapBool(ready, "ready", "failed"),
			Ready:        ready,
			StartedAt:    b.Certification.StartedAt,
			FinishedAt:   time.Now(),
			TTFTMillis:   ttft,
			TokensPerSec: tps,
			Checks:       checks,
			LastError:    lastError,
		}
		// Only SEED ObservedTPS from the probe when there's no live EWMA yet. A
		// re-cert (circuit-breaker recovery, model recheck, stuck-probing rescue)
		// must not throw a runtime-learned throughput back to the one-shot profiled
		// baseline — the live EWMA is more current. A genuine model change re-profiles
		// via the cold-start path (a different (id,model) profile), which resets the
		// backend anyway, so this never pins a stale rate across a real model swap.
		if b.ObservedTPS == 0 {
			b.ObservedTPS = tps
		}
		b.LastError = lastError
		if ready {
			b.Healthy = true
			b.Status = "ready"
			b.LastHealthy = time.Now()
			b.certFailures = 0
			b.nextCertifyAt = time.Time{}
			b.proxyFailures = 0
		} else {
			b.Healthy = false
			b.Status = "failed"
			b.certFailures++
			// Back off re-certification of a persistently-failing backend so it
			// isn't re-probed on every health tick (see dueForRecertify).
			b.nextCertifyAt = time.Now().Add(recertBackoff(b.certFailures))
		}
	}
}

// noteRejectedField records that an endpoint refused a request field it did not
// recognise AND then accepted the same request without it, so everything sent to
// it from now on omits that field. Reports whether this is the first time, which
// is what gates the log line.
//
// A re-learn restarts the clock: the field was just re-tested and refused again,
// so the next re-test belongs a full TTL away, not immediately.
func (r *Registry) noteRejectedField(id, field string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil {
		return false
	}
	if b.rejectedAt == nil {
		b.rejectedAt = map[string]time.Time{}
	}
	b.rejectedAt[field] = time.Now()
	for _, f := range b.RejectedFields {
		if f == field {
			return false
		}
	}
	b.RejectedFields = append(b.RejectedFields, field)
	sort.Strings(b.RejectedFields)
	return true
}

// rejectedFields returns the fields this endpoint is still believed to refuse,
// dropping any whose verdict has aged out so the next request re-tests it.
//
// The empty case — every backend, nearly always — costs one read lock and no
// allocation. The returned slice is a copy: the caller filters it, and the
// registry's own may be re-sorted by a concurrent noteRejectedField.
func (r *Registry) rejectedFields(id string) []string {
	r.mu.RLock()
	b := r.backends[id]
	empty := b == nil || len(b.RejectedFields) == 0
	r.mu.RUnlock()
	if empty {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	b = r.backends[id]
	if b == nil {
		return nil
	}
	live := make([]string, 0, len(b.RejectedFields))
	for _, f := range b.RejectedFields {
		// A missing timestamp is treated as expired: it can only come from a row
		// that was written before the clock was, and re-testing is the safe way to
		// resolve a verdict of unknown age.
		if at, ok := b.rejectedAt[f]; !ok || time.Since(at) > rejectedFieldTTL {
			delete(b.rejectedAt, f)
			log.Printf("backend=%s: re-testing %q — the endpoint may have started accepting it since it was learned", id, f)
			continue
		}
		live = append(live, f)
	}
	if len(live) == 0 {
		b.RejectedFields = nil
		return nil
	}
	b.RejectedFields = live
	return append([]string(nil), live...)
}

// noteProfileAbort counts one discarded background profile and returns the
// running total for this registration.
func (r *Registry) noteProfileAbort(id string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil {
		return 0
	}
	b.profileAborts++
	return b.profileAborts
}

// clearProfileAborts forgets the aborted attempts after one finally worked.
func (r *Registry) clearProfileAborts(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if b := r.backends[id]; b != nil {
		b.profileAborts = 0
	}
}

// setCheck updates ONE entry in a certified backend's check list without
// disturbing the certification around it. The background profile is the caller
// that needs it: it finishes long after finishCertification wrote the row, and
// how that run went (or what it cost when it went nowhere) belongs on
// /backends/{id} with the rest of the probe results rather than only in the log.
func (r *Registry) setCheck(id, name string, c Check) {
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil {
		return
	}
	if b.Certification.Checks == nil {
		b.Certification.Checks = map[string]Check{}
	}
	b.Certification.Checks[name] = c
}

func (r *Registry) setError(id string, lastError string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if b := r.backends[id]; b != nil {
		b.LastError = lastError
	}
}

// proxyFailureThreshold is the number of consecutive proxy failures (transport
// errors or 5xx after retries) that trips a backend's circuit breaker.
const proxyFailureThreshold = 3

// noteProxyResult feeds real request outcomes back into eligibility. A run of
// consecutive failures trips a circuit breaker: the backend is dropped from
// rotation and marked for re-certification, so a wedged-but-health-OK backend
// stops receiving traffic until it proves it can serve again. Any success
// resets the counter.
func (r *Registry) noteProxyResult(id string, success bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil {
		return
	}
	if success {
		b.proxyFailures = 0
		return
	}
	b.proxyFailures++
	if b.proxyFailures < proxyFailureThreshold {
		return
	}
	b.proxyFailures = 0
	b.Healthy = false
	b.Status = "unhealthy"
	b.Certification.Ready = false
	b.Certification.Status = "failed"
	b.Certification.LastError = fmt.Sprintf("ejected after %d consecutive proxy failures", proxyFailureThreshold)
	// Re-certify as soon as the next health probe succeeds (no backoff on the
	// first attempt after a trip); the health loop drives the recovery.
	b.nextCertifyAt = time.Time{}
	b.certFailures = 0
}

// dueForRecertify reports whether a backend may be re-certified now: a failed
// backend whose exponential backoff (set in finishCertification) has elapsed,
// or a backend stuck in "probing" — a certification that started long ago and
// never finished (e.g. it bailed on the in-flight-profile guard). The guard in
// certifyBackend de-dups if a profile is genuinely still running.
func (r *Registry) dueForRecertify(id string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	b := r.backends[id]
	if b == nil {
		return false
	}
	switch b.Certification.Status {
	case "failed":
		return b.nextCertifyAt.IsZero() || !time.Now().Before(b.nextCertifyAt)
	case "probing":
		return time.Since(b.Certification.StartedAt) > 2*time.Minute
	}
	return false
}

// recertBackoff grows the gap between re-certification attempts: 30s, 60s,
// 120s, … capped at 10 minutes.
func recertBackoff(failures int) time.Duration {
	const base = 30 * time.Second
	const ceiling = 10 * time.Minute
	if failures < 1 {
		failures = 1
	}
	d := base
	for i := 1; i < failures && d < ceiling; i++ {
		d *= 2
	}
	if d > ceiling {
		d = ceiling
	}
	return d
}

// activeCount is how many requests this router currently has in flight on a
// backend. Recorded on each request so DurationMillis can later be read as
// "slow model" or "busy fleet" rather than an unresolvable mixture of the two —
// a timing model trained without it learns to avoid whichever worker is
// popular.
//
// Call it AFTER incActive, so the returned count includes the request being
// logged. Returns 0 for a backend that is no longer registered, which reads as
// "not recorded" — the same as any other row that never captured it.
func (r *Registry) activeCount(id string) int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if b := r.backends[id]; b != nil {
		return b.ActiveRequests
	}
	return 0
}

func (r *Registry) incActive(id string, delta int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if b := r.backends[id]; b != nil {
		b.ActiveRequests += delta
		if b.ActiveRequests < 0 {
			b.ActiveRequests = 0
		}
	}
}

// minObserveTokens is the smallest streamed response we trust as a live
// throughput sample. Below it the decode window is too short to measure cleanly
// (a 1–2 token reply is dominated by jitter), so we skip it rather than poison
// the EWMA.
const minObserveTokens = 16

// minPrefillTokens is the smallest prompt we trust as a prefill-rate sample.
// Below it TTFT is dominated by fixed request overhead rather than prefill work,
// so the derived tok/s says nothing about how the worker handles a real prompt.
const minPrefillTokens = 256

// decodeSampleRefTokens is the generation length at which a live decode sample
// earns full EWMA weight; shorter samples are weighted down proportionally.
// llama.cpp CPU decode degrades as the KV cache grows, so a stream of short
// replies pushed one CPU worker's ObservedTPS to 51 tok/s when it sustained only
// 17 tok/s over 1700 tokens — a 3x overestimate on precisely the long generations
// where placement matters. GPU workers show no such gap, so weighting by length
// costs nothing there.
var decodeSampleRefTokens = 512

// observe folds one streamed request's measured first-token latency and decode
// throughput into the backend's live EWMAs. decodeWindow is the time spent
// generating *after* the first token (TTFT excluded), so decodeTPS reflects true
// generation speed rather than end-to-end latency — short or large-prompt
// requests no longer drag the number down. Only streamed requests can separate
// the phases; buffered requests don't call this.
//
// promptTokens sizes the prefill sample. thinking reports whether the router
// enabled a thinking phase for this request: TTFT is only comparable across
// workers when it didn't, because vLLM buffers reasoning (a thinking turn's whole
// think phase lands inside TTFT — measured 12.45s of a 13.15s turn) while
// llama.cpp streams it (0.7s on the same job). Folding both into one EWMA made
// the faster prefill engine look ~30x slower and pushed traffic to CPU workers
// while the GPU sat idle.
func (r *Registry) observe(id string, ttft, decodeWindow time.Duration, tokens, promptTokens int, thinking bool) {
	if tokens < minObserveTokens || decodeWindow <= 0 {
		return
	}
	decodeTPS := float64(tokens) / decodeWindow.Seconds()
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil {
		return
	}
	// Weight the sample by how much generation it actually observed.
	alpha := 0.3
	if decodeSampleRefTokens > 0 {
		alpha *= math.Min(1, float64(tokens)/float64(decodeSampleRefTokens))
	}
	ewma(&b.ObservedTPS, decodeTPS, alpha)
	// The same two samples again, kept per mode. Nothing here assumes the modes
	// agree; if they turn out to, the two EWMAs simply converge and cost nothing.
	// The generated-length EWMA uses a fixed weight rather than the
	// tokens-weighted alpha above: alpha exists to trust a long sample more about
	// a RATE, but for a length the long samples are precisely the observations
	// that must not dominate — they are what we are trying to predict.
	tpsSlot, genSlot := &b.ObservedTPSNoThink, &b.ObservedGenNoThink
	if thinking {
		tpsSlot, genSlot = &b.ObservedTPSThink, &b.ObservedGenThink
	}
	ewma(tpsSlot, decodeTPS, alpha)
	ewma(genSlot, float64(tokens), 0.2)
	if thinking || ttft <= 0 {
		return
	}
	// On a relay row the measured first-token latency contains the hop to the
	// other ROUTER as well as the endpoint's own prefill, and everything that
	// reads these two fields expects the endpoint's figure alone — the imported
	// profile carries it that way, and prefillSeconds adds the link back once.
	// Leaving the link in would make a locally-measured sample mean something
	// different from an imported one, and then charge for the link twice.
	//
	// A sample that is somehow shorter than the link itself measures nothing but
	// clock noise, so it is dropped rather than clamped.
	if wire := time.Duration(b.RelayRTTMillis) * time.Millisecond; wire > 0 {
		if ttft -= wire; ttft <= 0 {
			return
		}
	}
	// Fixed 0.3 for both, unlike the tokens-weighted alpha above: a prefill sample
	// is one prompt, and there is no "how much of it did we observe" to weight by.
	ewma(&b.ObservedTTFTMillis, float64(ttft.Milliseconds()), 0.3)
	if promptTokens >= minPrefillTokens {
		ewma(&b.ObservedPrefillTPS, float64(promptTokens)/ttft.Seconds(), 0.3)
	}
}

func cloneBackend(b *Backend) *Backend {
	cp := *b
	cp.Features = append([]string(nil), b.Features...)
	cp.RejectedFields = append([]string(nil), b.RejectedFields...)
	// The verdict CLOCK stays behind the registry lock. A clone that shared the
	// map would hand a reader something the next noteRejectedField writes to;
	// everything that needs the timestamps asks the registry (rejectedFields).
	cp.rejectedAt = nil
	if b.Certification.Checks != nil {
		cp.Certification.Checks = make(map[string]Check, len(b.Certification.Checks))
		for k, v := range b.Certification.Checks {
			cp.Certification.Checks[k] = v
		}
	}
	return &cp
}

func isExpired(b *Backend) bool {
	return time.Since(b.LastSeen) > time.Duration(b.TTLSeconds)*time.Second
}

func openLogStore(path string, maxBody int, persistSecret string) (*LogStore, error) {
	dir := filepath.Dir(path)
	// 0o700: the DB holds request bodies and the key file holds the
	// registration-encryption key — neither should be group/world readable.
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	box, err := newSecretBox(persistSecret, filepath.Join(dir, "persist.key"))
	if err != nil {
		return nil, fmt.Errorf("init persistence encryption: %w", err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if maxBody <= 0 {
		maxBody = 16384
	}
	store := &LogStore{db: db, maxBody: maxBody, box: box}
	if err := store.init(context.Background()); err != nil {
		db.Close()
		return nil, err
	}
	return store, nil
}

// secretBox encrypts small secrets (backend API keys) before they are written
// to the on-disk registration table, so the SQLite file never contains
// plaintext credentials. The key comes from ROUTER_PERSIST_SECRET when set,
// otherwise from an auto-generated 32-byte key file kept beside the database
// (0600). Sealed values are "enc:v1:<base64(nonce|ciphertext)>"; anything
// without that prefix is treated as legacy plaintext and returned as-is.
type secretBox struct {
	gcm cipher.AEAD
}

const encPrefix = "enc:v1:"

func newSecretBox(envSecret, keyPath string) (*secretBox, error) {
	var key []byte
	if strings.TrimSpace(envSecret) != "" {
		sum := sha256.Sum256([]byte(envSecret))
		key = sum[:]
	} else {
		k, err := loadOrCreateKeyFile(keyPath)
		if err != nil {
			return nil, err
		}
		key = k
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	return &secretBox{gcm: gcm}, nil
}

// loadOrCreateKeyFile reads a 32-byte base64 key from path, generating and
// persisting one (0600) on first run.
func loadOrCreateKeyFile(path string) ([]byte, error) {
	if raw, err := os.ReadFile(path); err == nil {
		key, err := base64.StdEncoding.DecodeString(strings.TrimSpace(string(raw)))
		if err != nil {
			return nil, fmt.Errorf("decode key file %s: %w", path, err)
		}
		if len(key) != 32 {
			return nil, fmt.Errorf("key file %s must decode to 32 bytes, got %d", path, len(key))
		}
		return key, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		return nil, err
	}
	if err := os.WriteFile(path, []byte(base64.StdEncoding.EncodeToString(key)), 0o600); err != nil {
		return nil, err
	}
	return key, nil
}

// seal encrypts a non-empty secret. Empty input stays empty (no marker).
func (s *secretBox) seal(plain string) (string, error) {
	if plain == "" {
		return "", nil
	}
	nonce := make([]byte, s.gcm.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return "", err
	}
	ct := s.gcm.Seal(nonce, nonce, []byte(plain), nil)
	return encPrefix + base64.StdEncoding.EncodeToString(ct), nil
}

// open reverses seal. Values without the marker are returned unchanged so any
// pre-existing plaintext rows keep working.
func (s *secretBox) open(stored string) (string, error) {
	if !strings.HasPrefix(stored, encPrefix) {
		return stored, nil
	}
	raw, err := base64.StdEncoding.DecodeString(stored[len(encPrefix):])
	if err != nil {
		return "", err
	}
	if len(raw) < s.gcm.NonceSize() {
		return "", errors.New("ciphertext too short")
	}
	nonce, ct := raw[:s.gcm.NonceSize()], raw[s.gcm.NonceSize():]
	plain, err := s.gcm.Open(nil, nonce, ct, nil)
	if err != nil {
		return "", err
	}
	return string(plain), nil
}

func (s *LogStore) init(ctx context.Context) error {
	stmts := []string{
		`PRAGMA journal_mode=WAL`,
		// The pool is capped at one connection so this process never contends
		// with itself, but WAL lets other processes open the file concurrently
		// — backup.sh and any manual sqlite3 session do. Without a busy timeout
		// those collide as an immediate SQLITE_BUSY, which surfaces as a lost
		// request log or a failed registration write.
		`PRAGMA busy_timeout=5000`,
		`CREATE TABLE IF NOT EXISTS request_logs (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			backend_id TEXT NOT NULL,
			backend_model TEXT NOT NULL,
			route TEXT NOT NULL,
			observed_tps REAL NOT NULL DEFAULT 0,
			certified_tps REAL NOT NULL DEFAULT 0,
			baseline_tps REAL NOT NULL DEFAULT 0,
			speed_score INTEGER NOT NULL DEFAULT 0,
			stream INTEGER NOT NULL,
			status_code INTEGER NOT NULL,
			duration_ms INTEGER NOT NULL,
			prompt_tokens INTEGER NOT NULL DEFAULT 0,
			completion_tokens INTEGER NOT NULL DEFAULT 0,
			ttft_ms INTEGER NOT NULL DEFAULT 0,
			thinking INTEGER NOT NULL DEFAULT 0,
			concurrency INTEGER NOT NULL DEFAULT 0,
			input TEXT NOT NULL,
			output TEXT NOT NULL,
			error TEXT NOT NULL
		)`,
		`CREATE INDEX IF NOT EXISTS idx_request_logs_backend_created ON request_logs (backend_id, created_at DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_request_logs_created ON request_logs (created_at DESC)`,
		`CREATE TABLE IF NOT EXISTS backend_registrations (
			id TEXT PRIMARY KEY,
			updated_at TEXT NOT NULL,
			registration_json TEXT NOT NULL
		)`,
		// worker_profiles is the cached cold-start profile for the whole fleet, and
		// re-measuring it costs hours of GPU time and, since P2, real money. Note
		// the key here is `id` alone while LoadWorkerProfile selects on
		// (id, model): the effect is that one id holds one profile, and a worker
		// that changes model re-profiles and replaces the row. That is correct
		// under P2's "one row per (endpoint, model)" rule — every servable thing
		// has its own id — so the mismatch is left alone rather than rebuilt,
		// because rebuilding a table is exactly how this data gets lost.
		`CREATE TABLE IF NOT EXISTS worker_profiles (
			id TEXT PRIMARY KEY,
			model TEXT NOT NULL,
			updated_at TEXT NOT NULL,
			profile_json TEXT NOT NULL
		)`,
		// Virtual keys (P3). key_hash is a SHA-256 of the plaintext; the plaintext
		// exists once, in the response to the call that created it.
		`CREATE TABLE IF NOT EXISTS api_keys (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			key_hash TEXT NOT NULL UNIQUE,
			prefix TEXT NOT NULL,
			name TEXT NOT NULL DEFAULT '',
			role TEXT NOT NULL,
			enabled INTEGER NOT NULL DEFAULT 1,
			created_at TEXT NOT NULL,
			last_used_at TEXT NOT NULL DEFAULT '',
			models TEXT NOT NULL DEFAULT '',
			token_budget INTEGER NOT NULL DEFAULT 0,
			tokens_used INTEGER NOT NULL DEFAULT 0
		)`,
		// Authentication reads this on every request that presents a bearer token,
		// so it is an index rather than a scan.
		`CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys (key_hash)`,
		// Named groups (P5). The name is stored already lowercased (see groupKey),
		// so the primary key is the lookup key; members are a JSON array because a
		// member is a model id and model ids are not a comma-free alphabet.
		`CREATE TABLE IF NOT EXISTS router_groups (
			name TEXT PRIMARY KEY,
			members TEXT NOT NULL,
			updated_at TEXT NOT NULL
		)`,
		// Key/value rather than a one-row table with a column per setting, which
		// would need a migration for every setting ever added. Holds the admin
		// password hash; the database is canonical for it.
		`CREATE TABLE IF NOT EXISTS router_settings (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL,
			updated_at TEXT NOT NULL
		)`,
		// One graded answer per (question, model, thinking mode, source). REPLACE
		// on that key, so a re-profile supersedes its predecessor rather than
		// accumulating beside it — a model's history would otherwise drag its
		// current estimate. See outcomes.go.
		//
		// Keyed by MODEL, not by worker. A graded answer is evidence about the
		// weights that produced it, so it outlives the box: decommission a worker
		// and its results stay, deploy the same model elsewhere and the profile is
		// already there. backend_id rides along as provenance — it is what tells a
		// latency this worker measured from one another deployment measured on
		// different hardware.
		//
		// The PRE-MODEL-HASH table is dropped by dropLegacyObservations below, NOT
		// from this list. Putting a DROP here made it run on EVERY startup rather
		// than once: init() is a plain statement list executed by NewLogStore each
		// boot, so the permacache was emptied on every restart, and the judged half
		// — up to maxJudgedQuestions production questions and the vectors persisted
		// precisely so they would survive — was destroyed with no second copy to
		// rebuild from. A one-shot migration has to be able to tell that it has
		// already run.
		`CREATE TABLE IF NOT EXISTS observations (
			qid TEXT NOT NULL,
			model_hash TEXT NOT NULL,
			backend_id TEXT NOT NULL DEFAULT '',
			thinking INTEGER NOT NULL,
			correct INTEGER NOT NULL,
			loose INTEGER NOT NULL DEFAULT 0,
			latency_ms INTEGER NOT NULL DEFAULT 0,
			source TEXT NOT NULL,
			created_at TEXT NOT NULL,
			PRIMARY KEY (qid, model_hash, thinking, source)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_observations_model ON observations (model_hash)`,
		// Upstream routers this one may relay through (see relay.go). The name is
		// stored already lowercased (see relayKey), so the primary key is the
		// lookup key. api_key is sealed with the same box that protects a backend's
		// credential; the backends this relay expands into are DERIVED and are not
		// persisted anywhere — the refresh loop rebuilds them from this row.
		`CREATE TABLE IF NOT EXISTS router_relays (
			name TEXT PRIMARY KEY,
			url TEXT NOT NULL,
			api_key TEXT NOT NULL DEFAULT '',
			enabled INTEGER NOT NULL DEFAULT 1,
			updated_at TEXT NOT NULL
		)`,
	}
	// Before the CREATEs, so the new table is created after the old one is gone.
	if err := s.dropLegacyObservations(ctx); err != nil {
		return err
	}
	for _, stmt := range stmts {
		if _, err := s.db.ExecContext(ctx, stmt); err != nil {
			return err
		}
	}
	// Columns added to a table that already exists in the field. SQLite has no
	// ADD COLUMN IF NOT EXISTS, so the duplicate is swallowed and everything else
	// is still an error — a typo must not pass as "already applied".
	for _, stmt := range []string{
		`ALTER TABLE request_logs ADD COLUMN observed_tps REAL NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN certified_tps REAL NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN baseline_tps REAL NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN speed_score INTEGER NOT NULL DEFAULT 0`,
		// Which key made the request. Rows written before P3 keep the empty
		// default, which reads as "the router required no credential" — true of
		// every request made under the old client-token-or-nothing scheme.
		`ALTER TABLE request_logs ADD COLUMN key_id TEXT NOT NULL DEFAULT ''`,
		// Where the request came in from, for the dashboard's usage graph. Every
		// row already in the table keeps the empty default and there is nothing to
		// backfill it from — the address was never recorded — so an existing
		// database shows twelve hours of "unknown" until it has been running with
		// this build for twelve hours. That is the honest rendering; the
		// alternative of hiding those rows would make a busy router look idle.
		`ALTER TABLE request_logs ADD COLUMN client_ip TEXT NOT NULL DEFAULT ''`,
		// Timing-model columns. Existing rows keep their defaults and there is
		// nothing to backfill them from: token counts were parsed and dropped,
		// and TTFT, thinking mode and concurrency were never recorded at all. A
		// timing model must therefore treat 0 as MISSING rather than as a
		// measurement, which is why thinkingMode's zero value is "unknown" and
		// why concurrency is recorded including the current request (so a real
		// value is never 0).
		`ALTER TABLE request_logs ADD COLUMN prompt_tokens INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN completion_tokens INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN ttft_ms INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN thinking INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN concurrency INTEGER NOT NULL DEFAULT 0`,
		// Whether a key belongs to a downstream router rather than a client
		// (see relay.go). Every key issued before relays existed is not one, and
		// 0 is what the default gives them.
		`ALTER TABLE api_keys ADD COLUMN relay INTEGER NOT NULL DEFAULT 0`,
	} {
		if _, err := s.db.ExecContext(ctx, stmt); err != nil && !strings.Contains(err.Error(), "duplicate column name") {
			return err
		}
	}
	return nil
}

func (s *LogStore) Close() error {
	return s.db.Close()
}

func (s *LogStore) Insert(ctx context.Context, entry RequestLog) error {
	_, err := s.db.ExecContext(ctx, `INSERT INTO request_logs
		(created_at, backend_id, backend_model, route, observed_tps, certified_tps, baseline_tps, speed_score, stream, status_code, duration_ms, prompt_tokens, completion_tokens, ttft_ms, thinking, concurrency, input, output, error, key_id, client_ip)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		entry.CreatedAt.Format(time.RFC3339Nano),
		entry.BackendID,
		entry.BackendModel,
		entry.Route,
		entry.ObservedTPS,
		entry.CertifiedTPS,
		entry.BaselineTPS,
		entry.SpeedScore,
		boolInt(entry.Stream),
		entry.StatusCode,
		entry.DurationMillis,
		entry.PromptTokens,
		entry.CompletionTokens,
		entry.TTFTMillis,
		entry.Thinking,
		entry.Concurrency,
		clipLog(entry.Input, s.maxBody),
		clipLog(entry.Output, s.maxBody),
		clipLog(entry.Error, s.maxBody),
		entry.KeyID,
		entry.ClientIP,
	)
	return err
}

// clipLog bounds a stored log field to maxBytes, trimming back to a UTF-8 rune
// boundary and appending a truncation marker. Keeps the request-log table from
// growing without limit on large prompts/responses.
func clipLog(s string, maxBytes int) string {
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s
	}
	cut := maxBytes
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + fmt.Sprintf("…[truncated %d bytes]", len(s)-cut)
}

func (s *LogStore) List(ctx context.Context, backendID string, limit int, offset int) ([]RequestLog, error) {
	var (
		rows *sql.Rows
		err  error
	)
	if backendID == "" {
		rows, err = s.db.QueryContext(ctx, `SELECT id, created_at, backend_id, backend_model, route, observed_tps, certified_tps, baseline_tps, speed_score, stream, status_code, duration_ms, prompt_tokens, completion_tokens, ttft_ms, thinking, concurrency, input, output, error, key_id, client_ip
			FROM request_logs ORDER BY created_at DESC LIMIT ? OFFSET ?`, limit, offset)
	} else {
		rows, err = s.db.QueryContext(ctx, `SELECT id, created_at, backend_id, backend_model, route, observed_tps, certified_tps, baseline_tps, speed_score, stream, status_code, duration_ms, prompt_tokens, completion_tokens, ttft_ms, thinking, concurrency, input, output, error, key_id, client_ip
			FROM request_logs WHERE backend_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?`, backendID, limit, offset)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []RequestLog{}
	for rows.Next() {
		var entry RequestLog
		var created string
		var stream int
		if err := rows.Scan(&entry.ID, &created, &entry.BackendID, &entry.BackendModel, &entry.Route, &entry.ObservedTPS, &entry.CertifiedTPS, &entry.BaselineTPS, &entry.SpeedScore, &stream, &entry.StatusCode, &entry.DurationMillis, &entry.PromptTokens, &entry.CompletionTokens, &entry.TTFTMillis, &entry.Thinking, &entry.Concurrency, &entry.Input, &entry.Output, &entry.Error, &entry.KeyID, &entry.ClientIP); err != nil {
			return nil, err
		}
		entry.CreatedAt, _ = time.Parse(time.RFC3339Nano, created)
		entry.Stream = stream != 0
		out = append(out, entry)
	}
	return out, rows.Err()
}

// ── Usage graph ─────────────────────────────────────────────────────────────
//
// What the dashboard's usage chart plots is ACTIVE SLOTS: how many requests were
// in flight at once, split by the address they came from. Not a request count —
// forty one-second embedding calls and one four-minute generation are wildly
// different load on a fleet whose workers have a max_concurrency in the single
// digits, and it is the second one that fills a slot and makes the next caller
// queue. Counting requests would draw those the same way round the wrong way.
//
// Per bucket the figure is the MEAN number in flight: busy seconds accumulated
// inside the bucket, over the length of the bucket (Little's law). It is a real
// number rather than an integer, and a value of 0.1 on a five-minute bucket
// means one thirty-second request, not "almost none".
const (
	usageWindow = 12 * time.Hour
	// Five minutes, which makes 144 columns across twelve hours. That is about
	// as many stacked columns as a chart a thousand pixels wide can draw and
	// still have each one readable — roughly six pixels of bar plus the two-pixel
	// gap that separates the segments of a stack. One-minute buckets would be 720
	// columns at under two pixels apiece: the stack ordering stops being legible,
	// and any client that is not the busiest disappears into a sub-pixel sliver.
	//
	// Buckets are aligned to absolute multiples of five minutes, never to "now".
	// The graph reloads every ten seconds, and a window anchored to the poll
	// instant would give every one of the 144 columns a slightly different value
	// each time — the whole chart would shimmer. Aligned, a refresh changes the
	// newest column and, twice an hour, drops the oldest.
	usageBucket = 5 * time.Minute
	// The page's categorical palette has eight validated slots and a ninth colour
	// would be indistinguishable from one of them under colour-blind simulation,
	// so the ninth-busiest client onwards is folded into a single grey band. See
	// seriesFill in dashboard.html.
	usageTopClients = 8
	// What a row with no address is drawn as. Every row written before the
	// client_ip column existed is one of these.
	usageUnknownClient = "unknown"
	usageOtherClients  = "other"
	// How long before the window a request may have STARTED and still have the
	// part of it that lands inside the window drawn.
	//
	// The scan has to be a range over created_at, because that is the indexed
	// column; "which requests were still running at 04:00" is not a question that
	// index can answer without reading the whole table, which this query does
	// every ten seconds. So it asks for rows that started within an hour of the
	// window and clamps them. A request that ran for longer than an hour loses
	// its in-window tail. Nothing this router serves comes close — slotMaxWait is
	// ten minutes and a generation on top of it is minutes, not hours — and the
	// alternative is a full table scan on every poll.
	usageMaxRequestAge = time.Hour
)

// UsageSeries is one render of the usage chart: a fixed frame of buckets and the
// bands stacked inside it.
type UsageSeries struct {
	BucketSeconds int64 `json:"bucket_seconds"`
	From          int64 `json:"from"`
	To            int64 `json:"to"`
	// Clients is the stacking order, busiest first, with "other" and "unknown"
	// pinned last so the identified traffic sits at the bottom of the stack where
	// it is easiest to compare between columns.
	Clients []UsageClient `json:"clients"`
	// Buckets always covers the whole window, empty ones included. The browser
	// gets a complete axis rather than a set of points with holes to interpolate,
	// which is what keeps a quiet hour looking quiet instead of looking absent.
	Buckets []UsageBucket `json:"buckets"`
}

// UsageClient is one band. Slots is the mean over the WHOLE window, so it ranks
// the bands against each other but is not the height of any single column; Peak
// is the busiest bucket that band had.
type UsageClient struct {
	ClientIP string  `json:"client_ip"`
	Slots    float64 `json:"slots"`
	Peak     float64 `json:"peak"`
	Requests int64   `json:"requests"`
}

// UsageBucket is one column. Slots and Requests are positional against
// UsageSeries.Clients: the addresses are written once for the whole payload
// instead of once per bucket, which is the difference between about 25KB every
// ten seconds and several times that.
type UsageBucket struct {
	At       int64     `json:"t"`
	Slots    []float64 `json:"slots"`
	Requests []int64   `json:"requests"`
}

// usageSeriesQuery buckets the request log into mean concurrency per client.
//
// All of the work is here rather than in the browser on purpose: twelve hours of
// this table is tens of thousands of rows carrying stored prompts and completions,
// and shipping that to a page to bucket it in JavaScript would move megabytes
// every ten seconds to draw a chart with 144 columns in it.
const usageSeriesQuery = `
WITH RECURSIVE
	-- Each logged request as a half-open interval of epoch seconds.
	--
	-- created_at is the moment the request STARTED, not the moment it finished:
	-- every insert site sets it from the same instant duration_ms is later measured
	-- against (proxyToBackend, proxyPassthrough, serveExpert and logSlotUnavailable
	-- all do CreatedAt: start.UTC(), where start := time.Now() on the way in, and
	-- their deferred DurationMillis is time.Since(start) on the way out). So the
	-- interval runs FORWARD from created_at.
	-- If that ever became a completion time this arithmetic has to run backwards
	-- instead, and until it does every column on the chart is displaced by one
	-- request duration — which is invisible, because the chart still looks
	-- perfectly reasonable.
	--
	-- strftime('%s') truncates to whole seconds, moving a request at most one
	-- second earlier. Against a five-minute bucket that is not worth the
	-- julianday arithmetic it would take to avoid.
	raw AS (
		SELECT
			CASE WHEN client_ip = '' THEN ? ELSE client_ip END AS ip,
			CAST(strftime('%s', created_at) AS INTEGER) AS s0,
			CAST(strftime('%s', created_at) AS INTEGER) + max(duration_ms, 0) / 1000.0 AS e0
		FROM request_logs
		WHERE created_at >= ? AND created_at < ?
	),
	-- Clamped to the window, so a request straddling either edge contributes only
	-- the part of itself that is on screen. e0 >= lo drops the rows that the
	-- look-back above dragged in but that had already finished.
	--
	-- MATERIALIZED because this is read twice below, and the alternative is
	-- scanning and clamping the whole window a second time to work out who the
	-- busiest callers were. Measured at 200k rows: 1.64s down to 1.21s.
	spans AS MATERIALIZED (
		SELECT ip, max(s0, ?) AS s, min(e0, ?) AS e FROM raw WHERE e0 >= ?
	),
	-- Which addresses get a band of their own. Ranked on busy time rather than on
	-- request count, for the same reason the chart plots slots rather than
	-- requests: the heaviest caller is the one holding workers, not the one
	-- making the most calls. Ties break on the address so the ranking is stable
	-- between two polls that see identical data.
	busiest AS (
		SELECT ip FROM spans GROUP BY ip ORDER BY sum(e - s) DESC, ip LIMIT ?
	),
	-- One row per (request, bucket it overlapped). The seed is the bucket the
	-- request started in; the recursion walks forward for as long as it was still
	-- running. That is what puts a four-minute generation into every bucket it
	-- spanned instead of only into the one it began in — the difference between a
	-- graph that shows sustained load and one that shows a spike at the start of
	-- it. It cannot run away: e is already clamped to the end of the window.
	covered(ip, s, e, b) AS (
		SELECT ip, s, e, (s / ?) * ? FROM spans
		UNION ALL
		SELECT ip, s, e, b + ? FROM covered WHERE b + ? < e
	)
SELECT
	c.b,
	coalesce(busiest.ip, ?) AS series,
	sum(min(c.e, c.b + ?) - max(c.s, c.b)) AS busy_seconds,
	count(*) AS requests
FROM covered c
LEFT JOIN busiest ON busiest.ip = c.ip
GROUP BY c.b, series
ORDER BY c.b`

// UsageSeries returns the chart frame for the window ending at now. It always
// returns a complete frame: an error is an error, but an empty table is a valid
// answer with every bucket at zero, because "nothing has happened for twelve
// hours" is a thing an operator needs to be able to see.
func (s *LogStore) UsageSeries(ctx context.Context, now time.Time, window, bucket time.Duration, top int) (*UsageSeries, error) {
	out := newUsageSeries(now, window, bucket)
	if top < 1 {
		top = 1
	}
	lo, hi, bucketSecs := out.From, out.To, out.BucketSeconds
	rows, err := s.db.QueryContext(ctx, usageSeriesQuery,
		usageUnknownClient,
		time.Unix(lo, 0).UTC().Add(-usageMaxRequestAge).Format(time.RFC3339Nano),
		time.Unix(hi, 0).UTC().Format(time.RFC3339Nano),
		lo, hi, lo,
		top,
		bucketSecs, bucketSecs, bucketSecs, bucketSecs,
		usageOtherClients,
		bucketSecs,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type cell struct {
		bucket   int
		busy     float64
		requests int64
	}
	cells := map[string][]cell{}
	totals := map[string]*UsageClient{}
	for rows.Next() {
		var (
			at       int64
			series   string
			busy     float64
			requests int64
		)
		if err := rows.Scan(&at, &series, &busy, &requests); err != nil {
			return nil, err
		}
		// The window bounds are compared against created_at as TEXT, and RFC3339
		// sorts one fractional second the wrong way round at an exact boundary
		// (".5Z" sorts before "Z"), so a row up to a second outside the window can
		// survive the WHERE. Discarding an out-of-frame bucket here is both the
		// fix and the guard against ever indexing outside the slice.
		idx := int((at - lo) / bucketSecs)
		if idx < 0 || idx >= len(out.Buckets) || busy < 0 {
			continue
		}
		cells[series] = append(cells[series], cell{bucket: idx, busy: busy, requests: requests})
		total, ok := totals[series]
		if !ok {
			total = &UsageClient{ClientIP: series}
			totals[series] = total
		}
		total.Slots += busy
		total.Requests += requests
		if slots := busy / float64(bucketSecs); slots > total.Peak {
			total.Peak = slots
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	windowSecs := float64(hi - lo)
	for _, total := range totals {
		total.Slots = roundSlots(total.Slots / windowSecs)
		total.Peak = roundSlots(total.Peak)
		out.Clients = append(out.Clients, *total)
	}
	// Busiest first, with the two bands that are not an identity — the tail fold
	// and the rows that predate the column — pinned to the end. They stack on top,
	// above the addresses, so that a band an operator can actually act on never
	// has an unattributable one wobbling underneath it and moving its baseline.
	sort.Slice(out.Clients, func(i, j int) bool {
		a, b := out.Clients[i], out.Clients[j]
		if ra, rb := usageBandRank(a.ClientIP), usageBandRank(b.ClientIP); ra != rb {
			return ra < rb
		}
		if a.Slots != b.Slots {
			return a.Slots > b.Slots
		}
		return a.ClientIP < b.ClientIP
	})
	for i := range out.Buckets {
		out.Buckets[i].Slots = make([]float64, len(out.Clients))
		out.Buckets[i].Requests = make([]int64, len(out.Clients))
	}
	for i, client := range out.Clients {
		for _, c := range cells[client.ClientIP] {
			out.Buckets[c.bucket].Slots[i] = roundSlots(c.busy / float64(bucketSecs))
			out.Buckets[c.bucket].Requests[i] = c.requests
		}
	}
	return out, nil
}

// newUsageSeries builds the empty frame: every bucket in the window, at zero.
// The nil-database case and the never-used-router case both answer with this, so
// the page has one shape to draw and no special case for "no data".
func newUsageSeries(now time.Time, window, bucket time.Duration) *UsageSeries {
	bucketSecs := int64(bucket / time.Second)
	if bucketSecs < 1 {
		bucketSecs = 60
	}
	count := int(int64(window/time.Second) / bucketSecs)
	if count < 1 {
		count = 1
	}
	// The frame ends at the end of the bucket currently in progress, so the newest
	// column is always the live one rather than a stale complete bucket.
	hi := (now.UTC().Unix()/bucketSecs + 1) * bucketSecs
	lo := hi - int64(count)*bucketSecs
	out := &UsageSeries{
		BucketSeconds: bucketSecs,
		From:          lo,
		To:            hi,
		Clients:       []UsageClient{},
		Buckets:       make([]UsageBucket, count),
	}
	for i := range out.Buckets {
		out.Buckets[i] = UsageBucket{At: lo + int64(i)*bucketSecs, Slots: []float64{}, Requests: []int64{}}
	}
	return out
}

// usageBandRank sorts the two non-identity bands after every real address.
func usageBandRank(client string) int {
	switch client {
	case usageOtherClients:
		return 1
	case usageUnknownClient:
		return 2
	}
	return 0
}

// roundSlots trims a concurrency figure to three decimals. Purely a payload
// size decision — 144 buckets times ten bands of full float64 precision is most
// of the JSON, and the chart cannot draw a thousandth of a slot anyway.
func roundSlots(v float64) float64 {
	return math.Round(v*1000) / 1000
}

func (s *LogStore) DeleteOlderThan(ctx context.Context, cutoff time.Time) (int64, error) {
	result, err := s.db.ExecContext(ctx, `DELETE FROM request_logs WHERE created_at < ?`, cutoff.UTC().Format(time.RFC3339Nano))
	if err != nil {
		return 0, err
	}
	return result.RowsAffected()
}

func (s *LogStore) SaveBackendRegistration(ctx context.Context, reg BackendRegistration) error {
	// Encrypt the api key so the SQLite file never holds a plaintext credential.
	sealed, err := s.box.seal(reg.APIKey)
	if err != nil {
		return fmt.Errorf("encrypt api key: %w", err)
	}
	stored := reg
	stored.APIKey = sealed
	data, err := json.Marshal(stored)
	if err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, `INSERT INTO backend_registrations (id, updated_at, registration_json)
		VALUES (?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at, registration_json = excluded.registration_json`,
		reg.ID,
		time.Now().UTC().Format(time.RFC3339Nano),
		string(data),
	)
	return err
}

// DeleteBackendRegistration removes a backend's persisted row so a deleted
// backend stays deleted across restarts — otherwise LoadBackendRegistrations
// resurrects it on the next startup.
func (s *LogStore) DeleteBackendRegistration(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM backend_registrations WHERE id = ?`, id)
	return err
}

func (s *LogStore) LoadBackendRegistrations(ctx context.Context) ([]BackendRegistration, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT registration_json FROM backend_registrations ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []BackendRegistration{}
	for rows.Next() {
		var raw string
		if err := rows.Scan(&raw); err != nil {
			return nil, err
		}
		var reg BackendRegistration
		if err := json.Unmarshal([]byte(raw), &reg); err != nil {
			return nil, err
		}
		if reg.APIKey != "" {
			plain, err := s.box.open(reg.APIKey)
			if err != nil {
				// A key we can't decrypt (e.g. ROUTER_PERSIST_SECRET changed) must
				// not block startup; drop it and let the worker re-register to
				// restore it. Log loudly so the cause is visible.
				log.Printf("decrypt persisted api key for backend %q failed: %v (backend will need re-registration to restore its key)", reg.ID, err)
				reg.APIKey = ""
			} else {
				reg.APIKey = plain
			}
		}
		out = append(out, reg)
	}
	return out, rows.Err()
}

func (r *Router) logRetentionLoop() {
	r.pruneLogs()
	ticker := time.NewTicker(24 * time.Hour)
	defer ticker.Stop()
	for range ticker.C {
		r.pruneLogs()
	}
}

func (r *Router) pruneLogs() {
	if r.cfg.LogRetention <= 0 {
		return
	}
	deleted, err := r.logs.DeleteOlderThan(context.Background(), time.Now().Add(-r.cfg.LogRetention))
	if err != nil {
		log.Printf("request log retention cleanup failed: %v", err)
		return
	}
	if deleted > 0 {
		log.Printf("deleted %d expired request logs", deleted)
	}
}

// authorizedAsWorker checks the request's Bearer token against the single
// ROUTER_WORKER_TOKEN. Used for worker self-registration endpoints. Empty
// configured token disables the check.
func authorizedAsWorker(req *http.Request, token string) bool {
	if token == "" {
		return true
	}
	got := req.Header.Get("Authorization")
	return subtle.ConstantTimeCompare([]byte(got), []byte("Bearer "+token)) == 1
}

// authorizedAsClient checks the request's Bearer token against the any-of
// list ROUTER_CLIENT_TOKENS. Used for /v1/*, read-only registry endpoints,
// and debug endpoints. Empty configured list disables the check.
func authorizedAsClient(req *http.Request, tokens []string) bool {
	if len(tokens) == 0 {
		return true
	}
	got := []byte(req.Header.Get("Authorization"))
	// Check every token without short-circuiting so the comparison work — and
	// thus the timing — doesn't depend on which token (if any) matched.
	authorized := false
	for _, t := range tokens {
		if subtle.ConstantTimeCompare(got, []byte("Bearer "+t)) == 1 {
			authorized = true
		}
	}
	return authorized
}

// maxRequestBytes caps inbound POST bodies. Chat requests with base64 images
// run to a few MB; nothing legitimate approaches this bound — without it a
// single multi-GB POST OOMs the router (and with no client tokens configured,
// auth doesn't gate that).
const maxRequestBytes = 64 << 20

func readRequestBody(w http.ResponseWriter, req *http.Request) ([]byte, error) {
	req.Body = http.MaxBytesReader(w, req.Body, maxRequestBytes)
	return io.ReadAll(req.Body)
}

func copyHeaders(dst, src http.Header) {
	for key, values := range src {
		lower := strings.ToLower(key)
		if lower == "content-length" || lower == "connection" {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func upstreamChatURL(backend *Backend) string {
	return upstreamPathURL(backend, "/v1/chat/completions")
}

// upstreamPathURL joins a registered backend's base URL to an OpenAI-style
// API path. If the backend already registered with a `/v1` suffix on its
// URL, the duplicate `/v1` is collapsed so we end up with one canonical
// path. `openAIPath` must start with `/v1/`.
func upstreamPathURL(backend *Backend, openAIPath string) string {
	base := strings.TrimRight(backend.URL, "/")
	if strings.HasSuffix(base, "/v1") {
		return base + strings.TrimPrefix(openAIPath, "/v1")
	}
	return base + openAIPath
}

func backendRootURL(backend *Backend) string {
	base := strings.TrimRight(backend.URL, "/")
	return strings.TrimSuffix(base, "/v1")
}

func publicBackends(backends []*Backend) []*Backend {
	out := make([]*Backend, 0, len(backends))
	for _, b := range backends {
		cp := cloneBackend(b)
		cp.APIKey = ""
		if isExpired(cp) {
			cp.Status = "expired"
		}
		out = append(out, cp)
	}
	return out
}

// readSSEStream drains a streamed completion, accumulating its content and
// reasoning text separately (both dialects' reasoning field names — see
// extract.go).
//
// firstChunk fires on EVERY non-empty delta of EITHER kind, and the caller
// latches the first (see speedProbe). Either kind, because for a thinking model
// the reasoning tokens ARE the first generated output, and stamping TTFT only on
// content would fold the whole reasoning phase into first-token latency.
//
// tokens prefers the stream's usage.completion_tokens (sent when the request
// asked for stream_options.include_usage), falling back to counting non-empty
// deltas.
//
// The fallback is a COUNT only for engines that emit one token per delta.
// Speculative decoding breaks that: vLLM with MTP packs each accepted
// multi-token step into ONE delta (measured on Qwen3.8-27B 2026-08-15: 52
// completion tokens arrived in 25 deltas), so delta-counting under-read the
// fleet's MTP workers ~2.5x — certified 37 tok/s against a benched 92. That is
// why the probe now requests include_usage and this prefers it.
func readSSEStream(body io.Reader, firstChunk func()) (content, reasoning string, tokens int, err error) {
	var out, think strings.Builder
	usageTokens := 0
	scanner := newLargeScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimSpace(line[6:])
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta sseDelta `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		// Cumulative in both dialects (vLLM final-chunk usage and llama.cpp
		// per-chunk usage) — the last value seen wins either way.
		if chunk.Usage != nil && chunk.Usage.CompletionTokens > 0 {
			usageTokens = chunk.Usage.CompletionTokens
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta
		rt := delta.reasoningText()
		if delta.Content != "" || rt != "" {
			tokens++
		}
		if delta.Content != "" {
			firstChunk()
			out.WriteString(delta.Content)
		}
		if rt != "" {
			firstChunk()
			think.WriteString(rt)
		}
	}
	if usageTokens > 0 {
		tokens = usageTokens
	}
	return out.String(), think.String(), tokens, scanner.Err()
}

func newLargeScanner(r io.Reader) *bufio.Scanner {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	return scanner
}

func truncate(value string, max int) string {
	value = strings.TrimSpace(value)
	if len(value) <= max {
		return value
	}
	return value[:max] + "..."
}

func truncateJSON(value any, max int) string {
	data, _ := json.Marshal(value)
	return truncate(string(data), max)
}

func mapBool(ok bool, yes string, no string) string {
	if ok {
		return yes
	}
	return no
}

// countSSETokens counts actual generated tokens from a captured SSE stream.
// Each "data: {...}" line with a non-empty content OR reasoning delta represents
// one token — see the loop body for why reasoning has to count too.
func countSSETokens(data []byte) int {
	tokens := 0
	for _, line := range bytes.Split(data, []byte("\n")) {
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		payload := line[6:]
		if bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		// Count a chunk carrying a non-empty content OR reasoning delta. Reasoning
		// tokens must count toward throughput: a model that spends most of a turn
		// emitting a long reasoning block before a short answer would otherwise look
		// an order of magnitude slower than it is, poisoning latency-aware routing.
		// vLLM ≥0.23 emits "reasoning" instead of "reasoning_content".
		if hasNonEmptyDelta(payload, `"content":"`) || hasNonEmptyDelta(payload, `"reasoning_content":"`) || hasNonEmptyDelta(payload, `"reasoning":"`) {
			tokens++
		}
	}
	return tokens
}

// hasNonEmptyDelta reports whether payload contains key immediately followed by a
// non-empty JSON string value (the char after the opening quote isn't the closing
// quote).
func hasNonEmptyDelta(payload []byte, key string) bool {
	idx := bytes.Index(payload, []byte(key))
	if idx < 0 {
		return false
	}
	after := idx + len(key)
	return after < len(payload) && payload[after] != '"'
}

// copyStreaming relays an SSE response to the client (and into capture),
// returning the timestamp of the first chunk — the worker's first token, ≈ TTFT.
// A zero time means nothing was ever read.
func copyStreaming(w http.ResponseWriter, src io.Reader, capture io.Writer, progress func()) (time.Time, error) {
	buf := make([]byte, 32*1024)
	flusher, _ := w.(http.Flusher)
	var firstByte time.Time
	for {
		n, readErr := src.Read(buf)
		if n > 0 {
			if progress != nil {
				progress()
			}
			if firstByte.IsZero() {
				firstByte = time.Now()
			}
			if capture != nil {
				if _, captureErr := capture.Write(buf[:n]); captureErr != nil {
					return firstByte, captureErr
				}
			}
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				return firstByte, writeErr
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				return firstByte, nil
			}
			return firstByte, readErr
		}
	}
}

// writeJSON is the single northbound response writer. A validationError is
// rendered into the OpenAI envelope here rather than at its ~20 construction
// sites, which is what lets those sites stay as short as
// `validationError{Message: err.Error()}` while the status code still picks the
// right error `type`.
func writeJSON(w http.ResponseWriter, status int, value any) {
	if ve, ok := value.(validationError); ok {
		value = ve.envelope(status)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		log.Printf("write json failed: %v", err)
	}
}

func methodNotAllowed(w http.ResponseWriter) {
	writeError(w, http.StatusMethodNotAllowed, "method not allowed")
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}

// dashboardTemplate renders the operator dashboard. The markup lives in
// dashboard.html rather than in a 395-line raw string literal here: as a real
// .html file it gets syntax highlighting and formatting, and main.go loses a
// tenth of its length to markup that has nothing to do with routing.
//
// NOTE: the Dockerfile must COPY dashboard.html — it previously copied only
// *.go, so an embed added without that change fails the build.
//
//go:embed dashboard.html
var dashboardHTML string

var dashboardTemplate = template.Must(template.New("dashboard").Parse(dashboardHTML))

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if value := os.Getenv(key); value != "" {
		parsed, err := strconv.Atoi(value)
		if err == nil {
			return parsed
		}
	}
	return fallback
}

// envDuration reads an integer-seconds env var and converts to Duration.
// Falls back to “fallback“ if the var is unset or unparseable.
func envDuration(key string, fallback time.Duration) time.Duration {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return fallback
	}
	return time.Duration(n) * time.Second
}

func envBoundedInt(value string, fallback int, min int, max int) int {
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	if parsed < min {
		return min
	}
	if parsed > max {
		return max
	}
	return parsed
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

// thinkingMode is what a request's thinking state was recorded as.
//
// The ZERO value is "unknown", and that is the whole point of the type. A
// RequestLog built without setting the field — a route added later, a path
// nobody thought about — must not silently claim a measurement. Recording
// unknown as "off" would pool two regimes whose output lengths differ by more
// than an order of magnitude, and the resulting timing model would describe
// neither.
type thinkingMode int

const (
	thinkingUnknown thinkingMode = iota
	thinkingOff
	thinkingOn
)

// thinkingLogValue flattens a thinking resolution to what the log stores.
//
// noThink is the field that means "this request WILL be served with thinking
// disabled" — the same one selection uses to pick between a worker's two
// quality scores. Reading `enable` instead would be wrong: it is only
// meaningful when `patch` is set, so a request that inherits the worker's
// default would be recorded as thinking-off regardless of what the worker
// actually did.
func thinkingLogValue(tr thinkingResolution) thinkingMode {
	if tr.noThink {
		return thinkingOff
	}
	if tr.patch && tr.enable {
		return thinkingOn
	}
	if tr.hardThink || tr.softThink {
		return thinkingOn
	}
	return thinkingUnknown
}

// ewma folds a sample into an exponentially weighted average in place, seeding
// it on the first observation so a cold metric does not spend its early life
// being dragged up from zero.
func ewma(slot *float64, sample, alpha float64) {
	if *slot == 0 {
		*slot = sample
		return
	}
	*slot = *slot*(1-alpha) + sample*alpha
}

// dropLegacyObservations removes the pre-model-hash observation tables, once.
//
// The old table was keyed (qid, backend_id, thinking, source) and its qids were an
// FNV of prompt+expect carrying no match mode and no grader version. Every row in
// it is unreachable under the content-addressed identity (see identity.go), so it
// is dropped rather than migrated: carrying a second, dead key format forever to
// preserve rows nothing can look up is worse than re-earning them.
//
// The migration is CONDITIONAL, and that is the whole point of it being a function
// rather than two statements in init()'s list. That list runs on every startup, so
// an unconditional DROP is not a migration at all — it is a boot-time wipe. It
// emptied the permacache on every restart, and took the judged half with it: those
// rows have no second copy, and the vectors beside them are persisted precisely
// because a judged question's text is never stored and its vector cannot be
// re-derived.
//
// "Already migrated" is read off the schema itself rather than a version counter:
// the new table has a model_hash column and the old one does not. A table that is
// absent entirely is a fresh database and needs nothing.
func (s *LogStore) dropLegacyObservations(ctx context.Context) error {
	rows, err := s.db.QueryContext(ctx, `PRAGMA table_info(observations)`)
	if err != nil {
		return err
	}
	defer rows.Close()
	columns, hasModelHash := 0, false
	for rows.Next() {
		var cid int
		var name, typ string
		var notNull int
		var dflt sql.NullString
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notNull, &dflt, &pk); err != nil {
			return err
		}
		columns++
		if name == "model_hash" {
			hasModelHash = true
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if columns == 0 || hasModelHash {
		return nil // fresh database, or already on the current schema
	}
	log.Printf("outcome matrix: dropping the pre-model-hash observation tables (one-off migration)")
	for _, stmt := range []string{
		`DROP TABLE IF EXISTS observations`,
		// The vectors go with them: observation_vectors is keyed by qid, and every
		// judged qid moved to the same hash the bank now uses.
		`DROP TABLE IF EXISTS observation_vectors`,
	} {
		if _, err := s.db.ExecContext(ctx, stmt); err != nil {
			return err
		}
	}
	return nil
}
