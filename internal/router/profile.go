package router

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"path"
	"regexp"
	"strings"
	"time"
)

// WorkerProfile is everything the router measures about a worker at cold start,
// so the worker itself declares almost nothing (just id/url/api_key). It's
// persisted per (id, model) and reused on warm restarts — see the /llm skill and
// llm-router/README.md.
type WorkerProfile struct {
	Model          string  `json:"model"`
	Quality        int     `json:"quality"`
	ContextK       int     `json:"context_k"`
	MaxConcurrency int     `json:"max_concurrency"`
	BaselineTPS    float64 `json:"baseline_tps"`
	TTFTMillis     int64   `json:"ttft_millis"`
	// PrefillTPS is prompt tokens per second, measured on a prompt of known length.
	// Unlike TTFTMillis it scales with the request, which is what routing needs: the
	// spread across the fleet is far wider for prefill than for decode (0.67s vs 37.2s
	// on the same ~4k prompt), so a flat TTFT average misprices long prompts badly.
	PrefillTPS float64 `json:"prefill_tps,omitempty"`
	// SpeedVersion is the version of the DECODE measurement BaselineTPS was taken
	// with. Separate from BenchVersion because the two have wildly different
	// re-measurement costs: correcting how decode is counted is one 64-token
	// generation per worker, while a BenchVersion bump re-runs the whole quality
	// suite on every worker in the fleet and parks each at a provisional quality
	// until it finishes. A speed-probe correction should not cost that.
	SpeedVersion int      `json:"speed_version,omitempty"`
	Features     []string `json:"features"`
	Thinking     bool     `json:"thinking"`
	// ThinkingDialect is the spelling of the thinking gate this endpoint was
	// MEASURED to honour (one of the thinkingDialect* constants).
	//
	// Empty means UNKNOWN, not unsupported: a profile cached before the dialect
	// was probed has the zero value, and so does a worker whose probe never got a
	// verdict. Unknown falls back to chat_template_kwargs, the spelling the whole
	// fleet has always spoken — a zero value that meant "cannot think" would
	// silently switch thinking off across the fleet on the first warm restart.
	ThinkingDialect string           `json:"thinking_dialect,omitempty"`
	BenchVersion    int              `json:"bench_version"`            // question-set version this quality was measured against
	QualityDetail   string           `json:"quality_detail,omitempty"` // per-tier + truncation breakdown
	Failed          []string         `json:"failed,omitempty"`         // labels of benchmark questions the worker missed
	BenchResults    []BenchResult    `json:"bench_results,omitempty"`  // full per-question Q&A from the most recent run
	Checks          map[string]Check `json:"checks,omitempty"`
	MeasuredAt      time.Time        `json:"measured_at"`
	ProfileMillis   int64            `json:"profile_ms,omitempty"` // wall time of the full cold-start profile (capacity ramp + quality benchmark)
	// Incomplete marks a profile where one or more capability probes never got a
	// verdict (transient errors exhausted their retries). The worker stays
	// routable on these values, but the profile must NOT be persisted — a cached
	// "not detected" from a probe blip would misroute traffic until the next
	// benchmarkVersion bump (the 2026-07-06 thinking incident's stickiness half).
	Incomplete bool `json:"-"`
}

// provisionalQuality is the conservative tier a worker gets between the quick
// profile and the background quality benchmark — low enough (on the 0–100 quality
// percentage scale) that an unproven worker only draws easy traffic until it earns more.
const provisionalQuality = 30

// A capacity-probe level is only believed to be the ceiling after this many
// consecutive failures, spaced by the delay below. Cheap insurance: the measured
// value is cached per (id, model) and never re-measured on its own, so a single
// false negative here permanently under-rates a worker.
const (
	capacityProbeAttempts   = 3
	capacityProbeRetryDelay = 3 * time.Second
)

// fmtProfileDuration renders a profile's wall time as "6m 48s" (minutes + seconds).
func fmtProfileDuration(ms int64) string {
	d := (time.Duration(ms) * time.Millisecond).Round(time.Second)
	return fmt.Sprintf("%dm %02ds", int(d/time.Minute), int((d%time.Minute)/time.Second))
}

// profileQuick runs the FAST half of profiling — capability discovery, speed, and
// context — so a brand-new worker becomes routable in seconds. Quality and
// capacity stay provisional (the declared seed or a conservative default) until
// profileBackend measures them in the background.
func (r *Router) profileQuick(b *Backend, model string) (*WorkerProfile, error) {
	checks := map[string]Check{}
	if err := r.chatProbe(b); err != nil {
		return nil, fmt.Errorf("chat probe failed: %w", err)
	}
	checks["chat"] = Check{OK: true}
	features := []string{"chat"}

	incomplete := false
	if err, inconclusive := r.capabilityProbe(func() error { return r.jsonProbe(b) }); err == nil {
		features = append(features, "json")
		checks["json"] = Check{OK: true}
	} else if inconclusive {
		incomplete = true
		checks["json"] = Check{Message: "inconclusive (probe errored, will re-probe): " + err.Error()}
	} else {
		checks["json"] = Check{Message: err.Error()}
	}
	if err, inconclusive := r.capabilityProbe(func() error { return r.toolProbe(b) }); err == nil {
		features = append(features, "tools")
		checks["tools"] = Check{OK: true}
	} else if inconclusive {
		incomplete = true
		checks["tools"] = Check{Message: "inconclusive (probe errored, will re-probe): " + err.Error()}
	} else {
		checks["tools"] = Check{Message: err.Error()}
	}
	thinking, dialect, thinkInconclusive := r.thinkingProbe(b)
	switch {
	case thinkInconclusive:
		incomplete = true
		checks["thinking"] = Check{OK: false, Message: "inconclusive (probe errored, will re-probe)"}
	case thinking:
		checks["thinking"] = Check{OK: true, Message: "supported via " + dialect}
	default:
		checks["thinking"] = Check{OK: false, Message: "not detected"}
	}

	// Vision is detected empirically — send an image, see if the worker reads it —
	// rather than trusted from a declaration (see visionProbe).
	vision, visionInconclusive := r.visionProbe(b)
	if visionInconclusive {
		incomplete = true
		checks["vision"] = Check{OK: false, Message: "inconclusive (probe errored, will re-probe)"}
	} else {
		checks["vision"] = Check{OK: vision, Message: mapBool(vision, "detected", "not detected")}
	}
	if vision {
		features = appendUnique(features, "vision")
	}

	// Preserve declared semantic tags the router can't fully settle by probing.
	for _, tag := range b.Features {
		switch tag {
		case "uncensored": // pure policy — unprobeable, so trust the declaration
			features = appendUnique(features, tag)
		case "vision": // probed above; keep a declared tag as a robustness backstop
			features = appendUnique(features, tag)
			if !vision {
				log.Printf("worker %s declares vision but the probe couldn't confirm it — keeping the declared tag; verify the model is multimodal", b.ID)
			}
		}
	}

	tps, ttft, err := r.speedProbe(b)
	if err != nil {
		return nil, fmt.Errorf("speed probe failed: %w", err)
	}
	checks["speed"] = Check{OK: true, Message: fmt.Sprintf("%.1f tok/s, ttft %dms", tps, ttft)}

	quality := b.Quality // declared seed, if any
	if quality <= 0 {
		quality = provisionalQuality
	}
	return &WorkerProfile{
		Model: model, Quality: quality, ContextK: r.queryContext(b), MaxConcurrency: 1,
		BaselineTPS: tps, TTFTMillis: ttft, SpeedVersion: speedProbeVersion,
		Features: features, Thinking: thinking, ThinkingDialect: dialect,
		Checks: checks, MeasuredAt: time.Now(), Incomplete: incomplete,
	}, nil
}

// profileBackend is the FULL cold-start measurement: profileQuick plus the slow
// capacity ramp and quality benchmark. Run in the background after the worker is
// provisionally ready; it refines the live values and is persisted for warm
// restarts. Returns an error only if the worker can't serve chat at all.
func (r *Router) profileBackend(b *Backend, model string) (*WorkerProfile, error) {
	start := time.Now()
	p, err := r.profileQuick(b, model)
	if err != nil {
		return nil, err
	}
	capN, capOK := r.capacityProbe(b)
	// Abort BEFORE the benchmark when the capacity probe already failed: a hung
	// worker otherwise burns 28 questions × attempts × the request timeout
	// (many hours) holding the profiling guard, only for the results to be
	// discarded here anyway. The worker keeps its provisional profile and the
	// caller schedules a retry.
	if !capOK {
		return nil, errors.New("worker unreachable during capacity probe; discarding partial measurements")
	}
	// The ramp infers capacity from aggregate throughput, which can over-report on a
	// worker that serialises: at n=2 a single-slot worker runs the two requests
	// back-to-back, and if the samples are noisy the second level still clears the
	// 1.15x knee test. Where the worker publishes its own slot count, that is ground
	// truth and outranks the inference.
	if slots := r.querySlots(b); slots > 0 && capN > slots {
		log.Printf("capacity probe: %s ramp suggested %d but the worker reports %d slot(s) — using %d",
			b.ID, capN, slots, slots)
		capN = slots
	}
	// Prefill rate is measured here rather than left to the live EWMA, which only
	// samples non-thinking requests and so never fills in for a thinking-heavy worker.
	// A failure is not fatal: routing falls back to the flat TTFT average as before.
	if rate, err := r.prefillProbe(b); err != nil {
		log.Printf("prefill probe failed for %s: %v — routing will price its TTFT from the flat average", b.ID, err)
	} else {
		p.PrefillTPS = rate
		p.Checks["prefill"] = Check{OK: true, Message: fmt.Sprintf("%.0f tok/s on a %d-token prompt", rate, prefillProbeTokens)}
	}
	// Deliberately BEFORE benchConc: the quality benchmark grades at this concurrency,
	// so an over-reported capacity makes half the questions queue behind a full
	// generation and blow benchAnswerDeadline. On llm-naples-deepseek-284B-q4 (probe
	// said 2, --parallel 1) that scored 46 questions "(too slow)" with not one wrong
	// answer, dragging a capable model down to quality 53.
	benchConc := capN
	if benchConc > 4 {
		benchConc = 4 // cap concurrency so profiling leaves headroom for live traffic
	}
	quality, qOK, qBreakdown, qFailed, qResults := r.runQualityBenchmark(b, benchConc)
	r.auditThinkingGate() // log where the reasoning gate disagrees with tier (model-independent, once)
	// If the worker went unreachable mid-benchmark, the quality number is
	// garbage — abort rather than persist an under-rating.
	if !qOK {
		return nil, errors.New("worker unreachable during quality benchmark; discarding partial measurements")
	}
	p.MaxConcurrency = capN
	p.Checks["capacity"] = Check{OK: true, Message: fmt.Sprintf("%d concurrent", capN)}
	p.Quality = quality
	p.BenchVersion = benchmarkVersion
	p.QualityDetail = qBreakdown
	p.Failed = qFailed
	p.BenchResults = qResults
	qMsg := fmt.Sprintf("%d%%", quality)
	if qBreakdown != "" {
		qMsg += " " + qBreakdown // per-tier + truncation breakdown, visible via /backends/{id}
	}
	p.Checks["quality"] = Check{OK: true, Message: qMsg}
	p.MeasuredAt = time.Now()
	p.ProfileMillis = time.Since(start).Milliseconds()
	return p, nil
}

func appendUnique(s []string, v string) []string {
	for _, x := range s {
		if x == v {
			return s
		}
	}
	return append(s, v)
}

// chatProbe verifies the worker serves chat completions at all.
func (r *Router) chatProbe(b *Backend) error {
	content, err := r.simpleCompletion(b, map[string]any{
		"model": probeModel(b), "stream": false, "max_tokens": 16,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
		"messages":             []map[string]string{{"role": "user", "content": "Reply with the single word: ready /no_think"}},
	})
	if err != nil {
		return err
	}
	if strings.TrimSpace(content) == "" {
		return errors.New("empty chat response")
	}
	return nil
}

// thinkingDialect names the spelling of the "think about this" gate an endpoint
// honours. Measured rather than assumed: enable_thinking inside
// chat_template_kwargs is a vLLM and llama.cpp extension, not part of the OpenAI
// API, so a provider either ignores it — thinking silently never happens, and
// the router believes it asked — or rejects the unknown field and fails the
// whole request, while reading reasoning_effort perfectly well.
const (
	thinkingDialectKwargs = "chat_template_kwargs" // enable_thinking nested in chat_template_kwargs
	thinkingDialectEffort = "reasoning_effort"     // the OpenAI-standard top-level field
	thinkingDialectNone   = "none"                 // neither gate produced a reasoning block
)

// thinkingProbeQuestion is genuinely multi-step on purpose — a trivial one lets
// an adaptive reasoning model skip its thinking block and false-negative the
// probe.
const thinkingProbeQuestion = "A train goes 60 km in 1.5 hours, then 90 km in 1 hour. What is its average speed for the whole trip in km/h? Show your reasoning step by step."

// thinkingProbe reports whether the worker emits a reasoning block when asked
// to, and WHICH spelling of the ask it honours. inconclusive=true means the
// probe never got a verdict (transient errors exhausted the retries) — the
// caller must not persist that as a durable "not detected" (see the Incomplete
// handling in profileQuick).
//
// The dialects are tried in fleet order: chat_template_kwargs first, because
// every local vLLM and llama.cpp worker speaks it and answers on the first
// request, then the standard reasoning_effort. Only a POSITIVE result settles
// it, so an endpoint that accepts the kwargs object and quietly discards it
// still gets its standard field tried before the router writes it off.
func (r *Router) thinkingProbe(b *Backend) (thinking bool, dialect string, inconclusive bool) {
	gates := []struct {
		dialect string
		field   string
		value   any
	}{
		{thinkingDialectKwargs, "chat_template_kwargs", map[string]bool{"enable_thinking": true}},
		// "high" rather than a milder level: the probe asks whether the field does
		// anything at all here, and the strongest ask is the least likely to be
		// answered without a reasoning block.
		{thinkingDialectEffort, "reasoning_effort", "high"},
	}
	unknown := false
	for _, g := range gates {
		payload := map[string]any{
			"model": probeModel(b), "stream": false, "max_tokens": 1024,
			"messages": []map[string]string{{"role": "user", "content": thinkingProbeQuestion}},
			g.field:    g.value,
		}
		thought, settled := r.thinkingProbeOnce(b, payload)
		if !settled {
			unknown = true
			continue
		}
		if thought {
			return true, g.dialect, false
		}
	}
	if unknown {
		return false, "", true
	}
	return false, thinkingDialectNone, false
}

// thinkingProbeOnce runs one dialect's probe under the same transient-retry
// discipline as capabilityProbe. settled=false means no verdict at all: every
// attempt hit a transient error.
func (r *Router) thinkingProbeOnce(b *Backend, payload map[string]any) (thinking, settled bool) {
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			time.Sleep(2 * time.Second)
		}
		raw, err := r.rawCompletion(b, payload)
		if err != nil {
			// A definitive 4xx settles THIS dialect — a strict endpoint refusing a
			// field it does not recognise — and says nothing about the other one.
			if isClientReject(err) {
				return false, true
			}
			continue // transient (load shedding, 5xx, transport) → retry
		}
		content, reasoning, _ := completionText(raw)
		return strings.TrimSpace(reasoning) != "" || strings.Contains(content, "<think>"), true
	}
	return false, false
}

// capabilityProbe runs an error-returning probe (json/tools) with the same
// transient-retry discipline as thinkingProbe/visionProbe: up to 3 attempts, a
// definitive 4xx reject or success short-circuits. inconclusive=true means
// retries were exhausted on transient errors only — the capability is UNKNOWN,
// not absent, and the profile must not be cached as if it were measured.
func (r *Router) capabilityProbe(probe func() error) (err error, inconclusive bool) {
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			time.Sleep(2 * time.Second)
		}
		err = probe()
		if err == nil || isClientReject(err) {
			return err, false
		}
	}
	return err, true
}

// capacityProbe estimates how many concurrent requests the worker handles before
// aggregate throughput stops scaling (the saturation knee) or it starts failing.
func (r *Router) capacityProbe(b *Backend) (capacity int, ok bool) {
	maxN := r.cfg.CapacityProbeMax
	if maxN < 1 {
		maxN = 16
	}
	best, prev := 1, 0.0
	for _, n := range []int{1, 2, 4, 8, 16, 32, 64} {
		if n > maxN {
			break
		}
		// Retry a failed level before believing it. measureConcurrent fails the
		// whole level if ANY of its n requests errors, so a single transient blip
		// — a worker still capturing CUDA graphs seconds after it first answers,
		// a momentary 503 — otherwise becomes a permanent verdict: the ramp
		// breaks, capacity sticks at the last good level, and the value is cached
		// per (id, model) so it never re-measures. That is exactly how
		// llm-6000pro (--max-num-seqs 6, ~4x aggregate throughput at n=6) ended
		// up pinned to serial dispatch, funnelling fleet traffic onto slower
		// backends until an operator noticed.
		var agg float64
		var good bool
		for attempt := 0; attempt < capacityProbeAttempts; attempt++ {
			if attempt > 0 {
				time.Sleep(capacityProbeRetryDelay)
				log.Printf("capacity probe: retrying n=%d for %s (attempt %d/%d)", n, b.ID, attempt+1, capacityProbeAttempts)
			}
			if agg, good = r.measureConcurrent(b, n); good {
				break
			}
		}
		if !good {
			if n == 1 {
				return 1, false // can't serve even one request → worker unreachable
			}
			log.Printf("capacity probe: %s failed n=%d after %d attempts; capacity=%d", b.ID, n, capacityProbeAttempts, best)
			break // repeatable errors at a higher level → that's the capacity ceiling
		}
		if n > 1 && agg < prev*1.15 {
			break // throughput plateaued → saturation knee
		}
		best, prev = n, agg
	}
	return best, true
}

// measureConcurrent fires n identical short completions at once and returns the
// aggregate throughput (approx tokens/sec). ok is false if any request failed.
func (r *Router) measureConcurrent(b *Backend, n int) (float64, bool) {
	payload := map[string]any{
		"model": probeModel(b), "stream": false, "max_tokens": 64,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
		"messages":             []map[string]string{{"role": "user", "content": "Write two sentences about the ocean. /no_think"}},
	}
	type res struct {
		toks int
		err  error
	}
	ch := make(chan res, n)
	start := time.Now()
	for i := 0; i < n; i++ {
		go func() {
			content, err := r.simpleCompletion(b, payload)
			ch <- res{len(strings.Fields(content)), err}
		}()
	}
	total := 0
	for i := 0; i < n; i++ {
		x := <-ch
		if x.err != nil {
			return 0, false
		}
		total += x.toks
	}
	elapsed := time.Since(start).Seconds()
	if elapsed <= 0 {
		return 0, false
	}
	return float64(total) / elapsed, true
}

// ModelMeta is what a worker's runtime reports about the weights it actually
// loaded, as distinct from the model *name*, which is only ever a string an
// operator chose. The two disagree routinely: an unaliased llama.cpp worker
// advertises its gguf path, one started with --alias advertises that alias, and
// neither has to describe what is in the file (llm-6000pro-deepseek-284B-q8
// serves MXFP4 weights). Zero fields mean the runtime doesn't publish that
// datum — vLLM exposes no parameter count — so consumers must degrade rather
// than render zeroes. Embedded in Backend, so these serialise flat into
// /backends alongside "model".
type ModelMeta struct {
	ModelPath      string `json:"model_path,omitempty"`       // weights file the worker loaded
	ModelParams    int64  `json:"model_params,omitempty"`     // parameter count read off the weights
	ModelQuant     string `json:"model_quant,omitempty"`      // ftype, e.g. "Q4_K - Medium", "MXFP4 MoE"
	ModelSizeBytes int64  `json:"model_size_bytes,omitempty"` // size of the loaded weights
	ModelCtxTrain  int    `json:"model_ctx_train,omitempty"`  // context length the model was trained at
	// The id the worker's own /v1/models advertises, "default" included — the
	// one spelling the runtime is guaranteed to accept in a request's "model"
	// field. llama.cpp ignores that field, but vLLM 404s anything else, so
	// forwarded bodies are rewritten to this (see patchForwardedBody).
	ServedID string `json:"served_model_id,omitempty"`
}

// probeModel is the model name a probe puts in its request. Every probe used to
// send the literal "default": harmless against a single-model vLLM or llama.cpp
// worker, which either publishes that id or ignores the field, and fatal against
// an endpoint that validates model names — a strict provider failed
// certification for reasons that had nothing to do with its capabilities.
//
// Precedence follows how much the router actually knows about the endpoint:
//
//  1. ServedID, what the endpoint's own /v1/models advertises, and therefore the
//     one spelling its validator is guaranteed to accept.
//  2. The registration's declared model, which is all there is on the very first
//     probe against a brand-new registration — queryModelInfo has not run yet,
//     or the endpoint publishes no catalogue to read.
//  3. "default", the historical literal, when even that is empty.
func probeModel(b *Backend) string {
	if b == nil {
		return "default"
	}
	if b.ServedID != "" {
		return b.ServedID
	}
	if b.Model != "" {
		return b.Model
	}
	return "default"
}

// queryModel asks the worker its served model id (the profile fingerprint).
func (r *Router) queryModel(b *Backend) string {
	id, _ := r.queryModelInfo(b)
	return id
}

// shardSuffix matches the multi-part GGUF shard counter ("-00001-of-00005")
// that a sharded model's first file carries in its name.
var shardSuffix = regexp.MustCompile(`-\d{5}-of-\d{5}$`)

// niceModelName is the human-readable name of what a backend serves, for
// display surfaces only (response headers, dashboards) — the profile
// fingerprint stays Backend.Model untouched, or renaming would re-profile the
// fleet. The advertised model id is used unless it is just the worker id
// echoed back (an --alias masking the real name), in which case the weights
// path recovers it; either way a file path reduces to its basename with the
// .gguf extension and any shard counter stripped, so
// "/models/UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf"
// displays as "DeepSeek-V4-Flash-0731-UD-Q4_K_XL".
func niceModelName(b *Backend) string {
	name := b.Model
	if (name == "" || name == "default" || name == b.ID) && b.ModelPath != "" {
		name = b.ModelPath
	}
	name = path.Base(name)
	name = strings.TrimSuffix(name, ".gguf")
	name = shardSuffix.ReplaceAllString(name, "")
	if name == "" || name == "." || name == "/" {
		return b.Model
	}
	return name
}

// queryModelInfo asks a worker what it is serving: the advertised model id (the
// profile cache fingerprint) and whatever hard metadata the runtime publishes
// about the weights behind that name. llama.cpp nests the metadata under the
// same /v1/models entry the id comes from, so the id costs nothing extra;
// /props adds the weights path, which is the only way to recover a real model
// name from a worker whose --alias masks it.
//
// Which entry that is comes from modelEntryFor, not from data[0] — see there.
func (r *Router) queryModelInfo(b *Backend) (string, ModelMeta) {
	var meta ModelMeta
	id := ""
	if raw, err := r.backendGET(b, "/v1/models"); err == nil {
		if m, ok := modelEntryFor(raw, b.Model); ok {
			if v, _ := m["id"].(string); v != "" {
				meta.ServedID = v
				if v != "default" {
					id = v
				}
			}
			if mm, ok := m["meta"].(map[string]any); ok {
				meta.ModelParams = int64(jsonNum(mm, "n_params"))
				meta.ModelSizeBytes = int64(jsonNum(mm, "size"))
				meta.ModelCtxTrain = int(jsonNum(mm, "n_ctx_train"))
				meta.ModelQuant, _ = mm["ftype"].(string)
			}
		}
	}
	// llama.cpp only (vLLM has no /props). Recent builds put the weights path at
	// the top level, older ones under default_generation_settings.model.
	if raw, err := r.backendGET(b, "/props"); err == nil {
		if p, _ := raw["model_path"].(string); p != "" {
			meta.ModelPath = p
		} else if dg, ok := raw["default_generation_settings"].(map[string]any); ok {
			meta.ModelPath, _ = dg["model"].(string)
		}
	}
	switch {
	case id != "":
	case b.Model != "" && b.Model != "default":
		id = b.Model
	default:
		id = b.ID
	}
	return id, meta
}

// modelEntryFor picks the entry of an OpenAI-shaped /v1/models response that
// describes what this backend serves.
//
// It used to be data[0], unconditionally. That is right for the single-model
// worker the router grew up with, and arbitrary for a catalogue endpoint serving
// hundreds — and the id it yields is stamped onto every forwarded request (see
// patchForwardedBody), so an arbitrary pick means every client request is
// rewritten to a model nobody asked for.
//
// So: the declared model wins when the endpoint's list actually contains it,
// because that is the row the registration is about. Otherwise only a
// single-entry list is unambiguous. A catalogue with no declared match yields
// nothing, which leaves the client's own model name to travel through untouched
// — the correct answer for an endpoint that serves whatever it is asked for.
func modelEntryFor(raw map[string]any, declared string) (map[string]any, bool) {
	data, ok := raw["data"].([]any)
	if !ok || len(data) == 0 {
		return nil, false
	}
	if declared != "" {
		for _, item := range data {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if id, _ := m["id"].(string); id == declared {
				return m, true
			}
		}
	}
	if len(data) != 1 {
		return nil, false
	}
	m, ok := data[0].(map[string]any)
	return m, ok
}

// jsonNum reads a numeric field from a decoded JSON object (every JSON number
// decodes to float64), returning 0 when it is absent or another type.
func jsonNum(m map[string]any, key string) float64 {
	v, _ := m[key].(float64)
	return v
}

// queryContext discovers the worker's context window (vLLM max_model_len /
// llama.cpp n_ctx), falling back to any declared value, in 1024-token units.
func (r *Router) queryContext(b *Backend) int {
	if k, ok := r.queryContextMeasured(b); ok {
		return k
	}
	return b.ContextK
}

// queryContextMeasured is queryContext without the fallback, reporting whether the
// worker actually answered.
//
// The fallback is right for a cold profile — no measurement, so keep whatever the
// registry holds — but wrong for refreshing a CACHED profile, where the two values
// come from different places and silently disagreeing is the whole bug being fixed.
// A failed probe there must leave the cached number alone, not overwrite it with the
// registry's, so the caller needs to tell "measured 256k" from "could not measure".
func (r *Router) queryContextMeasured(b *Backend) (int, bool) {
	if raw, err := r.backendGET(b, "/v1/models"); err == nil {
		// Same entry selection as queryModelInfo: on a catalogue endpoint,
		// data[0].max_model_len is some other model's context window.
		if m, ok := modelEntryFor(raw, b.Model); ok {
			if v := jsonNum(m, "max_model_len"); v > 0 {
				return int(v) / 1024, true
			}
		}
	}
	if raw, err := r.backendGET(b, "/props"); err == nil {
		if v, _ := raw["n_ctx"].(float64); v > 0 {
			return int(v) / 1024, true
		}
		if dg, ok := raw["default_generation_settings"].(map[string]any); ok {
			if v, _ := dg["n_ctx"].(float64); v > 0 {
				return int(v) / 1024, true
			}
		}
	}
	return 0, false
}

// querySlots reads the worker's OWN concurrent-request ceiling when it publishes one.
// llama.cpp's /props reports total_slots, which is exactly --parallel: a hard limit on
// how many sequences it will decode at once, regardless of what a throughput ramp
// appears to show. Used to cap capacityProbe, never to raise it — the ramp can still
// find a LOWER practical knee than the configured slot count.
//
// Returns 0 when the worker publishes nothing usable (vLLM does not expose
// max_num_seqs on any endpoint), which leaves the ramp's verdict untouched.
func (r *Router) querySlots(b *Backend) int {
	raw, err := r.backendGET(b, "/props")
	if err != nil {
		return 0
	}
	if v, _ := raw["total_slots"].(float64); v > 0 {
		return int(v)
	}
	return 0
}

// backendGET does an authenticated GET against a worker and decodes JSON.
func (r *Router) backendGET(b *Backend, path string) (map[string]any, error) {
	req, err := http.NewRequest("GET", strings.TrimRight(b.URL, "/")+path, nil)
	if err != nil {
		return nil, err
	}
	if b.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.APIKey)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: HTTP %d", path, resp.StatusCode)
	}
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// applyProfileIfGen copies measured values onto the live backend — declared
// values are only ever a seed; the measured profile wins. It applies only when
// the backend still exists at the registration generation the profile was
// measured against: results from a stale generation (the worker re-registered
// with new content, or was deleted, mid-measurement) are dropped, and the
// caller must not persist them either. gen 0 skips the check (tests, callers
// that hold no generation).
func (r *Registry) applyProfileIfGen(id string, gen int64, p *WorkerProfile) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	b := r.backends[id]
	if b == nil || (gen != 0 && b.profileGen != gen) {
		return false
	}
	if p.Model != "" {
		b.Model = p.Model
	}
	if p.Quality > 0 {
		b.Quality = p.Quality
	}
	if p.ContextK > 0 {
		b.ContextK = p.ContextK
	}
	if p.MaxConcurrency > 0 {
		b.MaxConcurrency = p.MaxConcurrency
	}
	if p.BaselineTPS > 0 {
		b.BaselineTPS = p.BaselineTPS
	}
	// Seed the prefill EWMA from the probe, but never overwrite a live one — same rule
	// as ObservedTPS. Without this seed a worker that mostly serves thinking traffic
	// never gets a prefill rate at all (observe() skips those samples) and every long
	// prompt is priced at the flat TTFT average.
	if p.PrefillTPS > 0 && b.ObservedPrefillTPS == 0 {
		b.ObservedPrefillTPS = p.PrefillTPS
	}
	if len(p.Features) > 0 {
		b.Features = append([]string(nil), p.Features...)
	}
	b.Thinking = p.Thinking
	b.ThinkingDialect = p.ThinkingDialect
	b.QualityDetail = p.QualityDetail
	b.Failed = append([]string(nil), p.Failed...)
	// Measured capacity becomes the slot cap — the bound that makes requests
	// queue at the router and spill across the fleet instead of piling onto a
	// saturated worker. Only a FULL profile carries a measured value
	// (BenchVersion is set); profileQuick's MaxConcurrency=1 is a provisional
	// placeholder that must not throttle a fresh worker to serial dispatch.
	if p.BenchVersion > 0 {
		r.syncSlotsLocked(id, p.MaxConcurrency)
	}
	return true
}

// SaveWorkerProfile/LoadWorkerProfile/DeleteWorkerProfile persist measured
// profiles so a warm restart (same id + model) skips re-profiling.
func (s *LogStore) SaveWorkerProfile(ctx context.Context, id string, p *WorkerProfile) error {
	data, err := json.Marshal(p)
	if err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, `INSERT INTO worker_profiles (id, model, updated_at, profile_json)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET model=excluded.model, updated_at=excluded.updated_at, profile_json=excluded.profile_json`,
		id, p.Model, time.Now().UTC().Format(time.RFC3339Nano), string(data))
	return err
}

func (s *LogStore) LoadWorkerProfile(ctx context.Context, id, model string) (*WorkerProfile, bool) {
	var raw string
	if err := s.db.QueryRowContext(ctx, `SELECT profile_json FROM worker_profiles WHERE id=? AND model=?`, id, model).Scan(&raw); err != nil {
		return nil, false
	}
	var p WorkerProfile
	if json.Unmarshal([]byte(raw), &p) != nil {
		return nil, false
	}
	return &p, true
}

// LoadWorkerProfileByID loads the most recent persisted profile for a worker id
// regardless of model (the table is keyed by id) — for the benchmark-inspection
// endpoint, which only has the id.
func (s *LogStore) LoadWorkerProfileByID(ctx context.Context, id string) (*WorkerProfile, bool) {
	var raw string
	if err := s.db.QueryRowContext(ctx, `SELECT profile_json FROM worker_profiles WHERE id=?`, id).Scan(&raw); err != nil {
		return nil, false
	}
	var p WorkerProfile
	if json.Unmarshal([]byte(raw), &p) != nil {
		return nil, false
	}
	return &p, true
}

func (s *LogStore) DeleteWorkerProfile(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM worker_profiles WHERE id=?`, id)
	return err
}
