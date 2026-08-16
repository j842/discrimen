package router

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// visionProbe detects multimodal (image) support empirically instead of trusting
// a declared tag. It sends a solid-red test image using the standard OpenAI
// image_url content schema and requires the worker to BOTH accept it AND correctly
// name the colour (a server that silently ignores the image part can't). That
// double check is robust to both failure modes.
//
// Rejections come in two shapes, because servers disagree on the status code for
// "this model has no image support": vLLM answers 4xx (isClientReject), llama.cpp
// answers 500 with an explicit message (isVisionUnsupported). Both are permanent
// verdicts and short-circuit. Anything else — transport, a bare 5xx, or a
// load-shedding 408/425/429 — is retried. Returns false on any doubt; vision is
// opt-in by evidence.
func (r *Router) visionProbe(b *Backend) (vision, inconclusive bool) {
	payload := map[string]any{
		"model":                probeModel(b),
		"stream":               false,
		"max_tokens":           16,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": "What single colour fills this image? Answer with one lowercase word."},
				{"type": "image_url", "image_url": map[string]string{"url": visionTestImageDataURL()}},
			},
		}},
	}
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			// A load-shedding worker that 429'd milliseconds ago will 429 an
			// immediate retry too; give it a beat.
			time.Sleep(2 * time.Second)
		}
		raw, err := r.rawCompletion(b, payload)
		if err != nil {
			if isClientReject(err) || isVisionUnsupported(err) {
				return false, false // definitive → the worker refuses image input (text-only model)
			}
			continue // transient (408/425/429, bare 5xx, transport) → retry
		}
		content, _ := completionContent(raw)
		return strings.Contains(strings.ToLower(content), "red"), false
	}
	// Retries exhausted on transient errors only — never a definitive rejection.
	// Report false ("unconfirmed", per the opt-in-by-evidence convention above)
	// rather than true (which would route image traffic to a worker that never
	// proved it can read one) — but flag it inconclusive so the profile is NOT
	// persisted as if vision were measured absent (cached certifications don't
	// re-probe). A declared vision tag survives regardless — profile.go keeps it
	// as a backstop.
	return false, true
}

// completionStatusPattern matches the exact prefix rawCompletion stamps on HTTP
// errors ("completion returned %d: <body>"). Anchoring to that known format is
// what keeps status-like digits inside an error body (e.g. an upstream proxy
// echoing "returned 404") from being mistaken for the worker's own status.
var completionStatusPattern = regexp.MustCompile(`^completion returned (\d{3}):`)

// completionStatusCode extracts the HTTP status from a rawCompletion error, or
// 0 when the error carries none (transport failure, JSON decode error, ...).
func completionStatusCode(err error) int {
	if err == nil {
		return 0
	}
	m := completionStatusPattern.FindStringSubmatch(err.Error())
	if m == nil {
		return 0
	}
	code, _ := strconv.Atoi(m[1])
	return code
}

// isClientReject reports whether a rawCompletion error was a definitive HTTP 4xx
// rejection of the request itself (e.g. "this model does not support image
// input") rather than something worth retrying. Not every 4xx is a verdict on
// the model: 408 (timeout), 425 (too early) and 429 (overloaded — what a busy
// production worker returns mid-recertification) are load conditions, so they
// classify as transient alongside 5xx/transport errors.
func isClientReject(err error) bool {
	switch code := completionStatusCode(err); code {
	case 408, 425, 429:
		return false // transient load shedding, not a statement about image support
	default:
		return code >= 400 && code < 500
	}
}

// visionUnsupportedPattern matches a worker stating outright that it cannot accept
// image input, whatever status code it wrapped that in.
//
// This exists because llama.cpp reports the condition as an HTTP 500 —
//
//	{"error":{"code":500,"message":"image input is not supported - hint: if this
//	 is unexpected, you may need to provide the mmproj","type":"server_error"}}
//
// — where vLLM uses a 4xx. A bare 500 is genuinely transient and isClientReject
// deliberately keeps it that way, so before this existed every text-only llama.cpp
// worker burned all three attempts on the same permanent error, reported
// inconclusive, and had its whole profile discarded unpersisted (profile.go's
// Incomplete handling). The visible symptom was a fleet that re-ran its full
// cold-start benchmark on every restart and never stored a profiling run.
var visionUnsupportedPattern = regexp.MustCompile(
	`(?i)image input is not supported|does not support image|no multimodal support|mmproj`)

// isVisionUnsupported reports whether err is a worker declaring it has no image
// support. That is a permanent capability verdict rather than a fault, so the
// probe should stop retrying and record a definitive "no".
//
// Matching on message text is deliberate: the status code alone cannot carry this
// meaning once servers disagree about it. A false positive is benign — the only
// consequence is "vision: not detected" on a worker that just told us it has no
// vision — and a declared `vision` tag still survives as a backstop in profile.go.
func isVisionUnsupported(err error) bool {
	return err != nil && visionUnsupportedPattern.MatchString(err.Error())
}

// visionTestImageDataURL builds a 32×32 solid pure-red PNG as a data URL. Pure
// red (255,0,0) is unambiguous, so any real vision model answers "red", while a
// text model that ignored the image won't reliably say it.
func visionTestImageDataURL() string {
	const n = 32
	img := image.NewRGBA(image.Rect(0, 0, n, n))
	red := color.RGBA{R: 255, G: 0, B: 0, A: 255}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			img.SetRGBA(x, y, red)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return ""
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}
