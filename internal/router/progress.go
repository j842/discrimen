package router

// What a cold-start profile is doing, while it does it.
//
// A profile is the longest thing this router ever runs — hours on a slow worker,
// and every worker at once after a benchmarkVersion bump — and until now the
// only thing it published was a boolean. `ask -l` said "profiling" and nothing
// else, so the operator's questions ("is it stuck?", "how much longer?", "is it
// actually doing anything?") had no answer short of reading container logs.
//
// The phases are deliberately named after what a person would recognise rather
// than after the functions that run them, because the audience for this is
// somebody watching a fleet come up, not somebody reading profile.go.

import (
	"fmt"
	"sync/atomic"
	"time"
)

// Profile phases, in the order they run.
const (
	phaseCapabilities = "capabilities" // chat/json/tools/thinking/vision + context + speed
	phaseCapacity     = "capacity"     // the concurrency ramp
	phasePrefill      = "prefill"      // prompt-processing rate
	phaseQuality      = "quality"      // the graded question set, thinking-on
	phaseQualityNT    = "quality/nothink"
	phaseContext      = "context" // the needle-in-a-haystack ladder
)

// ProfileProgress is one worker's live profiling state, published on /backends.
//
// The counters are atomics because they are written by the benchmark's worker
// goroutines (up to benchConc at once) and read by every /backends request. A
// mutex would be correct too and would serialise a hot read path against a hot
// write path for no benefit — nothing here needs the phase and the counter to
// be consistent with each other to the instant.
type ProfileProgress struct {
	phase    atomic.Value // string
	done     atomic.Int64
	total    atomic.Int64
	inFlight atomic.Int64
	// startedAt is the whole profile; phaseAt (unix nanos) is the current phase.
	// The ETA has to divide by the phase's own elapsed time, not the profile's:
	// quality runs fourth, so charging it with the capacity ramp's minutes would
	// inflate per-question cost by however long the earlier phases took and
	// project an ETA far past the truth.
	startedAt time.Time
	phaseAt   atomic.Int64
}

func newProfileProgress() *ProfileProgress {
	p := &ProfileProgress{startedAt: time.Now()}
	p.phase.Store(phaseCapabilities)
	p.phaseAt.Store(time.Now().UnixNano())
	return p
}

// enter starts a phase with a known amount of work. total <= 0 means the phase
// has no natural unit to count, which is most of them.
func (p *ProfileProgress) enter(phase string, total int) {
	if p == nil {
		return
	}
	p.phase.Store(phase)
	p.total.Store(int64(total))
	p.done.Store(0)
	p.phaseAt.Store(time.Now().UnixNano())
}

func (p *ProfileProgress) step() {
	if p != nil {
		p.done.Add(1)
	}
}

// begin/end bracket one in-flight generation, so an operator can see that the
// worker is being asked something rather than merely that a phase is open — the
// difference between "slow" and "stuck".
func (p *ProfileProgress) begin() {
	if p != nil {
		p.inFlight.Add(1)
	}
}

func (p *ProfileProgress) end() {
	if p != nil {
		p.inFlight.Add(-1)
	}
}

// snapshot renders the live state for the API. Returns nil for a nil receiver so
// callers do not have to branch.
func (p *ProfileProgress) snapshot() *ProfileProgressView {
	if p == nil {
		return nil
	}
	phase, _ := p.phase.Load().(string)
	v := &ProfileProgressView{
		Phase:     phase,
		Done:      int(p.done.Load()),
		Total:     int(p.total.Load()),
		InFlight:  int(p.inFlight.Load()),
		ElapsedMS: time.Since(p.startedAt).Milliseconds(),
	}
	// An ETA only from the phase that has a countable unit and enough of it to
	// extrapolate from. Guessing at the rest would put a number on the screen
	// that means nothing, which is worse than no number.
	if v.Total > 0 && v.Done >= profileETAMinSamples {
		phaseMS := time.Since(time.Unix(0, p.phaseAt.Load())).Milliseconds()
		per := float64(phaseMS) / float64(v.Done)
		v.RemainingMS = int64(per * float64(v.Total-v.Done))
	}
	return v
}

// profileETAMinSamples is how many questions must be graded before the estimate
// is published. Early questions are unrepresentative — the first few race the
// worker's own warm-up — and a wildly wrong first estimate is the thing that
// makes an operator stop believing the later, accurate ones.
const profileETAMinSamples = 8

// ProfileProgressView is the serialisable form.
type ProfileProgressView struct {
	Phase       string `json:"phase"`
	Done        int    `json:"done,omitempty"`
	Total       int    `json:"total,omitempty"`
	InFlight    int    `json:"in_flight"`
	ElapsedMS   int64  `json:"elapsed_ms"`
	RemainingMS int64  `json:"remaining_ms,omitempty"`
}

// String is the one-line form for a terminal.
func (v *ProfileProgressView) String() string {
	if v == nil {
		return ""
	}
	out := v.Phase
	if v.Total > 0 {
		out += fmt.Sprintf(" %d/%d", v.Done, v.Total)
	}
	if v.InFlight > 0 {
		out += fmt.Sprintf(" (%d in flight)", v.InFlight)
	}
	out += " " + compactDuration(time.Duration(v.ElapsedMS)*time.Millisecond)
	if v.RemainingMS > 0 {
		out += fmt.Sprintf(", ~%s left", compactDuration(time.Duration(v.RemainingMS)*time.Millisecond))
	}
	return out
}

// compactDuration renders a duration the way a person reads one off a progress
// line: minutes and hours, never "1h2m3.456s".
func compactDuration(d time.Duration) string {
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	default:
		return fmt.Sprintf("%dh%02dm", int(d.Hours()), int(d.Minutes())%60)
	}
}

// progressFor is the live tracker for a worker being profiled, or nil.
//
// Looked up by id rather than threaded through profileQuick/profileBackend/
// runQualityBenchmark and back, because the alternative is a parameter on a
// dozen signatures that exists only so a progress line can be drawn. Every
// method on *ProfileProgress is nil-safe, so callers never branch.
func (r *Router) progressFor(id string) *ProfileProgress {
	if r == nil {
		return nil
	}
	if v, ok := r.profiling.Load(id); ok {
		if p, ok := v.(*ProfileProgress); ok {
			return p
		}
	}
	return nil
}
