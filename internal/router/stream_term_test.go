package router

import (
	"io"
	"strings"
	"testing"
	"time"
)

// A stream must always END. copyStreaming exits only on bytes-then-EOF, a read
// error or a write error, so every way a source can misbehave has to resolve
// into one of those — otherwise the request, its slot and its ActiveRequests
// entry are held until the client disconnects.

// stalledReader is the pathological case the io.Reader contract discourages but
// permits: never any bytes, never any error.
type stalledReader struct{ reads int }

func (s *stalledReader) Read(p []byte) (int, error) { s.reads++; return 0, nil }

// A source returning (0, nil) forever used to spin a core inside the tool-call
// guard with no bytes emitted, so copyStreaming never called progress() and — if
// the idle watchdog was disabled — nothing ever stopped it. streamClient has no
// client-level timeout, so there was no second bound.
func TestStalledSourceTerminates(t *testing.T) {
	src := &stalledReader{}
	g := newToolCallGuard(io.NopCloser(src), &Backend{})
	done := make(chan error, 1)
	go func() {
		buf := make([]byte, 4096)
		_, err := g.Read(buf)
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("a stalled source returned success")
		}
		if src.reads > maxIdleReads+2 {
			t.Errorf("spun %d times before giving up, cap is %d", src.reads, maxIdleReads)
		}
	case <-time.After(3 * time.Second):
		t.Fatalf("guard did not terminate on a stalled source (spun %d times)", src.reads)
	}
}

// The ordinary path must be untouched: a real stream ends at EOF.
func TestNormalStreamTerminatesAtEOF(t *testing.T) {
	body := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
	g := newToolCallGuard(io.NopCloser(strings.NewReader(body)), &Backend{})
	var out strings.Builder
	buf := make([]byte, 64)
	for i := 0; i < 1000; i++ {
		n, err := g.Read(buf)
		out.Write(buf[:n])
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if i == 999 {
			t.Fatal("guard did not reach EOF within 1000 reads")
		}
	}
	if !strings.Contains(out.String(), "[DONE]") {
		t.Errorf("the stream's terminator did not survive the guard: %q", out.String())
	}
}

// A source that errors mid-stream must surface the error, not swallow it — the
// client needs an in-stream failure rather than a silently short answer.
func TestGuardPropagatesAMidStreamError(t *testing.T) {
	src := io.MultiReader(strings.NewReader("data: {\"a\":1}\n"), &erroringReader{})
	g := newToolCallGuard(io.NopCloser(src), &Backend{})
	buf := make([]byte, 4096)
	sawData := false
	for i := 0; i < 100; i++ {
		n, err := g.Read(buf)
		if n > 0 {
			sawData = true
		}
		if err != nil {
			if err == io.EOF {
				t.Fatal("a mid-stream error was reported as a clean EOF")
			}
			if !sawData {
				t.Error("the bytes before the error were dropped")
			}
			return
		}
	}
	t.Fatal("guard never terminated on an erroring source")
}

type erroringReader struct{}

func (erroringReader) Read(p []byte) (int, error) { return 0, io.ErrUnexpectedEOF }

// An unterminated tail must be forwarded rather than held: a stream whose last
// line has no newline still has to reach the client before the stream ends.
func TestGuardForwardsAnUnterminatedTail(t *testing.T) {
	g := newToolCallGuard(io.NopCloser(strings.NewReader("data: {\"a\":1}")), &Backend{})
	var out strings.Builder
	buf := make([]byte, 64)
	for i := 0; i < 100; i++ {
		n, err := g.Read(buf)
		out.Write(buf[:n])
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	if !strings.Contains(out.String(), `{"a":1}`) {
		t.Errorf("the unterminated tail was swallowed: %q", out.String())
	}
}

// A line longer than maxSSELine must be flushed rather than buffered forever —
// otherwise a source that never sends a newline grows the buffer without bound
// and emits nothing, which reads to everything upstream as a silent stream.
func TestGuardFlushesAnOverlongLine(t *testing.T) {
	g := newToolCallGuard(io.NopCloser(strings.NewReader(strings.Repeat("x", maxSSELine+1024))), &Backend{})
	buf := make([]byte, 4096)
	total := 0
	for i := 0; i < 10000; i++ {
		n, err := g.Read(buf)
		total += n
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	if total == 0 {
		t.Error("a newline-free stream produced no output; it was buffered indefinitely")
	}
}
