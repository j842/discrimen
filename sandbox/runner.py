"""The child process. Everything below this line shares an address space with
model-generated code, and nothing above it does.

Started by supervisor.py as ``python3 -I -S runner.py grade|decode``, with the
request on stdin and nothing else — no argv payload (a 3.4 MB private-test blob
is well past ARG_MAX), no scratch file (the expected answers would then be
sitting in a directory the submission can read, and a submission that reads its
own answer key grades as correct).

RESULTS ARE STREAMED, one JSON object per line, and that is a robustness
decision rather than a stylistic one. A submission that hangs is killed by the
parent's wall clock, and a child that only reported at the end would have
reported nothing — so a timeout on case 17 of 20 would be indistinguishable
from a timeout on case 0. Streaming means the parent already holds every
completed case when the SIGKILL lands, and "16 of 20 passed, then it hung" is a
sentence the benchmark can use.

FD LAYOUT, and why it is not simply stdout:

  0   the request, then /dev/null. Redirected as soon as it has been read, so a
      submission calling input() gets EOF rather than blocking on a pipe that
      the parent has closed anyway.
  1   /dev/null. A submission can write to fd 1 directly — os.write(1, …) is
      two lines past print() — and if fd 1 were still the result pipe it could
      inject records into the grading stream. The real pipe is dup'd away first
      and fd 1 is pointed at the bit bucket, so a gigabyte of print() output
      costs the CPU to produce it and nothing else.
  2   /dev/null, or the service's stderr when SANDBOX_DEBUG is set.
  n   the dup of the original stdout, private to this module. Every record
      carries a per-run nonce so a submission that goes looking for the
      descriptor and sprays JSON at it still cannot forge a passing case.

The nonce is a speed bump, not a proof, and the difference should be stated
plainly: the harness and the submission share one interpreter, so a determined
submission can reach into this module's globals and read the nonce, or rewrite
compare.values_equal, or set the results list to whatever it likes. That is
inherent to grading in-process, it is equally true of LiveCodeBench's own
harness, and it is not what the jail is for. The jail exists to stop a
submission harming the HOST. Keeping a benchmark honest is a different problem
and the answer to it is that the questions are secret, not that the grader is
unforgeable.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import pickle
import signal
import sys
import traceback
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare  # noqa: E402  (must follow the sys.path fix-up above)
import jail  # noqa: E402

# Names the preamble puts in a submission's namespace before its first line
# runs. LeetCode starter code is annotated ``List[int]`` with no import in
# sight, so without at least the typing names every second submission dies on a
# NameError that has nothing to do with whether it solved the problem.
# LiveCodeBench injects the same sort of block for the same reason. numpy is
# NOT here: it is not in the image, and a submission that imports it gets a
# clean ImportError reported as its failure rather than a silent absence.
PREAMBLE = """
import sys, os, math, cmath, collections, itertools, functools, heapq, bisect
import string, re, random, fractions, decimal, operator, array, copy, json, time
from collections import Counter, OrderedDict, defaultdict, deque, ChainMap, namedtuple
from itertools import accumulate, product, permutations, combinations
from itertools import combinations_with_replacement, groupby, chain, count, cycle, repeat
from functools import lru_cache, cache, reduce, cmp_to_key, partial
from heapq import heappush, heappop, heapify, heappushpop, heapreplace, nlargest, nsmallest
from bisect import bisect, bisect_left, bisect_right, insort, insort_left, insort_right
from math import inf, gcd, lcm, sqrt, isqrt, ceil, floor, factorial, comb, perm
from math import log, log2, log10, exp, hypot, pi, e, fabs
from typing import Any, Callable, DefaultDict, Deque, Dict, FrozenSet, Iterable
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
"""

# What one case may hold of a submission's stdout. Only the stdin testtype reads
# it back, and no expected output in this dataset is anywhere near this size —
# so anything past it is a flood, and the flood is discarded as it arrives
# rather than accumulated and truncated later.
STDOUT_CAPTURE_LIMIT = 1 << 20

_RESULT_FD = 1
_NONCE = ""


class _Deadline(Exception):
    """The wall clock or the CPU limit ran out. Distinct from anything a
    submission can raise, so the case loop can tell "it timed out" from "it
    raised TimeoutError", which are different results."""


# ── Result stream ───────────────────────────────────────────────────────────


def emit(**record) -> None:
    """One NDJSON record onto the private result descriptor.

    os.write in a loop rather than a buffered file object: a partial write is
    legal on a pipe, and a record that arrives half-formed is a line the parent
    silently drops — which for a case record means a passing case scored as
    never having run.
    """
    record["n"] = _NONCE
    try:
        payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8", "replace")
    except (TypeError, ValueError):
        payload = (json.dumps({"n": _NONCE, "t": "fatal", "error": "unserialisable result"}) + "\n").encode()
    view = memoryview(payload)
    while view:
        try:
            written = os.write(_RESULT_FD, view)
        except BrokenPipeError:
            return  # the parent has gone; nothing left to report to
        except InterruptedError:
            continue
        view = view[written:]


# ── Captured streams ────────────────────────────────────────────────────────


class CappedStdout(io.TextIOBase):
    """sys.stdout for a submission: keeps the first STDOUT_CAPTURE_LIMIT
    characters and counts the rest.

    write() returns the full length even when it kept none of it, because that
    is what the TextIOBase contract says and some code checks. Reporting a short
    write would make a caller retry the remainder forever, which is a hang
    manufactured by the thing that exists to prevent hangs.
    """

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, limit: int = STDOUT_CAPTURE_LIMIT):
        self._limit = limit
        self._parts: list[str] = []
        self._size = 0

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        length = len(text)
        room = self._limit - self._size
        if room > 0:
            self._parts.append(text[:room])
        self._size += length
        return length

    def flush(self) -> None:
        return None

    @property
    def buffer(self):
        # Competitive-programming solutions reach for sys.stdout.buffer to skip
        # the text layer. TextIOBase has no such attribute, and the AttributeError
        # would be reported as the submission's failure when it is ours.
        return _BinaryBridge(self)

    def value(self) -> str:
        return "".join(self._parts)

    def truncated(self) -> bool:
        return self._size > self._limit


class _BinaryBridge(io.RawIOBase):
    def __init__(self, text_stream: CappedStdout):
        self._text = text_stream

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        self._text.write(bytes(data).decode("utf-8", "replace"))
        return len(data)

    def flush(self) -> None:
        return None


class FedStdin(io.StringIO):
    """sys.stdin for a stdin-testtype case, with the .buffer a fair number of
    solutions read instead."""

    @property
    def buffer(self):
        return io.BytesIO(self.getvalue().encode("utf-8"))


# ── Building the submission's namespace ─────────────────────────────────────


def build_namespace() -> dict:
    """A fresh module namespace with the preamble already run in it.

    Builtins are deliberately NOT restricted. Handing the submission a cut-down
    __builtins__ is the classic move here and it is security theatre in CPython:
    ``().__class__.__base__.__subclasses__()`` walks back to everything that was
    removed, in one expression, and has since Python 2. The isolation in this
    service is the process boundary, the rlimits and the seccomp filter — all of
    which hold whatever the submission can name. Pretending the namespace is
    also a boundary would only make the real ones look optional.
    """
    namespace: dict = {"__name__": "__submission__"}
    exec(compile(PREAMBLE, "<preamble>", "exec"), namespace)
    return namespace


def load_submission(source: str, namespace: dict) -> str | None:
    """Execute the submission at module level. Returns a failure description, or
    None on success.

    A failure is returned rather than raised because it is not always fatal: exec
    populates the namespace as it goes, so a submission whose class is defined at
    the top and whose demo driver at the bottom raises has still defined the
    class. The caller tries to resolve the entry point before deciding, which
    turns a whole category of "model appended a __main__ block" from a zero into
    a graded answer.
    """
    try:
        exec(compile(source, "<submission>", "exec"), namespace)
    except SyntaxError as exc:
        return f"{type(exc).__name__}: {exc}"
    except BaseException as exc:  # noqa: BLE001 — including SystemExit and KeyboardInterrupt
        return f"{type(exc).__name__} while loading the submission: {exc}"
    return None


def resolve_entry(namespace: dict, entry: dict | None):
    """Find how to reach the callable a functional case should invoke.

    Returns a FACTORY, not the callable: calling it produces a freshly bound
    method on a freshly constructed instance. LiveCodeBench builds one
    Solution() and reuses it for every case, and that is a false-pass generator
    — a solution that memoises on self can answer case 4 out of a cache case 3
    filled, and once any cross-case state exists the cases have stopped being
    independent measurements. It cuts the other way too: state left behind by a
    case that raised can fail a case that would have passed. A fresh instance
    costs one __init__ per case and buys cases that mean what they say.

    Three tiers of lookup, and the third is the one that earns its place: models
    rename the class about as often as they keep it (``class Sol``, or the
    method hoisted to module level), while the func name is the part they do not
    change — it was in the starter code they were handed. Scanning for it is
    what stops a correct answer scoring zero over a class name nobody asked
    about.
    """
    func = (entry or {}).get("func") or ""
    cls = (entry or {}).get("class") or ""
    if not func:
        raise LookupError("no entry function was given for a functional test")

    if cls and isinstance(namespace.get(cls), type) and hasattr(namespace[cls], func):
        holder = namespace[cls]
        return lambda: getattr(holder(), func)

    candidate = namespace.get(func)
    if callable(candidate) and not isinstance(candidate, type):
        return lambda: candidate

    for name, value in namespace.items():
        if name.startswith("__") or not isinstance(value, type):
            continue
        if value.__module__ != "__submission__":
            continue  # a class the preamble imported, not one the submission wrote
        if hasattr(value, func):
            holder = value
            return lambda: getattr(holder(), func)

    raise LookupError(f"the submission defines no {cls + '.' if cls else ''}{func}")


# ── Grading ─────────────────────────────────────────────────────────────────


def run_grade(payload: dict) -> None:
    """Every case, in order, one record each.

    EVERY case, not "up to the first failure". LiveCodeBench stops at the first
    one and it is the cheaper choice, but the response contract asks for
    cases_passed as well as pass, and an early stop makes that field say
    "cases_run − 1" on every failing submission — a count of nothing. Running
    the rest turns it into a real fraction, which is the difference between "the
    model is wrong" and "the model is wrong on the empty-input edge case", and
    on a submission that passes (the case worth optimising for) it costs
    nothing at all. The cost lands only on a wrong AND slow submission, and that
    is what the wall clock is for. Callers who only need the boolean can send
    stop_on_first_failure.
    """
    source = compare.extract_source(payload.get("code") or "")
    tests = payload.get("tests") or []
    entry = payload.get("entry") or {}
    stop_early = bool(payload.get("stop_on_first_failure"))

    namespace = build_namespace()
    load_error = load_submission(source, namespace)

    # The entry point is resolved once, up front, against whatever namespace the
    # submission left behind — so a submission that defines nothing usable fails
    # the request with one clear message instead of the same message repeated
    # once per test case. A stdin-only run needs no entry point at all, and
    # demanding one would reject a perfectly gradeable request.
    factory = None
    if any((case.get("testtype") or "functional").strip().lower() != "stdin" for case in tests):
        try:
            factory = resolve_entry(namespace, entry)
        except Exception as exc:  # noqa: BLE001
            # The load error, when there is one, is the cause; the missing entry
            # point is only its symptom.
            emit(t="fatal", error=load_error or f"{type(exc).__name__}: {exc}")
            return

    for index, case in enumerate(tests):
        testtype = (case.get("testtype") or "functional").strip().lower()
        try:
            if testtype == "stdin":
                record = run_stdin_case(source, case)
            else:
                record = run_functional_case(factory, case)
        except _Deadline as exc:
            emit(t="fatal", error=str(exc) or "time limit exceeded", index=index)
            return
        except MemoryError:
            emit(t="fatal", error="memory limit exceeded", index=index)
            return
        except compare.TestDataError as exc:
            emit(t="fatal", error=f"malformed test case {index}: {exc}")
            return
        record["t"] = "case"
        record["index"] = index
        emit(**record)
        if stop_early and not record.get("pass"):
            emit(t="done", stopped_early=True)
            return

    emit(t="done")


def run_functional_case(factory, case: dict) -> dict:
    args = compare.parse_functional_args(case.get("input") or "")
    expected = compare.parse_expected(case.get("output") or "")

    try:
        call_target = factory()
    except (_Deadline, MemoryError):
        raise
    except BaseException as exc:  # noqa: BLE001
        return _failure(expected, None, f"constructing the entry point: {_describe(exc)}")

    sink = CappedStdout()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(CappedStdout(4096)):
            got = call_target(*args)
    except (_Deadline, MemoryError):
        raise
    except BaseException as exc:  # noqa: BLE001 — a submission may raise SystemExit
        return _failure(expected, None, _describe(exc))

    try:
        matched = compare.values_equal(expected, got)
    except RecursionError:
        # A submission that returned a self-referential structure. MAX_DEPTH
        # normally catches this first; this is the case where the recursion is
        # in a __eq__ the submission wrote.
        return _failure(expected, None, "the returned value could not be compared (recursive structure)")
    if matched:
        return {"pass": True}
    return _failure(expected, got, None)


def run_stdin_case(source: str, case: dict) -> dict:
    """A whole-program case: feed stdin, capture stdout, compare the text.

    The submission is re-executed from source for every case rather than being
    loaded once and called repeatedly. There is no callable to call — the answer
    IS the side effect of running the module — and a module only reads its stdin
    once, so any form of reuse would give case two an empty input.

    KNOWN LIMIT. The input is fed by replacing sys.stdin, so it reaches
    input(), sys.stdin.read() and sys.stdin.buffer — everything a competitive
    solution uses — but NOT a raw os.read(0, …), which by then is /dev/null and
    returns EOF. Feeding the real descriptor instead would mean handing the
    submission a pipe the parent has to keep writing to while it also drains
    results, and a submission that never reads would park that writer until the
    wall clock; the trade is a case that fails cleanly for a technique nothing
    in this dataset uses, against a class of hang in the code that exists to
    prevent hangs.
    """
    expected = case.get("output") or ""
    sink = CappedStdout()
    namespace = build_namespace()
    namespace["__name__"] = "__main__"

    saved_stdin = sys.stdin
    sys.stdin = FedStdin(case.get("input") or "")
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(CappedStdout(4096)):
            try:
                exec(compile(source, "<submission>", "exec"), namespace)
            except SystemExit as exc:
                # sys.exit(0) at the end of main() is idiomatic and is not a
                # failure; a non-zero code is the program saying it failed.
                if exc.code not in (None, 0):
                    return _failure(expected, sink.value(), f"exited with status {exc.code}")
    except (_Deadline, MemoryError):
        raise
    except BaseException as exc:  # noqa: BLE001
        return _failure(expected, sink.value(), _describe(exc))
    finally:
        sys.stdin = saved_stdin

    got = sink.value()
    if sink.truncated():
        return _failure(expected, got, f"stdout exceeded {STDOUT_CAPTURE_LIMIT} bytes")
    if compare.compare_stdout(expected, got):
        return {"pass": True}
    return _failure(expected, got, None)


def _failure(expected, got, error: str | None) -> dict:
    """One failed case, with both sides rendered for a human to read.

    Rendered through JSON even for a stdin case, where both sides are already
    strings and passing them through untouched would be the obvious thing. It is
    the wrong thing: the failure a stdin case most often has is a trailing
    newline or a stray space, and untouched text renders those as nothing at
    all. ``"16\\n"`` against ``"16"`` is a report somebody can act on.
    """
    return {
        "pass": False,
        "expected": compare.render(expected),
        "got": compare.render(got),
        "error": error,
    }


def _describe(exc: BaseException) -> str:
    """One line naming the exception and where in the SUBMISSION it came from.

    The traceback is walked for the innermost frame belonging to <submission>,
    because the outermost frames are this file and reporting them would tell an
    operator that runner.py raised IndexError.
    """
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<submission>":
            line = frame.lineno
    where = f" at line {line}" if line else ""
    detail = str(exc)
    if len(detail) > 400:
        detail = detail[:400] + "…"
    return f"{type(exc).__name__}{where}: {detail}" if detail else f"{type(exc).__name__}{where}"


# ── Private test decoding ───────────────────────────────────────────────────


def run_decode(payload: dict) -> None:
    """base64 → zlib → pickle, which is why it happens in here.

    pickle.loads on data fetched from the internet is arbitrary code execution
    by design — the format's whole job is to reconstruct objects by calling
    things — and no amount of validating the base64 first changes that. So the
    decode runs in the same jail as a submission, under the same limits, with
    the same seccomp filter: whatever a hostile pickle does it does with no
    network, a bounded address space and a process group about to be killed.

    The double decode at the end is LiveCodeBench's shape, not a guess: the
    pickle payload is a JSON *string*, so the object that comes out has to be
    parsed a second time. Rows that pickle the list directly are accepted too,
    because reading the format defensively is free and a version bump upstream
    that changed it would otherwise take every coding question offline.
    """
    blob = payload.get("blob") or ""
    try:
        raw = base64.b64decode(blob, validate=False)
    except Exception as exc:  # noqa: BLE001
        emit(t="fatal", error=f"blob is not valid base64: {exc}")
        return
    try:
        data = zlib.decompress(raw)
    except Exception as exc:  # noqa: BLE001
        emit(t="fatal", error=f"blob is not zlib-compressed: {exc}")
        return
    try:
        obj = pickle.loads(data)
    except Exception as exc:  # noqa: BLE001
        emit(t="fatal", error=f"unpickling failed: {type(exc).__name__}: {exc}")
        return

    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode("utf-8", "replace")
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as exc:
            emit(t="fatal", error=f"decoded payload is not JSON: {exc}")
            return
    if not isinstance(obj, list):
        emit(t="fatal", error=f"decoded payload is a {type(obj).__name__}, expected a list of test cases")
        return

    tests = []
    for item in obj:
        if not isinstance(item, dict):
            emit(t="fatal", error=f"decoded test case is a {type(item).__name__}, expected an object")
            return
        tests.append(
            {
                "input": _as_text(item.get("input", "")),
                "output": _as_text(item.get("output", "")),
                "testtype": _as_text(item.get("testtype", "functional")) or "functional",
            }
        )
    emit(t="tests", tests=tests)
    emit(t="done")


def _as_text(value) -> str:
    """Whatever the pickle held, as a string.

    Not str() on everything: a pickle can hold any object at all, and str() on a
    hostile one calls its __str__. The types that legitimately appear are handled
    by name and the rest are refused by shape rather than by execution.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, dict, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return ""
    return ""


# ── Entry point ─────────────────────────────────────────────────────────────


def _on_deadline(signum, _frame):
    name = "cpu limit exceeded" if signum == signal.SIGXCPU else "time limit exceeded"
    raise _Deadline(name)


def main(argv: list[str]) -> int:
    global _RESULT_FD, _NONCE

    mode = argv[1] if len(argv) > 1 else ""
    request = sys.stdin.buffer.read()

    # From here on fd 1 is the bit bucket and the result stream is private.
    _RESULT_FD = os.dup(1)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)

    try:
        payload = json.loads(request)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        emit(t="fatal", error=f"unreadable request: {exc}")
        return 2
    _NONCE = str(payload.get("nonce") or "")

    drop_to = payload.get("drop_to")
    try:
        status = jail.apply(
            int(payload.get("memory_mb") or 512),
            int(payload.get("cpu_seconds") or 10),
            (int(drop_to[0]), int(drop_to[1])) if drop_to else None,
        )
    except PermissionError as exc:
        # Refusing to run as root is the one setup failure worth aborting on:
        # every limit below would be a limit root can simply raise.
        emit(t="fatal", error=str(exc))
        return 2
    emit(t="jail", network=status)

    # The child's own deadline sits inside the parent's, so in the ordinary case
    # this fires first and the run ends with a record saying so. When the
    # submission is stuck somewhere a Python signal handler cannot run — a long
    # sort, a pathological regex — nothing here fires and the parent's SIGKILL
    # is what ends it. Both paths are covered; this one is just the readable one.
    signal.signal(signal.SIGALRM, _on_deadline)
    signal.signal(signal.SIGXCPU, _on_deadline)
    budget = float(payload.get("wall_seconds") or 10.0)
    signal.setitimer(signal.ITIMER_REAL, max(0.05, budget))

    try:
        if mode == "grade":
            run_grade(payload)
        elif mode == "decode":
            run_decode(payload)
        else:
            emit(t="fatal", error=f"unknown runner mode {mode!r}")
            return 2
    except _Deadline as exc:
        emit(t="fatal", error=str(exc))
    except MemoryError:
        emit(t="fatal", error="memory limit exceeded")
    except BaseException as exc:  # noqa: BLE001
        emit(t="fatal", error=f"grader crashed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    code = main(sys.argv)
    # _exit, not sys.exit: interpreter shutdown runs atexit handlers, module
    # __del__ methods and garbage collection, all of which are the submission's
    # code getting one more turn after the results have been reported. A
    # __del__ that loops would hang a process the parent has already accounted
    # for as finished.
    sys.stderr.flush()
    os._exit(code)
