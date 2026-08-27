"""The child process. Everything below this line shares an address space with
model-generated code, and nothing above it does.

Started by supervisor.py as ``python3 -I -S runner.py grade|decode``, with the
request on stdin and nothing else — no argv payload (a 3.4 MB private-test blob
is well past ARG_MAX), no scratch file (a submission can read its own working
directory, and anything about the run left lying in there is something it can
read about itself).

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
  2   /dev/null, or the service's stderr when SANDBOX_DEBUG is set. The only
      one of the four this file does not arrange for itself: the parent picks it
      at spawn time (supervisor._spawn) and nothing below touches fd 2.
  n   the dup of the original stdout, private to this module. Every record
      carries a per-run nonce so a submission that goes looking for the
      descriptor and sprays JSON at it still cannot forge a passing case.

THE VERDICT IS NOT DECIDED HERE. This process shares one interpreter with the
submission, so anything it computes the submission can rewrite — and the
cheapest version of that needs no knowledge of the question at all:

    import sys
    sys.modules['compare'].values_equal = lambda *a, **k: True

Two lines, and every case reports a pass. This docstring used to answer that
with "the questions are secret", which defends against a submission that knows
the ANSWER and not at all against one that redefines what CORRECT MEANS. So this
process now reports what the submission returned and main.py decides, in an
address space the submission never touches (see _decide).

AND THE ANSWER KEY IS NOT HERE EITHER, which is the half that moving the verdict
out did NOT fix and which this docstring used to claim it had. Deciding in the
parent stops a submission redefining correctness; it does nothing about one that
simply LOOKS UP the right answer and returns it, and the expected outputs used
to arrive in this process inside the request:

    fr = sys._getframe(1)
    while fr:                                    # → pass:true, 3/3
        if 'expected' in fr.f_locals: return fr.f_locals['expected']
        fr = fr.f_back

Three lines, no knowledge of the question, and a perfect score against
deliberately wrong expectations — confirmed live, along with the same walk for
``payload`` and for the raw ``request`` bytes, and with a full forged record set
emitted through ``sys.modules['__main__'].emit``. Every one of them is a
capability measurement inflated by a model that emits a test-harness idiom.

So the request the parent sends carries the INPUTS ONLY. Test cases reach this
process with no ``output`` field at all (main.handle_grade strips them, and
_strip_answer_key strips them again here in case a future caller does not),
which is why nothing below this line renders or compares an expected value.
Frame-walking now finds the questions and never the answers; forging a record
only lets a submission lie about what it RETURNED, which the parent compares
against an expected value this process has never seen. The nonce remains a
speed bump against a submission spraying JSON at the result descriptor.

Do not add the expected value back for a nicer failure report. The parent has
the test cases and renders both sides itself — see main._decide.
"""

from __future__ import annotations

import base64
import codecs
import contextlib
import io
import json
import os
import pickle
import signal
import sys
import traceback
import types
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


class FedStdin(io.TextIOBase):
    """sys.stdin for a stdin-testtype case, with the .buffer a fair number of
    solutions read instead.

    ONE stream, read from ONE cursor, because that is what a real interpreter
    gives a program and the previous shape did not: .buffer was a property that
    built a fresh BytesIO over the whole input on every access, so the standard
    competitive-programming opening

        n = int(input())
        data = sys.stdin.buffer.read()

    re-read the header it had already consumed. On "3\\n1 2 3\\n" that printed
    "3 9" instead of "3 6", and it reported error:null — indistinguishable from
    the model getting the arithmetic wrong. Every solution mixing the text and
    binary APIs was being graded on doubled input.

    The text side reads through the same BytesIO without reading ahead, so what
    input() leaves behind is exactly what .buffer.read() returns. CPython's own
    sys.stdin does read ahead — a TextIOWrapper pulls 8 KB at a time, which is
    why the idiom above yields b"" there on a small input and the tail of a large
    one. Matching that quirk would swap one confound for another: the same
    submission would pass the small cases and fail the big ones, which reads as a
    wrong answer too. The input here is already entirely in memory, so there is
    nothing for a read-ahead buffer to win.
    """

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, text: str):
        self._raw = io.BytesIO(text.encode("utf-8"))
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    @property
    def buffer(self):
        return self._raw

    def readable(self) -> bool:
        return True

    def read(self, size: int | None = -1) -> str:
        if size is None or size < 0:
            return self._decoder.decode(self._raw.read(), True)
        # By bytes, not by characters: a character is at least one byte, so
        # reading `size` bytes can never overshoot `size` characters, and a
        # multi-byte character split across the boundary is held by the
        # incremental decoder until the next pass tops it up.
        out = ""
        while len(out) < size:
            chunk = self._raw.read(size - len(out))
            if not chunk:
                out += self._decoder.decode(b"", True)
                break
            out += self._decoder.decode(chunk)
        return out

    def readline(self, size: int | None = -1) -> str:
        raw = self._raw.readline() if size is None or size < 0 else self._raw.readline(size)
        return self._decoder.decode(raw, not raw)


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


def _last_line(code: types.CodeType) -> int:
    """The highest source line this code object covers, nested defs included.

    The nesting matters: co_lines() on a method reports the line its inner
    ``def popcount`` sits on and none of that function's body, so a completion
    whose only contribution is inside a helper the prefix opened would otherwise
    look like it had contributed nothing.
    """
    last = code.co_firstlineno
    for _start, _end, line in code.co_lines():
        if line is not None and line > last:
            last = line
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            last = max(last, _last_line(const))
    return last


def _only_from_the_prefix(func, prefix_lines: int) -> bool:
    """Whether this callable's whole body came from the partial solution.

    A completion prompt hands the model a class with one method truncated
    part-way through, and an answer that restates that method at column 0 leaves
    the truncated stub in the class with its own copy out at module level. The
    two have the same name, so a lookup by name grades the stub — which returns
    None for every case, reporting 0/N with error:null. compare.assemble now
    re-indents that shape back into the class; this is the second line of
    defence, so a stub that survives anyway is refused by name rather than
    graded in silence.

    A COMPLIANT answer is not caught by it. Its ``def`` line does come from the
    prefix, but its body extends past the prefix's last line, so the code object
    reaches further than prefix_lines and this is False.
    """
    code = getattr(func, "__code__", None)
    if code is None or prefix_lines <= 0 or code.co_filename != "<submission>":
        return False
    return _last_line(code) <= prefix_lines


def resolve_entry(namespace: dict, entry: dict | None, prefix_lines: int = 0):
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
        if not _only_from_the_prefix(getattr(holder, func), prefix_lines):
            return lambda: getattr(holder(), func)
        # The class is still holding the prefix's truncated stub. If the model
        # wrote its own copy at module level, bind that to a fresh instance so
        # the ``self`` it kept in the signature has something to be — grading the
        # stub instead is the failure that scored a correct answer 0/2.
        rewritten = namespace.get(func)
        if (
            callable(rewritten)
            and not isinstance(rewritten, type)
            and not _only_from_the_prefix(rewritten, prefix_lines)
        ):
            return lambda: lambda *args, **kwargs: rewritten(holder(), *args, **kwargs)
        raise LookupError(
            f"the submission left {cls}.{func} as the partial solution's truncated stub"
        )

    candidate = namespace.get(func)
    if callable(candidate) and not isinstance(candidate, type):
        return lambda: candidate

    for name, value in namespace.items():
        if name.startswith("__") or not isinstance(value, type):
            continue
        if value.__module__ != "__submission__":
            continue  # a class the preamble imported, not one the submission wrote
        if hasattr(value, func) and not _only_from_the_prefix(getattr(value, func), prefix_lines):
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
    source, prefix_lines = compare.assemble(payload.get("prefix") or "", payload.get("code") or "")
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
            factory = resolve_entry(namespace, entry, prefix_lines)
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
    # ``case`` holds an input and a testtype and no expected output — the parent
    # strips it, so that no frame on this stack has the answer in it while the
    # submission is running. See the module docstring.
    args = compare.parse_functional_args(case.get("input") or "")

    try:
        call_target = factory()
    except (_Deadline, MemoryError):
        raise
    except BaseException as exc:  # noqa: BLE001
        return _failure(None, f"constructing the entry point: {_describe(exc)}")

    sink = CappedStdout()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(CappedStdout(4096)):
            got = call_target(*args)
    except (_Deadline, MemoryError):
        raise
    except BaseException as exc:  # noqa: BLE001 — a submission may raise SystemExit
        return _failure(None, _describe(exc))

    # NO VERDICT HERE. The child reports what the submission RETURNED; the
    # parent decides whether it is correct. See _verifiable.
    try:
        return _verifiable(got, None)
    except RecursionError:
        # A submission that returned a self-referential structure. MAX_DEPTH
        # normally catches this first; this is the case where the recursion is
        # in a __eq__ the submission wrote.
        return _failure(None, "the returned value could not be compared (recursive structure)")


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
                    return _failure(sink.value(), f"exited with status {exc.code}")
    except (_Deadline, MemoryError):
        raise
    except BaseException as exc:  # noqa: BLE001
        return _failure(sink.value(), _describe(exc))
    finally:
        sys.stdin = saved_stdin

    got = sink.value()
    if sink.truncated():
        return _failure(got, f"stdout exceeded {STDOUT_CAPTURE_LIMIT} bytes")
    return _verifiable(got, None, stdout=True)


def _verifiable(got, error: str | None, stdout: bool = False) -> dict:
    """One case, reported so the PARENT can decide it.

    The verdict deliberately does not come from this process. The harness and the
    submission share one interpreter, so a submission can reach into this
    module's globals or rewrite ``compare.values_equal`` — and unlike the forgery
    the nonce guards against, that attack needs no knowledge of the question:

        import sys
        sys.modules['compare'].values_equal = lambda *a, **k: True

    Two lines, and every case passes. The module docstring used to answer this
    with "the questions are secret", which is a defence against a submission that
    KNOWS the answer and no defence at all against one that redefines what
    correct means. So the child now emits the raw value and the parent compares,
    outside any namespace a submission can reach.

    No ``expected`` here, for the other half of the same argument: a value this
    process holds is a value the submission can read off the stack and return.
    The parent renders both sides of a failure from the test case it kept.

    ``value`` is JSON so it survives the pipe, and it is PLAIN json.dumps — no
    ``default=str``. The fallback made every object serialisable by printing it,
    so a submission returning an object whose __str__ happened to spell the
    expected string scored a pass; the outputs these questions have are all JSON
    by construction, so a return that will not serialise is not a right answer we
    failed to read, and reporting it as unverifiable is the honest result.
    """
    record = {
        "got": compare.render(got),
        "error": error,
        "stdout_case": stdout,
    }
    try:
        record["value"] = json.dumps(got)
    except (TypeError, ValueError, RecursionError) as exc:
        record["error"] = error or f"the returned value could not be serialised: {exc}"
    return record


def _failure(got, error: str | None) -> dict:
    """One failed case, with the submission's side rendered for a human to read.

    Rendered through JSON even for a stdin case, where it is already a string and
    passing it through untouched would be the obvious thing. It is the wrong
    thing: the failure a stdin case most often has is a trailing newline or a
    stray space, and untouched text renders those as nothing at all. ``"16\\n"``
    against ``"16"`` is a report somebody can act on. The expected side is
    rendered the same way by the parent, which is the only process that has it.
    """
    return {
        "pass": False,
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


def _strip_answer_key(payload: dict) -> None:
    """Drop every expected output from the request, before anything is executed.

    The parent already sends inputs only, so on the shipped path this finds
    nothing. It runs anyway because the cost of the caller being wrong about that
    is not a bad error message: an ``output`` reaching this process is a value
    sitting in a frame on the submission's own call stack, and three lines of
    frame walking turn it into a perfect score. Enforcing the contract at the
    boundary it protects is what stops a future caller reopening it by accident.
    """
    for case in payload.get("tests") or []:
        if isinstance(case, dict):
            case.pop("output", None)


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
    _strip_answer_key(payload)
    # The raw bytes are a local of this frame, and this frame is on the stack
    # for as long as the submission runs — a walk to it reads the whole request
    # back, stripped payload or not.
    del request
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
