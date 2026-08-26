"""The two encodings a test case has, and what it means for an answer to match.

LiveBench's coding rows carry LiveCodeBench test cases, and those are strings on
both sides — ``input`` and ``output`` are text, whatever the problem's actual
types are. This file owns the decode in one direction and the comparison in the
other, and they belong together because they are the same contract read twice:
whatever ``input`` is parsed AS is what ``output`` has to be compared AS.

Nothing here executes anything or touches the operating system. It is imported
by the jailed child, and it is also the half of the grader that can be unit
tested without a subprocess, which is why it is a file of its own.

COMPARISON, and the four decisions in it that are not obvious:

  Tuples equal lists. A Python solution that returns ``(3, 5)`` where the
  expected JSON is ``[3, 5]`` is right, and ``[3,5] == (3,5)`` is False. LeetCode
  compares by value, not by container type, and so does this.

  Booleans are NOT integers. ``True == 1`` in Python, so a submission that
  returned True where the answer is 1 would grade as correct under a plain ==.
  It is not correct: those are different answers to different questions, and the
  conflation only ever produces false passes.

  A float tolerance, applied only when the EXPECTED value is a float. Problems
  whose answer is a division or a square root are graded to about 1e-5 on
  LeetCode, and an exact == on those is a coin flip on the last bit. But
  applying the same relative tolerance to integers is a live source of false
  passes — isclose(1000000, 1000001, rel_tol=1e-6) is True — so an integer
  expectation is compared exactly, which still lets an integral float like 2.0
  match a 2.

  No unwrap-the-single-element fallback. LiveCodeBench's own checker retries a
  failed comparison against ``expected[0]``, which exists to paper over its
  line-list encoding of stdin outputs. Carried over to functional cases it turns
  "the answer is the list [5]" and "the answer is the number 5" into the same
  answer, and those are different answers. LiveBench's functional ``output`` is
  the JSON-encoded return value directly, so the fallback buys nothing and costs
  a class of false pass.
"""

from __future__ import annotations

import json
import math
import re
import reprlib

# Matches LeetCode's stated float tolerance closely enough (it accepts 1e-5)
# while staying tight enough that two genuinely different answers cannot land
# inside it. Applied to floats only — see the module docstring.
FLOAT_REL_TOL = 1e-6
FLOAT_ABS_TOL = 1e-6

# Structures nested deeper than this are not test data, they are a submission
# returning something self-referential. Bounded rather than caught, so the
# comparison ends in a grading result instead of a RecursionError that has to be
# distinguished from the submission's own.
MAX_DEPTH = 60

# How much of an expected/actual value travels back in first_failure. Enough to
# read a wrong answer at a glance, short enough that a submission returning a
# million-element list cannot make the HTTP response the problem.
RENDER_LIMIT = 2048

_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


class TestDataError(ValueError):
    """The test case could not be decoded — a fault in the question, not the answer."""


# ── Decoding a test case ────────────────────────────────────────────────────


def parse_functional_args(raw: str) -> list:
    """Arguments for a LeetCode-style call: one JSON value per line.

    The line-wise read is the format LiveCodeBench actually emits and is what
    makes a bare ``5`` on its own line an integer argument rather than a syntax
    error. The whole-blob fallback exists for the rows where a value was
    pretty-printed across several lines: raw_decode walks values off the front
    of the string and does not care where the newlines fell, so it reads those
    correctly, and it is second because it would ALSO happily read ``1 2`` as
    two arguments on one line, which the line-wise form correctly rejects.
    """
    lines = [line for line in raw.split("\n") if line.strip()]
    try:
        return [json.loads(line) for line in lines]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    args = []
    index = 0
    text = raw.strip()
    try:
        while index < len(text):
            value, index = decoder.raw_decode(text, index)
            args.append(value)
            while index < len(text) and text[index] in " \t\r\n,":
                index += 1
    except json.JSONDecodeError as exc:
        raise TestDataError(f"could not decode test input as JSON arguments: {exc}") from exc
    return args


def parse_expected(raw: str):
    """The expected return value of a functional case.

    A row whose ``output`` is not valid JSON is a data fault and is raised as
    one. Falling back to "compare it as a string" was the alternative and is
    worse than useless: every submission would then fail the case with a
    mismatch that reads like a wrong answer.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TestDataError(f"could not decode expected output as JSON: {exc}") from exc


def extract_source(code: str) -> str:
    """The submission, with markdown fences removed when they are in the way.

    Gated on the code not compiling, rather than on a fence being present. That
    ordering is the whole point: a submission that already parses is used
    verbatim, so a solution containing a ``\\`\\`\\`` inside a docstring is never
    rewritten, and the unfencing can only ever turn a certain SyntaxError into a
    possible pass. A model that wraps its answer in a code block is answering
    the question; scoring it zero measures its formatting.
    """
    if _compiles(code):
        return code

    blocks = _FENCE_RE.findall(code)
    # Python-tagged blocks first, then untagged, then everything else — a reply
    # that shows the wrong answer in a ```text block and the real one in
    # ```python must be graded on the ```python one.
    ordered = [body for tag, body in blocks if tag.lower() in ("python", "python3", "py")]
    ordered += [body for tag, body in blocks if tag == ""]
    ordered += [body for tag, body in blocks if tag.lower() not in ("python", "python3", "py", "")]
    for body in ordered:
        if _compiles(body):
            return body
    return code


def _compiles(source: str) -> bool:
    try:
        compile(source, "<submission>", "exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


# ── Comparing an answer ─────────────────────────────────────────────────────


def values_equal(expected, got, depth: int = 0) -> bool:
    """Value equality for a functional test case. See the module docstring."""
    if depth > MAX_DEPTH:
        return False

    if isinstance(expected, bool) or isinstance(got, bool):
        # Both branches: a bool expectation needs a bool answer, and a bool
        # answer is wrong wherever a number was expected.
        return isinstance(expected, bool) and isinstance(got, bool) and expected == got

    if expected is None:
        return got is None

    if isinstance(expected, int):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and got == expected

    if isinstance(expected, float):
        if not isinstance(got, (int, float)) or isinstance(got, bool):
            return False
        if math.isnan(expected) or math.isnan(got):
            return math.isnan(expected) and math.isnan(got)
        return math.isclose(got, expected, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL)

    if isinstance(expected, str):
        return isinstance(got, str) and expected == got

    if isinstance(expected, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(expected) != len(got):
            return False
        return all(values_equal(a, b, depth + 1) for a, b in zip(expected, got))

    if isinstance(expected, dict):
        if not isinstance(got, dict):
            return False
        if len(expected) != len(got):
            return False
        for key, value in expected.items():
            if key in got:
                other = got[key]
            else:
                # JSON has no non-string keys, so a solution returning a dict
                # keyed by int is right about the answer and wrong only about
                # the encoding it came back through. Matched by the string form
                # rather than by coercing every key, so {"1":…} and {1:…} in the
                # SAME dict still count as two keys and still fail.
                match = [k for k in got if str(k) == key]
                if len(match) != 1:
                    return False
                other = got[match[0]]
            if not values_equal(value, other, depth + 1):
                return False
        return True

    return expected == got


def compare_stdout(expected: str, got: str) -> bool:
    """Output equality for a stdin-driven case.

    Two passes, in this order:

      Line-wise, with trailing whitespace and trailing blank lines dropped. That
      is the whole of it for the overwhelming majority of cases, and it forgives
      exactly the two things nobody means to test — a missing final newline and
      a trailing space after the last field.

      Then token-wise with a float tolerance, for the problems whose answer is a
      real number. "0.3333333333" against "0.33333333333" is the same answer,
      and a checker that says otherwise is grading print formatting. The token
      counts must still match, so this cannot rescue an answer with the wrong
      number of values in it.
    """
    if _normalise_lines(expected) == _normalise_lines(got):
        return True
    return _tokens_match(expected, got)


def _normalise_lines(text: str) -> list[str]:
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in unified.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _tokens_match(expected: str, got: str) -> bool:
    left, right = expected.split(), got.split()
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a == b:
            continue
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if math.isnan(fa) or math.isnan(fb):
            return False
        if not math.isclose(fa, fb, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL):
            return False
    return True


# ── Rendering a value for the failure report ────────────────────────────────

_REPR = reprlib.Repr()
_REPR.maxstring = 256
_REPR.maxother = 256
_REPR.maxlist = _REPR.maxtuple = _REPR.maxset = _REPR.maxdict = 32
_REPR.maxlevel = 6


def render(value, limit: int = RENDER_LIMIT) -> str:
    """A bounded, safe string for an arbitrary object a submission returned.

    iterencode rather than dumps, because the value is under the submission's
    control: dumps on a returned ten-million-element list builds the whole
    string before anything can truncate it, and inside a jail with an address
    space limit that turns a wrong answer into a MemoryError. The generator form
    stops as soon as there is enough to show.

    reprlib is the fallback rather than a bare repr for the same reason, plus
    one more: repr() runs the submission's own __repr__, and reprlib bounds how
    much of the result is kept even when that method is hostile. It cannot bound
    how long the method RUNS for — that is what the CPU limit is for.
    """
    try:
        chunks: list[str] = []
        size = 0
        for chunk in json.JSONEncoder(default=_unencodable, ensure_ascii=False).iterencode(value):
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                break
        text = "".join(chunks)
    except Exception:
        try:
            text = _REPR.repr(value)
        except Exception:
            return "<unrepresentable>"
    return _truncate(text, limit)


def _unencodable(value) -> str:
    return f"<{type(value).__name__}>"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… (+{len(text) - limit} chars)"
