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

import ast
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
    return assemble("", code)[0]


def assemble_source(prefix: str, code: str) -> str:
    """The runnable program only. See assemble for how it is chosen."""
    return assemble(prefix, code)[0]


def assemble(prefix: str, code: str) -> tuple[str, int]:
    """The runnable program, and how many of its leading lines came from the prefix.

    The second number is not decoration. runner.resolve_entry uses it to tell a
    definition the MODEL wrote from the truncated one the prefix already had,
    which are otherwise indistinguishable by name. It is 0 whenever the chosen
    reading does not begin with prefix text.

    With no prefix this is plain unfencing. With one, the submission is usually
    a FRAGMENT — a function body resumed part-way through — and the difference
    matters more than it looks: a fragment never compiles on its own, so
    applying the usual "does it compile" gate to the fragment alone rejects
    every candidate and falls through to returning the raw reply, fences and
    all. Every completion answer would then be a syntax error.

    So the compile test is applied to prefix+candidate, which is the thing that
    actually has to run. That also makes the fenced-block choice meaningful for
    completions: the right block is the one that continues THIS prefix.
    """
    blocks = _FENCE_RE.findall(code)
    # Python-tagged blocks first, then untagged, then everything else — a reply
    # that shows the wrong answer in a ```text block and the real one in
    # ```python must be graded on the ```python one. The grouping is what keeps
    # that priority intact; see _readings for the order WITHIN a group.
    py = ("python", "python3", "py")
    ordered = _readings(blocks, lambda t: t in py)
    ordered += _readings(blocks, lambda t: t == "")
    ordered += _readings(blocks, lambda t: t not in py and t != "")
    candidates = [code] + ordered

    if not prefix:
        for cand in candidates:
            if _compiles(cand):
                return cand, 0
        # Nothing compiled. The best block beats the raw reply, so the failure
        # reads as "this program is wrong" rather than as a fence in a traceback.
        return (ordered[0] if ordered else code), 0

    head, member_indent, head_lines = _open_definition(prefix)
    prefix_lines = prefix.rstrip("\n").count("\n") + 1

    # Four readings of a completion answer, tried in this order, and ALL of them
    # are accepted. That is a deliberate choice about what this benchmark
    # measures: a completion prompt asks for the missing portion only, some
    # models comply and some restate the whole function, and grading only one
    # interpretation scores the other zero. Either way the task stops measuring
    # whether the model can solve the problem and starts measuring whether it
    # followed a formatting instruction — exactly the confound that made the two
    # strongest workers score 0% here and the weakest score highest.
    #
    # Accepting whichever reading actually runs drops the format axis entirely.
    # It cannot manufacture a pass: the tests still have to agree with the
    # reference implementation, and a wrong program fails under every reading.
    for cand in candidates:  # 1. a genuine continuation of the truncated body
        if _continues(cand, member_indent) and _compiles(join_prefix(prefix, cand)):
            return join_prefix(prefix, cand), prefix_lines
    for cand in candidates:  # 2. the truncated method, restated at column 0
        grafted = _graft(head, head_lines, member_indent, cand)
        if grafted is not None and _compiles(grafted[0]):
            return grafted
    for cand in candidates:  # 3. a whole class or module-level function, shadowing
        if _compiles(join_prefix(prefix, cand)):
            return join_prefix(prefix, cand), prefix_lines
    for cand in candidates:  # 4. the reply standing entirely on its own
        if _compiles(cand):
            return cand, 0
    return join_prefix(prefix, ordered[0] if ordered else code), prefix_lines


def join_prefix(prefix: str, body: str) -> str:
    """Append a body to a partial solution, verbatim.

    The task specifies literal appending ("directly appending your code after
    the partial code should produce a correct completion"), so the join must not
    reindent or strip leading whitespace — the fragment's own indentation is
    what places it inside the function. Only the newline between the two is
    normalised, since a fenced block arrives without the prefix's trailing one.
    """
    if not prefix:
        return body
    return prefix.rstrip("\n") + "\n" + body.lstrip("\n")


def _readings(blocks: list[tuple[str, str]], pred) -> list[str]:
    """One tag group's candidates: the whole group joined, then each block alone.

    JOINED FIRST, because a reply's fenced blocks are usually one program cut
    into pieces rather than a list of alternatives. Taking only the last block
    made an "Example usage" demo THE submission: a correct answer followed by a
    ```python block containing ``print(Solution().f(1))`` came back pass:false,
    cases_run:0, "NameError: name 'Solution' is not defined". Reproduced live. A
    reply that puts its imports in one block and its class in the next failed
    the same way, and a model that demonstrates its own answer is not a model
    that wrote no answer.

    A retracted draft followed by a correction still grades as the correction:
    two defs of the same name in one module leave the LATER one bound, which is
    the answer "last block wins" was reaching for — arrived at now by running
    the model's whole reply instead of by picking one piece of it.

    Then each block alone, latest first, for the replies the join cannot run —
    a draft whose module level raises before the correction is ever reached.
    """
    bodies = [body for tag, body in blocks if pred(tag.lower())]
    if len(bodies) < 2:
        return bodies
    return ["".join(b if b.endswith("\n") else b + "\n" for b in bodies)] + list(reversed(bodies))


# A ``def`` line, capturing its indentation. A completion prefix always breaks
# off inside one — that is what makes it a completion prompt — so this is how
# far in a genuine continuation of it has to be.
_DEF_RE = re.compile(r"^([ \t]*)(?:async[ \t]+)?def[ \t]")


def _open_definition(prefix: str) -> tuple[str, int, int]:
    """The prefix up to its last ``def``, that def's indent, and its line count.

    The indent is -1 when the prefix contains no def at all, which makes every
    candidate count as a continuation and leaves such a prefix grading exactly
    as it did before.
    """
    lines = prefix.split("\n")
    for index in range(len(lines) - 1, -1, -1):
        match = _DEF_RE.match(lines[index])
        if not match:
            continue
        start = index
        # A decorator belongs to the def below it, so cutting between the two
        # would leave a dangling @lru_cache that cannot compile.
        while start > 0 and lines[start - 1].lstrip().startswith("@"):
            start -= 1
        return "\n".join(lines[:start]), len(match.group(1).expandtabs(8)), start
    return "", -1, 0


def _indent_of(body: str) -> int | None:
    """The shallowest indentation in a block of code; None if it is all blank."""
    widths = [
        len(line) - len(line.lstrip())
        for line in body.expandtabs(8).split("\n")
        if line.strip()
    ]
    return min(widths) if widths else None


def _continues(body: str, def_indent: int) -> bool:
    """Whether this candidate resumes the prefix's last definition.

    Indentation is the whole test, and it is the one that was missing. A model
    that answers "complete this method" by restating the method at COLUMN 0
    produces a join that still compiles — the restatement lands at module level
    beside the class, and the class keeps the truncated stub the prefix ended on.
    resolve_entry then grades the stub, which returns None for every case: the
    run comes back 0/N with error:null, indistinguishable from a wrong answer.
    That is the shape that scored the fleet's strongest worker 0/50 on
    coding_completion while its weakest scored 8/50.
    """
    if def_indent < 0:
        return True  # there is no open definition to fall out of
    indent = _indent_of(body)
    return indent is None or indent > def_indent


def _graft(head: str, head_lines: int, indent: int, body: str) -> tuple[str, int] | None:
    """The prefix with its truncated definition REPLACED by a restated one.

    For the answer shape _continues rejects: the model wrote the whole method
    out again at column 0. Appending that leaves it at module level, where it is
    dead code — and, still carrying ``self``, uncallable on its own — so it is
    re-indented back into the class body instead and the stub it replaces is
    dropped. Everything above that def survives: the imports, the class header,
    any earlier method.

    Replacing rather than appending is what makes it work on a prefix cut
    mid-block. A prefix ending on ``for i in range(n):`` has an open suite, and
    a second def appended after it is an IndentationError.

    None when the candidate is not that shape. A restated CLASS, or a restated
    module-level function, must NOT be re-indented: at column 0 it already
    shadows the prefix's own, which is the reading that has always worked.
    """
    if indent < 0 or not _restates_a_method(body):
        return None
    delta = indent - (_indent_of(body) or 0)
    if delta < 0:
        return None
    pad = " " * delta
    shifted = "\n".join(pad + line if line.strip() else line for line in body.split("\n"))
    if not head.strip():
        return shifted, 0
    return head.rstrip("\n") + "\n" + shifted.lstrip("\n"), head_lines


def _restates_a_method(body: str) -> bool:
    """Whether this candidate is one or more bare METHODS and nothing else.

    Read off the parse tree rather than matched with a regex, because the thing
    that decides it is the first parameter: a top-level ``def f(self, n)`` only
    means anything inside a class, while ``def f(n)`` at column 0 is a complete
    module-level answer and has to be left exactly where the model put it.
    """
    try:
        tree = ast.parse(body)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    found = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        args = node.args.posonlyargs + node.args.args
        if not args or args[0].arg not in ("self", "cls"):
            return False
        found = True
    return found


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

    The second pass is applied LINE BY LINE, not to the whole output as one bag
    of tokens. Flattening it passed "1 2 3" against an expected "1\\n2\\n3" — a
    program that printed one line where three were asked for — because the
    tolerance pass was reached precisely when the line structure had already
    failed to match, and then ignored the thing that had just failed.
    """
    left, right = _normalise_lines(expected), _normalise_lines(got)
    if left == right:
        return True
    if len(left) != len(right):
        return False
    return all(_tokens_match(a, b) for a, b in zip(left, right))


def _normalise_lines(text: str) -> list[str]:
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in unified.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _tokens_match(expected: str, got: str) -> bool:
    """One LINE of expected output against one line of actual, token by token."""
    left, right = expected.split(), got.split()
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a == b:
            continue
        # An INTEGER expectation is compared exactly. The module docstring has
        # always said so and values_equal implements it, but this path did not:
        # the float tolerance was applied to every token, so a stdout answer of
        # 1000001 passed an expected 1000000, and 0.0000001 passed an expected 0.
        # A tolerance exists for accumulated floating-point error; an integer
        # result that is off by one is a wrong answer, not a rounding artefact.
        if _INT_RE.match(a) and _INT_RE.match(b):
            return False
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if math.isnan(fa) or math.isnan(fb):
            return False
        # An integer expectation against a non-integer answer is still exact:
        # tolerating it would re-admit the same class through the other door.
        if _INT_RE.match(a) and fa != fb:
            return False
        if not math.isclose(fa, fb, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL):
            return False
    return True


# _INT_RE marks a token that is an integer literal, which must match exactly.
_INT_RE = re.compile(r"^[+-]?\d+$")


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
