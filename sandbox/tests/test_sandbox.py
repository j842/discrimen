"""Tests for the grading sandbox.

Two halves, and they are testing different kinds of claim:

  The COMPARISON tests are ordinary unit tests. They pin the semantics in
  compare.py — tuple/list equivalence, int-versus-float, the bool rule, the
  float tolerance and where it does NOT apply — because those are the decisions
  most likely to be "simplified" later by someone who reads `==` and assumes it
  was an oversight.

  The CONTAINMENT tests are not really unit tests at all. Each one runs a
  genuinely hostile submission — an infinite loop, a fork bomb, a ten-gigabyte
  allocation — and asserts three things every time: that the answer came back,
  that it came back as a clean pass:false rather than an exception, and that
  nothing was left running afterwards. The third assertion is the one that
  matters most and the one a passing grade would not otherwise notice, so it is
  factored into assert_no_leaked_processes and applied everywhere.

Run with either runner:

    python3 -m unittest discover -s sandbox/tests -v
    pytest sandbox/tests
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
import zlib
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compare  # noqa: E402
import main  # noqa: E402
import supervisor  # noqa: E402


# ── helpers ─────────────────────────────────────────────────────────────────


def grade(code: str, tests: list[dict], func: str = "f", cls: str = "Solution", **kwargs) -> dict:
    request = {
        "language": "python",
        "code": code,
        "entry": {"class": cls, "func": func},
        "tests": tests,
        "timeout_ms": kwargs.pop("timeout_ms", 5000),
    }
    request.update(kwargs)
    return main.handle_grade(request)


def case(input_: str, output: str, testtype: str = "functional") -> dict:
    return {"input": input_, "output": output, "testtype": testtype}


def live_runner_pids() -> set[int]:
    """Every process on the box currently running our runner.py.

    Read out of /proc rather than shelled out to pgrep, because a pgrep pattern
    also matches the shell that ran it — which is how the first version of this
    check reported three leaked processes that were all the check itself.
    """
    pids = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                argv = handle.read().split(b"\0")
        except OSError:
            continue
        if any(part.endswith(b"/runner.py") for part in argv):
            pids.add(int(entry))
    return pids


class SandboxCase(unittest.TestCase):
    def setUp(self) -> None:
        self._before = live_runner_pids()

    def tearDown(self) -> None:
        self.assert_no_leaked_processes()

    def assert_no_leaked_processes(self) -> None:
        """Nothing this test started is still running.

        A small settle window, because a killed process is a zombie for however
        long it takes its parent to reap it, and a zombie is still a /proc entry.
        Anything still there after half a second is not being reaped, it is
        running.
        """
        for _ in range(50):
            leaked = live_runner_pids() - self._before
            if not leaked:
                return
            time.sleep(0.01)
        self.fail(f"leaked sandbox processes: {sorted(leaked)}")


# ── comparison semantics ────────────────────────────────────────────────────


class TestCompletionPrefix(unittest.TestCase):
    """A coding_completion answer is a FRAGMENT, graded as prefix+fragment.

    These pin the bug that made the task measure formatting instead of ability:
    the fragment does not compile alone, so the ordinary "does it compile" gate
    rejected every candidate and fell through to the raw reply — scoring the two
    strongest workers 0% on this task while the weakest, which ignored the
    instruction and restated the whole function, scored highest.
    """

    PREFIX = (
        "class Solution(object):\n"
        "    def f(self, n):\n"
        "        total = 0\n"
        "        for i in range(n):"
    )

    def _f5(self, answer: str):
        ns = {}
        exec(compile(compare.assemble_source(self.PREFIX, answer), "<s>", "exec"), ns)
        return ns["Solution"]().f(5)

    def test_compliant_fenced_fragment(self):
        self.assertEqual(self._f5("Sure:\n```python\n            total += i\n        return total\n```"), 10)

    def test_bare_fragment_without_fences(self):
        self.assertEqual(self._f5("            total += i\n        return total"), 10)

    def test_disobedient_whole_function_also_accepted(self):
        # Restating the whole class is not the requested format, but it is a
        # correct solution; grading it zero would measure instruction-following.
        whole = "```python\nclass Solution(object):\n    def f(self, n):\n        return sum(range(n))\n```"
        self.assertEqual(self._f5(whole), 10)

    def test_prose_around_the_block_is_ignored(self):
        self.assertEqual(
            self._f5("Here you go:\n```\n            total += i\n        return total\n```\nHope that helps!"),
            10,
        )

    def test_wrong_fragment_still_wrong(self):
        # The join must not manufacture a pass out of a wrong answer.
        self.assertEqual(self._f5("```python\n            total += i * 2\n        return total\n```"), 20)

    def test_no_prefix_is_unchanged(self):
        # Every non-completion question must grade exactly as before.
        self.assertEqual(compare.assemble_source("", "x = 1"), "x = 1")
        self.assertEqual(compare.extract_source("```python\nx = 1\n```").strip(), "x = 1")

    def test_indentation_is_preserved_verbatim(self):
        # The task specifies literal appending; reindenting would change meaning.
        out = compare.assemble_source(self.PREFIX, "            total += i\n        return total")
        self.assertIn("\n            total += i\n", out)




class TestComparison(unittest.TestCase):
    def test_json_spacing_is_irrelevant(self):
        # The whole point of comparing parsed values rather than strings.
        self.assertTrue(compare.values_equal(json.loads("[1,2]"), json.loads("[1, 2]")))

    def test_tuple_matches_list(self):
        self.assertTrue(compare.values_equal([3, 5], (3, 5)))
        self.assertTrue(compare.values_equal([[1, 2], [3]], ((1, 2), (3,))))

    def test_integral_float_matches_int(self):
        self.assertTrue(compare.values_equal(2, 2.0))
        self.assertTrue(compare.values_equal([1, 2], [1.0, 2.0]))

    def test_near_miss_integer_does_not_match(self):
        # The reason the float tolerance is not applied to an int expectation:
        # isclose(1000000, 1000001, rel_tol=1e-6) is True, and these are two
        # different answers.
        self.assertFalse(compare.values_equal(1000000, 1000001))
        self.assertFalse(compare.values_equal(1000000, 1000001.0))

    def test_float_tolerance_applies_to_float_expectations(self):
        self.assertTrue(compare.values_equal(0.3333333333, 0.33333333333))
        self.assertFalse(compare.values_equal(0.5, 0.6))

    def test_bool_is_not_an_integer(self):
        self.assertFalse(compare.values_equal(1, True))
        self.assertFalse(compare.values_equal(0, False))
        self.assertFalse(compare.values_equal(True, 1))
        self.assertTrue(compare.values_equal(True, True))

    def test_no_single_element_unwrapping(self):
        # LiveCodeBench's checker would pass this. It should not: "the answer is
        # the list [5]" and "the answer is 5" are different answers.
        self.assertFalse(compare.values_equal([5], 5))
        self.assertFalse(compare.values_equal(5, [5]))

    def test_nested_and_dict(self):
        self.assertTrue(compare.values_equal({"a": [1, 2]}, {"a": (1, 2)}))
        self.assertFalse(compare.values_equal({"a": 1}, {"a": 1, "b": 2}))
        # A dict keyed by int is right about the answer, wrong only about the
        # encoding JSON could carry it in.
        self.assertTrue(compare.values_equal({"1": "x"}, {1: "x"}))

    def test_order_matters_in_lists(self):
        self.assertFalse(compare.values_equal([1, 2], [2, 1]))

    def test_stdout_forgives_trailing_whitespace_only(self):
        self.assertTrue(compare.compare_stdout("16", "16\n"))
        self.assertTrue(compare.compare_stdout("1\n2", "1 \n2\n\n"))
        self.assertFalse(compare.compare_stdout("1\n2", "1\n3"))
        self.assertFalse(compare.compare_stdout("16", "1 6"))

    def test_stdout_float_tolerance(self):
        self.assertTrue(compare.compare_stdout("0.3333333333", "0.33333333333"))
        self.assertFalse(compare.compare_stdout("0.33", "0.44"))
        # Token counts must still match, so a tolerance cannot rescue an answer
        # with the wrong number of values in it.
        self.assertFalse(compare.compare_stdout("1 2", "1 2 3"))

    def test_functional_arguments_are_one_json_value_per_line(self):
        self.assertEqual(compare.parse_functional_args('[1,4,3,1]\n5\n"abc"'), [[1, 4, 3, 1], 5, "abc"])
        self.assertEqual(compare.parse_functional_args("[1,\n 2]"), [[1, 2]])
        with self.assertRaises(compare.TestDataError):
            compare.parse_functional_args("not json at all")

    def test_render_is_bounded(self):
        rendered = compare.render(list(range(100000)), limit=200)
        self.assertLessEqual(len(rendered), 260)

    def test_render_survives_a_hostile_repr(self):
        class Hostile:
            def __repr__(self):
                raise RuntimeError("no")

        self.assertIsInstance(compare.render([Hostile()]), str)

    def test_fences_are_stripped_only_when_the_code_will_not_compile(self):
        fenced = "Here you go:\n```python\nclass Solution:\n    def f(self): return 1\n```\n"
        self.assertIn("class Solution", compare.extract_source(fenced))
        self.assertNotIn("Here you go", compare.extract_source(fenced))
        # Code that already parses is returned byte for byte, so a docstring
        # containing a fence is never rewritten.
        intact = 'x = """\n```python\n"""\n'
        self.assertEqual(compare.extract_source(intact), intact)


# ── the two test types ──────────────────────────────────────────────────────


class TestGrading(SandboxCase):
    def test_correct_solution_passes(self):
        result = grade(
            "class Solution:\n"
            "    def minimumArrayLength(self, nums: List[int]) -> int:\n"
            "        m = min(nums)\n"
            "        if any(x % m for x in nums):\n"
            "            return 1\n"
            "        return (nums.count(m) + 1) // 2\n",
            [case("[1,4,3,1]", "1"), case("[5,5,5,10,5]", "2"), case("[2,3,4]", "1")],
            func="minimumArrayLength",
        )
        self.assertTrue(result["pass"], result)
        self.assertEqual(result["cases_run"], 3)
        self.assertEqual(result["cases_passed"], 3)
        self.assertIsNone(result["first_failure"])
        self.assertIsNone(result["error"])

    def test_wrong_answer_fails_and_names_the_case(self):
        result = grade(
            "class Solution:\n    def f(self, n):\n        return n * 2\n",
            [case("2", "4"), case("3", "9"), case("4", "8")],
        )
        self.assertFalse(result["pass"])
        # Every case runs, so cases_passed is a real fraction rather than
        # "cases_run minus one".
        self.assertEqual(result["cases_run"], 3)
        self.assertEqual(result["cases_passed"], 2)
        self.assertEqual(result["first_failure"]["index"], 1)
        self.assertEqual(result["first_failure"]["expected"], "9")
        self.assertEqual(result["first_failure"]["got"], "6")
        self.assertIsNone(result["error"])

    def test_stop_on_first_failure_short_circuits(self):
        result = grade(
            "class Solution:\n    def f(self, n):\n        return 0\n",
            [case("1", "1"), case("2", "2"), case("3", "3")],
            stop_on_first_failure=True,
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["cases_run"], 1)

    def test_stdin_testtype(self):
        result = main.handle_grade(
            {
                "code": "n = int(input())\nprint(n * n)\n",
                "tests": [case("4\n", "16", "stdin"), case("5\n", "25", "stdin")],
                "timeout_ms": 5000,
            }
        )
        self.assertTrue(result["pass"], result)
        self.assertEqual(result["cases_passed"], 2)

    def test_stdin_case_gets_a_fresh_program_each_time(self):
        # A module reads its stdin once. If the submission were loaded once and
        # reused, case two would see an empty stream.
        result = main.handle_grade(
            {
                "code": "import sys\nprint(sum(int(x) for x in sys.stdin.read().split()))\n",
                "tests": [case("1 2 3", "6", "stdin"), case("10 20", "30", "stdin")],
                "timeout_ms": 5000,
            }
        )
        self.assertTrue(result["pass"], result)

    def test_stdin_wrong_answer(self):
        result = main.handle_grade(
            {
                "code": "print(0)\n",
                "tests": [case("4\n", "16", "stdin")],
                "timeout_ms": 5000,
            }
        )
        self.assertFalse(result["pass"])
        # Both sides come back JSON-rendered, so the trailing newline that is
        # the whole substance of most stdin mismatches is visible in the report.
        self.assertEqual(json.loads(result["first_failure"]["got"]), "0\n")
        self.assertEqual(json.loads(result["first_failure"]["expected"]), "16")

    def test_mixed_testtypes_in_one_request(self):
        result = main.handle_grade(
            {
                "code": "class Solution:\n    def f(self, n):\n        return n + 1\n",
                "entry": {"class": "Solution", "func": "f"},
                "tests": [case("1", "2"), case("", "", "stdin")],
                "timeout_ms": 5000,
            }
        )
        # The functional case passes; the stdin case runs the same source as a
        # program, which prints nothing, which is what it was told to expect.
        self.assertTrue(result["pass"], result)

    def test_entry_point_is_found_when_the_model_renamed_the_class(self):
        result = grade(
            "class MySolution:\n    def f(self, n):\n        return n + 1\n",
            [case("1", "2")],
        )
        self.assertTrue(result["pass"], result)

    def test_entry_point_is_found_at_module_level(self):
        result = grade("def f(n):\n    return n + 1\n", [case("1", "2")])
        self.assertTrue(result["pass"], result)

    def test_missing_entry_point_is_a_clean_failure(self):
        result = grade("class Solution:\n    def other(self, n):\n        return n\n", [case("1", "1")])
        self.assertFalse(result["pass"])
        self.assertEqual(result["cases_run"], 0)
        self.assertIn("defines no", result["error"])

    def test_solution_may_not_carry_state_between_cases(self):
        # A fresh instance per case. With one shared instance this returns the
        # first case's answer forever and case two passes by accident.
        result = grade(
            "class Solution:\n"
            "    def __init__(self):\n"
            "        self.seen = None\n"
            "    def f(self, n):\n"
            "        if self.seen is None:\n"
            "            self.seen = n\n"
            "        return self.seen\n",
            [case("1", "1"), case("2", "2")],
        )
        self.assertTrue(result["pass"], result)

    def test_markdown_fenced_answer_is_graded(self):
        result = grade(
            "Sure! Here is the solution:\n\n```python\nclass Solution:\n    def f(self, n):\n        return n + 1\n```\n",
            [case("1", "2")],
        )
        self.assertTrue(result["pass"], result)


# ── containment ─────────────────────────────────────────────────────────────


class TestContainment(SandboxCase):
    def test_syntax_error_is_a_clean_fail(self):
        result = grade("class Solution:\n  def f(self, n)\n    return n\n", [case("1", "1")])
        self.assertFalse(result["pass"])
        self.assertEqual(result["cases_run"], 0)
        self.assertIn("SyntaxError", result["error"])

    def test_exception_fails_only_its_own_case(self):
        result = grade(
            "class Solution:\n    def f(self, n):\n        if n == 2:\n            raise ValueError('boom')\n        return n\n",
            [case("1", "1"), case("2", "2"), case("3", "3")],
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["cases_run"], 3)
        self.assertEqual(result["cases_passed"], 2)
        self.assertIn("ValueError", result["first_failure"]["error"])

    def test_nonzero_exit_is_a_failure_not_a_crash(self):
        result = main.handle_grade(
            {
                "code": "import sys\nsys.exit(3)\n",
                "tests": [case("", "", "stdin")],
                "timeout_ms": 5000,
            }
        )
        self.assertFalse(result["pass"])
        self.assertIn("status 3", result["first_failure"]["error"])

    def test_infinite_loop_is_killed(self):
        started = time.monotonic()
        result = grade(
            "class Solution:\n    def f(self, n):\n        while True:\n            pass\n",
            [case("1", "1")],
            timeout_ms=2000,
        )
        elapsed = time.monotonic() - started
        self.assertFalse(result["pass"])
        self.assertEqual(result["cases_passed"], 0)
        self.assertIsNotNone(result["error"])
        # Killed on its own budget, not left to some far larger backstop.
        self.assertLess(elapsed, 10.0, f"took {elapsed:.1f}s to kill a 2s budget")

    def test_sleeping_submission_is_killed_by_the_wall_clock(self):
        # The case no rlimit covers: it consumes no CPU and allocates nothing,
        # so only the parent's deadline ends it.
        started = time.monotonic()
        result = grade(
            "import time\nclass Solution:\n    def f(self, n):\n        time.sleep(600)\n",
            [case("1", "1")],
            timeout_ms=2000,
        )
        elapsed = time.monotonic() - started
        self.assertFalse(result["pass"])
        self.assertIn("limit exceeded", result["error"])
        self.assertLess(elapsed, 10.0)

    def test_memory_bomb_is_contained(self):
        result = grade(
            "class Solution:\n"
            "    def f(self, n):\n"
            "        blocks = []\n"
            "        while True:\n"
            "            blocks.append(bytearray(50_000_000))\n",
            [case("1", "1")],
            timeout_ms=8000,
            memory_mb=256,
        )
        self.assertFalse(result["pass"])
        self.assertIsNotNone(result["error"])
        self.assertIn("memory", result["error"].lower())

    def test_fork_bomb_is_contained(self):
        result = grade(
            "import os\n"
            "class Solution:\n"
            "    def f(self, n):\n"
            "        while True:\n"
            "            try:\n"
            "                os.fork()\n"
            "            except OSError:\n"
            "                pass\n",
            [case("1", "1")],
            timeout_ms=3000,
        )
        self.assertFalse(result["pass"])
        self.assertIsNotNone(result["error"])
        # tearDown's leak check is the real assertion here.

    def test_forked_children_do_not_outlive_a_successful_grade(self):
        # The nastier shape: the submission answers correctly and leaves
        # sleepers behind. Nothing about a clean exit reaps them, so the kill
        # has to happen on the success path too.
        result = grade(
            "import os, time\n"
            "class Solution:\n"
            "    def f(self, n):\n"
            "        for _ in range(4):\n"
            "            if os.fork() == 0:\n"
            "                time.sleep(300)\n"
            "                os._exit(0)\n"
            "        return n\n",
            [case("1", "1")],
            timeout_ms=5000,
        )
        self.assertTrue(result["pass"], result)

    def test_a_child_that_escapes_the_process_group_is_still_reaped(self):
        # setsid() puts it in a group killpg was never told about. The /proc
        # environ sweep is the only thing that finds it.
        result = grade(
            "import os, time\n"
            "class Solution:\n"
            "    def f(self, n):\n"
            "        if os.fork() == 0:\n"
            "            os.setsid()\n"
            "            time.sleep(300)\n"
            "            os._exit(0)\n"
            "        return n\n",
            [case("1", "1")],
            timeout_ms=5000,
        )
        self.assertTrue(result["pass"], result)

    def test_stdout_flood_does_not_wedge_the_service(self):
        # Two floods in one: print() goes through the captured stream, and a
        # direct write to fd 1 bypasses it entirely. Both have to be harmless,
        # and the correct answer still has to come back.
        result = grade(
            "import os\n"
            "class Solution:\n"
            "    def f(self, n):\n"
            "        for _ in range(20000):\n"
            "            print('x' * 500)\n"
            "            os.write(1, b'y' * 500)\n"
            "        return n + 1\n",
            [case("1", "2")],
            timeout_ms=15000,
        )
        self.assertTrue(result["pass"], result)
        # And the service still answers afterwards.
        self.assertTrue(grade("class Solution:\n    def f(self, n):\n        return n\n", [case("7", "7")])["pass"])

    def test_stdin_case_reports_a_flood_rather_than_buffering_it(self):
        result = main.handle_grade(
            {
                "code": "print('z' * 5_000_000)\n",
                "tests": [case("", "z", "stdin")],
                "timeout_ms": 10000,
            }
        )
        self.assertFalse(result["pass"])
        self.assertIn("exceeded", result["first_failure"]["error"])

    def test_network_access_is_refused(self):
        result = grade(
            "import socket\n"
            "class Solution:\n"
            "    def f(self, n):\n"
            "        s = socket.socket()\n"
            "        s.connect(('1.1.1.1', 80))\n"
            "        return n\n",
            [case("1", "1")],
            timeout_ms=8000,
        )
        self.assertFalse(result["pass"])
        self.assertIn("Error", result["first_failure"]["error"])

    def test_importing_ssl_and_urllib_still_works(self):
        # Regression: blocking the network by replacing socket.socket with a
        # plain function makes `class SSLSocket(socket)` in ssl.py call that
        # function as a metaclass, and every submission that imports urllib dies
        # at load time with a TypeError about function() arguments.
        result = grade(
            "import ssl, urllib.request, http.client\n"
            "class Solution:\n    def f(self, n):\n        return n + 1\n",
            [case("1", "2")],
        )
        self.assertTrue(result["pass"], result)

    def test_the_scratch_directory_is_removed(self):
        result = grade(
            "import os\n"
            "class Solution:\n"
            "    def f(self, n):\n"
            "        open('scratch.txt', 'w').write('x')\n"
            "        os.mkdir('locked')\n"
            "        os.chmod('locked', 0)\n"
            "        return os.getcwd()\n",
            [case("1", '""')],
        )
        # The answer is wrong (a path is not ""), which is how the cwd gets
        # reported back so the test can check it is gone.
        cwd = json.loads(result["first_failure"]["got"])
        self.assertFalse(os.path.exists(cwd), f"{cwd} survived the run")

    def test_the_submission_cannot_see_the_service_environment(self):
        result = grade(
            "import os, json\n"
            "class Solution:\n"
            "    def f(self, n):\n"
            "        return sorted(os.environ)\n",
            [case("1", "[]")],
        )
        seen = json.loads(result["first_failure"]["got"])
        self.assertNotIn("SANDBOX_TOKEN", seen)
        self.assertNotIn("PYTHONPATH", seen)

    def test_recursion_is_deep_enough_for_real_solutions(self):
        result = grade(
            "class Solution:\n"
            "    def f(self, n):\n"
            "        def down(k):\n"
            "            return 0 if k == 0 else 1 + down(k - 1)\n"
            "        return down(n)\n",
            [case("5000", "5000")],
        )
        self.assertTrue(result["pass"], result)

    def test_runaway_recursion_is_a_failure_not_a_crash(self):
        result = grade(
            "class Solution:\n    def f(self, n):\n        return self.f(n)\n",
            [case("1", "1")],
        )
        self.assertFalse(result["pass"])
        self.assertIsNotNone(result["first_failure"])

    def test_concurrent_grades_do_not_interfere(self):
        results: list[dict] = []
        lock = threading.Lock()

        def one(value: int) -> None:
            outcome = grade(
                "class Solution:\n    def f(self, n):\n        return n * n\n",
                [case(str(value), str(value * value))],
            )
            with lock:
                results.append(outcome)

        threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(r["pass"] for r in results), results)


# ── private test decoding ───────────────────────────────────────────────────


def encode_private(obj) -> str:
    """The shape LiveBench ships: base64 of zlib of a pickle of a JSON string."""
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(obj)))).decode()


class Detonator:
    """A pickle that runs code when it is loaded, which is what pickles do."""

    def __reduce__(self):
        return (eval, ("__import__('socket').socket()",))


class Spinner:
    def __reduce__(self):
        return (eval, ("[x for x in iter(int, 1)]",))


class TestPrivateTests(SandboxCase):
    def test_round_trip(self):
        cases = [
            {"input": "[1,2,3]", "output": "6", "testtype": "functional"},
            {"input": "5\n", "output": "25", "testtype": "stdin"},
        ]
        result = main.handle_decode({"blob": encode_private(cases)})
        self.assertEqual(result["tests"], cases)

    def test_a_list_pickled_directly_is_also_accepted(self):
        cases = [{"input": "1", "output": "1", "testtype": "functional"}]
        blob = base64.b64encode(zlib.compress(pickle.dumps(cases))).decode()
        self.assertEqual(main.handle_decode({"blob": blob})["tests"], cases)

    def test_decoded_cases_can_be_graded(self):
        blob = encode_private([{"input": "3", "output": "9", "testtype": "functional"}])
        tests = main.handle_decode({"blob": blob})["tests"]
        result = grade("class Solution:\n    def f(self, n):\n        return n * n\n", tests)
        self.assertTrue(result["pass"], result)

    def test_a_hostile_pickle_is_executed_inside_the_jail(self):
        # It DOES execute — that is unavoidable and is the reason the endpoint
        # exists — but it executes with no network, so the payload fails and the
        # failure is reported rather than served.
        blob = base64.b64encode(zlib.compress(pickle.dumps(Detonator()))).decode()
        with self.assertRaises(main.Rejected) as caught:
            main.handle_decode({"blob": blob})
        self.assertEqual(caught.exception.status, 422)
        self.assertIn("unpickling failed", caught.exception.message)

    def test_a_pickle_that_never_returns_is_killed(self):
        blob = base64.b64encode(zlib.compress(pickle.dumps(Spinner()))).decode()
        with self.assertRaises(main.Rejected) as caught:
            main.handle_decode({"blob": blob, "timeout_ms": 2000, "memory_mb": 128})
        self.assertEqual(caught.exception.status, 422)

    def test_a_large_blob_decodes_without_reading_it_a_byte_at_a_time(self):
        # A performance regression test, which is unusual but this one is worth
        # pinning: with bufsize=0 on the result pipe, stdout is a raw FileIO and
        # readline() costs one syscall per byte. That made the real 3.4 MB
        # private_test_cases blob take 1.7 seconds of pure syscall overhead
        # instead of 0.07, and it would come back silently the moment somebody
        # decided the pipe "should" be unbuffered.
        cases = [{"input": "x" * 4000, "output": "y" * 4000, "testtype": "functional"} for _ in range(2000)]
        started = time.monotonic()
        result = main.handle_decode({"blob": encode_private(cases), "timeout_ms": 60000})
        elapsed = time.monotonic() - started
        self.assertEqual(len(result["tests"]), 2000)
        self.assertLess(elapsed, 5.0, f"16 MB of test cases took {elapsed:.1f}s to come back")

    def test_rubbish_is_rejected_not_crashed_on(self):
        for blob in ("not base64 at all!!", base64.b64encode(b"not zlib").decode()):
            with self.assertRaises(main.Rejected) as caught:
                main.handle_decode({"blob": blob})
            self.assertEqual(caught.exception.status, 422)


# ── the HTTP surface ────────────────────────────────────────────────────────


class TestHTTP(SandboxCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), main.Handler)
        cls.server.daemon_threads = True
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, path: str, body, headers: dict | None = None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path, data=data, headers={"Content-Type": "application/json", **(headers or {})}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health(self):
        with urllib.request.urlopen(self.base + "/health", timeout=10) as response:
            self.assertEqual(json.loads(response.read()), {"status": "ok"})

    def test_grade_over_http(self):
        status, body = self.post(
            "/grade",
            {
                "language": "python",
                "code": "class Solution:\n    def minimumArrayLength(self, nums):\n        return len(nums)\n",
                "entry": {"class": "Solution", "func": "minimumArrayLength"},
                "tests": [{"input": "[1,2,3]", "output": "3", "testtype": "functional"}],
                "timeout_ms": 5000,
                "memory_mb": 256,
            },
        )
        self.assertEqual(status, 200)
        # prefix_applied is part of the wire contract, not an optional extra: the
        # caller treats its ABSENCE as "this sidecar predates prefix support" and
        # refuses to grade, rather than accepting a completion answer that was
        # silently run without its partial solution. Removing it here would make
        # every deployed router think every sandbox is stale.
        self.assertEqual(set(body), {"pass", "cases_run", "cases_passed", "first_failure", "error", "prefix_applied"})
        self.assertTrue(body["pass"], body)
        self.assertFalse(body["prefix_applied"], "no prefix was sent, so none can have been applied")

    def test_decode_over_http(self):
        status, body = self.post("/decode-private", {"blob": encode_private([{"input": "1", "output": "1"}])})
        self.assertEqual(status, 200)
        self.assertEqual(body["tests"][0]["testtype"], "functional")

    def test_a_bad_request_is_4xx_and_a_bad_submission_is_200(self):
        # The split that keeps "the router sent nonsense" distinguishable from
        # "the model wrote nonsense".
        self.assertEqual(self.post("/grade", {"code": "", "tests": [{}]})[0], 400)
        self.assertEqual(self.post("/grade", {"code": "x", "tests": []})[0], 400)
        self.assertEqual(self.post("/grade", {"code": "x", "tests": [{}], "language": "rust"})[0], 400)
        self.assertEqual(self.post("/grade", b"{not json")[0], 400)
        status, body = self.post(
            "/grade", {"code": "def (", "entry": {"func": "f"}, "tests": [{"input": "1", "output": "1"}]}
        )
        self.assertEqual(status, 200)
        self.assertFalse(body["pass"])

    def test_oversized_body_is_refused_before_it_is_read(self):
        original = main.MAX_BODY_BYTES
        main.MAX_BODY_BYTES = 64
        try:
            status, body = self.post("/grade", {"code": "x" * 4096, "tests": [{}]})
            self.assertEqual(status, 413)
            self.assertIn("exceeds", body["error"])
        finally:
            main.MAX_BODY_BYTES = original
        # And the shipped ceiling has room for the largest private_test_cases
        # blob in the LiveBench coding split, which is 3.4 MB of base64.
        self.assertGreater(main.MAX_BODY_BYTES, 8 * 1024 * 1024)

    def test_bearer_token_when_configured(self):
        original = main.TOKEN
        main.TOKEN = "sesame"
        try:
            self.assertEqual(self.post("/grade", {"code": "x", "tests": [{}]})[0], 401)
            status, _ = self.post(
                "/grade", {"code": "x", "tests": []}, headers={"Authorization": "Bearer sesame"}
            )
            self.assertEqual(status, 400)  # past auth, into validation
        finally:
            main.TOKEN = original

    def test_unknown_route(self):
        self.assertEqual(self.post("/nope", {})[0], 404)


# ── the jail itself ─────────────────────────────────────────────────────────


class TestJail(SandboxCase):
    def test_seccomp_is_installed(self):
        # Not a soft expectation. If this fails in CI the container is running
        # without the filter and the network story is only the python-level
        # block, which ctypes walks straight past.
        self.assertEqual(main._probe_network_isolation(), "seccomp")

    def test_a_run_that_produces_nothing_is_reported_not_hung(self):
        outcome = supervisor.run(
            "grade", {"code": "pass", "tests": [], "memory_mb": 128, "cpu_seconds": 5}, 5.0
        )
        self.assertTrue(outcome.completed())
        self.assertFalse(outcome.timed_out)

    def test_privilege_drop_is_a_no_op_for_a_non_root_service(self):
        # The shipped image runs as uid 10001, so this is the path every real
        # deployment takes. The root path is exercised by running the container
        # with --user 0:0, where the drop to 65534 either works (it has
        # CAP_SETUID) or the service refuses to start (it does not) — neither of
        # which a test process can produce.
        import jail

        self.assertFalse(jail.drop_privileges(65534, 65534))
        result = grade(
            "import os\nclass Solution:\n    def f(self, n):\n        return os.getuid()\n",
            [case("1", str(os.getuid()))],
        )
        self.assertTrue(result["pass"], result)

    def test_orphan_zombies_are_reaped(self):
        # In a container this service is pid 1 unless --init is passed, so a
        # fork bomb's killed grandchildren are reparented here and stay defunct
        # forever. Exercised directly because the reparenting only happens when
        # the process IS pid 1, which a test runner never is.
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        for _ in range(100):
            if supervisor.reap_orphans():
                break
            time.sleep(0.01)
        with self.assertRaises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)

    def test_reaping_leaves_a_concurrent_runs_child_alone(self):
        # The reason it is not a waitpid(-1) loop: a blind reap would consume
        # this run's exit status and report a killed process as a clean exit.
        results = []

        def one():
            results.append(grade("class Solution:\n    def f(self, n):\n        return n\n", [case("1", "1")]))

        noise = threading.Thread(target=lambda: [supervisor.reap_orphans() for _ in range(200)])
        workers = [threading.Thread(target=one) for _ in range(4)]
        noise.start()
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
        noise.join(timeout=60)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r["pass"] for r in results), results)

    def test_forged_records_without_the_nonce_are_dropped(self):
        # A submission can find the result descriptor. It must not be able to
        # write itself a passing case with it.
        result = grade(
            "import os, json\n"
            "class Solution:\n"
            "    def f(self, n):\n"
            "        for fd in range(3, 32):\n"
            "            try:\n"
            "                os.write(fd, (json.dumps({'t': 'case', 'index': 0, 'pass': True}) + chr(10)).encode())\n"
            "            except OSError:\n"
            "                pass\n"
            "        return 0\n",
            [case("1", "1")],
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["cases_run"], 1)
        self.assertEqual(result["cases_passed"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
