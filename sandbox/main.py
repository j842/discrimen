"""discrimen-sandbox: the code-execution grading sidecar.

LiveBench's coding split cannot be graded by comparing strings. Its rows carry
an EMPTY ground_truth on purpose — the answer to "write minimumArrayLength" is
not a token, it is a function, and the only thing that can tell a right one from
a wrong one is running it against the test cases. So the benchmark needs an
interpreter, and the router must not be the thing holding it: discrimen is a
long-lived process with the fleet's credentials, its request log and its
database in the same address space, and exec'ing a language model's output next
to any of that is not a risk worth reasoning about carefully. It is a risk worth
moving into a different container.

Hence a sidecar. It runs beside the router on loopback, it holds nothing worth
stealing, and every submission it is asked about is executed in a fresh
subprocess that is torn down before the response is written. The division is the
point: the router decides WHAT to grade and reads a boolean back; this service
is the only thing that ever runs the code, and it is built on the assumption
that everything it runs is hostile.

  POST /grade            a submission and its test cases → pass/fail and a count
  POST /decode-private   base64 → zlib → pickle, done in the jail because
                         unpickling is arbitrary code execution
  GET  /health           {"status":"ok"}

STDLIB ONLY, and that is a decision rather than an accident. The image is
python:3.12-slim with no pip install layer at all, so there is no dependency
that can pull a new transitive package into the one container in the deployment
whose whole job is to be the blast radius. It also keeps the image small enough
that pulling it is not a consideration when a host redeploys. http.server is
unfashionable and entirely adequate for two endpoints on loopback behind a
router that already speaks to it over a socket it opened itself.

WHAT ISOLATES WHAT — the layers, and the file each one lives in:

  jail.py        a privilege drop if this service is somehow root, rlimits
                 (cpu, address space, file size, processes, core) and a seccomp
                 filter that removes socket() from the process
  supervisor.py  a session per run, a wall-clock SIGKILL on the process group,
                 a /proc sweep for anything that escaped it, orphan reaping, and
                 a scratch directory destroyed on every exit path
  Dockerfile     non-root, read-only rootfs, all capabilities dropped,
                 no-new-privileges, a tmpfs for the scratch space
  start.sh       --pids-limit, --memory and --init, so the HOST is bounded even
                 if everything above it has been defeated

None of the four is sufficient alone. The rlimits do not stop a sleeping
process, the wall clock does not stop a fork bomb, seccomp does not stop a
memory bomb, and the container flags do not stop any of them from happening —
they stop them mattering to anything else.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import supervisor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("discrimen-sandbox")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        log.warning("%s is not an integer; using %d", name, default)
        return default


PORT = _env_int("SANDBOX_PORT", 8587)
# 0.0.0.0 inside the container, published to 127.0.0.1 on the host by start.sh.
# Binding to loopback in here would make the published port unreachable, since
# the port mapping arrives on the container's eth0 and not on its lo.
BIND = os.environ.get("SANDBOX_BIND", "0.0.0.0")
TOKEN = os.environ.get("SANDBOX_TOKEN", "").strip()

# Each run can saturate a core, so the cap is about CPUs and not about request
# rate. Two is the floor because one would make a single slow grade look like an
# outage to everything queued behind it.
MAX_CONCURRENCY = _env_int("SANDBOX_MAX_CONCURRENCY", max(2, min(4, os.cpu_count() or 2)))
QUEUE_WAIT_MS = _env_int("SANDBOX_QUEUE_WAIT_MS", 30000)

DEFAULT_TIMEOUT_MS = _env_int("SANDBOX_DEFAULT_TIMEOUT_MS", 10000)
# The ceiling is the service's promise to itself, not the caller's request. A
# router asking for a ten-minute grade has misread its own benchmark deadline,
# and honouring it would pin a concurrency slot for ten minutes.
MAX_TIMEOUT_MS = _env_int("SANDBOX_MAX_TIMEOUT_MS", 120000)
DEFAULT_MEMORY_MB = _env_int("SANDBOX_DEFAULT_MEMORY_MB", 512)
MAX_MEMORY_MB = _env_int("SANDBOX_MAX_MEMORY_MB", 4096)

# Sized for the largest private_test_cases blob in the LiveBench coding split
# (3.4 MB of base64) with room for JSON overhead and a much larger one later.
MAX_BODY_BYTES = _env_int("SANDBOX_MAX_BODY_BYTES", 32 * 1024 * 1024)

# Where per-run scratch directories are made. Set to the tmpfs the container
# mounts, so a submission writing files never touches a real filesystem and the
# space it can consume is capped by the mount rather than by the disk.
SCRATCH_ROOT = os.environ.get("SANDBOX_SCRATCH_DIR") or None
DEBUG = os.environ.get("SANDBOX_DEBUG", "").strip().lower() in ("1", "true", "yes")

_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)


class Rejected(Exception):
    """A request the caller got wrong. Distinct from anything a SUBMISSION got
    wrong, which is a 200 with pass:false — see handle_grade."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# ── Grading ─────────────────────────────────────────────────────────────────


def handle_grade(request: dict) -> dict:
    """POST /grade.

    The status-code split is the important part and it is worth stating
    outright: 4xx means the ROUTER sent something wrong, and 200 with
    pass:false means the MODEL did. A submission that fails to compile, hangs,
    forks, allocates ten gigabytes or exits non-zero is a grading result — the
    benchmark asked "is this code correct" and the answer is no. Returning 500
    for those would make an unparseable model answer indistinguishable from the
    sandbox being broken, and the router would retry something that will fail
    identically forever.
    """
    language = (request.get("language") or "python").strip().lower()
    if language not in ("python", "python3", "py"):
        raise Rejected(400, f"language {language!r} is not supported; this sandbox runs python only")

    code = request.get("code")
    if not isinstance(code, str) or not code.strip():
        raise Rejected(400, "code must be a non-empty string")

    tests = request.get("tests")
    if not isinstance(tests, list) or not tests:
        raise Rejected(400, "tests must be a non-empty list")
    for index, case in enumerate(tests):
        if not isinstance(case, dict):
            raise Rejected(400, f"tests[{index}] must be an object")

    entry = request.get("entry")
    if entry is not None and not isinstance(entry, dict):
        raise Rejected(400, "entry must be an object or null")

    # The partial solution a completion answer continues, or "" for the ordinary
    # case. Validated but NOT executed here — it is joined to the submission
    # inside the jail, on the far side of the isolation boundary, because a
    # prefix is untrusted dataset content exactly as the submission is.
    prefix = request.get("prefix") or ""
    if not isinstance(prefix, str):
        raise Rejected(400, "prefix must be a string")

    timeout_ms = _bounded(request.get("timeout_ms"), DEFAULT_TIMEOUT_MS, 100, MAX_TIMEOUT_MS)
    memory_mb = _bounded(request.get("memory_mb"), DEFAULT_MEMORY_MB, 16, MAX_MEMORY_MB)

    payload = {
        "code": code,
        "prefix": prefix,
        "entry": entry or {},
        "tests": tests,
        "memory_mb": memory_mb,
        # The CPU limit is the wall budget rounded up, not something separate. A
        # tighter CPU limit would fail a submission that is merely slow before
        # the wall clock it was actually given had run out; a looser one would
        # let a busy-loop outlive its own deadline on a machine under load.
        "cpu_seconds": max(1, (timeout_ms + 999) // 1000),
        "stop_on_first_failure": bool(request.get("stop_on_first_failure")),
    }
    outcome = _run("grade", payload, timeout_ms)
    return _assemble(outcome, len(tests), timeout_ms)


def _assemble(outcome: supervisor.Outcome, requested: int, timeout_ms: int) -> dict:
    cases = outcome.all("case")
    cases_run = len(cases)
    cases_passed = sum(1 for case in cases if case.get("pass"))

    error = _run_error(outcome, timeout_ms)
    first_failure = None
    for case in cases:
        if not case.get("pass"):
            first_failure = {
                "index": case.get("index", 0),
                "expected": case.get("expected") or "",
                "got": case.get("got") or "",
                "error": case.get("error"),
            }
            break

    # A run that died partway through failed AT a case, and saying which one is
    # most of the diagnostic value in a timeout report — "hung on case 0" and
    # "hung on case 340" are different bugs. The child tags its fatal record with
    # the index it was on precisely so this can be reconstructed.
    if first_failure is None and error is not None:
        fatal = outcome.first("fatal") or {}
        if "index" in fatal:
            first_failure = {"index": fatal["index"], "expected": "", "got": "", "error": error}

    passed = (
        error is None
        and cases_run == requested
        and cases_passed == cases_run
        and outcome.completed()
    )
    return {
        "pass": passed,
        "cases_run": cases_run,
        "cases_passed": cases_passed,
        "first_failure": first_failure,
        "error": error,
    }


def _run_error(outcome: supervisor.Outcome, timeout_ms: int) -> str | None:
    """The one thing that went wrong with the RUN, as opposed to with a case.

    Ordered by how much each explains. A fatal record is the child's own account
    and beats every inference the parent could make from an exit status; a
    timeout beats a signal, because the signal IS the timeout's SIGKILL and
    reporting it as "killed by signal 9" would describe the cure rather than the
    disease.
    """
    if outcome.spawn_error:
        return outcome.spawn_error
    fatal = outcome.first("fatal")
    if fatal:
        return str(fatal.get("error") or "the sandbox reported a failure with no detail")
    if outcome.overflowed:
        return f"the submission produced more than {supervisor.MAX_RESULT_BYTES} bytes of results"
    if outcome.timed_out:
        return f"timed out after {timeout_ms} ms"
    if outcome.signal:
        return f"the sandbox process was killed by signal {outcome.signal}"
    if not outcome.completed():
        return f"the sandbox process exited with status {outcome.returncode} before finishing"
    return None


# ── Private test decoding ───────────────────────────────────────────────────


def handle_decode(request: dict) -> dict:
    """POST /decode-private.

    Runs in the jail for one reason, which is the whole reason this endpoint
    exists as an endpoint at all: LiveBench's private_test_cases are base64 of
    zlib of a PICKLE, and pickle.loads is a documented arbitrary-code-execution
    primitive — the format works by naming callables and calling them. There is
    no way to validate the bytes first that is not itself a pickle
    implementation. So it is executed with exactly the same containment a model's
    own submission gets, and the router receives ordinary JSON.

    The memory allowance is larger than a submission's by default because the
    work is inflating megabytes of compressed data into Python objects, which is
    legitimate and which a 512 MB ceiling makes fail.
    """
    blob = request.get("blob")
    if not isinstance(blob, str) or not blob.strip():
        raise Rejected(400, "blob must be a non-empty base64 string")

    timeout_ms = _bounded(request.get("timeout_ms"), 60000, 100, MAX_TIMEOUT_MS)
    memory_mb = _bounded(request.get("memory_mb"), 1024, 64, MAX_MEMORY_MB)
    payload = {"blob": blob, "memory_mb": memory_mb, "cpu_seconds": max(1, (timeout_ms + 999) // 1000)}

    outcome = _run("decode", payload, timeout_ms)
    record = outcome.first("tests")
    if record is None:
        raise Rejected(422, _run_error(outcome, timeout_ms) or "the blob could not be decoded")
    return {"tests": record.get("tests") or []}


# ── Plumbing ────────────────────────────────────────────────────────────────


def _run(mode: str, payload: dict, timeout_ms: int) -> supervisor.Outcome:
    """Take a concurrency slot, run one child, give the slot back.

    The bound is on CPUs rather than on connections because a grade is CPU-bound
    by construction — it is somebody's algorithm running flat out — and letting
    twenty of them share four cores would make every one of them miss a deadline
    it would otherwise have met, turning a load spike into a wave of false
    timeouts. Queueing instead means a late answer; oversubscribing means a
    WRONG one.
    """
    if not _slots.acquire(timeout=QUEUE_WAIT_MS / 1000.0):
        raise Rejected(503, f"the sandbox is at capacity ({MAX_CONCURRENCY} concurrent runs)")
    try:
        return supervisor.run(mode, payload, timeout_ms / 1000.0, SCRATCH_ROOT, DEBUG)
    finally:
        _slots.release()


def _bounded(value, default: int, low: int, high: int) -> int:
    """A caller's number, or the default, clamped rather than rejected.

    Clamping because these are resource requests, not semantics: a router asking
    for 8 GB is asking for more than it will get, and refusing the whole grade
    over it would take the benchmark down for a knob nobody set deliberately.
    Anything that is not a number at all is a different matter and falls back to
    the default rather than guessing what was meant.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(low, min(high, int(value)))


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 so the router can keep the connection open across a benchmark's
    # worth of grades rather than paying a handshake per question. It obliges
    # every response to carry an accurate Content-Length, which _send does.
    protocol_version = "HTTP/1.1"
    server_version = "discrimen-sandbox"
    sys_version = ""

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/health":
            self._send(200, {"status": "ok"})
            return
        self._send(404, {"error": f"no route for GET {self.path}"})

    def do_POST(self) -> None:
        route = self.path.split("?")[0]
        try:
            self._authorise()
            request = self._read_json()
            if route == "/grade":
                self._send(200, handle_grade(request))
            elif route == "/decode-private":
                self._send(200, handle_decode(request))
            else:
                self._send(404, {"error": f"no route for POST {self.path}"})
        except Rejected as exc:
            self._send(exc.status, {"error": exc.message})
        except Exception as exc:  # noqa: BLE001
            # The last line of defence for the SERVICE, not for a submission —
            # a submission's failure never reaches here. Something got past the
            # handlers, and the one outcome worse than a 500 is a thread dying
            # with the connection still open and the router waiting on it.
            log.exception("unhandled error serving %s", route)
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _authorise(self) -> None:
        """Optional bearer token.

        Optional because the deployed shape publishes this port on 127.0.0.1
        only, and a token adds nothing against an attacker who is already
        executing on the host. It exists for the shapes that are not that one —
        a shared docker network, a developer forwarding the port — where the
        alternative is an unauthenticated code-execution endpoint.
        """
        if not TOKEN:
            return
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer ") or header[7:].strip() != TOKEN:
            raise Rejected(401, "unauthorized")

    def _read_json(self) -> dict:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            # http.server does not decode chunked bodies, and a handler that read
            # the raw stream as if it were the body would parse the size prefix
            # as JSON. Say so rather than fail obscurely.
            raise Rejected(411, "chunked request bodies are not supported; send Content-Length")
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            raise Rejected(400, "Content-Length is not a number") from None
        if length <= 0:
            raise Rejected(400, "a request body is required")
        if length > MAX_BODY_BYTES:
            raise Rejected(413, f"request body of {length} bytes exceeds the {MAX_BODY_BYTES} byte limit")

        body = self.rfile.read(length)
        if len(body) != length:
            raise Rejected(400, "the request body was shorter than its Content-Length")
        try:
            request = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise Rejected(400, f"invalid json: {exc}") from None
        if not isinstance(request, dict):
            raise Rejected(400, "the request body must be a json object")
        return request

    def _send(self, status: int, body: dict) -> None:
        try:
            payload = json.dumps(body).encode("utf-8")
        except (TypeError, ValueError):
            status, payload = 500, b'{"error":"response could not be serialised"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            # The router gave up while a grade was running. Its problem, not
            # ours, and not worth a stack trace in the log.
            pass

    def log_message(self, fmt: str, *args) -> None:
        # BaseHTTPRequestHandler writes one line per request straight to stderr.
        # At debug level it is useful; at info it is a wall of 200s that hides
        # the two lines anybody actually wants (the boot self-test, and a run
        # that could not be contained).
        log.debug("%s %s", self.address_string(), fmt % args)


def selftest() -> None:
    """Prove the jail works before accepting traffic, and say so in the log.

    Run at start-up rather than left to the first request, because the failure
    it catches is silent by nature: a container started without the seccomp
    filter grades exactly as correctly as one with it, and the difference only
    shows up as a submission that turned out to have a network. An operator who
    reads one line of the boot log should be able to tell which they have.
    """
    result = handle_grade(
        {
            "language": "python",
            "code": "class Solution:\n    def echo(self, n):\n        return n + 1\n",
            "entry": {"class": "Solution", "func": "echo"},
            "tests": [{"input": "1", "output": "2", "testtype": "functional"}],
            "timeout_ms": 5000,
        }
    )
    if not result.get("pass"):
        log.error("SELF-TEST FAILED: %s", json.dumps(result))
        if os.geteuid() == 0:
            # The likeliest cause by a distance, and the one whose error message
            # ("Operation not permitted") explains nothing on its own. Dropping
            # to another uid needs CAP_SETUID, so root plus --cap-drop=ALL is a
            # configuration in which the service can neither stay safe nor make
            # itself safe. Failing closed is the only honest answer: a grader
            # that runs a submission as root has no isolation at all, because
            # every rlimit it sets is one root can raise back.
            log.error(
                "This service is running as root. It drops each run to uid %d, which needs "
                "CAP_SETUID — and a container started with --cap-drop=ALL does not have it. "
                "Run the image as its own non-root user (the default; remove any --user root) "
                "rather than granting the capability back.",
                supervisor.SANDBOX_UID,
            )
        raise SystemExit("the sandbox cannot grade a trivial submission; refusing to serve")

    network = _probe_network_isolation()
    if network == "seccomp":
        log.info("jail: seccomp filter active — socket() returns EPERM inside a run")
    else:
        log.warning(
            "jail: SECCOMP FILTER NOT INSTALLED (%s). Executed code can still open sockets; "
            "the only thing between it and the network is the python-level block, which ctypes "
            "walks straight past. Check that the container has not disabled it.",
            network,
        )


def _probe_network_isolation() -> str:
    outcome = supervisor.run("grade", {"code": "pass", "tests": [], "memory_mb": 128, "cpu_seconds": 5},
                             5.0, SCRATCH_ROOT, DEBUG)
    record = outcome.first("jail") or {}
    return str(record.get("network") or "unknown")


def main() -> None:
    selftest()
    server = ThreadingHTTPServer((BIND, PORT), Handler)
    # Without this, a client holding a keep-alive connection open keeps a
    # non-daemon thread alive and the process will not shut down on SIGTERM —
    # which docker follows with a SIGKILL ten seconds later, so a redeploy
    # becomes a hard kill of whatever was mid-grade.
    server.daemon_threads = True
    log.info(
        "listening on %s:%d (max %d concurrent runs, default %d ms / %d MB, auth %s)",
        BIND, PORT, MAX_CONCURRENCY, DEFAULT_TIMEOUT_MS, DEFAULT_MEMORY_MB,
        "on" if TOKEN else "off",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
