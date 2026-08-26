"""The parent side of one jailed run: spawn it, read it, and be certain it is
gone.

Nothing in here trusts the child to finish, to answer, or to stay inside its own
process group. The three things it guarantees, in the order they matter:

  It ends. A wall-clock deadline kills the process GROUP, which is why the child
  is given its own session — a submission that forks leaves children that are
  no longer the child we waited on, and killing one pid would leave the rest
  running until the container did.

  It is reaped even when it escapes the group. A process that calls setsid() for
  itself is out of the group before the kill lands, so after any abnormal end
  the parent sweeps /proc for anything still carrying this run's token in its
  environment. The token is generated per request and set nowhere else, so the
  sweep cannot mistake the service for its own child.

  It leaves no zombies. Grandchildren orphaned by a killed run are reparented to
  pid 1 in this container, which without docker's --init is this service — and
  nothing in Python reaps them, so each one sits in the pid table as a corpse
  until the container's --pids-limit is full of them.

  It leaves no files. The scratch directory is created per run and removed on
  every exit path, including the timeout and the crash — through a chmod walk,
  because a submission that mkdir'd something 0000 would otherwise defeat rmtree
  and leave the tmpfs a little smaller every request.

WHY NOT preexec_fn. The rlimits are set by the child, in jail.py, not here. The
reason is in that file's docstring and it is not a preference: this parent is
multi-threaded, and running Python between fork and exec in a threaded process
deadlocks on whatever lock another thread held at the moment of the fork.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")

# The environment variable that marks a process as belonging to one run. Read by
# the straggler sweep out of /proc/<pid>/environ, which is inherited across
# fork() and therefore survives the double-fork-and-setsid that beats killpg.
RUN_TOKEN_VAR = "DS_SANDBOX_RUN"

# Time between the child's own deadline and the parent's. It covers interpreter
# start-up, the payload write and the last record's trip up the pipe, so in the
# ordinary case the child's SIGALRM fires first and the run ends with a record
# that says "time limit exceeded" rather than with a corpse the parent has to
# interpret. Too small and every timeout is reported as a kill; too large and a
# hung submission holds its concurrency slot for the difference.
PARENT_GRACE_SECONDS = 2.0

# Who a jailed run belongs to when this service was started as root. 65534 is
# nobody/nogroup on every base this image could plausibly use.
#
# The shipped container runs as uid 10001 and never uses these — the child
# inherits a non-root uid and there is nothing to drop. They are for the
# deployments that are not the shipped container: `docker run --user root`, a
# runtime that ignores USER, a developer running main.py on a host. In those,
# the child would inherit root, and every rlimit the jail sets is one root can
# raise back. The alternative — trusting the Dockerfile's USER line — makes the
# whole isolation story depend on a line anybody can override from the command
# line without being told what it was for.
SANDBOX_UID = int(os.environ.get("SANDBOX_RUN_UID", "") or 65534)
SANDBOX_GID = int(os.environ.get("SANDBOX_RUN_GID", "") or 65534)

# Ceiling on what one run may send back. Sized for /decode-private: the largest
# private_test_cases blob in the LiveBench coding split is 3.4 MB compressed and
# inflates well past that. A run that exceeds it is killed rather than truncated,
# because a half-read test list is worse than no test list.
MAX_RESULT_BYTES = 64 * 1024 * 1024

# Pids this process currently holds a Popen for, and the lock that makes the
# set mean something. Held across the fork itself so a child is never visible in
# /proc before it is in here — see reap_orphans for why that window would matter.
_live_children: set[int] = set()
_live_lock = threading.Lock()


class Outcome:
    """What one run produced. Deliberately not a grading result — assembling one
    of those is main.py's job, and keeping the two apart is what lets the
    supervisor be tested against a runner that only ever prints records."""

    def __init__(self) -> None:
        self.records: list[dict] = []
        self.timed_out = False
        self.returncode: int | None = None
        self.signal: int | None = None
        self.spawn_error: str | None = None
        self.overflowed = False
        self.duration_ms = 0

    def first(self, kind: str) -> dict | None:
        for record in self.records:
            if record.get("t") == kind:
                return record
        return None

    def all(self, kind: str) -> list[dict]:
        return [record for record in self.records if record.get("t") == kind]

    def completed(self) -> bool:
        return self.first("done") is not None


def run(mode: str, payload: dict, wall_seconds: float, scratch_root: str | None = None,
        debug: bool = False) -> Outcome:
    """One child, start to finish. Never raises for anything the child did."""
    outcome = Outcome()
    token = secrets.token_hex(16)
    nonce = secrets.token_hex(16)
    payload = dict(payload, nonce=nonce, wall_seconds=wall_seconds)

    started = time.monotonic()
    try:
        scratch = tempfile.mkdtemp(prefix="grade-", dir=scratch_root)
    except OSError as exc:
        outcome.spawn_error = f"could not create a scratch directory: {exc}"
        return outcome

    # When this service is root the child drops to SANDBOX_UID, and the scratch
    # directory has to be reachable by that uid before the child gets there —
    # mkdtemp made it 0700 root, and a dropped child would find its own cwd
    # unreadable.
    #
    # chmod and not chown, which looks like the weaker choice and is the only
    # one that works: chown to a DIFFERENT uid needs CAP_CHOWN even for root,
    # and the container this is most likely to be root in is one started with
    # --cap-drop=ALL. chmod on a directory we own needs nothing. The permission
    # is wide, and it is wide on a randomly-named directory inside a tmpfs that
    # is already mode 1777, in a container with one other user — so it grants
    # nothing that the mount above it did not already grant.
    drop_to = None
    if os.geteuid() == 0:
        drop_to = [SANDBOX_UID, SANDBOX_GID]
        try:
            os.chmod(scratch, 0o777)
        except OSError as exc:
            _force_rmtree(scratch)
            outcome.spawn_error = f"could not open the scratch directory to {SANDBOX_UID}: {exc}"
            return outcome
    payload["drop_to"] = drop_to

    try:
        # The lock spans the fork so that no pid exists in /proc as our child
        # before it is registered here. Any gap is a window in which another
        # thread's reap_orphans could see it, fail to recognise it, and consume
        # the exit status this run is about to wait for.
        with _live_lock:
            proc = _spawn(mode, scratch, token, debug)
            _live_children.add(proc.pid)
    except OSError as exc:
        _force_rmtree(scratch)
        outcome.spawn_error = f"could not start the sandbox process: {exc}"
        return outcome

    pgid = proc.pid  # start_new_session=True makes the child its own group leader
    body = json.dumps(payload).encode("utf-8")

    # Three threads and not one, because two of the pipes can block each other.
    # The child reads its whole request before it writes anything, so a payload
    # larger than a pipe buffer (64 KB — every /decode-private request) parks the
    # parent mid-write while the child is parked mid-read. Writing from its own
    # thread while another drains the results is what makes both sides progress.
    writer = threading.Thread(target=_write_request, args=(proc, body), daemon=True)
    drainer = threading.Thread(target=_drain, args=(proc, outcome, nonce), daemon=True)
    writer.start()
    drainer.start()

    # Wait WITHOUT reaping, then kill the group, then reap.
    #
    # The ordering is the whole reason this is not a plain proc.wait(). A
    # SIGKILL has to go to the process GROUP, because a submission that forks
    # leaves children that are no longer the pid we waited on — but a group is
    # named by the leader's pid, and once the leader has been reaped that pid is
    # free to be reused, so killpg after a reap is a signal aimed at whatever
    # happens to hold that number now. Leaving the leader as a zombie until the
    # kill has landed makes the group id provably still this run's.
    #
    # And the kill runs on EVERY path, not only on the timeout. A submission
    # that forks and then returns an answer exits cleanly, reports a result, and
    # leaves its children running: nothing about a normal exit reaps them, which
    # is exactly how the first version of this file leaked processes.
    exited = _wait_without_reaping(proc.pid, time.monotonic() + wall_seconds + PARENT_GRACE_SECONDS)
    if not exited:
        outcome.timed_out = True
    _kill_group(pgid)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Unkillable means stuck in uninterruptible I/O, which nothing here can
        # clear. Report it and let the container's pid limit hold the line.
        outcome.spawn_error = "the sandbox process did not die after SIGKILL"

    outcome.returncode = proc.returncode
    if proc.returncode is not None and proc.returncode < 0:
        outcome.signal = -proc.returncode

    # A short join, then a decision. The pipe reaches EOF when the last process
    # holding its write end exits — so a drainer still alive AFTER the group has
    # been killed is direct evidence that something escaped the group, which is
    # the only time the /proc sweep is worth its couple of milliseconds. On the
    # ordinary path this join returns immediately and no scan happens at all.
    drainer.join(timeout=0.25)
    if drainer.is_alive() or outcome.timed_out:
        _sweep(token)
        drainer.join(timeout=1.0)
    if drainer.is_alive():
        # Something is still holding the write end and the sweep could not find
        # it. Close this end so the reader stops blocking: a leaked thread per
        # request would outlast every process this file is trying to contain.
        try:
            proc.stdout.close()
        except (OSError, ValueError):
            pass
    writer.join(timeout=1.0)

    with _live_lock:
        _live_children.discard(proc.pid)
    reap_orphans()

    _force_rmtree(scratch)
    outcome.duration_ms = int((time.monotonic() - started) * 1000)
    return outcome


def reap_orphans() -> int:
    """Clear zombie children that are not ours.

    When a submission forks and its parent dies, the grandchildren are
    reparented to pid 1 in this container — which, without docker's --init, is
    this service. Nothing in Python reaps them, so each one stays a zombie
    holding a pid slot: a fork bomb whose eight processes are killed leaves
    eight defunct entries behind, every request, until the container's
    --pids-limit is full of corpses. Measured, not theorised — a containerised
    run of the fork-bomb test left exactly FORK_SLACK zombies behind.

    The obvious ``waitpid(-1, WNOHANG)`` loop is wrong here, and the reason is
    concurrency: this service runs several grades at once, so a blind wait would
    happily consume a SIBLING run's exit status and leave that run reporting a
    status of 0 for a process the kernel had killed. Instead the children are
    enumerated from /proc and anything a live Popen owns is skipped, so only
    genuine orphans are touched.

    --init makes this a no-op, and start.sh passes it. This exists so that the
    image is also correct for anyone who runs it without.
    """
    reaped = 0
    with _live_lock:
        for pid in _child_pids() - _live_children:
            try:
                done, _status = os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                continue
            if done:
                reaped += 1
    return reaped


def _child_pids() -> set[int]:
    """Every process the kernel currently calls a child of this one.

    Read per-thread, because /proc/<pid>/task/<tid>/children lists the children
    created by THAT thread and this service forks from a different thread per
    request. An empty result — a kernel without CONFIG_PROC_CHILDREN — makes
    reap_orphans a no-op and leaves the job to --init, which is the right way
    round: reaping nothing is a slow leak, reaping the wrong thing corrupts a
    concurrent run's result.
    """
    pids: set[int] = set()
    try:
        threads = os.listdir("/proc/self/task")
    except OSError:
        return pids
    for tid in threads:
        try:
            with open(f"/proc/self/task/{tid}/children") as handle:
                pids.update(int(part) for part in handle.read().split())
        except (OSError, ValueError):
            continue
    return pids


def _wait_without_reaping(pid: int, deadline: float) -> bool:
    """Poll for the child's exit, leaving it a zombie. True if it exited in time.

    waitid with WNOWAIT is the only wait that does not consume the exit status,
    and consuming it is what frees the pid. The poll interval opens from a
    millisecond to twenty so that a submission finishing in 3 ms is not billed
    for a fixed tick, while a sixty-second grade does not spend the whole minute
    waking up.
    """
    interval = 0.001
    while True:
        try:
            info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
        except (ChildProcessError, OSError):
            return True
        if info is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
        interval = min(interval * 1.5, 0.02)


def _spawn(mode: str, scratch: str, token: str, debug: bool) -> subprocess.Popen:
    # -I isolates: no PYTHONPATH, no user site directory, and (3.11+) no script
    # directory on sys.path, which is why runner.py fixes its own path up. -S
    # skips site.py, so a sitecustomize dropped into the image is not a way in
    # and site-packages is not importable by a submission.
    argv = [sys.executable, "-I", "-S", RUNNER, mode]
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        # Both point into the scratch directory: a submission that writes to
        # $TMPDIR or $HOME then writes somewhere that is removed with the run,
        # instead of somewhere shared with the next one.
        "HOME": scratch,
        "TMPDIR": scratch,
        RUN_TOKEN_VAR: token,
    }
    # Everything else the service holds — the bearer token, the bind address —
    # is left behind. There is no network to exfiltrate it over, but an
    # environment built by allow-list cannot leak the next secret somebody adds.
    return subprocess.Popen(
        argv,
        cwd=scratch,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None if debug else subprocess.DEVNULL,
        # setsid(), done in C after the fork. The Python-level equivalent in a
        # preexec_fn is what this whole file avoids.
        start_new_session=True,
        close_fds=True,
        # Buffered, which matters more than it looks. bufsize=0 makes stdout a
        # raw FileIO, and IOBase.readline on a raw stream reads ONE BYTE PER
        # SYSCALL — so _drain's line loop turned a 3.4 MB /decode-private
        # response into three and a half million read() calls and about a second
        # and a half of pure syscall overhead. Nothing needs the stdin side
        # unbuffered either: _write_request closes the pipe, which flushes.
        bufsize=-1,
    )


def _write_request(proc: subprocess.Popen, body: bytes) -> None:
    try:
        proc.stdin.write(body)
        proc.stdin.close()
    except (BrokenPipeError, ValueError, OSError):
        # The child died before reading its request. Nothing to do here: the
        # absence of records is what the caller will report.
        pass


def _drain(proc: subprocess.Popen, outcome: Outcome, nonce: str) -> None:
    """Read the NDJSON result stream, discarding anything that is not ours.

    Records without the run's nonce are dropped silently. A submission can find
    the result descriptor and write to it — see runner.py — and while nothing
    stops it forging the nonce as well, dropping unnonced lines means the cheap
    version of the attack produces a run with no records rather than a run with
    a fabricated pass.
    """
    total = 0
    try:
        for line in proc.stdout:
            total += len(line)
            if total > MAX_RESULT_BYTES:
                outcome.overflowed = True
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(record, dict) or record.get("n") != nonce:
                continue
            outcome.records.append(record)
    except (OSError, ValueError):
        pass
    finally:
        try:
            proc.stdout.close()
        except (OSError, ValueError):
            pass


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _sweep(token: str, passes: int = 8) -> int:
    """Kill stragglers until a pass finds none.

    One pass is not enough against a fork bomb, and the reason is a race the
    first version of this file lost: the scan takes milliseconds, and every
    process it has not reached yet is still forking while it walks /proc. So the
    survivors of pass one are the children born during pass one. Each pass
    strictly reduces the population — a killed process forks no more, and
    RLIMIT_NPROC caps how many can exist at once — so this converges in two or
    three, and the bound is only there so that a pathological case ends in a
    slow request rather than in a loop.
    """
    killed = 0
    for _ in range(passes):
        found = _kill_by_token(token)
        killed += found
        if found == 0:
            return killed
        time.sleep(0.01)
    return killed


def _kill_by_token(token: str) -> int:
    """SIGKILL anything still carrying this run's token.

    This is the answer to the one hole killpg has: a process that calls setsid()
    is in a group the parent never knew about, and no amount of killing the
    original group reaches it. /proc/<pid>/environ is inherited across fork() and
    is readable by the uid that owns the process — the same uid as this service,
    since the container runs as one non-root user — so the token identifies
    descendants regardless of what they did to their process group.

    Best effort, and honest about it: a straggler that exec'd a fresh binary with
    a scrubbed environment is invisible here. What still bounds that case is
    RLIMIT_NPROC (it can only ever be FORK_SLACK processes) and the container's
    --pids-limit.
    """
    needle = f"{RUN_TOKEN_VAR}={token}".encode()
    self_pid = os.getpid()
    killed = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == self_pid:
            continue
        try:
            with open(f"/proc/{pid}/environ", "rb") as handle:
                if needle not in handle.read():
                    continue
        except OSError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except OSError:
            pass
    return killed


def _force_rmtree(path: str) -> None:
    """Remove the scratch directory even when the submission made that awkward.

    A directory created 0000 cannot be listed or descended into, and rmtree
    gives up on it — so a submission that does one mkdir and one chmod costs the
    tmpfs a directory per request, forever. Everything under here belongs to our
    own uid, so the permission can always be restored.

    The chmod happens in a topdown walk rather than in rmtree's error handler,
    which was the first attempt and does not work: the handler is called with
    whatever function failed, and for a directory that is os.open with a dir_fd
    — retrying it as ``func(target)`` raises TypeError about a missing argument
    and the tree survives anyway. os.walk yields a directory BEFORE it descends
    into it, so fixing the permission as it is yielded is what lets the same
    walk reach the contents.
    """
    try:
        os.chmod(path, stat.S_IRWXU)
    except OSError:
        pass
    try:
        for root, dirs, _files in os.walk(path, topdown=True):
            for name in dirs:
                try:
                    os.chmod(os.path.join(root, name), stat.S_IRWXU)
                except OSError:
                    pass
    except OSError:
        pass
    shutil.rmtree(path, ignore_errors=True)
