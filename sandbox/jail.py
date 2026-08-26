"""Everything the child process does to itself before it touches a submission.

This file runs INSIDE the untrusted process, after fork and exec but before a
single byte of model-generated code has been compiled. That placement is the
whole design, and it is worth saying why it is not the obvious alternative.

The obvious alternative is subprocess's ``preexec_fn``, which runs between fork
and exec and is where a textbook judge sets its rlimits. It is unusable here.
The service is a ThreadingHTTPServer, so the parent is multi-threaded, and
``preexec_fn`` runs Python code in a forked child of a threaded process — a
child that holds whatever locks the other threads happened to be holding at the
instant of the fork. CPython's own documentation calls this out; the failure it
produces is not an error but a hang, in the one process whose job is to contain
hangs. So the parent uses only ``start_new_session=True`` (a setsid done in C,
in _posixsubprocess, safe against threads) and everything else is set here, by a
freshly exec'd single-threaded interpreter that owns no locks.

The gap this opens — between exec and the call to ``apply()`` — is a few
milliseconds of OUR code, not the submission's. Nothing untrusted has been read
in that window, so nothing untrusted benefits from it.

LAYERS, and what each one is actually for:

  uid          if the service was started as root, the child stops being root
               before anything else happens. Every limit below is one root can
               raise back, so this is the layer the others rest on.
  rlimits      bound one submission's appetite: CPU seconds, address space,
               bytes written to disk, processes forked, core dumps.
  seccomp      removes the socket() syscall, which is what "no network" means
               when you cannot have a network namespace.
  the parent   kills the process GROUP on the wall clock, and sweeps for
               anything that escaped the group (see supervisor.py).

None of the four is sufficient alone and the ordering between them matters:
rlimits stop the cheap resource attacks, seccomp stops exfiltration, and the
parent's kill is what handles a process that is simply asleep — which no rlimit
covers, because sleeping costs no CPU and allocates nothing.
"""

from __future__ import annotations

import ctypes
import os
import platform
import resource
import sys

# The address-space floor. RLIMIT_AS caps VIRTUAL memory, and a CPython
# interpreter has already mapped tens of megabytes of it before it reaches this
# function — so a caller who asks for memory_mb=8 is not asking for a tight
# jail, they are asking for a process that cannot allocate its own next object.
# Clamping up rather than failing keeps a mis-set knob from turning every
# submission into an identical MemoryError that looks like a grading result.
MIN_MEMORY_MB = 64

# What a submission may write to disk, per file. Not zero: a solution that opens
# a scratch file is doing something legitimate (and rare), and RLIMIT_FSIZE=0
# would kill it with SIGXFSZ on the first byte. Small enough that filling the
# container's tmpfs takes more files than RLIMIT_NOFILE allows.
FILE_SIZE_LIMIT = 8 * 1024 * 1024

# How many extra tasks a submission may create beyond what its uid already has.
#
# RLIMIT_NPROC is per REAL UID, not per process tree — it is checked at fork()
# against the total task count for the user, which in this container includes
# the service's own threads. That is exactly why the limit is computed as
# "whatever is running now, plus this" instead of being a constant: a constant
# would either be so high a fork bomb reaches it slowly, or so low that the
# service's own threads had already consumed it and the first fork of an honest
# submission failed.
#
# The asymmetry that makes this work: the limit consulted by fork() is the
# FORKING process's, so the child's low ceiling bounds the child while the
# service keeps its own inherited (high) ceiling and can still spawn the next
# grade. A fork bomb therefore costs FORK_SLACK tasks for the length of one
# request and nothing after it.
FORK_SLACK = 8

# Deep recursion is normal in this problem class (tree and DFS solutions), and
# the default 1000 fails honest submissions. CPython 3.11+ no longer consumes C
# stack for pure-Python calls, so a limit in this range is bounded by heap
# rather than by a segfault — which is what made LiveCodeBench's 6*10**5 unsafe
# and is why this is two orders of magnitude smaller.
RECURSION_LIMIT = 20000

# ── seccomp ─────────────────────────────────────────────────────────────────
#
# A BPF program, assembled by hand, that makes socket() return EPERM. There is
# no stdlib seccomp binding and libseccomp would be a compiled dependency in an
# image whose entire point is that it has none, so this is ctypes and a struct
# layout.
#
# Blocking socket() ALONE is deliberate and is sufficient. connect(), sendto()
# and the rest all need a descriptor that only socket() can mint, and the child
# inherits none: subprocess closes every fd above 2 before exec, and 0/1/2 are a
# pipe and /dev/null. A longer denylist is more code, more syscall numbers to
# get wrong per architecture, and no more secure.
#
# ERRNO rather than KILL as the action: a submission that tries to open a socket
# then sees PermissionError, which is an ordinary Python exception it can
# report — so grading says "this code tried to use the network" instead of the
# process vanishing and the whole run being scored as a crash.

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2

BPF_LD_W_ABS = 0x20  # BPF_LD | BPF_W | BPF_ABS
BPF_JEQ_K = 0x15  # BPF_JMP | BPF_JEQ | BPF_K
BPF_RET_K = 0x06  # BPF_RET | BPF_K

SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
EPERM = 1

# struct seccomp_data is { int nr; __u32 arch; __u64 ip; __u64 args[6]; }, so
# the syscall number is at offset 0 and the architecture token at offset 4.
SECCOMP_DATA_NR = 0
SECCOMP_DATA_ARCH = 4

# AUDIT_ARCH_* is the ELF machine id with the 64-bit and little-endian flags
# set. Checking it is not decoration: without the check, a process that made a
# syscall through the 32-bit compat entry point would be filtered against a
# table where 41 means something else entirely, and the filter would be
# blocking a syscall it did not mean to and allowing the one it did.
_ARCH = {
    "x86_64": (0xC000003E, 41),  # AUDIT_ARCH_X86_64, __NR_socket
    "aarch64": (0xC00000B7, 198),  # AUDIT_ARCH_AARCH64, __NR_socket (asm-generic)
    "arm64": (0xC00000B7, 198),
}


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_uint16),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


# Held at module scope for the lifetime of the process. The kernel copies the
# program in at prctl() time so a dangling pointer could not actually be
# dereferenced later, but a filter freed by the garbage collector while the
# call is still on the stack is the kind of bug that reproduces once a month.
_FILTER_KEEPALIVE = None


def install_seccomp() -> str:
    """Remove socket() from this process. Returns a one-word status for logging.

    Never raises. A kernel or a container runtime that refuses the filter is a
    deployment this service still has to run in — with a weaker network story,
    which the caller reports rather than discovers — and turning that into a
    startup crash would take the grader down over a defence in depth.
    """
    global _FILTER_KEEPALIVE

    arch = _ARCH.get(platform.machine())
    if arch is None:
        return "unsupported-arch"
    audit_arch, nr_socket = arch

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return "no-libc"

    # PR_SET_NO_NEW_PRIVS first, and it is not optional: without CAP_SYS_ADMIN
    # the kernel refuses PR_SET_SECCOMP outright unless no_new_privs is set,
    # because a filter that could be escaped by exec'ing a setuid binary is not
    # a filter. The container also sets it (--security-opt no-new-privileges),
    # and setting it twice costs nothing.
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        return "no-new-privs-failed"

    program = [
        # if data.arch != our arch: kill. Anything reaching the filter through
        # a foreign ABI is not a case this table describes.
        _SockFilter(BPF_LD_W_ABS, 0, 0, SECCOMP_DATA_ARCH),
        _SockFilter(BPF_JEQ_K, 1, 0, audit_arch),
        _SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_KILL_PROCESS),
        # if data.nr == socket: EPERM, else allow.
        _SockFilter(BPF_LD_W_ABS, 0, 0, SECCOMP_DATA_NR),
        _SockFilter(BPF_JEQ_K, 0, 1, nr_socket),
        _SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_ERRNO | EPERM),
        _SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW),
    ]
    array = (_SockFilter * len(program))(*program)
    prog = _SockFprog(len(program), array)
    _FILTER_KEEPALIVE = (array, prog)

    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(prog), 0, 0) != 0:
        return "seccomp-refused-errno-%d" % ctypes.get_errno()
    return "seccomp"


def block_network_in_python() -> None:
    """The fallback for when seccomp did not install.

    It is honestly weaker and the difference should not be papered over: this
    replaces the constructors the standard library reaches for, so urllib,
    http.client and socket.create_connection all fail — but ctypes can still
    reach the raw syscall, so it stops a submission that USES the network and
    not one that is trying to escape. It is applied unconditionally anyway, on
    top of a working filter, because a PermissionError that names the sandbox
    reads as a grading result and an EPERM from the kernel reads as a bug.

    socket.socket is replaced with a SUBCLASS of itself, not with a function,
    and that is not tidiness. ssl.py contains ``class SSLSocket(socket)``, so a
    plain function in that slot makes the class statement call the function as
    a metaclass: importing ssl then dies with "function() argument 'code' must
    be code, not str", and every submission that so much as imports urllib fails
    at load time with an error that mentions neither ssl nor the network. Found
    the hard way; the network test was the only one that caught it.
    """

    def refuse(*_args, **_kwargs):
        raise PermissionError("network access is disabled in the grading sandbox")

    try:
        import socket as _socket_mod
    except Exception:
        return

    class BlockedSocket(_socket_mod.socket):
        def __init__(self, *_args, **_kwargs):
            raise PermissionError("network access is disabled in the grading sandbox")

    _socket_mod.socket = BlockedSocket
    for name in (
        "socketpair",
        "create_connection",
        "create_server",
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
    ):
        if hasattr(_socket_mod, name):
            setattr(_socket_mod, name, refuse)


def _current_task_count(uid: int) -> int:
    """Tasks already owned by this uid, read out of /proc.

    Costs a couple of hundred stat() calls, once per grade, against a fork+exec
    that costs more. The alternative — a fixed RLIMIT_NPROC — is what the
    FORK_SLACK comment above rejects.

    A read failure returns 0, which makes the limit FORK_SLACK on its own. That
    is the conservative direction: too low a ceiling fails an honest submission
    visibly, where too high a one fails nothing and contains nothing.
    """
    count = 0
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            if os.stat("/proc/" + entry).st_uid == uid:
                # Threads count against RLIMIT_NPROC too, and /proc/<pid> hides
                # them — so read the kernel's own count rather than inferring 1.
                count += _thread_count("/proc/" + entry + "/status")
        except OSError:
            continue
    return count


def _thread_count(status_path: str) -> int:
    try:
        with open(status_path, "rb") as handle:
            for line in handle:
                if line.startswith(b"Threads:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 1


def _set(limit: int, soft: int, hard: int | None = None) -> None:
    """Lower one rlimit, never above the hard limit we were given.

    soft == hard on purpose everywhere it is used below. A soft limit can be
    raised back to the hard limit by the process that holds it, and the process
    that holds it here is about to execute somebody else's code — so leaving
    headroom between the two would make every limit in this file advisory.
    """
    if hard is None:
        hard = soft
    try:
        current_soft, current_hard = resource.getrlimit(limit)
    except (ValueError, OSError):
        return
    if current_hard != resource.RLIM_INFINITY:
        soft = min(soft, current_hard)
        hard = min(hard, current_hard)
    try:
        resource.setrlimit(limit, (soft, hard))
    except (ValueError, OSError):
        # A limit the kernel or the container will not accept is one fewer
        # layer, not a reason to abandon the others.
        pass


def drop_privileges(uid: int, gid: int) -> bool:
    """Become a non-root user, irreversibly. Reports whether it had to.

    The shipped image runs as uid 10001 and never reaches this, so it exists for
    the deployments that are not the shipped image: `docker run --user root`, a
    runtime that ignores USER, a developer running main.py on a host. In all of
    those the child would inherit root, and every rlimit in this file would be a
    limit root can raise — RLIMIT_NPROC and RLIMIT_AS are both explicitly waived
    for a process with the right capability.

    setgroups before setgid before setuid, in that order, because each one needs
    the privilege the next one gives up. Getting the order wrong does not fail
    loudly; it leaves the process in the supplementary groups it started with,
    which on a host is often the docker group.
    """
    if os.geteuid() != 0:
        return False
    try:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
    except OSError:
        # Refuse to continue as root. A grader that silently runs a submission
        # with full privileges is worse than one that reports a failure, because
        # the failure is the only signal anybody would ever get.
        raise PermissionError(f"could not drop privileges to {uid}:{gid}; refusing to execute as root") from None
    if os.geteuid() == 0 or os.getuid() == 0:
        raise PermissionError("privilege drop did not take effect; refusing to execute as root")
    return True


def apply(memory_mb: int, cpu_seconds: int, drop_to: tuple[int, int] | None = None) -> str:
    """Put this process in the jail. Returns the network-isolation status.

    Called once, from runner.py, after the payload has been read off stdin and
    before anything from the request has been compiled. Reading first matters
    for RLIMIT_AS: a 3.4 MB private-test blob inflates to tens of megabytes of
    parsed Python, and doing that under the submission's memory limit would
    charge the model for the size of the question.

    The privilege drop comes before the limits, not after, and the ordering is
    load-bearing: RLIMIT_NPROC is counted per real uid, so computing it while
    still root would measure root's task count — every process in the container
    — and then apply the answer to a uid that owns none of them.
    """
    if drop_to:
        drop_privileges(drop_to[0], drop_to[1])

    _set(resource.RLIMIT_CORE, 0)
    _set(resource.RLIMIT_FSIZE, FILE_SIZE_LIMIT)
    _set(resource.RLIMIT_NPROC, _current_task_count(os.getuid()) + FORK_SLACK)

    # The soft limit lands one second early so SIGXCPU arrives while the process
    # can still say what happened; the hard limit is the kernel's SIGKILL for a
    # submission stuck somewhere a Python signal handler cannot run (a long
    # sort, a big regex). Both are backstops — the parent's wall clock is the
    # primary, because it is the only one that catches a process that is asleep.
    cpu_seconds = max(1, cpu_seconds)
    _set(resource.RLIMIT_CPU, cpu_seconds, cpu_seconds + 1)

    # Raised, not lowered, and only as far as the hard limit allows: RECURSION_LIMIT
    # is worth nothing if the stack it recurses on is the default 8 MB.
    _set(resource.RLIMIT_STACK, 64 * 1024 * 1024, resource.RLIM_INFINITY)

    memory_bytes = max(MIN_MEMORY_MB, memory_mb) * 1024 * 1024
    _set(resource.RLIMIT_AS, memory_bytes)

    sys.setrecursionlimit(RECURSION_LIMIT)

    status = install_seccomp()
    block_network_in_python()
    return status
