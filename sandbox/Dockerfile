# One stage and no pip install, which is the point rather than an omission.
#
# This image exists to be the blast radius: it is the only container in the
# deployment that ever executes a language model's output, and its dependency
# list is the surface an attacker who got that far would inherit. Stdlib-only
# means there is nothing to inherit and nothing that a transitive bump can add
# — no wheel to compile, no lockfile to audit, and a `docker build` that is a
# COPY. The whole image is the python:3.12-slim base plus five files.
#
# 3.12 specifically, not 3.13 or "3": pure-Python recursion stopped consuming C
# stack in 3.11, which is what makes jail.py's raised recursion limit end in a
# RecursionError instead of a segfault, and a floating tag would eventually move
# the interpreter under a jail whose assumptions were measured against this one.
FROM python:3.12-slim

# A dedicated uid, and it is load-bearing in two places. RLIMIT_NPROC is per
# real uid, so the fork-bomb ceiling is computed against whatever this user is
# already running — which is this service and nothing else, because nothing else
# in the image runs as anybody. And running as root, even with every capability
# dropped, would let a submission write to the parts of the filesystem that are
# still writable when --read-only is not set.
RUN groupadd --gid 10001 sandbox \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin sandbox

WORKDIR /app

# Owned by root, run as sandbox: the service cannot rewrite its own grader even
# before --read-only is applied. Five files listed rather than a COPY of the
# directory, so the test suite and any stray scratch file stay out of the image
# without a .dockerignore having to be remembered.
COPY main.py supervisor.py runner.py jail.py compare.py /app/

# Scratch space. The service makes one directory per run under here and removes
# it afterwards; start.sh mounts a tmpfs over it (noexec, size-capped) so the
# files a submission writes never reach a real filesystem and cannot grow past
# the mount. Owned by the sandbox uid because a tmpfs mounted over it inherits
# the mount's own mode, and the un-mounted case has to work too.
RUN mkdir -p /scratch && chown 10001:10001 /scratch
VOLUME /scratch

ENV SANDBOX_PORT=8587 \
    SANDBOX_SCRATCH_DIR=/scratch \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER 10001:10001
EXPOSE 8587

# The healthcheck is python's own urllib, not curl. The slim base ships neither
# curl nor wget, and installing one for a single loopback GET would add an apt
# layer and a package feed to the image whose selling point is that it has
# neither. Note this runs in the CONTAINER, not in a jailed child, so the
# seccomp filter that removes socket() does not apply to it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import os,sys,urllib.request;\
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('SANDBOX_PORT','8587')+'/health',timeout=4).status==200 else 1)"

# -u as well as PYTHONUNBUFFERED, because a container that is killed mid-write
# should still have logged the line that explains why.
CMD ["python3", "-u", "/app/main.py"]
