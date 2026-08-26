# discrimen — dropshell template

Deploys two containers:

- `${SERVICE}` — `ghcr.io/j842/discrimen:latest` on `--network host`, with one
  Docker volume mounted at `/data` holding the SQLite database.
- `${SERVICE}_sandbox` — `ghcr.io/j842/discrimen-sandbox:latest` on the default
  bridge, published to `127.0.0.1:8587` and nowhere else. See
  [The grading sandbox](#the-grading-sandbox).

Both images are built and published by GitHub Actions. The target host pulls
them, so nothing needs a Go toolchain and every host runs the same binaries.

## Deploying over an existing llm-router service

This template replaces the private `llm-router` one. The service on `epyc.home`
is named `llm-router`, and the template sets `CONTAINER_NAME="${SERVICE}"`, so
the container is `llm-router` and the volume is `llm-router_data`. That volume
holds the `worker_profiles` table: the cold-start profiles measured for every
worker in the fleet.

**Keep the service name.** Do not create a new service called `discrimen`
alongside the old one. A new name derives a new volume name, orphans
`llm-router_data`, and re-benchmarks the whole fleet from cold — hours of GPU
time, and real money as soon as metered endpoints join.

Point the existing service at the new template instead. Register this repo as a
template source — templates are found by scanning for `template_info.env`, so
the repo root is enough, and the template's name is its directory name:

```
ds add-template /path/to/discrimen
```

Then edit the service's own config and change the template it points at:

```
ds edit llm-router epyc.home     # set TEMPLATE="discrimen"
ds install llm-router epyc.home
```

`SERVICE` stays `llm-router`, so the container name, the volume and the database
are all untouched. The install pulls the new image, `start.sh` sees the image
reference has changed and recreates the container against the same volume.

`LOG_DB_PATH` defaults to `/data/llm-router/logs.sqlite` and is frozen there for
backwards compatibility. Tidying it to `/data/discrimen/` starts an empty
database and leaves the profiles and log history stranded in the volume.

The old service.env carries variables this template no longer forwards. They are
harmless — leave them or delete them.

## Configuration

Fourteen variables, all in `config/service.env`. The test each one passed is
whether it describes something only the operator can know. Ports, credentials,
retention and how long a caller is willing to queue are facts about a site;
learning rates and tier bands are the router's own decisions and are now
constants in the binary.

Every variable has a default in `start.sh`, so a service.env missing a key still
starts.

| Variable | Default | What it does |
| --- | --- | --- |
| `ROUTER_PORT` | `8585` | Port the router listens on. Passed as an env var, not a `-p` mapping — the container is on the host network. |
| `LOG_DB_PATH` | `/data/llm-router/logs.sqlite` | Database path inside the volume. Frozen; see above. |
| `ROUTER_ADMIN_PASSWORD` | empty | Dashboard login. Bootstrap only: once set, the database is canonical, so changing this later does not rotate a stored password. |
| `ROUTER_WORKER_TOKEN` | empty | Bearer token workers present to register and unregister. Blank lets anything that can reach the port register itself as a worker. |
| `ROUTER_CLIENT_TOKENS` | empty | Comma-separated client tokens, one per consuming service so each rotates independently. Any of them authorises `/v1/*` and the read-only endpoints. Blank disables client auth. |
| `ROUTER_PERSIST_SECRET` | empty | Encrypts stored endpoint API keys at rest. Blank means the router generates `persist.key` (0600) beside the database instead, so encryption is on either way. Change it and previously persisted keys cannot be decrypted; those workers must re-register. |
| `LOG_RETENTION_DAYS` | `30` | How long request and response rows are kept. A privacy decision, and it varies with who is calling you. |
| `LOG_MAX_BODY_BYTES` | `16384` | Cap on stored body length per log row, so large prompts cannot grow the database without bound. |
| `HEALTH_INTERVAL_SECONDS` | `15` | Health poll interval across registered workers. Scales with fleet size. |
| `BACKEND_TIMEOUT_SECONDS` | `600` | Whole-exchange cap for buffered (non-streaming) requests and probes. Streaming is deliberately not capped by this. |
| `BACKEND_IDLE_TIMEOUT_SECONDS` | `120` | Streaming idle watchdog: abort when a backend sends no bytes at all for this long. This is what frees a hung worker's slot instead of pinning it for the full `BACKEND_TIMEOUT_SECONDS`. `0` disables. |
| `ROUTER_SLOT_MAX_WAIT_SECONDS` | `600` | How long a caller queues for a slot before getting a 503. A promise to clients, not a routing decision. |
| `DEFAULT_MAX_TOKENS` | `4096` | Token budget applied when a client declares none of its own. |
| `ROUTER_AUTO_ROUTING` | `true` | One switch for the whole automatic layer: difficulty classification, thinking mode, online adaptation and background judging. `false` makes discrimen a plain load balancer and drops the dependency on the embeddings worker (`ghcr.io/j842/discrimen-embeddings`, port 8586). |

Plus seven for the sandbox: `SANDBOX_PORT`, `SANDBOX_TOKEN`,
`SANDBOX_MAX_CONCURRENCY`, `SANDBOX_DEFAULT_TIMEOUT_MS`,
`SANDBOX_DEFAULT_MEMORY_MB`, `SANDBOX_CONTAINER_MEMORY`, `SANDBOX_PIDS_LIMIT`
and `SANDBOX_SCRATCH_MB`. Each is commented in `config/service.env`.

## The grading sandbox

LiveBench's coding questions carry an **empty `ground_truth`**. That is not an
oversight in the data: the answer to "write `minimumArrayLength`" is a function,
and the only thing that can tell a correct one from a plausible one is running
it against the test cases. So grading those questions needs an interpreter, and
the router must not be the thing holding it — it is a long-lived process with
the fleet's credentials, its request log and its database in one address space.

`${SERVICE}_sandbox` is that interpreter, in a different container:

```
POST /grade            code + tests   → {"pass":…, "cases_run":…, "cases_passed":…}
POST /decode-private   base64 blob    → {"tests":[…]}
GET  /health                          → {"status":"ok"}
```

`/decode-private` exists because LiveBench's `private_test_cases` are base64 of
zlib of a **pickle**, and unpickling is arbitrary code execution by design.
Doing it inside the jail is the whole reason it is an endpoint rather than a
function in the router.

### What contains what

Four layers, none sufficient alone:

| Layer | Where | Against |
| --- | --- | --- |
| `RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_FSIZE`, `RLIMIT_NPROC`, `RLIMIT_CORE` | `sandbox/jail.py`, set by the child on itself | busy loops, memory bombs, disk filling, fork bombs |
| a seccomp BPF filter removing `socket()` | `sandbox/jail.py` | exfiltration; there is no network namespace to take away |
| own session, wall-clock `SIGKILL` on the process group, a `/proc` sweep for anything that escaped it, a scratch dir destroyed on every path | `sandbox/supervisor.py` | sleeping processes, `setsid()` escapes, leftover files |
| `--read-only`, `--cap-drop=ALL`, `no-new-privileges`, `--pids-limit`, `--memory`, `--init`, non-root uid, tmpfs `noexec` scratch | `start.sh` | everything above being wrong |

The layers cover different things on purpose. An rlimit does not stop a process
that is asleep — it consumes no CPU and allocates nothing. A wall clock does not
stop a fork bomb from spawning while it waits. seccomp does not stop a memory
bomb. And the container flags stop none of them happening; they stop them
mattering to anything else on the host.

`no-new-privileges` is the one flag that is not merely defence in depth: the
kernel refuses `PR_SET_SECCOMP` from an unprivileged process without it, so
removing that line removes the network isolation too.

### Why the port is on 127.0.0.1

The router is on `--network host`; the sandbox is not. It sits on the default
bridge with `-p 127.0.0.1:${SANDBOX_PORT}:${SANDBOX_PORT}`, so the router
reaches it over the loopback interface they share and nothing off the machine
can reach a code-execution endpoint at all.

`ports.sh` deliberately does not report it. A loopback-only port listed as a
service port is an invitation to expose it.

That is also why `SANDBOX_TOKEN` is empty by default — an attacker who can
already connect to `127.0.0.1` can run code on the box anyway. Set it before you
publish the port anywhere else, and be clear that publishing it means publishing
remote code execution.

### When it is missing

A router with no sandbox still routes, still serves `/v1/*` and still benchmarks
every question that is not a coding question. So `install.sh` **warns** rather
than failing when the sandbox does not come up — failing would take a working
fleet offline over a feature that degrades.

But it degrades silently: an ungradeable coding question reads as a worker that
got it wrong, so a dead sandbox shows up as the whole fleet being worse at code
than it is. That is why `status.sh` reports a stopped or unhealthy sandbox as
`Error`, the same as a dead router.

A service installed before the sandbox existed has no such container, and that
is not an error — it is a template that has not been reinstalled yet.

## Reachability

A worker registers a URL and the router dials it. The URL has to be reachable
**from the router**, not from wherever you ran the registration.

`localhost` is the trap. This template runs the router with `--network host`, so
a worker on the same box can register `http://localhost:8080` and it will work —
but the moment the router moves to another host, or is run in a bridge network,
every `localhost` registration in the fleet points at the wrong machine and the
worker drops out at the next health poll. Register a LAN hostname or address the
router can resolve, and the registration keeps working wherever either end runs.

The symptom is a worker that registers cleanly and then never serves anything.
Check `/backends` and the container log.

## Behaviour on install

`IDEMPOTENT=true`, so dropshell skips the uninstall step and the router keeps
serving while template files are swapped.

- `install-pre.sh` pulls both images while the old containers are still up, and
  aborts the whole install if either pull fails, before anything is torn down.
- `install.sh` does not stop or remove anything. It calls `start.sh`, polls the
  router's `/health` for up to 10 seconds and dies if it never answers, then
  polls the sandbox's for up to 15 and only warns.
- `start.sh` converges both containers: it hashes each run command into a
  `ds.spec` label and inspects the image reference separately, so a newly pulled
  image or an edited service.env recreates that container, while an install that
  changed neither leaves it running. The sandbox is started first, because a
  router that comes up to find nothing on the port scores coding questions zero
  rather than failing visibly.

`uninstall.sh` preserves the volume. Only `destroy.sh` removes it, and doing so
throws away the measured fleet. Neither has anything to preserve for the
sandbox: it has no volume, and every run's scratch space is a tmpfs directory
deleted with the run.
