# discrimen — dropshell template

Deploys `ghcr.io/j842/discrimen:latest` as a single container on `--network host`,
with one Docker volume mounted at `/data` holding the SQLite database.

The image is built and published by GitHub Actions. The target host pulls it, so
nothing needs a Go toolchain and every host runs the same binary.

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

- `install-pre.sh` pulls the image while the old container is still up, and
  aborts the whole install if the pull fails, before anything is torn down.
- `install.sh` does not stop or remove anything. It calls `start.sh`, then polls
  `/health` for up to 10 seconds.
- `start.sh` converges: it hashes the run command into a `ds.spec` label and
  inspects the image reference separately, so a newly pulled image or an edited
  service.env recreates the container, while an install that changed neither
  leaves it running.

`uninstall.sh` preserves the volume. Only `destroy.sh` removes it, and doing so
throws away the measured fleet.
