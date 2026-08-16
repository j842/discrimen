# discrimen

A self-measuring, OpenAI-compatible LLM router.

Point it at some endpoints — local vLLM and llama.cpp workers, metered internet
providers, anything that speaks an OpenAI base URL — and it publishes one
`/v1/chat/completions` over all of them. Clients send a plain OpenAI request
with no routing fields, and discrimen decides which endpoint answers it.

The two ideas it is built on:

**Operators don't tune routing.** There are no quality tiers to hand-set and no
speed cutoffs to pick. The router scores each prompt and derives its targets
from the fleet it can see. Every configuration surface in this repository
governs which endpoints exist and who may call them, never how a request is
routed.

**Measure, don't trust.** An endpoint declares almost nothing: an id, a URL, an
API key, and the handful of tags that cannot be probed. Quality, speed,
capacity, context and capabilities are all measured, by running an objective
benchmark and a set of probes against it. Internet providers are profiled on
exactly the same terms as local ones.

A client that *does* want to choose says so in plain OpenAI — `model`,
`reasoning_effort` — and gets what it asked for. Both live on the one port:
automatic is what you get by saying nothing, not a mode you switch into.

---

## Quickstart

```bash
git clone https://github.com/j842/discrimen
cd discrimen
cp .env.example .env      # then set the tokens
docker compose up -d
```

That brings up two containers, and you want both. Auto-routing works by
embedding the prompt, and the only thing that can embed it is the embeddings
worker. Without one, discrimen does not fail — it falls back to plain
quality-and-speed ranking and says so on `/health` as
`"auto_routing": "degraded"`. Running the router alone is a legitimate
configuration; set `ROUTER_AUTO_ROUTING=false` and mean it. Running it alone by
accident is the failure the compose file exists to prevent.

Check it came up:

```bash
curl -s localhost:8585/health | jq
```

If you left the tokens empty in `.env`, discrimen generates them on first run
and prints them to its log. Grab them before you do anything else:

```bash
docker compose logs discrimen | grep -i bootstrap
```

Leaving them empty and ignoring the log means no authentication at all — fine
on a trusted LAN, wrong anywhere else.

## Adding a model

Two ways in, and they meet in the same registry.

**A worker registers itself.** This is how the local fleet works. The worker
POSTs to `/backends/register` on startup and re-posts every 60 seconds as a
keepalive, gated on its own health endpoint:

```bash
curl -X POST localhost:8585/backends/register \
  -H "Authorization: Bearer $ROUTER_WORKER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
        "id":  "workstation-qwen",
        "url": "http://192.168.1.50:8000",
        "model": "Qwen3.6-27B-FP8"
      }'
```

That is the whole payload. No quality, no speed, no context window, no
concurrency — the router measures all of it.

**You enter it on the web page.** For an endpoint that will never call you: an
OpenAI-compatible provider, a box you do not control, anything without a beacon.
Same registry row, marked `manual` instead of `beacon`, and owned by you — probes
refine a manual row but never overwrite a value you declared.

### The reachability footgun

`url` is the address **the router** uses to reach the endpoint. Not the
endpoint's view of itself, and not yours.

`localhost` from inside a container is that container. A worker on the same host
as the router in a bridge network that registers `http://localhost:8000` has
just pointed the router at its own port. Registration succeeds, health checks
fail or — worse — succeed against the wrong thing, and every request to that
worker breaks. It is the most common way a first deployment goes wrong, and it
breaks quietly.

Use an address that resolves and connects **from the router's network
namespace**: the service name under compose, a LAN address or hostname
otherwise.

### What the first few minutes look like

The first time discrimen sees an endpoint it has no profile for, it measures it
properly, and that takes a while.

Profiling splits in two so a fresh deployment does not black out the fleet. The
quick half — capabilities, speed, context — makes the endpoint routable in
**seconds**. The slow half — a concurrency ramp and a 102-question graded
benchmark — runs in the background and then refines the live values. Until the
benchmark finishes, an unproven endpoint holds a conservative provisional
quality of 30, so it only draws easy traffic.

The result is cached per `(endpoint id, model)` in SQLite, so a restart is
**instant**: same id, same model, profile reloaded, no re-measurement. Only a
brand-new endpoint or a changed model pays the cost again.

On a paid endpoint, that cost is money. The benchmark is 102 questions, most of
them graded thinking-on with a 16k token ceiling, so a cold profile lands
somewhere near 200-300k output tokens. Once. See
[docs/benchmark.md](docs/benchmark.md) for the whole question set and its answer
key.

## How a request is routed

For each `POST /v1/chat/completions`:

1. **Hard filters.** Drop endpoints that cannot serve it: not the model the
   client named, insufficient context (estimated from messages plus tools plus a
   nominal answer reserve — *not* the client's `max_tokens` ceiling), missing
   required features (`tools` detected from the request's `tools` field,
   `vision` from image content), or thinking required and unsupported.

2. **Difficulty to target quality.** An embedding-centroid classifier scores the
   prompt's difficulty in `[0,1]` and maps it onto the benchmark's absolute
   0-100 scale, clamped to the best quality actually registered. The bar is a
   property of the question, not of the fleet, so registering a high-quality
   slow endpoint cannot re-tier questions the existing fleet was already
   clearing. The online adapter adds a learned upward bias for that difficulty
   region.

3. **Reasoning to thinking mode.** A second centroid pair scores whether the
   prompt needs reasoning, and if so the router turns thinking on — in whichever
   dialect it measured the chosen endpoint to speak. Simple prompts run with
   thinking off.

4. **Rank by expected completion time.** Among the endpoints that clear the
   quality bar, take the one that will **finish soonest**: prefill time for this
   prompt, plus decode time for the expected output, plus queue occupancy. This
   replaces "cheapest tier". Slow endpoints lose on latency by themselves, with
   no speed cutoff to tune, and a busy fast endpoint sheds load to idle ones
   automatically.

   The estimate is per *request*, not per endpoint. Prefill scales with the
   prompt — an agent turn's system prompt and tool schemas run to thousands of
   tokens — and decode scales with the expected output, which is roughly six
   times longer once thinking is on. That matters because the two phases have
   very different cross-endpoint spreads: on the fleet this was built against, a
   4k-token prompt prefills in 0.67s on a GPU worker and 37.2s on a CPU one, a
   55x spread, while decode differs only about 2x. Ranking every request as a
   fixed 256-token job made a long thinking turn look identical to a short chat
   turn everywhere, and sent agent turns to CPU workers that needed over two
   minutes for them while the GPU sat idle.

   Two measurement rules keep the inputs comparable. **TTFT and prefill are
   sampled only from non-thinking turns**, because vLLM buffers reasoning so a
   thinking turn's whole think phase lands inside TTFT — 12.45s of a 13.15s turn
   — while llama.cpp streams it, 0.7s on the same job. Mixing them made the
   faster prefill engine look 30x slower. And **decode samples are weighted by
   generation length**, because llama.cpp CPU decode degrades as the KV cache
   grows: unweighted short replies had one CPU worker reporting 51 tok/s when it
   sustained 17 over 1700 tokens.

5. **Cost.** Among endpoints that clear the quality bar, prefer the free ones.
   Spill to a paid endpoint only when nothing free clears the bar, or every free
   candidate is saturated past the grace period below. Price is a declared fact
   about an endpoint, in the same category as an `uncensored` tag — not a
   routing knob.

6. **Spill.** Walk the ranked list and take the first endpoint with a free
   concurrency slot, so a saturated top pick overflows to the next. Two bounded
   preferences can briefly hold a request first: the quality floor waits for an
   above-target endpoint to free a slot, and while a tool loop is open the
   session lock waits for the incumbent. Neither can refuse a request; they only
   reorder who gets tried first.

The decision is reported back in headers: `X-LLM-Route` (`route:d=0.92,q>=10`
for an automatic pick, `route` for an explicit one), `X-LLM-Backend-Model` for
who answered, `X-LLM-Session` for what affinity did, `X-LLM-Escalated` when an
empty answer was repaired, `X-LLM-Group` when a group fell back to automatic.

**Want the decision without paying for the answer?** `POST /v1/route-preview`
with the same body runs the whole pipeline and returns what it *would* do:
classification, tier, thinking mode, session state, the ranked candidates with
their completion-time estimates, and why each excluded endpoint was excluded. It
contacts no endpoint and changes no state.

## Session affinity

Routing every turn from scratch is right for a one-shot prompt and wrong for a
tool loop. Moving turn N+1 elsewhere throws away the KV cache and re-prefills
the whole system prompt and tool schemas — exactly the prompt shape where
prefill dominates — and switching mid-loop hands a tool result to a model that
never emitted the matching tool call.

A conversation is identified by hashing the **head** of the message list, the
part that stays identical as turns accumulate. Hashing the conversation minus
its last message, the obvious first attempt, yields a different key every turn
and matches nothing. The tracker is in-memory, TTL'd and bounded, and evicts by
oldest use so a burst of one-shot prompts cannot push out a live agent session.

Two effects, both advisory. Neither hard-filters, 503s, or holds a request on an
endpoint that has left the candidate set:

| | effect |
|---|---|
| normal turn | prefill discount on the incumbent, inside the completion-time ranking |
| mid-tool-loop | incumbent preferred outright for a bounded grace, then spills |

Expressing stickiness as a prefill discount rather than a bias gives it the
right shape for free: it grows with the conversation, vanishes for a short
one-off, and a genuinely faster endpoint still wins.

The tool-loop lock deliberately outranks the quality floor. Half a tool loop
served by two models is worse than all of it served by the cheaper one.

Honest limit: the router cannot see an endpoint's KV cache. That needs
llm-d or Dynamo-style cache-event streams from the engine, not an OpenAI HTTP
body, so the discount is a proxy for cache locality rather than a measurement of
it.

## Inline escalation

An inadequate answer used to only nudge the tier adapter: the region got routed
higher *next* time, while the caller who hit the problem was handed the empty
response. Now, when an endpoint returns a 2xx with nothing in it, the request is
re-dispatched to a strictly better endpoint before replying.

Four deliberate boundaries:

- **Non-streamed only.** Once SSE bytes are on the wire they cannot be recalled.
- **Empty only, not truncated.** A length-capped answer hit the *caller's* token
  ceiling; a bigger model runs into the same wall and bills twice for it.
- **Router-chosen routes only.** A client that named a model or pinned an
  endpoint asked for that one. Answering from a different one would be a worse
  surprise than the empty reply.
- **One hop.** If the better endpoint is also empty, the original response is
  returned.

The escalation is fed to the adapter as "this bin needed a better model".
Without that, the repair would teach it the opposite, since the body it finally
returns looks clean.

## Profiling, and self-improvement

Profiling is the cold-start rating. Two runtime mechanisms correct the drift a
static benchmark misses.

**Online tier adapter.** A per-difficulty-bin, upward-only bias. It rises when a
response comes back inadequate and decays on clean ones, and is added before the
tier is computed, so regions that keep failing get routed higher over time.
Persisted, so learning survives a restart.

**Background LLM-as-judge.** A sampled fraction of answers served by a
cheaper-than-best endpoint are graded in the background by the best model, good
or bad. A bad verdict raises that bin's floor. This is what makes a fast-but-dim
endpoint safe rather than merely contained: a complete-but-wrong answer looks
like success to the inadequacy check, but the judge catches it. Judging is
dormant until you run something below the top tier, since the best model has
nothing better to grade it. It prefers the best *free* model, and falls back to
a paid one only under a budget cap — otherwise the arrival of a paid frontier
model would turn background grading into continuous spend on ordinary traffic.

**Throughput accounting** counts both content and reasoning tokens, so a
thinking-heavy endpoint is not mistaken for a slow one, which would poison the
latency ranking.

## Client guidance

Two kinds of client, one port, told apart by nothing more than the standard
OpenAI fields.

**Services: say nothing, get everything.** Send `"model": "default"` and no
`reasoning_effort`, and the router picks the tier, the thinking mode and the
endpoint. Model-tier selection is always automatic; there are deliberately no
`min_quality`, `min_speed` or `preference` overrides, and any such field an old
client still sends is ignored.

**Harnesses: ask in plain OpenAI.** A coding agent written against a normal
OpenAI server should not have to learn a private vocabulary:

| field | effect |
|---|---|
| `model: "default"` or absent | Automatic. The router chooses. Also published in `/v1/models` so it can be picked deliberately |
| `model: "<id from /v1/models>"` | Serve this model. Every endpoint running it stays a candidate, so a named model still load-balances. 404 if nothing serves it |
| `model: "<group name>"` | Serve from this group's preference list, falling back to automatic routing if no member qualifies |
| `model: "<endpoint id>"` | Same, narrowed to one endpoint |
| `reasoning_effort` absent | Automatic. The reasoning classifier decides |
| `reasoning_effort: "none"` | Thinking off |
| `reasoning_effort: <level>` | Thinking on at that level, hard-filtered to thinking-capable endpoints |

Levels are passed through verbatim rather than validated: the meaningful set
belongs to the endpoint's chat template, not to the router. DeepSeek branches on
`high` and `max`, other templates on other words.

A named model reports as `X-LLM-Route: model:d=…` instead of `route:d=…`, which
is also how the adapter and the judge know to learn only from tiers the *router*
chose.

`max_tokens` is a ceiling, not a reservation. The context filter charges a
nominal answer rather than the client's declared cap, and the cap is trimmed to
fit the chosen endpoint on the way out — so a harness declaring a huge budget no
longer excludes the cheap fleet from its own one-word prompts.

Clients must **not** append `/no_think` to prompts or set `chat_template_kwargs`
themselves. The router translates to the endpoint's measured dialect on every
route. Setting the chat-template gate directly remains supported as a low-level
escape hatch and wins over everything above, at the cost of switching the
automatic decision off.

## Endpoints

| Method | Path | Scope | Purpose |
|---|---|---|---|
| POST | `/v1/chat/completions` | client | Automatically routed chat |
| POST | `/v1/completions` | client | Legacy completions |
| POST | `/v1/embeddings` | client | Embeddings |
| GET | `/v1/models` | client | The model menu, aliases included |
| GET | `/v1/models/{id}` | client | One model |
| POST | `/v1/route-preview` | client | What this request *would* do. No generation, no state change |
| POST | `/v1/route-feedback` | client | Report a route outcome; feeds the adapter |
| GET | `/health` | none | Health and `auto_routing` status |
| POST | `/backends/register` | worker | Endpoint self-registration. Frozen interface |
| DELETE | `/backends/{id}` | worker or admin | Remove an entry, its persisted row and its cached profile |
| GET | `/backends` | admin | The fleet: quality, throughput, features, status |
| GET | `/logs` | admin | Stored request logs |
| — | `/admin/providers[/{id}]` | admin | CRUD over manually-entered endpoints |
| — | `/admin/keys[/{id}]` | admin | Issue, list, disable and delete API keys |
| — | `/admin/groups[/{id}]` | admin | CRUD over named groups |
| POST | `/admin/login`, `/admin/logout` | password | Session cookie |
| GET | `/` | none | Dashboard shell. Discloses nothing; the fleet table is fetched client-side with a token |

`/logs` and `/backends` are admin-scoped, not client-scoped. Any client token
used to read every stored prompt and response in the log, which is acceptable
for a private fleet and not acceptable the moment tokens go to people you do not
administer.

`/workers` and `/workers/register` are accepted as aliases of the `/backends`
spellings. Both are frozen: a worker deployed against an older version must keep
working with no edits.

## Configuration

Fourteen environment variables. The test for whether a setting belongs here is
whether it describes something only the operator can know — hardware, network,
ports, credentials, retention, how long a caller is willing to queue. Learning
rates, classifier thresholds, latency estimates and tier bands are not: they are
the router's own decisions, and a site that has to set them has been handed the
problem the router exists to solve. They are constants in the binary.

| Variable | Default | |
|---|---|---|
| `ROUTER_PORT` | `8585` | |
| `LOG_DB_PATH` | `/data/llm-router/logs.sqlite` | Also where the adapter state and the persistence keyfile land |
| `ROUTER_ADMIN_PASSWORD` | *(empty)* | Bootstrap only. The database is canonical afterwards |
| `ROUTER_WORKER_TOKEN` | *(empty)* | Bearer token an endpoint presents to register |
| `ROUTER_CLIENT_TOKENS` | *(empty)* | Comma-separated client bearer tokens |
| `ROUTER_PERSIST_SECRET` | *(empty)* | Encrypts stored endpoint API keys at rest. Blank generates a keyfile |
| `LOG_RETENTION_DAYS` | `30` | |
| `LOG_MAX_BODY_BYTES` | `16384` | Main driver of database growth |
| `HEALTH_INTERVAL_SECONDS` | `15` | Scales with fleet size |
| `BACKEND_TIMEOUT_SECONDS` | `600` | Whole-exchange cap for non-streaming requests |
| `BACKEND_IDLE_TIMEOUT_SECONDS` | `120` | Streaming idle watchdog. 0 disables |
| `ROUTER_SLOT_MAX_WAIT_SECONDS` | `600` | How long a caller queues before a 503 |
| `DEFAULT_MAX_TOKENS` | `4096` | Used when the client declares no budget |
| `ROUTER_AUTO_ROUTING` | `true` | One switch for the whole automatic layer |

`ROUTER_AUTO_ROUTING=false` turns discrimen into a plain load balancer with no
embeddings dependency: no difficulty scoring, no thinking detection, no
adaptation, no judging. That is a legitimate thing to want, and the only reason
to touch any of this.

The full `.env.example` explains each one in context.

## Groups

A group is a name and an ordered preference list, where a member is a model id,
an alias or an endpoint id. Call it as a model:

```json
{"model": "coding", "messages": [...]}
```

The resolver walks the list in order and takes the first member that is
registered, healthy and past the hard filters. If none qualify it drops the
group filter entirely, routes automatically, and says so with
`X-LLM-Group: fallback` and a log line — a group is a preference, not a
constraint that can fail a request.

Groups also fix a display wrinkle. Two endpoints running different builds of the
same family have different raw model ids that reduce to the same alias, which
the menu then suppresses as ambiguous even though routing still pools them
correctly. A group over both restores the readable name.

## Measuring the router

The cold-start benchmark measures **endpoints**. `discrimen arena` measures the
**router** — whether the classifier actually sends each prompt to the cheapest
endpoint that can answer it, which is the one claim the whole design rests on.

```bash
discrimen arena run -router http://localhost:8585 -dataset sub_10.jsonl -oracle -robustness
discrimen arena report -in arena-results.json
```

It follows [RouterArena](https://github.com/RouteWorks/RouterArena)'s shape —
accuracy, cost, routing optimality, robustness, routing latency — with two
departures. **Cost is worker-seconds, not dollars**, because there is no
per-token price on a self-hosted fleet and what a request costs is the time it
occupies. And **robustness measures the decision, not the answer**: perturbing a
prompt and re-asking costs another full pass, while asking `/v1/route-preview`
whether it still routes the same way costs milliseconds and tests the classifier
directly.

`-oracle` runs every question on every endpoint — questions × endpoints
generations, expensive, and the only way optimality can be computed at all. The
report splits the two failure directions that matter: **overspend**, where the
answer was right but a cheaper endpoint was also right, and **undershoot**,
where the answer was wrong and some endpoint had it. Grading uses the production
`checkAnswer`, so the harness cannot score a policy that never ran.

## Deploying

The compose file is the supported path and it is enough for most people.

If you use [dropshell](https://github.com/j842/dropshell), there is a template
at [`deploy/dropshell/discrimen`](deploy/dropshell/discrimen) that pulls the
published image, converges the container, and backs the data volume up through
restic. Its README covers the one migration trap: the Docker volume name is
derived from the container name, so renaming the container orphans the volume,
takes the cached profiles with it, and re-benchmarks the entire fleet from cold.

Images are published multi-arch (amd64 and arm64) to `ghcr.io/j842/discrimen`
and `ghcr.io/j842/discrimen-embeddings` on every push to main.

## Building

```bash
go build ./...
go test ./... -count=1
```

One direct dependency, `modernc.org/sqlite`, which is pure Go — hence
`CGO_ENABLED=0` and an alpine runtime image. Everything else is the standard
library.

## Source map

| | |
|---|---|
| `main.go` | server, registry, proxy, selection, persistence, health and certification loops |
| `difficulty.go` | embedding-centroid classifier, target quality, ranking, latency estimates |
| `profile.go` | cold-start profiling — capability, speed, context and capacity probes, and their persistence |
| `benchmark.go`, `benchmark_data.go` | the tiered quality benchmark |
| `session.go` | conversation identity, tool-loop detection, affinity tracker |
| `escalate.go` | buffered dispatch, inline escalation, strip-and-retry |
| `adapter.go` | online tier adapter |
| `judge.go` | background LLM-as-judge |
| `preview.go` | `/v1/route-preview` |
| `arena.go` | the router-level regression gate |
| `benchgen*.go` | the benchmark refresh pipeline: fetch, calibrate, emit |

Everything lives in `internal/router/`.

## Licence

MIT — see [LICENSE](LICENSE).

Two third-party snapshots ride along, both recorded in [NOTICE](NOTICE):

- **LiveBench** questions and ground-truth answers, under the Apache License
  2.0. Parts of that snapshot derive from material LiveBench does not own —
  competition problems copyright the Mathematical Association of America and
  the United Kingdom Mathematics Trust, both non-commercial-use-only. NOTICE
  carries those terms forward and says which subsets are affected. **Read it
  before putting the benchmark data to commercial use.**
- **LiteLLM**'s model price and context-window table, under MIT, used to seed
  prices on a manually-entered provider. Nothing to watch out for there.
