# discrimen implementation plan

## What this is

discrimen is an OpenAI-compatible LLM router. It started life as `llm-router`
inside a private deployment repo, where it has been routing production traffic
across a mixed fleet of vLLM and llama.cpp workers. This document is the plan
for extracting it into a standalone public project and adding the pieces it
needs to serve endpoints you pay for and people you don't administer.

Nothing below is built yet. The routing itself already works; the work here is
packaging, configuration, and the parts that only start to matter once the
fleet includes a metered API and the client list includes strangers.

## What already works

Worth stating plainly, because most of the target design is already in the
code:

- OpenAI surface on one port: `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/models`, with SSE streaming.
- Automatic routing with no per-request configuration. The prompt is embedded
  once; a centroid pair decides whether it needs reasoning, and the embedding
  is looked up against every graded benchmark answer the fleet has produced to
  rank candidates by predicted correctness and then by speed.
- Manual selection through the same standard fields. Name a model, an alias or
  a worker id and you get it; say nothing and the router chooses.
- Cold-start profiling. The first time the router sees a worker it measures
  capabilities, context, speed, concurrency and quality, caches the result per
  (worker id, model), and reuses it instantly on a warm restart. The individual
  graded answers are cached separately and permanently, keyed by question and
  by model, so they outlive both the profiling run and the worker.
- Self-correction at runtime, from a background LLM-as-judge that grades a
  sample of answers served by a cheaper-than-best worker and writes the verdict
  into the same evidence table profiling fills.
- Push registration. Workers POST to `/backends/register` and re-post from a
  beacon sidecar every 60 seconds, gated on their own health endpoint.
- Model aliasing, so `/v1/models` publishes `qwen3.8` rather than a
  quant-encrusted file path, with the raw id preserved as `root`.
- A read-only dashboard and a SQLite request log.

## Goals

1. A public, standalone container that runs on its own, with a web page for
   configuring downstream OpenAI-compatible endpoints (local and internet) and
   for issuing access tokens.
2. Profiling that rates every endpoint, unchanged and still aggressive.
3. The fleet exposed through a standard OpenAI API.
4. Automatic routing kept, with manual model choice alongside it.
5. A readable model menu, aliases included.
6. Named groups such as `coding` or `research`, each with an ordered
   preference list, falling back to automatic routing when no member is
   available.
7. The existing push registration still working, so vanilla endpoints are the
   only ones needing a web-page entry.

## Design principles

These are inherited from the existing router and are not up for renegotiation
in this work.

### Operators don't tune routing

There are no quality tiers to hand-set and no speed cutoffs to pick. The router
asks which endpoints have got questions like this one right, and picks the
fastest of those. Every configuration surface added below governs which endpoints
exist and who may call them, never how a request is routed.

(When this was written the router derived a quality *target* from the fleet it
could see. The principle survived the mechanism: the outcome matrix replaced the
target, and there is still nothing to tune.)

### Measure, don't trust

A worker declares almost nothing: an id, a URL, an API key, and the handful of
tags that cannot be probed. Quality, speed, capacity, context and capabilities
are all measured. This stays true for internet providers, which get profiled
on exactly the same terms as local ones.

### Lenient to endpoints, careful about ourselves

The router adapts to whatever dialect an endpoint speaks rather than demanding
a strict one. No adapter sidecars, no per-provider translation layer. Where an
endpoint is fussy, the router discovers that by probing and works around it.
The northbound API, by contrast, is held to the standard properly, because
that is the contract people write clients against.

## Compatibility contract

Deployed workers must keep working with no edits. That means these stay
frozen:

- `POST /backends/register` and `/workers/register`, same payload shape
  (`id`, `url`, `model`, `features`, `api_key`), same bearer token, same
  upsert semantics, same 90-second TTL, same health refresh.
- `DELETE /backends/{id}`.
- The `ROUTER_*` environment variable names, including `ROUTER_WORKER_TOKEN`.
- The `X-LLM-Route` and `X-LLM-Backend-*` response headers. Clients parse
  these. The route string's *content* is not frozen and has moved with the
  routing policy — it now reads `route:outcome:p=…,n=…,sup=…` on a matrix-routed
  request. What is frozen is the `route:` / `model:` prefix, which is the
  router's own record of whether it or the client chose, and which is what keeps
  the router from learning from a decision a harness made for it.
- Port 8585.
- `LOG_DB_PATH` at `/data/llm-router/logs.sqlite`, and the container name used
  by the deployment template. The template derives its Docker volume name from
  the container name, so renaming the container orphans the volume, taking both
  `worker_profiles` and every graded answer with it and re-benchmarking the
  entire fleet from cold. That volume is the only thing the fleet's measured
  history lives in; back it up.

So: the new name goes on the box, the old names stay on the plumbing. A
registration that arrives without a `provider` field defaults to `local` at
zero cost per token, which is what every existing worker sends.

## Configuration surface

The router reads 42 environment variables today. After P0 it reads 14. The
test for whether a variable survives is whether it describes something only
the operator can know.

Hardware, network, ports, credentials, retention and how long a caller is
willing to queue are all facts about a site. Learning rates, classifier
thresholds, latency estimates, prefill discounts and tier bands are not: they
are the router's own decisions, and a site that has to set them has been
handed a problem the router was supposed to solve. The evidence that nobody
can reason about them is already in the deployment template, where
`ROUTER_QUALITY_FLOOR_WAIT_SECONDS` is 10 in one file and 3 in another, and
where the five `ROUTER_SESSION_*` variables are read by the binary but never
forwarded into the container at all.

These stay:

| Variable | Why it survives |
| --- | --- |
| `ROUTER_PORT` | Deployment. |
| `LOG_DB_PATH` | Deployment. |
| `ROUTER_ADMIN_PASSWORD` | New in P3. Bootstrap only; the database is canonical. |
| `ROUTER_WORKER_TOKEN` | Bootstrap. Frozen by the compatibility contract. |
| `ROUTER_CLIENT_TOKENS` | Bootstrap, until P3 keys take over. |
| `ROUTER_PERSIST_SECRET` | Encrypts stored endpoint keys at rest. |
| `LOG_RETENTION_DAYS` | Privacy policy, and it varies by who is calling you. |
| `LOG_MAX_BODY_BYTES` | Storage policy. |
| `HEALTH_INTERVAL_SECONDS` | Scales with fleet size. |
| `BACKEND_TIMEOUT_SECONDS` | Depends on how slow the slowest endpoint is. |
| `BACKEND_IDLE_TIMEOUT_SECONDS` | Same. |
| `ROUTER_SLOT_MAX_WAIT_SECONDS` | How long a caller queues before a 503, which is a promise to clients rather than a routing decision. |
| `DEFAULT_MAX_TOKENS` | What to do when a client declares no budget. |
| `ROUTER_AUTO_ROUTING` | New. One switch for the whole automatic layer: prompt classification (and therefore the outcome-matrix lookup it feeds), thinking detection, and judging. Off turns discrimen into a plain load balancer with no embeddings dependency, which is a legitimate thing to want and the only reason to touch any of this. Profiling is not part of it and keeps running. |

Everything else becomes a constant in the binary: the four `ROUTER_ADAPT_*`
rates, `ROUTER_REASONING_THRESHOLD`, `ROUTER_DIFFICULTY_QUALITY_BANDS`,
`ROUTER_DIFFICULTY_TEMP`, `ROUTER_DIFFICULTY_CACHE_SIZE`,
`ROUTER_DIFFICULTY_MAX_CHARS`, both `ROUTER_LATENCY_EST_*`,
`ROUTER_CONTEXT_ANSWER_RESERVE`, `ROUTER_DECODE_SAMPLE_REF_TOKENS`,
`ROUTER_DEADLINE_SAFETY_FACTOR`, `ROUTER_SPEED_SCORE_FULL_TPS`,
`ROUTER_UNCAPPED_NOMINAL_SLOTS`, `ROUTER_QUALITY_FLOOR_WAIT_SECONDS`, the five
`ROUTER_SESSION_*`, both `ROUTER_ESCALATE_*`, `ROUTER_PROFILE_WORKERS`,
`ROUTER_CAPACITY_PROBE_MAX`, and the three `ROUTER_AUTO_*` and
`ROUTER_JUDGE_SAMPLE_RATE` that `ROUTER_AUTO_ROUTING` replaces.

Two of those need a replacement rather than a plain constant.
`ROUTER_CAPACITY_PROBE_MAX` is superseded by the per-row declared
concurrency in P2, which is better targeted than a global ceiling.
`ROUTER_DIFFICULTY_TIMEOUT_SECONDS` becomes derived: the router already
measures every worker, so it can measure the embeddings worker's latency at
certification and set the classifier deadline from that. A fixed two seconds
silently disables classification on a slow box, and the health endpoint still
reports the worker as present, so the failure is invisible.

## Plan

### P0. Split the repo

Publish two multi-arch images from this repo, the router and the embeddings
worker, built by GitHub Actions to ghcr.io. Ship a `docker-compose.yml` that
brings up both, because auto-routing has a hard dependency on the embeddings
worker and a router without one silently falls back to plain quality and speed
ranking.

Move the routing defaults into the binary and cut the configuration surface
from 42 variables to the 14 listed above. `loadConfig` currently defaults
`ROUTER_AUTO_DIFFICULTY`, `ROUTER_AUTO_THINKING` and `ROUTER_ONLINE_ADAPT` to
false and the judge sample rate to zero, with only the deployment template
turning them on. A bare `docker run` therefore gives you the least interesting
version of the product. Flipping them has a useful side effect: once the
template only overrides deviations, forgetting to forward a variable can no
longer disable a feature by accident.

Confirm the LiveBench licence permits redistributing the question snapshot,
and carry its attribution into the repo. The generated half of the pool comes
from LiveBench (ICLR 2025) through the HuggingFace datasets API, and the
source URL is already recorded in the pool file.

Generate bootstrap credentials on first run and print them to the container
log. An empty client token list currently means no client authentication at
all, which is fine on a trusted LAN and wrong for a public image.

Write the README for someone who has never heard of the deployment tool:
quickstart, pointing it at a model both ways, the reachability footgun (a
worker registers a URL the router must be able to reach, and `localhost` from
inside a container is not it), what cold-start profiling will do to the first
few minutes, then the client guidance and endpoint reference. Deployment with
the private template gets a short section at the end.

The private repo's template shrinks to an image tag, a `service.env` for ports
and tokens and the data volume, and a short `start.sh`.

### P1. Southbound leniency, northbound politeness

Strip the router's own fields (`requirements`, `classify_text`, `deadline_ms`)
from the forwarded body. They are ours, no endpoint has ever needed them, and
a strict one will reject them.

Probe the thinking dialect instead of declaring it. The profiler already has a
thinking probe; extend it to work out which spelling an endpoint honours,
`chat_template_kwargs.enable_thinking` or `reasoning_effort` or neither, and
store that on the profile like every other measured capability.

Add strip-and-retry as the backstop. When an endpoint rejects a request with
an unknown-parameter error, retry once without the offending field and
remember the result against that backend.

Derive the classifier deadline from the embeddings worker's measured latency
instead of the fixed two seconds, as described under the configuration
surface.

Fix the two outright bugs in the table below, then bring the northbound API up
to standard: the OpenAI error envelope with a JSON content type, `GET
/v1/models/{id}`, `Retry-After` on 503, streaming failures as SSE error events
rather than a truncated stream, and acceptance of the legacy `function` role
that the validator currently rejects.

Scope: OpenAI-compatible base URLs only. Anthropic, Google and most others
publish one. Bedrock's SigV4 and its like are out, and that is a fine place to
stop.

### P2. Providers

Registry rows gain a `provider` (default `local`), a `source` of `beacon` or
`manual`, input and output prices per million tokens (default zero), and a
declared `max_concurrency` that outranks the capacity ramp. That last field is
the whole accommodation for metered endpoints: profiling stays fully
aggressive, but a rate-limited provider cannot have its capacity permanently
under-measured by a burst of 429s, in the same way that llama.cpp's published
slot count already outranks the ramp today.

One row per (endpoint, model), since a catalogue endpoint serves hundreds and
the router treats a row as one servable thing.

Manual rows are operator-owned: probes refine them, but never overwrite a
declared value. Beacon rows behave exactly as they do now. Seed prices and
context windows from LiteLLM's public price data where the model id matches.

Add the admin write API. The push registration endpoint is untouched.

### P3. Virtual keys and admin scope

An `api_keys` table: hash, displayed prefix, name, role of admin or client or
worker, enabled flag, creation and last-used timestamps, an optional model
allow-list, an optional token budget. Keys are `sk-` prefixed, hashed with
SHA-256 at rest, and shown once.

Admin access is a single password, hashed in the database, bootstrapped from
the environment, held in a session cookie. No OIDC and no external identity
provider.

Move `/logs` and `/backends` to admin scope. Today any client token can read
every prompt and response in the log, which is acceptable for a private fleet
and not acceptable the moment tokens go to other people. Stamp the key id on
every log row.

### P4. Cost in the ranker

Price is a declared fact about an endpoint, in the same category as the
`uncensored` tag, not a routing knob. Local workers cost nothing, which makes
the rule simple enough to need no tuning: among the workers that clear the
quality bar, prefer the free ones, and spill to a paid endpoint only when
nothing free clears the bar or every free candidate is saturated past the
existing grace period. Per-key budgets handle the rest.

(As built, "clear the quality bar" became "are in the band the outcome matrix
judged interchangeable on correctness". The scoping is what matters and it turned
out to matter a great deal: applied to the whole ranked list rather than to a
band, the cost preference sent every hard prompt to the one free local worker
however confident the matrix was that it would answer wrong.)

One interaction to fix here. The background judge grades with the best model
in the fleet, on a sampled fraction of cheap answers. Once the best model is a
paid one, that becomes a continuous spend on ordinary traffic. The judge
should prefer the best free model and fall back to a paid one only under a
budget cap.

### P5. Groups

A `groups` table holding a name and an ordered list of members, where a member
is a model id, an alias or a worker id. A resolver runs ahead of backend
selection: walk the list in order, take the first member that is registered,
healthy and past the hard filters, and if none qualify, drop the group filter
entirely and route automatically with an `X-LLM-Group: fallback` header and a
log line.

Groups appear in `/v1/models` owned by the router. They also fix a display
wrinkle: two workers running different builds of the same family have
different raw model ids that reduce to the same alias, which the menu then
suppresses as ambiguous even though routing still pools them correctly. A
group over both restores the readable name.

### P6. Admin UI

Extend the existing single-file dashboard, which has no build step, with tabs
for the fleet, providers, keys, groups and logs, behind the admin password.

### P7. Relay

Two fleets, run by the same operator in different places, where one may route
to part of the other. The obvious construction — register the far workers
directly, or point a provider row at the far router's port — is wrong in three
ways, and P7 is the three fixes.

**Slot accounting has to stay in one place.** Two routers dispatching to the
same GPUs each keep their own view of how busy those GPUs are, and both are
wrong by the size of the other's queue. So a relayed request goes to the other
ROUTER, which acquires the slot, queues and spills for both fleets at once.
The downstream always names the model on the way out, so the upstream is a
slot broker rather than a second classifier deciding a tier the downstream had
already decided.

**A measurement should cross rather than be repeated.** The quality benchmark
is 401 questions and roughly a million output tokens of the upstream's GPU time,
and it has already been run against exactly these workers. Running it again from
the downstream would spend the same GPUs to learn the same answer. "Measure,
don't trust" survives intact — they ARE measured, and `bench_version` on the wire
is the statement that the two measurements were made the same way. A mismatch is
not papered over: capacity, context and
capabilities still cross, because those are facts about the deployment rather
than the question set, and the quality is held at the provisional tier an
unproven worker gets. Re-measuring locally is not an option a relay has, which
is why the version is reported on `/health` rather than quietly worked around.

The one value that cannot cross is the thinking dialect. It describes how to
spell the gate to the endpoint that finally serves the request, and a relay's
immediate peer is a router — which speaks the client-facing OpenAI spelling
and translates onward. So it is `reasoning_effort` by construction.

**Content and accounting are different things.** A relay key marks its caller,
and a marked caller's request and response bodies never reach the log store.
The row does: which endpoint served, what it cost, how long it took, which key
spent it. An operator who could not see those could not answer "what is my
fleet doing" about the traffic they did not send, and a per-key budget that
stopped counting would not be a budget. Relayed outcomes are also kept out of
the judge, on the same reasoning that already excludes a named-model route: it
learns about a route THIS router chose, and a relayed prompt was chosen
downstream against a fleet this router cannot see — so feeding it here would
count one signal twice and write evidence about somebody else's decision into
this router's matrix.

Relay is a flag on a client key rather than a fourth role: the roles map onto
the three surfaces, and a relay calls the OpenAI one. The per-key model
allow-list, which already exists, is what limits a relay to part of a fleet —
and it holds through `allowsBackend`, so it constrains the auto route too, not
just what the caller may name.

Cycles are refused by construction. Each router generates and persists its own
id, every relayed request carries the chain of ids it has passed through, and
a router that finds itself in the chain answers 508. Generated rather than
configured: an environment variable two deployments copy from each other is
precisely how a cycle becomes undetectable.

The link is priced, not ignored, and it is priced in exactly one place. Every
latency term on a relayed row describes the far ENDPOINT with the link
excluded: imported that way, and stripped back out of the downstream's own
live samples. The round trip measured on the fleet poll is then added once,
where the estimate is built.

The obvious alternative — fold the round trip into the imported TTFT and skip
the prefill rate, on the grounds that a rate has nowhere to put a constant
offset — is wrong, and wrong in the expensive direction. Without a rate a
remote model is priced at a FLAT first-token latency however long the prompt
is, so against local workers that prefill at thousands of tokens a second it
wins every long-context comparison it should lose. Nor does it correct itself:
the live sampler only measures non-thinking turns, so a reasoning worker never
earns a rate that way — which is why a local worker's rate is seeded from its
probe for the same reason.

## Findings from the code review

Line numbers refer to the pre-extraction `llm-router` tree.

| Finding | Where | Consequence |
| --- | --- | --- |
| `ServedID` is taken from `/v1/models` `data[0]` | `profile.go:448-455` | On a catalogue endpoint that is an arbitrary model, and `patchForwardedBody` then stamps it onto every client request. Pin from the declared model instead. |
| Probes hardcode `"model": "default"` | `profile.go:242,353`, `main.go:2548,2577` | Any endpoint that validates model names rejects every probe. |
| Router-only fields ride through to the endpoint | `main.go:195-212` | Documented as harmless, and it is, right up until the endpoint validates its input. |
| Thinking is injected as `chat_template_kwargs` | `difficulty.go:1037` | A vLLM and llama.cpp extension, not standard. |
| Errors are not the OpenAI envelope | `main.go:233`, `main.go:479` | `validationError` serialises to `{"message":...}`, and the 401 path uses `http.Error`, which sends a JSON body with a `text/plain` content type. |
| `GET /v1/models/{id}` is missing | `main.go:327` | An exact-match route, so the path falls through to the dashboard handler and 404s. |
| The legacy `function` role is rejected | `main.go:2789-2792` | Stricter than OpenAI, and inconsistent with the session tracker, which handles that role at `session.go:167`. |
| `/logs` is client-scoped | `main.go:600` | Any client token reads every stored prompt and response. |
| Routing features default to off | `main.go:385-401` | A standalone container runs without the behaviour that distinguishes it. |
| Context and capacity discovery is engine-specific | `profile.go:502-555` | vLLM's `max_model_len` and llama.cpp's `/props`, with no third path, so a cloud endpoint reports an unknown context. |

Health checking already works for internet endpoints without a code change,
since `health_path` is per-registration and `/v1/models` is a fine choice for
a provider with no health route.

## Known costs

Three things this plan accepts rather than solves.

The benchmark is published in full, questions and answer key both. That
exposes the grading set to anything that scrapes GitHub, which over time means
models trained on it. The LiveBench-derived half is already public so nothing
changes there, but the hand-written trap questions have not been public
before. The mitigation is the one LiveBench itself was designed around: refresh
from upstream periodically, so the fleet is measured against questions the models
have not seen. Publishing is the right call anyway, because a quality score
nobody can inspect is a number you have to take on faith, and the whole point of
this router is not taking numbers on faith.

**A refresh is much cheaper than this paragraph originally assumed.** It used to
mean bumping the benchmark version, invalidating every cached profile, and
re-running the whole set on every worker. A graded answer is now filed under a
hash of the question and a hash of the model, so a refreshed question is simply a
new question with no cached verdict, while the questions that survived the
refresh keep theirs on every worker. Bumping `benchmarkVersion` is for a change
to the profiling *method*, not to the questions.

Profiling a paid model costs real money. The set is 401 questions, 393 of them
graded thinking-on with a 32k token ceiling, so a cold profile lands somewhere
near 0.8-1.2M output tokens, plus around 400k prompt tokens for the long-context
questions across the two passes. That is once per **model**, not per worker, and
it survives the worker: decommission the box and redeploy the same weights
elsewhere and nothing is re-asked. Warm restarts reuse the profile, and a version
bump re-runs the probes rather than the questions. Profiling stays aggressive by
choice. The UI should show what a run cost once it has finished, so the number is
at least visible.

A relay is trusted, and cannot be anything else. The downstream adopts the
upstream's quality, capacity and capabilities as measured, so an upstream that
claims a quality it has not earned is believed — there is no cheap way to
verify it that is not simply running the benchmark again, which is the cost the
import exists to avoid. That is the right trade between two halves of one fleet
under one operator and the wrong one anywhere else, and the mitigation is to say
so rather than to build a verification nobody would want to pay for: a relay is
a router you run, and a stranger's endpoint belongs in `/admin/providers`, which
measures what it is told.

It also couples the two deployments' benchmark versions. Bump one side and the
imported quality stops being adoptable until the other follows, which is a
version skew nothing else in this design has. The alternative — a version-free
quality — would be worse, because it is exactly the number that must not be
comparable by accident. A mismatch is not an outage: capacity and capabilities
still cross and the imported worker is held at the provisional 30 until the
versions agree, which `/health` reports under `relays`.
