# discrimen

A self-measuring, OpenAI-compatible LLM router.

Point it at some endpoints (local vLLM and llama.cpp workers, metered internet
providers, anything that speaks an OpenAI base URL) and it publishes one
`/v1/chat/completions` over all of them. Clients send a plain OpenAI request
with no routing fields, and discrimen decides which endpoint answers it.

The two ideas it is built on:

**Operators don't tune routing.** There are no quality tiers to hand-set and no
speed cutoffs to pick. The router asks a different question instead: *which of
these endpoints has got questions like this one right, and which of those is
fastest*. It answers from graded evidence it collected itself. Every
configuration surface in this repository governs which endpoints exist and who
may call them, never how a request is routed.

**Measure, don't trust.** An endpoint declares almost nothing: an id, a URL, an
API key, and the handful of tags that cannot be probed. Quality, speed,
capacity, context and capabilities are all measured, by running an objective
benchmark and a set of probes against it. Internet providers are profiled on
exactly the same terms as local ones.

**Nothing measured is ever thrown away.** A graded answer is filed under a hash
of the question (its text, its expected answer, how it is matched, and the
version of the grader that scores it) and a hash of the model (served id,
parameter count, quantisation, file size, trained context, serving engine). So
it is evidence about the *weights*, not about the box: decommission a worker and
its results stay, deploy the same model on another host and it inherits the
whole profile instead of re-earning it, and fixing one grader re-asks that
grader's questions and nothing else.

A client that *does* want to choose says so in plain OpenAI, with `model` and
`reasoning_effort`, and gets what it asked for. Both live on the one port:
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
worker. Without one, discrimen does not fail. It falls back to plain
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
docker compose logs discrimen | grep -i -A4 -E 'bootstrap|GENERATED'
```

There are three banners, one each for the client token, the worker token and the
admin password. The credential sits three lines below the heading it belongs to,
so the `-A4` is doing the real work here. Without it you match three headings and
print no secrets.

Leaving them empty and ignoring the log means no authentication at all: fine
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
concurrency. The router measures all of it.

**You enter it on the web page.** For an endpoint that will never call you: an
OpenAI-compatible provider, a box you do not control, anything without a beacon.
Same registry row, marked `manual` instead of `beacon`, and owned by you: probes
refine a manual row but never overwrite a value you declared.

### The reachability footgun

`url` is the address **the router** uses to reach the endpoint. Not the
endpoint's view of itself, and not yours.

`localhost` from inside a container is that container. A worker on the same host
as the router in a bridge network that registers `http://localhost:8000` has
just pointed the router at its own port. Registration succeeds, health checks
fail or, worse, succeed against the wrong thing, and every request to that
worker breaks. It is the most common way a first deployment goes wrong, and it
breaks quietly.

Use an address that resolves and connects **from the router's network
namespace**: the service name under compose, a LAN address or hostname
otherwise.

### What the first few minutes look like

The first time discrimen sees an endpoint it has no profile for, it measures it
properly, and that takes a while.

Profiling splits in two so a fresh deployment does not black out the fleet. The
quick half (capabilities, speed, context) makes the endpoint routable in
**seconds**. The slow half, a concurrency ramp and a 401-question graded
benchmark, runs in the background and then refines the live values. Until the
benchmark finishes, an unproven endpoint holds a conservative provisional
quality of 30, so it only draws easy traffic.

While the slow half runs, `GET /backends` publishes what it is doing rather than
a bare `profiling` boolean: a `profile_progress` object with the phase
(`capabilities`, `capacity`, `prefill`, `quality`, `quality/nothink`,
`context`), how many questions of that phase are done, how many are in flight,
how long it has been going and — once eight questions have finished, which is
enough for the rate to mean anything — an estimate of how much longer. `ask -l`
renders it as a progress bar under the worker's row. A profile is the longest
thing this router runs, and "is it stuck?" used to have no answer short of
reading container logs.

The result is cached per `(endpoint id, model)` in SQLite, so a restart is
**instant**: same id, same model, profile reloaded, no re-measurement.

And when something *does* force a re-profile, the expensive part usually does not
run again. The probes are re-run — they are seconds — but every question the
model has already answered is served from the permanent cache described above,
because the cache is keyed by question and model rather than by profiling run.
So a grader fix re-asks that grader's questions on every worker and leaves the
rest alone, and a worker that comes back after a rebuild inherits its own
grading. The one thing that genuinely costs a full pass is a model the router
has never graded — a new endpoint, or a real change to the weights behind an
existing one.

The context window is the exception, because it is a deployment choice rather
than a property of the model: raise `--ctx-size` (or vLLM's `--max-model-len`),
restart, and the cache key does not move, so the profile would keep advertising
the old window. Discrimen re-reads it — one metadata GET, no benchmark — on
every certification and again on the health loop's periodic check, so a change
lands within ten minutes of the restart with nothing to do by hand.

Alongside the claim there is a needle-in-a-haystack ladder that climbs 4K, 8K,
16K… and stops when a rung fails, because a model advertising 256K that loses
facts past 64K passes an advertised-context filter and then answers badly —
which shows up as "the agent got confused", never as an error. Routing filters
on the *measured* window when the ladder found an edge. It does **not** when the
ladder merely ran out of room: the rungs double, and the next one has to fit
inside the claim to be attempted at all, so a 256K endpoint proves 128K and is
never asked about 256K. That is not a measurement, and reading it as one halved
every power-of-two window in the fleet — which is every large one. A claim the
ladder declined to test stands.

The ladder fills its haystacks using the same chars-per-token the hard filter
sizes prompts by, and that matters more than it looks: a rung licenses the filter
to admit a prompt it *calls* N tokens, which it can only do honestly if it tested
the amount of text the filter would call N tokens. The two used to disagree — the
probe filling at 4 characters per token while the filter divided by 3 — so a rung
labelled 128K was built from 512K characters the filter would have sized at 170K,
and the ladder was proving a different length from the one it reported.

On a paid endpoint, that cost is money. The benchmark is 401 questions: 393
graded thinking-on against a 32768-token ceiling, and 8 — the two control tiers —
graded thinking-off against a 1024-token one. A reasoning answer on this set runs
roughly 2000 to 3000 output tokens, well short of the ceiling, so a cold profile
lands near **0.8-1.2M output tokens**. Once, per model. Nine of the questions are
deliberately enormous on the *input* side (see below), which adds about 200k
prompt tokens to the thinking-on pass and the same again to the no-think one.

Two things bound that. The permanent cache means the number above is what a model
the router has never seen costs, not what every profiling run costs. And a
thinking-on pass is capped at 90 minutes of wall clock once at least 48 questions
are away, so a very slow endpoint is scored on what it managed rather than
holding a slot indefinitely. See [docs/benchmark.md](docs/benchmark.md) for the
question set and its answer key.

An endpoint that can think is scored **twice**: the headline quality grades the
hard tiers thinking-on (the mode discrimen serves them in when it decides), and
a second pass re-asks those same hard tiers with thinking disabled, merging in
the easy-tier answers that already ran thinking-off. The result is a separate
`quality_nothink` score, because a reasoning MoE with its thinking suppressed
can be a different, far worse model than the one the headline score describes —
measured on this fleet, a 35B A3B at quality 84 thinking-on wrote deterministic
garbage SQL thinking-off. A request that will be served without thinking (an
explicit `requirements.thinking: "off"`, `reasoning_effort: "none"`, a pinned
`enable_thinking: false`, or a direct verdict from the auto classifier) is
answered from the fleet's no-think evidence. That separation runs the whole way
down: every graded answer in the outcome matrix records which mode produced it,
and a lookup only ever matches rows from the same mode, so a model's thinking-on
record cannot vouch for its thinking-off behaviour. `quality_nothink` is the
human-readable summary of the same split.

### Nine very long questions

Since the bank gained long-context questions, three of the tiers also test how
much of a large prompt a model actually uses. Three synthetic machine logs at
roughly 4K, 16K and 48K tokens, each asked three ways: sum a set of values
scattered through the log, order a set of events and name the third, and find the
one record that contradicts a roster stated in the header. Every answer is
settled at least halfway down, and the first half of each log yields a *different*
answer, so a model that skims the opening and stops cannot pass.

They are not a needle-in-a-haystack test — the context probe already does single
needle lookup, and that is deliberately the one shape not used here.

**Their difficulty tiers (6, 9 and 10) are provisional and uncalibrated.** Every
other tier in the bank is a measured fleet pass-rate band; these three are an
author's ordering of three input lengths, assigned without running a single
worker. Treat the ordering as the only defensible claim — 48K is not easier than
16K, which is not easier than 4K — and re-band them from measured pass rates
before reading anything into the numbers. `docs/benchmark.md` says how.

A worker whose context window is too small for one of these is scored as having
**missed** it, decided before the request is dispatched so nothing is spent. That
is the honest answer for a weakness map: it used to be sent anyway, rejected by
the server, and recorded as an error — and an errored question stays out of the
denominator, so a 32K worker read as *unmeasured* at the 48K rung rather than as
unable to do it.

## How a request is routed

For each `POST /v1/chat/completions`:

1. **Hard filters.** Drop endpoints that cannot serve it: not the model the
   client named, insufficient context (estimated from messages plus tools plus a
   nominal answer reserve, *not* the client's `max_tokens` ceiling), missing
   required features (`tools` detected from the request's `tools` field,
   `vision` from image content), or thinking required and unsupported.

   The context estimate is *per candidate*, because a token is not a fixed
   amount of text: the same 700KB of Go and JSON is a different number of tokens
   to two models, by more than the margin that decides whether it fits. The
   divisor is **measured** — the endpoint reports `usage.prompt_tokens` on every
   response, its own tokenizer's count of the text just sent, and the router
   learns a chars-per-token per model from it (visible as `chars_per_token` in
   `/health`). It moves toward caution fast and away from it slowly, is clamped
   at both ends, and only samples prompts large enough that the chat template's
   fixed overhead is noise. Before that it was a flat 3.0 for every model, which
   on traffic that really runs 3.5 inflated a 200K prompt to a 233K estimate —
   enough to have every worker in the fleet refuse a prompt every one of them
   could hold.

   And when context is the *only* thing left standing between a request and a
   worker, the filter does not refuse it. A missing feature is a fact; context is
   this router's estimate, and the endpoint holds the truth. So the request goes
   to the widest window and the real tokenizer rules: either it fits, and the
   caller gets their answer plus a calibration sample, or the engine returns a
   400 that says by how much in exact tokens — which is a better answer than a
   503, and is *also* a sample, so the estimate that caused the refusal is
   corrected by the request it turned away. `X-LLM-Context-Overflow` reports it,
   and `/v1/route-preview` explains it.

2. **Embed the prompt, once.** An embeddings worker turns the prompt into a
   vector. Everything below reads that one vector: the reasoning score, and the
   lookup into the outcome matrix. A second centroid pair on the same vector
   scores prompt *difficulty*, which survives as a fallback and is what
   `X-LLM-Route` reports as `d=` when the matrix is not available.

3. **Reasoning to thinking mode.** A centroid pair scores whether the prompt
   needs reasoning, and if so the router turns thinking on, in whichever dialect
   it measured the chosen endpoint to speak. Simple prompts run with thinking
   off. Thinking-on and thinking-off are treated as two different models
   everywhere below, because on this fleet they measure like two different
   models.

4. **Rank by predicted correctness, then by speed.** This is the whole routing
   policy, and it is not a quality bar.

   The router finds the graded questions nearest this prompt in embedding space
   (up to 12 of them, cosine ≥ 0.55) and asks, per candidate, how that model did
   on those questions in this thinking mode. That gives a similarity-weighted hit
   rate, a confidence, and the weight of evidence behind it. Candidates fall into
   three bands, ranked in this order:

   | band | meaning | ordered by |
   |---|---|---|
   | **able** | predicted to get this right | correctness margin, then speed |
   | **unmeasured** | nothing like this has been graded on it | speed |
   | **unable** | graded on questions like this, and got them wrong | predicted correctness |

   Admission to the *able* band is on the hit rate **discounted for how thin the
   evidence is**, not on the raw rate — the penalty falls off as the square root
   of the evidence weight. That discount is not cosmetic: a worker that got one
   of two nearby questions right sat exactly on the 0.5 floor, was admitted, and
   because the band then sorted purely on speed it beat a worker with a dozen
   observations at 0.95 on a 3% speed edge.

   Nothing is ever *dropped* here, only ordered. An unmeasured worker is not a
   bad worker, and the ranked list is what failover and escalation walk along —
   shrinking it leaves them nowhere to go at exactly the moment the first choice
   has failed.

   **Exploration.** One in twenty opportunities (an opportunity being a request
   where there is both something known-good and something unmeasured to learn
   about), the fastest unmeasured worker is promoted to the head instead. That is
   how a newly registered endpoint earns evidence without waiting for a full
   profile, and it is the only place this step does not return its own best
   answer. Explored requests protect no ordering downstream, so the cost step
   below cannot quietly undo them.

   **When the matrix has nothing to say** about a prompt — which is the ordinary
   case for traffic unlike the question bank — it falls back to each model's
   overall graded hit rate, taking the fastest worker within 15 points of the
   best. A model never graded in this mode reads as a fleet-neutral 0.5 rather
   than as zero, so a fresh endpoint is not excluded from everything.

   **Where the time estimate comes from.** "Fastest" is prefill time for this
   prompt, plus decode time for the expected output, plus queue occupancy, plus
   one term the live figures cannot supply: how long questions *like this one*
   have historically made this worker generate for.

   The estimate is per *request*, not per endpoint. Prefill scales with the
   prompt (an agent turn's system prompt and tool schemas run to thousands of
   tokens) and decode scales with the expected output, which is roughly six
   times longer once thinking is on. That matters because the two phases have
   very different cross-endpoint spreads: on the fleet this was built against, a
   4k-token prompt prefills in 0.67s on a GPU worker and 37.2s on a CPU one, a
   55x spread, while decode differs only about 2x. Ranking every request as a
   fixed 256-token job made a long thinking turn look identical to a short chat
   turn everywhere, and sent agent turns to CPU workers that needed over two
   minutes for them while the GPU sat idle.

   Two measurement rules keep the inputs comparable. **TTFT and prefill are
   sampled only from non-thinking turns**, because vLLM buffers reasoning so a
   thinking turn's whole think phase lands inside TTFT, 12.45s of a 13.15s turn,
   while llama.cpp streams it, 0.7s on the same job. Mixing them made the
   faster prefill engine look 30x slower. And **decode samples are weighted by
   generation length**, because llama.cpp CPU decode degrades as the KV cache
   grows: unweighted short replies had one CPU worker reporting 51 tok/s when it
   sustained 17 over 1700 tokens.

5. **Cost, inside the correctness band.** Among the workers the matrix judged
   interchangeable — the *able* band, or every candidate when there is no
   correctness judgement to protect — prefer the free ones, and among those the
   local ones. Spill to a paid endpoint only when nothing free is left in the
   band, or every free candidate is saturated past the grace period below. Price
   is a declared fact about an endpoint, in the same category as an `uncensored`
   tag, not a routing knob.

   **The band is the point.** Preferring local is not really about the link being
   slow — prefill already prices the round trip, and it is milliseconds against a
   generation measured in seconds. It is that the two occupancy numbers are not
   equally trustworthy: a local queue depth is exact and live, while a relayed
   one is a snapshot up to 15 seconds old and blind to the upstream router's own
   clients. Ranking on remote completion times alone systematically over-picks
   remote, and this corrects that. What it must not do is reach *across* a
   correctness boundary to do it — applied to the whole ranked list, it sent
   every hard prompt to the one free local CPU worker no matter how confident the
   matrix was that the worker would get it wrong.

6. **Spill.** Walk the ranked list and take the first endpoint with a free
   concurrency slot, so a saturated top pick overflows to the next. Two bounded
   preferences can briefly hold a request first: the cost preference above waits
   inside its band, and while a tool loop is open the session lock waits for the
   incumbent. Neither can refuse a request; they only reorder who gets tried
   first.

The decision is reported back in headers: `X-LLM-Route`
(`route:outcome:p=0.95,n=12,sup=8.3` for an automatic pick — predicted
correctness, observation count and evidence weight for the worker at the head —
`route:outcome:explore,1in20` when the request was spent on exploration,
`route:outcome:unknown,fallback-speed,q=0.71` when nothing similar had been
graded, and bare `route` when the classifier was unavailable),
`X-LLM-Backend-Model` for who answered, `X-LLM-Session` for what affinity did,
`X-LLM-Escalated` when an empty answer was repaired, `X-LLM-Group` when a group
fell back to automatic. A route the *client* chose reports `model:` in place of
`route:`, which is how the router keeps its own learning off decisions a harness
made for it.

**Want the decision without paying for the answer?** `POST /v1/route-preview`
with the same body runs the whole pipeline and returns what it *would* do:
classification, thinking mode, session state, group and ensemble resolution, and
which worker would serve. An admin caller additionally gets, for every candidate,
the matrix's prediction — band, predicted correctness, the evidence-discounted
figure band admission actually used, confidence, evidence weight, observation
count, how many of those came from the background judge — alongside the
completion-time estimates and the reason each excluded endpoint was excluded. It
contacts no endpoint and changes no state.

Note what it does *not* return on a matrix-routed request: `target_quality` is 0
and `above_bar` is absent, because there is no quality bar involved. Both belong
to the difficulty-tier ranker described under "What is left of the tier system"
below.

## Session affinity

Routing every turn from scratch is right for a one-shot prompt and wrong for a
tool loop. Moving turn N+1 elsewhere throws away the KV cache and re-prefills
the whole system prompt and tool schemas, which is exactly the prompt shape
where prefill dominates, and switching mid-loop hands a tool result to a model that
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

The tool-loop lock deliberately outranks the cost preference, and the correctness
band with it. Half a tool loop served by two models is worse than all of it
served by the weaker one.

Honest limit: the router cannot see an endpoint's KV cache. That needs
llm-d or Dynamo-style cache-event streams from the engine, not an OpenAI HTTP
body, so the discount is a proxy for cache locality rather than a measurement of
it.

## Inline escalation

An inadequate answer used to be a lesson for the *next* caller only: something
was nudged, and the caller who hit the problem was handed the empty response.
Now, when an endpoint returns a 2xx with nothing in it, the request is
re-dispatched to a strictly better endpoint before replying.

Four deliberate boundaries:

- **Non-streamed only.** Once SSE bytes are on the wire they cannot be recalled.
- **Empty only, not truncated.** A length-capped answer hit the *caller's* token
  ceiling; a bigger model runs into the same wall and bills twice for it.
  Truncation gets a different repair — see the length rescue below.
- **Router-chosen routes only.** A client that named a model or pinned an
  endpoint asked for that one. Answering from a different one would be a worse
  surprise than the empty reply.
- **One hop.** If the better endpoint is also empty, the original response is
  returned.

An escalation repairs the request in front of it and teaches routing nothing
directly. What the outcome matrix learns from the same exchange arrives by the
other route: the background judge grades the answer that
was finally returned and attributes it to the worker that produced it, which is
the escalated one. The worker that returned nothing is not currently credited
with having done so.

## Length rescue

A hybrid reasoning model can pour its whole token budget into the thinking block
and stop at the cap before writing a word of the answer. What comes back is a
200 with `finish_reason: "length"`, a full reasoning trace, and `content: ""`.
To the caller that is a failed turn; it is not even *empty* by the classifier's
reckoning, because the reasoning trace counts as output — so escalation never
fired on it and the caller was handed the nothing.

The repair is one cheap follow-up turn to the **same** endpoint: replay the
conversation, append the working notes as an incomplete assistant turn, and ask
for the conclusion with thinking off and a small budget. The prefix is already
in that worker's KV cache, so the second pass is mostly decode. The reply is
spliced into the original response — the trace is kept, the usage of both
generations is added up, and `finish_reason` becomes `stop`, because it is now
true. `X-LLM-Rescue: length` reports it.

Same endpoint, deliberately: the fix is not a better model — a bigger one hits
the same ceiling and bills twice for it — it is asking this one to stop
thinking. The off-switch is written in the spelling that endpoint was *measured*
to honour, which is why this lives in the router rather than in a per-worker
sidecar.

It fires only on a truncation with **no content and a reasoning trace**. A
truncation that produced real text is the caller's `max_tokens` doing its job,
and re-asking would bill them for a shorter answer they did not want; a response
carrying a tool call is not a failed turn whatever its finish reason; and a
truncation with nothing in it at all is escalation's case, since there is
nothing to conclude from. Non-streamed only, one attempt, and it never returns
anything worse than what it was given.

## Profiling, and self-improvement

Profiling is the cold-start rating. One runtime mechanism keeps it honest against
the traffic the question bank does not resemble.

**Background LLM-as-judge.** A sampled fraction of answers served by a
cheaper-than-best endpoint are graded in the background by a stronger model, good
or bad, and the verdict is written into the outcome matrix as evidence about that
model on that kind of prompt — the same table profiling writes into, alongside
the prompt's embedding so future prompts can find it. This is what makes a
fast-but-dim endpoint safe rather than merely contained: a complete-but-wrong
answer looks like success to the inadequacy check, but the judge catches it.

The grader is picked per *prompt*, not per worker average: the judge reuses the
embedding routing already computed and asks which worker is strong on prompts
*like this one*. Choosing on overall hit rate sent every verdict to whichever
model happened to score best on the bank, including for subjects it was weak at.
It prefers the best *free* model and falls back to a paid one only under a budget
cap — otherwise the arrival of a paid frontier model would turn background
grading into continuous spend on ordinary traffic.

Judged evidence is deliberately weaker than bench evidence. It counts half as
much toward the "is there enough to go on" threshold, so two judged verdicts do
not qualify a worker on their own, and it is discounted again in the
evidence-weighted rate. It was graded by another model, on a prompt that merely
resembles the one being routed; it is good routing evidence and it is not a
substitute for asking the question. Judged rows are also the only rows that are
ever pruned — the oldest are dropped past a cap of 4000 questions, while a
question with any bench grading behind it is never pruned at all, because the
bank is the fixed instrument every worker's summary is computed over.

**Throughput accounting** counts both content and reasoning tokens, so a
thinking-heavy endpoint is not mistaken for a slow one, which would poison the
latency ranking.

### What is left of the tier system

Routing used to work the other way round: score the prompt's difficulty, map that
onto a target quality on the benchmark's 0-100 scale, and take the fastest worker
clearing the bar. The outcome matrix replaced it, for a reason worth stating
plainly — a single number per worker cannot say that a model is excellent at
maths and poor at code, and compressing a fleet's measured strengths into one
scalar is exactly the information the router needed to keep.

Three pieces of the old system are still in the tree and it is worth knowing what
they now do:

- **The quality scores** (`quality`, `quality_nothink`) are still measured and
  still published on `/backends` and `ask -l`. They are a good summary for a
  person. Routing no longer ranks on them.
- **The difficulty-tier ranker** is still there as a fallback for a router with
  no outcome matrix, which in practice means the test suite. On a deployed
  router the matrix ranks rather than declining, so the tier branch is not
  reached.
- **The online tier adapter is gone.** It learned a per-difficulty-bin upward
  bias from inadequate answers and fed the tier branch above, which nothing
  reaches. Removed outright rather than left switched on and inert: its config
  knobs, its `tier_adapter.json` persistence, its "online tier adaptation
  enabled" boot line and the `POST /v1/route-feedback` endpoint that was its
  only remaining writer have all gone with it. Its response-adequacy classifier
  was never really the adapter's and lives on in
  `internal/router/inadequacy.go`, where escalation and the expert panel use it.

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
| `model: "expert"` | Ask every model, then have the best one write the final answer. See below |
| `reasoning_effort` absent | Automatic. The reasoning classifier decides |
| `reasoning_effort: "none"` | Thinking off |
| `reasoning_effort: <level>` | Thinking on at that level, hard-filtered to thinking-capable endpoints |

Levels are passed through verbatim rather than validated: the meaningful set
belongs to the endpoint's chat template, not to the router. DeepSeek branches on
`high` and `max`, other templates on other words.

A named model reports as `X-LLM-Route: model:…` instead of `route:…`, so an
operator can tell a client's choice from the router's at a glance. Nothing
branches on the spelling — the judge is gated on a structural flag set where the
plan is built — but it is the first thing worth knowing when a route looks
wrong.

`max_tokens` is a ceiling, not a reservation. The context filter charges a
nominal answer rather than the client's declared cap, and the cap is trimmed to
fit the chosen endpoint on the way out, so a harness declaring a huge budget no
longer excludes the cheap fleet from its own one-word prompts.

Clients must **not** append `/no_think` to prompts or set `chat_template_kwargs`
themselves. The router translates to the endpoint's measured dialect on every
route. Setting the chat-template gate directly remains supported as a low-level
escape hatch and wins over everything above, at the cost of switching the
automatic decision off.

## expert

`"model": "expert"` is a reserved name that routes differently from everything
else here. Instead of choosing one endpoint, it asks **every model in the fleet
the same question**, then hands all the answers to the highest-measured-quality
endpoint and asks it to write the final answer.

```json
{"model": "expert", "messages": [{"role": "user", "content": "..."}]}
```

You get back an ordinary chat completion containing that final answer and
nothing else. No panel, no candidate list, no working: the members' reasoning
traces are stripped before the synthesiser ever sees them, and stripped again
from the reply. If you asked for a stream, the synthesis is streamed.

One member per distinct model, not per endpoint, so three endpoints running the
same model count once. A member that errors, is saturated, or comes back empty
is simply absent. With one usable answer it is returned directly, because there
is nothing to gather. With none, 503.

Two things are worth knowing before you point traffic at it.

**It costs what it sounds like.** N generations plus one synthesis, every time.
Per-key budgets apply and the reported `usage` is the true total across the
whole panel, so it is metered honestly rather than cheaply.

**Tool calls cannot be merged.** A request carrying `tools`, or one arriving
mid-tool-loop, silently falls back to normal automatic routing and says so with
`X-LLM-Expert: fallback=tools`. A synthesiser inventing one tool call out of
five disagreeing ones would be worse than any single model's answer.

The route reports itself in headers: `X-LLM-Route: expert` and
`X-LLM-Expert: members=3,skipped=1,synth=<endpoint id>`. It deliberately does
not look like a router-chosen route, so the background judge learns nothing from
it: the answer is a synthesis of the whole fleet's work, and crediting it to the
synthesiser would record a hit rate for a model that did not earn it alone.

## Endpoints

| Method | Path | Scope | Purpose |
|---|---|---|---|
| POST | `/v1/chat/completions` | client | Automatically routed chat |
| POST | `/v1/completions` | client | Legacy completions |
| POST | `/v1/embeddings` | client | Embeddings |
| GET | `/v1/models` | client | The model menu, aliases included |
| GET | `/v1/models/{id}` | client | One model |
| POST | `/v1/route-preview` | client | What this request *would* do. No generation, no state change. An admin caller additionally gets the per-worker outcome predictions and time estimates, and the reason each worker was excluded |
| GET | `/health` | none | Health and `auto_routing` status |
| POST | `/backends/register` | worker | Endpoint self-registration. Frozen interface |
| DELETE | `/backends/{id}` | worker or admin | Remove an entry, its persisted row and its cached profile. **Its graded answers stay**: they are evidence about the model, not about the box |
| GET | `/backends` | admin | The fleet: measured hit rate and its per-topic breakdown in both thinking modes, quality scores, throughput, features, status, and live `profile_progress` while a cold profile runs |
| GET | `/backends/{id}` | admin | One endpoint's row in full |
| GET | `/backends/{id}/benchmark` | admin | Per-question results for that endpoint's stored profile, plus the per-category breakdown (coding / maths / reasoning / general) in both thinking modes. Answers served from the permanent cache are marked as cached rather than re-asked |
| POST | `/debug/backends/{id}/certify`, `/debug/backends/{id}/chat` | admin | Re-profile one endpoint, or prompt it directly |
| GET | `/logs` | admin | Stored request logs |
| GET | `/admin/usage` | admin | Requests in flight over time, bucketed. `?range=1h` for the overview's one-hour frame in one-minute columns, otherwise twelve hours in five-minute ones; `?by=backend` cuts the bands on the worker that served each request instead of the address it came from. Carries a `totals` block counting the window once per request |
| GET | `/admin/outcomes` | admin | Whether the matrix's predictions hold up: held-out accuracy over the graded evidence it is routing on |
| any | `/admin/providers[/{id}]` | admin | CRUD over manually-entered endpoints |
| any | `/admin/keys[/{id}]` | admin | Issue, list, disable and delete API keys |
| any | `/admin/groups[/{id}]` | admin | CRUD over named groups |
| any | `/admin/relays[/{name}]` | admin | CRUD over upstream routers this one relays to |
| GET | `/relay/fleet` | relay | What a downstream router may see of this fleet: one entry per model it is allowed, with the measured profile and live occupancy |
| POST | `/admin/login`, `/admin/logout` | password | Session cookie |
| GET | `/` | none | Dashboard shell. Discloses nothing; the fleet table is fetched client-side with the admin session cookie |

The dashboard opens on an **overview**: what the fleet is holding right now
against the concurrency the router measured for it, one meter per worker, and the
last hour of load in one-minute columns — split either by the address the work
came from or by the worker that carried it. Behind that are tabs for the fleet,
twelve hours of the same chart, providers, keys, groups, relays and the request
log.

Two things on that page measure different clocks and the labels say so.
`/backends` is live: what is in flight at the instant of the poll. The hour
figures come from the request log, and **a request is written to the log when it
finishes**, so they trail reality by however long the in-flight requests have been
running. During a burst the two will not add up.

The dashboard holds no bearer token. It signs in through `POST /admin/login`,
which sets an HttpOnly session cookie, and every fetch it makes is same-origin so
the browser attaches the cookie by itself. The previous version prompted for a
key and kept it in `sessionStorage`; the current page deletes any key that
version left behind rather than migrating it.

`GET /backends`, `GET /workers`, `GET /backends/{id}`,
`GET /backends/{id}/benchmark`, `GET /logs` and
`POST /debug/backends/{id}/{certify,chat}` are **admin** scope, not client scope.
A client token could previously read every stored prompt and response in the log,
which was acceptable for a private fleet and stops being acceptable the moment
tokens go to people you do not administer. `DELETE /backends/{id}` takes a worker
credential or admin, and no longer a client token.

That does not mean carrying two credentials. **One admin-role API key satisfies
both scopes.** Issue it by POSTing `{"name": "ops", "role": "admin"}` to
`/admin/keys`, and that single key covers `/v1/chat/completions` and the rest of
the OpenAI surface, `/backends`, `/logs`, `/admin/*` and worker registration.
There is no authority a client has that an admin does not, so the OpenAI surface
accepts an admin key rather than making an operator hold a second credential to
test what they just configured.

**Relay** is not a fourth role. It is a flag on a client key, and the only thing
it opens is `GET /relay/fleet` — everything else it does is subtractive (see
[Relay](#relay)). A client key without it is refused there, because the fleet's
measured capacity and live occupancy is most of what moving `/backends` behind
the admin gate was meant to protect.

`/workers` and `/workers/register` are accepted as aliases of the `/backends`
spellings. Both are frozen: a worker deployed against an older version must keep
working with no edits.

## Configuration

Fourteen environment variables. The test for whether a setting belongs here is
whether it describes something only the operator can know: hardware, network,
ports, credentials, retention, how long a caller is willing to queue. Learning
rates, classifier thresholds, latency estimates, the correctness floor, the
exploration rate and the old tier bands are not: they are the router's own
decisions, and a site that has to set them has been handed the problem the router
exists to solve. They are constants in the binary.

| Variable | Default | |
|---|---|---|
| `ROUTER_PORT` | `8585` | |
| `LOG_DB_PATH` | `/data/llm-router/logs.sqlite` | The request log, and in the same file the outcome matrix — so a profile's graded results and the requests that produced them are backed up and restored together. Its directory is also where the persistence keyfile lands |
| `ROUTER_ADMIN_PASSWORD` | *(empty)* | Seeds the admin password, and resets it on any start where it is set. Unset leaves a stored password alone |
| `ROUTER_WORKER_TOKEN` | *(empty)* | Bearer token an endpoint presents to register |
| `ROUTER_CLIENT_TOKENS` | *(empty)* | Comma-separated client bearer tokens |
| `ROUTER_PERSIST_SECRET` | *(empty)* | Encrypts stored endpoint API keys at rest. Blank generates a keyfile |
| `LOG_RETENTION_DAYS` | `30` | |
| `LOG_MAX_BODY_BYTES` | `16384` | Main driver of database growth. An oversized body is stored head-and-tail with a marker naming the dropped middle, so the system prompt and the turn that was actually asked both survive |
| `HEALTH_INTERVAL_SECONDS` | `15` | Scales with fleet size |
| `BACKEND_TIMEOUT_SECONDS` | `600` | Whole-exchange cap for non-streaming requests |
| `BACKEND_IDLE_TIMEOUT_SECONDS` | `120` | Streaming idle watchdog. 0 disables |
| `ROUTER_SLOT_MAX_WAIT_SECONDS` | `600` | How long a caller queues before a 503 |
| `DEFAULT_MAX_TOKENS` | `16384` | Used when the client declares no budget |
| `ROUTER_AUTO_ROUTING` | `true` | One switch for the whole automatic layer |

`ROUTER_AUTO_ROUTING=false` turns discrimen into a plain load balancer with no
embeddings dependency: no prompt classification, so no outcome-matrix lookup and
no thinking detection, and no judging. Ranking falls back to quality and speed.
Profiling still runs — the benchmark is how the fleet is measured, not part of
the automatic layer — so the scores and the graded answers keep accumulating.
That is a legitimate thing to want, and the only reason to touch any of this.

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
`X-LLM-Group: fallback` and a log line. A group is a preference, not a
constraint that can fail a request.

Groups also fix a display wrinkle. Two endpoints running different builds of the
same family have different raw model ids that reduce to the same alias, which
the menu then suppresses as ambiguous even though routing still pools them
correctly. A group over both restores the readable name.

## Relay

A relay is a second discrimen, somewhere else, whose fleet this one may route
to. Add it once and it expands into one backend per model the other router
publishes to you; they appear in `/v1/models` and rank against your local
workers like anything else.

**On the upstream** — issue a client key with the relay flag, and an allow-list
saying which models it may reach:

```bash
curl -X POST http://upstream:8585/admin/keys \
  -H "Authorization: Bearer $ADMIN_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"home","role":"client","relay":true,"models":["model-a","model-b"]}'
```

**On the downstream** — point a relay at it with that key:

```bash
curl -X POST http://localhost:8585/admin/relays \
  -H "Authorization: Bearer $ADMIN_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"work","url":"http://upstream:8585","api_key":"sk-…"}'
```

That is the whole configuration. Both ends also have a tab on the dashboard.

The allow-list is what limits a relay to part of a fleet, and it holds through
`allowsBackend`, so it constrains the automatic route as well as what the
downstream may name. Write it in whichever spelling you prefer — a model id, an
endpoint id or an alias: `/relay/fleet` publishes each model back under the name
this key was actually issued for, so everything the downstream discovers is
something it is allowed to say.

It is not the same thing as pointing a provider row at the other router's port,
and the difference is three things.

**The upstream keeps the slots.** A relayed request is dispatched to the other
router, which acquires the worker slot, queues, ranks and spills exactly as it
does for its own traffic. Register the same workers directly on two routers and
each keeps its own idea of how busy those GPUs are, and both are wrong by the
size of the other's queue.

**The profile crosses instead of being re-measured.** Quality, capacity,
context, capabilities and the thinking dialect were already measured upstream by
the same graded benchmark this binary carries — 401 questions and roughly a
million output tokens of somebody's GPU time. `bench_version` on the wire is what
says the two measurements were made the same way; when it does not match,
capacity and capabilities still cross and the quality is held at the provisional
30 an unproven worker gets, rather than adopting a score derived by a different
method. `/health` reports the mismatch under `relays`.

**Relayed traffic leaves no prompts upstream.** A relay key marks its caller,
and a marked caller's request and response bodies are dropped before the log row
is written. The row itself stays — which endpoint served, what it cost, how long
it took, which key spent it — because that is capacity accounting rather than
content, and a per-key budget that stopped counting would be a budget nobody
could enforce. Relayed outcomes are also not judged, so they never reach the
outcome matrix that ranks this fleet: the downstream router classified that
prompt against its own fleet and is already learning from the same answer, and
recording it twice would weight one exchange as two.

**Trust.** A relay is a router you run. The downstream adopts the upstream's
measurements sight unseen, so an upstream that claims quality 100 is believed.
That is the right trade between two halves of one fleet and the wrong one for a
stranger's endpoint — for those there is `/admin/providers`, which measures what
it is told.

**Cycles.** Every relayed request carries `X-LLM-Relay`, a chain of router ids.
A router that finds its own id in the chain answers 508, and so does one past
`relayMaxHops`. Each router's id is generated on first run and persisted; there
is nothing to configure and nothing two deployments can copy from each other.

**What it costs.** Two hops of network, and the downstream prices them. Every
latency figure it imports — first-token latency, prefill rate — describes the far
endpoint on the upstream's own LAN, and the round trip between the two routers is
added once, where the estimate is built. So a remote fleet is neither compared
against local workers as though it were in the next rack, nor charged for its
link twice. The downstream always names the model on the way out, so the upstream
is a slot broker rather than a second classifier.

## Measuring the router

The cold-start benchmark measures **endpoints**. `discrimen arena` measures the
**router**: whether it actually sends each prompt to the cheapest endpoint that
can answer it, which is the one claim the whole design rests on.

```bash
discrimen arena run -router http://localhost:8585 -dataset sub_10.jsonl -oracle -robustness
discrimen arena report -in arena-results.json
```

It follows [RouterArena](https://github.com/RouteWorks/RouterArena)'s shape,
measuring accuracy, cost, routing optimality, robustness and routing latency,
with two departures. **Cost is worker-seconds, not dollars**, because there is no
per-token price on a self-hosted fleet and what a request costs is the time it
occupies. And **robustness measures the decision, not the answer**: perturbing a
prompt and re-asking costs another full pass, while asking `/v1/route-preview`
whether it still routes the same way costs milliseconds and tests the classifier
directly.

`-oracle` runs every question on every endpoint: questions × endpoints
generations, expensive, and the only way optimality can be computed at all. The
report splits the two failure directions that matter: **overspend**, where the
answer was right but a cheaper endpoint was also right, and **undershoot**,
where the answer was wrong and some endpoint had it. Grading uses the production
`checkAnswer`, so the harness cannot score a policy that never ran.

## The grading sandbox

LiveBench's coding questions carry an **empty `ground_truth`**, and that is not
a gap in the data. The answer to "write `minimumArrayLength`" is a function, and
the only thing that can tell a correct one from a plausible one is running it
against the test cases. Grading them needs an interpreter.

The router must not be the thing holding it. It is a long-lived process with the
fleet's credentials, the request log and the database in one address space, and
`exec`-ing a language model's output next to any of that is not a risk to reason
about carefully — it is a risk to move into another container.

So `sandbox/` is a separate service, on loopback, holding nothing worth
stealing:

```
POST /grade            code + tests   → {"pass":…, "cases_run":…, "cases_passed":…, "first_failure":…}
POST /decode-private   base64 blob    → {"tests":[…]}
GET  /health                          → {"status":"ok"}
```

`/decode-private` is an endpoint rather than a function because LiveBench's
`private_test_cases` are base64 of zlib of a **pickle**, and unpickling is
arbitrary code execution by design — the format works by naming callables and
calling them. There is no way to validate the bytes first that is not itself a
pickle implementation, so the decode happens under the same containment a
submission gets.

Every submission runs in a fresh subprocess with `RLIMIT_CPU`, `RLIMIT_AS`,
`RLIMIT_FSIZE` and `RLIMIT_NPROC` set, a seccomp filter that removes `socket()`,
its own session, a wall-clock `SIGKILL` on the process group, a `/proc` sweep
for anything that escaped the group, and a scratch directory destroyed on every
exit path. The container is non-root, read-only, capability-free and pid-capped.
Details, and the reasoning for each layer, are in `sandbox/main.py` and
[`deploy/dropshell/discrimen/README.md`](deploy/dropshell/discrimen/README.md).

It is stdlib-only Python with no pip install layer at all: the one container in
the deployment whose job is to be the blast radius should not have a dependency
tree.

```bash
python3 -m unittest discover -s sandbox/tests -v
```

Most of those tests run a genuinely hostile submission — an infinite loop, a
fork bomb, a ten-gigabyte allocation, a `setsid()` escape — and assert that the
answer came back, that it came back as a clean `pass:false`, and that nothing
was left running.

## Deploying

The compose file is the supported path and it is enough for most people.

Upgrading an existing `llm-router` deployment: the commands are in
[docs/upgrading.md](docs/upgrading.md) under "Do this", and the rest of that page
is why. It upgrades in place on the same database, but 29 environment variables
stop being read, seven endpoints move to admin scope, and the whole fleet
re-benchmarks once.

If you use [dropshell](https://github.com/j842/dropshell), there is a template
at [`deploy/dropshell/discrimen`](deploy/dropshell/discrimen) that pulls the
published image, converges the container, and backs the data volume up through
restic. Its README covers the one migration trap: the Docker volume name is
derived from the container name, so renaming the container orphans the volume —
and the volume is where both the stored profiles and the graded answers live, so
losing it is the one event that genuinely costs the fleet a cold re-benchmark.
Back it up.

Images are published multi-arch (amd64 and arm64) to `ghcr.io/j842/discrimen`,
`ghcr.io/j842/discrimen-embeddings` and `ghcr.io/j842/discrimen-sandbox` on every
push to main. The sandbox builds per-arch despite compiling nothing: its seccomp
filter carries a syscall number per architecture, so an amd64-only manifest run
under emulation on an arm64 host would be checking the wrong table.

## Building

```bash
go build ./...
go test ./... -count=1
```

One direct dependency, `modernc.org/sqlite`, which is pure Go. That is why the
build is `CGO_ENABLED=0` and the runtime image is alpine. Everything else is the
standard library.

## Source map

| | |
|---|---|
| `main.go` | server, registry, proxy, selection, persistence, health and certification loops |
| `outcomes.go`, `outcomes_bank.go` | **the routing policy**: the outcome matrix, the neighbour lookup, the three bands, exploration, and the backfill that rebuilds it from stored profiles |
| `outcomes_validate.go` | `/admin/outcomes` — whether the matrix's predictions actually hold up |
| `identity.go` | what makes a question and a model identifiable, and therefore what the permanent cache is keyed by |
| `difficulty.go` | embedding-centroid classifier, and the superseded target-quality ranker it still hosts |
| `profile.go` | cold-start profiling: capability, speed, context and capacity probes, and their persistence |
| `progress.go` | what a cold-start profile is doing while it does it |
| `benchmark.go`, `benchmark_data*.go` | the tiered quality benchmark, its graders, and the question bank |
| `session.go` | conversation identity, tool-loop detection, affinity tracker |
| `escalate.go` | buffered dispatch, inline escalation, strip-and-retry |
| `rescue.go` | the length rescue — asking an endpoint that spent its whole budget thinking for its conclusion |
| `tokens.go` | prompt sizing: the measured chars-per-token per model, and reading a prompt size out of an endpoint's refusal |
| `relay.go` | routing through another discrimen: the relay key, `/relay/fleet`, the loop guard, and the fleet import |
| `inadequacy.go` | the response-adequacy classifier — is a 2xx body actually an answer, and is it empty (the worker's failure, escalated) or truncated (the caller's ceiling, not escalated). Split out when the online tier adapter that used to share the file was removed |
| `judge.go` | background LLM-as-judge |
| `preview.go` | `/v1/route-preview` |
| `arena.go` | the router-level regression gate |
| `benchgen*.go` | the benchmark refresh pipeline: fetch, calibrate, emit |

Everything above lives in `internal/router/`. The two sidecars are separate
services and separate images:

| | |
|---|---|
| `embeddings/main.py` | the embeddings worker auto-routing needs, self-registering |
| `sandbox/main.py` | the grading sidecar's HTTP surface and the request/response contract |
| `sandbox/supervisor.py` | one jailed run: spawn, drain, wall-clock kill, straggler sweep, scratch teardown |
| `sandbox/jail.py` | what the child does to itself: rlimits and the seccomp filter |
| `sandbox/runner.py` | the child. The only file that shares an address space with model-generated code |
| `sandbox/compare.py` | test-case decoding and the answer-comparison semantics |

## Licence

MIT. See [LICENSE](LICENSE).

Two third-party snapshots ride along, both recorded in [NOTICE](NOTICE):

- **LiveBench** questions and ground-truth answers, under the Apache License
  2.0. Parts of that snapshot derive from material LiveBench does not own:
  competition problems copyright the Mathematical Association of America and
  the United Kingdom Mathematics Trust, both non-commercial-use-only. NOTICE
  carries those terms forward and says which subsets are affected. **Read it
  before putting the benchmark data to commercial use.**
- **LiteLLM**'s model price and context-window table, under MIT, used to seed
  prices on a manually-entered provider. Nothing to watch out for there.
