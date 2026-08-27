# Upgrading from llm-router

## Do this

Replace `SERVICE` and `SERVER` with your own. The service name must not change:
the Docker volume is derived from it, and a new name means a cold re-profile of
the whole fleet.

```bash
# 1. Back up. Everything below is recoverable from this.
ds backupdata SERVICE SERVER
ds list-backups SERVICE

# 2. Register the template source.
ds add-template /home/j/code/github/discrimen
ds list-templates | grep discrimen        # must print

# 3. Set the admin password. Do this BEFORE the first start.
ds edit SERVICE SERVER
#      add:     ROUTER_ADMIN_PASSWORD="<pick one>"
#      change:  TEMPLATE="llm-router"  ->  TEMPLATE="discrimen"
#      leave alone: CONTAINER_NAME, DATA_VOLUME, LOG_DB_PATH

# 4. First install FAILS BY DESIGN. Dropshell rewrites service.env and stops.
ds install SERVICE SERVER

# 5. Check the rewrite kept your password, then install for real.
ds edit SERVICE SERVER                    # confirm ROUTER_ADMIN_PASSWORD is still set
ds install SERVICE SERVER

# 6. Verify.
ds run SERVICE SERVER status.sh           # Running
curl -sf http://SERVER:8585/health

# 7. Issue one admin key. /backends, /logs and /debug/* need it now.
curl -sS -c /tmp/cj -X POST http://SERVER:8585/admin/login \
  -H 'Content-Type: application/json' -d '{"password":"<the one from step 3>"}'
curl -sS -b /tmp/cj -X POST http://SERVER:8585/admin/keys \
  -H 'Content-Type: application/json' -d '{"name":"ops","role":"admin"}'
# copy "secret" from the response — it is shown once

# 8. Put that key where your tooling reads its token.
#    An admin key also passes client scope, so one key covers chat and /backends.
```

Then expect every worker to re-benchmark once, in the background, over the
following hours. The fleet keeps serving throughout at a provisional quality, and
`GET /backends` (or `ask -l`) shows each worker's phase and an ETA while it runs.

This is a one-off. Graded answers are now cached permanently, keyed by the
question and the model, so once the fleet is through this pass a future upgrade
re-runs the probes and re-asks nothing it has already answered.

To roll back: set `TEMPLATE` back to `llm-router` and run `ds install` twice
again, for the same reason as step 4.

---

The rest of this page is why, and what breaks if you skip a step.

discrimen is llm-router extracted and renamed. An existing deployment upgrades in
place: same SQLite file, same `/data/llm-router/logs.sqlite` path, same
`/backends/register` interface, same worker beacons. Nothing needs re-registering
and no schema migration has to be run by hand.

Six things change underneath, and every one of them is silent. The sections below
are the list.

## Configuration that is gone

Twenty-nine `ROUTER_*` variables the old binary read are now constants in the
Go source. They are not read, not validated, and not warned about: setting one in
your environment file has no effect at all, and the router starts normally.

The rule applied was whether a setting describes something only the operator can
know. Hardware, network, ports, credentials, retention and how long a caller is
willing to queue are facts about a site and are still variables. Learning rates,
classifier thresholds, latency estimates and tier bands are the router's own
decisions, and a site that has to set them has been handed the problem the router
exists to solve.

| Removed variable | Now |
|---|---|
| `ROUTER_ADAPT_BINS` | `adaptBins` = 10 (`main.go`) |
| `ROUTER_ADAPT_LR_DOWN` | `adaptLRDown` = 0.01 (`main.go`) |
| `ROUTER_ADAPT_LR_UP` | `adaptLRUp` = 0.04 (`main.go`) |
| `ROUTER_ADAPT_MAX_BIAS` | `adaptMaxBias` = 0.30 (`main.go`) |
| `ROUTER_CAPACITY_PROBE_MAX` | `capacityProbeMax` = 16 (`main.go`) |
| `ROUTER_CONTEXT_ANSWER_RESERVE` | `contextAnswerReserve` = 8192 (`difficulty.go`) |
| `ROUTER_DEADLINE_SAFETY_FACTOR` | `deadlineSafetyFactor` = 0.8 (`difficulty.go`) |
| `ROUTER_DECODE_SAMPLE_REF_TOKENS` | `decodeSampleRefTokens` = 512 (`main.go`) |
| `ROUTER_DIFFICULTY_CACHE_SIZE` | `difficultyCacheSize` = 4096 (`main.go`) |
| `ROUTER_DIFFICULTY_MAX_CHARS` | `difficultyMaxChars` = 4000 (`main.go`) |
| `ROUTER_DIFFICULTY_QUALITY_BANDS` | `DifficultyBands: ""` in `loadConfig`, hard-wired empty, so bands are always fleet-derived |
| `ROUTER_DIFFICULTY_TEMP` | `difficultyTemp` = 0.10 (`main.go`) |
| `ROUTER_DIFFICULTY_TIMEOUT_SECONDS` | `difficultyTimeoutFallback` = 2s (`main.go`), now only the pre-measurement fallback |
| `ROUTER_ESCALATE_INLINE` | `EscalateInline: true` in `loadConfig` |
| `ROUTER_ESCALATE_SLOT_WAIT_SECONDS` | `escalateSlotWait` = 15s (`escalate.go`) |
| `ROUTER_JUDGE_SAMPLE_RATE` | `judgeSampleRate` = 0.2 (`main.go`), gated on `ROUTER_AUTO_ROUTING` |
| `ROUTER_LATENCY_EST_THINK_TOKENS` | `latencyEstThinkTokens` = 1500 (`difficulty.go`) |
| `ROUTER_LATENCY_EST_TOKENS` | `latencyEstTokens` = 256 (`difficulty.go`) |
| `ROUTER_ONLINE_ADAPT` | Removed with the online tier adapter itself, along with `tier_adapter.json` and `POST /v1/route-feedback`. Nothing consulted its learned bias: it fed only the difficulty-tier branch, which the outcome matrix supersedes on every classified request |
| `ROUTER_PROFILE_WORKERS` | `ProfileWorkers: true` in `loadConfig` |
| `ROUTER_QUALITY_FLOOR_WAIT_SECONDS` | `qualityFloorWait` = 10s (`main.go`) |
| `ROUTER_REASONING_THRESHOLD` | `reasoningThreshold` = 0.35 (`main.go`) |
| `ROUTER_SESSION_LOCK_WAIT_SECONDS` | `sessionLockWait` = 5s (`session.go`) |
| `ROUTER_SESSION_MAX` | `sessionMax` = 4096 (`session.go`) |
| `ROUTER_SESSION_PREFILL_DISCOUNT` | `sessionPrefillDiscount` = 0.8 (`session.go`) |
| `ROUTER_SESSION_STICKY` | `sessionSticky` = true (`session.go`) |
| `ROUTER_SESSION_TTL_SECONDS` | `sessionTTL` = 30m (`session.go`) |
| `ROUTER_SPEED_SCORE_FULL_TPS` | `speedScoreFullTPS` = 150 (`main.go`) |
| `ROUTER_UNCAPPED_NOMINAL_SLOTS` | `uncappedNominalSlots` = 4 (`difficulty.go`) |

Every constant in that table is the value the reference llm-router deployment was
already running, checked one by one against its `template_info.env`,
`config/service.env` and `start.sh`. A fleet on the stock template sees no
behaviour change from any of them. Two look like exceptions when you read the
deployment scripts, and are not:

- **`ROUTER_JUDGE_SAMPLE_RATE`** looks in `start.sh` as though it defaulted to
  `0`, because the flag reads `${ROUTER_JUDGE_SAMPLE_RATE:-0}`. It did not.
  Dropshell sources `template_info.env` before it runs `start.sh`, that file set
  `ROUTER_JUDGE_SAMPLE_RATE=0.2`, and a `:-` fallback only fires on an unset
  variable. So the stock template ran the judge at 0.2 and the constant matches
  it. Worth checking only if you deployed without that `template_info.env`, in
  which case judging was genuinely off and 0.2 is new spend on your best model.
- **`ROUTER_QUALITY_FLOOR_WAIT_SECONDS`** was set twice and to different values:
  3 in `template_info.env` and the old binary's compiled default, 10 in the
  template's `config/service.env`. Service config wins, so the stock deployment
  ran 10 and the constant matches it. If you overrode it back to 3, or ran
  without the template's `service.env`, a request that wants an above-target
  endpoint now waits up to 10 seconds for a slot rather than 3 before serving
  from a lower one.

That disagreement is itself the argument for the change. The value was set in two
files, neither of which was obviously authoritative, and reading the deployment
scripts gave the wrong answer twice.

### The two levers people actually used

Twenty-seven of those twenty-nine were tuning knobs nobody touched. Two were not.

**`ROUTER_PROFILE_WORKERS=false`** was the documented way to stop the router
re-measuring a fleet: it made the router trust declared values and skip the
cold-start benchmark. There is no replacement. Profiling is unconditional, and
the only lever left over what gets measured is `ROUTER_AUTO_ROUTING=false`, which
turns off prompt classification, the outcome-matrix lookup it feeds, thinking
detection and judging but does **not** stop the benchmark. If you were using this
to avoid spending money on a metered endpoint, plan for the cold profile
described below instead.

**`ROUTER_DIFFICULTY_QUALITY_BANDS`** let you hand-set the band table mapping
difficulty to a target quality. `loadConfig` now hard-wires it to the empty
string, which is what the old binary read as "automatic, fleet-derived". The
reasoning is that a hand-set band table is a claim about which endpoint is good
enough for which prompt, and that claim is the measurement the router already
makes.

Since then the whole quality-target mechanism has been superseded: routing ranks
by predicted correctness on graded questions near the prompt, then by speed, and
computes no target at all. A band table is not merely ignored now — there is no
longer anything for it to override on the live path. See "How a request is
routed" in the README.

### Not removed, renamed

Three more variables changed rather than disappearing.

- `ROUTER_AUTO_DIFFICULTY` and `ROUTER_AUTO_THINKING` are replaced by the single
  `ROUTER_AUTO_ROUTING`, which also governs background judging (and the online
  adapter, now removed). It defaults to **true**. The old
  pair defaulted to false in the binary and true in the deployment template.
- `ROUTER_CLIENT_TOKEN` (singular) was the default for the `-token` flag on the
  `arena` and `bench` subcommands. It is now `ROUTER_ADMIN_KEY`, because both
  subcommands list the fleet through `GET /backends`, which is admin scope.
  `ROUTER_CLIENT_TOKENS` (plural), the actual client token list, is unchanged.

## Endpoints that changed scope

These moved from client scope to admin scope:

| Endpoint | Was | Is |
|---|---|---|
| `GET /backends` | client token | admin |
| `GET /workers` | client token | admin |
| `GET /backends/{id}` | client token | admin |
| `GET /backends/{id}/benchmark` | client token | admin |
| `GET /logs` | client token | admin |
| `POST /debug/backends/{id}/certify` | client token | admin |
| `POST /debug/backends/{id}/chat` | client token | admin |

`DELETE /backends/{id}` moved the other way: it used to accept a worker token
**or** any client token, and now accepts a worker credential or admin. A client
token no longer evicts workers. It is also a smaller action than it was — it
removes the registration and the stored profile and deliberately keeps the
worker's graded answers, which are evidence about the model rather than about the
box.

The reasoning is `/logs` in particular. Its rows carry every stored prompt and
response, so a client token there reads every other client's traffic. That was
acceptable when the only client token was yours and stops being acceptable the
moment a token goes to someone else. `/debug/backends/{id}/certify` re-runs a
cold-start profile, which takes the worker out of rotation while the probes run —
though it is now much cheaper than it was, because the questions the model has
already answered come from the permanent cache rather than being re-asked.

**The fix is one key, not two.** An admin-role API key satisfies client scope and
worker scope as well, because there is no authority a client has that an admin
does not:

```bash
curl -X POST localhost:8585/admin/keys \
  -H "Authorization: Bearer $ADMIN_KEY_OR_LOG_IN_FIRST" \
  -H 'Content-Type: application/json' \
  -d '{"name": "ops", "role": "admin"}'
```

The plaintext key is in that response and nowhere else: the table stores a
SHA-256 and the displayed prefix, so there is no endpoint that could re-read it.
That one key then works for `/v1/chat/completions`, `/backends`, `/logs`,
`/admin/*` and `/backends/register`. Scripts and monitoring that used to poll
`/backends` with a client token need this key substituted in; nothing else about
the call changes.

The browser dashboard is separate. It signs in with `POST /admin/login` and holds
an HttpOnly session cookie, and it deletes any bearer key the previous version of
the page left in `sessionStorage`.

## Set ROUTER_ADMIN_PASSWORD before the first start

There was no admin password in llm-router. On the first start against a database
that has none, discrimen generates one, prints it to the log once, stores the
hash, and never prints it again. The same happens for a client key and a worker
key if `ROUTER_CLIENT_TOKENS` and `ROUTER_WORKER_TOKEN` are empty.

Recover them with:

```bash
docker compose logs discrimen | grep -i -A4 -E 'bootstrap|GENERATED'
```

If you miss it, set `ROUTER_ADMIN_PASSWORD` and restart. When that variable is
set it is authoritative on every start, not only against a virgin database: if it
does not match the stored hash it replaces it, and the router logs that it did
without logging the password. So the recovery is one environment variable and a
restart, rather than stopping the router and running `sqlite3` against the volume
by hand.

Leaving the variable unset does not touch a stored password, so a password
rotated later through the dashboard survives a restart. You can blank the
variable again once the database holds the hash.

Setting it before the first start is still the better move. It means never
depending on catching a line in a log.

## You cannot delete your way back to an open router

llm-router treated an empty `ROUTER_WORKER_TOKEN` as "no worker authentication",
and the same for clients. discrimen keeps that, but it also mints a key when the
variable is empty, which means the gate is shut from the first start.

That created a trapdoor: deleting the single bootstrap worker key from the Keys
tab would have made `/backends/register` anonymous again, and deleting the last
client and admin key would have made `/v1/chat/completions` anonymous. Both are
now refused with a 409 naming the gate that would reopen. Issue the replacement
first, or set the corresponding environment variable, and the delete is allowed.

A gate that is already open is not defended, so a deployment that genuinely runs
without authentication can still tidy its keys table.

## The fleet re-benchmarks once

`benchmarkVersion` moved from **32** to **43**, and a cached profile is reused
only on an exact version match:

```go
if prof, ok := r.logs.LoadWorkerProfile(ctx, id, model); ok && prof.BenchVersion == benchmarkVersion {
```

So every worker in `worker_profiles` re-profiles on the first start after the
upgrade. There is no partial reuse of the *profile* and no way to opt out.

That is a much smaller event than it was when this page was first written, and
the difference is worth understanding before you budget for it. A re-profile
re-runs the capability, speed, capacity and context probes — seconds — and then
asks the question bank. **The questions themselves are cached separately, keyed
by the question and the model rather than by the profiling run, and that cache is
never invalidated by a version bump.** So a worker whose weights have not changed
answers nothing it has already answered.

Coming from `llm-router` you are crossing that change, so this particular upgrade
does pay a genuine cold pass: the old binary predates the cache and the old
`observations` table used an unreachable key format, so there is nothing to hit.
Budget for it once.

The benchmark is now **401 questions across 12 tiers**, up from 102 across 11:
130 hand-authored, 262 generated from LiveBench, and 9 long-context. 393 are
graded thinking-on against a 32768-token ceiling and 8 thinking-off against a
1024-token one, so budget roughly **0.8-1.2M output tokens** per endpoint, plus
about 400k prompt tokens for the long-context questions across the two passes. On
a local fleet that is GPU time. On a metered provider it is a bill. A thinking-on
pass gives up after 90 minutes of wall clock once 48 questions are away, which
bounds the worst case on a slow endpoint. See [benchmark.md](benchmark.md) for
the whole question set.

Profiling splits in two, so the fleet does not black out while this runs: the
quick half (capabilities, speed, context) makes each endpoint routable in
seconds, and the benchmark runs in the background and then refines the live
quality. An endpoint whose benchmark has not finished holds a provisional quality
of 30 and only draws easy traffic. `GET /backends` publishes a `profile_progress`
object while it runs — phase, questions done, in flight, elapsed, and an ETA — so
you can watch it rather than guess; `ask -l` renders it as a progress bar.

Afterwards the profile is cached and looked up by **(endpoint id, model)**. A
restart reloads it with no re-measurement. The table holds one row per endpoint
id, so changing a worker's model replaces the row.

**The graded answers are keyed differently, and this is the part that changes how
you operate the fleet.** They are filed under a hash of the question (its text,
expected answer, match mode and grader version) and a hash of the model (served
id, parameter count, quantisation, size, trained context, engine) — not the
endpoint id. Three consequences:

- **Deleting a worker does not delete its grading.** `DELETE /backends/{id}`
  removes the registration and the profile row and deliberately leaves the graded
  answers alone. Decommissioning a box costs you its speed and capacity
  measurements, not its hours of grading.
- **The same weights on another host inherit the whole thing.** Move a model to a
  different machine and it starts with its record already earned. Only the
  latency is re-measured, because a duration is a property of the box and a
  verdict is a property of the weights.
- **A grader fix is cheap.** Bumping one entry in `graderVersions` re-asks that
  mode's questions and leaves every other verdict addressable. Bumping
  `benchmarkVersion` for it — which is what the old rule said to do — used to
  discard about 5,600 gradings across the fleet to re-derive verdicts that had
  not changed.

One deployment trap worth repeating here: the dropshell template derives its
Docker volume name from the container name, and both `worker_profiles` and the
graded answers live inside that volume. Renaming the container orphans the volume
and re-benchmarks the whole fleet from cold — which is now the *only* thing that
does. Back it up.

## Error bodies changed shape

Error responses are now the OpenAI envelope. Old:

```json
{"message": "backend \"gpu-1\" not found"}
```

```json
{"error": "unauthorized"}
```

New, for both:

```json
{"error": {"message": "backend \"gpu-1\" not found", "type": "invalid_request_error", "param": null, "code": null}}
```

```json
{"error": {"message": "unauthorized: send a bearer token in the Authorization header", "type": "authentication_error", "param": null, "code": null}}
```

**Anything parsing `.error` as a string has to read `.error.message` instead.**
A `jq -r .error` that used to print `unauthorized` now prints a JSON object, and
a client that string-compares it will not match anything. A `jq -r .message` on
the old validation shape now prints `null`.

`type` is set from the HTTP status: `authentication_error` for 401,
`permission_error` for 403, `not_found_error` for 404, `rate_limit_error` for
429, `service_unavailable` for 503, `server_error` for any other 5xx, and
`invalid_request_error` otherwise. `param` names the offending request field when
the router knows it and is null otherwise. `param` and `code` are always present
as explicit nulls rather than omitted, because several SDKs index into them and a
missing key reads differently from a null.

This is why the change was made: the old bare `{"message": …}` is read as an
empty error by every client written against the standard, since the SDKs look
under `error.message` and find nothing there.

## Rolling back

The old binary opens a database the new one has migrated, without error. This was
tested. Two properties make it work:

- Every table the new binary adds (`api_keys`, `router_groups`,
  `router_settings`) is created with `CREATE TABLE IF NOT EXISTS`, and the old
  binary's own statements are a subset of the new ones, so nothing collides.
- `request_logs.key_id` is `TEXT NOT NULL DEFAULT ''`, and the old binary's
  `INSERT` names its columns explicitly without it, so the default fills in and
  old inserts keep working.

What rolling back costs you:

**The fleet re-profiles again, and this time the questions really are re-asked.**
The old binary carries `benchmarkVersion = 32`, which does not match the 43
written into every profile by the new one, so every worker cold-profiles a second
time against the old 102-question set. It predates the permanent answer cache, so
none of the grading the new binary accumulated is available to it — and its
questions are a different set anyway. Rolling forward again re-profiles a third
time; the forward pass is the cheap one, because by then the new binary's cache
holds the answers.

**And the graded answers themselves go.** The rollback DDL is not the only thing
that matters here: the `observations` table the new binary writes is keyed in a
format the old one does not understand, and the new binary drops and re-creates
it when it finds an unrecognised key format. Take a copy of the SQLite file
before you roll back if you want the grading to survive the round trip.

**Provider pricing stops applying.** Prices live in the `registration_json` blob
on `backend_registrations`, and the old binary's registration struct has no
price, provider or source fields. It does not delete them, and it only rewrites
that row when something registers under the same id, so the stored values usually
survive in SQLite. But the running old binary cannot see them, so a metered
provider is indistinguishable from a free local worker and takes ordinary traffic
at ordinary rates. If a worker does re-register under that id, the rewrite drops
the fields for good.

**Virtual keys stop being enforced, and that may open the router.** The old
binary has no idea `api_keys` exists. The rows survive untouched, but nothing
reads them, so authentication falls back entirely to `ROUTER_CLIENT_TOKENS` and
`ROUTER_WORKER_TOKEN`. If you relied on generated or issued keys and left those
variables empty, the old binary's `authorizedAsClient` returns true for an empty
token list, and the router accepts anything from anyone. Set both variables
before rolling back, or do not expose the port while you do.

**Named groups stop resolving.** `router_groups` is likewise unknown to the old
binary. A request sending `"model": "coding"` no longer resolves to a preference
list. The rows survive and start working again when you roll forward.

**The admin surface disappears.** `/admin/*` and the cookie-session dashboard do
not exist in the old binary, and the endpoints listed under
[Endpoints that changed scope](#endpoints-that-changed-scope) revert to accepting
a client token. So does `DELETE /backends/{id}`.
