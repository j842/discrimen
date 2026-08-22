#!/usr/bin/env bash
# profile-worker.sh — run discrimen's full cold-start profiling suite against
# one OpenAI-compatible endpoint, using a throwaway single-use router.
#
#   scripts/profile-worker.sh http://127.0.0.1:8093 [worker-api-key]
#
# What it does:
#   1. builds the discrimen image from this working tree (cached after the
#      first run; --no-build to require a prebuilt image),
#   2. starts a scratch router on a side port (host network, ephemeral /data,
#      random credentials, auto-routing off — profiling doesn't need
#      embeddings) that never touches any production discrimen,
#   3. registers the target; the router's cold-start profiler then runs the
#      whole suite by itself: capability probes (chat/json/tools/thinking),
#      context detection, speed probe, concurrency ramp, and the tiered
#      quality benchmark,
#   4. streams progress as each check lands,
#   5. writes a report (human-readable + raw JSON) to
#      ~/.discrimen/profiles/YYYYMMDD_<host>.txt and prints it,
#   6. tears everything down and VERIFIES it: container gone, its anonymous
#      /data volume gone, port free.
#
# Because the scratch router starts with an empty worker_profiles table, every
# run is a genuine cold-start profile — which also makes this a clean A/B
# harness: profile, change one thing on the worker, profile again, diff the
# two report files.
#
# Needs: docker, curl, jq. The target URL must be reachable from this host's
# network namespace (the router runs with --network host, so 127.0.0.1
# targets work when the worker runs on this machine).
#
# NOTE for thinking models: the quality benchmark waits through <think>
# blocks, so expect the run to take a while; --timeout bounds it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] <worker-url> [worker-api-key]

  <worker-url>      OpenAI-compatible endpoint base URL (e.g. http://127.0.0.1:8093)
  [worker-api-key]  sent as the worker's api_key if it requires one

Options:
  -o, --out FILE     report path (default ~/.discrimen/profiles/YYYYMMDD_<host>.txt)
  -t, --timeout SEC  give up after SEC seconds (default 3600)
  -p, --port PORT    scratch router port (default 18585)
      --id ID        worker id used in the scratch registry (default profile-target)
      --no-build     don't build; use the existing local image
  -h, --help         this text
EOF
    exit "${1:-0}"
}

OUT=""
TIMEOUT=3600
PORT=18585
WORKER_ID="profile-target"
BUILD=true
TARGET_URL=""
TARGET_KEY=""

while [ $# -gt 0 ]; do
    case "$1" in
        -o|--out)     OUT="$2"; shift 2 ;;
        -t|--timeout) TIMEOUT="$2"; shift 2 ;;
        -p|--port)    PORT="$2"; shift 2 ;;
        --id)         WORKER_ID="$2"; shift 2 ;;
        --no-build)   BUILD=false; shift ;;
        -h|--help)    usage 0 ;;
        -*)           echo "unknown option: $1" >&2; usage 2 ;;
        *)  if   [ -z "${TARGET_URL}" ]; then TARGET_URL="$1"
            elif [ -z "${TARGET_KEY}" ]; then TARGET_KEY="$1"
            else echo "unexpected argument: $1" >&2; usage 2; fi
            shift ;;
    esac
done
[ -n "${TARGET_URL}" ] || usage 2
case "${TARGET_URL}" in http://*|https://*) ;; *)
    echo "worker-url must start with http:// or https://" >&2; exit 2 ;;
esac

for dep in docker curl jq; do
    command -v "$dep" >/dev/null || { echo "missing dependency: $dep" >&2; exit 1; }
done

# Root-owned docker daemons (no docker group): fall back to passwordless sudo
# transparently, so the script works both on dropshell hosts (root) and on a
# dev laptop. The function shadows the binary for every later call.
DOCKER_CMD="docker"
if ! docker info >/dev/null 2>&1; then
    if sudo -n docker info >/dev/null 2>&1; then
        DOCKER_CMD="sudo -n docker"
    else
        echo "cannot talk to the docker daemon (tried plain and passwordless sudo)" >&2
        exit 1
    fi
fi
docker() { ${DOCKER_CMD} "$@"; }

IMAGE="discrimen-profile:local"
CONTAINER="discrimen-profile-$$"
ROUTER="http://127.0.0.1:${PORT}"
TMPDIR_="$(mktemp -d)"
COOKIES="${TMPDIR_}/cookies"
KEEPALIVE_PID=""
CONTAINER_STARTED=false
START_TS=$(date +%s)

rand_hex() { head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
WORKER_TOKEN="$(rand_hex)"
CLIENT_TOKEN="$(rand_hex)"
ADMIN_PASSWORD="$(rand_hex)"

IS_TTY=false; [ -t 1 ] && IS_TTY=true
SPINNER='|/-\'
SPIN_I=0
spin_clear() { $IS_TTY && printf '\r\033[K' || true; }
spin_draw() {  # $1 = status text
    $IS_TTY || return 0
    SPIN_I=$(( (SPIN_I + 1) % 4 ))
    printf '\r\033[K %s %s' "${SPINNER:${SPIN_I}:1}" "$1"
}
event() {  # $1 = glyph, $2 = text
    spin_clear
    printf '%s %s %s\n' "$(date +%H:%M:%S)" "$1" "$2"
}

port_in_use() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&- 3<&-; return 0; } || return 1; }

# ── Teardown: always runs, verifies, and says so ────────────────────────────
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    spin_clear
    [ -n "${KEEPALIVE_PID}" ] && kill "${KEEPALIVE_PID}" 2>/dev/null || true
    if $CONTAINER_STARTED; then
        echo "Tearing down scratch router..."
        # Capture the anonymous /data volume BEFORE removal so we can verify it
        # went with the container (-v removes anonymous volumes).
        local vols
        vols="$(docker inspect -f '{{range .Mounts}}{{.Name}} {{end}}' "${CONTAINER}" 2>/dev/null || true)"
        docker rm -fv "${CONTAINER}" >/dev/null 2>&1 || true
        # Double-check 1: container really gone (any state, not just running).
        local i
        for i in 1 2 3 4 5; do
            [ -z "$(docker ps -aq --filter "name=^${CONTAINER}$")" ] && break
            sleep 1
            docker rm -fv "${CONTAINER}" >/dev/null 2>&1 || true
        done
        if [ -n "$(docker ps -aq --filter "name=^${CONTAINER}$")" ]; then
            echo "  WARNING: container ${CONTAINER} still present — remove it manually." >&2
        else
            echo "  container removed: ${CONTAINER}"
        fi
        # Double-check 2: its volumes are gone (belt over the -v braces).
        local v
        for v in ${vols}; do
            if docker volume inspect "$v" >/dev/null 2>&1; then
                docker volume rm "$v" >/dev/null 2>&1 || true
            fi
            if docker volume inspect "$v" >/dev/null 2>&1; then
                echo "  WARNING: volume $v still present — remove it manually." >&2
            else
                echo "  volume removed: $v"
            fi
        done
        # Double-check 3: the port is free again.
        if port_in_use "${PORT}"; then
            echo "  WARNING: port ${PORT} still in use." >&2
        else
            echo "  port ${PORT} free"
        fi
    fi
    rm -rf "${TMPDIR_}"
    exit "$rc"
}
trap cleanup EXIT INT TERM

# ── 1. Image ────────────────────────────────────────────────────────────────
if $BUILD; then
    echo "Building ${IMAGE} from ${REPO_ROOT} (cached after first run)..."
    docker build -q -t "${IMAGE}" "${REPO_ROOT}" >/dev/null
else
    docker image inspect "${IMAGE}" >/dev/null 2>&1 \
        || { echo "--no-build but ${IMAGE} is not present" >&2; exit 1; }
fi

# ── 2. Scratch router ───────────────────────────────────────────────────────
if port_in_use "${PORT}"; then
    echo "port ${PORT} is already in use — pick another with --port" >&2
    exit 1
fi
echo "Starting scratch router on :${PORT} (host network, ephemeral /data, auto-routing off)..."
docker run -d --name "${CONTAINER}" \
    --network host \
    -e ROUTER_PORT="${PORT}" \
    -e ROUTER_WORKER_TOKEN="${WORKER_TOKEN}" \
    -e ROUTER_CLIENT_TOKENS="${CLIENT_TOKEN}" \
    -e ROUTER_ADMIN_PASSWORD="${ADMIN_PASSWORD}" \
    -e ROUTER_AUTO_ROUTING=false \
    "${IMAGE}" >/dev/null
CONTAINER_STARTED=true

for _ in $(seq 1 30); do
    curl -sf "${ROUTER}/health" >/dev/null 2>&1 && break
    if [ -z "$(docker ps -q --filter "name=^${CONTAINER}$")" ]; then
        echo "scratch router exited during startup:" >&2
        docker logs "${CONTAINER}" 2>&1 | tail -20 >&2
        exit 1
    fi
    sleep 1
done
curl -sf "${ROUTER}/health" >/dev/null 2>&1 || { echo "router did not become healthy" >&2; exit 1; }

# Admin session for /backends (admin scope): password → cookie.
curl -sf -c "${COOKIES}" -X POST "${ROUTER}/admin/login" \
    -H 'Content-Type: application/json' \
    -d "{\"password\":\"${ADMIN_PASSWORD}\"}" >/dev/null \
    || { echo "admin login against the scratch router failed" >&2; exit 1; }

# ── 3. Register the target; keepalive so the TTL can't prune it mid-run ────
REG_PAYLOAD="$(jq -cn --arg id "${WORKER_ID}" --arg url "${TARGET_URL}" --arg key "${TARGET_KEY}" \
    '{id:$id, url:$url} + (if $key != "" then {api_key:$key} else {} end)')"
register() {
    curl -sf -X POST "${ROUTER}/backends/register" \
        -H "Authorization: Bearer ${WORKER_TOKEN}" \
        -H 'Content-Type: application/json' \
        -d "${REG_PAYLOAD}" >/dev/null
}
register || { echo "registration failed — is ${TARGET_URL} reachable from this host?" >&2; exit 1; }
( while :; do sleep 45; register 2>/dev/null || true; done ) &
KEEPALIVE_PID=$!
event "•" "registered ${TARGET_URL} as '${WORKER_ID}' — profiling starts now"

# ── 4. Watch the checks land ────────────────────────────────────────────────
# Completion signal: the registry's `profiling` flag is true while the
# background quality+capacity run is in flight and cleared when it finishes —
# on BOTH the persisted and the not-persisted (inconclusive-probe) paths. The
# checks.profile message is NOT a completion signal on a cold profile: it
# stays "provisional — …" after a successful fresh run ("cached: …" only ever
# appears on a warm restart, which a scratch router never has).
declare -A SEEN
BACKEND=""
DONE=false
SAW_PROFILING=false
while :; do
    NOW=$(date +%s); ELAPSED=$((NOW - START_TS))
    if [ "${ELAPSED}" -ge "${TIMEOUT}" ]; then
        spin_clear
        echo "timed out after ${TIMEOUT}s — partial state below; raise --timeout for slow/thinking workers" >&2
        break
    fi
    BACKEND="$(curl -sf -b "${COOKIES}" "${ROUTER}/backends" 2>/dev/null \
        | jq -c --arg id "${WORKER_ID}" '.backends[] | select(.id == $id)' 2>/dev/null || true)"
    if [ -n "${BACKEND}" ]; then
        # New/changed checks become timestamped event lines.
        while IFS=$'\t' read -r name ok msg; do
            [ -z "${name}" ] && continue
            key="${name}=${ok}:${msg}"
            if [ "${SEEN[${name}]:-}" != "${key}" ]; then
                SEEN[${name}]="${key}"
                glyph="✓"; [ "${ok}" = "true" ] || glyph="✗"
                event "${glyph}" "${name}: ${msg}"
            fi
        done < <(jq -r '((.certification.checks // {}) + (.checks // {})) | to_entries[] | [.key, (.value.ok|tostring), (.value.message // "")] | @tsv' <<<"${BACKEND}")

        STATUS="$(jq -r '.status // "?"' <<<"${BACKEND}")"
        PROFILING="$(jq -r '.profiling // false' <<<"${BACKEND}")"
        PROFMSG="$(jq -r '((.certification.checks // {}) + (.checks // {})).profile.message // ""' <<<"${BACKEND}")"
        SUMMARY="$(jq -r '"status=\(.status // "?") q=\(.quality // "?") \(.baseline_tps // 0 | floor) tok/s ctx=\(.context_k // "?")k"' <<<"${BACKEND}")"
        spin_draw "[${ELAPSED}s] ${SUMMARY} — $([ "${PROFILING}" = "true" ] && echo "quality+capacity benchmark running" || echo "waiting")"

        if [ "${PROFILING}" = "true" ]; then
            $SAW_PROFILING || event "•" "background quality+capacity profile started"
            SAW_PROFILING=true
        elif $SAW_PROFILING; then
            DONE=true
            spin_clear
            event "★" "profile complete: ${SUMMARY#status=* }"
            break
        elif curl -sf -o /dev/null -b "${COOKIES}" "${ROUTER}/backends/${WORKER_ID}/benchmark" 2>/dev/null; then
            # Fallback: a persisted run exists but we never caught the in-flight
            # flag (profile finished inside one poll interval).
            DONE=true
            spin_clear
            event "★" "profile complete (persisted): ${SUMMARY}"
            break
        fi
        if [[ "${PROFMSG}" == abandoned* ]]; then
            event "✗" "profiler gave up: ${PROFMSG}"
            break
        fi
        if [ "${STATUS}" = "failed" ] || [ "${STATUS}" = "unhealthy" ]; then
            ERR="$(jq -r '.last_error // "?"' <<<"${BACKEND}")"
            if [ "${LAST_ERR:-}" != "${STATUS}:${ERR}" ]; then
                LAST_ERR="${STATUS}:${ERR}"
                event "✗" "worker status=${STATUS}: ${ERR}"
            fi
        fi
    else
        spin_draw "[${ELAPSED}s] waiting for registry entry..."
    fi
    sleep 3
done

# ── 5. Report ───────────────────────────────────────────────────────────────
BENCH="$(curl -sf -b "${COOKIES}" "${ROUTER}/backends/${WORKER_ID}/benchmark" 2>/dev/null || true)"

if [ -z "${OUT}" ]; then
    SLUG="$(sed -e 's#^[a-z]*://##' -e 's#[^A-Za-z0-9._-]#-#g' -e 's#-*$##' <<<"${TARGET_URL}")"
    OUT="${HOME}/.discrimen/profiles/$(date +%Y%m%d)_${SLUG}.txt"
fi
mkdir -p "$(dirname "${OUT}")"

{
    echo "discrimen worker profile"
    echo "========================"
    echo "target:     ${TARGET_URL}"
    echo "date:       $(date -Iseconds)"
    echo "wall clock: $(( $(date +%s) - START_TS ))s"
    echo "complete:   ${DONE}"
    echo
    if [ -n "${BACKEND}" ]; then
        jq -r '
            "model:            \(.model // "?")",
            "quality:          \(.quality // "?")/10\(if .quality_detail then "  (" + .quality_detail + ")" else "" end)",
            "thinking:         \(.thinking // false)\(if .thinking_dialect then "  (dialect: " + .thinking_dialect + ")" else "" end)",
            "context:          \(.context_k // "?")k",
            "baseline decode:  \(if .baseline_tps then (.baseline_tps * 10 | round / 10 | tostring) else "?" end) tok/s",
            "observed decode:  \(if .observed_tps then (.observed_tps * 10 | round / 10 | tostring) else "n/a" end) tok/s",
            "observed prefill: \(if .observed_prefill_tps then (.observed_prefill_tps * 10 | round / 10 | tostring) else "n/a" end) tok/s",
            "observed TTFT:    \(if .observed_ttft_ms then (.observed_ttft_ms | round | tostring) else "n/a" end) ms",
            "max concurrency:  \(.max_concurrency // "?")",
            "features:         \((.features // []) | join(", "))",
            "",
            "checks:",
            (((.certification.checks // {}) + (.checks // {})) | to_entries[] | "  [\(if .value.ok then "ok" else "FAIL" end)] \(.key): \(.value.message // "")")
        ' <<<"${BACKEND}"
        FAILED="$(jq -r '(.failed_benchmarks // []) | length' <<<"${BACKEND}")"
        echo
    fi
    if [ -n "${BENCH}" ]; then
        jq -r '
            "benchmark (\(.profiled_in // "?")\(if .profile_cost_measured then ", " + ((.profile_prompt_tokens // 0)|tostring) + " prompt + " + ((.profile_output_tokens // 0)|tostring) + " output tokens" else "" end)):",
            "  passed \([.results[] | select(.pass)] | length)/\(.results | length) — by tier: \(
                [.results | group_by(.tier)[] | "T\(.[0].tier): \([.[] | select(.pass)] | length)/\(length)"] | join("  "))",
            "",
            "  failures:",
            (if ([.results[] | select(.pass | not)] | length) == 0 then "    (none)" else
                (.results[] | select(.pass | not) |
                 "    [T\(.tier)\(if .truncated then " truncated" elif .errored then " errored" elif .slow then " slow" else "" end)] \(.prompt | gsub("\n"; " ") | .[0:90])\n        expected: \(.expect | gsub("\n"; " ") | .[0:80])\n        got:      \(.got | gsub("\n"; " ") | .[0:80])")
             end)
        ' <<<"${BENCH}" 2>/dev/null || echo "  (benchmark details unavailable)"
    else
        echo "benchmark: no stored run (incomplete profile?)"
    fi
    echo
    echo "raw registry entry (JSON):"
    jq . <<<"${BACKEND:-null}"
    if [ -n "${BENCH}" ]; then
        echo
        echo "raw benchmark run (JSON):"
        jq . <<<"${BENCH}"
    fi
} | tee "${OUT}"

echo
echo "Saved: ${OUT}"
$DONE || exit 1
# cleanup runs via the EXIT trap and verifies the teardown.
