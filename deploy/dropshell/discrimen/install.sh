#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME" "IMAGE_REGISTRY" "IMAGE_REPO" "IMAGE_TAG" \
    "SANDBOX_IMAGE_REPO" "SANDBOX_IMAGE_TAG"
_check_docker_installed || _die "Docker test failed"

# install-pre.sh has normally pulled these already; repeated because install.sh
# is also runnable on its own, and a pull of an image already present is free.
docker pull "$IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG" \
    || _die "Failed to pull $IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG"
docker pull "$IMAGE_REGISTRY/$SANDBOX_IMAGE_REPO:$SANDBOX_IMAGE_TAG" \
    || _die "Failed to pull $IMAGE_REGISTRY/$SANDBOX_IMAGE_REPO:$SANDBOX_IMAGE_TAG"

# No teardown here: start.sh converges, so a newly pulled image or a changed run
# command recreates the container while an install that changed neither leaves
# it serving — no dropped requests and no re-measuring of the fleet.
bash ./start.sh || _die "Failed to start container"

echo "Waiting for discrimen to start..."
for _ in $(seq 1 10); do
    curl -sf "http://localhost:${ROUTER_PORT:-8585}/health" >/dev/null 2>&1 && break
    sleep 1
done
curl -sf "http://localhost:${ROUTER_PORT:-8585}/health" >/dev/null 2>&1 \
    || _die "discrimen failed to start"

# The sidecar is polled separately and is NOT fatal.
#
# The asymmetry is deliberate. A router with no grading sandbox still routes,
# still serves /v1/*, and still benchmarks every question that is not a coding
# question — so failing the install would take a working fleet offline over a
# feature that degrades. But it degrades SILENTLY, as coding questions that
# every worker gets wrong, so the warning has to be loud and has to name the
# container to look in.
echo "Waiting for the grading sandbox..."
for _ in $(seq 1 15); do
    curl -sf "http://127.0.0.1:${SANDBOX_PORT:-8587}/health" >/dev/null 2>&1 && break
    sleep 1
done
if ! curl -sf "http://127.0.0.1:${SANDBOX_PORT:-8587}/health" >/dev/null 2>&1; then
    echo
    echo "WARNING: the grading sandbox on port ${SANDBOX_PORT:-8587} is not answering."
    echo "  The router is up and routing. Coding benchmark questions cannot be graded"
    echo "  without it, and they fail as wrong answers rather than as errors — so every"
    echo "  worker's coding tier will read as zero until this is fixed."
    echo "  Look at:  docker logs ${CONTAINER_NAME}_sandbox"
    echo
fi

echo "Installation of ${CONTAINER_NAME} complete"

# The router prints a generated credential to its log exactly once, on the first
# start against a database that has none, and never again. That line is easy to
# lose in a wall of container output, and until you have it the dashboard,
# /backends, /logs and /debug/* are all shut. So repeat it here, where the
# operator is already looking.
#
# These are secrets on a terminal by design: they are the operator's own
# credentials on their own machine, and showing them is the whole point. They do
# land in whatever transcript ds install is running under.
#
# Read from the HEAD of the log, not the tail — the banners are written during
# startup, so on a container that has been serving for a week they are the oldest
# lines, not the newest. head also bounds the read on a large log.
#
# The banner spans four lines (heading, rule, blank, value) and there are three of
# them, so reproducing it verbatim buries the three strings that matter in twelve
# lines of punctuation. Pull out the title and the value and print one line each.
# If the banner format ever changes the awk stops matching, so fall back to the
# raw lines rather than silently reporting nothing.
BOOTSTRAP_LOG="$(docker logs "${CONTAINER_NAME}" 2>&1 | head -n 200 || true)"
BOOTSTRAP_CREDS="$(printf '%s\n' "${BOOTSTRAP_LOG}" | awk '
    /BOOTSTRAP [A-Z]/ {
        title = $0
        sub(/.*BOOTSTRAP /, "", title)
        match(title, /^[A-Z][A-Z ]*[A-Z]/)
        title = substr(title, 1, RLENGTH)
        pending = 1
        next
    }
    pending && /^      [^ ]/ { printf "  %-16s %s\n", tolower(title) ":", $1; pending = 0 }
')"
if [ -z "${BOOTSTRAP_CREDS}" ]; then
    BOOTSTRAP_CREDS="$(printf '%s\n' "${BOOTSTRAP_LOG}" | grep -A3 'BOOTSTRAP ' || true)"
fi
ADMIN_URL="http://${SSH_HOST:-localhost}:${ROUTER_PORT:-8585}"

echo
echo "=================================================================="
echo "  discrimen admin"
echo "=================================================================="
echo "  dashboard:  ${ADMIN_URL}/"
echo
if [ -n "${BOOTSTRAP_CREDS}" ]; then
    echo "  Generated on first start. Copy them now, they are not shown again:"
    echo
    echo "${BOOTSTRAP_CREDS}"
elif [ -n "${ROUTER_ADMIN_PASSWORD:-}" ]; then
    echo "  admin password:  ${ROUTER_ADMIN_PASSWORD}"
    echo
    echo "  Set from ROUTER_ADMIN_PASSWORD in service.env, which wins on every"
    echo "  start. Change it there and reinstall to reset a lost password."
else
    echo "  Nothing was generated on this start, so the credentials were set on"
    echo "  an earlier one. They are hashed and cannot be read back."
    echo
    echo "  To reset the admin password: set ROUTER_ADMIN_PASSWORD in service.env"
    echo "  and run ds install again."
fi
echo
echo "  /backends, /logs and /debug/* need an admin credential, not a client"
echo "  token. For scripts and CLIs, issue a reusable admin key once:"
echo
echo "    curl -sS -c /tmp/cj -X POST ${ADMIN_URL}/admin/login \\"
echo "      -H 'Content-Type: application/json' -d '{\"password\":\"PASSWORD\"}'"
echo "    curl -sS -b /tmp/cj -X POST ${ADMIN_URL}/admin/keys \\"
echo "      -H 'Content-Type: application/json' -d '{\"name\":\"ops\",\"role\":\"admin\"}'"
echo
echo "  An admin key also passes client scope, so one key covers chat as well."
echo "=================================================================="
