#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME" "DATA_VOLUME" "IMAGE_REGISTRY" "IMAGE_REPO" "IMAGE_TAG" \
    "SANDBOX_IMAGE_REPO" "SANDBOX_IMAGE_TAG"

docker volume create "$DATA_VOLUME" >/dev/null

SANDBOX_CONTAINER="${CONTAINER_NAME}_sandbox"
SANDBOX_PORT="${SANDBOX_PORT:-8587}"

# ── The code-execution grading sidecar ──────────────────────────────────────
#
# Started FIRST, because the router's coding benchmark calls it and a router
# that comes up to find nothing on the port scores every worker zero on those
# questions rather than failing visibly.
#
# NETWORKING IS ASYMMETRIC AND THAT IS THE POINT. The router is on --network
# host; this is not. It sits on the default bridge with its port published to
# 127.0.0.1 only, so the router reaches it at http://127.0.0.1:$SANDBOX_PORT
# through the loopback interface they share, and nothing off this machine can
# reach a code-execution endpoint at all. Putting it on the host network would
# have been one fewer flag and would have handed every submission the LAN: the
# sidecar's own seccomp filter removes socket() from the process running the
# code, but a defence that is the ONLY defence is a defence one bug from
# nothing.
#
# The hardening flags, and what each one is actually for:
#
#   --read-only        the submission's own view of / is immutable, so the
#                      grader cannot be rewritten by the thing it is grading
#   --tmpfs /scratch   the one writable path, noexec so nothing written there
#                      can be run, and size-capped so a disk-filling submission
#                      fills 256 MB of RAM and stops
#   --tmpfs /tmp       python's default temp dir; without it a read-only rootfs
#                      turns an ordinary tempfile call into a crash
#   --cap-drop=ALL     it needs none. It binds a port above 1024 and forks.
#   no-new-privileges  also a precondition for the seccomp filter the sidecar
#                      installs on itself, which the kernel refuses from an
#                      unprivileged process without it
#   --pids-limit       the ceiling under the sidecar's own RLIMIT_NPROC. That
#                      limit bounds what one submission can fork; this bounds
#                      what the whole container can, so a fork bomb that beat
#                      every layer above still cannot reach the host's pid table
#   --memory           likewise under the per-run address-space limit
#   --init             a real pid 1 to reap orphans. The sidecar reaps them
#                      itself as well, so this is belt and braces — but it is
#                      the layer that keeps working if that code is ever wrong.
#
# --memory is sized for concurrency x per-run memory plus the service itself.
# Set SANDBOX_MEMORY_MB above a quarter of it and the container's own limit,
# not the per-run one, becomes what a grading run hits.
SANDBOX_RUN_CMD="docker run -d \
    --name $(_q "$SANDBOX_CONTAINER") \
    --restart unless-stopped \
    --init \
    -p $(_q "127.0.0.1:${SANDBOX_PORT}:${SANDBOX_PORT}") \
    --read-only \
    --tmpfs $(_q "/scratch:rw,noexec,nosuid,nodev,size=${SANDBOX_SCRATCH_MB:-256}m,mode=1777") \
    --tmpfs $(_q "/tmp:rw,noexec,nosuid,nodev,size=32m,mode=1777") \
    --cap-drop=ALL \
    --security-opt no-new-privileges \
    --pids-limit $(_q "${SANDBOX_PIDS_LIMIT:-512}") \
    --memory $(_q "${SANDBOX_CONTAINER_MEMORY:-3g}") \
    -e $(_q "SANDBOX_PORT=${SANDBOX_PORT}") \
    -e $(_q "SANDBOX_TOKEN=${SANDBOX_TOKEN:-}") \
    -e $(_q "SANDBOX_MAX_CONCURRENCY=${SANDBOX_MAX_CONCURRENCY:-4}") \
    -e $(_q "SANDBOX_DEFAULT_TIMEOUT_MS=${SANDBOX_DEFAULT_TIMEOUT_MS:-10000}") \
    -e $(_q "SANDBOX_DEFAULT_MEMORY_MB=${SANDBOX_DEFAULT_MEMORY_MB:-512}") \
    $(_q "$IMAGE_REGISTRY/$SANDBOX_IMAGE_REPO:$SANDBOX_IMAGE_TAG")"

if ! _converge_container "$SANDBOX_RUN_CMD" "$SANDBOX_CONTAINER" \
        "$IMAGE_REGISTRY/$SANDBOX_IMAGE_REPO:$SANDBOX_IMAGE_TAG"; then
    _is_container_exists "$SANDBOX_CONTAINER" && _get_container_logs "$SANDBOX_CONTAINER"
    _die "Failed to start the grading sandbox"
fi

# ── The router ──────────────────────────────────────────────────────────────
#
# --network host with no -p mapping: the router dials worker URLs outbound
# across the LAN, and the port it listens on is whatever ROUTER_PORT says. A
# published port would pin the mapping to one number and break the moment
# ROUTER_PORT moved.
#
# Sixteen -e flags, and that is the whole surface the binary reads; everything
# the old template forwarded for routing internals is now a constant. Each one
# carries its default here, so a service.env written before a variable existed
# still starts. Adding a variable the binary reads without adding it here
# silently disables whatever it controls.
#
# SANDBOX_URL is 127.0.0.1 and not the container name for the reason above: the
# router is in the host network namespace, so container DNS does not resolve
# there and the published loopback port is what it can actually reach.
DOCKER_RUN_CMD="docker run -d \
    --name $(_q "$CONTAINER_NAME") \
    --restart unless-stopped \
    --network host \
    -v $(_q "${DATA_VOLUME}:/data") \
    -e $(_q "ROUTER_PORT=${ROUTER_PORT:-8585}") \
    -e $(_q "LOG_DB_PATH=${LOG_DB_PATH:-/data/llm-router/logs.sqlite}") \
    -e $(_q "ROUTER_ADMIN_PASSWORD=${ROUTER_ADMIN_PASSWORD:-}") \
    -e $(_q "ROUTER_WORKER_TOKEN=${ROUTER_WORKER_TOKEN:-}") \
    -e $(_q "ROUTER_CLIENT_TOKENS=${ROUTER_CLIENT_TOKENS:-}") \
    -e $(_q "ROUTER_PERSIST_SECRET=${ROUTER_PERSIST_SECRET:-}") \
    -e $(_q "LOG_RETENTION_DAYS=${LOG_RETENTION_DAYS:-30}") \
    -e $(_q "LOG_MAX_BODY_BYTES=${LOG_MAX_BODY_BYTES:-16384}") \
    -e $(_q "HEALTH_INTERVAL_SECONDS=${HEALTH_INTERVAL_SECONDS:-15}") \
    -e $(_q "BACKEND_TIMEOUT_SECONDS=${BACKEND_TIMEOUT_SECONDS:-600}") \
    -e $(_q "BACKEND_IDLE_TIMEOUT_SECONDS=${BACKEND_IDLE_TIMEOUT_SECONDS:-120}") \
    -e $(_q "ROUTER_SLOT_MAX_WAIT_SECONDS=${ROUTER_SLOT_MAX_WAIT_SECONDS:-600}") \
    -e $(_q "DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-16384}") \
    -e $(_q "ROUTER_AUTO_ROUTING=${ROUTER_AUTO_ROUTING:-true}") \
    -e $(_q "SANDBOX_URL=http://127.0.0.1:${SANDBOX_PORT}") \
    -e $(_q "SANDBOX_TOKEN=${SANDBOX_TOKEN:-}") \
    $(_q "$IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG")"

# Hashes the run command into a ds.spec label and inspects the image reference
# separately, so a newly pulled image or an edited service.env recreates the
# container while an install that changed neither leaves it running.
_converge_container "$DOCKER_RUN_CMD" "$CONTAINER_NAME" "$IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG" \
    || _die "Failed to start container"
