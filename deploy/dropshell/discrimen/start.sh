#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME" "DATA_VOLUME" "IMAGE_REGISTRY" "IMAGE_REPO" "IMAGE_TAG"

docker volume create "$DATA_VOLUME" >/dev/null

# --network host with no -p mapping: the router dials worker URLs outbound
# across the LAN, and the port it listens on is whatever ROUTER_PORT says. A
# published port would pin the mapping to one number and break the moment
# ROUTER_PORT moved.
#
# Fourteen -e flags, and that is the whole surface the binary reads; everything
# the old template forwarded for routing internals is now a constant. Each one
# carries its default here, so a service.env written before a variable existed
# still starts. Adding a variable the binary reads without adding it here
# silently disables whatever it controls.
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
    -e $(_q "DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-4096}") \
    -e $(_q "ROUTER_AUTO_ROUTING=${ROUTER_AUTO_ROUTING:-true}") \
    $(_q "$IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG")"

# Hashes the run command into a ds.spec label and inspects the image reference
# separately, so a newly pulled image or an edited service.env recreates the
# container while an install that changed neither leaves it running.
_converge_container "$DOCKER_RUN_CMD" "$CONTAINER_NAME" "$IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG" \
    || _die "Failed to start container"
