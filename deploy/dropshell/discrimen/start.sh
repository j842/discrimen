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
    --name \"$CONTAINER_NAME\" \
    --restart unless-stopped \
    --network host \
    -v \"${DATA_VOLUME}:/data\" \
    -e \"ROUTER_PORT=${ROUTER_PORT:-8585}\" \
    -e \"LOG_DB_PATH=${LOG_DB_PATH:-/data/llm-router/logs.sqlite}\" \
    -e \"ROUTER_ADMIN_PASSWORD=${ROUTER_ADMIN_PASSWORD:-}\" \
    -e \"ROUTER_WORKER_TOKEN=${ROUTER_WORKER_TOKEN:-}\" \
    -e \"ROUTER_CLIENT_TOKENS=${ROUTER_CLIENT_TOKENS:-}\" \
    -e \"ROUTER_PERSIST_SECRET=${ROUTER_PERSIST_SECRET:-}\" \
    -e \"LOG_RETENTION_DAYS=${LOG_RETENTION_DAYS:-30}\" \
    -e \"LOG_MAX_BODY_BYTES=${LOG_MAX_BODY_BYTES:-16384}\" \
    -e \"HEALTH_INTERVAL_SECONDS=${HEALTH_INTERVAL_SECONDS:-15}\" \
    -e \"BACKEND_TIMEOUT_SECONDS=${BACKEND_TIMEOUT_SECONDS:-600}\" \
    -e \"BACKEND_IDLE_TIMEOUT_SECONDS=${BACKEND_IDLE_TIMEOUT_SECONDS:-120}\" \
    -e \"ROUTER_SLOT_MAX_WAIT_SECONDS=${ROUTER_SLOT_MAX_WAIT_SECONDS:-600}\" \
    -e \"DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-4096}\" \
    -e \"ROUTER_AUTO_ROUTING=${ROUTER_AUTO_ROUTING:-true}\" \
    \"$IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG\""

# Hashes the run command into a ds.spec label and inspects the image reference
# separately, so a newly pulled image or an edited service.env recreates the
# container while an install that changed neither leaves it running.
_converge_container "$DOCKER_RUN_CMD" "$CONTAINER_NAME" "$IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG" \
    || _die "Failed to start container"
