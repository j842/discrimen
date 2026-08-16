#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME" "IMAGE_REGISTRY" "IMAGE_REPO" "IMAGE_TAG"
_check_docker_installed || _die "Docker test failed"

# install-pre.sh has normally pulled this already; repeated because install.sh
# is also runnable on its own, and a pull of an image already present is free.
docker pull "$IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG" \
    || _die "Failed to pull $IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG"

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

echo "Installation of ${CONTAINER_NAME} complete"
