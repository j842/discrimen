#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "IMAGE_REGISTRY" "IMAGE_REPO" "IMAGE_TAG"

# Runs before the old service is touched, so the download happens while the old
# container is still serving. Fatal on failure: an install that cannot get the
# image must abort here, before it tears anything down.
docker pull "$IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG" \
    || _die "Failed to pull $IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG"

echo "Pre-install complete"
