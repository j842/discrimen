#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "IMAGE_REGISTRY" "IMAGE_REPO" "IMAGE_TAG" "SANDBOX_IMAGE_REPO" "SANDBOX_IMAGE_TAG"

# Runs before the old service is touched, so the download happens while the old
# container is still serving. Fatal on failure: an install that cannot get the
# image must abort here, before it tears anything down.
docker pull "$IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG" \
    || _die "Failed to pull $IMAGE_REGISTRY/$IMAGE_REPO:$IMAGE_TAG"

# The sidecar, pulled with the same rule and for the same reason. Fatal too:
# start.sh brings both up together, and an install that got one image and not
# the other would leave the router running with a grader that cannot start —
# which shows up much later, as a coding benchmark that scores every worker
# zero rather than as an install that failed.
docker pull "$IMAGE_REGISTRY/$SANDBOX_IMAGE_REPO:$SANDBOX_IMAGE_TAG" \
    || _die "Failed to pull $IMAGE_REGISTRY/$SANDBOX_IMAGE_REPO:$SANDBOX_IMAGE_TAG"

echo "Pre-install complete"
