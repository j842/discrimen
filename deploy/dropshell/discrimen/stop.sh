#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME"

docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true

# The sidecar, best effort. A service installed before the sandbox existed has
# no such container, and stop.sh must still succeed for it — which is also why
# nothing here is fatal.
docker stop "${CONTAINER_NAME}_sandbox" >/dev/null 2>&1 || true
