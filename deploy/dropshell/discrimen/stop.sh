#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME"

docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
