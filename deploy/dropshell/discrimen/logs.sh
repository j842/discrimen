#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME"

docker logs -f --tail=200 "$CONTAINER_NAME"
