#!/bin/bash
source "${AGENT_PATH}/common.sh"
source "${AGENT_PATH}/datacommands_v2.sh"
_check_required_env_vars "CONTAINER_NAME" "DATA_VOLUME"

restore_items "volume:data:${DATA_VOLUME}"
