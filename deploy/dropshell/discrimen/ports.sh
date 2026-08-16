#!/bin/bash
source "${AGENT_PATH}/common.sh"

# Default matches start.sh: this must answer even when service.env omits the key.
echo "${ROUTER_PORT:-8585}"
