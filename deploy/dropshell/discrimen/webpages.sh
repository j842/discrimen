#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "SSH_HOST" "ROUTER_PORT"

echo "http://${SSH_HOST}:${ROUTER_PORT}|discrimen dashboard"
