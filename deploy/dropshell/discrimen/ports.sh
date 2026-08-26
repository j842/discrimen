#!/bin/bash
source "${AGENT_PATH}/common.sh"

# The router's port, and only the router's. Default matches start.sh: this must
# answer even when service.env omits the key.
#
# The grading sandbox's port is deliberately not listed. It is published to
# 127.0.0.1 only, so reporting it as a service port would advertise an address
# nothing off this machine can reach — and the one thing that must not happen to
# a code-execution endpoint is somebody being told it is a service to expose.
echo "${ROUTER_PORT:-8585}"
