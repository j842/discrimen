#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME"

if ! docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Unknown"
    exit 0
fi

STATE=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)
case "$STATE" in
    running)
        # The image carries a HEALTHCHECK against /health, so a router that is
        # up but not answering is an error rather than a healthy service.
        if docker inspect -f '{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "unhealthy"; then
            echo "Error"
        else
            echo "Running"
        fi
        ;;
    exited|stopped) echo "Stopped" ;;
    dead) echo "Error" ;;
    *) echo "Unknown" ;;
esac
