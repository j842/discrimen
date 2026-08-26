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
            exit 0
        fi
        # The sidecar counts too, because the failure it produces is invisible
        # otherwise: coding benchmark questions cannot be graded without it and
        # they fail as WRONG ANSWERS, so a fleet with a dead sandbox looks
        # healthy and simply rates every worker worse. Reported as Error rather
        # than Stopped so it shows up the same way a dead router does.
        #
        # A service that predates the sandbox has no such container at all, and
        # that is not an error — it is a template that has not been reinstalled
        # yet, and calling it broken would make every one of them red.
        SANDBOX="${CONTAINER_NAME}_sandbox"
        if docker ps -a --format "{{.Names}}" | grep -q "^${SANDBOX}$"; then
            SANDBOX_STATE=$(docker inspect -f '{{.State.Status}}' "$SANDBOX" 2>/dev/null)
            if [ "$SANDBOX_STATE" != "running" ]; then
                echo "Error"
                exit 0
            fi
            if docker inspect -f '{{.State.Health.Status}}' "$SANDBOX" 2>/dev/null | grep -q "unhealthy"; then
                echo "Error"
                exit 0
            fi
        fi
        echo "Running"
        ;;
    exited|stopped) echo "Stopped" ;;
    dead) echo "Error" ;;
    *) echo "Unknown" ;;
esac
