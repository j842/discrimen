#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME"

bash ./stop.sh || _die "Failed to stop container"
_remove_container "$CONTAINER_NAME" || _die "Failed to remove container"

# Not fatal, and not for the same reason as the router's removal: the sandbox
# holds no state and a service installed before it existed has no such
# container, so "it was not there" is a normal outcome rather than a failure.
_remove_container "${CONTAINER_NAME}_sandbox" >/dev/null 2>&1 || true

# The data volume stays. It holds the worker profiles, which cost hours of GPU
# time to rebuild; only destroy.sh removes it. The sandbox has no volume at all
# — every run's scratch space is a tmpfs directory that is deleted with the run.
echo "Uninstallation of ${CONTAINER_NAME} complete"
echo "Note: Data volume has been preserved. To remove all data, use destroy.sh"
