#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME"

bash ./stop.sh || _die "Failed to stop container"
_remove_container "$CONTAINER_NAME" || _die "Failed to remove container"

# The data volume stays. It holds the worker profiles, which cost hours of GPU
# time to rebuild; only destroy.sh removes it.
echo "Uninstallation of ${CONTAINER_NAME} complete"
echo "Note: Data volume has been preserved. To remove all data, use destroy.sh"
