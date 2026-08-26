#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME" "DATA_VOLUME"

# The only script that removes the volume. Taking it also takes the cached
# worker profiles, so the next start re-benchmarks the entire fleet from cold.
echo "WARNING: This will PERMANENTLY DELETE all data for ${CONTAINER_NAME}"

bash ./stop.sh
_remove_container "$CONTAINER_NAME" || true
_remove_container "${CONTAINER_NAME}_sandbox" >/dev/null 2>&1 || true
_remove_volume "$DATA_VOLUME"

echo "Service and all data destroyed"
