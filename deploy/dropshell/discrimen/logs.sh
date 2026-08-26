#!/bin/bash
source "${AGENT_PATH}/common.sh"
_check_required_env_vars "CONTAINER_NAME"

# The router only. `docker logs -f` follows one container, and interleaving two
# streams would need a background job whose lifetime nothing here manages — a
# tail left running after Ctrl-C is worse than a second command to type.
echo "(grading sandbox: docker logs -f ${CONTAINER_NAME}_sandbox)" >&2
docker logs -f --tail=200 "$CONTAINER_NAME"
