#!/bin/bash
source "${AGENT_PATH}/common.sh"
source "${AGENT_PATH}/datacommands_v2.sh"
_check_required_env_vars "CONTAINER_NAME" "DATA_VOLUME" "IMAGE_REGISTRY" "IMAGE_REPO" "IMAGE_TAG"

restore_items "volume:data:${DATA_VOLUME}"

# The restored volume holds two copies of the database: the live files as restic
# found them mid-write, and the consistent one backup.sh wrote with VACUUM INTO.
# Promote the consistent one.
#
# Deleting -wal and -shm is the half that matters. They belong to the live copy
# being overwritten, and a WAL sitting beside a database it did not come from is
# not inert — it is a set of frames SQLite may decide to replay. The router
# converts the promoted file back to WAL itself on first open.
#
# No snapshot in the archive means it was taken by a template older than this
# one, or by a run where the snapshot step failed and said so. Leave the
# restored files alone in that case and let WAL recovery do what it can: that is
# the behaviour every existing snapshot in the repository was taken under, and
# it beats refusing to restore.
#
# Safe to write here because restore_items has already force-removed every
# container referencing this volume in order to replace it, so nothing holds the
# database open. Dropshell starts the service again after this script returns.
DB_PATH="${LOG_DB_PATH:-/data/llm-router/logs.sqlite}"
SNAPSHOT_PATH="/data/snapshot/logs.sqlite"

docker run --rm --network none -v "${DATA_VOLUME}:/data" \
    "${IMAGE_REGISTRY}/${IMAGE_REPO}:${IMAGE_TAG}" \
    sh -c "
        set -e
        if [ ! -f '${SNAPSHOT_PATH}' ]; then
            echo 'No consistent snapshot in this archive — keeping the restored database files as-is.'
            exit 0
        fi
        rm -f '${DB_PATH}-wal' '${DB_PATH}-shm'
        cp '${SNAPSHOT_PATH}' '${DB_PATH}'
        echo 'Promoted the consistent snapshot over the restored database.'
    " || _die "Failed to promote the consistent database snapshot"
