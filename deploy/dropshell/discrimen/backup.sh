#!/bin/bash
source "${AGENT_PATH}/common.sh"
source "${AGENT_PATH}/datacommands_v2.sh"
_check_required_env_vars "CONTAINER_NAME" "DATA_VOLUME" "IMAGE_REGISTRY" "IMAGE_REPO" "IMAGE_TAG"

# Where the consistent copy lands. Inside the data volume, so the single
# manifest line at the bottom carries it without restic needing to know it
# exists. Hard-coded rather than derived from LOG_DB_PATH because restore.sh has
# to find it again from the far side of a restic restore, and the two only agree
# if neither of them is computing it.
SNAPSHOT_PATH="/data/snapshot/logs.sqlite"
SNAPSHOT_CONTAINER="${CONTAINER_NAME}_snapshot"
SNAPSHOT_IMAGE="${IMAGE_REGISTRY}/${IMAGE_REPO}:${IMAGE_TAG}"

# Dropshell's restic pass mounts this volume read-only and reads the raw files
# while the router is still serving, so logs.sqlite, -wal and -shm each get
# captured at a slightly different instant and the restored set is only as good
# as WAL recovery makes it. `discrimen snapshot` holds one read transaction
# across the whole database and writes a single self-contained file, which
# restore.sh promotes back into place.
#
# A one-shot container rather than `docker exec` on the running router: VACUUM
# INTO takes no write lock and needs nothing from the running process, so one
# code path covers a service that is up and a service that is stopped, and the
# snapshot is always at least as new as the files beside it. `docker exec` would
# have needed a second branch for the stopped case, and that branch is where a
# snapshot from a previous run survives to be restored over newer data.
#
# --network none because copying a database has no business dialling anything.
# The timeout is for the deploy window where the template has been installed but
# the image has not: an older binary does not know this subcommand, and before
# it learned to reject unknown arguments it would start a whole second router
# here and block the backup until something killed it.
echo "Snapshotting the router database (VACUUM INTO ${SNAPSHOT_PATH})..."
docker rm -f "${SNAPSHOT_CONTAINER}" >/dev/null 2>&1 || true
if ! timeout 600 docker run --rm --name "${SNAPSHOT_CONTAINER}" --network none \
        -v "${DATA_VOLUME}:/data" \
        -e "LOG_DB_PATH=${LOG_DB_PATH:-/data/llm-router/logs.sqlite}" \
        "${SNAPSHOT_IMAGE}" \
        discrimen snapshot "${SNAPSHOT_PATH}"; then
    # Never fatal. A hot copy is a worse backup than a consistent one and a far
    # better backup than none, so this warns and lets the restic pass proceed.
    #
    # The stale snapshot has to go with it. Leaving it means the next restore
    # promotes a copy older than the live files it was archived beside, silently
    # rolling the database back to whenever the last snapshot succeeded — which
    # is worse than the hot copy this is falling back to.
    echo "WARNING: consistent snapshot failed. Backing up the live database files as-is;"
    echo "         restoring them will depend on SQLite WAL recovery."
    docker rm -f "${SNAPSHOT_CONTAINER}" >/dev/null 2>&1 || true
    docker run --rm --network none -v "${DATA_VOLUME}:/data" \
        "${SNAPSHOT_IMAGE}" \
        rm -f "${SNAPSHOT_PATH}" "${SNAPSHOT_PATH}-wal" "${SNAPSHOT_PATH}-shm" >/dev/null 2>&1 || true
fi

backup_items "volume:data:${DATA_VOLUME}"
