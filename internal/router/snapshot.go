package router

import (
	"context"
	"database/sql"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

// `discrimen snapshot DEST` writes a transactionally consistent copy of the
// request-log database to DEST.
//
// It exists for the deployment template's backup.sh. Dropshell's restic pass
// mounts the data volume read-only and reads the raw files out from under a
// router that is still serving, so what reaches the repository is logs.sqlite,
// logs.sqlite-wal and logs.sqlite-shm as each of them stood at a slightly
// different instant. WAL recovery survives most of that. "Most" is not a
// property a backup should have, and the table being protected is
// worker_profiles — the cached cold-start profile for the whole fleet, whose
// loss re-benchmarks every worker from cold.
//
// VACUUM INTO holds a single read transaction across the entire copy, so DEST
// is the database as of one instant no matter what is written meanwhile. It
// takes no write lock and modifies nothing, which is why this is preferable to
// stopping the container for the length of the restic run: the alternative
// consistency guarantee is an outage on the service every client routes
// through.
//
// This lives in the router binary because the pure-Go SQLite driver is already
// linked here (see the Dockerfile's CGO_ENABLED=0 note) and neither the alpine
// runtime image nor the host it runs on has a sqlite3 to call instead.
func runSnapshotCommand(argv []string) bool {
	if len(argv) < 2 || argv[1] != "snapshot" {
		return false
	}
	fs := flag.NewFlagSet("snapshot", flag.ExitOnError)
	fs.Usage = snapshotUsage
	// Defaulted after parsing rather than here, so building the usage string
	// does not depend on the environment the command happens to run in.
	src := fs.String("db", "", "source database (default: $LOG_DB_PATH, exactly as the server resolves it)")
	_ = fs.Parse(argv[2:])

	dest := fs.Arg(0)
	if dest == "" {
		snapshotUsage()
		os.Exit(2)
	}
	from := *src
	if from == "" {
		// loadConfig rather than a second envOr with the same fallback baked in.
		// The frozen default path is stated once, and a snapshot that read a
		// different database than the server writes would back up the wrong file
		// while reporting success.
		from = loadConfig().LogDBPath
	}

	size, err := snapshotDatabase(context.Background(), from, dest)
	if err != nil {
		fmt.Fprintf(os.Stderr, "snapshot: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Snapshot written: %s (%.1f MB, consistent as of one instant)\n", dest, float64(size)/(1<<20))
	return true
}

func snapshotUsage() {
	fmt.Fprint(os.Stderr, `discrimen snapshot — write a consistent copy of the request-log database

  snapshot DEST [-db SOURCE]

DEST must not already exist as a directory, and is replaced if it exists as a
file. SOURCE defaults to $LOG_DB_PATH, which is what the running server opens.

The copy is taken with SQLite's VACUUM INTO: one read transaction across the
whole database, no write lock, nothing blocked. Intended for backup.sh, which
writes it inside the data volume so the volume backup carries it, and for
restore.sh, which promotes it over the raw files restic captured mid-write.
`)
}

// snapshotDatabase VACUUMs src INTO dest and returns the size of the result.
func snapshotDatabase(ctx context.Context, src, dest string) (int64, error) {
	if src == "" {
		return 0, errors.New("no source database (set -db or LOG_DB_PATH)")
	}
	if dest == "" {
		return 0, errors.New("no destination path")
	}

	// sql.Open on a path that does not exist CREATES an empty database, so
	// without this a typo'd -db would vacuum nothing into dest and report
	// success — the one failure mode a backup must not have.
	if _, err := os.Stat(src); err != nil {
		return 0, fmt.Errorf("source database %s: %w", src, err)
	}

	// 0o700 for the same reason openLogStore uses it: this is a full copy of a
	// database holding request bodies, and it lands beside nothing that should
	// widen its permissions.
	if err := os.MkdirAll(filepath.Dir(dest), 0o700); err != nil {
		return 0, err
	}
	// VACUUM INTO refuses to overwrite, so without this every run after the
	// first fails. The -wal and -shm go too: they can only belong to some
	// earlier copy, and a WAL left beside a database it did not come from is
	// not inert — it is frames SQLite may choose to replay.
	for _, stale := range []string{dest, dest + "-wal", dest + "-shm"} {
		if err := os.Remove(stale); err != nil && !errors.Is(err, os.ErrNotExist) {
			return 0, fmt.Errorf("clear previous snapshot %s: %w", stale, err)
		}
	}

	db, err := sql.Open("sqlite", src)
	if err != nil {
		return 0, err
	}
	defer db.Close()
	// One connection, so the busy_timeout below is set on the same connection
	// that runs the VACUUM rather than on whichever one the pool hands out.
	db.SetMaxOpenConns(1)
	// The router is a live writer on this file. A reader that gave up on the
	// first SQLITE_BUSY would fail a backup for a lock held for microseconds;
	// 30s is far past any write this router makes and still bounded.
	if _, err := db.ExecContext(ctx, `PRAGMA busy_timeout=30000`); err != nil {
		return 0, fmt.Errorf("set busy timeout: %w", err)
	}
	if _, err := db.ExecContext(ctx, `VACUUM INTO ?`, dest); err != nil {
		return 0, fmt.Errorf("vacuum into %s: %w", dest, err)
	}

	info, err := os.Stat(dest)
	if err != nil {
		return 0, fmt.Errorf("snapshot reported success but %s is not there: %w", dest, err)
	}
	return info.Size(), nil
}
