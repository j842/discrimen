package router

import (
	"context"
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

// openSnapshotTestStore builds a real request-log database — same schema, same
// WAL mode, same single connection as production — and returns it with its path.
func openSnapshotTestStore(t *testing.T) (*LogStore, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "logs.sqlite")
	store, err := openLogStore(path, 16384, "test-secret")
	if err != nil {
		t.Fatalf("openLogStore: %v", err)
	}
	t.Cleanup(func() { store.Close() })
	return store, path
}

// TestSnapshotIsConsistentWhileTheRouterWrites is the whole point of the
// command. backup.sh runs it against the database a live router is writing to,
// and what must come out is a file that opens cleanly and still has
// worker_profiles in it — because the alternative, restic copying logs.sqlite,
// -wal and -shm at three different instants, is what this replaces.
func TestSnapshotIsConsistentWhileTheRouterWrites(t *testing.T) {
	store, path := openSnapshotTestStore(t)
	ctx := context.Background()

	if err := store.SaveWorkerProfile(ctx, "llm-6000pro", &WorkerProfile{
		Model: "qwen38-27b", Quality: 8, ContextK: 256, MaxConcurrency: 6, BaselineTPS: 95,
	}); err != nil {
		t.Fatalf("seed profile: %v", err)
	}

	// A writer running for the duration of the copy. VACUUM INTO takes a read
	// transaction, so the snapshot must be the database as of one instant
	// regardless of how many of these land while it runs.
	var writes atomic.Int64
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			if err := store.SaveWorkerProfile(ctx, "churn", &WorkerProfile{
				Model: "churn-model", Quality: i % 10,
			}); err != nil {
				return
			}
			writes.Add(1)
		}
	}()

	// The seeded profile is still sitting in the write-ahead log at this point,
	// not in logs.sqlite — nothing has checkpointed. Asserting that here is what
	// makes the LoadWorkerProfile check below mean something: it proves the copy
	// includes committed data the main database file does not yet hold, which is
	// exactly what a file-level copy of logs.sqlite alone would miss.
	wal, err := os.Stat(path + "-wal")
	if err != nil || wal.Size() == 0 {
		t.Fatalf("expected an unflushed WAL to copy from (size=%v err=%v)", wal, err)
	}

	dest := filepath.Join(t.TempDir(), "snapshot", "logs.sqlite")
	size, err := snapshotDatabase(ctx, path, dest)
	close(stop)
	wg.Wait()

	if err != nil {
		t.Fatalf("snapshotDatabase: %v", err)
	}
	if size <= 0 {
		t.Fatalf("snapshot is %d bytes", size)
	}
	if writes.Load() == 0 {
		t.Fatal("no concurrent writes landed — the test proved nothing about a live database")
	}

	// The snapshot must be a database in its own right, not a torn copy.
	snap, err := sql.Open("sqlite", dest)
	if err != nil {
		t.Fatalf("open snapshot: %v", err)
	}
	defer snap.Close()
	var integrity string
	if err := snap.QueryRowContext(ctx, `PRAGMA integrity_check`).Scan(&integrity); err != nil {
		t.Fatalf("integrity_check: %v", err)
	}
	if integrity != "ok" {
		t.Fatalf("integrity_check = %q, want ok", integrity)
	}

	// And it must still hold the row the fleet cannot cheaply re-earn.
	restored, err := openLogStore(dest, 16384, "test-secret")
	if err != nil {
		t.Fatalf("reopen snapshot as a store: %v", err)
	}
	defer restored.Close()
	p, ok := restored.LoadWorkerProfile(ctx, "llm-6000pro", "qwen38-27b")
	if !ok {
		t.Fatal("worker_profiles row did not survive the snapshot")
	}
	if p.Quality != 8 || p.BaselineTPS != 95 {
		t.Fatalf("profile came back altered: %+v", p)
	}
}

// TestSnapshotReplacesAPreviousSnapshot covers the sharp edge: VACUUM INTO
// refuses to write to a path that already exists, so without the unlink every
// backup after the first would fail.
func TestSnapshotReplacesAPreviousSnapshot(t *testing.T) {
	store, path := openSnapshotTestStore(t)
	ctx := context.Background()
	if err := store.SaveWorkerProfile(ctx, "w", &WorkerProfile{Model: "m", Quality: 5}); err != nil {
		t.Fatalf("seed: %v", err)
	}

	dest := filepath.Join(t.TempDir(), "snapshot", "logs.sqlite")
	if _, err := snapshotDatabase(ctx, path, dest); err != nil {
		t.Fatalf("first snapshot: %v", err)
	}
	// A stale WAL beside the previous snapshot has to go too: it belongs to a
	// database that no longer exists, and SQLite may try to replay it.
	if err := os.WriteFile(dest+"-wal", []byte("stale frames"), 0o600); err != nil {
		t.Fatalf("plant stale wal: %v", err)
	}
	if _, err := snapshotDatabase(ctx, path, dest); err != nil {
		t.Fatalf("second snapshot: %v", err)
	}
	if _, err := os.Stat(dest + "-wal"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("stale -wal survived the second snapshot (err=%v)", err)
	}
}

// TestSnapshotRefusesAMissingSource is the failure mode a backup must not have.
// sql.Open creates the file it cannot find, so a mistyped path would otherwise
// vacuum a brand-new empty database into the destination and exit 0 — a backup
// reporting success while archiving nothing.
func TestSnapshotRefusesAMissingSource(t *testing.T) {
	dir := t.TempDir()
	missing := filepath.Join(dir, "not-here.sqlite")
	dest := filepath.Join(dir, "snapshot.sqlite")

	if _, err := snapshotDatabase(context.Background(), missing, dest); err == nil {
		t.Fatal("snapshotDatabase accepted a source that does not exist")
	}
	if _, err := os.Stat(missing); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("snapshotDatabase created the missing source database")
	}
	if _, err := os.Stat(dest); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("snapshotDatabase wrote a destination for a source that does not exist")
	}
}
