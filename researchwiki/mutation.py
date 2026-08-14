"""All-or-nothing multi-file mutations for the wiki tree.

`fsatomic` makes a single write atomic. That is not enough for the operations
that touch several files at once: `promote_to_wiki` writes a page, moves a PDF,
splices back-links into N existing pages, and appends to `index.md` and
`log.md`. Each step is individually atomic and nothing binds them, so a failure
after step two leaves a paper half-landed — a page on disk with no PDF, no
back-links, no catalogue entry (see CHANGELOG 0.3.2, and `prompts/recovery.md`
§ Half-landed promote).

The shape here is adapted from OpenKB's `mutation.py` (Apache-2.0), reduced to
the four ideas that carry the weight:

1. **Declare the touched paths up front.** Everything that will be written is
   backed up before anything is written. A path that doesn't exist yet is
   recorded as "created" so rollback removes it.
2. **Journal with an explicit commit point.** `mark_committed()` flips the
   journal the instant the work is durable. `discard()` is *post-commit*
   cleanup and is best-effort: it runs after the commit point, so its failure
   must never trigger a rollback. That ordering is the subtle part.
3. **Recover on the next run, not via a repair command.** `recover_pending()`
   drains stale journals: `active` rolls back, `committed` is discarded.
4. **Cap rollback attempts.** A deterministically-failing rollback (ENOSPC, a
   permission problem) must surface rather than spin on every subsequent run.

Deliberately *not* adopted: OpenKB's hardlink backups. They are an O(1)-per-file
optimisation for whole-tree snapshots, they carry real edge cases (EXDEV, and
cloud-sync folders that refuse hardlinks), and this repo's `wiki/` and `papers/`
are commonly symlinks into a synced vault — exactly that environment. A promote
touches a handful of files; plain copies are fine.

**The transaction spans two storage systems.** File state is journalled and
survives a crash; the DB row written by `wiki.commit_page` is not. Callers
register an in-process undo with `also_undo()`, which runs on an in-process
rollback but *cannot* run during crash recovery — nothing is persisted to
replay it. That is deliberate rather than an oversight: the markdown page is
canonical and the DB is a derived index, so a crash-recovered rollback removes
the page and the next `db rebuild` reconciles the orphaned row. `db verify`
reports the drift in the meantime.

Set `RW_MUTATION_JOURNAL=0` to bypass journalling entirely (mutations run
unwrapped, exactly as before this module existed). An escape hatch for one
release, not a supported mode.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .fsatomic import write_json_atomic
from .log import log
from .paths import mutation_dir

# A rollback that keeps failing for the same reason must stop retrying. Without
# a cap it is re-attempted on every subsequent run forever, re-doing the failed
# work and never releasing its backup directory.
MAX_ROLLBACK_ATTEMPTS = 5

JOURNAL_VERSION = 1


def journalling_enabled() -> bool:
    """False when `RW_MUTATION_JOURNAL=0`. Any other value (or unset) is on."""
    return os.environ.get("RW_MUTATION_JOURNAL", "1").strip() != "0"


@dataclass
class Snapshot:
    """A declared set of paths, backed up, with a journal recording the intent."""

    operation: str
    journal_path: Path
    backup_dir: Path
    details: dict = field(default_factory=dict)
    # target -> backup path, or None when the target did not exist (rollback
    # removes it rather than restoring it).
    entries: dict[Path, Path | None] = field(default_factory=dict)
    attempts: int = 0
    committed: bool = False
    # In-process only, never persisted — see the module docstring on why crash
    # recovery cannot replay these.
    _undo: list[Callable[[], None]] = field(default_factory=list, repr=False)

    # ---------- registration ----------

    def also_undo(self, fn: Callable[[], None]) -> None:
        """Register a non-file undo to run on an in-process rollback.

        For state that isn't a file — the `state.db` row `commit_page` writes.
        Runs last-registered-first, after the files are restored.
        """
        self._undo.append(fn)

    # ---------- journal ----------

    def _payload(self, status: str) -> dict:
        return {
            "version": JOURNAL_VERSION,
            "operation": self.operation,
            "status": status,
            "attempts": self.attempts,
            "created_at": _dt.datetime.now().replace(microsecond=0).isoformat(),
            "backup_dir": str(self.backup_dir),
            "details": self.details,
            "entries": [
                {"target": str(t), "backup": (str(b) if b else None)}
                for t, b in self.entries.items()
            ],
        }

    def _write(self, status: str) -> None:
        write_json_atomic(self.journal_path, self._payload(status))

    def mark_committed(self) -> None:
        """The commit point. After this, recovery discards rather than rolls back."""
        self.committed = True
        self._write("committed")

    # ---------- outcomes ----------

    def discard(self) -> None:
        """Post-commit cleanup: drop the backups and the journal.

        Best-effort by contract. This runs *after* the commit point, so a
        failure here means some bytes are left in `.mutation/` — untidy, never
        incorrect, and never a reason to undo committed work.
        """
        try:
            shutil.rmtree(self.backup_dir, ignore_errors=True)
            self.journal_path.unlink(missing_ok=True)
        except OSError as e:
            log(f"WARN: could not clean up journal {self.journal_path.name}: {e}",
                tag="mutation")

    def rollback(self) -> None:
        """Restore every declared path, then run the registered undos.

        Raises the first failure after recording the attempt, so the journal
        survives for the recovery drain to retry.
        """
        self.attempts += 1
        self._write("active")
        _restore_entries(self.entries)
        for fn in reversed(self._undo):
            try:
                fn()
            except Exception as e:  # an undo must not mask the original failure
                log(f"WARN: rollback undo failed: {e}", tag="mutation")
        self.discard()


def _restore_entries(entries: dict[Path, Path | None]) -> None:
    """Copy each backup back over its target; remove targets that were created."""
    for target, backup in entries.items():
        if backup is None:
            # Didn't exist before the mutation — a rollback un-creates it.
            try:
                Path(target).unlink(missing_ok=True)
            except OSError as e:
                raise RuntimeError(f"rollback could not remove {target}: {e}") from e
            continue
        src = Path(backup)
        if not src.exists():
            # Backup lost (manual cleanup, or a partial recovery). Leave the
            # target as-is rather than deleting content we cannot restore.
            log(f"WARN: backup missing for {target}; leaving current file in place",
                tag="mutation")
            continue
        try:
            dest = Path(target)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".rollback-tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, dest)
        except OSError as e:
            raise RuntimeError(f"rollback could not restore {target}: {e}") from e


def _new_journal_paths() -> tuple[Path, Path]:
    root = mutation_dir()
    root.mkdir(parents=True, exist_ok=True)
    ident = f"{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    return root / f"{ident}.json", root / ident


def snapshot(
    paths: Iterable[Path | str],
    *,
    operation: str,
    details: dict | None = None,
) -> Snapshot:
    """Back up `paths` and open a journal. Duplicates and None entries are ignored.

    Every path is resolved, so two spellings of the same file (a relative and an
    absolute, or one reached through a symlinked parent) collapse to one entry
    rather than backing the file up twice and restoring it twice.
    """
    journal_path, backup_dir = _new_journal_paths()
    backup_dir.mkdir(parents=True, exist_ok=True)

    entries: dict[Path, Path | None] = {}
    for i, raw in enumerate(paths):
        if raw is None:
            continue
        target = Path(raw).resolve()
        if target in entries:
            continue
        if target.exists():
            backup = backup_dir / f"{i:03d}-{target.name}"
            shutil.copy2(target, backup)
            entries[target] = backup
        else:
            entries[target] = None

    snap = Snapshot(
        operation=operation,
        journal_path=journal_path,
        backup_dir=backup_dir,
        details=details or {},
        entries=entries,
    )
    snap._write("active")
    return snap


@contextmanager
def mutation(
    paths: Iterable[Path | str],
    *,
    operation: str,
    details: dict | None = None,
) -> Iterator[Snapshot]:
    """Run a multi-file mutation transactionally.

        with mutation(paths, operation="promote") as snap:
            ...                 # do the work
            snap.mark_committed()

    Leaving the block without `mark_committed()` — by exception *or* by an
    early `return` — rolls everything back. That is deliberate: `promote`
    returns early on failure, and the shape that makes "forgot to commit"
    behave like "failed" is the safe one.

    A no-op passthrough when `RW_MUTATION_JOURNAL=0`.
    """
    if not journalling_enabled():
        yield Snapshot(
            operation=operation,
            journal_path=Path(os.devnull),
            backup_dir=Path(os.devnull),
            details=details or {},
        )
        return

    snap = snapshot(paths, operation=operation, details=details)
    try:
        yield snap
    except BaseException:
        try:
            snap.rollback()
        except Exception as e:
            log(f"ERROR: rollback failed for {operation}; journal retained at "
                f"{snap.journal_path} — the next ingest will retry it: {e}",
                tag="mutation")
        raise
    else:
        if snap.committed:
            snap.discard()
        else:
            snap.rollback()


# ---------- recovery ----------

def pending_journals() -> list[dict]:
    """Every journal on disk, parsed. Read-only — safe for `status`.

    Each dict carries its own `journal_path` so a caller can report or drain it.
    """
    root = mutation_dir()
    if not root.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data["journal_path"] = str(path)
        out.append(data)
    return out


def recover_pending() -> list[str]:
    """Drain stale journals. Returns a human-readable line per journal handled.

    `active` rolls back, `committed` is discarded. A journal whose rollback has
    already failed `MAX_ROLLBACK_ATTEMPTS` times is left alone and reported, so
    a deterministic failure surfaces instead of being retried forever.

    Called at the start of the write paths (`agent ingest`, `ingest`). Never
    from a read-only command.
    """
    notes: list[str] = []
    for data in pending_journals():
        journal_path = Path(data["journal_path"])
        status = data.get("status")
        operation = data.get("operation", "?")
        backup_dir = Path(data.get("backup_dir", ""))

        if status == "committed":
            shutil.rmtree(backup_dir, ignore_errors=True)
            journal_path.unlink(missing_ok=True)
            notes.append(f"discarded committed journal for {operation}")
            continue

        attempts = int(data.get("attempts", 0))
        if attempts >= MAX_ROLLBACK_ATTEMPTS:
            notes.append(
                f"journal for {operation} left in place after {attempts} failed "
                f"rollback attempts — inspect {journal_path} by hand"
            )
            continue

        entries = {
            Path(e["target"]): (Path(e["backup"]) if e.get("backup") else None)
            for e in data.get("entries", [])
        }
        data["attempts"] = attempts + 1
        data["status"] = "active"
        write_json_atomic(journal_path, {k: v for k, v in data.items()
                                         if k != "journal_path"})
        try:
            _restore_entries(entries)
        except Exception as e:
            notes.append(f"rollback of {operation} failed (attempt {attempts + 1}): {e}")
            continue
        shutil.rmtree(backup_dir, ignore_errors=True)
        journal_path.unlink(missing_ok=True)
        notes.append(
            f"rolled back interrupted {operation} ({len(entries)} path(s) restored)"
        )
    return notes
