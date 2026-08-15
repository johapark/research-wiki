"""All-or-nothing multi-file mutations for the wiki tree.

`fsatomic` makes a single write atomic, which is not enough for the operations
that touch several files at once: `promote_to_wiki` writes a page, moves a PDF,
splices back-links into N existing pages, and appends to `index.md` and `log.md`.
Every step is individually atomic and nothing binds them, so a failure after step
two leaves a paper half-landed — a page on disk with no PDF, no back-links and no
catalogue entry (see CHANGELOG 0.3.2 and `prompts/recovery.md` § Half-landed
promote).

Four properties carry the weight:

1. **Declare the touched paths up front.** Everything that will be written is
   copied aside before anything is written. A path that does not exist yet is
   recorded with no backup, so rolling back deletes it instead of restoring it.
2. **An explicit commit point.** `mark_committed()` flips the journal the moment
   the work is durable. `discard()` is *post-commit* cleanup and is best-effort by
   contract: it runs after the commit point, so its failure must never trigger a
   rollback. That ordering is the subtle part, and getting it backwards would let
   a tidy-up error undo landed work.
3. **Recovery happens on the next run, not through a repair command.**
   `recover_pending()` drains whatever journals it finds: `active` rolls back,
   `committed` is cleaned up.
4. **Rollback attempts are capped.** A rollback failing deterministically —
   ENOSPC, a permission problem — must eventually surface to a human instead of
   being re-attempted on every subsequent run forever.

**Backups are plain copies, deliberately.** Hardlinking a backup is an
O(1)-per-file trick that pays off when snapshotting whole trees, but it brings
EXDEV and "this filesystem refuses hardlinks" handling with it, and `wiki/` and
`papers/` here are commonly symlinks into a cloud-synced vault — precisely the
environment where that breaks. A promote touches a handful of files, so copying
them is cheap and has no edge cases.

**The transaction spans two storage systems.** File state is journalled and
survives a crash; the `state.db` row written by `wiki.commit_page` is not. Callers
register an in-process undo with `also_undo()`, which runs on an in-process
rollback but *cannot* run during crash recovery, because nothing about it is
persisted to replay. That is a decision rather than an oversight: the markdown
page is canonical and the DB is a derived index, so a crash-recovered rollback
removes the page and the next `db rebuild` reconciles the orphaned row. `db
verify` reports the drift meanwhile.

Set `RW_MUTATION_JOURNAL=0` to bypass journalling entirely — mutations then run
unwrapped, exactly as they did before this module existed. An escape hatch for one
release, not a supported mode.
"""

from __future__ import annotations

import datetime
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

#: Give up re-trying a rollback after this many failures. A rollback that fails
#: for a standing reason will fail identically next time, and without a ceiling it
#: is retried on every run forever — redoing the failed work each time and never
#: releasing its backup directory.
MAX_ROLLBACK_ATTEMPTS = 5

#: On-disk journal schema. Bump only for a change old journals cannot be read
#: through; `recover_pending` must keep draining journals written by the release
#: before this one, or an upgrade mid-mutation loses its rollback.
JOURNAL_VERSION = 1

_ACTIVE = "active"
_COMMITTED = "committed"


def journalling_enabled() -> bool:
    """False only when `RW_MUTATION_JOURNAL=0`. Unset or anything else is on."""
    return os.environ.get("RW_MUTATION_JOURNAL", "1").strip() != "0"


@dataclass(frozen=True)
class BackedUpPath:
    """One declared path and the copy taken of it, if there was anything to copy.

    A record per path rather than a `{target: backup_or_None}` mapping, because
    `None` there had to mean "no backup exists *because the file did not*", and
    that sentinel reading is what the undo logic turns on. Naming it
    `existed_before` puts the distinction in the type instead of in a comment.
    """

    target: Path
    backup: Path | None

    @property
    def existed_before(self) -> bool:
        return self.backup is not None

    def undo(self) -> None:
        """Put this path back the way it was. Raises `RuntimeError` on failure."""
        if not self.existed_before:
            self._uncreate()
        else:
            self._restore()

    def _uncreate(self) -> None:
        """The mutation brought this path into being, so undoing means removing it."""
        try:
            if self.target.is_dir() and not self.target.is_symlink():
                shutil.rmtree(self.target)
            else:
                self.target.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"rollback could not remove {self.target}: {exc}") from exc

    def _restore(self) -> None:
        source = self.backup
        assert source is not None       # guarded by `existed_before`
        if not source.exists():
            # The backup is gone — hand-cleaned `.mutation/`, or a recovery that
            # got part-way. Overwriting the target with nothing would destroy
            # content we have no copy of, so leave whatever is there and say so.
            log(f"WARN: no backup left for {self.target}; current file kept as-is",
                tag="mutation")
            return
        try:
            self.target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                # Directory targets (a paper's `.supp/`, a figures cache) stage
                # a full copy beside the target, then swap — same
                # old-or-new-never-half guarantee the file path gets from
                # `os.replace`, at directory granularity.
                staged = self.target.parent / (self.target.name + ".rollback-tmp")
                if staged.exists():
                    shutil.rmtree(staged, ignore_errors=True)
                shutil.copytree(source, staged, symlinks=True)
                if self.target.is_dir() and not self.target.is_symlink():
                    shutil.rmtree(self.target)
                else:
                    self.target.unlink(missing_ok=True)
                os.replace(staged, self.target)
            else:
                staged = self.target.with_suffix(self.target.suffix + ".rollback-tmp")
                shutil.copy2(source, staged)
                os.replace(staged, self.target)
        except (OSError, shutil.Error) as exc:
            raise RuntimeError(f"rollback could not restore {self.target}: {exc}") from exc

    def as_journal_record(self) -> dict:
        return {"target": str(self.target),
                "backup": str(self.backup) if self.backup else None}

    @classmethod
    def from_journal_record(cls, record: dict) -> "BackedUpPath":
        raw = record.get("backup")
        return cls(target=Path(record["target"]), backup=Path(raw) if raw else None)


def _undo_each(paths: Iterable[BackedUpPath]) -> None:
    """Undo every declared path, or raise on the first one that will not budge.

    Raising rather than continuing is intentional: a half-restored tree that
    reported success is worse than one that stopped and left its journal behind
    for `recover_pending` to try again.
    """
    for entry in paths:
        entry.undo()


@dataclass
class Snapshot:
    """A declared set of paths, copied aside, with a journal recording the intent."""

    operation: str
    journal_path: Path
    backup_dir: Path
    details: dict[str, object] = field(default_factory=dict)
    entries: list[BackedUpPath] = field(default_factory=list)
    attempts: int = 0
    committed: bool = False
    # In-process only and never serialized — see the module docstring for why
    # crash recovery is unable to replay these.
    _undo_hooks: list[Callable[[], None]] = field(default_factory=list, repr=False)

    # ---------- registration ----------

    def also_undo(self, fn: Callable[[], None]) -> None:
        """Register a non-file undo, run on an in-process rollback.

        For state that is not a file — the `state.db` row `commit_page` writes.
        Hooks run in reverse registration order, after the files are back.
        """
        self._undo_hooks.append(fn)

    # ---------- journal ----------

    def _journal_document(self, status: str) -> dict:
        stamped = datetime.datetime.now().replace(microsecond=0).isoformat()
        return {
            "version": JOURNAL_VERSION,
            "operation": self.operation,
            "status": status,
            "attempts": self.attempts,
            "created_at": stamped,
            "backup_dir": str(self.backup_dir),
            "details": self.details,
            "entries": [e.as_journal_record() for e in self.entries],
        }

    def _persist(self, status: str) -> None:
        write_json_atomic(self.journal_path, self._journal_document(status))

    def mark_committed(self) -> None:
        """The commit point. Past here, recovery cleans up instead of undoing."""
        self.committed = True
        self._persist(_COMMITTED)

    # ---------- outcomes ----------

    def discard(self) -> None:
        """Post-commit cleanup: drop the copies and the journal.

        Best-effort by contract. It runs *after* the commit point, so failing here
        leaves some bytes in `.mutation/` — untidy, never incorrect, and never a
        reason to undo work that already landed.
        """
        try:
            shutil.rmtree(self.backup_dir, ignore_errors=True)
            self.journal_path.unlink(missing_ok=True)
        except OSError as exc:
            log(f"WARN: could not clean up journal {self.journal_path.name}: {exc}",
                tag="mutation")

    def rollback(self) -> None:
        """Put every declared path back, then run the registered hooks.

        The attempt is recorded before the work starts, so a failure leaves an
        `active` journal with an incremented count for the recovery drain to pick
        up rather than a journal that looks untried.
        """
        self.attempts += 1
        self._persist(_ACTIVE)
        _undo_each(self.entries)
        for hook in reversed(self._undo_hooks):
            try:
                hook()
            except Exception as exc:   # a hook must not mask the original failure
                log(f"WARN: rollback undo failed: {exc}", tag="mutation")
        self.discard()


def _allocate_journal() -> tuple[Path, Path]:
    """Reserve a journal file and its sibling backup directory under `.mutation/`."""
    root = mutation_dir()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    ident = f"{stamp}-{uuid.uuid4().hex[:8]}"
    return root / f"{ident}.json", root / ident


def snapshot(
    paths: Iterable[Path | str],
    *,
    operation: str,
    details: dict | None = None,
) -> Snapshot:
    """Copy `paths` aside and open a journal. `None` and repeats are ignored.

    Each path is resolved first, so two spellings of one file — a relative and an
    absolute, or one reached through a symlinked parent — collapse to a single
    entry instead of being backed up twice and restored twice.
    """
    journal_path, backup_dir = _allocate_journal()
    backup_dir.mkdir(parents=True, exist_ok=True)

    entries: list[BackedUpPath] = []
    claimed: set[Path] = set()
    try:
        for raw in paths:
            if raw is None:
                continue
            target = Path(raw).resolve()
            if target in claimed:
                continue
            claimed.add(target)
            if target.is_dir():
                # Directories are real declared targets here — `remove` snapshots
                # a paper's `.supp/` and cache dirs before rmtree-ing them.
                # `copy2` on one raises, which used to kill the whole mutation.
                backup = backup_dir / f"{len(entries):03d}-{target.name}"
                shutil.copytree(target, backup, symlinks=True)
                entries.append(BackedUpPath(target=target, backup=backup))
            elif target.exists():
                backup = backup_dir / f"{len(entries):03d}-{target.name}"
                shutil.copy2(target, backup)
                entries.append(BackedUpPath(target=target, backup=backup))
            else:
                entries.append(BackedUpPath(target=target, backup=None))
    except BaseException:
        # The journal is only written below, so a copy failure here would
        # otherwise strand a backup directory with no journal beside it —
        # invisible to `recover_pending`, which drains `*.json` only.
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise

    snap = Snapshot(
        operation=operation,
        journal_path=journal_path,
        backup_dir=backup_dir,
        details=dict(details or {}),
        entries=entries,
    )
    snap._persist(_ACTIVE)
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

    Leaving the block without `mark_committed()` rolls everything back — whether
    it left by exception or by an early `return`. That is deliberate: `promote`
    returns early on failure, and the shape where "forgot to commit" behaves like
    "failed" is the safe one to get wrong.

    A no-op passthrough when `RW_MUTATION_JOURNAL=0`.
    """
    if not journalling_enabled():
        yield Snapshot(
            operation=operation,
            journal_path=Path(os.devnull),
            backup_dir=Path(os.devnull),
            details=dict(details or {}),
        )
        return

    snap = snapshot(paths, operation=operation, details=details)
    try:
        yield snap
    except BaseException:
        try:
            snap.rollback()
        except Exception as exc:
            log(f"ERROR: rollback failed for {operation}; journal retained at "
                f"{snap.journal_path} — the next ingest will retry it: {exc}",
                tag="mutation")
        raise
    else:
        if snap.committed:
            snap.discard()
        else:
            snap.rollback()


# ---------- recovery ----------

def pending_journals() -> list[dict]:
    """Every journal currently on disk, parsed. Read-only, so `status` may call it.

    Each dict carries its own `journal_path` so a caller can report on it or drain
    it. Unparseable files are skipped rather than raised on: a stray or truncated
    `.json` must not brick the write paths that drain this.
    """
    root = mutation_dir()
    if not root.is_dir():
        return []
    found: list[dict] = []
    for path in sorted(root.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        document["journal_path"] = str(path)
        found.append(document)
    return found


def _removable_backup_dir(raw: object) -> Path | None:
    """Validate a journal's recorded `backup_dir` before anything rmtree-s it.

    Returns the path only when it is *strictly inside* `.mutation/`; anything
    else — and above all a missing key, whose old `Path("")` fallback resolved
    to the current working directory and handed the whole wiki root to
    `shutil.rmtree` — comes back as None, meaning "do not remove anything".
    Journal entries still restore individually; only the bulk delete is gated.
    """
    if not raw or not isinstance(raw, str):
        return None
    root = mutation_dir().resolve()
    candidate = Path(raw).resolve()
    if candidate == root or root not in candidate.parents:
        return None
    return candidate


def _clean_up(journal_path: Path, backup_dir: Path | None) -> None:
    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)
    journal_path.unlink(missing_ok=True)


def recover_pending() -> list[str]:
    """Drain stale journals. Returns one human-readable line per journal handled.

    `active` is rolled back and `committed` is cleaned up. A journal whose
    rollback has already failed `MAX_ROLLBACK_ATTEMPTS` times is left untouched
    and reported, so a standing failure reaches a human instead of being retried
    forever. A journal from a *newer* schema than this release understands is
    also left in place — draining it with old code could lose its rollback.

    Called at the start of the write paths (`agent ingest`, `ingest`), never from
    a read-only command.
    """
    notes: list[str] = []
    for document in pending_journals():
        journal_path = Path(document["journal_path"])
        operation = document.get("operation", "?")
        backup_dir = _removable_backup_dir(document.get("backup_dir"))

        try:
            version = int(document.get("version", 0))
        except (TypeError, ValueError):
            version = 0
        if version > JOURNAL_VERSION:
            notes.append(
                f"journal for {operation} uses schema v{version}, newer than this "
                f"release understands (v{JOURNAL_VERSION}) — left in place at "
                f"{journal_path}"
            )
            continue

        if document.get("status") == _COMMITTED:
            _clean_up(journal_path, backup_dir)
            notes.append(f"discarded committed journal for {operation}")
            continue

        already_tried = int(document.get("attempts", 0))
        if already_tried >= MAX_ROLLBACK_ATTEMPTS:
            notes.append(
                f"journal for {operation} left in place after {already_tried} failed "
                f"rollback attempts — inspect {journal_path} by hand"
            )
            continue

        entries = [BackedUpPath.from_journal_record(r)
                   for r in document.get("entries", [])]

        # Record this attempt before making it, so a crash mid-rollback still
        # counts against the cap and cannot loop forever.
        this_attempt = already_tried + 1
        document["attempts"] = this_attempt
        document["status"] = _ACTIVE
        write_json_atomic(journal_path,
                          {k: v for k, v in document.items() if k != "journal_path"})
        try:
            _undo_each(entries)
        except Exception as exc:
            notes.append(f"rollback of {operation} failed (attempt {this_attempt}): {exc}")
            continue
        _clean_up(journal_path, backup_dir)
        notes.append(
            f"rolled back interrupted {operation} ({len(entries)} path(s) restored)"
        )

    # A backup directory with no `.json` beside it is the residue of a snapshot
    # that died mid-copy (or of a journal cleaned by hand). Nothing will ever
    # reference it again, but deleting it here could race a snapshot being taken
    # by a concurrent process right now — so it is reported, not removed.
    root = mutation_dir()
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and not (root / f"{entry.name}.json").exists():
                notes.append(
                    f"orphaned backup directory {entry} has no journal — left by an "
                    f"interrupted snapshot; safe to delete by hand"
                )
    return notes
