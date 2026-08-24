"""Atomic, utf-8, optionally cross-process-locked file writes.

Every page/index/cache writer in the package historically called `.write_text()`
directly: non-atomic (a crash mid-write truncates the file) and, for the shared
files edited by concurrent `agent ingest` subprocesses (`wiki/index.md`, back-link
target pages), racy. This module is the single primitive that fixes both.

- `write_text_atomic` / `write_json_atomic` — write to a sibling `.tmp` then
  `os.replace`, which is atomic on POSIX. A crash leaves either the old file or
  the new one, never a truncated one.
- `read_json` — guarded read: a truncated/corrupt cache file reads as a miss
  instead of crashing every subsequent run (the pattern S2's provider already used).
- `update_locked` — read-modify-write under an exclusive `flock`, so two ingest
  subprocesses splicing into the same shared file can't clobber each other.

All text I/O is utf-8 so nothing depends on the ambient locale (`LANG=C` safe).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

try:
    import fcntl  # POSIX only; absent on Windows.
except ImportError:  # pragma: no cover - not our platform
    fcntl = None  # type: ignore[assignment]


def write_text_atomic(path: Path | str, text: str) -> None:
    """Write `text` to `path` atomically (utf-8).

    Writes a sibling `{path}.tmp`, fsyncs it, then `os.replace`s it over the
    target. `os.replace` is atomic on POSIX, so a reader (or a crash) never sees
    a partially-written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path | str, obj) -> None:
    """Atomically write `obj` as indented JSON (utf-8, non-ASCII preserved).

    Consolidates the two private `_atomic_write_json` helpers that previously
    lived in `agents/relay.py` and `tasks/_ingest_batch.py`.
    """
    write_text_atomic(path, json.dumps(obj, indent=2, ensure_ascii=False))


def read_text(path: Path | str, default: str | None = None) -> str | None:
    """Read `path` as utf-8; return `default` if the file is missing."""
    path = Path(path)
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def file_sha256(path: Path | str) -> str:
    """Content hash of a file, read in chunks so a large PDF is not slurped.

    File *identity* where a path cannot provide it. Two callers need the same
    question answered — "is the file already sitting at the destination the same
    file I was about to copy there?" — and a name comparison cannot answer it:
    exporters and reference managers both produce names that carry no identity
    (`Nature-2026.pdf`), so same-name-different-paper is a real case.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path | str, default=None):
    """Read JSON from `path`, returning `default` on missing/corrupt/undecodable.

    A cache file truncated by an interrupted write (or bad bytes) is treated as a
    miss rather than raising — the caller re-fetches instead of crashing forever.
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return default


def _lock_path_for(path: Path) -> Path:
    """Lock-file path for `path`, in a host-temp dir keyed by the target's
    absolute path.

    Kept OUT of the repo tree: a sibling `{page}.lock` would litter the Obsidian
    vault and git status, and deleting such a lockfile after use is unsafe (the
    classic unlink race breaks mutual exclusion). All processes on the host
    resolve the same absolute target path, so they agree on the lock file.
    """
    lock_dir = Path(tempfile.gettempdir()) / "researchwiki-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    return lock_dir / f"{digest}.lock"


def update_locked(
    path: Path | str,
    mutate: Callable[[str], str],
    *,
    missing_ok: bool = True,
) -> bool:
    """Read-modify-write `path` under an exclusive cross-process lock.

    Acquires `flock(LOCK_EX)` on a sibling `{path}.lock`, reads the current text
    (utf-8; `""` when missing and `missing_ok`), calls `mutate(old) -> new`, and
    writes atomically iff `new != old`. Returns True if the file was written.

    `mutate` should return the current text unchanged to signal "no change"
    (e.g. an idempotent insert that finds the content already present) — that
    yields no write and a False return.

    Where `fcntl` is unavailable (Windows), degrades to atomic-write-only: still
    crash-safe, but not protected against a concurrent writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _apply() -> bool:
        if path.exists():
            old = path.read_text(encoding="utf-8")
        elif missing_ok:
            old = ""
        else:
            raise FileNotFoundError(path)
        new = mutate(old)
        if new is None or new == old:
            return False
        write_text_atomic(path, new)
        return True

    if fcntl is None:  # pragma: no cover - not our platform
        return _apply()

    lock_path = _lock_path_for(path)
    with open(lock_path, "w") as lock_fp:
        fcntl.flock(lock_fp, fcntl.LOCK_EX)
        try:
            return _apply()
        finally:
            fcntl.flock(lock_fp, fcntl.LOCK_UN)


@contextmanager
def exclusive_lock(path: Path | str):
    """Cross-process exclusive lock keyed by an arbitrary shared resource.

    Unlike :func:`update_locked`, this guards a multi-file operation. Page-level
    index updates rewrite Tantivy plus the aligned semantic ``.npy``/JSON pair,
    so no single target file can represent the whole critical section.
    """
    path = Path(path)
    if fcntl is None:  # pragma: no cover - Windows fallback
        yield
        return
    lock_path = _lock_path_for(path)
    with open(lock_path, "w") as lock_fp:
        fcntl.flock(lock_fp, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fp, fcntl.LOCK_UN)
