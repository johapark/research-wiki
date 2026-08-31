"""Atomic, utf-8, optionally cross-process-locked file writes.

Every page/index/cache writer in the package historically called `.write_text()`
directly: non-atomic (a crash mid-write truncates the file) and, for the shared
files edited by concurrent `agent ingest` subprocesses (`wiki/index.md`, back-link
target pages), racy. This module is the single primitive that fixes both.

- `write_text_atomic` / `write_json_atomic` — serialize writers per target, write
  to a unique sibling temp file, then `os.replace`. A crash leaves either the old
  file or the new one, never a truncated one; unique names keep cleanup local to
  the writer that created each temporary file.
- `read_json` — guarded read: a truncated/corrupt cache file reads as a miss
  instead of crashing every subsequent run (the pattern S2's provider already used).
- `update_locked` — read-modify-write under a cross-platform file lock, so two
  ingest subprocesses splicing into the same shared file can't clobber each other.

All text I/O is utf-8 so nothing depends on the ambient locale (`LANG=C` safe).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable


def _open_unique_sibling(path: Path) -> tuple[int, Path]:
    """Create a unique sibling using the caller's umask for its initial mode."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    for _ in range(100):
        candidate = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
        try:
            # 0666 is the normal text-file request; os.open applies the process
            # umask atomically, unlike chmodding mkstemp's 0600 inode afterward.
            return os.open(candidate, flags, 0o666), candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique temporary file for {path}")

def _write_text_atomic_unlocked(path: Path, text: str) -> None:
    """Write atomically while the caller holds this target's writer lock."""
    fd, tmp = _open_unique_sibling(path)
    try:
        # Preserve the current target's mode as close to replacement as
        # practical. A new target keeps the mode the kernel derived from umask.
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except FileNotFoundError:
            pass
        else:
            # `os.fchmod` is Unix-only; chmodding our unique path is portable.
            os.chmod(tmp, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def write_text_atomic(path: Path | str, text: str) -> None:
    """Write `text` to `path` atomically and serialize same-target writers.

    Writes a unique sibling temporary file, fsyncs it, then `os.replace`s it
    over the target. Keeping the temp in the same directory preserves atomic
    rename semantics; making it unique prevents cleanup from touching another
    writer's temporary path. The per-target lock is required on Windows, where
    concurrent `os.replace` calls can otherwise fail with ``WinError 5``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(_lock_path_for(path)):
        _write_text_atomic_unlocked(path, text)


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


def _file_lock(lock_path: Path):
    """Construct lazily so metadata-only CLI commands need no runtime extras."""
    from filelock import FileLock

    return FileLock(str(lock_path))


def update_locked(
    path: Path | str,
    mutate: Callable[[str], str],
    *,
    missing_ok: bool = True,
) -> bool:
    """Read-modify-write `path` under an exclusive cross-process lock.

    Acquires a cross-platform lock keyed by `path`, reads the current text
    (utf-8; `""` when missing and `missing_ok`), calls `mutate(old) -> new`, and
    writes atomically iff `new != old`. Returns True if the file was written.

    `mutate` should return the current text unchanged to signal "no change"
    (e.g. an idempotent insert that finds the content already present) — that
    yields no write and a False return.

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
        # The surrounding lock already serializes this target. Calling the
        # public writer here would acquire a second FileLock instance for the
        # same path, which is not reliably re-entrant across platforms.
        _write_text_atomic_unlocked(path, new)
        return True

    lock_path = _lock_path_for(path)
    with _file_lock(lock_path):
        return _apply()


@contextmanager
def exclusive_lock(path: Path | str):
    """Cross-process exclusive lock keyed by an arbitrary shared resource.

    Unlike :func:`update_locked`, this guards a multi-file operation. Page-level
    index updates rewrite Tantivy plus the aligned semantic ``.npy``/JSON pair,
    so no single target file can represent the whole critical section.
    """
    path = Path(path)
    lock_path = _lock_path_for(path)
    with _file_lock(lock_path):
        yield
