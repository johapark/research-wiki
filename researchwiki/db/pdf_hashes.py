"""Persistent PDF content hashes for read-path freshness checks.

Claim retrieval compares each claim's recorded `source_pdf_sha256` with its
PDF. Hashing every PDF on every query reads the whole of `papers/` — gigabytes,
and on a synced folder with cloud-only files a download of each PDF — so read
paths take the hash from `pdf_hashes` while the file's size, mtime and inode
still match what was recorded.

**Read paths only.** A stat match is strong evidence, not proof: a tool that
rewrites a file in place and restores its mtime would get the old hash back.
Grade persistence, PDF chunk-index builds, and rebuild's grade invalidation
therefore keep calling `fsatomic.file_sha256` on the bytes. The cost of a wrong
cache hit here is bounded by that: a stale grade could be *shown* as current,
but never written against the wrong PDF.

A cache problem never breaks the query: an unavailable or locked database
falls back to hashing the file directly.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from ..fsatomic import file_sha256


class PdfHashCache:
    """Hash lookups bound to one open state-DB connection.

    One instance per query batch: lookups read through the caller's
    connection, and misses are written in one transaction by `flush()`.
    """

    def __init__(self, conn: sqlite3.Connection | None):
        self._conn = conn
        self._pending: list[tuple[str, int, int, int, str, int]] = []

    def sha256(self, path: Path | str) -> str:
        """Content hash of `path`, from the cache when the file is unchanged."""
        resolved = Path(path).resolve()
        info = resolved.stat()
        stamp = (info.st_size, info.st_mtime_ns, info.st_ino)
        key = str(resolved)
        row = self._lookup(key)
        if row is not None and tuple(row[:3]) == stamp:
            return row[3]
        digest = file_sha256(resolved)
        # Re-stat: a file replaced while it was being read must not be cached
        # under the newer stat with the older bytes' hash.
        after = resolved.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ino) == stamp:
            self._pending.append((key, *stamp, digest, int(time.time())))
        return digest

    def flush(self) -> None:
        """Persist this batch's misses. Best-effort; a failure only costs a re-hash."""
        if not self._pending or self._conn is None:
            self._pending.clear()
            return
        try:
            with self._conn:
                self._conn.executemany(
                    "INSERT INTO pdf_hashes (path, size, mtime_ns, inode, sha256, hashed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(path) DO UPDATE SET size = excluded.size, "
                    "mtime_ns = excluded.mtime_ns, inode = excluded.inode, "
                    "sha256 = excluded.sha256, hashed_at = excluded.hashed_at",
                    self._pending,
                )
        except sqlite3.Error:
            pass
        finally:
            self._pending.clear()

    def _lookup(self, key: str):
        if self._conn is None:
            return None
        try:
            return self._conn.execute(
                "SELECT size, mtime_ns, inode, sha256 FROM pdf_hashes WHERE path = ?",
                (key,),
            ).fetchone()
        except sqlite3.Error:
            # An older DB opened without schema init, or a locked read.
            return None
