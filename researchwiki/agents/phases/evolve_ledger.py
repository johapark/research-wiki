"""Judged-pair ledger for memory_evolve — an idempotency cache.

Backing file: `.evolve-cache/judged.db` (gitignored; safe to delete — pairs are
re-judged on the next run). It is a *separate* SQLite file, NOT `state.db`,
because it stores LLM verdicts, which violate `state.db`'s deterministic /
rebuildable invariant — the same reason `.claim-graph/edges.db` is separate.

**Only "none" verdicts are cached, deliberately.** Rationale:

  - "none" is the overwhelming majority of judgments (~87% in dogfooding), and
    skipping a cached "none" on a re-run is *always correct*: source paper pages
    are immutable per Rule 3, and the cache entry is keyed on content hashes of
    both the source and target pages — if either changed, it's a cache miss and
    the pair is re-judged.
  - Actionable verdicts (refine / enhance / contrast) are rare and are NOT
    cached. They are re-judged on a re-run so a fresh, reviewable proposal is
    re-emitted if the previous one wasn't applied yet. (Once applied, the target
    page changes → cache miss → re-judge → the new verdict is "none, already
    covered", which then caches.)

This makes the ledger a pure cost-saver for re-runs (re-ingest, batch
`--resume`, a future `evolve --sweep`) with no risk of suppressing a real edit.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from ...paths import evolve_cache_dir

_SCHEMA = """
CREATE TABLE IF NOT EXISTS judged_none (
    source_stem TEXT NOT NULL,
    target_key  TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    target_hash TEXT NOT NULL,
    judged_at   INTEGER NOT NULL,
    PRIMARY KEY (source_stem, target_key)
);
"""


def page_hash(text: str) -> str:
    """Content hash of a page body — the cache-invalidation key."""
    return hashlib.blake2s(text.encode("utf-8"), digest_size=16).hexdigest()


def ledger_path() -> Path:
    d = evolve_cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "judged.db"


def open_ledger(path: Path | None = None) -> sqlite3.Connection:
    """Open the ledger, initializing the schema idempotently."""
    conn = sqlite3.connect(str(path if path is not None else ledger_path()))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def is_cached_none(conn: sqlite3.Connection, source_stem: str, target_key: str,
                   source_hash: str, target_hash: str) -> bool:
    """True iff this exact (source, target) pair was judged "none" at these
    content hashes — i.e. re-judging would be a wasted, unchanged "none"."""
    row = conn.execute(
        "SELECT 1 FROM judged_none WHERE source_stem=? AND target_key=? "
        "AND source_hash=? AND target_hash=?",
        (source_stem, target_key, source_hash, target_hash),
    ).fetchone()
    return row is not None


def record_none(conn: sqlite3.Connection, source_stem: str, target_key: str,
                source_hash: str, target_hash: str, *, now: int | None = None) -> None:
    """Cache a fresh "none" verdict (upsert — refreshes hashes/timestamp)."""
    conn.execute(
        "INSERT OR REPLACE INTO judged_none "
        "(source_stem, target_key, source_hash, target_hash, judged_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (source_stem, target_key, source_hash, target_hash,
         now if now is not None else int(time.time())),
    )
    conn.commit()


def clear(conn: sqlite3.Connection, source_stem: str, target_key: str) -> None:
    """Drop any cached "none" for a pair — called when the verdict turns
    actionable, so a stale "none" can never shadow a real proposal."""
    conn.execute(
        "DELETE FROM judged_none WHERE source_stem=? AND target_key=?",
        (source_stem, target_key),
    )
    conn.commit()
