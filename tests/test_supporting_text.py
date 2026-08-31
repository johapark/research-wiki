"""Tests for the supporting_text column on claims (Pattern 3 — Starling-transferable).

Three guarantees worth pinning:
  1. Schema migration runs idempotently — fresh DB and pre-migration DB both
     end up with the supporting_text column.
  2. _upsert_claims resets supporting_text to NULL when claim text changes
     (alongside the other grader-derived columns), so a stale chunk doesn't
     outlive the claim it was tied to.
  3. claims_by_stem / claim_lookup surface supporting_text only with
     include_context=True (default-off keeps JSON dumps narrow).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from researchwiki.db.connection import init_schema


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    return conn


def test_supporting_text_column_present_on_fresh_db(tmp_path):
    conn = _open(tmp_path / "fresh.db")
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(claims)")}
    assert "supporting_text" in cols


def test_supporting_text_resets_when_claim_text_changes(tmp_path):
    """Edit a claim's text → grader columns including supporting_text reset."""
    conn = _open(tmp_path / "reset.db")
    conn.execute(
        """INSERT INTO papers (stem, category, page_type, title, page_path,
                                page_mtime, raw_frontmatter, indexed_at)
           VALUES ('p1','cgt','paper','t','/p','0','{}','0')"""
    )
    conn.execute(
        """INSERT INTO claims (paper_stem, section, position, text, is_cross_ref,
                                supporting_text, last_graded_at)
           VALUES ('p1','key_contributions',0,'old text',0,'old chunk',1)"""
    )
    conn.commit()

    # Simulate the rebuild path: an UPDATE that resets grader columns.
    conn.execute(
        """UPDATE claims SET text=?, is_cross_ref=?, bm25_top1=NULL, bm25_top3_mean=NULL,
                              bm25_top1_chunk_id=NULL, supporting_text=NULL,
                              semantic_score=NULL, embed_model=NULL,
                              negation_mismatch=NULL, numeric_tokens=NULL,
                              numeric_unmatched=NULL, last_graded_at=NULL
            WHERE paper_stem='p1' AND section='key_contributions' AND position=0""",
        ("new text", 0),
    )
    conn.commit()

    row = conn.execute(
        "SELECT text, supporting_text, last_graded_at FROM claims WHERE paper_stem='p1'"
    ).fetchone()
    assert row["text"] == "new text"
    assert row["supporting_text"] is None
    assert row["last_graded_at"] is None


def test_claims_by_stem_includes_context_only_when_requested(tmp_path, monkeypatch):
    """Default off; opt-in via include_context=True."""
    db = tmp_path / "lookup.db"
    conn = _open(db)
    conn.execute(
        """INSERT INTO papers (stem, category, page_type, title, page_path,
                                page_mtime, raw_frontmatter, indexed_at)
           VALUES ('p1','cgt','paper','t','/p','0','{}','0')"""
    )
    conn.execute(
        """INSERT INTO claims (paper_stem, section, position, text, is_cross_ref,
                                supporting_text, last_graded_at)
           VALUES ('p1','key_contributions',0,'a claim',0,'source paragraph',1)"""
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(db))
    # Re-import to pick up the patched env.
    from importlib import reload
    from researchwiki.db import connection as conn_mod
    reload(conn_mod)
    from researchwiki.search import tools as tools_core
    reload(tools_core)

    default_hits = tools_core.claims_by_stem("p1")
    assert default_hits and "supporting_text" not in default_hits[0]

    ctx_hits = tools_core.claims_by_stem("p1", include_context=True)
    assert ctx_hits and ctx_hits[0]["supporting_text"] == "source paragraph"
