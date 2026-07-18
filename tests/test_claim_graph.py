"""Tests for the claim-graph foundation — slugs + edge cache + reconcile.

Covers:
  - slug determinism
  - collision disambiguation
  - normalize spec (frozen v1)
  - edge upsert + query
  - status lifecycle (candidate → confirmed / rejected)
  - rejected-edge idempotence (skip_if_rejected)
  - reconcile marks edges stale on endpoint drift + version drift
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from researchwiki.claim_graph import (
    SLUG_SCHEME_VERSION,
    Edge,
    compute_claim_slug,
    disambiguate_slug,
    normalize_claim_text,
    open_edges_db,
    query,
    reconcile,
    set_status,
    upsert_edge,
)


# ---------- slug determinism + normalize spec ----------


def test_compute_claim_slug_is_deterministic():
    a = compute_claim_slug("results", "The model achieves 92.3% accuracy on task X.")
    b = compute_claim_slug("results", "The model achieves 92.3% accuracy on task X.")
    assert a == b
    assert a.startswith("res-")
    # 8 hex chars after the "res-" prefix.
    assert len(a) == len("res-") + 8


def test_slug_prefix_from_known_sections():
    assert compute_claim_slug("key_contributions", "X").startswith("kc-")
    assert compute_claim_slug("results", "X").startswith("res-")
    assert compute_claim_slug("limitations", "X").startswith("lim-")
    assert compute_claim_slug("methodology", "X").startswith("met-")


def test_slug_prefix_fallback_for_unknown_section():
    # First three chars of the section name, lowercased alnum only.
    slug = compute_claim_slug("data_availability", "some text")
    assert slug.startswith("dat-")


def test_normalize_strips_wikilinks_and_footnotes():
    got = normalize_claim_text("A finding [[foo-2024]] with a note[^ref].")
    assert "[[" not in got
    assert "[^" not in got
    # Whitespace collapses; trailing punctuation drops.
    assert got == "a finding with a note"


def test_normalize_collapses_whitespace_and_trims_punct():
    # Leading whitespace + surrounding quote stripped; internal whitespace collapsed;
    # trailing `.` dropped. Internal quotes are NOT stripped — they'd change meaning.
    got = normalize_claim_text('  "A claim with spaces."  ')
    assert got == "a claim with spaces"


def test_normalize_ignores_markdown_emphasis():
    got = normalize_claim_text("**Prime** editing *reduces* off-target `effects`.")
    assert "*" not in got and "`" not in got
    assert got == "prime editing reduces off-target effects"


def test_disambiguate_appends_position():
    assert disambiguate_slug("kc-abcd1234", 3) == "kc-abcd1234-3"


# ---------- edge cache lifecycle ----------


@pytest.fixture
def edges_conn(tmp_path):
    conn = open_edges_db(tmp_path / "edges.db")
    yield conn
    conn.close()


def _make_edge(src="paper-a", src_slug="kc-aaaa1111",
               tgt="paper-b", tgt_slug="kc-bbbb2222",
               relation="contradicts", status="candidate") -> Edge:
    return Edge(
        src_stem=src, src_slug=src_slug, tgt_stem=tgt, tgt_slug=tgt_slug,
        relation=relation, slug_scheme_version=SLUG_SCHEME_VERSION,
        confidence=0.9, rationale="stub", judge_phase="test", judge_model="test-1",
        status=status,
    )


def test_upsert_edge_inserts_then_updates(edges_conn):
    kind, id1 = upsert_edge(edges_conn, _make_edge())
    assert kind == "inserted"
    assert id1 > 0
    # Second call with the same identity → update, not a new row.
    e = _make_edge()
    e.rationale = "refined"
    kind, id2 = upsert_edge(edges_conn, e)
    assert kind == "updated"
    assert id2 == id1
    rows = query(edges_conn, relation="contradicts")
    assert len(rows) == 1
    assert rows[0].rationale == "refined"


def test_upsert_rejects_unknown_relation(edges_conn):
    e = _make_edge()
    e.relation = "not-a-relation"
    with pytest.raises(ValueError):
        upsert_edge(edges_conn, e)


def test_upsert_skips_when_rejected(edges_conn):
    kind, edge_id = upsert_edge(edges_conn, _make_edge())
    assert kind == "inserted"
    # Human rejects — durably.
    set_status(edges_conn, edge_id, "rejected")
    # A fresh judge run re-proposes the same pair; it should be skipped.
    kind, id2 = upsert_edge(edges_conn, _make_edge(status="candidate"))
    assert kind == "skipped"
    assert id2 == edge_id
    rows = query(edges_conn, status="rejected")
    assert len(rows) == 1


def test_query_by_stem_matches_either_endpoint(edges_conn):
    upsert_edge(edges_conn, _make_edge(src="paper-a", tgt="paper-b"))
    upsert_edge(edges_conn, _make_edge(
        src="paper-c", src_slug="kc-ccc11111",
        tgt="paper-a", tgt_slug="kc-aaaa1111",
    ))
    hits = query(edges_conn, stem="paper-a")
    assert len(hits) == 2


# ---------- reconcile against a stub state.db ----------


def _stub_state_db(tmp_path: Path, *, slug_version: int, claims: list[tuple[str, str]],
                   concept_stems: list[str] | None = None) -> sqlite3.Connection:
    """Build a state.db-compatible sqlite with just the shape reconcile() needs."""
    p = tmp_path / "state.db"
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE papers (stem TEXT PRIMARY KEY, page_type TEXT);
        CREATE TABLE claims (
            paper_stem TEXT, claim_slug TEXT,
            UNIQUE(paper_stem, claim_slug)
        );
    """)
    conn.execute("INSERT INTO schema_meta VALUES ('slug_scheme_version', ?)",
                 (str(slug_version),))
    for stem, slug in claims:
        conn.execute("INSERT INTO claims VALUES (?, ?)", (stem, slug))
    for stem in concept_stems or []:
        conn.execute("INSERT INTO papers VALUES (?, 'concept')", (stem,))
    conn.commit()
    return conn


def test_reconcile_marks_stale_when_slug_missing(edges_conn, tmp_path):
    upsert_edge(edges_conn, _make_edge(
        src="paper-a", src_slug="kc-aaaa1111",
        tgt="paper-b", tgt_slug="kc-bbbb2222",
    ))
    # state.db has only paper-a's claim — paper-b's is gone.
    state = _stub_state_db(
        tmp_path, slug_version=SLUG_SCHEME_VERSION,
        claims=[("paper-a", "kc-aaaa1111")],
    )
    try:
        stats = reconcile(edges_conn, state)
    finally:
        state.close()
    assert stats.marked_stale_missing == 1
    # Row is now stale.
    stale = query(edges_conn, status="stale")
    assert len(stale) == 1


def test_reconcile_marks_stale_on_version_drift(edges_conn, tmp_path):
    upsert_edge(edges_conn, _make_edge(
        src="paper-a", src_slug="kc-aaaa1111",
        tgt="paper-b", tgt_slug="kc-bbbb2222",
    ))
    # Both slugs present in state.db, but under a newer version.
    state = _stub_state_db(
        tmp_path, slug_version=SLUG_SCHEME_VERSION + 1,
        claims=[("paper-a", "kc-aaaa1111"), ("paper-b", "kc-bbbb2222")],
    )
    try:
        stats = reconcile(edges_conn, state)
    finally:
        state.close()
    assert stats.marked_stale_version == 1


def test_reconcile_leaves_healthy_edges_active(edges_conn, tmp_path):
    upsert_edge(edges_conn, _make_edge())
    state = _stub_state_db(
        tmp_path, slug_version=SLUG_SCHEME_VERSION,
        claims=[("paper-a", "kc-aaaa1111"), ("paper-b", "kc-bbbb2222")],
    )
    try:
        stats = reconcile(edges_conn, state)
    finally:
        state.close()
    assert stats.marked_stale_missing == 0
    assert stats.marked_stale_version == 0
    assert stats.active == 1


def test_reconcile_resolves_instantiates_against_concepts_table(edges_conn, tmp_path):
    # instantiates: (claim → concept-page-slug). Target lives in papers, not claims.
    upsert_edge(edges_conn, Edge(
        src_stem="paper-a", src_slug="kc-aaaa1111",
        tgt_stem="concepts", tgt_slug="prime-editing",
        relation="instantiates",
        slug_scheme_version=SLUG_SCHEME_VERSION,
    ))
    state = _stub_state_db(
        tmp_path, slug_version=SLUG_SCHEME_VERSION,
        claims=[("paper-a", "kc-aaaa1111")],
        concept_stems=["prime-editing"],
    )
    try:
        stats = reconcile(edges_conn, state)
    finally:
        state.close()
    assert stats.marked_stale_missing == 0
    assert stats.active == 1


def test_reconcile_marks_instantiates_stale_when_concept_gone(edges_conn, tmp_path):
    upsert_edge(edges_conn, Edge(
        src_stem="paper-a", src_slug="kc-aaaa1111",
        tgt_stem="concepts", tgt_slug="prime-editing",
        relation="instantiates",
        slug_scheme_version=SLUG_SCHEME_VERSION,
    ))
    state = _stub_state_db(
        tmp_path, slug_version=SLUG_SCHEME_VERSION,
        claims=[("paper-a", "kc-aaaa1111")],
        concept_stems=[],  # concept page deleted.
    )
    try:
        stats = reconcile(edges_conn, state)
    finally:
        state.close()
    assert stats.marked_stale_missing == 1
