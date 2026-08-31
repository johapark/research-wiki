"""Tests for `researchwiki claim-graph promote` — propose / apply loop.

Covers:
  - propose_promotions drafts one proposal per (edge × matching synthesis)
  - proposals land under `.ingest/{stem}-claim-edges/{edge_id}.md`
  - confirmed edges with no synthesis referencing both endpoints are skipped
  - apply_promotions inserts the drafted bullet under the target section,
    creates the section if missing, refreshes generated_at, promotes the edge,
    and removes the proposal file
  - rejected proposals (deleted before --apply) leave synthesis untouched
  - alert hook prints ⚠ lines and returns a stats dict
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.claim_graph import (
    Edge, SLUG_SCHEME_VERSION, open_edges_db, set_status, upsert_edge,
)
from researchwiki.claim_graph.promote import (
    _find_target_syntheses,
    _insert_bullet_under_section,
    _CONTRADICTS_HEADINGS,
    apply_promotions,
    propose_promotions,
)


# ---------- section-insertion primitive ----------


def test_insert_under_existing_heading():
    text = (
        "# Title\n\n"
        "Intro paragraph.\n\n"
        "## Tensions / open questions\n\n"
        "- existing bullet\n\n"
        "## Next section\n\n"
        "more\n"
    )
    bullet = "- new bullet\n"
    out = _insert_bullet_under_section(
        text, _CONTRADICTS_HEADINGS, _CONTRADICTS_HEADINGS[0], bullet,
    )
    assert "- existing bullet" in out
    assert "- new bullet" in out
    # Order: existing bullet first, new one after.
    assert out.index("existing bullet") < out.index("new bullet")


def test_insert_matches_heading_variant():
    """Accepts `## Tensions` even when the canonical form is `## Tensions / open questions`."""
    text = "## Tensions\n\n- old\n"
    out = _insert_bullet_under_section(
        text, _CONTRADICTS_HEADINGS, _CONTRADICTS_HEADINGS[0], "- new\n",
    )
    assert "## Tensions\n" in out
    assert "- old" in out and "- new" in out


def test_insert_appends_section_when_missing():
    text = "# Title\n\nSome prose.\n\n## Evidence\n\n- old\n"
    out = _insert_bullet_under_section(
        text, _CONTRADICTS_HEADINGS, _CONTRADICTS_HEADINGS[0], "- new\n",
    )
    # Should have added the Tensions heading, kept existing Evidence intact.
    assert "## Tensions / open questions" in out
    assert "- old" in out
    assert "- new" in out


# ---------- target-synthesis discovery ----------


def test_find_target_synthesis_matches_when_both_stems_referenced():
    index = [
        {"stem": "syn-a", "page_type": "synthesis", "path": Path("wiki/synthesis/syn-a.md"),
         "referenced_stems": {"paper-x", "paper-y", "paper-z"}},
        {"stem": "syn-b", "page_type": "synthesis", "path": Path("wiki/synthesis/syn-b.md"),
         "referenced_stems": {"paper-x", "paper-w"}},  # missing paper-y
    ]
    got = _find_target_syntheses(index, "paper-x", "paper-y")
    stems = [s["stem"] for s in got]
    assert stems == ["syn-a"]


def test_synthesis_wins_over_idea_when_tied():
    index = [
        {"stem": "idea-a", "page_type": "idea", "path": Path("wiki/ideas/idea-a.md"),
         "referenced_stems": {"paper-x", "paper-y"}},
        {"stem": "syn-a", "page_type": "synthesis", "path": Path("wiki/synthesis/syn-a.md"),
         "referenced_stems": {"paper-x", "paper-y"}},
    ]
    got = _find_target_syntheses(index, "paper-x", "paper-y")
    assert got[0]["stem"] == "syn-a"


# ---------- propose_promotions (integration; stubs state.db + edges.db) ----------


@pytest.fixture
def isolated_wiki(tmp_path, monkeypatch):
    """Redirect wiki_dir, ingest_dir, and both DBs into tmp_path."""
    root = tmp_path
    wiki = root / "wiki"
    (wiki / "synthesis").mkdir(parents=True)
    ingest = root / ".ingest"
    ingest.mkdir()
    monkeypatch.chdir(root)
    yield root


def _mk_paper_row(stem, page_type, referenced_stems=None, path_str=None):
    """Row shape for the papers table (subset used by promote)."""
    import json as _json
    fm = {"referenced_papers": [f"[[{s}]]" for s in (referenced_stems or [])]}
    return {
        "stem": stem, "page_type": page_type, "path_str": path_str or f"wiki/{page_type}/{stem}.md",
        "raw_frontmatter": _json.dumps(fm),
    }


def _seed_state_db(monkeypatch, papers=None, claims=None):
    """Stand up a minimal state.db shape so promote's SQL runs.

    Uses a real sqlite3.Connection so promote's own queries hit real SQL rather
    than a mock. papers: list of dicts from _mk_paper_row. claims: list of
    (paper_stem, claim_slug, text) tuples.
    """
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE papers (
            stem TEXT PRIMARY KEY,
            page_type TEXT,
            page_path TEXT,
            raw_frontmatter TEXT
        );
        CREATE TABLE claims (
            paper_stem TEXT,
            claim_slug TEXT,
            text TEXT,
            UNIQUE(paper_stem, claim_slug)
        );
    """)
    for p in (papers or []):
        conn.execute("INSERT INTO papers VALUES (?, ?, ?, ?)",
                     (p["stem"], p["page_type"], p["path_str"], p["raw_frontmatter"]))
    for stem, slug, text in (claims or []):
        conn.execute("INSERT INTO claims VALUES (?, ?, ?)", (stem, slug, text))
    conn.commit()

    def fake_get_connection():
        return conn

    monkeypatch.setattr("researchwiki.db.connection.get_connection", fake_get_connection)
    return conn


def _seed_edge(edges_conn, src_stem, src_slug, tgt_stem, tgt_slug, relation, status="confirmed"):
    _, eid = upsert_edge(edges_conn, Edge(
        src_stem=src_stem, src_slug=src_slug,
        tgt_stem=tgt_stem, tgt_slug=tgt_slug,
        relation=relation, directed=False, confidence=0.9,
        rationale=f"synthetic {relation}",
        slug_scheme_version=SLUG_SCHEME_VERSION,
        status=status,
    ))
    edges_conn.commit()
    return eid


def test_propose_drafts_one_proposal_per_confirmed_edge(isolated_wiki, monkeypatch):
    # Two graded papers with one claim each; one confirmed contradicts edge.
    synth_path = isolated_wiki / "wiki" / "synthesis" / "combo.md"
    synth_path.write_text(
        "---\ntitle: combo\ntype: synthesis\ncategory: [ai]\n"
        "referenced_papers:\n  - [[paper-a]]\n  - [[paper-b]]\n"
        "generated_at: 2020-01-01\ntopic_seed: x\n---\n\n"
        "## Approaches\n\n- overview\n"
    )
    _seed_state_db(
        monkeypatch,
        papers=[_mk_paper_row("combo", "synthesis",
                              referenced_stems=["paper-a", "paper-b"],
                              path_str=str(synth_path))],
        claims=[
            ("paper-a", "kc-aaaa1111", "A claims X = 40%."),
            ("paper-b", "kc-bbbb2222", "B claims X = 60%."),
        ],
    )
    edges = open_edges_db()
    try:
        eid = _seed_edge(edges, "paper-a", "kc-aaaa1111", "paper-b", "kc-bbbb2222",
                         "contradicts")
    finally:
        edges.close()

    stats = propose_promotions()
    assert stats.proposals_written == 1
    assert stats.edges_no_target == 0

    proposal = isolated_wiki / ".ingest" / "combo-claim-edges" / f"{eid}.md"
    assert proposal.exists()
    text = proposal.read_text()
    # Frontmatter carries the edge + endpoints.
    assert f"edge_id: {eid}" in text
    assert "relation: contradicts" in text
    assert "src: paper-a#kc-aaaa1111" in text
    assert "tgt: paper-b#kc-bbbb2222" in text
    # Drafted bullet cites both sides.
    assert "[[paper-a#kc-aaaa1111]]" in text
    assert "[[paper-b#kc-bbbb2222]]" in text


def test_propose_skips_edge_with_no_matching_synthesis(isolated_wiki, monkeypatch):
    # Synthesis only references paper-a, not paper-b — edge has no target.
    synth_path = isolated_wiki / "wiki" / "synthesis" / "solo.md"
    synth_path.write_text("---\ntitle: solo\ntype: synthesis\ncategory: [ai]\n"
                          "referenced_papers:\n  - [[paper-a]]\ngenerated_at: 2020-01-01\n"
                          "topic_seed: x\n---\n")
    _seed_state_db(
        monkeypatch,
        papers=[_mk_paper_row("solo", "synthesis", referenced_stems=["paper-a"],
                              path_str=str(synth_path))],
        claims=[("paper-a", "kc-aaaa1111", "A"), ("paper-b", "kc-bbbb2222", "B")],
    )
    edges = open_edges_db()
    try:
        _seed_edge(edges, "paper-a", "kc-aaaa1111", "paper-b", "kc-bbbb2222", "contradicts")
    finally:
        edges.close()

    stats = propose_promotions()
    assert stats.proposals_written == 0
    assert stats.edges_no_target == 1


# ---------- apply_promotions ----------


def test_apply_inserts_bullet_and_promotes_edge(isolated_wiki, monkeypatch):
    synth_path = isolated_wiki / "wiki" / "synthesis" / "combo.md"
    synth_path.write_text(
        "---\ntitle: combo\ntype: synthesis\ncategory: [ai]\n"
        "referenced_papers:\n  - [[paper-a]]\n  - [[paper-b]]\n"
        "generated_at: 2020-01-01\ntopic_seed: x\n---\n\n"
        "## Approaches\n\n- overview\n"
    )
    _seed_state_db(
        monkeypatch,
        papers=[_mk_paper_row("combo", "synthesis",
                              referenced_stems=["paper-a", "paper-b"],
                              path_str=str(synth_path))],
        claims=[
            ("paper-a", "kc-aaaa1111", "A claims X = 40%."),
            ("paper-b", "kc-bbbb2222", "B claims X = 60%."),
        ],
    )
    edges = open_edges_db()
    try:
        eid = _seed_edge(edges, "paper-a", "kc-aaaa1111", "paper-b", "kc-bbbb2222",
                         "contradicts")
    finally:
        edges.close()

    propose_promotions()

    # Apply: bullet inserted, edge promoted, proposal removed, generated_at refreshed.
    stats = apply_promotions()
    assert stats.applied == 1
    assert stats.skipped_stale == 0

    updated = synth_path.read_text()
    assert "## Tensions / open questions" in updated
    assert "[[paper-a#kc-aaaa1111]]" in updated
    assert "[[paper-b#kc-bbbb2222]]" in updated
    assert "generated_at: 2020-01-01" not in updated  # bumped
    # Proposal file gone.
    proposal = isolated_wiki / ".ingest" / "combo-claim-edges" / f"{eid}.md"
    assert not proposal.exists()
    # Edge status promoted.
    edges2 = open_edges_db()
    try:
        row = edges2.execute("SELECT status FROM edges WHERE id=?", (eid,)).fetchone()
    finally:
        edges2.close()
    assert row["status"] == "promoted"


def test_apply_commits_target_page_to_db(isolated_wiki, monkeypatch):
    """Applying a promotion rewrites `generated_at:`, so the DB row must be
    reconciled at write time rather than left for the next `db rebuild`.

    Spied rather than asserted through `db verify`, because this fixture seeds
    `papers` synthetically instead of walking the wiki — the call is what's
    under test.
    """
    from researchwiki.claim_graph import promote as promote_mod

    synth_path = isolated_wiki / "wiki" / "synthesis" / "combo.md"
    synth_path.write_text(
        "---\ntitle: combo\ntype: synthesis\ncategory: [ai]\n"
        "referenced_papers:\n  - [[paper-a]]\n  - [[paper-b]]\n"
        "generated_at: 2020-01-01\ntopic_seed: x\n---\n\n"
        "## Approaches\n\n- overview\n"
    )
    _seed_state_db(
        monkeypatch,
        papers=[_mk_paper_row("combo", "synthesis",
                              referenced_stems=["paper-a", "paper-b"],
                              path_str=str(synth_path))],
        claims=[
            ("paper-a", "kc-aaaa1111", "A claims X = 40%."),
            ("paper-b", "kc-bbbb2222", "B claims X = 60%."),
        ],
    )
    edges = open_edges_db()
    try:
        _seed_edge(edges, "paper-a", "kc-aaaa1111", "paper-b", "kc-bbbb2222",
                   "contradicts")
    finally:
        edges.close()

    propose_promotions()

    committed: list = []
    monkeypatch.setattr(promote_mod, "commit_page", committed.append)

    stats = apply_promotions()

    assert stats.applied == 1
    assert committed == [synth_path]


def test_apply_leaves_synthesis_untouched_when_proposal_deleted(isolated_wiki, monkeypatch):
    """User rejects a proposal by deleting the file before --apply."""
    synth_path = isolated_wiki / "wiki" / "synthesis" / "combo.md"
    synth_path.write_text(
        "---\ntitle: combo\ntype: synthesis\ncategory: [ai]\n"
        "referenced_papers:\n  - [[paper-a]]\n  - [[paper-b]]\n"
        "generated_at: 2020-01-01\ntopic_seed: x\n---\n\n## Approaches\n\n- overview\n"
    )
    _seed_state_db(
        monkeypatch,
        papers=[_mk_paper_row("combo", "synthesis",
                              referenced_stems=["paper-a", "paper-b"],
                              path_str=str(synth_path))],
        claims=[("paper-a", "kc-a", "A"), ("paper-b", "kc-b", "B")],
    )
    edges = open_edges_db()
    try:
        _seed_edge(edges, "paper-a", "kc-a", "paper-b", "kc-b", "contradicts")
    finally:
        edges.close()

    propose_promotions()
    # Simulate reject: user deletes every proposal file.
    for p in (isolated_wiki / ".ingest").glob("*-claim-edges/*.md"):
        p.unlink()
    before = synth_path.read_text()

    stats = apply_promotions()
    assert stats.proposals_seen == 0
    assert synth_path.read_text() == before  # untouched


def test_apply_skips_when_edge_no_longer_confirmed(isolated_wiki, monkeypatch):
    """After proposal is drafted, human rejects the edge in `review` (→ status=rejected).
    A subsequent --apply should NOT apply the drafted proposal."""
    synth_path = isolated_wiki / "wiki" / "synthesis" / "combo.md"
    synth_path.write_text(
        "---\ntitle: combo\ntype: synthesis\ncategory: [ai]\n"
        "referenced_papers:\n  - [[paper-a]]\n  - [[paper-b]]\n"
        "generated_at: 2020-01-01\ntopic_seed: x\n---\n"
    )
    _seed_state_db(
        monkeypatch,
        papers=[_mk_paper_row("combo", "synthesis",
                              referenced_stems=["paper-a", "paper-b"],
                              path_str=str(synth_path))],
        claims=[("paper-a", "kc-a", "A"), ("paper-b", "kc-b", "B")],
    )
    edges = open_edges_db()
    try:
        eid = _seed_edge(edges, "paper-a", "kc-a", "paper-b", "kc-b", "contradicts")
    finally:
        edges.close()

    propose_promotions()
    # Flip the edge to `rejected` between propose and apply.
    edges = open_edges_db()
    try:
        set_status(edges, eid, "rejected")
        edges.commit()
    finally:
        edges.close()

    before = synth_path.read_text()
    stats = apply_promotions()
    assert stats.applied == 0
    assert stats.skipped_stale == 1
    assert synth_path.read_text() == before
