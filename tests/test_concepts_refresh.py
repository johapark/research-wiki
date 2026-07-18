"""Tests for `researchwiki concepts refresh <slug>` — draft Cross-domain
connections from typed edges among a hub's member claims.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.claim_graph import (
    Edge, SLUG_SCHEME_VERSION, open_edges_db, upsert_edge,
)
from researchwiki.concepts import refresh as concepts_mod


@pytest.fixture
def isolated_wiki(tmp_path, monkeypatch):
    """Redirect wiki_dir + ingest_dir into tmp_path, seed a hub file."""
    root = tmp_path
    wiki = root / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "cgt").mkdir()
    (wiki / "compbio").mkdir()
    (root / ".ingest").mkdir()

    monkeypatch.chdir(root)
    monkeypatch.setattr("researchwiki.concepts.refresh.wiki_dir", lambda: wiki)

    (wiki / "concepts" / "prime-editing.md").write_text(
        '---\ntitle: "prime editing"\ntype: concept\ncategory: [cgt]\n'
        "referenced_papers:\n  - [[cgt/paper-a]]\n  - [[compbio/paper-b]]\n"
        'concept_span: 2\ngenerated_at: 2020-01-01\ntopic_seed: "prime editing"\n'
        "tags: [concept, prime-editing]\n---\n\n## Definition\nA thing.\n"
    )
    return root, wiki


def _stub_stem_categories(monkeypatch, mapping: dict[str, str]):
    monkeypatch.setattr(concepts_mod, "_resolve_stem_categories",
                        lambda stems: {s: mapping.get(s, "") for s in stems})


def _seed_edges(cases: list[tuple[str, str, str, str, str, bool]]):
    """cases: [(src_stem, src_slug, tgt_stem, tgt_slug, relation, directed)]."""
    conn = open_edges_db()
    try:
        for src_stem, src_slug, tgt_stem, tgt_slug, rel, directed in cases:
            upsert_edge(conn, Edge(
                src_stem=src_stem, src_slug=src_slug,
                tgt_stem=tgt_stem, tgt_slug=tgt_slug,
                relation=rel, directed=directed,
                confidence=0.9, rationale=f"{rel} rationale",
                slug_scheme_version=SLUG_SCHEME_VERSION, status="candidate",
            ))
        conn.commit()
    finally:
        conn.close()


def test_refresh_returns_zero_when_no_member_claims(isolated_wiki, monkeypatch):
    _, wiki = isolated_wiki
    _stub_stem_categories(monkeypatch, {})
    result = concepts_mod.refresh_concept("prime-editing")
    assert result["n_member_claims"] == 0
    assert result["draft_path"] is None


def test_refresh_returns_zero_when_no_cross_category_bridges(isolated_wiki, monkeypatch):
    """Two member claims, both in the same category → no bridge to draft."""
    _stub_stem_categories(monkeypatch, {"paper-a": "cgt", "paper-b": "cgt"})
    _seed_edges([
        # instantiates: paper-a and paper-b both point at prime-editing hub
        ("paper-a", "kc-a1", "concepts", "prime-editing", "instantiates", True),
        ("paper-b", "kc-b1", "concepts", "prime-editing", "instantiates", True),
        # A typed edge between them, but same-category → not a bridge
        ("paper-a", "kc-a1", "paper-b", "kc-b1", "corroborates", False),
    ])
    result = concepts_mod.refresh_concept("prime-editing")
    assert result["n_member_claims"] == 2
    assert result["n_bridges_found"] == 0
    assert result["draft_path"] is None


def test_refresh_drafts_cross_domain_block(isolated_wiki, monkeypatch):
    """One typed edge across two categories → drafts a Cross-domain
    connections block citing both endpoints via [[stem#slug]]."""
    root, wiki = isolated_wiki
    _stub_stem_categories(monkeypatch, {
        "paper-a": "cgt", "paper-b": "compbio",
    })
    _seed_edges([
        ("paper-a", "kc-a1", "concepts", "prime-editing", "instantiates", True),
        ("paper-b", "kc-b1", "concepts", "prime-editing", "instantiates", True),
        ("paper-a", "kc-a1", "paper-b", "kc-b1", "corroborates", False),
    ])
    result = concepts_mod.refresh_concept("prime-editing")
    assert result["n_bridges_found"] == 1
    assert result["draft_path"] is not None
    draft = Path(result["draft_path"]).read_text()
    assert "## Cross-domain connections" in draft
    assert "### corroborates across cgt ↔ compbio" in draft
    assert "[[paper-a#kc-a1]]" in draft
    assert "[[paper-b#kc-b1]]" in draft
    assert "corroborates rationale" in draft


def test_refresh_dry_run_writes_nothing(isolated_wiki, monkeypatch):
    root, wiki = isolated_wiki
    _stub_stem_categories(monkeypatch, {"paper-a": "cgt", "paper-b": "compbio"})
    _seed_edges([
        ("paper-a", "kc-a1", "concepts", "prime-editing", "instantiates", True),
        ("paper-b", "kc-b1", "concepts", "prime-editing", "instantiates", True),
        ("paper-a", "kc-a1", "paper-b", "kc-b1", "refines", True),
    ])
    result = concepts_mod.refresh_concept("prime-editing", dry_run=True)
    assert result["n_bridges_found"] == 1
    assert result["draft_path"] is None
    # No draft dir either.
    assert not (root / ".ingest" / "prime-editing-concept-refresh").exists()


def test_refresh_directed_relations_use_arrow(isolated_wiki, monkeypatch):
    _stub_stem_categories(monkeypatch, {"paper-a": "cgt", "paper-b": "compbio"})
    _seed_edges([
        ("paper-a", "kc-a1", "concepts", "prime-editing", "instantiates", True),
        ("paper-b", "kc-b1", "concepts", "prime-editing", "instantiates", True),
        # refines is directed
        ("paper-a", "kc-a1", "paper-b", "kc-b1", "refines", True),
    ])
    result = concepts_mod.refresh_concept("prime-editing")
    draft = Path(result["draft_path"]).read_text()
    assert "→" in draft
    assert "### refines across cgt ↔ compbio" in draft


def test_refresh_deleted_draft_is_a_rejection(isolated_wiki, monkeypatch):
    """The plan's reject-by-delete story — after refresh writes a draft, the
    author deletes the file. The hub is untouched, and a second refresh call
    just re-drafts (idempotent from the hub's POV; author sees the same draft
    proposal again to redecide)."""
    root, wiki = isolated_wiki
    hub_path = wiki / "concepts" / "prime-editing.md"
    before_hub = hub_path.read_text()

    _stub_stem_categories(monkeypatch, {"paper-a": "cgt", "paper-b": "compbio"})
    _seed_edges([
        ("paper-a", "kc-a1", "concepts", "prime-editing", "instantiates", True),
        ("paper-b", "kc-b1", "concepts", "prime-editing", "instantiates", True),
        ("paper-a", "kc-a1", "paper-b", "kc-b1", "corroborates", False),
    ])
    result = concepts_mod.refresh_concept("prime-editing")
    draft_path = Path(result["draft_path"])
    assert draft_path.exists()
    # Simulate the author rejecting by deleting.
    draft_path.unlink()
    # Hub must be untouched.
    assert hub_path.read_text() == before_hub
    # Second call re-drafts under the same path.
    result2 = concepts_mod.refresh_concept("prime-editing")
    assert Path(result2["draft_path"]) == draft_path
    assert draft_path.exists()


def test_refresh_raises_for_missing_hub(isolated_wiki, monkeypatch):
    _stub_stem_categories(monkeypatch, {})
    with pytest.raises(ValueError):
        concepts_mod.refresh_concept("does-not-exist")
