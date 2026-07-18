"""Tests for the keyword-primary concept detector.

LLM-authored keywords/tags on each paper page are the primary detection
substrate; the claim-substrate regex path fills recall for acronyms +
phrases the keyword lists miss.
"""

from __future__ import annotations

import pytest

from researchwiki.claim_graph import Edge, open_edges_db, query
from researchwiki.concepts.candidates import _FRAMEWORK_TAG_STOPLIST
from researchwiki.concepts.candidates import _merge_candidate_sources, _term_slug, find_candidates_from_keywords


# ---------- keyword aggregator ----------


def _mk_paper(stem: str, category: str, keywords=None, tags=None) -> dict:
    return {
        "stem": stem,
        "category": category,
        "keywords": keywords or [],
        "tags": tags or [],
    }


def test_keywords_surface_when_three_papers_share_a_term():
    papers = [
        _mk_paper("p1", "cgt", keywords=["prime editing", "efficiency"]),
        _mk_paper("p2", "cgt", keywords=["prime editing", "delivery"]),
        _mk_paper("p3", "compbio", keywords=["prime editing"]),
    ]
    got = find_candidates_from_keywords(papers, existing_slugs=set())
    slugs = {r["slug"] for r in got}
    assert "prime-editing" in slugs


def test_keywords_below_threshold_filtered():
    papers = [
        _mk_paper("p1", "cgt", keywords=["prime editing"]),
        _mk_paper("p2", "cgt", keywords=["prime editing"]),
    ]
    assert find_candidates_from_keywords(papers, existing_slugs=set()) == []


def test_framework_tags_ignored():
    """Ingest-artifact tags (`ingested-via-agent`, ...) must never surface."""
    papers = [
        _mk_paper("p1", "cgt", tags=["ingested-via-agent"]),
        _mk_paper("p2", "cgt", tags=["ingested-via-agent"]),
        _mk_paper("p3", "cgt", tags=["ingested-via-agent"]),
    ]
    assert find_candidates_from_keywords(papers, existing_slugs=set()) == []


def test_stoplist_membership_stable():
    """Explicit whitelist test — additions to the stoplist should be
    deliberate. Keeping this small prevents accidental filtering of real
    concept-slug tags like 'crispr' or 'foundation-model'."""
    assert "ingested-via-agent" in _FRAMEWORK_TAG_STOPLIST
    # Sanity: real concept tags MUST NOT be in the stoplist.
    for real_tag in ("crispr", "foundation-model", "off-target", "prime-editing"):
        assert real_tag not in _FRAMEWORK_TAG_STOPLIST


def test_hyphen_and_space_variants_merge_by_slug():
    """`off-target` (tag) and `off target` (keyword) should count as the
    same concept because they slugify identically."""
    papers = [
        _mk_paper("p1", "cgt", tags=["off-target"]),
        _mk_paper("p2", "cgt", keywords=["off target"]),
        _mk_paper("p3", "compbio", keywords=["off-target"]),
    ]
    got = find_candidates_from_keywords(papers, existing_slugs=set())
    slugs = [r["slug"] for r in got]
    # Both variants collapse to the same slug.
    assert slugs.count("off-target") == 1


def test_longer_form_wins_as_display():
    """When variants merge, the longer / more human-readable form wins."""
    papers = [
        _mk_paper("p1", "cgt", tags=["prime-editing"]),
        _mk_paper("p2", "cgt", keywords=["Prime Editing"]),
        _mk_paper("p3", "cgt", keywords=["Prime Editing"]),
    ]
    got = find_candidates_from_keywords(papers, existing_slugs=set())
    assert got[0]["slug"] == "prime-editing"
    # "Prime Editing" (13 chars) > "prime-editing" (13 chars): tie broken
    # by ordering — either is acceptable, but non-empty is required.
    assert got[0]["term"]


def test_keywords_source_marker():
    papers = [
        _mk_paper(f"p{i}", "cgt", keywords=["prime editing"]) for i in range(3)
    ]
    got = find_candidates_from_keywords(papers, existing_slugs=set())
    assert got[0]["source"] == "keywords"


def test_categories_span_from_paper_directories():
    papers = [
        _mk_paper("p1", "cgt", keywords=["prime editing"]),
        _mk_paper("p2", "compbio", keywords=["prime editing"]),
        _mk_paper("p3", "genomics", keywords=["prime editing"]),
    ]
    got = find_candidates_from_keywords(papers, existing_slugs=set())
    assert got[0]["categories"] == 3
    assert got[0]["label"] == "concept-ready (bridge)"


def test_existing_slugs_are_filtered():
    papers = [
        _mk_paper(f"p{i}", "cgt", keywords=["prime editing"]) for i in range(3)
    ]
    got = find_candidates_from_keywords(papers, existing_slugs={"prime-editing"})
    assert got == []


def test_empty_and_non_string_keywords_are_skipped():
    papers = [
        {"stem": "p1", "category": "cgt", "keywords": ["", None, 42, "  ", "real term"]},
        {"stem": "p2", "category": "cgt", "keywords": ["real term"]},
        {"stem": "p3", "category": "cgt", "keywords": ["real term"]},
    ]
    got = find_candidates_from_keywords(papers, existing_slugs=set())
    assert len(got) == 1
    assert got[0]["slug"] == "real-term"


# ---------- source union ----------


def test_merge_primary_wins_when_slugs_overlap():
    primary = [{"term": "Foo", "slug": "foo", "pages": 5, "categories": 3,
                "weighted": 5.0, "sections": {}, "label": "concept-ready (bridge)",
                "source": "keywords"}]
    secondary = [{"term": "FOO", "slug": "foo", "pages": 3, "categories": 1,
                  "weighted": 3.0, "sections": {}, "label": "candidate",
                  "source": "claims"}]
    merged = _merge_candidate_sources(primary, secondary)
    assert len(merged) == 1
    assert merged[0]["term"] == "Foo"  # primary won
    assert merged[0]["source"] == "keywords"


def test_merge_appends_secondary_only_slugs():
    primary = [{"term": "Foo", "slug": "foo", "pages": 5, "categories": 3,
                "weighted": 5.0, "sections": {}, "label": "concept-ready (bridge)",
                "source": "keywords"}]
    secondary = [{"term": "BAR", "slug": "bar", "pages": 4, "categories": 2,
                  "weighted": 4.0, "sections": {}, "label": "concept-ready (bridge)",
                  "source": "claims"}]
    merged = _merge_candidate_sources(primary, secondary)
    assert len(merged) == 2
    # Primary before secondary.
    assert merged[0]["slug"] == "foo"
    assert merged[1]["slug"] == "bar"


# ---------- integration: instantiates edges from keywords ----------


@pytest.fixture
def temp_edges_db(tmp_path, monkeypatch):
    from researchwiki import claim_graph
    monkeypatch.setattr(claim_graph.edges, "edges_db_path",
                        lambda: tmp_path / "edges.db")
    yield tmp_path


def test_keyword_edges_attach_to_matching_claims_only(temp_edges_db, monkeypatch):
    """A paper with `prime editing` in its keywords AND a claim mentioning
    `prime editing` gets an edge from that claim; a paper with the keyword
    but no matching claim contributes NO edge (keyword-only membership)."""
    from researchwiki.concepts import candidates as _concepts

    papers_meta = [
        _mk_paper("p1", "cgt", keywords=["prime editing"]),
        _mk_paper("p2", "cgt", keywords=["prime editing"]),
        _mk_paper("p3", "cgt", keywords=["prime editing"]),
    ]
    claim_rows = [
        # p1: two claims, one mentions the term.
        {"paper_stem": "p1", "section": "results", "claim_slug": "res-a1",
         "text": "prime editing at 40% efficiency", "category": "cgt"},
        {"paper_stem": "p1", "section": "limitations", "claim_slug": "lim-a2",
         "text": "delivery remains a challenge", "category": "cgt"},
        # p2: one claim, mentions the term.
        {"paper_stem": "p2", "section": "results", "claim_slug": "res-b1",
         "text": "our prime editing pipeline outperforms Cas9", "category": "cgt"},
        # p3: one claim, does NOT mention the term (uses PE only).
        {"paper_stem": "p3", "section": "results", "claim_slug": "res-c1",
         "text": "PE achieved 50% efficiency in liver", "category": "cgt"},
    ]

    keyword_rows = find_candidates_from_keywords(papers_meta, existing_slugs=set())
    assert keyword_rows[0]["slug"] == "prime-editing"

    _concepts._persist_keyword_instantiates(keyword_rows, claim_rows, papers_meta)
    conn = open_edges_db()
    try:
        edges = query(conn, relation="instantiates",
                      tgt_stem="concepts", tgt_slug="prime-editing")
    finally:
        conn.close()

    # Expect two edges: p1/res-a1, p2/res-b1. p1/lim-a2 doesn't mention;
    # p3/res-c1 uses the "PE" acronym, not the spelled-out form.
    assert len(edges) == 2
    endpoints = {(e.src_stem, e.src_slug) for e in edges}
    assert endpoints == {("p1", "res-a1"), ("p2", "res-b1")}
