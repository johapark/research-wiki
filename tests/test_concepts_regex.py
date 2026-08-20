"""Tests for the claim-substrate concept detector (regex over claims.text).

Covers:
  - fixtures with `prime editing`, `off-target effects`, `foundation model`
    across ≥3 papers each → surface in candidates output at appropriate labels
  - stop-list filters `this work`, `the method`, etc. even at ≥5 papers
  - ranking: (pages=5, cats=3) outranks (pages=8, cats=1)
  - threshold labels: bridge / deep / candidate
  - instantiates edges written when persist_edges=True
  - term-slug normalization matches _slugify (scaffold + edge target agree)
"""

from __future__ import annotations

import pytest

from researchwiki.claim_graph import Edge, open_edges_db, query, upsert_edge
from researchwiki.concepts.candidates import _extract_terms, _label_for, _normalize_lc_phrase, _term_slug, find_candidates_from_claims
from researchwiki.concepts.term_claims import _promote_instantiates_edges_for


# ---------- extractor primitives ----------


def test_extract_terms_finds_acronym():
    assert "CRISPR" in _extract_terms("CRISPR-Cas9 introduces DSBs at target sites")


def test_extract_terms_finds_lowercase_head_noun_phrase():
    assert "prime editing" in _extract_terms(
        "prime editing achieves 40% efficiency without double-strand breaks"
    )


def test_extract_terms_finds_off_target_effects():
    got = _extract_terms("off-target effects were assessed in HEK293T cells")
    assert "off-target effects" in got
    assert "HEK293T" in got


def test_extract_terms_captures_numeric_prefixed_entity_whole():
    # "1000 Genomes Project" must be captured whole, not truncated to the
    # fragment "Genomes Project" — the fragment otherwise surfaces as its own
    # inflated candidate (double-counting the same entity).
    got = _extract_terms("variants from the 1000 Genomes Project reference panel")
    assert "1000 Genomes Project" in got
    assert "Genomes Project" not in got


def test_extract_terms_strips_leading_determiner():
    # "the foundation model" → "foundation model", not both.
    got = _extract_terms("the foundation model was pre-trained on 200B tokens")
    assert "foundation model" in got
    assert "the foundation model" not in got


def test_extract_terms_filters_leading_verb_and_short_residual():
    # "outperforms nuclease editing" — verb consumed; only "nuclease editing" left.
    got = _extract_terms("base editing outperforms nuclease editing on most benchmarks")
    assert "nuclease editing" in got
    assert "outperforms nuclease editing" not in got
    # "most benchmarks" — "most" consumed; single-word residual dropped.
    assert "most benchmarks" not in got
    assert "benchmarks" not in got


def test_extract_terms_filters_claim_stop_phrases():
    # "this work" would end in a head noun (`work` isn't in HEAD_NOUNS, so
    # the regex wouldn't match anyway) — but "the method" would (if `method`
    # were a HEAD_NOUN, which it isn't). Test with something that WOULD match
    # were it not stop-listed: "the model" → after stripping "the" → "model" →
    # single-word residual → dropped. Second check: "the framework" — same.
    assert "the model" not in _extract_terms("the model achieves 92% accuracy")
    assert "the framework" not in _extract_terms("the framework outperforms baselines")


def test_normalize_lc_phrase_returns_none_for_single_word_residual():
    # After stripping leading stopwords, a bare head noun isn't concept-worthy.
    assert _normalize_lc_phrase("the model") is None
    assert _normalize_lc_phrase("a design") is None


def test_normalize_lc_phrase_strips_possessive_s():
    got = _normalize_lc_phrase("author's design system")
    assert got is not None
    assert "'" not in got


# ---------- ranking + labels ----------


def test_label_bridge_when_span_ge_2_and_pages_ge_3():
    assert _label_for(pages=3, categories=2) == "concept-ready (bridge)"
    assert _label_for(pages=10, categories=7) == "concept-ready (bridge)"


def test_label_deep_when_span_1_and_pages_ge_5():
    assert _label_for(pages=5, categories=1) == "concept-ready (deep)"
    assert _label_for(pages=8, categories=1) == "concept-ready (deep)"


def test_label_candidate_otherwise():
    assert _label_for(pages=3, categories=1) == "candidate"
    assert _label_for(pages=4, categories=1) == "candidate"


def test_ranking_prefers_span_over_raw_page_count():
    # (pages=5, cats=3) outranks (pages=8, cats=1) per §3.3.
    #   5 * sqrt(3) ≈ 8.66   >   8 * sqrt(1) = 8.0
    rows = [
        {"paper_stem": f"p1-{i}", "section": "results", "text": "alpha bridge design",
         "category": f"cat-{i}", "claim_slug": f"res-{i:04d}"} for i in range(5)
    ] + [
        {"paper_stem": f"p2-{i}", "section": "results", "text": "beta deep design",
         "category": "one-cat", "claim_slug": f"res-{i+100:04d}"} for i in range(8)
    ]
    # Distribute categories: alpha across cat-0..cat-2 (3 categories); beta all "one-cat".
    for i, r in enumerate(rows[:5]):
        r["category"] = f"cat-{i % 3}"
    got = find_candidates_from_claims(rows, existing_slugs=set())
    terms_ranked = [r["term"] for r in got]
    # alpha (bridge, pages=5, cats=3) should come before beta (deep, pages=8, cats=1).
    a_idx = terms_ranked.index("alpha bridge design")
    b_idx = terms_ranked.index("beta deep design")
    assert a_idx < b_idx


# ---------- corpus-shape fixtures ----------


def _mk(stem: str, section: str, text: str, cat: str, slug_suffix: str):
    return {"paper_stem": stem, "section": section, "text": text,
            "category": cat, "claim_slug": f"{section[:3]}-{slug_suffix}"}


def test_fixture_prime_editing_surfaces_across_papers():
    rows = [
        _mk("p1", "key_contributions",
            "prime editing installs targeted edits without a DSB", "cgt", "aaaa1111"),
        _mk("p2", "results",
            "prime editing efficiency reached 40% in HEK293T", "cgt", "bbbb2222"),
        _mk("p3", "key_contributions",
            "improves prime editing with a novel PE architecture", "cgt", "cccc3333"),
    ]
    got = find_candidates_from_claims(rows, existing_slugs=set())
    slugs = {r["slug"] for r in got}
    assert "prime-editing" in slugs


def test_fixture_off_target_effects_surfaces():
    # Note: leading verbs (`we observed`, `we noted`) are absorbed by the
    # greedy regex — the extractor cannot separate them from the phrase
    # without POS tagging. Claim text from real papers is written as short
    # declarative bullets, not narrative sentences, so this is the shape
    # the detector is tuned for. Head-noun-anchored regexes are bounded,
    # not general — same-position ambiguity is unresolved.
    # Three papers must carry the term in a *contribution* section to clear the
    # floor — membership counts the sections `find_members` matches over, so the
    # limitations row is evidence the term recurs but is not a member. p4 is what
    # makes this a 3-page candidate.
    rows = [
        _mk("p1", "results", "off-target effects were minimal in liver", "cgt", "a1"),
        _mk("p2", "results", "off-target effects at three sites", "cgt", "b2"),
        _mk("p3", "limitations", "off-target effects remain uncharacterised", "compbio", "c3"),
        _mk("p4", "key_contributions", "off-target effects quantified genome-wide", "compbio", "d4"),
    ]
    got = find_candidates_from_claims(rows, existing_slugs=set())
    slugs = {r["slug"] for r in got}
    assert "off-target-effects" in slugs


def test_fixture_foundation_model_surfaces():
    rows = [
        _mk("p1", "key_contributions",
            "the foundation model was pre-trained on 200B tokens", "compbio", "a1"),
        _mk("p2", "results",
            "foundation model outperforms all baselines", "compbio", "b2"),
        _mk("p3", "methodology",
            "we adapt a foundation model to protein prediction", "compbio", "c3"),
    ]
    got = find_candidates_from_claims(rows, existing_slugs=set())
    slugs = {r["slug"] for r in got}
    assert "foundation-model" in slugs


def test_fixture_stop_phrases_filtered_even_at_high_count():
    # "the method" across 5 papers — must NOT surface.
    rows = [
        _mk(f"p{i}", "key_contributions", "the method achieves state of the art results",
            f"cat{i % 3}", f"a{i}")
        for i in range(5)
    ]
    got = find_candidates_from_claims(rows, existing_slugs=set())
    slugs = {r["slug"] for r in got}
    assert "the-method" not in slugs
    assert "method" not in slugs


# ---------- instantiates edges ----------


@pytest.fixture
def temp_edges_db(tmp_path, monkeypatch):
    """Point .claim-graph/ at a tmp dir so tests don't touch the real cache."""
    from researchwiki import claim_graph
    monkeypatch.setattr(claim_graph.edges, "edges_db_path",
                        lambda: tmp_path / "edges.db")
    yield tmp_path


def test_persist_edges_writes_instantiates(temp_edges_db):
    rows = [
        _mk("p1", "results", "prime editing at 40% efficiency", "cgt", "a1"),
        _mk("p2", "results", "prime editing outperforms Cas9 base", "cgt", "b2"),
        _mk("p3", "key_contributions", "prime editing pipeline for liver", "cgt", "c3"),
    ]
    find_candidates_from_claims(rows, existing_slugs=set(), persist_edges=True)
    conn = open_edges_db()
    try:
        edges = query(conn, relation="instantiates",
                      tgt_stem="concepts", tgt_slug="prime-editing")
        assert len(edges) == 3
        assert {e.src_stem for e in edges} == {"p1", "p2", "p3"}
        assert {e.status for e in edges} == {"candidate"}
    finally:
        conn.close()


def test_promotion_transitions_candidate_to_confirmed(temp_edges_db):
    # Seed a couple of candidate instantiates edges via API.
    from researchwiki.claim_graph import SLUG_SCHEME_VERSION
    conn = open_edges_db()
    try:
        for stem in ("p1", "p2"):
            upsert_edge(conn, Edge(
                src_stem=stem, src_slug="kc-aaaa1111",
                tgt_stem="concepts", tgt_slug="off-target-effects",
                relation="instantiates", directed=True,
                slug_scheme_version=SLUG_SCHEME_VERSION, status="candidate",
            ))
        conn.commit()
    finally:
        conn.close()

    n = _promote_instantiates_edges_for("off-target effects")
    assert n == 2

    conn = open_edges_db()
    try:
        confirmed = query(conn, relation="instantiates",
                          tgt_stem="concepts", tgt_slug="off-target-effects",
                          status="confirmed")
        candidates = query(conn, relation="instantiates",
                           tgt_stem="concepts", tgt_slug="off-target-effects",
                           status="candidate")
        assert len(confirmed) == 2
        assert len(candidates) == 0
    finally:
        conn.close()


# ---------- slug helper ----------


def test_term_slug_matches_synthesize_slugify():
    """The concept-page filename slug and the edge target slug MUST agree —
    otherwise scaffold promotion misses its own edges."""
    from researchwiki.tasks.synthesize import _slugify
    assert _term_slug("Prime Editing") == _slugify("Prime Editing")
    assert _term_slug("Off-Target Effects") == _slugify("Off-Target Effects")
    assert _term_slug("CRISPR") == _slugify("CRISPR")
