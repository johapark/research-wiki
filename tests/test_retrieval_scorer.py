"""Retrieval-fixture scorer — pure scoring math.

Tests build synthetic top-K result lists and verify MRR / nDCG@K / recall /
must_not_hits compute correctly. No embedding model loaded, no DB hit; the
scorer takes RetrievedClaim/RetrievedPage lists directly.
"""
from __future__ import annotations

from researchwiki.benchmark.fixture import (
    ExpectedClaim,
    ExpectedPage,
    NegativeAnchor,
    RetrievalFixture,
)
from researchwiki.benchmark.retrieval import (
    RetrievedClaim,
    RetrievedPage,
    diff_retrieval_scores,
    score_claims_fixture,
    score_pages_fixture,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _claims_fixture(expected, must_not=(), k=10) -> RetrievalFixture:
    return RetrievalFixture(
        fixture_id="test", fixture_type="claims",
        query="test query", k=k,
        expected_claims=list(expected),
        expected_pages=[],
        must_not_appear=list(must_not),
    )


def _pages_fixture(expected, must_not=(), k=5) -> RetrievalFixture:
    return RetrievalFixture(
        fixture_id="test", fixture_type="pages",
        query="test query", k=k,
        expected_claims=[],
        expected_pages=list(expected),
        must_not_appear=list(must_not),
    )


def _claim(stem, section, position, score=1.0):
    return RetrievedClaim(paper_stem=stem, section=section, position=position, score=score)


def _page(stem, score=1.0):
    return RetrievedPage(paper_stem=stem, score=score)


# ─────────────────────────────────────────────────────────────────────
# Claim-level scoring
# ─────────────────────────────────────────────────────────────────────

def test_claims_perfect_score():
    """Every expected claim hits at the right position."""
    f = _claims_fixture([
        ExpectedClaim("a", "kc", 0, "critical"),
        ExpectedClaim("b", "kc", 0, "high"),
    ])
    retrieved = [_claim("a", "kc", 0), _claim("b", "kc", 0)]
    s = score_claims_fixture(f, retrieved, "bge")
    assert s.expected_recall == 1.0
    assert s.expected_recall_critical == 1.0
    assert s.mrr == 1.0
    assert s.ndcg_at_k == 1.0
    assert s.must_not_hits == 0


def test_claims_complete_miss():
    """No expected claims appear; recall = 0, MRR = 0, nDCG = 0."""
    f = _claims_fixture([
        ExpectedClaim("a", "kc", 0, "critical"),
    ])
    retrieved = [_claim("z", "kc", 0)]
    s = score_claims_fixture(f, retrieved, "bge")
    assert s.expected_recall == 0.0
    assert s.expected_recall_critical == 0.0
    assert s.mrr == 0.0
    assert s.ndcg_at_k == 0.0


def test_claims_mrr_prefers_critical_over_normal():
    """MRR returns reciprocal of TOP critical hit, not top-of-any-tier hit."""
    f = _claims_fixture([
        ExpectedClaim("a", "kc", 0, "normal"),     # at rank 1, but only normal
        ExpectedClaim("b", "kc", 0, "critical"),   # at rank 3
    ])
    retrieved = [
        _claim("a", "kc", 0),  # rank 1, normal
        _claim("z", "kc", 0),
        _claim("b", "kc", 0),  # rank 3, critical
    ]
    s = score_claims_fixture(f, retrieved, "bge")
    # MRR uses rank of best critical = 1/3, not rank of best any = 1/1.
    assert s.mrr == 1 / 3
    assert s.expected_recall == 1.0
    assert s.expected_recall_critical == 1.0


def test_claims_mrr_falls_back_to_any_when_no_critical():
    f = _claims_fixture([
        ExpectedClaim("a", "kc", 0, "high"),
        ExpectedClaim("b", "kc", 0, "normal"),
    ])
    retrieved = [_claim("z", "kc", 0), _claim("a", "kc", 0)]   # high at rank 2
    s = score_claims_fixture(f, retrieved, "bge")
    assert s.mrr == 0.5  # 1 / 2


def test_claims_must_not_hits_counted_stem_level():
    """Must-not violations match on stem (any section/position)."""
    f = _claims_fixture(
        expected=[ExpectedClaim("a", "kc", 0, "critical")],
        must_not=[NegativeAnchor("evil-paper", "shouldn't appear")],
    )
    retrieved = [
        _claim("a", "kc", 0),                  # expected hit
        _claim("evil-paper", "results", 4),    # must_not violation (different section/position)
    ]
    s = score_claims_fixture(f, retrieved, "bge")
    assert s.must_not_hits == 1


def test_claims_ndcg_partial_recall():
    """nDCG@K reflects partial recall and ranking quality."""
    f = _claims_fixture([
        ExpectedClaim("a", "kc", 0, "critical"),
        ExpectedClaim("b", "kc", 0, "critical"),
        ExpectedClaim("c", "kc", 0, "high"),
    ])
    # 2/3 recall: 'a' at rank 1, 'b' at rank 2, 'c' missing
    retrieved = [
        _claim("a", "kc", 0),
        _claim("b", "kc", 0),
        _claim("z", "kc", 0),
    ]
    s = score_claims_fixture(f, retrieved, "bge")
    assert s.expected_recall == 2 / 3
    assert s.expected_recall_critical == 1.0
    assert 0 < s.ndcg_at_k < 1.0  # imperfect because 'c' missed


def test_claims_first_occurrence_wins_for_rank():
    """When a (stem, section, pos) appears twice in top-K (it shouldn't, but
    test the reduction is well-defined), the first occurrence sets the rank."""
    f = _claims_fixture([ExpectedClaim("a", "kc", 0, "critical")])
    retrieved = [
        _claim("z", "kc", 0),
        _claim("a", "kc", 0),
        _claim("a", "kc", 0),  # duplicate
    ]
    s = score_claims_fixture(f, retrieved, "bge")
    assert s.per_item[0].rank == 2


def test_claims_top_k_truncation():
    """Items beyond fixture.k aren't counted as hits."""
    f = _claims_fixture(
        expected=[ExpectedClaim("a", "kc", 0, "critical")],
        k=3,
    )
    retrieved = [
        _claim("z", "kc", 0),
        _claim("z", "kc", 1),
        _claim("z", "kc", 2),
        _claim("a", "kc", 0),  # rank 4, beyond k=3
    ]
    s = score_claims_fixture(f, retrieved, "bge")
    assert s.expected_recall == 0.0


# ─────────────────────────────────────────────────────────────────────
# Page-level scoring
# ─────────────────────────────────────────────────────────────────────

def test_pages_basic_recall():
    f = _pages_fixture([
        ExpectedPage("a", "critical"),
        ExpectedPage("b", "high"),
    ])
    retrieved = [_page("a"), _page("b"), _page("z")]
    s = score_pages_fixture(f, retrieved, "bge")
    assert s.expected_recall == 1.0
    assert s.mrr == 1.0


def test_pages_expected_rank_violation():
    """When expected_rank is set, hitting at any other rank is a violation."""
    f = _pages_fixture([
        ExpectedPage("a", "critical", expected_rank=1),
    ])
    retrieved = [_page("z"), _page("a")]   # 'a' lands at rank 2, expected 1
    s = score_pages_fixture(f, retrieved, "bge")
    assert s.expected_recall == 1.0      # still recalled
    assert s.rank_violations == 1
    assert s.per_item[0].rank == 2
    assert s.per_item[0].rank_violation is True


def test_pages_expected_rank_satisfied():
    f = _pages_fixture([
        ExpectedPage("a", "critical", expected_rank=1),
    ])
    retrieved = [_page("a"), _page("z")]
    s = score_pages_fixture(f, retrieved, "bge")
    assert s.rank_violations == 0


def test_pages_category_prefix_matches_bare_stem():
    """Fixture writes bare stem; retrieval returns category/stem keys.
    Matching falls back to bare-stem comparison."""
    f = _pages_fixture([ExpectedPage("khattab-2020-foo", "critical")])
    retrieved = [_page("ai/khattab-2020-foo")]
    s = score_pages_fixture(f, retrieved, "bge")
    assert s.expected_recall == 1.0  # bare-stem fallback matched


def test_pages_must_not_matches_bare_stem():
    f = _pages_fixture(
        expected=[ExpectedPage("a", "critical")],
        must_not=[NegativeAnchor("evil")],
    )
    retrieved = [_page("a"), _page("ai/evil")]   # category prefix shouldn't matter
    s = score_pages_fixture(f, retrieved, "bge")
    assert s.must_not_hits == 1


# ─────────────────────────────────────────────────────────────────────
# A/B diff
# ─────────────────────────────────────────────────────────────────────

def test_diff_null_when_identical():
    f = _claims_fixture([ExpectedClaim("a", "kc", 0, "critical")])
    retrieved = [_claim("a", "kc", 0)]
    a = score_claims_fixture(f, retrieved, "x")
    b = score_claims_fixture(f, retrieved, "y")
    d = diff_retrieval_scores(a, b)
    assert d.delta_mrr == 0
    assert d.delta_recall == 0
    assert d.improved == []
    assert d.regressed == []
    assert len(d.unchanged) == 1


def test_diff_improvement_caught():
    f = _claims_fixture([
        ExpectedClaim("a", "kc", 0, "critical"),
        ExpectedClaim("b", "kc", 0, "high"),
    ])
    base_hits = [_claim("a", "kc", 0)]                       # 'b' missing
    cand_hits = [_claim("a", "kc", 0), _claim("b", "kc", 0)]  # both hit
    a = score_claims_fixture(f, base_hits, "base")
    b = score_claims_fixture(f, cand_hits, "cand")
    d = diff_retrieval_scores(a, b)
    assert d.delta_recall > 0
    assert len(d.improved) == 1
    keys = [k for k, _, _ in d.improved]
    assert any("b" in k for k in keys)


def test_diff_regression_caught():
    f = _claims_fixture([
        ExpectedClaim("a", "kc", 0, "critical"),
        ExpectedClaim("b", "kc", 0, "high"),
    ])
    base_hits = [_claim("a", "kc", 0), _claim("b", "kc", 0)]
    cand_hits = [_claim("a", "kc", 0)]   # 'b' dropped
    a = score_claims_fixture(f, base_hits, "base")
    b = score_claims_fixture(f, cand_hits, "cand")
    d = diff_retrieval_scores(a, b)
    assert d.delta_recall < 0
    assert len(d.regressed) == 1


def test_diff_must_not_left_top_k():
    """Hybrid drops a must_not that BM25 had — should report it as 'left'."""
    f = _claims_fixture(
        expected=[ExpectedClaim("a", "kc", 0, "critical")],
        must_not=[NegativeAnchor("bad-paper")],
    )
    base_hits = [_claim("a", "kc", 0), _claim("bad-paper", "kc", 0)]
    cand_hits = [_claim("a", "kc", 0)]   # bad-paper dropped
    a = score_claims_fixture(f, base_hits, "base")
    b = score_claims_fixture(f, cand_hits, "cand")
    d = diff_retrieval_scores(a, b)
    assert d.delta_must_not_hits == -1
    assert any("bad-paper" in k for k in d.must_not_left)


def test_diff_must_not_entered_top_k():
    """Candidate introduces a must_not that baseline didn't have."""
    f = _claims_fixture(
        expected=[ExpectedClaim("a", "kc", 0, "critical")],
        must_not=[NegativeAnchor("bad-paper")],
    )
    base_hits = [_claim("a", "kc", 0)]
    cand_hits = [_claim("a", "kc", 0), _claim("bad-paper", "kc", 0)]
    a = score_claims_fixture(f, base_hits, "base")
    b = score_claims_fixture(f, cand_hits, "cand")
    d = diff_retrieval_scores(a, b)
    assert d.delta_must_not_hits == 1
    assert any("bad-paper" in k for k in d.must_not_entered)


def test_diff_neutral_unexpected_not_flagged_as_must_not():
    """A non-expected, non-must_not stem appearing only in candidate should
    NOT show up in must_not_entered (it's just an unrelated extra hit)."""
    f = _claims_fixture(
        expected=[ExpectedClaim("a", "kc", 0, "critical")],
        must_not=[NegativeAnchor("bad-paper")],
    )
    base_hits = [_claim("a", "kc", 0)]
    cand_hits = [_claim("a", "kc", 0), _claim("neutral-paper", "kc", 0)]
    a = score_claims_fixture(f, base_hits, "base")
    b = score_claims_fixture(f, cand_hits, "cand")
    d = diff_retrieval_scores(a, b)
    assert d.must_not_entered == []
    assert d.delta_must_not_hits == 0


def test_diff_fixture_id_mismatch_raises():
    """Diffing scores from different fixtures should fail loudly."""
    import pytest
    f1 = _claims_fixture([ExpectedClaim("a", "kc", 0, "critical")])
    f2 = RetrievalFixture(
        fixture_id="other",
        fixture_type="claims",
        query="q", k=10,
        expected_claims=[ExpectedClaim("a", "kc", 0, "critical")],
        expected_pages=[], must_not_appear=[],
    )
    a = score_claims_fixture(f1, [_claim("a", "kc", 0)], "x")
    b = score_claims_fixture(f2, [_claim("a", "kc", 0)], "y")
    with pytest.raises(ValueError, match="fixture_id mismatch"):
        diff_retrieval_scores(a, b)
