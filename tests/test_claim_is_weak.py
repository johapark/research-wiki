"""ClaimDetail.is_weak — gatekeeper for which claims reach the critic phase.

Adds a BM25-conjunctive-with-semantic floor on top of the existing checks
(numeric drift / negation mismatch / very-low semantic). Important to pin
precisely: a regression here either over-flags legitimate paraphrases
(noises the critic with no-change-needed cases) or under-flags drift
(claims with low surface-overlap and only-borderline semantic that should
have been revised slip past the critic entirely).
"""

from researchwiki.agents.phases.grade import ClaimDetail


def _claim(*, bm25: float, semantic: float | None,
           numeric_unmatched=None, negation_mismatch: bool = False) -> ClaimDetail:
    return ClaimDetail(
        section="key_contributions", position=0, text="(test)",
        bm25=bm25, semantic=semantic,
        negation_mismatch=negation_mismatch,
        numeric_unmatched=list(numeric_unmatched or []),
    )


# ---------- existing rules unchanged ----------

def test_numeric_drift_is_weak():
    c = _claim(bm25=20.0, semantic=0.9, numeric_unmatched=["47.73%"])
    assert c.is_weak()


def test_negation_mismatch_is_weak():
    c = _claim(bm25=20.0, semantic=0.9, negation_mismatch=True)
    assert c.is_weak()


def test_very_low_semantic_is_weak():
    # Below 0.40 is weak even without BM25 signal.
    c = _claim(bm25=25.0, semantic=0.30)
    assert c.is_weak()


# ---------- BM25 + semantic conjunctive floor ----------

def test_bm25_low_and_semantic_low_is_weak():
    """Drift signal: low surface overlap AND borderline semantic."""
    c = _claim(bm25=4.0, semantic=0.50)
    assert c.is_weak()


def test_bm25_low_but_semantic_high_is_not_weak():
    """Legitimate paraphrase: low BM25 (different surface form) but high
    semantic similarity. Critic shouldn't be asked to revise these."""
    c = _claim(bm25=4.0, semantic=0.70)
    assert not c.is_weak()


def test_bm25_borderline_and_semantic_borderline_is_not_weak():
    """Both at threshold = not strictly below = not weak. Conjunction
    boundary."""
    c = _claim(bm25=8.0, semantic=0.55)
    assert not c.is_weak()


def test_bm25_high_and_semantic_low_is_not_weak_unless_below_040():
    """Low semantic alone (but ≥ 0.40) and high BM25 → not weak under
    the new conjunctive rule. The 0.40 single-signal rule still holds."""
    c = _claim(bm25=20.0, semantic=0.45)
    assert not c.is_weak()
    c2 = _claim(bm25=20.0, semantic=0.35)
    assert c2.is_weak()


def test_strong_claim_is_not_weak():
    c = _claim(bm25=25.0, semantic=0.85)
    assert not c.is_weak()


def test_semantic_none_falls_through_bm25_floor():
    """When semantic scoring was unavailable, BM25 floor still applies via
    the `(self.semantic or 0.0) < 0.55` clause — None coerces to 0.0,
    triggering weak when BM25 also below 8."""
    c = _claim(bm25=4.0, semantic=None)
    assert c.is_weak()


def test_semantic_none_high_bm25_is_not_weak():
    c = _claim(bm25=25.0, semantic=None)
    assert not c.is_weak()
