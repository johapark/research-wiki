"""Promotion-gate logic (`should_auto_promote`) and its structural helpers.

The gate is the load-bearing decision between auto-promote and sandbox. These
tests pin each individual failure reason plus the all-pass path, and double as
executable documentation of the thresholds (which the prose docs got wrong).
"""

from types import SimpleNamespace

from researchwiki.agents.promote import (
    MIN_GRADED_CLAIMS,
    MIN_KEY_CONTRIBUTIONS,
    SEMANTIC_MEAN_THRESHOLD,
    SEMANTIC_MEAN_THRESHOLD_REVIEW,
    _count_key_contributions,
    _extract_section,
    detect_publication_status,
    should_auto_promote,
)


def _passing_scores(**overrides):
    scores = {
        "semantic_available": True,
        "semantic_score": 0.70,
        "n_drift": 0,
        "n_graded": MIN_GRADED_CLAIMS,
    }
    scores.update(overrides)
    return scores


def _no_broken():
    return SimpleNamespace(broken=[])


# ---------- should_auto_promote ----------

def test_all_gates_pass():
    gate = should_auto_promote(
        _passing_scores(), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS,
    )
    assert gate.promoted is True
    assert gate.reasons == []


def test_low_semantic_blocks():
    gate = should_auto_promote(
        _passing_scores(semantic_score=SEMANTIC_MEAN_THRESHOLD - 0.1),
        _no_broken(), n_key_contributions=MIN_KEY_CONTRIBUTIONS,
    )
    assert gate.promoted is False
    assert any("semantic" in r for r in gate.reasons)


def test_review_uses_relaxed_threshold():
    # A score that fails the research bar but clears the (lower) review bar.
    sem = (SEMANTIC_MEAN_THRESHOLD + SEMANTIC_MEAN_THRESHOLD_REVIEW) / 2
    assert SEMANTIC_MEAN_THRESHOLD_REVIEW <= sem < SEMANTIC_MEAN_THRESHOLD
    research = should_auto_promote(
        _passing_scores(semantic_score=sem), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS, paper_type="research",
    )
    review = should_auto_promote(
        _passing_scores(semantic_score=sem), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS, paper_type="review",
    )
    assert research.promoted is False
    assert review.promoted is True


def test_disabled_semantic_scorer_blocks():
    gate = should_auto_promote(
        _passing_scores(semantic_available=False), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS,
    )
    assert gate.promoted is False
    assert any("semantic scorer was disabled" in r for r in gate.reasons)


def test_numeric_drift_blocks():
    gate = should_auto_promote(
        _passing_scores(n_drift=2), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS,
    )
    assert gate.promoted is False
    assert any("numeric drift" in r for r in gate.reasons)


def test_unsupported_claims_block():
    """The per-claim entailment veto (grade.support): any unsupported claim
    blocks auto-promote, the qualitative analogue of numeric drift."""
    gate = should_auto_promote(
        _passing_scores(n_unsupported=3), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS,
    )
    assert gate.promoted is False
    assert any("unsupported claim" in r for r in gate.reasons)


def test_zero_unsupported_does_not_block():
    """Absent / zero n_unsupported is inert — the check didn't run or every
    claim was supported, so the gate is unaffected."""
    gate = should_auto_promote(
        _passing_scores(n_unsupported=0), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS,
    )
    assert gate.promoted is True
    assert not any("unsupported claim" in r for r in gate.reasons)


def test_broken_wikilinks_warn_but_dont_block():
    """Verify already strips broken targets from the cleaned text. The list
    records what the drafter *attempted* to link to (typically external
    baselines hallucinated as wiki pages). The page is still shippable, so
    the gate reports a warning instead of blocking — drift / KC / semantic
    gates remain the load-bearing fail conditions."""
    gate = should_auto_promote(
        _passing_scores(), SimpleNamespace(broken=["cgt/nope", "compbio/nope2"]),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS,
    )
    assert gate.promoted is True
    assert gate.reasons == []
    assert any("broken wikilink" in w for w in gate.warnings)


def test_too_few_graded_claims_blocks():
    gate = should_auto_promote(
        _passing_scores(n_graded=MIN_GRADED_CLAIMS - 1), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS,
    )
    assert gate.promoted is False
    assert any("graded claims" in r for r in gate.reasons)


def test_too_few_key_contributions_blocks():
    gate = should_auto_promote(
        _passing_scores(), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS - 1,
    )
    assert gate.promoted is False
    assert any("Key Contribution" in r for r in gate.reasons)


def test_multiple_failures_all_reported():
    gate = should_auto_promote(
        _passing_scores(semantic_score=0.1, n_drift=3, n_graded=1),
        _no_broken(), n_key_contributions=0,
    )
    assert gate.promoted is False
    # 4 hard fails: semantic, drift, n_graded, n_kc. Broken is now a warning
    # and would not contribute to fails even if non-empty.
    assert len(gate.reasons) >= 4


# ---------- structural helpers ----------

_BODY = """---
title: x
---

## Summary
A one-line summary of the paper. It has two sentences. Here is the second.

## Key Contributions
- First contribution
- Second contribution
- Third contribution

## Results
- some number
"""


def test_count_key_contributions():
    assert _count_key_contributions(_BODY) == 3


def test_count_key_contributions_missing_section():
    assert _count_key_contributions("## Summary\nno KC here\n") == 0


def test_extract_section_is_case_insensitive():
    assert "First contribution" in _extract_section(_BODY, "key contributions")


def test_extract_section_absent_returns_empty():
    assert _extract_section(_BODY, "limitations") == ""


# ---------- detect_publication_status (DOI-prefix branch) ----------

def test_arxiv_doi_detected():
    assert detect_publication_status(None, "10.48550/arXiv.2401.00001") == "arxiv-preprint"


def test_biorxiv_doi_detected():
    assert detect_publication_status(None, "10.1101/2021.11.18.469088") == "biorxiv-preprint"


def test_published_doi_is_none():
    assert detect_publication_status(None, "10.1126/science.ado2243") is None
