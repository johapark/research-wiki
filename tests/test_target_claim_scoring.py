"""Live ingest scoring against importance-triaged target claims."""

import pytest

from researchwiki.agents.phases.grade import _score_target_claims
from researchwiki.agents.phases.target_claims import TargetClaim, TargetClaimsOutput


def _claim(content: str, importance: str, claim_type="capability") -> TargetClaim:
    return TargetClaim(
        type=claim_type,
        content=content,
        importance=importance,
        location="Results",
    )


def test_target_claim_score_weights_critical_miss_three_times_normal_match():
    targets = TargetClaimsOutput(claims=[
        _claim("Central catalyst improves genome editing precision", "critical", "headline"),
        _claim("Secondary assay measures protein abundance", "normal"),
        _claim("Validation includes primary cell cultures", "normal"),
        _claim("Software exports tabular result files", "normal"),
    ])
    draft = (
        "Secondary assay measures protein abundance. "
        "Validation includes primary cell cultures. "
        "Software exports tabular result files."
    )

    scores = _score_target_claims(
        stem="smith-2026-example",
        draft_text=draft,
        target_claims=targets,
        use_semantic=False,
    )

    # Three normal matches earn 3 points; one critical miss loses 3 points.
    assert scores["target_claim_score"] == pytest.approx(0.5)
    assert scores["critical_target_claim_recall"] == 0.0
    assert scores["normal_target_claim_recall"] == 1.0
    assert scores["n_critical_target_claims_missed"] == 1
    assert scores["missed_target_claims"][0]["importance"] == "critical"
    assert scores["missed_target_claims"][0]["axis"] == "target_claims"


def test_target_claim_score_ignores_invalid_or_empty_extractor_items():
    targets = TargetClaimsOutput(claims=[
        _claim("", "critical"),
        _claim("Unsupported type should not enter denominator", "high", "context"),
        _claim("Valid limitation remains in denominator", "high", "limitation"),
    ])

    scores = _score_target_claims(
        stem="smith-2026-example",
        draft_text="Valid limitation remains in denominator.",
        target_claims=targets,
        use_semantic=False,
    )

    assert scores["n_target_claims"] == 1
    assert scores["target_claim_score"] == 1.0
