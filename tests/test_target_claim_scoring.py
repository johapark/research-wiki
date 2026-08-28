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
    assert scores["n_target_claim_sources"] == 1
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


def test_target_claim_source_diversity_counts_structural_locations():
    targets = TargetClaimsOutput(claims=[
        _claim("Headline claim from the abstract", "critical", "headline"),
        TargetClaim(
            type="capability", content="Benchmark claim from Figure 3",
            importance="high", location="Figure 3",
        ),
        TargetClaim(
            type="limitation", content="Caveat acknowledged by the authors",
            importance="high", location="Discussion: Limitations",
        ),
    ])
    scores = _score_target_claims(
        stem="smith-2026-example",
        draft_text="Headline claim from the abstract. Benchmark claim from Figure 3. "
                   "Caveat acknowledged by the authors.",
        target_claims=targets,
        use_semantic=False,
    )
    # `_claim` defaults to Results; the other two normalize to figures and
    # discussion. Free-form subsection names must not inflate this count.
    assert scores["n_target_claim_sources"] == 3


def test_unplaceable_locations_do_not_penalize_against_omitting_them():
    """Vague provenance must never score worse than no provenance.

    A location the normalizer cannot place ("p. 4") used to collapse into one
    `other` family, which set `n_target_claim_sources: 1` and halved
    `target_claim_confidence` — while an extractor that emitted no location at
    all omitted the key and kept full confidence. That inversion punished the
    more informative output, so an unplaceable location now contributes no
    family and the two cases weigh the same.
    """
    from researchwiki.agents.fitness import target_claim_confidence

    def _score(location):
        return _score_target_claims(
            stem="smith-2026-example",
            draft_text="Alpha finding holds. Beta finding holds.",
            target_claims=TargetClaimsOutput(claims=[
                TargetClaim(type="headline", content="Alpha finding holds",
                            importance="critical", location=location),
                TargetClaim(type="capability", content="Beta finding holds",
                            importance="high", location=location),
            ]),
            use_semantic=False,
        )

    unplaceable = _score("p. 4")
    absent = _score(None)
    assert "n_target_claim_sources" not in unplaceable
    assert "n_target_claim_sources" not in absent
    assert target_claim_confidence(unplaceable) == target_claim_confidence(absent)

    # A single *placeable* family still discounts, which is the axis's point:
    # the multiplier rewards spread across a paper's structure.
    single_family = _score("Results")
    assert single_family["n_target_claim_sources"] == 1
    assert target_claim_confidence(single_family) < target_claim_confidence(absent)
