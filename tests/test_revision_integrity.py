"""Critical target IDs are constraints on both revision operators."""

from types import SimpleNamespace

import pytest

from researchwiki.agents.fitness import is_evolve_improvement, is_strict_improvement


def _draft(score, misses):
    return SimpleNamespace(scores={
        "semantic_score": score, "n_drift": 0, "n_target_claims": 5,
        "target_claim_score": score, "target_claim_extraction_status": "extracted",
        "missed_target_claims": [
            {"id": key, "importance": importance} for key, importance in misses
        ],
    })


@pytest.mark.parametrize("rule", [is_evolve_improvement, is_strict_improvement])
@pytest.mark.parametrize("old_misses,new_misses", [
    ([], [("target-001", "critical")]),
    ([("target-001", "critical")], [("target-002", "critical")]),
])
def test_new_critical_miss_vetoes_higher_quality(rule, old_misses, new_misses):
    assert not rule(_draft(.95, new_misses), _draft(.60, old_misses))


@pytest.mark.parametrize("rule", [is_evolve_improvement, is_strict_improvement])
@pytest.mark.parametrize("new_misses", [[], [("target-001", "critical")], [("target-002", "high")]])
def test_improvement_without_new_critical_miss_is_accepted(rule, new_misses):
    assert rule(_draft(.95, new_misses), _draft(.60, [("target-001", "critical")]))


@pytest.mark.parametrize("rule", [is_evolve_improvement, is_strict_improvement])
@pytest.mark.parametrize("field,value", [
    ("missed_target_claims", None), ("target_claim_extraction_status", "error"),
    ("n_target_claims", 0), ("n_target_claims", 6),
])
def test_unavailable_or_incomparable_evaluation_preserves_legacy_rule(rule, field, value):
    prior = _draft(.60, [])
    new = _draft(.95, [("target-001", "critical")])
    if value is None:
        prior.scores.pop(field)
    else:
        prior.scores[field] = value
    assert rule(new, prior)
