"""Per-claim support (entailment) check — the qualitative analogue of the
numeric-drift veto.

Pins the pure core (`check_support` / `count_unsupported`) and the judge-
response parser (`_parse_verdicts`) with a stub classifier, so the veto logic
is verified without touching a provider. The gate is opt-in: only a flat
`unsupported` verdict counts toward the veto (`partial` does not), mirroring
the zero-tolerance-but-hard-failure-only posture of the numeric-drift veto.
"""

from __future__ import annotations

import pytest

from researchwiki.grade.support import (
    ClaimSupport,
    check_support,
    count_unsupported,
    unsupported_claims,
    _parse_verdicts,
)


def _claims():
    # (section, position, claim_text, chunk_text)
    return [
        ("Results", 0, "Method X reaches 90% accuracy.", "X reaches 90% accuracy on the test set."),
        ("Results", 1, "Method X invented a new table.", "The paper reports only qualitative results."),
        ("Summary", 2, "X is related to Y.", "X builds on Y in section 2."),
    ]


def test_verdicts_join_back_to_claims_in_order():
    verdicts = ["supported", "unsupported", "partial"]
    out = check_support(_claims(), lambda pairs: verdicts)
    assert [s.verdict for s in out] == verdicts
    assert [(s.section, s.position) for s in out] == [
        ("Results", 0), ("Results", 1), ("Summary", 2)
    ]


def test_only_unsupported_counts_toward_veto():
    out = check_support(_claims(), lambda pairs: ["supported", "unsupported", "partial"])
    assert count_unsupported(out) == 1  # 'partial' is not a veto


def test_all_supported_is_clean():
    out = check_support(_claims(), lambda pairs: ["supported"] * 3)
    assert count_unsupported(out) == 0


def test_unsupported_claims_keeps_identity_for_review():
    out = check_support(_claims(), lambda pairs: ["supported", "unsupported", "partial"])
    failed = unsupported_claims(out)
    assert len(failed) == 1                       # only the flat-unsupported one
    assert failed[0].section == "Results"
    assert failed[0].position == 1
    assert failed[0].text == "Method X invented a new table."
    # count is defined in terms of the same filter
    assert count_unsupported(out) == len(failed)


def test_empty_chunks_are_skipped_not_sent():
    claims = [
        ("Results", 0, "A claim with evidence.", "supporting chunk"),
        ("Results", 1, "A claim with no retrieved chunk.", ""),
        ("Results", 2, "Another with only whitespace.", "   "),
    ]
    seen = {}

    def classify(pairs):
        seen["n"] = len(pairs)
        return ["unsupported"] * len(pairs)

    out = check_support(claims, classify)
    assert seen["n"] == 1                 # only the claim with a chunk is judged
    assert len(out) == 1
    assert out[0].position == 0


def test_unknown_verdict_falls_back_to_partial_not_veto():
    out = check_support(_claims()[:1], lambda pairs: ["bogus"])
    assert out[0].verdict == "partial"
    assert count_unsupported(out) == 0


def test_classifier_length_mismatch_raises():
    with pytest.raises(ValueError):
        check_support(_claims(), lambda pairs: ["supported"])  # 1 verdict, 3 claims


def test_no_claims_returns_empty():
    assert check_support([], lambda pairs: []) == []


# ---- parser robustness --------------------------------------------------

def test_parse_verdicts_orders_by_id_and_defaults_partial():
    text = '{"verdicts": [{"id": 2, "verdict": "unsupported"}, {"id": 0, "verdict": "supported"}]}'
    out = _parse_verdicts(text, n=3)
    assert out == ["supported", "partial", "unsupported"]  # id 1 missing → partial


def test_parse_verdicts_tolerates_code_fence_and_prose():
    text = 'Here you go:\n```json\n{"verdicts": [{"id": 0, "verdict": "supported"}]}\n```'
    assert _parse_verdicts(text, n=1) == ["supported"]


def test_parse_verdicts_unparseable_is_all_partial_never_vetoes():
    out = _parse_verdicts("the model refused to answer", n=2)
    assert out == ["partial", "partial"]
    assert count_unsupported(
        [ClaimSupport("s", i, "t", v) for i, v in enumerate(out)]
    ) == 0
