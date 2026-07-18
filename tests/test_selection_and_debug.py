"""Selection policy (`_is_strict_improvement`) and the DEBUG issue classifier.

`_is_strict_improvement` is π_sel — the comparison rule that decides whether an
evolve/debug draft replaces the current winner. Its lexicographic ordering
(semantic, then drift veto, then BM25) prevents replacement cycles, so the
thresholds matter. `detect_structural_gate_issues` decides which gate failures
DEBUG will attempt to repair vs. which go straight to sandbox.
"""

from types import SimpleNamespace

from researchwiki.agents.phases.revise import detect_structural_gate_issues
from researchwiki.agents.runner import _is_strict_improvement


def _draft(**scores):
    return SimpleNamespace(scores=scores)


# ---------- _is_strict_improvement ----------

def test_higher_semantic_improves():
    new = _draft(semantic_score=0.70)
    old = _draft(semantic_score=0.60)
    assert _is_strict_improvement(new, old) is True


def test_lower_semantic_rejected():
    new = _draft(semantic_score=0.50)
    old = _draft(semantic_score=0.70)
    assert _is_strict_improvement(new, old) is False


def test_semantic_within_epsilon_is_a_tie():
    # 0.005 difference < 0.01 epsilon → not an improvement on the primary axis.
    new = _draft(semantic_score=0.605, n_drift=0, mean_bm25=4.0)
    old = _draft(semantic_score=0.600, n_drift=0, mean_bm25=4.0)
    assert _is_strict_improvement(new, old) is False


def test_drift_breaks_semantic_tie():
    new = _draft(semantic_score=0.60, n_drift=0, mean_bm25=4.0)
    old = _draft(semantic_score=0.60, n_drift=2, mean_bm25=4.0)
    assert _is_strict_improvement(new, old) is True


def test_more_drift_rejected_on_tie():
    new = _draft(semantic_score=0.60, n_drift=2, mean_bm25=9.0)
    old = _draft(semantic_score=0.60, n_drift=0, mean_bm25=4.0)
    assert _is_strict_improvement(new, old) is False


def test_bm25_tiebreak_requires_margin():
    # BM25 only breaks a full tie, and only past a 0.5 margin.
    clear = _draft(semantic_score=0.60, n_drift=0, mean_bm25=5.0)
    base = _draft(semantic_score=0.60, n_drift=0, mean_bm25=4.0)
    assert _is_strict_improvement(clear, base) is True

    marginal = _draft(semantic_score=0.60, n_drift=0, mean_bm25=4.3)
    assert _is_strict_improvement(marginal, base) is False


def test_falls_through_when_semantic_unavailable():
    # No semantic scores at all → decided on drift / BM25.
    new = _draft(n_drift=0, mean_bm25=5.0)
    old = _draft(n_drift=0, mean_bm25=4.0)
    assert _is_strict_improvement(new, old) is True


# ---------- detect_structural_gate_issues ----------

def test_drift_reason_maps_to_drift():
    assert detect_structural_gate_issues(["2 numeric drift claim(s)"]) == ["drift"]


def test_kc_reason_maps_to_n_kc():
    reason = "only 2 Key Contribution bullets (need 4 — page may be incomplete)"
    assert detect_structural_gate_issues([reason]) == ["n_kc"]


def test_graded_reason_maps_to_n_graded():
    assert detect_structural_gate_issues(["only 3 graded claims (need 5)"]) == ["n_graded"]


def test_non_structural_reason_ignored():
    # A semantic-threshold failure is not something DEBUG repairs.
    assert detect_structural_gate_issues(["mean semantic 0.40 < 0.55 threshold"]) == []


def test_multiple_issues_deduped_and_ordered():
    reasons = [
        "3 numeric drift claim(s)",
        "only 1 Key Contribution bullets (need 4)",
        "only 2 graded claims (need 5)",
        "another numeric drift claim(s)",  # duplicate code, must not repeat
    ]
    assert detect_structural_gate_issues(reasons) == ["drift", "n_kc", "n_graded"]


def test_empty_reasons():
    assert detect_structural_gate_issues([]) == []
    assert detect_structural_gate_issues(None) == []
