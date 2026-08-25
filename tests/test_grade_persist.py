"""Regression coverage for commit-time claim-grade telemetry."""

from types import SimpleNamespace

from researchwiki.agents.phases.grade_persist import persist_grades
from researchwiki.db.connection import get_connection
from researchwiki.db.iterations import read_attempt
from researchwiki.grade.fidelity import paper as fidelity_paper


def test_persist_grades_stores_scores_as_json_object(monkeypatch):
    report = SimpleNamespace(
        n_claims=3,
        n_graded=3,
        n_cross_refs=0,
        mean_top1=21.5,
        median_top1=20.0,
        weakest_score=12.0,
        semantic_available=True,
        semantic_score=0.81,
        n_with_numeric_drift=0,
        n_negation_mismatches=0,
    )
    monkeypatch.setattr(fidelity_paper, "grade_page", lambda *_args, **_kwargs: report)

    ctx = SimpleNamespace(
        attempt_id="attempt-1",
        paper_stem="example-2026-paper",
        pdf_filename="example.pdf",
        iteration=4,
    )
    conn = get_connection()
    persist_grades(ctx, conn)
    conn.close()

    [row] = read_attempt("attempt-1")
    assert row.grader_scores == {
        "n_claims": 3,
        "n_graded": 3,
        "mean_top1": 21.5,
        "median_top1": 20.0,
        "semantic_score": 0.81,
        "n_negation_mismatches": 0,
        "n_with_numeric_drift": 0,
    }
