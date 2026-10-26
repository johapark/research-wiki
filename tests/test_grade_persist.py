"""Regression coverage for commit-time claim-grade telemetry."""

from types import SimpleNamespace

import pytest

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


def _score(text="A source claim", position=0):
    return fidelity_paper.ClaimScore(
        section="key_contributions", position=position, text=text, is_cross_ref=False,
        top1_score=10, top3_mean=5, top1_chunk_id=0, supporting_text="Source evidence",
        supporting_provenance="p. 1", numeric_tokens=[], numeric_unmatched=[],
        semantic_score=None, negation_mismatch=False, graded=True,
    )


def test_required_grade_persistence_cannot_silently_skip_an_unavailable_db(monkeypatch):
    def unavailable():
        raise OSError("database unavailable")

    monkeypatch.setattr(fidelity_paper, "_db_connection", unavailable)
    fidelity_paper._persist_scores("example", [_score()], embed_model=None)  # legacy best-effort callers
    with pytest.raises(OSError):
        fidelity_paper._persist_scores("example", [_score()], embed_model=None, strict=True)


def test_required_grade_persistence_rolls_back_if_a_claim_row_is_missing(tmp_path, monkeypatch):
    from researchwiki.db.rebuild import rebuild

    monkeypatch.chdir(tmp_path)
    page = tmp_path / "wiki/other/example.md"
    page.parent.mkdir(parents=True)
    page.write_text('---\ntitle: Example\ntype: paper\n---\n## Key Contributions\n- A source claim\n')
    rebuild()
    with pytest.raises(RuntimeError, match="Claim row missing"):
        fidelity_paper._persist_scores("example", [_score(), _score("Missing claim", 1)],
                                       embed_model=None, strict=True)
    conn = get_connection()
    try:
        row = conn.execute("SELECT last_graded_at FROM claims WHERE paper_stem = 'example'").fetchone()
        assert row is not None and row[0] is None
    finally:
        conn.close()


def test_commit_time_grader_requests_strict_persistence(monkeypatch):
    seen = []

    def fail(stem, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("persistence failed")

    monkeypatch.setattr(fidelity_paper, "grade_page", fail)
    ctx = SimpleNamespace(attempt_id="strict-test", paper_stem="example", pdf_filename="example.pdf", iteration=1)
    conn = get_connection()
    try:
        receipt = persist_grades(ctx, conn)
    finally:
        conn.close()
    assert seen == [{"persist": True, "require_persistence": True}]
    assert "error" in receipt


def test_commentary_has_no_claim_grade_projection(monkeypatch):
    monkeypatch.setattr(fidelity_paper, "grade_page", lambda *a, **k: pytest.fail("commentary must not enter claims DB"))
    ctx = SimpleNamespace(attempt_id="commentary-test", paper_stem="commentary", pdf_filename="commentary.pdf",
                          iteration=1, metadata={"page_type": "commentary"})
    conn = get_connection()
    try:
        receipt = persist_grades(ctx, conn)
    finally:
        conn.close()
    assert receipt == {"n_claims": 0, "n_graded": 0, "not_applicable": True}
    assert read_attempt("commentary-test")[0].decision == "not_applicable"
