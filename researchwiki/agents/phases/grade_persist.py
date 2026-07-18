"""Post-commit grading phase — run the per-paper fidelity grader on the
freshly-committed page and persist per-claim scores to the DB.

Distinct from `phases.grade` (the per-draft scorer used during the
tournament): this runs *after* commit, against the canonical wiki page,
and writes `claims.bm25_top1`, `semantic_score`, `negation_mismatch`,
`numeric_unmatched`, etc. into the structured DB so other read-side
surfaces (`researchwiki claims`, claim-id lookups in synthesis,
weak-page surfacing) have signal.

Failures are recorded in `ingest_iterations` but never fatal — the page
is already on disk and grading can be retried later via
`researchwiki grade paper <stem>` or `researchwiki grade regression`.

Lifted out of `runner.py` to keep that file focused on orchestration;
the function reads naturally as "the commit-time grader" rather than as
yet another phase wrapper threading state.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from ...db.iterations import write_iteration
from ...log import log

if TYPE_CHECKING:
    from ..context import Context


def grade_persist(ctx: "Context", conn) -> dict:
    """Grade the freshly-committed paper page and write per-claim scores.

    Returns a small summary dict (n_claims, n_graded, mean_top1,
    semantic_score) for the runner's print line. The substantive output
    is the persisted `claims` rows, not the return value.
    """
    from ...grade.fidelity.paper import grade_page

    t0 = time.time()
    try:
        report = grade_page(ctx.paper_stem, persist=True)
    except Exception as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        write_iteration(
            attempt_id=ctx.attempt_id,
            paper_stem=ctx.paper_stem,
            pdf_filename=ctx.pdf_filename,
            iteration=ctx.iteration,
            role="grade_persist",
            decision="error",
            decision_reason=f"{type(e).__name__}: {e}; elapsed={elapsed_ms}ms",
            model_used="(local)",
            conn=conn,
        )
        log(f"grade    ⚠ {type(e).__name__}: {e}", tag="agent")
        return {"error": str(e)}

    elapsed_ms = int((time.time() - t0) * 1000)
    summary = (
        f"n_claims={report.n_claims} graded={report.n_graded} "
        f"xref={report.n_cross_refs} mean_top1={report.mean_top1:.3f} "
        f"weakest={report.weakest_score:.3f}"
        if report.n_graded
        else f"n_claims={report.n_claims} graded=0 (cross-refs only or no claims)"
    )
    if report.semantic_available and report.semantic_score is not None:
        summary += f" semantic_score={report.semantic_score:.3f}"
    if report.n_with_numeric_drift:
        summary += f" numeric_drift={report.n_with_numeric_drift}"
    if report.n_negation_mismatches:
        summary += f" negation_mismatches={report.n_negation_mismatches}"

    write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="grade_persist",
        decision="persisted" if report.n_graded else "no_gradable_claims",
        decision_reason=f"{summary}; elapsed={elapsed_ms}ms",
        grader_scores=json.dumps({
            "n_claims": report.n_claims,
            "n_graded": report.n_graded,
            "mean_top1": report.mean_top1,
            "median_top1": report.median_top1,
            "semantic_score": report.semantic_score,
            "n_negation_mismatches": report.n_negation_mismatches,
            "n_with_numeric_drift": report.n_with_numeric_drift,
        }),
        model_used="(local)",
        conn=conn,
    )
    log(f"grade    → {summary}", tag="agent")
    return {
        "n_claims": report.n_claims,
        "n_graded": report.n_graded,
        "mean_top1": report.mean_top1,
        "semantic_score": report.semantic_score,
    }
