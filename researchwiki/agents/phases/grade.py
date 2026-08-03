"""Grade phase — Phase-1 grader on a draft.

Wraps `researchwiki.grade.fidelity.paper.grade_page` to produce both an
aggregate score dict (used by `tournament` for argmax) and a per-claim
detail list (used by `critic` and `debug` to identify weak / drifting
claims).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...grade.fidelity.paper import grade_page
from ...grade.support import Classifier, check_support, unsupported_claims
from .draft import _wrap_with_frontmatter

# Per-anchor text budget for the `missed_anchors` entries carried in the
# aggregate. The list is already capped at `salience._TOP_K_MISSED` entries, so
# this bounds one grade row's `grader_scores` JSON at a few KB. Generous enough
# that the critic sees a whole abstract sentence (the longest useful anchors run
# ~600 chars); anything longer is a run-on the sentence splitter failed on.
_ANCHOR_TEXT_CAP = 700


@dataclass
class ClaimDetail:
    """Per-claim grader output, used by the critic to spot weak claims."""
    section: str
    position: int
    text: str
    bm25: float
    semantic: float | None
    negation_mismatch: bool
    numeric_unmatched: list[str]

    def is_weak(self) -> bool:
        """A claim is weak if it has unmatched numbers, a flagged negation
        mismatch, a low semantic-similarity score, or BM25 + semantic both
        below their respective floors (drift signal that semantic alone
        misses).

        Thresholds:
          - semantic < 0.40 alone — the critic should see more weak claims
            than the promote-gate hard-fails on (0.55), so the author has
            a chance to revise them in the evolve loop.
          - BM25 < 8 AND semantic < 0.55 — drift signal. BM25 alone would
            over-flag legitimate paraphrases (high semantic, low surface
            overlap); conjoining with semantic separates "paraphrase too
            far" from "honest paraphrase." 8 is well below the typical
            supported-claim BM25 range; matches the empirical floor where
            agent-page Limitations sections drift on existing fixtures.
        """
        if self.numeric_unmatched:
            return True
        if self.negation_mismatch:
            return True
        if self.semantic is not None and self.semantic < 0.40:
            return True
        if self.bm25 < 8.0 and (self.semantic or 0.0) < 0.55:
            return True
        return False


def grade_draft(
    *,
    stem: str | None,
    draft_text: str,
    metadata: dict,
    sandbox_dir: Path,
    pdf_path: Path | str | None = None,
    use_semantic: bool = True,
    support_classifier: Classifier | None = None,
) -> tuple[dict, list[ClaimDetail]]:
    """Grade a draft via the Phase 1 grader.

    Returns (aggregate_scores, claim_details). The aggregate is for tournament
    selection; the claim_details list is for the critic phase, which needs
    per-claim weak-spot identification rather than a page-level summary.

    `pdf_path` is the agent's source PDF (may be in inbox/, papers/, or
    anywhere else); the grader's per-PDF index is built from that path
    rather than from `papers/{stem}.pdf` so the agent works pre-ingest.

    `use_semantic` toggles the bi-encoder semantic scorer. The semantic
    signal is load-bearing for tournament selection (BM25 alone proved
    insufficient in earlier eval); skip only for offline / no-deps tests.
    """
    if not stem:
        return {"error": "no stem", "mean_bm25": 0.0, "semantic_score": None}, []

    sandbox_dir.mkdir(parents=True, exist_ok=True)
    temp_path = sandbox_dir / f"{stem}-attempt-grade.md"
    temp_path.write_text(_wrap_with_frontmatter(draft_text, metadata, stem), encoding="utf-8")

    report = grade_page(
        stem=stem,
        page_path=str(temp_path),
        semantic=use_semantic,
        persist=False,
        pdf_path=pdf_path,
    )

    # Salience: PDF-anchor recall surfaced from grade.fidelity.paper. Lifted into the
    # aggregate so the tournament fitness can use it as a co-primary signal
    # alongside semantic_score. Falls back to None / 0 when --no-salience is in
    # effect or no anchors are recoverable from the PDF structure.
    sal = report.salience
    salience_score = sal.salience_score if sal is not None else None
    n_anchors = sal.n_anchors if sal is not None else 0
    n_anchors_matched = sal.n_match if sal is not None else 0
    n_anchors_missed = sal.n_miss if sal is not None else 0
    # The missed anchors themselves — not just the count. Without these the
    # recall axis is measurable but not *actionable*: the critic can only flag
    # claims that ARE on the page (every `ClaimDetail.is_weak()` predicate is a
    # precision test), so an omission produced no revision signal and the
    # evolve loop broke on "no weak claims". Carried in the aggregate rather
    # than a new return value because the aggregate already travels to the
    # critic on `draft.scores` and is persisted as `grader_scores` — which also
    # makes "which anchors do we systematically miss?" answerable after the
    # fact. Text is truncated: the critic prompt truncates anyway, and this
    # lands in every grade row in the DB.
    missed_anchors = [
        {**m, "text": (m.get("text") or "")[:_ANCHOR_TEXT_CAP]}
        for m in (sal.missed_anchors if sal is not None else [])
    ]

    aggregate = {
        "n_claims": report.n_claims,
        "n_graded": report.n_graded,
        "mean_bm25": report.mean_top1,
        "median_bm25": report.median_top1,
        "semantic_score": report.semantic_score,
        "semantic_median": report.semantic_median,
        "n_negation_mismatches": report.n_negation_mismatches,
        "n_drift": report.n_with_numeric_drift,
        "weakest_score": report.weakest_score,
        "semantic_available": report.semantic_available,
        "salience_score": salience_score,
        "n_anchors": n_anchors,
        "n_anchors_matched": n_anchors_matched,
        "n_anchors_missed": n_anchors_missed,
        "missed_anchors": missed_anchors,
    }
    details = [
        ClaimDetail(
            section=c.section,
            position=c.position,
            text=c.text,
            bm25=c.top1_score,
            semantic=c.semantic_score,
            negation_mismatch=c.negation_mismatch,
            numeric_unmatched=list(c.numeric_unmatched or []),
        )
        for c in report.claims
        if c.graded
    ]

    # Opt-in per-claim support (entailment) check. Off by default: no classifier
    # → no LLM call and no `n_unsupported` key, so the promote-gate veto stays
    # inert. When supplied, judges each graded claim against its already-
    # retrieved top-1 chunk (`supporting_text`) and surfaces both the count and
    # the identity of the unsupported claims, so a reviewer sees *which* claims
    # the source doesn't entail.
    if support_classifier is not None:
        graded_claims = [
            (c.section, c.position, c.text, c.supporting_text)
            for c in report.claims
            if c.graded
        ]
        supports = check_support(graded_claims, support_classifier)
        unsupported = unsupported_claims(supports)
        aggregate["n_support_checked"] = len(supports)
        aggregate["n_unsupported"] = len(unsupported)
        aggregate["unsupported_claims"] = [
            {"section": s.section, "position": s.position, "text": s.text}
            for s in unsupported
        ]

    return aggregate, details
