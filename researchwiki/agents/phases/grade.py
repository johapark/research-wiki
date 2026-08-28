"""Grade phase — Phase-1 grader on a draft.

Wraps `researchwiki.grade.fidelity.paper.grade_page` to produce both an
aggregate score dict (used by `tournament` for argmax) and a per-claim
detail list (used by `critic` and `debug` to identify weak / drifting
claims).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...benchmark.fixture import ContentFixture, FixtureItem
from ...grade.fidelity.paper import grade_page
from ...grade.scorer import IMPORTANCE_WEIGHTS, VERDICT_SCORES, score_text
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
    target_claims=None,
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
    n_anchor_sources = sal.n_anchor_sources if sal is not None else 0
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

    target_scores = _score_target_claims(
        stem=stem,
        draft_text=draft_text,
        target_claims=target_claims,
        use_semantic=use_semantic,
    )
    # Target claims come from a paper-wide extraction pass and therefore make
    # omissions actionable beyond the structurally extracted salience anchors.
    # Preserve the existing critic transport rather than introducing a second
    # gap channel. Exact misses only: partial coverage already affects fitness,
    # but should not automatically trigger additive prose.
    missed_anchors.extend(target_scores.get("missed_target_claims", []))

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
        "n_anchor_sources": n_anchor_sources,
        "n_anchors_matched": n_anchors_matched,
        "n_anchors_missed": n_anchors_missed,
        "missed_anchors": missed_anchors,
        **target_scores,
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


def _score_target_claims(
    *,
    stem: str,
    draft_text: str,
    target_claims,
    use_semantic: bool,
) -> dict:
    """Score the authored page against the extractor's triaged claim set.

    The shared fixture scorer already implements the desired importance
    weighting (critical=3, high=2, normal=1), numeric-integrity checks, and
    paraphrase-tolerant semantic matching. Reusing it keeps benchmark and live
    ingest semantics aligned instead of growing a second coverage heuristic.
    """
    claims = list(getattr(target_claims, "claims", None) or [])
    if not claims:
        return {}

    axes: dict[str, list[FixtureItem]] = {
        "headline": [],
        "capability": [],
        "limitation": [],
    }
    claim_by_id = {}
    location_families: set[str] = set()
    for position, claim in enumerate(claims):
        claim_type = getattr(claim, "type", "")
        importance = getattr(claim, "importance", "normal")
        content = (getattr(claim, "content", "") or "").strip()
        if claim_type not in axes or importance not in IMPORTANCE_WEIGHTS or not content:
            continue
        item_id = f"target-{position:03d}"
        location = getattr(claim, "location", None)
        axes[claim_type].append(FixtureItem(
            id=item_id,
            importance=importance,
            verbalization=content,
            location=location,
        ))
        claim_by_id[item_id] = claim
        family = _location_family(location)
        if family:
            location_families.add(family)

    n_valid = len(claim_by_id)
    if not n_valid:
        return {}

    fixture = ContentFixture(
        paper_stem=stem,
        paper_type="research",
        title="",
        notes="live target-claim evaluation",
        headline_claims=axes["headline"],
        capabilities=axes["capability"],
        limitations=axes["limitation"],
        related_papers=[],
    )
    report = score_text(fixture, draft_text, use_semantic=use_semantic)
    verdicts = [
        verdict
        for axis in ("headline_claims", "capabilities", "limitations")
        for verdict in report.axes[axis].items
    ]

    tier_recall: dict[str, float | None] = {}
    for tier, weight in IMPORTANCE_WEIGHTS.items():
        tier_items = [v for v in verdicts if v.importance == tier]
        possible = len(tier_items) * weight
        achieved = sum(VERDICT_SCORES[v.verdict] * weight for v in tier_items)
        tier_recall[tier] = (achieved / possible) if possible else None

    misses = []
    for verdict in verdicts:
        if verdict.verdict != "miss":
            continue
        claim = claim_by_id[verdict.item_id]
        misses.append({
            "axis": "target_claims",
            "id": verdict.item_id,
            "importance": verdict.importance,
            "text": (getattr(claim, "content", "") or "")[:_ANCHOR_TEXT_CAP],
            "location": getattr(claim, "location", None),
        })

    result = {
        "target_claim_score": report.overall_weighted_recall,
        "n_target_claims": n_valid,
        "n_target_claims_matched": sum(v.verdict == "match" for v in verdicts),
        "n_target_claims_partial": sum(v.verdict == "partial" for v in verdicts),
        "n_target_claims_missed": len(misses),
        "n_critical_target_claims": sum(v.importance == "critical" for v in verdicts),
        "n_critical_target_claims_missed": sum(
            v.importance == "critical" and v.verdict == "miss" for v in verdicts
        ),
        "target_claim_recall_by_importance": tier_recall,
        "critical_target_claim_recall": tier_recall["critical"],
        "high_target_claim_recall": tier_recall["high"],
        "normal_target_claim_recall": tier_recall["normal"],
        "missed_target_claims": misses,
    }
    # Preserve legacy/full confidence when the extractor supplied no usable
    # locations at all; a present value means the diversity was measurable.
    if location_families:
        result["n_target_claim_sources"] = len(location_families)
    return result


def _location_family(location: str | None) -> str | None:
    """Normalize a free-form claim location to a structural source family.

    Returns None when the location is absent *or* names no recognizable
    structure, which are treated identically on purpose. An earlier `other`
    bucket counted the unrecognizable case as one measurable family, and that
    inverted the incentive: an extractor emitting vague locations ("p. 4") was
    scored 0.5x by `target_claim_confidence`, while one omitting the field
    entirely kept full confidence — supplying weaker provenance cost more than
    supplying none. The multiplier is meant to reward spread across the paper's
    structure, so a location it cannot place must not count as spread.
    """
    value = (location or "").strip().lower()
    if not value:
        return None
    families = (
        ("abstract", ("abstract",)),
        ("introduction", ("introduction", "background")),
        ("methods", ("method", "methodology", "approach")),
        ("results", ("result", "experiment", "evaluation")),
        ("discussion", ("discussion", "conclusion", "limitation")),
        ("figures", ("figure", "fig.", "table")),
        ("supplement", ("extended data", "supplement", "appendix")),
    )
    for family, needles in families:
        if any(needle in value for needle in needles):
            return family
    return None
