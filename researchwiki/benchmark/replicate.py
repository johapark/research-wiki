"""Replicated author scoring — variance estimation for pipeline-change validation.

The benchmark-fixture harness in single-shot mode scores ONE wiki page against a
fixture. That's appropriate for "how does this committed page compare to
ground truth" but inadequate for "did this pipeline change actually move the
needle" because the agent's tournament-among-stance-varied-drafts produces
±10pp variance run-to-run on prose-heavy papers.

This driver isolates author-side variance. Reconcile + extract + crosslinks
+ semantic candidate proposal all run ONCE (deterministic given the source
PDF + S2 cache + paper-page index). The author phase runs N times against
that shared upstream state, each draft is scored independently, and the
report aggregates to mean / SD / min / max per axis. N=3 is usually enough
to detect lever effects ≥10pp; N=5 brings the SE of the mean down to ~4-5pp
which catches the L4-class effects that single-shot couldn't attribute.

Cost note: this is opt-in for methodology validation, not the day-to-day
ingest path. `researchwiki agent ingest <pdf>` stays one-shot — one PDF,
one wiki page, no replication. Replication is invoked via
`researchwiki benchmark-fixture <stem> --repeat N`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from ..agents import phases
from ..agents.model_config import for_phase
from ..agents.phases.draft import stance_for_slot
from ..paths import resolve_pdf
from .fixture import ContentFixture
from ..grade.scorer import ScoreReport, score_text


# Phases the replicate path exercises (author + upstream deterministic
# extractors) plus the judge invoked when --llm scoring is on. Recorded
# in ReplicateReport so config A/B outputs are self-describing.
_REPLICATE_PHASES = ("author", "reconcile", "target_claims", "link_generation")
_JUDGE_PHASE = "eval_judge"


def resolve_models(use_llm: bool) -> dict[str, str]:
    """Return {phase: 'model@provider'} for the phases this run touches.

    Includes eval_judge only when the judge will actually run. Failures on
    any single phase resolve to '?' rather than aborting — the point is
    diagnostics, not correctness."""
    out: dict[str, str] = {}
    phases_to_probe = list(_REPLICATE_PHASES) + ([_JUDGE_PHASE] if use_llm else [])
    for p in phases_to_probe:
        try:
            cfg = for_phase(p)
            out[p] = f"{cfg.model}@{cfg.provider}"
        except Exception as e:
            out[p] = f"?({type(e).__name__})"
    return out


@dataclass
class ReplicateRun:
    """One replicate's per-axis scores plus diagnostic metadata."""
    run_index: int
    stance: str
    temperature: float
    overall: float
    axes: dict[str, float]
    input_tokens: int
    output_tokens: int
    report: ScoreReport = field(repr=False)


@dataclass
class ReplicateReport:
    paper_stem: str
    n_runs: int
    use_llm: bool
    runs: list[ReplicateRun]
    mean: float
    sd: float
    min_overall: float
    max_overall: float
    axis_mean: dict[str, float]
    axis_sd: dict[str, float]
    # Resolved (phase → "model@provider") for every LLM-touched phase
    # this run exercised. Self-documents which config produced the report
    # so downstream A/Bs can label arms without external bookkeeping.
    resolved_models: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_stem": self.paper_stem,
            "n_runs": self.n_runs,
            "use_llm": self.use_llm,
            "mean": self.mean,
            "sd": self.sd,
            "min_overall": self.min_overall,
            "max_overall": self.max_overall,
            "axis_mean": self.axis_mean,
            "axis_sd": self.axis_sd,
            "resolved_models": self.resolved_models,
            "runs": [
                {
                    "run_index": r.run_index,
                    "stance": r.stance,
                    "temperature": r.temperature,
                    "overall": r.overall,
                    "axes": r.axes,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                }
                for r in self.runs
            ],
        }


def replicate_score(
    fixture: ContentFixture,
    *,
    n: int = 3,
    use_llm: bool = False,
    use_stub: bool = False,
    verbose: bool = True,
) -> ReplicateReport:
    """Run the author phase N times and score each draft against the fixture.

    Reconcile + extract + crosslinks run once and the result is reused for
    each author replicate. Stance cycles through `phases.DRAFT_STANCES`
    (balanced → skeptical → comprehensive → ...); temperature follows the
    runner's per-slot schedule (`min(1.0, 0.2 + 0.4 * slot)`).

    Returns a ReplicateReport with per-run scores, mean/SD/min/max overall,
    and per-axis mean/SD. The caller decides how to display.
    """
    try:
        pdf_path = resolve_pdf(fixture.paper_stem)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"cannot replicate: {e}") from None

    import sys as _sys
    def _v(msg: str) -> None:
        if verbose:
            print(msg, file=_sys.stderr)
    _v(f"[replicate] {fixture.paper_stem} — n={n} use_llm={use_llm}")
    _v(f"[replicate] reconcile + extract + crosslinks (once, shared)")

    # 1. Reconcile (once). Returns the metadata dict the runner uses.
    metadata = phases.reconcile_metadata(pdf_path, use_llm=not use_stub)

    # 2. Extract sections + full text (once).
    sections, full_text = phases.extract_sections(pdf_path)

    # 3. Crosslink candidates: citation-graph (S2/Crossref) + topical (semantic).
    citation_cands = phases.crosslink_candidates(pdf_path, metadata)
    excluded = frozenset(c.wikilink for c in citation_cands)
    try:
        topical_cands = phases.propose_crosslinks(
            metadata, sections, exclude_keys=excluded, use_stub=use_stub,
        )
    except Exception as e:
        _v(f"[replicate] propose_crosslinks failed ({e}); skipping topical")
        topical_cands = []
    candidates = citation_cands + topical_cands

    # 3.5. Target-claims extraction (L3) — once, shared across all
    # replicates. Failure is graceful (empty TargetClaimsOutput → author
    # prompt skips the block, behavior matches pre-L3).
    target_claims = None
    try:
        from ..agents.phases.target_claims import extract_target_claims
        target_claims = extract_target_claims(
            metadata=metadata, sections=sections,
            pdf_full_text=full_text, use_stub=use_stub,
        )
        if target_claims.is_empty() and target_claims.error:
            _v(f"[replicate] target_claims failed: {target_claims.error[:100]}")
        elif target_claims.is_empty():
            _v(f"[replicate] target_claims returned empty list")
        else:
            _v(f"[replicate] target_claims extracted {len(target_claims.claims)}")
    except Exception as e:
        _v(f"[replicate] target_claims raised ({e}); skipping")

    # 4. Author × N. Each replicate is one LLM call; reconcile/extract/
    # crosslinks/target_claims results are passed in unchanged.
    runs: list[ReplicateRun] = []
    for i in range(n):
        stance = stance_for_slot(i)
        temperature = min(1.0, 0.2 + 0.4 * i)
        _v(f"[replicate] author #{i + 1}/{n} stance={stance[0]} t={temperature:.1f}")
        draft = phases.author(
            metadata=metadata,
            sections=sections,
            temperature=temperature,
            candidates=candidates,
            target_claims=target_claims,
            stance=stance,
            pdf_full_text=full_text,
            use_stub=use_stub,
        )
        # Score this draft directly (no commit to disk).
        report = score_text(
            fixture,
            draft.text,
            page_path=f"(replicate run #{i + 1})",
            use_llm=use_llm,
        )
        runs.append(ReplicateRun(
            run_index=i,
            stance=stance[0],
            temperature=temperature,
            overall=report.overall_weighted_recall,
            axes={k: v.weighted_recall for k, v in report.axes.items()},
            input_tokens=draft.input_tokens,
            output_tokens=draft.output_tokens,
            report=report,
        ))
        _v(f"[replicate]   → overall {report.overall_weighted_recall:.1%} "
           f"({draft.input_tokens} in / {draft.output_tokens} out)")

    # 5. Aggregate.
    overalls = [r.overall for r in runs]
    mean = statistics.mean(overalls)
    sd = statistics.stdev(overalls) if len(overalls) > 1 else 0.0
    axis_names = list(runs[0].axes.keys()) if runs else []
    axis_mean = {
        ax: statistics.mean([r.axes[ax] for r in runs])
        for ax in axis_names
    }
    axis_sd = {
        ax: (statistics.stdev([r.axes[ax] for r in runs]) if len(runs) > 1 else 0.0)
        for ax in axis_names
    }

    return ReplicateReport(
        paper_stem=fixture.paper_stem,
        n_runs=n,
        use_llm=use_llm,
        runs=runs,
        mean=mean,
        sd=sd,
        min_overall=min(overalls) if overalls else 0.0,
        max_overall=max(overalls) if overalls else 0.0,
        axis_mean=axis_mean,
        axis_sd=axis_sd,
        resolved_models=resolve_models(use_llm=use_llm),
    )
