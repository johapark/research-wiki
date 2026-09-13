"""Focused helpers for the ingest runner: early phases and budget teardown."""

from __future__ import annotations

import time

from ..db.iterations import write_iteration
from ..errors import EnvironmentFailure
from ..fsatomic import write_text_atomic
from ..log import log
from . import fitness, phases
from .budget import BudgetExhausted, BudgetTracker, IngestBudget


def usage_costs(result) -> dict[str, int]:
    """Map a phase result's token usage onto iteration cost columns."""
    return {
        "cost_input_tokens": result.input_tokens,
        "cost_output_tokens": result.output_tokens,
        "cost_cache_read_tokens": getattr(result, "cache_read_tokens", 0),
        "cost_cache_write_tokens": getattr(result, "cache_write_tokens", 0),
    }


def make_budget_tracker(*, max_model_calls=None, max_tokens=None,
                        max_cost_usd=None, max_wall_seconds=None):
    limits = IngestBudget(max_model_calls, max_tokens, max_cost_usd, max_wall_seconds)
    return BudgetTracker(limits) if limits.active() else None


def handle_budget_exhausted(ctx, conn, exc: BudgetExhausted) -> None:
    """Persist the terminal event and preserve the best graded partial."""
    graded = [draft for draft in ctx.drafts if getattr(draft, "scores", None)]
    partial = ctx.winner or (max(graded, key=fitness.tournament_key) if graded else None)
    if partial is not None:
        from .phases import _wrap_with_frontmatter
        ctx.sandbox_dir.mkdir(parents=True, exist_ok=True)
        stem = ctx.paper_stem or f"unknown-stem-{ctx.attempt_id[:8]}"
        partial_path = ctx.sandbox_dir / f"{stem}.budget-partial.md"
        write_text_atomic(partial_path, _wrap_with_frontmatter(partial.text, ctx.metadata, stem))
        ctx.committed_path = partial_path
        exc.partial_path = partial_path
    ctx.budget_exhausted = {
        "dimension": exc.dimension,
        "limit": exc.limit,
        "used_or_reserved": exc.used,
        "partial_artifact": str(ctx.committed_path) if ctx.committed_path else None,
        **exc.snapshot,
    }
    ctx.next_iter()
    write_iteration(
        attempt_id=ctx.attempt_id, paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename, iteration=ctx.iteration, role="budget",
        parent_iteration_id=(ctx.winner.iteration_id if ctx.winner else None),
        decision="budget-exhausted", decision_reason=str(exc),
        gate_metrics=ctx.budget_exhausted, conn=conn,
    )


def record_revision_decision(ctx, conn, draft, *, operator: str,
                             accepted: bool, writer=write_iteration) -> None:
    """Append the keep/reject verdict instead of rewriting the draft event."""
    ctx.next_iter()
    writer(
        attempt_id=ctx.attempt_id, paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename, iteration=ctx.iteration, role="tournament",
        parent_iteration_id=draft.iteration_id,
        grader_scores=draft.scores,
        decision="kept" if accepted else "discarded",
        decision_reason=f"operator={operator}; fitness={'improved' if accepted else 'not improved'}",
        gate_metrics={"revision_accepted": accepted, "operator": operator}, conn=conn,
    )


def record_timed_subphase(ctx, conn, *, role: str, started: float,
                          decision: str = "observed", reason: str = "",
                          writer=write_iteration, suspend_budget: bool = False) -> None:
    """Append a nested commit-subphase duration event.

    These rows make commit internals visible; reports mark them nested so they
    are never added to the parent commit timer when computing attempt totals.
    """
    if suspend_budget and ctx.budget_tracker is not None:
        ctx.budget_tracker.suspend()
    ctx.next_iter()
    writer(
        attempt_id=ctx.attempt_id, paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename, iteration=ctx.iteration, role=role,
        decision=decision, decision_reason=reason,
        duration_ms=int((time.monotonic() - started) * 1000), conn=conn,
    )


def finalize_attempt_timing(ctx, conn, *, started: float,
                            error: BaseException | None,
                            writer=write_iteration) -> None:
    """Best-effort terminal wall timer, independent of phase-work totals."""
    if ctx.budget_exhausted:
        decision = "budget-exhausted"
    elif error is None:
        decision = "completed"
    elif isinstance(error, (KeyboardInterrupt, SystemExit)):
        decision = "interrupted"
    else:
        decision = "failed"
    reason = "all phases returned" if error is None else f"{type(error).__name__}: {error}"
    # Do not call next_iter(): an exhausted wall budget must not suppress its
    # own terminal telemetry row.
    ctx.iteration += 1
    try:
        writer(
            attempt_id=ctx.attempt_id, paper_stem=ctx.paper_stem,
            pdf_filename=ctx.pdf_filename, iteration=ctx.iteration, role="attempt",
            decision=decision, decision_reason=reason[:500],
            duration_ms=int((time.monotonic() - started) * 1000), conn=conn,
        )
    except Exception as exc:
        log(f"timing   ⚠ could not persist terminal attempt timer: {exc}", tag="agent")


def keyword_body_gaps(keywords: list[str], body_text: str) -> list[str]:
    body_lc = body_text.lower()
    missing = []
    for keyword in keywords:
        first = (keyword.split() or [""])[0].lower().strip(".,;:()[]{}")
        if first and first not in body_lc:
            missing.append(keyword)
    return missing


def prepare_keywords(ctx, conn, cleaned_text: str, *, writer=write_iteration):
    """Reuse target-pass keywords, falling back to one dedicated source call."""
    ctx.next_iter()
    started = time.monotonic()
    extracted = list(getattr(ctx.target_claims, "keywords", None) or [])
    if len(extracted) >= phases.MIN_KEYWORDS:
        out = phases.KeywordsOutput(
            keywords=extracted,
            model=getattr(ctx.target_claims, "model", "") or "(target_claims)",
        )
        reason = "reused target-claims extraction"
    else:
        out = phases.propose_keywords(
            metadata=ctx.metadata, draft_text=cleaned_text,
            sections=ctx.sections, full_pdf_text=ctx.pdf_full_text,
            use_stub=ctx.use_stub,
        )
        reason = "fallback keyword extraction"
    writer(
        attempt_id=ctx.attempt_id, paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename, iteration=ctx.iteration, role="keywords",
        decision="kept" if out.keywords else "rejected",
        decision_reason=f"{reason}: {out.keywords!r}", model_used=out.model,
        **usage_costs(out), duration_ms=int((time.monotonic() - started) * 1000),
        conn=conn,
    )
    log(f"keywords  → {out.keywords}", tag="agent")
    missing = keyword_body_gaps(out.keywords, cleaned_text) if out.keywords else []
    if missing:
        log(f"kw-coverage → {len(out.keywords) - len(missing)}/{len(out.keywords)} "
            f"keywords appear in body; missing: {missing[:5]}", tag="agent")
    elif out.keywords:
        log(f"kw-coverage → all {len(out.keywords)} keywords appear in body", tag="agent")
    return out


def find_exact_duplicate(ctx, prior_stem: str | None):
    """Return ``(stem, page_path)`` for an identical deposited PDF."""
    from ..fsatomic import file_sha256
    from ..paths import papers_dir
    from ..wiki import find_stem_collision

    existing_stem = (
        ctx.paper_stem
        if ctx.allow_rename and prior_stem and prior_stem != ctx.paper_stem
        else prior_stem or ctx.paper_stem
    )
    if not existing_stem:
        return None
    canonical_pdf = papers_dir() / f"{existing_stem}.pdf"
    try:
        if (not canonical_pdf.exists()
                or ctx.pdf_path.resolve() == canonical_pdf.resolve()
                or file_sha256(ctx.pdf_path) != file_sha256(canonical_pdf)):
            return None
    except OSError:
        return None
    page = find_stem_collision(existing_stem)
    return (existing_stem, page) if page is not None else None


_FINDING_SECTIONS = ("results", "discussion", "conclusion", "findings")


def warn_thin_extraction(sections: dict, full_text: str | None = None) -> None:
    if full_text:
        from ..pdf.sections import assess_section_health
        health = assess_section_health(full_text, sections)
        if not health.healthy:
            log(
                "extract ⚠ unhealthy section structure "
                f"({', '.join(health.reasons)}; "
                f"intro={health.introduction_fraction:.0%}) — using "
                "document-stratified prompt fallback",
                tag="agent",
            )
            return
    names = {str(key).lower() for key in (sections or {})}
    if not names:
        log("extract ⚠ no sections recovered at all — the page will rest on the "
            "abstract and full-text retrieval alone", tag="agent")
    elif not any(any(finding in name for finding in _FINDING_SECTIONS) for name in names):
        log(f"extract ⚠ no findings section recovered (got {sorted(names)}) — "
            "claims will be graded against full text only, so treat the Results "
            "section of the draft with extra care", tag="agent")


def phase_reconcile(ctx, conn):
    t0 = time.monotonic()
    meta = phases.reconcile_metadata(
        ctx.pdf_path, doi_override=ctx.doi_override, title_override=ctx.title_override,
        year_override=ctx.year_override, authors_override=ctx.authors_override,
        use_llm=ctx.use_llm_reconcile,
    )
    usage = meta.pop("_reconcile_usage", {})
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    iteration_id = write_iteration(
        attempt_id=ctx.attempt_id, pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration, role="reconcile", paper_stem=meta.get("stem"),
        decision="observed",
        decision_reason=f"reconciled in {elapsed_ms}ms; sources={meta.get('sources', [])}",
        critic_notes=str(meta), duration_ms=elapsed_ms,
        model_used=usage.get("model"),
        cost_input_tokens=usage.get("input_tokens", 0),
        cost_output_tokens=usage.get("output_tokens", 0),
        cost_cache_read_tokens=usage.get("cache_read_tokens", 0),
        cost_cache_write_tokens=usage.get("cache_write_tokens", 0),
        conn=conn,
    )
    return meta, iteration_id


def phase_extract(ctx, conn):
    t0 = time.monotonic()
    sections, full_text = phases.extract_sections(ctx.pdf_path)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    from ..pdf.sections import assess_section_health
    health = assess_section_health(full_text, sections)
    write_iteration(
        attempt_id=ctx.attempt_id, paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename, iteration=ctx.iteration, role="extract",
        decision="observed",
        decision_reason=(f"extracted in {elapsed_ms}ms; sections={list(sections)}; "
                         f"full_text_chars={len(full_text)}; "
                         f"healthy={health.healthy}; reasons={list(health.reasons)}"),
        duration_ms=elapsed_ms, conn=conn,
    )
    return sections, full_text


def phase_target_claims(ctx, conn):
    from .phases.target_claims import extract_target_claims
    t0 = time.monotonic()
    out = extract_target_claims(
        metadata=ctx.metadata or {}, sections=ctx.sections or {},
        pdf_full_text=ctx.pdf_full_text, use_stub=ctx.use_stub,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if out.error:
        decision, reason = "error", f"{out.error[:200]}; elapsed={elapsed_ms}ms"
    elif out.is_empty():
        decision, reason = "empty", f"no claims extracted; elapsed={elapsed_ms}ms"
    else:
        critical = sum(claim.importance == "critical" for claim in out.claims)
        high = sum(claim.importance == "high" for claim in out.claims)
        decision = "extracted"
        reason = f"n={len(out.claims)} (critical={critical}, high={high}); elapsed={elapsed_ms}ms"
    write_iteration(
        attempt_id=ctx.attempt_id, paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename, iteration=ctx.iteration, role="target_claims",
        decision=decision, decision_reason=reason, model_used=out.model or "(no calls)",
        cost_input_tokens=out.input_tokens, cost_output_tokens=out.output_tokens,
        cost_cache_read_tokens=out.cache_read_tokens,
        cost_cache_write_tokens=out.cache_write_tokens,
        duration_ms=elapsed_ms,
        gate_metrics={
            "target_claims": len(out.claims),
            "keywords": len(getattr(out, "keywords", None) or []),
            "error": bool(out.error),
            "empty": out.is_empty(),
        },
        conn=conn,
    )
    return out


def run_entailment_check(ctx, conn, cleaned_text: str) -> None:
    """Run the explicitly requested support veto against the final draft.

    Provider/environment failures propagate. Malformed or incomplete output is
    recorded as a hard gate failure, so a requested verification can never
    silently turn into an unchecked promotion.
    """
    from ..grade.support import llm_support_classifier

    started = time.monotonic()
    try:
        support_scores, _ = phases.grade_draft(
            stem=ctx.paper_stem,
            draft_text=cleaned_text,
            metadata=ctx.metadata,
            sandbox_dir=ctx.sandbox_dir,
            pdf_path=ctx.pdf_path,
            use_semantic=ctx.use_semantic,
            support_classifier=llm_support_classifier,
        )
        n_unsupported = support_scores.get("n_unsupported", 0)
        n_checked = support_scores.get("n_support_checked", 0)
        unsupported = support_scores.get("unsupported_claims", [])
        ctx.winner.scores["support_check_complete"] = support_scores.get(
            "support_check_complete", False
        )
        ctx.winner.scores["n_support_expected"] = support_scores.get(
            "n_support_expected", n_checked
        )
        ctx.winner.scores["n_unsupported"] = n_unsupported
        ctx.winner.scores["n_support_checked"] = n_checked
        ctx.winner.scores["unsupported_claims"] = unsupported
        log(f"support  → {n_unsupported} unsupported / {n_checked} checked", tag="agent")
        for claim in unsupported:
            log(f"           ✗ [{claim['section']}] {claim['text'][:80]}", tag="agent")
        ctx.next_iter()
        write_iteration(
            attempt_id=ctx.attempt_id,
            paper_stem=ctx.paper_stem,
            pdf_filename=ctx.pdf_filename,
            iteration=ctx.iteration,
            role="claim_support",
            parent_iteration_id=ctx.winner.iteration_id,
            grader_scores={
                "support_check_complete": support_scores.get(
                    "support_check_complete", False
                ),
                "n_support_expected": support_scores.get(
                    "n_support_expected", n_checked
                ),
                "n_unsupported": n_unsupported,
                "n_support_checked": n_checked,
                "unsupported_claims": unsupported,
            },
            decision="observed",
            decision_reason=f"{n_unsupported} unsupported / {n_checked} checked",
            duration_ms=int((time.monotonic() - started) * 1000),
            gate_metrics={
                "unsupported_claims": n_unsupported,
                "claims_checked": n_checked,
            },
            conn=conn,
        )
    except EnvironmentFailure:
        raise
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        ctx.winner.scores["support_check_failed"] = True
        ctx.winner.scores["support_check_error"] = message
        ctx.winner.scores["support_check_complete"] = False
        log(f"support  → check failed; promotion vetoed: {message}", tag="agent")
        ctx.next_iter()
        try:
            write_iteration(
                attempt_id=ctx.attempt_id,
                paper_stem=ctx.paper_stem,
                pdf_filename=ctx.pdf_filename,
                iteration=ctx.iteration,
                role="claim_support",
                parent_iteration_id=ctx.winner.iteration_id,
                grader_scores={
                    "support_check_failed": True,
                    "support_check_complete": False,
                    "support_check_error": message,
                },
                decision="failed",
                decision_reason=message[:500],
                duration_ms=int((time.monotonic() - started) * 1000),
                gate_metrics={"verification_failed": True},
                conn=conn,
            )
        except Exception as telemetry_exc:
            # The safety outcome already lives on the winner and must reach the
            # promotion gate even if optional failure telemetry cannot be saved.
            log(f"support  ⚠ could not persist failure telemetry: {telemetry_exc}",
                tag="agent")


def run_post_promote_memory_evolution(ctx, conn, *, source_key: str) -> None:
    """Re-arm the budget for optional memory evolution, then pause it again.

    A promoted page is already canonical, so exhaustion here is a recorded
    optional skip rather than a terminal partial-ingest failure. Required
    maintenance runs with enforcement suspended so the paper cannot remain
    half-maintained when promotion itself crosses the wall deadline.

    `EnvironmentFailure` is skipped on the same terms, and for a sharper reason
    than budget: by the time this runs, the page, the PDF move, the back-links,
    the `index.md` bullet and the `log.md` entry have all landed. Letting a
    provider outage or an unanswered chat-relay prompt escape here made the
    worker exit 2, which `_should_retry` treats as retryable — and since the PDF
    has already left `inbox/`, `--resume` then filed a complete, twice-gated
    paper as `unresumable` and advised deleting the page as a half-landed
    promote. House rule 2 in `errors.py`.
    """
    if ctx.budget_tracker is not None:
        ctx.budget_tracker.resume()
    try:
        ctx.next_iter()
        phases.evolve_memory(ctx, conn, source_key=source_key)
    except BudgetExhausted as exc:
        if ctx.budget_tracker is not None:
            ctx.budget_tracker.suspend()
        ctx.next_iter()
        write_iteration(
            attempt_id=ctx.attempt_id, paper_stem=ctx.paper_stem,
            pdf_filename=ctx.pdf_filename, iteration=ctx.iteration,
            role="memory_evolve", decision="skipped",
            decision_reason=f"post-promotion budget exhausted: {exc}",
            gate_metrics={"budget_dimension": exc.dimension, **exc.snapshot},
            conn=conn,
        )
        log(f"evolve   → skipped ({exc})", tag="agent")
    except EnvironmentFailure as exc:
        if ctx.budget_tracker is not None:
            ctx.budget_tracker.suspend()
        ctx.next_iter()
        write_iteration(
            attempt_id=ctx.attempt_id, paper_stem=ctx.paper_stem,
            pdf_filename=ctx.pdf_filename, iteration=ctx.iteration,
            role="memory_evolve", decision="skipped",
            decision_reason=(
                f"post-promotion environment failure: "
                f"{type(exc).__name__}: {exc}"
            ),
            conn=conn,
        )
        log(f"evolve   → skipped ({type(exc).__name__}: {exc})", tag="agent")
        log("evolve   → the page is fully promoted; re-run "
            "`researchwiki evolve` when the provider is reachable", tag="agent")
    finally:
        if ctx.budget_tracker is not None:
            ctx.budget_tracker.suspend()
