"""Derived-cache and audit helpers for the agent commit phase."""

from __future__ import annotations

from ..log import log
from ..paths import index_path, log_path, wiki_dir


def promotion_decision_reason(result) -> str:
    """Stable audit string for both successful and failed promotion results."""
    reason = (
        f"category={result.category}; wiki={result.wiki_path}; "
        f"pdf={result.pdf_path}; backlinks_added={result.backlinks_added}; "
        f"index_updated={result.index_updated}; log_appended={result.log_appended}"
    )
    if result.warnings:
        reason += f"; warnings={result.warnings}"
    return reason


def update_indexes_after_promotion(result) -> dict | None:
    """Upsert every parseable wiki page changed by a successful promotion.

    Indexes are derived caches, so failure is recorded as a recoverable warning
    rather than rolling back a valid page/PDF commit.
    """
    try:
        from ..index.incremental import update_page_indexes

        changed_pages = [result.wiki_path]
        changed_pages.extend(
            wiki_dir() / f"{key}.md" for key in result.backlinks_added
        )
        if result.index_updated:
            changed_pages.append(index_path())
        if result.log_appended:
            changed_pages.append(log_path())
        outcome = update_page_indexes(changed_pages)
        log(
            f"indexes  → {outcome['n_pages']} page(s) "
            f"(bm25={outcome['bm25_mode']}, semantic={outcome['semantic_mode']})",
            tag="agent",
        )
        return outcome
    except Exception as exc:
        warning = f"incremental index update failed: {exc}; run researchwiki reindex"
        result.warnings.append(warning)
        log(f"indexes  ⚠ {warning}", tag="agent")
        return None


def evaluate_promotion_gate(ctx, conn, cleaned_text, n_kc, verification, *, debug, entailment):
    """Apply promotion modes, bounded structural repairs and the final support veto.

    Callbacks preserve the runner's observed DEBUG phase and entailment boundary.
    Omission observations stay in the enclosing commit capture scope.
    """
    from . import phases, promote as promote_mod
    from ..provenance import specific_author_model
    from ..records.lineage import note_phase_omission

    # Promotion-mode dispatch.
    if ctx.promote_mode == "never":
        note_phase_omission("debug", "forced_sandbox",
                            {"promotion_mode": ctx.promote_mode})
        gate = promote_mod.GateResult(
            promoted=False, reasons=["--force-sandbox flag set"]
        )
    else:
        gate = promote_mod.should_auto_promote(
            scores=ctx.winner.scores,
            verification=verification,
            n_key_contributions=n_kc,
            paper_type=(ctx.metadata or {}).get("paper_type", "research"),
            commentary_signals=(ctx.metadata or {}).get("commentary_signals"),
        )
        if gate.promoted:
            reason = ("gate_passed_before_entailment_check" if ctx.verify_claim_entailment
                      else "promotion_gate_passed")
            note_phase_omission("debug", reason,
                                {"promotion_mode": ctx.promote_mode,
                                 "entailment_check_enabled": ctx.verify_claim_entailment})
        # DEBUG operator: when the gate rejects on a structural issue
        # (numeric drift, too few KC bullets, too few graded claims) we
        # try up to `max_debug` targeted repair passes before falling back
        # to sandbox. Skipped under --auto-promote (caller already overrode
        # the gate) and --force-sandbox (gate is informational only there).
        if not gate.promoted and ctx.promote_mode == "auto":
            if ctx.max_debug <= 0:
                note_phase_omission("debug", "debug_budget_zero",
                                    {"max_debug": ctx.max_debug})
            for _ in range(ctx.max_debug):
                if gate.promoted:
                    note_phase_omission("debug", "gate_passed_after_repair",
                                        {"max_debug": ctx.max_debug})
                    break
                structural_issues = phases.detect_structural_gate_issues(gate.reasons)
                if not structural_issues:
                    note_phase_omission("debug", "no_structural_gate_issues",
                                        {"max_debug": ctx.max_debug})
                    break
                prior_gate = gate
                cleaned_text, n_kc, gate, verification = debug(
                    ctx, conn, cleaned_text, n_kc, gate, verification,
                    issues=structural_issues,
                )
                if gate is prior_gate:
                    # _phase_debug reverted (repair didn't strictly improve)
                    # and handed back the same gate object — another pass
                    # would re-run the identical repair against the same
                    # winner. Stop burning the budget.
                    note_phase_omission("debug", "repair_did_not_improve",
                                        {"max_debug": ctx.max_debug})
                    break
            else:
                if ctx.max_debug > 0:
                    note_phase_omission("debug", "debug_budget_exhausted",
                                        {"max_debug": ctx.max_debug})
        elif not gate.promoted:
            note_phase_omission("debug", "debug_bypassed_by_promotion_mode",
                                {"promotion_mode": ctx.promote_mode})

        # Opt-in per-claim entailment veto (grade.support). Off unless
        # --verify-claim-entailment is set. Runs on the FINAL winner/text —
        # after any DEBUG repair has settled — then re-evaluates the gate so the
        # veto lands on the draft that actually gets promoted. (Running it
        # before DEBUG let a successful repair rebuild the winner's scores and
        # silently drop the n_unsupported veto.) It's the qualitative analogue
        # of the numeric-drift veto that BM25/semantic similarity can't catch.
        if ctx.verify_claim_entailment:
            entailment(ctx, conn, cleaned_text)
            gate = promote_mod.should_auto_promote(
                scores=ctx.winner.scores,
                verification=verification,
                n_key_contributions=n_kc,
                paper_type=(ctx.metadata or {}).get("paper_type", "research"),
                commentary_signals=(ctx.metadata or {}).get("commentary_signals"),
            )

        if ctx.promote_mode == "always" and not gate.promoted:
            log(f"promote  → --auto-promote forced over failed gates: {gate.reasons}", tag="agent")
            gate = promote_mod.GateResult(
                promoted=True, reasons=gate.reasons, warnings=gate.warnings,
            )

    # `promote_to_wiki` refuses a page it cannot attribute, so no mode —
    # `--auto-promote` included — can land one. Decide that here, where the
    # sandbox branch keeps the paid draft for review and the run exits 1 as a
    # review outcome. Reaching promote instead raised PromoteFailed: exit 2,
    # which batch `--resume` retries by re-running the whole paid pipeline into
    # the same refusal. Preflight normally stops a config like this before the
    # first call; this catches what it cannot see.
    if gate.promoted:
        winner_model = ctx.winner.model if ctx.winner else None
        if not specific_author_model(winner_model):
            gate = promote_mod.GateResult(
                promoted=False,
                reasons=[
                    f"author_model: winning draft's model {winner_model!r} is "
                    "not an exact model id, so the page cannot record which "
                    "model authored it"
                ],
                warnings=gate.warnings,
            )

    return cleaned_text, n_kc, gate, verification
