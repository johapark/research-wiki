"""Optional visual phase telemetry, immutable evidence packet and publication veto."""
from __future__ import annotations

from dataclasses import asdict
import time

from ..db.iterations import write_iteration
from ..fsatomic import write_json_atomic
from ..log import log
from ..operations.lifecycle import observed_phase
from ..provenance import specific_author_model
from .visual_evidence import apply_evidence


@observed_phase('visual_inspect')
def phase_visual_inspect(ctx, conn) -> None:
    from .phases.visuals import inspect_visuals
    ctx.next_iter()
    started = time.monotonic()
    result = inspect_visuals(pdf_path=ctx.pdf_path, pages=ctx.source_pages,
        sections=ctx.visual_source_sections, inventory=ctx.visual_inventory,
        target_claims=ctx.target_claims, use_stub=ctx.use_stub, expected_source_hash=ctx.visual_source_hash)
    ctx.visual_result = result
    apply_evidence(ctx.target_claims, result)
    write_iteration(attempt_id=ctx.attempt_id, paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename, iteration=ctx.iteration, role='visual_inspect',
        decision=result.status, decision_reason=result.error or result.status,
        model_used=result.model or '(no calls)', cost_input_tokens=result.input_tokens,
        cost_output_tokens=result.output_tokens, gate_metrics=result.metrics(),
        duration_ms=int((time.monotonic() - started) * 1000), conn=conn)
    log(f'visual → {result.status}; pages={len({item["page"] for item in result.selected})}; '
        f'critical unresolved={result.critical_unresolved}; cache={result.cache_hit}', tag='agent')
    ctx.sandbox_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(ctx.sandbox_dir / f'{ctx.paper_stem}-{ctx.attempt_id}.visual.json', asdict(result))


def final_publication_gate(ctx, gate):
    """Final veto survives DEBUG score replacement and forced promotion."""
    from .promote import GateResult
    if gate.promoted:
        winner_model = ctx.winner.model if ctx.winner else None
        if not specific_author_model(winner_model):
            gate = GateResult(False, reasons=[
                f"author_model: winning draft's model {winner_model!r} is not an exact model id, "
                'so the page cannot record which model authored it'], warnings=gate.warnings)
    if getattr(ctx, 'visual_inspection', False):
        result = getattr(ctx, 'visual_result', None)
        metrics = result.metrics() if result else {'visual_status': 'incomplete'}
        if ctx.winner:
            ctx.winner.scores['visual_coverage'] = metrics
        reasons = []
        if result is None or result.selection_status != 'complete':
            reasons.append('visual selection incomplete; source-image review required')
        if result and result.error:
            reasons.append('visual inspection incomplete: ' + result.error)
        if result and result.critical_unresolved:
            reasons.append('critical visual evidence requires source-image review: ' +
                           ', '.join(result.critical_unresolved))
        warnings = list(gate.warnings)
        if result and (result.unselected or result.unresolved):
            warnings.append('Visual coverage is selective; see the .visual.json review receipt.')
        gate = GateResult(gate.promoted and not reasons, reasons=list(gate.reasons) + reasons,
                          warnings=warnings)
    return gate


def visual_commit_metrics(ctx) -> dict:
    result = getattr(ctx, 'visual_result', None)
    return result.metrics() if result is not None else {}
