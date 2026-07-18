"""Memory-evolution phase — propose edits to existing synthesis pages
based on a freshly-ingested paper.

Runs after a successful promote_to_wiki: finds top-k semantic neighbors
among synthesis pages, applies a cosine-similarity prefilter, and writes
actionable proposals as markdown to
`.ingest/{stem}-evolution-proposals/`. Never auto-applies — the human
reviews the directory.

Named `memory_evolve` to disambiguate from the agent's draft-revision
`evolve` (which revises a draft based on critic notes — same word,
different concept). The iteration log uses `role="memory_evolve"` for
the same reason.

The cosine prefilter is the cost-saving knob. KNN is free; the LLM call
per neighbor is not. Skipping a candidate with cos < 0.65 saves ~$0.005
and almost never costs us a real edit (low-cosine pairs almost always
return "none" anyway).

Lifted out of `runner.py` because the body is substantive (~80 lines of
filesystem + iteration logging) rather than a thin wrapper. Sits
naturally alongside `phases.evolution` (the underlying proposal
generator) — this module is the runner-side orchestration that turns
proposals into on-disk markdown files.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ...db.iterations import write_iteration
from ...fsatomic import write_text_atomic
from ...paths import ingest_dir
from .evolution import propose_evolution, render_proposal_md
from ...log import log

if TYPE_CHECKING:
    from ..context import Context


def evolve_memory(ctx: "Context", conn, *, source_key: str) -> dict:
    """Propose neighbor-page edits and write actionable proposals to disk.

    Returns the stats dict from `propose_evolution` augmented with the
    list of files actually written. Stub mode skips entirely (proposal
    generation hits the LLM in current shape, and stub-mode runs assume
    zero network).
    """
    if ctx.use_stub:
        return {"skipped": "stub-mode"}

    t0 = time.time()
    try:
        proposals, stats = propose_evolution(source_key)
    except FileNotFoundError as e:
        log(f"evolve   ⚠ {e}", tag="agent")
        return {"error": str(e)}
    elapsed_ms = int((time.time() - t0) * 1000)

    actionable = [p for p in proposals if p.is_actionable()]
    proposal_dir = ingest_dir() / f"{ctx.paper_stem}-evolution-proposals"
    written: list[str] = []
    if actionable:
        proposal_dir.mkdir(parents=True, exist_ok=True)
        for p in actionable:
            target_safe = p.target_key.replace("/", "__")
            out = proposal_dir / f"{p.verdict}__{target_safe}.md"
            write_text_atomic(out, render_proposal_md(p))
            written.append(out.name)

    summary = (
        f"knn={stats['n_knn']} above_thr={stats['n_above_threshold']} "
        f"judged={stats['n_judged']} cached={stats.get('n_cached_skipped', 0)} "
        f"dropped={stats.get('n_dropped_adaptive', 0)} "
        f"actionable={stats['n_actionable']}"
    )
    decision = (
        "wrote_proposals" if written else
        "no_actionable" if stats["n_judged"] else
        "no_candidates"
    )
    write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="memory_evolve",
        decision=decision,
        decision_reason=f"{summary}; elapsed={elapsed_ms}ms"
                        + (f"; wrote: {', '.join(written)}" if written else ""),
        model_used=(stats.get("model") or "unknown") if stats["n_judged"] else "(no calls)",
        cost_input_tokens=stats["input_tokens"],
        cost_output_tokens=stats["output_tokens"],
        conn=conn,
    )

    if written:
        log(f"evolve   → {len(written)} proposal(s) at {proposal_dir}/  "
              f"({summary})", tag="agent")
    elif stats["n_judged"]:
        log(f"evolve   → no actionable proposals ({summary})", tag="agent")
    else:
        log(f"evolve   → no candidates above threshold ({summary})", tag="agent")

    return {**stats, "written": written}
