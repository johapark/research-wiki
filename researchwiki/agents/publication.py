"""Post-publication source attachments and required derived-state maintenance."""

import time

from ..log import log
from ..operations.lifecycle import maintenance, phase
from .runner_support import record_timed_subphase


def finish_publication(ctx, conn, result, *, indexer, grader, memory, writer):
    # Stage supplementary files now that the wiki page is on disk:
    # copy each into papers/{stem}.supp/ (an inbox/ source is moved,
    # so it doesn't linger as backlog), normalize the filename, and
    # append an entry to the page's `supplementary:` YAML block.
    # Failures on individual files are surfaced as warnings but
    # don't abort the run — the primary page is already committed.
    if ctx.supplementary:
        from ..supplementary import stage_supplementary, insert_supplementary_entry
        for sp in ctx.supplementary:
            try:
                staged = stage_supplementary(ctx.paper_stem, sp)
                insert_supplementary_entry(
                    result.wiki_path,
                    staged["filename"], staged["kind"],
                )
                origin = " (moved out of inbox/)" if staged["consumed"] else ""
                log(f"supp     → papers/{ctx.paper_stem}.supp/"
                      f"{staged['filename']} ({staged['kind']}){origin}", tag="agent")
            except (FileNotFoundError, FileExistsError, ValueError) as e:
                log(f"supp     ⚠ skipped {sp}: {e}", tag="agent")

    # Update only the new/edited pages. Parallel ingests serialize in the
    # index layer; failures remain recoverable derived-cache warnings.
    index_t0 = time.monotonic()
    with phase("index_update", ctx=ctx):
        outcome = indexer(result)
    keyword_ok = isinstance(outcome, dict) and outcome.get("bm25_mode") in ("incremental", "rebuilt")
    semantic_ok = isinstance(outcome, dict) and outcome.get("semantic") is not None
    maintenance("keyword_index", "completed" if keyword_ok else "failed")
    maintenance("semantic_index", "completed" if semantic_ok else "failed")
    record_timed_subphase(
        ctx, conn, role="index_update", started=index_t0,
        reason="incremental page and claim indexes", writer=writer,
    )

    # Coverage grading: score the just-committed page's claims against
    # its source PDF and write the per-claim signal back into the DB.
    # Without this, the `claims` table's grader columns stay NULL and
    # the read-side surfaces (claim-id lookup, weak-page surfacing,
    # synthesis grounding) have no signal to consume.
    ctx.next_iter()
    with phase("grade_persist", ctx=ctx):
        grade = grader(ctx, conn)
    persisted = isinstance(grade, dict) and "n_claims" in grade and "error" not in grade
    maintenance("claim_grades", "completed" if persisted else "failed")

    # Synthesis maintenance is model-backed corpus enrichment, so it is
    # opt-in on the ingest hot path. Required canonical-page grading above
    # always finishes first. The helper re-arms this ingest's budget only
    # around the optional work.
    if ctx.run_memory_evolve:
        memory(
            ctx, conn, source_key=f"{result.category}/{ctx.paper_stem}"
        )
