"""State-machine runner for the ingest agent.

The outer loop is a fixed sequence:

    reconcile → extract → author × N → grade × N → tournament → commit

Each phase has a thin wrapper that:
  1. Calls the phase function (deterministic or LLM-backed).
  2. Writes one or more rows to ingest_iterations capturing what happened.
  3. Updates the shared `Context` so downstream phases see the result.

The framework controls phase order and persistence; the LLM only produces
text inside author / critic phases. See plan-v1-research-agent.md and the
explanation in the conversation log for the design rationale.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from ..db.connection import get_connection
from ..db.iterations import write_iteration, update_paper_stem
from ..grade import coherence
from . import fitness, model_config, phases
from .context import Context, ReconcileFailed, StemRenameRefused
from ..fsatomic import write_text_atomic
from ..log import log


def _keyword_body_gaps(keywords: list[str], body_text: str) -> list[str]:
    """Return the subset of `keywords` whose first token doesn't appear in
    `body_text` (case-insensitive). Used by the runner to flag source-derived
    keywords absent from the page body — a structural coverage signal.

    The check is on the keyword's first whitespace-split token rather than
    the full keyword string because the body often expresses a multi-word
    keyword via a paraphrase or abbreviation. Matching on the first token
    gives a generous-presence signal (false-positive rate low; missing is
    actually missing) without being so loose it surfaces nothing.
    """
    body_lc = body_text.lower()
    missing: list[str] = []
    for kw in keywords:
        first_tok = (kw.split() or [""])[0].lower().strip(".,;:()[]{}")
        if not first_tok:
            continue
        if first_tok not in body_lc:
            missing.append(kw)
    return missing


def run_ingest(
    pdf_path: str | Path,
    *,
    use_stub: bool = False,
    use_semantic: bool = True,
    verify_claim_entailment: bool = False,
    n_drafts: int = 1,
    max_evolve: int = 1,
    max_debug: int = 1,
    promote_mode: str = "auto",
    doi_override: str | None = None,
    title_override: str | None = None,
    year_override: int | None = None,
    authors_override: list[str] | None = None,
    author_prompt_override: str | None = None,
    supplementary: list[Path] | None = None,
    use_llm_reconcile: bool = True,
    allow_rename: bool = False,
) -> Context:
    """Drive a single ingest attempt end-to-end.

    Returns the populated Context — caller can inspect attempt_id, paper_stem,
    committed_path. Failure is signaled by raising; the partial trace is still
    in ingest_iterations under the attempt_id.
    """
    pdf_path = Path(pdf_path).resolve()
    ctx = Context(
        attempt_id=str(uuid.uuid4()),
        pdf_path=pdf_path,
        pdf_filename=pdf_path.name,
        use_stub=use_stub,
        use_semantic=use_semantic,
        verify_claim_entailment=verify_claim_entailment,
        max_evolve=max_evolve,
        max_debug=max_debug,
        promote_mode=promote_mode,
        doi_override=doi_override,
        title_override=title_override,
        year_override=year_override,
        authors_override=authors_override,
        author_prompt_override=author_prompt_override,
        supplementary=supplementary,
        use_llm_reconcile=use_llm_reconcile,
        allow_rename=allow_rename,
    )

    log(f"attempt_id={ctx.attempt_id}", tag="agent")
    log(f"pdf={pdf_path.name}", tag="agent")
    _author_cfg = model_config.for_phase("author")
    _mode = "stub" if use_stub else f"real ({_author_cfg.provider}/{_author_cfg.model})"
    log(f"mode={_mode}", tag="agent")
    log(f"n_drafts={n_drafts}  use_semantic={use_semantic}  "
          f"verify_claim_entailment={verify_claim_entailment}  max_evolve={max_evolve}", tag="agent")

    # Open one connection for the whole attempt so all phases share the same
    # transaction view. Each write_iteration commits its own row, which is
    # what we want — append-only, durable as soon as the row is written.
    conn = get_connection()
    try:
        # Phase 1: reconcile
        ctx.next_iter()
        meta, parent_id = _phase_reconcile(ctx, conn)
        ctx.metadata = meta
        ctx.paper_stem = meta.get("stem")
        if ctx.paper_stem:
            update_paper_stem(ctx.attempt_id, ctx.paper_stem, conn=conn)
            conn.commit()
        log(
            f"reconcile → stem={ctx.paper_stem} year={meta.get('year')} "
            f"type={meta.get('paper_type', 'research')}", tag="agent"
        )

        # Refuse silent stem renames. When reconcile's DOI lookup finds an
        # existing wiki page AND the new derived stem differs, the default
        # is to abort before any state mutation. This catches the failure
        # mode where a re-ingest drifts year/title and orphans the prior
        # page (avsec-2025 → avsec-2026, linder-2023 → linder-2024 during
        # the 2026-06-14 bulk re-ingest). Pass --allow-rename to opt in.
        prior_stem = (meta or {}).get("prior_stem")
        if prior_stem and ctx.paper_stem and prior_stem != ctx.paper_stem and not ctx.allow_rename:
            raise StemRenameRefused(prior_stem=prior_stem, new_stem=ctx.paper_stem)

        # Bail loudly when reconcile produced no stem — running the
        # downstream phases on broken metadata wastes ~5 LLM calls and
        # ends with a sandbox file named `unknown-stem-<8hex>.md` that
        # nobody can act on. The CLI handler turns this into a focused
        # error message + exit-2 (no stack trace).
        if not ctx.paper_stem:
            missing = [
                k for k in ("title", "year", "doi", "authors")
                if not meta.get(k)
            ]
            raise ReconcileFailed(
                sources=meta.get("sources") or [],
                missing=missing,
            )

        # Phase 2: extract
        ctx.next_iter()
        sections, claim_count, full_text = _phase_extract(ctx, conn)
        ctx.sections = sections
        ctx.pdf_full_text = full_text
        ctx.claims_count = claim_count
        log(f"extract → sections={list(sections.keys())} pdf_claims={claim_count}", tag="agent")

        # Phase 2.4: target-claims extraction (L3) — structured list of
        # claims the page should preserve. Surfaces a coverage target the
        # author phase consumes; failure is graceful (empty target_claims
        # → author falls back to pre-L3 prompt shape).
        ctx.next_iter()
        ctx.target_claims = _phase_target_claims(ctx, conn)
        if ctx.target_claims is not None and not ctx.target_claims.is_empty():
            n = len(ctx.target_claims.claims)
            n_crit = len(ctx.target_claims.by_importance("critical"))
            n_high = len(ctx.target_claims.by_importance("high"))
            log(f"target-claims → {n} extracted "
                  f"({n_crit} critical, {n_high} high)", tag="agent")
        elif ctx.target_claims is not None and ctx.target_claims.error:
            log(f"target-claims → skipped ({ctx.target_claims.error[:60]})", tag="agent")

        # Phase 2.5: crosslink candidates (verified surface for Related Papers)
        ctx.next_iter()
        ctx.crosslink_candidates = _phase_crosslinks(ctx, conn)
        log(f"crosslinks → {len(ctx.crosslink_candidates)} verified candidate(s)", tag="agent")

        # Phase 3: author × N — parallel.
        # Each draft is an independent LLM call with the same input (only
        # temperature differs), so they parallelize trivially. The Anthropic
        # SDK is sync, so we use a ThreadPoolExecutor for IO concurrency.
        # `write_iteration` already commits per-call (append-only event log),
        # so per-thread DB writes don't race. We don't share the runner's
        # `conn` across threads — each _phase_author call opens its own
        # commit window via the no-conn fallback in iterations.write_iteration.
        from concurrent.futures import ThreadPoolExecutor

        # Per-draft temperature, clamped to Anthropic's valid [0, 1] range: the
        # raw 0.2 + 0.4*i formula exceeds 1.0 once i >= 2 (n_drafts >= 4), which
        # the API rejects. Default n_drafts=1 → [0.2] (single draft, tournament
        # is a no-op); opting into -n 2 gives [0.2, 0.6]. Real draft diversity
        # comes from per-slot drafting stances (see _phase_author → slot);
        # temperature is a secondary nudge on top.
        author_inputs = [
            (i, min(1.0, 0.2 + 0.4 * i))
            for i in range(n_drafts)
        ]
        # Reserve N iteration counters before fan-out so iteration numbers
        # in the trace stay monotonic and deterministic per draft slot.
        author_iter_ids = [ctx.next_iter() for _ in range(n_drafts)]

        def _author_one(slot: int, iter_no: int, temperature: float):
            # Build a per-thread Context view with the right iteration number.
            # Only the iteration field varies; everything else is read-only
            # from the thread's perspective.
            local_ctx = type(ctx)(**{**ctx.__dict__, "iteration": iter_no})
            return slot, _phase_author(local_ctx, conn=None, temperature=temperature, slot=slot)

        with ThreadPoolExecutor(max_workers=n_drafts) as pool:
            futures = [
                pool.submit(_author_one, slot, iter_no, t)
                for (slot, t), iter_no in zip(author_inputs, author_iter_ids)
            ]
            # Place results back in slot order so ctx.drafts[i] corresponds
            # to temperature 0.2+0.4*i, matching the prior sequential layout.
            results: list[phases.Draft | None] = [None] * n_drafts
            for fut in futures:
                slot, draft = fut.result()
                results[slot] = draft

        for slot, draft in enumerate(results):
            ctx.drafts.append(draft)
            t = author_inputs[slot][1]
            log(f"author #{slot + 1} → stance={draft.stance} t={t:.1f} {len(draft.text)} chars", tag="agent")

        # Phase 4: grade each draft (BM25 + bi-encoder semantic similarity +
        # PDF-anchor salience + structural coherence)
        for d in ctx.drafts:
            ctx.next_iter()
            _phase_grade(ctx, conn, d)
            sal = d.scores.get("salience_score")
            sal_str = f"{sal:.2f}" if sal is not None else "n/a"
            sem = d.scores.get("semantic_score")
            sem_str = f"{sem:.2f}" if sem is not None else "n/a"
            log(
                f"grade   → draft {d.iteration_id} "
                f"sem={sem_str} sal={sal_str} "
                f"coh={d.scores.get('coherence_score', 0):.2f} "
                f"bm25={d.scores.get('mean_bm25', 0):.2f}", tag="agent"
            )

        # Phase 5: tournament — pick the highest-scored draft
        ctx.next_iter()
        ctx.winner = _phase_tournament(ctx, conn)
        log(f"tournament → winner draft {ctx.winner.iteration_id}", tag="agent")

        # Phase 5.5: critic + evolve loop. Skip when the grader found neither a
        # weakly-supported claim (precision) nor an uncovered load-bearing PDF
        # anchor (recall) — don't burn tokens critiquing a draft it's happy with.
        # Both axes have to gate the loop: weak claims alone let the worst-recall
        # drafts exit here, since a page that omits content has less text to be
        # weak about.
        for round_num in range(ctx.max_evolve):
            ctx.next_iter()
            critique = _phase_critic(ctx, conn, ctx.winner)
            if not critique.weak_claims and not critique.coverage_gaps:
                log(f"critic  → no weak claims or coverage gaps, skipping evolve", tag="agent")
                break
            log(f"critic  → flagged {len(critique.weak_claims)} weak claims, "
                  f"{len(critique.coverage_gaps)} coverage gaps "
                  f"({critique.cost_input_tokens} in / {critique.cost_output_tokens} out)", tag="agent")

            ctx.next_iter()
            evolved = _phase_evolve(ctx, conn, ctx.winner, critique)
            log(f"evolve  → revised draft {evolved.iteration_id} "
                  f"({evolved.input_tokens} in / {evolved.output_tokens} out)", tag="agent")

            ctx.next_iter()
            _phase_grade(ctx, conn, evolved)
            log(f"grade   → evolved sem={(evolved.scores.get('semantic_score') or 0):.2f} "
                  f"bm25={(evolved.scores.get('mean_bm25') or 0):.2f}", tag="agent")

            if fitness.is_evolve_improvement(evolved, ctx.winner):
                ctx.winner = evolved
                log(f"swap    → evolved beats prior winner; promoted", tag="agent")
            else:
                log(f"keep    → evolved did not improve (mean/floor/drift); reverting", tag="agent")
                break

        # Phase 6: commit — write the winning draft to sandbox
        ctx.next_iter()
        ctx.committed_path = _phase_commit(ctx, conn)
        log(f"commit  → {ctx.committed_path}", tag="agent")

    finally:
        conn.close()

    return ctx


def _phase_reconcile(ctx: Context, conn) -> tuple[dict, int]:
    """Wrapper for the reconcile phase. Persists one row before returning."""
    t0 = time.time()
    meta = phases.reconcile_metadata(
        ctx.pdf_path,
        doi_override=ctx.doi_override,
        title_override=ctx.title_override,
        year_override=ctx.year_override,
        authors_override=ctx.authors_override,
        use_llm=ctx.use_llm_reconcile,
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    iter_id = write_iteration(
        attempt_id=ctx.attempt_id,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="reconcile",
        paper_stem=meta.get("stem"),
        decision="observed",
        decision_reason=f"reconciled in {elapsed_ms}ms; sources={meta.get('sources', [])}",
        critic_notes=str(meta),
        conn=conn,
    )
    return meta, iter_id


def _phase_extract(ctx: Context, conn) -> tuple[dict, int, str]:
    """Wrapper for the extract phase. Persists one row."""
    t0 = time.time()
    sections, n_claims, full_text = phases.extract_sections(ctx.pdf_path)
    elapsed_ms = int((time.time() - t0) * 1000)
    write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="extract",
        decision="observed",
        decision_reason=(
            f"extracted in {elapsed_ms}ms; sections={list(sections.keys())}; "
            f"claims_pdf_anchored={n_claims}; full_text_chars={len(full_text)}"
        ),
        conn=conn,
    )
    return sections, n_claims, full_text


def _phase_target_claims(ctx: Context, conn):
    """Wrapper for the target-claims phase (L3). Persists one row.

    Returns a TargetClaimsOutput. On any LLM/JSON failure, returns the
    output object with `error` set and `claims=[]` — the caller treats
    empty claims as a no-op and the author phase reverts to the pre-L3
    prompt shape. So target-claims failure is non-fatal.
    """
    from .phases.target_claims import extract_target_claims
    t0 = time.time()
    out = extract_target_claims(
        metadata=ctx.metadata or {},
        sections=ctx.sections or {},
        pdf_full_text=ctx.pdf_full_text,
        use_stub=ctx.use_stub,
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    if out.error:
        decision = "error"
        reason = f"{out.error[:200]}; elapsed={elapsed_ms}ms"
    elif out.is_empty():
        decision = "empty"
        reason = f"no claims extracted; elapsed={elapsed_ms}ms"
    else:
        n = len(out.claims)
        n_crit = sum(1 for c in out.claims if c.importance == "critical")
        n_high = sum(1 for c in out.claims if c.importance == "high")
        decision = "extracted"
        reason = (
            f"n={n} (critical={n_crit}, high={n_high}); "
            f"elapsed={elapsed_ms}ms"
        )
    write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="target_claims",
        decision=decision,
        decision_reason=reason,
        model_used=out.model or "(no calls)",
        cost_input_tokens=out.input_tokens,
        cost_output_tokens=out.output_tokens,
        conn=conn,
    )
    return out


def _phase_crosslinks(ctx: Context, conn) -> list:
    """Wrapper for the crosslink-candidates phase. Persists one row.

    Two-source pipeline:
      1. Citation-graph (deterministic): S2 references/citations, Crossref,
         PDF-text DOI scan. Produces `cited_by_source` / `cites_source` candidates.
      2. Semantic-KNN + LLM judge: adds `topical` candidates the
         citation graph misses (e.g. independent contemporaneous work).
         Skipped under --use-stub so offline runs stay deterministic.

    Both lists are unioned with citation-graph winning on duplicate wikilinks
    (its `kind` carries directional info the topical path can't recover).
    """
    t0 = time.time()
    cite_cands = phases.crosslink_candidates(ctx.pdf_path, ctx.metadata or {})
    cite_keys = frozenset(c.wikilink for c in cite_cands)
    topical_cands = phases.propose_crosslinks(
        ctx.metadata or {},
        ctx.sections or {},
        use_stub=ctx.use_stub,
        exclude_keys=cite_keys,
    )
    cands = list(cite_cands) + list(topical_cands)
    # Drop any self-reference. On re-ingest the paper's own prior page is
    # already in the wiki, so both the DOI path (own DOI in wiki_dois) and
    # the semantic path (own identical page is the top hit) can propose the
    # paper as a cross-link to itself. Filter on the stem suffix of the
    # 'category/stem' wikilink.
    if ctx.paper_stem:
        cands = [c for c in cands if c.wikilink.rsplit("/", 1)[-1] != ctx.paper_stem]
    elapsed_ms = int((time.time() - t0) * 1000)
    summary = "; ".join(f"{c.wikilink} ({c.kind})" for c in cands[:10])
    write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="crosslinks",
        decision="observed",
        decision_reason=(
            f"found {len(cands)} candidates in {elapsed_ms}ms "
            f"({len(cite_cands)} citation-graph + {len(topical_cands)} topical)"
            + (f"; first 10: {summary}" if cands else "")
        ),
        conn=conn,
    )
    return cands


def _phase_author(ctx: Context, conn, temperature: float, slot: int = 0):
    """Wrapper for one author call. Returns a Draft (with iteration_id filled).

    `slot` selects the drafting stance (instruction-level draft diversity) so
    parallel drafts differ by stance, not just temperature/sampling noise.
    """
    stance = phases.stance_for_slot(slot)
    draft = phases.author(
        metadata=ctx.metadata,
        sections=ctx.sections,
        temperature=temperature,
        candidates=ctx.crosslink_candidates,
        use_stub=ctx.use_stub,
        system_prompt_override=ctx.author_prompt_override,
        stance=stance,
        pdf_full_text=ctx.pdf_full_text,
        target_claims=ctx.target_claims,
    )
    iter_id = write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="author",
        section="page",
        draft_text=draft.text,
        decision="kept",
        decision_reason=f"author draft #{len(ctx.drafts) + 1} t={temperature:.2f} stance={draft.stance}",
        model_used=draft.model,
        temperature=draft.temperature,
        cost_input_tokens=draft.input_tokens,
        cost_output_tokens=draft.output_tokens,
        conn=conn,
    )
    draft.iteration_id = iter_id
    return draft


def _phase_grade(ctx: Context, conn, draft) -> None:
    """Wrapper for grading one draft. Updates draft.scores + claim_details in place.

    Also runs the cheap structural-conformance ('coherence') check from
    `grade.coherence` and merges its score into draft.scores so the
    tournament can see it. Coherence is regex-only and adds no noticeable
    latency.
    """
    scores, details = phases.grade_draft(
        stem=ctx.paper_stem,
        draft_text=draft.text,
        metadata=ctx.metadata,
        sandbox_dir=ctx.sandbox_dir,
        pdf_path=ctx.pdf_path,
        use_semantic=ctx.use_semantic,
    )
    coh = coherence.score_coherence(draft.text, page_type="paper")
    scores["coherence_score"] = coh.score
    scores["coherence_violations"] = coh.violations
    draft.scores = scores
    draft.claim_details = details
    sem_label = (
        "with semantic" if ctx.use_semantic and scores.get("semantic_available") else "BM25-only"
    )
    write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="grade",
        parent_iteration_id=draft.iteration_id,
        grader_scores=scores,
        decision="observed",
        decision_reason=f"{sem_label} grade of draft {draft.iteration_id}",
        conn=conn,
    )


def _phase_tournament(ctx: Context, conn):
    """Wrapper for tournament phase. Picks the highest-scored draft."""
    winner, rationale = phases.tournament(ctx.drafts)
    write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="tournament",
        parent_iteration_id=winner.iteration_id,
        decision="kept",
        decision_reason=rationale,
        grader_scores=winner.scores,
        conn=conn,
    )
    return winner


def _phase_critic(ctx: Context, conn, draft):
    """Wrapper for critic phase. Persists one row, returns CritiqueOutput
    (with cost fields the runner uses for the print line)."""
    critique = phases.critic(
        draft=draft,
        metadata=ctx.metadata,
        use_stub=ctx.use_stub,
    )
    n_weak, n_gaps = len(critique.weak_claims), len(critique.coverage_gaps)
    if n_weak or n_gaps:
        reason = f"flagged {n_weak} weak claims, {n_gaps} coverage gaps"
    else:
        reason = "no weak claims or coverage gaps; pass-through"
    write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="critic",
        parent_iteration_id=draft.iteration_id,
        critic_notes=critique.notes,
        decision="kept" if (n_weak or n_gaps) else "observed",
        decision_reason=reason,
        model_used=critique.model,
        cost_input_tokens=critique.input_tokens,
        cost_output_tokens=critique.output_tokens,
        conn=conn,
    )
    # Stash cost on the object so the runner's print line works without
    # plumbing it through a separate return tuple.
    critique.cost_input_tokens = critique.input_tokens
    critique.cost_output_tokens = critique.output_tokens
    return critique


def _phase_evolve(ctx: Context, conn, prior_draft, critique):
    """Wrapper for evolve phase. Returns a new Draft (graded by next phase)."""
    out = phases.evolve(
        draft=prior_draft,
        critique=critique,
        metadata=ctx.metadata,
        sections=ctx.sections,
        use_stub=ctx.use_stub,
    )
    new_draft = phases.Draft(
        text=out.text,
        model=out.model,
        temperature=out.temperature,
        input_tokens=out.input_tokens,
        output_tokens=out.output_tokens,
    )
    iter_id = write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="author",
        section="page",
        draft_text=new_draft.text,
        parent_iteration_id=prior_draft.iteration_id,
        decision="kept",
        decision_reason="evolved from critic notes",
        model_used=new_draft.model,
        temperature=new_draft.temperature,
        cost_input_tokens=new_draft.input_tokens,
        cost_output_tokens=new_draft.output_tokens,
        conn=conn,
    )
    new_draft.iteration_id = iter_id
    ctx.drafts.append(new_draft)
    return new_draft


# DEBUG and other generic call sites use the conservative strict rule; evolve
# uses the floor-aware variant. Both live in `agents.fitness` (per-operator
# fitness). Kept as a module-level alias for the DEBUG call site below and for
# backward-compat with callers importing it from `runner`.
_is_strict_improvement = fitness.is_strict_improvement


def _phase_debug(
    ctx: Context, conn,
    cleaned_text: str, n_kc: int, gate, verification,
    *, issues: list[str],
):
    """DEBUG phase — one targeted repair pass when the gate rejects on a
    structural issue. Returns the (possibly updated) (cleaned_text, n_kc,
    gate, verification) so the commit phase can re-evaluate.

    On success: ctx.winner is replaced with the debug draft, scores +
    cross-link verification + KC count are refreshed, and the gate is
    re-run. On failure (debug doesn't strictly improve), we keep the
    original winner and the original gate result.
    """
    from . import promote as promote_mod

    log(f"debug    → structural issues: {issues}", tag="agent")
    ctx.next_iter()
    out = phases.debug(
        draft=ctx.winner,
        issues=issues,
        gate_reasons=gate.reasons,
        metadata=ctx.metadata,
        sections=ctx.sections,
        use_stub=ctx.use_stub,
    )
    new_draft = phases.Draft(
        text=out.text,
        model=out.model,
        temperature=out.temperature,
        input_tokens=out.input_tokens,
        output_tokens=out.output_tokens,
    )
    iter_id = write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="debug",
        section="page",
        draft_text=new_draft.text,
        parent_iteration_id=ctx.winner.iteration_id,
        decision="kept",
        decision_reason=f"debug repair targeting {issues}",
        model_used=new_draft.model,
        temperature=new_draft.temperature,
        cost_input_tokens=new_draft.input_tokens,
        cost_output_tokens=new_draft.output_tokens,
        conn=conn,
    )
    new_draft.iteration_id = iter_id
    ctx.drafts.append(new_draft)

    # Re-grade against the original PDF.
    ctx.next_iter()
    _phase_grade(ctx, conn, new_draft)
    log(
        f"grade    → debug sem={(new_draft.scores.get('semantic_score') or 0):.2f} "
        f"bm25={(new_draft.scores.get('mean_bm25') or 0):.2f} "
        f"drift={new_draft.scores.get('n_drift') or 0}", tag="agent"
    )

    # Strict-improvement gate on the repair: keep the debug draft only when
    # it actually beats the original on the comparison rule. This prevents
    # DEBUG from making things worse in pursuit of a structural fix.
    if not _is_strict_improvement(new_draft, ctx.winner):
        log("debug    → did not strictly improve; reverting to prior winner", tag="agent")
        return cleaned_text, n_kc, gate, verification

    log("debug    → repaired draft beats prior winner; promoted", tag="agent")
    ctx.winner = new_draft
    cleaned_text, verification = phases.verify_crosslinks(
        new_draft.text, ctx.crosslink_candidates
    )
    n_kc = promote_mod._count_key_contributions(cleaned_text)
    new_gate = promote_mod.should_auto_promote(
        scores=new_draft.scores,
        verification=verification,
        n_key_contributions=n_kc,
        paper_type=(ctx.metadata or {}).get("paper_type", "research"),
    )
    if new_gate.promoted:
        log("gate     → DEBUG repair passed all gates", tag="agent")
    else:
        log(f"gate     → still failing after DEBUG: {new_gate.reasons}", tag="agent")
    return cleaned_text, n_kc, new_gate, verification


def _run_entailment_check(ctx: Context, conn, cleaned_text: str) -> None:
    """Per-claim entailment (support) check on the FINAL promoted text.

    Runs one batched entailment judge over `cleaned_text`, merges
    `n_unsupported` into the winner's scores (so should_auto_promote can veto),
    surfaces each unsupported claim, and writes a durable `claim_support`
    iteration row. Best-effort: a classifier failure logs and leaves the
    winner's scores untouched (no veto rather than a crash).

    MUST run *after* any DEBUG repair: a successful repair rebuilds the
    winner's scores from a fresh grade, which silently drops an `n_unsupported`
    set before it — the bug this ordering fixes. Caller re-evaluates the gate
    afterward so the veto lands on the draft that actually gets promoted.
    """
    from ..grade.support import llm_support_classifier
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
        n_unsup = support_scores.get("n_unsupported", 0)
        n_checked = support_scores.get("n_support_checked", 0)
        unsupported = support_scores.get("unsupported_claims", [])
        ctx.winner.scores["n_unsupported"] = n_unsup
        ctx.winner.scores["n_support_checked"] = n_checked
        ctx.winner.scores["unsupported_claims"] = unsupported
        log(f"support  → {n_unsup} unsupported / {n_checked} checked", tag="agent")
        # Name the offending claims so a reviewer of a sandboxed page knows
        # which ones to re-check, not just that N failed.
        for uc in unsupported:
            log(f"           ✗ [{uc['section']}] {uc['text'][:80]}", tag="agent")
        # Durable record: one iteration row carrying the verdict detail, so the
        # failed claims survive past the console for post-hoc review.
        ctx.next_iter()
        write_iteration(
            attempt_id=ctx.attempt_id,
            paper_stem=ctx.paper_stem,
            pdf_filename=ctx.pdf_filename,
            iteration=ctx.iteration,
            role="claim_support",
            parent_iteration_id=ctx.winner.iteration_id,
            grader_scores={
                "n_unsupported": n_unsup,
                "n_support_checked": n_checked,
                "unsupported_claims": unsupported,
            },
            decision="observed",
            decision_reason=f"{n_unsup} unsupported / {n_checked} checked",
            conn=conn,
        )
    except Exception as e:
        log(f"support  → check failed, veto skipped: {e}", tag="agent")


def _phase_commit(ctx: Context, conn) -> Path:
    """Commit phase: cross-link verification → promote-or-sandbox decision.

    Always runs verify_crosslinks first so the cleaned-text and verification
    report are available to both branches. Then evaluates promotion gates
    and either:
      - writes to wiki/{category}/{stem}.md and updates index/log/back-links
        (auto-promote); or
      - falls back to .agent-output/{stem}.md (sandbox).
    """
    from . import promote as promote_mod

    cleaned_text, verification = phases.verify_crosslinks(
        ctx.winner.text, ctx.crosslink_candidates
    )
    n_kc = promote_mod._count_key_contributions(cleaned_text)

    # Print verifier outcome (covers both paths).
    if verification.unverified or verification.broken:
        log(f"verify   → kept {len(verification.verified)} wikilinks; "
              f"stripped {len(verification.unverified)} unverified, "
              f"{len(verification.broken)} broken", tag="agent")
    else:
        log(f"verify   → all {len(verification.verified)} wikilinks verified", tag="agent")

    # Promotion-mode dispatch.
    if ctx.promote_mode == "never":
        gate = promote_mod.GateResult(
            promoted=False, reasons=["--force-sandbox flag set"]
        )
    else:
        gate = promote_mod.should_auto_promote(
            scores=ctx.winner.scores,
            verification=verification,
            n_key_contributions=n_kc,
            paper_type=(ctx.metadata or {}).get("paper_type", "research"),
        )
        # DEBUG operator: when the gate rejects on a structural issue
        # (numeric drift, too few KC bullets, too few graded claims) we
        # try up to `max_debug` targeted repair passes before falling back
        # to sandbox. Skipped under --auto-promote (caller already overrode
        # the gate) and --force-sandbox (gate is informational only there).
        if not gate.promoted and ctx.promote_mode == "auto":
            for _ in range(ctx.max_debug):
                if gate.promoted:
                    break
                structural_issues = phases.detect_structural_gate_issues(gate.reasons)
                if not structural_issues:
                    break
                prior_gate = gate
                cleaned_text, n_kc, gate, verification = _phase_debug(
                    ctx, conn, cleaned_text, n_kc, gate, verification,
                    issues=structural_issues,
                )
                if gate is prior_gate:
                    # _phase_debug reverted (repair didn't strictly improve)
                    # and handed back the same gate object — another pass
                    # would re-run the identical repair against the same
                    # winner. Stop burning the budget.
                    break

        # Opt-in per-claim entailment veto (grade.support). Off unless
        # --verify-claim-entailment is set. Runs on the FINAL winner/text —
        # after any DEBUG repair has settled — then re-evaluates the gate so the
        # veto lands on the draft that actually gets promoted. (Running it
        # before DEBUG let a successful repair rebuild the winner's scores and
        # silently drop the n_unsupported veto.) It's the qualitative analogue
        # of the numeric-drift veto that BM25/semantic similarity can't catch.
        if ctx.verify_claim_entailment:
            _run_entailment_check(ctx, conn, cleaned_text)
            gate = promote_mod.should_auto_promote(
                scores=ctx.winner.scores,
                verification=verification,
                n_key_contributions=n_kc,
                paper_type=(ctx.metadata or {}).get("paper_type", "research"),
            )

        if ctx.promote_mode == "always" and not gate.promoted:
            log(f"promote  → --auto-promote forced over failed gates: {gate.reasons}", tag="agent")
            gate = promote_mod.GateResult(
                promoted=True, reasons=gate.reasons, warnings=gate.warnings,
            )

    # Surface non-blocking warnings (e.g. drafter hallucinated wikilinks
    # that verify already stripped) regardless of which path we take.
    for w in gate.warnings:
        log(f"verify   ⚠ {w}", tag="agent")

    if gate.promoted:
        # Short-name proposal — only worth the LLM call when we're committing
        # to the wiki (sandbox pages don't appear in index.md anyway).
        ctx.next_iter()
        sn_out = phases.propose_short_name(
            metadata=ctx.metadata,
            draft_text=cleaned_text,
            use_stub=ctx.use_stub,
        )
        write_iteration(
            attempt_id=ctx.attempt_id,
            paper_stem=ctx.paper_stem,
            pdf_filename=ctx.pdf_filename,
            iteration=ctx.iteration,
            role="short_name",
            decision="kept" if sn_out.name != "TODO" else "rejected",
            decision_reason=f"proposed: {sn_out.name!r}",
            model_used=sn_out.model,
            cost_input_tokens=sn_out.input_tokens,
            cost_output_tokens=sn_out.output_tokens,
            conn=conn,
        )
        log(f"shortname → {sn_out.name!r}", tag="agent")

        # Keywords — same pattern as short_name, written into YAML for the
        # search index (BM25 + semantic) and for `lint` quality checks.
        # Source-derived: the extract phase's section excerpts are the
        # primary input, not the winning draft. This makes the YAML
        # `keywords:` field a structural coverage signal — terms named in
        # the source but missing from the page body are flagged in the
        # body-coverage log line below.
        ctx.next_iter()
        kw_out = phases.propose_keywords(
            metadata=ctx.metadata,
            draft_text=cleaned_text,
            sections=ctx.sections,
            full_pdf_text=ctx.pdf_full_text,
            use_stub=ctx.use_stub,
        )
        write_iteration(
            attempt_id=ctx.attempt_id,
            paper_stem=ctx.paper_stem,
            pdf_filename=ctx.pdf_filename,
            iteration=ctx.iteration,
            role="keywords",
            decision="kept" if kw_out.keywords else "rejected",
            decision_reason=f"proposed: {kw_out.keywords!r}",
            model_used=kw_out.model,
            cost_input_tokens=kw_out.input_tokens,
            cost_output_tokens=kw_out.output_tokens,
            conn=conn,
        )
        log(f"keywords  → {kw_out.keywords}", tag="agent")

        # Body-coverage signal. With source-derived keywords, any keyword
        # that doesn't appear (even partially) in the page body is a
        # candidate omission worth a reviewer's eye. We log this rather
        # than gate on it — a small body/keyword gap is normal (e.g., the
        # body uses a synonym, or the keyword names a paper-internal
        # subsystem the wiki page deliberately summarized at higher
        # altitude). Whole-token presence test (`keyword in body`,
        # case-insensitive); multi-word keywords pass if any token-bigram
        # of the keyword appears, since "panel of normals" matching the
        # body's `(PoN)` is a different shape of presence.
        if kw_out.keywords:
            missing = _keyword_body_gaps(kw_out.keywords, cleaned_text)
            if missing:
                log(
                    f"kw-coverage → {len(kw_out.keywords) - len(missing)}/"
                    f"{len(kw_out.keywords)} keywords appear in body; "
                    f"missing: {missing[:5]}", tag="agent"
                )
            else:
                log(f"kw-coverage → all {len(kw_out.keywords)} keywords appear in body", tag="agent")

        # `ctx.winner.model` is the model that produced the final page text.
        # It's set by either the author phase (initial draft) or the evolve
        # phase (when critic+evolve produced an improvement that beat the
        # tournament winner — in that case ctx.winner was reassigned to the
        # evolved Draft). Either way, the model attribution on disk reflects
        # the LLM that authored the committed prose.
        author_model_id = (ctx.winner.model if ctx.winner else None) or None
        result = promote_mod.promote_to_wiki(
            stem=ctx.paper_stem,
            draft_text=cleaned_text,
            metadata=ctx.metadata,
            # Back-links go to ALL verified candidates (not just the ones the
            # author wrote forward links for). Forward-link discretion stays
            # with the author; back-links are structural — if the citation graph
            # confirms paper A cites page B, B's page should record that even
            # if A's author chose not to feature B in Related Papers. This is
            # the fix for orphan pages whose author-phase wrote `(none)`.
            candidates=[c for c in ctx.crosslink_candidates if c.verified],
            source_pdf_path=ctx.pdf_path,
            attempt_id=ctx.attempt_id,
            short_name=sn_out.name,
            keywords=kw_out.keywords,
            author_model=author_model_id,
        )
        # Carry gate-level warnings (e.g. broken-wikilinks-stripped) through
        # to the persisted decision_reason so post-hoc audits see the signal.
        if gate.warnings:
            result.warnings.extend(gate.warnings)
        if result.pdf_upgrade:
            ctx.next_iter()
            write_iteration(
                attempt_id=ctx.attempt_id,
                paper_stem=ctx.paper_stem,
                pdf_filename=ctx.pdf_filename,
                iteration=ctx.iteration,
                role="pdf_upgrade",
                decision="upgraded",
                decision_reason=(
                    f"verdict=journal-upgrade; "
                    f"old_doi={result.pdf_upgrade.get('old_doi')}; "
                    f"new_doi={result.pdf_upgrade.get('new_doi')}"
                ),
                conn=conn,
            )
            log(f"pdf_upgrade → swapped preprint for journal version "
                  f"(old_doi={result.pdf_upgrade.get('old_doi')}, "
                  f"new_doi={result.pdf_upgrade.get('new_doi')})", tag="agent")
        decision = "committed-to-wiki"
        decision_reason = (
            f"category={result.category}; "
            f"wiki={result.wiki_path}; pdf={result.pdf_path}; "
            f"backlinks_added={result.backlinks_added}; "
            f"index_updated={result.index_updated}; log_appended={result.log_appended}"
        )
        if result.warnings:
            decision_reason += f"; warnings={result.warnings}"
        log(f"promote  → wiki/{result.category}/{ctx.paper_stem}.md "
              f"({len(result.backlinks_added)} back-links added)", tag="agent")
        if result.warnings:
            for w in result.warnings:
                log(f"           ⚠ {w}", tag="agent")

        # Stage supplementary files now that the wiki page is on disk:
        # copy each into papers/{stem}.supp/, normalize the filename,
        # and append an entry to the page's `supplementary:` YAML block.
        # Failures on individual files are surfaced as warnings but
        # don't abort the run — the primary page is already committed.
        if ctx.supplementary:
            from .. tasks.attach import stage_supplementary, insert_supplementary_entry
            for sp in ctx.supplementary:
                try:
                    staged = stage_supplementary(ctx.paper_stem, sp)
                    insert_supplementary_entry(
                        result.wiki_path,
                        staged["filename"], staged["kind"],
                    )
                    log(f"supp     → papers/{ctx.paper_stem}.supp/"
                          f"{staged['filename']} ({staged['kind']})", tag="agent")
                except (FileNotFoundError, FileExistsError, ValueError) as e:
                    log(f"supp     ⚠ skipped {sp}: {e}", tag="agent")

        # Memory evolution: now that the new page is on disk, ask whether
        # any neighboring synthesis pages need updating.
        # Cosine prefilter inside skips zero-signal candidates so the cost
        # bill stays bounded (~$0.04 worst-case per ingest).
        ctx.next_iter()
        phases.evolve_memory(ctx, conn, source_key=f"{result.category}/{ctx.paper_stem}")

        # Coverage grading: score the just-committed page's claims against
        # its source PDF and write the per-claim signal back into the DB.
        # Without this, the `claims` table's grader columns stay NULL and
        # the read-side surfaces (claim-id lookup, weak-page surfacing,
        # synthesis grounding) have no signal to consume.
        ctx.next_iter()
        phases.persist_grades(ctx, conn)

        out_path = result.wiki_path
    else:
        # Sandbox path: write the agent's draft to .agent-output (was the
        # commit phase's behavior in Phase 2.6).
        #
        # Run the same Tantivy neighbor-vote categorizer here that the
        # auto-promote path uses, so a sandboxed page carries a useful
        # category hint instead of `[TODO]`. The categorizer is read-only
        # and cheap; bypassing it just because gates failed forced the
        # user to re-categorize from scratch on every false-positive
        # sandbox.
        from .phases import _wrap_with_frontmatter
        ctx.sandbox_dir.mkdir(parents=True, exist_ok=True)
        # Stem may be None when reconcile fails outright. Disambiguate the
        # fallback name with the attempt-id short hash so two failed
        # ingests in the same session don't clobber each other's trace.
        stem_for_path = ctx.paper_stem or f"unknown-stem-{ctx.attempt_id[:8]}"
        out_path = ctx.sandbox_dir / f"{stem_for_path}.md"
        summary_text = promote_mod._extract_section(cleaned_text, "summary")
        cat_suggestion, cat_strength = promote_mod._suggest_category(
            (ctx.metadata or {}).get("title") or "", summary_text
        )
        write_text_atomic(out_path, _wrap_with_frontmatter(
            cleaned_text, ctx.metadata, stem_for_path,
            category=cat_suggestion,
            category_strength=cat_strength,
        ))
        if cat_suggestion:
            log(f"sandbox  → category suggested: {cat_suggestion} "
                  f"({cat_strength})", tag="agent")
        decision = "committed-to-sandbox"
        decision_reason = (
            f"sandbox={out_path}; gates_failed: " + "; ".join(gate.reasons or ["unknown"])
        )
        log(f"sandbox  → {out_path}", tag="agent")
        for r in gate.reasons:
            log(f"           ✗ {r}", tag="agent")

    write_iteration(
        attempt_id=ctx.attempt_id,
        paper_stem=ctx.paper_stem,
        pdf_filename=ctx.pdf_filename,
        iteration=ctx.iteration,
        role="commit",
        parent_iteration_id=ctx.winner.iteration_id,
        decision=decision,
        decision_reason=decision_reason,
        conn=conn,
    )
    return out_path
