"""Ingest-time contradiction alert.

Post-ingest hook that runs the cross-paper contradiction judge scoped to
just the new paper's claims, then prints one-line `⚠ contradicts` alerts
per detected disagreement. Silent no-op on any failure — this hook must
NEVER break ingest.

Same shape as `claim_overlap.run_after_ingest` and
`concepts.attach_after_ingest`: called from the agent path right after
promotion. The judge is LLM-call-heavy but bounded by max_pairs; when
--stub / --no-semantic (no embedder / no judge) is in play, this returns
early with no alerts.
"""

from __future__ import annotations


from ..log import log
from ..search.refs import format_claim_ref


def alert_after_ingest(
    stem: str,
    committed_path,
    *,
    sim_threshold: float = 0.85,
    max_pairs: int = 20,
) -> dict | None:
    """Detect contradictions between the new paper's claims and the graded
    corpus, print `⚠ contradicts [[existing#slug]]` per hit, and return a
    stats dict. `max_pairs` is intentionally lower than lint's default (50)
    since we're scoping to one paper and don't want to explode LLM cost.

    Returns {stem, n_alerts} on success, None on error. Errors are logged
    at INFO level and swallowed.
    """
    try:
        # Reuse the judge + persistence path from the lint check; only-stem
        # scoping keeps the LLM call count bounded to N × new-paper claims.
        from ..tasks.lint.cross_paper import find_cross_paper_contradictions

        hits = find_cross_paper_contradictions(
            sim_threshold=sim_threshold,
            max_pairs=max_pairs,
            only_stem=stem,
        )
        if not hits:
            return {"stem": stem, "n_alerts": 0}

        print()
        for h in hits:
            a, b = h["pair"]
            other = b if a["paper_stem"] == stem else a
            # Resolve the OTHER side's claim_slug via state.db (the judge
            # infra doesn't include it in the return dict).
            slug = _resolve_slug(other["paper_stem"], other["section"], other["position"])
            citation = format_claim_ref({"paper_stem": other["paper_stem"], "claim_slug": slug})
            print(f"⚠ contradicts {citation}  ({h['verdict']}, sim={h['similarity']:.2f})")
            if h.get("rationale"):
                print(f"    {h['rationale']}")
        log(f"contradiction-alert {stem}: {len(hits)} disagreement(s) flagged",
            tag="claim_graph")
        return {"stem": stem, "n_alerts": len(hits)}
    except Exception as e:
        log(f"contradiction-alert skipped: {type(e).__name__}: {e}",
            tag="claim_graph")
        return None


def _resolve_slug(stem: str, section: str, position: int) -> str | None:
    """Look up claim_slug for a (stem, section, position) triple."""
    try:
        from ..db.connection import get_connection
        conn = get_connection()
    except Exception:
        return None
    try:
        row = conn.execute(
            "SELECT claim_slug FROM claims "
            " WHERE paper_stem = ? AND section = ? AND position = ?",
            (stem, section, position),
        ).fetchone()
    finally:
        conn.close()
    return row["claim_slug"] if row and row["claim_slug"] else None
