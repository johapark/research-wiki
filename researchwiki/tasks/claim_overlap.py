"""Proactively cross-link a newly-ingested paper via claim overlap.

Finds existing wiki papers whose claims are near-paraphrases of the new paper's
claims (cheap cosine, no LLM), then asks a small LLM judge whether each is a
*real* relationship — the new paper builds on / extends / directly contrasts /
measures the same thing as the existing one — versus mere vocabulary overlap.
Confirmed matches get a reciprocal `[[wikilink]]` auto-added to both pages'
Related Papers. Coincidences are dropped.

Runs AFTER `db rebuild` (claim rows are inserted there, from the final page —
never drafts), so it belongs in the post-ingest sequence, not the ingest agent.
LLM-gated on purpose: cosine alone is topical adjacency, which the framework
forbids as a linking basis (see the cross-link corollary in CLAUDE.md).

Usage:
  researchwiki claim-overlap <stem>              # judge + auto-apply
  researchwiki claim-overlap <stem> --dry-run    # show candidates + verdicts, write nothing
  researchwiki claim-overlap <stem> --json

Exit codes: 0 = ran (including "no candidates"); 1 = unknown stem; 2 = env
(DB/bi-encoder unreachable — nothing to do).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable

from ..backlinks import append_related_paper
from ..log import log
from ..wiki import read_pages

_NOTE = "claim-grounded match (auto-added; claim-overlap)"

# Typed relation vocabulary — a superset of the earlier binary
# {cross_link, coincidence} vocabulary. Any non-"none" verdict counts as a
# cross-link (the reciprocal `[[wikilink]]` in Related Papers is applied),
# but the specific verdict is what the claim-graph edge cache stores. `none`
# replaces the earlier `coincidence`; the older term is still accepted as a
# fallback for callers that haven't migrated yet.
_JUDGE_SYSTEM = """\
You classify the relation between two research-paper claims for a research wiki.

Claim NEW is from a newly-added paper; claim EXISTING is from a paper already
in the wiki. They were retrieved as near-paraphrases by embedding similarity —
so they use similar words. Similar words are NOT enough.

Pick ONE verdict:

  corroborates    Independent papers affirming the SAME finding on the same
                  entity/benchmark, arriving from different setups. Both can
                  be true simultaneously and they strengthen each other.
  measures_same   Both claims measure the same quantity or benchmark, but on
                  legitimately DIFFERENT cohorts / datasets / conditions —
                  the values may differ without contradicting. Use this
                  instead of `corroborates` when the setups are not directly
                  comparable.
  refines         NEW sharpens, conditions, or qualifies the finding in
                  EXISTING (e.g. narrows the applicable regime, adds a
                  constraint, corrects a scope).
  builds_on      NEW generalizes, extends, or reuses the method/result of
                  EXISTING. Directed: NEW → EXISTING.
  none            The embeddings clustered them but the claims describe
                  DIFFERENT things — different entities, domains, datasets,
                  or contexts that merely sound alike. No cross-link.

Be conservative on non-none verdicts. When multiple could apply, prefer the
weakest (measures_same < corroborates < refines < builds_on). When the
setups are unclear, use `none`.

Output strict JSON: {"verdict": "corroborates" | "measures_same" | "refines" | "builds_on" | "none", "rationale": "one short sentence"}.
"""

_JUDGE_SCHEMA = {
    "type": "object",
    "required": ["verdict", "rationale"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["corroborates", "measures_same", "refines", "builds_on", "none"],
        },
        "rationale": {"type": "string"},
    },
}

# Which verdicts imply a paper-level cross-link. `refines` and `builds_on`
# are directed (NEW → EXISTING); symmetric verdicts still get a reciprocal
# [[wikilink]] on both sides because the wiki's Related Papers section is
# undirected. Directedness only affects the edge record.
_CROSS_LINK_VERDICTS = frozenset({
    "corroborates", "measures_same", "refines", "builds_on",
    # Legacy alias — old prompt returned this and older callers rely on it.
    "cross_link",
})
_DIRECTED_VERDICTS = frozenset({"refines", "builds_on"})


def _relation_from_verdict(v: str) -> str | None:
    """Map a judge verdict to a claim-graph relation, or None when the pair
    should not be linked. Legacy `cross_link` maps to `builds_on` — the
    closest typed cousin — so old callers still emit a well-typed edge."""
    if v in ("corroborates", "measures_same", "refines", "builds_on"):
        return v
    if v == "cross_link":
        return "builds_on"
    return None


def _persist_typed_edge(
    new_stem: str, new_claim: dict,
    existing_stem: str, existing_claim: dict,
    relation: str, rationale: str, cosine: float,
) -> None:
    """Side-write a typed edge to `.claim-graph/edges.db`. Slug-computed on
    both sides (deterministic — same as _upsert_claims). Silent no-op on
    error; the cross-link itself is the primary outcome, edge write is a
    bonus for the claim graph."""
    try:
        from ..claim_graph import (
            Edge, SLUG_SCHEME_VERSION, compute_claim_slug, open_edges_db, upsert_edge,
        )
        new_slug = compute_claim_slug(new_claim["section"], new_claim["text"])
        old_slug = compute_claim_slug(existing_claim["section"], existing_claim["text"])
        directed = relation in _DIRECTED_VERDICTS
        conn = open_edges_db()
        try:
            upsert_edge(conn, Edge(
                src_stem=new_stem, src_slug=new_slug,
                tgt_stem=existing_stem, tgt_slug=old_slug,
                relation=relation, directed=directed,
                confidence=float(cosine),
                rationale=rationale[:400] if rationale else "",
                judge_phase="claim_overlap_judge",
                slug_scheme_version=SLUG_SCHEME_VERSION,
                status="candidate",
            ))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        log(f"claim-overlap edge persistence skipped: {type(e).__name__}: {e}",
            tag="claim-overlap")


def _format_prompt(new_stem: str, existing_stem: str, cand) -> str:
    nc, ec = cand.new_claim, cand.existing_claim
    return (
        f"Claim NEW — paper [[{new_stem}]] ({nc['section']}#{nc['position']}):\n"
        f"  {nc['text']}\n\n"
        f"Claim EXISTING — paper [[{existing_stem}]] ({ec['section']}#{ec['position']}):\n"
        f"  {ec['text']}\n\n"
        f"(embedding cosine {cand.cosine})\n\nOutput JSON only."
    )


def _default_judge(prompt: str) -> dict | None:
    from ..agents.judge import run_llm_judge
    return run_llm_judge(
        phase="claim_overlap_judge", system=_JUDGE_SYSTEM, prompt=prompt, schema=_JUDGE_SCHEMA,
    )


def claims_from_page(stem: str, page_path) -> list[dict]:
    """Parse a committed page's gradable claims into overlap-input dicts.

    Used by the ingest hook so cross-linking can run right after promote —
    before `db rebuild` has inserted the new paper's claim rows.
    """
    from pathlib import Path
    from ..grade.parser import parse_claims
    from ..wiki import read_page
    page = read_page(Path(page_path))
    if page is None:
        return []
    return [
        {"paper_stem": stem, "section": c.section, "position": c.position, "text": c.text}
        for c in parse_claims(page) if not c.is_cross_ref
    ]


def run(
    stem: str,
    *,
    new_claims: list[dict] | None = None,
    sim_threshold: float = 0.83,
    top_papers: int = 10,
    dry_run: bool = False,
    judge_fn: Callable[[str], dict | None] | None = None,
    conn=None,
) -> dict:
    """Core: find overlaps → judge → (optionally) apply reciprocal links.

    `new_claims` lets the caller supply the new paper's claims directly (ingest
    hook, pre-rebuild); when None they're read from the DB. Returns a decisions
    dict. `judge_fn`/`conn` are injectable for tests. Raises LookupError if
    `stem` isn't a wiki page.
    """
    pages = {p.stem: p for p in read_pages()}
    if stem not in pages:
        raise LookupError(stem)

    from ..grade.claim_overlap import find_claim_overlaps
    candidates = find_claim_overlaps(
        stem, new_claims=new_claims, sim_threshold=sim_threshold,
        top_papers=top_papers, conn=conn,
    )

    new_page = pages[stem]
    new_key = new_page.key
    new_body = new_page.path.read_text() if new_page.path.exists() else ""
    judge = judge_fn or _default_judge

    applied, coincidence, skipped, judge_failed = [], [], [], []
    for cand in candidates:
        ex = pages.get(cand.existing_stem)
        if ex is None:
            continue
        ex_key = ex.key
        # Already linked either direction → no judge call, no work.
        if f"[[{ex_key}]]" in new_body or f"[[{new_key}]]" in (
            ex.path.read_text() if ex.path.exists() else ""
        ):
            skipped.append({"existing": ex_key, "reason": "already-linked", "cosine": cand.cosine})
            continue

        verdict = judge(_format_prompt(stem, cand.existing_stem, cand))
        if verdict is None:
            # No verdict ≠ coincidence — the judge was unreachable / returned
            # unparseable output. Surface it distinctly so a provider/config
            # problem doesn't masquerade as "nothing to link".
            judge_failed.append({"existing": ex_key, "cosine": cand.cosine})
            continue
        v = verdict.get("verdict")
        rationale = verdict.get("rationale", "")
        if v not in _CROSS_LINK_VERDICTS:
            coincidence.append({"existing": ex_key, "cosine": cand.cosine,
                                "rationale": rationale, "verdict": v})
            continue

        relation = _relation_from_verdict(v)
        record = {"existing": ex_key, "cosine": cand.cosine,
                  "rationale": rationale, "relation": relation}
        if not dry_run:
            wrote_new = append_related_paper(new_page.path, ex_key, note=_NOTE)
            wrote_ex = append_related_paper(ex.path, new_key, note=_NOTE)
            record["wrote"] = {"new_page": wrote_new, "existing_page": wrote_ex}
            if wrote_new:  # keep in-memory body current so a later candidate sees the link
                new_body = new_page.path.read_text()
            # Typed edge lands regardless of whether the wikilink was already
            # present — the edge cache is orthogonal to Related Papers.
            if relation is not None:
                _persist_typed_edge(
                    stem, cand.new_claim,
                    cand.existing_stem, cand.existing_claim,
                    relation, rationale, cand.cosine,
                )
        applied.append(record)

    return {
        "stem": stem,
        "n_candidates": len(candidates),
        "applied": applied,
        "coincidence": coincidence,
        "skipped": skipped,
        "judge_failed": judge_failed,
        "dry_run": dry_run,
    }


def run_after_ingest(stem: str, committed_path, *, sim_threshold: float = 0.83) -> dict | None:
    """Ingest-hook entry point: cross-link a just-promoted paper.

    Reads the new paper's claims from its committed page (they aren't in the DB
    until `db rebuild`), runs the judge + auto-apply against the existing corpus,
    and prints a short summary. Returns the decisions dict, or None when there
    was nothing to do / the page isn't in wiki/ (sandboxed) / infra is missing —
    all non-fatal so a hiccup never fails the ingest.
    """
    try:
        new_claims = claims_from_page(stem, committed_path)
        if not new_claims:
            return None
        result = run(stem, new_claims=new_claims, sim_threshold=sim_threshold)
    except LookupError:
        return None  # page not promoted to wiki/ (sandboxed) — nothing to link
    except Exception as e:  # never let cross-linking break an otherwise-good ingest
        log(f"claim-overlap hook skipped: {type(e).__name__}: {e}", tag="claim-overlap")
        return None

    n = len(result["applied"])
    if n:
        print()
        print(f"claim-overlap → auto-added {n} cross-link(s):")
        for a in result["applied"]:
            print(f"  ✓ [[{a['existing']}]]  (cos {a['cosine']}) — {a['rationale']}")
    if result["judge_failed"]:
        print(f"  ⚠ judge gave no verdict for {len(result['judge_failed'])} candidate(s) "
              f"(LLM provider unreachable?) — not linked.")
    log(f"claim-overlap {stem}: {n} link(s), {len(result['coincidence'])} coincidence, "
        f"{len(result['skipped'])} already-linked", tag="claim-overlap")
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki claim-overlap",
        description="Auto cross-link a new paper to existing papers via claim overlap (LLM-gated).",
    )
    parser.add_argument("stem", help="Stem of the newly-ingested paper (e.g. smith-2024-...).")
    parser.add_argument("--sim", type=float, default=0.83, help="Cosine threshold (default 0.83).")
    parser.add_argument("--top", type=int, default=10, help="Max existing papers to judge (default 10).")
    parser.add_argument("--dry-run", action="store_true", help="Show verdicts, write no links.")
    parser.add_argument("--json", action="store_true", help="Emit the decisions as JSON.")
    args = parser.parse_args(argv)

    try:
        result = run(args.stem, sim_threshold=args.sim, top_papers=args.top, dry_run=args.dry_run)
    except LookupError:
        print(f"researchwiki claim-overlap: no wiki page for stem `{args.stem}`", file=sys.stderr)
        return 1
    except Exception as e:
        # DB or bi-encoder unreachable (numpy/torch import, embedding load, or
        # state.db access) — environment failure per the exit-code contract.
        print(f"researchwiki claim-overlap: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if result["n_candidates"] == 0:
        print(f"No claim-overlap candidates for {args.stem} (nothing ≥ {args.sim} cosine).")
        return 0
    verb = "would link" if args.dry_run else "linked"
    for a in result["applied"]:
        print(f"  ✓ {verb} [[{a['existing']}]]  (cos {a['cosine']}) — {a['rationale']}")
    for c in result["coincidence"]:
        print(f"  · skipped [[{c['existing']}]]  (cos {c['cosine']}, coincidence)")
    for s in result["skipped"]:
        print(f"  – {s['reason']} [[{s['existing']}]]  (cos {s['cosine']})")
    if result["judge_failed"]:
        print(f"  ⚠ judge returned no verdict for {len(result['judge_failed'])} candidate(s) — "
              f"LLM provider unreachable or misconfigured (check config/models.yaml + RW_LLM_BASE_URL). "
              f"These were NOT linked.")
    n_applied = len(result["applied"])
    log(f"claim-overlap {args.stem}: {n_applied} cross-link(s) "
        f"{'(dry-run)' if args.dry_run else 'applied'}, "
        f"{len(result['coincidence'])} coincidence, {len(result['skipped'])} already-linked",
        tag="claim-overlap")
    return 0
