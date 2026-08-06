"""Proactively cross-link a newly-ingested paper via claim overlap.

Finds existing wiki papers whose claims are near-paraphrases of the new paper's
claims (cheap cosine, no LLM), then asks a small LLM judge whether each is a
*real* relationship — the new paper builds on, extends, refines, or corroborates
the existing one — versus mere vocabulary overlap. Those get a reciprocal
`[[wikilink]]` auto-added to both pages' Related Papers.

A `measures_same` verdict is real but weaker: both claims measure the same
quantity on different cohorts, which in practice means shared methodology
rather than one paper engaging the other. It records a typed claim-graph edge
and no bullet (see `_EDGE_ONLY_VERDICTS`). Coincidences are dropped entirely.

Runs AFTER `db rebuild` (claim rows are inserted there, from the final page —
never drafts), so it belongs in the post-ingest sequence, not the ingest agent.
LLM-gated on purpose: cosine alone is topical adjacency, which the framework
forbids as a linking basis (see the cross-link corollary in CLAUDE.md).

Usage:
  researchwiki claim-overlap <stem>              # judge + auto-apply
  researchwiki claim-overlap <stem> --dry-run    # show candidates + verdicts, write nothing
  researchwiki claim-overlap <stem> --json

Exit codes: 0 = ran (including "no candidates"); 1 = unknown stem; 2 = env
(state.db or the search index unreachable — nothing to do). A missing
bi-encoder dependency is *not* 2: it's an ImportError, which propagates to the
CLI funnel as 3 with a traceback, because a broken install needs the traceback
to diagnose rather than a one-line "environment error".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Callable

from ..backlinks import append_related_paper
from ..log import log
from ..wiki import read_pages

_NOTE = "claim-grounded match (auto-added; claim-overlap)"

# Typed relation vocabulary — a superset of the earlier binary
# {cross_link, coincidence} vocabulary. The specific verdict decides BOTH what
# the claim-graph edge cache stores and whether a Related Papers bullet is
# written: see `_CROSS_LINK_VERDICTS` vs `_EDGE_ONLY_VERDICTS` below. `none`
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

# Verdicts that earn a `[[wikilink]]` in Related Papers. `refines` and
# `builds_on` are directed (NEW → EXISTING); symmetric verdicts still get a
# reciprocal bullet on both sides because Related Papers is undirected.
# Directedness only affects the edge record.
_CROSS_LINK_VERDICTS = frozenset({
    "corroborates", "refines", "builds_on",
    # Legacy alias — old prompt returned this and older callers rely on it.
    "cross_link",
})

# Verdicts that earn a typed claim-graph edge but NOT a Related Papers bullet.
#
# `measures_same` is defined in `_JUDGE_SYSTEM` as "both claims measure the same
# quantity or benchmark, but on legitimately DIFFERENT cohorts / datasets /
# conditions", and the prompt ranks it the weakest accepted relation. In practice
# it fires on shared methodology — "both quantify indel frequency by deep
# sequencing, on different CRISPR systems" — which is not the source explicitly
# citing, building on, or contrasting the other paper, so it fails CLAUDE.md's
# cross-link corollary. Measured over a 56-stem backlog drain it was 6 of 9
# confirmed links, i.e. it dominated the output with its weakest tier.
#
# The edge is still worth keeping: `concepts/refresh.py` consumes
# `measures_same` when seeding hub spokes, and "these two papers measure the
# same thing under different conditions" is a real fact about the corpus. It
# just doesn't earn Related Papers real estate.
_EDGE_ONLY_VERDICTS = frozenset({"measures_same"})

# Any verdict the judge returned that means something — used to separate a real
# relation from `none`/unparseable, independent of whether it earns a bullet.
_RELATION_VERDICTS = _CROSS_LINK_VERDICTS | _EDGE_ONLY_VERDICTS

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


def claims_fingerprint(claims: list[dict]) -> str:
    """Stable hash over a stem's claim texts.

    Order-independent (claim row order is not meaningful and `db rebuild`
    reassigns ids), so the same claim set always hashes the same. Section is
    included because the same sentence under a different H2 is a different
    claim for grading purposes.
    """
    parts = sorted(
        f"{(c.get('section') or '')}\x1f{(c.get('text') or '').strip()}"
        for c in claims
    )
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def _default_conn():
    from ..db.connection import get_connection
    return get_connection()


def _claims_for_stem(conn, stem: str) -> list[dict]:
    """Graded, non-cross-ref claims for `stem`, as the overlap finder sees them."""
    rows = conn.execute(
        "SELECT section, text FROM claims "
        " WHERE paper_stem = ? AND is_cross_ref = 0",
        (stem,),
    ).fetchall()
    return [{"section": r[0], "text": r[1]} for r in rows]


def record_run(conn, stem: str, *, fingerprint: str, n_claims: int,
               n_candidates: int, n_judged: int, n_confirmed: int,
               sim_threshold: float, ran_at: int | None = None,
               source: str = "run") -> None:
    """Mark `stem` as processed, so the backlog can exclude it.

    Upsert rather than insert: a re-run supersedes the previous record, which
    is what makes draining the backlog idempotent. `source` distinguishes a
    real execution from a `--mark-covered` back-record (see schema.sql).
    """
    conn.execute(
        "INSERT INTO claim_overlap_runs "
        "  (paper_stem, ran_at, claims_fingerprint, n_claims, n_candidates, "
        "   n_judged, n_confirmed, sim_threshold, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(paper_stem) DO UPDATE SET "
        "  ran_at=excluded.ran_at, claims_fingerprint=excluded.claims_fingerprint, "
        "  n_claims=excluded.n_claims, n_candidates=excluded.n_candidates, "
        "  n_judged=excluded.n_judged, n_confirmed=excluded.n_confirmed, "
        "  sim_threshold=excluded.sim_threshold, source=excluded.source",
        (stem, int(ran_at if ran_at is not None else time.time()), fingerprint,
         n_claims, n_candidates, n_judged, n_confirmed, float(sim_threshold),
         source),
    )
    conn.commit()


def mark_covered(*, dry_run: bool = False, conn=None) -> dict:
    """Back-record the papers the old auto-on-ingest hook already processed.

    One-time migration for this table's introduction. Until now claim-overlap
    ran automatically inside `agent ingest`, so every agent-ingested paper was
    examined — but that left no record, and without one the whole corpus looks
    like an untouched backlog.

    Restricted to papers with a row in `ingest_iterations`, because an ingest
    attempt recorded there is the evidence the hook ran. Digest-path papers were
    never covered (the hook is agent-only) and stay pending, which is the honest
    answer rather than a convenient one. A paper ingested with `--no-cross-link`
    is the one false positive this cannot detect; re-running that stem by hand is
    the remedy.

    This used to key off the `ingested-via-agent` tag, which was dropped from
    paper frontmatter on 2026-08-06 (it was the only tag 334 of 391 paper pages
    carried, and `keywords:` does the topical job). Telemetry is the better
    evidence where both exist — a recorded event rather than a self-declared
    label — but it is not a superset: measured at the swap, 326 papers carried
    the tag and 298 had telemetry, so **39 stems this once matched are no longer
    reachable**. That only matters on a machine whose `state.db` has not run this
    migration yet, since derived state is per-machine; the cost there is a
    claim-overlap backlog that over-reports rather than any incorrect data, and
    draining it is idempotent.
    """
    c = conn or _default_conn()
    rows = c.execute(
        "SELECT p.stem FROM papers p "
        " WHERE p.page_type='paper' "
        "   AND EXISTS (SELECT 1 FROM ingest_iterations i "
        "                WHERE i.paper_stem = p.stem) "
        " ORDER BY p.stem"
    ).fetchall()
    marked, no_claims = [], []
    for (stem,) in rows:
        claims = _claims_for_stem(c, stem)
        if not claims:
            no_claims.append(stem)
            continue
        if not dry_run:
            record_run(
                c, stem, fingerprint=claims_fingerprint(claims),
                n_claims=len(claims), n_candidates=0, n_judged=0, n_confirmed=0,
                sim_threshold=0.83, source="marked",
            )
        marked.append(stem)
    return {"marked": len(marked), "skipped_no_claims": len(no_claims),
            "dry_run": dry_run}


def find_backlog(conn=None) -> list[str]:
    """Paper stems with graded claims that claim-overlap has not covered.

    A stem is pending when it has no run row, or when its claims have changed
    since the recorded run (fingerprint mismatch) — a regrade or re-ingest
    means the earlier comparison no longer describes the current page.

    Paper pages only: synthesis / idea / concept / reference pages carry no
    graded paper claims, so there is nothing for the overlap finder to match.
    """
    from ..db.connection import get_connection
    c = conn or get_connection()
    stems = [
        r[0] for r in c.execute(
            "SELECT DISTINCT p.stem FROM papers p "
            "  JOIN claims cl ON cl.paper_stem = p.stem AND cl.is_cross_ref = 0 "
            " WHERE p.page_type = 'paper' ORDER BY p.stem"
        )
    ]
    recorded = {
        r[0]: r[1] for r in c.execute(
            "SELECT paper_stem, claims_fingerprint FROM claim_overlap_runs"
        )
    }
    pending = []
    for s in stems:
        want = recorded.get(s)
        if want is None or want != claims_fingerprint(_claims_for_stem(c, s)):
            pending.append(s)
    return pending


# Backlog nudge tunables. Deliberately the same shape as the `wiki/other/`
# saturation warning in `categories.py`: a size threshold plus a decay stamp, so
# `status` mentions the backlog only when it is worth one batch and then stays
# quiet for the window. Size-gated on purpose — a per-ingest reminder is exactly
# what moving claim-overlap out of the ingest path was meant to avoid.
#
# Consequence worth knowing: a backlog that never reaches the threshold is never
# surfaced. Given the measured yield (~1 confirmed link per 10 papers), an
# unprocessed tail of <10 stems costs close to nothing, so that is the intended
# trade rather than an oversight.
BACKLOG_THRESHOLD = 10
BACKLOG_DECAY_DAYS = 7
BACKLOG_STAMP = ".claim-overlap-stamp"


def _backlog_stamp_path():
    from ..paths import wiki_root
    return wiki_root() / BACKLOG_STAMP


def write_backlog_stamp() -> None:
    """Touch the dismissal stamp — called when the nudge is surfaced."""
    _backlog_stamp_path().write_text(str(int(time.time())), encoding="utf-8")


def backlog_stamp_age_days() -> float | None:
    """Days since the stamp was written, or None if absent/unreadable."""
    p = _backlog_stamp_path()
    if not p.exists():
        return None
    try:
        ts = int(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return (time.time() - ts) / 86400.0


def backlog_warning(*, touch: bool = True) -> str | None:
    """Nudge string when the backlog is worth draining, else None.

    `touch=False` peeks without affecting decay state (tests, or a caller that
    wants to compute the string and defer the "shown" semantics).
    """
    try:
        n = len(find_backlog())
    except Exception:
        return None      # no DB / cold install — nothing to say
    if n < BACKLOG_THRESHOLD:
        return None
    age = backlog_stamp_age_days()
    if age is not None and age < BACKLOG_DECAY_DAYS:
        return None
    if touch:
        write_backlog_stamp()
    return (
        f"Claim-overlap backlog: {n} stem(s) pending\n"
        f"  → researchwiki claim-overlap --backlog"
    )


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
    new_body = new_page.path.read_text(encoding="utf-8") if new_page.path.exists() else ""
    judge = judge_fn or _default_judge

    applied, edge_only, coincidence, skipped, judge_failed = [], [], [], [], []
    for cand in candidates:
        ex = pages.get(cand.existing_stem)
        if ex is None:
            continue
        ex_key = ex.key
        # Already linked either direction → no judge call, no work.
        if f"[[{ex_key}]]" in new_body or f"[[{new_key}]]" in (
            ex.path.read_text(encoding="utf-8") if ex.path.exists() else ""
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
        if v not in _RELATION_VERDICTS:
            coincidence.append({"existing": ex_key, "cosine": cand.cosine,
                                "rationale": rationale, "verdict": v})
            continue

        relation = _relation_from_verdict(v)
        record = {"existing": ex_key, "cosine": cand.cosine,
                  "rationale": rationale, "relation": relation}
        links = v in _CROSS_LINK_VERDICTS
        if not dry_run:
            # Typed edge lands for every real relation, and regardless of
            # whether a wikilink was already present — the edge cache is
            # orthogonal to Related Papers.
            if relation is not None:
                _persist_typed_edge(
                    stem, cand.new_claim,
                    cand.existing_stem, cand.existing_claim,
                    relation, rationale, cand.cosine,
                )
            if links:
                wrote_new = append_related_paper(new_page.path, ex_key, note=_NOTE)
                wrote_ex = append_related_paper(ex.path, new_key, note=_NOTE)
                record["wrote"] = {"new_page": wrote_new, "existing_page": wrote_ex}
                if wrote_new:  # keep in-memory body current so a later candidate sees the link
                    new_body = new_page.path.read_text(encoding="utf-8")
        (applied if links else edge_only).append(record)

    # Mark the stem covered so `find_backlog` stops returning it. Not on a dry
    # run: nothing was applied, so claiming coverage would hide real work.
    #
    # The fingerprint is taken over the claims actually compared. On the ingest
    # hook path those come from the committed page rather than the DB (claims
    # land at `db rebuild`), so if rebuild normalises text differently the
    # fingerprint won't match later and the stem resurfaces once — after which
    # the DB-derived fingerprint is recorded and matches thereafter. Costs one
    # redundant re-run in the worst case; never silently drops a stem.
    if not dry_run and not judge_failed:
        compared = new_claims if new_claims is not None else _claims_for_stem(
            conn or _default_conn(), stem
        )
        try:
            record_run(
                conn or _default_conn(), stem,
                fingerprint=claims_fingerprint(compared),
                n_claims=len(compared),
                n_candidates=len(candidates),
                n_judged=len(applied) + len(edge_only) + len(coincidence),
                n_confirmed=len(applied),
                sim_threshold=sim_threshold,
            )
        except Exception as e:      # never fail the cross-link over bookkeeping
            log(f"could not record run for {stem}: {type(e).__name__}: {e}",
                tag="claim-overlap")

    return {
        "stem": stem,
        "n_candidates": len(candidates),
        "applied": applied,
        # Real relation, typed edge written, deliberately no bullet — see
        # _EDGE_ONLY_VERDICTS. Distinct from `coincidence`, which is the
        # judge saying the pair is unrelated.
        "edge_only": edge_only,
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


def _run_backlog(args) -> int:
    """Drain the uncovered-stem backlog, one stem at a time.

    Sequential on purpose: each stem's judge calls are cheap (~3) but the
    finder holds a bi-encoder, and a confirmed link mutates both pages — so a
    later stem must see the links an earlier one wrote, or the two runs
    disagree about what is already linked.
    """
    pending = find_backlog()
    if not pending:
        print("claim-overlap backlog is empty — every paper page with claims "
              "has been covered at its current claim set.")
        return 0

    withheld = 0
    if args.limit is not None and len(pending) > args.limit:
        withheld = len(pending) - args.limit
        pending = pending[:args.limit]
    if not pending:
        # `--limit 0`. Distinguished from a genuinely empty backlog above,
        # so the report can't claim coverage that doesn't exist.
        print(f"claim-overlap backlog: {withheld} stem(s) pending, all withheld "
              f"by --limit 0.")
        return 0

    print(f"backlog: {len(pending)} stem(s)"
          + (f", {withheld} withheld by --limit" if withheld else "")
          + (" — DRY RUN, no links written" if args.dry_run else ""))

    results, failed = [], []
    tot = {"candidates": 0, "judged": 0, "confirmed": 0, "edge_only": 0}
    for i, stem in enumerate(pending, 1):
        try:
            r = run(stem, sim_threshold=args.sim, top_papers=args.top,
                    dry_run=args.dry_run)
        except LookupError:
            # In the DB but not on disk — a page deleted without a rebuild.
            failed.append((stem, "no wiki page"))
            continue
        results.append(r)
        edge_only = len(r.get("edge_only", []))
        judged = len(r["applied"]) + edge_only + len(r["coincidence"])
        tot["candidates"] += r["n_candidates"]
        tot["judged"] += judged
        tot["confirmed"] += len(r["applied"])
        tot["edge_only"] += edge_only
        if not args.json:
            mark = "✓" if r["applied"] else " "
            print(f"  {mark} [{i}/{len(pending)}] {stem[:56]:56s} "
                  f"cand {r['n_candidates']:2d}  judged {judged:2d}  "
                  f"confirmed {len(r['applied'])}")
            for a in r["applied"]:
                print(f"        → [[{a['existing']}]] (cos {a['cosine']}) {a['rationale'][:70]}")

    if args.json:
        print(json.dumps({"pending": len(pending), "withheld": withheld,
                          "totals": tot, "failed": failed,
                          "results": results}, ensure_ascii=False, indent=2))
        return 0

    print(f"\n  {len(results)} stem(s) processed: {tot['candidates']} candidate(s), "
          f"{tot['judged']} judged, {tot['confirmed']} confirmed link(s), "
          f"{tot['edge_only']} edge-only (typed edge, no bullet)")
    for stem, why in failed:
        print(f"  ! skipped {stem}: {why}")
    if not args.dry_run and tot["confirmed"]:
        print("  links written to both pages — run `researchwiki db rebuild && "
              "researchwiki reindex`")
    log(f"claim-overlap backlog: {len(results)} stem(s), {tot['confirmed']} link(s) "
        f"{'(dry-run)' if args.dry_run else 'applied'}", tag="claim-overlap")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki claim-overlap",
        description="Auto cross-link a new paper to existing papers via claim overlap (LLM-gated).",
    )
    parser.add_argument("stem", nargs="?",
                        help="Stem of the newly-ingested paper (e.g. smith-2024-...).")
    parser.add_argument("--backlog", action="store_true",
                        help="Process every stem claim-overlap has not covered yet.")
    parser.add_argument("--mark-covered", action="store_true",
                        help="One-time: back-record agent-ingested papers the old "
                             "auto-on-ingest hook already processed.")
    parser.add_argument("--limit", type=int, default=None,
                        help="With --backlog: process at most N stems.")
    parser.add_argument("--sim", type=float, default=0.83, help="Cosine threshold (default 0.83).")
    parser.add_argument("--top", type=int, default=10, help="Max existing papers to judge (default 10).")
    parser.add_argument("--dry-run", action="store_true", help="Show verdicts, write no links.")
    parser.add_argument("--json", action="store_true", help="Emit the decisions as JSON.")
    args = parser.parse_args(argv)

    if args.mark_covered:
        if args.stem or args.backlog:
            parser.error("--mark-covered takes no stem and does not combine with --backlog")
        res = mark_covered(dry_run=args.dry_run)
        verb = "would mark" if res["dry_run"] else "marked"
        print(f"{verb} {res['marked']} agent-ingested paper(s) as covered"
              + (f"; {res['skipped_no_claims']} had no claims and stay pending"
                 if res["skipped_no_claims"] else ""))
        print(f"remaining backlog: {len(find_backlog())} stem(s)")
        return 0

    if args.backlog and args.stem:
        parser.error("--backlog processes the whole backlog; don't also pass a stem")
    if not args.backlog and not args.stem:
        parser.error("pass a stem, --backlog, or --mark-covered")

    if args.backlog:
        return _run_backlog(args)

    try:
        result = run(args.stem, sim_threshold=args.sim, top_papers=args.top, dry_run=args.dry_run)
    except LookupError:
        print(f"researchwiki claim-overlap: no wiki page for stem `{args.stem}`", file=sys.stderr)
        return 1
    # Deliberately no `except Exception: return 2` here. state.db and the search
    # index raise `EnvironmentFailure`, which the funnel already reports as 2;
    # an ImportError on numpy/torch is a broken install, which the funnel's
    # code-3 traceback diagnoses far better than a one-line message would.

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if result["n_candidates"] == 0:
        print(f"No claim-overlap candidates for {args.stem} (nothing ≥ {args.sim} cosine).")
        return 0
    verb = "would link" if args.dry_run else "linked"
    for a in result["applied"]:
        print(f"  ✓ {verb} [[{a['existing']}]]  (cos {a['cosine']}) — {a['rationale']}")
    for e in result.get("edge_only", []):
        print(f"  ~ edge only [[{e['existing']}]]  (cos {e['cosine']}, "
              f"{e['relation']}) — typed edge recorded, no Related Papers bullet")
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
        f"{len(result.get('edge_only', []))} edge-only, "
        f"{len(result['coincidence'])} coincidence, {len(result['skipped'])} already-linked",
        tag="claim-overlap")
    return 0
