"""Judge claim overlap, verify bibliography identity, and apply reversible links.

Stem and exact-pair review share canonical A/B judgments and explicit direction.
A relationship writes a typed graph edge; Related Papers additionally requires
entry-scoped PDF citation proof. Receipts bind actual insertions to the existing
mutation journal, and recover completion records after interrupted maintenance.

Usage:
  researchwiki claim-overlap STEM [--dry-run] [--json]
  researchwiki claim-overlap --pair A#slug B#slug
  researchwiki claim-overlap --backlog --progress-json PATH
  researchwiki claim-overlap --revert RUN_ID [--stem STEM] [--link A B] [--apply]

Exit codes: 0 ran, 1 invalid stem/pair/revert, 2 environment failure, 3 internal
error. JSON errors retain a traceback on stderr for internal failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

from ..errors import EnvironmentFailure
from ..overlap import judging as _judging
from ..log import log
from ..wiki import read_pages
from ..paths import papers_dir

_NOTE = "claim-grounded match (auto-added; claim-overlap)"


def _source_cites_target(source, target, *, verifier=None) -> bool:
    from ..pdf.citation import CitationVerifier, target_identity
    own_doi = source.str_field("doi", "").strip().lower()
    if own_doi and own_doi == target.str_field("doi", "").strip().lower():
        return False
    return (verifier or CitationVerifier()).verify(
        papers_dir() / f"{source.stem}.pdf", source.stem, target_identity(target))["verified"]


def _pair_has_explicit_citation(page_a, page_b) -> bool:
    from ..pdf.citation import CitationVerifier
    verifier = CitationVerifier()
    return (_source_cites_target(page_a, page_b, verifier=verifier)
            or _source_cites_target(page_b, page_a, verifier=verifier))


_CROSS_LINK_VERDICTS = _judging.CROSS_LINK_VERDICTS
_EDGE_ONLY_VERDICTS = _judging.EDGE_ONLY_VERDICTS
_RELATION_VERDICTS = _judging.RELATION_VERDICTS


def _relation_from_verdict(value):
    return value if value in _RELATION_VERDICTS else None


_DIRECTED_VERDICTS = frozenset({"refines", "builds_on"})


def _persist_typed_edge(
    new_stem: str, new_claim: dict,
    existing_stem: str, existing_claim: dict,
    relation: str, rationale: str, cosine: float, *, judge_model: str = "",
) -> None:
    """Persist a canonical edge with actual provenance, preserving human rows.

    Failures propagate to receipt maintenance; they never silently close a stem.
    """
    try:
        from ..claim_graph import (
            Edge, SLUG_SCHEME_VERSION, compute_claim_slug, open_edges_db, upsert_edge,
        )
        new_slug = compute_claim_slug(new_claim["section"], new_claim["text"])
        old_slug = compute_claim_slug(existing_claim["section"], existing_claim["text"])
        directed = relation in _DIRECTED_VERDICTS
        if not directed and (new_stem, new_slug) > (existing_stem, old_slug):
            new_stem, existing_stem = existing_stem, new_stem
            new_slug, old_slug = old_slug, new_slug
        conn = open_edges_db()
        try:
            upsert_edge(conn, Edge(
                src_stem=new_stem, src_slug=new_slug,
                tgt_stem=existing_stem, tgt_slug=old_slug,
                relation=relation, directed=directed,
                confidence=float(cosine),
                rationale=rationale[:400] if rationale else "",
                judge_phase="claim_overlap_judge",
                judge_model=judge_model,
                slug_scheme_version=SLUG_SCHEME_VERSION,
                status="candidate",
            ))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        log(f"claim-overlap edge persistence failed: {type(e).__name__}: {e}",
            tag="claim-overlap")
        raise


def _parse_claim_ref(raw: str) -> tuple[str, str]:
    """Parse ``stem#claim_slug`` or ``[[category/stem#claim_slug]]``.

    Category prefixes are optional because the durable citation form emitted by
    ``claims`` and ``candidates pairs`` is ``[[stem#slug]]``.  Exact review is
    deliberately claim-addressed: a paper-level pair would reintroduce the
    threshold/top-k ambiguity that the discovery queue was built to avoid.
    """
    text = raw.strip()
    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2].split("|", 1)[0].strip()
    if "#" not in text:
        raise ValueError(f"claim reference needs #claim_slug: {raw!r}")
    stem, slug = (part.strip() for part in text.rsplit("#", 1))
    stem = stem.rsplit("/", 1)[-1]
    if not stem or not slug:
        raise ValueError(f"invalid claim reference: {raw!r}")
    return stem, slug


def _exact_claim(ref: str, conn) -> dict:
    """Resolve a durable claim reference to one current paper-claim row."""
    stem, slug = _parse_claim_ref(ref)
    row = conn.execute(
        "SELECT c.paper_stem, c.claim_slug, c.section, c.position, c.text, p.category "
        "  FROM claims c JOIN papers p ON p.stem = c.paper_stem "
        " WHERE c.paper_stem = ? AND c.claim_slug = ? "
        "   AND c.is_cross_ref = 0 AND p.page_type = 'paper'",
        (stem, slug),
    ).fetchone()
    if row is None:
        raise ValueError(f"no graded paper claim matches {ref!r}")
    return dict(row)


def _exact_pair_cosine(claim_a: dict, claim_b: dict) -> float:
    """Best-effort cached cosine for the edge record.

    The human selected the pair, so a cache miss must not prevent judging it.
    Do *not* call ``get_claim_embeddings`` here: that writer rewrites the
    cache to the supplied row set and an exact two-claim review would evict the
    corpus-wide cache that discovery relies on. ``0.0`` means unavailable, not
    unrelated.
    """
    try:
        from ..index.claim_embeddings import load_cached_claim_embeddings
        loaded = load_cached_claim_embeddings([claim_a, claim_b])
        if loaded is not None:
            vecs, indices = loaded
            if indices == [0, 1] and len(vecs) == 2:
                return round(float(vecs[0] @ vecs[1]), 4)
    except Exception:
        pass
    return 0.0


def review_pair(ref_a: str, ref_b: str, *, dry_run=False, judge_fn=None, conn=None,
                session=None, run_id=None) -> dict:
    from types import SimpleNamespace
    from ..overlap.workflow import evaluate
    from ..overlap.receipts import recover
    c = conn or _default_conn()
    close = conn is None
    try:
        if not dry_run:
            recover(conn=c)
        a, b = _exact_claim(ref_a, c), _exact_claim(ref_b, c)
        if a["paper_stem"] == b["paper_stem"]:
            raise ValueError("exact review needs claims from two different papers")
        pages = {p.stem: p for p in read_pages()}
        if a["paper_stem"] not in pages or b["paper_stem"] not in pages:
            raise ValueError("one or both claim papers are absent from wiki/")
        cosine = _exact_pair_cosine(a, b)
        cand = SimpleNamespace(existing_stem=b["paper_stem"], new_claim=a,
                               existing_claim=b, cosine=cosine)
        result = evaluate(a["paper_stem"], [cand], pages, compared=[], conn=c,
                          sim_threshold=0.83, dry_run=dry_run, judge_fn=judge_fn,
                          session=session, run_id=run_id, exact=True)
        result["pair"] = [f"[[{p['paper_stem']}#{p['claim_slug']}]]" for p in (a, b)]
        result["cosine"] = cosine
        return result
    finally:
        if close:
            c.close()


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
    recorded = {
        r[0]: r[1] for r in c.execute(
            "SELECT paper_stem, claims_fingerprint FROM claim_overlap_runs"
        )
    }
    # One pass over every paper's claims, grouped in Python. This used to issue
    # one query per paper (~480 on a 500-paper corpus), which made the check
    # over a second of `status` on its own. The rows and filter are exactly
    # `_claims_for_stem`'s, so fingerprints match those `record_run` stored.
    claims_by_stem: dict[str, list[dict]] = {}
    for stem, section, text in c.execute(
        "SELECT cl.paper_stem, cl.section, cl.text FROM claims cl "
        "  JOIN papers p ON p.stem = cl.paper_stem "
        " WHERE p.page_type = 'paper' AND cl.is_cross_ref = 0"
    ):
        claims_by_stem.setdefault(stem, []).append({"section": section, "text": text})
    pending = []
    for stem in sorted(claims_by_stem):
        want = recorded.get(stem)
        if want is None or want != claims_fingerprint(claims_by_stem[stem]):
            pending.append(stem)
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
    # Decay check first: it is one stat(), and the backlog scan reads every
    # claim in the corpus. Checked after the scan, `status` paid for that scan
    # on every run of the quiet window only to discard the answer.
    age = backlog_stamp_age_days()
    if age is not None and age < BACKLOG_DECAY_DAYS:
        return None
    try:
        n = len(find_backlog())
    except Exception:
        return None      # no DB / cold install — nothing to say
    if n < BACKLOG_THRESHOLD:
        return None
    if touch:
        write_backlog_stamp()
    return (
        f"Claim-overlap backlog: {n} stem(s) pending\n"
        f"  → researchwiki claim-overlap --backlog"
    )


def run(stem: str, *, new_claims=None, sim_threshold=0.83, top_papers=10,
        dry_run=False, judge_fn=None, conn=None, session=None, run_id=None) -> dict:
    from ..grade.claim_overlap import find_claim_overlaps
    from ..overlap.workflow import evaluate
    from ..overlap.receipts import recover
    c = conn or _default_conn()
    close = conn is None
    try:
        if not dry_run:
            recover(conn=c)
        pages = {p.stem: p for p in read_pages()}
        if stem not in pages:
            raise LookupError(stem)
        compared = new_claims if new_claims is not None else _claims_for_stem(c, stem)
        candidates = find_claim_overlaps(stem, new_claims=new_claims,
            sim_threshold=sim_threshold, top_papers=top_papers, conn=c)
        return evaluate(stem, candidates, pages, compared=compared, conn=c,
                        sim_threshold=sim_threshold, dry_run=dry_run, judge_fn=judge_fn,
                        session=session, run_id=run_id)
    finally:
        if close:
            c.close()


def run_after_ingest(stem: str, committed_path, *, sim_threshold: float = 0.83,
                     render: bool = True) -> dict | None:
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

    if render:
        print_ingest_result(result)
    log(f"claim-overlap {stem}: {len(result['applied'])} link(s), {len(result['coincidence'])} coincidence, "
        f"{len(result['skipped'])} already-linked", tag="claim-overlap")
    return result


def print_ingest_result(result: dict) -> None:
    """Render an already-computed receipt without repeating model work."""
    n = len(result["applied"])
    if n:
        print()
        print(f"claim-overlap → auto-added {n} cross-link(s):")
        for a in result["applied"]:
            print(f"  ✓ [[{a['existing']}]]  (cos {a['cosine']}) — {a['rationale']}")
    if result["judge_failed"]:
        print(f"  ⚠ judge gave no verdict for {len(result['judge_failed'])} candidate(s) "
              f"(LLM provider unreachable?) — not linked.")


def _run_backlog(args) -> int:
    from ..overlap.judging import JudgeSession
    from ..overlap.receipts import new_run_id, recover, run_path
    from ..overlap.progress import emit
    from ..fsatomic import write_json_atomic
    if not args.dry_run:
        recover()
    pending = find_backlog()
    all_pending = list(pending)
    withheld = 0
    if args.limit is not None:
        withheld = max(0, len(pending) - args.limit)
        pending = pending[:args.limit]
    run_id = new_run_id()
    progress = getattr(args, "progress_json", None)
    if not args.json:
        if not all_pending:
            print("claim-overlap backlog is empty — every paper page with claims has been covered.")
        else:
            print(f"backlog: {len(pending)} stem(s), {withheld} withheld by --limit"
                  + (" — DRY RUN, no links written" if args.dry_run else ""))
    results, failed = [], []
    retryable, completed, maintenance = [], [], []
    stopped_early = None
    totals = {k: 0 for k in ("candidates", "judged", "confirmed", "edge_only",
                             "judged_pairs", "reused_judgments", "committed_link_pairs")}
    session = JudgeSession(dry_run=args.dry_run)
    for i, stem in enumerate(pending, 1):
        halted = False
        try:
            r = run(stem, sim_threshold=args.sim, top_papers=args.top,
                    dry_run=args.dry_run, session=session, run_id=run_id)
        except LookupError as exc:
            if type(exc) is not LookupError:
                raise
            failed.append((stem, "no wiki page"))
            retryable.append(stem)
            emit(progress, run_id=run_id, stem=stem, completed=len(completed), total=len(pending),
                 result={"status": "incomplete", "judge_failed": [{}]})
            continue
        except EnvironmentFailure as exc:
            stopped_early = f"{type(exc).__name__}: {exc}"
            r = getattr(exc, "claim_overlap_result", None)
            retryable.extend(pending[i:] if r else pending[i - 1:])
            if r is None:
                break
            halted = True
        r.setdefault("stem", stem)
        results.append(r)
        if r.get("judge_failed"):
            retryable.append(stem)
        elif r.get("status") == "maintenance-pending":
            maintenance.append(stem)
            retryable.append(stem)
        else:
            completed.append(stem)
        judged = sum(len(r.get(k, [])) for k in ("applied", "edge_only", "coincidence"))
        totals["candidates"] += r["n_candidates"]
        totals["judged"] += judged
        totals["confirmed"] += len(r["applied"])
        totals["edge_only"] += len(r.get("edge_only", []))
        for k in ("judged_pairs", "reused_judgments", "committed_link_pairs"):
            totals[k] += r.get(k, 0)
        emit(progress, run_id=run_id, stem=stem, completed=len(completed), total=len(pending), result=r)
        if not args.json:
            print(f"  [{i}/{len(pending)}] {stem}: {judged} judged, {len(r['applied'])} confirmed")
        if halted:
            break
    summary = {"schema_version": 1, "record_type": "run", "run_id": run_id,
               "pending": len(pending), "withheld": withheld, "totals": totals,
               "failed": failed, "stopped_early": stopped_early, "results": results,
               "completed_stems": completed, "retryable_stems": retryable,
               "maintenance_pending_stems": maintenance}
    if not args.dry_run:
        path = run_path(run_id) / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, summary)
    emit(progress, run_id=run_id, completed=len(completed), total=len(pending), retryable=retryable,
         result={"status": "incomplete" if retryable else "completed"})
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"\n  {len(results)} stem(s) processed: {totals['candidates']} candidate(s), "
              f"{totals['judged']} judged, {totals['confirmed']} confirmed link(s), "
              f"{totals['edge_only']} edge-only")
        for stem, why in failed:
            print(f"  ! skipped {stem}: {why}")
        if stopped_early:
            print(f"STOPPED EARLY after {len(results)} of {len(pending)} stem(s) — {stopped_early}",
                  file=sys.stderr)
            print("Remaining stems are still in the backlog; re-run to continue.", file=sys.stderr)
    return 0 if stopped_early is None else 2


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki claim-overlap",
        description="Judge claim relationships, verify citations, and apply or revert paper links.",
    )
    parser.add_argument("stem", nargs="?",
                        help="Stem of the newly-ingested paper (e.g. smith-2024-...).")
    parser.add_argument("--pair", nargs=2, metavar=("CLAIM_A", "CLAIM_B"),
                        help="Judge exactly these claim references (stem#claim_slug or "
                             "[[category/stem#claim_slug]]); bypasses threshold retrieval.")
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
    parser.add_argument("--revert", metavar="RUN_ID", help="Preview reverting this run's owned link insertions.")
    parser.add_argument("--stem", dest="revert_stem", help="With --revert: restrict to one processed stem.")
    parser.add_argument("--link", nargs=2, metavar=("STEM_A", "STEM_B"), help="With --revert: restrict to this paper pair.")
    parser.add_argument("--apply", action="store_true", help="Apply the previewed revert.")
    parser.add_argument("--progress-json", metavar="PATH", help="Append flushed per-stem JSON Lines progress.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.revert:
        from ..overlap.receipts import revert
        if args.stem or args.pair or args.backlog or args.mark_covered:
            parser.error("--revert does not combine with a positional stem or another review mode")
        if args.apply and args.dry_run:
            parser.error("--apply and --dry-run are mutually exclusive")
        result = revert(args.revert, stem=args.revert_stem, link=args.link, apply=args.apply)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for item in result["reverted"]:
                print(f"  {item['outcome']}: {item['path']}")
            print(f"{'Applied' if args.apply else 'Preview'}: {len(result['reverted'])} owned insertion(s)")
        return 0
    if args.apply or args.link or args.revert_stem:
        parser.error("--apply, --link and --stem require --revert")


    if args.pair:
        if args.stem or args.backlog or args.mark_covered:
            parser.error("--pair does not combine with a stem, --backlog, or --mark-covered")
        result = review_pair(*args.pair, dry_run=args.dry_run)
        _progress_result(args, result)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if result["applied"]:
            record = result["applied"][0]
            verb = "would link" if args.dry_run else "linked"
            print(f"  ✓ {verb} exact pair (cos {record['cosine']}) — {record['rationale']}")
        elif result["edge_only"]:
            record = result["edge_only"][0]
            print(f"  ~ exact pair recorded as {record['relation']} edge only "
                  f"(cos {record['cosine']}) — {record['rationale']}")
        elif result["coincidence"]:
            print("  · exact pair judged unrelated; no link written.")
        elif result["skipped"]:
            print("  – exact pair already linked; no judge call made.")
        elif result["judge_failed"]:
            print("  ⚠ exact-pair judge returned no valid verdict; no link written.")
        return 0

    if args.mark_covered:
        if args.stem or args.backlog:
            parser.error("--mark-covered takes no stem and does not combine with --backlog")
        if not args.dry_run:
            from ..overlap.receipts import recover
            recover()
        res = mark_covered(dry_run=args.dry_run)
        if args.json:
            print(json.dumps(dict(res, remaining_backlog=len(find_backlog()))))
            return 0
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

    result = run(args.stem, sim_threshold=args.sim, top_papers=args.top, dry_run=args.dry_run)
    _progress_result(args, result)

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


def _progress_result(args, result):
    from ..overlap.progress import emit
    if args.progress_json:
        emit(args.progress_json, run_id=result.get("run_id", "unknown"), stem=result.get("stem"),
             completed=0 if result.get("judge_failed") else 1, result=result)


def main(argv: list[str]) -> int:
    try:
        return _main(argv)
    except Exception as exc:
        known = (isinstance(exc, EnvironmentFailure) or type(exc) is LookupError
                 or (isinstance(exc, ValueError) and any(
                     value == flag or value.startswith(flag + "=")
                     for value in argv for flag in ("--pair", "--revert"))))
        if not known and "--json" not in argv:
            raise
        code = (2 if isinstance(exc, EnvironmentFailure) else 1) if known else 3
        result = getattr(exc, "claim_overlap_result", None) or {}
        from ..overlap.receipts import new_run_id
        from ..overlap.progress import emit
        for i, value in enumerate(argv):
            if value == "--progress-json" and i + 1 < len(argv):
                emit(argv[i + 1], run_id=result.get("run_id") or new_run_id(), result=result,
                     retryable=[result["stem"]] if result.get("stem") else [])
        if "--json" in argv:
            print(json.dumps(dict(result, error=f"{type(exc).__name__}: {exc}",
                                  exit_code=code, status="incomplete")))
        print(f"researchwiki claim-overlap: {exc}", file=sys.stderr)
        if not known:
            import traceback
            traceback.print_exc()
        return code
