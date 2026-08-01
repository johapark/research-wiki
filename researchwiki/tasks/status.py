"""One-screen health check of the local wiki state.

✅ Use when: you want a quick snapshot (paper counts, cross-link density,
   orphans, inbox backlog, recent additions). Safe to run at any time.
❌ Don't use: for structured issue lists (use `lint`). Not a search tool.

Fast (no network calls) — reads only local YAML frontmatter, wiki page text,
and `inbox/` / `.ingest/` filesystem state. For a full citation-graph audit
(which does make Semantic Scholar calls) run `researchwiki audit`.

Exit code: 0 always (reports state, never fails on content).
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime, timedelta

from pathlib import Path

from ..paths import (
    inbox_dir, ingest_dir, papers_dir, search_index_dir, semantic_cache_dir,
    wiki_dir, wiki_root,
)
from ..wiki import read_pages


# Model pricing — public Anthropic rates per million tokens. Used only for
# the cost-estimate line in `status`; values rot, but the user can mentally
# re-multiply if rates change. See https://www.anthropic.com/pricing.
#
# When you re-verify rates against Anthropic's current published pricing,
# update both _PRICING and _PRICING_AS_OF in the same edit so the label in
# the status output reflects the date of the last verification.
_PRICING_AS_OF = "2026-05"
_PRICING = {
    "claude-opus-4-7":    {"in": 15.00, "out": 75.00},
    "claude-sonnet-4-6":  {"in":  3.00, "out": 15.00},
    "claude-haiku-4-5":   {"in":  0.80, "out":  4.00},
}

# `other`-saturation warning lives in researchwiki.categories — same helper
# is called from status (here) and from both ingest paths (digest + agent)
# so the threshold + decay constants stay in one place.

# Matches [[target]], [[target|display]], [[target#anchor]], [[target#anchor|display]].
WIKILINK_RE = re.compile(r"\[\[([^\]\|#]+?)(?:#[^\]\|]*)?(?:\|[^\]]+)?\]\]")


def _extract_wikilinks(text: str) -> list[str]:
    return [m.group(1).strip() for m in WIKILINK_RE.finditer(text)]


def _format_relative(t: datetime) -> str:
    now = datetime.now()
    delta = now - t
    if delta < timedelta(minutes=1):
        return "just now"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() / 60)}m ago"
    if delta < timedelta(hours=24):
        return f"{int(delta.total_seconds() / 3600)}h ago"
    if delta < timedelta(days=30):
        return f"{delta.days}d ago"
    return t.strftime("%Y-%m-%d")


def _pending_pdfs(directory):
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pdf"))


def _pending_digests(directory):
    if not directory.exists():
        return []
    return sorted(directory.glob("*.md"))


def _index_status() -> tuple[dict, dict]:
    """Snapshot of Tantivy + semantic page indexes for the status output.

    Each dict carries: exists, age_seconds (since last build), and index-specific
    stats (n_pages / dim / model). Missing indexes get exists=False and the
    rest of the keys absent — caller renders a "run reindex" hint.
    """
    import json
    import time

    tantivy: dict = {"exists": False}
    tdir = search_index_dir()
    marker = tdir / ".managed.json"
    if marker.exists():
        tantivy["exists"] = True
        tantivy["age_seconds"] = time.time() - marker.stat().st_mtime

    semantic: dict = {"exists": False}
    sdir = semantic_cache_dir()
    meta = sdir / "pages_meta.json"
    if meta.exists():
        try:
            obj = json.loads(meta.read_text())
            semantic["exists"] = True
            semantic["n_pages"] = len(obj.get("rows", []))
            semantic["dim"] = obj.get("dim", 0)
            semantic["model"] = obj.get("model", "?")
            semantic["age_seconds"] = time.time() - meta.stat().st_mtime
        except (OSError, ValueError):
            pass
    return tantivy, semantic


def _evolution_proposal_dirs() -> list[tuple[Path, int, float]]:
    """List .ingest/*-evolution-proposals/ directories.

    Each entry: (path, n_proposal_files, age_seconds). Sorted by age
    descending — the oldest unactioned proposals surface first.
    """
    import time
    base = ingest_dir()
    if not base.exists():
        return []
    out: list[tuple[Path, int, float]] = []
    for d in base.glob("*-evolution-proposals"):
        if not d.is_dir():
            continue
        n = sum(1 for f in d.glob("*.md"))
        age = time.time() - d.stat().st_mtime
        out.append((d, n, age))
    out.sort(key=lambda x: -x[2])
    return out


def _claim_grading_coverage() -> dict:
    """Aggregate the `claims` table into a one-line coverage summary.

    Returns {n_total, n_ungraded, n_papers_with_ungraded, mean_top1, semantic_score}
    or {} when the DB is unreachable / empty. Cross-ref claims (skipped by
    the grader) are excluded from the totals so the percentage doesn't get
    pulled down by claims that are NULL by design.
    """
    from ..db.safe import safe_read

    # Keep the closure narrow to raw fetches. Post-processing (int/float
    # coercion, empty-DB short-circuit) sits outside so real bugs there
    # surface loudly rather than getting silently converted to `{}` by
    # safe_read's `except Exception`.
    def _fetch(conn):
        row = conn.execute(
            """
            SELECT COUNT(*)                                       AS n_total,
                   SUM(CASE WHEN last_graded_at IS NULL THEN 1 ELSE 0 END) AS n_ungraded,
                   AVG(bm25_top1)                                 AS mean_top1,
                   AVG(semantic_score)                            AS semantic_score
              FROM claims
             WHERE is_cross_ref = 0
            """
        ).fetchone()
        ungraded_pages = conn.execute(
            """
            SELECT COUNT(DISTINCT paper_stem) AS n_papers
              FROM claims
             WHERE is_cross_ref = 0 AND last_graded_at IS NULL
            """
        ).fetchone()
        return row, ungraded_pages

    fetched = safe_read(_fetch, default=None, label="status.claim_grading")
    if fetched is None:
        return {}
    row, ungraded_pages = fetched
    if not row or row["n_total"] == 0:
        return {}
    return {
        "n_total": int(row["n_total"] or 0),
        "n_ungraded": int(row["n_ungraded"] or 0),
        "n_papers_with_ungraded": int((ungraded_pages or {"n_papers": 0})["n_papers"] or 0),
        "mean_top1": float(row["mean_top1"]) if row["mean_top1"] is not None else 0.0,
        "semantic_score": float(row["semantic_score"]) if row["semantic_score"] is not None else None,
    }


def _recent_ingest_costs(days: int = 7) -> dict:
    """Aggregate ingest_iterations rows over the last N days into per-paper cost.

    Returns a dict with: n_attempts, total_input, total_output, by_model,
    estimated_usd. Empty when the DB is missing or the window has no rows.
    """
    import time
    from ..db.safe import safe_read

    cutoff = int(time.time()) - days * 86400

    def _query(conn):
        return conn.execute(
            "SELECT attempt_id, model_used, "
            "       COALESCE(cost_input_tokens, 0) AS in_tok, "
            "       COALESCE(cost_output_tokens, 0) AS out_tok "
            "FROM ingest_iterations "
            "WHERE created_at >= ? AND model_used IS NOT NULL "
            "      AND model_used <> 'stub' AND model_used <> '(skipped)'",
            (cutoff,),
        ).fetchall()

    rows = safe_read(_query, default=[], label="status.recent_ingest_costs")
    if not rows:
        return {}

    by_model: dict[str, dict] = {}
    attempts: set = set()
    usd = 0.0
    for r in rows:
        attempt_id, model, in_tok, out_tok = r
        attempts.add(attempt_id)
        slot = by_model.setdefault(model, {"in": 0, "out": 0})
        slot["in"] += in_tok
        slot["out"] += out_tok
        rates = _PRICING.get(model)
        if rates:
            usd += (in_tok / 1_000_000) * rates["in"] + (out_tok / 1_000_000) * rates["out"]

    total_in = sum(s["in"] for s in by_model.values())
    total_out = sum(s["out"] for s in by_model.values())
    return {
        "days": days,
        "n_attempts": len(attempts),
        "total_input": total_in,
        "total_output": total_out,
        "by_model": by_model,
        "estimated_usd": usd,
    }


def _failed_parsing_entries(root) -> list[str]:
    from ..paths import pdfs_failed_parsing_path
    path = pdfs_failed_parsing_path()
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        # Lines shaped like: - **{stem}**
        m = re.match(r"\s*-\s+\*\*([a-z0-9-]+)\*\*\s*$", line)
        if m:
            entries.append(m.group(1))
    return entries


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki status",
        description="One-screen wiki health check (local-only, no network).",
    )
    parser.add_argument("--recent", type=int, default=5,
                        help="Number of recent additions to show (default: 5)")
    args = parser.parse_args(argv)

    root = wiki_root()
    wdir = wiki_dir()
    # Scan pages directly rather than `read_wiki_papers()`, which silently
    # drops any page missing a `doi:` field from its result — that erased
    # doi-less papers (e.g. an arXiv survey never assigned one) from the
    # cross-link graph entirely, so every edge pointing at or from them
    # vanished too (`lint`'s `missing_doi` still flags these separately).
    # Filter on `type: paper` instead of just excluding page-type dirs, so
    # housekeeping pages (e.g. the Dataview dashboard at `wiki/views.md`,
    # category "wiki") can't leak in as a "paper".
    PAGE_TYPE_DIRS = ("synthesis", "references", "ideas", "concepts")
    papers = [
        {"stem": p.stem, "category": p.category}
        for p in read_pages()
        if p.fm.get("type") == "paper" and p.category not in PAGE_TYPE_DIRS
    ]

    # --- edges from [[wikilink]] occurrences in wiki pages
    known_pages = {f"{p['category']}/{p['stem']}" for p in papers}
    page_type_sets: dict[str, set[str]] = {}
    for pt in PAGE_TYPE_DIRS:
        ptdir = wdir / pt
        if ptdir.exists():
            page_type_sets[pt] = {f"{pt}/{m.stem}" for m in ptdir.glob("*.md")}
        else:
            page_type_sets[pt] = set()

    # Every page type can be a [[wikilink]] target: papers + all page-type dirs.
    all_link_targets = known_pages | set().union(*page_type_sets.values())

    def _scan_links(page_path: Path, src_key: str) -> set[str]:
        text = page_path.read_text()
        targets: set[str] = set()
        for link in _extract_wikilinks(text):
            if link == src_key:
                continue
            if "/" in link:
                if link in all_link_targets:
                    targets.add(link)
            else:
                # Bare stem — resolve across categories
                for kp in known_pages:
                    if kp.split("/", 1)[1] == link and kp != src_key:
                        targets.add(kp)
                        break
        return targets

    edges_by_src: dict[str, set[str]] = {}
    edges_by_tgt: dict[str, set[str]] = {}

    for p in papers:
        page = wdir / p["category"] / f"{p['stem']}.md"
        if not page.exists():
            continue
        src_key = f"{p['category']}/{p['stem']}"
        targets = _scan_links(page, src_key)
        if targets:
            edges_by_src[src_key] = targets
        for t in targets:
            edges_by_tgt.setdefault(t, set()).add(src_key)

    # Synthesis/idea/concept/reference pages cite papers via [[wikilink]]s
    # and footnote defs, but were never scanned as edge *sources* — so a
    # paper cited only by, say, a dedicated idea page's footnotes (e.g.
    # zandieh-2025-turboquant) showed up as a false orphan. Scan them too.
    for keys in page_type_sets.values():
        for src_key in keys:
            page = wdir / f"{src_key}.md"
            if not page.exists():
                continue
            targets = _scan_links(page, src_key)
            if targets:
                edges_by_src[src_key] = targets
            for t in targets:
                edges_by_tgt.setdefault(t, set()).add(src_key)

    # Density/directional stats stay paper-to-paper only — a citation from a
    # synthesis/idea page isn't a paper-to-paper cross-link, so it shouldn't
    # inflate a metric sized against `n choose 2` paper pairs. Orphan
    # detection below uses the full edge sets so those citations still count.
    directional = sum(
        1 for src, tgts in edges_by_src.items() if src in known_pages
        for tgt in tgts if tgt in known_pages
    )
    undirected = {
        tuple(sorted([src, tgt]))
        for src, tgts in edges_by_src.items() if src in known_pages
        for tgt in tgts if tgt in known_pages
    }
    undirected_count = len(undirected)

    n = len(papers)
    max_possible = n * (n - 1) // 2 if n >= 2 else 0
    density = (undirected_count / max_possible) if max_possible else 0.0

    orphans = sorted(
        f"{p['category']}/{p['stem']}"
        for p in papers
        if f"{p['category']}/{p['stem']}" not in edges_by_src
        and f"{p['category']}/{p['stem']}" not in edges_by_tgt
    )

    cat_counts = Counter(p["category"] for p in papers)

    inbox_files = _pending_pdfs(inbox_dir())
    ingest_files = _pending_digests(ingest_dir())
    failed = _failed_parsing_entries(root)

    all_pages = []
    for p in papers:
        page = wdir / p["category"] / f"{p['stem']}.md"
        if page.exists():
            all_pages.append((page, f"{p['category']}/{p['stem']}"))
    for keys in page_type_sets.values():
        for key in keys:
            page = wdir / f"{key}.md"
            if page.exists():
                all_pages.append((page, key))
    all_pages.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    recent = all_pages[: args.recent]

    # --- render
    print("Research Wiki — status")
    print()

    page_type_labels = {
        "synthesis": "synthesis",
        "references": "reference",
        "ideas": "idea",
        "concepts": "concept",
    }
    total_pages = n + sum(len(s) for s in page_type_sets.values())
    parts = [f"{n} paper"]
    for pt in PAGE_TYPE_DIRS:
        count = len(page_type_sets[pt])
        if count:
            parts.append(f"{count} {page_type_labels[pt]}")
    print(f"Pages: {total_pages}  ({', '.join(parts)})")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:<12} {count}")
    for pt in PAGE_TYPE_DIRS:
        count = len(page_type_sets[pt])
        if count:
            print(f"  {pt:<12} {count}")
    print()

    # `other` saturation warning. Centralised helper — touches the stamp
    # when surfaced so the next status / ingest suppresses for the decay
    # window. See researchwiki.categories.other_saturation_warning for the
    # threshold + decay tunables.
    from ..categories import other_saturation_warning
    msg = other_saturation_warning()
    if msg:
        print(msg)
        print()

    # Within-category divergence nudge. Structural-only (no LLM) and
    # decay-stamped like the `other` warning — fires when a populated category
    # has grown a sub-cluster distinct enough to consider splitting out. See
    # researchwiki.tasks.suggest_splits.divergence_warning.
    from .suggest_splits import divergence_warning
    div_msg = divergence_warning()
    if div_msg:
        print(div_msg)
        print()

    print("Cross-link graph (from `[[wikilinks]]` in wiki pages):")
    print(f"  directional links:   {directional}")
    print(f"  unique paper pairs:  {undirected_count}")
    print(f"  density:             {density:.1%}  ({undirected_count} / {max_possible} max)")
    if orphans:
        print(f"  orphans (0 in/out):  {len(orphans)}")
        for o in orphans[:10]:
            print(f"    - {o}")
        if len(orphans) > 10:
            print(f"    ... ({len(orphans) - 10} more)")
    else:
        print("  orphans:             none — every paper is linked")
    print()

    print("Workflow state:")
    print(f"  inbox/ PDFs awaiting ingest:    {len(inbox_files)}")
    for f in inbox_files[:5]:
        print(f"    - {f.name}")
    if len(inbox_files) > 5:
        print(f"    ... ({len(inbox_files) - 5} more)")
    print(f"  .ingest/ digests awaiting page: {len(ingest_files)}")
    for f in ingest_files[:5]:
        print(f"    - {f.name}")
    if len(ingest_files) > 5:
        print(f"    ... ({len(ingest_files) - 5} more)")
    print(f"  PDFs failed parsing:            {len(failed)}")
    for s in failed[:5]:
        print(f"    - {s}")
    print()

    print(f"Recent additions (top {args.recent} by file mtime):")
    for path, key in recent:
        rel = _format_relative(datetime.fromtimestamp(path.stat().st_mtime))
        print(f"  {key:<65} {rel}")
    print()

    # --- active model config (which file drives ingest LLM routing)
    try:
        import os as _os
        from ..agents.model_config import config_path
        cfg = config_path()
        rel = cfg.name if cfg.parent == (root / "config") else str(cfg)
        marker = "" if cfg.exists() else "  (missing — using hardcoded defaults)"
        origin = " [RW_MODELS_CONFIG]" if _os.environ.get("RW_MODELS_CONFIG") else ""
        print(f"Model config:          {rel}{origin}{marker}")
        print()
    except Exception:
        pass

    # --- index health
    tantivy, semantic = _index_status()
    print("Index health:")
    if tantivy.get("exists"):
        age = _format_duration(tantivy["age_seconds"])
        print(f"  Tantivy BM25:        built {age} ago")
    else:
        print(f"  Tantivy BM25:        not built — run `researchwiki reindex`")
    if semantic.get("exists"):
        age = _format_duration(semantic["age_seconds"])
        print(f"  Semantic page idx:   built {age} ago "
              f"({semantic.get('n_pages',0)} pages, dim={semantic.get('dim','?')})")
    else:
        print(f"  Semantic page idx:   not built — run `researchwiki reindex`")

    # --- structured DB drift (cheap; runs `db verify` over current wiki/)
    try:
        from ..db.verify import verify as _db_verify
        report = _db_verify()
        drift = len(report.missing) + len(report.extra) + len(report.stale) + len(report.moved)
        if drift == 0:
            print(f"  Structured DB:       in sync ({report.papers_in_db} rows)")
        else:
            parts = []
            if report.missing: parts.append(f"{len(report.missing)} missing")
            if report.extra:   parts.append(f"{len(report.extra)} extra")
            if report.stale:   parts.append(f"{len(report.stale)} stale")
            if report.moved:   parts.append(f"{len(report.moved)} moved")
            print(f"  Structured DB:       drift — {', '.join(parts)} "
                  f"(run `researchwiki lint --fix` or `db rebuild`)")
    except Exception as e:
        print(f"  Structured DB:       check failed ({e})")

    # --- claim grading coverage (read-side surfaces depend on it)
    coverage = _claim_grading_coverage()
    if coverage:
        if coverage["n_ungraded"] == 0 and coverage["n_total"] > 0:
            line = (
                f"{coverage['n_total']} claims graded "
                f"(mean BM25={coverage['mean_top1']:.2f}"
            )
            if coverage.get("semantic_score") is not None:
                line += f", semantic={coverage['semantic_score']:.2f}"
            line += ")"
            print(f"  Claim grading:       {line}")
        elif coverage["n_total"] > 0:
            graded = coverage["n_total"] - coverage["n_ungraded"]
            line = (
                f"{graded}/{coverage['n_total']} graded "
                f"({coverage['n_papers_with_ungraded']} papers need backfill — "
                f"run `researchwiki grade regression`)"
            )
            print(f"  Claim grading:       {line}")
    print()

    # --- pending evolution proposals
    proposals = _evolution_proposal_dirs()
    if proposals:
        n_total = sum(n for _, n, _ in proposals)
        print(f"Pending evolution proposals: {len(proposals)} dir(s), "
              f"{n_total} total file(s)")
        for path, nf, age in proposals[:5]:
            rel = _format_duration(age)
            stem = path.name.replace("-evolution-proposals", "")
            print(f"  {stem:<55} {nf} file(s), {rel} old")
        if len(proposals) > 5:
            print(f"  ... ({len(proposals) - 5} more)")
        print(f"  Review and apply, then `rm -rf` the directory.")
        print()

    # --- supplementary files (S1 of supplementary-files.md)
    pdir = papers_dir()
    if pdir.exists():
        supp_dirs = [d for d in pdir.glob("*.supp") if d.is_dir()]
        if supp_dirs:
            n_files = sum(sum(1 for f in d.iterdir() if f.is_file()) for d in supp_dirs)
            total_bytes = sum(
                f.stat().st_size for d in supp_dirs for f in d.iterdir() if f.is_file()
            )
            mb = total_bytes / (1024 * 1024)
            print(f"Supplementary files: {n_files} file(s) across {len(supp_dirs)} paper(s), {mb:,.1f} MB")
            print()

    # --- ingest cost telemetry
    costs = _recent_ingest_costs(days=7)
    if costs:
        n_a = costs["n_attempts"]
        in_k = costs["total_input"] / 1000
        out_k = costs["total_output"] / 1000
        per_paper = (costs["total_input"] + costs["total_output"]) / max(n_a, 1) / 1000
        print(f"Ingest cost (last {costs['days']} days):")
        print(f"  attempts:           {n_a}")
        print(f"  total tokens:       {in_k:,.0f}K input + {out_k:,.0f}K output")
        print(f"  mean per attempt:   {per_paper:,.0f}K tokens")
        print(f"  estimated cost:     ${costs['estimated_usd']:.2f}  (Anthropic {_PRICING_AS_OF} rates)")
        for model, slot in sorted(costs["by_model"].items()):
            print(f"    {model:<22} {slot['in']/1000:>7,.0f}K in, "
                  f"{slot['out']/1000:>7,.0f}K out")
        print()

    # --- concept-hub candidates (opportunity signal; not a defect)
    # No outer try/except here: `n_bridge_candidates` already swallows its own
    # failures and reports them as None, so wrapping it again would only hide a
    # bug in these three prints.
    from ..concepts import TRIAGE_THRESHOLD, n_bridge_candidates
    n_bridges = n_bridge_candidates()
    if n_bridges is None:
        print("Concept-hub candidates: scan failed — count unknown "
              "(`researchwiki candidates concepts --bridges` for the error)")
        print()
    elif n_bridges >= TRIAGE_THRESHOLD:
        print(f"Concept-hub candidates: {n_bridges} bridge term(s) — likely dominated by "
              "extraction noise. Batch-triage with "
              "`researchwiki candidates concepts --triage` (--dry-run to preview).")
        print()
    elif n_bridges > 0:
        print(f"Concept-hub candidates: {n_bridges} bridge term(s) with no hub yet "
              "(`researchwiki candidates concepts --bridges`)")
        print()

    print("For the full citation-graph report run: researchwiki audit")
    return 0


def _format_duration(seconds: float) -> str:
    """Compact human-readable duration. `_format_relative` was for absolute times;
    this works on raw second counts (e.g. index age, proposal-dir age)."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h"
    return f"{int(seconds / 86400)}d"
