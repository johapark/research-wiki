"""Structured citation scouting across all current wiki papers.

✅ Use when: looking for cross-link gaps after batch ingests, refreshing
   `wiki/synthesis/suggested-additions.md`, or deciding which paper to
   ingest next. Agents: pass `--json` and feed the output into the
   suggested-additions update.
❌ Don't use: for a quick local health check (use `status`). Don't run on
   every session — S2 has rate limits; results are cached under
   `.s2-cache/` so re-runs are cheap, but fresh data costs quota.

For each paper in wiki/, fetches refs + citations + recommendations via the
default provider, then reports:

  * Cross-wiki citation edges (who cites whom among current wiki papers)
  * Top recommended DOIs not yet in the wiki (by multi-paper overlap)
  * Most-cited references across the wiki (shared citation anchors)

Output is Markdown to stdout by default. Pass `--json` for structured output.

Exit codes: 0 = ran to completion (even if 0 edges / 0 recommendations);
2 = provider unreachable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date

from ..categories import PAGE_TYPE_DIRS
from ..fsatomic import write_json_atomic
from ..log import log
from ..paths import s2_cache_dir
from ..providers import ScholarlyArticle, get_default_provider
from ..wiki import read_pages, read_wiki_papers


def _doi_key(value: object) -> str | None:
    """Return one canonical DOI key for graph joins and counters."""
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value or None


def _load_paper_inventory(log_tag: str) -> tuple[list[dict], int, list[str], int]:
    """Load scoutable papers and account for every intentional DOI omission."""
    papers = read_wiki_papers()
    # Surface the operational gap: paper-type pages without `doi:` are
    # silently filtered out of `read_wiki_papers()` and so can't be scouted.
    total_paper_pages = 0
    intentional_no_doi: list[str] = []
    for page in read_pages():
        if page.fm.get("type", "paper") != "paper":
            continue
        if page.category in PAGE_TYPE_DIRS:
            continue
        total_paper_pages += 1
        # Only effective no-DOI pages are intentional omissions. A stale reason
        # on a page that now has a DOI must not distort the denominator.
        if not page.fm.get("doi") and (page.fm.get("no_doi_reason") or "").strip():
            intentional_no_doi.append(page.stem)

    skipped_no_doi = total_paper_pages - len(papers) - len(intentional_no_doi)
    eligible = total_paper_pages - len(intentional_no_doi)
    if skipped_no_doi > 0:
        log(f"WARN: {skipped_no_doi} paper page(s) skipped (no DOI in YAML); "
            f"scout denominator is {len(papers)}/{eligible}. "
            f"Run `researchwiki lint --json | jq .missing_doi` to fix.",
            tag=log_tag)
    if intentional_no_doi:
        log(f"Note: {len(intentional_no_doi)} page(s) excluded from scout by "
            f"`no_doi_reason:` field (legitimate no-DOI cases).",
            tag=log_tag)
    log(f"Wiki papers (scoutable): {len(papers)}", tag=log_tag)
    return papers, total_paper_pages, intentional_no_doi, skipped_no_doi


def _build_doi_inventory(papers: list[dict], log_tag: str) -> dict:
    """Build unambiguous DOI joins and expose duplicate assignments."""
    doi_to_stems: dict[str, list[str]] = {}
    for paper in papers:
        doi = _doi_key(paper.get("doi"))
        if doi:
            doi_to_stems.setdefault(doi, []).append(paper["stem"])
    duplicate_dois = [
        {"doi": doi, "stems": sorted(stems)}
        for doi, stems in sorted(doi_to_stems.items())
        if len(stems) > 1
    ]
    if duplicate_dois:
        log(
            f"WARN: {len(duplicate_dois)} DOI(s) occur on multiple wiki pages; "
            "ambiguous DOI edges are skipped",
            tag=log_tag,
        )
    ambiguous_stems = {
        stem
        for entry in duplicate_dois
        for stem in entry["stems"]
    }
    return {
        "wiki_dois": set(doi_to_stems),
        "doi_to_stem": {
            doi: stems[0]
            for doi, stems in doi_to_stems.items()
            if len(stems) == 1
        },
        "duplicate_dois": duplicate_dois,
        "ambiguous_stems": ambiguous_stems,
        "stem_to_category": {p["stem"]: p["category"] for p in papers},
        "aggregate_papers": len(papers) - len(ambiguous_stems),
    }


def _fetch_bundles(papers: list[dict], provider, log_tag: str) -> tuple[dict, list[dict]]:
    """Fetch metadata and graph neighborhoods for every scoutable paper."""
    dois_to_batch = [p["doi"].strip() for p in papers if _doi_key(p.get("doi"))]
    batch_metadata: dict[str, ScholarlyArticle] = {}
    if dois_to_batch:
        log(f"batch-fetching metadata for {len(dois_to_batch)} papers", tag=log_tag)
        batch_lookup = getattr(provider, "get_batch_metadata", None)
        if callable(batch_lookup):
            raw_batch = batch_lookup(dois_to_batch) or {}
            if isinstance(raw_batch, dict):
                batch_metadata = {
                    key: article
                    for raw_key, article in raw_batch.items()
                    if (key := _doi_key(raw_key))
                }

    bundles: dict[str, dict] = {}
    s2_missing: list[dict] = []
    for paper in papers:
        stem = paper["stem"]
        log(f"Paper {stem}...", tag=log_tag)
        article = None
        if paper.get("doi"):
            article = batch_metadata.get(_doi_key(paper["doi"]) or "")
            if article is None:
                article = provider.get_by_doi(paper["doi"])
        if article is None:
            article = ScholarlyArticle()
            if paper.get("doi"):
                s2_missing.append({"stem": stem, "doi": paper["doi"]})
        bundles[stem] = {
            "article": article,
            "refs": provider.get_references(article) if article.doi else [],
            "cites": provider.get_citations(article) if article.doi else [],
            "recs": provider.get_recommendations(article) if article.doi else [],
        }
    return bundles, s2_missing


def main(
    argv: list[str],
    *,
    prog: str = "researchwiki scout citations",
    log_tag: str = "scout",
    report_title: str = "# Semantic Scholar Citation Scout",
) -> int:
    """Run the citation scout under a caller-selected CLI/log identity.

    The deprecated ``researchwiki audit`` wrapper supplies its legacy identity;
    both paths intentionally share the same stdout and cache contract.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Scout the wiki's citation graph through Semantic Scholar.",
    )
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit a structured JSON object instead of the Markdown report. "
                             "Keys: papers_skipped_no_doi, total_paper_pages, papers (now "
                             "with category), cross_wiki_citations, edge_summary, "
                             "recommended_additions, shared_citation_anchors (each entry now "
                             "also carries categories, category_breadth, count_normalized), "
                             "anchor_groups (multi_category / single_category / high_count "
                             "— a structural index into the top-40 anchors, not editorial "
                             "tiers). Snapshot cached to .s2-cache/audit-{date}.json.")
    parser.add_argument(
        "--refresh-cache", dest="refresh_cache", type=int, nargs="?", const=0, default=None,
        metavar="DAYS",
        help="Bypass cached S2 responses older than DAYS days for this run. "
             "Bare `--refresh-cache` busts everything (DAYS=0). Without this "
             "flag, positive caches are honored indefinitely and negative "
             "caches honor a 30-day TTL. Use to retry papers that were "
             "previously 404 in S2 but may now be indexed.",
    )
    args = parser.parse_args(argv)

    papers, total_paper_pages, intentional_no_doi, skipped_no_doi = (
        _load_paper_inventory(log_tag)
    )
    doi_inventory = _build_doi_inventory(papers, log_tag)
    wiki_dois = doi_inventory["wiki_dois"]
    doi_to_stem = doi_inventory["doi_to_stem"]
    duplicate_dois = doi_inventory["duplicate_dois"]
    ambiguous_stems = doi_inventory["ambiguous_stems"]
    stem_to_category = doi_inventory["stem_to_category"]
    aggregate_papers = doi_inventory["aggregate_papers"]
    provider = get_default_provider(
        log_tag=log_tag, force_refresh_days=args.refresh_cache,
    )
    bundles, s2_missing = _fetch_bundles(papers, provider, log_tag)

    # Cross-wiki edges (outgoing + incoming views; union)
    edges: set[tuple[str, str]] = set()
    for src_stem, b in bundles.items():
        if src_stem in ambiguous_stems:
            continue
        for r in b["refs"]:
            tgt_doi = _doi_key(r.doi)
            if tgt_doi and tgt_doi in doi_to_stem:
                tgt_stem = doi_to_stem[tgt_doi]
                if tgt_stem != src_stem:
                    edges.add((src_stem, tgt_stem))
    for tgt_stem, b in bundles.items():
        if tgt_stem in ambiguous_stems:
            continue
        for c in b["cites"]:
            src_doi = _doi_key(c.doi)
            if src_doi and src_doi in doi_to_stem:
                src_stem = doi_to_stem[src_doi]
                if src_stem != tgt_stem:
                    edges.add((src_stem, tgt_stem))

    # Aggregate recommendations (DOIs we don't have yet)
    rec_counter: dict[str, dict] = {}
    for stem, b in bundles.items():
        if stem in ambiguous_stems:
            continue
        seen: set[str] = set()
        for r in b["recs"]:
            d = _doi_key(r.doi)
            if not d or d in wiki_dois or d in seen:
                continue
            seen.add(d)
            entry = rec_counter.setdefault(d, {
                "count": 0, "title": r.title, "year": r.year,
                "sources": set(), "categories": set(),
            })
            entry["count"] += 1
            entry["sources"].add(stem)
            entry["categories"].add(stem_to_category.get(stem, "other"))

    # Aggregate shared-citation anchors (refs cited by 2+ wiki papers)
    ref_counter: dict[str, dict] = {}
    for stem, b in bundles.items():
        if stem in ambiguous_stems:
            continue
        seen: set[str] = set()
        for r in b["refs"]:
            d = _doi_key(r.doi)
            if not d or d in wiki_dois or d in seen:
                continue
            seen.add(d)
            entry = ref_counter.setdefault(d, {
                "count": 0, "title": r.title, "year": r.year,
                "sources": set(), "categories": set(),
            })
            entry["count"] += 1
            entry["sources"].add(stem)
            entry["categories"].add(stem_to_category.get(stem, "other"))

    ranked_recs = sorted(rec_counter.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
    ranked_refs = sorted(ref_counter.items(), key=lambda kv: (-kv[1]["count"], kv[0]))

    # --- Structural derived fields (emitted on each entry). The LLM weights
    # these; no policy/scoring baked in here.
    def _entry(doi: str, info: dict, *, sources_key: str) -> dict:
        cats = sorted(info["categories"])
        return {
            "doi": doi,
            "title": info["title"],
            "year": info["year"],
            "multi_paper_count": info["count"],
            "count_normalized": round(info["count"] / max(aggregate_papers, 1), 4),
            "categories": cats,
            "category_breadth": len(cats),
            sources_key: sorted(info["sources"]),
        }

    # --- Build structural anchor groups. Labels are descriptive (what the
    # data *is*), not editorial (what it *means*). Same anchor can appear in
    # multiple groups — this is an additive index over the same top-40 the
    # `shared_citation_anchors` field reports, so the two views stay
    # consistent.
    anchor_entries = [_entry(doi, info, sources_key="cited_by")
                      for doi, info in ranked_refs if info["count"] >= 2]
    top_anchors = anchor_entries[:40]

    # `high_count` = top quintile by count (p80) within the top-40. Threshold
    # is derived from the data each run; no hard-coded number.
    if top_anchors:
        counts = sorted((e["multi_paper_count"] for e in top_anchors), reverse=True)
        p80_idx = max(1, len(counts) // 5)
        high_count_threshold = counts[p80_idx - 1]
    else:
        high_count_threshold = 0

    anchor_groups = {
        "multi_category": [e for e in top_anchors if e["category_breadth"] >= 2],
        "single_category": [e for e in top_anchors if e["category_breadth"] == 1],
        "high_count": [e for e in top_anchors
                       if e["multi_paper_count"] >= high_count_threshold and high_count_threshold >= 2],
    }

    # --- Edge summary: compact view for when the raw edge list gets long.
    in_deg: Counter[str] = Counter()
    out_deg: Counter[str] = Counter()
    flow: Counter[tuple[str, str]] = Counter()
    for src, tgt in edges:
        out_deg[src] += 1
        in_deg[tgt] += 1
        flow[(stem_to_category.get(src, "other"), stem_to_category.get(tgt, "other"))] += 1

    edge_summary = {
        "total": len(edges),
        "top_hubs_incoming": [{"stem": s, "in_degree": n} for s, n in in_deg.most_common(5)],
        "top_hubs_outgoing": [{"stem": s, "out_degree": n} for s, n in out_deg.most_common(5)],
        "by_category_flow": [
            {"src_category": src, "tgt_category": tgt, "count": n}
            for (src, tgt), n in sorted(flow.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
    }

    if args.as_json:
        out = {
            "papers_skipped_no_doi": skipped_no_doi,
            "papers_intentional_no_doi": intentional_no_doi,
            "total_paper_pages": total_paper_pages,
            "papers": [
                {
                    "stem": p["stem"],
                    "category": p["category"],
                    "year": p.get("year", ""),
                    "doi": p.get("doi", ""),
                    "reference_count": bundles[p["stem"]]["article"].reference_count,
                    "citation_count": bundles[p["stem"]]["article"].citation_count,
                }
                for p in papers
            ],
            "cross_wiki_citations": [
                {"src": src, "tgt": tgt} for src, tgt in sorted(edges)
            ],
            "edge_summary": edge_summary,
            "recommended_additions": [
                _entry(doi, info, sources_key="recommended_for")
                for doi, info in ranked_recs[:30]
            ],
            "shared_citation_anchors": top_anchors,
            "anchor_groups": anchor_groups,
            "s2_missing": s2_missing,
            "duplicate_dois": duplicate_dois,
        }
        print(json.dumps(out, indent=2))

        # Cache the snapshot so `researchwiki lint` can cross-reference it
        # for the `p2_entries_with_anchor_hits` check. Gitignored via
        # `.s2-cache/` entry.
        try:
            cache_dir = s2_cache_dir()
            cache_dir.mkdir(exist_ok=True)
            snapshot = cache_dir / f"audit-{date.today().isoformat()}.json"
            write_json_atomic(snapshot, out)
            log(f"snapshot cached: {snapshot.name}", tag=log_tag)
        except OSError as e:
            log(f"snapshot write failed: {e}", tag=log_tag)

        return 0

    # Prose report (default)
    print(report_title)
    print()
    print(f"Wiki papers: {len(papers)}")
    for p in papers:
        stem = p["stem"]
        article = bundles[stem]["article"]
        nref = article.reference_count if article.reference_count is not None else "?"
        nci = article.citation_count if article.citation_count is not None else "?"
        print(f"- {stem} ({p.get('year', '')}) — refs {nref}, cited by {nci}")
    print()

    print("## Cross-wiki citations (source-supported cross-links)")
    print()
    if edges:
        by_target: dict[str, list[str]] = {}
        for src, tgt in sorted(edges):
            by_target.setdefault(tgt, []).append(src)
        for tgt, srcs in sorted(by_target.items()):
            print(f"### Cited by: {tgt}")
            for src in srcs:
                print(f"- **{src}** → cites → **{tgt}**")
            print()
    else:
        print("(none)")
    print()

    print("## Top recommended DOIs not yet in wiki (by multi-paper overlap)")
    print()
    for doi, info in ranked_recs[:30]:
        srcs = ", ".join(sorted(info["sources"]))
        print(f"- [{info['count']}×] **{info['title']}** ({info['year']}) `{doi}` — recommended for: {srcs}")
    print()

    print("## Most-cited references across the wiki (shared citation anchors)")
    print()
    for doi, info in ranked_refs[:40]:
        if info["count"] < 2:
            break
        srcs = ", ".join(sorted(info["sources"]))
        print(f"- [{info['count']}×] **{info['title']}** ({info['year']}) `{doi}` — cited by: {srcs}")
    print()

    if s2_missing:
        print("## Papers whose DOI is not in Semantic Scholar")
        print()
        print(f"{len(s2_missing)} paper(s). These 404 in S2 — typically a wrongly-formatted")
        print("DOI (e.g. `10.48550/arXiv.X` instead of the journal DOI) or a paper S2")
        print("hasn't indexed yet. Negative-cached for 30 days; pass `--refresh-cache`")
        print("to force a retry sooner.")
        print()
        for entry in s2_missing:
            print(f"- {entry['stem']} `{entry['doi']}`")
        print()

    if duplicate_dois:
        print("## Duplicate DOI assignments (ambiguous graph targets)")
        print()
        for entry in duplicate_dois:
            stems = ", ".join(entry["stems"])
            print(f"- `{entry['doi']}` — {stems}")
        print()

    return 0
