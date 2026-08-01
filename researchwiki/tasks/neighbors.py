"""Find papers cited-by / citing / similar-to a wiki page (or any DOI).

Use this when you have one paper in hand and want to discover what's around
it in the citation graph — typically the "is X the most up-to-date in this
field?" or "what should I ingest next?" question.

Returns Semantic Scholar's structured fields only: title, year, DOI, venue,
citation count. No prose, no Rule 1 leak. Each hit is cross-referenced
against the wiki's known DOI set so [needs-ingest] / [in-wiki] is visible
at a glance.

Usage:
    researchwiki neighbors compbio/avsec-2025-alphagenome-...
    researchwiki neighbors 10.64898/2026.02.12.705572
    researchwiki neighbors <stem> --mode references --year 2024-2026
    researchwiki neighbors <stem> --needs-ingest --limit 20
    researchwiki neighbors <stem> --json
"""

from __future__ import annotations

import argparse
import json as _json
import sys

from ..log import log
from ..paths import wiki_dir
from ..providers import ScholarlyArticle, get_default_provider
from ..wiki import read_pages, read_wiki_dois


_DOI_PREFIX = "10."


def _looks_like_doi(s: str) -> bool:
    return s.startswith(_DOI_PREFIX) and "/" in s


def _resolve_doi(identifier: str) -> tuple[str | None, str | None]:
    """Map an input string to (doi, stem). Either may be None on failure."""
    if _looks_like_doi(identifier):
        return identifier.lower().strip(), None

    # Treat as a stem (`category/stem` or bare). Look up DOI from the wiki page.
    target_stem: str | None = None
    if "/" in identifier:
        cat, stem = identifier.split("/", 1)
        path = wiki_dir() / cat / f"{stem}.md"
        target_stem = stem if path.exists() else None
    else:
        target_stem = identifier
        # Walk wiki/ for a matching stem
        if not any(p for p in wiki_dir().rglob(f"{identifier}.md")):
            return None, None

    if target_stem is None:
        return None, None

    for p in read_pages():
        if p.stem != target_stem:
            continue
        doi_raw = (p.fm.get("doi") or "").strip().strip('"').strip("'").lower()
        if not doi_raw or doi_raw in ("todo", "none"):
            return None, target_stem
        return doi_raw, target_stem

    return None, target_stem


def _format_article(a: ScholarlyArticle, in_wiki_dois: set[str]) -> dict:
    doi = a.doi_lower or ""
    return {
        "doi": doi or None,
        "title": (a.title or "")[:120],
        "year": a.year,
        "venue": a.venue or "",
        "citation_count": a.citation_count,
        "in_wiki": bool(doi and doi in in_wiki_dois),
    }


def _filter_year(items: list[dict], year_range: tuple[int, int] | None) -> list[dict]:
    if not year_range:
        return items
    lo, hi = year_range
    return [it for it in items if it.get("year") and lo <= it["year"] <= hi]


def _parse_year_range(s: str) -> tuple[int, int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    y = int(s)
    return y, y


def _gather(provider, article: ScholarlyArticle, mode: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    wiki_dois = set(read_wiki_dois())
    if mode in ("references", "all"):
        refs = provider.get_references(article)
        out["references"] = [_format_article(a, wiki_dois) for a in refs]
    if mode in ("citations", "all"):
        cites = provider.get_citations(article)
        out["citations"] = [_format_article(a, wiki_dois) for a in cites]
    if mode in ("recommendations", "all"):
        recs = provider.get_recommendations(article)
        out["recommendations"] = [_format_article(a, wiki_dois) for a in recs]
    return out


def _sort_and_trim(items: list[dict], limit: int) -> list[dict]:
    return sorted(
        items,
        key=lambda x: (-(x.get("citation_count") or 0), -(x.get("year") or 0)),
    )[:limit]


_LABEL = {
    "references": "Papers cited by this paper",
    "citations": "Papers citing this paper",
    "recommendations": "S2 /recommendations (similar papers)",
}


def _render_section(label: str, items: list[dict]) -> None:
    print(f"## {label} ({len(items)})")
    if not items:
        print("_(none)_")
        print()
        return
    for it in items:
        tag = "[in-wiki]" if it["in_wiki"] else "[needs-ingest]"
        cites = it["citation_count"] if it["citation_count"] is not None else "?"
        year = it["year"] if it["year"] is not None else "?"
        title = it["title"] or "(no title)"
        doi = it["doi"] or "(no DOI)"
        print(f"- {tag:<14} cites={cites:<5} {year}  `{doi}`")
        print(f"    {title}")
    print()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki neighbors",
        description="List papers in the S2 citation graph around a wiki page or DOI.",
    )
    parser.add_argument("identifier",
                        help="DOI (e.g. 10.64898/2026.02.12.705572) or wiki "
                             "stem (e.g. compbio/avsec-2025-... or bare "
                             "stem). Stems must have a `doi:` field on the page.")
    parser.add_argument("--mode", choices=("references", "citations", "recommendations", "all"),
                        default="all", help="Which neighbor relation to fetch (default: all)")
    parser.add_argument("--year", default=None,
                        help="Year filter: 'YYYY' or 'YYYY-YYYY'")
    parser.add_argument("--limit", type=int, default=30,
                        help="Per-section row limit (default: 30, sorted by citation count)")
    parser.add_argument("--needs-ingest", action="store_true",
                        help="Only show neighbors not yet in the wiki "
                             "(filters out [in-wiki] entries).")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit a structured JSON object instead of prose")
    args = parser.parse_args(argv)

    doi, stem = _resolve_doi(args.identifier)
    if doi is None:
        if stem is None:
            log(f"could not resolve identifier: {args.identifier!r} — not a "
                f"DOI and no matching wiki stem found", tag="neighbors")
            return 1
        log(f"wiki page `{stem}` exists but has no `doi:` field; can't query S2. "
            f"Run `researchwiki lint --json | jq .missing_doi` to find similar pages.",
            tag="neighbors")
        return 1

    year_range = None
    if args.year:
        try:
            year_range = _parse_year_range(args.year)
        except ValueError:
            print(f"neighbors: invalid --year {args.year!r}; "
                  f"expected YYYY or YYYY-YYYY", file=sys.stderr)
            return 1

    provider = get_default_provider()
    article = ScholarlyArticle(doi=doi)
    sections = _gather(provider, article, args.mode)

    for label, items in list(sections.items()):
        items = _filter_year(items, year_range)
        if args.needs_ingest:
            items = [it for it in items if not it["in_wiki"]]
        sections[label] = _sort_and_trim(items, args.limit)

    if args.as_json:
        print(_json.dumps({"source_doi": doi, "source_stem": stem, **sections}, indent=2))
        return 0

    print(f"# Neighbors for `{doi}`" + (f"  ({stem})" if stem else ""))
    if year_range:
        print(f"_year filter: {year_range[0]}–{year_range[1]}_")
    if args.needs_ingest:
        print(f"_filtered to [needs-ingest] only_")
    print()
    for key in ("references", "citations", "recommendations"):
        if key in sections:
            _render_section(_LABEL[key], sections[key])

    n_needs = sum(
        sum(1 for it in items if not it["in_wiki"])
        for items in sections.values()
    )
    if n_needs and not args.needs_ingest:
        print(f"_{n_needs} item(s) flagged [needs-ingest]; drop their PDFs into `inbox/`._")
    return 0
