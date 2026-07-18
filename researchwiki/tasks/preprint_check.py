"""Preprint ↔ journal-version pairing via bioRxiv / medRxiv details API.

✅ Use when: checking whether a bioRxiv-DOI paper in the wiki has since
   been published in a journal (stem-collision / version-tracking case
   already documented in CLAUDE.md's "Preprint → journal version" rule),
   or auditing the whole wiki for preprints whose journal version is now
   detectable. Rule-1 structured-API carve-out.
❌ Don't use: for title / author lookup (S2 and Crossref cover those).
   Don't use for abstracts — bioRxiv does expose `abstract` via the same
   endpoint but this tool does not re-export it (the prose ban still
   applies).

Two use modes:
  --doi X          Check a single DOI.
  --all            Scan every wiki paper whose DOI looks like a bioRxiv
                   / medRxiv preprint (10.1101/*, 10.64898/*) and report
                   any with a detected journal version.

Exit codes:
  0 — ran to completion (zero flagged is still 0)
  1 — user-input error (neither --doi nor --all)
  2 — environment error (bioRxiv unreachable after retries)
"""

from __future__ import annotations

import argparse
import json

from ..log import log
from ..providers.biorxiv import lookup
from ..wiki import read_wiki_papers

# DOI prefixes bioRxiv uses. New prefix (`10.64898/`) was added in 2026;
# both are served by the biorxiv API. Update as prefixes evolve.
PREPRINT_DOI_PREFIXES = ("10.1101/", "10.64898/", "10.31219/", "10.20944/")


def _is_preprint_doi(doi: str) -> bool:
    d = (doi or "").lower()
    return any(d.startswith(p) for p in PREPRINT_DOI_PREFIXES)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki preprint-check",
        description="Check whether a preprint has a detected journal version, or "
                    "scan every preprint-DOI in the wiki for journal updates.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doi", help="Single DOI to check")
    group.add_argument("--all", dest="scan_all", action="store_true",
                       help="Scan every wiki paper whose DOI is a known preprint prefix")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit structured JSON instead of prose output. Record "
                             "schema: {doi, server, title, version, date_posted, "
                             "category, type, published_doi, source, fetched_at}.")
    args = parser.parse_args(argv)

    if args.doi:
        record = lookup(args.doi)
        if args.as_json:
            print(json.dumps(record, indent=2))
        else:
            _print_prose(record, stem=None, in_wiki_dois=set())
        return 0

    # --all path
    papers = read_wiki_papers()
    wiki_dois = {(p.get("doi") or "").lower() for p in papers}
    preprint_papers = [p for p in papers if _is_preprint_doi(p.get("doi", ""))]
    log(f"Scanning {len(preprint_papers)} preprint-DOI papers (of {len(papers)} total)",
        tag="preprint")

    results: list[dict] = []
    for p in preprint_papers:
        record = lookup(p["doi"])
        record["stem"] = p["stem"]
        record["category"] = record.get("category") or p.get("category", "")
        # `published_in_wiki` is the actionable signal: if this preprint's
        # journal version already has its own wiki page, the preprint page
        # is a candidate for YAML updates (per CLAUDE.md's preprint→journal
        # rule: keep the stem, update title/doi/venue in YAML).
        record["published_in_wiki"] = (
            record["published_doi"] is not None
            and record["published_doi"] in wiki_dois
        )
        results.append(record)

    flagged = [r for r in results if r["published_doi"] is not None]

    if args.as_json:
        print(json.dumps({
            "scanned": len(results),
            "flagged": flagged,
            "unknown_on_biorxiv": [
                {"stem": r["stem"], "doi": r["doi"]}
                for r in results if r["server"] is None
            ],
        }, indent=2))
        return 0

    print(f"Scanned: {len(results)} preprint-DOI wiki papers")
    print(f"With detected journal version: {len(flagged)}")
    print()
    if flagged:
        for r in flagged:
            _print_prose(r, stem=r["stem"], in_wiki_dois=wiki_dois)
    else:
        print("_no preprints in the wiki have a bioRxiv-detected journal version yet._")
    unknown = [r for r in results if r["server"] is None]
    if unknown:
        print()
        print(f"_{len(unknown)} papers with preprint-like DOIs not found on bioRxiv/medRxiv_ "
              f"(could be new-prefix preprints, custom DOIs, or indexing lag)")
    return 0


def _print_prose(record: dict, stem: str | None, in_wiki_dois: set[str]) -> None:
    label = f"{stem} · {record['doi']}" if stem else record["doi"]
    if record["server"] is None:
        print(f"- **{label}** — not found on bioRxiv / medRxiv")
        return
    pub = record["published_doi"]
    if pub is None:
        print(f"- **{label}** (v{record['version']}, {record['date_posted']}, "
              f"{record['server']}/{record['category']}) — no journal version detected")
        return
    in_wiki_note = ""
    if pub in in_wiki_dois:
        in_wiki_note = " ⚠ **journal DOI already in wiki** (possible duplicate page — see CLAUDE.md's preprint→journal rule)"
    else:
        in_wiki_note = " (journal DOI not yet in wiki — candidate YAML update: update `doi:` + `title:` + venue, keep the stem)"
    print(f"- **{label}** (v{record['version']}, {record['date_posted']}) "
          f"→ published as `{pub}`{in_wiki_note}")
