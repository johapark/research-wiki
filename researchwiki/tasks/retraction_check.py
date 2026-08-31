"""Retraction-status lookup via PubMed E-utilities.

✅ Use when: auditing whether a wiki paper (or a candidate in
   `suggested-additions.md`) has been retracted or is itself a retraction
   notice. Safety-metadata only — does not fetch titles, authors, or
   abstracts. Rule-1 structured-API carve-out.
❌ Don't use: for title / author / year lookup (S2 and Crossref cover
   those). Don't use for retraction reasons / free-text retraction notes —
   those are prose and remain behind the Rule-1 prose ban.

Exit codes:
  0 — ran to completion (zero retractions is still 0)
  1 — user-input error (no DOI provided and --all not set)
  2 — environment error (PubMed unreachable after retries)
"""

from __future__ import annotations

import argparse
import json

from ..log import log
from ..providers.pubmed import retraction_status
from ..wiki import read_wiki_papers


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki retraction-check",
        description="Look up PubMed retraction status for a DOI or all wiki papers.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doi", help="Single DOI to check")
    group.add_argument("--all", dest="scan_all", action="store_true",
                       help="Scan every paper in the wiki and report retractions")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit structured JSON instead of prose output. "
                             "Schema: {doi, pmid, retracted, is_retraction_notice, "
                             "pubtypes, pubdate, source, fetched_at}.")
    args = parser.parse_args(argv)

    if args.doi:
        record = retraction_status(args.doi)
        if args.as_json:
            print(json.dumps(record, indent=2))
        else:
            _print_prose(record, stem=None)
        # A completed lookup is informational even when PubMed has no match.
        # Transport failures raise StructuredProviderUnavailable and the CLI's
        # shared failure funnel maps them to exit 2.
        return 0

    # --all path
    papers = read_wiki_papers()
    log(f"Scanning {len(papers)} papers for retraction status", tag="retract")
    results: list[dict] = []
    for p in papers:
        doi = p.get("doi")
        if not doi:
            continue
        record = retraction_status(doi)
        record["stem"] = p["stem"]
        record["category"] = p.get("category", "")
        results.append(record)

    flagged = [r for r in results if r["retracted"] or r["is_retraction_notice"]]

    if args.as_json:
        print(json.dumps({
            "scanned": len(results),
            "flagged": flagged,
            "unindexed_in_pubmed": [
                {"stem": r["stem"], "doi": r["doi"]}
                for r in results if r["pmid"] is None
            ],
        }, indent=2))
        return 0

    print(f"Scanned: {len(results)} wiki papers with DOIs")
    print(f"Retracted or retraction notice: {len(flagged)}")
    print()
    if flagged:
        for r in flagged:
            _print_prose(r, stem=r["stem"])
    else:
        print("_no retractions detected._")
    unindexed = [r for r in results if r["pmid"] is None]
    if unindexed:
        print()
        print(f"_{len(unindexed)} papers not indexed in PubMed (common for "
              f"preprints, CS/ML venues, some pre-2000 material) — absence of "
              f"`retracted: True` here is not evidence against retraction._")
    return 0


def _print_prose(record: dict, stem: str | None) -> None:
    label = f"{stem} · {record['doi']}" if stem else record["doi"]
    if record["pmid"] is None:
        print(f"- **{label}** — not indexed in PubMed (no signal)")
        return
    status = []
    if record["retracted"]:
        status.append("**RETRACTED**")
    if record["is_retraction_notice"]:
        status.append("**(is a retraction notice)**")
    if not status:
        status.append("ok")
    print(f"- **{label}** (PMID {record['pmid']}) — {' '.join(status)} "
          f"· pubtypes={record['pubtypes']} · pubdate={record['pubdate']} "
          f"· fetched {record['fetched_at']}")
