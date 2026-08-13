"""Search the pre-graded claims table for grounded citations.

✅ Use when: you make a factual statement and want a durable `[[stem#slug]]`
   anchor to cite, or you need the support scores (semantic + BM25) for a
   claim. Each hit is an atomic Key-Contribution / Results bullet anchored
   to a paper + section, with a graded score showing how well its source
   PDF backs it.
❌ Don't use: to find *pages* by topic — that's `researchwiki search`. Reach for
   `pdf-search` only when no claim covers the detail (claims are pre-vetted).

Usage:
  researchwiki claims "AlphaFold 3 protein-ligand accuracy"
  researchwiki claims "off-target rate" --k 8 --json
  researchwiki claims --by-stem smith-2024-paper-title-slug

Exit codes: 0 = hits returned; 1 = no claim contains any query token,
            or `--by-stem` resolved to a paper with zero claims.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..search import claim_lookup, claims_by_stem


def _fmt(hit: dict) -> str:
    # `[[stem#slug]]` is what the author types in a page (durable,
    # content-addressed). Falls back to bare `[[stem]]` for rows that
    # predate the slug migration.
    from ..search import format_claim_ref
    cite = format_claim_ref(hit)
    head = f"{cite}  ({hit['section']}#{hit['position']})"
    sem = hit.get("semantic_score")
    bm = hit.get("bm25_top1")
    parts = []
    if "match_score" in hit:
        parts.append(f"match={hit['match_score']}")
    if sem is not None:
        parts.append(f"semantic={sem:.2f}")
    if bm is not None:
        parts.append(f"bm25={bm:.2f}")
    if hit.get("graded") is False:
        parts.append("ungraded")
    scores = "      " + "  ".join(parts) if parts else "      (no scores)"
    text = (hit.get("text") or "").strip()
    out = f"{head}\n{scores}\n      › {text}"
    if "supporting_text" in hit and hit["supporting_text"]:
        # Indent the chunk under the claim so it's visually subordinate.
        chunk = hit["supporting_text"].replace("\n", " ").strip()
        # Where in the PDF that chunk sits — '§results, p. 7'. Absent on rows
        # graded before the chunk index carried provenance; they pick it up on
        # the next `grade`, since a stale-format index is rebuilt on read.
        where = (hit.get("supporting_provenance") or "").strip()
        label = f"source ({where})" if where else "source"
        out += f"\n      ⤷ {label}: {chunk}"
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki claims",
        description="Grounded-citation search over the pre-graded claims table.",
    )
    parser.add_argument("query", nargs="?", default=None,
                        help="Search terms (e.g. \"off-target rate\"). "
                             "Omit when using --by-stem.")
    parser.add_argument("--by-stem", default=None,
                        help="Dump every claim for this paper stem instead of a query search. "
                             "Useful when authoring a synthesis page that references the paper.")
    parser.add_argument("--k", type=int, default=5, help="Max claims to return (default: 5). Ignored with --by-stem.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit a JSON array instead of formatted prose.")
    parser.add_argument("--include-context", dest="include_context", action="store_true",
                        help="Surface each claim's supporting_text — the verbatim source-PDF chunk "
                             "the grader matched (≤500 chars). Off by default to keep dumps narrow.")
    args = parser.parse_args(argv)

    if args.by_stem and args.query:
        parser.error("Pass either a query or --by-stem, not both.")
    if not args.by_stem and not args.query:
        parser.error("Pass a query or --by-stem.")

    if args.by_stem:
        hits = claims_by_stem(args.by_stem, include_context=args.include_context)
    else:
        hits = claim_lookup(args.query, k=args.k, include_context=args.include_context)

    if args.as_json:
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return 0 if hits else 1

    if not hits:
        if args.by_stem:
            print(f"No claims for paper stem `{args.by_stem}` "
                  f"(unknown stem, or zero gradable claims).", file=sys.stderr)
        else:
            print("No matching claims.", file=sys.stderr)
        return 1
    for h in hits:
        print(_fmt(h))
        print()
    return 0
