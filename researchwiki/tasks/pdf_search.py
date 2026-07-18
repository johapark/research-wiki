"""BM25 search inside one paper's PDF for a passage matching a query.

✅ Use when: the wiki page summary lacks a specific detail — an exact number, a
   method parameter, a paragraph the page didn't quote — and you want the raw
   passage from the source PDF (Rule 3: re-read the PDF when the wiki is thin).
❌ Don't use: to find which paper to read — that's `researchwiki search`. Prefer
   `researchwiki claims` first; its hits are pre-graded.

The chunk index is built lazily on first call and cached under
`.grade-cache/{stem}/`, so the first query on a paper is slower.

Usage:
  researchwiki pdf-search smith-2024-paper-title-slug "training data composition"
  researchwiki pdf-search <stem> "off-target rate" --k 5 --json

Exit codes: 0 = passages returned; 1 = zero matches; 2 = no PDF / index error.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..search import pdf_section_search


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki pdf-search",
        description="BM25 search inside one paper's PDF chunks.",
    )
    parser.add_argument("stem", help="Paper stem (basename of papers/{stem}.pdf).")
    parser.add_argument("query", help="Search terms (e.g. \"training data composition\").")
    parser.add_argument("--k", type=int, default=3, help="Max passages to return (default: 3).")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit a JSON array instead of formatted prose.")
    args = parser.parse_args(argv)

    hits = pdf_section_search(args.stem, args.query, k=args.k)

    # The primitive signals failure as a single-element list carrying a `note`.
    if len(hits) == 1 and "note" in hits[0] and "chunk_id" not in hits[0]:
        if args.as_json:
            print(json.dumps(hits, indent=2, ensure_ascii=False))
        else:
            print(hits[0]["note"], file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return 0 if hits else 1

    if not hits:
        print("No matching passages.", file=sys.stderr)
        return 1
    for h in hits:
        text = (h.get("text") or "").strip()
        print(f"chunk:{h['chunk_id']}  score={h['score']:.2f}")
        print(f"  › {text}")
        print()
    return 0
