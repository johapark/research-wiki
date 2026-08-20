"""`researchwiki candidates pairs` — claim pairs worth a look, ranked.

The third opportunity signal beside `candidates concepts` (un-scaffolded hub
terms) and `candidates synthesis` (un-covered clusters). Same contract as its
siblings: read-only, zero tokens, advisory, and the only state it writes is the
decline list.

It ranks cross-paper claim pairs that sit *below* `claim-overlap`'s auto-link
threshold — the layer that command deliberately never judges. See
`claim_discover` for why lowering that threshold is not the alternative.

Vocabulary is deliberately identical to `candidates concepts`:
`--decline`/`--reason`, `--undecline`, `--list-declined`. Nothing new to learn.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki candidates pairs",
        description="Cross-paper claim pairs below the auto-link threshold, "
                    "ranked by shared rare-term mass. Read-only, zero tokens.",
    )
    parser.add_argument("--cross-category", action="store_true",
                        help="Only pairs bridging two categories — the ones no "
                             "other structure in the wiki connects.")
    parser.add_argument("--limit", type=int, default=40,
                        help="Max pairs to report (default 40).")
    parser.add_argument("--json", action="store_true",
                        help="Emit the ranked pairs as JSON.")
    parser.add_argument("--decline", nargs=2, metavar=("STEM_A", "STEM_B"), default=None,
                        help="Suppress a pair permanently (needs --reason). "
                             "Order-independent.")
    parser.add_argument("--undecline", nargs=2, metavar=("STEM_A", "STEM_B"), default=None,
                        help="Restore a declined pair to the list.")
    parser.add_argument("--reason", default=None,
                        help="Required with --decline: why the pair is not a relation.")
    parser.add_argument("--list-declined", action="store_true",
                        help="Print the declined-pair list and exit.")
    args = parser.parse_args(argv)

    if args.list_declined or args.decline or args.undecline:
        return _manage(args, parser)
    return _report(args)


def _manage(args, parser) -> int:
    from .pair_dismissals import (
        add_dismissal, is_stale, load_dismissals, remove_dismissal,
    )

    if args.list_declined:
        entries = load_dismissals()
        if args.json:
            print(json.dumps({k: {**v, "stale": is_stale(v)}
                              for k, v in entries.items()},
                             ensure_ascii=False, indent=2))
            return 0
        if not entries:
            print("No declined pairs.")
            return 0
        n_stale = sum(1 for e in entries.values() if is_stale(e))
        suffix = f" ({n_stale} stale — evidence changed, back in the list)" if n_stale else ""
        print(f"{len(entries)} declined pair(s){suffix}:")
        for entry in sorted(entries.values(), key=lambda e: e.get("dismissed_at", "")):
            a, b = entry.get("stems", ["?", "?"])
            mark = "  [STALE]" if is_stale(entry) else ""
            print(f"  {a}\n  {b}{mark}")
            print(f"    {entry.get('dismissed_at', '?')} "
                  f"[{entry.get('source', 'manual')}] {entry.get('reason', '')}")
        return 0

    if args.undecline:
        a, b = args.undecline
        if remove_dismissal(a, b):
            print(f"restored: {a} ↔ {b}")
            return 0
        print(f"researchwiki candidates pairs: no decline recorded for {a} ↔ {b}",
              file=sys.stderr)
        return 1

    a, b = args.decline
    if not (args.reason or "").strip():
        parser.error("--decline requires --reason: record why the pair is not a "
                     "relation, or the list becomes unreviewable")
    if a.strip() == b.strip():
        parser.error("--decline needs two different stems")
    add_dismissal(a, b, args.reason.strip())
    print(f"declined: {a} ↔ {b}")
    return 0


def _report(args) -> int:
    from .claim_discover import DEFAULT_COS_HI, DEFAULT_COS_LO, discover_pairs

    pairs = discover_pairs(limit=args.limit,
                           cross_category_only=args.cross_category)

    if args.json:
        print(json.dumps({
            "cos_lo": DEFAULT_COS_LO, "cos_hi": DEFAULT_COS_HI,
            "cross_category_only": bool(args.cross_category),
            "pairs": [
                {
                    "stem_a": p.stem_a, "stem_b": p.stem_b,
                    "category_a": p.category_a, "category_b": p.category_b,
                    "cross_category": p.cross_category,
                    "citation_a": p.citation_a(), "citation_b": p.citation_b(),
                    "cosine": round(p.cosine, 3), "idf_mass": round(p.idf_mass, 1),
                    "shared_terms": p.shared_terms,
                    "text_a": p.text_a, "text_b": p.text_b,
                }
                for p in pairs
            ],
        }, ensure_ascii=False, indent=2))
        return 0

    if not pairs:
        print(f"No pair candidates (cosine {DEFAULT_COS_LO}–{DEFAULT_COS_HI}). "
              "A cold claim-embedding cache also reads as zero — warm it with "
              "any `claim-overlap` run.")
        return 0

    n_cross = sum(1 for p in pairs if p.cross_category)
    print(f"{len(pairs)} pair(s) in the cosine {DEFAULT_COS_LO}–{DEFAULT_COS_HI} band, "
          f"ranked by shared rare-term mass ({n_cross} cross-category). "
          f"Nothing judged, nothing written.")
    print()
    for i, p in enumerate(pairs, 1):
        tag = "  [cross-category]" if p.cross_category else ""
        print(f"  {i:>3}. idf={p.idf_mass:5.1f}  cos={p.cosine:.3f}{tag}")
        print(f"       {p.citation_a()}")
        print(f"          › {p.text_a[:104]}")
        print(f"       {p.citation_b()}")
        print(f"          › {p.text_b[:104]}")
        print(f"       shared: {', '.join(p.shared_terms[:7])}")
        print()
    print("Unjudged. To act on one, run the judged path on either stem:")
    print(f"    researchwiki claim-overlap {pairs[0].stem_a}")
    return 0
