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
        # `.pair-dismissals.json` is hand-editable and `load_dismissals` only
        # validates the top level, so a malformed entry must not traceback the
        # management command — `dismissed_pairs` already skips these silently.
        # Report the count rather than hiding it: a dropped entry is a decision
        # that has stopped taking effect.
        raw = load_dismissals()
        entries = {k: e for k, e in raw.items()
                   if isinstance(e, dict)
                   and isinstance(e.get("stems"), list) and len(e["stems"]) == 2}
        malformed = len(raw) - len(entries)

        if args.json:
            print(json.dumps({k: {**e, "stale": is_stale(e)}
                              for k, e in entries.items()},
                             ensure_ascii=False, indent=2))
            return 0
        if not entries:
            print("No declined pairs."
                  + (f" ({malformed} malformed entr(y/ies) skipped)" if malformed else ""))
            return 0
        n_stale = sum(1 for e in entries.values() if is_stale(e))
        suffix = f" ({n_stale} stale — evidence changed, back in the list)" if n_stale else ""
        print(f"{len(entries)} declined pair(s){suffix}:")
        for entry in sorted(entries.values(), key=lambda e: e.get("dismissed_at", "")):
            a, b = entry["stems"]
            mark = "  [STALE]" if is_stale(entry) else ""
            print(f"  {a}\n  {b}{mark}")
            print(f"    {entry.get('dismissed_at', '?')} "
                  f"[{entry.get('source', 'manual')}] {entry.get('reason', '')}")
        if malformed:
            print(f"\n  {malformed} malformed entr(y/ies) in "
                  f".pair-dismissals.json were skipped and suppress nothing.")
        return 0

    if args.undecline:
        a, b = args.undecline
        if remove_dismissal(a, b):
            print(f"restored: {a} ↔ {b}")
            return 0
        # 0, matching `candidates concepts --undecline`: "it wasn't on the
        # list" is a no-op the caller asked for, not a failure.
        print(f"`{a} ↔ {b}` was not on the declined list.")
        return 0

    a, b = args.decline
    # Printed + `return 1` rather than `parser.error`, matching how
    # `candidates concepts --decline` reports the same omission.
    if not (args.reason or "").strip():
        print("researchwiki candidates pairs --decline: --reason is required "
              "(one sentence — why these two papers are not related).",
              file=sys.stderr)
        return 1
    if a.strip() == b.strip():
        print("researchwiki candidates pairs --decline: needs two different stems.",
              file=sys.stderr)
        return 1
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
                    "review_command": p.review_command(),
                    "cosine": round(p.cosine, 3), "idf_mass": round(p.idf_mass, 1),
                    "shared_terms": p.shared_terms,
                    "text_a": p.text_a, "text_b": p.text_b,
                }
                for p in pairs
            ],
        }, ensure_ascii=False, indent=2))
        return 0

    if not pairs:
        # Don't assert a cause. Zero here has three very different meanings —
        # no claims, a cold cache, or a genuinely quiet band — and the first two
        # are logged by `discover_pairs` itself as they happen.
        print(f"No pair candidates (cosine {DEFAULT_COS_LO}–{DEFAULT_COS_HI}).")
        print("  If a reason was logged above, fix that first; otherwise the "
              "band is genuinely empty.")
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
        print(f"       review: {p.review_command()}")
        print()
    print("Unjudged. Each review command judges exactly the displayed claim pair.")
    return 0
