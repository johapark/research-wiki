"""CLI wrapper for the concept-hub scaffolder + refresh + upgrade-spokes utilities.

The detection, scaffolding, and refresh logic all live in
`researchwiki.concepts` (split across `candidates.py`, `term_claims.py`,
`scaffold.py`, `refresh.py`). This file is the thin argparse layer that:

  - dispatches `concepts refresh <slug>` to `refresh_concept`
  - dispatches `concepts --upgrade-spokes` to `upgrade_spokes`
  - dispatches `concepts <term>` (with `--thesis`, `--aliases`, …) to `run`

For un-scaffolded candidates, see `researchwiki candidates concepts` —
the candidate lister is under `tasks/candidates.py`.

Usage:
  researchwiki concepts "RAG"                       # scaffold hub for a term
  researchwiki concepts "Virtual Cell" --title "Virtual cell modeling"
  researchwiki concepts "RAG" --dry-run             # show members + span
  researchwiki concepts "RAG" --json                # scaffold decisions JSON
  researchwiki concepts refresh <slug>              # draft Cross-domain connections from edges
  researchwiki concepts --upgrade-spokes            # backfill [[stem#slug]] on existing hubs

Exit codes: 0 = success (stub written or dry-run);
1 = user-input error (fewer than --min-members, empty slug, or target
exists without --force).
"""

from __future__ import annotations

import argparse
import json
import sys

from ..concepts import refresh_concept, run, upgrade_spokes


def _run_refresh(argv: list[str]) -> int:
    """`concepts refresh <slug>` — draft Cross-domain connections for a hub."""
    parser = argparse.ArgumentParser(
        prog="researchwiki concepts refresh",
        description="Draft a `## Cross-domain connections` block for the given "
                    "concept hub from typed edges among its member claims.",
    )
    parser.add_argument("slug", help="Concept-hub slug (e.g. `prime-editing`).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute the draft but don't write the proposal file.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = refresh_concept(args.slug, dry_run=args.dry_run)
    except ValueError as e:
        print(f"researchwiki concepts refresh: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if result["n_member_claims"] == 0:
        print(f"concepts refresh: no member claims found for hub `{args.slug}` "
              f"(any `instantiates` edges pointing at it?).")
        return 0
    if result["n_bridges_found"] == 0:
        print(f"concepts refresh: {result['n_member_claims']} member claim(s) "
              f"but no cross-category typed edges among them.")
        print(f"Run `researchwiki claim-overlap <stem>` on member papers to "
              f"seed typed edges (corroborates / refines / builds_on / measures_same).")
        return 0
    verb = "would draft" if args.dry_run else "drafted"
    print(f"concepts refresh: {verb} Cross-domain connections block for `{args.slug}` "
          f"({result['n_bridges_found']} bridge(s) across "
          f"{result['n_member_claims']} member claim(s)).")
    if result["draft_path"]:
        print(f"  → {result['draft_path']}")
        print("Review the draft, then copy the block into the hub's "
              "`## Cross-domain connections` section (or delete the draft to reject).")
    return 0


def _resolve_thesis(term: str, from_arg: str | None) -> str | None:
    """Return the concept thesis for this scaffold, or None if unavailable.

    Precedence: explicit `--thesis` beats the interactive prompt. Prompt only
    fires when stdin is a TTY (so `researchwiki concepts foo --json` piped
    in a script fails fast instead of hanging). Empty input → None → caller
    refuses.
    """
    if from_arg is not None:
        return from_arg.strip() or None
    if not sys.stdin.isatty():
        return None
    prompt = (
        f"\nScaffolding `{term}`. In one sentence, why is this a *concept* "
        "(an idea the corpus\ndisagrees about or elaborates on) rather than a "
        "*glossary* term (vocabulary\npapers use consistently) or a *synthesis* "
        "topic (a comparison of approaches)?\n"
        "Answer, or Ctrl-C to abort:\n> "
    )
    try:
        answer = input(prompt)
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return None
    return answer.strip() or None


def main(argv: list[str]) -> int:
    # Subcommand: `concepts refresh <slug>` peels off first — it's a distinct
    # mode over an existing hub rather than a term-scaffold operation.
    if argv and argv[0] == "refresh":
        return _run_refresh(argv[1:])

    parser = argparse.ArgumentParser(
        prog="researchwiki concepts",
        description="Scaffold a concept-hub note from a recurring term. For un-scaffolded "
                    "candidates, use `researchwiki candidates concepts`.",
    )
    parser.add_argument("term", nargs="?", default=None,
                        help='Concept term as it appears in `candidates concepts` output '
                             '(e.g. "RAG", "Virtual Cell"). Required unless --upgrade-spokes.')
    parser.add_argument("--upgrade-spokes", action="store_true",
                        help="Backfill: rewrite bare `[[stem]]` spokes in every wiki/concepts/*.md to "
                             "`[[stem#claim_slug]]` where a matching contribution claim exists. "
                             "Idempotent; already-slug-cited spokes are untouched. Nothing else runs "
                             "in this mode (term arg is ignored).")
    parser.add_argument("--title", default=None, help="Human-readable page title (default: the term).")
    parser.add_argument("--slug", default=None, help="Filename stem (default: slugified term).")
    parser.add_argument("--min-members", type=int, default=3,
                        help="Minimum member papers required to create the page (default 3).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing concept page.")
    parser.add_argument("--dry-run", action="store_true", help="Show members + span, write nothing.")
    parser.add_argument("--json", action="store_true", help="Emit scaffold decisions as JSON.")
    parser.add_argument("--thesis", default=None,
                        help="One-sentence answer to *why is this a concept, not a glossary term "
                             "or a synthesis topic?* — required. Stored as `concept_thesis:` in "
                             "the hub's YAML and rendered as a blockquote under the H1. If "
                             "omitted and stdin is a TTY, the user is prompted; if omitted in "
                             "non-interactive mode, the scaffold refuses.")
    parser.add_argument("--aliases", default=None,
                        help="Comma-separated vocabulary variants that also identify this "
                             "concept in the corpus (e.g. `--aliases \"DMS,saturation "
                             "mutagenesis,MAVE\"`). find_members expands the term-search across "
                             "all of them. Stored as `topic_seed_aliases:` in the hub YAML so "
                             "downstream hooks (attach, refresh) see the same alias set.")
    args = parser.parse_args(argv)

    if args.upgrade_spokes:
        stats = upgrade_spokes(dry_run=args.dry_run)
        if args.json:
            # per_hub keys aren't strings-only-compat by design (they're stem
            # names), but json.dumps handles them fine.
            print(json.dumps(stats, ensure_ascii=False, indent=2))
            return 0
        n_hubs = stats["hubs_scanned"]
        upd = stats["hubs_updated"]
        upgraded = stats["spokes_upgraded"]
        skipped = stats["spokes_skipped_no_claim"]
        verb = "would upgrade" if args.dry_run else "upgraded"
        print(f"upgrade-spokes: scanned {n_hubs} hub(s), {verb} {upgraded} spoke(s) "
              f"across {upd} hub(s). Skipped {skipped} spoke(s) with no matching claim.")
        if stats["per_hub"]:
            print()
            for stem, s in sorted(stats["per_hub"].items()):
                print(f"  {stem:<40} +{s['upgraded']} upgraded, {s['skipped']} skipped")
        return 0

    if not args.term:
        parser.print_usage(sys.stderr)
        print("researchwiki concepts: term is required unless --upgrade-spokes "
              "is passed. For un-scaffolded candidates, use "
              "`researchwiki candidates concepts`.", file=sys.stderr)
        return 1

    thesis = _resolve_thesis(args.term, args.thesis)
    if thesis is None:
        # Empty answer or non-interactive without `--thesis`. `run()` would
        # raise the same error, but we surface it here with the actionable
        # remedy before doing any DB work.
        print(
            f"researchwiki concepts: refusing to scaffold `{args.term}` without a "
            "`concept_thesis`. Pass `--thesis \"<one sentence>\"`, or run "
            "interactively and answer the prompt. See docs/concept-vs-glossary.md "
            "for the discipline this enforces.",
            file=sys.stderr,
        )
        return 1

    aliases_list = (
        [a.strip() for a in args.aliases.split(",") if a.strip()]
        if args.aliases else []
    )

    try:
        result = run(
            args.term, thesis=thesis, aliases=aliases_list,
            title=args.title, slug=args.slug,
            min_members=args.min_members, force=args.force, dry_run=args.dry_run,
        )
    except ValueError as e:
        print(f"researchwiki concepts: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    span = result["span"]
    n = len(result["members"])
    if args.dry_run:
        print(f"'{args.term}': {n} member(s) across {span} categor{'y' if span == 1 else 'ies'} "
              f"→ would write wiki/concepts/{result['slug']}.md")
        for k in result["members"]:
            print(f"  · [[{k}]]")
        return 0

    print(f"wrote {result['path']}  ({n} members, span {span}, "
          f"{len(result['linked'])} reciprocal link(s) added)")
    print("Next: fill Definition + spoke one-liners, then "
          f"`researchwiki check-grounding {result['path']}` + `grade synthesis`.")
    return 0
