"""Surface opportunity signals — un-scaffolded concept hubs, un-covered synthesis
clusters, and unreviewed claim pairs.

✅ Use when: `status` prints a nonzero `Concept-hub candidates: N bridge term(s)`
   line (concepts target), OR you want to ask "what synthesis pages are we
   missing?" after a batch of ingests (synthesis target), OR `status` prints a
   nonzero `Claim-pair discovery: N unreviewed cross-category pair(s)` line
   (pairs target).
❌ Don't use: as a substitute for human curation. All three are advisory.

Three targets:

  researchwiki candidates concepts   [--bridges] [--persist-edges] [--json]
                                      [--decline TERM --reason TEXT]
                                      [--undecline TERM] [--list-declined]
                                      [--triage [--dry-run]]
  researchwiki candidates synthesis  [--min-cluster N] [--threshold F] [--judge]
                                      [--write-proposals] [--json]
  researchwiki candidates pairs      [--cross-category] [--limit N] [--json]
                                      [--decline A B --reason TEXT]
                                      [--undecline A B] [--list-declined]

**concepts** — recurring vocabulary terms mentioned by ≥3 wiki papers with no
`wiki/concepts/{slug}.md` yet. Cheap (local, sub-second), no LLM. Bridge tier
(--bridges: span ≥ 2 categories) is the highest-leverage — those are the terms
the citation graph and semantic-KNN don't naturally connect. `status`
auto-surfaces the bridge count, so treat that line as the trigger.

Detection is stateless — it re-derives candidates from scratch every call, so
a term that fails the concept-vs-glossary thesis test (see
prompts/concept-page-author.md) resurfaces every time. `--decline TERM --reason
TEXT` records a permanent suppression in `.concept-declines.json`
(`--undecline` reverses it, `--list-declined` shows the current list) so a
rejected candidate stops appearing here and in `status`'s bridge count.

`--triage` automates that judgment when noise accumulates: one batch LLM call
classifies every candidate (concept/glossary/fragment/redundant/uncertain)
against the same thesis test and auto-declines the noise verdicts (tagged
`source=llm-triage`, reversible). `--dry-run` previews without writing. Genuine
`concept`/`uncertain` verdicts stay surfaced — triage never scaffolds. `status`
recommends this once the bridge count crosses `TRIAGE_THRESHOLD`.

**synthesis** — dense paper clusters (wikilinks + semantic cosine ≥ 0.65 +
keyword Jaccard ≥ 0.2, Louvain communities) not covered by any existing
synthesis page. Higher noise rate than concepts, so **not** auto-surfaced by
`status` — the run cadence is user-initiated: after ≥5 ingests since the last
run, or when a cross-paper question lands in unfamiliar territory. The default
is a local, non-mutating structural preview; `--judge` opts into configured-model
scope verdicts and `--write-proposals` persists review artifacts.

**pairs** — cross-paper claim pairs sitting *below* `claim-overlap`'s auto-link
threshold, ranked by shared rare-term mass inside a cosine band. Local,
sub-second, no LLM. Lowering `claim-overlap --sim` is not the alternative: at
0.70, 80% of all possible paper pairs qualify. `--cross-category` narrows to the
pair candidates that no other structure in the wiki connects. Each row prints an exact
`researchwiki claim-overlap --pair A#slug B#slug` command, which judges those
two claims without re-running the >=0.83 stem-wide path.

Exit code: concepts/pairs return 0 for advisory results; synthesis returns 2
when its semantic index is absent. Concepts can update `.concept-declines.json`
and pairs can update `.pair-dismissals.json`; synthesis writes `.ingest/`
proposal artifacts only with `--write-proposals`.
"""

from __future__ import annotations

import argparse
import sys


def _run_concepts(argv: list[str]) -> int:
    """Dispatch to the concepts-candidate collector without going through
    the top-level `concepts` command (which now only scaffolds / refreshes).
    Reuses the same `collect_candidates` implementation."""
    import json

    from ..concepts import (add_decline, apply_triage, collect_candidates,
                            load_declines, remove_decline, triage_candidates)

    parser = argparse.ArgumentParser(
        prog="researchwiki candidates concepts",
        description="List un-scaffolded concept-hub candidates.",
    )
    parser.add_argument("--bridges", action="store_true",
                        help="Restrict to bridge-tier candidates (span ≥ 2 categories).")
    parser.add_argument("--persist-edges", action="store_true",
                        help="Side-write `instantiates` edges (claim → concept-term-slug) into "
                             ".claim-graph/edges.db. Off by default; enable to seed the graph.")
    parser.add_argument("--json", action="store_true",
                        help="Emit as JSON.")
    parser.add_argument("--decline", metavar="TERM",
                        help="Permanently suppress TERM from every future listing (this command "
                             "and `status`'s bridge count) — for a candidate that failed the "
                             "concept-vs-glossary thesis test (prompts/concept-page-author.md) and "
                             "would otherwise keep resurfacing, since detection here is stateless. "
                             "Requires --reason.")
    parser.add_argument("--reason", metavar="TEXT",
                        help="One-sentence reason recorded with --decline: why TERM is glossary/"
                             "redundant rather than a concept. Same discipline as the scaffold-time "
                             "thesis prompt — forces the judgment to be written out.")
    parser.add_argument("--undecline", metavar="TERM",
                        help="Remove TERM from the suppression list so it can resurface.")
    parser.add_argument("--list-declined", action="store_true",
                        help="Print the current suppression list instead of the candidate list.")
    parser.add_argument("--triage", action="store_true",
                        help="Batch-LLM classify all candidates (concept/glossary/fragment/"
                             "redundant/uncertain) against the concept-vs-glossary thesis test "
                             "and auto-decline the noise verdicts (tagged source=llm-triage, "
                             "reversible via --undecline). Use --dry-run to preview without writing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --triage, print the verdicts and write nothing.")
    args = parser.parse_args(argv)

    if args.decline:
        if not args.reason or not args.reason.strip():
            print("researchwiki candidates concepts --decline: --reason is required "
                  "(one sentence — why this is glossary/redundant, not a concept).",
                  file=sys.stderr)
            return 1
        slug = add_decline(args.decline, args.reason.strip())
        print(f"Declined `{args.decline}` (slug: {slug}). It will no longer appear "
              f"in `candidates concepts` or `status`'s bridge count.")
        return 0

    if args.undecline:
        removed = remove_decline(args.undecline)
        if removed:
            print(f"Removed `{args.undecline}` from the suppression list.")
        else:
            print(f"`{args.undecline}` was not on the suppression list.")
        return 0

    if args.list_declined:
        declines = load_declines()
        if not declines:
            print("_no declined concept terms._")
            return 0
        print(f"Declined concept terms ({len(declines)}):")
        print()
        for slug, entry in sorted(declines.items()):
            src = entry.get("source", "manual")
            print(f"- `{entry['term']}` (slug: {slug}, declined {entry['declined_at']}, via {src})")
            print(f"  {entry['reason']}")
        return 0

    if args.triage:
        # Honor --bridges: `status` recommends --triage off the *bridge* count,
        # so `--bridges --triage` must scope the run to that tier rather than
        # silently auto-declining across every tier.
        results = triage_candidates(collect_candidates(bridges_only=args.bridges))
        if not results:
            print("_no concept candidates to triage._")
            return 0
        summary = apply_triage(results, dry_run=args.dry_run)
        counts, declined, kept = summary["counts"], summary["declined"], summary["kept"]
        # Nothing actionable and everything kept as uncertain → the LLM path
        # degraded (provider down, prompt missing, unparsable) — say so plainly.
        if not declined and all(k["verdict"] == "uncertain" for k in kept):
            print("concept triage unavailable or produced no actionable verdicts — no changes.")
            print("(LLM provider unreachable, prompt missing, or every term kept as uncertain.)")
            return 0
        print(f"Concept triage — {summary['total']} candidate(s): "
              + ", ".join(f"{n} {v}" for v, n in sorted(counts.items())))
        print()
        if declined:
            verb = "Would decline" if summary["dry_run"] else "Declined"
            print(f"{verb} ({len(declined)}) as noise:")
            for d in declined:
                print(f"- `{d['term']}` [{d['verdict']}] — {d['reason']}")
            print()
        concepts_kept = [k for k in kept if k["verdict"] == "concept"]
        uncertain_kept = [k for k in kept if k["verdict"] == "uncertain"]
        if concepts_kept:
            print(f"Surfaced as genuine concepts ({len(concepts_kept)}) — scaffold with "
                  '`researchwiki concepts "<term>" --thesis "..."`:')
            for k in concepts_kept:
                print(f"- `{k['term']}` — {k['reason']}")
            print()
        if uncertain_kept:
            print(f"Kept as uncertain ({len(uncertain_kept)}) — left surfaced for review.")
            print()
        if summary["dry_run"]:
            print("_Dry run — nothing written. Re-run without --dry-run to apply the declines._")
        else:
            print("_Declines tagged `source: llm-triage`; reverse any with "
                  '`researchwiki candidates concepts --undecline "<term>"`._')
        return 0

    cands = collect_candidates(bridges_only=args.bridges, persist_edges=args.persist_edges)[:30]
    if args.json:
        print(json.dumps(cands, ensure_ascii=False, indent=2))
        return 0
    if not cands:
        print("_no concept candidates — every ≥3-page term already has a page, or the corpus is too small._")
        return 0

    header = "Concept candidates (≥3 paper mentions, not yet a page)"
    if args.bridges:
        header += " — bridge tier (span ≥ 2)"
    print(header)
    print()
    for c in cands:
        pages = c["pages"]
        cats = c.get("categories") or 1
        cat_str = f", {cats} categories" if cats > 1 else ""
        label = c.get("label")
        label_str = f"  ← {label}" if label else ""
        print(f"- `{c['term']}` — {pages} paper pages{cat_str}{label_str}")
    print()
    print('_Triage manually. `researchwiki concepts "<term>"` scaffolds a hub_')
    print('_(bridges — span ≥ 2 — first); `researchwiki synthesize` for a cross-paper argument._')
    print('_`glossary-suspect` = bare acronym or corpus-ubiquitous term — demoted, not a bridge;_')
    print('_scaffold one only if you can write a genuine concept-thesis for it (see prompts/concept-page-author.md)._')
    print("_Pass --persist-edges to write `instantiates` edges into `.claim-graph/edges.db`._")
    print('_Fails the thesis test? `--decline "<term>" --reason "..."` suppresses it for good._')
    print('_Too many to triage by hand? `--triage` batch-classifies all candidates and auto-declines the noise (`--dry-run` to preview)._')
    return 0


def _run_synthesis(argv: list[str]) -> int:
    """Delegate to the hidden `_synthesis_candidates` module. Its `main` still
    parses its own flags so this dispatcher stays thin."""
    from . import _synthesis_candidates
    return _synthesis_candidates.main(argv)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        # Bespoke help so we can surface the two targets clearly without
        # forcing users through argparse's less-informative default output.
        print(__doc__.strip())
        return 0 if argv else 2

    target, rest = argv[0], argv[1:]
    if target == "concepts":
        return _run_concepts(rest)
    if target == "synthesis":
        return _run_synthesis(rest)
    if target == "pairs":
        from . import _pair_candidates
        return _pair_candidates.main(rest)

    print(f"researchwiki candidates: unknown target '{target}'. "
          f"Available: concepts, synthesis, pairs", file=sys.stderr)
    return 1
