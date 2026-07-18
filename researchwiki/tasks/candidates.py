"""Surface opportunity signals — un-scaffolded concept hubs and un-covered synthesis clusters.

✅ Use when: `status` prints a nonzero `Concept-hub candidates: N bridge term(s)`
   line (concepts target), OR you want to ask "what synthesis pages are we
   missing?" after a batch of ingests (synthesis target).
❌ Don't use: as a substitute for human curation. Both surfaces are advisory.

Two targets:

  researchwiki candidates concepts   [--bridges] [--persist-edges] [--json]
  researchwiki candidates synthesis  [--min-cluster N] [--threshold F] ...

**concepts** — recurring vocabulary terms mentioned by ≥3 wiki papers with no
`wiki/concepts/{slug}.md` yet. Cheap (local, sub-second), no LLM. Bridge tier
(--bridges: span ≥ 2 categories) is the highest-leverage — those are the terms
the citation graph and semantic-KNN don't naturally connect. `status`
auto-surfaces the bridge count, so treat that line as the trigger.

**synthesis** — dense paper clusters (wikilinks + semantic cosine ≥ 0.65 +
keyword Jaccard ≥ 0.2, connected components) not covered by any existing
synthesis page. Higher noise rate than concepts, so **not** auto-surfaced by
`status` — the run cadence is user-initiated: after ≥5 ingests since the last
run, or when a cross-paper question lands in unfamiliar territory. LLM-judged
by default (--no-judge to skip); ~30s + a few Anthropic calls per run.

Exit code: 0 always. Both targets are read-only opportunity signals; the only
output that mutates state is the synthesis-target's proposal files under
`.ingest/synthesis-candidates/` (skipped with --dry-run).
"""

from __future__ import annotations

import argparse
import sys


def _run_concepts(argv: list[str]) -> int:
    """Dispatch to the concepts-candidate collector without going through
    the top-level `concepts` command (which now only scaffolds / refreshes).
    Reuses the same `collect_candidates` implementation."""
    import json

    from ..concepts import collect_candidates

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
    args = parser.parse_args(argv)

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
    print('_scaffold one only if you can write a genuine concept-thesis for it (see docs/concept-vs-glossary.md)._')
    print("_Pass --persist-edges to write `instantiates` edges into `.claim-graph/edges.db`._")
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

    print(f"researchwiki candidates: unknown target '{target}'. "
          f"Available: concepts, synthesis", file=sys.stderr)
    return 2
