"""Render the wiki's link graph as a self-contained interactive HTML page.

✅ Use when: you want to *see* corpus structure — which clusters exist, which
   papers bridge them, where contradictions concentrate. Best after a batch of
   ingests, or when deciding whether a cluster deserves a synthesis page.
❌ Don't use: to answer a factual question. The graph shows structure, not
   claims — `researchwiki claims` and `search` are for content, and
   `claim-graph --tensions` gives you the contradiction list in text.

Two edge kinds are drawn. Wikilinks are the structure page authors wrote; claim
edges come from `.claim-graph/edges.db` and are judged relations between
individual claims (`builds_on`, `refines`, `corroborates`, `measures_same`,
`contradicts`, `instantiates`). `contradicts` is drawn loud because a list of
tensions cannot show you that four of them cluster on one paper.

Only live claim statuses are drawn by default (`candidate`, `confirmed`,
`promoted`). `stale` and `rejected` are excluded for a semantic reason, not a
volume one: `stale` means the claim the edge was judged against has changed, so
it is an assertion nobody currently stands behind. (Volume would be a bad
argument — parallel claim pairs collapse per page pair, so widening to `stale` on
a corpus with 13,535 of them adds only ~20 visible edges.) `--claim-status`
overrides; `--no-claims` drops them entirely.

Zero tokens, no network. Layout runs client-side in the browser, so this command
only decides what goes in the file.

Exit codes: 0 on success (an empty wiki still writes a valid page and exits 0);
2 if `wiki/` is missing or the output path can't be written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..paths import wiki_dir
from ..visualize import LIVE_CLAIM_STATUSES, build_graph, render_html

DEFAULT_OUT = Path("output") / "graph.html"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki visualize",
        description="Render the wiki link graph as a self-contained HTML page.",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output path for the HTML file (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--no-claims", action="store_true",
        help="Draw only [[wikilinks]]; skip the typed claim-graph edges.",
    )
    parser.add_argument(
        "--claim-status", action="append", metavar="STATUS",
        help="Claim-edge status to include; repeatable. "
             f"Default: {', '.join(LIVE_CLAIM_STATUSES)}. "
             "Pass `--claim-status stale` to see retracted edges too.",
    )
    parser.add_argument(
        "--open", dest="open_browser", action="store_true",
        help="Open the result in your browser afterwards.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the graph as JSON on stdout instead of writing HTML.",
    )
    args = parser.parse_args(argv)

    if not wiki_dir().exists():
        print(f"error: {wiki_dir()} not found — run this from the wiki root.", file=sys.stderr)
        return 2

    statuses = tuple(args.claim_status) if args.claim_status else LIVE_CLAIM_STATUSES
    graph = build_graph(
        claim_statuses=statuses,
        include_claims=not args.no_claims,
    )

    if args.json:
        json.dump(graph.as_json(), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    out = args.out
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(graph), encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write {out}: {exc}", file=sys.stderr)
        return 2

    m = graph.meta
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({size_kb:.0f} KB, self-contained)")
    print(f"  {m['n_nodes']} pages · {m['n_wikilinks']} wikilinks · "
          f"{m['n_claim_edges']} claim edges")
    if m["type_counts"]:
        print("  types: " + ", ".join(f"{n} {t}" for t, n in m["type_counts"].items()))
    if m["relation_counts"]:
        print("  relations: " + ", ".join(f"{n} {r}" for r, n in m["relation_counts"].items()))
    if m["claim_edges_dropped"]:
        # Not a warning to swallow: these are edges whose paper is gone from the
        # wiki, which means `claim-graph reconcile` has work to do.
        print(f"  ⚠ {m['claim_edges_dropped']} claim edge(s) skipped — endpoint has no "
              f"wiki page (run `researchwiki claim-graph reconcile`)")
    if not m["n_nodes"]:
        print("  (no pages yet — the page will render an empty-state message)")

    if args.open_browser:
        import webbrowser
        # resolve() so a relative --out still yields a valid file:// URI
        if not webbrowser.open(out.resolve().as_uri()):
            print("  (couldn't launch a browser — open the file above manually)")

    return 0
