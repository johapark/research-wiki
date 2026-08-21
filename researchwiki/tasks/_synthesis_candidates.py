"""CLI wrapper for the synthesis-candidates detector.

Hidden from `researchwiki` command auto-discovery (leading underscore per
`__main__.py:_discover_tasks`); invoked via `researchwiki candidates synthesis`
through `tasks.candidates._run_synthesis`.

The detection logic — graph build, LLM judge, proposal rendering — lives in
`researchwiki.synthesis_candidates`. This file is just argument parsing and
the CLI output shape (JSON vs prose report).

Where crosslink discovery surfaces topical relationships at ingest and
`evolve` surfaces edits to existing synthesis pages, this surfaces *missing*
synthesis pages — emergent clusters of related papers that no existing
synthesis page covers.

✅ Use when: after ≥5 ingests since the last run, or whenever you want to
   ask "what synthesis pages are we missing?" Output is proposals; nothing
   in `wiki/` mutates.
❌ Don't use: as a substitute for human curation — many clusters will be
   topical noise. The point is to surface candidates, not to author syntheses.

The structural scan is local by default. ``--judge`` opts into configured-model
calls; ``--write-proposals`` persists the review artifacts under `.ingest/`.

Exit codes: 0 = completed (including no candidates); 2 = semantic index absent.
"""

from __future__ import annotations

import argparse
import json
import time

from ..index.graph import EDGE_THRESHOLD
from ..fsatomic import write_text_atomic
from ..paths import ingest_dir
from ..synthesis_candidates import (
    Candidate,
    DEFAULT_COVERED,
    DEFAULT_EXTEND,
    DEFAULT_MIN_CLUSTER,
    find_candidates,
    render_proposal,
)


def _verdict_tag(c: Candidate, key: str) -> str:
    """Inline verdict marker for CLI output: ✓ in_scope, · tangential, ✗ out_of_scope, - un-judged."""
    if not c.judged:
        return "-"
    for v in c.member_verdicts:
        if v.key == key:
            return {"in_scope": "✓", "tangential": "·", "out_of_scope": "✗"}.get(v.verdict, "?")
    return "?"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki candidates synthesis",
        description="Detect emergent paper clusters that could become synthesis pages.",
    )
    parser.add_argument("--min-cluster", type=int, default=DEFAULT_MIN_CLUSTER,
                        help=f"Minimum cluster size to surface (default: {DEFAULT_MIN_CLUSTER})")
    parser.add_argument("--threshold", type=float, default=EDGE_THRESHOLD,
                        help=f"Edge weight threshold for cluster membership (default: {EDGE_THRESHOLD})")
    parser.add_argument("--covered-threshold", type=float, default=DEFAULT_COVERED,
                        help=f"Synthesis-overlap above which a cluster is treated as already covered "
                             f"and skipped (default: {DEFAULT_COVERED})")
    parser.add_argument("--extend-threshold", type=float, default=DEFAULT_EXTEND,
                        help=f"Synthesis-overlap above which the verdict is 'extend' rather than "
                             f"'new' (default: {DEFAULT_EXTEND}). Below this → 'new'.")
    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument("--judge", action="store_true",
                             help="Run the configured-model editorial-scoping pass. "
                                  "Without it the scan is local and structural only.")
    judge_group.add_argument("--no-judge", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--write-proposals", action="store_true",
                        help="Write proposal markdown to .ingest/synthesis-candidates/. "
                             "Without it the command is a non-mutating preview.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print candidates but don't write proposal files.")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit a JSON object instead of the prose report.")
    args = parser.parse_args(argv)
    if args.dry_run and args.write_proposals:
        parser.error("--dry-run and --write-proposals cannot be combined")
    if args.as_json and args.write_proposals:
        parser.error("--json is a report; omit --write-proposals or run the prose mode")

    t0 = time.time()
    candidates, stats = find_candidates(
        min_cluster=args.min_cluster,
        edge_threshold=args.threshold,
        covered_threshold=args.covered_threshold,
        extend_threshold=args.extend_threshold,
        judge=args.judge,
    )
    dt = time.time() - t0

    # Error check runs before the JSON emit so a failed run (e.g. semantic index
    # not built) returns the documented exit 2 for JSON callers too, not 0.
    if "error" in stats:
        print(f"error: {stats['error']}", file=sys.stderr)
        return 2

    if args.as_json:
        out = {
            "stats": stats,
            "elapsed_seconds": round(dt, 2),
            "candidates": [
                {
                    "slug": c.slug,
                    "verdict": c.verdict,
                    "n_members": len(c.members),
                    "members": c.members,
                    "members_missing_from_nearest": c.members_missing_from_nearest,
                    "density": round(c.density, 3),
                    "edge_signal_counts": c.edge_signal_counts,
                    "common_keywords": c.common_keywords[:10],
                    "nearest_synthesis": c.nearest_synthesis,
                    "nearest_synthesis_overlap": round(c.nearest_synthesis_overlap, 2),
                    "judged": c.judged,
                    "judge_batches": c.judge_batches,
                    "judge_topic": c.judge_topic,
                    "member_verdicts": [
                        {"key": v.key, "verdict": v.verdict, "rationale": v.rationale}
                        for v in c.member_verdicts
                    ],
                }
                for c in candidates
            ],
        }
        print(json.dumps(out, indent=2))
        return 0

    print(f"Synthesis-candidate scan ({dt:.1f}s)")
    print(f"  papers scanned:      {stats['n_papers']}")
    print(f"  edges above thr:     {stats['n_edges_above_threshold']}")
    print(f"  clusters ≥ {args.min_cluster}:        {stats['n_clusters_found']}")
    print(f"  already covered:     {stats['n_already_covered']}")
    print(f"  → 'extend' verdicts: {stats.get('n_extend', 0)}")
    print(f"  → 'new' verdicts:    {stats.get('n_new', 0)}")
    if stats.get("n_judge_batches", 0):
        in_tok = stats.get("judge_input_tokens", 0)
        out_tok = stats.get("judge_output_tokens", 0)
        print(f"  judge (B4.5):         {stats.get('n_judged', 0)} complete, "
              f"{stats['n_judge_batches']} batch(es) "
              f"(~{in_tok//1000}K in / ~{out_tok//1000}K out)")
    print()

    if not candidates:
        print("No actionable synthesis candidates. (Either everything is already "
              "covered, or no clusters reached min-size with current thresholds.)")
        return 0

    # Render extend verdicts first so they're visible at the top — they're
    # usually the more actionable case (smaller diff against existing page).
    for c in sorted(candidates, key=lambda c: (c.verdict != "extend", -c.density)):
        verdict_tag = "EXTEND" if c.verdict == "extend" else "NEW"
        print(f"## [{verdict_tag}] {c.slug}  ({len(c.members)} papers, density {c.density:.2f})")
        if c.judged and c.verdict == "new" and c.judge_topic:
            print(f"  judge-proposed topic: {c.judge_topic!r}")
        if c.verdict == "extend":
            print(f"  → augment [[{c.nearest_synthesis}]]  "
                  f"(currently covers {c.nearest_synthesis_overlap:.0%})")
            print(f"  → {len(c.members_missing_from_nearest)} member(s) missing from synthesis:")
            for k in c.members_missing_from_nearest:
                tag = _verdict_tag(c, k)
                print(f"        {tag} {k}  — {c.titles.get(k, '')[:80]}")
        else:
            for k in c.members:
                tag = _verdict_tag(c, k)
                print(f"  {tag} {k}  — {c.titles.get(k, '')[:90]}")
            if c.nearest_synthesis:
                print(f"  closest existing synthesis: {c.nearest_synthesis}  "
                      f"(overlap {c.nearest_synthesis_overlap:.2f})")
        if c.judged:
            in_count = sum(1 for v in c.member_verdicts if v.verdict == "in_scope")
            tan_count = sum(1 for v in c.member_verdicts if v.verdict == "tangential")
            out_count = sum(1 for v in c.member_verdicts if v.verdict == "out_of_scope")
            print(f"  judge: {in_count} in_scope, {tan_count} tangential, "
                  f"{out_count} out_of_scope")
        print(f"  signals: {c.edge_signal_counts}")
        print(f"  common keywords: {', '.join(c.common_keywords[:10]) or '(none)'}")
        print()

    if not args.write_proposals:
        print("(preview — no files written. Re-run with --write-proposals to persist review artifacts.)")
        return 0

    out_dir = ingest_dir() / "synthesis-candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for c in candidates:
        out_path = out_dir / f"{c.slug}.md"
        write_text_atomic(out_path, render_proposal(c))
        written += 1
    print(f"Wrote {written} proposal(s) to {out_dir}/")
    print("Review each proposal, then run the `synthesize` command shown inside "
          "to scaffold the actual page. Discard the proposal when done.")
    return 0
