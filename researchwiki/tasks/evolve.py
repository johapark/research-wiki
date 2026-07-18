"""Generate memory-evolution proposals for an existing paper page.

Given a paper P already in the wiki, find top-k semantically-related synthesis
pages and ask an LLM whether each should be edited in light of P. Proposals
are written to `.ingest/{P-stem}-evolution-proposals/` as markdown files.
Nothing in `wiki/` is modified — review and apply manually.

✅ Use when: you want to (re-)run evolution against an already-ingested paper
   on demand — e.g. after the synthesis coverage has grown, or for the
   digest ingest path, which (unlike the agent path) does not evolve for you.
   Run with `--dry-run` first to see the verdict distribution before letting
   it write proposals.
ℹ️ Note: the agent ingest path already runs this automatically post-promote
   (the `memory_evolve` phase in the runner), so you usually don't invoke the
   CLI by hand after `agent ingest`. It stays a separate command for the
   digest path and for re-running on demand.

Usage:
  researchwiki evolve cgt/du-2025-...           # propose against neighbors of Du 2025
  researchwiki evolve cgt/du-2025-... --k 5     # narrower neighbor set
  researchwiki evolve cgt/du-2025-... --dry-run # show verdicts; don't write files

Exit code: 0 always (failures are reported per-neighbor; aggregate is exit 0).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..agents.phases import render_proposal_md
from ..agents.phases.evolution import (
    DEFAULT_MIN_COSINE,
    auto_apply_proposal,
    propose_evolution,
)
from ..fsatomic import write_text_atomic
from ..paths import ingest_dir


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki evolve",
        description="Propose edits to neighboring synthesis pages "
                    "for a given source paper.",
    )
    parser.add_argument("source", metavar="CATEGORY/STEM",
                        help="Wiki key of the source paper, e.g. "
                             "'cgt/du-2025-a-versatile-crisprcas9-system-off-target'.")
    parser.add_argument("--k", type=int, default=8,
                        help="Max number of semantic neighbors to consider (default: 8).")
    parser.add_argument("--min-cosine", type=float, default=None,
                        help="Cosine-similarity threshold below which neighbors "
                             "are skipped (no LLM call). Default uses the value "
                             "in evolution.DEFAULT_MIN_COSINE.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print verdict distribution but don't write proposal files.")
    parser.add_argument("--auto-apply", action="store_true",
                        help="For each generated proposal, attempt to materialize it "
                             "in place under a conservative gate: only `refine` "
                             "bullet-append proposals at confidence >= 0.9 with passing "
                             "structural checks. Line-replace refines, enhances, and "
                             "contrasts always go to disk for human review. "
                             "Default off — opt in once you trust the proposer.")
    args = parser.parse_args(argv)

    if "/" not in args.source:
        parser.error("source must be in the form 'category/stem' (e.g. 'cgt/du-2025-...')")

    min_cosine = args.min_cosine if args.min_cosine is not None else DEFAULT_MIN_COSINE
    print(f"Generating evolution proposals for [[{args.source}]] "
          f"(k={args.k}, min_cosine={min_cosine:.2f})...")
    print()

    t0 = time.time()
    try:
        proposals, stats = propose_evolution(args.source, k=args.k, min_cosine=min_cosine)
    except FileNotFoundError as e:
        print(f"error: {e}")
        return 1
    dt = time.time() - t0

    print(f"KNN candidates: {stats['n_knn']} → "
          f"{stats['n_above_threshold']} above cosine {min_cosine:.2f} → "
          f"{stats['n_judged']} judged ({dt:.1f}s)")
    if stats["n_above_threshold"] == 0:
        print(f"All KNN hits below threshold; no LLM calls made. "
              f"Either no related synthesis pages, or relax --min-cosine.")
        return 0

    if not proposals:
        print(f"No proposals generated. (Likely all neighbors already "
              f"reference [[{args.source}]].)")
        return 0

    actionable = [p for p in proposals if p.is_actionable()]
    by_verdict: dict[str, int] = {}
    for p in proposals:
        by_verdict[p.verdict] = by_verdict.get(p.verdict, 0) + 1

    print(f"Verdict distribution ({len(actionable)} actionable):")
    for v, n in sorted(by_verdict.items(), key=lambda x: -x[1]):
        print(f"  {v:8s} {n}")
    print()

    for p in proposals:
        marker = "✓" if p.is_actionable() else "·"
        print(f"  {marker} {p.verdict:8s} (conf={p.confidence:.2f})  [[{p.target_key}]]")
        if p.rationale:
            print(f"      {p.rationale[:160]}")

    if args.dry_run:
        print()
        print("(dry run — no files written. Re-run without --dry-run to commit.)")
        return 0

    if not actionable:
        print()
        print("No actionable proposals to write.")
        return 0

    out_dir = _proposal_dir(args.source)
    out_dir.mkdir(parents=True, exist_ok=True)

    applied: list = []                          # list[EvolutionProposal]
    sandboxed: list[tuple] = []                 # list[(EvolutionProposal, reason)]
    for p in actionable:
        if args.auto_apply:
            ok, reason = auto_apply_proposal(p)
            if ok:
                applied.append(p)
                continue
            sandboxed.append((p, reason))
        out_path = out_dir / _proposal_filename(p)
        write_text_atomic(out_path, render_proposal_md(p))

    print()
    if applied:
        print(f"Auto-applied {len(applied)} proposal(s) in place:")
        for p in applied:
            print(f"  ✓ {p.verdict:8s} (conf={p.confidence:.2f})  [[{p.target_key}]]")
        print()

    n_sandboxed = len(actionable) - len(applied)
    if n_sandboxed > 0:
        print(f"Wrote {n_sandboxed} proposal file(s) to {out_dir}/ (review + apply manually):")
        if args.auto_apply and sandboxed:
            for p, reason in sandboxed:
                print(f"  · {p.verdict:8s} (conf={p.confidence:.2f})  [[{p.target_key}]]  — {reason}")
    elif applied:
        # Everything auto-applied; remove the empty dir we created.
        try:
            out_dir.rmdir()
        except OSError:
            pass
    return 0


def _proposal_dir(source_key: str) -> Path:
    """`.ingest/{stem}-evolution-proposals/` (stem only, not category, since
    stems are already unique within the wiki)."""
    stem = source_key.split("/", 1)[-1]
    return ingest_dir() / f"{stem}-evolution-proposals"


def _proposal_filename(prop) -> str:
    """e.g. `extend__cgt__crispr-off-target-strategies.md` — verdict prefix
    so a directory listing groups by edit kind."""
    target_safe = prop.target_key.replace("/", "__")
    return f"{prop.verdict}__{target_safe}.md"
