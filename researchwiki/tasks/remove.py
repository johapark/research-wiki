"""Retract a wiki page — remove it and every generated trace of it.

✅ Use when: a paper was ingested in error, retracted upstream, superseded by a
   version under a different stem, or landed as a mis-typed commentary — or
   when any other page (synthesis, idea, concept, reference doc) is being
   retired and its `index.md` bullet and inbound back-links should go with it.
❌ Don't use: to re-ingest with corrected metadata — that's
   `prompts/recovery.md`, which keeps the PDF and the back-link graph. Don't
   use it to "clean up" a paper you might want back: the PDF goes too unless
   you pass `--keep-pdf`.

**The target is resolved by filename stem, not by `type:`.** Every page type is
removable the same way; on a non-paper target the paper-shaped machinery (PDF,
supplementary dir, grade/figure caches, `claims` rows) simply finds nothing.
`index.md` and `log.md` are excluded from the scan and can never be the target.

Note the asymmetry: authored prose is protected on *citing* pages, never on the
target. `remove <synthesis-slug> --apply` deletes a hand-authored page that has
passed both gates, and `wiki/` may be gitignored — the dry run is the guard.

Dry-run is the default. Nothing is written until `--apply`.

    researchwiki remove <stem>              # show what would go
    researchwiki remove <stem> --apply      # do it
    researchwiki remove <stem> --apply --keep-pdf

**Generated text is removed; authored text is reported.** Back-link bullets and
the `index.md` entry were written by `promote`, so they go. A sentence on a
synthesis or idea page citing `[[stem#slug]]` was written by a human and has
passed both gates, so it is listed for you to resolve and then re-run
`grade synthesis`. Expect `lint` to report `dangling_claim_anchors` on exactly
those pages afterwards — that is the to-do queue, not a defect.

Run `researchwiki db rebuild && researchwiki reindex` afterwards; the command
prints the reminder.

Exit codes: 0 = scanned or applied; 1 = no such stem; 2 = removal failed.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..log import append_log_md
from ..mutation import mutation
from ..removal import apply as apply_removal, scan


def _print_plan(plan, *, keep_pdf: bool) -> None:
    print(f"researchwiki remove: {plan.stem}")
    if plan.page_path:
        print(f"\n  page          {plan.page_path}  (type: {plan.page_type})")
    else:
        print("\n  page          (none on disk)")

    if plan.files:
        print("  files")
        for f in plan.files:
            skip = "  [kept: --keep-pdf]" if keep_pdf and f.suffix == ".pdf" else ""
            print(f"                {f}{skip}")

    if plan.backlink_pages:
        total = sum(n for _, n in plan.backlink_pages)
        print(f"  back-links    {total} bullet(s) on {len(plan.backlink_pages)} page(s)")
        for page, n in plan.backlink_pages[:10]:
            print(f"                {page}  ({n})")
        if len(plan.backlink_pages) > 10:
            print(f"                ... ({len(plan.backlink_pages) - 10} more)")

    if plan.index_bullet:
        print("  index.md      1 bullet")

    if plan.concept_hubs:
        print(f"  concept hubs  {len(plan.concept_hubs)} spoke registr(y/ies)")
        for hub in plan.concept_hubs:
            print(f"                {hub}")

    if plan.db_rows:
        rows = ", ".join(f"{t}={n}" for t, n in sorted(plan.db_rows.items()))
        print(f"  db rows       {rows}")
    if plan.edge_count:
        print(f"  claim edges   {plan.edge_count}")

    if plan.commentary_pages:
        print("\n  ⚠ commentary page(s) name this as their primary_paper — the "
              "commentary will be left pointing at a paper that no longer exists:")
        for page in plan.commentary_pages:
            print(f"      {page}")

    if plan.prose_refs:
        print(f"\n  ⚠ {len(plan.prose_refs)} authored citation(s) on "
              f"{len({r.path for r in plan.prose_refs})} page(s). "
              f"NOT edited — yours to resolve:")
        for ref in plan.prose_refs[:20]:
            print(f"      {ref.path}:{ref.line_no}  ({ref.page_type})")
            print(f"        {ref.line[:120]}")
        if len(plan.prose_refs) > 20:
            print(f"      ... ({len(plan.prose_refs) - 20} more)")
        print("    After removal these become `dangling_claim_anchors` in lint. "
              "Edit each page, then re-run `researchwiki grade synthesis <page>`.")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki remove",
        description="Retract a wiki page: remove it and every generated trace.",
    )
    parser.add_argument(
        "stem",
        help="Page stem — the page's filename without .md (for a paper, the "
             "basename of papers/{stem}.pdf). Any page type is accepted.",
    )
    parser.add_argument("--apply", action="store_true",
                        help="Actually remove. Without this, nothing is written.")
    parser.add_argument("--keep-pdf", dest="keep_pdf", action="store_true",
                        help="Leave papers/{stem}.pdf in place (re-ingestable).")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit the plan as JSON instead of prose.")
    args = parser.parse_args(argv)

    plan = scan(args.stem)
    if not plan.exists:
        print(f"researchwiki remove: nothing found for stem {args.stem!r}.",
              file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps({
            "stem": plan.stem,
            "page": str(plan.page_path) if plan.page_path else None,
            "page_type": plan.page_type,
            "files": [str(f) for f in plan.files],
            "backlink_pages": [[str(p), n] for p, n in plan.backlink_pages],
            "index_bullet": plan.index_bullet,
            "concept_hubs": [str(p) for p in plan.concept_hubs],
            "commentary_pages": [str(p) for p in plan.commentary_pages],
            "prose_refs": [
                {"path": str(r.path), "page_type": r.page_type,
                 "line_no": r.line_no, "line": r.line}
                for r in plan.prose_refs
            ],
            "db_rows": plan.db_rows,
            "edge_count": plan.edge_count,
            "applied": False,
        }, indent=2))
        if not args.apply:
            return 0

    if not args.as_json:
        _print_plan(plan, keep_pdf=args.keep_pdf)

    if not args.apply:
        print(f"\n  dry run — nothing written. Re-run with --apply to remove.")
        return 0

    try:
        with mutation(
            plan.touched_paths, operation="remove",
            details={"stem": plan.stem},
        ) as snap:
            res = apply_removal(plan, keep_pdf=args.keep_pdf)
            snap.mark_committed()
    except Exception as e:
        print(f"researchwiki remove: failed, rolled back: {e}", file=sys.stderr)
        return 2

    # log.md is append-only: a removal records that it happened rather than
    # editing the paper's ingest entry out of the history.
    append_log_md(
        "remove",
        f"{plan.stem}",
        f"Removed {len(res.removed_files)} file(s), {res.backlinks_removed} "
        f"back-link bullet(s), "
        f"{'1' if res.index_bullet_removed else '0'} index.md bullet, "
        f"{sum(res.db_rows_deleted.values())} db row(s), "
        f"{res.edges_deleted} claim edge(s)."
        + (f" {len(plan.prose_refs)} authored citation(s) left for review."
           if plan.prose_refs else ""),
    )

    # The index bullet is named separately rather than folded into the counts:
    # the plan promises it above, so a summary that never mentions it leaves the
    # reader unable to tell whether it went.
    print(f"\n  removed       {len(res.removed_files)} file(s), "
          f"{res.backlinks_removed} back-link bullet(s), "
          f"{'1' if res.index_bullet_removed else '0'} index.md bullet, "
          f"{sum(res.db_rows_deleted.values())} db row(s), "
          f"{res.edges_deleted} claim edge(s)")
    if res.concept_hubs_updated:
        print(f"  concept hubs  {len(res.concept_hubs_updated)} updated")
    for w in res.warnings:
        print(f"  ⚠ {w}")
    print("\n  next: researchwiki db rebuild && researchwiki reindex")
    if plan.prose_refs:
        print("        then resolve the authored citations listed above and "
              "re-run `researchwiki grade synthesis` on those pages.")
    return 0
