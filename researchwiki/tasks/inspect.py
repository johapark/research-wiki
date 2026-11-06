"""Render what an ingest gate held back as one page for human review.

✅ Use when: `agent ingest --visual-inspection` left drafts in `.agent-output/`
   and you want to clear them. The page puts each flagged figure's rendered PDF
   page beside the model's reading of it — the comparison the receipt asks for
   but cannot show.
❌ Don't use: for factual questions, or for papers already in `wiki/`. This
   reports a queue.

    researchwiki inspect visuals          # → output/inspect/visuals.html
    researchwiki inspect visuals --json   # the same data, on stdout

Read-only and zero-token: it reads receipts, the page renders the ingest already
cached, and (best-effort) the gate reasons from the ingest log. It writes one
file under `output/` and records no decisions — a reviewer acts with the
ordinary commands.

The sub-mode is positional so other review queues (impact reviews, evolution
proposals) can join as `inspect <source>` without changing this surface.

Exit codes: 0 on success, including an empty queue, which renders a page saying
so; 1 for an unknown sub-mode or a bad `--max-bytes`; 2 if `wiki/` is missing or
the output path can't be written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..paths import wiki_dir
from ..review_queue import DEFAULT_IMAGE_BUDGET, build_queue, render_html

SOURCES = ("visuals",)
DEFAULT_OUT = Path("output") / "inspect" / "visuals.html"


def main(argv: list[str]) -> int:
    source = "visuals"
    if argv and not argv[0].startswith("-"):
        source, argv = argv[0], argv[1:]
        if source not in SOURCES:
            print(
                f"researchwiki inspect: unknown review source {source!r} "
                f"(have: {', '.join(SOURCES)})",
                file=sys.stderr,
            )
            return 1

    parser = argparse.ArgumentParser(
        prog="researchwiki inspect visuals",
        description="Render held visual-inspection reviews as a self-contained HTML page.",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output path for the HTML file (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Print the queue as JSON on stdout instead of writing HTML.",
    )
    parser.add_argument(
        "--all", action="store_true", dest="include_resolved",
        help="Include receipts nobody is waiting on — papers already promoted, "
             "and inspections that passed. Default shows only what is pending.",
    )
    parser.add_argument(
        "--all-attempts", action="store_true",
        help="Show every attempt per paper, not just the newest. A re-ingest "
             "writes another receipt, and an older one carries a stale reading.",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Leave out the embedded page renders (a much smaller page).",
    )
    parser.add_argument(
        "--max-bytes", type=int, default=DEFAULT_IMAGE_BUDGET, metavar="N",
        help=f"Budget for embedded images (default: {DEFAULT_IMAGE_BUDGET}). Pages "
             "past it are listed without their render rather than shrunk — a held "
             "paper's PDF is still in inbox/, so it cannot be re-rendered smaller.",
    )
    parser.add_argument(
        "--open", dest="open_browser", action="store_true",
        help="Open the result in your browser afterwards.",
    )
    args = parser.parse_args(argv)

    if args.max_bytes < 0:
        print("researchwiki inspect: --max-bytes cannot be negative", file=sys.stderr)
        return 1
    if not wiki_dir().exists():
        print(
            f"error: {wiki_dir()} not found — run this from the wiki root.",
            file=sys.stderr,
        )
        return 2

    queue = build_queue(
        include_resolved=args.include_resolved,
        all_attempts=args.all_attempts,
        images=not args.no_images,
        image_budget_bytes=args.max_bytes,
    )

    if args.as_json:
        json.dump(queue.as_json(), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    out = args.out
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(queue), encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot write {out}: {exc}", file=sys.stderr)
        return 2

    m = queue.meta
    print(f"Wrote {out}  ({out.stat().st_size / 1024:.0f} KB, self-contained)")
    if not m["n_papers"]:
        print("  nothing pending — no held draft has an unresolved figure reading")
        return 0
    print(f"  {m['n_papers']} draft(s) held · {m['n_blocking']} blocking figure(s) · "
          f"{m['n_images']} page image(s)")
    print(f"  {m['n_ready']} ready to judge · {m['n_papers'] - m['n_ready']} "
          f"needing a re-run first")
    if m["n_blocking"] and m["n_blocking_with_reading"] < m["n_blocking"]:
        print(f"  {m['n_blocking'] - m['n_blocking_with_reading']} blocking figure(s) "
              f"have no reading to compare — those papers need re-ingesting")
    if m["n_images_omitted"]:
        print(f"  ⚠ {m['n_images_omitted']} page image(s) omitted to stay under "
              f"--max-bytes ({args.max_bytes}); raise it or narrow the queue")
    if m["n_unreadable_receipts"]:
        print(f"  ⚠ {m['n_unreadable_receipts']} receipt(s) could not be parsed")

    if args.open_browser:
        import webbrowser
        # resolve() so a relative --out still yields a valid file:// URI
        if not webbrowser.open(out.resolve().as_uri()):
            print("  (couldn't launch a browser — open the file above manually)")

    return 0
