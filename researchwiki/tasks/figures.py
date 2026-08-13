"""List a paper's figures, and render one when a question turns on it.

✅ Use when: an answer depends on what a figure *shows* — a trend, an axis
   range, a panel the text only gestures at — and the caption alone doesn't
   settle it. Rule 3 (re-read the PDF) for the case `pdf-search` can't serve,
   because the passage says "see Fig. 4" and Fig. 4 is where the number lives.
❌ Don't use: as a first move. Try `researchwiki claims` (pre-graded), then
   `researchwiki pdf-search` (raw passages), then the caption list below —
   captions carry the quantitative results often enough that the list alone
   usually answers the question. Never render a paper "to have a look".

Two modes:

  researchwiki figures <stem>                # list captions — text, no render
  researchwiki figures <stem> --figure 3     # render the page carrying Fig. 3
  researchwiki figures <stem> --page 7       # render page 7 directly

Listing is free. Rendering costs no tokens either — it is local compute — but
the PNG costs *context* when you `Read` it, in proportion to its pixel area, so
one page is rendered per invocation unless `--pages` names more. The reported
dimensions are what that spend is proportional to.

Renders land in `.figures-cache/{stem}/` (gitignored, safe to delete).

Exit codes: 0 = listed or rendered; 1 = no figures found / unknown figure;
2 = no PDF for that stem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..paths import figures_cache_dir, resolve_pdf
from ..pdf import figures as figlib
from ..pdf.render import DEFAULT_DPI, render_page


def _parse_pages(spec: str) -> list[int]:
    """Parse `--pages 3,7,9` into [3, 7, 9]. Raises ValueError on junk."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            raise ValueError(f"page numbers are 1-based: {part!r}")
        out.append(n)
    if not out:
        raise ValueError("no page numbers given")
    return out


def _render_and_report(stem: str, pdf: Path, pages: list[int], dpi: int,
                       as_json: bool) -> int:
    cache = figures_cache_dir() / stem
    rendered = []
    for page in pages:
        dest = cache / f"p{page}@{dpi}.png"
        try:
            rendered.append(render_page(pdf, page, dest, dpi=dpi))
        except ValueError as e:
            print(f"researchwiki figures: {e}", file=sys.stderr)
            return 1

    if as_json:
        print(json.dumps([
            {"path": str(r.path), "page": r.page, "width": r.width,
             "height": r.height, "dpi": r.dpi, "grayscale": r.grayscale}
            for r in rendered
        ], indent=2))
        return 0

    for r in rendered:
        print(f"{r.path}")
        print(f"  page {r.page} · {r.width}x{r.height}px @ {r.dpi} DPI"
              f"{' · grayscale' if r.grayscale else ''}")
    if len(rendered) == 1:
        print("\nRead this file to view the figure. Re-render is cheap — delete "
              "the cache dir any time.")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki figures",
        description="List a paper's figure captions; render a page on request.",
    )
    parser.add_argument("stem", help="Paper stem (basename of papers/{stem}.pdf).")
    parser.add_argument(
        "--figure", metavar="REF",
        help='Render the page carrying this caption: "3", "table 2", "ed 1".',
    )
    parser.add_argument(
        "--page", type=int, metavar="N",
        help="Render this 1-based PDF page directly (escape hatch when a "
             "caption isn't detected).",
    )
    parser.add_argument(
        "--pages", metavar="A,B,C",
        help="Render several pages. Explicit by design — each page read into "
             "context costs in proportion to its pixel area.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help=f"Render resolution (default: {DEFAULT_DPI}).")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit JSON instead of formatted text.")
    args = parser.parse_args(argv)

    try:
        pdf = resolve_pdf(args.stem)
    except Exception as e:
        print(f"researchwiki figures: {e}", file=sys.stderr)
        return 2
    if not Path(pdf).exists():
        print(f"researchwiki figures: no PDF at {pdf}", file=sys.stderr)
        return 2

    selectors = [bool(args.figure), args.page is not None, bool(args.pages)]
    if sum(selectors) > 1:
        print("researchwiki figures: pass only one of --figure / --page / --pages",
              file=sys.stderr)
        return 1

    # Direct page renders skip caption detection entirely — that's the point of
    # the escape hatch, and it keeps a paper whose captions don't parse usable.
    if args.page is not None:
        return _render_and_report(args.stem, pdf, [args.page], args.dpi, args.as_json)
    if args.pages:
        try:
            pages = _parse_pages(args.pages)
        except ValueError as e:
            print(f"researchwiki figures: {e}", file=sys.stderr)
            return 1
        return _render_and_report(args.stem, pdf, pages, args.dpi, args.as_json)

    refs = figlib.locate_figures(pdf)

    if args.figure:
        ref = figlib.resolve(refs, args.figure)
        if ref is None:
            print(f"researchwiki figures: no caption matching {args.figure!r} in "
                  f"{args.stem}.", file=sys.stderr)
            if refs:
                print("  found: " + ", ".join(r.label for r in refs), file=sys.stderr)
                print("  (or use --page N to render a page directly)", file=sys.stderr)
            else:
                print("  no captions detected at all — use --page N.", file=sys.stderr)
            return 1
        print(f"{ref.label} → page {ref.page}")
        print(f"  {ref.caption}")
        print()
        return _render_and_report(args.stem, pdf, [ref.page], args.dpi, args.as_json)

    # Default: list. No render, no image tokens.
    if args.as_json:
        print(json.dumps([
            {"kind": r.kind, "number": r.number, "page": r.page,
             "extended": r.extended, "label": r.label, "caption": r.caption}
            for r in refs
        ], indent=2, ensure_ascii=False))
        return 0 if refs else 1

    if not refs:
        print(f"No figure or table captions detected in {args.stem}.", file=sys.stderr)
        print("  Either the paper has none, or its caption style isn't matched — "
              "`--page N` renders a page directly.", file=sys.stderr)
        return 1

    for r in refs:
        print(f"  p{r.page:<4} {r.label:<22} {r.caption[:96]}")
    print(f"\n{len(refs)} caption(s). These often answer the question on their "
          f"own; render only if not:\n  researchwiki figures {args.stem} --figure N")
    return 0
