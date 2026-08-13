"""Locate a paper's figure and table captions, and the page each sits on.

Complements `pdf/sections.py`, which extracts caption *text* for the claim
pipeline from the joined document text. That API deliberately has no page
granularity, and its `CAPTION_START_RE` requires pipe-style (`Fig. 1 | ...`)
to avoid body-prose false positives. Showing someone a figure needs the
opposite trade: page numbers, and period/colon styles too, since
`Figure 1: ...` is how most non-Nature venues typeset.

Detection is line-anchored plus a title-shape check. A caption line opens with
the figure token and continues with a *title* — an uppercase letter, a digit,
or an opening quote/paren. A body sentence that happens to start with the same
token continues with a lowercase verb: "Table 2 summarizes the various
datasets...". Requiring the title shape removed every false positive across the
sample papers checked while building this (RAPTOR's "Table 11 shows the prompt
used", Boltz-2's "Table 2 summarizes", "Table 14 reports").

Known miss, accepted: a caption whose title opens on a lowercase proper noun
("Table 9: mdCATH test set...") is filtered out with the false positives.
`figures --page N` is the escape hatch, and a caption list that is one short
is a better failure than one carrying three sentences that aren't captions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pypdfium2
import pypdfium2.raw

_PAGEOBJ_PATH = pypdfium2.raw.FPDF_PAGEOBJ_PATH
_PAGEOBJ_IMAGE = pypdfium2.raw.FPDF_PAGEOBJ_IMAGE

# Group 1: the label token (may carry an "Extended Data" prefix).
# Group 2: the number.
# Then a separator and the start of a title — uppercase, digit, or quote/paren.
#
# The separator set is the venue-variable part, and each member is a style seen
# in the corpus or the benchmark fixtures: `Fig. 1 | Title` (Nature),
# `Figure 1: Title` (most preprints), `Fig 1. Title` (PLOS), `Fig. 1 Title`
# (BMC, no separator at all), and `Figure 1- Title` / `Figure 2 - Title`
# (fonseca-2026, hyphen with or without a leading space). Requiring whitespace
# *after* the separator is what keeps a cross-reference range out: "Figure 1-3
# show..." has a digit where the space must be.
_CAPTION_RE = re.compile(
    r'^[ \t]*((?:Extended\s+Data\s+)?(?:Fig(?:ure)?|Table)\.?)'
    r'\s*(\d+)\s*[|.:–—-]?\s+(["“(]?[A-Z0-9])',
    re.MULTILINE,
)

# Cap the scan the same way the chunk index does: a figure past page 80 is
# beyond anything the grading corpus covers, and the scan is O(pages).
MAX_SCAN_PAGES = 80


@dataclass(frozen=True)
class FigureRef:
    kind: str        # "Figure" | "Table"
    number: int
    page: int        # 1-based, as printed by the renderer
    extended: bool   # Extended Data / supplementary series
    caption: str     # the caption's first line, trimmed

    @property
    def label(self) -> str:
        prefix = "Extended Data " if self.extended else ""
        return f"{prefix}{self.kind} {self.number}"


def page_texts(pdf_path: Path | str, max_pages: int = MAX_SCAN_PAGES) -> list[str]:
    """Per-page text, 0-indexed list. Separate from `pdf.text.extract_pdf`,
    which joins pages and so cannot answer "which page is this on".
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = pypdfium2.PdfDocument(str(pdf_path))
    out: list[str] = []
    try:
        for i in range(min(len(doc), max_pages)):
            page = doc[i]
            try:
                tp = page.get_textpage()
                try:
                    out.append(tp.get_text_bounded() or "")
                finally:
                    tp.close()
            finally:
                page.close()
    finally:
        doc.close()
    return out


def locate_figures(
    pdf_path: Path | str, max_pages: int = MAX_SCAN_PAGES
) -> list[FigureRef]:
    """Every figure/table caption in the PDF, in (kind, number) order."""
    return locate_in_texts(page_texts(pdf_path, max_pages))


def locate_in_texts(pages: list[str]) -> list[FigureRef]:
    """Caption detection over already-extracted per-page text.

    Split from `locate_figures` so the detection rules — which are the part
    with real judgement in them — can be exercised against text fixtures
    rather than requiring a PDF that renders the awkward case.

    First occurrence wins: a paper that reprints "Figure 3" in a supplementary
    recap should resolve to the page where it was introduced.
    """
    seen: dict[tuple[str, bool, int], FigureRef] = {}
    for page_no, text in enumerate(pages, start=1):
        for m in _CAPTION_RE.finditer(text):
            token = m.group(1)
            kind = "Table" if "table" in token.lower() else "Figure"
            extended = "extended" in token.lower()
            number = int(m.group(2))
            key = (kind, extended, number)
            if key in seen:
                continue
            line = text[m.start():].split("\n", 1)[0].strip()
            seen[key] = FigureRef(
                kind=kind,
                number=number,
                page=page_no,
                extended=extended,
                caption=line,
            )
    return sorted(
        seen.values(), key=lambda f: (f.kind, f.extended, f.number)
    )


def graphics_per_page(
    pdf_path: Path | str, max_pages: int = MAX_SCAN_PAGES
) -> list[int]:
    """Count drawable objects (paths + images) per page, 0-indexed.

    Exists for one real corpus shape: an accepted manuscript that collects all
    figure captions onto their own page and puts the artwork several pages
    later. `fonseca-2026` does exactly this — captions on p29-30, artwork on
    p34-37 — so resolving "Figure 1" to its caption page and rendering that
    yields a page of caption text and no figure.

    Counting objects is the cheap way to tell the two apart: a caption page has
    ~0 drawables, an artwork page has hundreds. Used only to *warn* and point
    at candidates; the caller still decides what to render, because rendering
    an extra page it didn't ask for would double the context spend silently.
    """
    pdf_path = Path(pdf_path)
    doc = pypdfium2.PdfDocument(str(pdf_path))
    out: list[int] = []
    try:
        for i in range(min(len(doc), max_pages)):
            page = doc[i]
            try:
                out.append(sum(
                    1 for o in page.get_objects(max_depth=8)
                    if o.type in (_PAGEOBJ_PATH, _PAGEOBJ_IMAGE)
                ))
            finally:
                page.close()
    finally:
        doc.close()
    return out


# A page with fewer drawables than this is text; a figure page has hundreds.
# Set well above 0 because a plain page still carries a rule or a logo.
GRAPHICS_FLOOR = 8


def artwork_candidates(
    densities: list[int], caption_page: int, window: int = 10
) -> list[int]:
    """1-based artwork pages after `caption_page`, **nearest first**.

    Nearest rather than densest: in this layout the plates follow the caption
    block in the same order the captions are listed, so the page closest to
    Figure 1's caption is Figure 1's plate. Ordering by object count instead
    put `fonseca-2026`'s 963-object Figure 4 page ahead of Figure 1's, which
    is confidently wrong rather than merely unhelpful.
    """
    start = caption_page  # 0-indexed index of the page after the caption page
    return [
        i + 1
        for i in range(start, min(len(densities), start + window))
        if densities[i] >= GRAPHICS_FLOOR
    ]


def resolve(
    refs: list[FigureRef], spec: str
) -> FigureRef | None:
    """Resolve a user's `--figure` argument against located captions.

    Accepts a bare number ("3" → Figure 3), an explicit kind ("table 2"),
    and the Extended Data series ("ed 1", "extended data fig 1").
    """
    s = spec.strip().lower()
    extended = bool(re.match(r"^(ed|extended)\b", s))
    s = re.sub(r"^(extended\s+data|extended|ed)\s*", "", s)
    kind = "Table" if s.startswith(("table", "tab")) else "Figure"
    s = re.sub(r"^(figures?|fig\.?|tables?|tab\.?)\s*", "", s)
    m = re.match(r"^(\d+)$", s.strip())
    if not m:
        return None
    number = int(m.group(1))
    for ref in refs:
        if ref.kind == kind and ref.extended == extended and ref.number == number:
            return ref
    return None
