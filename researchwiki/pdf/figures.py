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

A caption is not always on the same page as its figure. `artwork_coverage` and
`prefer_artwork_page` handle the two layouts in this corpus where it isn't.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import pypdfium2
import pypdfium2.raw

from .repair import _INTERIOR_CTRL_RE

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
    also_on: tuple[int, ...] = ()   # later pages repeating this label

    @property
    def label(self) -> str:
        prefix = "Extended Data " if self.extended else ""
        return f"{prefix}{self.kind} {self.number}"


def page_texts(pdf_path: Path | str, max_pages: int = MAX_SCAN_PAGES) -> list[str]:
    """Per-page text, 0-indexed list. Separate from `pdf.text.extract_pdf`,
    which joins pages and so cannot answer "which page is this on".

    Interior control bytes are stripped, the same way `pdf_shape` does it and
    for the same two reasons: a soft-hyphenated `Fig\\x02ure 3 | ...` should
    still match the caption pattern, and a caption is printed straight to the
    terminal, where a raw control byte is garbage on the reader's screen. The
    full ligature repair is deliberately not run — it costs the 1.7 MB
    dictionary to fix words nobody is grading against.
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
                    out.append(_INTERIOR_CTRL_RE.sub("", tp.get_text_bounded() or ""))
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

    The first occurrence is the caption of record — a paper that reprints
    "Figure 3" in a supplementary recap should resolve to where it was
    introduced. Later occurrences are kept in `also_on` rather than dropped,
    because in the append-the-plates layout the label is printed on the plate
    as well, and that is where the artwork is: see `prefer_artwork_page`.
    """
    seen: dict[tuple[str, bool, int], FigureRef] = {}
    repeats: dict[tuple[str, bool, int], list[int]] = {}
    for page_no, text in enumerate(pages, start=1):
        for m in _CAPTION_RE.finditer(text):
            token = m.group(1)
            kind = "Table" if "table" in token.lower() else "Figure"
            extended = "extended" in token.lower()
            number = int(m.group(2))
            key = (kind, extended, number)
            if key in seen:
                # Keep later occurrences rather than dropping them. In the
                # append-the-plates-at-the-end layout the label is printed on
                # the plate too, so a repeat is often where the artwork
                # actually is — see `prefer_artwork_page`.
                if page_no != seen[key].page:
                    repeats.setdefault(key, []).append(page_no)
                continue
            line = text[m.start():].split("\n", 1)[0].strip()
            seen[key] = FigureRef(
                kind=kind,
                number=number,
                page=page_no,
                extended=extended,
                caption=line,
            )
    out = [
        replace(ref, also_on=tuple(sorted(set(repeats.get(key, ())))))
        for key, ref in seen.items()
    ]
    return sorted(out, key=lambda f: (f.kind, f.extended, f.number))


def prefer_artwork_page(ref: FigureRef, coverage: list[float]) -> FigureRef:
    """Re-point `ref` at a repeat of its label that actually carries artwork.

    Only moves when the first occurrence has no artwork *and* a later one does,
    so an ordinary paper — caption and figure on the same page — is untouched.

    This is evidence, not a guess: the page it moves to prints the same figure
    label. `aygun-2026` is the case it was written for — legends collected on
    p30-31, plates appended on p37+ with "Extended Data Fig. 1" printed on the
    plate itself, which first-occurrence-wins was discarding.
    """
    def cov(page: int) -> float:
        return coverage[page - 1] if 1 <= page <= len(coverage) else 0.0

    if cov(ref.page) >= ARTWORK_FLOOR:
        return ref
    for page in ref.also_on:
        if cov(page) >= ARTWORK_FLOOR:
            return replace(ref, page=page)
    return ref


def artwork_coverage(
    pdf_path: Path | str, max_pages: int = MAX_SCAN_PAGES
) -> list[float]:
    """Fraction of each page's area covered by drawables, 0-indexed.

    Exists for the layouts where a caption and its artwork are on *different*
    pages, so resolving a caption and rendering its page shows text and no
    figure. Two real shapes in this corpus:

      - accepted manuscripts that collect every caption onto one page with the
        plates a few pages later (`fonseca-2026`: captions p29-30, plates
        p31-37);
      - preprints that run the whole manuscript and then append the figures at
        the end, which puts a much larger gap between the two.

    **Area, not object count.** Counting objects gets this backwards in both
    directions: a page holding one full-page raster figure has a single image
    object (muslu-2026 p7 — one object covering 38% of the page), while a
    plain text page with a header rule and a logo has two. Summing bounding-box
    areas separates them cleanly — measured on this corpus, caption-only and
    plain text pages land at ~1.5% while every real figure page checked ran
    15-98%.

    Used only to *warn* and point at candidates; the caller still chooses what
    to render, because rendering a page it didn't ask for spends context it
    didn't agree to spend.
    """
    pdf_path = Path(pdf_path)
    doc = pypdfium2.PdfDocument(str(pdf_path))
    out: list[float] = []
    try:
        for i in range(min(len(doc), max_pages)):
            page = doc[i]
            try:
                width, height = page.get_size()
                page_area = width * height
                if page_area <= 0:
                    out.append(0.0)
                    continue
                total = 0.0
                for obj in page.get_objects(max_depth=8):
                    if obj.type not in (_PAGEOBJ_PATH, _PAGEOBJ_IMAGE):
                        continue
                    try:
                        left, bottom, right, top = obj.get_bounds()
                    except Exception:
                        continue  # malformed object; contributes nothing
                    total += max(0.0, right - left) * max(0.0, top - bottom)
                # Overlapping objects can sum past 1.0; the cap keeps the value
                # readable without changing any comparison against the floor.
                out.append(min(total / page_area, 1.0))
            finally:
                page.close()
    finally:
        doc.close()
    return out


# Below this fraction a page is text. Caption-only and plain prose pages
# measured ~1.5% across this corpus; the lowest real figure page was ~15%.
ARTWORK_FLOOR = 0.06


def artwork_candidates(
    coverage: list[float], caption_page: int, window: int | None = None
) -> list[int]:
    """1-based artwork pages after `caption_page`, **nearest first**.

    Nearest rather than largest: plates follow the caption block in roughly the
    order the captions are listed, so the page closest to Figure 1's caption is
    the better guess. Ordering by size instead put `fonseca-2026`'s densest
    (last) plate ahead of the first one — confidently wrong rather than merely
    unhelpful.

    `window` defaults to the rest of the document. A bounded window was the
    first design and it broke on the append-at-the-end layout, where the gap
    between a caption and its plate runs to 20+ pages.
    """
    start = caption_page  # 0-indexed index of the page after the caption page
    end = len(coverage) if window is None else min(len(coverage), start + window)
    return [i + 1 for i in range(start, end) if coverage[i] >= ARTWORK_FLOOR]


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
