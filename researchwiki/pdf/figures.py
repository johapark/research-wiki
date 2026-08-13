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

# Group 1: the label token (may carry an "Extended Data" prefix).
# Group 2: the number.
# Then a separator and the start of a title — uppercase, digit, or quote/paren.
_CAPTION_RE = re.compile(
    r'^[ \t]*((?:Extended\s+Data\s+)?(?:Fig(?:ure)?|Table)\.?)'
    r'\s*(\d+)\s*[|.:]?\s+(["“(]?[A-Z0-9])',
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
