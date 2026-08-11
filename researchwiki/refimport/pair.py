"""Match export records to PDFs on disk.

Three rungs, tried in order, each consuming what it matches:

  1. `declared` — a path the export itself named
  2. `doi`      — the DOI printed in the PDF equals the record's DOI
  3. `title`    — the PDF's first page contains the record's title

**Rung 1 is a fast path, not the backbone.** A real ReadCube library exported as
both RIS and BibTeX contained zero attachment paths in either file, and CSL-JSON
carries none from any exporter seen. Zotero/BBT does write them, which is the
only reason the rung exists.

**No per-tool directory conventions.** Zotero's `storage/<key>/`, Paperpile's
Drive layout and ReadCube's tree all drift across versions and sync modes;
encoding them means a format the docs claim to support silently pairs zero
files. The content-based rungs degrade honestly instead, and every pairing
records which rung produced it so a bad run is diagnosable rather than mysterious.

**One extraction per PDF, ever.** The naive shape — for each record, scan each
PDF — is O(records x pdfs), which on a 532-record library with ~500 PDFs turns a
free phase into an overnight one. `build_pdf_index` makes a single pass, and its
output also feeds the text-layer and page-count gates in `triage`, so no file is
opened twice for any reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..pdf.text import detect_doi, extract_pdf, pdf_shape
from ..stems import strip_diacritics
from .parse import ExportItem

#: Pages read per PDF. Enough for a title, a DOI and a representative
#: characters-per-page figure; small enough to keep a 500-file pass to minutes.
INDEX_PAGES = 3

#: Token-set F1 above which a title match is trusted outright.
TITLE_ACCEPT = 0.75

#: Below this, a title match isn't reported at all — it's noise.
TITLE_FLOOR = 0.55

#: Characters of first-page text compared against a record's title. A full page
#: dilutes the token set with abstract and affiliations; the title lives at the top.
_TITLE_WINDOW = 700

#: Filenames that announce themselves as supplementary material.
_SUPP_RE = re.compile(r"(?i)(supp|_si\b|\bsi[-_ ]|supporting|appendix|extended[-_ ]data)")

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class PdfFacts:
    """Everything one pass over a PDF yields. Shared by pairing and triage."""

    path: Path
    page_count: int | None = None
    text: str = ""
    doi: str | None = None
    title_tokens: frozenset[str] = frozenset()

    @property
    def chars_per_page(self) -> float | None:
        """Characters of extractable text per page, or None if unreadable.

        The single most important number in the import: a scanned PDF extracts
        to almost nothing, ingest logs a thin-extraction warning nobody reads,
        and the resulting page passes every downstream gate on a grounding
        corpus that isn't there.
        """
        if self.page_count is None:
            return None
        pages = min(self.page_count, INDEX_PAGES) or 1
        return len(self.text) / pages


@dataclass
class Pairing:
    item: ExportItem
    primary: Path | None = None
    supplementary: list[Path] = field(default_factory=list)
    rung: str | None = None
    confidence: float = 0.0
    candidates: list[tuple[Path, float]] = field(default_factory=list)


def _tokens(text: str) -> frozenset[str]:
    """Content tokens for title comparison.

    Folded through `stems.strip_diacritics` so the comparison sees the same
    string stem derivation will — a Unicode-dash title must tokenize the same
    way its ASCII-hyphen twin does, or the two spellings of one paper score as
    different papers.
    """
    return frozenset(w for w in _WORD_RE.findall(strip_diacritics(text or "").lower())
                     if len(w) > 2)


#: A title with fewer content tokens than this is not matched by coverage at
#: all. Coverage of a 1-2 token title is satisfied by chance — "Evo 2" reduces
#: to {evo} — and a chance match assigns the wrong PDF to a record, which is
#: the most expensive error this module can make.
_MIN_TITLE_TOKENS = 3


def _coverage(title: frozenset[str], text: frozenset[str]) -> float:
    """How much of the title appears in the PDF's opening text. Asymmetric.

    Symmetric F1 is the intuitive choice here and it is wrong: a first page
    always carries authors, affiliations and an abstract on top of the title,
    so the title's tokens are a small fraction of the window's. An *exact*
    title match against a real page scored 0.615 by F1 — below any threshold
    worth setting — because the extra page content counted against it.

    The question this rung actually asks is "does this PDF contain this title",
    which is coverage of the title, and an exact match then scores 1.0 as it
    should. The risk coverage carries instead is a short title being contained
    by accident, which `_MIN_TITLE_TOKENS` handles.
    """
    if len(title) < _MIN_TITLE_TOKENS or not text:
        return 0.0
    return len(title & text) / len(title)


def build_pdf_index(pdf_root: Path) -> list[PdfFacts]:
    """One extraction pass over every PDF under `pdf_root`.

    Failures are recorded, not raised: an encrypted or truncated PDF becomes a
    `PdfFacts` with `page_count=None`, which triage reports as `pdf-unreadable`.
    A whole import should not stop because one file is broken.
    """
    facts = []
    for path in sorted(pdf_root.rglob("*.pdf")):
        if not path.is_file():
            continue
        f = PdfFacts(path=path)
        try:
            f.page_count, _ = pdf_shape(path)
            if f.page_count is not None:
                text, meta = extract_pdf(path, max_pages=INDEX_PAGES)
                f.text = text
                f.doi = detect_doi(meta, text)
                f.title_tokens = _tokens(text[:_TITLE_WINDOW])
        except Exception:
            f.page_count = None
        facts.append(f)
    return facts


def _resolve_declared(raw: str, pdf_root: Path, export_dir: Path,
                      by_name: dict[str, list[Path]]) -> Path | None:
    """A declared path → a file that exists.

    Tried against `pdf_root`, then the export file's own directory, then by
    basename. The basename fallback is not laziness: sync clients relocate
    trees constantly, and an export written before a move names a directory
    that no longer exists while the file itself is right there.
    """
    candidate = Path(raw.strip()).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    for base in (pdf_root, export_dir):
        p = (base / candidate).resolve()
        if p.is_file():
            return p
    hits = by_name.get(candidate.name.lower())
    return hits[0] if hits and len(hits) == 1 else None


def pair_items(items: list[ExportItem], facts: list[PdfFacts], *,
               pdf_root: Path, export_dir: Path) -> tuple[list[Pairing], list[PdfFacts]]:
    """`(pairings, unclaimed)` — one `Pairing` per item, plus leftover PDFs.

    Rungs run as three passes over all items rather than three attempts per
    item, so a confident DOI match always wins a file over a merely plausible
    title match on another record, regardless of record order.
    """
    pairings = {id(i): Pairing(item=i) for i in items}
    taken: set[Path] = set()

    by_name: dict[str, list[Path]] = {}
    for f in facts:
        by_name.setdefault(f.path.name.lower(), []).append(f.path)
    by_doi: dict[str, list[PdfFacts]] = {}
    for f in facts:
        if f.doi:
            by_doi.setdefault(f.doi.lower(), []).append(f)

    # Rung 1 — declared paths.
    for item in items:
        p = pairings[id(item)]
        for raw in item.declared_files:
            hit = _resolve_declared(raw, pdf_root, export_dir, by_name)
            if hit and hit not in taken:
                taken.add(hit)
                if p.primary is None:
                    p.primary, p.rung, p.confidence = hit, "declared", 1.0
                else:
                    p.supplementary.append(hit)

    # Rung 2 — DOI printed in the PDF.
    for item in items:
        p = pairings[id(item)]
        if p.primary is not None or not item.has_usable_doi:
            continue
        for f in by_doi.get(item.doi.lower(), []):
            if f.path not in taken:
                taken.add(f.path)
                p.primary, p.rung, p.confidence = f.path, "doi", 0.9
                break

    # Rung 3 — title similarity, best-match-first across all remaining pairs so
    # a strong match is never lost to a weaker one that was merely earlier.
    remaining = [f for f in facts if f.path not in taken and f.title_tokens]
    scored: list[tuple[float, ExportItem, PdfFacts]] = []
    for item in items:
        if pairings[id(item)].primary is not None or not item.title:
            continue
        it = _tokens(item.title)
        for f in remaining:
            score = _coverage(it, f.title_tokens)
            if score >= TITLE_FLOOR:
                scored.append((score, item, f))
    for score, item, f in sorted(scored, key=lambda t: -t[0]):
        p = pairings[id(item)]
        if p.primary is not None or f.path in taken:
            continue
        p.candidates.append((f.path, round(score, 3)))
        if score >= TITLE_FLOOR:
            taken.add(f.path)
            p.primary, p.rung, p.confidence = f.path, "title", round(score, 3)

    _attach_supplementary(pairings.values(), facts, taken)
    return list(pairings.values()), [f for f in facts if f.path not in taken]


def _attach_supplementary(pairings, facts: list[PdfFacts], taken: set[Path]) -> None:
    """Claim supplementary-looking siblings of a paired PDF.

    This is where the export beats a flat folder of files: the manager knows two
    files belong to one item. Restricted to same-directory siblings whose name
    announces them as supplementary, because the alternative — guessing from
    content — produces two paper pages for one paper, which is worse than
    missing an appendix.
    """
    by_dir: dict[Path, list[PdfFacts]] = {}
    for f in facts:
        by_dir.setdefault(f.path.parent, []).append(f)
    for p in pairings:
        if p.primary is None:
            continue
        for f in by_dir.get(p.primary.parent, []):
            if f.path in taken or not _SUPP_RE.search(f.path.stem):
                continue
            if f.path.stem.lower().startswith(p.primary.stem.lower()[:12]):
                taken.add(f.path)
                p.supplementary.append(f.path)
