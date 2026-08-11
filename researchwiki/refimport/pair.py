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

#: How far the best-scoring record must beat the runner-up for a title pairing
#: to count as unambiguous. A near-tie means the score came from vocabulary the
#: two records share rather than from identity.
#:
#: Measured against 313 DOI-confirmed pairs from a real library (the DOI gives
#: ground truth, so title matching can be scored against it):
#:
#:     margin   correct   wrong   precision
#:       0.00       282       6       0.979
#:       0.05       278       0       1.000
#:       0.15       267       0       1.000
#:
#: 0.05 removes every wrong pairing for four correct ones, and those four are
#: not lost — they land in `review` with their candidates listed, rather than
#: silently attaching the wrong PDF to a record.
TITLE_MARGIN = 0.05

#: Characters of leading text compared against a record's title.
#:
#: Kept deliberately narrow, against the intuition that a wider window finds
#: more. Widening it was measured on the same 313 ground-truth pairs and is
#: strictly worse — body text covers a generic title by chance, and wrong
#: records start outscoring right ones:
#:
#:     window   correct   wrong   precision
#:        700       282       6       0.979
#:       1500       288      16       0.947
#:       3000       274      37       0.881
#:    3 pages       201     110       0.646
#:
#: The failure this *doesn't* fix is a masthead-first page (`nature
#: biotechnology VOLUME 36 …`) pushing the title past 700 chars. Those records
#: are better recovered by their DOI, which is why the DOI rung runs first.
_TITLE_WINDOW = 700

#: Filenames that announce themselves as supplementary material.
_SUPP_RE = re.compile(r"(?i)(supp|_si\b|\bsi[-_ ]|supporting|appendix|extended[-_ ]data)")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _looks_supplementary(path: Path) -> bool:
    return bool(_SUPP_RE.search(path.stem))


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
    #: Best score any *other* record achieved against this same PDF. Only
    #: meaningful for the title rung. Triage compares it against
    #: `TITLE_MARGIN` — a near-tie is an ambiguous match, not a confident one.
    rival: float = 0.0
    candidates: list[tuple[Path, float]] = field(default_factory=list)

    @property
    def margin(self) -> float:
        return round(self.confidence - self.rival, 3)


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


def _canonical(path: Path) -> Path:
    """One spelling per file.

    Everything in this module is keyed on a `Path` — `taken`, `by_name`,
    `by_dir`, `unclaimed` — and `triage` looks each pairing's primary up in a
    dict keyed on `PdfFacts.path`. `Path` equality is string equality, not file
    identity, so two spellings of one file escape all of them at once: the file
    is handed to a record *and* reported unclaimed, a second record re-pairs it,
    its supplementary siblings are not found, and triage calls it unreadable.

    `resolve()` rather than `absolute()` because the spellings that actually
    collide differ by more than a leading slash: a relative root
    (`import inspect lib.ris pdfs`) and a symlinked library, where the export
    names the location the sync client wrote and the user names the link.
    Falls back to `absolute()` on a symlink loop, which at least fixes the
    relative case.
    """
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def build_pdf_index(pdf_root: Path) -> list[PdfFacts]:
    """One extraction pass over every PDF under `pdf_root`.

    Failures are recorded, not raised: an encrypted or truncated PDF becomes a
    `PdfFacts` with `page_count=None`, which triage reports as `pdf-unreadable`.
    A whole import should not stop because one file is broken.

    The root is canonicalized first, so the spelling published here — which every
    other structure keys on — does not depend on how the caller spelled it.
    """
    facts = []
    for path in sorted(_canonical(pdf_root).rglob("*.pdf")):
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
                      by_name: dict[str, list[Path]],
                      by_canonical: dict[Path, Path]) -> Path | None:
    """A declared path → a file that exists, spelled the way the index spells it.

    Tried against `pdf_root`, then the export file's own directory, then by
    basename. The basename fallback is not laziness: sync clients relocate
    trees constantly, and an export written before a move names a directory
    that no longer exists while the file itself is right there.

    Each of those three rungs finds the file a different way — verbatim,
    joined, by basename — and returning whichever spelling the winning rung
    produced is what let one file sit in `taken` under a name nothing else in
    this module could match. Hits are mapped back through `by_canonical` to the
    spelling `build_pdf_index` published; see `_canonical`. A declared file that
    is not in the index at all still resolves, canonicalized, since `export_dir`
    is legitimately allowed to sit outside `pdf_root`.
    """
    candidate = Path(raw.strip()).expanduser()
    found: Path | None = None
    if candidate.is_absolute() and candidate.is_file():
        found = candidate
    else:
        for base in (pdf_root, export_dir):
            p = base / candidate
            if p.is_file():
                found = p
                break
    if found is not None:
        canon = _canonical(found)
        return by_canonical.get(canon, canon)
    hits = by_name.get(candidate.name.lower())
    return hits[0] if hits and len(hits) == 1 else None


def pair_items(items: list[ExportItem], facts: list[PdfFacts], *,
               pdf_root: Path, export_dir: Path) -> tuple[list[Pairing], list[PdfFacts]]:
    """`(pairings, unclaimed)` — one `Pairing` per item, plus leftover PDFs.

    Rungs run as three passes over all items rather than three attempts per
    item, so a confident DOI match always wins a file over a merely plausible
    title match on another record, regardless of record order.
    """
    pdf_root, export_dir = _canonical(pdf_root), _canonical(export_dir)
    pairings = {id(i): Pairing(item=i) for i in items}
    taken: set[Path] = set()

    by_name: dict[str, list[Path]] = {}
    for f in facts:
        by_name.setdefault(f.path.name.lower(), []).append(f.path)
    # Whatever spelling a declared path resolves to → the spelling the index
    # published for that same file.
    by_canonical: dict[Path, Path] = {_canonical(f.path): f.path for f in facts}
    by_doi: dict[str, list[PdfFacts]] = {}
    for f in facts:
        if f.doi:
            by_doi.setdefault(f.doi.lower(), []).append(f)

    # Rung 1 — declared paths.
    for item in items:
        p = pairings[id(item)]
        for raw in item.declared_files:
            hit = _resolve_declared(raw, pdf_root, export_dir, by_name,
                                    by_canonical)
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
            if f.path not in taken and not _looks_supplementary(f.path):
                taken.add(f.path)
                p.primary, p.rung, p.confidence = f.path, "doi", 0.9
                break

    # Rung 3 — title similarity, best-match-first across all remaining pairs so
    # a strong match is never lost to a weaker one that was merely earlier.
    # Supplementary files are excluded from *primary* candidacy on the content
    # rungs. They carry the paper's title and often its DOI, so they score as
    # well as the paper itself — and `sorted(rglob)` can hand them the file
    # first ("Title - Supplementary.pdf" sorts before "Title.pdf", space < dot).
    # The paper then lands in `unclaimed` and cannot be recovered, because
    # `_attach_supplementary` matches on the *primary's* name. The page would be
    # authored from the appendix. Rung 1 is exempt: a declared path is the
    # export naming that exact file on purpose.
    remaining = [f for f in facts if f.path not in taken and f.title_tokens
                 and not _looks_supplementary(f.path)]
    scored: list[tuple[float, ExportItem, PdfFacts]] = []
    # Scores per PDF, carrying *which record* produced each. A bare score list
    # let a record that was not the top scorer for a PDF read its own score back
    # as the rival, reporting margin 0.
    #
    # Worth being precise: that never changed a verdict. The old list included
    # the winner's own score, so margin came out as exactly 0 — still under
    # `TITLE_MARGIN`, still flagged. What was wrong was the number published in
    # the manifest and `--json`, which a human reads to judge whether the flag
    # is fair; "rival 0.889" when the real contender scored 1.0 understates the
    # contest.
    per_pdf: dict[Path, list[tuple[float, int]]] = {}
    for item in items:
        if pairings[id(item)].primary is not None or not item.title:
            continue
        it = _tokens(item.title)
        for f in remaining:
            score = _coverage(it, f.title_tokens)
            if score <= 0:
                continue
            per_pdf.setdefault(f.path, []).append((score, id(item)))
            if score >= TITLE_FLOOR:
                scored.append((score, item, f))

    for score, item, f in sorted(scored, key=lambda t: -t[0]):
        p = pairings[id(item)]
        # Every PDF this record could plausibly have taken, winner or not — the
        # report promises "each names its candidate PDFs", and a record that
        # loses every candidate to a higher scorer would otherwise carry an
        # empty list and be reported as `no-pdf`, sending the user off to
        # download a file already sitting in their library.
        p.candidates.append((f.path, round(score, 3)))
        if p.primary is not None or f.path in taken:
            continue
        taken.add(f.path)
        rivals = [s for s, owner in per_pdf.get(f.path, []) if owner != id(item)]
        p.primary, p.rung, p.confidence = f.path, "title", round(score, 3)
        p.rival = round(max(rivals), 3) if rivals else 0.0

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
