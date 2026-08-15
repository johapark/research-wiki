"""PDF text extraction, DOI detection, and reference-list DOI harvesting (pypdfium2-based).

Migrated from pypdf to pypdfium2 on 2026-05-28 after a 122-paper benchmark showed
pypdfium2 is ~22× faster (1.07s vs 23.5s on a 6-paper representative sample) at
equivalent text quality (mean +47 chars per first page across the wiki). PDFium
is the C++ engine Chrome ships for PDF rendering — battle-tested at scale,
permissively licensed (BSD-3-Clause + Apache-2.0). pypdfium2 is the Python
binding distributed as pre-built wheels.

**Failure modes are split on purpose, so callers know which contract they get.**
`extract_pdf` and `extract_pdf_page_texts` *raise* (`PdfiumError` on an
unparseable file, `FileNotFoundError` on a missing one): they are the ingest and
grading paths, where a PDF that cannot be read means the operation cannot
proceed and a silent empty string would land a page grounded in nothing. The
three probing functions — `pdf_shape`, `extract_ref_dois`, `cites_reference` —
return an empty sentinel instead (`(None, "")`, `[]`, `False`), because each
answers an optional question whose "don't know" is a legitimate answer and whose
callers must not be aborted by one bad file in a corpus walk.

API note: pypdfium2 only exposes the standard PDF info dictionary (Title,
Author, Subject, Keywords, Creator, Producer, CreationDate, ModDate) as bare
keys, not custom XMP fields like Springer's `/doi`. We normalize the keys to
slash-prefixed form (`/Title`, `/Subject`, ...) for backward compatibility
with the original `detect_doi` lookup chain. The `/doi` custom-XMP path is
gone; `detect_doi` falls through to `/Subject` (which Springer/Nature reliably
populate as `"Nature Genetics, doi:10.xxxx/yyyy"`) and first-page text regex.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pypdfium2

from .repair import _INTERIOR_CTRL_RE, repair_text

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

# arXiv IDs in references — both bare-form `arXiv:YYMM.NNNNN(vN)?` and
# URL-form `arxiv.org/abs/YYMM.NNNNN(vN)?`. Captures the numeric ID only;
# version suffix is dropped because the canonical arXiv DOI form
# `10.48550/arXiv.YYMM.NNNNN` doesn't carry the version.
_ARXIV_ID_RE = re.compile(
    r"(?:arxiv[:\s]*|arxiv\.org/abs/)\s*(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)

# SSRN abstract URLs — `ssrn.com/abstract=NNNNNNN`. SSRN preprints typeset
# this URL in the page footer instead of the canonical DOI `10.2139/ssrn.NNNN`
# form, so the bare-DOI scanner misses them. Recognized here separately and
# converted to canonical DOI form. Triggered the Jang 2025 (sub-1% somatic
# WGS) recovery flow this would have prevented.
_SSRN_ABSTRACT_RE = re.compile(r"ssrn\.com/abstract=(\d{4,12})", re.IGNORECASE)

# Where a paper's reference list starts. Widened after `kim-2019-spcas9-…`
# turned out to have no match at all: Science/AAAS prints `REFERENCES AND NOTES`,
# which the original end-anchored form rejected, so `extract_ref_dois` silently
# fell back to `text[3000:]` on the whole Science family.
#
# Measured on a 90-PDF corpus sample: the original found a heading in 74, this
# finds 80. The six gained are `REFERENCES AND NOTES`, singular `Reference`, and
# page-number-glued forms (`78REFERENCES`, `9References`) where extraction ran a
# running page number into the heading — hence the leading `\d+` allowance, which
# stays safe because the whole line must still end after the heading word.
REFS_HEADING_RE = re.compile(
    r"(?im)^\s*(?:\d+\.?\s*)?(references?|bibliography|works\s+cited|literature\s+cited)"
    r"(?:\s+(?:and|&)\s+notes?)?\s*:?\s*$"
)

# Trailing characters to strip from a DOI captured by DOI_RE — publishers often
# typeset DOIs followed by a period, comma, or closing paren.
_DOI_TRIM_RE = re.compile(r"[.,;:)\]]+$")

# Short alpha suffixes that paste onto DOIs without whitespace in some PDFs
# (e.g. a running footer `https://doi.org/10.xxxx/yyydoi: bioRxiv preprint`
# causes the regex to greedy-capture `...yyydoi`). Stripped after the main
# trim so we don't corrupt legitimate DOIs ending in letters.
_DOI_SUFFIX_NOISE = ("doi", "author", "preprint", "https")


def _extract_all_pages(pdf: pypdfium2.PdfDocument, max_pages: int | None) -> list[str]:
    """Iterate the document with explicit per-page resource cleanup.

    pypdfium2's PdfPage and PdfTextPage hold native handles that the binding
    cleans up on garbage-collection, but explicit close() is recommended for
    deterministic release — especially when iterating large documents from
    long-running processes (the agent runner, the grader's chunk-cache build).
    """
    n = len(pdf)
    upper = n if max_pages is None else min(n, max_pages)
    parts: list[str] = []
    for i in range(upper):
        page = pdf[i]
        try:
            tp = page.get_textpage()
            try:
                parts.append(tp.get_text_bounded() or "")
            finally:
                tp.close()
        finally:
            page.close()
    return parts


def pdf_shape(path: Path) -> tuple[int | None, str]:
    """Return `(page_count, first_page_text)` — the cheap structural read.

    Deliberately *not* folded into `extract_pdf`: callers of that function want
    concatenated body text and don't care where the page boundaries were, while
    the commentary guard (`agents.commentary`) needs exactly two facts — how
    many pages the document has, and what the FIRST page alone says. Slicing
    the first N chars off `extract_pdf`'s joined output is not a substitute: on
    a 30-page article that slice runs into the introduction, where a sentence
    like "as a recent News & Views argued" would trip a section-label match
    that only belongs in a masthead.

    Only page 0's textpage is materialized, so this is O(1) in document length.
    Ligature repair is skipped (the dictionary pass costs ~1.7 MB resident and
    the guard's patterns are plain ASCII words); interior control bytes are
    stripped so a soft-hyphenated `News & Vi\\x02ews` still matches.

    Returns `(None, "")` when the file is missing or PDFium can't parse it —
    the guard treats unknown page count as "no structural signal" rather than
    failing the ingest.
    """
    try:
        pdf = pypdfium2.PdfDocument(str(path))
    except Exception:
        return None, ""
    try:
        n = len(pdf)
        first = ""
        if n:
            page = pdf[0]
            try:
                tp = page.get_textpage()
                try:
                    first = tp.get_text_bounded() or ""
                finally:
                    tp.close()
            finally:
                page.close()
    except Exception:
        return None, ""
    finally:
        pdf.close()
    return n, _INTERIOR_CTRL_RE.sub("", first)


def _slash_prefix_metadata(md: dict[str, Any]) -> dict[str, Any]:
    """Normalize pypdfium2's bare-key metadata to slash-prefixed form.

    pypdf returned keys like `/Title`, `/Subject`, `/doi` (the last being a
    Springer-specific XMP field). pypdfium2 returns bare keys like `Title`,
    `Subject` from the standard info dict — no XMP custom fields. We slash-
    prefix what pypdfium2 gives us so the downstream `detect_doi` lookup
    chain (which iterates `/doi`, `/DOI`, `/Subject`) still works for the
    `/Subject` fallback. The `/doi` custom-XMP path is unreachable post-
    migration, but `/Subject` carries the DOI text for Springer/Nature/etc.
    """
    return {f"/{k}": v for k, v in (md or {}).items()}


def extract_pdf(path: Path, max_pages: int = 20) -> tuple[str, dict[str, Any]]:
    """Return (concatenated first-N-pages text, slash-prefixed metadata dict).

    Output is post-processed by `repair.repair_text` to fix the common ligature-
    glyph dropout pattern in scientific PDFs (see module-level commentary).
    """
    pdf = pypdfium2.PdfDocument(str(path))
    try:
        meta = _slash_prefix_metadata(pdf.get_metadata_dict() or {})
        parts = _extract_all_pages(pdf, max_pages=max_pages)
    finally:
        pdf.close()
    return repair_text("\n\n".join(parts)), meta


PAGE_SEPARATOR = "\n\n"


def extract_pdf_page_texts(path: Path, max_pages: int = 20) -> list[str]:
    """Per-page text, ligature-repaired, one string per page.

    **Invariant:** `PAGE_SEPARATOR.join(extract_pdf_page_texts(p, n))` equals
    `extract_pdf(p, n)[0]` byte for byte, which is what lets a caller compute
    the page a character offset falls on without re-deriving the text. Pinned
    by `tests/test_chunk_provenance.py`.

    Repair is applied per page rather than to the join. The two agree because
    `repair_text`'s patterns are word-shaped and no word survives a page
    break — verified across 15 corpus papers when this was introduced — and
    per-page is the order that keeps page lengths meaningful *after* repair,
    which is the thing offsets are measured against.
    """
    pdf = pypdfium2.PdfDocument(str(path))
    try:
        parts = _extract_all_pages(pdf, max_pages=max_pages)
    finally:
        pdf.close()
    return [repair_text(part) for part in parts]


def page_offsets(page_texts: list[str]) -> list[int]:
    """Start offset of each page within `PAGE_SEPARATOR.join(page_texts)`."""
    offsets: list[int] = []
    cursor = 0
    for i, text in enumerate(page_texts):
        offsets.append(cursor)
        cursor += len(text) + (len(PAGE_SEPARATOR) if i < len(page_texts) - 1 else 0)
    return offsets


def page_for_offset(
    offsets: list[int], offset: int, total_len: int | None = None
) -> int | None:
    """1-based page number containing `offset`. None when out of range.

    `offsets` is ascending, so this is a right-bisect: the page is the last one
    whose start is <= the offset.

    Page starts alone cannot say where the *last* page ends, so an offset past
    the end of the text is indistinguishable from one on the final page unless
    the caller says how long the text is. Pass `total_len` (the length of the
    joined text these offsets index into) and an offset at or beyond it returns
    None, as the docstring has always promised; omit it and the old behaviour
    stands, since a caller that joined the pages itself cannot produce an
    out-of-range offset anyway.
    """
    if not offsets or offset < 0:
        return None
    if total_len is not None and offset >= total_len:
        return None
    lo, hi = 0, len(offsets)
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= offset:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo >= 1 else None


def _trim_doi_noise(doi: str) -> str:
    """Strip trailing punctuation + alpha-suffix noise from a DOI capture.

    Originally written for pypdf, where text extraction sometimes ran lines
    together so a footer like `https://doi.org/10.xxxx/yyydoi: bioRxiv preprint`
    produced a greedy DOI capture of `10.xxxx/yyydoi`. The Boltz-2 ingest hit
    this exact mode (DOI ended `...659707doi:` and the trailing `doi:` prevented
    S2 lookup, silently breaking the metadata pipeline). pypdfium2's text is
    cleaner but the trim is still needed for the same edge cases — kept as
    a defense in depth across both `detect_doi` and `extract_ref_dois`.
    """
    out = _DOI_TRIM_RE.sub("", doi)
    for _ in range(2):
        for noise in _DOI_SUFFIX_NOISE:
            if out.lower().endswith(noise) and len(out) > len(noise) + 10:
                out = out[:-len(noise)]
        out = _DOI_TRIM_RE.sub("", out)
    return out


def detect_doi(pdf_meta: dict[str, Any], pdf_text: str) -> str | None:
    """Try metadata first (Springer `/Subject` carries the DOI as text), then
    first-page DOI regex, then arXiv-ID fallback for preprints whose header
    carries `arXiv:NNNN.NNNNN` instead of a 10.xxxx/ DOI.

    `/doi` and `/DOI` keys are checked for backward compatibility but won't
    populate post-pypdfium2 — the binding doesn't surface custom XMP fields.
    The `/Subject` path covers Springer/Nature, which is the bulk of where
    `/doi` used to hit anyway.
    """
    for key in ("/doi", "/DOI", "/Subject"):
        v = pdf_meta.get(key) or ""
        if isinstance(v, str):
            m = DOI_RE.search(v)
            if m:
                return _trim_doi_noise(m.group(0).strip())
    m = DOI_RE.search(pdf_text[:4000])
    if m:
        return _trim_doi_noise(m.group(0).strip())
    # arXiv preprints often print the bare ID in the upper-right of the
    # first page rather than a real DOI. Convert to canonical arXiv-DOI form.
    am = _ARXIV_ID_RE.search(pdf_text[:4000])
    if am:
        return f"10.48550/arXiv.{am.group(1)}"
    return None


def find_url_doi_candidates(
    pdf_text: str, *, stop_at_references: bool = True
) -> list[tuple[str, str]]:
    """Hunt for DOI candidates encoded as preprint-server URLs (not yet in
    `10.X/Y` form). Use as a fallback **after** `detect_doi` returns None.

    Returns a list of `(provenance, doi)` tuples in PDF order, deduplicated
    on the DOI value. Provenance is a short tag (`"ssrn-url"`,
    `"arxiv-url"`) for logging — surfaces which pattern fired.

    This is the complement to `detect_doi`: that function handles DOIs
    already in canonical form, while this one handles preprint-server
    URL forms whose mapping to a DOI is mechanical:

      - `ssrn.com/abstract=NNNN`  → `10.2139/ssrn.NNNN`
      - `arxiv.org/abs/YYMM.NNNNN` → `10.48550/arXiv.YYMM.NNNNN`

    Caller is expected to validate each candidate against an authoritative
    source (Crossref) before adopting — these URL → DOI mappings are
    **mechanical**, not semantic, so the URL itself doesn't prove the DOI
    actually resolves. A typo'd SSRN URL would map to a non-existent DOI.

    Scans more text than `detect_doi` (no 4000-char cap) because page
    footers carrying the SSRN URL repeat across all pages and the first
    one may sit past the abstract block.

    `stop_at_references` truncates the scan at the References heading, and is
    the default because **every URL past that heading belongs to a different
    paper**. Without it, `kim-2019-spcas9-activity-prediction-by-deepspcas9`
    yielded `arxiv.org/abs/1412.6980` from its reference list — the Adam
    optimizer, cited as the training optimizer — as a candidate for the paper's
    own DOI. The body pages are still scanned in full, so the repeated-SSRN-
    footer case this function exists for is unaffected: that footer appears on
    page 1 onward, long before any reference list. Pass False to restore the
    old whole-document behaviour.
    """
    if stop_at_references:
        m = REFS_HEADING_RE.search(pdf_text)
        if m:
            pdf_text = pdf_text[:m.start()]

    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    for m in _SSRN_ABSTRACT_RE.finditer(pdf_text):
        doi = f"10.2139/ssrn.{m.group(1)}"
        if doi not in seen:
            seen.add(doi)
            out.append(("ssrn-url", doi))

    for m in _ARXIV_ID_RE.finditer(pdf_text):
        doi = f"10.48550/arXiv.{m.group(1)}"
        if doi not in seen:
            seen.add(doi)
            out.append(("arxiv-url", doi))

    return out


def cites_reference(path: Path, surname: str, year: int | str, *, window: int = 300) -> bool:
    """Does this PDF's reference list contain an entry by `surname` from `year`?

    Answers the one question that separates "these two papers are topically
    related" from "this paper cites that one" — cheaply, with no LLM and no
    network, using text we already have to read.

    It exists because the alternative is a fabricated claim. `propose_crosslinks`
    labels every unverified pairing `topical`, and `promote` then writes that
    phrasing onto the target page; where the citation is real, the wiki
    permanently understates a relationship it could have proven, and where a
    later pass "upgrades" the note by guessing, it asserts support that was never
    checked (13 such bullets were found wrong and corrected on 2026-08-05 —
    including three claiming a 2019 paper cited work from 2023).

    Match rule: the normalised surname appears in the reference section with the
    year within `window` characters after it. That co-occurrence window is what
    keeps "Doench 2016 (rule set 2)" in the body from satisfying a query for
    Doench 2014 — reference entries put author and year close together, prose
    discussion generally does not.

    Deliberately not DOI-based, though `extract_ref_dois` exists: Science and
    many other publishers print no DOIs in their reference lists, so on the very
    PDF that motivated this the DOI route returns nothing.

    Returns False on unreadable PDFs or empty inputs — never raises, and never
    guesses. A False means "not proven", not "proven absent", so callers should
    fall back to the weaker phrasing rather than asserting a non-citation.
    """
    if not surname or year in (None, ""):
        return False
    try:
        pdf = pypdfium2.PdfDocument(str(path))
    except Exception:
        return False
    try:
        chunks = _extract_all_pages(pdf, max_pages=None)
    finally:
        pdf.close()
    text = "\n\n".join(chunks)

    m = REFS_HEADING_RE.search(text)
    # Same fallback `extract_ref_dois` uses: past the front matter, which skips
    # the paper's own title/abstract without needing the heading to be found.
    section = text[m.end():] if m else text[3000:]
    section = _fold_ascii(re.sub(r"\s+", " ", section)).lower()

    needle = _fold_ascii(surname).lower()
    if not needle:
        return False
    y = str(year)
    for mm in re.finditer(re.escape(needle), section):
        if y in section[mm.start():mm.start() + window]:
            return True
    return False


def _fold_ascii(s: str) -> str:
    """NFKD-fold to ASCII so `García` matches a reference list's `Garcia`."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def extract_ref_dois(path: Path, own_doi: str | None = None) -> list[str]:
    """Harvest unique DOIs from the References section of a PDF.

    Reads all pages (no page cap, unlike `extract_pdf`) because references
    typically sit past the main-text pages. Needed as a fallback when the
    Semantic Scholar `/references` endpoint elides references — Springer
    Nature and bioRxiv both do this for most post-2024 papers.

    If the References heading isn't found, falls back to scanning all text
    past the first ~3k characters (which skips the paper's own DOI on page 1).

    `own_doi` is excluded from the result if supplied.
    """
    try:
        pdf = pypdfium2.PdfDocument(str(path))
    except Exception:
        return []

    try:
        chunks = _extract_all_pages(pdf, max_pages=None)
    finally:
        pdf.close()
    text = "\n\n".join(chunks)

    m = REFS_HEADING_RE.search(text)
    section = text[m.end():] if m else text[3000:]

    own_doi_l = (own_doi or "").lower().strip()
    dois: set[str] = set()
    for mm in DOI_RE.finditer(section):
        doi = _trim_doi_noise(mm.group(0)).lower()
        if doi and doi != own_doi_l:
            dois.add(doi)

    # arXiv IDs → canonical arXiv-DOI form. S2 indexes a fair fraction of
    # these, and Crossref doesn't carry arXiv references at all, so the
    # PDF text is the only place to find them for preprint-heavy bibliographies.
    for am in _ARXIV_ID_RE.finditer(section):
        arxiv_id = am.group(1)
        arxiv_doi = f"10.48550/arxiv.{arxiv_id}"
        if arxiv_doi != own_doi_l:
            dois.add(arxiv_doi)

    return sorted(dois)


def dois_as_s2_refs(dois: list[str]) -> list[dict]:
    """Wrap bare DOIs in the S2 reference-list shape, so they can pass through
    the same `intersect_crosslinks` pipeline that consumes S2 results."""
    return [
        {"externalIds": {"DOI": d}, "title": "(from PDF references)", "year": None}
        for d in dois
    ]
