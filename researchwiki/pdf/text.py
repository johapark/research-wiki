"""PDF text extraction, DOI detection, and reference-list DOI harvesting (pypdfium2-based).

Migrated from pypdf to pypdfium2 on 2026-05-28 after a 122-paper benchmark showed
pypdfium2 is ~22× faster (1.07s vs 23.5s on a 6-paper representative sample) at
equivalent text quality (mean +47 chars per first page across the wiki). PDFium
is the C++ engine Chrome ships for PDF rendering — battle-tested at scale,
permissively licensed (BSD-3-Clause + Apache-2.0). pypdfium2 is the Python
binding distributed as pre-built wheels.

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
from pathlib import Path
from typing import Any

import pypdfium2

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

# References / Bibliography heading at the start of a line.
REFS_HEADING_RE = re.compile(
    r"(?im)^\s*(?:\d+\.\s*)?(references|bibliography|works\s+cited|literature\s+cited)\s*$"
)

# Trailing characters to strip from a DOI captured by DOI_RE — publishers often
# typeset DOIs followed by a period, comma, or closing paren.
_DOI_TRIM_RE = re.compile(r"[.,;:)\]]+$")

# Short alpha suffixes that paste onto DOIs without whitespace in some PDFs
# (e.g. a running footer `https://doi.org/10.xxxx/yyydoi: bioRxiv preprint`
# causes the regex to greedy-capture `...yyydoi`). Stripped after the main
# trim so we don't corrupt legitimate DOIs ending in letters.
_DOI_SUFFIX_NOISE = ("doi", "author", "preprint", "https")


# --- PDF-extraction noise repair ---------------------------------------------
# Two failure modes show up in pypdfium2 output:
#
# 1. Ligature dropouts. Some PDFs ship Type 1 / Type 3 fonts that include
#    ligature glyphs (`fi`, `ff`, `ffi`, `ffl`, `fl`, `ft`, `tt`) but no
#    `ToUnicode` CMap, or a CMap that omits the ligature CIDs. PDFium falls
#    back to the raw CID, which usually lands in the C1 control range
#    (U+0080–U+009F). The result is words like `effective`, `efficient`,
#    `difference`, `off-target` coming out with a single non-printing byte
#    where the ligature used to be:
#
#        'e<\x82>ective'  →  effective
#        'di<\x85>erence' →  difference
#        'o<\x97>-target' →  off-target
#
#    The byte value isn't stable — different fonts use different CIDs for the
#    same ligature, sometimes within one paper. What's stable is the
#    *surrounding letters*: only one ligature substitution yields a valid
#    English word. We exploit that with a dictionary lookup over a bundled
#    30K-word frequency-sorted list.
#
#    Rarer cousin: the ligature is dropped entirely with no replacement byte,
#    producing words like `ne-grained` (was `fine-grained`). Same dictionary
#    trick works — try inserting each ligature at each gap, accept the unique
#    fix that resolves to a known word.
#
# 2. Soft-hyphen / line-break-hyphenation artifacts. PDFs hyphenate at line
#    ends; some encodings preserve the soft hyphen as a C0 control char
#    (most commonly U+0002 STX, also U+0012, U+001E, U+001F). The corpus
#    shows 7,500+ occurrences of these between letter clusters — by far the
#    largest class of noise. Stripped unconditionally between two letters.
#
# The dictionary is loaded lazily on first repair call and cached in a
# module-global, so paths that don't extract PDFs (`status`, `search`,
# `claims`, etc.) don't pay the ~1.7 MB resident cost.

_C1 = r"[\x80-\x9f]"
_C1_RE = re.compile(_C1)
# A "word" candidate for repair: ASCII letters + apostrophes + hyphens, with
# at least one C1 control byte embedded. We skip words without C1 since the
# dictionary check on a clean word would just say "yes" and do nothing.
_C1_WORD_RE = re.compile(rf"[A-Za-z]*{_C1}[A-Za-z'-]*")

# Mode-B candidates: clean words with no C1 byte that *might* be missing a
# ligature. We require the word boundary on both sides to be free of C1
# bytes — otherwise we'd "repair" fragments left behind when Mode A failed
# to fix a C1 word, e.g. `pre\x93lter` → fragment `lter` → "filter" but the
# `\x93` is still there. Capital first letter is allowed because legitimate
# damage shows up at sentence starts (`Dierent` was `Different`); the
# dictionary check rejects author surnames like `Gurevych` naturally.
# Bound work by only probing 3-15-letter words.
_BARE_WORD_RE = re.compile(r"(?<![\x80-\x9f])\b[A-Za-z][A-Za-z'-]{2,14}\b(?![\x80-\x9f])")

# Candidate ligatures, ordered by typical frequency in English. When two
# candidates both yield real words, the first one wins (mostly cosmetic;
# ambiguity is rare in practice).
_LIGATURES = ("fi", "ff", "ffi", "fl", "ffl", "ft", "tt")

# Soft-hyphen / line-break artifacts: any C0 control char (except \t\n\r)
# sitting between two letters is treated as an elidable soft hyphen.
_INTERIOR_CTRL_RE = re.compile(
    r"(?<=[A-Za-z])[\x00-\x08\x0b\x0c\x0e-\x1f](?=[A-Za-z])"
)

_DICTIONARY: frozenset[str] | None = None
_DICTIONARY_PATH = Path(__file__).resolve().parent / "data" / "english_words.txt"


def _load_dictionary() -> frozenset[str]:
    """Lazy-load the bundled wordlist. Returns empty set if missing (graceful
    degradation: ligature repair becomes a no-op rather than crashing on a
    fresh checkout that hasn't built the wordlist yet)."""
    global _DICTIONARY
    if _DICTIONARY is None:
        try:
            words = _DICTIONARY_PATH.read_text(encoding="utf-8").split()
            _DICTIONARY = frozenset(w.lower() for w in words if w)
        except FileNotFoundError:
            _DICTIONARY = frozenset()
    return _DICTIONARY


def _is_known_word(candidate: str, dictionary: frozenset[str]) -> bool:
    """Whole-word lookup with hyphen-split fallback.

    The bundled list is unigrams (no hyphenated entries), so for hyphenated
    candidates like `off-target` or `fine-grained` we accept the candidate
    iff every hyphen-separated part is itself a dictionary word. Apostrophes
    are stripped (English contractions: `don't` → `dont` not in dict, but
    `don` is — kept simple by lookup of the leading part).
    """
    cand = candidate.lower()
    if cand in dictionary:
        return True
    if "-" in cand:
        parts = [p for p in cand.split("-") if p]
        return bool(parts) and all(p in dictionary for p in parts)
    return False


def _repair_c1_word(word: str, dictionary: frozenset[str]) -> str:
    """Try each ligature in turn at each C1 byte; accept the first
    substitution that yields a dictionary word. If none does, return the
    original (possibly with multiple C1 bytes still in it)."""
    # Multiple C1 bytes in one word (e.g. `Co\x83\x83dence`): collapse runs
    # of consecutive C1 bytes into one slot. This handles the common case
    # where the same byte represents one ligature; rare cases where two
    # adjacent unmapped CIDs represent different things (e.g. `\x83` = `n`
    # and `\x83` = `fi` in `Confidence`) are out of scope.
    collapsed = re.sub(rf"{_C1}+", "\x00", word)
    if "\x00" not in collapsed:
        return word
    parts = collapsed.split("\x00")
    if len(parts) - 1 == 1:
        prefix, suffix = parts
        for lig in _LIGATURES:
            candidate = prefix + lig + suffix
            if _is_known_word(candidate, dictionary):
                return candidate
    elif len(parts) - 1 == 2:
        a, b, c = parts
        for lig1 in _LIGATURES:
            for lig2 in _LIGATURES:
                candidate = a + lig1 + b + lig2 + c
                if _is_known_word(candidate, dictionary):
                    return candidate
    # 3+ gaps: don't try; combinatorial blow-up and almost never seen.
    return word


def _repair_bare_word(word: str, dictionary: frozenset[str]) -> str:
    """For a word *without* C1 bytes that's missing from the dictionary, try
    inserting each ligature at each gap. Accept only if exactly one
    insertion produces a dictionary word — otherwise leave alone."""
    if _is_known_word(word, dictionary):
        return word
    matches: list[str] = []
    for i in range(len(word) + 1):
        for lig in _LIGATURES:
            candidate = word[:i] + lig + word[i:]
            if _is_known_word(candidate, dictionary):
                matches.append(candidate)
                if len(matches) > 1:
                    return word  # ambiguous, bail
    return matches[0] if len(matches) == 1 else word


def _repair_ligatures(text: str) -> str:
    """Repair PDF-extraction noise: soft-hyphen joins and ligature dropouts.

    Two passes: (1) elide soft-hyphen control bytes between letters, then
    (2) for each word containing a C1 byte, try the ligature substitutions
    against the bundled English wordlist and accept the unique fix.

    Mode-B repair (whole-word drops with no C1 byte) is bounded — we only
    probe lowercase words 3-11 letters long that fail dictionary lookup,
    which keeps the work proportional to actual damage. Ambiguous matches
    (two ligatures both yield valid words) are left alone rather than
    guessed at; the LLM downstream of extraction tolerates a few damaged
    words better than it tolerates a confidently-wrong repair.
    """
    text = _INTERIOR_CTRL_RE.sub("", text)

    dictionary = _load_dictionary()
    if not dictionary:
        return text  # graceful degradation: no wordlist, no repair

    # Mode A: words containing C1 bytes
    text = _C1_WORD_RE.sub(lambda m: _repair_c1_word(m.group(), dictionary), text)
    # Mode B: bare lowercase words that fail dictionary lookup
    text = _BARE_WORD_RE.sub(lambda m: _repair_bare_word(m.group(), dictionary), text)
    return text


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

    Output is post-processed by `_repair_ligatures` to fix the common ligature-
    glyph dropout pattern in scientific PDFs (see module-level commentary).
    """
    pdf = pypdfium2.PdfDocument(str(path))
    try:
        meta = _slash_prefix_metadata(pdf.get_metadata_dict() or {})
        parts = _extract_all_pages(pdf, max_pages=max_pages)
    finally:
        pdf.close()
    return _repair_ligatures("\n\n".join(parts)), meta


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


def find_url_doi_candidates(pdf_text: str) -> list[tuple[str, str]]:
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
    """
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
