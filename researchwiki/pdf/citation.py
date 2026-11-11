"""Citation identity proof from one bounded bibliography entry and extraction."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from . import text as pdf_text

VERSION = 'citation-entry-v3'
_NUMBERED = re.compile(r'(?m)^[ \t]*(?:\[\d{1,4}\]|\d{1,4}[.)]|\d{1,4}[ \t]+(?=[A-Z]))[ \t]*')
# Headingless fallback: "1. Surname, I." / "[1] Surname I" runs, the layout
# Nature-family PDFs print with no "References" line at all.
_LIST_ITEM = re.compile(r"(?m)^[ \t]*(?:\d{1,5}[ \t]+)?(?:\[(\d{1,4})\]|(\d{1,4})\.)[ \t]+"
                        r"(?=[A-Z][\w’'-]+[,\s])")
_YEAR = re.compile(r'\b((?:19|20)\d{2})[a-z]?\b')
# Years that are part of a DOI, URL, or page/article range are not entry years.
_YEAR_NOISE = re.compile(r'(?i)(?:\b10\.\d{4,9}/\S+|https?://\S+|\b\d{1,6}[–-]\d{1,6}\b'
                         r'|\b\d{4}\.\d{4,5}(?:v\d+)?\b)')
_MIN_RUN = 5
_MAX_ENTRY = 2000
_AUTHOR = re.compile(r"(?m)^[ \t]*(?:[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ’']+(?:-[A-ZÀ-ÖØ-Ý]?[a-zà-öø-ÿ’']+)*"
                     r"(?:[ -][A-Z][a-z]+)?),?[ \t]+(?:[A-Z]{1,4}(?:[.,]|"
                     r"(?=[ \t]+(?:et[ \t]+al\b|(?:19|20)\d{2}\b)))|et[ \t]+al\b)")
_END = re.compile(r'(?im)^\s*(?:acknowledg(?:e)?ments|author contributions|competing interests|'
                  r'online methods|methods|supplementary information|extended data|'
                  r'data availability|code availability|funding|ethics declarations)\s*[:.]?\s*$')


def file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def target_identity(page):
    stem = re.match(r'^(.+?)-((?:19|20)\d{2})[a-z]?-.+$', page.stem)
    # The YAML version year is primary, but a preprint→journal upgrade keeps
    # the stem's preprint year (CLAUDE.md), and sources cite either version.
    stem_year = int(stem.group(2)) if stem else None
    year = page.year_int() or stem_year
    years = sorted({y for y in (year, stem_year) if y})
    return {'stem': page.stem, 'title': page.str_field('title', ''),
            'doi': page.str_field('doi', '').strip().lower(),
            'surname': stem.group(1) if stem else '', 'year': year, 'years': years}


def _accepted_years(expected):
    years = expected.get('years') or [expected.get('year')]
    return {str(y) for y in years if y}


def _entry_years(entry):
    return set(_YEAR.findall(_YEAR_NOISE.sub(' ', entry)))


def _tokens(value):
    value = re.sub(r'(?<=\d),(?=\d{3}\b)', '', pdf_text._fold_ascii(value).lower())
    return [w for w in re.findall(r'[a-z0-9]+', value) if w != 'of']


def _title_span(entry, title):
    wanted = _tokens(title)
    if len(wanted) < 4:
        return None
    # Keep raw offsets while comparing normalized tokens. Thousands commas
    # are included in a token; all other punctuation acts as a separator.
    words = [(m, _tokens(m.group())) for m in re.finditer(r'[\w]+(?:,\d{3})*', entry)]
    words = [(m, t[0]) for m, t in words if len(t) == 1]
    for i in range(len(words) - len(wanted) + 1):
        if [w for _, w in words[i:i + len(wanted)]] != wanted:
            continue
        start, end = words[i][0].start(), words[i + len(wanted) - 1][0].end()
        # Full title must terminate; a longer title sharing a prefix fails.
        if re.match(r"\s*[\"')\]]*(?:[.;?!]|\((?:19|20)\d{2}\))", entry[end:]):
            return start, end
    return None


def _section_end(text, start, stop=None):
    """The next section/reference heading bounds every bibliography region."""
    stop = len(text) if stop is None else stop
    boundaries = (_END.search(text, start, stop), pdf_text.REFS_HEADING_RE.search(text, start, stop))
    return min([stop] + [m.start() for m in boundaries if m])


def _sections(text):
    # Figures also label a genome "Reference". Accept a heading only when
    # a recognizable first bibliography entry follows it, not distant prose.
    # Every qualifying heading counts: Nature-style papers print a second,
    # Methods-only reference list after the main one.
    for heading in pdf_text.REFS_HEADING_RE.finditer(text):
        finish = _section_end(text, heading.end())
        section = text[heading.end():finish]
        starts = list(_NUMBERED.finditer(section))
        numbered = (starts and starts[0].start() < 250
                    and int(re.search(r'\d+', starts[0].group()).group()) == 1)
        if not numbered:
            starts = list(_AUTHOR.finditer(section))
        if not starts or section[:starts[0].start()].strip():
            continue
        yield [(heading.end() + m.start(),
                heading.end() + (starts[i + 1].start() if i + 1 < len(starts) else len(section)))
               for i, m in enumerate(starts)]


def _numbered_runs(text):
    """Headingless bibliographies: runs of consecutively numbered entries.

    A run must count up by one for at least `_MIN_RUN` entries, so section
    numbering and short enumerated prose do not qualify. A shorter run is kept
    only when it continues an accepted run's numbering: a Methods list picks
    up where the main list stopped (e.g. 47, 48 after 1–46).
    """
    items = [(m.start(), int(m.group(1) or m.group(2))) for m in _LIST_ITEM.finditer(text)]
    candidates, current = [], []
    for start, number in items:
        if (current and number == current[-1][1] + 1
                and _section_end(text, current[-1][0], start) == start):
            current.append((start, number))
            continue
        if current:
            candidates.append(current)
        current = [(start, number)]
    if current:
        candidates.append(current)
    runs = []
    for run in candidates:
        continues = runs and run[0][1] == runs[-1][-1][1] + 1
        if len(run) >= _MIN_RUN or continues:
            runs.append(run)
    for run in runs:
        stop = _section_end(text, run[-1][0])
        bounds = [s for s, _ in run] + [stop]
        yield [(bounds[i], bounds[i + 1]) for i in range(len(run))]


def bibliography_entries(text):
    """Non-overlapping entry spans from every identified reference list.

    Heading-scoped lists come first; a consecutively numbered run outside them
    (a headingless list, or a Methods list after the main one) is added too.
    Each entry ends at the next entry's start and never exceeds `_MAX_ENTRY`,
    so one entry's text cannot appear in another and manufacture ambiguity.
    """
    sections = list(_sections(text))
    entries = [e for section in sections for e in section]
    covered = [(s[0][0], s[-1][1]) for s in sections]
    # With a headed bibliography, only runs after it count (back-matter lists);
    # numbered prose in the body is never a reference list.
    floor = covered[0][0] if covered else 0
    # A Methods list may continue the main numbering, but is a separate run
    # so the preceding entry cannot absorb intervening Methods prose.
    for run in _numbered_runs(text):
        entries.extend((s, e) for s, e in run
                       if s >= floor and not any(lo <= s < hi for lo, hi in covered))
    starts = sorted({s: e for s, e in sorted(entries, reverse=True)}.items())
    return [(s, min(e, starts[i + 1][0] if i + 1 < len(starts) else e, s + _MAX_ENTRY))
            for i, (s, e) in enumerate(starts)]


def _unsegmented_identifiers(text):
    """Exact DOI/arXiv identifiers in a bibliography that cannot be segmented.

    Require an explicit bibliography heading and stop at the next section.
    Singular "Reference" can be a figure label, so it only qualifies through
    the entry parser. Headingless PDFs need an identified numbered list; an
    arbitrary character cutoff cannot establish citation context.
    """
    found = []
    for heading in pdf_text.REFS_HEADING_RE.finditer(text):
        if heading.group(1).lower() == 'reference':
            continue
        start, end = heading.end(), _section_end(text, heading.end())
        identifiers = [(pdf_text._trim_doi_noise(m.group()).lower(), m)
                       for m in pdf_text.DOI_RE.finditer(text, start, end)]
        identifiers.extend(('10.48550/arxiv.' + m.group(1), m)
                           for m in pdf_text._ARXIV_ID_RE.finditer(text, start, end))
        for ident, m in identifiers:
            excerpt = text[max(start, m.start() - 200):min(end, m.end() + 200)]
            found.append((ident, m.start(), m.end(), excerpt, [start, end]))
    return found


class CitationVerifier:
    """Per-run extraction memo; every proof pins the PDF hash and coordinates.

    Only the bibliography spans are retained, in a bounded LRU, so a corpus-wide
    backlog does not hold every PDF's full text for the life of the process.
    """
    MAX_EXTRACTIONS = 64

    def __init__(self, max_extractions=MAX_EXTRACTIONS):
        from collections import OrderedDict
        self.extractions = OrderedDict()
        self.max_extractions = max_extractions
        self.proofs = {}

    def verify(self, path, source_stem, expected):
        path = Path(path)
        digest = file_hash(path)
        base = {'source_stem': source_stem, 'target_stem': expected['stem'],
                'source_pdf_sha256': digest, 'expected_identity': expected,
                'verifier_version': VERSION, 'verified': False, 'method': None,
                'page': None, 'excerpt': None, 'match_span': None}
        if digest is None:
            return dict(base, reason='missing or unreadable PDF')
        key = (digest, repr(sorted(expected.items())), VERSION)
        if key in self.proofs:
            return dict(self.proofs[key], source_stem=source_stem)
        if digest in self.extractions:
            self.extractions.move_to_end(digest)
        else:
            try:
                pages = pdf_text.extract_pdf_page_texts(path, max_pages=None)
                text = pdf_text.PAGE_SEPARATOR.join(pages)
                if file_hash(path) != digest:
                    return dict(base, reason='PDF changed during extraction')
                # Keep only each entry's absolute span and text, not the PDF.
                entries = [((s, e), text[s:e]) for s, e in bibliography_entries(text)]
                loose = [] if entries else _unsegmented_identifiers(text)
                self.extractions[digest] = (entries, loose, pdf_text.page_offsets(pages), len(text))
                while len(self.extractions) > self.max_extractions:
                    self.extractions.popitem(last=False)
            except Exception as exc:
                return dict(base, reason=f'unreadable PDF: {type(exc).__name__}')
        entries, loose, offsets, total = self.extractions[digest]
        matches = []
        doi = expected.get('doi')
        for ident, s, e, excerpt, region in loose:
            if doi and ident == doi:
                matches.append(dict(base, verified=True, reason='verified',
                                    method='exact-doi-unsegmented',
                                    page=pdf_text.page_for_offset(offsets, s, total),
                                    excerpt=excerpt.strip(), entry_span=None, match_span=[s, e],
                                    bibliography_span=region))
        # The same DOI printed twice is still one identity, not ambiguity.
        matches = matches[:1]
        for (start, end), entry in entries:
            span = next(((m.start(), m.end()) for m in pdf_text.DOI_RE.finditer(entry)
                         if doi and pdf_text._trim_doi_noise(m.group()).lower() == doi), None)
            method = 'exact-doi' if span else None
            if not span and doi.startswith('10.48550/arxiv.'):
                span = next(((m.start(), m.end()) for m in pdf_text._ARXIV_ID_RE.finditer(entry)
                             if '10.48550/arxiv.' + m.group(1) == doi), None)
                method = 'exact-doi' if span else None
            if not span:
                title_span = _title_span(entry, expected.get('title', ''))
                surname = pdf_text._surname_pattern(expected.get('surname', ''))
                accepted = _accepted_years(expected)
                author_text = entry[:title_span[0]] if title_span else ''
                # `_LIST_ITEM` also strips a preprint's line-number gutter.
                number = _LIST_ITEM.match(author_text) or _NUMBERED.match(author_text)
                author_text = author_text[number.end():] if number else author_text.lstrip()
                # Author identity belongs before the title, not in another
                # entry or in a title mentioning the same surname.
                if title_span and surname and accepted and surname.match(
                        pdf_text._fold_ascii(author_text).lower()):
                    years = _entry_years(entry)
                    # Several distinct publication years could be another,
                    # unsegmented reference: never borrow an arbitrary later
                    # year. Page ranges and DOI-embedded dates are not years.
                    if len(years) == 1 and years <= accepted:
                        span, method = title_span, 'title-author-year'
            if span:
                absolute = [start + span[0], start + span[1]]
                matches.append(dict(base, verified=True, reason='verified', method=method,
                                    page=pdf_text.page_for_offset(offsets, absolute[0], total),
                                    excerpt=entry.strip(), entry_span=[start, end], match_span=absolute))
        if len(matches) == 1:
            proof = matches[0]
        else:
            reason = ('ambiguous bibliography identity' if matches else
                      'target identity absent from one bibliography entry' if entries else
                      'bibliography entries unidentified; no exact DOI in references')
            proof = dict(base, reason=reason)
        self.proofs[key] = proof
        return proof
