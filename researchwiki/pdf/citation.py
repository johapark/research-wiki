"""Citation identity proof from one bounded bibliography entry and extraction."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from . import text as pdf_text

VERSION = 'citation-entry-v1'
_NUMBERED = re.compile(r'(?m)^[ \t]*(?:\[\d{1,4}\]|\d{1,4}[.)]|\d{1,4}[ \t]+(?=[A-Z]))[ \t]*')
_AUTHOR = re.compile(r"(?m)^[ \t]*(?:[A-ZÀ-ÖØ-Ý][a-zà-öø-ÿ’']+(?:-[A-ZÀ-ÖØ-Ý]?[a-zà-öø-ÿ’']+)*"
                     r"(?:[ -][A-Z][a-z]+)?),?[ \t]+(?:[A-Z]{1,4}(?:[.,]|"
                     r"(?=[ \t]+(?:et[ \t]+al\b|(?:19|20)\d{2}\b)))|et[ \t]+al\b)")
_END = re.compile(r'(?im)^\s*(?:acknowledg(?:e)?ments|author contributions|competing interests|'
                  r'online methods|methods|supplementary information|extended data)\s*[:.]?\s*$')


def file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def target_identity(page):
    stem = re.match(r'^(.+?)-((?:19|20)\d{2})[a-z]?-.+$', page.stem)
    # The current YAML version year is authoritative when present.
    year = page.year_int() or (int(stem.group(2)) if stem else None)
    return {'stem': page.stem, 'title': page.str_field('title', ''),
            'doi': page.str_field('doi', '').strip().lower(),
            'surname': stem.group(1) if stem else '', 'year': year}


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


def bibliography_entries(text):
    # Figures also label a genome "Reference". Accept a heading only when
    # a recognizable first bibliography entry follows it, not distant prose.
    for heading in pdf_text.REFS_HEADING_RE.finditer(text):
        end = _END.search(text, heading.end())
        finish = end.start() if end else len(text)
        section = text[heading.end():finish]
        starts = list(_NUMBERED.finditer(section))
        numbered = (starts and starts[0].start() < 250
                    and int(re.search(r'\d+', starts[0].group()).group()) == 1)
        if not numbered:
            starts = list(_AUTHOR.finditer(section))
        if not starts or section[:starts[0].start()].strip():
            continue
        return [(heading.end() + m.start(),
                 heading.end() + (starts[i + 1].start() if i + 1 < len(starts) else len(section)))
                for i, m in enumerate(starts)]
    return []



class CitationVerifier:
    """Per-run extraction memo; every proof pins the PDF hash and coordinates."""
    def __init__(self):
        self.extractions = {}
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
        if digest not in self.extractions:
            try:
                pages = pdf_text.extract_pdf_page_texts(path, max_pages=None)
                text = pdf_text.PAGE_SEPARATOR.join(pages)
                if file_hash(path) != digest:
                    return dict(base, reason='PDF changed during extraction')
                self.extractions[digest] = (text, pdf_text.page_offsets(pages))
            except Exception as exc:
                return dict(base, reason=f'unreadable PDF: {type(exc).__name__}')
        text, offsets = self.extractions[digest]
        entries = bibliography_entries(text)
        matches = []
        for start, end in entries:
            entry = text[start:end]
            doi = expected.get('doi')
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
                year = expected.get('year')
                author_text = entry[:title_span[0]] if title_span else ''
                number = _NUMBERED.match(author_text)
                author_text = author_text[number.end():] if number else author_text.lstrip()
                # Author identity belongs before the title, not in another
                # entry or in a title mentioning the same surname.
                if (title_span and surname and surname.match(pdf_text._fold_ascii(author_text).lower())
                        and year and re.search(rf'\b{year}[a-z]?\b', entry)):
                    years = set(re.findall(r'\b((?:19|20)\d{2})[a-z]?\b', entry))
                    # Multiple different years could be another unsegmented
                    # reference. Do not borrow an arbitrary later year.
                    if years == {str(year)}:
                        span, method = title_span, 'title-author-year'
            if span:
                absolute = [start + span[0], start + span[1]]
                matches.append(dict(base, verified=True, reason='verified', method=method,
                                    page=pdf_text.page_for_offset(offsets, absolute[0], len(text)),
                                    excerpt=entry.strip(), entry_span=[start, end], match_span=absolute))
        if len(matches) == 1:
            proof = matches[0]
        else:
            reason = ('ambiguous bibliography identity' if matches else
                      'target identity absent from one bibliography entry' if entries else
                      'bibliography entries unidentified')
            proof = dict(base, reason=reason)
        self.proofs[key] = proof
        return proof
