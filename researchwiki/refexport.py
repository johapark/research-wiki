"""Emit the corpus as BibTeX / RIS / CSL-JSON — the inverse of `refimport`.

Zero tokens, no network: everything comes from `wiki/` frontmatter. Where the
corpus is short of data the run *reports* it rather than guessing, which is the
one rule the whole module is arranged around.

**Names are not parsed for BibTeX or RIS.** Both formats understand
`First von Last` themselves — verified: `{A. van der Graaf and Christopher Ré}`
and `AU  - A. van der Graaf` both round-trip through `refimport.parse` exactly. So
the transformation there is "replace the separator", and 58 nobiliary-particle
names plus 76 four-token names cannot be corrupted by a boundary guess we never
make. Only CSL-JSON wants structured `family`/`given`, and `names.as_family_given`
declines to split anything ambiguous — CSL's own `literal` field is the faithful
record of a name with no given/family structure.

**Frontmatter, not the state DB.** The DB is derived, and its schema header says
markdown wins on drift; a `.bib` pasted into a manuscript must not be able to
disagree with the pages. It also keeps the command free of a `db rebuild`
freshness gate.

**The citekey is the page stem**, because a citekey sitting in someone's
manuscript must never change. Any key computed at export time renumbers its
disambiguating letters when a sibling paper is ingested.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import names
from .metadata_sanity import is_venue_furniture
from .refimport import clean_doi
from .wiki import Page, is_preprint_doi, read_pages

#: Page types that describe a document somebody else published. Everything else
#: in `wiki/` — synthesis, idea, concept, and the `meta`/`dashboard` bookkeeping
#: pages — is the user's own unpublished analysis, with no DOI, venue or year of
#: record. An entry for one would assert, once pasted into a manuscript, a
#: publication that does not exist. That is a citation-integrity harm rather than
#: a formatting one, so there is deliberately no flag to include them; the
#: shareable-document workflow (`prompts/share-page.md`) is the right path.
EXPORTABLE_TYPES = ("paper", "commentary", "whitepaper", "guidance", "book")

#: Wiki page type → the kind of thing we emit. Refined below for preprints and
#: for papers with no recorded venue.
_PAGE_KIND = {
    "paper": "article",
    "commentary": "article",     # a real publication; a manager should hold it
    "whitepaper": "report",
    "guidance": "report",
    "book": "book",
}

#: kind → (BibTeX entry, RIS TY, CSL type).
#:
#: An explicit *outbound* table, not an inversion of `refimport.parse`'s inbound
#: maps: those are many-to-one (`inproceedings|conference|article → article`), so
#: inverting silently picks whichever key iterated last, and their keys are
#: reference-manager types where ours are wiki page types.
#:
#: Every value here must be readable back by our own parser — pinned by a test
#: asserting each is a key of the corresponding inbound map. `misc` for a
#: preprint (rather than `article`) is what arXiv's own BibTeX button emits, and
#: pairs with the `eprint` field that makes `parse._normalize_type` recognize it.
#: CSL `manuscript` is a documented wart: CSL 1.0.2's `preprint` reads better in
#: Zotero, but our parser has no key for it, so round-trip wins.
ENTRY_TYPES = {
    "article":  ("article",    "JOUR", "article-journal"),
    "preprint": ("misc",       "INPR", "manuscript"),
    "report":   ("techreport", "RPRT", "report"),
    "book":     ("book",       "BOOK", "book"),
    "misc":     ("misc",       "GEN",  "document"),
}

_YEAR_RE = re.compile(r"(19|20)\d{2}")

#: Every character that cannot appear literally in a BibTeX field value.
#:
#: Applied as a single character-for-character pass, which is what makes it
#: order-free. Escaping `\` in its own pass before the rest is the obvious
#: implementation and it is wrong: the `{}` in `\textbackslash{}` is LaTeX syntax,
#: and a following brace pass escapes it into `\textbackslash\{\}`. One pass
#: cannot re-examine what it just wrote.
_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


# ---------------------------------------------------------------- records

@dataclass
class Record:
    """One bibliography entry, format-neutral."""

    key: str                      # citekey == page stem
    kind: str                     # a key of ENTRY_TYPES
    title: str
    authors: list[str] = field(default_factory=list)   # display order, as printed
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    institution: str | None = None     # `issuer`, for a techreport
    url: str | None = None             # `source_url`
    note: str | None = None            # no_doi_reason / document_id / status
    eprint: str | None = None          # `arxiv_id`


@dataclass
class Report:
    """What the run did, and what the corpus is short of.

    Every selected page lands in `records` or `skipped`, so
    `records + len(skipped)` is the number selected. That equality is what makes
    "report, don't guess" checkable rather than aspirational.
    """

    records: int = 0
    by_entry_type: dict[str, int] = field(default_factory=dict)
    venue_missing: list[str] = field(default_factory=list)
    venue_furniture: list[dict] = field(default_factory=list)
    doi_missing: list[dict] = field(default_factory=list)
    authors_unparseable: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def as_dict(self, fmt: str) -> dict:
        return {
            "format": fmt,
            "records": self.records,
            "by_entry_type": self.by_entry_type,
            "venue_missing": self.venue_missing,
            "venue_furniture": self.venue_furniture,
            "doi_missing": self.doi_missing,
            "authors_unparseable": self.authors_unparseable,
            "skipped": self.skipped,
        }


# ---------------------------------------------------------------- LaTeX

def latex_escape(s: str) -> str:
    """Make a string safe as a BibTeX field value.

    UTF-8 is emitted as-is: biber, XeLaTeX, Pandoc and Zotero all read it, the
    real 532-item ReadCube export we validated the importer against contains zero
    backslashes, and a Unicode→macro table would have to be *correct* or it
    corrupts the 86 corpus author fields carrying non-ASCII. A pdfLaTeX-only
    pipeline needs `\\usepackage[utf8]{inputenc}` or biber.
    """
    s = _CONTROL_RE.sub("", (s or "")).replace("\xa0", " ")
    return "".join(_ESCAPES.get(c, c) for c in s)


def _needs_protection(token: str) -> bool:
    """Whether a title token would be destroyed by a title-lowercasing style.

    Orthographic, so it needs no vocabulary: an uppercase letter past position 0
    (`CRISPR`, `AlphaFold`, `mRNA`, `PLoS`), all-caps of length ≥2 (`DNA`, `AI`),
    or a digit adjacent to a letter (`Cas9`, `SARS-CoV-2`). A token whose only
    capital is initial is left alone — that is sentence case or a proper noun the
    style may legitimately re-case.
    """
    word = token.strip(".,;:!?()[]\"'")
    if not word:
        return False
    if any(c.isupper() for c in word[1:]):
        return True
    if word.isupper() and len(word) >= 2:
        return True
    return bool(re.search(r"[A-Za-z]\d|\d[A-Za-z]", word))


def bibtex_value(s: str, *, protect: bool = False) -> str:
    """A BibTeX field value: escaped, and optionally brace-protected per token.

    Order matters. The protection decision is made on the *raw* token and the
    brace wraps the *escaped* one — deciding after escaping would test
    backslashes and braces this function just introduced. The whole token is
    wrapped, never a piece of it: a brace mid-word breaks hyphenation.

    248 of 421 corpus titles contain a token that needs this, so for titles it is
    not optional — `CRISPR-Cas9` becomes `Crispr-cas9` under `plain.bst` without.
    """
    if not protect:
        return latex_escape(s)
    out = []
    for token in (s or "").split():
        esc = latex_escape(token)
        out.append("{" + esc + "}" if _needs_protection(token) else esc)
    return " ".join(out)


# ---------------------------------------------------------------- collect

def _year_of(page: Page) -> int | None:
    """`year:`, else a year inside `issuance_date`, else the stem's.

    All `guidance` pages carry an `issuance_date` and no `year:`.
    """
    if (y := page.year_int()) is not None:
        return y
    for source in (page.str_field("issuance_date"), page.stem):
        if m := _YEAR_RE.search(source):
            return int(m.group(0))
    return None


def _note_of(page: Page, doi: str | None) -> str | None:
    """The one field where free text is emitted verbatim rather than mined.

    `document_id` holds things like `Nature 654:324-326`, which *contains* a
    volume and a page range — and FDA guidance numbers occupy the same field with
    entirely different structure. Parsing it is judgement, so it is passed
    through whole. `no_doi_reason` is human-authored prose explaining a citation
    gap, which is exactly what a reader of the `.bib` wants to see.
    """
    parts = [page.str_field("document_id"), page.str_field("status")]
    if not doi:
        parts.append(page.str_field("no_doi_reason"))
    joined = ". ".join(p for p in parts if p)
    return joined or None


def _record_for(page: Page, report: Report) -> Record | None:
    title = page.str_field("title")
    if not title:
        report.skipped.append({"stem": page.stem, "reason": "no title"})
        return None

    doi = clean_doi(page.str_field("doi") or None)
    if not doi:
        report.doi_missing.append({"stem": page.stem,
                                   "reason": page.str_field("no_doi_reason") or None})

    venue = page.str_field("venue") or None
    if venue and is_venue_furniture(venue):
        # The one place this command could print a falsehood: a masthead artifact
        # like `Journal of LaTeX Class Files` recorded as the journal.
        report.venue_furniture.append({"stem": page.stem, "venue": venue})
        venue = None

    author_field = page.fm.get("authors")
    authors = names.split_author_field(author_field)
    if author_field and not authors:
        report.authors_unparseable.append(page.stem)

    kind = _PAGE_KIND[page.page_type]
    if kind == "article":
        if is_preprint_doi(doi) or page.str_field("arxiv_id"):
            kind = "preprint"
        elif not venue:
            # `@article` with no `journal` makes bibtex merely *warn*, surfacing
            # as buried lines in a LaTeX log weeks later. `@misc` surfaces in the
            # bibliography itself and in this report.
            report.venue_missing.append(page.stem)
            kind = "misc"

    return Record(
        key=page.stem,
        kind=kind,
        title=title,
        authors=authors,
        year=_year_of(page),
        doi=doi,
        venue=venue,
        institution=page.str_field("issuer") or None,
        url=page.str_field("source_url") or None,
        note=_note_of(page, doi),
        eprint=page.str_field("arxiv_id") or None,
    )


def collect(*, categories: list[str] | None = None,
            years: tuple[int, int] | None = None,
            stems: list[str] | None = None) -> tuple[list[Record], Report]:
    """Every exportable page matching the filters, in a deterministic order."""
    report = Report()
    wanted = set(stems or ())
    records: list[Record] = []

    for page in read_pages():
        if page.page_type not in EXPORTABLE_TYPES:
            continue
        if categories and page.category not in categories:
            continue
        if wanted and page.stem not in wanted:
            continue
        if years is not None:
            y = _year_of(page)
            if y is None or not (years[0] <= y <= years[1]):
                continue
        if (rec := _record_for(page, report)) is not None:
            records.append(rec)

    records.sort(key=lambda r: r.key)
    report.records = len(records)
    for r in records:
        entry = ENTRY_TYPES[r.kind][0]
        report.by_entry_type[entry] = report.by_entry_type.get(entry, 0) + 1
    return records, report


# ---------------------------------------------------------------- render

def render_bibtex(records: list[Record]) -> str:
    out = []
    for r in records:
        entry = ENTRY_TYPES[r.kind][0]
        fields: list[tuple[str, str]] = [("title", bibtex_value(r.title, protect=True))]
        if r.authors:
            # ` and `-joined and otherwise untouched: BibTeX parses `First von
            # Last` itself, so inverting here could only introduce errors.
            fields.append(("author", " and ".join(latex_escape(a) for a in r.authors)))
        if r.venue:
            fields.append(("journal", bibtex_value(r.venue, protect=True)))
        if r.year is not None:
            fields.append(("year", str(r.year)))
        if r.doi:
            fields.append(("doi", r.doi))
        if r.eprint:
            fields.extend([("eprint", latex_escape(r.eprint)),
                           ("archivePrefix", "arXiv")])
        if r.institution and r.kind == "report":
            fields.append(("institution", bibtex_value(r.institution, protect=True)))
        if r.url:
            fields.append(("url", r.url))
        if r.note:
            fields.append(("note", bibtex_value(r.note)))

        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
        out.append(f"@{entry}{{{r.key},\n{body},\n}}")
    return "\n\n".join(out) + ("\n" if out else "")


def render_ris(records: list[Record]) -> str:
    out = []
    for r in records:
        lines = [f"TY  - {ENTRY_TYPES[r.kind][1]}", f"ID  - {r.key}",
                 f"TI  - {r.title}"]
        # One AU per author, un-inverted: `_ris_record_to_item` normalizes either
        # spelling back to `Given Family`, so the simpler one is also faithful.
        lines += [f"AU  - {a}" for a in r.authors]
        if r.year is not None:
            lines.append(f"PY  - {r.year}")
        if r.venue:
            lines.append(f"T2  - {r.venue}")
        if r.doi:
            lines.append(f"DO  - {r.doi}")
        if r.url:
            lines.append(f"UR  - {r.url}")
        if r.institution and r.kind == "report":
            lines.append(f"PB  - {r.institution}")
        if r.note:
            lines.append(f"N1  - {r.note}")
        lines.append("ER  - ")
        out.append("\n".join(lines))
    return "\n\n".join(out) + ("\n" if out else "")


def _csl_author(raw: str) -> dict:
    split = names.as_family_given(raw)
    if split is None:
        return {"literal": raw}
    family, given = split
    return {"family": family, "given": given} if given else {"family": family}


def render_csl_json(records: list[Record]) -> str:
    items = []
    for r in records:
        item: dict = {"id": r.key, "type": ENTRY_TYPES[r.kind][2], "title": r.title}
        if r.authors:
            item["author"] = [_csl_author(a) for a in r.authors]
        if r.year is not None:
            item["issued"] = {"date-parts": [[r.year]]}
        if r.doi:
            item["DOI"] = r.doi
        if r.venue:
            item["container-title"] = r.venue
        if r.url:
            item["URL"] = r.url
        if r.institution and r.kind == "report":
            item["publisher"] = r.institution
        if r.note:
            item["note"] = r.note
        items.append(item)
    return json.dumps(items, ensure_ascii=False, indent=2) + "\n"


RENDERERS = {
    "bibtex": render_bibtex,
    "ris": render_ris,
    "csl-json": render_csl_json,
}
