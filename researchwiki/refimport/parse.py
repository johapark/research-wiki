"""Bibliographic exports → `ExportItem` records.

Three formats, one output shape. What matters here is *tolerance*: these files
are written by consumer applications, not by a spec-conformant serializer, and
the ones that break a strict parser are the ones users actually have.

Observed in a single real 532-item ReadCube library:

  - a **4-character `PMID`** RIS tag, where the convention is 2
  - a junk `XX  - ` tag with an empty value, on 385 of 532 records
  - BibTeX citekeys containing `:` (55 records) and non-ASCII (16) — both
    illegal in strict BibTeX, so a validating parser rejects the entire file
  - CRLF line endings, which is only a trap because Python's text-mode read
    translates them, making a literal `"\\r\\n"` split return one giant record

None of that is malformed enough to be worth refusing. The rule throughout is:
skip what you cannot understand, keep what you can, and never let one bad record
take down the file.

Deliberately no `bibtexparser` dependency. Eight fields do not justify a new
package in a project that already pins `numpy<2` on x86_64 macOS to keep torch
importable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .latex import delatex

#: Normalized item types. `article` is the only one that proceeds to ingest
#: without a second look; the rest are triage signals.
ITEM_TYPES = ("article", "preprint", "book", "chapter", "thesis",
              "report", "webpage", "other")

_BIBTEX_TYPE_MAP = {
    "article": "article", "inproceedings": "article", "conference": "article",
    "incollection": "chapter", "inbook": "chapter",
    "book": "book", "booklet": "book",
    "phdthesis": "thesis", "mastersthesis": "thesis",
    "techreport": "report", "manual": "report",
    "online": "webpage", "electronic": "webpage", "www": "webpage",
    "misc": "other", "unpublished": "other",
}

_RIS_TYPE_MAP = {
    "JOUR": "article", "EJOUR": "article", "CPAPER": "article",
    "CONF": "article", "INPR": "preprint",
    "BOOK": "book", "EBOOK": "book", "CHAP": "chapter",
    "THES": "thesis", "RPRT": "report",
    "ELEC": "webpage", "ICOMM": "webpage", "BLOG": "webpage",
    "GEN": "other", "UNPB": "other",
}

#: One known cross-format asymmetry, left in deliberately: a record that RIS
#: types `ELEC` and BibTeX types `@misc` lands as `webpage` from one file and
#: `other` from the other. Both are honest readings of what the exporter wrote,
#: and forcing them to agree would mean overriding one tool's own type. It costs
#: nothing because triage never relies on the type field (531 of 532 records in
#: a real library are `JOUR`/`@article`, books included) — the record that
#: triggers this in practice has no DOI, author or year either, so the
#: `unresolvable` gate catches it from both formats regardless.
_CSL_TYPE_MAP = {
    "article-journal": "article", "paper-conference": "article",
    "article": "article", "manuscript": "preprint",
    "book": "book", "chapter": "chapter", "thesis": "thesis",
    "report": "report", "webpage": "webpage", "post-weblog": "webpage",
    "post": "webpage", "document": "other",
}

#: A DOI as registered: `10.` + registrant + `/` + suffix. Used to validate
#: before a value is handed to `agent ingest --doi`, because a malformed DOI
#: there costs a failed lookup rather than being ignored.
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

#: RIS tag line. **Not** the conventional `^[A-Z][A-Z0-9]  - ` — see the module
#: docstring for the 4-character `PMID` tag that motivates the width range.
_RIS_TAG = re.compile(r"^([A-Z][A-Z0-9]{1,5})\s+-\s?(.*)$")

#: RIS tags that name an attachment. `L1` is the primary full-text link.
_RIS_FILE_TAGS = ("L1", "L2", "L4", "LK", "FI")


@dataclass
class ExportItem:
    """One bibliographic record, format-independent.

    `raw` keeps every field the parser saw, including ones this project has no
    use for. It costs nothing and makes an unexpected export diagnosable
    without re-running the parser under a debugger.
    """

    key: str
    item_type: str = "other"
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    declared_files: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def has_usable_doi(self) -> bool:
        return bool(self.doi and DOI_RE.match(self.doi))

    @property
    def is_preprint_doi(self) -> bool:
        """True when the DOI belongs to a preprint server.

        Used to pick the survivor of a preprint/published pair — the single
        highest-value dedupe in a real library: 10 such pairs in 532 records
        with **zero** duplicate DOIs, so nothing else finds them.

        Delegates to `wiki.is_preprint_doi` rather than testing `10.1101/`
        here. That shared list also covers medRxiv, OSF, Preprints.org and
        arXiv, and the real library contains a `10.64898/` record a
        bioRxiv-only check would have missed. One list, one place to extend.
        """
        from ..wiki import is_preprint_doi
        return is_preprint_doi(self.doi)

    def as_dict(self) -> dict:
        return {
            "key": self.key, "item_type": self.item_type, "title": self.title,
            "authors": self.authors, "year": self.year, "doi": self.doi,
            "venue": self.venue, "declared_files": self.declared_files,
        }


# ---------- shared field normalization ----------

def _clean_doi(value: str | None) -> str | None:
    """Strip the URL wrappers exporters put around a DOI, lowercase, validate.

    Returns None for anything that is not a registered-shape DOI, so a caller
    can treat "has a DOI" as "has a DOI worth passing to `--doi`".
    """
    if not value:
        return None
    s = value.strip().rstrip(".,;")
    s = re.sub(r"(?i)^(?:https?://)?(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"(?i)^doi:\s*", "", s).strip()
    return s.lower() if DOI_RE.match(s) else None


#: A plausible publication year, guarded on both sides by "not a digit" rather
#: than by `\b`. Two reasons, both from real values:
#:
#:   - `\b` fails on `c2015` (copyright form, common in catalogue-derived
#:     records) because `c`→`2` is not a word boundary.
#:   - dropping the guard entirely would match inside longer numbers: the ISBN
#:     `9781590595` contains `1590`, which is in range.
#:
#: Digit-guards accept a year preceded by a letter and reject one embedded in a
#: number, which is exactly the distinction wanted.
_YEAR_RE = re.compile(r"(?<![0-9])(1[5-9]\d{2}|20\d{2}|21\d{2})(?![0-9])")


def _clean_year(value) -> int | None:
    """First plausible 4-digit year in the value, or None.

    Exporters write `2015`, `2015/03/01`, `c2015`, `2015-2016` and `in press`.
    The range bound also rejects a page range (`1473--1475`) that would
    otherwise read as a year.
    """
    if value is None:
        return None
    m = _YEAR_RE.search(str(value))
    return int(m.group(1)) if m else None


def _split_authors(raw: str) -> list[str]:
    """`"Last, First and Last, First"` → display-order names.

    Both BibTeX (` and `) and RIS (one tag per author) arrive here as a list of
    `Surname, Given` strings. `stems.first_author_surname` handles the
    comma form natively, so the flip to `Given Surname` is for readability in
    the manifest and for `--authors`, not for stem derivation.
    """
    out = []
    for part in re.split(r"\s+and\s+", raw):
        name = delatex(part).strip().rstrip(",")
        if not name:
            continue
        if "," in name:
            surname, _, given = name.partition(",")
            name = f"{given.strip()} {surname.strip()}".strip()
        out.append(name)
    return out


def _normalize_type(mapped: str | None, item: ExportItem) -> str:
    """Refine a mapped type using fields the type field itself doesn't carry.

    An `@misc`/`GEN` record with an arXiv `eprint` is a preprint, and so is
    anything under the bioRxiv DOI prefix — both are worth knowing before
    triage, since a preprint competes with its own published version.
    """
    if mapped in (None, "other"):
        if item.raw.get("eprint") or item.raw.get("archiveprefix"):
            return "preprint"
    if item.is_preprint_doi:
        return "preprint"
    return mapped or "other"


# ---------- RIS ----------

def parse_ris(text: str) -> list[ExportItem]:
    """RIS → items. Unknown tags are ignored, not fatal.

    Records open on `TY` and close on `ER`. A line that matches no tag is
    treated as a continuation of the previous one — RIS producers wrap long
    abstracts and occasionally titles, and dropping the continuation silently
    truncates a title, which silently changes the derived stem.
    """
    items: list[ExportItem] = []
    cur: dict[str, list[str]] | None = None
    last_tag: str | None = None

    for line in text.splitlines():
        m = _RIS_TAG.match(line)
        if m:
            tag, value = m.group(1), m.group(2).strip()
            if tag == "TY":
                cur, last_tag = {"TY": [value]}, "TY"
            elif tag == "ER":
                if cur is not None:
                    items.append(_ris_record_to_item(cur, len(items)))
                cur, last_tag = None, None
            elif cur is not None:
                cur.setdefault(tag, []).append(value)
                last_tag = tag
            continue
        if cur is not None and last_tag and line.strip():
            cur[last_tag][-1] = f"{cur[last_tag][-1]} {line.strip()}".strip()

    # A final record whose `ER` the exporter forgot is still a record.
    if cur is not None:
        items.append(_ris_record_to_item(cur, len(items)))
    return items


def _ris_record_to_item(rec: dict[str, list[str]], index: int) -> ExportItem:
    def one(*tags: str) -> str | None:
        for t in tags:
            if rec.get(t) and rec[t][0].strip():
                return rec[t][0].strip()
        return None

    item = ExportItem(
        key=one("ID") or f"ris-{index + 1}",
        title=delatex(one("TI", "T1", "CT")) or None,
        authors=[a for raw in rec.get("AU", []) or rec.get("A1", [])
                 for a in _split_authors(raw)],
        year=_clean_year(one("PY", "Y1", "DA")),
        doi=_clean_doi(one("DO", "DI")),
        venue=one("T2", "JO", "JF"),
        declared_files=[v for t in _RIS_FILE_TAGS for v in rec.get(t, [])
                        if v.strip() and not v.strip().lower().startswith("http")],
        raw={k: (v[0] if len(v) == 1 else v) for k, v in rec.items()},
    )
    item.item_type = _normalize_type(_RIS_TYPE_MAP.get(one("TY") or ""), item)
    return item


# ---------- BibTeX ----------

def parse_bibtex(text: str) -> list[ExportItem]:
    """BibTeX → items, tolerantly.

    Entries are located by `@type{` at a line start rather than by parsing the
    file as a grammar, so a malformed entry costs itself and not the rest of
    the file. Citekeys are taken verbatim: real ones contain `:` and non-ASCII,
    and validating them buys nothing — the key is a join handle here, never an
    identifier we resolve.
    """
    items: list[ExportItem] = []
    for m in re.finditer(r"(?mi)^@([A-Za-z]+)\s*\{", text):
        kind = m.group(1).lower()
        if kind in ("string", "comment", "preamble"):
            continue
        body = _brace_span(text, m.end() - 1)
        if body is None:
            continue
        key, _, rest = body.partition(",")
        item = _bibtex_fields_to_item(kind, key.strip(), rest, len(items))
        if item is not None:
            items.append(item)
    return items


def _brace_span(text: str, open_idx: int) -> str | None:
    """Content between `{` at `open_idx` and its matching `}`, or None.

    Depth-counting rather than a regex, because a brace-protected word
    (`{CRISPR}`) or an accent group (`{\\"u}`) nests inside field values and a
    non-greedy `\\{.*?\\}` would end the entry at the first inner close.
    """
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
    return None


def _bibtex_fields(body: str) -> dict[str, str]:
    """`name = value` pairs from an entry body. Values may be braced, quoted or
    bare; braced values may nest."""
    fields: dict[str, str] = {}
    i = 0
    while i < len(body):
        m = re.compile(r"([A-Za-z][A-Za-z0-9_-]*)\s*=\s*").search(body, i)
        if not m:
            break
        name = m.group(1).lower()
        j = m.end()
        if j >= len(body):
            break
        if body[j] == "{":
            val = _brace_span(body, j)
            if val is None:
                break
            i = j + len(val) + 2
        elif body[j] == '"':
            k = body.find('"', j + 1)
            if k == -1:
                break
            val, i = body[j + 1:k], k + 1
        else:
            k = body.find(",", j)
            k = len(body) if k == -1 else k
            val, i = body[j:k].strip(), k
        fields[name] = val
    return fields


def _bibtex_files(value: str) -> list[str]:
    """Better BibTeX writes `description:path:mimetype`, `;`-separated.

    Split on `;`, then take the middle colon-field when there are three. A bare
    path (no colons) is kept as-is, and a Windows path (`C:\\…`) is protected by
    requiring the triple form to have exactly three parts.
    """
    out = []
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        out.append(parts[1].strip() if len(parts) == 3 and parts[1].strip() else entry)
    return out


def _bibtex_fields_to_item(kind: str, key: str, body: str, index: int) -> ExportItem | None:
    f = _bibtex_fields(body)
    if not f:
        return None
    item = ExportItem(
        key=key or f"bib-{index + 1}",
        title=delatex(f.get("title")) or None,
        authors=_split_authors(f["author"]) if f.get("author") else [],
        year=_clean_year(f.get("year") or f.get("date")),
        doi=_clean_doi(f.get("doi")),
        venue=delatex(f.get("journal") or f.get("booktitle") or "") or None,
        declared_files=_bibtex_files(f["file"]) if f.get("file") else [],
        raw=f,
    )
    item.item_type = _normalize_type(_BIBTEX_TYPE_MAP.get(kind), item)
    return item


# ---------- CSL-JSON ----------

def _csl_year(issued) -> int | None:
    """Year from a CSL `issued` field, whatever shape it arrives in.

    The spec says a mapping with `date-parts`, and exporters also emit a bare
    string (`"2015"`) and the `{"raw": "2015"}` form. Assuming the mapping
    raised `AttributeError` on the string form, which escaped `parse_export`
    (it only catches `JSONDecodeError`) as exit 3 — an internal bug — against a
    module whose stated rule is that one bad record never takes down the file.
    """
    if isinstance(issued, dict):
        parts = issued.get("date-parts") or []
        if parts and parts[0]:
            return _clean_year(parts[0][0])
        return _clean_year(issued.get("raw") or issued.get("literal"))
    if isinstance(issued, (str, int)):
        return _clean_year(issued)
    return None


def parse_csl_json(text: str) -> list[ExportItem]:
    """CSL-JSON → items. Carries no attachment paths in any exporter we've seen,
    so pairing for this format always falls through to the content-based rungs."""
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("items") or [data]
    items = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            continue
        authors = []
        for a in rec.get("author") or []:
            if not isinstance(a, dict):
                continue
            name = " ".join(x for x in (a.get("given"), a.get("family")) if x)
            if not name and a.get("literal"):
                name = a["literal"]
            if name:
                authors.append(name.strip())
        year = _csl_year(rec.get("issued"))
        item = ExportItem(
            key=str(rec.get("id") or f"csl-{i + 1}"),
            title=(rec.get("title") or None),
            authors=authors,
            year=year,
            doi=_clean_doi(rec.get("DOI")),
            venue=rec.get("container-title") or None,
            raw=rec,
        )
        item.item_type = _normalize_type(_CSL_TYPE_MAP.get(rec.get("type") or ""), item)
        items.append(item)
    return items


# ---------- entry point ----------

def sniff_format(text: str) -> str | None:
    """Identify the format from content, not from the file extension.

    Users rename exports (`library.txt`, `My Library.ris.bak`) and some tools
    write `.txt` by default, so the extension is the least reliable signal
    available.
    """
    head = text.lstrip()[:4000]
    if not head:
        return None
    if re.search(r"(?mi)^@[A-Za-z]+\s*\{", head):
        return "bibtex"
    if re.search(r"(?m)^TY\s+-\s", head):
        return "ris"
    if head[0] in "[{":
        return "csl-json"
    return None


def parse_export(path: Path) -> tuple[str, list[ExportItem]]:
    """`(format, items)` for a bibliographic export.

    Raises `ValueError` when the format cannot be identified or the file is
    unreadable — the caller maps that onto exit code 1, since it means the
    argument was wrong, not that the environment is broken. An identified
    format that yields zero items is *not* an error here; that's a result the
    caller reports.
    """
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e}") from e

    fmt = sniff_format(text)
    if fmt == "bibtex":
        return fmt, parse_bibtex(text)
    if fmt == "ris":
        return fmt, parse_ris(text)
    if fmt == "csl-json":
        try:
            return fmt, parse_csl_json(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} looks like JSON but does not parse: {e}") from e
    raise ValueError(
        f"cannot identify the export format of {path} — expected BibTeX "
        f"(@article{{…}}), RIS (TY  - …) or CSL-JSON ([{{…}}])"
    )
