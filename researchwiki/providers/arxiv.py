"""arXiv API thin client — keyword search for `scout search`.

Returns arXiv id + version, title, authors, submitted/updated dates, primary
category, the journal DOI / journal-ref the authors registered, and the
verbatim abstract (`summary`). The author `comment` is not read. Results are
discovery leads (CLAUDE.md Rule 1): they steer which PDF to ingest and never
support a claim themselves.

API docs: https://info.arxiv.org/help/api/user-manual.html. The API asks for
one request every three seconds and answers bursts with HTTP 503.

**Errors arrive as HTTP 200.** A malformed query returns a one-entry feed
whose `<id>` points at `…/api/errors#…` and whose summary is the message, so
the parser checks for that before treating the entry as a paper.
"""

from __future__ import annotations

import re
import time
import urllib.parse
from datetime import date

from ..paths import web_cache_dir
from ._cache import read_text_cache, safe_cache_key, write_text_cache
from ._http import ProviderRequestRejected, curl_body
from ._xml import parse_xml

API_URL = "https://export.arxiv.org/api/query"
POLITE_SLEEP = 3.0
_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
SEARCH_FIELDS = (
    "arxiv_id", "version", "doi", "title", "authors", "published", "updated",
    "year", "primary_category", "journal_doi", "journal_ref", "abstract",
)
#: `ti:`, `abs:`, `au:`, `cat:`, … — a query that already uses arXiv's field
#: syntax is passed through untouched.
_FIELD_PREFIX_RE = re.compile(r"\b(?:ti|au|abs|co|jr|cat|rn|id|all|submittedDate):", re.I)
_ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(?:v(\d+))?$")
_BOOLEAN = {"AND", "OR", "ANDNOT"}


def _curl_body(url: str, retries: int = 3) -> str | None:
    return curl_body(url, provider="arxiv", retries=retries, reject_400=True)


def build_query(query: str, since: date | None = None) -> str:
    """Plain words → `all:"w1" AND all:"w2"`; field syntax passes through.

    Quoted phrases in the input stay phrases. `since` adds a
    `submittedDate:[YYYYMMDD0000 TO <today>2359]` clause.
    """
    q = query.strip()
    if not _FIELD_PREFIX_RE.search(q):
        terms = re.findall(r'"[^"]+"|\S+', q)
        parts = []
        for t in terms:
            if t.upper() in _BOOLEAN:
                parts.append(t.upper())
                continue
            phrase = t.strip('"')
            parts.append(f'all:"{phrase}"' if " " in phrase else f"all:{phrase}")
        joined: list[str] = []
        for part in parts:
            if joined and joined[-1] not in _BOOLEAN and part not in _BOOLEAN:
                joined.append("AND")
            joined.append(part)
        q = " ".join(joined)
    if since:
        q = (f"({q}) AND submittedDate:[{since.strftime('%Y%m%d')}0000 TO "
             f"{date.today().strftime('%Y%m%d')}2359]")
    return q


def _text(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_feed(xml_text: str) -> list[dict]:
    """Atom feed → fixed-schema records (`SEARCH_FIELDS`).

    Raises `ProviderRequestRejected` for an error entry: arXiv's way of
    saying the query was malformed.
    """
    root = parse_xml(xml_text, provider="arxiv")
    out: list[dict] = []
    for entry in root.findall("a:entry", _NS):
        raw_id = _text(entry.find("a:id", _NS))
        if "/api/errors" in raw_id:
            raise ProviderRequestRejected(
                f"arxiv rejected the query: {_text(entry.find('a:summary', _NS)) or raw_id}"
            )
        m = _ID_RE.search(raw_id)
        if not m:
            continue
        arxiv_id, version = m.group(1), m.group(2)
        published = _text(entry.find("a:published", _NS))[:10]
        primary = entry.find("arxiv:primary_category", _NS)
        journal_doi = _text(entry.find("arxiv:doi", _NS)).lower() or None
        out.append({
            "arxiv_id": arxiv_id,
            "version": version or "",
            "doi": f"10.48550/arxiv.{arxiv_id.lower()}",
            "title": _text(entry.find("a:title", _NS)),
            "authors": [_text(a.find("a:name", _NS)) for a in entry.findall("a:author", _NS)],
            "published": published,
            "updated": _text(entry.find("a:updated", _NS))[:10],
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "primary_category": primary.get("term", "") if primary is not None else "",
            "journal_doi": journal_doi,
            "journal_ref": _text(entry.find("arxiv:journal_ref", _NS)),
            "abstract": _text(entry.find("a:summary", _NS)),
        })
    return out


def search(query: str, *, limit: int = 20, since: date | None = None,
           max_age_days: float = 1) -> list[dict]:
    """Relevance-ordered keyword search. Records follow `SEARCH_FIELDS`."""
    params = {"search_query": build_query(query, since), "start": "0",
              "max_results": str(limit), "sortBy": "relevance", "sortOrder": "descending"}
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    cache_dir = web_cache_dir() / "search"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"arxiv_query__{safe_cache_key(url)}.json"
    body = read_text_cache(cache, max_age_days=max_age_days)
    if body is None:
        body = _curl_body(url) or ""
        records = parse_feed(body) if body else []
        # Cache only what parsed: an error entry raises before this line.
        write_text_cache(cache, body, url=url, fmt="atom")
        time.sleep(POLITE_SLEEP)
        return records
    return parse_feed(body) if body else []
