"""openFDA thin client: approved-product labels and device approvals for `scout search`.

Three endpoints, each returning its own fixed-schema record because a label, a
PMA and a 510(k) are different documents:

- `drug/label` → `LABEL_FIELDS`. Structured product labels (SPL) for drugs and
  biologics, including CBER cell and gene therapies (`product_type: CELLULAR
  THERAPY`). Limited to labels carrying a BLA or NDA, which drops OTC,
  unapproved and generic-only (ANDA) labels. A generic label that also cites
  its innovator NDA is kept. The verbatim *Indications and Usage* section is
  the triage text; the other sections are not stored.
- `device/pma` → `PMA_FIELDS`. Premarket approvals, **original approvals only**
  (`supplement_number:""`): one NGS panel can carry fifty supplements. The
  verbatim approval-order statement (`ao_statement`) is the triage text.
- `device/510k` → `K510_FIELDS`. 510(k) clearances, which on this endpoint
  also include De Novo grants (`DEN…` numbers).

Records are regulatory documents, not papers. They are discovery leads (CLAUDE.md
Rule 1): their prose fields steer what to read and never support a wiki claim.
No openFDA record is ever downloaded. Each one carries a URL for a person to
open: DailyMed for a label, the accessdata.fda.gov database page for a device.

API docs: https://open.fda.gov/apis/. Probed 2026-11-04:

- Space-separated terms are **OR'd** (`prime editing` matched 2,279 labels),
  so `build_query` joins plain words with AND, as `arxiv.build_query` does.
- **Results are not relevance-ranked.** They come back in index order (labels
  by `set_id`), so with a `limit` the first N are effectively arbitrary. Every
  endpoint is therefore sorted newest first (`effective_time:desc` /
  `decision_date:desc`); fit to the wiki is ranked locally afterwards.
- **A bare word matches any field**, prose and address lines alike. Plain
  `"gene" AND "therapy"` matched 910 labels led by Mekinist and Entresto, and
  on 510(k) it matched a 1983 radiation simulator whose contact was named
  Gene. Each plain word is therefore scoped to what the record *is*:
  `LABEL_TEXT_FIELDS` (names, pharmacologic class, Indications, Description,
  Mechanism of Action — 142 labels led by Encelto, Elevidys, Lyfgenia),
  `PMA_TEXT_FIELDS`, `K510_TEXT_FIELDS`. A query written in field syntax
  passes through unscoped, which is how to search a whole record.
- No match is HTTP 404 `NOT_FOUND`, which the transport returns as `{}`. A
  misspelled field name is also a plain 404, so it reads as zero results.
- `limit` > 1000 is HTTP 400. A Lucene syntax error is HTTP 500 with
  `"details": "[parse_exception] …"`; `_parse_error` turns that one 500 into
  a rejected query (exit 1) rather than an outage (exit 2).
- Date ranges: `effective_time:[20250101 TO 20261104]` on labels (the SPL
  revision date, *not* the approval date — openFDA's label endpoint carries
  none); `decision_date:[2025-01-01 TO 2026-11-04]` on both device endpoints.
- There is no field projection: one full label is ~110 KB. The cache holds
  the parsed records only, so label prose beyond Indications never lands on
  disk.
- accessdata.fda.gov answers curl with a 302 to an abuse-detection page, so its
  URLs are for a browser. DailyMed's `lookup.cfm?setid=` answers 200.
- 510(k) fields used here: `k_number`, `device_name`, `applicant`,
  `decision_date`, `decision_code` (`SESE`, `DENG`, …),
  `decision_description` ("Substantially Equivalent"; "Unknown" on De Novo
  rows), `clearance_type` (`Traditional`, `Direct`, …), `advisory_committee_description`,
  `product_code`, `openfda.device_class`.

`OPENFDA_API_KEY` (optional) lifts the per-IP limits of 240 requests/minute
and 1,000/day. It travels as a secret query parameter: never in argv, a log
line, or a cache filename.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
from datetime import date

from ..paths import web_cache_dir
from ._cache import read_cache, safe_cache_key, write_stamped_cache
from ._http import curl_json

API_BASE = "https://api.fda.gov"
DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}"
PMA_URL = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpma/pma.cfm?id={number}"
PMN_URL = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID={number}"
DENOVO_URL = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/denovo.cfm?ID={number}"
POLITE_SLEEP = 0.3
KEYED_SLEEP = 0.1

LABEL_FILTER = "openfda.application_number:(BLA* OR NDA*)"
PMA_FILTER = 'supplement_number:""'
#: Where a plain query word must appear on a label: what the product is and is for.
LABEL_TEXT_FIELDS = (
    "openfda.brand_name", "openfda.generic_name", "openfda.pharm_class_epc",
    "openfda.pharm_class_moa", "indications_and_usage", "description", "mechanism_of_action",
)
PMA_TEXT_FIELDS = ("trade_name", "generic_name", "ao_statement", "openfda.device_name")
#: 510(k) records carry no summary text; the device name and its product-code
#: name are what the record is about. The applicant is left out (a word would
#: match a company name, which is not a topic).
K510_TEXT_FIELDS = ("device_name", "openfda.device_name")

LABEL_FIELDS = (
    "set_id", "spl_id", "effective_date", "brand_names", "generic_names", "manufacturer",
    "application_numbers", "product_type", "routes", "pharm_classes", "has_boxed_warning",
    "indications", "url",
)
PMA_FIELDS = (
    "pma_number", "trade_name", "generic_name", "applicant", "decision_date",
    "decision_code", "advisory_committee", "product_code", "device_class",
    "expedited_review", "ao_statement", "url",
)
K510_FIELDS = (
    "k_number", "device_name", "applicant", "decision_date", "decision_code",
    "decision_description", "clearance_type", "advisory_committee", "product_code",
    "device_class", "is_de_novo", "url",
)
#: Application numbers a label's lead is keyed by. ANDA numbers are kept out:
#: the label filter already requires a BLA or NDA, and a generic's ANDA would
#: give one innovator product a second identity.
_APP_RE = re.compile(r"(?:BLA|NDA)\d{6}")
#: openFDA field syntax (`indications_and_usage:`, `openfda.brand_name:`,
#: `_exists_:`), so a query that already uses it is passed through untouched.
#: A colon followed by a space (`Kymriah: tisagenlecleucel`) is prose, not a field.
_FIELD_PREFIX_RE = re.compile(r"(?:^|[\s(])[A-Za-z_][\w.]*:(?=\S)")
#: A quoted phrase, a grouping parenthesis, or a bare word.
_TOKEN_RE = re.compile(r'"[^"]+"|[()]|[^\s()"]+')
_BOOLEAN = {"AND", "OR", "NOT"}


def _parse_error(status: str, body: str) -> bool:
    """openFDA's answer to a malformed query: HTTP 500 naming `parse_exception`."""
    if status != "500":
        return False
    try:
        details = (json.loads(body).get("error") or {}).get("details") or ""
    except (ValueError, AttributeError):
        return False
    return "parse_exception" in str(details)


def _secret() -> tuple[dict[str, str], float]:
    key = (os.environ.get("OPENFDA_API_KEY") or "").strip()
    return ({"api_key": key} if key else {}), (KEYED_SLEEP if key else POLITE_SLEEP)


def _curl_json(url: str, *, secret_query=None, retries: int = 3) -> dict:
    return curl_json(url, provider="openfda", retries=retries, secret_query=secret_query,
                     reject_400=True, reject_when=_parse_error)


def build_query(query: str, *, fields: tuple[str, ...] = (), filters: str = "",
                date_field: str = "", since: date | None = None,
                today: date | None = None) -> str:
    """Plain words → `"w1" AND "w2"`; field syntax passes through.

    Quoted phrases stay phrases; `AND`/`OR`/`NOT` and parentheses are kept.
    Every other word is quoted, so Lucene metacharacters in it (`car-t`,
    `il-2/il-15`) are literal. With `fields`, each word must appear in one of
    them: `(f1:"w" OR f2:"w")`. The endpoint's own filter and, with `since`, a
    `date_field` range are AND-ed on outside the user's query, so a user's
    `OR` cannot escape them.
    """
    q = query.strip()
    if not _FIELD_PREFIX_RE.search(q):
        parts = []
        for t in _TOKEN_RE.findall(q):
            if t.upper() in _BOOLEAN:
                parts.append(t.upper())
            elif t in "()":
                parts.append(t)
            elif (word := t.strip('"').replace("\\", " ").strip()):
                parts.append("(" + " OR ".join(f'{f}:"{word}"' for f in fields) + ")"
                             if fields else f'"{word}"')
        joined: list[str] = []
        for part in parts:
            if (joined and joined[-1] not in _BOOLEAN and joined[-1] != "("
                    and part not in _BOOLEAN and part != ")"):
                joined.append("AND")
            joined.append(part)
        q = " ".join(joined)
    clauses = [f"({q})"] if q else []
    if filters:
        clauses.append(filters)
    if since and date_field:
        end = today or date.today()
        if date_field == "effective_time":
            lo, hi = since.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        else:
            lo, hi = since.isoformat(), end.isoformat()
        clauses.append(f"{date_field}:[{lo} TO {hi}]")
    return " AND ".join(clauses)


#: C1 control characters: Windows-1252 bytes stored as Latin-1 upstream, so a
#: trademark sign arrives as U+0099 (`Signatera\x99 CDx`). Decoded as the
#: cp1252 they were; the five bytes cp1252 leaves undefined are dropped.
_C1 = {c: bytes([c]).decode("cp1252", errors="ignore") or None for c in range(0x80, 0xA0)}


def _str(value) -> str:
    return str(value or "").translate(_C1)


def _first(value) -> str:
    if isinstance(value, list):
        return _str(value[0]) if value else ""
    return _str(value)


def _strs(value) -> list[str]:
    if isinstance(value, list):
        return [_str(v) for v in value if v]
    return [_str(value)] if value else []


def _iso(yyyymmdd: str) -> str:
    v = str(yyyymmdd or "")
    return f"{v[:4]}-{v[4:6]}-{v[6:8]}" if re.fullmatch(r"\d{8}", v) else v


def parse_label(r: dict) -> dict:
    """One `drug/label` result → a `LABEL_FIELDS` record."""
    o = r.get("openfda") or {}
    set_id = str(r.get("set_id") or "")
    return {
        "set_id": set_id,
        "spl_id": str(r.get("id") or ""),
        "effective_date": _iso(r.get("effective_time")),
        "brand_names": _strs(o.get("brand_name")),
        "generic_names": _strs(o.get("generic_name")),
        "manufacturer": _first(o.get("manufacturer_name")),
        "application_numbers": [a for a in _strs(o.get("application_number")) if _APP_RE.fullmatch(a)],
        "product_type": _first(o.get("product_type")),
        "routes": _strs(o.get("route")),
        "pharm_classes": _strs(o.get("pharm_class_epc")),
        "has_boxed_warning": bool(r.get("boxed_warning")),
        "indications": " ".join(_first(r.get("indications_and_usage")).split()),
        "url": DAILYMED_URL.format(set_id=set_id) if set_id else None,
    }


def parse_pma(r: dict) -> dict:
    """One `device/pma` result → a `PMA_FIELDS` record."""
    number = str(r.get("pma_number") or "")
    supplement = str(r.get("supplement_number") or "")
    return {
        "pma_number": number,
        "trade_name": _str(r.get("trade_name")),
        "generic_name": _str(r.get("generic_name")),
        "applicant": _str(r.get("applicant")),
        "decision_date": str(r.get("decision_date") or ""),
        "decision_code": str(r.get("decision_code") or ""),
        "advisory_committee": _str(r.get("advisory_committee_description")),
        "product_code": str(r.get("product_code") or ""),
        "device_class": str((r.get("openfda") or {}).get("device_class") or ""),
        "expedited_review": r.get("expedited_review_flag") == "Y",
        "ao_statement": " ".join(_str(r.get("ao_statement")).split()),
        "url": PMA_URL.format(number=number + supplement) if number else None,
    }


def parse_510k(r: dict) -> dict:
    """One `device/510k` result → a `K510_FIELDS` record."""
    number = str(r.get("k_number") or "")
    de_novo = number.upper().startswith("DEN")
    return {
        "k_number": number,
        "device_name": _str(r.get("device_name")),
        "applicant": _str(r.get("applicant")),
        "decision_date": str(r.get("decision_date") or ""),
        "decision_code": str(r.get("decision_code") or ""),
        "decision_description": _str(r.get("decision_description")),
        "clearance_type": str(r.get("clearance_type") or ""),
        "advisory_committee": _str(r.get("advisory_committee_description")),
        "product_code": str(r.get("product_code") or ""),
        "device_class": str((r.get("openfda") or {}).get("device_class") or ""),
        "is_de_novo": de_novo,
        "url": (DENOVO_URL if de_novo else PMN_URL).format(number=number) if number else None,
    }


def _search(endpoint: str, parse, query: str, *, fields: tuple[str, ...] = (), filters: str,
            date_field: str, limit: int, since: date | None, max_age_days: float) -> list[dict]:
    params = {"search": build_query(query, fields=fields, filters=filters,
                                    date_field=date_field, since=since),
              "sort": f"{date_field}:desc", "limit": str(limit)}
    url = f"{API_BASE}/{endpoint}.json?{urllib.parse.urlencode(params)}"
    cache_dir = web_cache_dir() / "search"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"openfda__{safe_cache_key(url)}.json"
    data = read_cache(cache, max_age_days=max_age_days)
    if data is None:
        secret, sleep = _secret()
        raw = _curl_json(url, secret_query=secret or None)
        data = {"results": [parse(r) for r in raw.get("results") or []]}
        write_stamped_cache(cache, data)
        time.sleep(sleep)
    return list(data.get("results") or [])


def search_labels(query: str, *, limit: int = 20, since: date | None = None,
                  max_age_days: float = 1) -> list[dict]:
    """BLA/NDA labels matching `query`, newest revision first. `since` is on the revision date."""
    return _search("drug/label", parse_label, query, fields=LABEL_TEXT_FIELDS, filters=LABEL_FILTER,
                   date_field="effective_time", limit=limit, since=since,
                   max_age_days=max_age_days)


def search_pma(query: str, *, limit: int = 20, since: date | None = None,
               max_age_days: float = 1) -> list[dict]:
    """Original PMA approvals matching `query`, newest first. `since` is on the decision date."""
    return _search("device/pma", parse_pma, query, fields=PMA_TEXT_FIELDS, filters=PMA_FILTER,
                   date_field="decision_date", limit=limit, since=since,
                   max_age_days=max_age_days)


def search_510k(query: str, *, limit: int = 20, since: date | None = None,
                max_age_days: float = 1) -> list[dict]:
    """510(k) clearances and De Novo grants, newest first. `since` is on the decision date."""
    return _search("device/510k", parse_510k, query, fields=K510_TEXT_FIELDS, filters="",
                   date_field="decision_date", limit=limit, since=since,
                   max_age_days=max_age_days)
