"""Europe PMC REST thin client — bioRxiv/medRxiv keyword search and OA status.

Two uses:

- **Preprint search** (`search_preprints`). api.biorxiv.org has no keyword
  search, so `scout search --source biorxiv/medrxiv` queries Europe PMC's
  preprint index (`SRC:PPR`) restricted to those publishers, keeping the
  verbatim abstract for triage. Results are discovery leads (Rule 1).
- **Open-access status** (`oa_status`), for `scout search fetch`:
  `isOpenAccess`, `license`, and the `fullTextUrlList` entries Europe PMC
  marks open access — the structured facts the download decision rests on.

API docs: https://europepmc.org/RestfulWebService. Probed 2026-10-03:
`PUBLISHER:"bioRxiv"` filters preprints by server; `FIRST_PDATE:[a TO b]`
filters by first posting date; a busy index answers HTTP 200 with
`{"errCode": 503}` in the body, which is treated as an outage. Europe PMC
reports `isOpenAccess: "N"` for bioRxiv preprints whatever their licence, so
preprint licences come from `biorxiv.lookup`, not from here. Its
`?pdf=render` links answer HTTP 403 to non-browser clients, so `fetch` lists
them for a manual download rather than pretending it can retrieve them.
"""

from __future__ import annotations

import time
import urllib.parse
from datetime import date

from ..paths import web_cache_dir
from ._cache import read_cache, safe_cache_key, write_stamped_cache
from ._http import StructuredProviderUnavailable, curl_json

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
POLITE_SLEEP = 0.2
SERVERS = {"biorxiv": "bioRxiv", "medrxiv": "medRxiv"}
SEARCH_FIELDS = (
    "id", "doi", "server", "title", "authors", "pub_date", "year", "abstract",
)
OA_FIELDS = ("is_open_access", "license", "pmid", "pmcid", "doi", "pdf_urls")


def _curl_json(url: str, retries: int = 3) -> dict:
    return curl_json(url, provider="europepmc", retries=retries, reject_400=True)


def _query(params: dict, *, max_age_days: float | None) -> dict:
    url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
    cache_dir = web_cache_dir() / "search"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"europepmc__{safe_cache_key(url)}.json"
    data = read_cache(cache, max_age_days=max_age_days)
    if data is not None:
        return data
    data = _curl_json(url)
    if data.get("errCode"):
        raise StructuredProviderUnavailable(
            f"europepmc API unavailable ({data.get('errCode')}: {data.get('errMsg') or 'no message'})"
        )
    write_stamped_cache(cache, data)
    time.sleep(POLITE_SLEEP)
    return data


def _authors(record: dict) -> list[str]:
    names = []
    for a in (record.get("authorList") or {}).get("author") or []:
        name = a.get("fullName") or a.get("collectiveName") or ""
        if name:
            names.append(str(name))
    if names:
        return names
    return [s.strip() for s in str(record.get("authorString") or "").rstrip(".").split(",") if s.strip()]


def search_preprints(query: str, *, servers: list[str], limit: int = 20,
                     since: date | None = None, max_age_days: float = 1) -> list[dict]:
    """Preprints on the named servers matching `query`, relevance-ordered."""
    publishers = " OR ".join(f'PUBLISHER:"{SERVERS[s]}"' for s in servers)
    q = f"({query}) AND SRC:PPR AND ({publishers})"
    if since:
        q += f" AND FIRST_PDATE:[{since.isoformat()} TO {date.today().isoformat()}]"
    data = _query({"query": q, "format": "json", "resultType": "core",
                   "pageSize": str(limit)}, max_age_days=max_age_days)
    out = []
    for r in (data.get("resultList") or {}).get("result") or []:
        publisher = str((r.get("bookOrReportDetails") or {}).get("publisher") or "")
        pub_date = str(r.get("firstPublicationDate") or "")
        year = str(r.get("pubYear") or "")
        out.append({
            "id": str(r.get("id") or ""),
            "doi": str(r.get("doi") or "").lower() or None,
            "server": publisher.lower() or None,
            "title": " ".join(str(r.get("title") or "").split()),
            "authors": _authors(r),
            "pub_date": pub_date,
            "year": int(year) if year.isdigit() else None,
            "abstract": str(r.get("abstractText") or ""),
        })
    return out


def oa_status(key: str) -> dict:
    """Open-access status for `pmid:<n>`, `pmcid:pmc<n>`, or a DOI.

    Fixed schema (`OA_FIELDS`); `is_open_access` is None when Europe PMC has
    no record. DOI-keyed lookups never expire (a DOI's licence is stable).
    """
    if key.startswith("pmid:"):
        q = f"EXT_ID:{key[5:]} AND SRC:MED"
    elif key.startswith("pmcid:"):
        q = f"PMCID:{key[6:].upper()}"
    else:
        q = f'DOI:"{key}"'
    data = _query({"query": q, "format": "json", "resultType": "core", "pageSize": "1"},
                  max_age_days=30)
    results = (data.get("resultList") or {}).get("result") or []
    out = {"is_open_access": None, "license": None, "pmid": None, "pmcid": None,
           "doi": None, "pdf_urls": []}
    if not results:
        return out
    r = results[0]
    out.update({
        "is_open_access": str(r.get("isOpenAccess") or "").upper() == "Y",
        "license": r.get("license") or None,
        "pmid": r.get("pmid") or None,
        "pmcid": r.get("pmcid") or None,
        "doi": str(r.get("doi") or "").lower() or None,
        "pdf_urls": [
            str(u.get("url")) for u in (r.get("fullTextUrlList") or {}).get("fullTextUrl") or []
            if u.get("availabilityCode") == "OA" and u.get("documentStyle") == "pdf" and u.get("url")
        ],
    })
    return out
