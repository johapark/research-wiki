"""bioRxiv / medRxiv details API — thin client.

Narrow-use provider for the Rule-1 structured-API whitelist. Exposes only
structured fields: server, version, date, type, category, `published`
(the journal-version DOI if bioRxiv's Crossref cross-referencing has
detected one). Prose fields present in the API response — `abstract`
(verbatim authors' text, comparable to S2's abstract exception) and
`jatsxml` (URL to full-text XML, banned) — are deliberately NOT re-exposed
by this helper. Callers get the fields they need for structural
decisions (preprint↔journal pairing, version tracking) and nothing else.

API docs: https://api.biorxiv.org/

Uses the same `curl`-subprocess + on-disk cache pattern as
`crossref.py` and `pubmed.py`.
"""

from __future__ import annotations

import time
import urllib.parse
from datetime import date

from ..paths import web_cache_dir
from ._cache import negative_sentinel, read_cache, safe_cache_key, write_cache
from ._http import StructuredProviderUnavailable, curl_json

BIORXIV_BASE = "https://api.biorxiv.org/details"
POLITE_SLEEP = 0.4


def _curl_json(url: str, retries: int = 3) -> dict | None:
    return curl_json(url, provider="biorxiv", retries=retries)


def _fetch_server(server: str, doi: str) -> dict | None:
    cache_dir = web_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"biorxiv_{server}__{safe_cache_key(doi)}.json"
    cached = read_cache(cache)
    if cached is not None:
        return cached
    url = f"{BIORXIV_BASE}/{server}/{urllib.parse.quote(doi, safe='/.')}"
    data = _curl_json(url)
    if data is None:
        # Defensive for injected/custom transports that still use the old
        # sentinel. The built-in transport raises this directly.
        raise StructuredProviderUnavailable("biorxiv API returned no response")
    time.sleep(POLITE_SLEEP)
    # 404 or empty collection = "not on this server (yet)"; TTL it so a
    # preprint bioRxiv hasn't posted at query time can be re-checked later.
    if not (data.get("collection") or []):
        data = negative_sentinel(data)
    write_cache(cache, data)
    return data


def lookup(doi: str) -> dict:
    """Return a structured preprint record for a DOI.

    Always returns a record with a fixed schema (all keys present) so
    callers can rely on shape:

    {
      "doi": str,                    # normalized lowercase DOI
      "server": "biorxiv" | "medrxiv" | None,  # None if not found on either
      "title": str,                  # structured — safe to re-expose
      "version": str,                # latest version number as string
      "date_posted": str,            # YYYY-MM-DD from bioRxiv's `date`
      "category": str,               # bioRxiv taxonomy, e.g. "molecular biology"
      "type": str,                   # "new results" / "confirmatory results" / etc.
      "published_doi": str | None,   # journal DOI if detected, else None
      "source": "biorxiv",
      "fetched_at": "YYYY-MM-DD",
    }

    If the DOI is not on bioRxiv or medRxiv (common for non-preprint DOIs),
    `server` is None and structural fields are empty.
    """
    out = {
        "doi": (doi or "").strip().lower(),
        "server": None,
        "title": "",
        "version": "",
        "date_posted": "",
        "category": "",
        "type": "",
        "published_doi": None,
        "source": "biorxiv",
        "fetched_at": date.today().isoformat(),
    }
    if not out["doi"]:
        return out

    for server in ("biorxiv", "medrxiv"):
        data = _fetch_server(server, out["doi"])
        col = data.get("collection") or []
        if not col:
            continue  # empty collection = not on this server
        # Use the latest version (bioRxiv appends them chronologically).
        latest = col[-1]
        out["server"] = (latest.get("server") or server).lower()
        out["title"] = str(latest.get("title") or "")
        out["version"] = str(latest.get("version") or "")
        out["date_posted"] = str(latest.get("date") or "")
        out["category"] = str(latest.get("category") or "")
        out["type"] = str(latest.get("type") or "")
        pub = latest.get("published")
        # bioRxiv uses "NA" string for "no journal version detected yet".
        if pub and pub != "NA":
            out["published_doi"] = str(pub).lower()
        return out
    return out
