"""ORCID Public API v3.0 — thin client.

Narrow-use provider for the Rule-1 structured-API whitelist. Exposes only
structured fields: canonical given-name / family-name / credit-name,
other-names list, and the most-recent employment (organization + role +
start date). Prose fields served by ORCID — `biography`, `researcher-urls`,
`keywords` — are deliberately NOT re-exposed by this helper. Authentication
is not required for public records.

API docs: https://info.orcid.org/documentation/api-tutorials/api-tutorial-read-data-on-a-record/

Uses the same `curl`-subprocess + on-disk cache pattern as
`pubmed.py` and `biorxiv.py`.
"""

from __future__ import annotations

import re
import time
import urllib.parse
from datetime import date
from pathlib import Path

from ..paths import web_cache_dir
from ._cache import negative_sentinel, read_cache, safe_cache_key, write_cache
from ._http import StructuredProviderUnavailable, curl_json

ORCID_BASE = "https://pub.orcid.org/v3.0"
POLITE_SLEEP = 0.3  # ORCID public API: 24 req/sec is safe; 0.3s is comfortably below.

ORCID_ID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")


def _curl_json(url: str, retries: int = 3) -> dict | None:
    return curl_json(
        url, provider="orcid", retries=retries, headers=("Accept: application/json",)
    )


def _cache_path(kind: str, key: str) -> Path:
    cache_dir = web_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"orcid_{kind}__{safe_cache_key(key)}.json"


def _empty_record(orcid_id: str = "") -> dict:
    """Fixed-shape empty record so callers can rely on the schema."""
    return {
        "orcid": orcid_id,
        "given_names": "",
        "family_name": "",
        "credit_name": "",
        "other_names": [],
        "latest_affiliation": None,
        "source": "orcid",
        "fetched_at": date.today().isoformat(),
    }


def lookup_by_id(orcid_id: str) -> dict:
    """Fetch personal-details + employments for a specific ORCID.

    Returns a fixed-shape record:
      {
        "orcid": str,                      # normalized "0000-0000-0000-0000"
        "given_names": str,
        "family_name": str,
        "credit_name": str,                # "" if not set
        "other_names": [str, ...],
        "latest_affiliation": {            # None if no public employments
          "organization": str,
          "role": str,
          "department": str,
          "start_year": int | None,
          "end_year": int | None,          # None = current
        } | None,
        "source": "orcid",
        "fetched_at": "YYYY-MM-DD",
      }

    On HTTP 404 (unknown ORCID) returns an empty-shape record with
    the requested `orcid` echoed back.
    """
    orcid_norm = (orcid_id or "").strip().upper()
    if not ORCID_ID_RE.match(orcid_norm):
        out = _empty_record(orcid_norm)
        return out

    out = _empty_record(orcid_norm)

    # Personal details
    pd_cache = _cache_path("personal", orcid_norm)
    pd = read_cache(pd_cache)
    if pd is None:
        pd = _curl_json(f"{ORCID_BASE}/{orcid_norm}/personal-details")
        if pd is None:
            raise StructuredProviderUnavailable("orcid API returned no response")
        time.sleep(POLITE_SLEEP)
        # Empty dict = HTTP 404 (unknown ORCID) — TTL it so a since-registered
        # ORCID isn't rejected forever; a real 200 payload always has a `name`.
        write_cache(pd_cache, negative_sentinel(pd) if not pd else pd)

    name = pd.get("name") or {}
    gn = (name.get("given-names") or {}).get("value") or ""
    fn = (name.get("family-name") or {}).get("value") or ""
    cn = (name.get("credit-name") or {}).get("value") or ""
    other = (pd.get("other-names") or {}).get("other-name") or []
    out["given_names"] = str(gn)
    out["family_name"] = str(fn)
    out["credit_name"] = str(cn)
    out["other_names"] = [str(o.get("content") or "") for o in other if o.get("content")]
    # Biography is deliberately dropped — prose, banned by Rule 1.

    # Employments — latest only (simplest structural signal).
    em_cache = _cache_path("employments", orcid_norm)
    em = read_cache(em_cache)
    if em is None:
        em = _curl_json(f"{ORCID_BASE}/{orcid_norm}/employments")
        if em is None:
            raise StructuredProviderUnavailable("orcid API returned no response")
        time.sleep(POLITE_SLEEP)
        # Same 404-vs-real-payload TTL as the personal-details cache above.
        write_cache(em_cache, negative_sentinel(em) if not em else em)

    groups = em.get("affiliation-group") or []
    if groups:
        # Pick the group with the latest start-date (most-recent affiliation).
        def _start_year(group):
            summaries = group.get("summaries") or []
            if not summaries:
                return -1
            s = summaries[0].get("employment-summary") or {}
            sd = s.get("start-date") or {}
            y = (sd.get("year") or {}).get("value") if isinstance(sd, dict) else None
            try:
                return int(y) if y else -1
            except (TypeError, ValueError):
                return -1

        latest_group = max(groups, key=_start_year)
        summaries = latest_group.get("summaries") or []
        if summaries:
            s = summaries[0].get("employment-summary") or {}
            org = (s.get("organization") or {}).get("name") or ""
            role = s.get("role-title") or ""
            dept = s.get("department-name") or ""
            sd = s.get("start-date") or {}
            ed = s.get("end-date")
            sy = (sd.get("year") or {}).get("value") if isinstance(sd, dict) else None
            ey = (ed.get("year") or {}).get("value") if (isinstance(ed, dict) and ed) else None
            try:
                sy_int = int(sy) if sy else None
            except (TypeError, ValueError):
                sy_int = None
            try:
                ey_int = int(ey) if ey else None
            except (TypeError, ValueError):
                ey_int = None
            out["latest_affiliation"] = {
                "organization": str(org),
                "role": str(role),
                "department": str(dept),
                "start_year": sy_int,
                "end_year": ey_int,
            }
    return out


def search_by_name(given: str = "", family: str = "", limit: int = 5) -> list[dict]:
    """Search ORCID for candidates matching a (given, family) name pair.

    Returns up to `limit` structured records (same shape as lookup_by_id).
    Empty list on zero matches; provider outages raise
    StructuredProviderUnavailable. Use when you don't know the ORCID ID but do
    know the author's name — typical entry point when normalising YAML
    `authors` against a canonical source.
    """
    given = (given or "").strip()
    family = (family or "").strip()
    if not family and not given:
        return []

    terms = []
    if given:
        terms.append(f"given-names:{given}")
    if family:
        terms.append(f"family-name:{family}")
    query = "+AND+".join(terms)
    cache = _cache_path("search", f"{given}__{family}__{limit}")
    data = read_cache(cache)
    if data is None:
        url = (f"{ORCID_BASE}/search/"
               f"?q={urllib.parse.quote(query, safe=':+')}&rows={limit}")
        data = _curl_json(url)
        if data is None:
            raise StructuredProviderUnavailable("orcid API returned no response")
        time.sleep(POLITE_SLEEP)
        # Empty dict = HTTP 404; TTL it like the lookups above.
        write_cache(cache, negative_sentinel(data) if not data else data)

    results = data.get("result") or []
    records: list[dict] = []
    for r in results[:limit]:
        oid = (r.get("orcid-identifier") or {}).get("path") or ""
        if oid:
            records.append(lookup_by_id(oid))
    return records
