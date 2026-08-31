"""PubMed E-utilities thin client — retraction status only.

Narrow-use provider, not a full metadata source. PubMed is included in the
Rule-1 structured-API whitelist specifically for retraction / erratum
detection (the `esummary.pubtype` field flags retracted or retracting
publications with fixed NLM-defined strings). We don't fetch titles /
authors / abstracts from here — S2 and Crossref cover those with less
latency and no NCBI rate limit.

The returned record is a fixed schema: {pmid, retracted, retraction_of_pmid,
retracted_by_pmid, pubtypes, pubdate, source, fetched_at}. No free-form
prose fields are exposed — prose-based retraction notes remain behind the
Rule-1 prose ban.

Uses the same `curl`-subprocess + on-disk cache pattern as
`crossref.py` and `semantic_scholar.py`.
"""

from __future__ import annotations

import time
import urllib.parse
from datetime import date

from ..paths import web_cache_dir
from ._cache import negative_sentinel, read_cache, safe_cache_key, write_cache
from ._http import StructuredProviderUnavailable, curl_json

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Polite-pool rate: NCBI asks for ≤3 req/sec without an API key.
POLITE_SLEEP = 0.4


def _curl_json(url: str, retries: int = 3) -> dict | None:
    """Fetch JSON via curl subprocess. Mirrors crossref.py's pattern to
    bypass Python stdlib SSL issues behind enterprise proxies."""
    return curl_json(url, provider="pubmed", retries=retries)


def doi_to_pmid(doi: str) -> str | None:
    """Return the PubMed ID for a DOI, or None if not indexed in PubMed."""
    if not doi:
        return None
    doi_norm = doi.strip().lower()
    cache_dir = web_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"pubmed_esearch__{safe_cache_key(doi_norm)}.json"
    data = read_cache(cache)
    if data is None:
        url = (f"{EUTILS_BASE}/esearch.fcgi"
               f"?db=pubmed&retmode=json&term={urllib.parse.quote(doi_norm)}%5Baid%5D")
        data = _curl_json(url)
        if data is None:
            raise StructuredProviderUnavailable("pubmed API returned no response")
        time.sleep(POLITE_SLEEP)
        # Empty dict = HTTP 404; TTL it so a DOI PubMed hasn't indexed yet
        # (common for fresh publications) is re-checked later.
        write_cache(cache, negative_sentinel(data) if not data else data)
    ids = (data.get("esearchresult") or {}).get("idlist") or []
    return ids[0] if ids else None


def retraction_status(doi: str) -> dict:
    """Return structured retraction status for a DOI.

    Schema (all keys always present so callers can rely on shape):
    {
      "doi": str,
      "pmid": str | None,           # None if not indexed in PubMed
      "retracted": bool,            # pubtype includes "Retracted Publication"
      "is_retraction_notice": bool, # pubtype includes "Retraction of Publication"
      "pubtypes": list[str],        # raw NLM pubtype strings
      "pubdate": str,               # YYYY-MM-DD from esummary.pubdate (possibly partial)
      "source": "pubmed",
      "fetched_at": "YYYY-MM-DD",
    }

    A missing PMID returns the record with pmid=None and retracted=False.
    Transport failure raises StructuredProviderUnavailable so it cannot be
    mistaken for evidence that the paper is not retracted.
    """
    out = {
        "doi": doi.strip().lower() if doi else "",
        "pmid": None,
        "retracted": False,
        "is_retraction_notice": False,
        "pubtypes": [],
        "pubdate": "",
        "source": "pubmed",
        "fetched_at": date.today().isoformat(),
    }
    pmid = doi_to_pmid(doi)
    if not pmid:
        return out
    out["pmid"] = pmid

    cache_dir = web_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"pubmed_esummary__{pmid}.json"
    data = read_cache(cache)
    if data is None:
        url = f"{EUTILS_BASE}/esummary.fcgi?db=pubmed&retmode=json&id={pmid}"
        data = _curl_json(url)
        if data is None:
            raise StructuredProviderUnavailable("pubmed API returned no response")
        time.sleep(POLITE_SLEEP)
        # Empty dict = HTTP 404; TTL it like the esearch cache above.
        write_cache(cache, negative_sentinel(data) if not data else data)

    result = (data.get("result") or {}).get(pmid) or {}
    pubtypes = result.get("pubtype") or []
    if not isinstance(pubtypes, list):
        pubtypes = [str(pubtypes)]
    out["pubtypes"] = list(pubtypes)
    out["pubdate"] = str(result.get("pubdate") or "")
    # NLM-defined pubtype strings. Exact equality on the canonical forms —
    # we don't paraphrase or interpret free-text fields.
    pt_set = {p for p in pubtypes if isinstance(p, str)}
    out["retracted"] = "Retracted Publication" in pt_set
    out["is_retraction_notice"] = "Retraction of Publication" in pt_set
    return out
