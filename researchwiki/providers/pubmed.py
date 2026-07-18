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

import json
import subprocess
import time
import urllib.parse
from datetime import date

from ..log import log
from ..paths import web_cache_dir
from ._cache import read_cache, write_cache

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "researchwiki/0.1 (https://github.com/anthropic/claude-code; mailto:noreply@example.com)"
# Polite-pool rate: NCBI asks for ≤3 req/sec without an API key.
POLITE_SLEEP = 0.4


def _curl_json(url: str, retries: int = 3) -> dict | None:
    """Fetch JSON via curl subprocess. Mirrors crossref.py's pattern to
    bypass Python stdlib SSL issues behind enterprise proxies."""
    for attempt in range(retries):
        if attempt > 0:
            time.sleep(2 ** attempt)
            log(f"  retry {attempt}", tag="pubmed")
        log(f"  fetch {url}", tag="pubmed")
        try:
            proc = subprocess.run(
                ["curl", "-sS", "-w", "\n%{http_code}", "-A", USER_AGENT, url],
                capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            log("  timeout", tag="pubmed")
            continue
        if proc.returncode != 0:
            log(f"  curl error: {proc.stderr.strip()}", tag="pubmed")
            continue
        body, _, status = proc.stdout.rpartition("\n")
        status = status.strip()
        if status == "404":
            log(f"  HTTP 404: {url}", tag="pubmed")
            return {}
        if status != "200":
            log(f"  HTTP {status}", tag="pubmed")
            continue
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            log(f"  JSON parse error: {e}", tag="pubmed")
            continue
    return None


def doi_to_pmid(doi: str) -> str | None:
    """Return the PubMed ID for a DOI, or None if not indexed in PubMed."""
    if not doi:
        return None
    doi_norm = doi.strip().lower()
    cache_dir = web_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    safe = doi_norm.replace("/", "_").replace(":", "_")
    cache = cache_dir / f"pubmed_esearch__{safe[-160:]}.json"
    data = read_cache(cache)
    if data is None:
        url = (f"{EUTILS_BASE}/esearch.fcgi"
               f"?db=pubmed&retmode=json&term={urllib.parse.quote(doi_norm)}%5Baid%5D")
        data = _curl_json(url)
        if data is None:
            return None
        time.sleep(POLITE_SLEEP)
        write_cache(cache, data)
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

    On network / missing-PMID failure, returns the record with pmid=None
    and retracted=False (absence of evidence, not evidence of absence —
    callers should check `pmid` before trusting `retracted=False`).
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
            return out
        time.sleep(POLITE_SLEEP)
        write_cache(cache, data)

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
