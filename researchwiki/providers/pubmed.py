"""PubMed E-utilities thin client — retraction status and keyword search.

Two narrow uses of one Rule-1 whitelisted API:

- **Retraction status** (`retraction_status`, for `retraction-check`): the
  `esummary.pubtype` field flags retracted or retracting publications with
  fixed NLM-defined strings. Titles/authors for *wiki papers* still come from
  S2 and Crossref, which have less latency and no NCBI rate limit.
- **Keyword search** (`search`, for `scout search`): esearch + efetch, returning
  discovery-only leads. See the section at the bottom of this module.

The returned record is a fixed schema: {pmid, retracted, retraction_of_pmid,
retracted_by_pmid, pubtypes, pubdate, source, fetched_at}. No free-form
prose fields are exposed — prose-based retraction notes remain behind the
Rule-1 prose ban.

Uses the same `curl`-subprocess + on-disk cache pattern as
`crossref.py` and `semantic_scholar.py`.
"""

from __future__ import annotations

import os
import time
import urllib.parse
from datetime import date

from ..paths import web_cache_dir
from ._cache import (
    negative_sentinel,
    read_cache,
    read_text_cache,
    safe_cache_key,
    write_cache,
    write_stamped_cache,
    write_text_cache,
)
from ._http import StructuredProviderUnavailable, _json_object, curl_body, curl_json
from ._xml import parse_xml

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


# ---------- keyword search (`scout search`) ----------
#
# A second use of the same API: esearch for PMIDs, then efetch for the
# records. The verbatim abstract travels under its own key; `scout search`
# ranks and displays it as a discovery lead, never as wiki evidence.

SEARCH_FIELDS = (
    "pmid", "doi", "pmcid", "title", "authors", "journal", "pub_date", "year",
    "pubtypes", "retracted", "abstract",
)
#: NCBI allows 10 requests/second with an API key, 3 without.
KEYED_SLEEP = 0.11
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def _curl_body(url: str, *, secret_query=None, retries: int = 3) -> str | None:
    return curl_body(url, provider="pubmed", retries=retries,
                     secret_query=secret_query, reject_400=True)


def _identity() -> tuple[dict[str, str], dict[str, str], float]:
    """(public params, secret params, polite sleep) from the environment.

    `NCBI_API_KEY` raises the rate limit; `RW_CONTACT_EMAIL` is NCBI's
    requested contact for heavy users. Both are optional and both travel as
    secrets: the key is a credential and the address is personal, so neither
    belongs in a log line or a cache filename.
    """
    secret: dict[str, str] = {}
    key = (os.environ.get("NCBI_API_KEY") or "").strip()
    email = (os.environ.get("RW_CONTACT_EMAIL") or "").strip()
    if key:
        secret["api_key"] = key
    if email:
        secret["email"] = email
    return {"tool": "researchwiki"}, secret, (KEYED_SLEEP if key else POLITE_SLEEP)


def _text(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _date_of(el) -> str:
    """ISO date, as precise as the record is: `YYYY-MM-DD`, `YYYY-MM`, `YYYY`."""
    if el is None:
        return ""
    year = _text(el.find("Year"))
    if not year:
        medline = _text(el.find("MedlineDate"))
        return medline[:4] if medline[:4].isdigit() else ""
    month = _text(el.find("Month")).lower()
    mm = _MONTHS.get(month[:3]) if not month.isdigit() else int(month)
    if not mm:
        return year
    day = _text(el.find("Day"))
    return f"{year}-{mm:02d}-{int(day):02d}" if day.isdigit() else f"{year}-{mm:02d}"


def _author(el) -> str:
    collective = _text(el.find("CollectiveName"))
    if collective:
        return collective
    last = _text(el.find("LastName"))
    initials = _text(el.find("Initials"))
    return " ".join(x for x in (last, initials) if x)


def parse_efetch(xml_text: str) -> list[dict]:
    """PubMed efetch XML → fixed-schema records (`SEARCH_FIELDS`)."""
    root = parse_xml(xml_text, provider="pubmed")
    out: list[dict] = []
    for art in root.iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        if cit is None:
            continue
        article = cit.find("Article")
        ids = {
            (i.get("IdType") or "").lower(): _text(i)
            for i in art.findall("PubmedData/ArticleIdList/ArticleId")
        }
        pubtypes = [_text(p) for p in cit.findall("Article/PublicationTypeList/PublicationType")]
        parts = []
        for ab in cit.findall("Article/Abstract/AbstractText"):
            label = (ab.get("Label") or "").strip()
            body = _text(ab)
            if body:
                parts.append(f"{label}: {body}" if label else body)
        electronic = next(
            (d for d in cit.findall("Article/ArticleDate") if d.get("DateType") == "Electronic"),
            None,
        )
        pub_date = _date_of(electronic) or _date_of(cit.find("Article/Journal/JournalIssue/PubDate"))
        pmcid = ids.get("pmc", "")
        out.append({
            "pmid": _text(cit.find("PMID")),
            "doi": ids.get("doi", "").lower() or None,
            "pmcid": pmcid.upper() if pmcid else None,
            "title": _text(article.find("ArticleTitle") if article is not None else None).rstrip("."),
            "authors": [a for a in (_author(x) for x in cit.findall("Article/AuthorList/Author")) if a],
            "journal": _text(cit.find("Article/Journal/Title")),
            "pub_date": pub_date,
            "year": int(pub_date[:4]) if pub_date[:4].isdigit() else None,
            "pubtypes": pubtypes,
            "retracted": "Retracted Publication" in pubtypes,
            "abstract": "\n".join(parts),
        })
    return out


def search(query: str, *, limit: int = 20, since: date | None = None,
           max_age_days: float = 1) -> list[dict]:
    """Keyword search, relevance-ordered. Records follow `SEARCH_FIELDS`.

    `since` filters on publication date (`datetype=pdat`). Raises
    `ProviderRequestRejected` for a query PubMed refuses and
    `StructuredProviderUnavailable` for an outage.
    """
    public, secret, sleep = _identity()
    params = {"db": "pubmed", "retmode": "json", "sort": "relevance",
              "retmax": str(limit), "term": query, **public}
    if since:
        params.update({"datetype": "pdat", "mindate": since.strftime("%Y/%m/%d"),
                       "maxdate": date.today().strftime("%Y/%m/%d")})
    url = f"{EUTILS_BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    cache_dir = web_cache_dir() / "search"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"pubmed_esearch__{safe_cache_key(url)}.json"
    data = read_cache(cache, max_age_days=max_age_days)
    if data is None:
        body = _curl_body(url, secret_query=secret)
        data = _json_object(body) if body else {}
        write_stamped_cache(cache, data)
        time.sleep(sleep)
    pmids = [str(p) for p in (data.get("esearchresult") or {}).get("idlist") or []]
    if not pmids:
        return []

    fetch_params = {"db": "pubmed", "retmode": "xml", "id": ",".join(pmids), **public}
    fetch_url = f"{EUTILS_BASE}/efetch.fcgi?{urllib.parse.urlencode(fetch_params)}"
    fetch_cache = cache_dir / f"pubmed_efetch__{safe_cache_key(fetch_url)}.json"
    xml_text = read_text_cache(fetch_cache, max_age_days=max_age_days)
    if xml_text is None:
        xml_text = _curl_body(fetch_url, secret_query=secret) or "<PubmedArticleSet/>"
        write_text_cache(fetch_cache, xml_text, url=fetch_url, fmt="xml")
        time.sleep(sleep)
    records = {r["pmid"]: r for r in parse_efetch(xml_text)}
    # efetch returns PMID order, not relevance order; esearch's order wins.
    return [records[p] for p in pmids if p in records]
