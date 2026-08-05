"""Crossref references + DOI-existence fallback.

Intentionally NOT a full `ScholarlyDatabaseProvider` — Crossref is used here
for two narrow purposes:

1. **References**. Recover reference lists when Semantic Scholar's
   `/references` endpoint returns `data: null` because the publisher has
   elided the field (Springer Nature and bioRxiv now do this for most
   post-2024 papers). See `fetch_crossref_refs`.
2. **DOI verification**. Confirm a candidate DOI actually resolves to an
   indexed work. Used by the reconcile phase's URL→DOI hunt waterfall
   to validate preprint-server URL conversions (SSRN, arXiv) before
   adopting them — Crossref indexes SSRN/arXiv DOIs that Semantic
   Scholar hasn't ingested yet (common for fresh preprints). See
   `verify_doi_via_crossref`.

For title / abstract / recommendations we stay on Semantic Scholar — those
are the S2 fields that motivated the whole provider abstraction, and
Crossref either doesn't expose them or exposes them in formats that add
complexity without new signal.

Uses the same `curl`-subprocess + on-disk cache pattern as
`semantic_scholar.py` to sidestep Python stdlib SSL handling behind
enterprise proxies.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.parse

from ..log import log
from ..paths import crossref_cache_dir
from ._cache import negative_sentinel, read_cache, safe_cache_key, write_cache

CROSSREF_BASE = "https://api.crossref.org/works"
# Polite-pool User-Agent per Crossref API guidelines.
USER_AGENT = "researchwiki/0.1 (https://github.com/anthropic/claude-code; mailto:noreply@example.com)"


def _fetch_crossref_work(
    doi: str, retries: int = 3, sleep_sec: float = 0.5
) -> dict | None:
    """Fetch a Crossref `/works/{doi}` payload, with on-disk cache.

    Returns the parsed JSON dict on HTTP 200, the sentinel `{"message": {}}`
    on HTTP 404 (unknown DOI — distinguishable from network error because the
    sentinel is also cached), or None when all retries fail (network error /
    timeout / unparseable response).

    Shared between `fetch_crossref_refs` (which extracts the references
    list) and `verify_doi_via_crossref` (which reads title/year from the
    same payload). Both callers benefit from the cache: a DOI we verified
    minutes ago doesn't re-hit the network when we later need its refs.
    """
    if not doi:
        return None
    doi_norm = doi.strip().lower()
    cache_dir = crossref_cache_dir()
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"crossref__{safe_cache_key(doi_norm)}.json"
    cached = read_cache(cache)
    if cached is not None:
        return cached

    url = f"{CROSSREF_BASE}/{urllib.parse.quote(doi_norm, safe='/.')}"
    data: dict | None = None
    for attempt in range(retries):
        if attempt > 0:
            backoff = 2 ** attempt
            log(f"  retry {attempt} after {backoff}s", tag="crossref")
            time.sleep(backoff)
        log(f"  fetch {url}", tag="crossref")
        try:
            proc = subprocess.run(
                ["curl", "-sS", "-w", "\n%{http_code}", "-A", USER_AGENT, url],
                capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            log(f"  timeout on {url}", tag="crossref")
            continue
        if proc.returncode != 0:
            log(f"  curl error: {proc.stderr.strip()}", tag="crossref")
            continue
        body, _, status = proc.stdout.rpartition("\n")
        status = status.strip()
        if status == "404":
            log(f"  HTTP 404 — unknown DOI on Crossref: {doi_norm}", tag="crossref")
            # TTL'd negative: a fresh preprint DOI Crossref hasn't indexed yet
            # gives a real 404; without expiry it would be rejected forever.
            data = negative_sentinel({"message": {}})
            break
        if status != "200":
            log(f"  HTTP {status}", tag="crossref")
            continue
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            log(f"  JSON parse error: {e}", tag="crossref")
            continue
        time.sleep(sleep_sec)
        break
    if data is None:
        log(f"  giving up on {doi_norm} after {retries} retries", tag="crossref")
        return None
    write_cache(cache, data)
    return data


def fetch_crossref_refs(doi: str, retries: int = 3, sleep_sec: float = 0.5) -> list[str]:
    """Return lowercase DOIs from Crossref's `reference` array for this work.

    Empty list on cache miss + network failure, unknown DOI, or works with
    no deposited references (common for bioRxiv preprints).
    """
    data = _fetch_crossref_work(doi, retries=retries, sleep_sec=sleep_sec)
    if data is None:
        return []
    refs = (data.get("message") or {}).get("reference") or []
    out: list[str] = []
    seen: set[str] = set()
    for r in refs:
        ref_doi = (r.get("DOI") or "").lower().strip()
        if ref_doi and ref_doi not in seen:
            seen.add(ref_doi)
            out.append(ref_doi)
    return out


def _cached_crossref_work(doi: str) -> dict | None:
    """Read a previously-cached `/works/{doi}` payload without touching the
    network. Returns None on cache miss.

    Split out from `_fetch_crossref_work` (which is cache-*first*, network-
    second) so callers that must not add a request — the commentary guard's
    default path — can consult whatever the ingest already fetched and
    otherwise degrade to "signal unknown".
    """
    if not doi:
        return None
    cache = crossref_cache_dir() / f"crossref__{safe_cache_key(doi.strip().lower())}.json"
    return read_cache(cache)


def crossref_structural_signals(doi: str, *, allow_fetch: bool = False) -> dict:
    """Structural (non-prose) fields used to tell a commentary from an article.

    Returns `{"reference_count": int|None, "page": str|None, "type": str|None,
    "subtype": str|None}` — all None when the DOI isn't in the cache and
    `allow_fetch` is False, or when Crossref doesn't know the DOI.

    Whitelist note (CLAUDE.md Rule 1): every field here is structural metadata
    Crossref already exposes to `ingest`. Nothing prose-y is read, and nothing
    is paraphrased into a page — the fields inform one promotion decision.

    `type`/`subtype` are returned for logging only. They do **not** discriminate
    a Research Highlight from the article it summarizes: the Nature Genetics
    highlight that motivated this function reports `type: journal-article`,
    `subtype: None`, exactly like a 32-page primary research paper. The
    load-bearing fields are `reference-count` (0 for the highlight) and `page`
    (`"1458-1458"` — a one-page extent).

    `allow_fetch` defaults False so calling this never costs a request; the
    ingest opts in only after a cheap *local* pre-trigger has already fired
    (see `agents.commentary.crossref_lookup_worthwhile`).
    """
    data = _cached_crossref_work(doi)
    if data is None and allow_fetch:
        data = _fetch_crossref_work(doi)
    msg = ((data or {}).get("message") or {}) if data else {}
    ref_count = msg.get("reference-count")
    if not isinstance(ref_count, int):
        ref_count = None
    page = msg.get("page")
    return {
        "reference_count": ref_count,
        "page": page if isinstance(page, str) and page.strip() else None,
        "type": msg.get("type") or None,
        "subtype": msg.get("subtype") or None,
    }


def verify_doi_via_crossref(doi: str) -> dict | None:
    """Confirm `doi` resolves to an indexed Crossref work; return a small
    metadata dict on hit, or None on miss / network failure.

    Used by the reconcile phase's URL→DOI hunt waterfall to validate
    preprint-server URL conversions (SSRN, arXiv) before adopting them.
    Crossref indexes many SSRN/arXiv DOIs that Semantic Scholar hasn't
    ingested yet — this is the gap the URL hunt fills.

    Return shape (subset of Crossref's `message`):
        {
            "doi":   <lowercased canonical DOI>,
            "title": <first title string, or None>,
            "year":  <int, or None — pulled from `posted` ∨ `published-*`>,
            "venue": <first container-title (journal of record), or None>,
        }

    Returns None for HTTP 404 (DOI doesn't exist) and for network errors.
    Caller should treat both as "candidate failed verification" — the
    distinction doesn't matter for the hunt waterfall (try the next
    candidate either way).
    """
    data = _fetch_crossref_work(doi)
    if data is None:
        return None
    msg = (data or {}).get("message") or {}
    if not msg or not msg.get("DOI"):
        return None

    title_list = msg.get("title") or []
    title = title_list[0].strip() if title_list and isinstance(title_list[0], str) else None

    # `container-title` is the journal/book of record — the authoritative venue
    # even for an accepted-manuscript/preprint PDF whose first page lacks a
    # journal masthead. It's a list (like `title`); take the first entry.
    container_list = msg.get("container-title") or []
    venue = container_list[0].strip() if container_list and isinstance(container_list[0], str) else None

    year: int | None = None
    for key in ("posted", "published-print", "published-online", "issued", "created"):
        parts = ((msg.get(key) or {}).get("date-parts") or [[]])
        if parts and parts[0]:
            try:
                y = int(parts[0][0])
            except (TypeError, ValueError):
                continue
            if 1900 <= y <= 2099:
                year = y
                break

    return {"doi": (msg["DOI"] or "").lower(), "title": title, "year": year, "venue": venue}
