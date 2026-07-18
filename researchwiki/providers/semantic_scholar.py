"""Semantic Scholar Graph API + Recommendations API provider.

Uses `curl` under the hood to sidestep Python's stdlib SSL-cert handling
(enterprise proxy compatibility). Caches all responses under `.s2-cache/`.

Per CLAUDE.md Rule 1 this is the one permitted external data source for
structural metadata plus the verbatim `abstract` and draft-only `tldr`.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

from ..log import log
from ..paths import s2_cache_dir
from .base import ScholarlyArticle, ScholarlyDatabaseProvider

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_RECS = "https://api.semanticscholar.org/recommendations/v1"

DETAIL_FIELDS = "title,authors.name,year,externalIds,venue,abstract,tldr,referenceCount,citationCount"
REFCITE_FIELDS = "title,year,externalIds,venue,citationCount"
REC_FIELDS = "title,year,externalIds,venue,citationCount"

# Negative-cache shape — written to the same cache path on a permanent
# failure so re-runs don't re-fetch and re-suffer the retry-backoff cost
# (~6-10s per stale DOI). Recognized on read and turned back into a None
# return. The TTL bounds the silence so a paper S2 hadn't indexed yet
# auto-recovers on the next audit past the deadline.
_NEG_KEY = "_negative_cached"
_NEG_AT = "_cached_at"
_NEG_STATUS = "_status"
DEFAULT_NEG_TTL_DAYS = 30


class SemanticScholarProvider(ScholarlyDatabaseProvider):
    """Reference implementation of the provider interface for Semantic Scholar."""

    def __init__(
        self,
        sleep_sec: float = 1.1,
        retries: int = 3,
        log_tag: str = "ingest",
        *,
        negative_ttl_days: int = DEFAULT_NEG_TTL_DAYS,
        force_refresh_days: int | None = None,
    ) -> None:
        self.sleep_sec = sleep_sec
        self.retries = retries
        self._log_tag = log_tag
        # `negative_ttl_days` — how long a 404 sentinel stays valid before
        # the provider re-fetches. Default 30d; user-tunable per audit run.
        self.negative_ttl_days = negative_ttl_days
        # `force_refresh_days` — when set, BOTH positive and negative caches
        # older than this are bypassed for this run. None = honor caches as
        # written; 0 = bust everything; N>0 = refresh anything older than N.
        self.force_refresh_days = force_refresh_days
        # Tracks every URL whose final outcome this run was a negative-cache
        # decision (sentinel honored OR sentinel freshly written). The audit
        # task reads this to surface the list of papers S2 doesn't index.
        self.negative_cache_hits: list[str] = []

    @property
    def name(self) -> str:
        return "semantic-scholar"

    # ---------- transport ----------

    def _cache_path(self, url: str) -> Path:
        safe = url.replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_")
        return s2_cache_dir() / f"s2__{safe[-160:]}.json"

    def _read_cache(self, cache: Path, url: str) -> tuple[bool, dict[str, Any] | None]:
        """Read a cache file with negative-cache + force-refresh semantics.

        Returns (consumed, value):
          - (False, None) — miss, caller must fetch.
          - (True, dict)  — positive cache hit, return the dict directly.
          - (True, None)  — negative cache hit (sentinel honored), return None.

        Force-refresh semantics:
          - If `force_refresh_days is None`: caches always honored.
          - If `force_refresh_days == 0`: ALL caches bypassed (positive + negative).
          - If `force_refresh_days > 0`: caches older than N days bypassed.
            Positive caches don't carry a write-timestamp today (the file is
            just the API payload), so we fall back to the file's mtime for them.

        Negative-cache TTL: a sentinel older than `negative_ttl_days` is
        ignored even without a force-refresh, so a paper S2 didn't index yet
        auto-recovers on the next audit past the deadline.
        """
        if not cache.exists():
            return False, None
        try:
            data = json.loads(cache.read_text())
        except (json.JSONDecodeError, OSError):
            return False, None  # corrupt → treat as miss

        is_negative = isinstance(data, dict) and data.get(_NEG_KEY) is True

        # Force-refresh check applies before the negative-TTL check; a manual
        # bust supersedes the automatic one.
        if self.force_refresh_days is not None:
            if self.force_refresh_days == 0:
                return False, None
            age_days = self._cache_age_days(cache, data, is_negative)
            if age_days is not None and age_days > self.force_refresh_days:
                return False, None

        if is_negative:
            age_days = self._cache_age_days(cache, data, is_negative=True)
            if age_days is not None and age_days > self.negative_ttl_days:
                return False, None  # TTL expired → re-fetch
            self.negative_cache_hits.append(url)
            return True, None

        return True, data

    @staticmethod
    def _cache_age_days(
        cache: Path, data: Any, is_negative: bool,
    ) -> float | None:
        """Days since the cache entry was written. Negative caches carry an
        explicit `_cached_at` ISO timestamp; positive caches use file mtime."""
        if is_negative and isinstance(data, dict):
            ts = data.get(_NEG_AT)
            if isinstance(ts, str):
                try:
                    written = _dt.datetime.fromisoformat(ts)
                except ValueError:
                    return None
                return (_dt.datetime.now() - written).total_seconds() / 86400.0
        try:
            mtime = cache.stat().st_mtime
        except OSError:
            return None
        return (time.time() - mtime) / 86400.0

    def _write_negative_cache(self, cache: Path, status: int | str, url: str) -> None:
        """Persist a negative-cache sentinel. Caller is the giving-up path."""
        sentinel = {
            _NEG_KEY: True,
            _NEG_STATUS: status,
            _NEG_AT: _dt.datetime.now().replace(microsecond=0).isoformat(),
        }
        try:
            cache.write_text(json.dumps(sentinel, indent=2))
            self.negative_cache_hits.append(url)
        except OSError as e:
            log(f"  WARN: could not write negative-cache for {url}: {e}",
                tag=self._log_tag)

    def _fetch(self, url: str) -> dict[str, Any] | None:
        """curl-backed GET with retries + disk cache. Returns None on failure."""
        s2_cache_dir().mkdir(exist_ok=True)
        cache = self._cache_path(url)
        consumed, value = self._read_cache(cache, url)
        if consumed:
            return value

        # Track whether any attempt saw a 404 — that's the "permanent" signal
        # we use to negative-cache. Retrying a 404 won't change the answer, but
        # we still let the existing retry loop run (no behavior regression on
        # transient 404s) and decide at the end based on what we observed.
        saw_404 = False
        for attempt in range(self.retries):
            if attempt > 0:
                backoff = 2 ** attempt
                log(f"  retry {attempt} after {backoff}s", tag=self._log_tag)
                time.sleep(backoff)
            log(f"  fetch {url}", tag=self._log_tag)
            try:
                proc = subprocess.run(
                    ["curl", "-sS", "-w", "\n%{http_code}",
                     "-A", "research-wiki-provider/0.1", url],
                    capture_output=True, text=True, timeout=60,
                )
            except subprocess.TimeoutExpired:
                log(f"  timeout on {url}", tag=self._log_tag)
                continue
            if proc.returncode != 0:
                log(f"  curl error: {proc.stderr.strip()}", tag=self._log_tag)
                continue
            body, _, status = proc.stdout.rpartition("\n")
            status = status.strip()
            if status == "429":
                log(f"  rate limited, sleeping 5s", tag=self._log_tag)
                time.sleep(5)
                continue
            if status == "404":
                saw_404 = True
                log(f"  HTTP 404 on {url}", tag=self._log_tag)
                continue
            if status != "200":
                log(f"  HTTP {status} on {url}", tag=self._log_tag)
                continue
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                log(f"  JSON parse error on {url}: {e}", tag=self._log_tag)
                continue
            cache.write_text(json.dumps(data, indent=2))
            time.sleep(self.sleep_sec)
            return data
        log(f"  giving up on {url} after {self.retries} retries", tag=self._log_tag)
        # Negative-cache only when the dominant failure was 404 (permanent
        # "not in S2"). Transient failures — timeouts, 5xx, 429, curl errors —
        # leave the cache empty so the next run gets to retry afresh.
        if saw_404:
            self._write_negative_cache(cache, status=404, url=url)
        return None

    def _post_fetch(self, url: str, payload: dict[str, Any], cache_path: Path) -> Any | None:
        """curl-backed POST with retries + disk cache. Returns None on failure.

        Uses a longer 429 sleep (30s) and more retries than _fetch because batch
        POSTs are less frequent but more valuable — worth waiting longer to succeed.
        """
        s2_cache_dir().mkdir(exist_ok=True)
        consumed, value = self._read_cache(cache_path, url)
        if consumed:
            return value
        payload_str = json.dumps(payload)
        retries = max(5, self.retries)
        saw_404 = False
        for attempt in range(retries):
            if attempt > 0:
                backoff = 2 ** attempt
                log(f"  retry {attempt} after {backoff}s", tag=self._log_tag)
                time.sleep(backoff)
            n_ids = len(payload.get("ids", []))
            log(f"  fetch POST {url} ({n_ids} ids)", tag=self._log_tag)
            try:
                proc = subprocess.run(
                    ["curl", "-sS", "-w", "\n%{http_code}",
                     "-A", "research-wiki-provider/0.1",
                     "-X", "POST", "-H", "Content-Type: application/json",
                     "-d", payload_str, url],
                    capture_output=True, text=True, timeout=60,
                )
            except subprocess.TimeoutExpired:
                log(f"  timeout on POST {url}", tag=self._log_tag)
                continue
            if proc.returncode != 0:
                log(f"  curl error: {proc.stderr.strip()}", tag=self._log_tag)
                continue
            body, _, status = proc.stdout.rpartition("\n")
            if status.strip() == "429":
                log(f"  rate limited, sleeping 30s", tag=self._log_tag)
                time.sleep(30)
                continue
            if status.strip() == "404":
                saw_404 = True
                log(f"  HTTP 404 on POST {url}", tag=self._log_tag)
                continue
            if status.strip() != "200":
                log(f"  HTTP {status.strip()} on POST {url}", tag=self._log_tag)
                continue
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                log(f"  JSON parse error on POST {url}: {e}", tag=self._log_tag)
                continue
            cache_path.write_text(json.dumps(data, indent=2))
            time.sleep(self.sleep_sec)
            return data
        log(f"  giving up on POST {url} after {retries} retries", tag=self._log_tag)
        if saw_404:
            self._write_negative_cache(cache_path, status=404, url=url)
        return None

    # ---------- public interface ----------

    def get_by_doi(self, doi: str) -> ScholarlyArticle | None:
        url = f"{S2_BASE}/paper/DOI:{urllib.parse.quote(doi)}?fields={DETAIL_FIELDS}"
        data = self._fetch(url)
        return self._to_article(data) if data else None

    def search_by_title(self, title: str) -> ScholarlyArticle | None:
        q = urllib.parse.quote(title[:200])
        url = f"{S2_BASE}/paper/search/match?query={q}&fields={DETAIL_FIELDS}"
        data = self._fetch(url)
        if data and isinstance(data, dict):
            hits = data.get("data") or []
            if hits:
                return self._to_article(hits[0])
        return None

    def get_references(self, article: ScholarlyArticle) -> list[ScholarlyArticle]:
        pid = self._paper_id(article)
        if not pid:
            return []
        url = f"{S2_BASE}/paper/{pid}/references?fields={REFCITE_FIELDS}&limit=1000"
        data = self._fetch(url)
        if not data:
            return []
        items = (data.get("data") or [])
        return [
            self._to_article(item.get("citedPaper") or {})
            for item in items
            if item.get("citedPaper")
        ]

    def get_citations(self, article: ScholarlyArticle) -> list[ScholarlyArticle]:
        pid = self._paper_id(article)
        if not pid:
            return []
        url = f"{S2_BASE}/paper/{pid}/citations?fields={REFCITE_FIELDS}&limit=1000"
        data = self._fetch(url)
        if not data:
            return []
        items = (data.get("data") or [])
        return [
            self._to_article(item.get("citingPaper") or {})
            for item in items
            if item.get("citingPaper")
        ]

    def get_recommendations(self, article: ScholarlyArticle) -> list[ScholarlyArticle]:
        pid = self._paper_id(article)
        if not pid:
            return []
        url = f"{S2_RECS}/papers/forpaper/{pid}?fields={REC_FIELDS}&limit=30"
        data = self._fetch(url)
        if not data:
            return []
        items = data.get("recommendedPapers") or data.get("data") or []
        return [self._to_article(item) for item in items if item]

    def get_batch_metadata(self, dois: list[str]) -> dict[str, ScholarlyArticle]:
        """Batch-fetch metadata for multiple papers in chunks of 500.

        Returns {doi_lower -> ScholarlyArticle}. Papers unknown to S2 are omitted.
        As a side-effect, prefills per-paper GET caches so subsequent get_by_doi
        calls return immediately from cache without network requests.
        """
        result: dict[str, ScholarlyArticle] = {}
        for start in range(0, len(dois), 500):
            chunk = dois[start : start + 500]
            url = f"{S2_BASE}/paper/batch?fields={DETAIL_FIELDS}"
            key = hashlib.md5("|".join(sorted(chunk)).encode()).hexdigest()
            cache_path = s2_cache_dir() / f"s2_batch__{key}.json"
            items = self._post_fetch(url, {"ids": [f"DOI:{d}" for d in chunk]}, cache_path)
            if not items or not isinstance(items, list):
                continue
            for doi, item in zip(chunk, items):
                if not item:
                    continue
                article = self._to_article(item)
                result[doi.lower()] = article
                # Prefill per-paper GET cache so get_by_doi hits cache
                per_url = (
                    f"{S2_BASE}/paper/DOI:{urllib.parse.quote(doi)}"
                    f"?fields={DETAIL_FIELDS}"
                )
                per_cache = self._cache_path(per_url)
                if not per_cache.exists():
                    per_cache.write_text(json.dumps(item, indent=2))
        return result

    # ---------- helpers ----------

    @staticmethod
    def _paper_id(article: ScholarlyArticle) -> str | None:
        if not article.doi:
            return None
        return f"DOI:{urllib.parse.quote(article.doi)}"

    @staticmethod
    def _to_article(d: dict[str, Any] | None) -> ScholarlyArticle:
        if not d:
            return ScholarlyArticle(raw={})
        ext = d.get("externalIds") or {}
        tldr_obj = d.get("tldr") or {}
        tldr_text = tldr_obj.get("text") if isinstance(tldr_obj, dict) else ""
        authors = [a.get("name", "") for a in d.get("authors") or []]
        return ScholarlyArticle(
            title=d.get("title") or "",
            authors=authors,
            year=d.get("year"),
            venue=d.get("venue") or "",
            doi=(ext.get("DOI") or "").lower() or None,
            abstract=d.get("abstract") or "",
            tldr=tldr_text or "",
            external_ids={k: str(v) for k, v in ext.items() if v is not None},
            reference_count=d.get("referenceCount"),
            citation_count=d.get("citationCount"),
            raw=d,
        )
