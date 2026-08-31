"""Shared on-disk cache helpers for the whitelist providers.

Used directly by Crossref, bioRxiv, ORCID, and PubMed (`read_cache` /
`write_cache` / `negative_sentinel`). Semantic Scholar keeps its own
sentinel format (`_negative_cached`/`_cached_at`, plus force-refresh and
negative-cache-hit tracking that don't apply to the other four) — existing
`.s2-cache/` entries already use those key names, and switching to this
module's `_rw_negative`/`_rw_cached_at` would make old cached negatives
silently misread as real data. S2 instead calls `fsatomic.read_json` /
`write_json_atomic` directly for the same guarded-read/atomic-write
guarantees, and this module's `safe_cache_key` for its cache filenames.

Two guarantees every provider gets, one way or the other:

- **Guarded reads / atomic writes.** A cache file truncated by an interrupted
  write reads as a miss (re-fetch) instead of crashing every subsequent run,
  and writes go through `fsatomic` so a crash can't leave a partial file.
- **Negative-cache TTL.** A transient HTTP 404 — e.g. a freshly-registered
  preprint DOI that Crossref/bioRxiv hasn't indexed yet — is stamped so it
  auto-expires and is re-fetched after `NEG_TTL_DAYS`, instead of condemning
  the DOI as "not found" forever.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from pathlib import Path

from ..fsatomic import read_json, write_json_atomic

NEG_TTL_DAYS = 30
_NEG_KEY = "_rw_negative"
_NEG_AT = "_rw_cached_at"


def negative_sentinel(payload: dict) -> dict:
    """Mark `payload` as a negative-cache entry stamped at the current time.

    `payload` keeps whatever shape the caller's readers expect (e.g.
    `{"message": {}}` for Crossref), so existing field accesses still work;
    the extra keys only drive TTL expiry here.
    """
    return {
        **payload,
        _NEG_KEY: True,
        _NEG_AT: _dt.datetime.now().replace(microsecond=0).isoformat(),
    }


def _stale_negative(data) -> bool:
    if not (isinstance(data, dict) and data.get(_NEG_KEY)):
        return False
    ts = data.get(_NEG_AT)
    try:
        written = _dt.datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return False  # missing/invalid stamp → treat as fresh, don't churn
    return (_dt.datetime.now() - written).total_seconds() / 86400.0 > NEG_TTL_DAYS


def read_cache(cache: Path):
    """Return cached JSON, or None on miss / corrupt / expired-negative."""
    data = read_json(cache)
    if data is None or _stale_negative(data):
        return None
    return data


def write_cache(cache: Path, data) -> None:
    """Atomically persist `data` (utf-8)."""
    write_json_atomic(cache, data)


def safe_cache_key(raw: str, max_len: int = 160) -> str:
    """Return a collision-resistant, cross-platform-safe cache-key fragment.

    A readable tail is useful when inspecting caches by hand, but it is not an
    identity: punctuation replacement and truncation can map distinct DOIs or
    URLs onto the same filename. The SHA-256 suffix carries identity while the
    conservative ASCII fragment keeps the result valid on Windows and POSIX.

    Cache filenames written before the hash suffix are intentionally treated as
    misses. Those files are derived state, and their lossy identity means there
    is no safe way to decide which original request they belong to.
    """
    if max_len < 8:
        raise ValueError("max_len must be at least 8")
    raw = str(raw)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    budget = max_len - len(digest) - 1
    if budget <= 0 or not readable:
        return digest[:max_len]
    return f"{readable[-budget:]}-{digest}"
