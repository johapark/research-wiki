"""Shared on-disk cache helpers for the whitelist providers.

Two guarantees the providers previously lacked (S2 already had them):

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
