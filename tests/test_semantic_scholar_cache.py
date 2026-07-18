"""Negative-cache + force-refresh semantics for the Semantic Scholar provider.

Pins the contract documented at
researchwiki.providers.semantic_scholar:_read_cache:

  - 200 → positive cache, honored indefinitely without --refresh-cache
  - 404 (after retries exhausted) → negative-cache sentinel, honored within
    `negative_ttl_days`, re-fetched after expiry
  - 5xx / timeout / curl error → NOT cached (transient)
  - force_refresh_days == 0 → bypass everything
  - force_refresh_days == N → bypass entries older than N days
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from researchwiki.providers.semantic_scholar import (
    DEFAULT_NEG_TTL_DAYS,
    SemanticScholarProvider,
    _NEG_AT,
    _NEG_KEY,
    _NEG_STATUS,
)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Point the provider's cache directory at an isolated tmp_path."""
    monkeypatch.setattr(
        "researchwiki.providers.semantic_scholar.s2_cache_dir",
        lambda: tmp_path,
    )
    return tmp_path


# ---------- positive cache ----------

def test_positive_cache_returned_immediately(tmp_cache):
    provider = SemanticScholarProvider()
    cache = provider._cache_path("https://example/p1")
    cache.write_text(json.dumps({"data": "real"}))

    consumed, value = provider._read_cache(cache, "https://example/p1")
    assert consumed is True
    assert value == {"data": "real"}


# ---------- negative cache write + read ----------

def test_negative_cache_sentinel_honored(tmp_cache):
    provider = SemanticScholarProvider()
    cache = provider._cache_path("https://example/dead")
    provider._write_negative_cache(cache, status=404, url="https://example/dead")

    # Sentinel is on disk with the documented shape.
    payload = json.loads(cache.read_text())
    assert payload[_NEG_KEY] is True
    assert payload[_NEG_STATUS] == 404
    assert _NEG_AT in payload

    # Read returns (consumed=True, value=None) — caller treats as cache miss
    # *result*, but the URL is recorded as a negative-cache hit.
    new_provider = SemanticScholarProvider()  # fresh hit-list
    consumed, value = new_provider._read_cache(cache, "https://example/dead")
    assert consumed is True
    assert value is None
    assert "https://example/dead" in new_provider.negative_cache_hits


def test_negative_cache_expires_past_ttl(tmp_cache):
    provider = SemanticScholarProvider(negative_ttl_days=30)
    cache = provider._cache_path("https://example/old")
    # Hand-write a 60-day-old sentinel.
    expired_at = (_dt.datetime.now() - _dt.timedelta(days=60)).isoformat()
    cache.write_text(json.dumps({
        _NEG_KEY: True, _NEG_STATUS: 404, _NEG_AT: expired_at,
    }))

    consumed, _ = provider._read_cache(cache, "https://example/old")
    # Past TTL → caller must fetch.
    assert consumed is False


def test_negative_cache_within_ttl_honored(tmp_cache):
    provider = SemanticScholarProvider(negative_ttl_days=30)
    cache = provider._cache_path("https://example/recent")
    recent_at = (_dt.datetime.now() - _dt.timedelta(days=5)).isoformat()
    cache.write_text(json.dumps({
        _NEG_KEY: True, _NEG_STATUS: 404, _NEG_AT: recent_at,
    }))

    consumed, value = provider._read_cache(cache, "https://example/recent")
    assert consumed is True
    assert value is None


# ---------- force_refresh_days ----------

def test_force_refresh_days_zero_busts_positive_cache(tmp_cache):
    """`--refresh-cache` with no days = force_refresh_days=0 bypasses every
    cache entry, positive or negative, regardless of age."""
    provider = SemanticScholarProvider(force_refresh_days=0)
    cache = provider._cache_path("https://example/p1")
    cache.write_text(json.dumps({"data": "real"}))

    consumed, _ = provider._read_cache(cache, "https://example/p1")
    assert consumed is False  # fresh fetch required


def test_force_refresh_days_zero_busts_negative_cache(tmp_cache):
    provider = SemanticScholarProvider(force_refresh_days=0)
    cache = provider._cache_path("https://example/dead")
    cache.write_text(json.dumps({
        _NEG_KEY: True, _NEG_STATUS: 404,
        _NEG_AT: _dt.datetime.now().isoformat(),
    }))

    consumed, _ = provider._read_cache(cache, "https://example/dead")
    assert consumed is False


def test_force_refresh_days_n_bypasses_entries_older_than_n(tmp_cache):
    """Positive cache ages by file mtime; force_refresh_days=10 bypasses
    files older than 10 days."""
    provider = SemanticScholarProvider(force_refresh_days=10)
    cache = provider._cache_path("https://example/old-positive")
    cache.write_text(json.dumps({"data": "old"}))
    # Stat the file 30 days into the past.
    old_ts = (_dt.datetime.now() - _dt.timedelta(days=30)).timestamp()
    os.utime(cache, (old_ts, old_ts))

    consumed, _ = provider._read_cache(cache, "https://example/old-positive")
    assert consumed is False


def test_force_refresh_days_n_keeps_recent_entries(tmp_cache):
    provider = SemanticScholarProvider(force_refresh_days=30)
    cache = provider._cache_path("https://example/fresh-positive")
    cache.write_text(json.dumps({"data": "fresh"}))
    # mtime defaults to now → fresh; no utime adjustment.

    consumed, value = provider._read_cache(cache, "https://example/fresh-positive")
    assert consumed is True
    assert value == {"data": "fresh"}


# ---------- transient failures NOT cached ----------

def test_transient_failure_does_not_write_negative_cache(tmp_cache, monkeypatch):
    """A run where every attempt times out (or returns 5xx) must leave the
    cache empty so the next run can retry afresh — only confirmed 404s get
    negative-cached."""
    provider = SemanticScholarProvider(retries=2, sleep_sec=0)

    class _FakeProc:
        def __init__(self, returncode=0, stdout="\n500"):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    # Every curl call returns a 500 — same retry path as a real S2 outage.
    def fake_run(*args, **kwargs):
        return _FakeProc(returncode=0, stdout="\n500")

    monkeypatch.setattr(
        "researchwiki.providers.semantic_scholar.subprocess.run", fake_run
    )
    monkeypatch.setattr(
        "researchwiki.providers.semantic_scholar.time.sleep", lambda *_: None
    )
    result = provider._fetch("https://example/transient")
    assert result is None
    cache = provider._cache_path("https://example/transient")
    assert not cache.exists(), (
        "transient (5xx) must not write negative-cache — only confirmed 404 does"
    )


def test_persistent_404_writes_negative_cache(tmp_cache, monkeypatch):
    provider = SemanticScholarProvider(retries=2, sleep_sec=0)

    class _FakeProc:
        def __init__(self):
            self.returncode = 0
            self.stdout = "\n404"
            self.stderr = ""

    monkeypatch.setattr(
        "researchwiki.providers.semantic_scholar.subprocess.run",
        lambda *a, **k: _FakeProc(),
    )
    monkeypatch.setattr(
        "researchwiki.providers.semantic_scholar.time.sleep", lambda *_: None
    )
    result = provider._fetch("https://example/perma-404")
    assert result is None

    # Sentinel must be on disk.
    cache = provider._cache_path("https://example/perma-404")
    assert cache.exists()
    payload = json.loads(cache.read_text())
    assert payload[_NEG_KEY] is True
    assert payload[_NEG_STATUS] == 404


def test_default_ttl_is_documented(tmp_cache):
    """The default 30-day TTL is the published contract; pin it so a future
    edit doesn't silently retune steady-state cost."""
    assert DEFAULT_NEG_TTL_DAYS == 30
    provider = SemanticScholarProvider()
    assert provider.negative_ttl_days == 30
