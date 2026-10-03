"""Shared structured-provider cache identity and portability."""

from __future__ import annotations

import pytest

from researchwiki.providers._cache import safe_cache_key


def test_lossy_readable_fragments_do_not_collide():
    """Both pairs collapsed to one filename before the hash carried identity."""
    assert safe_cache_key("10.1234/a:b") != safe_cache_key("10.1234/a/b")
    assert safe_cache_key("https://x.test/a?b=c") != safe_cache_key(
        "https://x.test/a:b=c"
    )


def test_cache_key_is_windows_safe_bounded_and_stable():
    raw = 'A\\B:C*D?E"F<G>H|I/' + "x" * 300
    first = safe_cache_key(raw, max_len=80)
    assert first == safe_cache_key(raw, max_len=80)
    assert len(first) <= 80
    assert not set('\\/:*?"<>|') & set(first)


def test_cache_key_rejects_too_short_bound():
    with pytest.raises(ValueError, match="at least 8"):
        safe_cache_key("x", max_len=7)


def test_max_age_expires_positive_entries_only_when_asked(tmp_path, monkeypatch):
    import datetime as dt
    from researchwiki.providers import _cache

    path = tmp_path / "c.json"
    _cache.write_stamped_cache(path, {"x": 1})
    assert _cache.read_cache(path)["x"] == 1
    assert _cache.read_cache(path, max_age_days=1)["x"] == 1
    assert _cache.read_cache(path, max_age_days=0) is None

    later = dt.datetime.now() + dt.timedelta(days=3)

    class _Later(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return later
    monkeypatch.setattr(_cache._dt, "datetime", _Later)
    assert _cache.read_cache(path, max_age_days=2) is None
    assert _cache.read_cache(path)["x"] == 1  # default: never expires


def test_text_cache_round_trips_xml(tmp_path):
    from researchwiki.providers import _cache

    path = tmp_path / "t.json"
    _cache.write_text_cache(path, "<feed/>", url="https://x.test", fmt="atom")
    assert _cache.read_text_cache(path, max_age_days=1) == "<feed/>"
    assert _cache.read_text_cache(path, max_age_days=0) is None
    path.write_text("{truncated", encoding="utf-8")
    assert _cache.read_text_cache(path) is None
