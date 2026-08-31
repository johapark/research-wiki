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
