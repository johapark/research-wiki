"""Retry policy for the OpenAI-compatible client (`call_openai_compatible`).

OpenAI intermittently returns spurious 401 "insufficient permissions" on
otherwise-valid keys (observed 2026-07-18: some calls in a run 401 while
others on the same key + model succeed). 401 is therefore in `_RETRY_STATUS`
so a transient one self-heals via the existing backoff instead of aborting
the whole ingest. Other 4xx (e.g. 400) must still fail fast.

`time.sleep` is monkeypatched to a no-op so the backoff adds no wall-clock
delay, and `urlopen` is faked so no network call happens.
"""
from __future__ import annotations

import email.message
import io
import json
import urllib.error

import pytest

from researchwiki.agents import llm


def _http_error(url: str, code: int, msg: str) -> urllib.error.HTTPError:
    body = json.dumps({"error": {"message": msg}}).encode("utf-8")
    return urllib.error.HTTPError(
        url, code, msg, email.message.Message(), io.BytesIO(body)
    )


class _FakeResp:
    """Minimal context-manager response with a `.read()` of JSON bytes."""

    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


_OK_PAYLOAD = {
    "choices": [{"message": {"content": "hello"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}


def test_openai_401_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(req.full_url, 401, "insufficient permissions")
        return _FakeResp(_OK_PAYLOAD)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    resp = llm.call_openai_compatible(
        model="gpt-5.6-luna",
        prompt="p",
        base_url="https://api.openai.com/v1",
        max_tokens=10,
        temperature=0.2,
    )
    assert resp.text == "hello"
    assert calls["n"] == 2  # first attempt 401'd, retry succeeded


def test_openai_400_is_not_retried(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(req.full_url, 400, "bad request")

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        llm.call_openai_compatible(
            model="gpt-5.6-luna",
            prompt="p",
            base_url="https://api.openai.com/v1",
            max_tokens=10,
            temperature=0.2,
        )
    assert calls["n"] == 1  # non-retryable 4xx fails fast
