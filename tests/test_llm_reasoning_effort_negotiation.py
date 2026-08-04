"""`reasoning_effort` negotiation on the OpenAI-compatible path.

Models disagree about this field's vocabulary and disagree with a hard 400,
not a warning. Observed 2026-08-03: `disable_thinking` maps to "minimal",
gpt-5.6-luna accepts only none/low/medium/high/xhigh, and the resulting 400
killed the memory-evolution judge on every ingest while the phase reported
`judged=0` rather than failing — the silence is what made it expensive.

These tests drive `call_openai_compatible` against a fake urlopen so no
network call happens.
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from researchwiki.agents import llm


# Verbatim shape of the error that motivated this (OpenAI /v1/chat/completions).
_LUNA_400 = json.dumps({
    "error": {
        "message": (
            "Unsupported value: 'reasoning_effort' does not support 'minimal' "
            "with this model. Supported values are: 'none', 'low', 'medium', "
            "'high', and 'xhigh'."
        ),
        "type": "invalid_request_error",
        "param": "reasoning_effort",
        "code": "unsupported_value",
    }
})


@pytest.fixture(autouse=True)
def _clear_negotiation_cache():
    """The caches are module-level and process-lived; isolate each test."""
    llm._EFFORT_REJECTED.clear()
    llm._EFFORT_SUPPORTED.clear()
    yield
    llm._EFFORT_REJECTED.clear()
    llm._EFFORT_SUPPORTED.clear()


def _http_error(body: str, code: int = 400) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://x/v1/chat/completions", code=code, msg="Bad Request",
        hdrs=None, fp=io.BytesIO(body.encode("utf-8")),
    )


def _ok_response() -> io.BytesIO:
    payload = json.dumps({
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    })
    return io.BytesIO(payload.encode("utf-8"))


class _FakeUrlopen:
    """Fails with `errors` in order, then succeeds. Records each request body."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.bodies: list[dict] = []

    def __call__(self, req, timeout=None):
        self.bodies.append(json.loads(req.data.decode("utf-8")))
        if self.errors:
            raise self.errors.pop(0)
        resp = _ok_response()
        resp.__enter__ = lambda: resp          # type: ignore[attr-defined]
        resp.__exit__ = lambda *a: False       # type: ignore[attr-defined]
        return resp


def _call(monkeypatch, fake, **kw):
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return llm.call_openai_compatible(
        model=kw.pop("model", "gpt-5.6-luna"),
        prompt="p",
        base_url="http://x/v1",
        **kw,
    )


def test_rejected_value_is_renegotiated_not_raised(monkeypatch):
    """The bug: a 400 on 'minimal' used to abort the call outright."""
    fake = _FakeUrlopen([_http_error(_LUNA_400)])
    resp = _call(monkeypatch, fake, reasoning_effort="minimal")

    assert resp.text == "hi"
    assert [b.get("reasoning_effort") for b in fake.bodies] == ["minimal", "none"]


def test_advertised_vocabulary_is_honored(monkeypatch):
    """'none' is chosen over 'low' — nearest on the scale, ties go lower."""
    fake = _FakeUrlopen([_http_error(_LUNA_400)])
    _call(monkeypatch, fake, reasoning_effort="minimal")
    assert fake.bodies[1]["reasoning_effort"] == "none"


def test_negotiated_value_is_cached_per_endpoint_and_model(monkeypatch):
    """Only the first call pays for discovery."""
    fake1 = _FakeUrlopen([_http_error(_LUNA_400)])
    _call(monkeypatch, fake1, reasoning_effort="minimal")

    fake2 = _FakeUrlopen([])          # second call must not need a retry
    _call(monkeypatch, fake2, reasoning_effort="minimal")
    assert [b.get("reasoning_effort") for b in fake2.bodies] == ["none"]


def test_field_is_dropped_when_server_names_no_alternatives(monkeypatch):
    """Rejecting the field (not a value) drops it in one step, no thrashing."""
    generic = json.dumps({"error": {"message": "reasoning_effort not supported",
                                    "param": "reasoning_effort"}})
    fake = _FakeUrlopen([_http_error(generic)])
    resp = _call(monkeypatch, fake, reasoning_effort="minimal")

    assert resp.text == "hi"
    assert len(fake.bodies) == 2
    assert "reasoning_effort" not in fake.bodies[1]


def test_unsupported_field_is_remembered_not_re_litigated(monkeypatch):
    """Later calls to the same endpoint skip the field outright."""
    generic = json.dumps({"error": {"message": "reasoning_effort not supported",
                                    "param": "reasoning_effort"}})
    _call(monkeypatch, _FakeUrlopen([_http_error(generic)]),
          reasoning_effort="minimal")

    fake2 = _FakeUrlopen([])
    _call(monkeypatch, fake2, reasoning_effort="minimal")
    assert len(fake2.bodies) == 1
    assert "reasoning_effort" not in fake2.bodies[0]


def test_unrelated_400_still_raises(monkeypatch):
    """Masking a real bad request would be worse than the original bug."""
    other = json.dumps({"error": {"message": "model not found", "param": "model"}})
    fake = _FakeUrlopen([_http_error(other)])
    with pytest.raises(RuntimeError, match="HTTP 400"):
        _call(monkeypatch, fake, reasoning_effort="minimal")
    assert len(fake.bodies) == 1


def test_renegotiation_does_not_consume_transient_retry_budget(monkeypatch):
    """A 400 isn't transient; it must not eat the 429/5xx budget."""
    errors = [_http_error(_LUNA_400)]
    errors += [_http_error("busy", code=429)] * (llm._RETRY_MAX_ATTEMPTS - 1)
    fake = _FakeUrlopen(errors)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)

    resp = _call(monkeypatch, fake, reasoning_effort="minimal")
    assert resp.text == "hi"


def test_no_reasoning_effort_means_no_field(monkeypatch):
    fake = _FakeUrlopen([])
    _call(monkeypatch, fake)
    assert "reasoning_effort" not in fake.bodies[0]


@pytest.mark.parametrize(
    "requested,supported,expected",
    [
        # The observed case: minimal unavailable, none is nearest below.
        ("minimal", {"none", "low", "medium", "high", "xhigh"}, "none"),
        # Classic o-series vocabulary: no "none", so step up to low.
        ("minimal", {"low", "medium", "high"}, "low"),
        # A high request degrades downward, not to the floor.
        ("high", {"none", "minimal", "low", "medium"}, "medium"),
        # Provider-specific token we don't model: unusable once rejected.
        ("turbo", {"low", "high"}, None),
    ],
)
def test_nearest_effort_across_model_vocabularies(requested, supported, expected):
    assert llm._nearest_effort(requested, {requested}, supported) == expected


def test_parse_supported_ignores_the_rejected_value():
    """'minimal' is quoted in the same sentence; it must not be read as supported."""
    got = llm._parse_supported_efforts(_LUNA_400)
    assert got == {"none", "low", "medium", "high", "xhigh"}
    assert "minimal" not in got
