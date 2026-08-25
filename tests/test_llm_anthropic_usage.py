from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from researchwiki.agents import llm
from researchwiki.agents.provider_errors import friendly_provider_error


def _install_fake_anthropic(monkeypatch, create):
    class BadRequestError(Exception):
        pass

    module = SimpleNamespace(
        Anthropic=lambda: SimpleNamespace(messages=SimpleNamespace(create=create)),
        BadRequestError=BadRequestError,
    )
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    return BadRequestError


def test_anthropic_usage_normalizes_cache_buckets_into_total_input(monkeypatch):
    response = SimpleNamespace(
        content=[SimpleNamespace(text="ok")],
        usage=SimpleNamespace(
            input_tokens=8,
            output_tokens=3,
            cache_read_input_tokens=20,
            cache_creation_input_tokens=5,
        ),
    )
    _install_fake_anthropic(monkeypatch, lambda **kwargs: response)

    got = llm.call_anthropic(model="claude-test", prompt="p")

    assert (got.input_tokens, got.output_tokens) == (33, 3)
    assert (got.cache_read_tokens, got.cache_write_tokens) == (20, 5)


def test_anthropic_access_failure_becomes_provider_unavailable(monkeypatch):
    pending = {}

    def fail(**kwargs):
        raise pending["error"]

    error_type = _install_fake_anthropic(monkeypatch, fail)
    error = error_type("model_not_found")
    error.status_code = 404
    error.body = {"type": "not_found_error", "message": "model_not_found"}
    pending["error"] = error
    with pytest.raises(llm.ProviderUnavailable, match="cannot access"):
        llm.call_anthropic(model="claude-missing", prompt="p")


@pytest.mark.parametrize("status,body", [
    (404, "unrelated route missing"),
    (400, "ordinary invalid request"),
])
def test_unknown_client_errors_keep_the_debugging_path(status, body):
    assert friendly_provider_error(
        "OpenAI-compatible", "model", status=status, body=body,
    ) is None


def test_provider_diagnostic_does_not_reflect_response_body():
    secret = "acct-secret-123"
    message = friendly_provider_error(
        "OpenAI-compatible", "model", status=402,
        body=f"insufficient_quota for {secret}",
    )
    assert message is not None and secret not in message
