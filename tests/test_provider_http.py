"""Structured-provider transport keeps outages distinct from valid misses."""

from __future__ import annotations

import json
import subprocess

import pytest

from researchwiki.providers import _http, biorxiv, orcid, pubmed, semantic_scholar
from researchwiki.providers._http import StructuredProviderUnavailable


class _Proc:
    def __init__(self, stdout: str, *, returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_transport_returns_empty_object_for_real_404(monkeypatch):
    monkeypatch.setattr(_http.subprocess, "run", lambda *a, **k: _Proc("body\n404"))
    assert _http.curl_json("https://example.test/missing", provider="test") == {}


def test_transport_raises_after_transient_failures(monkeypatch):
    monkeypatch.setattr(_http.subprocess, "run", lambda *a, **k: _Proc("\n503"))
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)
    with pytest.raises(StructuredProviderUnavailable, match="503"):
        _http.curl_json("https://example.test/down", provider="test", retries=2)


def test_transport_reports_missing_curl_as_environment_failure(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("curl")

    monkeypatch.setattr(_http.subprocess, "run", missing)
    with pytest.raises(StructuredProviderUnavailable, match="not installed"):
        _http.curl_json("https://example.test", provider="test")


def test_semantic_scholar_reports_missing_curl(monkeypatch, tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError("curl")

    monkeypatch.setattr(semantic_scholar, "s2_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(semantic_scholar.subprocess, "run", missing)
    provider = semantic_scholar.SemanticScholarProvider(retries=1)
    with pytest.raises(StructuredProviderUnavailable, match="not installed"):
        provider._fetch("https://example.test")


def test_transport_uses_current_project_identity(monkeypatch):
    seen: list[str] = []

    def run(cmd, **kwargs):
        seen.extend(cmd)
        return _Proc(json.dumps({"ok": True}) + "\n200")

    monkeypatch.setattr(_http.subprocess, "run", run)
    assert _http.curl_json("https://example.test", provider="test") == {"ok": True}
    user_agent = seen[seen.index("-A") + 1]
    assert "github.com/johapark/research-wiki" in user_agent
    assert "anthropic/claude-code" not in user_agent


@pytest.mark.parametrize(
    ("module", "call", "cache_name"),
    [
        (biorxiv, lambda: biorxiv.lookup("10.1101/2026.01.01.1"), "bio"),
        (orcid, lambda: orcid.lookup_by_id("0000-0002-1825-0097"), "orcid"),
        (pubmed, lambda: pubmed.retraction_status("10.1000/journal"), "pubmed"),
    ],
)
def test_provider_outage_never_becomes_empty_success(
    tmp_path, monkeypatch, module, call, cache_name
):
    cache = tmp_path / cache_name
    cache.mkdir()
    monkeypatch.setattr(module, "web_cache_dir", lambda: cache)
    monkeypatch.setattr(
        module,
        "_curl_json",
        lambda *a, **k: (_ for _ in ()).throw(
            StructuredProviderUnavailable("simulated outage")
        ),
    )
    with pytest.raises(StructuredProviderUnavailable, match="simulated outage"):
        call()


def test_timeout_type_remains_available_for_transport_mocks():
    """Keep the public test seam explicit: timeout comes from subprocess."""
    assert issubclass(subprocess.TimeoutExpired, Exception)
