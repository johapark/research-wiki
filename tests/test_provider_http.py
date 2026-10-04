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


# ---------- keyword-search transport (`scout search`) ----------

def test_400_is_retried_by_default_but_rejected_fast_when_opted_in(monkeypatch):
    """DOI lookups keep their old behaviour; search callers opt into exit 1."""
    calls = []
    monkeypatch.setattr(_http.subprocess, "run",
                        lambda *a, **k: calls.append(1) or _Proc("bad field\n400"))
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)
    with pytest.raises(StructuredProviderUnavailable, match="400"):
        _http.curl_json("https://example.test", provider="test", retries=2)
    assert len(calls) == 2
    calls.clear()
    with pytest.raises(_http.ProviderRequestRejected, match="bad field"):
        _http.curl_json("https://example.test", provider="test", retries=3, reject_400=True)
    assert len(calls) == 1


def test_curl_body_returns_text_and_none_on_404(monkeypatch):
    monkeypatch.setattr(_http.subprocess, "run", lambda *a, **k: _Proc("<feed/>\n200"))
    assert _http.curl_body("https://example.test", provider="test") == "<feed/>"
    monkeypatch.setattr(_http.subprocess, "run", lambda *a, **k: _Proc("\n404"))
    assert _http.curl_body("https://example.test", provider="test") is None


def test_secret_query_never_reaches_argv_or_logs(monkeypatch, capsys):
    seen = {}

    def run(cmd, **kwargs):
        seen["cmd"], seen["input"] = cmd, kwargs.get("input")
        return _Proc("{}\n200")

    monkeypatch.setattr(_http.subprocess, "run", run)
    _http.curl_json("https://example.test/q?term=x", provider="test",
                    secret_query={"api_key": "sekrit"})
    assert not any("sekrit" in part for part in seen["cmd"])
    assert seen["cmd"][-2:] == ["-K", "-"]
    assert 'url = "https://example.test/q?term=x&api_key=sekrit"' in seen["input"]
    assert "sekrit" not in capsys.readouterr().err


def test_redact_blanks_secret_looking_params():
    assert _http.redact("https://x.test/a?term=q&api_key=k1&email=a@b") == (
        "https://x.test/a?term=q&api_key=***&email=***")


def _download_run(responses):
    """Fake curl: each call pops (status, redirect, body) and writes the -o file."""
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd[-1 - cmd[::-1].index("-o") + 1] if "-o" in cmd else None)
        status, redirect, body = responses.pop(0)
        out = cmd[cmd.index("-o") + 1]
        with open(out, "wb") as fh:
            fh.write(body)
        return _Proc(f"{status}\n{redirect}")
    return run, calls


def test_download_follows_allowed_redirects_and_lands_atomically(tmp_path, monkeypatch):
    run, _ = _download_run([
        ("302", "https://b.test/real.pdf", b""),
        ("200", "", b"%PDF-1.7 body"),
    ])
    monkeypatch.setattr(_http.subprocess, "run", run)
    dest = tmp_path / "x.pdf"
    assert _http.curl_download("https://a.test/x", dest, provider="t",
                               allowed_hosts={"a.test", "b.test"}, max_bytes=1000) == dest
    assert dest.read_bytes().startswith(b"%PDF-")
    assert [p.name for p in tmp_path.iterdir()] == ["x.pdf"]


@pytest.mark.parametrize(("responses", "reason"), [
    ([("302", "https://evil.test/x.pdf", b"")], "off-allowlist"),
    ([("302", "http://a.test/x.pdf", b"")], "not-https"),
    ([("200", "", b"<!DOCTYPE html>challenge")], "not-pdf"),
    ([("403", "", b"denied")], "http-403"),
])
def test_download_refusals_leave_no_file(tmp_path, monkeypatch, responses, reason):
    run, _ = _download_run(responses)
    monkeypatch.setattr(_http.subprocess, "run", run)
    with pytest.raises(_http.DownloadRefused) as exc:
        _http.curl_download("https://a.test/x", tmp_path / "x.pdf", provider="t",
                            allowed_hosts={"a.test"}, max_bytes=1000)
    assert exc.value.reason == reason
    assert list(tmp_path.iterdir()) == []


def test_download_never_overwrites(tmp_path, monkeypatch):
    dest = tmp_path / "x.pdf"
    dest.write_bytes(b"%PDF-original")
    monkeypatch.setattr(_http.subprocess, "run", lambda *a, **k: pytest.fail("fetched"))
    with pytest.raises(FileExistsError):
        _http.curl_download("https://a.test/x", dest, provider="t",
                            allowed_hosts={"a.test"}, max_bytes=1000)
    assert dest.read_bytes() == b"%PDF-original"


def test_download_refuses_an_off_list_start_url_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setattr(_http.subprocess, "run", lambda *a, **k: pytest.fail("fetched"))
    with pytest.raises(_http.DownloadRefused, match="off-allowlist"):
        _http.curl_download("https://evil.test/x", tmp_path / "x.pdf", provider="t",
                            allowed_hosts={"a.test"}, max_bytes=1000)
