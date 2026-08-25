"""Status validates model profiles and reports effective endpoint provenance."""

from __future__ import annotations

import pytest

from researchwiki import __main__ as cli
from researchwiki.agents import model_config as mc


@pytest.fixture(autouse=True)
def _clean_routing(monkeypatch):
    for name in (
        "RW_MODELS_CONFIG", "RW_LLM_BASE_URL", "RW_LLM_PROVIDER", "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    mc.clear_caches()
    yield
    mc.clear_caches()


def _empty_wiki(tmp_path, monkeypatch):
    (tmp_path / "wiki").mkdir()
    (tmp_path / "config").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path / "config"


def _write_openai_config(path, endpoint: str) -> None:
    path.write_text(
        f"base_url: {endpoint}\n"
        "roles:\n"
        "  author: {provider: openai-compatible, model: test-model}\n",
        encoding="utf-8",
    )


def test_status_with_malformed_explicit_config_exits_2(tmp_path, monkeypatch, capsys):
    config_dir = _empty_wiki(tmp_path, monkeypatch)
    broken = config_dir / "models.broken.yaml"
    broken.write_text("roles: [unterminated\n", encoding="utf-8")
    monkeypatch.setenv("RW_MODELS_CONFIG", broken.name)

    assert cli.main(["status"]) == 2

    captured = capsys.readouterr()
    assert "cannot be parsed" in captured.err
    assert "Traceback" not in captured.err


def test_status_with_malformed_implicit_config_exits_2(tmp_path, monkeypatch, capsys):
    config_dir = _empty_wiki(tmp_path, monkeypatch)
    (config_dir / "models.yaml").write_text("roles: [unterminated\n", encoding="utf-8")

    assert cli.main(["status"]) == 2

    captured = capsys.readouterr()
    assert "implicit model config" in captured.err
    assert "cannot be parsed" in captured.err
    assert "Traceback" not in captured.err


def test_status_with_ambiguous_implicit_endpoint_exits_2(
    tmp_path, monkeypatch, capsys,
):
    config_dir = _empty_wiki(tmp_path, monkeypatch)
    (config_dir / "models.yaml").write_text(
        "roles:\n  author: {provider: openai-compatible, model: test-model}\n",
        encoding="utf-8",
    )

    assert cli.main(["status"]) == 2

    captured = capsys.readouterr()
    assert "does not declare a top-level base_url" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(("use_env", "expected_url", "expected_source"), [
    (False, "https://config.invalid/v1", "config"),
    (True, "https://env.invalid/v1", "env"),
])
def test_status_reports_effective_openai_endpoint_source(
    tmp_path, monkeypatch, capsys, use_env, expected_url, expected_source,
):
    config_dir = _empty_wiki(tmp_path, monkeypatch)
    config = config_dir / "models.selected.yaml"
    _write_openai_config(config, "https://config.invalid/v1")
    monkeypatch.setenv("RW_MODELS_CONFIG", config.name)
    if use_env:
        monkeypatch.setenv("RW_LLM_BASE_URL", "https://env.invalid/v1")

    assert cli.main(["status"]) == 0

    out = capsys.readouterr().out
    assert f"Model endpoint:        {expected_url} [source={expected_source}]" in out


def test_status_reports_zero_config_fallback_source(tmp_path, monkeypatch, capsys):
    _empty_wiki(tmp_path, monkeypatch)

    assert cli.main(["status"]) == 0

    out = capsys.readouterr().out
    assert "Model endpoint:        https://api.openai.com/v1 [source=fallback]" in out


def test_status_does_not_show_openai_endpoint_for_anthropic_only_config(
    tmp_path, monkeypatch, capsys,
):
    config_dir = _empty_wiki(tmp_path, monkeypatch)
    config = config_dir / "models.anthropic.yaml"
    role_lines = "\n".join(
        f"  {name}: {{provider: anthropic, model: claude-test}}"
        for name in ("author", "critic", "judge", "classifier", "proposer", "extractor")
    )
    config.write_text(f"roles:\n{role_lines}\n", encoding="utf-8")
    monkeypatch.setenv("RW_MODELS_CONFIG", config.name)

    assert cli.main(["status"]) == 0

    out = capsys.readouterr().out
    assert "Model provider(s):     anthropic" in out
    assert "Model endpoint:        (Anthropic SDK default) [source=fallback]" in out
    assert "api.openai.com" not in out
    assert "localhost:1234" not in out


def test_status_rejects_unsafe_anthropic_endpoint_without_echo(
    tmp_path, monkeypatch, capsys,
):
    config_dir = _empty_wiki(tmp_path, monkeypatch)
    config = config_dir / "models.anthropic.yaml"
    role_lines = "\n".join(
        f"  {name}: {{provider: anthropic, model: claude-test}}"
        for name in ("author", "critic", "judge", "classifier", "proposer", "extractor")
    )
    config.write_text(f"roles:\n{role_lines}\n", encoding="utf-8")
    unsafe = "http://remote.invalid/anthropic?token=must-not-echo"
    monkeypatch.setenv("RW_MODELS_CONFIG", config.name)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", unsafe)

    assert cli.main(["status"]) == 2

    captured = capsys.readouterr()
    assert "ANTHROPIC_BASE_URL" in captured.err
    assert unsafe not in captured.err
    assert "Traceback" not in captured.err
