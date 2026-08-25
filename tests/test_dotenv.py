"""Project-root .env loading semantics."""

from __future__ import annotations

from researchwiki import __main__ as cli


_LOAD_DOTENV = cli._load_dotenv


def test_load_dotenv_accepts_export_and_environment_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITELLM_API_KEY", "secret-from-shell")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RW_LLM_BASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "export RW_LLM_BASE_URL=http://litellm.test/v1\n"
        'export OPENAI_API_KEY="${LITELLM_API_KEY}"\n',
        encoding="utf-8",
    )

    _LOAD_DOTENV()

    assert cli.os.environ["RW_LLM_BASE_URL"] == "http://litellm.test/v1"
    assert cli.os.environ["OPENAI_API_KEY"] == "secret-from-shell"


def test_load_dotenv_does_not_override_explicit_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_LLM_BASE_URL", "http://explicit.test/v1")
    (tmp_path / ".env").write_text(
        "RW_LLM_BASE_URL=http://dotenv.test/v1\n",
        encoding="utf-8",
    )

    _LOAD_DOTENV()

    assert cli.os.environ["RW_LLM_BASE_URL"] == "http://explicit.test/v1"


def test_load_dotenv_skips_missing_environment_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=${MISSING_SECRET}\n",
        encoding="utf-8",
    )

    _LOAD_DOTENV()

    assert "OPENAI_API_KEY" not in cli.os.environ
