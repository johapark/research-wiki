"""Provider preflight for `agent ingest` — fail before spending, not mid-run.

Before this existed, an ingest with no configured provider ran PDF extraction
and metadata reconcile first, then died in the author phase with an uncaught
`RuntimeError` — exit 3 ("internal bug — file a report") for what is a plain
configuration error. The diagnostic a new user got was a 401 quoting the
literal string `lm-studio`, which is what `call_openai_compatible` substitutes
for an unset `OPENAI_API_KEY`; nobody had typed it.

What these pin:
  - The check resolves the endpoint exactly the way `llm.call` does, so a
    preflight verdict and a call-site outcome can't disagree.
  - Loopback endpoints need no key (LM Studio / vLLM / llama.cpp / ollama all
    ignore the Bearer token), and chat-relay needs no credentials at all.
  - An Anthropic key set against an OpenAI-routed config still fails, with the
    config-copy hint — README and prompts/init.md both name this as the
    failure that actually happens, and it is the one `has_synchronous_llm()`
    (any-key-anywhere) waves through.
  - Argv errors are diagnosed before environment errors, so a typo'd path
    stays exit 1 rather than being masked by exit 2.

Hermetic: no network, no LLM, no state.db.
"""

from __future__ import annotations

import pytest

from researchwiki.agents import llm
from researchwiki.errors import EnvironmentFailure
from researchwiki.tasks import agent as agent_cli


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """No credentials and no routing overrides unless a test sets them.

    `conftest._no_dotenv_leak` already blocks the repo `.env`, but a developer
    running pytest with keys exported in their shell would otherwise silently
    skip every negative case here.
    """
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "RW_LLM_BASE_URL", "RW_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_models_config(monkeypatch, tmp_path):
    """Force the zero-config path: no `config/models.yaml`, so every role
    resolves to the OpenAI-compatible fallback at api.openai.com."""
    monkeypatch.setattr(llm.model_config, "config_path", lambda: tmp_path / "models.yaml")
    monkeypatch.setattr(llm.model_config, "base_url", lambda: llm.model_config._FALLBACK_BASE_URL)
    llm.model_config._config.cache_clear()
    yield
    llm.model_config._config.cache_clear()


# ---------- endpoint resolution ----------

@pytest.mark.parametrize("url,is_local", [
    ("http://localhost:1234/v1", True),
    ("http://127.0.0.1:8000/v1", True),
    ("http://0.0.0.0:11434/v1", True),
    ("http://[::1]:1234/v1", True),
    ("https://api.openai.com/v1", False),
    ("https://generativelanguage.googleapis.com/v1beta/openai", False),
    ("not a url", False),
])
def test_is_local_endpoint(url, is_local):
    assert llm._is_local_endpoint(url) is is_local


def test_base_url_precedence_matches_call_site(monkeypatch, no_models_config):
    """RW_LLM_BASE_URL wins over the config's base_url, which wins over the
    LM Studio default — the same order `llm.call` applies."""
    assert llm._resolved_openai_base_url() == "https://api.openai.com/v1"
    monkeypatch.setenv("RW_LLM_BASE_URL", "http://localhost:9999/v1")
    assert llm._resolved_openai_base_url() == "http://localhost:9999/v1"


def test_falls_back_to_lm_studio_default_when_config_declares_none(monkeypatch):
    monkeypatch.setattr(llm.model_config, "base_url", lambda: None)
    assert llm._resolved_openai_base_url() == llm._DEFAULT_LOCAL_BASE_URL


# ---------- credential diagnosis ----------

def test_remote_openai_without_key_is_a_problem(no_models_config):
    problems = llm.missing_provider_credentials()
    assert len(problems) == 1
    assert "OPENAI_API_KEY is not set" in problems[0]
    # The endpoint is named, so the user can see *why* a key is required.
    assert "https://api.openai.com/v1" in problems[0]


def test_remote_openai_with_key_is_clean(monkeypatch, no_models_config):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-whatever")
    assert llm.missing_provider_credentials() == []


def test_local_endpoint_needs_no_key(monkeypatch, no_models_config):
    """The whole point of the local path: no key, nothing leaves the machine."""
    monkeypatch.setenv("RW_LLM_BASE_URL", "http://localhost:1234/v1")
    assert llm.missing_provider_credentials() == []


def test_chat_relay_needs_no_credentials(monkeypatch, no_models_config):
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    assert llm.missing_provider_credentials() == []


def test_anthropic_route_without_key_is_a_problem(monkeypatch, no_models_config):
    monkeypatch.setenv("RW_LLM_PROVIDER", "anthropic")
    problems = llm.missing_provider_credentials()
    assert len(problems) == 1
    assert "ANTHROPIC_API_KEY is not set" in problems[0]


def test_anthropic_key_against_openai_config_still_fails(monkeypatch, no_models_config):
    """The documented trap. `has_synchronous_llm()` returns True here — any key
    counts — which is exactly why the preflight doesn't use it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxxx")
    assert llm.has_synchronous_llm() is True

    problems = llm.missing_provider_credentials()
    assert len(problems) == 1
    assert "OPENAI_API_KEY is not set" in problems[0]
    assert "cp config/models.anthropic.yaml config/models.yaml" in problems[0]


def test_hint_absent_when_no_anthropic_key(no_models_config):
    """The config-copy hint is for the mixed-up case only — don't tell a user
    with no keys at all to copy the Anthropic template."""
    (problem,) = llm.missing_provider_credentials()
    assert "models.anthropic.yaml" not in problem


# ---------- the raising wrapper ----------

def test_preflight_raises_environment_failure(no_models_config):
    """Must derive from EnvironmentFailure so `__main__`'s funnel maps it to
    exit 2 — a config error, not exit 3 "internal bug"."""
    assert issubclass(llm.ProviderUnavailable, EnvironmentFailure)
    with pytest.raises(llm.ProviderUnavailable) as e:
        llm.preflight_providers()
    msg = str(e.value)
    assert "no usable LLM provider" in msg
    # Names the config it actually read, flagged when absent.
    assert "missing — using built-in defaults" in msg
    assert "README.md" in msg


def test_preflight_silent_when_configured(monkeypatch, no_models_config):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-whatever")
    llm.preflight_providers()  # must not raise


# ---------- CLI ordering ----------

def test_ingest_without_provider_exits_2(tmp_path, monkeypatch, no_models_config):
    """End-to-end through the CLI funnel: exit 2, no traceback."""
    from researchwiki import __main__ as cli
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    assert cli.main(["agent", "ingest", str(pdf)]) == 2


def test_bad_path_still_exits_1_without_a_provider(no_models_config):
    """Argv is validated before the environment: a path that doesn't exist is
    a user-input error (1) even when credentials are also missing, so the user
    is told about the typo they can fix rather than the key they may not need."""
    with pytest.raises(SystemExit) as e:
        agent_cli.main(["ingest", "/does/not/exist.pdf"])
    assert e.value.code == 1


def test_stub_ingest_skips_preflight(tmp_path, monkeypatch, no_models_config):
    """`--stub` never reaches a provider, so it must run with no credentials —
    it's the offline framework test."""
    seen: list[str] = []
    monkeypatch.setattr(llm, "preflight_providers",
                        lambda: seen.append("preflight"))
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    def _fake_run_ingest(*a, **kw):
        seen.append("run_ingest")
        raise RuntimeError("stop here — nothing past this point is under test")

    monkeypatch.setattr(agent_cli, "run_ingest", _fake_run_ingest)
    agent_cli.main(["ingest", str(pdf), "--stub"])
    # Reached the ingest without ever consulting the provider environment.
    assert seen == ["run_ingest"]
