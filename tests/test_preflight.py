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
    for var in (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "RW_LLM_BASE_URL",
        "RW_LLM_PROVIDER", "RW_MODELS_CONFIG", "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def no_models_config(monkeypatch, tmp_path):
    """Force the zero-config path: no `config/models.yaml`, so every role
    resolves to the OpenAI-compatible fallback at api.openai.com."""
    monkeypatch.setattr(llm.model_config, "config_path", lambda: tmp_path / "models.yaml")
    monkeypatch.setattr(llm.model_config, "base_url", lambda: llm.model_config._FALLBACK_BASE_URL)
    llm.model_config.clear_caches()
    yield
    llm.model_config.clear_caches()


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
    resolved = llm.resolve_openai_endpoint()
    assert (resolved.url, resolved.source) == ("https://api.openai.com/v1", "fallback")
    monkeypatch.setenv("RW_LLM_BASE_URL", "http://localhost:9999/v1")
    resolved = llm.resolve_openai_endpoint()
    assert (resolved.url, resolved.source) == ("http://localhost:9999/v1", "env")


def test_unsafe_env_endpoint_fails_before_resolution_or_availability(
    monkeypatch, no_models_config,
):
    unsafe = "http://remote.invalid/v1?token=must-not-echo"
    monkeypatch.setenv("RW_LLM_BASE_URL", unsafe)
    with pytest.raises(llm.model_config.ModelConfigUnavailable) as exc:
        llm.resolve_openai_endpoint()
    assert "RW_LLM_BASE_URL" in str(exc.value)
    assert unsafe not in str(exc.value)
    with pytest.raises(llm.model_config.ModelConfigUnavailable):
        llm.has_synchronous_llm()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(llm.model_config.ModelConfigUnavailable):
        llm.has_synchronous_llm()
    with pytest.raises(llm.model_config.ModelConfigUnavailable):
        llm.call_openai_compatible(model="test-model", prompt="must not be sent")


def test_explicit_unsafe_openai_endpoint_fails_before_request(monkeypatch):
    monkeypatch.delenv("RW_LLM_BASE_URL", raising=False)
    with pytest.raises(llm.model_config.ModelConfigUnavailable):
        llm.call_openai_compatible(
            model="test-model",
            prompt="must not be sent",
            base_url="http://remote.invalid/v1?token=must-not-echo",
        )


def test_falls_back_to_lm_studio_default_when_config_declares_none(monkeypatch):
    monkeypatch.setattr(llm.model_config, "base_url", lambda: None)
    resolved = llm.resolve_openai_endpoint()
    assert (resolved.url, resolved.source) == (llm._DEFAULT_LOCAL_BASE_URL, "fallback")


def test_endpoint_resolution_reports_config_source(monkeypatch, tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text("base_url: https://provider.invalid/v1\n", encoding="utf-8")
    monkeypatch.setattr(llm.model_config, "config_path", lambda: config)
    monkeypatch.setattr(
        llm.model_config, "base_url", lambda: "https://provider.invalid/v1",
    )
    resolved = llm.resolve_openai_endpoint()
    assert (resolved.url, resolved.source) == ("https://provider.invalid/v1", "config")


def test_endpoint_source_uses_the_same_cached_file_snapshot(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "models.yaml"
    monkeypatch.setattr(llm.model_config, "wiki_root", lambda: tmp_path)
    llm.model_config.clear_caches()
    try:
        resolved = llm.resolve_openai_endpoint()
        assert (resolved.url, resolved.source) == (
            llm.model_config._FALLBACK_BASE_URL,
            "fallback",
        )

        config.write_text(
            "base_url: https://provider.invalid/v1\n",
            encoding="utf-8",
        )
        resolved = llm.resolve_openai_endpoint()
        assert (resolved.url, resolved.source) == (
            llm.model_config._FALLBACK_BASE_URL,
            "fallback",
        )

        llm.model_config.clear_caches()
        resolved = llm.resolve_openai_endpoint()
        assert (resolved.url, resolved.source) == (
            "https://provider.invalid/v1",
            "config",
        )
        config.unlink()
        resolved = llm.resolve_openai_endpoint()
        assert (resolved.url, resolved.source) == (
            "https://provider.invalid/v1",
            "config",
        )
    finally:
        llm.model_config.clear_caches()


def test_endpoint_display_does_not_expose_url_credentials():
    resolved = llm.EndpointResolution(
        "https://user:secret@provider.invalid/v1?api_key=secret#fragment", "config",
    )
    assert resolved.display_url == "https://provider.invalid/v1"


def test_lmstudio_yaml_base_url_counts_as_synchronous(tmp_path, monkeypatch):
    """Selecting the LM Studio profile is itself the local-server signal.

    Users normally select it with ``RW_MODELS_CONFIG=models.lmstudio.yaml``;
    they should not also have to duplicate its ``base_url:`` in an environment
    variable merely to enable synchronous-only features.
    """
    config = tmp_path / "models.lmstudio.yaml"
    config.write_text("base_url: http://localhost:1234/v1\n", encoding="utf-8")
    monkeypatch.setattr(llm.model_config, "config_path", lambda: config)
    llm.model_config.clear_caches()
    try:
        assert llm.has_synchronous_llm() is True
    finally:
        llm.model_config.clear_caches()


@pytest.mark.parametrize("routing", [
    (
        'roles:\n  author: {provider: " OpenAI-Compatible ", model: test-model}\n'
        "phases:\n  author: {role: author}\n"
    ),
    (
        "roles:\n  author: {provider: anthropic, model: test-model}\n"
        'phases:\n  author: {role: author, provider: " OPENAI "}\n'
    ),
])
def test_canonical_provider_id_matches_actual_dispatch(
    tmp_path, monkeypatch, routing,
):
    config = tmp_path / "config"
    config.mkdir()
    (config / "models.yaml").write_text(
        "base_url: https://provider.invalid/v1\n" + routing,
        encoding="utf-8",
    )
    monkeypatch.setattr(llm.model_config, "wiki_root", lambda: tmp_path)
    dispatched = []
    sentinel = object()

    def fake_openai(**kwargs):
        dispatched.append(kwargs["base_url"])
        return sentinel

    monkeypatch.setattr(llm, "call_openai_compatible", fake_openai)
    llm.model_config.clear_caches()
    try:
        assert llm.model_config.for_phase("author").provider in {
            "openai-compatible", "openai",
        }
        assert llm.call(prompt="test", phase="author") is sentinel
        assert dispatched == ["https://provider.invalid/v1"]
    finally:
        llm.model_config.clear_caches()


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


def test_unsafe_anthropic_endpoint_fails_preflight_and_call_without_echo(
    monkeypatch, no_models_config,
):
    unsafe = "http://remote.invalid/anthropic?token=must-not-echo"
    monkeypatch.setenv("RW_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", unsafe)

    with pytest.raises(llm.model_config.ModelConfigUnavailable) as exc:
        llm.missing_provider_credentials()
    assert "ANTHROPIC_BASE_URL" in str(exc.value)
    assert unsafe not in str(exc.value)
    with pytest.raises(llm.model_config.ModelConfigUnavailable):
        llm.call_anthropic(model="claude-test", prompt="must not be sent")


def test_glm_anthropic_endpoint_is_accepted_by_preflight(monkeypatch, no_models_config):
    monkeypatch.setenv("RW_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.z.ai/api/anthropic")
    assert llm.missing_provider_credentials() == []


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
