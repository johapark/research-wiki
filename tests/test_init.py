"""Unit tests for the `researchwiki init` wizard's pure helpers.

The interactive I/O isn't exercised here — these pin the load-bearing pure
functions: `.env` upsert (create / replace / preserve), slug validation,
provider→template/env mapping, and the invariant that the scaffolded dashboard
carries no `category:` (a root-level page with one trips lint's category-drift
check — see researchwiki/tasks/lint/yaml_checks.py:find_category_drift).
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from researchwiki import env_profiles
from researchwiki.errors import EnvironmentFailure
from researchwiki.tasks import init
from researchwiki.tasks import _provider_setup


ROOT = Path(__file__).resolve().parents[1]


def _install_template(config: Path, name: str) -> None:
    """Give an isolated wizard test the template a real clone contains."""
    (config / name).write_bytes((ROOT / "config" / name).read_bytes())


# ── _template_for_provider / _env_updates_for_provider ───────────────────────

def test_template_mapping():
    assert init._template_for_provider("anthropic") == "models.anthropic.yaml"
    assert init._template_for_provider("openai-compatible") == "models.openai-compatible.yaml"
    assert init._template_for_provider("local") == "models.lmstudio.yaml"
    # Chat-relay reuses the anthropic template (it's an env override).
    assert init._template_for_provider("chat-relay") == "models.anthropic.yaml"


def test_openai_maps_to_no_template():
    """The default provider is zero-config: the fallback roles already point at
    OpenAI, so the right `config/models.yaml` for it is no file at all.

    Not `models.chatgpt.yaml` — that template puts author/critic/judge on
    gpt-5.6-terra (~$0.071/paper per its own header) where the fallback uses
    gpt-5.6-luna (~$0.009/paper), so copying it would make the recommended
    choice ~7x dearer than choosing nothing."""
    assert init._template_for_provider("openai") is None


def test_openai_is_the_recommended_default():
    """Menu entry 1 is what `_ask_choice`'s default selects, and README plus
    `model_config._FALLBACK_ROLES` both make OpenAI the default. This pins the
    three in agreement — they were not, and the wizard steered new users onto
    the ~10x-dearer provider while calling it the default."""
    assert init._PROVIDER_MENU[0][0] == "openai"
    assert "RECOMMENDED" in init._PROVIDER_MENU[0][2]
    # Every menu id must be routable, or the step raises picking it.
    for pid, _label, _blurb in init._PROVIDER_MENU:
        assert pid in init._TEMPLATE_BY_PROVIDER
    # Anthropic is still offered — it just isn't the default any more.
    assert "anthropic" in {pid for pid, _, _ in init._PROVIDER_MENU}


def test_env_updates_anthropic():
    assert init._env_updates_for_provider("anthropic", api_key="sk-ant") == {
        "ANTHROPIC_API_KEY": "sk-ant"
    }
    # No key supplied → nothing to write (user sets it later).
    assert init._env_updates_for_provider("anthropic") == {}


def test_env_updates_openai_writes_key_but_no_base_url():
    """The zero-config path must not park RW_LLM_BASE_URL in .env: the built-in
    fallback already points at api.openai.com, and README keeps that var a shell
    export so swapping backends doesn't touch the file."""
    assert init._env_updates_for_provider("openai", api_key="sk-x") == {
        "OPENAI_API_KEY": "sk-x"
    }
    u = init._env_updates_for_provider("openai", api_key="sk-x",
                                       base_url="https://api.openai.com/v1")
    assert "RW_LLM_BASE_URL" not in u


def test_env_updates_openai_compatible():
    assert init._env_updates_for_provider(
        "openai-compatible", api_key="sk-x", base_url="https://api.openai.com/v1"
    ) == {"OPENAI_API_KEY": "sk-x"}


def test_env_updates_local_has_no_key():
    u = init._env_updates_for_provider("local", base_url="http://localhost:1234/v1")
    assert u == {"RW_LLM_BASE_URL": "http://localhost:1234/v1"}
    assert "OPENAI_API_KEY" not in u and "ANTHROPIC_API_KEY" not in u


def test_env_updates_chat_relay_is_fixed():
    assert init._env_updates_for_provider("chat-relay") == {"RW_LLM_PROVIDER": "chat-relay"}


def test_stale_routing_keys_follow_provider_ownership():
    assert init._stale_routing_keys("openai") == {
        "RW_MODELS_CONFIG", "RW_LLM_PROVIDER", "RW_LLM_BASE_URL",
    }
    assert init._stale_routing_keys("openai-compatible") == {
        "RW_MODELS_CONFIG", "RW_LLM_PROVIDER", "RW_LLM_BASE_URL",
    }
    assert init._stale_routing_keys("local") == {
        "RW_MODELS_CONFIG", "RW_LLM_PROVIDER",
    }
    assert init._stale_routing_keys("anthropic") == {
        "RW_MODELS_CONFIG", "RW_LLM_PROVIDER", "RW_LLM_BASE_URL",
        "ANTHROPIC_BASE_URL",
    }
    assert init._stale_routing_keys("chat-relay") == {
        "RW_MODELS_CONFIG", "RW_LLM_BASE_URL",
    }
    assert init._stale_routing_keys(
        "openai-compatible", preserve_models_config=True,
    ) == {"RW_LLM_PROVIDER", "RW_LLM_BASE_URL"}


def test_active_env_path_prefers_cli_selected_profile(tmp_path, monkeypatch):
    assert init._active_env_path(tmp_path) == tmp_path / ".env"
    selected = tmp_path / ".env.litellm"
    monkeypatch.setenv(init._ACTIVE_ENV_FILE_VAR, str(selected))
    assert init._active_env_path(tmp_path) == selected


def test_effective_provider_prefers_forced_provider(tmp_path, monkeypatch):
    models = tmp_path / "models.yaml"
    models.write_text("roles:\n  author:\n    provider: anthropic\n")
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    monkeypatch.delenv("RW_MODELS_CONFIG", raising=False)

    assert init._effective_provider(models) == ("chat-relay", "RW_LLM_PROVIDER")


def test_effective_provider_reads_named_models_config(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "models.named.yaml").write_text(
        "roles:\n  author:\n    provider: openai-compatible\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RW_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("RW_MODELS_CONFIG", "models.named.yaml")

    assert init._effective_provider(config / "models.yaml") == (
        "openai-compatible", "RW_MODELS_CONFIG='models.named.yaml'",
    )


def test_effective_provider_reports_explicit_empty_models_override(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("RW_MODELS_CONFIG", "")
    monkeypatch.delenv("RW_LLM_PROVIDER", raising=False)

    assert init._effective_provider(tmp_path / "models.yaml") == (
        "unavailable", "RW_MODELS_CONFIG=''",
    )


def test_remove_env_keys_preserves_credentials_and_restricts_mode(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# keep this\n"
        'OPENAI_API_KEY="secret-not-printed"\n'
        'RW_LLM_BASE_URL="http://old.invalid/v1"\n'
        'export RW_LLM_PROVIDER="chat-relay"\n'
    )
    env.chmod(0o644)

    removed = init._remove_env_keys(
        env, {"RW_LLM_BASE_URL", "RW_LLM_PROVIDER"}
    )

    assert removed == {"RW_LLM_BASE_URL", "RW_LLM_PROVIDER"}
    text = env.read_text()
    assert "# keep this" in text
    assert "OPENAI_API_KEY" in text
    assert "RW_LLM_BASE_URL" not in text
    assert "RW_LLM_PROVIDER" not in text
    assert env.stat().st_mode & 0o077 == 0


def test_provider_base_url_requires_absolute_http_url():
    assert init._valid_provider_base_url("https://api.groq.com/openai/v1")
    assert init._valid_provider_base_url("http://localhost:1234/v1")
    assert init._valid_provider_base_url("http://api.groq.com/openai/v1")
    assert init._valid_provider_base_url("http://10.212.23.212/v1")
    assert init._valid_provider_base_url("http://192.168.1.10:1234/v1")
    assert not init._valid_provider_base_url("api.groq.com/openai/v1")
    assert not init._valid_provider_base_url("file:///tmp/provider")
    assert not init._valid_provider_base_url("https://example.com:bad/v1")
    assert not init._valid_provider_base_url("https://example.com:/v1")
    assert not init._valid_provider_base_url("https://example.com:0/v1")
    assert not init._valid_provider_base_url("https://example.com:99999/v1")
    assert not init._valid_provider_base_url("https://example.com/v1?key=x")
    assert not init._valid_provider_base_url("https://example.com/v1#fragment")
    assert not init._valid_provider_base_url("https://user:secret@example.com/v1")
    assert not init._valid_provider_base_url("https://exa mple.com/v1")
    assert not init._valid_provider_base_url("https://exa\tmple.com/v1")
    assert not init._valid_provider_base_url("https://exa\nmple.com/v1")
    assert not init._valid_provider_base_url("https://exa\u00a0mple.com/v1")
    assert not init._valid_provider_base_url("https://exa\u200bmple.com/v1")
    assert not init._valid_provider_base_url("")


def test_endpoint_change_can_explicitly_reuse_existing_key(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.setenv("OPENAI_API_KEY", "old-provider-secret")
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: True)

    proceed, replacement = init._choose_openai_api_key(
        tmp_path / ".env",
        previous_endpoint="https://api.openai.com/v1",
        selected_endpoint="https://api.groq.com/openai/v1",
        prompt="unused",
    )

    assert proceed is True and replacement is None
    assert "Reusing OPENAI_API_KEY" in capsys.readouterr().out


def test_endpoint_change_replaces_key_owned_by_selected_profile(
    tmp_path, monkeypatch,
):
    profile = tmp_path / ".env.provider"
    profile.write_text('export OPENAI_API_KEY="old-provider-secret"\n')
    _mark_profile_owned(
        monkeypatch, profile, OPENAI_API_KEY="old-provider-secret"
    )
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: False)
    monkeypatch.setattr(init, "_ask", lambda *args, **kwargs: "new-provider-secret")

    proceed, replacement = init._choose_openai_api_key(
        profile,
        previous_endpoint="https://api.openai.com/v1",
        selected_endpoint="https://api.groq.com/openai/v1",
        prompt="unused",
    )

    assert proceed is True
    assert replacement == "new-provider-secret"


def test_endpoint_change_cannot_replace_parent_shell_key(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.setenv("OPENAI_API_KEY", "parent-shell-secret")
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: False)

    proceed, replacement = init._choose_openai_api_key(
        tmp_path / ".env",
        previous_endpoint="https://api.openai.com/v1",
        selected_endpoint="https://api.groq.com/openai/v1",
        prompt="unused",
    )

    assert proceed is False and replacement is None
    out = capsys.readouterr().out
    assert "parent shell" in out
    assert "Unset it there" in out


def test_equal_value_in_profile_does_not_disguise_parent_shell_key(
    tmp_path, monkeypatch, capsys,
):
    profile = tmp_path / ".env.provider"
    profile.write_text('OPENAI_API_KEY="same-secret"\n')
    profile.chmod(0o600)
    monkeypatch.setenv("OPENAI_API_KEY", "same-secret")
    monkeypatch.setenv(init._ACTIVE_ENV_FILE_VAR, str(profile.resolve()))
    monkeypatch.setenv(
        env_profiles.DOTENV_PROVENANCE_VAR,
        json.dumps({"path": str(profile.resolve()), "keys": []}),
    )
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: False)

    proceed, replacement = init._choose_openai_api_key(
        profile,
        previous_endpoint="https://api.openai.com/v1",
        selected_endpoint="https://api.groq.com/openai/v1",
        prompt="unused",
    )

    assert proceed is False and replacement is None
    assert "parent shell" in capsys.readouterr().out


def test_reuse_external_key_rejects_different_shadowed_profile_key(
    tmp_path, monkeypatch, capsys,
):
    profile = tmp_path / ".env.provider"
    profile.write_text('OPENAI_API_KEY="future-profile-secret"\n')
    profile.chmod(0o600)
    monkeypatch.setenv("OPENAI_API_KEY", "current-shell-secret")
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: True)

    proceed, replacement = init._choose_openai_api_key(
        profile,
        previous_endpoint="https://api.openai.com/v1",
        selected_endpoint="https://api.groq.com/openai/v1",
        prompt="unused",
    )

    assert proceed is False and replacement is None
    assert "different OPENAI_API_KEY" in capsys.readouterr().out


def test_anthropic_endpoint_change_requires_explicit_key_reuse(
    tmp_path, monkeypatch,
):
    profile = tmp_path / ".env.provider"
    profile.write_text('ANTHROPIC_API_KEY="custom-endpoint-secret"\n')
    _mark_profile_owned(
        monkeypatch, profile,
        ANTHROPIC_API_KEY="custom-endpoint-secret",
    )
    confirmations = []
    monkeypatch.setattr(
        init,
        "_confirm",
        lambda prompt, default=False: confirmations.append((prompt, default)) or True,
    )

    proceed, replacement = init._choose_anthropic_api_key(
        profile,
        previous_endpoint="https://compat.example/anthropic",
        prompt="unused",
    )

    assert proceed is True and replacement is None
    assert len(confirmations) == 1
    assert "ANTHROPIC_API_KEY" in confirmations[0][0]
    assert confirmations[0][1] is False


def test_local_endpoint_change_cannot_hide_parent_shell_key(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.setenv("OPENAI_API_KEY", "parent-shell-secret")
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: False)

    proceed, remove_key = init._prepare_local_api_key(
        tmp_path / ".env",
        previous_endpoint="https://api.openai.com/v1",
        selected_endpoint="http://localhost:1234/v1",
    )

    assert proceed is False and remove_key is False
    out = capsys.readouterr().out
    assert "parent shell" in out
    assert "Unset it there" in out


def test_local_endpoint_change_can_remove_profile_owned_key(
    tmp_path, monkeypatch,
):
    profile = tmp_path / ".env.local"
    profile.write_text('export OPENAI_API_KEY="cloud-secret"\n')
    _mark_profile_owned(monkeypatch, profile, OPENAI_API_KEY="cloud-secret")
    answers = iter((False, True))
    monkeypatch.setattr(
        init, "_confirm", lambda *args, **kwargs: next(answers)
    )

    proceed, remove_key = init._prepare_local_api_key(
        profile,
        previous_endpoint="https://api.openai.com/v1",
        selected_endpoint="http://localhost:1234/v1",
    )

    assert proceed is True and remove_key is True


def test_local_endpoint_can_remove_reference_loaded_profile_key(
    tmp_path, monkeypatch,
):
    profile = tmp_path / ".env.local"
    profile.write_text('OPENAI_API_KEY="${LITELLM_API_KEY}"\n')
    monkeypatch.setenv("LITELLM_API_KEY", "cloud-secret")
    _mark_profile_owned(monkeypatch, profile, OPENAI_API_KEY="cloud-secret")
    answers = iter((False, True))
    monkeypatch.setattr(
        init, "_confirm", lambda *args, **kwargs: next(answers)
    )

    proceed, remove_key = init._prepare_local_api_key(
        profile,
        previous_endpoint="https://api.openai.com/v1",
        selected_endpoint="http://localhost:1234/v1",
    )

    assert proceed is True and remove_key is True


# ── _ask_choice ──────────────────────────────────────────────────────────────

def _answers(monkeypatch, *replies):
    """Feed `_ask` a scripted sequence of answers."""
    seq = iter(replies)
    monkeypatch.setattr(init, "_ask", lambda prompt, default=None: next(seq))


def _mark_profile_owned(monkeypatch, profile, **values):
    profile.chmod(0o600)
    monkeypatch.setenv(init._ACTIVE_ENV_FILE_VAR, str(profile.resolve()))
    monkeypatch.setenv(
        env_profiles.DOTENV_PROVENANCE_VAR,
        json.dumps({"path": str(profile.resolve()), "keys": sorted(values)}),
    )
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_ask_choice_accepts_valid(monkeypatch):
    _answers(monkeypatch, "3")
    assert init._ask_choice(5) == 2  # 0-based


def test_ask_choice_reprompts_instead_of_defaulting(monkeypatch, capsys):
    """A typo must not silently pick entry 1. The old code printed
    "defaulting to Anthropic" and proceeded, so a slip chose the dearest
    provider on the menu."""
    _answers(monkeypatch, "yes", "9", "2")
    assert init._ask_choice(5) == 1
    out = capsys.readouterr().out
    assert "isn't a number" in out
    assert "out of range" in out


def test_category_menu_reprompts_too(monkeypatch, tmp_path, capsys):
    """Both wizard menus go through `_ask_choice`. The category menu used to
    compare the raw string to "1" and treat everything else as "manual", so a
    typo silently chose an option the user hadn't picked."""
    monkeypatch.setattr(init, "content_categories", lambda: frozenset({"other"}))
    from researchwiki.tasks.bootstrap_categories import MIN_INBOX_FOR_BOOTSTRAP
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for i in range(MIN_INBOX_FOR_BOOTSTRAP):
        (inbox / f"p{i}.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(init, "inbox_dir", lambda: inbox)
    seen = []
    monkeypatch.setattr(init, "_bootstrap_categories", lambda: seen.append("bootstrap"))
    monkeypatch.setattr(init, "_manual_categories", lambda root: seen.append("manual"))
    _answers(monkeypatch, "y", "1")   # "y" is not a choice → re-prompt, then pick 1

    init._step_categories(tmp_path)
    assert seen == ["bootstrap"]
    assert "isn't a number" in capsys.readouterr().out


def test_categories_defer_until_corpus_reaches_bootstrap_threshold(
    monkeypatch, tmp_path, capsys,
):
    """A first-time user should not design taxonomy before seeing one ingest."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "first.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(init, "inbox_dir", lambda: inbox)
    monkeypatch.setattr(init, "content_categories", lambda: frozenset({"other"}))
    monkeypatch.setattr(
        init, "_ask_choice",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not prompt")),
    )

    init._step_categories(tmp_path)

    out = capsys.readouterr().out
    assert "Using wiki/other/ for now" in out
    assert "bootstrap-categories --apply" in out


def test_confirm_does_not_offer_ingest_while_doctor_is_blocked(
    monkeypatch, capsys,
):
    from researchwiki.tasks import doctor

    monkeypatch.setattr(doctor, "main", lambda argv: 1)

    assert init._step_confirm() == 1
    out = capsys.readouterr().out
    assert "Setup still needs attention" in out
    assert "Next: add any PDF path" not in out


def test_confirm_offers_arbitrary_pdf_path_when_ready(monkeypatch, capsys):
    from researchwiki.tasks import doctor

    monkeypatch.setattr(doctor, "main", lambda argv: 0)

    assert init._step_confirm() == 0
    out = capsys.readouterr().out
    assert "Next: add any PDF path" in out
    assert "researchwiki add /path/to/paper.pdf" in out


def test_ask_choice_empty_takes_the_default(monkeypatch):
    """Bare Enter accepts the recommendation — `_ask` returns the default,
    which is also how EOF resolves, so this terminates on a closed stdin."""
    monkeypatch.setattr(init, "_ask", lambda prompt, default=None: default)
    assert init._ask_choice(5) == 0


def test_bootstrap_threshold_is_not_restated(monkeypatch, tmp_path, capsys):
    """The wizard must source the PDF threshold from `bootstrap_categories`,
    not restate it. It hardcoded 5 against the real value of 3, so users with
    3-4 PDFs were told bootstrap was unavailable when it would have worked."""
    from researchwiki.tasks.bootstrap_categories import MIN_INBOX_FOR_BOOTSTRAP

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for i in range(MIN_INBOX_FOR_BOOTSTRAP):        # exactly at the threshold
        (inbox / f"p{i}.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(init, "inbox_dir", lambda: inbox)

    called = {}
    import researchwiki.tasks.bootstrap_categories as bc
    monkeypatch.setattr(bc, "main", lambda argv: called.setdefault("argv", argv))

    init._bootstrap_categories()
    assert called.get("argv") == ["--apply"], "threshold blocked a valid PDF count"
    assert "manually" not in capsys.readouterr().out


# ── _write_models_config ─────────────────────────────────────────────────────

def _config_dir(tmp_path):
    d = tmp_path / "config"
    d.mkdir()
    (d / "models.anthropic.yaml").write_text("roles:\n  author:\n    provider: anthropic\n")
    return d


def test_openai_choice_writes_no_config(tmp_path, capsys):
    cfg = _config_dir(tmp_path)
    models = cfg / "models.yaml"
    assert init._write_models_config(cfg, models, "openai") is True
    assert not models.exists()
    assert "built-in defaults" in capsys.readouterr().out


def test_openai_choice_removes_a_stale_config(tmp_path, monkeypatch):
    """A leftover models.yaml would override the choice just made and the
    wizard would report success for a provider the user didn't pick."""
    cfg = _config_dir(tmp_path)
    models = cfg / "models.yaml"
    models.write_text("roles:\n  author:\n    provider: anthropic\n")
    monkeypatch.setattr(init, "_confirm", lambda prompt, default=True: True)
    assert init._write_models_config(cfg, models, "openai") is True
    assert not models.exists()


def test_openai_choice_warns_when_stale_config_kept(tmp_path, monkeypatch, capsys):
    cfg = _config_dir(tmp_path)
    models = cfg / "models.yaml"
    models.write_text("roles:\n  author:\n    provider: anthropic\n")
    monkeypatch.setattr(init, "_confirm", lambda prompt, default=True: False)
    assert init._write_models_config(cfg, models, "openai") is False
    assert models.exists()  # user's call, honored
    assert "overrides this choice" in capsys.readouterr().out


def test_non_openai_choice_copies_its_template(tmp_path):
    cfg = _config_dir(tmp_path)
    models = cfg / "models.yaml"
    assert init._write_models_config(cfg, models, "anthropic") is True
    assert "provider: anthropic" in models.read_text()


def test_generic_compatible_config_rewrites_endpoint_and_all_role_models(tmp_path):
    models = tmp_path / "models.yaml"
    models.write_text(
        "# comments stay\n"
        "base_url: https://api.openai.com/v1\n"
        "roles:\n"
        "  author:\n"
        "    provider: openai-compatible\n"
        "    model: gemini-example\n"
        "  judge:\n"
        "    provider: openai-compatible\n"
        "    model: gemini-example\n"
        "  classifier:\n"
        "    provider: openai-compatible\n"
        "    model: gemini-lite-example\n"
        "  critic:\n"
        "    provider: openai-compatible\n"
        "    model: gemini-example\n"
        "  proposer:\n"
        "    provider: openai-compatible\n"
        "    model: gemini-lite-example\n"
        "  extractor:\n"
        "    provider: openai-compatible\n"
        "    model: gemini-lite-example\n"
        "phases:\n"
        "  author: {role: author}\n"
    )

    _provider_setup.customize_openai_compatible_config(
        models,
        base_url="https://api.groq.com/openai/v1",
        quality_model="llama-quality",
        utility_model="llama-fast",
    )

    text = models.read_text()
    assert 'base_url: "https://api.groq.com/openai/v1"' in text
    assert text.count('model: "llama-quality"') == 3
    assert text.count('model: "llama-fast"') == 3
    assert "gemini-example" not in text
    assert "# comments stay" in text


def test_shipped_generic_template_passes_six_role_validation():
    root = Path(__file__).resolve().parents[1]
    template = (root / "config" / "models.openai-compatible.yaml").read_text()

    customized = init._customize_openai_compatible_text(
        template,
        base_url="https://api.groq.com/openai/v1",
        quality_model="llama-quality",
        utility_model="llama-fast",
    )

    assert customized.count('model: "llama-quality"') == 3
    assert customized.count('model: "llama-fast"') == 3


def test_generic_template_validation_happens_before_active_config_overwrite(
    tmp_path, monkeypatch, capsys,
):
    config = tmp_path / "config"
    config.mkdir()
    models = config / "models.yaml"
    original = "roles:\n  author:\n    provider: anthropic\n"
    models.write_text(original)
    malformed_template = (
        "base_url: https://api.openai.com/v1\n"
        "roles:\n"
        "  author:\n"
        "    provider: openai-compatible\n"
        "    model: example\n"
        "phases:\n"
        "  author: {role: author}\n"
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RW_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RW_MODELS_CONFIG", raising=False)
    monkeypatch.delenv("RW_LLM_BASE_URL", raising=False)
    monkeypatch.setattr(init, "_effective_provider", lambda _models: None)
    monkeypatch.setattr(init, "_effective_openai_base_url", lambda: None)
    monkeypatch.setattr(init, "_ask_choice", lambda _n: 1)
    _answers(
        monkeypatch,
        "https://api.groq.com/openai/v1",
        "",
        "llama-quality",
        "llama-fast",
    )
    monkeypatch.setattr(
        init, "model_template_text", lambda *_args, **_kwargs: malformed_template
    )

    init._step_provider(tmp_path)

    assert models.read_text() == original
    assert "left unchanged" in capsys.readouterr().out


def test_generic_reconfigure_replaces_selected_profile_key(
    tmp_path, monkeypatch,
):
    config = tmp_path / "config"
    config.mkdir()
    _install_template(config, "models.openai-compatible.yaml")
    profile = tmp_path / ".env.provider"
    profile.write_text('export OPENAI_API_KEY="old-provider-secret"\n')

    _mark_profile_owned(
        monkeypatch, profile, OPENAI_API_KEY="old-provider-secret"
    )
    monkeypatch.delenv("RW_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RW_MODELS_CONFIG", raising=False)
    monkeypatch.delenv("RW_LLM_BASE_URL", raising=False)
    monkeypatch.setattr(init, "_effective_provider", lambda _models: None)
    monkeypatch.setattr(
        init, "_effective_openai_base_url", lambda: "https://api.openai.com/v1"
    )
    monkeypatch.setattr(init, "_ask_choice", lambda _n: 1)
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: False)
    _answers(
        monkeypatch,
        "https://api.groq.com/openai/v1",
        "new-provider-secret",
        "llama-quality",
        "llama-fast",
        "config/profiles/provider.yaml",
    )
    monkeypatch.setattr(init, "_report_readiness", lambda _provider: None)
    monkeypatch.setattr(init, "_warn_gitignore", lambda _root, _path: None)

    init._step_provider(tmp_path)

    assert profile.read_text() == (
        'export OPENAI_API_KEY="new-provider-secret"\n'
        'RW_MODELS_CONFIG="config/profiles/provider.yaml"\n'
    )
    assert not (config / "models.yaml").exists()
    models = (config / "profiles" / "provider.yaml").read_text()
    assert 'base_url: "https://api.groq.com/openai/v1"' in models
    assert models.count('model: "llama-quality"') == 3
    assert models.count('model: "llama-fast"') == 3


def test_named_openai_profile_never_removes_global_config(
    tmp_path, monkeypatch,
):
    config = tmp_path / "config"
    config.mkdir()
    _install_template(config, "models.openai.yaml")
    global_models = config / "models.yaml"
    original = "roles:\n  author:\n    provider: anthropic\n"
    global_models.write_text(original)
    profile = tmp_path / ".env.openai"
    profile.write_text("")
    _mark_profile_owned(monkeypatch, profile)

    for key in (
        "OPENAI_API_KEY", "RW_LLM_PROVIDER", "RW_MODELS_CONFIG",
        "RW_LLM_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(init, "_effective_provider", lambda _models: None)
    monkeypatch.setattr(init, "_effective_openai_base_url", lambda: None)
    monkeypatch.setattr(init, "_ask_choice", lambda _n: 0)
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: False)
    _answers(monkeypatch, "openai-secret")
    monkeypatch.setattr(init, "_report_readiness", lambda _provider: None)
    monkeypatch.setattr(init, "_warn_gitignore", lambda _root, _path: None)

    init._step_provider(tmp_path)

    assert global_models.read_text() == original
    assert 'RW_MODELS_CONFIG="models.openai.yaml"' in profile.read_text()
    assert 'OPENAI_API_KEY="openai-secret"' in profile.read_text()
    assert (config / "models.openai.yaml").exists()
    assert not (config / "profiles").exists()


def test_generic_reconfigure_migrates_tracked_template_to_profile_config(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config"
    config.mkdir()
    _install_template(config, "models.openai-compatible.yaml")
    selected = config / "models.profile.yaml"
    selected.write_text(
        "base_url: https://old.example/v1\n"
        "roles:\n  author: {provider: openai-compatible, model: old}\n"
    )
    profile = tmp_path / ".env.provider"
    profile.write_text(
        'RW_MODELS_CONFIG="models.profile.yaml"\n'
        'OPENAI_API_KEY="old-provider-secret"\n'
    )
    _mark_profile_owned(
        monkeypatch,
        profile,
        RW_MODELS_CONFIG="models.profile.yaml",
        OPENAI_API_KEY="old-provider-secret",
    )
    monkeypatch.delenv("RW_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RW_LLM_BASE_URL", raising=False)
    monkeypatch.setattr(init, "_effective_provider", lambda _models: None)
    monkeypatch.setattr(
        init, "_effective_openai_base_url", lambda: "https://old.example/v1"
    )
    monkeypatch.setattr(init, "_ask_choice", lambda _n: 1)
    decisions = iter((False, True))
    monkeypatch.setattr(
        init, "_confirm", lambda *args, **kwargs: next(decisions)
    )
    _answers(
        monkeypatch,
        "http://10.212.23.212/v1",
        "new-provider-secret",
        "llama-quality",
        "llama-fast",
        "config/profiles/provider.yaml",
    )
    monkeypatch.setattr(init, "_report_readiness", lambda _provider: None)
    monkeypatch.setattr(init, "_warn_gitignore", lambda _root, _path: None)

    init._step_provider(tmp_path)

    profile_text = profile.read_text()
    assert 'RW_MODELS_CONFIG="config/profiles/provider.yaml"' in profile_text
    assert 'OPENAI_API_KEY="new-provider-secret"' in profile_text
    assert os.environ["RW_MODELS_CONFIG"] == "config/profiles/provider.yaml"
    assert not (config / "models.yaml").exists()
    assert selected.read_text() == (
        "base_url: https://old.example/v1\n"
        "roles:\n  author: {provider: openai-compatible, model: old}\n"
    )
    profile_config = config / "profiles" / "provider.yaml"
    profile_config_text = profile_config.read_text()
    assert 'base_url: "http://10.212.23.212/v1"' in profile_config_text
    assert profile_config_text.count('model: "llama-quality"') == 3
    assert profile_config_text.count('model: "llama-fast"') == 3


def test_invalid_local_url_stops_before_config_write(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config"
    config.mkdir()
    models = config / "models.yaml"
    original = "roles:\n  author:\n    provider: anthropic\n"
    models.write_text(original)

    monkeypatch.setattr(init, "_effective_provider", lambda _models: None)
    monkeypatch.setattr(init, "_effective_openai_base_url", lambda: None)
    monkeypatch.setattr(init, "_ask_choice", lambda _n: 3)
    _answers(monkeypatch, "file:///tmp/not-an-http-server")

    init._step_provider(tmp_path)

    assert models.read_text() == original
    assert "setup cancelled" in capsys.readouterr().out


def test_local_reconfigure_removes_profile_key_before_readiness(
    tmp_path, monkeypatch,
):
    config = tmp_path / "config"
    config.mkdir()
    _install_template(config, "models.lmstudio.yaml")
    profile = tmp_path / ".env.local"
    profile.write_text('export OPENAI_API_KEY="cloud-secret"\n')

    _mark_profile_owned(monkeypatch, profile, OPENAI_API_KEY="cloud-secret")
    monkeypatch.delenv("RW_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RW_MODELS_CONFIG", raising=False)
    # Ensure monkeypatch owns an undo record for the value the wizard writes
    # directly through os.environ during this test.
    monkeypatch.setenv("RW_LLM_BASE_URL", "test-undo-sentinel")
    monkeypatch.delenv("RW_LLM_BASE_URL")
    monkeypatch.setattr(init, "_effective_provider", lambda _models: None)
    monkeypatch.setattr(
        init, "_effective_openai_base_url", lambda: "https://api.openai.com/v1"
    )
    monkeypatch.setattr(init, "_ask_choice", lambda _n: 3)
    _answers(monkeypatch, "http://localhost:1234/v1")
    decisions = iter((False, True))
    monkeypatch.setattr(
        init, "_confirm", lambda *args, **kwargs: next(decisions)
    )
    readiness_key = []
    monkeypatch.setattr(
        init,
        "_report_readiness",
        lambda _provider: readiness_key.append(os.environ.get("OPENAI_API_KEY")),
    )
    monkeypatch.setattr(init, "_warn_gitignore", lambda _root, _path: None)

    init._step_provider(tmp_path)

    assert readiness_key == [None]
    assert "OPENAI_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in profile.read_text()
    assert 'RW_LLM_BASE_URL="http://localhost:1234/v1"' in profile.read_text()
    assert 'RW_MODELS_CONFIG="models.lmstudio.yaml"' in profile.read_text()
    assert not (config / "models.yaml").exists()
    assert not (config / "profiles").exists()


def test_anthropic_reselect_replaces_key_and_removes_profile_endpoint(
    tmp_path, monkeypatch,
):
    config = tmp_path / "config"
    config.mkdir()
    _install_template(config, "models.anthropic.yaml")
    profile = tmp_path / ".env.provider"
    profile.write_text(
        'ANTHROPIC_BASE_URL="https://compat.example/anthropic"\n'
        'ANTHROPIC_API_KEY="custom-endpoint-secret"\n'
    )
    _mark_profile_owned(
        monkeypatch,
        profile,
        ANTHROPIC_BASE_URL="https://compat.example/anthropic",
        ANTHROPIC_API_KEY="custom-endpoint-secret",
    )
    for key in ("RW_MODELS_CONFIG", "RW_LLM_PROVIDER", "RW_LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(init, "_effective_provider", lambda _models: None)
    monkeypatch.setattr(init, "_effective_openai_base_url", lambda: None)
    monkeypatch.setattr(init, "_ask_choice", lambda _n: 2)
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: False)
    _answers(monkeypatch, "official-anthropic-secret")
    monkeypatch.setattr(init, "_report_readiness", lambda _provider: None)
    monkeypatch.setattr(init, "_warn_gitignore", lambda _root, _path: None)

    init._step_provider(tmp_path)

    text = profile.read_text()
    assert "ANTHROPIC_BASE_URL" not in text
    assert 'ANTHROPIC_API_KEY="official-anthropic-secret"' in text
    assert "ANTHROPIC_BASE_URL" not in os.environ
    assert os.environ["ANTHROPIC_API_KEY"] == "official-anthropic-secret"
    assert 'RW_MODELS_CONFIG="models.anthropic.yaml"' in text
    assert not (config / "models.yaml").exists()
    assert not (config / "profiles").exists()


# ── _valid_slug ──────────────────────────────────────────────────────────────

def test_valid_slugs():
    assert init._valid_slug("prime-editing")
    assert init._valid_slug("rna-biology")
    assert init._valid_slug("ai")
    assert init._valid_slug("evo-2")


def test_invalid_slugs():
    assert not init._valid_slug("Bad Slug")     # space + uppercase
    assert not init._valid_slug("Immunology")   # uppercase
    assert not init._valid_slug("-leading")     # leading hyphen
    assert not init._valid_slug("trailing-")    # trailing hyphen
    assert not init._valid_slug("under_score")  # underscore
    assert not init._valid_slug("")             # empty


def test_page_type_dirs_rejected_as_slugs():
    # Page-type dirs are structural, never content categories.
    for reserved in ("synthesis", "ideas", "references"):
        assert not init._valid_slug(reserved)


# ── _upsert_env ──────────────────────────────────────────────────────────────

def test_upsert_env_creates_file(tmp_path):
    env = tmp_path / ".env"
    init._upsert_env(env, {"ANTHROPIC_API_KEY": "sk-ant"})
    assert env.read_text() == 'ANTHROPIC_API_KEY="sk-ant"\n'


def test_upsert_env_replaces_existing_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text('ANTHROPIC_API_KEY="old"\n')
    init._upsert_env(env, {"ANTHROPIC_API_KEY": "new"})
    assert env.read_text() == 'ANTHROPIC_API_KEY="new"\n'


def test_upsert_env_replaces_exported_key_and_removes_old_duplicates(tmp_path):
    env = tmp_path / ".env.local"
    env.write_text(
        'export RW_LLM_BASE_URL="https://old.example/v1"\n'
        'RW_LLM_BASE_URL="https://older.example/v1"\n'
    )

    init._upsert_env(
        env, {"RW_LLM_BASE_URL": "http://localhost:1234/v1"}
    )

    assert env.read_text() == (
        'export RW_LLM_BASE_URL="http://localhost:1234/v1"\n'
    )
    assert init._dotenv_value(env, "RW_LLM_BASE_URL") == (
        "http://localhost:1234/v1"
    )


def test_upsert_env_replaces_bom_prefixed_first_key_without_duplicate(tmp_path):
    env = tmp_path / ".env.windows"
    env.write_bytes(b'\xef\xbb\xbfexport OPENAI_API_KEY="old"\r\nKEEP="yes"\r\n')

    init._upsert_env(env, {"OPENAI_API_KEY": "new"})

    text = env.read_text(encoding="utf-8")
    assert text.count("OPENAI_API_KEY=") == 1
    assert 'export OPENAI_API_KEY="new"' in text
    assert 'KEEP="yes"' in text
    assert env.stat().st_mode & 0o077 == 0


def test_upsert_env_preserves_comments_and_other_vars(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# my env\n"
        'ANTHROPIC_API_KEY="keep"\n'
        "\n"
        "# a comment\n"
    )
    init._upsert_env(env, {"OPENAI_API_KEY": "sk-x"})
    text = env.read_text()
    assert "# my env" in text
    assert '# a comment' in text
    assert 'ANTHROPIC_API_KEY="keep"' in text          # untouched
    assert 'OPENAI_API_KEY="sk-x"' in text             # appended


def test_upsert_env_is_idempotent(tmp_path):
    env = tmp_path / ".env"
    init._upsert_env(env, {"RW_LLM_BASE_URL": "http://x"})
    first = env.read_text()
    init._upsert_env(env, {"RW_LLM_BASE_URL": "http://x"})
    assert env.read_text() == first  # no duplicate line


def test_upsert_env_noop_on_empty(tmp_path):
    env = tmp_path / ".env"
    init._upsert_env(env, {})
    assert not env.exists()


def test_upsert_env_sets_restrictive_mode(tmp_path):
    env = tmp_path / ".env"
    init._upsert_env(env, {"ANTHROPIC_API_KEY": "sk-ant"})
    assert (env.stat().st_mode & 0o777) == 0o600


def test_secure_env_write_fails_before_replace_when_fchmod_fails(
    tmp_path, monkeypatch,
):
    env = tmp_path / ".env"
    env.write_text('OPENAI_API_KEY="old"\n')
    env.chmod(0o600)
    original = env.read_bytes()

    def fail_fchmod(_fd, _mode):
        raise OSError("permission denied")

    monkeypatch.setattr(env_profiles.os, "fchmod", fail_fchmod)
    with pytest.raises(EnvironmentFailure, match="cannot securely write"):
        init._upsert_env(env, {"OPENAI_API_KEY": "new"})

    assert env.read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_init_rewrites_existing_group_readable_profile_as_private(tmp_path):
    profile = tmp_path / ".env"
    profile.write_text('OPENAI_API_KEY="secret"\n')
    profile.chmod(0o644)

    init._upsert_env(profile, {"OPENAI_API_KEY": "updated"})

    assert profile.read_text() == 'OPENAI_API_KEY="updated"\n'
    assert profile.stat().st_mode & 0o777 == 0o600


def test_provider_transaction_rolls_back_config_before_restoring_old_key(
    tmp_path, monkeypatch,
):
    models = tmp_path / "models.yaml"
    models.write_text('base_url: "https://old.example/v1"\n')
    profile = tmp_path / ".env.provider"
    profile.write_text('OPENAI_API_KEY="old-secret"\n')
    _mark_profile_owned(monkeypatch, profile, OPENAI_API_KEY="old-secret")
    original_profile = profile.read_bytes()
    real_write = env_profiles.write_profile_atomic
    writes = 0

    def fail_final_profile_write(path, text):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise env_profiles.EnvProfileFailure("simulated final env failure")
        real_write(path, text)

    monkeypatch.setattr(
        env_profiles, "write_profile_atomic", fail_final_profile_write
    )

    def apply_new_config():
        models.write_text('base_url: "https://new.example/v1"\n')
        return True

    with pytest.raises(EnvironmentFailure, match="previous config and env profile"):
        env_profiles.commit_profile_and_config(
            config_path=models,
            apply_config=apply_new_config,
            env_path=profile,
            updates={"OPENAI_API_KEY": "new-secret"},
            removals=set(),
            remove_openai_key=False,
            replace_openai_key=True,
        )

    assert models.read_text() == 'base_url: "https://old.example/v1"\n'
    assert profile.read_bytes() == original_profile
    assert os.environ["OPENAI_API_KEY"] == "old-secret"
    assert writes == 2


def test_anthropic_endpoint_transaction_rolls_back_before_restoring_key(
    tmp_path, monkeypatch,
):
    models = tmp_path / "models.yaml"
    models.write_text('provider: old-compatible\n')
    profile = tmp_path / ".env.provider"
    profile.write_text(
        'ANTHROPIC_BASE_URL="https://compat.example/anthropic"\n'
        'ANTHROPIC_API_KEY="old-secret"\n'
    )
    _mark_profile_owned(
        monkeypatch,
        profile,
        ANTHROPIC_BASE_URL="https://compat.example/anthropic",
        ANTHROPIC_API_KEY="old-secret",
    )
    original_profile = profile.read_bytes()
    real_write = env_profiles.write_profile_atomic
    writes = 0

    def fail_final_profile_write(path, text):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise env_profiles.EnvProfileFailure("simulated final env failure")
        real_write(path, text)

    monkeypatch.setattr(
        env_profiles, "write_profile_atomic", fail_final_profile_write
    )

    def apply_new_config():
        models.write_text('provider: anthropic\n')
        return True

    with pytest.raises(EnvironmentFailure, match="previous config and env profile"):
        env_profiles.commit_profile_and_config(
            config_path=models,
            apply_config=apply_new_config,
            env_path=profile,
            updates={"ANTHROPIC_API_KEY": "new-secret"},
            removals={"ANTHROPIC_BASE_URL"},
            remove_openai_key=False,
            replace_openai_key=False,
            protected_credential_keys={"ANTHROPIC_API_KEY"},
        )

    assert models.read_text() == 'provider: old-compatible\n'
    assert profile.read_bytes() == original_profile
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://compat.example/anthropic"
    assert os.environ["ANTHROPIC_API_KEY"] == "old-secret"
    assert writes == 2


def test_anthropic_endpoint_transaction_reinstates_explicitly_reused_key(
    tmp_path, monkeypatch,
):
    models = tmp_path / "models.yaml"
    models.write_text('provider: old-compatible\n')
    profile = tmp_path / ".env.provider"
    profile.write_text(
        'ANTHROPIC_BASE_URL="https://compat.example/anthropic"\n'
        'ANTHROPIC_API_KEY="reused-secret"\n'
    )
    _mark_profile_owned(
        monkeypatch,
        profile,
        ANTHROPIC_BASE_URL="https://compat.example/anthropic",
        ANTHROPIC_API_KEY="reused-secret",
    )

    committed, _, _ = env_profiles.commit_profile_and_config(
        config_path=models,
        apply_config=lambda: models.write_text('provider: anthropic\n') is not None,
        env_path=profile,
        updates={},
        removals={"ANTHROPIC_BASE_URL"},
        remove_openai_key=False,
        replace_openai_key=False,
        protected_credential_keys={"ANTHROPIC_API_KEY"},
    )

    assert committed is True
    assert "ANTHROPIC_BASE_URL" not in profile.read_text()
    assert 'ANTHROPIC_API_KEY="reused-secret"' in profile.read_text()
    assert os.environ["ANTHROPIC_API_KEY"] == "reused-secret"


def test_provider_transaction_rolls_back_on_keyboard_interrupt(
    tmp_path, monkeypatch,
):
    models = tmp_path / "models.yaml"
    models.write_text('base_url: "https://old.example/v1"\n')
    profile = tmp_path / ".env.provider"
    profile.write_text('OPENAI_API_KEY="old-secret"\n')
    _mark_profile_owned(monkeypatch, profile, OPENAI_API_KEY="old-secret")
    original_profile = profile.read_bytes()

    def interrupt_after_config_write():
        models.write_text('base_url: "https://new.example/v1"\n')
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        env_profiles.commit_profile_and_config(
            config_path=models,
            apply_config=interrupt_after_config_write,
            env_path=profile,
            updates={"OPENAI_API_KEY": "new-secret"},
            removals=set(),
            remove_openai_key=False,
            replace_openai_key=True,
        )

    assert models.read_text() == 'base_url: "https://old.example/v1"\n'
    assert profile.read_bytes() == original_profile
    assert os.environ["OPENAI_API_KEY"] == "old-secret"


def test_failed_transaction_rollback_keeps_profile_credentialless(
    tmp_path, monkeypatch,
):
    models = tmp_path / "models.yaml"
    models.write_text('base_url: "https://old.example/v1"\n')
    profile = tmp_path / ".env.provider"
    profile.write_text('OPENAI_API_KEY="old-secret"\n')
    _mark_profile_owned(monkeypatch, profile, OPENAI_API_KEY="old-secret")
    real_write = env_profiles.write_profile_atomic
    writes = 0

    def fail_final_profile_write(path, text):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise env_profiles.EnvProfileFailure("simulated final env failure")
        real_write(path, text)

    monkeypatch.setattr(
        env_profiles, "write_profile_atomic", fail_final_profile_write
    )
    monkeypatch.setattr(
        env_profiles,
        "_restore_config",
        lambda *_args: (_ for _ in ()).throw(
            EnvironmentFailure("simulated config rollback failure")
        ),
    )

    def apply_new_config():
        models.write_text('base_url: "https://new.example/v1"\n')
        return True

    with pytest.raises(EnvironmentFailure, match="rollback was incomplete"):
        env_profiles.commit_profile_and_config(
            config_path=models,
            apply_config=apply_new_config,
            env_path=profile,
            updates={"OPENAI_API_KEY": "new-secret"},
            removals=set(),
            remove_openai_key=False,
            replace_openai_key=True,
        )

    assert models.read_text() == 'base_url: "https://new.example/v1"\n'
    assert "OPENAI_API_KEY" not in profile.read_text()
    assert "OPENAI_API_KEY" not in os.environ


def test_rerun_reports_named_effective_provider_not_builtin_openai(
    tmp_path, monkeypatch, capsys,
):
    config = tmp_path / "config"
    config.mkdir()
    (config / "models.compatible.yaml").write_text(
        "roles:\n  author:\n    provider: openai-compatible\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_MODELS_CONFIG", "models.compatible.yaml")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    monkeypatch.delenv("RW_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: False)

    init._step_provider(tmp_path)

    out = capsys.readouterr().out
    assert "openai-compatible" in out
    assert "RW_MODELS_CONFIG='models.compatible.yaml'" in out
    assert "built-in defaults" not in out


def test_keep_existing_provider_still_audits_credential_profile(
    tmp_path, monkeypatch,
):
    profile = tmp_path / ".env"
    profile.write_text('OPENAI_API_KEY="secret"\n')
    _mark_profile_owned(monkeypatch, profile, OPENAI_API_KEY="secret")
    monkeypatch.setattr(
        init, "_effective_provider", lambda _models: ("openai", "built-in defaults")
    )
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: False)
    audited = []
    monkeypatch.setattr(
        init, "_warn_gitignore", lambda root, path: audited.append((root, path))
    )

    init._step_provider(tmp_path)

    assert audited == [(tmp_path, profile)]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("RW_MODELS_CONFIG", ""),
        ("RW_LLM_PROVIDER", "chat-relay"),
        ("RW_LLM_BASE_URL", "https://old.example/v1"),
        ("ANTHROPIC_BASE_URL", "https://compat.example/anthropic"),
    ],
)
def test_parent_shell_routing_override_aborts_before_any_mutation(
    tmp_path, monkeypatch, capsys, key, value,
):
    config = tmp_path / "config"
    config.mkdir()
    models = config / "models.yaml"
    original = "roles:\n  author:\n    provider: anthropic\n"
    models.write_text(original)
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(init, "_effective_provider", lambda _models: None)
    monkeypatch.setattr(init, "_effective_openai_base_url", lambda: None)
    monkeypatch.setattr(
        init,
        "_ask_choice",
        lambda _n: pytest.fail("menu must not run with an external override"),
    )

    init._step_provider(tmp_path)

    assert models.read_text() == original
    assert not (tmp_path / ".env").exists()
    out = capsys.readouterr().out
    assert key in out
    assert "parent shell" in out


def test_warn_gitignore_checks_the_actual_selected_profile(tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(
        ["git", "init", "-q", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    (root / ".gitignore").write_text(".env.*\n")
    ignored = root / ".env.provider"
    ignored.write_text('OPENAI_API_KEY="secret"\n')
    unsafe = root / "secrets.prod"
    unsafe.write_text('OPENAI_API_KEY="secret"\n')

    init._warn_gitignore(root, ignored)
    assert capsys.readouterr().out == ""

    init._warn_gitignore(root, unsafe)
    warning = capsys.readouterr().out
    assert "secrets.prod" in warning
    assert "not confirmed gitignored" in warning

    subprocess.run(
        ["git", "-C", str(root), "add", "secrets.prod"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert init._warn_gitignore(root, unsafe) is False
    tracked_warning = capsys.readouterr().out
    assert "tracked by git" in tracked_warning
    assert "git rm --cached" in tracked_warning


# ── dashboard template invariant ─────────────────────────────────────────────

def test_scaffold_only_creates_dashboard_without_overwriting_it(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    assert init._scaffold(quiet=True) == 0
    views = tmp_path / "wiki" / "views.md"
    assert views.read_text(encoding="utf-8") == init.VIEWS_MD_TEMPLATE

    views.write_text("personal dashboard\n", encoding="utf-8")
    assert init._scaffold(quiet=True) == 0
    assert views.read_text(encoding="utf-8") == "personal dashboard\n"

def test_unignored_credential_profile_is_rejected_before_config_change(
    tmp_path, monkeypatch, capsys,
):
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    config = tmp_path / "config"
    config.mkdir()
    models = config / "models.yaml"
    original = "roles:\n  author:\n    provider: anthropic\n"
    models.write_text(original)
    profile = tmp_path / "secrets.prod"
    monkeypatch.setenv(init._ACTIVE_ENV_FILE_VAR, str(profile.resolve()))
    monkeypatch.setattr(init, "_effective_provider", lambda _models: None)
    monkeypatch.setattr(init, "_effective_openai_base_url", lambda: None)
    monkeypatch.setattr(init, "_ask_choice", lambda _n: 0)
    _answers(monkeypatch, "new-secret")
    monkeypatch.setattr(init, "_confirm", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        init,
        "_report_readiness",
        lambda _provider: pytest.fail("readiness must not run"),
    )

    init._step_provider(tmp_path)

    assert models.read_text() == original
    assert not profile.exists()
    out = capsys.readouterr().out
    assert "not confirmed gitignored" in out
    assert "before any config or credential change" in out


def test_readiness_uses_public_model_config_cache_reset(monkeypatch, capsys):
    from researchwiki.agents import llm, model_config

    cleared = []
    monkeypatch.setattr(model_config, "clear_caches", lambda: cleared.append(True))
    monkeypatch.setattr(llm, "missing_provider_credentials", lambda: [])

    init._report_readiness("openai")

    assert cleared == [True]
    assert "Provider configured" in capsys.readouterr().out


def test_views_template_contract():
    # A root-level page with `category:` trips lint's category-drift check.
    template = init.VIEWS_MD_TEMPLATE
    assert "category:" not in template
    assert "type: dashboard" in template
    # Every table requires the canonical provenance stamp; filesystem times
    # describe edits and back-link splices, never addition/generation dates.
    assert "file.mtime" not in template
    assert "file.ctime" not in template
    assert 'WHERE type = "paper" AND ingested_at' in template
    assert 'WHERE type = "synthesis" AND generated_at' in template
    assert 'WHERE type = "concept" AND generated_at' in template
    assert 'WHERE type = "idea" AND generated_at' in template
    assert template.index("## Recent papers") < template.index("## Recent ideas")
    assert template.index("## Recent ideas") < template.index("## Recent synthesis pages")
    assert template.index("## Recent synthesis pages") < template.index(
        "## Recent concept hubs"
    )
    ideas = template.split("## Recent ideas", 1)[1].split(
        "## Recent synthesis pages", 1
    )[0]
    assert "LIMIT 10" in ideas
    paper = template.split("## Recent papers", 1)[1].split("## Recent ideas", 1)[0]
    assert 'link(file.link, file.name) AS "Stem"' in paper
    assert "short_name" not in paper
    assert paper.index('AS "Category"') < paper.index('AS "Journal"')
    assert 'venue AS "Journal"' in paper
    # Synthesis stays JavaScript-free and has no duplicate member registry;
    # concepts own an explicit spoke registry and can count it directly.
    assert "```dataviewjs" not in template
    synthesis = template.split("## Recent synthesis pages", 1)[1].split(
        "## Recent concept hubs", 1
    )[0]
    assert "Members" not in synthesis
    assert "referenced_papers" not in synthesis
    concept = template.split("## Recent concept hubs", 1)[1].split(
        "## Recent ideas", 1
    )[0]
    assert 'length(referenced_papers) AS "Members"' in concept


def test_dashboard_prompt_matches_executable_member_contract():
    prompt = (ROOT / "prompts" / "init.md").read_text(encoding="utf-8")
    for signal in (
        "Keep this dashboard JavaScript-free",
        "No synthesis member-count column",
        "canonical spoke registry",
        'WHERE type = "concept" AND generated_at',
    ):
        assert signal in prompt


def test_wizard_steps_are_numbered_consecutively():
    """Folding the dashboard into scaffold removed step 3; the confirm header was
    left at 4, so the first-run flow printed 1, 2, 4."""
    source = Path(init.__file__).read_text(encoding="utf-8")
    numbers = [int(n) for n in re.findall(r'_header\("Step (\d+) —', source)]
    assert numbers == list(range(1, len(numbers) + 1)), numbers


def test_no_orphaned_wizard_steps():
    """A `_step_*` helper nobody calls is dead code ruff cannot see, and it takes
    its user-facing guidance out of the wizard with it."""
    source = Path(init.__file__).read_text(encoding="utf-8")
    for name in re.findall(r"^def (_step_\w+)", source, re.MULTILINE):
        assert source.count(name) > 1, f"{name} is defined but never called"


def _isolated_wiki(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    return tmp_path / "wiki" / "views.md"


def test_refresh_dashboard_adopts_template_and_backs_up_the_old_one(tmp_path, monkeypatch, capsys):
    """The upgrade path for a dashboard scaffolded before the concept-hub section:
    without it, `dashboard_contract_violations` has no resolving command."""
    views = _isolated_wiki(tmp_path, monkeypatch)
    views.write_text("## Recent papers (top 15)\nmy own notes\n", encoding="utf-8")

    assert init.main(["--refresh-dashboard"]) == 0

    assert views.read_text(encoding="utf-8") == init.VIEWS_MD_TEMPLATE
    backups = list((tmp_path / ".ingest").glob("views-*.md.bak"))
    assert len(backups) == 1
    assert "my own notes" in backups[0].read_text(encoding="utf-8")
    assert "Backed up your dashboard" in capsys.readouterr().out


def test_refresh_dashboard_is_idempotent_and_makes_no_second_backup(tmp_path, monkeypatch):
    views = _isolated_wiki(tmp_path, monkeypatch)
    views.write_text("stale\n", encoding="utf-8")

    assert init.main(["--refresh-dashboard"]) == 0
    assert init.main(["--refresh-dashboard"]) == 0

    assert len(list((tmp_path / ".ingest").glob("views-*.md.bak"))) == 1


def test_refresh_dashboard_creates_a_missing_dashboard_without_a_backup(tmp_path, monkeypatch):
    views = _isolated_wiki(tmp_path, monkeypatch)

    assert init.main(["--refresh-dashboard"]) == 0

    assert views.read_text(encoding="utf-8") == init.VIEWS_MD_TEMPLATE
    assert not (tmp_path / ".ingest").exists()


def test_refresh_dashboard_needs_no_tty(tmp_path, monkeypatch):
    """It has to work on an upgrade, where the interactive wizard refuses to run."""
    views = _isolated_wiki(tmp_path, monkeypatch)
    views.write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(init.sys.stdin, "isatty", lambda: False)

    assert init.main(["--refresh-dashboard"]) == 0
    assert views.read_text(encoding="utf-8") == init.VIEWS_MD_TEMPLATE


@pytest.mark.parametrize(
    "argv, expected_code",
    [
        (["--refresh-dashboard", "--definitely-invalid"], 2),
        (["--refresh-dashboard", "--scaffold-only"], 2),
        (["--refresh-dashboard", "--help"], 0),
    ],
)
def test_refresh_dashboard_help_and_bad_argv_never_mutate(
    tmp_path, monkeypatch, argv, expected_code,
):
    views = _isolated_wiki(tmp_path, monkeypatch)
    views.write_text("custom dashboard\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        init.main(argv)

    assert exc.value.code == expected_code
    assert views.read_text(encoding="utf-8") == "custom dashboard\n"
    assert not (tmp_path / ".ingest").exists()


def test_refresh_dashboard_backups_never_collide_within_one_second(
    tmp_path, monkeypatch,
):
    from researchwiki.tasks import init_dashboard

    views = _isolated_wiki(tmp_path, monkeypatch)

    class FixedDateTime(init_dashboard.dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 1, 12, 34, 56)

    monkeypatch.setattr(init_dashboard.dt, "datetime", FixedDateTime)
    views.write_text("first custom dashboard\n", encoding="utf-8")
    assert init.main(["--refresh-dashboard"]) == 0

    views.write_text("second custom dashboard\n", encoding="utf-8")
    assert init.main(["--refresh-dashboard"]) == 0

    backups = sorted((tmp_path / ".ingest").glob("views-*.md.bak"))
    assert len(backups) == 2
    assert {p.read_text(encoding="utf-8") for p in backups} == {
        "first custom dashboard\n", "second custom dashboard\n",
    }


def test_shipped_dashboard_template_satisfies_its_own_lint_contract(tmp_path):
    """The template and the checker must not be able to disagree — otherwise
    `--refresh-dashboard` writes a file that lint immediately flags."""
    from researchwiki.tasks.lint.dashboard_contract import (
        find_dashboard_contract_violations,
    )

    views = tmp_path / "views.md"
    views.write_text(init.VIEWS_MD_TEMPLATE, encoding="utf-8")
    assert find_dashboard_contract_violations(views) == []
