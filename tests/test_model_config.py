"""Model-config loader, role/phase resolution, and the reasoning_effort plumb.

Covers `researchwiki/agents/model_config.py`:
  - Fallback path when YAML missing or malformed.
  - YAML loading of role + phase entries (incl. reasoning_effort).
  - Per-phase override semantics, including explicit `null` to clear an
    inherited reasoning_effort (the short_name-on-thinking-role case).
  - `RW_LLM_PROVIDER` env override applied last.
  - Failure modes: unknown phase, phase pointing at unknown role, role
    missing `model:`, role with malformed numeric value.

Hermetic — no network, no LLM calls. Each test patches `wiki_root` to a
tmp_path so the live `config/models.yaml` doesn't bleed in, and clears
the `_config` lru_cache around every test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.agents import model_config


# ---------- fixtures ----------

@pytest.fixture(autouse=True)
def reset_cache_and_env(monkeypatch):
    """Clear lru_cache, env-var overrides, and one-shot warning flag."""
    model_config._config.cache_clear()
    model_config._ingest_settings.cache_clear()
    model_config.base_url.cache_clear()
    model_config._env_override_warned = False
    monkeypatch.delenv("RW_LLM_PROVIDER", raising=False)
    yield
    model_config._config.cache_clear()
    model_config._ingest_settings.cache_clear()
    model_config.base_url.cache_clear()
    model_config._env_override_warned = False


def _write_yaml(tmp_path: Path, contents: str) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.yaml").write_text(contents)
    return tmp_path


def _use_yaml(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(model_config, "wiki_root", lambda: root)
    model_config._config.cache_clear()
    model_config._ingest_settings.cache_clear()
    model_config.base_url.cache_clear()


# ---------- fallback (no YAML) ----------

def test_fallback_when_yaml_missing(tmp_path, monkeypatch):
    """Missing config/models.yaml → hardcoded fallback config used
    (the chatgpt default: openai-compatible / gpt-5.6-luna)."""
    monkeypatch.setattr(model_config, "wiki_root", lambda: tmp_path)
    cfg = model_config.for_phase("author")
    assert cfg.provider == "openai-compatible"
    assert cfg.model == "gpt-5.6-luna"
    assert cfg.reasoning_effort is None


def test_fallback_roles_have_reasoning_effort_none():
    """Every hardcoded fallback role defaults reasoning_effort to None.

    Guards against a future fallback edit that hard-codes a value — the
    default chatgpt config leaves reasoning_effort unset, so the fallback
    mirror must too.
    """
    for name, cfg in model_config._FALLBACK_ROLES.items():
        assert cfg.reasoning_effort is None, f"role {name}: {cfg.reasoning_effort!r}"


def test_fallback_covers_every_phase_in_anthropic_template(tmp_path, monkeypatch):
    """Every phase declared in `config/models.anthropic.yaml` must resolve
    under the hardcoded fallback.

    The fallback is the safety net for fresh checkouts where the user
    hasn't created `config/models.yaml` yet. When a new phase is added to
    the Anthropic template (e.g. `reconcile`/`extractor`), this test
    forces the matching `_FALLBACK_PHASES` / `_FALLBACK_ROLES` update —
    otherwise that phase silently degrades on every fresh install.
    """
    pytest.importorskip("yaml")
    import yaml

    template = (
        Path(__file__).resolve().parent.parent
        / "config" / "models.anthropic.yaml"
    )
    declared_phases = list((yaml.safe_load(template.read_text()) or {})
                           .get("phases", {}).keys())
    assert declared_phases, "no phases parsed from models.anthropic.yaml"

    monkeypatch.setattr(model_config, "wiki_root", lambda: tmp_path)
    for phase in declared_phases:
        # Resolves end-to-end: phase → role → ModelConfig. Raises KeyError
        # if either the phase or its target role is missing from fallback.
        model_config.for_phase(phase)


# ---------- YAML parsing of reasoning_effort ----------

def test_yaml_role_with_reasoning_effort(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: openai-compatible
    model: gemini-2.5-flash
    temperature: 0.5
    max_tokens: 2500
    reasoning_effort: low
phases:
  author: {role: author}
""")
    _use_yaml(monkeypatch, root)
    cfg = model_config.for_phase("author")
    assert cfg.reasoning_effort == "low"
    assert cfg.model == "gemini-2.5-flash"


def test_yaml_role_without_reasoning_effort_defaults_to_none(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.5
    max_tokens: 2500
phases:
  author: {role: author}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.for_phase("author").reasoning_effort is None


def test_yaml_role_with_explicit_null_reasoning_effort(tmp_path, monkeypatch):
    """`reasoning_effort: null` (YAML) parses to None — same as omitting it."""
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: openai-compatible
    model: gemini-2.5-flash
    temperature: 0.5
    max_tokens: 2500
    reasoning_effort: null
phases:
  author: {role: author}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.for_phase("author").reasoning_effort is None


# ---------- per-phase resolution + override ----------

def test_phase_inherits_role_reasoning_effort(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: openai-compatible
    model: gemini-2.5-flash
    temperature: 0.5
    max_tokens: 2500
    reasoning_effort: low
phases:
  evolve: {role: author}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.for_phase("evolve").reasoning_effort == "low"


def test_phase_overrides_reasoning_effort(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: openai-compatible
    model: gemini-2.5-flash
    temperature: 0.5
    max_tokens: 2500
    reasoning_effort: low
phases:
  evolve: {role: author, reasoning_effort: high}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.for_phase("evolve").reasoning_effort == "high"


def test_phase_clears_inherited_reasoning_effort_with_null(tmp_path, monkeypatch):
    """A phase may set `reasoning_effort: null` to disable thinking on a
    role that otherwise enables it.

    Concrete case: short_name binds to `proposer`, which on a Gemini Flash
    config has reasoning_effort=low for general-purpose proposing — but
    short_name has a 32-token budget and CAN'T afford any thinking. The
    phase entry must be able to override the inherited value back to None.
    """
    root = _write_yaml(tmp_path, """
roles:
  proposer:
    provider: openai-compatible
    model: gemini-2.5-flash
    temperature: 0.3
    max_tokens: 200
    reasoning_effort: low
phases:
  short_name: {role: proposer, reasoning_effort: null, max_tokens: 32}
""")
    _use_yaml(monkeypatch, root)
    cfg = model_config.for_phase("short_name")
    assert cfg.reasoning_effort is None
    assert cfg.max_tokens == 32
    # Role-level reasoning_effort still set; phase only cleared its own copy.
    assert model_config.for_role("proposer").reasoning_effort == "low"


def test_phase_temperature_and_max_tokens_overrides_still_work(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.5
    max_tokens: 2500
phases:
  debug: {role: author, temperature: 0.2, max_tokens: 1000}
""")
    _use_yaml(monkeypatch, root)
    cfg = model_config.for_phase("debug")
    assert cfg.temperature == 0.2
    assert cfg.max_tokens == 1000
    # Untouched fields still come from the role.
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-4-6"


# ---------- rpm (rate-limit cap) ----------

def test_yaml_role_with_rpm(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: openai-compatible
    model: gemini-3.1-flash-lite
    temperature: 0.5
    max_tokens: 2500
    rpm: 15
phases:
  author: {role: author}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.for_phase("author").rpm == 15


def test_yaml_role_without_rpm_defaults_to_none(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.5
    max_tokens: 2500
phases:
  author: {role: author}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.for_phase("author").rpm is None


def test_fallback_roles_have_rpm_none():
    """Hardcoded fallback roles are unthrottled (rpm=None)."""
    for name in model_config.list_roles():
        assert model_config.for_role(name).rpm is None, name


def test_phase_inherits_and_overrides_rpm(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: openai-compatible
    model: gemini-3.5-flash
    temperature: 0.5
    max_tokens: 2500
    rpm: 5
phases:
  evolve: {role: author}
  debug: {role: author, rpm: 2}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.for_phase("evolve").rpm == 5   # inherited
    assert model_config.for_phase("debug").rpm == 2    # overridden


# ---------- ingest.n_drafts (pipeline default) ----------

def test_ingest_n_drafts_from_yaml(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
ingest:
  n_drafts: 1
roles:
  author: {provider: openai-compatible, model: m, temperature: 0.5, max_tokens: 100}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.default_n_drafts() == 1


def test_ingest_n_drafts_absent_is_none(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author: {provider: openai-compatible, model: m, temperature: 0.5, max_tokens: 100}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.default_n_drafts() is None


def test_ingest_n_drafts_missing_file_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(model_config, "wiki_root", lambda: tmp_path)
    model_config._ingest_settings.cache_clear()
    assert model_config.default_n_drafts() is None


def test_ingest_n_drafts_below_one_treated_as_unset(tmp_path, monkeypatch):
    """A tournament needs >= 1 draft; 0 / negative fall back to None."""
    root = _write_yaml(tmp_path, """
ingest:
  n_drafts: 0
roles:
  author: {provider: openai-compatible, model: m, temperature: 0.5, max_tokens: 100}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.default_n_drafts() is None


def test_ingest_n_drafts_non_int_is_none(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
ingest:
  n_drafts: "lots"
roles:
  author: {provider: openai-compatible, model: m, temperature: 0.5, max_tokens: 100}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.default_n_drafts() is None


# ---------- ingest.target_claims_max_chars ----------

def test_target_claims_max_chars_from_yaml(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
ingest:
  target_claims_max_chars: 40000
roles:
  author: {provider: openai-compatible, model: m, temperature: 0.5, max_tokens: 100}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.target_claims_max_chars() == 40000


def test_target_claims_max_chars_defaults_generous(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author: {provider: openai-compatible, model: m, temperature: 0.5, max_tokens: 100}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.target_claims_max_chars() == model_config._DEFAULT_TARGET_CLAIMS_MAX_CHARS


def test_target_claims_max_chars_invalid_falls_back(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
ingest:
  target_claims_max_chars: 0
roles:
  author: {provider: openai-compatible, model: m, temperature: 0.5, max_tokens: 100}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.target_claims_max_chars() == model_config._DEFAULT_TARGET_CLAIMS_MAX_CHARS


# ---------- base_url (folded-in endpoint) ----------

def test_base_url_from_yaml(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
base_url: https://api.upstage.ai/v1
roles:
  author: {provider: openai-compatible, model: m, temperature: 0.5, max_tokens: 100}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.base_url() == "https://api.upstage.ai/v1"


def test_base_url_absent_is_none(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author: {provider: openai-compatible, model: m, temperature: 0.5, max_tokens: 100}
""")
    _use_yaml(monkeypatch, root)
    assert model_config.base_url() is None


def test_base_url_missing_file_is_fallback(tmp_path, monkeypatch):
    """No config file → the OpenAI endpoint paired with the openai-compatible
    fallback roles, so a fresh checkout reaches OpenAI (not localhost)."""
    monkeypatch.setattr(model_config, "wiki_root", lambda: tmp_path)
    model_config.base_url.cache_clear()
    assert model_config.base_url() == model_config._FALLBACK_BASE_URL


# ---------- env-var override ----------

def test_rw_llm_provider_env_var_wins_over_yaml(tmp_path, monkeypatch):
    """RW_LLM_PROVIDER replaces the resolved provider on every phase but
    leaves model / temperature / max_tokens / reasoning_effort intact."""
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.5
    max_tokens: 2500
    reasoning_effort: low
phases:
  author: {role: author}
""")
    _use_yaml(monkeypatch, root)
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    cfg = model_config.for_phase("author")
    assert cfg.provider == "chat-relay"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.reasoning_effort == "low"


# ---------- env-override-defeats-mixing warning ----------
#
# Documented scar (CLAUDE.md § Model providers): `RW_LLM_PROVIDER` in `.env`
# silently defeats per-role provider mixing. These tests pin the four cases
# of the truth table: env-set × mixing-declared, with the banner firing only
# on the (set, mixed) corner.

_MIXED_YAML = """
roles:
  author:
    provider: anthropic
    model: claude-sonnet-4-6
    temperature: 0.5
    max_tokens: 2500
  extractor:
    provider: openai-compatible
    model: gemini-2.5-flash
    temperature: 0.0
    max_tokens: 800
phases:
  author: {role: author}
  extractor: {role: extractor}
"""

# Every role declared with one provider — genuinely uniform after the
# fallback merge (the fallback is openai-compatible, so a partial YAML would
# read as *mixed*; this fixture pins the truly-uniform corner).
_UNIFORM_YAML = """
roles:
  author:     {provider: anthropic, model: claude-sonnet-4-6, temperature: 0.5, max_tokens: 2500}
  critic:     {provider: anthropic, model: claude-sonnet-4-6, temperature: 0.3, max_tokens: 2500}
  judge:      {provider: anthropic, model: claude-sonnet-4-6, temperature: 0.2, max_tokens: 1500}
  classifier: {provider: anthropic, model: claude-sonnet-4-6, temperature: 0.1, max_tokens: 200}
  proposer:   {provider: anthropic, model: claude-sonnet-4-6, temperature: 0.3, max_tokens: 200}
  extractor:  {provider: anthropic, model: claude-haiku-4-5-20251001, temperature: 0.0, max_tokens: 800}
phases:
  author: {role: author}
"""


def test_env_override_with_mixing_fires_stderr_banner(tmp_path, monkeypatch, capsys):
    """Env override + non-uniform per-role providers → banner fires."""
    _use_yaml(monkeypatch, _write_yaml(tmp_path, _MIXED_YAML))
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    model_config.for_phase("author")
    err = capsys.readouterr().err
    assert "RW_LLM_PROVIDER" in err
    assert "chat-relay" in err
    assert "overrides per-role providers" in err


def test_env_override_without_mixing_does_not_fire(tmp_path, monkeypatch, capsys):
    """Env override + uniform config → no banner (nothing hidden)."""
    _use_yaml(monkeypatch, _write_yaml(tmp_path, _UNIFORM_YAML))
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    model_config.for_phase("author")
    assert "RW_LLM_PROVIDER" not in capsys.readouterr().err


def test_no_env_override_with_mixing_does_not_fire(tmp_path, monkeypatch, capsys):
    """Mixed config but no env override → mixing works, no banner."""
    _use_yaml(monkeypatch, _write_yaml(tmp_path, _MIXED_YAML))
    model_config.for_phase("author")
    assert "RW_LLM_PROVIDER" not in capsys.readouterr().err


def test_env_override_banner_fires_only_once(tmp_path, monkeypatch, capsys):
    """Repeated `for_phase` calls in one process → banner prints exactly once.

    Guards against a chatty banner in long-running jobs (agent ingest calls
    for_phase many times per PDF).
    """
    _use_yaml(monkeypatch, _write_yaml(tmp_path, _MIXED_YAML))
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    for _ in range(5):
        model_config.for_phase("author")
    err = capsys.readouterr().err
    # "overrides per-role providers" appears exactly once per banner.
    assert err.count("overrides per-role providers") == 1


# ---------- failure modes ----------

def test_unknown_phase_raises_keyerror(tmp_path, monkeypatch):
    monkeypatch.setattr(model_config, "wiki_root", lambda: tmp_path)
    with pytest.raises(KeyError):
        model_config.for_phase("does_not_exist")


def test_phase_pointing_at_unknown_role_raises_keyerror(tmp_path, monkeypatch):
    """A phase whose role isn't in YAML *or* fallbacks raises clearly."""
    root = _write_yaml(tmp_path, """
roles: {}
phases:
  newphase: {role: not_a_real_role}
""")
    _use_yaml(monkeypatch, root)
    with pytest.raises(KeyError):
        model_config.for_phase("newphase")


def test_unknown_role_via_for_role_raises_keyerror(tmp_path, monkeypatch):
    monkeypatch.setattr(model_config, "wiki_root", lambda: tmp_path)
    with pytest.raises(KeyError):
        model_config.for_role("not_a_real_role")


def test_role_missing_model_field_silently_skipped(tmp_path, monkeypatch):
    """Role without `model:` is dropped silently; fallback fills the gap."""
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: openai-compatible
    temperature: 0.5
phases:
  author: {role: author}
""")
    _use_yaml(monkeypatch, root)
    cfg = model_config.for_phase("author")
    # YAML's broken `author` was skipped; fallback's `author` won.
    assert cfg.provider == "openai-compatible"
    assert cfg.model == "gpt-5.6-luna"


def test_role_with_bad_numeric_value_skipped_with_stderr(tmp_path, monkeypatch, capsys):
    """A non-numeric temperature triggers a stderr log and skips the role.

    Loader is intentionally lenient: a typo in one role shouldn't take
    down the whole config — fallbacks fill in.
    """
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: openai-compatible
    model: gemini-2.5-flash
    temperature: "not-a-number"
phases:
  author: {role: author}
""")
    _use_yaml(monkeypatch, root)
    cfg = model_config.for_phase("author")
    assert cfg.provider == "openai-compatible"  # fallback won
    assert "bad role" in capsys.readouterr().err


def test_malformed_yaml_falls_back_silently(tmp_path, monkeypatch, capsys):
    """An unparseable YAML body logs to stderr and falls back."""
    root = _write_yaml(tmp_path, "::: not yaml :::\n  - [unbalanced")
    _use_yaml(monkeypatch, root)
    cfg = model_config.for_phase("author")
    assert cfg.provider == "openai-compatible"
    assert "could not parse" in capsys.readouterr().err


# ---------- introspection ----------

def test_list_roles_and_phases_include_yaml_additions(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  custom_role:
    provider: openai-compatible
    model: my-model
    temperature: 0.0
    max_tokens: 100
phases:
  custom_phase: {role: custom_role}
""")
    _use_yaml(monkeypatch, root)
    roles = model_config.list_roles()
    phases = model_config.list_phases()
    assert "custom_role" in roles
    assert "custom_phase" in phases
    # Fallback entries remain visible — YAML is additive, not replacing.
    assert "author" in roles
    assert "author" in phases


def test_for_role_returns_yaml_config_with_reasoning_effort(tmp_path, monkeypatch):
    root = _write_yaml(tmp_path, """
roles:
  author:
    provider: openai-compatible
    model: gemini-2.5-flash
    temperature: 0.4
    max_tokens: 3000
    reasoning_effort: medium
""")
    _use_yaml(monkeypatch, root)
    cfg = model_config.for_role("author")
    assert cfg.provider == "openai-compatible"
    assert cfg.model == "gemini-2.5-flash"
    assert cfg.temperature == 0.4
    assert cfg.max_tokens == 3000
    assert cfg.reasoning_effort == "medium"
