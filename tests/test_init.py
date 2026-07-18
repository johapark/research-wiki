"""Unit tests for the `researchwiki init` wizard's pure helpers.

The interactive I/O isn't exercised here — these pin the load-bearing pure
functions: `.env` upsert (create / replace / preserve), slug validation,
provider→template/env mapping, and the invariant that the scaffolded dashboard
carries no `category:` (a root-level page with one trips lint's category-drift
check — see researchwiki/tasks/lint/yaml_checks.py:find_category_drift).
"""

from researchwiki.tasks import init


# ── _template_for_provider / _env_updates_for_provider ───────────────────────

def test_template_mapping():
    assert init._template_for_provider("anthropic") == "models.anthropic.yaml"
    assert init._template_for_provider("openai-compatible") == "models.openai-compatible.yaml"
    assert init._template_for_provider("local") == "models.lmstudio.yaml"
    # Chat-relay reuses the anthropic template (it's an env override).
    assert init._template_for_provider("chat-relay") == "models.anthropic.yaml"


def test_env_updates_anthropic():
    assert init._env_updates_for_provider("anthropic", api_key="sk-ant") == {
        "ANTHROPIC_API_KEY": "sk-ant"
    }
    # No key supplied → nothing to write (user sets it later).
    assert init._env_updates_for_provider("anthropic") == {}


def test_env_updates_openai_compatible():
    assert init._env_updates_for_provider(
        "openai-compatible", api_key="sk-x", base_url="https://api.openai.com/v1"
    ) == {"OPENAI_API_KEY": "sk-x", "RW_LLM_BASE_URL": "https://api.openai.com/v1"}


def test_env_updates_local_has_no_key():
    u = init._env_updates_for_provider("local", base_url="http://localhost:1234/v1")
    assert u == {"RW_LLM_BASE_URL": "http://localhost:1234/v1"}
    assert "OPENAI_API_KEY" not in u and "ANTHROPIC_API_KEY" not in u


def test_env_updates_chat_relay_is_fixed():
    assert init._env_updates_for_provider("chat-relay") == {"RW_LLM_PROVIDER": "chat-relay"}


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


# ── dashboard template invariant ─────────────────────────────────────────────

def test_views_template_has_no_category():
    # A root-level page with `category:` trips lint's category-drift check.
    assert "category:" not in init.VIEWS_MD_TEMPLATE
    assert "type: dashboard" in init.VIEWS_MD_TEMPLATE
    # The three cuts the plan promises.
    assert 'WHERE type = "paper"' in init.VIEWS_MD_TEMPLATE
    assert 'WHERE type = "synthesis"' in init.VIEWS_MD_TEMPLATE
    assert 'WHERE type = "idea"' in init.VIEWS_MD_TEMPLATE
