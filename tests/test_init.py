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
    ) == {"OPENAI_API_KEY": "sk-x", "RW_LLM_BASE_URL": "https://api.openai.com/v1"}


def test_env_updates_local_has_no_key():
    u = init._env_updates_for_provider("local", base_url="http://localhost:1234/v1")
    assert u == {"RW_LLM_BASE_URL": "http://localhost:1234/v1"}
    assert "OPENAI_API_KEY" not in u and "ANTHROPIC_API_KEY" not in u


def test_env_updates_chat_relay_is_fixed():
    assert init._env_updates_for_provider("chat-relay") == {"RW_LLM_PROVIDER": "chat-relay"}


# ── _ask_choice ──────────────────────────────────────────────────────────────

def _answers(monkeypatch, *replies):
    """Feed `_ask` a scripted sequence of answers."""
    seq = iter(replies)
    monkeypatch.setattr(init, "_ask", lambda prompt, default=None: next(seq))


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
    seen = []
    monkeypatch.setattr(init, "_bootstrap_categories", lambda: seen.append("bootstrap"))
    monkeypatch.setattr(init, "_manual_categories", lambda root: seen.append("manual"))
    _answers(monkeypatch, "y", "1")   # "y" is not a choice → re-prompt, then pick 1

    init._step_categories(tmp_path)
    assert seen == ["bootstrap"]
    assert "isn't a number" in capsys.readouterr().out


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
    init._write_models_config(cfg, models, "openai")
    assert not models.exists()
    assert "built-in defaults" in capsys.readouterr().out


def test_openai_choice_removes_a_stale_config(tmp_path, monkeypatch):
    """A leftover models.yaml would override the choice just made and the
    wizard would report success for a provider the user didn't pick."""
    cfg = _config_dir(tmp_path)
    models = cfg / "models.yaml"
    models.write_text("roles:\n  author:\n    provider: anthropic\n")
    monkeypatch.setattr(init, "_confirm", lambda prompt, default=True: True)
    init._write_models_config(cfg, models, "openai")
    assert not models.exists()


def test_openai_choice_warns_when_stale_config_kept(tmp_path, monkeypatch, capsys):
    cfg = _config_dir(tmp_path)
    models = cfg / "models.yaml"
    models.write_text("roles:\n  author:\n    provider: anthropic\n")
    monkeypatch.setattr(init, "_confirm", lambda prompt, default=True: False)
    init._write_models_config(cfg, models, "openai")
    assert models.exists()  # user's call, honored
    assert "overrides this choice" in capsys.readouterr().out


def test_non_openai_choice_copies_its_template(tmp_path):
    cfg = _config_dir(tmp_path)
    models = cfg / "models.yaml"
    init._write_models_config(cfg, models, "anthropic")
    assert "provider: anthropic" in models.read_text()


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
