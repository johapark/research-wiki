"""RW_MODELS_CONFIG selects which models-config file drives LLM routing.

Lets you switch backends without copying files over your active config
(e.g. `RW_MODELS_CONFIG=models.glm.yaml researchwiki agent ingest …`).
"""

from pathlib import Path

import pytest

from researchwiki.agents import model_config as mc
from researchwiki.errors import EnvironmentFailure


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    for name in (
        "RW_MODELS_CONFIG", "RW_LLM_BASE_URL", "RW_LLM_PROVIDER",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    mc.clear_caches()
    yield
    mc.clear_caches()


def test_default_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert mc.config_path() == tmp_path / "config" / "models.yaml"


def test_bare_name_resolves_under_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_MODELS_CONFIG", "models.glm.yaml")
    assert mc.config_path() == tmp_path / "config" / "models.glm.yaml"


def test_absolute_path_used_verbatim(monkeypatch):
    monkeypatch.setenv("RW_MODELS_CONFIG", "/etc/rw/custom.yaml")
    assert mc.config_path() == Path("/etc/rw/custom.yaml")


def test_relative_path_with_separator_used_verbatim(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_MODELS_CONFIG", "sub/dir/models.yaml")
    assert mc.config_path() == Path("sub/dir/models.yaml")


def test_override_actually_loads_the_named_file(monkeypatch, tmp_path):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "models.custom.yaml").write_text(
        "base_url: https://provider.invalid/v1\n"
        "roles:\n  author:\n    provider: anthropic\n    model: my-test-model\n"
        "    temperature: 0.4\n    max_tokens: 100\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_MODELS_CONFIG", "models.custom.yaml")
    mc.clear_caches()
    assert mc.for_phase("author").model == "my-test-model"


def test_missing_override_file_fails_closed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_MODELS_CONFIG", "does-not-exist.yaml")
    mc.clear_caches()
    with pytest.raises(mc.ModelConfigUnavailable, match="does not exist"):
        mc.for_phase("author")
    assert issubclass(mc.ModelConfigUnavailable, EnvironmentFailure)


def test_malformed_override_file_fails_closed(monkeypatch, tmp_path):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "models.broken.yaml").write_text("roles: [unterminated\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_MODELS_CONFIG", "models.broken.yaml")
    mc.clear_caches()
    with pytest.raises(mc.ModelConfigUnavailable, match="cannot be parsed"):
        mc.for_phase("author")


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_empty_override_is_an_explicit_error(monkeypatch, value):
    monkeypatch.setenv("RW_MODELS_CONFIG", value)
    with pytest.raises(mc.ModelConfigUnavailable, match="is empty"):
        mc.config_path()


@pytest.mark.parametrize(("body", "message"), [
    (
        "as_of: 2026-08-20\nmodels:\n  gpt: {input: 1, output: 2}\n",
        "unknown top-level",
    ),
    ("roles: [author]\n", "roles must be a mapping"),
    ("base_url: provider.invalid/v1\n", "safe absolute endpoint"),
    (
        "roles:\n  author:\n    provider: anthropic\n",
        "missing required field 'model'",
    ),
    (
        "roles:\n  author:\n    model: m\n    max_tokens: many\n",
        "max_tokens must be a positive integer",
    ),
    (
        "phases:\n  custom:\n    role: not-a-role\n",
        "references unknown role",
    ),
    (
        "phases:\n  custom:\n    max_tokens: 100\n",
        "needs a non-empty role",
    ),
])
def test_explicit_schema_errors_fail_closed(monkeypatch, tmp_path, body, message):
    config = tmp_path / "models.invalid.yaml"
    config.write_text(body, encoding="utf-8")
    monkeypatch.setenv("RW_MODELS_CONFIG", str(config))
    with pytest.raises(mc.ModelConfigUnavailable, match=message):
        mc.validate_config()


def test_clear_caches_resets_all_routing_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mc._config()
    mc._ingest_settings()
    mc.base_url()
    mc._env_override_warned = True
    mc._env_model_mismatch_warned = True
    mc._missing_base_url_warned = True

    mc.clear_caches()

    assert mc._load_routing_snapshot.cache_info().currsize == 0
    assert mc._env_override_warned is False
    assert mc._env_model_mismatch_warned is False
    assert mc._missing_base_url_warned is False


@pytest.mark.parametrize(("body", "message"), [
    ("roles: [unterminated\n", "cannot be parsed"),
    ("- roles\n- phases\n", "must contain a YAML mapping"),
    ("as_of: 2026-08-20\nmodels: {}\n", "unknown top-level"),
    ("roles: [author]\n", "roles must be a mapping"),
    ("roles:\n  author: {provider: anthropic}\n", "missing required field"),
])
def test_present_invalid_implicit_config_fails_closed(
    monkeypatch, tmp_path, body, message,
):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "models.yaml").write_text(body, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(mc.ModelConfigUnavailable, match=message):
        mc.validate_config()


def test_unreadable_implicit_config_fails_closed(monkeypatch, tmp_path):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    config = cfgdir / "models.yaml"
    config.write_text("base_url: https://provider.invalid/v1\n", encoding="utf-8")
    original = Path.read_text

    def denied(path, *args, **kwargs):
        if path == config:
            raise PermissionError("denied for test")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(mc.ModelConfigUnavailable, match="cannot be read"):
        mc.base_url()


def test_broken_implicit_config_symlink_fails_closed(monkeypatch, tmp_path):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "models.yaml").symlink_to(tmp_path / "missing-target.yaml")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(mc.ModelConfigUnavailable, match="broken symbolic link"):
        mc.validate_config()


@pytest.mark.parametrize("url", [
    "http://provider.invalid/v1",
    "https://user:secret@provider.invalid/v1",
    "https://provider.invalid/v1?token=secret",
    "https://provider.invalid/v1#fragment",
    "https://provider.invalid:0/v1",
    "https://provider.invalid:/v1",
])
def test_unsafe_base_url_in_implicit_config_fails_closed(monkeypatch, tmp_path, url):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "models.yaml").write_text(
        f"base_url: {url}\nroles:\n  author: {{provider: openai-compatible, model: m}}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(mc.ModelConfigUnavailable, match="safe absolute endpoint"):
        mc.validate_config()


@pytest.mark.parametrize("url", [
    "http://localhost:1234/v1",
    "http://127.0.0.1:8000/v1",
    "http://[::1]:11434/v1",
])
def test_loopback_http_base_url_remains_valid(monkeypatch, tmp_path, url):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "models.yaml").write_text(
        f"base_url: {url}\nroles:\n  author: {{provider: openai-compatible, model: m}}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert mc.base_url() == url


@pytest.mark.parametrize("url", [
    "http://remote.invalid/v1",
    "https://user:secret@remote.invalid/v1",
    "https://remote.invalid/v1?token=secret",
    "https://remote.invalid/v1#fragment",
    " https://remote.invalid/v1",
    "https://remote invalid/v1",
    "https://remote.invalid/\u200bv1",
    "https://remote.invalid/v1\n",
])
def test_unsafe_env_base_url_fails_closed_without_echoing_value(
    monkeypatch, tmp_path, url,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_LLM_BASE_URL", url)
    with pytest.raises(mc.ModelConfigUnavailable, match="RW_LLM_BASE_URL") as exc:
        mc.validate_config()
    assert url not in str(exc.value)


@pytest.mark.parametrize("url", [
    "https://remote.invalid/v1",
    "http://localhost:1234/v1",
    "http://127.0.0.1:8000/v1",
])
def test_safe_env_base_url_remains_valid(monkeypatch, tmp_path, url):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_LLM_BASE_URL", url)
    mc.validate_config()


@pytest.mark.parametrize("url", [
    "http://remote.invalid/anthropic",
    "https://user:secret@remote.invalid/anthropic",
    "https://remote.invalid/anthropic?token=secret",
    "https://remote.invalid/anthropic\u200b",
])
def test_unsafe_anthropic_env_endpoint_fails_closed_without_echoing_value(
    monkeypatch, tmp_path, url,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
    with pytest.raises(mc.ModelConfigUnavailable, match="ANTHROPIC_BASE_URL") as exc:
        mc.validate_config()
    assert url not in str(exc.value)


def test_shipped_glm_anthropic_endpoint_remains_valid(monkeypatch):
    config = Path(__file__).resolve().parent.parent / "config" / "models.glm.yaml"
    monkeypatch.setenv("RW_MODELS_CONFIG", str(config))
    monkeypatch.setenv(
        "ANTHROPIC_BASE_URL", "https://api.z.ai/api/anthropic",
    )
    mc.validate_config()


def test_all_readers_share_one_document_snapshot(monkeypatch, tmp_path):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    config = cfgdir / "models.yaml"
    config.write_text(
        "base_url: https://first.invalid/v1\n"
        "ingest: {n_drafts: 1}\n"
        "roles:\n  author: {provider: openai-compatible, model: first-model}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert mc.for_role("author").model == "first-model"

    config.write_text(
        "base_url: https://second.invalid/v1\n"
        "ingest: {n_drafts: 2}\n"
        "roles:\n  author: {provider: openai-compatible, model: second-model}\n",
        encoding="utf-8",
    )
    assert mc.default_n_drafts() == 1
    assert mc.base_url() == "https://first.invalid/v1"

    mc.clear_caches()
    assert mc.for_role("author").model == "second-model"
    assert mc.default_n_drafts() == 2
    assert mc.base_url() == "https://second.invalid/v1"


@pytest.mark.parametrize(
    "config",
    sorted((Path(__file__).resolve().parent.parent / "config").glob("models*.yaml")),
    ids=lambda path: path.name,
)
def test_every_shipped_model_config_passes_strict_validation(monkeypatch, config):
    monkeypatch.setenv("RW_MODELS_CONFIG", str(config))
    mc.validate_config()
