"""RW_MODELS_CONFIG selects which models-config file drives LLM routing.

Lets you switch backends without copying files over your active config
(e.g. `RW_MODELS_CONFIG=models.glm.yaml researchwiki agent ingest …`).
"""

import os
from pathlib import Path

import pytest

from researchwiki.agents import model_config as mc


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    monkeypatch.delenv("RW_MODELS_CONFIG", raising=False)
    mc._config.cache_clear()
    yield
    mc._config.cache_clear()


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
        "roles:\n  author:\n    provider: anthropic\n    model: my-test-model\n"
        "    temperature: 0.4\n    max_tokens: 100\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_MODELS_CONFIG", "models.custom.yaml")
    mc._config.cache_clear()
    assert mc.for_phase("author").model == "my-test-model"


def test_missing_override_file_falls_back_to_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_MODELS_CONFIG", "does-not-exist.yaml")
    mc._config.cache_clear()
    # Fallback defaults resolve author → the hardcoded author role model.
    assert mc.for_phase("author").model  # non-empty; didn't raise
