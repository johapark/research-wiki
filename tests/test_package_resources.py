"""Distribution regressions for setup resources used outside a git clone."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from researchwiki import package_resources


ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ROOT / "researchwiki" / "resources"


def test_bundled_env_template_matches_repository_copy():
    source = (ROOT / ".env.template").read_text(encoding="utf-8")
    assert package_resources.bundled_env_template_text() == source


def test_bundled_model_templates_exactly_match_repository_copies():
    source = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "config").glob("models.*.yaml"))
    }
    bundled = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((BUNDLED / "config").glob("models.*.yaml"))
    }

    assert bundled == source
    for name, contents in source.items():
        assert package_resources.bundled_model_template_text(name) == contents


def test_local_template_takes_precedence(tmp_path):
    name = "models.anthropic.yaml"
    local = tmp_path / name
    local.write_text("local checkout template\n", encoding="utf-8")
    assert package_resources.model_template_text(tmp_path, name) == (
        "local checkout template\n"
    )


def test_model_template_name_rejects_resource_traversal():
    import pytest

    with pytest.raises(ValueError, match="invalid models template name"):
        package_resources.bundled_model_template_text("../models.secret.yaml")


def test_wheel_and_sdist_contain_setup_resources(tmp_path):
    """Exercise setuptools itself so a package-data typo cannot pass CI."""
    project = tmp_path / "project"
    project.mkdir()
    for name in (
        "pyproject.toml", "README.md", "LICENSE", "MANIFEST.in",
        ".env.template",
    ):
        shutil.copy2(ROOT / name, project / name)
    shutil.copytree(ROOT / "researchwiki", project / "researchwiki")
    # Include the source tests in the build fixture so the distribution-level
    # exclusion is exercised instead of passing because the fixture omitted the
    # very tree MANIFEST.in is meant to prune.
    shutil.copytree(
        ROOT / "tests",
        project / "tests",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (project / "config").mkdir()
    for template in (ROOT / "config").glob("models.*.yaml"):
        shutil.copy2(template, project / "config" / template.name)

    dist = tmp_path / "dist"
    dist.mkdir()
    # setuptools' legacy command machinery mutates process-global argv state,
    # so exercise each PEP 517 hook in a fresh interpreter just as a real
    # frontend does.
    for hook in ("build_sdist", "build_wheel"):
        script = (
            "import os, setuptools.build_meta as backend, sys; "
            "os.chdir(sys.argv[1]); "
            f"print(backend.{hook}(sys.argv[2]))"
        )
        subprocess.run(
            [sys.executable, "-c", script, str(project), str(dist)],
            check=True,
            capture_output=True,
            text=True,
        )

    expected = {
        "researchwiki/resources/.env.template",
        *{
            f"researchwiki/resources/config/{path.name}"
            for path in (ROOT / "config").glob("models.*.yaml")
        },
    }
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
        assert expected <= wheel_names
        assert not any(name == "tests" or name.startswith("tests/")
                       for name in wheel_names)

    sdist = next(dist.glob("*.tar.gz"))
    with tarfile.open(sdist, "r:gz") as archive:
        names = {name.split("/", 1)[-1] for name in archive.getnames()}
        assert expected <= names
        assert ".env.template" in names
        assert {
            f"config/{path.name}"
            for path in (ROOT / "config").glob("models.*.yaml")
        } <= names
        assert not any(name == "tests" or name.startswith("tests/")
                       for name in names)
