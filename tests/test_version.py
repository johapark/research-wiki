"""Release invariants: one version literal, and a changelog entry that matches it.

`0.1.0` used to be written twice — `pyproject.toml` and `researchwiki/__init__.py`
— with nothing tying them together. `--version` reads only the second (see
`researchwiki.__main__._build_parser`), so a bump that touched the packaging metadata
alone would ship a CLI reporting the previous release. The fix was structural: `pyproject.toml` declares
the version `dynamic` and resolves it from the package literal, which makes drift
impossible rather than merely detectable. `test_pyproject_has_no_second_version_literal`
is what stops someone reinstating the copy.

The changelog assertions pin the other half of a release: bumping the version and
recording what's in it have to happen in the same commit, or a tagged release ships
with either no notes or somebody else's.

Deliberately absent: any tag-vs-version check. `actions/checkout` fetches no tags by
default, so a test would pass in CI for the wrong reason. That guard lives in
`.github/workflows/release.yml`, where the tag *is* the trigger.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import researchwiki

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"

# MAJOR.MINOR.PATCH with an optional pre-release suffix (`-rc.1`, `-alpha.2`).
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")

# A released section heading: `## [0.2.0] - 2026-08-07`. `## [Unreleased]` is
# deliberately not matched — it's the staging area, not a release.
_RELEASED_HEADING = re.compile(
    r"^## \[(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)\] - (\d{4}-\d{2}-\d{2})\s*$",
    re.MULTILINE,
)


def _released_versions() -> list[str]:
    return [m.group(1) for m in _RELEASED_HEADING.finditer(CHANGELOG.read_text("utf-8"))]


def _sort_key(version: str) -> tuple[int, int, int]:
    return tuple(int(p) for p in version.split("-")[0].split("."))  # type: ignore[return-value]


# --- the single literal ----------------------------------------------------


def test_version_is_semver():
    assert _SEMVER.fullmatch(researchwiki.__version__), (
        f"__version__ = {researchwiki.__version__!r} is not MAJOR.MINOR.PATCH"
    )


def test_pyproject_declares_the_version_dynamic():
    """Parsed as text, not with `tomllib`: CI runs Python 3.10, where that module
    doesn't exist, and CONTRIBUTING forbids adding a dependency for the suite."""
    text = PYPROJECT.read_text("utf-8")
    assert 'dynamic = ["version"]' in text, (
        'pyproject.toml [project] must declare `dynamic = ["version"]`'
    )
    assert 'version = {attr = "researchwiki.__version__"}' in text, (
        "pyproject.toml must resolve the version from the package literal via "
        '[tool.setuptools.dynamic] version = {attr = "researchwiki.__version__"}'
    )


def test_pyproject_has_no_second_version_literal():
    """The regression this file exists for. A `version = "x.y.z"` line anywhere in
    pyproject.toml means the number is written twice again."""
    offenders = [
        line for line in PYPROJECT.read_text("utf-8").splitlines()
        if re.match(r'^\s*version\s*=\s*["\']', line)
    ]
    assert not offenders, (
        "pyproject.toml carries a static version literal "
        f"({offenders[0].strip()!r}). The version lives in researchwiki/__init__.py "
        "and is resolved via [tool.setuptools.dynamic] — see CONTRIBUTING.md § Releasing."
    )


def test_build_backend_resolves_the_dynamic_version():
    """Guards the wiring end to end: if the attr path breaks, every built artifact
    is misversioned, and nothing else in the suite would notice.

    Skipped where setuptools isn't importable: it's the *build* backend, and a
    Python 3.12+ venv doesn't ship it at runtime. Asserting on it unconditionally
    made the suite fail in an environment that installs only what it needs.
    """
    pytest.importorskip(
        "setuptools", reason="build backend not installed in this environment"
    )
    from setuptools.config.pyprojecttoml import read_configuration

    resolved = read_configuration(str(PYPROJECT))["project"].get("version")
    assert resolved == researchwiki.__version__, (
        f"setuptools resolves {resolved!r} but __version__ is "
        f"{researchwiki.__version__!r}"
    )


# --- the changelog --------------------------------------------------------


def test_changelog_exists_and_stages_unreleased_work():
    text = CHANGELOG.read_text("utf-8")
    assert "## [Unreleased]" in text, (
        "CHANGELOG.md needs a standing `## [Unreleased]` section — it's where entries "
        "accumulate between releases."
    )


def test_top_released_section_matches_version():
    """Forces the bump and its entry into one commit."""
    released = _released_versions()
    assert released, "CHANGELOG.md has no released section"
    assert released[0] == researchwiki.__version__, (
        f"CHANGELOG.md's newest released section is {released[0]}, but __version__ is "
        f"{researchwiki.__version__}. A release commit moves `## [Unreleased]` into a "
        f"dated section and bumps the literal together."
    )


def test_released_versions_descend():
    """Catches an entry inserted in the wrong place, which would silently make the
    previous test compare against the wrong section."""
    released = _released_versions()
    keys = [_sort_key(v) for v in released]
    assert keys == sorted(keys, reverse=True), (
        f"released sections are out of order: {released}"
    )


def test_no_duplicate_released_versions():
    released = _released_versions()
    assert len(released) == len(set(released)), f"duplicate section: {released}"


def test_every_released_version_has_a_link_reference(label="Unreleased"):
    """Keep a Changelog's compare links are how a reader gets from an entry to the
    diff. A missing one renders as literal `[0.2.0]` text."""
    text = CHANGELOG.read_text("utf-8")
    for version in [label, *_released_versions()]:
        assert f"[{version}]: https://" in text, (
            f"CHANGELOG.md is missing the link reference for [{version}]"
        )


# --- the release-notes extractor ------------------------------------------
#
# `release.yml` publishes whatever this returns, so a bug here is a bug in the
# release notes of every version. The specific regression: terminating a section
# only at the next `## [` heading let the oldest section run to end-of-file and
# absorb the link-reference block.

_EXTRACTOR = ROOT / ".github" / "scripts" / "changelog_section.py"


def _load_extractor():
    import importlib.util

    spec = importlib.util.spec_from_file_location("changelog_section", _EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_FIXTURE = (
    "# Changelog\n\n"
    "## [Unreleased]\n\n### Added\n\n- pending thing\n\n"
    "## [0.2.0] - 2026-08-07\n\n### Added\n\n- the new thing\n\n"
    "## [0.1.0] - 2026-07-18\n\nInitial tagged release.\n\n"
    "[Unreleased]: https://example.invalid/compare/v0.2.0...HEAD\n"
    "[0.2.0]: https://example.invalid/compare/v0.1.0...v0.2.0\n"
    "[0.1.0]: https://example.invalid/releases/tag/v0.1.0\n"
)


def test_extractor_stops_at_the_next_section():
    body = _load_extractor().section_body(_FIXTURE, "0.2.0")
    assert body == "### Added\n\n- the new thing"


def test_extractor_stops_at_the_link_reference_block():
    """The oldest section has no following heading — it must not absorb the links."""
    body = _load_extractor().section_body(_FIXTURE, "0.1.0")
    assert body == "Initial tagged release."
    assert "https://" not in body


def test_extractor_accepts_a_v_prefixed_tag():
    """`release.yml` passes `$GITHUB_REF_NAME`, which carries the `v`."""
    module = _load_extractor()
    assert module.section_body(_FIXTURE, "v0.2.0") == module.section_body(_FIXTURE, "0.2.0")


def test_extractor_never_returns_the_unreleased_section():
    """Tagging a version whose entries are still staged under [Unreleased] must
    fail loudly rather than publish the staging area as release notes."""
    module = _load_extractor()
    with pytest.raises(ValueError, match="no released section"):
        module.section_body(_FIXTURE, "0.3.0")
    with pytest.raises(ValueError, match="no released section"):
        module.section_body(_FIXTURE, "Unreleased")


def test_extractor_rejects_an_empty_section():
    module = _load_extractor()
    empty = "## [0.4.0] - 2026-09-01\n\n## [0.3.0] - 2026-08-01\n\nreal body\n"
    with pytest.raises(ValueError, match="is empty"):
        module.section_body(empty, "0.4.0")


def test_current_version_section_extracts_cleanly():
    """End to end against the real file: whatever the next tag is, the notes it
    would publish are non-empty and free of link-reference lines."""
    module = _load_extractor()
    body = module.section_body(CHANGELOG.read_text("utf-8"), researchwiki.__version__)
    assert body
    assert "]: https://" not in body
