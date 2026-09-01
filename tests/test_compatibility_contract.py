"""Executable checks for the beta compatibility and deprecation contract."""

from __future__ import annotations

from datetime import date
from importlib import resources
from pathlib import Path

import yaml

from researchwiki import __main__ as cli
from researchwiki.db.iterations import VALID_ROLES
from researchwiki.tasks.lint.contracts import LINT_JSON_KEYS


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "researchwiki" / "data" / "deprecations.yaml"


def _ledger() -> dict:
    return yaml.safe_load(LEDGER.read_text(encoding="utf-8"))


def _version(value: object) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in str(value).split("."))
    assert len(parts) == 3
    return parts


def test_contract_promises_both_time_and_release_windows():
    text = (ROOT / "COMPATIBILITY.md").read_text(encoding="utf-8")
    assert "at least 90 days" in text
    assert "entire next minor release line" in text
    assert "second subsequent minor release" in text
    assert "Deprecation notices go to stderr" in text
    assert "dual emission" in text
    assert "Readers accept both deprecated and replacement fields" in text


def test_active_deprecations_satisfy_minimum_windows_and_keep_aliases():
    data = _ledger()
    assert data["schema_version"] == 1
    rows = data["deprecations"]
    assert len({row["id"] for row in rows}) == len(rows)
    commands = cli._discover_tasks()

    for row in rows:
        assert row["status"] in {"active", "removed"}
        announced = row["announced_on"]
        removable = row["removal_not_before_date"]
        if not isinstance(announced, date):
            announced = date.fromisoformat(str(announced))
        if not isinstance(removable, date):
            removable = date.fromisoformat(str(removable))
        assert (removable - announced).days >= 90

        announced_version = _version(row["announced_in"])
        removal_version = _version(row["removal_not_before_version"])
        assert removal_version[0] == announced_version[0]
        assert removal_version[1] >= announced_version[1] + 2

        if row["status"] == "active" and row["surface"] == "cli-command":
            old_top_level = row["deprecated"].split()[1]
            assert old_top_level in commands


def test_persisted_roles_are_append_only_and_inventory_is_current():
    inventory = set(_ledger()["persisted_phase_roles"])
    assert inventory == VALID_ROLES


def test_lint_json_contract_carries_separate_legacy_acknowledgments():
    assert "missing_author_model" in LINT_JSON_KEYS
    assert "acknowledged_legacy_provenance" in LINT_JSON_KEYS


def test_deprecation_ledger_is_in_the_installed_package():
    packaged = resources.files("researchwiki").joinpath("data/deprecations.yaml")
    assert "audit-command" in packaged.read_text(encoding="utf-8")
