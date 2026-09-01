"""Executable checks for the beta compatibility and deprecation contract."""

from __future__ import annotations

import contextlib
import io
from datetime import date
import importlib
from importlib import resources
from pathlib import Path

import pytest

import yaml

import researchwiki as researchwiki_pkg
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


def _alias_modules() -> dict[str, str]:
    """Task modules that behave as a deprecated CLI alias.

    The ledger is only a contract if it is complete, and completeness cannot be
    checked from the ledger's own rows. Keyed on the stderr rename notice rather
    than on a docstring phrase: the notice is the behavior COMPATIBILITY.md
    actually requires, and a docstring scan already proved too weak — it read
    `eval_classifier.py`, whose summary line describes the evaluation, as not an
    alias at all, so the check passed vacuously on half the aliases it covers.
    """
    out: dict[str, str] = {}
    tasks_dir = Path(researchwiki_pkg.__file__).parent / "tasks"
    for cli_name, module_name in cli._discover_tasks().items():
        source = tasks_dir / f"{module_name}.py"
        if not source.is_file():
            continue
        text = source.read_text(encoding="utf-8")
        if "note: `" in text and "is now" in text:
            out[cli_name] = module_name
    return out


def test_every_deprecated_alias_has_a_ledger_row():
    """A new alias must not be able to ship without a deprecation window.

    The converse check (rows point at live commands) is above; this is the
    direction the contract actually depends on — an alias with no row has no
    announced date, no removal floor, and nothing in the changelog.
    """
    ledger_commands = {
        row["deprecated"].split()[1]
        for row in _ledger()["deprecations"]
        if row["surface"] == "cli-command"
    }
    for cli_name in _alias_modules():
        assert cli_name in ledger_commands, (
            f"`researchwiki {cli_name}` calls itself a deprecated alias but has no "
            f"row in deprecations.yaml"
        )


#: Each active CLI alias, with the downstream worker to stub so the notice can be
#: observed without a wiki or a network.
_ALIAS_WORKERS = {
    "audit": ("researchwiki.tasks.audit", "citations"),
    "eval-classifier": ("researchwiki.tasks.eval_classifier", "evaluate"),
}


@pytest.mark.parametrize("command", sorted(_ALIAS_WORKERS))
def test_active_cli_aliases_announce_on_stderr_and_keep_stdout_clean(command, monkeypatch):
    """COMPATIBILITY.md promises the notice goes to stderr, not stdout.

    Asserting that sentence appears in the markdown (above) tests the document.
    This tests the behavior it promises — the property that keeps `--json` stdout
    parseable for the whole window an alias is still warning.
    """
    module_name, worker = _ALIAS_WORKERS[command]
    module = importlib.import_module(module_name)
    if worker == "citations":
        monkeypatch.setattr(module.citations, "main", lambda *a, **k: 0)
    else:
        monkeypatch.setattr(module, worker, lambda *a, **k: 0)

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = module.main([])

    assert rc == 0
    assert "is now" in err.getvalue(), f"{command} printed no deprecation notice to stderr"
    assert "is now" not in out.getvalue(), f"{command} leaked its notice onto stdout"


def test_alias_notices_survive_argparse_failure():
    """The notice must not be contingent on the rest of the argv parsing.

    `audit` printed before parsing and `eval-classifier` after, so `--help` (and
    any bad flag) announced nothing on one of them. A user discovering the new
    spelling from `--help` is exactly the case the notice exists for.
    """
    for command, (module_name, _worker) in _ALIAS_WORKERS.items():
        module = importlib.import_module(module_name)
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            with contextlib.suppress(SystemExit):
                module.main(["--__definitely_not_a_flag__"])
        assert "is now" in err.getvalue(), (
            f"`researchwiki {command}` announced nothing when argv failed to parse"
        )


def test_alias_discovery_covers_every_known_alias():
    """Guard the guard: an empty or shrunken alias set makes the two checks above
    pass without testing anything."""
    assert set(_alias_modules()) == set(_ALIAS_WORKERS)
