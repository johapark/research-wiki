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


# ---------- deprecated-in-place surfaces ----------
#
# Concept hubs have no replacement *command* to delegate to, so they are
# deprecated in place: every published surface keeps working and warns. These
# pin that it does, for the whole window.


def _concept_row() -> dict:
    return next(r for r in _ledger()["deprecations"] if r["id"] == "concept-hubs")


def test_concept_hub_notice_matches_the_ledger():
    from researchwiki.concepts import DEPRECATION_NOTICE

    row = _concept_row()
    assert row["status"] == "active"
    assert row["compatibility"] == "deprecated-in-place"
    assert f"no earlier than {row['removal_not_before_version']}" in DEPRECATION_NOTICE
    assert str(row["removal_not_before_date"]) in DEPRECATION_NOTICE


@pytest.mark.parametrize("argv", [
    ["concepts", "--__definitely_not_a_flag__"],
    ["candidates", "concepts", "--__definitely_not_a_flag__"],
])
def test_concept_commands_warn_on_stderr_even_when_argv_fails(argv):
    """Warned before parsing, so `--help` and a typo still announce it."""
    module = importlib.import_module(
        "researchwiki.tasks.concepts" if argv[0] == "concepts"
        else "researchwiki.tasks.candidates")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        with contextlib.suppress(SystemExit):
            module.main(argv[1:])
    assert "concept hubs are deprecated" in err.getvalue()
    assert "deprecated" not in out.getvalue()


def test_candidates_concepts_json_stdout_stays_parseable(monkeypatch):
    import json as _json

    from researchwiki.tasks import candidates

    import researchwiki.concepts as concepts_pkg
    # `_run_concepts` imports it from the package inside the function.
    monkeypatch.setattr(concepts_pkg, "collect_candidates", lambda **kw: [])
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = candidates.main(["concepts", "--json"])
    assert rc == 0
    _json.loads(out.getvalue())
    assert "deprecated" in err.getvalue()


def test_deprecated_concept_surfaces_are_still_published():
    """Removal is not allowed before both gates, so each surface must exist."""
    commands = cli._discover_tasks()
    assert "concepts" in commands and "candidates" in commands
    assert "concept_contract_violations" in LINT_JSON_KEYS
    from researchwiki.provenance import AUTHORED_PAGE_TYPES
    assert "concept" in AUTHORED_PAGE_TYPES


# ---------- experimental surfaces ----------
#
# The mirror of the deprecation rows above: a deprecated surface is frozen until
# removal, an experimental one is exempt from the freeze. Both promises are only
# worth as much as the agreement between the ledger, the prose contract and what
# the commands actually print.


def _experimental_rows() -> dict:
    rows = _ledger()["experimental"]
    assert isinstance(rows, list) and rows
    return {row["id"]: row for row in rows}


def test_experimental_ids_are_unique_and_disjoint_from_deprecations():
    rows = _ledger()["experimental"]
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    # A surface cannot be both exempt from the window and frozen by it.
    assert not set(ids) & {row["id"] for row in _ledger()["deprecations"]}


def test_every_experimental_row_names_what_it_exempts():
    """The exemption is the only thing the marker buys, so it must be explicit."""
    for entry_id, row in _experimental_rows().items():
        assert row["status"] == "active", entry_id
        assert _version(row["announced_in"]) <= _version(researchwiki_pkg.__version__)
        assert str(row.get("reason") or "").strip(), entry_id
        exempt = row["exempt_from_beta_contract"]
        assert isinstance(exempt, list) and exempt, entry_id


def test_visual_inspection_is_experimental_and_matches_its_notice():
    from researchwiki.experimental import (
        VISUAL_INSPECTION,
        VISUAL_INSPECTION_NOTICE,
        load_experimental,
    )

    row = _experimental_rows()[VISUAL_INSPECTION]
    assert VISUAL_INSPECTION in load_experimental()
    assert "experimental" in VISUAL_INSPECTION_NOTICE
    assert str(row["announced_in"]) in VISUAL_INSPECTION_NOTICE


def test_the_visual_phase_role_is_never_exempted_by_the_marker():
    """`db rebuild` cannot reconstruct telemetry, so a shipped role is forever.

    The marker must not read as permission to withdraw it — hence both the
    ledger's own `not_exempt` note and this pin against VALID_ROLES.
    """
    row = _experimental_rows()["visual-inspection"]
    not_exempt = " ".join(row["not_exempt"])
    assert "visual_inspect" in not_exempt
    assert "visual_inspect" in VALID_ROLES
    assert "visual_inspect" in set(_ledger()["persisted_phase_roles"])


def test_the_contract_documents_the_experimental_clause():
    text = (ROOT / "COMPATIBILITY.md").read_text(encoding="utf-8")
    assert "### Experimental surfaces" in text
    # The two limits that keep the exemption honest.
    assert "Persisted phase-role identifiers are never exempt" in text
    assert "stderr" in text


def test_experimental_ledger_is_in_the_installed_package():
    packaged = resources.files("researchwiki").joinpath("data/deprecations.yaml")
    assert "visual-inspection" in packaged.read_text(encoding="utf-8")


def test_inspect_warns_on_stderr_and_keeps_json_stdout_parseable(tmp_path, monkeypatch):
    import json as _json

    from researchwiki import paths
    from researchwiki.tasks import inspect as inspect_cli

    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = inspect_cli.main(["visuals", "--json"])
    assert rc == 0
    assert _json.loads(out.getvalue())["schema_version"]
    assert "experimental" in err.getvalue()
    assert "experimental" not in out.getvalue()


def test_inspect_warns_even_when_argv_fails():
    """Warned before parsing, so `--help` and a typo still announce it."""
    from researchwiki.tasks import inspect as inspect_cli

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        with contextlib.suppress(SystemExit):
            inspect_cli.main(["--__definitely_not_a_flag__"])
    assert "experimental" in err.getvalue()


def test_ingest_announces_the_marker_only_when_the_flag_is_set(monkeypatch):
    """An ordinary ingest must not grow a warning it cannot act on."""
    from researchwiki.tasks import agent as agent_cli

    parser = agent_cli.build_parser()
    for argv, expected in ((["ingest", "p.pdf", "--stub"], False),
                           (["ingest", "p.pdf", "--stub", "--visual-inspection"], True)):
        args = parser.parse_args(argv)
        # Stop immediately after the notice: `-w 0` is rejected before any work,
        # so this exercises the announce path without running an ingest.
        args.workers = 0
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            agent_cli._execute_ingest(args)
        assert ("experimental" in err.getvalue()) is expected, argv
        assert "experimental" not in out.getvalue()
