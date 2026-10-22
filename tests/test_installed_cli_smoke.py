"""Subprocess smoke checks for the CLI as a *user* invokes it.

The gap this closes: `beta-acceptance` runs
`.github/scripts/fresh_install_acceptance.py` outside pytest, and nothing under
`tests/` referenced it. So when that lifecycle broke, 3900 passing tests said
nothing — the failure was only visible on three CI jobs. Every other test in
this suite imports the package in-process, which never exercises the console
entry point, argparse wiring, or exit-code funnel the way a shell does.

These are deliberately cheap and hermetic: no network, no API calls, no writes
outside `tmp_path`, and no ML stack beyond what importing the package already
needs. They are not a replacement for the acceptance lifecycle, which installs
into a clean venv and runs a real ingest. They are the fast signal that catches
a broken entry point or a mis-wired exit code before CI does.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _cli(*args: str, cwd=None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Invoke the CLI the way a shell does: a fresh interpreter, `-m`."""
    return subprocess.run(
        [sys.executable, "-m", "researchwiki", *args],
        cwd=cwd, capture_output=True, encoding="utf-8", timeout=timeout,
    )


def test_bare_invocation_prints_help_and_exits_1():
    """Exit-code contract: no command is user-input error, not success."""
    proc = _cli()
    assert proc.returncode == 1
    assert "COMMAND" in proc.stderr or "usage" in proc.stderr.lower()


def test_help_exits_0_and_lists_the_core_commands():
    proc = _cli("--help")
    assert proc.returncode == 0, proc.stderr
    for command in ("add", "agent", "ingest", "lint", "status", "search", "remove"):
        assert command in proc.stdout, f"{command!r} missing from --help"


def test_version_is_reported_and_matches_the_package():
    from researchwiki import __version__

    proc = _cli("--version")
    assert proc.returncode == 0, proc.stderr
    assert __version__ in (proc.stdout + proc.stderr)


def test_an_unknown_command_is_a_user_error_not_a_crash():
    proc = _cli("definitely-not-a-command")
    assert proc.returncode == 1, proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("command", ["lint", "status", "search", "remove", "export"])
def test_every_entry_point_can_at_least_render_its_help(command):
    """A command whose module fails to import still has a parser, so `--help`
    is the cheapest proof that each entry point is wired up at all."""
    proc = _cli(command, "--help")
    assert proc.returncode == 0, f"{command} --help failed: {proc.stderr[:400]}"
    assert "Traceback" not in proc.stderr


def _repo_root():
    from pathlib import Path as _Path
    return _Path(__file__).resolve().parent.parent


def _checkout_like(tmp_path):
    """`init` refuses a directory that is not a checkout, so give it one.

    It verifies `config/pricing.yaml` and `prompts/author-system-research.md`
    exist before scaffolding — the same reason `fresh_install_acceptance.py`
    copytrees both directories into its temp root. Symlinked here rather than
    copied: these are read-only inputs and the prompts tree is not small.
    """
    root = _repo_root()
    for name in ("config", "prompts"):
        (tmp_path / name).symlink_to(root / name, target_is_directory=True)
    return tmp_path


def test_scaffold_init_then_status_round_trips_in_a_fresh_directory(tmp_path):
    """The first two commands a new user runs, in a directory with no wiki.

    `fresh_install_acceptance.py` opens with exactly this pair; running it here
    means a break in the scaffold shows up in pytest rather than only in the
    three beta-acceptance jobs.
    """
    _checkout_like(tmp_path)
    init = _cli("init", "--scaffold-only", cwd=tmp_path)
    assert init.returncode == 0, init.stderr
    for created in ("wiki", "papers", "inbox"):
        assert (tmp_path / created).is_dir(), f"{created}/ was not scaffolded"

    status = _cli("status", cwd=tmp_path)
    assert status.returncode == 0, status.stderr
    assert "Research Wiki" in status.stdout


def test_lint_on_a_fresh_scaffold_is_clean_and_json_parses(tmp_path):
    import json

    _checkout_like(tmp_path)
    assert _cli("init", "--scaffold-only", cwd=tmp_path).returncode == 0
    proc = _cli("lint", "--json", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)  # the whole stream must be one document
    assert payload["pages_scanned"] >= 0
    assert payload["broken_wikilinks"] == []
