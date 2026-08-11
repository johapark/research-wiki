"""The exit-code contract CLAUDE.md publishes, pinned at the CLI funnel.

`researchwiki.__main__.main` is the single entry point (`[project.scripts]`
points straight at it), so it is the one place the contract can be enforced:

  | 0 | success                                                      |
  | 1 | user-input error — fix your command line                      |
  | 2 | environment error — missing index, unreachable provider, disk |
  | 3 | internal bug / uncaught exception                             |

These codes are deliberately *not* argparse's convention, where 2 means a
usage error. `AGENTS.md` is a symlink to CLAUDE.md, so every other agent tool
(Codex, Cursor, Aider, Gemini CLI) reads this table verbatim and branches on
it: 1 says "I typo'd a flag", 2 says "go look at the machine". Conflating the
two sends a caller off inspecting a healthy disk.

Two regressions this file guards:
  - Code 3 was unreachable. An uncaught exception in a task module escaped as
    Python's own exit 1, which the table labels a user-input error.
  - argparse's SystemExit(2) reached the shell untouched, so `researchwiki db
    --nope` reported an environment failure.

Hermetic: fake task modules, no real command runs.
"""

from __future__ import annotations

import argparse
import sys

import pytest

from researchwiki import __main__ as cli


@pytest.fixture(autouse=True)
def _in_a_wiki_dir(tmp_path, monkeypatch):
    """Run every test from a directory that looks like a wiki root.

    `main()` guards on `Path.cwd()/"wiki"` being a directory and returns 2
    ("environment error — cd to the wiki root") before dispatching. Without
    this, every assertion below silently measured that guard instead of the
    code under test: the file claims to be hermetic, but it was passing only
    because pytest happened to be invoked from the repo root, and failed from
    anywhere else — including most IDE test runners.

    A tmp dir keeps it hermetic *and* exercises the real guard path rather than
    patching it out.
    """
    (tmp_path / "wiki").mkdir()
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fake_task(monkeypatch):
    """Register a synthetic `researchwiki faketask` whose `main` we control.

    Injecting into `sys.modules` is enough — `main()` reaches the module via
    `importlib.import_module`, which returns the cached entry without touching
    the filesystem.
    """
    import types

    def install(fn):
        mod = types.ModuleType("researchwiki.tasks.faketask")
        mod.__doc__ = "Synthetic task for exit-code tests."
        mod.main = fn
        monkeypatch.setitem(sys.modules, "researchwiki.tasks.faketask", mod)
        monkeypatch.setattr(cli, "_discover_tasks", lambda: {"faketask": "faketask"})
        return mod
    return install


# ---------- 0: success ----------

def test_success_returns_0(fake_task):
    fake_task(lambda argv: 0)
    assert cli.main(["faketask"]) == 0


def test_none_return_normalizes_to_0(fake_task):
    # Task modules are allowed to fall off the end of `main` without returning.
    fake_task(lambda argv: None)
    assert cli.main(["faketask"]) == 0


# ---------- 1: user-input error ----------

def test_no_args_returns_1(capsys):
    # Bare `researchwiki` prints help to stderr. argparse would call this 2.
    assert cli.main([]) == 1
    assert "usage:" in capsys.readouterr().err


def test_unknown_command_returns_1(capsys):
    assert cli.main(["no-such-command"]) == 1
    assert "unknown command" in capsys.readouterr().err


def test_argparse_usage_error_remapped_to_1(fake_task, capsys):
    """The load-bearing remap. All 34 task modules build their own
    ArgumentParser, and every one of them exits 2 on a bad flag. Rewriting each
    parser is not viable; the funnel is."""
    def task_main(argv):
        p = argparse.ArgumentParser(prog="researchwiki faketask")
        p.add_argument("--count", type=int)
        p.parse_args(argv)          # raises SystemExit(2) on bad input
        return 0
    fake_task(task_main)

    assert cli.main(["faketask", "--count", "not-a-number"]) == 1
    assert cli.main(["faketask", "--nonexistent-flag"]) == 1
    # The argparse diagnostic still reaches the user; only the code changed.
    assert "usage:" in capsys.readouterr().err


def test_task_level_sys_exit_1_passes_through(fake_task):
    # Helpers deep in a task (e.g. `_ingest_batch._resolve_inputs`) reject bad
    # argv with `sys.exit(1)`. That's already on-contract — don't touch it.
    def task_main(argv):
        sys.exit(1)
    fake_task(task_main)
    assert cli.main(["faketask"]) == 1


def test_keyboard_interrupt_returns_1(fake_task, capsys):
    def task_main(argv):
        raise KeyboardInterrupt
    fake_task(task_main)
    assert cli.main(["faketask"]) == 1
    assert "interrupted" in capsys.readouterr().err


# ---------- 2: environment error ----------

def test_unimportable_task_module_returns_2(monkeypatch, capsys):
    """A task the discovery walk lists but that won't import means a missing
    dependency or a broken install — the caller's command line was fine."""
    monkeypatch.setattr(cli, "_discover_tasks",
                        lambda: {"ghost": "module_that_does_not_exist"})
    assert cli.main(["ghost"]) == 2
    assert "cannot load task module" in capsys.readouterr().err


# ---------- 3: internal bug ----------

def test_uncaught_exception_returns_3(fake_task, capsys):
    """Code 3's whole reason to exist, and previously unreachable: without the
    handler in `main`, this surfaced as Python's own exit 1 — indistinguishable
    from "you passed a bad flag"."""
    def task_main(argv):
        raise RuntimeError("boom")
    fake_task(task_main)

    assert cli.main(["faketask"]) == 3
    err = capsys.readouterr().err
    assert "RuntimeError: boom" in err     # traceback preserved for debugging
    assert "internal error" in err


def test_uncaught_exception_is_not_confused_with_environment(fake_task):
    # An ImportError raised *inside* a task's body is a bug in that task, not
    # a broken install — only a failure to import the module itself is 2.
    def task_main(argv):
        raise ImportError("optional dep missing at call time")
    fake_task(task_main)
    assert cli.main(["faketask"]) == 3


# ---------- pass-through cases that must not be swept up ----------

def test_broken_pipe_returns_0(fake_task):
    # `researchwiki search ... | head` closes the pipe early. Not a failure.
    def task_main(argv):
        raise BrokenPipeError
    fake_task(task_main)
    assert cli.main(["faketask"]) == 0


def test_subparser_help_still_exits_0(fake_task, capsys):
    """`researchwiki faketask --help` raises SystemExit(0) from argparse. The
    remap only touches 2, so this must stay 0 — otherwise every `--help` in the
    tool starts reporting failure."""
    def task_main(argv):
        p = argparse.ArgumentParser(prog="researchwiki faketask")
        p.parse_args(argv)
        return 0
    fake_task(task_main)
    assert cli.main(["faketask", "--help"]) == 0
    assert "usage:" in capsys.readouterr().out


def test_top_level_help_returns_0(capsys):
    assert cli.main(["--help"]) == 0
    assert "usage: researchwiki" in capsys.readouterr().out


# ---------- the page-gate family's local convention ----------
#
# CLAUDE.md tells the author to run `check-grounding`, `grade synthesis` and
# `check-coverage` on the same page, and to require exit 0 from the first two.
# All three therefore document 0/1/2 as clean / gate-found-something / bad input,
# and for them exit 1 is the *expected* review-triggering outcome rather than a
# complaint about argv. Folding "that file isn't there" into 1 would make a
# workflow that retries on nonzero spin forever on a filename typo — so these
# three keep 2 for bad input, and the value of that is that they *agree*.
#
# This is the guard against a well-meant sweep aligning one of them to the
# general contract and silently splitting the trio.

@pytest.mark.parametrize("module_name,argv_prefix", [
    ("check_grounding", []),
    ("_grade_synthesis", []),
    ("check_coverage", []),
])
def test_page_gates_agree_on_missing_path(module_name, argv_prefix, tmp_path, capsys):
    import importlib
    mod = importlib.import_module(f"researchwiki.tasks.{module_name}")
    missing = str(tmp_path / "no-such-page.md")
    assert mod.main([*argv_prefix, missing]) == 2, (
        f"{module_name} disagrees with its sibling gates on a missing page. "
        "All three use 2 for bad input so that exit 1 can keep meaning "
        "'the gate found something' — see this test's preamble."
    )
    capsys.readouterr()


def test_page_gates_reserve_1_for_findings():
    """The reason the convention exists: 1 is a result, not a complaint. Pinned
    against the docstrings so the three stay documented the same way."""
    import importlib
    for module_name in ("check_grounding", "_grade_synthesis", "check_coverage"):
        doc = importlib.import_module(f"researchwiki.tasks.{module_name}").__doc__ or ""
        assert "Exit codes:" in doc, f"{module_name} lost its exit-code table"
        line = next(ln for ln in doc.splitlines() if ln.strip().startswith("2 "))
        assert "input" in line.lower(), \
            f"{module_name} documents code 2 as {line.strip()!r}, not bad input"


# ---------- import ----------
#
# `import` is the counter-example to the page gates above: it is not a gate, so
# it follows the *table* in CLAUDE.md rather than the gate exception. Its phases
# differ deliberately in what "nothing to do" means, and that is the part worth
# pinning — `inspect` finding nothing importable is a **result**, not an error,
# while `apply` (stage 4) having nothing to act on is a failure, because acting
# is the only thing it does.
#
# Reached through `importlib` because `researchwiki/tasks/import.py` is named
# for a keyword; that is deliberate, and its module docstring says why.

import importlib
import pathlib

FIXTURE_RIS = pathlib.Path(__file__).parent / "refimport-fixtures" / "readcube-sample.ris"


def _import_task():
    return importlib.import_module("researchwiki.tasks.import")


def test_import_command_is_discovered_under_its_keyword_name():
    """The reason the filename is a keyword: `_discover_tasks` derives the CLI
    name from it, so `import.py` is what makes the command `researchwiki
    import`. Renaming the file to something importable silently renames the
    command."""
    assert cli._discover_tasks().get("import") == "import"


def test_import_command_dispatches_through_the_cli_funnel(tmp_path, capsys):
    assert cli.main(["import", "preflight", str(tmp_path / "nope.ris")]) == 1
    capsys.readouterr()


def test_import_preflight_bad_input_returns_1(tmp_path, capsys):
    assert _import_task().main(["preflight", str(tmp_path / "nope.ris")]) == 1
    capsys.readouterr()


def test_import_unidentifiable_export_returns_1(tmp_path, capsys):
    p = tmp_path / "notes.txt"
    p.write_text("not an export at all")
    assert _import_task().main(["preflight", str(p)]) == 1
    capsys.readouterr()


def test_import_inspect_returns_0_when_nothing_is_importable(tmp_path, capsys):
    """The normal metadata-only run: every record skipped for want of a PDF.
    Exit 1 here would make the expected outcome look like a failure.

    `_in_a_wiki_dir` has already chdir'd into `tmp_path` and made `wiki/`.
    """
    (tmp_path / ".ingest").mkdir(exist_ok=True)
    assert _import_task().main(["inspect", str(FIXTURE_RIS)]) == 0
    capsys.readouterr()


def test_import_documents_its_exit_codes():
    assert "Exit codes:" in (_import_task().__doc__ or "")
