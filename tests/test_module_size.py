"""Keep every module small enough for an agent to hold in context.

A file that does not fit in a context window is a file that gets edited blind, so
the ceiling is a legibility budget rather than a matter of taste. It lives in the
test suite because that is the only place a budget actually holds: a convention in
a style guide is advisory, and a `lint` finding is something you can run past.

**Existing debt is pinned, not pardoned.** `_DEBT` maps a module to the size it
was when this gate landed, so a listed file may shrink as much as you like and
cannot gain a single line. The obvious alternative — a plain set of exempt paths —
grants a permanent licence to grow, which is how `agents/runner.py` got to 1214
lines with no test ever objecting. Pinning the number means the next fifty lines
have to justify themselves in review.

`test_debt_list_has_no_dead_entries` is the other half of the ratchet: a module
that has been split back under the ceiling looks *identical to a passing gate*, so
without that check the list would quietly turn into a monument to work already
finished.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

import pytest

import researchwiki

MAX_LINES = 800

# Locate the package through the import system rather than by walking up from
# `__file__`. If this test is ever moved, path arithmetic would silently point at
# a directory with no modules in it and the gate would pass by scanning nothing.
_PACKAGE = Path(researchwiki.__file__).resolve().parent
_REPO = _PACKAGE.parent

# Module (repo-relative, posix) -> pinned ceiling, set 2026-08-14.
#
# Every entry is an admission that a file is too long and was not split that day.
# Editing a number upward is the same admission again, so do it in a commit that
# says why. When a module drops under MAX_LINES, delete its line.
_DEBT: dict[str, int] = {
    # Agent-ingest orchestrator: phase sequencing, the retry/DEBUG loop, and
    # sandbox-vs-promote routing. Splits along the phase boundary.
    "researchwiki/agents/runner.py": 1214,
    # Metadata reconcile: PDF-side extraction against provider records, plus the
    # sanity gate. The gate is the separable half.
    "researchwiki/agents/phases/reconcile.py": 1131,
    # Concept candidates: term mining, scoring, triage labelling. Mining and
    # labelling do not need to share a module.
    "researchwiki/concepts/candidates.py": 1074,
    # Transactional promote: five journalled steps and their rollback. Each step
    # could stand alone under a thin coordinator.
    "researchwiki/agents/promote.py": 920,
    # Retrieval benchmark: fixture loading, scoring and reporting in one file.
    "researchwiki/benchmark/retrieval.py": 909,
    # Provider client: request shaping, cache_control placement, retry, and the
    # per-provider quirks. Splits per provider family.
    "researchwiki/agents/llm.py": 902,
    # Backfill targets (hook / keywords / doi) share only a work-list idiom.
    "researchwiki/tasks/backfill.py": 854,
    # Memory evolution: candidate selection, proposal drafting, emit.
    "researchwiki/agents/phases/evolution.py": 843,
    # `researchwiki/tasks/lint/__init__.py` sat here at 831 until its two emitters
    # moved to `lint/report.py`; at 265 lines it no longer needs an entry, and the
    # entry is gone rather than commented out. This list is current debt, not a log.
}


class Oversize(NamedTuple):
    """One module that fails the gate, and which of the two ways it failed."""

    path: str
    lines: int
    ceiling: int | None      # None => never had an exemption

    @property
    def is_debt_growth(self) -> bool:
        return self.ceiling is not None

    def describe(self) -> str:
        if self.ceiling is None:
            return f"{self.path}: {self.lines} lines"
        return (f"{self.path}: {self.lines} lines, pinned at {self.ceiling} "
                f"(+{self.lines - self.ceiling})")


def count_lines(module: Path) -> int:
    """Physical lines in `module`.

    Counted from bytes via `splitlines()`, which treats `\\n`, `\\r\\n` and a lone
    `\\r` as breaks alike — a file saved with classic-Mac endings would otherwise
    read as one enormous line and sail through. Bytes also mean a stray non-UTF-8
    character fails that file's own decode rather than crashing the whole gate.
    """
    return len(module.read_bytes().splitlines())


def python_modules(root: Path) -> Iterator[Path]:
    for module in sorted(root.rglob("*.py")):
        if "__pycache__" not in module.parts:
            yield module


def scan(
    *, package: Path, repo: Path, ceiling: int, debt: dict[str, int]
) -> list[Oversize]:
    """Every module that either crosses `ceiling` or outgrew its pinned size."""
    failures: list[Oversize] = []
    for module in python_modules(package):
        rel = module.relative_to(repo).as_posix()
        lines = count_lines(module)
        pinned = debt.get(rel)
        if pinned is not None:
            if lines > pinned:
                failures.append(Oversize(rel, lines, pinned))
        elif lines >= ceiling:
            failures.append(Oversize(rel, lines, None))
    return failures


def _write(directory: Path, name: str, lines: int) -> Path:
    target = directory / name
    target.write_text("value = 0\n" * lines, encoding="utf-8")
    return target


def _scan_tmp(directory: Path, *, ceiling: int, debt=None) -> list[Oversize]:
    return scan(package=directory, repo=directory, ceiling=ceiling, debt=debt or {})


# --------------------------------------------------------------------------
# The detector, exercised on throwaway trees. A gate nobody tested is a gate
# that reports "all clear" for whatever reason it likes.
# --------------------------------------------------------------------------


def test_only_the_long_module_is_reported(tmp_path):
    _write(tmp_path, "sprawling.py", 6)
    _write(tmp_path, "tidy.py", 2)
    assert [f.path for f in _scan_tmp(tmp_path, ceiling=4)] == ["sprawling.py"]


def test_a_module_sitting_exactly_on_the_ceiling_fails():
    # The docstring promises modules stay *under* the budget, so landing on it is
    # already too big. Asserted because `>=` versus `>` is a one-character slip
    # that widens the budget without anyone noticing.
    assert MAX_LINES == 800


def test_the_ceiling_is_inclusive(tmp_path):
    _write(tmp_path, "borderline.py", 4)
    assert [f.path for f in _scan_tmp(tmp_path, ceiling=4)] == ["borderline.py"]


def test_generated_directories_are_not_scanned(tmp_path):
    cached = tmp_path / "__pycache__"
    cached.mkdir()
    _write(cached, "stale.py", 50)
    assert _scan_tmp(tmp_path, ceiling=4) == []


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"])
def test_every_line_ending_counts(tmp_path, ending):
    target = tmp_path / "endings.py"
    target.write_bytes(b"value = 0" + ending + (b"value = 1" + ending) * 6)
    assert count_lines(target) == 7


def test_a_bad_byte_fails_its_own_file_not_the_gate(tmp_path):
    target = tmp_path / "latin.py"
    target.write_bytes(b"label = '\xff'\n" * 3)
    assert count_lines(target) == 3


def test_pinned_module_within_its_size_passes(tmp_path):
    _write(tmp_path, "known.py", 9)
    assert _scan_tmp(tmp_path, ceiling=4, debt={"known.py": 9}) == []


def test_pinned_module_may_shrink(tmp_path):
    _write(tmp_path, "known.py", 6)
    assert _scan_tmp(tmp_path, ceiling=4, debt={"known.py": 9}) == []


def test_pinned_module_may_not_grow(tmp_path):
    # The entire reason the debt list stores a number instead of a bare path.
    _write(tmp_path, "known.py", 11)
    failures = _scan_tmp(tmp_path, ceiling=4, debt={"known.py": 9})
    assert [(f.path, f.lines, f.ceiling) for f in failures] == [("known.py", 11, 9)]
    assert failures[0].is_debt_growth


def test_the_two_failure_modes_are_told_apart(tmp_path):
    _write(tmp_path, "known.py", 11)
    _write(tmp_path, "fresh.py", 7)
    failures = {f.path: f.is_debt_growth
                for f in _scan_tmp(tmp_path, ceiling=4, debt={"known.py": 9})}
    assert failures == {"known.py": True, "fresh.py": False}


# --------------------------------------------------------------------------
# The gate.
# --------------------------------------------------------------------------


def test_no_module_is_too_long():
    modules = list(python_modules(_PACKAGE))
    assert modules, f"scanned {_PACKAGE} and found no modules — the gate is inert"

    failures = scan(package=_PACKAGE, repo=_REPO, ceiling=MAX_LINES, debt=_DEBT)
    if not failures:
        return

    complaints: list[str] = []
    fresh = [f for f in failures if not f.is_debt_growth]
    grown = [f for f in failures if f.is_debt_growth]
    if fresh:
        complaints.append(
            f"Over the {MAX_LINES}-line budget:\n"
            + "\n".join(f"  - {f.describe()}" for f in fresh)
            + "\n  Fix: lift a cohesive group into its own module. Pinning a new "
              "file in _DEBT is not the intended way out — that list is for debt "
              "this gate inherited."
        )
    if grown:
        complaints.append(
            "Already-pinned modules that grew:\n"
            + "\n".join(f"  - {f.describe()}" for f in grown)
            + "\n  Fix: put the new code in a focused sibling, or split the file "
              "while you are in it. Raising the pin is available and is a decision "
              "to record in the commit message, not a formality."
        )
    raise AssertionError("\n\n".join(complaints))


def test_debt_list_has_no_dead_entries():
    """Prune entries that no longer hold anything back.

    An entry goes dead two ways: the module was split under the budget, or it was
    renamed away. Neither is visible from the gate above — a dead entry produces a
    green run, which is precisely why it needs its own assertion.
    """
    dead: list[str] = []
    for rel, pinned in sorted(_DEBT.items()):
        module = _REPO / rel
        if not module.exists():
            dead.append(f"  - {rel}: gone (renamed or deleted)")
        elif (lines := count_lines(module)) < MAX_LINES:
            dead.append(f"  - {rel}: down to {lines} lines, inside the "
                        f"{MAX_LINES} budget (pinned at {pinned})")
    if dead:
        raise AssertionError(
            "Dead entries in _DEBT:\n" + "\n".join(dead)
            + "\n\nFix: delete them. Someone did the work; the list should show "
              "what is still outstanding."
        )
