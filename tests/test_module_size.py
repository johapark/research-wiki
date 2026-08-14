"""Enforce a per-module line limit so files stay legible to agents.

A module an agent cannot hold in context is a module an agent edits blind. The
limit is a legibility budget, not a style preference, which is why it is a test
rather than a lint rule: it fails the build the moment a new file crosses it.

**The grandfather list is a ratchet, not an amnesty.** Existing debt is exempted
*at its current size* — `path -> ceiling` — so a listed module may shrink freely
but cannot grow. A bare set (the shape this pattern is usually written with)
exempts a file forever and is how `agents/runner.py` reached 1214 lines without
anything objecting. Recording the count means the next 50 lines have to argue
for themselves.

Adapted from OpenKB's `tests/test_file_size.py` (Apache 2.0), which supplied the
limit, the remediation-in-the-failure-message idea, and the tests-for-the-
detector discipline. The dict-shaped ratchet and the stale-entry check are ours.
"""

from __future__ import annotations

from pathlib import Path

import researchwiki

LIMIT = 800

# Resolve the package from the imported module rather than by path math off
# `__file__`, so moving this test file can never make the gate silently vacuous.
_PKG = Path(researchwiki.__file__).resolve().parent
_REPO_ROOT = _PKG.parent

# Grandfathered debt: posix path relative to the repo root -> line ceiling,
# which is the module's size when the gate was introduced (2026-08-14).
#
# Adding an entry is a deliberate act — it records that a module is too long and
# nobody split it today. Raising a ceiling is the same act again. The direction
# of travel is down; when a module drops under LIMIT, delete its entry (
# `test_no_stale_grandfather_entries` insists on it).
_GRANDFATHERED: dict[str, int] = {
    # Agent-ingest orchestrator: phase sequencing, retry/DEBUG loop, sandbox vs
    # promote routing. Split by phase boundary.
    "researchwiki/agents/runner.py": 1214,
    # Metadata reconcile phase: PDF-side extraction vs provider records, plus
    # the sanity gate. Split the gate out from the reconciliation.
    "researchwiki/agents/phases/reconcile.py": 1131,
    # Concept-candidate detection: term mining, scoring, triage labelling.
    # Mining and labelling are separable.
    "researchwiki/concepts/candidates.py": 1074,
    # Transactional promote: five journalled steps plus rollback. Steps could
    # each be a unit under a thin coordinator.
    "researchwiki/agents/promote.py": 920,
    # Retrieval benchmark: fixture loading, scoring, reporting in one file.
    "researchwiki/benchmark/retrieval.py": 909,
    # Provider client: request shaping, cache_control placement, retry, and the
    # per-provider quirks. Split per provider family.
    "researchwiki/agents/llm.py": 902,
    # Backfill targets (hook / keywords / doi) share only a work-list idiom;
    # one module per target is the natural shape.
    "researchwiki/tasks/backfill.py": 854,
    # Memory-evolution phase: candidate selection, proposal drafting, emit.
    "researchwiki/agents/phases/evolution.py": 843,
    # `researchwiki/tasks/lint/__init__.py` was here at 831. Its two emitters
    # moved to `lint/report.py` (2026-08-14) and the dispatcher is now 261 lines,
    # so the entry is gone rather than commented out — the list is the remaining
    # debt, not a changelog.
}


def _line_count(path: Path) -> int:
    """Physical line count.

    `splitlines()` on bytes handles `\\n`, `\\r\\n` and bare `\\r` alike, so an
    unusual line-ending style cannot under-count and slip a long file past the
    gate. Reading bytes also avoids a decode error on a stray non-UTF-8 byte
    turning a size check into a crash.
    """
    return len(path.read_bytes().splitlines())


def _py_files(pkg: Path) -> list[Path]:
    return [p for p in sorted(pkg.rglob("*.py")) if "__pycache__" not in p.parts]


def _violations(
    root: Path, pkg: Path, limit: int, grandfathered: dict[str, int]
) -> tuple[list[tuple[str, int]], list[tuple[str, int, int]]]:
    """Split every module into the two ways it can fail the gate.

    Returns `(over_limit, grown)`:
      - `over_limit`: not grandfathered and at/over `limit` — `(rel, n)`.
      - `grown`: grandfathered but now larger than its recorded ceiling —
        `(rel, n, ceiling)`.

    Two lists rather than one because the remediation differs: a new offender
    should be split, whereas a grown one has already been argued about and the
    question is only whether the growth was worth it.
    """
    over: list[tuple[str, int]] = []
    grown: list[tuple[str, int, int]] = []
    for path in _py_files(pkg):
        rel = path.relative_to(root).as_posix()
        n = _line_count(path)
        ceiling = grandfathered.get(rel)
        if ceiling is not None:
            if n > ceiling:
                grown.append((rel, n, ceiling))
            continue
        if n >= limit:
            over.append((rel, n))
    return over, grown


# --------------------------------------------------------------------------
# Detector tests — the gate is only as trustworthy as its own coverage.
# --------------------------------------------------------------------------


def test_detector_flags_oversize(tmp_path):
    (tmp_path / "big.py").write_text("x = 1\n" * 5)
    (tmp_path / "small.py").write_text("x = 1\n" * 2)
    over, grown = _violations(tmp_path, tmp_path, limit=3, grandfathered={})
    assert [name for name, _ in over] == ["big.py"]
    assert grown == []


def test_boundary_is_inclusive(tmp_path):
    # The module docstring promises files stay *under* the limit, so one sitting
    # exactly on it violates. Pinned because `>=` vs `>` is a one-character slip
    # that silently widens the budget.
    (tmp_path / "edge.py").write_text("x = 1\n" * 3)
    over, _ = _violations(tmp_path, tmp_path, limit=3, grandfathered={})
    assert [name for name, _ in over] == ["edge.py"]


def test_bare_cr_line_endings_are_counted(tmp_path):
    (tmp_path / "cr.py").write_bytes(b"x = 1\r" * 10)
    assert _line_count(tmp_path / "cr.py") == 10


def test_undecodable_bytes_do_not_crash_the_count(tmp_path):
    # A stray latin-1 byte in a source file should fail the *file*, not the
    # gate — reading bytes is what makes that true.
    (tmp_path / "odd.py").write_bytes(b"x = '\xff'\n" * 4)
    assert _line_count(tmp_path / "odd.py") == 4


def test_grandfathered_under_ceiling_is_exempt(tmp_path):
    (tmp_path / "old.py").write_text("x = 1\n" * 5)
    over, grown = _violations(tmp_path, tmp_path, limit=3, grandfathered={"old.py": 5})
    assert over == []
    assert grown == []


def test_grandfathered_may_shrink(tmp_path):
    (tmp_path / "old.py").write_text("x = 1\n" * 4)
    over, grown = _violations(tmp_path, tmp_path, limit=3, grandfathered={"old.py": 5})
    assert (over, grown) == ([], [])


def test_grandfathered_growth_is_flagged(tmp_path):
    # The whole point of the dict shape: exempt at recorded size, not forever.
    (tmp_path / "old.py").write_text("x = 1\n" * 6)
    over, grown = _violations(tmp_path, tmp_path, limit=3, grandfathered={"old.py": 5})
    assert over == []
    assert grown == [("old.py", 6, 5)]


# --------------------------------------------------------------------------
# The gate itself.
# --------------------------------------------------------------------------


def test_no_module_exceeds_limit():
    files = _py_files(_PKG)
    assert files, f"no Python files found under {_PKG} — the scan would be vacuous"

    over, grown = _violations(_REPO_ROOT, _PKG, LIMIT, _GRANDFATHERED)

    problems: list[str] = []
    if over:
        listing = "\n".join(f"  - {rel}: {n} lines" for rel, n in over)
        problems.append(
            f"These modules reach or exceed the {LIMIT}-line limit:\n{listing}\n"
            "How to fix: split cohesive groups into focused modules by "
            "responsibility. Grandfathering a NEW file is not the intended "
            "escape hatch — the list is for debt that predates this gate."
        )
    if grown:
        listing = "\n".join(
            f"  - {rel}: {n} lines, ceiling {ceiling} (+{n - ceiling})"
            for rel, n, ceiling in grown
        )
        problems.append(
            f"These grandfathered modules grew past their recorded ceiling:\n{listing}\n"
            "How to fix: move the new code into a focused sibling module, or "
            "take the opportunity to split the file. If the growth is genuinely "
            "unavoidable, raise the ceiling in _GRANDFATHERED — but that is a "
            "decision to record, not a formality: the direction of travel is down."
        )
    if problems:
        raise AssertionError("\n\n".join(problems))


def test_no_stale_grandfather_entries():
    """A ratchet that never releases is just a list of excuses.

    Two ways an entry goes stale: the module was split below `LIMIT` (so the
    exemption is no longer doing anything), or it was renamed/deleted (so the
    entry silently protects nothing). Both should be pruned, and neither is
    visible from `test_no_module_exceeds_limit` — passing the gate is exactly
    what a stale entry looks like.
    """
    stale: list[str] = []
    for rel, ceiling in sorted(_GRANDFATHERED.items()):
        path = _REPO_ROOT / rel
        if not path.exists():
            stale.append(f"  - {rel}: no such file (renamed or deleted?)")
            continue
        n = _line_count(path)
        if n < LIMIT:
            stale.append(
                f"  - {rel}: now {n} lines, under the {LIMIT} limit "
                f"(ceiling {ceiling}) — exemption no longer needed"
            )
    if stale:
        listing = "\n".join(stale)
        raise AssertionError(
            "Stale _GRANDFATHERED entries in this test:\n"
            f"{listing}\n\n"
            "How to fix: delete the entry. Someone did the work — the list "
            "should show what is left, not what used to be true."
        )
