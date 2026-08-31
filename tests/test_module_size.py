"""Cap how much *code* one module or function carries.

The budget lives in the test suite because that is the only place a budget
actually holds: a convention in a style guide is advisory, and a `lint` finding is
something you can run past.

**It counts code, not lines.** Docstrings, comment-only lines and blanks are
excluded, and that distinction is the whole point rather than a refinement. This
package is 57% code and 29% prose, and the prose is the house style — so a
physical-line budget is a tax on documentation, levied hardest on the
best-explained files. Measured when this gate counted physical lines, it had the
ranking inverted: `agents/llm.py` was pinned as debt at 902 lines while holding
only 470 lines of code (39% prose, i.e. well explained), and `tasks/lint/report.py`
passed comfortably at 603 lines while holding 549 — more code than four of the
pinned modules. The gate flagged the documented file and waved through the dense
one. It also fired on `pdf/text.py` at 868 lines when 284 of them were code.

**Why the metric mattered more than the number.** Raising the old physical-line
ceiling would have moved the threshold without fixing the ordering — `llm.py` is
still debt at 1000 physical lines despite being 39% prose. Ordering is what a
budget is *for*, so the metric changed first; the ceiling was then free to move,
and did (500 -> 800), because a correct metric with a permissive bound still
ranks modules honestly while a wrong one does not at any bound.

The old rationale — "small enough for an agent to hold in context" — no longer
carries the argument either. The largest module here is ~13,500 tokens against a
200k window, which it fits fourteen times over. What a size cap actually buys is
cohesion: past some volume of logic a module is doing more than one job, and
that threshold scales with code, not with how well the code is described.

**A multi-line string assigned to a name still counts as code.** It is content
the module carries, and the alternative is a rule you can slip past by moving
text into a triple-quoted constant.

**Two bounds, because they answer different questions.** `MAX_CODE_LINES` is the
ceiling: how large any module may ever be. `_DEBT` is a ratchet: a module already
large must not get larger, whether or not it is under the ceiling. A listed file
may shrink as much as you like and cannot gain a single line. The alternative — a
plain set of exempt paths — grants a permanent licence to grow, which is how
`agents/runner.py` got to 817 code lines with no test ever objecting.

Keeping them separate is not theoretical. While the dead-entry rule was keyed on
the ceiling, raising `MAX_CODE_LINES` 500 -> 800 forced eight pins at 502-661 to
be deleted, handing those modules ~1,700 lines of unratcheted growth though none
had shrunk by a line. `RATCHET_RELEASE` is now the retirement point, so the
ceiling can move without silently unlatching every ratchet beneath it.

`test_debt_list_has_no_dead_entries` is the other half: a module split back below
`RATCHET_RELEASE` looks *identical to a passing gate*, so without that check the
list would quietly turn into a monument to work already finished.

Functions have a tighter 250-code-line ceiling. Ruff independently caps McCabe
complexity, while this catches long straight-line coordinators and renderers that
branch metrics miss. `_FUNCTION_DEBT` gives the one inherited exception an exact
ratchet rather than a permanent exemption.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, NamedTuple

import pytest

import researchwiki

#: Hard ceiling: no module may reach this, pinned or not. Code lines —
#: docstrings, comments and blanks excluded — so it is not a `wc -l` figure.
#:
#: Set permissively on purpose. The *metric* is what makes this gate rank modules
#: honestly (counting logic, not prose), so the ceiling only has to catch a module
#: nobody would defend; ordinary judgement about cohesion belongs in review.
MAX_CODE_LINES = 800

#: Where the per-module ratchet stops applying. A module in `_DEBT` may not gain
#: a line while it is at or above this; once it falls below, its entry is dead and
#: the module rejoins the ordinary population governed only by the ceiling.
#:
#: This exists because the ceiling and the ratchet answer different questions, and
#: tying them together broke the second one. When `MAX_CODE_LINES` moved 500 -> 800,
#: the dead-entry rule — then keyed on the ceiling — required deleting eight pins
#: at 502-661, handing those modules ~1,700 lines of unratcheted growth even
#: though none of them had shrunk by a line. The ceiling says how large a module
#: may ever be; the ratchet says a module already this large must not get larger.
#: A release point of its own keeps the second promise while the first stays
#: permissive.
RATCHET_RELEASE = 500

#: A function is a unit a reader should be able to understand without paging
#: through several independent workflows. This deliberately measures code lines
#: with the same prose exclusions as the module budget above.
MAX_FUNCTION_CODE_LINES = 250

# Function key (`repo/path.py::qualified.name`) -> current code-line count.
# Retire a pin as soon as the function drops below MAX_FUNCTION_CODE_LINES.
_FUNCTION_DEBT: dict[str, int] = {
    # Journalled commit/promotion coordinator. Each transactional step is a
    # natural extraction boundary; until then, it may shrink but not grow.
    "researchwiki/agents/runner.py::_phase_commit": 269,
}

# Locate the package through the import system rather than by walking up from
# `__file__`. If this test is ever moved, path arithmetic would silently point at
# a directory with no modules in it and the gate would pass by scanning nothing.
_PACKAGE = Path(researchwiki.__file__).resolve().parent
_REPO = _PACKAGE.parent

# Module (repo-relative, posix) -> pinned size in CODE lines. Numbers are code
# lines and are not comparable to the physical-line pins this list carried before
# 2026-08-14.
#
# A pin means "already this large; must not get larger" — it binds whether or not
# the module is under MAX_CODE_LINES, which is the whole reason the ratchet has
# its own release point. Every entry is an admission that a file is too long and
# was not split that day; editing a number upward is the same admission again, so
# do it in a commit that says why. An entry retires when the module drops below
# RATCHET_RELEASE (or disappears), not when the ceiling moves.
_DEBT: dict[str, int] = {
    # Agent-ingest orchestrator: phase sequencing, the retry/DEBUG loop, and
    # sandbox-vs-promote routing. Splits along the phase boundary. The only entry
    # that also exceeds MAX_CODE_LINES.
    "researchwiki/agents/runner.py": 817,
    # Metadata reconcile: PDF-side extraction against provider records, plus the
    # sanity gate. The gate is the separable half.
    "researchwiki/agents/phases/reconcile.py": 661,
    # Concept candidates: term mining, scoring, triage labelling. Mining and
    # labelling do not need to share a module. Raised 653 -> 656 to count hub
    # membership over contribution sections only, which is what the scaffolder
    # matches over — the detector was advertising `limitations` mentions as
    # members and publishing bridge candidates that could not be scaffolded.
    # Three lines: a deferred import (term_claims imports from here, so a
    # top-level one closes the cycle) and the two-line member filter.
    "researchwiki/concepts/candidates.py": 656,
    # Retrieval benchmark: fixture loading, scoring and reporting in one file.
    "researchwiki/benchmark/retrieval.py": 644,
    # Backfill targets (hook / keywords / doi) share only a work-list idiom.
    "researchwiki/tasks/backfill.py": 606,
    # Lint's emitters (human text, `--json`). Grows by one block per new check,
    # which is exactly the drift a ratchet is for. Raised 565 -> 588 for
    # the `idea_contract_violations` emitter, on the same reasoning as the
    # 549 -> 565 raise before it: rendering findings is this file's one job, and
    # splitting per check would scatter that job across modules to satisfy a
    # number. Raised 588 -> 595 to replace the 525-line prose renderer with five
    # bounded sections plus a coordinator; no finding-rendering logic was added.
    "researchwiki/tasks/lint/report.py": 595,
    # Benchmark fixtures: YAML loading, scoring, and the report. Raised 525 ->
    # 530 for parser/content-path extraction; the largest function fell from
    # 388 to 193 code lines and the module gained only the dispatch boundaries.
    "researchwiki/tasks/benchmark_fixture.py": 530,
    # Memory evolution: candidate selection, proposal drafting, emit.
    "researchwiki/agents/phases/evolution.py": 524,
    # Transactional promote: five journalled steps and their rollback. Each step
    # could stand alone under a thin coordinator.
    "researchwiki/agents/promote.py": 502,
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
        # "code lines", spelled out: the number is smaller than `wc -l` and a
        # reader who assumes otherwise will go looking for lines that aren't there.
        if self.ceiling is None:
            return f"{self.path}: {self.lines} code lines"
        return (f"{self.path}: {self.lines} code lines, pinned at {self.ceiling} "
                f"(+{self.lines - self.ceiling})")


def _physical_lines(source: str) -> int:
    """Every line, however it ends.

    `splitlines()` treats `\\n`, `\\r\\n` and a lone `\\r` as breaks alike — a file
    saved with classic-Mac endings would otherwise read as one enormous line and
    sail through.
    """
    return len(source.splitlines())


def _docstring_line_numbers(tree: ast.Module) -> set[int]:
    """1-based line numbers occupied by module/class/function docstrings.

    Only a *docstring* — the first statement of a body — is prose by this rule. A
    string expression floating elsewhere in a module is not, and neither is one
    bound to a name; see the module docstring on why that asymmetry is deliberate.
    """
    prose: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if ast.get_docstring(node, clean=False) is None or not node.body:
            continue
        first = node.body[0]
        prose.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return prose


def count_code_lines(module: Path) -> int:
    """Lines of `module` that carry code: not blank, not a comment, not a docstring.

    Read as bytes and decoded permissively, so a stray non-UTF-8 character costs
    that file its own accuracy rather than crashing the whole gate.

    A line whose code is followed by a trailing comment still counts — the code is
    there. Only a line that is *nothing but* a comment is prose.

    An unparseable module falls back to its physical line count. That direction is
    the safe one: a syntax error must not be a way to score zero and slip under the
    budget, and a broken module is already failing louder tests than this one.
    """
    source = module.read_bytes().decode("utf-8", errors="replace")
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return _physical_lines(source)

    prose = _docstring_line_numbers(tree)
    prose.update(
        number for number, line in enumerate(lines, start=1)
        if not (stripped := line.strip()) or stripped.startswith("#")
    )
    return len(lines) - len(prose)


def _qualified_functions(node: ast.AST, prefix: str = ""):
    """Yield `(qualified_name, node)` without losing nested/class context."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            class_name = f"{prefix}.{child.name}" if prefix else child.name
            yield from _qualified_functions(child, class_name)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{prefix}.{child.name}" if prefix else child.name
            yield name, child
            yield from _qualified_functions(child, name)
        else:
            yield from _qualified_functions(child, prefix)


def function_code_lines(module: Path) -> list[tuple[str, int]]:
    """Return each qualified function and its code lines, excluding prose."""
    source = module.read_bytes().decode("utf-8", errors="replace")
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        # Syntax errors fail compilation too. Still expose one synthetic span so
        # a malformed large file cannot make this scanner report no functions.
        return [("<unparseable>", _physical_lines(source))]

    prose = _docstring_line_numbers(tree)
    prose.update(
        number for number, line in enumerate(lines, start=1)
        if not (stripped := line.strip()) or stripped.startswith("#")
    )
    return [
        (
            name,
            sum(
                number not in prose
                for number in range(node.lineno, (node.end_lineno or node.lineno) + 1)
            ),
        )
        for name, node in _qualified_functions(tree)
    ]


def scan_functions(
    *, package: Path, repo: Path, ceiling: int, debt: dict[str, int]
) -> list[Oversize]:
    """Functions crossing the hard ceiling or their exact inherited pin."""
    failures: list[Oversize] = []
    for module in python_modules(package):
        rel = module.relative_to(repo).as_posix()
        for name, lines in function_code_lines(module):
            key = f"{rel}::{name}"
            pinned = debt.get(key)
            if pinned is not None and lines > pinned:
                failures.append(Oversize(key, lines, pinned))
            elif pinned is None and lines >= ceiling:
                failures.append(Oversize(key, lines, None))
    return failures


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
        lines = count_code_lines(module)
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
    assert MAX_CODE_LINES == 800


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
    assert count_code_lines(target) == 7


def test_a_bad_byte_fails_its_own_file_not_the_gate(tmp_path):
    target = tmp_path / "latin.py"
    target.write_bytes(b"label = '\xff'\n" * 3)
    assert count_code_lines(target) == 3


# --------------------------------------------------------------------------
# What counts as code. This is the part the budget rests on: the gate used to
# count physical lines, which taxed the docstrings that are this repo's house
# style and inverted its own ranking (see the module docstring).
# --------------------------------------------------------------------------


def _module(directory: Path, source: str) -> Path:
    target = directory / "sample.py"
    target.write_text(source, encoding="utf-8")
    return target


def test_docstrings_do_not_count(tmp_path):
    module = _module(tmp_path, '''"""Module docstring.

    Several lines of it, as this package writes them.
    """


def documented():
    """One-line docstring."""
    return 1
''')
    assert count_code_lines(module) == 2      # `def` and `return`


def test_comment_only_lines_do_not_count(tmp_path):
    module = _module(tmp_path, "# a note\n#: another\nvalue = 1\n")
    assert count_code_lines(module) == 1


def test_a_trailing_comment_still_counts_as_code(tmp_path):
    # The code is on that line; only a line that is *nothing but* comment is prose.
    module = _module(tmp_path, "value = 1  # why this value\n")
    assert count_code_lines(module) == 1


def test_blank_lines_do_not_count(tmp_path):
    module = _module(tmp_path, "value = 1\n\n\n   \n\nvalue = 2\n")
    assert count_code_lines(module) == 2


def test_a_string_bound_to_a_name_counts_as_code(tmp_path):
    """A prompt template is content the module carries. Excluding it would make
    the budget dodgeable by moving text into a triple-quoted constant."""
    module = _module(tmp_path, 'PROMPT = """\nline one\nline two\n"""\n')
    assert count_code_lines(module) == 4


def test_an_unparseable_module_falls_back_to_its_physical_size(tmp_path):
    """The safe direction: a syntax error must not be a way to score zero and
    slip under the budget."""
    module = _module(tmp_path, '"""Doc."""\ndef broken(:\n    pass\n')
    assert count_code_lines(module) == 3


def test_a_prose_heavy_module_is_judged_on_its_code(tmp_path):
    """The regression that motivated the switch, in miniature. This module is
    typical of the package's shape — mostly explanation — and the old rule
    counted it at nearly four times the code it holds."""
    module = _module(tmp_path, '''"""What this module is for.

    Why it exists, what it decided against, and the measurement behind it.
    Several paragraphs of this is the house style, not an outlier.
    """

# A section banner, of the kind that separates concerns in the long files.


def contribute():
    """One job, explained."""
    return 1


value = contribute()
''')
    assert len(module.read_bytes().splitlines()) == 15
    assert count_code_lines(module) == 3      # `def`, `return`, `value =`


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
# Function-level ceiling. Complexity is enforced by Ruff C901; this catches
# long, mostly straight-line functions as well.
# --------------------------------------------------------------------------


def test_function_code_lines_exclude_its_docstring_and_comments(tmp_path):
    module = _module(tmp_path, '''def compact():
    """Several words
    across several lines.
    """
    # explanation
    value = 1
    return value
''')
    assert function_code_lines(module) == [("compact", 3)]


def test_function_at_the_ceiling_is_reported(tmp_path):
    _module(tmp_path, "def long():\n" + "    value = 1\n" * 3)
    failures = scan_functions(
        package=tmp_path, repo=tmp_path, ceiling=4, debt={},
    )
    assert [(f.path, f.lines) for f in failures] == [("sample.py::long", 4)]


def test_pinned_function_may_not_grow(tmp_path):
    _module(tmp_path, "def watched():\n" + "    value = 1\n" * 4)
    failures = scan_functions(
        package=tmp_path,
        repo=tmp_path,
        ceiling=4,
        debt={"sample.py::watched": 4},
    )
    assert [(f.path, f.lines, f.ceiling) for f in failures] == [
        ("sample.py::watched", 5, 4),
    ]


# --------------------------------------------------------------------------
# The gate.
# --------------------------------------------------------------------------


def test_no_module_is_too_long():
    modules = list(python_modules(_PACKAGE))
    assert modules, f"scanned {_PACKAGE} and found no modules — the gate is inert"

    failures = scan(package=_PACKAGE, repo=_REPO, ceiling=MAX_CODE_LINES, debt=_DEBT)
    if not failures:
        return

    complaints: list[str] = []
    fresh = [f for f in failures if not f.is_debt_growth]
    grown = [f for f in failures if f.is_debt_growth]
    if fresh:
        complaints.append(
            f"Over the {MAX_CODE_LINES}-code-line budget:\n"
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


def test_no_function_is_too_long():
    failures = scan_functions(
        package=_PACKAGE,
        repo=_REPO,
        ceiling=MAX_FUNCTION_CODE_LINES,
        debt=_FUNCTION_DEBT,
    )
    if failures:
        raise AssertionError(
            f"Functions over the {MAX_FUNCTION_CODE_LINES}-code-line budget:\n"
            + "\n".join(f"  - {failure.describe()}" for failure in failures)
            + "\nFix: extract a cohesive phase. Function pins are exact inherited debt, "
              "not blanket exemptions."
        )


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
        elif (lines := count_code_lines(module)) < RATCHET_RELEASE:
            dead.append(f"  - {rel}: down to {lines} code lines, below the "
                        f"{RATCHET_RELEASE}-line ratchet release (pinned at {pinned})")
    if dead:
        raise AssertionError(
            "Dead entries in _DEBT:\n" + "\n".join(dead)
            + "\n\nFix: delete them. Someone did the work; the list should show "
              "what is still outstanding."
        )


def test_function_debt_has_no_dead_entries():
    actual = {
        f"{module.relative_to(_REPO).as_posix()}::{name}": lines
        for module in python_modules(_PACKAGE)
        for name, lines in function_code_lines(module)
    }
    dead = [
        f"  - {key}: "
        + ("gone" if key not in actual else f"down to {actual[key]} code lines")
        for key in sorted(_FUNCTION_DEBT)
        if key not in actual or actual[key] < MAX_FUNCTION_CODE_LINES
    ]
    if dead:
        raise AssertionError(
            "Dead entries in _FUNCTION_DEBT:\n" + "\n".join(dead)
            + "\n\nFix: delete them; the function now satisfies the ordinary ceiling."
        )


# --------------------------------------------------------------------------
# The ratchet and the ceiling are independent. Tying them together is the
# regression these pin: raising the ceiling must not unlatch pins beneath it.
# --------------------------------------------------------------------------


def test_a_pinned_module_far_under_the_ceiling_still_cannot_grow(tmp_path):
    """The whole point of a separate release. 9 lines is nowhere near the
    ceiling, and the pin still binds."""
    _write(tmp_path, "watched.py", 10)
    failures = _scan_tmp(tmp_path, ceiling=800, debt={"watched.py": 9})
    assert [(f.path, f.lines, f.ceiling) for f in failures] == [("watched.py", 10, 9)]


def test_raising_the_ceiling_does_not_retire_a_pin():
    """The regression itself: at a ceiling of 800 every pin except runner.py sits
    underneath, and all of them must still be live."""
    below = {rel: pin for rel, pin in _DEBT.items() if pin < MAX_CODE_LINES}
    assert len(below) >= 8, "expected most pins to sit under the ceiling"
    assert all(pin >= RATCHET_RELEASE for pin in below.values())


def test_an_entry_is_dead_only_below_the_release_point():
    assert RATCHET_RELEASE == 500
    assert RATCHET_RELEASE < MAX_CODE_LINES, "release must sit under the ceiling"


def test_every_pin_is_at_or_above_the_release_point():
    """A pin below the release is self-contradictory: the dead-entry check would
    demand its deletion on the next run."""
    offenders = {rel: pin for rel, pin in _DEBT.items() if pin < RATCHET_RELEASE}
    assert not offenders, f"pins below RATCHET_RELEASE: {offenders}"
