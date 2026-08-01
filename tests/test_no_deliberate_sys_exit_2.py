"""The CLI funnel's `SystemExit(2) -> 1` remap depends on nobody in the
package meaning code 2 by it.

`researchwiki.__main__.main` remaps every `SystemExit(2)` that reaches the
dispatch call to exit code 1, because argparse itself raises exactly that for
every usage error (bad flag, missing required argument, unparseable type) —
see `__main__.py`'s docstring and `tests/test_exit_codes.py
::test_argparse_usage_error_remapped_to_1`. The remap is only correct because
nothing in `researchwiki/` deliberately calls `sys.exit(2)` or raises
`SystemExit(2)` to *mean* "environment error, code 2" — if something did, the
funnel would silently downgrade that deliberate signal to "bad argv" (code 1),
which is a different message to whoever is scripting against these codes.

That absence was previously established by a one-off grep during review, with
nothing to stop it from becoming false the next time someone adds a
`sys.exit(2)` guard by analogy with argparse's own convention. This test is
that grep, made permanent: an AST scan for a literal `2` passed to `sys.exit`
or raised via `SystemExit`, across every module in the package.

Scanned with no exclusions. `_ingest_batch.py` is the one module that reaches
for bare `sys.exit` (its `_resolve_inputs` rejects a non-PDF or missing input
that way), which makes it the *most* likely place for a `sys.exit(2)` to
reappear by analogy with argparse — so it is deliberately in scope rather than
waved through. Every module must use code 1 for bad input or raise
`EnvironmentFailure` for environment errors, per `researchwiki/errors.py` and
`tests/test_environment_failure.py`.

Deliberately does NOT flag `parser.error(...)` (that's argparse's own path to
`SystemExit(2)`, which is exactly what the funnel exists to catch) or
`argparse.ArgumentParser.exit(...)` — only an explicit, literal `2` passed to
`sys.exit` or `SystemExit`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import researchwiki

PACKAGE_ROOT = Path(researchwiki.__file__).parent


def _python_files() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py"))


def _is_literal_two(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == 2 and not isinstance(node.value, bool)


def _deliberate_exit_2_calls(tree: ast.AST) -> list[int]:
    """Line numbers of `sys.exit(2)` calls or `SystemExit(2)` raises."""
    hits: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_sys_exit = (
                isinstance(func, ast.Attribute) and func.attr == "exit"
                and isinstance(func.value, ast.Name) and func.value.id == "sys"
            )
            is_system_exit = isinstance(func, ast.Name) and func.id == "SystemExit"
            if (is_sys_exit or is_system_exit) and node.args and _is_literal_two(node.args[0]):
                hits.append(node.lineno)
    return hits


def test_no_module_deliberately_exits_with_code_2():
    offenders: list[str] = []
    for f in _python_files():
        rel = f.relative_to(PACKAGE_ROOT.parent)
        tree = ast.parse(f.read_text(), filename=str(f))
        for lineno in _deliberate_exit_2_calls(tree):
            offenders.append(f"{rel}:{lineno}")

    assert not offenders, (
        "Found sys.exit(2) / SystemExit(2) outside argparse's own path:\n  "
        + "\n  ".join(offenders)
        + "\n\nThe CLI funnel (__main__.py) remaps every SystemExit(2) that "
        "reaches it to exit code 1, on the assumption that only argparse "
        "raises that code — see this test's module docstring. Use exit code "
        "1 for bad input directly, or raise researchwiki.errors."
        "EnvironmentFailure (or a subclass) for an environment error; don't "
        "sys.exit(2)."
    )


# ---------- the detector itself ----------

def test_detector_flags_sys_exit_2():
    tree = ast.parse("import sys\nsys.exit(2)\n")
    assert _deliberate_exit_2_calls(tree) == [2]


def test_detector_flags_system_exit_2():
    tree = ast.parse("raise SystemExit(2)\n")
    assert _deliberate_exit_2_calls(tree) == [1]


def test_detector_ignores_other_exit_codes():
    tree = ast.parse("import sys\nsys.exit(1)\nsys.exit(0)\nsys.exit()\n")
    assert _deliberate_exit_2_calls(tree) == []


def test_detector_ignores_parser_error():
    # parser.error(...) is argparse's own path to SystemExit(2) — exactly
    # what the funnel remap exists to handle, not a second deliberate one.
    tree = ast.parse("parser.error('bad flag')\n")
    assert _deliberate_exit_2_calls(tree) == []


def test_detector_ignores_unrelated_two_argument_calls():
    tree = ast.parse("sys.exit\nfoo.exit(2)\nSystemExit\n")
    assert _deliberate_exit_2_calls(tree) == []


def test_detector_scans_a_nonempty_tree():
    # Guards against the whole test silently passing on zero files.
    assert len(_python_files()) > 100
