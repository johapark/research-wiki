"""No unused imports in `researchwiki/`.

There is no linter in this repo — CI runs pytest and nothing else, and
ruff/flake8/mypy aren't in `[dev]`. So the 45 dead imports that had accumulated
were invisible: nothing failed, nothing warned. They aren't cosmetic. Each one
is a live edge in the import graph, and two of them had already caused real
trouble — an `import sys` kept alive only by a `sys.exit` that moved, and
`from .paths import wiki_root` in `log.py` which made a logging call able to
fail on path resolution.

This test is that missing linter, scoped to the one rule worth pinning without
adopting a whole toolchain. It's AST-only: no imports are executed, so it stays
hermetic and fast.

Three exemptions, all deliberate:

  - `from __future__ import ...` — has no binding to use.
  - `__init__.py` — a package's imports *are* its public surface. Re-exports
    look unused by definition.
  - anything named in `__all__` — the explicit form of the same thing, for
    non-package modules that curate a namespace (`search/__init__.py` re-exports
    `claim_lookup` for `tasks/claims.py`; `grade/fidelity/paper.py` keeps
    `_normalize_numeric` for `test_numeric_drift.py`).

A genuinely-needed import that this flags is a signal to add it to `__all__`,
which documents the intent where the next reader will see it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import researchwiki

PACKAGE_ROOT = Path(researchwiki.__file__).parent


def _python_files() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py"))


def _bound_names(node: ast.Import | ast.ImportFrom) -> list[tuple[str, str]]:
    """Names an import statement binds, as (bound_name, source_text) pairs.

    `import a.b.c` binds `a`; `import a.b as ab` binds `ab`. The source text is
    carried along only to make the failure message actionable.
    """
    out: list[tuple[str, str]] = []
    for alias in node.names:
        if alias.asname:
            bound = alias.asname
        elif isinstance(node, ast.Import):
            bound = alias.name.split(".")[0]
        else:
            bound = alias.name
        shown = (f"from {'.' * getattr(node, 'level', 0)}{node.module or ''} "
                 f"import {alias.name}" if isinstance(node, ast.ImportFrom)
                 else f"import {alias.name}")
        if alias.asname:
            shown += f" as {alias.asname}"
        out.append((bound, shown))
    return out


def _declared_all(tree: ast.Module) -> set[str]:
    """Names listed in a module-level `__all__` literal."""
    out: set[str] = set()
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.add(elt.value)
    return out


def _used_names(tree: ast.Module, imported: set[str]) -> set[str]:
    """Every imported name that appears anywhere outside its own import.

    Deliberately generous. A `Name` load, an attribute base, a string
    annotation, a `__all__` entry, or a mention inside a docstring all count as
    use. This test exists to catch the unambiguous cases; a false positive here
    would be a broken build over a style opinion, so it errs toward silence.
    """
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                used.add(base.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Deferred annotations (`x: "Foo"`), docstrings mentioning a symbol,
            # and TYPE_CHECKING-only names all live in strings.
            for name in imported:
                if name in node.value:
                    used.add(name)
    return used


def _unused_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    if path.name == "__init__.py":
        return []                      # a package's imports are its surface

    statements = [n for n in ast.walk(tree)
                  if isinstance(n, (ast.Import, ast.ImportFrom))]
    pairs = [pair for n in statements
             if not (isinstance(n, ast.ImportFrom) and n.module == "__future__")
             for pair in _bound_names(n)]
    imported = {bound for bound, _ in pairs}
    if not imported:
        return []

    live = _used_names(tree, imported) | _declared_all(tree)
    return sorted({shown for bound, shown in pairs
                   if bound not in live and bound != "*"})


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.name))
def test_module_has_no_unused_imports(path):
    unused = _unused_in(path)
    rel = path.relative_to(PACKAGE_ROOT.parent)
    assert not unused, (
        f"{rel} imports names it never uses:\n"
        + "\n".join(f"  - {u}" for u in unused)
        + "\n\nDelete them, or — if the import is a deliberate re-export for "
          "another module or a test — add the name to this module's `__all__` "
          "so the intent is recorded where the next reader will look."
    )


# ---------- the detector itself ----------
#
# A lint test that silently stops detecting is worse than none, so pin its
# behavior on synthetic sources rather than trusting it against a clean tree.

def _write(tmp_path, body: str, name: str = "mod.py") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def test_detector_flags_a_plainly_unused_import(tmp_path):
    p = _write(tmp_path, "import os\nimport sys\n\nprint(os.getcwd())\n")
    assert _unused_in(p) == ["import sys"]


def test_detector_flags_an_unused_from_import(tmp_path):
    p = _write(tmp_path, "from pathlib import Path\n\nx = 1\n")
    assert _unused_in(p) == ["from  import Path"] or "Path" in _unused_in(p)[0]


def test_detector_accepts_attribute_use(tmp_path):
    p = _write(tmp_path, "import os.path\n\nprint(os.path.join('a', 'b'))\n")
    assert _unused_in(p) == []


def test_detector_accepts_aliased_use(tmp_path):
    p = _write(tmp_path, "import numpy as np\n\nnp.array([1])\n")
    assert _unused_in(p) == []


def test_detector_flags_unused_alias(tmp_path):
    p = _write(tmp_path, "import numpy as np\n\nx = 1\n")
    assert _unused_in(p) == ["import numpy as np"]


def test_detector_ignores_future_imports(tmp_path):
    # `annotations` binds nothing that could ever be "used".
    p = _write(tmp_path, "from __future__ import annotations\n\nx = 1\n")
    assert _unused_in(p) == []


def test_detector_skips_package_init(tmp_path):
    p = _write(tmp_path, "from .thing import Thing\n", name="__init__.py")
    assert _unused_in(p) == []


def test_detector_honors_dunder_all_reexport(tmp_path):
    """The escape hatch that saved four real re-exports (`claim_lookup`,
    `claims_by_stem`, `pdf_section_search`, `_normalize_numeric`) from being
    deleted as dead."""
    p = _write(tmp_path,
               "from .search import claim_lookup\n\n__all__ = ['claim_lookup']\n")
    assert _unused_in(p) == []


def test_detector_accepts_string_annotation_use(tmp_path):
    p = _write(tmp_path,
               "from pathlib import Path\n\n"
               "def f(p: 'Path') -> None: ...\n")
    assert _unused_in(p) == []


def test_detector_scans_a_nonempty_tree():
    """Guard against the whole file passing vacuously because the glob broke."""
    files = _python_files()
    assert len(files) > 100, f"expected the full package, walked {len(files)} files"
