"""The ML stack must stay lazily imported — no module-scope `import torch`.

`torch`, `transformers` and `sentence-transformers` are core dependencies on
purpose (the auto-promote gate needs the grader), but every import of them is
function-local. That one property is what makes three things true at once: the
unit suite runs without them installed, `--no-semantic` degrades to BM25 instead
of failing at import time, and a lean install works for the MCP server and the
offline commands. A single hoisted import ends all three — silently, because a
developer machine has them installed.

Checked by reading the source rather than by importing it, so the test means the
same thing whether or not the ML stack is present.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "researchwiki"
HEAVY = {"torch", "transformers", "sentence_transformers"}


def _module_scope_imports(source: str) -> set[str]:
    """Top-level packages imported when the module is imported.

    Only `tree.body` — imports nested in a function, a class, or an
    `if TYPE_CHECKING:` block don't run at import time, which is precisely what
    this invariant permits.
    """
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_ml_stack_is_imported_lazily():
    files = sorted(PACKAGE.rglob("*.py"))
    assert files, f"no sources found under {PACKAGE}"
    offenders = [
        f"{path.relative_to(PACKAGE.parent)}: {name}"
        for path in files
        for name in sorted(_module_scope_imports(path.read_text(encoding="utf-8")) & HEAVY)
    ]
    assert not offenders, "module-scope ML imports:\n  " + "\n  ".join(offenders)
