"""No re-exported name may collide with a submodule name.

`phases/__init__.py` re-exports phase functions. If a re-exported name equals a
submodule name, the function shadows the module — so `phases.grade` would be the
function while `phases.draft` stays a module, an unpredictable split that
produces confusing AttributeErrors when monkeypatching what you assumed was a
module.

The `verb_object` naming convention (`grade.py` defines `grade_draft()`) keeps
those namespaces disjoint. This is expressed as an *invariant* rather than a
hand-maintained map of known collisions, so it holds automatically for phases
added later — there is no list anyone has to remember to update.
"""
from __future__ import annotations

import importlib
import types
from pathlib import Path

from researchwiki.agents import phases


def _submodule_names() -> set[str]:
    pkg_dir = Path(phases.__file__).parent
    return {p.stem for p in pkg_dir.glob("*.py") if p.stem != "__init__"}


def test_no_reexport_collides_with_a_submodule():
    collisions = set(phases.__all__) & _submodule_names()
    assert not collisions, (
        "these names are re-exported AND are submodule names, so the function "
        f"shadows the module: {sorted(collisions)}. Rename the function to "
        "verb_object form (e.g. grade -> grade_draft) per the convention in "
        "phases/__init__.py."
    )


def test_every_submodule_is_importable_by_full_path():
    """The invariant must not come at the cost of reaching a module."""
    for name in sorted(_submodule_names()):
        mod = importlib.import_module(f"researchwiki.agents.phases.{name}")
        assert isinstance(mod, types.ModuleType)
        assert mod.__name__ == f"researchwiki.agents.phases.{name}"


def test_every_reexport_actually_resolves():
    """`__all__` must not advertise a name the package doesn't provide."""
    missing = [n for n in phases.__all__ if not hasattr(phases, n)]
    assert not missing, f"__all__ lists names that don't exist: {missing}"
