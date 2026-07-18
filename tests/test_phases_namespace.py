"""Pin which `phases.X` names are functions and which are modules.

Six phase functions share a name with the module defining them, so the
re-export in `phases/__init__.py` shadows the submodule: `phases.commit` is the
function. The split is not uniform — `phases.draft` *is* a module — and that
unpredictability is the actual trap (it produces a confusing AttributeError
when you monkeypatch what you assumed was a module).

These tests make the map an explicit contract rather than an accident, so
adding or dropping a re-export is a deliberate, reviewed change.
"""
from __future__ import annotations

import importlib
import types

from researchwiki.agents import phases

# Phase functions that deliberately shadow their same-named submodule.
SHADOWED_BY_FUNCTION = {
    "commit",
    "extract",
    "grade",
    "grade_persist",
    "memory_evolve",
    "reconcile",
}

# Submodules with no same-named re-export, so the module stays reachable.
REACHABLE_AS_MODULE = {
    "crosslinks",
    "draft",
    "evolution",
    "evolve_ledger",
    "revise",
    "target_claims",
}


def _from_package_import(name: str):
    """Emulate `from researchwiki.agents.phases import <name>`.

    Mirrors Python's actual semantics: attribute first, then fall back to
    importing the submodule. The fallback matters — a submodule nothing
    imports at package-init (e.g. `target_claims`) is not an attribute yet,
    but `from ... import` still resolves it to the module. Testing via bare
    getattr would make this contract falsely import-order dependent.
    """
    try:
        return getattr(phases, name)
    except AttributeError:
        return importlib.import_module(f"researchwiki.agents.phases.{name}")


def test_shadowed_names_resolve_to_functions():
    for name in sorted(SHADOWED_BY_FUNCTION):
        attr = _from_package_import(name)
        assert not isinstance(attr, types.ModuleType), (
            f"phases.{name} is now a module — the re-export was dropped. "
            "Update SHADOWED_BY_FUNCTION and the __init__ docstring."
        )
        assert callable(attr), f"phases.{name} should be callable"


def test_unshadowed_names_resolve_to_modules():
    for name in sorted(REACHABLE_AS_MODULE):
        attr = _from_package_import(name)
        assert isinstance(attr, types.ModuleType), (
            f"phases.{name} is no longer a module — a same-named symbol is now "
            "re-exported. Update REACHABLE_AS_MODULE and the __init__ docstring."
        )


def test_shadowed_modules_are_still_importable_by_full_path():
    """The documented escape hatch must keep working."""
    for name in sorted(SHADOWED_BY_FUNCTION):
        mod = importlib.import_module(f"researchwiki.agents.phases.{name}")
        assert isinstance(mod, types.ModuleType)
        assert mod.__name__ == f"researchwiki.agents.phases.{name}"


def test_map_covers_every_submodule():
    """No submodule may be silently omitted from the contract above."""
    pkg_dir = importlib.import_module("researchwiki.agents.phases").__path__[0]
    from pathlib import Path

    on_disk = {
        p.stem
        for p in Path(pkg_dir).glob("*.py")
        if p.stem != "__init__"
    }
    documented = SHADOWED_BY_FUNCTION | REACHABLE_AS_MODULE
    assert on_disk == documented, (
        "phases submodules changed — reconcile the namespace contract. "
        f"on disk but undocumented: {sorted(on_disk - documented)}; "
        f"documented but missing: {sorted(documented - on_disk)}"
    )
