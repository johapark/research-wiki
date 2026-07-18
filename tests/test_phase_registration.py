"""Every `phase=` literal must resolve, and config errors must not be swallowed.

An unregistered phase raises from `for_phase`. `run_llm_judge` returns None on
any exception so callers can treat "no verdict" uniformly — which used to
swallow that raise, turning a whole feature into a silent no-op indistinguishable
from "the judge had no opinion". Two guards:

  1. a source scan asserting every `phase="..."` literal in the package is
     registered, so the bug can't ship at all;
  2. behavioural tests that `PhaseNotRegistered` propagates through
     `run_llm_judge` while ordinary runtime failures still yield None.

Note `judge_phase=` is a *different* parameter — a provenance column on the
claim-graph `Edge` recording which judge produced an edge. It never reaches
`for_phase`, so the scan must not match it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import researchwiki
from researchwiki.agents import judge, llm
from researchwiki.agents.model_config import PhaseNotRegistered, for_phase, list_phases

# `phase=` not preceded by an identifier character — excludes `judge_phase=`.
_PHASE_LITERAL = re.compile(r"""(?<![A-Za-z0-9_])phase=\s*["']([a-z_]+)["']""")

_PKG_ROOT = Path(researchwiki.__file__).parent


def _phase_literals_in_source() -> dict[str, list[str]]:
    """Map each `phase=` literal in the package to the files that use it."""
    found: dict[str, list[str]] = {}
    for py in _PKG_ROOT.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        for name in _PHASE_LITERAL.findall(text):
            found.setdefault(name, []).append(str(py.relative_to(_PKG_ROOT)))
    return found


def test_every_phase_literal_is_registered():
    used = _phase_literals_in_source()
    assert used, "scan found no phase= literals — the regex probably broke"

    registered = set(list_phases())
    missing = {n: sorted(set(f)) for n, f in used.items() if n not in registered}
    assert not missing, (
        "unregistered phase(s) — add them to _FALLBACK_PHASES in "
        f"agents/model_config.py: {missing}"
    )


def test_every_phase_literal_actually_resolves():
    """Registration isn't enough — the bound role must exist too."""
    for name in _phase_literals_in_source():
        for_phase(name)  # raises PhaseNotRegistered if the role is missing


def test_judge_phase_provenance_is_not_mistaken_for_a_phase():
    """`judge_phase=` is an Edge provenance column, not a model-config phase."""
    assert _PHASE_LITERAL.findall('judge_phase="concepts_detector",') == []
    assert _PHASE_LITERAL.findall('phase="critic",') == ["critic"]


def test_unknown_phase_raises_phase_not_registered():
    with pytest.raises(PhaseNotRegistered):
        for_phase("definitely_not_a_registered_phase")
    # subclasses KeyError, so pre-existing handlers keep working
    assert issubclass(PhaseNotRegistered, KeyError)


def test_run_llm_judge_propagates_phase_not_registered(monkeypatch):
    def boom(**kw):
        raise PhaseNotRegistered("phase 'nope' not in model config")

    monkeypatch.setattr(llm, "call", boom)
    with pytest.raises(PhaseNotRegistered):
        judge.run_llm_judge(phase="nope", system="s", prompt="p")


def test_run_llm_judge_still_tolerates_runtime_failures(monkeypatch):
    """Transient failures must keep returning None — that tolerance is intended."""
    def boom(**kw):
        raise RuntimeError("server unreachable")

    monkeypatch.setattr(llm, "call", boom)
    assert judge.run_llm_judge(phase="critic", system="s", prompt="p") is None
