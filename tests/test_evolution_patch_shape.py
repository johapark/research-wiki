"""`evolution._coerce_patch` — the judge's `patch` field is untrusted input.

Observed 2026-08-04 batch-ingesting a Molecular Biology and Evolution paper:
the judge returned `patch` as a LIST. `_EVOLUTION_SCHEMA` declares it an object
but is only honored by chat-relay, so nothing enforced it; the old
`parsed.get("patch") or {}` let a non-empty list through (truthy) into a field
annotated `dict`, and `render_proposal_md` died on `.get()`.

The failure mode is what makes this worth pinning: the crash landed in
`_phase_commit` *after* the page was promoted, so the batch reported
`FAIL rc=3` for an ingest that had actually written a correct page. A
misreported failure is worse than a real one — it makes you distrust the whole
batch.
"""
from __future__ import annotations

import pytest

from researchwiki.agents.phases.evolution import _coerce_patch, render_proposal_md


@pytest.mark.parametrize("raw", [
    None, {}, [], "text", 0, [{"a": 1}, {"b": 2}], ["str"], [[{"a": 1}]],
])
def test_unusable_shapes_become_empty_dict(raw):
    """Every rejection still returns a dict, so `.get()` downstream is safe."""
    out = _coerce_patch(raw)
    assert out == {} and isinstance(out, dict)


def test_dict_passes_through_unchanged():
    p = {"add_bullet_under": "## Results", "bullet_text": "[[c/s]] — x"}
    assert _coerce_patch(p) is p


def test_single_element_list_is_unwrapped():
    """The common LLM slip — `[{...}]` for `{...}` — is recovered, not dropped."""
    p = {"add_bullet_under": "## Results"}
    assert _coerce_patch([p]) == p


class _Prop:
    """Minimal stand-in for EvolutionProposal's render surface."""
    def __init__(self, patch, verdict="refine"):
        self.source_key, self.target_key = "compbio/new-2026-a", "synthesis/t"
        self.verdict, self.confidence = verdict, 0.9
        self.rationale, self.claim_ids = "r", []
        self.patch = patch


@pytest.mark.parametrize("verdict", ["refine", "enhance", "contrast"])
def test_render_survives_an_empty_patch(verdict):
    """Belt and braces: even if an empty patch reaches render, it must not raise.

    The caller drops non-"none" verdicts with an unusable patch, so this should
    be unreachable — but `render_proposal_md` has ~12 `.get()` calls and a
    future caller shouldn't be able to crash a committed ingest.
    """
    out = render_proposal_md(_Prop({}, verdict))
    assert isinstance(out, str) and verdict.upper() in out


def test_render_would_have_crashed_on_a_list():
    """Guards the actual regression: a list must never reach render as-is."""
    with pytest.raises(AttributeError):
        render_proposal_md(_Prop([{"add_bullet_under": "## Results"}]))
    # ...which is exactly why the coercion happens before construction.
    assert isinstance(_coerce_patch([{"add_bullet_under": "## Results"}]), dict)
