"""Revision prompts preserve the whole draft and retain source fallback."""

from types import SimpleNamespace

import pytest

from researchwiki.agents.phases import revise
from researchwiki.agents.phases.draft import Draft
from researchwiki.agents.phases.revise import (
    _build_debug_prompt,
    _build_evolve_prompt,
)


def _long_inputs():
    prior = "## Summary\n" + ("draft body " * 500) + "\n## Limitations\nKEEP_THIS_CAVEAT"
    full_text = ("paper source " * 3000) + "ONLY_SOURCE_EVIDENCE"
    sections = {"introduction": "introduction only"}
    return prior, full_text, sections


def test_evolve_prompt_preserves_late_draft_and_full_text_fallback():
    prior, full_text, sections = _long_inputs()
    prompt = _build_evolve_prompt(
        prior, "repair one claim", {}, sections, pdf_full_text=full_text,
    )
    assert "KEEP_THIS_CAVEAT" in prompt
    assert "ONLY_SOURCE_EVIDENCE" in prompt
    assert "document-stratified fallback" in prompt


def test_debug_prompt_preserves_late_draft_and_full_text_fallback():
    prior, full_text, sections = _long_inputs()
    prompt = _build_debug_prompt(
        prior_draft=prior,
        issues=["n_graded"],
        gate_reasons=["only 3 graded claims"],
        drift_details=[],
        sections=sections,
        pdf_full_text=full_text,
    )
    assert "KEEP_THIS_CAVEAT" in prompt
    assert "ONLY_SOURCE_EVIDENCE" in prompt
    assert "document-stratified fallback" in prompt


@pytest.mark.parametrize("operator", ["evolve", "debug"])
def test_repair_preserves_custom_system_and_full_page(monkeypatch, tmp_path, operator):
    custom = "Preserve this custom priority.\n" * 1500 + "FINAL_CUSTOM_INSTRUCTION\n"
    path = tmp_path / "author.md"
    path.write_text(custom)
    prior = "## Summary\nFinding.\n## Limitations\nCaveat.\n## Custom Appendix\nKeep me."
    draft = Draft(text=prior, model="m", temperature=0, input_tokens=0, output_tokens=0)
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=prior, model="m", temperature=0,
                               input_tokens=1, output_tokens=1)

    monkeypatch.setattr(revise.llm, "call", call)
    kwargs = dict(draft=draft, metadata={}, sections={}, system_prompt_override=str(path))
    if operator == "evolve":
        revise.evolve(**kwargs, critique=revise.CritiqueOutput(
            "Fix one claim", [], "m", 0, 0,
        ))
    else:
        revise.debug(**kwargs, issues=["n_graded"], gate_reasons=[])
    assert len(calls) == 1
    assert calls[0]["system"] == custom
    assert prior in calls[0]["prompt"]
    assert "all existing sections" in calls[0]["prompt"]


def test_clean_draft_skips_model_and_retrieval(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a clean draft must not call a provider or retrieve evidence")

    monkeypatch.setattr(revise.llm, "call", forbidden)
    from researchwiki.index import pdf_chunks
    monkeypatch.setattr(pdf_chunks, "query_pdf", forbidden)
    draft = Draft(text="clean", model="m", temperature=0, input_tokens=0, output_tokens=0)
    critique = revise.critic(draft=draft, metadata={})
    assert not critique.weak_claims and not critique.coverage_gaps
    assert revise.debug(draft=draft, issues=[], gate_reasons=[], metadata={}, sections={}).text == "clean"
