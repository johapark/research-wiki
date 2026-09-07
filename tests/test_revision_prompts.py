"""Revision prompts preserve the whole draft and retain source fallback."""

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
