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


def test_critic_scopes_omission_and_evolve_splices_only_that_section(monkeypatch):
    import json

    gap = {"id": "t1", "axis": "target_claims", "importance": "critical",
           "text": "The authors reported a second comparison showing improved accuracy across three datasets."}
    prior = ("## Summary\r\nOriginal summary.\r\n\r\n## Results\r\n\r\n"
             "Earlier result.\r\n\r\n## Custom Appendix\r\nKeep exactly.  \r\n")
    draft = Draft(prior, "m", 0, 0, 0, scores={"missed_anchors": [gap]})
    calls = []

    def call(**kwargs):
        calls.append(kwargs)
        if kwargs["phase"] == "critic":
            assert "REPAIR_SCOPE" in kwargs["prompt"]
            text = "Add the missing comparison to Results.\nREPAIR_SCOPE: Results"
        else:
            assert kwargs["phase"] == "evolve" and kwargs["schema"]["required"] == ["body"]
            assert prior in kwargs["prompt"]
            assert "FULL revised draft" not in kwargs["prompt"]
            text = json.dumps({"body": "Earlier result.\nSecond comparison."})
        return SimpleNamespace(text=text, model="m", temperature=0, input_tokens=11, output_tokens=7)

    monkeypatch.setattr(revise.llm, "call", call)
    critique = revise.critic(draft=draft, metadata={})
    assert critique.repair_section == "Results"
    assert "REPAIR_SCOPE" not in critique.notes
    out = revise.evolve(draft=draft, critique=critique, metadata={}, sections={})
    assert out.text == prior.replace("Earlier result.", "Earlier result.\r\nSecond comparison.")
    assert out.output_tokens == 7 and len(calls) == 2
    assert out.evidence_metrics["revision_scope"] == "section"
    assert out.evidence_metrics["section_response_valid"] is True


@pytest.mark.parametrize("reply", [
    '{"body": "incomplete', '{"body": ""}', '{"body": 42}',
    '{"body": "Repaired.\\n## Replacement appendix\\nOther edits."}',
    '{"body": "Open block\\n```python"}', 'A full page with no JSON envelope.',
])
def test_invalid_scoped_response_preserves_parent_draft(monkeypatch, reply):
    prior = "## Summary\nPreserve.\n## Results\nOld result."
    draft = Draft(prior, "m", 0, 0, 0)
    critique = revise.CritiqueOutput("repair", [SimpleNamespace(section="results", text="Old result.", position=0)],
                                     "m", 0, 0)
    monkeypatch.setattr(revise.llm, "call", lambda **kw:
                        SimpleNamespace(text=reply, model="m", temperature=.3, input_tokens=8, output_tokens=4))
    out = revise.evolve(draft=draft, critique=critique, metadata={}, sections={})
    assert out.text == prior
    assert out.evidence_metrics["section_response_valid"] is False
    assert out.input_tokens == 8 and out.output_tokens == 4


@pytest.mark.parametrize("extra,weak,gaps,hint", [
    ("\n## Results\nRepeated heading.", ["results"], [], None),
    ("\n```\n## Example\n```", ["results"], [], None),
    ("", ["results", "limitations"], [], None),
    ("", [], [{"text": "omission"}], None),
    ("", ["results"], [{"text": "omission"}], "Limitations"),
    ("", ["missing-section"], [], None),
])
def test_ambiguous_or_multisection_edits_keep_full_draft_repair(monkeypatch, extra, weak, gaps, hint):
    prior = "## Results\nOriginal result.\n## Limitations\nKeep this." + extra
    draft = Draft(prior, "m", 0, 0, 0)
    details = [SimpleNamespace(section=k, text="result", position=0) for k in weak]
    critique = revise.CritiqueOutput("repair", details, "m", 0, 0, coverage_gaps=gaps, repair_section=hint)

    def call(**kwargs):
        assert "schema" not in kwargs and "FULL revised draft" in kwargs["prompt"]
        return SimpleNamespace(text="full response", model="m", temperature=0, input_tokens=1, output_tokens=2)

    monkeypatch.setattr(revise.llm, "call", call)
    out = revise.evolve(draft=draft, critique=critique, metadata={}, sections={})
    assert out.text == "full response" and out.evidence_metrics["revision_scope"] == "page"
