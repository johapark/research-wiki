"""Runner propagation and revision observations exercise the actual phases."""

from types import SimpleNamespace

import pytest

from researchwiki.agents import runner
from researchwiki.agents.context import Context
from researchwiki.agents.phases.draft import Draft
from researchwiki.agents.phases.grade import ClaimDetail
from researchwiki.agents.phases.revise import CritiqueOutput
from researchwiki.index import pdf_chunks


@pytest.mark.parametrize("operator", ["evolve", "debug"])
def test_runner_repairs_keep_author_prompt_pdf_and_evidence_metrics(monkeypatch, tmp_path, operator):
    prompt_file = tmp_path / "author.md"
    prompt_file.write_text("KEEP_CUSTOM_PRIORITY\n")
    pdf = tmp_path / "incoming.pdf"
    detail = ClaimDetail("results", 1, "periodic glycine steering 999", 0, .1, False, ["999"])
    draft = Draft("## Summary\nKeep this.\n## Limitations\nKeep that.", "m", 0, 0, 0,
                  claim_details=[detail], handle="Handle", hook="Hook")
    ctx = Context("attempt", pdf, pdf.name, author_prompt_override=str(prompt_file),
                  paper_stem="paper", sections={}, metadata={}, winner=draft)
    calls, queries, rows = [], [], []

    def query(stem, text, **kwargs):
        queries.append(kwargs)
        return [pdf_chunks.RetrievedChunk(9, 5, "periodic glycine steering 123", 17, 17)]

    def call(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=draft.text, model="m", temperature=0,
                               input_tokens=1, output_tokens=1)

    monkeypatch.setattr(pdf_chunks, "query_pdf", query)
    monkeypatch.setattr(runner.phases.revise.llm, "call", call)
    monkeypatch.setattr(runner, "write_iteration", lambda **kw: rows.append(kw) or 10)
    if operator == "evolve":
        revised = runner._phase_evolve(ctx, None, draft, CritiqueOutput("fix number", [detail], "m", 0, 0))
    else:
        monkeypatch.setattr(runner, "_phase_grade", lambda *a, **kw: None)
        monkeypatch.setattr(runner, "_is_strict_improvement", lambda *a: False)
        runner._phase_debug(ctx, None, draft.text, 4, SimpleNamespace(reasons=[]), None, issues=["drift"])
        revised = ctx.drafts[-1]
    assert calls[0]["system"] == prompt_file.read_text()
    assert queries == [{"topk": 2, "pdf_path": pdf}]
    assert rows[0]["gate_metrics"]["evidence_route"] == "targeted"
    assert rows[0]["gate_metrics"]["evidence_queries"] == 1
    assert revised.handle == draft.handle and revised.hook == draft.hook


def test_revision_rejection_records_lost_critical_ids():
    from researchwiki.agents.runner_support import record_revision_decision

    def scores(misses):
        return {"target_claim_extraction_status": "extracted", "n_target_claims": 5,
                "target_claim_score": .7, "missed_target_claims": misses}

    ctx = SimpleNamespace(next_iter=lambda: None, attempt_id="a", paper_stem="s",
                          pdf_filename="s.pdf", iteration=2,
                          winner=SimpleNamespace(scores=scores([])))
    revised = SimpleNamespace(iteration_id=9, scores=scores([
        {"id": "target-002", "importance": "critical"},
    ]))
    rows = []
    record_revision_decision(ctx, None, revised, operator="evolve", accepted=False,
                             writer=lambda **kwargs: rows.append(kwargs))
    assert rows[0]["decision"] == "discarded"
    assert rows[0]["gate_metrics"]["new_critical_misses"] == ["target-002"]
    assert "target-002" in rows[0]["decision_reason"]
