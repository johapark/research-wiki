"""Portable draft lineage with synthetic pages, no model or network calls."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from itertools import count
import json
from pathlib import Path
import shutil
import uuid

import pytest

from researchwiki.agents import llm, model_config, relay, runner, phases
from researchwiki.agents.context import Context
from researchwiki.agents.phases.revise import CritiqueOutput, EvolveOutput
from researchwiki.agents.runner_support import record_revision_decision
from researchwiki.operations.lifecycle import observed_phase
from researchwiki.operations.observation import submit_observed
from researchwiki.records import RecordStore, RecordStoreUnavailable
from researchwiki.records.export import export_attempt
from researchwiki.records.jsonio import decode_json
from researchwiki.records.lifecycle import capture_operation
from researchwiki.records.lineage import current_scope
from tests.test_ingest_retention_lifecycle import library as library


@pytest.fixture
def lineage(library, monkeypatch):
    root, request = library
    ids = count(1)
    monkeypatch.setattr(runner, "write_iteration", lambda **kwargs: next(ids))
    with capture_operation(None) as owner:
        ctx = Context(attempt_id=str(uuid.uuid4()), pdf_path=Path(request.source_pdf),
                      pdf_filename="synthetic.pdf", paper_stem="synthetic", metadata={"title": "Synthetic"},
                      use_stub=True, use_semantic=False)
        owner.bind(ctx)
        ctx.sections, ctx.pdf_full_text = owner.records.extract(ctx.processing_pdf_path)
        yield root, ctx, owner
    assert current_scope() is None


def artifacts(owner, slot):
    records = owner.records
    manifest = records.store.read_attempt(records.spec.attempt_id)
    ids = next(item.artifact_ids for item in manifest.inventory if item.slot == slot)
    return [(item, decode_json(records.store.read_blob(item.payload))) for item in manifest.artifacts if item.artifact_id in ids]


def draft(text="synthetic draft", **kwargs):
    return phases.Draft(text=text, model="synthetic-author-1", temperature=0.2,
                        input_tokens=0, output_tokens=0, **kwargs)


def author(lineage, monkeypatch, text="synthetic draft", slot=0):
    _, ctx, _ = lineage
    monkeypatch.setattr(phases, "author", lambda **kwargs: draft(text, stance=kwargs["stance"]))
    result = runner._phase_author(ctx, None, 0.2, slot=slot)
    ctx.drafts.append(result)
    return result


def terminal(owner):
    owner.finish({"scope": "synthetic-lineage-test", "execution": "finished", "outcome": "draft_ready",
                  "publication": "not_published", "projections": "not_applicable"})
    return owner.records.store.read_attempt(owner.ctx.attempt_id)


def test_all_candidates_and_real_tournament_rationale_survive(lineage, monkeypatch):
    _, ctx, owner = lineage
    first = author(lineage, monkeypatch, "first loser " * 300)
    second = author(lineage, monkeypatch, "second winner", slot=1)
    first.scores = {"mean_bm25": 1}
    second.scores = {"mean_bm25": 10}
    assert runner._phase_tournament(ctx, None) is second
    retained = artifacts(owner, "lineage.drafts")
    assert {value["draft"]["text"] for _, value in retained} == {first.text, second.text}
    selection = next(value for _, value in artifacts(owner, "lineage.tournament.output") if value["boundary"] == "selection_decision")
    assert selection["rationale"].startswith("kept ")
    inputs = artifacts(owner, "lineage.tournament.input")[0][1]
    assert [c["key"][-2] for c in inputs["candidates"]] == [1, 10]
    assert selection["selected"] == inputs["candidates"][1]["draft"]
    manifest = terminal(owner)
    assert manifest.capture_state == "complete"
    assert next(i.status for i in manifest.inventory if i.slot == "lineage.author.request") == "not_run"
    assert next(i.status for i in manifest.inventory if i.slot == "lineage.evolve.input") == "not_run"


def test_rejected_revision_retains_parent_critique_and_decision(lineage, monkeypatch):
    _, ctx, owner = lineage
    prior = author(lineage, monkeypatch, "prior winner")
    ctx.winner = prior
    critique = CritiqueOutput("full feedback " * 300, [], "synthetic-critic-1", 0, 0)
    monkeypatch.setattr(phases, "critic", lambda **kwargs: critique)
    actual = runner._phase_critic(ctx, None, prior)
    monkeypatch.setattr(phases, "evolve", lambda **kwargs: EvolveOutput("rejected revision " * 300, "synthetic-reviser-1", 0.3, 0, 0))
    revised = runner._phase_evolve(ctx, None, prior, actual)
    record_revision_decision(ctx, None, revised, operator="evolve", accepted=False, writer=lambda **kwargs: None)
    evolved_input = artifacts(owner, "lineage.evolve.input")[0]
    critique_output = artifacts(owner, "lineage.critic.output")[0]
    assert critique_output[0].artifact_id in evolved_input[0].parents
    assert evolved_input[1]["critique"]["notes"] == critique.notes
    decision = artifacts(owner, "lineage.decisions")[0]
    assert decision[1]["accepted"] is False and decision[1]["prior"] != decision[1]["candidate"]
    assert ctx.winner is prior
    assert any(v["draft"]["text"] == revised.text for _, v in artifacts(owner, "lineage.drafts"))
    terminal(owner)


def test_empty_critique_is_observed_without_inventing_a_call(lineage, monkeypatch):
    _, ctx, owner = lineage
    prior = author(lineage, monkeypatch)
    monkeypatch.setattr(phases, "critic", lambda **kwargs: CritiqueOutput("", [], "(no-call)", 0, 0))
    runner._phase_critic(ctx, None, prior)
    manifest = terminal(owner)
    assert artifacts(owner, "lineage.critic.output")[0][1]["critique"]["notes"] == ""
    assert next(i.status for i in manifest.inventory if i.slot == "lineage.critic.request") == "not_run"


@pytest.mark.parametrize("route", ["openai-compatible", "chat-relay"])
def test_author_requests_and_untrimmed_answers_are_separate_from_draft(lineage, monkeypatch, route):
    root, ctx, owner = lineage
    ctx.use_stub = False
    response_text = "## Summary\nSynthetic answer.\n\nHANDLE: Synthetic\nHOOK: Synthetic fixture."
    cfg = model_config.ModelConfig(provider=route, model="synthetic-author-1", temperature=0.2, max_tokens=2000)
    monkeypatch.setattr(model_config, "for_phase", lambda phase: cfg)
    monkeypatch.setenv("OPENAI_API_KEY", "never-retain-secret")
    monkeypatch.setenv("RW_LLM_BASE_URL", "https://fixture.invalid/v1")
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(json.dumps({
        "model": "reported-author", "choices": [{"message": {"content": response_text}}],
        "usage": {"prompt_tokens": 17, "completion_tokens": 9}}).encode()))
    monkeypatch.setattr(relay, "wiki_root", lambda: root)
    def handoff(pending, completed, *args, **kwargs):
        relay._write_atomic_json(completed, {"via": "fixture/synthetic-author-1", "response": response_text})
    monkeypatch.setattr(relay, "_emit_handoff_message", handoff)
    result = runner._phase_author(ctx, None, 0.2, slot=2)
    requests = artifacts(owner, "lineage.author.request")
    answers = artifacts(owner, "lineage.author.response")
    assert requests and answers and not artifacts(owner, "target.response")
    visible = next(v for _, v in answers if v["boundary"] == "visible_answer")
    assert (visible["answer"] if route == "openai-compatible" else json.loads(visible["answer"])["response"]) == response_text
    assert all(v["run_id"] == requests[0][1]["run_id"] for _, v in answers)
    output = artifacts(owner, "lineage.author.output")[-1]
    draft_id = output[1]["draft"]
    item, value = next(pair for pair in artifacts(owner, "lineage.drafts") if pair[0].artifact_id == draft_id)
    assert value["draft"]["text"] == result.text
    assert any(a.artifact_id in item.parents for a, _ in answers)
    assert "never-retain-secret" not in json.dumps([v for _, v in requests + answers])


def test_phase_failure_and_interruption_leave_inputs_and_safe_error_only(lineage):
    _, ctx, owner = lineage
    @observed_phase("author")
    def fail(ctx, conn, temperature, slot=0):
        raise KeyboardInterrupt("do-not-retain-exception-secret")
    with pytest.raises(KeyboardInterrupt):
        fail(ctx, None, 0.2)
    output = artifacts(owner, "lineage.author.output")[0][1]
    assert output["status"] == "interrupted" and output["error_code"] == "KeyboardInterrupt"
    assert "do-not-retain-exception-secret" not in json.dumps(output)
    assert current_scope() is None


def test_input_capture_failure_prevents_author_and_latches(lineage, monkeypatch):
    _, ctx, owner = lineage
    original = owner.records.store.record_artifact
    def reject(slot, item):
        if slot == "lineage.author.input":
            raise RecordStoreUnavailable("injected input failure")
        return original(slot, item)
    monkeypatch.setattr(owner.records.store, "record_artifact", reject)
    calls = []
    monkeypatch.setattr(phases, "author", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(RecordStoreUnavailable):
        runner._phase_author(ctx, None, 0.2)
    assert not calls and owner.records.failure is not None


def test_parallel_author_slots_have_independent_runs_and_draft_ids(lineage):
    _, ctx, owner = lineage
    @observed_phase("author")
    def produce(ctx, conn, temperature, slot=0):
        return draft(f"slot {slot}", stance=str(slot))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [submit_observed(pool, produce, replace(ctx, iteration=i), None, 0.2, slot=i) for i in range(3)]
        assert [f.result().text for f in futures] == ["slot 0", "slot 1", "slot 2"]
    inputs = [v for _, v in artifacts(owner, "lineage.author.input")]
    outputs = [v for _, v in artifacts(owner, "lineage.author.output")]
    assert len({v["run_id"] for v in inputs}) == 3
    assert {v["run_id"] for v in inputs} == {v["run_id"] for v in outputs}
    assert sorted(v["slot"] for v in inputs) == [0, 1, 2]
    assert len({v["draft"] for v in outputs}) == 3
    assert current_scope() is None


def test_export_retains_losing_draft_and_selection_after_source_deletion(lineage, monkeypatch, tmp_path):
    root, ctx, owner = lineage
    first = author(lineage, monkeypatch, "portable loser " * 300)
    second = author(lineage, monkeypatch, "portable winner")
    second.scores = {"mean_bm25": 10}
    ctx.winner = runner._phase_tournament(ctx, None)
    manifest = terminal(owner)
    destination = tmp_path / "portable-lineage"
    export_attempt(owner.records.store, ctx.attempt_id, destination)
    shutil.rmtree(root)
    moved = RecordStore(destination)
    assert moved.verify(ctx.attempt_id).manifest == manifest
    assert any(first.text in moved.read_blob(a.payload).decode() for a in manifest.artifacts if a.kind == "parse_trace")
    assert current_scope() is None


def test_grade_observation_links_to_exact_draft_and_prior_output(lineage, monkeypatch):
    _, ctx, owner = lineage
    result = author(lineage, monkeypatch, "## Key Contributions\n- Synthetic fixture result.")
    monkeypatch.setattr(phases, "grade_draft", lambda **kwargs: ({"n_graded": 1, "mean_bm25": 10}, []))
    runner._phase_grade(ctx, None, result)
    authored = artifacts(owner, "lineage.author.output")[0][1]["draft"]
    grade_input = artifacts(owner, "lineage.grade.input")[0][0]
    output = artifacts(owner, "lineage.grade.output")[0][1]
    assert authored in grade_input.parents
    assert output["draft"] == authored
    assert output["scores"]["n_graded"] == 1


def test_real_grading_bundle_has_lineage_parent(lineage, monkeypatch):
    _, ctx, owner = lineage
    from researchwiki.index import pdf_chunks, embeddings
    monkeypatch.setattr(pdf_chunks, "_build_embeddings_cache", lambda *args: None)
    monkeypatch.setattr(embeddings, "_get_model", lambda *args: None)
    result = author(lineage, monkeypatch, "## Key Contributions\n- Synthetic fixture result.")
    runner._phase_grade(ctx, None, result)
    input_id = artifacts(owner, "lineage.grade.input")[0][0].artifact_id
    started = next(item for item, value in artifacts(owner, "grading.input") if value["boundary"] == "run_started")
    assert input_id in started.parents


def test_failed_draft_output_capture_stops_and_preserves_raw_answer(lineage, monkeypatch):
    _, ctx, owner = lineage
    from researchwiki.records.observation import observe_call
    original = owner.records.store.record_artifact
    def reject(slot, item):
        if slot == "lineage.drafts":
            raise RecordStoreUnavailable("injected draft capture failure")
        return original(slot, item)
    monkeypatch.setattr(owner.records.store, "record_artifact", reject)
    @observed_phase("author")
    def produce(ctx, conn, temperature, slot=0):
        with observe_call("author", uuid.uuid4().hex, {"model": "fixture"}) as capture:
            capture.answer("full visible response", model="fixture")
        return draft()
    with pytest.raises(RecordStoreUnavailable):
        produce(ctx, None, 0.2)
    assert artifacts(owner, "lineage.author.response")[0][1]["answer"] == "full visible response"
    assert owner.records.failure is not None and current_scope() is None


def test_unavailable_author_answer_cannot_claim_complete_capture(lineage):
    _, ctx, owner = lineage
    from researchwiki.records.observation import observe_call
    @observed_phase("author")
    def produce(ctx, conn, temperature, slot=0):
        with observe_call("author", uuid.uuid4().hex, {"model": "fixture"}) as capture:
            capture.answer(None)
        return draft()
    produce(ctx, None, 0.2)
    manifest = terminal(owner)
    assert manifest.capture_state == "incomplete"
    assert next(i.status for i in manifest.inventory if i.slot == "lineage.author.response") == "unavailable"


def test_cli_lineage_after_export_and_default_envelope_unchanged(lineage, monkeypatch, tmp_path, capsys):
    root, ctx, owner = lineage
    from researchwiki.tasks.records import main
    first = author(lineage, monkeypatch, "portable text")
    ctx.winner = first
    runner._phase_tournament(ctx, None)
    terminal(owner)
    output = tmp_path / "export-inspection"
    export_attempt(owner.records.store, ctx.attempt_id, output)
    monkeypatch.chdir(tmp_path)
    shutil.rmtree(root)
    assert main(["--root", str(output), "show", ctx.attempt_id, "--lineage", "--json"]) == 0
    values = json.loads(capsys.readouterr().out)
    assert any(v["payload"].get("draft", {}).get("text") == "portable text"
               for v in values["lineage"] if isinstance(v["payload"].get("draft"), dict))
    assert main(["--root", str(output), "show", ctx.attempt_id, "--json"]) == 0
    assert "lineage" not in json.loads(capsys.readouterr().out)


def test_rejected_debug_draft_is_retained_with_nested_grade_parent(lineage, monkeypatch):
    _, ctx, owner = lineage
    from researchwiki.agents.promote import GateResult
    from researchwiki.agents.phases.crosslinks import VerificationReport
    prior = author(lineage, monkeypatch, "prior")
    prior.scores = {"mean_bm25": 10}
    ctx.winner = prior
    monkeypatch.setattr(phases, "debug", lambda **kwargs: EvolveOutput("rejected debug body", "synthetic-debug-1", 0.2, 0, 0))
    monkeypatch.setattr(phases, "grade_draft", lambda **kwargs: ({"mean_bm25": 1, "n_graded": 1}, []))
    gate = GateResult(False, ["synthetic structural issue"])
    verification = VerificationReport([], [], [])
    output = runner._phase_debug(ctx, None, prior.text, 0, gate, verification, issues=["synthetic"])
    assert output[2] is gate and ctx.winner is prior
    body = next(item for item, value in artifacts(owner, "lineage.drafts") if value["draft"]["text"] == "rejected debug body")
    debug_input = artifacts(owner, "lineage.debug.input")[0][0]
    assert debug_input.artifact_id in body.parents
    decision = artifacts(owner, "lineage.decisions")[0][1]
    assert decision["candidate"] == body.artifact_id and decision["accepted"] is False
