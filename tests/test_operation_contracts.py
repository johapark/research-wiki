"""Application contracts preserve gate truth, uncertain commits, and relay waits."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchwiki.operations import (
    CheckResult, ContractError, IngestEvent, IngestOptions, IngestRequest,
    IngestResult, OperationError, RelayHandoff, UsageReport,
    error_from_exception, result_from_context, transition_execution,
)


def request(tmp_path, **options):
    return IngestRequest(
        request_id="request-1", library_root=str(tmp_path),
        source_pdf=str(tmp_path / "incoming.pdf"), options=IngestOptions(**options),
    )


def test_request_round_trip_keeps_options_and_defers_per_phase_provider_routing(tmp_path):
    original = request(
        tmp_path, authors=("Ada Example",), verify_claim_entailment=True,
        supplementary=(str(tmp_path / "supp.pdf"),), max_tokens=4000,
    )
    data = json.loads(json.dumps(original.to_dict(), allow_nan=False))
    assert IngestRequest.from_dict(data) == original
    assert data["options"]["n_drafts"] is None  # resolve from model config later
    assert "provider" not in data and "mode" not in data
    assert "api_key" not in data


@pytest.mark.parametrize("update", [
    {"schema_version": 2}, {"schema_version": True}, {"api_key": "secret"},
    {"options": {"max_tokens": True}}, {"options": {"max_model_calls": 0}},
    {"options": {"max_cost_usd": float("nan")}}, {"options": {"max_evolve": -1}},
    {"options": {"n_drafts": 1.5}}, {"options": {"use_semantic": "false"}},
    {"options": {"unknown_option": True}}, {"source_pdf": "relative.pdf"},
    {"source_pdf": ""}, {"request_id": " "}, {"provider_profile": "relative.env"},
])
def test_bad_wire_requests_fail_before_any_provider_or_filesystem_work(tmp_path, update):
    data = request(tmp_path).to_dict()
    data.update(update)
    with pytest.raises(ContractError):
        IngestRequest.from_dict(data)


def test_direct_construction_is_validated_too():
    with pytest.raises(ContractError):
        IngestOptions(max_tokens=True)
    with pytest.raises(ContractError):
        UsageReport(status="unavailable", input_tokens=0)
    with pytest.raises(ContractError):
        IngestOptions(authors=("",))


@pytest.mark.parametrize("engine,outcome,published", [
    ("promoted", "published", True), ("sandboxed", "draft_ready", False),
    ("already_present", "already_present", True),
])
def test_terminal_context_mapping_does_not_invent_projection_success(tmp_path, engine, outcome, published):
    ctx = SimpleNamespace(
        outcome=engine, committed_path=tmp_path / "wiki/methods/example.md",
        paper_stem="example", attempt_id="attempt-1", promote_mode="auto", gate_reasons=[],
    )
    result = result_from_context(request(tmp_path), ctx)
    assert result.outcome == outcome
    assert result.publication == ("published" if published else "not_published")
    assert result.projections == ("unknown" if published else "not_applicable")
    assert (result.page_path is not None) is published
    assert (result.draft_path is not None) is not published
    assert result.usage.status == "unavailable"
    assert IngestResult.from_dict(json.loads(json.dumps(result.to_dict()))) == result


def test_forced_publication_keeps_failed_checks_and_gate_reasons(tmp_path):
    ctx = SimpleNamespace(
        outcome="promoted", committed_path=tmp_path / "wiki/methods/example.md",
        paper_stem="example", attempt_id="attempt-1", promote_mode="always",
        gate_reasons=["numeric drift"],
        winner=SimpleNamespace(scores={"n_graded": 5, "n_drift": 1}),
    )
    result = result_from_context(request(tmp_path, promotion="always"), ctx,
                                 projections="maintenance_pending")
    assert result.outcome == "published"
    assert result.promotion == "always"
    assert result.gate_reasons == ("numeric drift",)
    assert CheckResult(name="numeric", status="failed") in result.checks
    assert CheckResult(name="retrieval", status="completed") in result.checks
    assert CheckResult(name="entailment", status="not_requested") in result.checks


@pytest.mark.parametrize("update", [
    {"execution": "finished", "outcome": None},
    {"execution": "cancelled", "publication": "unknown"},
    {"execution": "running", "outcome": "failed"},
    {"execution": "finished", "outcome": "published", "publication": "unknown"},
    {"execution": "finished", "outcome": "draft_ready", "publication": "published"},
    {"execution": "finished", "outcome": "failed", "publication": "published"},
    {"projections": "current", "publication": "unknown"},
])
def test_status_cannot_misrepresent_publication_or_cancellation(update):
    data = dict(request_id="r", execution="interrupted", outcome=None, publication="unknown")
    data.update(update)
    with pytest.raises(ContractError):
        IngestResult(**data)


def test_interrupted_commit_uncertainty_is_round_trippable():
    fixture = Path(__file__).parent / "operation-fixtures/interrupted-v1.json"
    result = IngestResult.from_dict(json.loads(fixture.read_text()))
    assert result.execution == "interrupted" and result.publication == "unknown"
    assert result.error.retry == "reconcile_first"


def test_retry_cannot_silently_restart_an_interrupted_or_terminal_job():
    assert transition_execution("queued", "running") == "running"
    assert transition_execution("interrupted", "finished") == "finished"
    for current, target in (("interrupted", "running"), ("running", "cancelled"),
                            ("finished", "running"), ("cancelled", "queued")):
        with pytest.raises(ContractError):
            transition_execution(current, target)


def test_relay_timeout_is_environment_failure_and_does_not_leak_error_text():
    from researchwiki.agents.relay import RelayTimeout
    error = error_from_exception(RelayTimeout("confidential provider response"))
    assert error == OperationError(
        code="relay_timeout", category="environment", message="RelayTimeout",
        retry="after_environment_recovery",
    )
    assert "confidential" not in json.dumps(error.to_dict())


def test_publication_failure_requires_reconciliation_even_when_a_path_exists(tmp_path):
    from researchwiki.agents.context import PromoteFailed
    error = error_from_exception(PromoteFailed(stem="example", page_path=tmp_path / "page.md"))
    assert error.retry == "reconcile_first"


def test_relay_handoff_round_trip_keeps_exact_mailbox_identity(tmp_path):
    handoff = RelayHandoff(
        op_id="op-2", phase="author", prompt_path=str(tmp_path / "op-2.prompt.json"),
        response_path=str(tmp_path / "op-2.response.json"), timeout_seconds=600,
        retry_of="op-1",
    )
    event = IngestEvent(
        request_id="r", sequence=1, timestamp="2026-10-18T00:00:00Z",
        kind="relay_waiting", call_id="call-1", relay=handoff,
    )
    assert IngestEvent.from_dict(json.loads(json.dumps(event.to_dict()))) == event
    with pytest.raises(ContractError):
        IngestEvent(request_id="r", sequence=1, timestamp="2026-10-18T00:00:00",
                    kind="model_call_started", call_id="call-1")
