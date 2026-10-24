"""Synthetic extraction observations: no external providers or library writes."""

from contextlib import contextmanager
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import urllib.error
import uuid

import pytest

from researchwiki.agents import llm, model_config, relay
from researchwiki.agents.budget import BudgetTracker, IngestBudget, activate
from researchwiki.agents.context import Context, processing_pdf
from researchwiki.agents.phases.extract import extract_sections
from researchwiki.agents.phases.target_claims import (
    extract_target_claims, parse_target_claims, render_for_author_prompt,
)
from researchwiki.pdf.representation import extract_representation
from researchwiki.pdf.text import extract_pdf
from researchwiki.records import (
    AttemptFinished, AttemptSpec, CapturePolicy, Producer, RecordConflict,
    RecordStore, RecordStoreUnavailable, extraction_requirements,
)
from researchwiki.records.extraction import ExtractionRecorder
from researchwiki.records.jsonio import decode_json
from researchwiki.records.observation import current_call, observe_extraction


def claim(content="Synthetic benchmark AUC 0.918", **kwargs):
    return {"type": "headline", "content": content, "importance": "high", "location": "Results", **kwargs}


def payload(**kwargs):
    return {"claims": [claim()], "keywords": ["Synthetic benchmark", "AUC", "precision recall",
                                               "variant calling", "genome benchmark"], **kwargs}


@pytest.fixture
def recorder(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "extraction_pdf_fixture", Path(__file__).resolve().parents[1] / ".github/scripts/fresh_install_acceptance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "input.pdf"
    module.write_fixture_pdf(path, title="Synthetic Extraction Fixture", author="Ada Lovelace")
    store = RecordStore.create(tmp_path / "archive")
    attempt = AttemptSpec(attempt_id=str(uuid.uuid4()), policy=CapturePolicy(mode="required"),
                          producer=Producer(name="synthetic-extraction-test", version="1"),
                          requirements=extraction_requirements(), created_at="2026-10-23T12:00:00+00:00")
    store.start_attempt(attempt)
    records = ExtractionRecorder(store, attempt.attempt_id)
    source = records.snapshot_source(path)
    extract_sections(source.processing_path, records=records)
    return records


def contents(records, kind):
    manifest = records.store.read_attempt(records.spec.attempt_id)
    return [decode_json(records.store.read_blob(item.payload)) for item in manifest.artifacts if item.kind == kind]


def run_target(records, **kwargs):
    return extract_target_claims(metadata={"title": "Synthetic"}, sections=records.sections,
                                 pdf_full_text=records.full_text, records=records, **kwargs)


@pytest.fixture
def api(monkeypatch):
    calls = []
    response = {"model": "reported-model-2", "choices": [{"message": {"content": json.dumps(payload())},
                                                           "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 9}}
    def urlopen(request, **kwargs):
        calls.append(json.loads(request.data))
        return BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    cfg = model_config.ModelConfig(provider="openai-compatible", model="configured-model", temperature=0.1, max_tokens=4096)
    monkeypatch.setattr(model_config, "for_phase", lambda phase: cfg)
    monkeypatch.setenv("RW_LLM_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "NEVER-RETAIN-THIS-KEY")
    return calls, response


def assert_no_credentials(records):
    all_json = b"".join(path.read_bytes() for path in records.store.root.rglob("*") if path.is_file())
    assert b"NEVER-RETAIN-THIS-KEY" not in all_json
    assert b"api.example.test" not in all_json


def test_source_snapshot_processing_ownership_and_representation_are_exact(recorder):
    records = recorder
    source = records.source
    legacy, _ = extract_pdf(source.original_path, max_pages=80)
    assert records.full_text == legacy
    representation = contents(records, "representation")[0]
    assert representation["text"] == legacy and representation["page_mapping"] == "exact"
    for page in representation["pages"]:
        assert legacy[page["start"]:page["end"]] == page["repaired_text"]
    assert representation["unread_pages"] == []
    ctx = Context(attempt_id=records.spec.attempt_id, pdf_path=source.original_path, pdf_filename="input.pdf")
    records.bind_context(ctx)
    source.original_path.write_bytes(b"%PDF-replacement")
    assert ctx.pdf_path.read_bytes() == b"%PDF-replacement"
    assert processing_pdf(ctx) == source.processing_path
    assert processing_pdf(ctx).read_bytes() != ctx.pdf_path.read_bytes()
    assert extract_sections(processing_pdf(ctx), records=records)[1] == legacy
    with pytest.raises(RecordConflict):
        records.extract(ctx.pdf_path)
    with pytest.raises(RecordConflict):
        extract_target_claims(metadata={}, sections={"results": "unrelated"}, records=records)


def test_page_limits_and_cross_page_repair_do_not_change_legacy_text(tmp_path, monkeypatch):
    import researchwiki.pdf.representation as representation
    class PDF:
        def __init__(self, _): pass
        def __len__(self): return 3
        def get_metadata_dict(self): return {"Title": "Synthetic"}
        def close(self): pass
    monkeypatch.setattr(representation.pypdfium2, "PdfDocument", PDF)
    monkeypatch.setattr(representation, "_extract_all_pages", lambda pdf, maximum: ["first", "second"][:maximum])
    monkeypatch.setattr(representation, "repair_text", lambda text: text.replace("first\n\nsecond", "cross-page"))
    result = extract_representation(tmp_path / "opaque.pdf", max_pages=2)
    assert result.text == "cross-page"
    assert result.record["pages_read"] == 2 and result.record["unread_pages"] == [3]
    assert result.record["page_mapping"] == "unavailable_cross_page_repair"
    assert all(page["start"] is None for page in result.record["pages"])


def test_mutating_returned_sections_cannot_change_retained_input_identity(recorder):
    sections, text = extract_sections(recorder.source.processing_path, records=recorder)
    sections["results"] = "unrelated substituted results"
    with pytest.raises(RecordConflict, match="target inputs differ"):
        recorder.validate_inputs(sections, text)


def test_parser_retains_invalid_duplicate_capped_and_late_critical_candidates():
    candidates = [claim(f"Synthetic result {i}") for i in range(37)]
    candidates += [claim("Synthetic result 0."), claim("Central result", importance="critical"),
                   {"type": "headline", "content": 123}, "invalid item"]
    out, trace = parse_target_claims("```json\n" + json.dumps(payload(claims=candidates)) + "\n```")
    assert len(out.claims) == 35 and out.claims[0].content == "Central result"
    assert len(trace["candidates"]) == len(candidates)
    assert trace["candidates"][37]["duplicate_of"] == "candidate-0000"
    assert trace["candidates"][36]["selection_reason"] == "cap"
    assert trace["candidates"][38]["selected_for_checklist"]
    assert "content_not_string" in trace["candidates"][39]["validation_errors"]
    assert trace["candidates"][40]["validation_errors"] == ["not_an_object"]
    assert trace["keywords_before"] == trace["keywords_after"]


def test_parser_partial_spans_no_object_invalid_shape_and_nonfinite_values():
    text = '```json\n{"claims":[' + json.dumps(claim()) + ',{"type":"headline","content":"unfinished'
    out, trace = parse_target_claims(text)
    assert len(out.claims) == 1 and trace["status"] == "partial_salvage"
    start, end = trace["candidates"][0]["response_span"]
    assert json.loads(text[start:end]) == claim()
    for raw in ('plain malformed reply', '{"claims":"bad shape"}', '{"claims":[{"type":NaN}]}'):
        out, trace = parse_target_claims(raw)
        assert out.is_empty() and (out.error or trace["candidates"])
        assert json.dumps(trace, allow_nan=False)


def test_api_exact_request_answer_parser_and_relocation_without_database(recorder, api, tmp_path):
    calls, response = api
    response["choices"][0]["message"]["content"] = "```json\n" + json.dumps(payload()) + "\n```"
    out = run_target(recorder)
    assert len(out.claims) == 1 and len(calls) == 1
    requests = contents(recorder, "model_request")
    assert requests[0]["boundary"] == "logical" and requests[0]["schema"]["type"] == "object"
    assert requests[1]["payload"]["messages"] == calls[0]["messages"]
    assert all(req["transport_enforcement"] == "not_applied" for req in requests)
    responses = contents(recorder, "model_response")
    assert responses[0]["answer"] == response["choices"][0]["message"]["content"]
    assert responses[0]["model"] == "reported-model-2" and responses[0]["model_origin"] == "provider_reported"
    assert responses[1]["model"] == "configured-model" and responses[1]["model_origin"] == "configured"
    assert_no_credentials(recorder)
    render_for_author_prompt(out, records=recorder)
    recorder.store.finish_attempt(recorder.spec.attempt_id, AttemptFinished(
        execution="finished", outcome="draft_ready", publication="not_published", projections="not_applicable"))
    assert recorder.store.read_attempt(recorder.spec.attempt_id).capture_state == "complete"
    assert current_call() is None
    relocated = tmp_path / "relocated"
    shutil.move(recorder.store.root, relocated)
    recorder.source.original_path.unlink()
    store = RecordStore(relocated)
    assert store.verify(recorder.spec.attempt_id).manifest.capture_state == "complete"
    assert not list(tmp_path.rglob("*.db"))


def test_stub_has_synthetic_result_and_no_fabricated_calls(recorder, monkeypatch):
    monkeypatch.setattr(llm, "call", lambda **kwargs: pytest.fail("stub must not call an LLM"))
    out = run_target(recorder, use_stub=True)
    assert out.model == "(stub)" and not contents(recorder, "model_request")
    assert not contents(recorder, "model_response")
    manifest = recorder.store.read_attempt(recorder.spec.attempt_id)
    assert all(item.status == "not_run" for item in manifest.inventory if item.slot in ("target.request", "target.response"))
    assert all(item.origin == "synthetic" for item in manifest.artifacts if item.kind in ("extraction_result", "parse_trace"))


@contextmanager
def fail_capture(records, monkeypatch, kind, predicate=lambda value: True):
    original = records.store.record_artifact
    def write(slot, item):
        if item.kind == kind and predicate(decode_json(records.store.read_blob(item.payload))):
            raise RecordStoreUnavailable("injected archive failure")
        return original(slot, item)
    with monkeypatch.context() as patch:
        patch.setattr(records.store, "record_artifact", write)
        yield


@pytest.mark.parametrize("boundary", ["logical", "client_dispatch", "visible_answer", "phase_input"])
def test_capture_failure_stops_calls_and_settles_budget_once(recorder, api, monkeypatch, boundary):
    calls, _ = api
    monkeypatch.setattr(model_config, "estimate_usd", lambda model, inp, out, **kwargs: (inp + out) / 1000)
    tracker = BudgetTracker(IngestBudget(max_model_calls=3))
    kind = "model_request" if boundary in ("logical", "client_dispatch") else "model_response"
    with activate(tracker), fail_capture(recorder, monkeypatch, kind, lambda value: value.get("boundary") == boundary):
        with pytest.raises(RecordStoreUnavailable):
            run_target(recorder)
        with pytest.raises(RecordStoreUnavailable):
            run_target(recorder)
    snapshot = tracker.snapshot()
    assert snapshot["reserved_tokens"] == 0 and snapshot["reserved_cost_usd"] == 0
    assert len(calls) == (0 if boundary in ("logical", "client_dispatch") else 1)
    assert snapshot["model_calls"] == len(calls)
    assert snapshot["tokens"] == (26 if calls else 0)
    assert snapshot["estimated_cost_usd"] == (0.026 if calls else 0)


def configure_relay(monkeypatch, tmp_path, responses):
    monkeypatch.delenv("RW_RELAY_FRESH", raising=False)
    cfg = model_config.ModelConfig(provider="chat-relay", model="configured-relay", temperature=0.1, max_tokens=4096)
    monkeypatch.setattr(model_config, "for_phase", lambda phase: cfg)
    monkeypatch.setattr(relay, "wiki_root", lambda: tmp_path)
    emitted = []
    def handoff(pending, completed, *args, **kwargs):
        emitted.append(json.loads(pending.read_text()))
        data = {"via": "test/synthetic-model-1", "structured": responses[len(emitted) - 1],
                "authorization": "NEVER-RETAIN-THIS-KEY"}
        relay._write_atomic_json(completed, data)
    monkeypatch.setattr(relay, "_emit_handoff_message", handoff)
    return emitted


def test_relay_rejected_answers_survive_retry_cleanup(recorder, monkeypatch, tmp_path):
    emitted = configure_relay(monkeypatch, tmp_path, [{"claims": "invalid"}, payload()])
    out = run_target(recorder)
    assert len(out.claims) == 1 and len(emitted) == 2
    responses = contents(recorder, "model_response")
    assert [item["validation"] for item in responses[:2]] == ["rejected", "accepted"]
    assert json.loads(responses[0]["answer"]) == {"structured": {"claims": "invalid"}}
    assert all(item["usage_status"] == "unavailable" for item in responses[:2])
    assert responses[1]["model_origin"] == "self_reported"
    assert not list((tmp_path / ".llm-relay").rglob("*.response.json"))
    assert not list((tmp_path / ".llm-relay").rglob("*.prompt.json"))
    assert_no_credentials(recorder)
    requests = contents(recorder, "model_request")
    assert requests[-1]["retry_of"] == requests[-2]["op_id"]
    artifacts = recorder.store.read_attempt(recorder.spec.attempt_id).artifacts
    dispatched = [item for item in artifacts if item.kind == "model_request" and item.dispatch_id]
    assert len({item.call_id for item in dispatched}) == 1 and len({item.dispatch_id for item in dispatched}) == 2


@pytest.mark.parametrize("accepted", [False, True])
def test_relay_capture_failure_preserves_mailbox_and_does_not_retry(recorder, monkeypatch, tmp_path, accepted):
    emitted = configure_relay(monkeypatch, tmp_path, [payload() if accepted else {"claims": "invalid"}])
    with fail_capture(recorder, monkeypatch, "model_response"):
        with pytest.raises(RecordStoreUnavailable):
            run_target(recorder)
    assert len(emitted) == 1
    assert len(list((tmp_path / ".llm-relay").rglob("*.response.json"))) == 1


def test_api_retries_have_separate_dispatches_without_secret_error_dump(recorder, api, monkeypatch):
    _, response = api
    count = []
    def urlopen(request, **kwargs):
        count.append(request)
        if len(count) == 1:
            raise urllib.error.HTTPError("secret-url", 500, "error", {}, BytesIO(b"NEVER-RETAIN-THIS-KEY"))
        return BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    assert not run_target(recorder).is_empty()
    assert len(count) == 2
    requests = contents(recorder, "model_request")
    assert len(requests) == 3
    traces = contents(recorder, "parse_trace")
    assert traces[0]["boundary"] == "transport_rejection" and traces[0]["http_status"] == 500
    assert_no_credentials(recorder)


def test_anthropic_controlled_temperature_and_reasoning_exclusion(recorder, monkeypatch):
    cfg = model_config.ModelConfig(provider="anthropic", model="claude-test", temperature=0.1, max_tokens=4096)
    monkeypatch.setattr(model_config, "for_phase", lambda phase: cfg)
    from tests.test_llm_anthropic_usage import _install_fake_anthropic
    seen = []
    error_type = None
    def create(**kwargs):
        seen.append(kwargs)
        if "temperature" in kwargs:
            raise error_type("temperature is not supported")
        return SimpleNamespace(model="reported-claude-model", stop_reason="end_turn",
            content=[SimpleNamespace(type="thinking", text="DO-NOT-RETAIN-REASONING"),
                     SimpleNamespace(type="text", text=json.dumps(payload()))],
            usage=SimpleNamespace(input_tokens=8, output_tokens=3, cache_read_input_tokens=2, cache_creation_input_tokens=1))
    error_type = _install_fake_anthropic(monkeypatch, create)
    out = run_target(recorder)
    assert len(seen) == 2 and len(out.claims) == 1
    answers = contents(recorder, "model_response")
    assert answers[0]["temperature"] == {"value": None, "status": "provider_controlled"}
    assert answers[0]["usage"]["input_tokens"] == 11
    assert answers[0]["model"] == "reported-claude-model"
    assert "DO-NOT-RETAIN-REASONING" not in json.dumps(answers)
    assert json.dumps(answers, allow_nan=False)


def test_openai_hidden_thought_prefix_is_excluded_from_archive(recorder, api):
    _, response = api
    response["choices"][0]["message"]["content"] = "<thought>DO-NOT-RETAIN-REASONING</thought>" + json.dumps(payload())
    assert not run_target(recorder).is_empty()
    assert "DO-NOT-RETAIN-REASONING" not in json.dumps(contents(recorder, "model_response"))


def test_other_phases_and_scope_exit_do_not_leak_retention(recorder, api):
    before = recorder.store.read_attempt(recorder.spec.attempt_id)
    with observe_extraction(recorder):
        llm.call(phase="author", prompt="unrelated author input")
    llm.call(phase="target_claims", prompt="outside retention scope")
    assert recorder.store.read_attempt(recorder.spec.attempt_id) == before
    assert current_call() is None


def test_unknown_usage_on_capture_failure_is_charged_conservatively(recorder, api, monkeypatch):
    calls, response = api
    del response["usage"]
    tracker = BudgetTracker(IngestBudget(max_model_calls=2))
    monkeypatch.setattr(model_config, "estimate_usd", lambda model, inp, out, **kwargs: (inp + out) / 1000)
    with activate(tracker), fail_capture(recorder, monkeypatch, "model_response"):
        with pytest.raises(RecordStoreUnavailable):
            run_target(recorder)
    assert len(calls) == 1 and tracker.calls == 1 and tracker.tokens > 4096
    assert tracker.cost_usd > 4 and tracker.reserved_tokens == tracker.reserved_cost_usd == 0


def test_rejection_capture_failure_does_not_trigger_api_retry(recorder, api, monkeypatch):
    count = []
    def urlopen(request, **kwargs):
        count.append(request)
        raise urllib.error.HTTPError("secret-url", 429, "error", {}, BytesIO(b"NEVER-RETAIN-THIS-KEY"))
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: pytest.fail("must not retry after capture failure"))
    tracker = BudgetTracker(IngestBudget(max_model_calls=2))
    with activate(tracker), fail_capture(recorder, monkeypatch, "parse_trace"):
        with pytest.raises(RecordStoreUnavailable):
            run_target(recorder)
    assert len(count) == 1 and tracker.calls == 1 and tracker.tokens > 0
    assert tracker.reserved_tokens == tracker.reserved_cost_usd == 0


def test_relay_cached_reply_reuse_is_recorded_without_new_handoff(recorder, monkeypatch, tmp_path):
    from researchwiki.agents.phases.target_claims import _build_prompt
    emitted = configure_relay(monkeypatch, tmp_path, [])
    prompt = _build_prompt({"title": "Synthetic"}, recorder.sections, recorder.full_text,
                           model_config.target_claims_max_chars())
    op_id = relay._stable_op_id("target_claims", prompt)
    _, completed = relay._paths_for(op_id)
    relay._write_atomic_json(completed, {"via": "test/synthetic-model-1", "structured": payload()})
    assert not run_target(recorder).is_empty() and not emitted
    requests = contents(recorder, "model_request")
    assert requests[-1]["response_reused"] is True
    assert contents(recorder, "model_response")[0]["response_reused"] is True
    assert not completed.exists()


@pytest.mark.parametrize("raw", [b'{"bad":', b'\xff'])
def test_invalid_relay_envelope_is_unavailable_not_complete(recorder, monkeypatch, tmp_path, raw):
    configure_relay(monkeypatch, tmp_path, [])
    def handoff(pending, completed, *args, **kwargs):
        completed.write_bytes(raw)
    monkeypatch.setattr(relay, "_emit_handoff_message", handoff)
    out = run_target(recorder)
    assert out.error and out.is_empty()
    responses = contents(recorder, "model_response")
    assert responses[0]["answer"] is None and responses[0]["answer_status"] == "unavailable"
    assert list((tmp_path / ".llm-relay").rglob("*.response.json"))
    manifest = recorder.store.finish_attempt(recorder.spec.attempt_id, AttemptFinished(
        execution="finished", outcome="failed", publication="not_published", projections="not_applicable"))
    assert manifest.capture_state == "incomplete"
    assert next(item for item in manifest.inventory if item.slot == "target.response").status == "unavailable"


def test_mixed_routing_captures_actual_phase_route(recorder, api, monkeypatch, tmp_path):
    calls, _ = api
    cfg_api = model_config.for_phase("author")
    emitted = configure_relay(monkeypatch, tmp_path, [payload()])
    cfg_relay = model_config.for_phase("target_claims")
    monkeypatch.setattr(model_config, "for_phase", lambda phase: cfg_relay if phase == "target_claims" else cfg_api)
    with observe_extraction(recorder):
        llm.call(phase="author", prompt="Synthetic author phase routed to API")
        run_target(recorder)
    assert len(calls) == 1 and len(emitted) == 1
    requests = contents(recorder, "model_request")
    assert {item["provider"] for item in requests} == {"chat-relay"}
    assert all("Synthetic author phase" not in json.dumps(item) for item in requests)


def test_author_exposure_retains_normal_item_cap_separately_from_extraction(recorder, api):
    _, response = api
    response["choices"][0]["message"]["content"] = json.dumps(payload(
        claims=[claim(f"Normal result {i}", importance="normal") for i in range(7)]))
    out = run_target(recorder)
    text = render_for_author_prompt(out, max_normal=2, records=recorder)
    assert len(out.claims) == 7 and "Normal result 1" in text and "Normal result 2" not in text
    trace = contents(recorder, "parse_trace")[-1]
    assert trace["boundary"] == "author_exposure" and trace["max_normal"] == 2 and trace["text"] == text
    assert len(contents(recorder, "extraction_result")[0]["output"]["claims"]) == 7


def test_negotiated_effort_is_retained_as_actual_dispatch_settings(recorder, api, monkeypatch):
    _, response = api
    model_cfg = model_config.for_phase("target_claims")
    from dataclasses import replace
    monkeypatch.setattr(model_config, "for_phase", lambda phase: replace(model_cfg, reasoning_effort="minimal"))
    monkeypatch.setattr(llm, "_EFFORT_REJECTED", {})
    monkeypatch.setattr(llm, "_EFFORT_SUPPORTED", {})
    seen = []
    def urlopen(request, **kwargs):
        body = json.loads(request.data)
        seen.append(body)
        if body.get("reasoning_effort") == "minimal":
            raise urllib.error.HTTPError("unused", 400, "error", {}, BytesIO(
                b"reasoning_effort does not support 'minimal'. Supported values are: 'none', 'low'."))
        return BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    assert not run_target(recorder).is_empty()
    assert [body.get("reasoning_effort") for body in seen] == ["minimal", "none"]
    requests = contents(recorder, "model_request")
    assert requests[0]["reasoning_effort"] == "minimal"
    assert [request["payload"]["reasoning_effort"] for request in requests[1:]] == ["minimal", "none"]


def test_transport_exception_text_is_not_archived_as_a_diagnostic(recorder, api):
    _, response = api
    del response["choices"]
    response["debug"] = "NEVER-RETAIN-THIS-KEY"
    out = run_target(recorder)
    assert out.error and "NEVER-RETAIN-THIS-KEY" in out.error
    assert contents(recorder, "extraction_result")[0]["output"]["error"] == "RuntimeError"
    assert_no_credentials(recorder)
