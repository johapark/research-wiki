"""Exercise API dispatch and the real relay mailbox with deterministic responders."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest

from researchwiki.agents import llm, model_config, relay
from researchwiki.operations import EventRecorder, observe_models
from researchwiki.operations.observation import current_recorder, submit_observed


@pytest.fixture
def mailbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(relay, "_current_pdf", None)
    monkeypatch.setattr(relay, "_current_stem", None)
    return tmp_path


def respond(monkeypatch, *, structured=None):
    def answer(path, timeout):
        data = {"schema_version": 1, "op_id": path.name.split(".")[0], "via": "test/model-1"}
        data.update({"response": "answer"} if structured is None else {"structured": structured})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    monkeypatch.setattr(relay, "_poll_until_exists", answer)


@pytest.mark.parametrize("provider", ["anthropic", "openai-compatible", "lmstudio"])
def test_api_progress_reports_actual_model_and_usage_without_prompt_text(monkeypatch, provider):
    events = []
    response = llm.LLMResponse("confidential answer", "actual-model", 0.2, 9, 3)
    monkeypatch.setattr(llm, "call_anthropic", lambda **kwargs: response)
    monkeypatch.setattr(llm, "call_openai_compatible", lambda **kwargs: response)
    got = llm.call(model="model-1", provider=provider, prompt="secret prompt",
                   system="secret system", event_sink=EventRecorder("r", events.append))
    assert got is response
    assert [e.kind for e in events] == ["model_call_started", "model_call_finished"]
    assert events[-1].model == "actual-model"
    assert events[-1].usage.input_tokens == 9
    assert events[-1].usage.status == "reported"
    assert events[0].call_id == events[-1].call_id
    serialized = json.dumps([e.to_dict() for e in events])
    assert all(text not in serialized for text in ("secret", "confidential"))


def test_relay_wait_is_visible_before_response_and_preserves_via(mailbox, monkeypatch, capsys):
    events = []
    recorder = EventRecorder("r", events.append)

    def answer(path, timeout):
        handoff, = recorder.pending_relays
        assert handoff.response_path == str(path.resolve())
        assert handoff.timeout_seconds == timeout
        prompt = json.loads(Path(handoff.prompt_path).read_text())
        assert prompt["schema_version"] == 1
        assert prompt["op_id"] == handoff.op_id
        assert prompt["prompt"] == "secret prompt"
        path.write_text(json.dumps({"via": "test/model-1", "response": "answer"}))

    monkeypatch.setattr(relay, "_poll_until_exists", answer)
    got = llm.call(model="configured-hint", provider="chat-relay", prompt="secret prompt",
                   event_sink=recorder)
    assert got.model == "chat-relay:test/model-1"
    assert [e.kind for e in events] == [
        "model_call_started", "relay_waiting", "relay_response_received", "model_call_finished",
    ]
    assert len({e.call_id for e in events}) == 1
    assert events[-1].usage.status == "unavailable"  # relay's zero counters aren't measured usage
    assert recorder.pending_relays == ()
    assert "secret prompt" not in json.dumps([e.to_dict() for e in events])
    assert relay.HANDOFF_PREFIX in capsys.readouterr().err  # old foreground protocol remains
    assert not list((mailbox / ".llm-relay/pending").glob("*.json"))


def test_relay_timeout_keeps_pending_prompt_without_automatic_retry(mailbox, monkeypatch):
    events = []
    recorder = EventRecorder("r", events.append)
    polls = []

    def timeout(path, seconds):
        polls.append(path)
        raise relay.RelayTimeout("responder absent")

    monkeypatch.setattr(relay, "_poll_until_exists", timeout)
    with pytest.raises(relay.RelayTimeout):
        llm.call(model="hint", provider="chat-relay", prompt="hello", event_sink=recorder)
    assert len(polls) == 1
    assert [e.kind for e in events][-2:] == ["relay_timeout", "model_call_failed"]
    assert Path(events[-2].relay.prompt_path).exists()
    assert recorder.pending_relays == ()  # mailbox remains, but no active waiter


def test_schema_retry_has_new_handoff_but_keeps_call_identity(mailbox, monkeypatch):
    events = []
    recorder = EventRecorder("r", events.append)
    replies = iter(({"invalid": True}, {"ok": True}))

    def answer(path, timeout):
        path.write_text(json.dumps({"via": "test/model-1", "structured": next(replies)}))

    monkeypatch.setattr(relay, "_poll_until_exists", answer)
    llm.call(model="hint", provider="chat-relay", prompt="hello", event_sink=recorder,
             schema={"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}})
    waits = [e for e in events if e.kind == "relay_waiting"]
    assert len(waits) == 2
    assert waits[1].relay.retry_of == waits[0].relay.op_id
    assert waits[1].relay.op_id != waits[0].relay.op_id
    assert len({e.call_id for e in events}) == 1
    assert any(e.kind == "relay_response_rejected" for e in events)
    assert recorder.pending_relays == ()


def test_cached_relay_response_does_not_invent_a_wait(mailbox, monkeypatch):
    events = []
    op_id = relay._stable_op_id(None, "cached prompt")
    _, response = relay._paths_for(op_id)
    response.parent.mkdir(parents=True, exist_ok=True)
    response.write_text(json.dumps({"via": "test/model-1", "response": "cached"}))
    llm.call(model="hint", provider="chat-relay", prompt="cached prompt",
             event_sink=EventRecorder("r", events.append))
    received, = [e for e in events if e.kind == "relay_response_received"]
    assert received.relay.response_reused
    assert all(e.kind != "relay_waiting" for e in events)


def test_one_observation_scope_supports_mixed_per_phase_routing(mailbox, monkeypatch):
    base = model_config.for_phase("author")
    monkeypatch.setattr(model_config, "for_phase", lambda phase: replace(
        base, model="model-1", provider="chat-relay" if phase == "critic" else "anthropic", rpm=None,
    ))
    monkeypatch.setattr(llm, "call_anthropic", lambda **kwargs: llm.LLMResponse("api", "api-model", 0.2, 4, 2))
    respond(monkeypatch)
    events = []
    with observe_models(EventRecorder("r", events.append, attempt_id="attempt-1")):
        llm.call(phase="author", prompt="a")
        llm.call(phase="critic", prompt="b")
    assert current_recorder() is None
    assert {e.provider for e in events if e.provider is not None} == {"anthropic", "chat-relay"}
    assert all(e.attempt_id == "attempt-1" for e in events)


def test_parallel_slots_keep_order_and_observers_do_not_leak(monkeypatch):
    monkeypatch.setattr(llm, "call_anthropic", lambda **kwargs: llm.LLMResponse("api", "model-1", 0.2, 1, 1))
    events = []
    recorder = EventRecorder("r", events.append)
    with ThreadPoolExecutor(4) as pool:
        with observe_models(recorder):
            futures = [submit_observed(pool, llm.call, model="model-1", provider="anthropic", prompt=str(i))
                       for i in range(12)]
            for future in futures:
                future.result()
        assert pool.submit(current_recorder).result() is None
    assert [e.sequence for e in events] == list(range(1, 25))
    assert len({e.call_id for e in events}) == 12


def test_parallel_relay_slots_expose_both_owned_mailboxes(mailbox, monkeypatch):
    events = []
    recorder = EventRecorder("r", events.append)
    arrived, inspected = Barrier(2), Barrier(2)
    snapshots = []

    def answer(path, timeout):
        arrived.wait(timeout=5)
        snapshots.append({handoff.op_id for handoff in recorder.pending_relays})
        inspected.wait(timeout=5)
        path.write_text(json.dumps({"via": "test/model-1", "response": "answer"}))

    monkeypatch.setattr(relay, "_poll_until_exists", answer)
    with ThreadPoolExecutor(2) as pool, observe_models(recorder):
        futures = [submit_observed(pool, llm.call, model="hint", provider="chat-relay", prompt=f"slot-{i}")
                   for i in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert all(len(snapshot) == 2 for snapshot in snapshots)
    assert snapshots[0] == snapshots[1]
    assert recorder.pending_relays == ()
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert len({event.call_id for event in events}) == 2


def test_sink_failure_never_consumes_a_second_relay_response(mailbox, monkeypatch, capsys):
    def broken(event):
        raise OSError("secret response body")

    recorder = EventRecorder("r", broken)
    respond(monkeypatch)
    response = llm.call(model="hint", provider="chat-relay", prompt="hello", event_sink=recorder)
    assert response.text == "answer"
    assert recorder.delivery_errors == 4
    assert "secret" not in capsys.readouterr().err


def test_api_failure_is_observed_without_changing_exception_or_exposing_details(monkeypatch):
    failure = llm.ProviderUnavailable("secret provider body")

    def fail(**kwargs):
        raise failure

    monkeypatch.setattr(llm, "call_anthropic", fail)
    events = []
    with pytest.raises(llm.ProviderUnavailable) as exc:
        llm.call(model="model-1", provider="anthropic", prompt="hello",
                 event_sink=EventRecorder("r", events.append))
    assert exc.value is failure
    assert [e.kind for e in events] == ["model_call_started", "model_call_failed"]
    assert events[-1].error_code == "ProviderUnavailable"
    assert "secret" not in json.dumps([e.to_dict() for e in events])


def test_runner_binds_attempt_and_observes_model_calls_in_a_phase(mailbox, monkeypatch):
    from researchwiki.agents import runner

    original = model_config.for_phase("author")
    monkeypatch.setattr(model_config, "for_phase", lambda phase: replace(
        original, model="model-1", provider="chat-relay" if phase == "reconcile" else "anthropic", rpm=None,
    ))
    monkeypatch.setattr(llm, "call_anthropic", lambda **kwargs: llm.LLMResponse("api", "api-model", 0.2, 4, 2))
    respond(monkeypatch)

    def reconcile(ctx, conn):
        llm.call(phase="reconcile", prompt="metadata")
        llm.call(phase="author", prompt="api helper")
        return {"stem": "example", "year": 2026}, None

    monkeypatch.setattr(runner, "_phase_reconcile", reconcile)
    monkeypatch.setattr(runner, "_find_exact_duplicate", lambda *args: ("example", mailbox / "wiki/other/example.md"))
    events = []
    recorder = EventRecorder("r", events.append)
    ctx = runner.run_ingest(mailbox / "incoming.pdf", event_sink=recorder)
    assert ctx.outcome == "already_present"
    assert {e.attempt_id for e in events} == {ctx.attempt_id}
    assert {e.provider for e in events if e.provider is not None} == {"anthropic", "chat-relay"}
    assert current_recorder() is None


def test_nested_observation_failure_restores_prior_scope(monkeypatch):
    outer = EventRecorder("outer", lambda event: None)
    inner = EventRecorder("inner", lambda event: None)
    with observe_models(outer):
        with pytest.raises(RuntimeError), observe_models(inner):
            assert current_recorder() is inner
            raise RuntimeError("failed phase")
        assert current_recorder() is outer
    assert current_recorder() is None
