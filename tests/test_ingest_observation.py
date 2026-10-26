"""The normal CLI can observe API and relay ingestion without replacing it."""

import json
from types import SimpleNamespace

import pytest

from researchwiki.agents import llm, relay
from researchwiki.operations import IngestEvent, IngestResult
from researchwiki.tasks import agent


@pytest.mark.parametrize("provider", ["anthropic", "chat-relay"])
def test_cli_reports_preserve_real_dispatch_and_typed_result(tmp_path, monkeypatch, provider):
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "incoming.pdf"
    pdf.write_bytes(b"%PDF")
    events_path, result_path = tmp_path / "events.jsonl", tmp_path / "result.json"
    monkeypatch.setattr(agent, "_drain_pending_mutations", lambda: None)
    monkeypatch.setattr(agent, "_finish_ingest", lambda ctx, args: 0)
    monkeypatch.setattr(llm, "call_anthropic", lambda **kw: llm.LLMResponse("answer", "actual-model", 0.2, 2, 1))

    def answer(path, timeout):
        path.write_text(json.dumps({"via": "test/model-1", "response": "answer"}))

    monkeypatch.setattr(relay, "_poll_until_exists", answer)

    def runner(path, *, event_sink, **kwargs):
        event_sink.bind_attempt("attempt-test")
        llm.call(model="configured-model", provider=provider, prompt="private paper text", event_sink=event_sink)
        return SimpleNamespace(
            outcome="promoted", paper_stem="example", attempt_id="attempt-test",
            committed_path=tmp_path / "wiki/other/example.md", promote_mode="auto", gate_reasons=[],
        )

    monkeypatch.setattr(agent, "run_ingest", runner)
    assert agent.main(["ingest", str(pdf), "--stub", "-n", "1",
                       "--events-file", str(events_path), "--result-file", str(result_path)]) == 0
    events = [IngestEvent.from_dict(json.loads(line)) for line in events_path.read_text().splitlines()]
    result = IngestResult.from_dict(json.loads(result_path.read_text()))
    assert result.outcome == "published" and result.publication == "published"
    assert result.projections == "unknown"  # don't invent maintenance success
    assert {event.request_id for event in events} == {result.request_id}
    assert {event.attempt_id for event in events if event.attempt_id is not None} == {result.attempt_id}
    assert all(event.attempt_id == result.attempt_id for event in events
               if event.kind.startswith(("model_call_", "relay_")))
    assert "private paper text" not in events_path.read_text()
    assert ("relay_waiting" in [event.kind for event in events]) == (provider == "chat-relay")


def test_relay_timeout_result_does_not_claim_nothing_was_published(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    pdf = tmp_path / "incoming.pdf"
    pdf.write_bytes(b"%PDF")
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(agent, "_drain_pending_mutations", lambda: None)

    def runner(*args, event_sink, **kwargs):
        event_sink.bind_attempt("attempt-test")
        raise relay.RelayTimeout("responder unavailable")

    monkeypatch.setattr(agent, "run_ingest", runner)
    from researchwiki.__main__ import main
    assert main(["agent", "ingest", str(pdf), "--stub", "--result-file", str(result_path)]) == 2
    result = IngestResult.from_dict(json.loads(result_path.read_text()))
    assert result.outcome == "failed" and result.publication == "unknown"
    assert result.error.code == "relay_timeout"
    assert result.error.retry == "after_environment_recovery"


@pytest.mark.parametrize("extra", [["--workers", "1"], ["--resume", "a-batch"]])
def test_reports_reject_batch_without_creating_files(tmp_path, extra):
    report = tmp_path / "report.json"
    assert agent.main(["ingest", str(tmp_path / "input.pdf"), "--result-file", str(report), *extra]) == 1
    assert not report.exists()


def test_report_cannot_overwrite_existing_file_or_canonical_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "incoming.pdf"
    pdf.write_bytes(b"keep-this-pdf")
    assert agent.main(["ingest", str(pdf), "--result-file", str(pdf)]) == 1
    assert pdf.read_bytes() == b"keep-this-pdf"
    target = tmp_path / "wiki/result.json"
    assert agent.main(["ingest", str(pdf), "--result-file", str(target)]) == 1
    assert not target.exists()


def test_reports_cannot_share_a_path(tmp_path):
    path = str(tmp_path / "report.json")
    assert agent.main(["ingest", str(tmp_path / "input.pdf"), "--events-file", path,
                       "--result-file", path]) == 1
    assert not (tmp_path / "report.json").exists()


def test_post_runner_model_work_keeps_the_same_observation_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "incoming.pdf"
    pdf.write_bytes(b"%PDF")
    report = tmp_path / "events.jsonl"
    monkeypatch.setattr(agent, "_drain_pending_mutations", lambda: None)
    monkeypatch.setattr(llm, "call_anthropic", lambda **kw: llm.LLMResponse("answer", "model-1", 0.2, 1, 1))

    def runner(*args, event_sink, **kwargs):
        event_sink.bind_attempt("attempt-test")
        return SimpleNamespace(
            outcome="already_present", attempt_id="attempt-test", paper_stem="example",
            committed_path=tmp_path / "wiki/other/example.md", promote_mode="auto", gate_reasons=[],
        )

    def finish(ctx, args):
        llm.call(model="model-1", provider="anthropic", prompt="post-run work")
        return 0

    monkeypatch.setattr(agent, "run_ingest", runner)
    monkeypatch.setattr(agent, "_finish_ingest", finish)
    assert agent.main(["ingest", str(pdf), "--stub", "--events-file", str(report)]) == 0
    events = [IngestEvent.from_dict(json.loads(line)) for line in report.read_text().splitlines()]
    assert [event.kind for event in events if event.kind.startswith("model_call_")] == ["model_call_started", "model_call_finished"]
    assert {event.attempt_id for event in events if event.attempt_id is not None} == {"attempt-test"}


def test_progress_write_and_close_failure_do_not_repeat_or_fail_ingest(tmp_path, monkeypatch, capsys):
    from researchwiki.tasks._ingest_observation import CliObservation

    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "incoming.pdf"
    pdf.write_bytes(b"%PDF")
    result_path = tmp_path / "result.json"
    calls = []
    writes = []
    monkeypatch.setattr(agent, "_drain_pending_mutations", lambda: None)
    monkeypatch.setattr(agent, "_finish_ingest", lambda ctx, args: 0)

    class BrokenStream:
        def write(self, text):
            writes.append(text)
            raise OSError("confidential details")

        def close(self):
            raise OSError("confidential details")

    open_original = CliObservation.open

    def open_with_broken_progress(self):
        open_original(self)
        self.stream = BrokenStream()

    monkeypatch.setattr(CliObservation, "open", open_with_broken_progress)

    def api(**kwargs):
        calls.append(1)
        return llm.LLMResponse("answer", "model-1", 0.2, 1, 1)

    def runner(*args, event_sink, **kwargs):
        event_sink.bind_attempt("attempt-test")
        llm.call(model="model-1", provider="anthropic", prompt="hello", event_sink=event_sink)
        return SimpleNamespace(
            outcome="promoted", attempt_id="attempt-test", paper_stem="example",
            committed_path=tmp_path / "wiki/other/example.md", promote_mode="auto", gate_reasons=[],
        )

    monkeypatch.setattr(llm, "call_anthropic", api)
    monkeypatch.setattr(agent, "run_ingest", runner)
    assert agent.main(["ingest", str(pdf), "--stub", "--result-file", str(result_path)]) == 0
    result = IngestResult.from_dict(json.loads(result_path.read_text()))
    assert result.outcome == "published"
    assert len(calls) == 1
    assert result.warnings == (f"{len(writes)} progress event delivery failure(s)",)
    assert "confidential" not in capsys.readouterr().err
