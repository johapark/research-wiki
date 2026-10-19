"""The application and CLI share the real execution boundary, without paid calls."""

import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchwiki.operations import EventRecorder, IngestOptions, IngestRequest, IngestResult, ingest
from researchwiki.operations.ingest import execute_ingest


@pytest.fixture
def ingest_request(tmp_path, monkeypatch):
    from researchwiki.env_profiles import ACTIVE_ENV_FILE_VAR

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ACTIVE_ENV_FILE_VAR, raising=False)
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    return IngestRequest(request_id="ingest_request-1", library_root=str(tmp_path), source_pdf=str(pdf),
                         options=IngestOptions(use_stub=True, n_drafts=1))


def context(ingest_request, outcome="promoted", mode="auto"):
    return SimpleNamespace(
        attempt_id="attempt-1", paper_stem="example", outcome=outcome,
        committed_path=Path(ingest_request.library_root) / (
            ".agent-output/example.md" if outcome == "sandboxed" else "wiki/other/example.md"),
        promote_mode=mode, gate_reasons=["critical claim missing"] if outcome == "sandboxed" else [],
        supplementary=None, budget_tracker=None,
    )


@pytest.mark.parametrize("outcome,mode,expected,publication", [
    ("promoted", "auto", "published", "published"),
    ("promoted", "always", "published", "published"),
    ("already_present", "auto", "already_present", "published"),
    ("sandboxed", "auto", "draft_ready", "not_published"),
    ("sandboxed", "never", "draft_ready", "not_published"),
])
def test_application_and_cli_share_terminal_results(ingest_request, monkeypatch, outcome, mode, expected, publication):
    from researchwiki.tasks import agent

    ctx = context(ingest_request, outcome, mode)
    seen = []

    def runner(path, *, event_sink, **kwargs):
        event_sink.bind_attempt(ctx.attempt_id)
        seen.append(kwargs)
        return ctx

    monkeypatch.setattr("researchwiki.agents.runner.run_ingest", runner)
    monkeypatch.setattr(agent, "run_ingest", runner)
    monkeypatch.setattr("researchwiki.categories.other_saturation_warning", lambda: None)
    ingest_request = replace(ingest_request, options=replace(ingest_request.options, promotion=mode))
    direct = ingest(ingest_request)
    report = Path(ingest_request.library_root) / "result.json"
    flags = ["--auto-promote"] if mode == "always" else ["--force-sandbox"] if mode == "never" else []
    code = agent.main(["ingest", ingest_request.source_pdf, "--stub", "-n", "1",
                       "--result-file", str(report), *flags])
    cli = IngestResult.from_dict(json.loads(report.read_text()))
    assert cli == replace(direct, request_id=cli.request_id)
    assert direct.outcome == expected and direct.publication == publication
    assert direct.post_work == "not_requested"
    assert code == (1 if outcome == "sandboxed" and mode == "auto" else 0)
    assert seen[0] == seen[1]


def test_options_reach_the_runner_without_resetting_defaults(ingest_request, monkeypatch):
    from researchwiki.agents import model_config

    monkeypatch.setattr(model_config, "default_n_drafts", lambda: 3)
    author = Path(ingest_request.library_root) / "author.md"
    author.write_text("custom instructions")
    supp = Path(ingest_request.library_root) / "table.csv"
    supp.write_text("a,b")
    options = replace(ingest_request.options, n_drafts=None, max_debug=2, max_evolve=4, promotion="never",
                      use_semantic=False, verify_claim_entailment=True, doi="10.1234/example",
                      title="Example", year=2026, authors=("A Author",), allow_rename=True,
                      supplementary=(str(supp),), author_prompt_file=str(author),
                      use_llm_reconcile=False, max_model_calls=9, max_tokens=1000,
                      max_cost_usd=1.5, max_wall_seconds=60, memory_evolve=True)
    seen = {}

    def runner(path, **kwargs):
        seen.update(kwargs)
        return context(ingest_request, "sandboxed", "never")

    result = execute_ingest(replace(ingest_request, options=options), runner=runner).result
    assert result.outcome == "draft_ready"
    assert seen['n_drafts'] == 3 and seen['max_debug'] == 2 and seen['max_evolve'] == 4
    assert seen['supplementary'] == [supp] and seen['author_prompt_override'] == str(author)
    assert seen['authors_override'] == ["A Author"] and seen['allow_rename'] is True
    assert seen['use_semantic'] is False and seen['verify_claim_entailment'] is True
    assert seen['run_memory_evolve'] is True and seen['use_llm_reconcile'] is False
    assert seen['doi_override'] == options.doi and seen['year_override'] == 2026
    assert seen['max_model_calls'] == 9 and seen['max_tokens'] == 1000
    assert seen['max_cost_usd'] == 1.5 and seen['max_wall_seconds'] == 60


@pytest.mark.parametrize("failure", ["root", "profile", "source", "supplementary", "prompt", "recorder"])
def test_invalid_runtime_does_not_change_process_or_start_paid_work(ingest_request, monkeypatch, failure):
    import os
    from researchwiki.env_profiles import ACTIVE_ENV_FILE_VAR

    original_cwd, original_env = Path.cwd(), dict(os.environ)
    changes = {}
    if failure == "root":
        changes['library_root'] = str(original_cwd / "elsewhere")
    elif failure == "profile":
        changes['provider_profile'] = str(original_cwd / "another.env")
    elif failure == "source":
        changes['source_pdf'] = str(original_cwd / "missing.pdf")
    elif failure in ("supplementary", "prompt"):
        options = ({"supplementary": (str(original_cwd / "missing.csv"),)} if failure == "supplementary"
                   else {"author_prompt_file": str(original_cwd / "missing.md")})
        changes['options'] = replace(ingest_request.options, **options)
    recorder = EventRecorder("another-ingest_request", lambda event: None) if failure == "recorder" else None
    result = execute_ingest(replace(ingest_request, **changes), event_sink=recorder,
                            runner=lambda *a, **k: pytest.fail("runner must not start"),
                            recover=lambda: pytest.fail("recovery must not start")).result
    assert result.error.code == "invalid_request"
    assert result.publication == "not_published" and result.projections == "not_applicable"
    assert Path.cwd() == original_cwd and dict(os.environ) == original_env
    assert ACTIVE_ENV_FILE_VAR not in os.environ


@pytest.mark.parametrize("recovery", ["unresolved", "exception"])
def test_recovery_failure_blocks_the_runner_and_is_structured(ingest_request, monkeypatch, recovery):
    from researchwiki import mutation

    def recover():
        if recovery == "exception":
            raise OSError("SENSITIVE_TEST_CONTENT path")
        return ["tried recovery"]

    monkeypatch.setattr(mutation, "recover_pending", recover)
    monkeypatch.setattr(mutation, "pending_journals", lambda: [{"status": "active"}])
    result = execute_ingest(ingest_request, runner=lambda *a, **k: pytest.fail("no runner after failed recovery")).result
    assert result.error.code == "environment_unavailable"
    assert result.error.retry == "after_environment_recovery"
    assert result.publication == "not_published" and "SENSITIVE_TEST_CONTENT" not in json.dumps(result.to_dict())


def test_api_preflight_failure_is_a_typed_result_before_runner(ingest_request, monkeypatch):
    from researchwiki.errors import EnvironmentFailure

    def fail():
        raise EnvironmentFailure("credential secret")

    monkeypatch.setattr("researchwiki.agents.llm.preflight_providers", fail)
    result = execute_ingest(replace(ingest_request, options=replace(ingest_request.options, use_stub=False)),
                            runner=lambda *a, **k: pytest.fail("no paid work")).result
    assert result.error.code == "environment_unavailable" and result.publication == "not_published"
    assert "secret" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("failure", ["budget", "relay", "promotion", "interrupt", "internal"])
def test_failures_keep_identity_and_uncertain_publication(ingest_request, failure):
    from researchwiki.agents.budget import BudgetExhausted
    from researchwiki.agents.context import PromoteFailed
    from researchwiki.agents.relay import RelayTimeout

    errors = {
        "budget": BudgetExhausted("model_calls", 1, 2, {}),
        "relay": RelayTimeout("SENSITIVE_TEST_CONTENT provider response"),
        "promotion": PromoteFailed(stem="example", page_path=None),
        "interrupt": KeyboardInterrupt(), "internal": RuntimeError("SENSITIVE_TEST_CONTENT paper text"),
    }
    if failure == "budget":
        errors[failure].partial_path = Path('.agent-output/partial.md')

    def runner(path, *, event_sink, **kwargs):
        event_sink.bind_attempt("attempt-failed")
        raise errors[failure]

    execution = execute_ingest(ingest_request, runner=runner)
    result = execution.result
    assert result.attempt_id == "attempt-failed" and result.publication == "unknown"
    assert result.execution == ("interrupted" if failure == "interrupt" else "finished")
    assert result.outcome == (None if failure == "interrupt" else "failed")
    assert "SENSITIVE_TEST_CONTENT" not in json.dumps(result.to_dict())
    if failure == "promotion":
        assert result.paper_stem == "example" and result.error.retry == "reconcile_first"
    if failure == "budget":
        assert result.draft_path == str(Path(ingest_request.library_root) / '.agent-output/partial.md')
        assert result.error.code == "budget_exhausted"


@pytest.mark.parametrize("mode", ["api", "relay", "mixed"])
def test_direct_service_preserves_api_relay_and_post_work_observation(ingest_request, monkeypatch, mode, capsys):
    from researchwiki.agents import llm, relay

    calls = []
    events = []
    monkeypatch.setattr(llm, "preflight_providers", lambda: None)
    monkeypatch.setattr(llm, "call_anthropic", lambda **k: llm.LLMResponse("ok", "api/model-1", 0.1, 2, 1))

    def answer(path, timeout):
        path.write_text(json.dumps({"via": "relay/model-1", "response": "ok"}))

    monkeypatch.setattr(relay, "_poll_until_exists", answer)
    monkeypatch.setattr("researchwiki.impact_review.scan", lambda key: {"state": "complete", "candidates": []})

    def model_call(provider):
        response = llm.call(model="configured", provider=provider, prompt="SENSITIVE_TEST_CONTENT source prose")
        calls.append(response.model)

    def runner(path, *, event_sink, **kwargs):
        event_sink.bind_attempt("attempt-1")
        model_call("anthropic" if mode in ("api", "mixed") else "chat-relay")
        return context(ingest_request)

    def overlap(*a, **kwargs):
        assert kwargs['render'] is False
        model_call("anthropic" if mode == "api" else "chat-relay")
        return {"applied": [], "judge_failed": []}

    monkeypatch.setattr("researchwiki.tasks.claim_overlap.run_after_ingest", overlap)
    options = replace(ingest_request.options, use_stub=False, claim_overlap=True)
    result = execute_ingest(replace(ingest_request, options=options), runner=runner, event_sink=events.append).result
    assert result.outcome == "published" and result.post_work == "completed"
    assert len(calls) == 2
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert {event.request_id for event in events} == {ingest_request.request_id}
    assert {event.attempt_id for event in events if event.attempt_id is not None} == {"attempt-1"}
    assert all(event.attempt_id == "attempt-1" for event in events
               if event.kind.startswith(("model_call_", "relay_")))
    assert "SENSITIVE_TEST_CONTENT" not in json.dumps([e.to_dict() for e in events])
    finished = [e for e in events if e.kind == "model_call_finished"]
    assert [e.model for e in finished] == calls
    assert [e.usage.status for e in finished] == ["unavailable" if e.provider == "chat-relay" else "reported" for e in finished]
    assert ("relay_waiting" in [e.kind for e in events]) == (mode != "api")
    if mode != "api":
        assert relay.HANDOFF_PREFIX in capsys.readouterr().err


@pytest.mark.parametrize("hook_failure", ["exception", "budget", "unavailable", "judge"])
def test_optional_hook_failure_never_repeats_or_relabels_publication(ingest_request, monkeypatch, hook_failure):
    from researchwiki.agents.budget import BudgetExhausted, BudgetTracker, IngestBudget

    monkeypatch.setattr("researchwiki.agents.llm.preflight_providers", lambda: None)
    monkeypatch.setattr("researchwiki.impact_review.scan", lambda key: {"state": "complete", "candidates": []})
    tracker = BudgetTracker(IngestBudget(max_model_calls=1))
    tracker.suspend()
    calls = []

    def runner(*args, **kwargs):
        calls.append(1)
        ctx = context(ingest_request)
        ctx.budget_tracker = tracker
        return ctx

    def hook(*a, **k):
        assert not tracker.suspended
        if hook_failure == "exception":
            raise RuntimeError("SENSITIVE_TEST_CONTENT source")
        if hook_failure == "budget":
            raise BudgetExhausted("model_calls", 1, 2, {})
        return None if hook_failure == "unavailable" else {"judge_failed": ["missing"]}

    monkeypatch.setattr("researchwiki.tasks.claim_overlap.run_after_ingest", hook)
    options = replace(ingest_request.options, use_stub=False, claim_overlap=True)
    result = execute_ingest(replace(ingest_request, options=options), runner=runner).result
    assert calls == [1] and tracker.suspended
    assert result.outcome == "published" and result.publication == "published"
    assert result.post_work == "partial" and result.error is None and result.warnings
    assert "SENSITIVE_TEST_CONTENT" not in json.dumps(result.to_dict())


def test_incomplete_impact_scan_is_preserved_without_republication(ingest_request, monkeypatch):
    monkeypatch.setattr("researchwiki.agents.llm.preflight_providers", lambda: None)

    def fail(*a):
        raise OSError("SENSITIVE_TEST_CONTENT path")

    monkeypatch.setattr("researchwiki.impact_review.scan", fail)
    monkeypatch.setattr("researchwiki.impact_review.record_unscanned", fail)
    result = execute_ingest(replace(ingest_request, options=replace(ingest_request.options, use_stub=False)),
                            runner=lambda *a, **k: context(ingest_request)).result
    assert result.outcome == "published" and result.post_work == "partial"
    assert "SENSITIVE_TEST_CONTENT" not in json.dumps(result.to_dict())


def test_public_entry_point_remains_callable_after_repeated_module_imports(ingest_request, monkeypatch):
    import researchwiki.operations as operations

    monkeypatch.setattr("researchwiki.agents.runner.run_ingest", lambda *a, **k: context(ingest_request))
    assert operations.ingest(ingest_request).outcome == "published"
    importlib.import_module('researchwiki.operations.ingest')
    assert operations.ingest(ingest_request).outcome == "published"


def test_relay_only_operation_needs_no_api_credentials(ingest_request, monkeypatch):
    from researchwiki.agents import llm, model_config

    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    monkeypatch.setattr(model_config, "config_path", lambda: Path(ingest_request.library_root) / 'absent.yaml')
    model_config.clear_caches()
    try:
        assert llm.missing_provider_credentials() == []
        result = execute_ingest(
            replace(ingest_request, options=replace(ingest_request.options, use_stub=False)),
            runner=lambda *a, **k: context(ingest_request, "already_present"),
        ).result
    finally:
        model_config.clear_caches()
    assert result.outcome == "already_present" and result.error is None


def test_interrupt_after_runner_publication_preserves_published_state(ingest_request, monkeypatch):
    module = importlib.import_module('researchwiki.operations.ingest')

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "run_post_work", interrupt)
    result = execute_ingest(ingest_request, runner=lambda *a, **k: context(ingest_request)).result
    assert result.execution == "interrupted" and result.outcome == "published"
    assert result.publication == "published" and result.error.code == "interrupted"
    assert result.error.retry == "reconcile_first"


def test_cli_invalid_options_remain_an_input_error(ingest_request, monkeypatch):
    from researchwiki.tasks import agent

    monkeypatch.setattr(agent, "run_ingest", lambda *a, **k: pytest.fail("invalid option reached runner"))
    assert agent.main(["ingest", ingest_request.source_pdf, "--stub", "-n", "0"]) == 1
