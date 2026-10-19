"""Lifecycle receipts follow real canonical commit boundaries, never log text."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from researchwiki.agents import promote, runner
from researchwiki.agents.context import Context
from researchwiki.agents.publication import finish_publication
from researchwiki.operations import ContractError, EventRecorder, IngestEvent, IngestOptions, IngestRequest
from researchwiki.operations.ingest import execute_ingest
from researchwiki.operations.lifecycle import IngestFacts, gate_evaluated, observe_lifecycle
from researchwiki.operations.observation import observe_models, submit_observed


@pytest.fixture
def library(tmp_path, monkeypatch):
    from researchwiki.env_profiles import ACTIVE_ENV_FILE_VAR

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ACTIVE_ENV_FILE_VAR, raising=False)
    for folder in ('wiki/other', 'inbox', 'papers'):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / 'wiki/index.md').write_text('# Index\n\n## other\n')
    source = tmp_path / 'inbox/paper.pdf'
    source.write_bytes(b'%PDF-1.4 test source')
    monkeypatch.setattr(promote, '_suggest_category', lambda *a, **k: ('other', 'strong'))
    return IngestRequest(
        request_id='request-commit', library_root=str(tmp_path), source_pdf=str(source),
        options=IngestOptions(use_stub=True, n_drafts=1),
    )


def publish(request, *, event_sink, gate_reasons=()):
    event_sink.bind_attempt('attempt-commit')
    ctx = Context(attempt_id='attempt-commit', pdf_path=Path(request.source_pdf), pdf_filename='paper.pdf')
    ctx.paper_stem = 'smith-2026-a-test-paper'
    ctx.winner = SimpleNamespace(scores={'n_graded': 5, 'n_drift': 0})
    gate_evaluated(ctx, promote.GateResult(promoted=True, reasons=list(gate_reasons)))
    result = promote.promote_to_wiki(
        stem=ctx.paper_stem, draft_text='## Summary\nA source-grounded test.\n\n## Key Contributions\n- Alpha improves speed.\n',
        metadata={'title': 'A test paper', 'authors': 'Smith', 'year': 2026, 'doi': '10.1234/test'},
        candidates=[], source_pdf_path=ctx.pdf_path, attempt_id=ctx.attempt_id,
        author_model='test-model', use_llm_classify=False,
    )
    if not result.promoted:
        from researchwiki.agents.context import PromoteFailed
        raise PromoteFailed(stem=ctx.paper_stem, page_path=result.wiki_path, warnings=result.warnings)
    return ctx, result


def complete(ctx, result, *, index=None, grade=None, memory=None):
    finish_publication(
        ctx, None, result,
        indexer=index or (lambda res: {'bm25_mode': 'incremental', 'semantic': {'mode': 'incremental'}}),
        grader=grade or (lambda *a: {'n_claims': 1, 'n_graded': 1}),
        memory=memory or (lambda *a, **k: None), writer=lambda **k: None,
    )
    ctx.outcome = 'promoted'
    ctx.committed_path = result.wiki_path
    return ctx


def test_real_commit_precedes_maintenance_and_terminal_events(library):
    events = []

    def execute(path, *, event_sink, **kwargs):
        ctx, result = publish(library, event_sink=event_sink)
        return complete(ctx, result)

    result = execute_ingest(library, runner=execute, event_sink=events.append).result
    assert result.outcome == 'published' and result.projections == 'current'
    kinds = [e.kind for e in events]
    assert kinds[0] == 'operation_started' and kinds[-1] == 'operation_finished'
    publication = next(e for e in events if e.kind == 'publication_committed')
    assert publication.publication == 'published' and publication.projections == 'maintenance_pending'
    assert kinds.index('publication_committed') < next(i for i, e in enumerate(events) if e.phase == 'index_update')
    assert kinds.index('gate_evaluated') < kinds.index('publication_committed')
    assert events[-1].projections == 'current'
    assert [e.sequence for e in events] == list(range(1, len(events) + 1))
    assert all(e.schema_version == 2 for e in events)
    assert {e.attempt_id for e in events if e.attempt_id} == {'attempt-commit'}
    assert Path(result.pdf_path).read_bytes() == b'%PDF-1.4 test source'
    assert 'source-grounded' not in json.dumps([e.to_dict() for e in events])


@pytest.mark.parametrize('fault', ['rollback', 'mark'])
def test_no_publication_event_before_successful_commit(library, monkeypatch, fault):
    def fail(*args, **kwargs):
        raise OSError('SENSITIVE_FAILURE_CONTENT')

    if fault == 'rollback':
        monkeypatch.setattr(promote, '_append_index_entry', fail)
    else:
        monkeypatch.setattr('researchwiki.mutation.Snapshot.mark_committed', fail)
    events = []
    result = execute_ingest(library, runner=lambda path, *, event_sink, **kw:
                            publish(library, event_sink=event_sink), event_sink=events.append).result
    assert result.publication == 'unknown' and result.outcome == 'failed'
    assert not any(e.kind == 'publication_committed' for e in events)
    assert Path(library.source_pdf).exists()
    assert not (Path(library.library_root) / 'papers/smith-2026-a-test-paper.pdf').exists()
    assert 'SENSITIVE_FAILURE_CONTENT' not in json.dumps([e.to_dict() for e in events])


@pytest.mark.parametrize('failure', ['index', 'semantic', 'grades', 'database'])
def test_failed_maintenance_preserves_canonical_publication(library, monkeypatch, failure):
    if failure == 'database':
        monkeypatch.setattr('researchwiki.mutation.Snapshot.reconcile',
                            lambda *a: (_ for _ in ()).throw(OSError('SENSITIVE_FAILURE_CONTENT')))

    def execute(path, *, event_sink, **kwargs):
        ctx, result = publish(library, event_sink=event_sink)
        return complete(
            ctx, result,
            index=(lambda res: None) if failure == 'index' else
                  (lambda res: {'bm25_mode': 'incremental', 'semantic': None}) if failure == 'semantic' else None,
            grade=(lambda *a: {'error': 'SENSITIVE_FAILURE_CONTENT'}) if failure == 'grades' else None,
        )

    events = []
    result = execute_ingest(library, runner=execute, event_sink=events.append).result
    assert result.outcome == 'published' and result.projections == 'maintenance_pending'
    assert result.warnings and Path(result.page_path).exists()
    assert events[-1].publication == 'published' and events[-1].projections == 'maintenance_pending'
    assert any(e.kind == 'maintenance_finished' and e.status == 'failed' for e in events)
    assert 'SENSITIVE_FAILURE_CONTENT' not in json.dumps(result.to_dict())


@pytest.mark.parametrize('sink', ['normal', 'absent', 'broken'])
def test_exception_after_commit_is_known_even_without_event_delivery(library, sink):
    calls, events = [], []

    def fail_sink(event):
        raise OSError('SENSITIVE_FAILURE_CONTENT')

    def execute(path, *, event_sink, **kwargs):
        calls.append(1)
        publish(library, event_sink=event_sink, gate_reasons=('numeric drift',))
        raise OSError('SENSITIVE_FAILURE_CONTENT')

    callback = events.append if sink == 'normal' else fail_sink if sink == 'broken' else None
    execution = execute_ingest(library, runner=execute, event_sink=callback)
    result = execution.result
    assert calls == [1] and execution.context is None
    assert result.outcome == 'published' and result.publication == 'published'
    assert result.projections == 'maintenance_pending' and result.error.retry == 'reconcile_first'
    assert result.gate_reasons == ('numeric drift',)
    assert result.checks and Path(result.page_path).exists() and Path(result.pdf_path).exists()
    assert 'SENSITIVE_FAILURE_CONTENT' not in json.dumps(result.to_dict())


def test_journal_records_request_attempt_and_job_before_canonical_writes(library, monkeypatch):
    captured = []
    original = promote._finish_promote

    def inspect(**kwargs):
        from researchwiki.mutation import pending_journals
        captured.extend(pending_journals())
        return original(**kwargs)

    monkeypatch.setattr(promote, '_finish_promote', inspect)

    def execute(path, *, event_sink, **kwargs):
        return complete(*publish(library, event_sink=event_sink))

    recorder = EventRecorder(library.request_id, lambda event: None, job_id='job-test')
    result = execute_ingest(library, runner=execute, event_sink=recorder).result
    assert result.job_id == 'job-test'
    assert captured[0]['status'] == 'active'
    assert captured[0]['details']['request_id'] == library.request_id
    assert captured[0]['details']['attempt_id'] == 'attempt-commit'
    assert captured[0]['details']['job_id'] == 'job-test'


def test_cli_does_not_return_retryable_failure_after_known_commit(library, monkeypatch, capsys):
    from researchwiki.tasks import agent

    calls = []

    def execute(path, *, event_sink, **kwargs):
        calls.append(1)
        publish(library, event_sink=event_sink)
        raise OSError('maintenance interrupted')

    monkeypatch.setattr(agent, 'run_ingest', execute)
    output = Path(library.library_root) / 'result.json'
    assert agent.main(['ingest', library.source_pdf, '--stub', '--result-file', str(output)]) == 0
    result = json.loads(output.read_text())
    assert calls == [1] and result['publication'] == 'published'
    assert result['error']['retry'] == 'reconcile_first'
    assert 'canonical content preserved' in capsys.readouterr().out


def test_parallel_native_author_phases_keep_slot_identity(library, monkeypatch):
    from researchwiki.agents import phases
    from researchwiki.agents.phases.draft import Draft

    barrier = Barrier(2)

    def author(**kwargs):
        barrier.wait(timeout=5)
        return Draft(text='SENSITIVE_DRAFT_BODY', model='test-model', temperature=kwargs['temperature'],
                     input_tokens=1, output_tokens=1)

    monkeypatch.setattr(phases, 'author', author)
    monkeypatch.setattr(runner, 'write_iteration', lambda **kwargs: kwargs['iteration'])
    events = []
    recorder = EventRecorder(library.request_id, events.append, attempt_id='attempt-1')
    with ThreadPoolExecutor(2) as pool, observe_models(recorder):
        futures = []
        for slot in range(2):
            ctx = Context(attempt_id='attempt-1', pdf_path=Path(library.source_pdf), pdf_filename='paper.pdf', iteration=slot+3)
            futures.append(submit_observed(pool, runner._phase_author, ctx, None, temperature=0.2, slot=slot))
        for future in futures:
            future.result(timeout=10)
    starts = [e for e in events if e.kind == 'phase_started']
    finishes = [e for e in events if e.kind == 'phase_finished']
    assert {e.slot for e in starts} == {0, 1}
    assert {e.iteration for e in starts} == {3, 4}
    assert {e.call_id for e in starts} == {e.call_id for e in finishes}
    assert len({e.call_id for e in starts}) == 2
    assert [e.sequence for e in events] == list(range(1, 5))
    assert 'SENSITIVE_DRAFT_BODY' not in json.dumps([e.to_dict() for e in events])


def test_v1_event_reader_preserves_legacy_wire_shape():
    legacy = dict(request_id='r', sequence=1, timestamp='2026-10-19T00:00:00Z',
                  kind='model_call_started', call_id='c', schema_version=1)
    parsed = IngestEvent.from_dict(legacy)
    assert parsed.to_dict()['schema_version'] == 1
    assert 'status' not in parsed.to_dict()
    assert IngestEvent.from_dict(parsed.to_dict()) == parsed
    with pytest.raises(ContractError):
        IngestEvent.from_dict({**legacy, 'kind': 'publication_committed'})
    with pytest.raises(ContractError):
        IngestEvent.from_dict({**legacy, 'status': 'completed'})
    with pytest.raises(ContractError):
        IngestEvent.from_dict({**legacy, 'schema_version': 3})


def test_mismatched_recorder_cannot_receive_another_requests_events(library):
    events = []
    recorder = EventRecorder('another-request', events.append, attempt_id='another-attempt', job_id='another-job')
    result = execute_ingest(library, event_sink=recorder,
                            runner=lambda *a, **k: pytest.fail('invalid recorder reached runner')).result
    assert result.error.code == 'invalid_request' and result.request_id == library.request_id
    assert result.attempt_id is None and result.job_id is None and events == []


@pytest.mark.parametrize('output,state', [
    ({'n_above_threshold': 0, 'n_judged': 0, 'n_cached_skipped': 0, 'written': []}, 'completed'),
    ({'n_above_threshold': 2, 'n_judged': 1, 'n_cached_skipped': 0, 'written': []}, 'partial'),
    ({'n_above_threshold': 2, 'n_judged': 1, 'n_cached_skipped': 1, 'written': []}, 'completed'),
    ({'skipped': 'stub-mode'}, 'partial'), ({'error': 'private'}, 'failed'), (None, 'unknown'),
])
def test_memory_completion_requires_an_authoritative_receipt(library, monkeypatch, output, state):
    from researchwiki.agents import runner_support

    monkeypatch.setattr(runner_support.phases, 'evolve_memory', lambda *a, **k: output)
    facts = IngestFacts()
    ctx = Context(attempt_id='attempt-1', pdf_path=Path(library.source_pdf), pdf_filename='paper.pdf')
    with observe_lifecycle(facts):
        runner_support.run_post_promote_memory_evolution(ctx, None, source_key='other/example')
    assert facts.memory_work == state


def test_phase_failure_is_paired_and_scopes_are_restored(library):
    from researchwiki.operations.lifecycle import current_facts, phase
    from researchwiki.operations.observation import current_recorder

    events = []
    with observe_models(EventRecorder('r', events.append)), observe_lifecycle(IngestFacts()):
        with pytest.raises(RuntimeError), phase('test-stage'):
            raise RuntimeError('SENSITIVE_FAILURE_CONTENT')
    assert [e.kind for e in events] == ['phase_started', 'phase_failed']
    assert events[0].call_id == events[1].call_id and events[1].error_code == 'RuntimeError'
    assert current_facts() is None and current_recorder() is None
    assert 'SENSITIVE_FAILURE_CONTENT' not in json.dumps([e.to_dict() for e in events])
