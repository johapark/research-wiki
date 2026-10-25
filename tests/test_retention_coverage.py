"""Invocation and call coverage must survive aggregation and archive relocation."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from researchwiki.agents import llm, model_config, relay, runner
from researchwiki.agents.phases.revise import CritiqueOutput
from researchwiki.agents.promote import GateResult
from researchwiki.agents.phases.crosslinks import VerificationReport
from researchwiki.operations.lifecycle import observed_phase
from researchwiki.operations.observation import submit_observed
from researchwiki.records import AttemptSpec, RecordStore, RecordStoreUnavailable
from researchwiki.records.coverage import audit_coverage
from researchwiki.records.export import export_attempt
from researchwiki.records.grading import GradingRun
from researchwiki.records.jsonio import decode_json
from researchwiki.records.lineage import PHASES, note_call_omission, note_phase_omission
from researchwiki.records.observation import observe_call
from researchwiki.tasks.records import main as records_main
from tests.test_draft_lineage_retention import (
    draft, lineage as lineage, library as library, terminal,
)
from tests.test_grading_retention import grading as grading, grade, run_support
from tests.test_ingest_records_store import artifact as old_artifact, finish as old_finish


def test_one_author_invocation_cannot_cover_an_unexplained_sibling(lineage):
    _, ctx, owner = lineage
    @observed_phase('author')
    def produce(ctx, conn, temperature, slot=0):
        if slot == 0:
            with observe_call('author', uuid.uuid4().hex, {'model': 'fixture'}) as capture:
                capture.answer('observed answer', model='fixture')
        return draft()
    produce(ctx, None, 0.2, slot=0)
    produce(ctx, None, 0.2, slot=1)
    assert terminal(owner, complete_fixture=True).capture_state == 'incomplete'


def test_one_logical_call_cannot_cover_another_missing_answer(lineage):
    _, ctx, owner = lineage
    @observed_phase('author')
    def produce(ctx, conn, temperature):
        for index in range(2):
            with observe_call('author', uuid.uuid4().hex, {'model': 'fixture'}) as capture:
                if index == 0:
                    capture.answer('observed answer', model='fixture')
        return draft()
    produce(ctx, None, 0.2)
    assert terminal(owner, complete_fixture=True).capture_state == 'incomplete'


def test_later_dispatch_cannot_cover_an_unaccounted_retry(lineage):
    _, ctx, owner = lineage
    @observed_phase('author')
    def produce(ctx, conn, temperature):
        with observe_call('author', uuid.uuid4().hex, {'model': 'fixture'}) as capture:
            capture.dispatch({'provider': 'fixture'})
            capture.dispatch({'provider': 'fixture'})
            capture.answer('later answer', model='fixture')
        return draft()
    produce(ctx, None, 0.2)
    assert terminal(owner, complete_fixture=True).capture_state == 'incomplete'


@pytest.mark.parametrize('phase', ['critic', 'evolve', 'debug'])
def test_repeated_revision_invocations_are_independently_accounted(lineage, phase):
    _, ctx, owner = lineage
    ctx.winner = draft()
    critique = CritiqueOutput('', [], 'fixture', 0, 0)
    @observed_phase(phase)
    def produce(ctx, conn, draft=None, prior_draft=None, critique=None, cleaned_text='',
                n_kc=0, issues=(), gate=None, verification=None, missing=False):
        if not missing:
            with observe_call(phase, uuid.uuid4().hex, {'model': 'fixture'}) as capture:
                capture.answer('observed answer', model='fixture')
        return critique if phase == 'critic' else (cleaned_text, n_kc, gate, verification) if phase == 'debug' else ctx.winner
    kwargs = dict(draft=ctx.winner, prior_draft=ctx.winner, critique=critique,
                  gate=GateResult(False, []), verification=VerificationReport([], [], []))
    produce(ctx, None, **kwargs)
    produce(ctx, None, missing=True, **kwargs)
    assert terminal(owner, complete_fixture=True).capture_state == 'incomplete'


@pytest.mark.parametrize('accounted', [False, True])
def test_parallel_author_call_and_deliberate_omission_remain_separate(lineage, accounted):
    _, ctx, owner = lineage
    @observed_phase('author')
    def produce(ctx, conn, temperature, slot=0):
        if slot == 0:
            with observe_call('author', uuid.uuid4().hex, {'model': 'fixture'}) as capture:
                capture.answer('observed answer', model='fixture')
        elif accounted:
            note_call_omission('author', 'stub_response_used')
        return draft()
    with ThreadPoolExecutor(2) as pool:
        futures = [submit_observed(pool, produce, ctx, None, 0.2, slot=i) for i in range(2)]
        for future in futures:
            future.result()
    manifest = terminal(owner, complete_fixture=True)
    assert manifest.capture_state == ('complete' if accounted else 'incomplete')
    assert audit_coverage(owner.records.store, manifest)['phase_runs'] == 2


@pytest.mark.parametrize('accounted', [False, True])
def test_keyword_call_cannot_satisfy_classifier_obligation(lineage, accounted):
    _, ctx, owner = lineage
    ctx.winner = draft()
    @observed_phase('commit')
    def produce(ctx, conn):
        note_call_omission('short_name', 'ingest_uses_author_trailer')
        with observe_call('keywords', uuid.uuid4().hex, {'model': 'fixture'}) as capture:
            capture.answer('keywords', model='fixture')
        if accounted:
            note_call_omission('classifier', 'classifier_disabled')
        return Path('fixture.md')
    produce(ctx, None)
    manifest = terminal(owner, complete_fixture=True)
    assert manifest.capture_state == ('complete' if accounted else 'incomplete')


def test_explicit_retry_rejection_followed_by_answer_is_complete(lineage):
    _, ctx, owner = lineage
    @observed_phase('author')
    def produce(ctx, conn, temperature):
        with observe_call('author', uuid.uuid4().hex, {'model': 'fixture'}) as capture:
            capture.dispatch({'provider': 'fixture'})
            capture.rejected('unsupported_effort', status=400)
            capture.dispatch({'provider': 'fixture'})
            capture.answer('later answer', model='fixture')
        return draft()
    produce(ctx, None, 0.2)
    manifest = terminal(owner, complete_fixture=True)
    assert manifest.capture_state == 'complete'
    assert audit_coverage(owner.records.store, manifest)['dispatches'] == 2


@pytest.mark.parametrize('route', ['openai-compatible', 'chat-relay'])
def test_real_adapter_normalization_is_not_a_second_invocation_input(lineage, monkeypatch, route):
    root, ctx, owner = lineage
    ctx.use_stub = False
    text = '## Summary\nSynthetic output.\n\nHANDLE: Fixture\nHOOK: Synthetic test.'
    config = model_config.ModelConfig(provider=route, model='fixture-author-1', temperature=0.2, max_tokens=2000)
    monkeypatch.setattr(model_config, 'for_phase', lambda phase: config)
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture-secret')
    monkeypatch.setenv('RW_LLM_BASE_URL', 'https://fixture.invalid/v1')
    monkeypatch.setattr(llm.urllib.request, 'urlopen', lambda *a, **k: BytesIO(json.dumps({
        'choices': [{'message': {'content': text}}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
    }).encode()))
    monkeypatch.setattr(relay, 'wiki_root', lambda: root)
    def handoff(pending, completed, *a, **k):
        relay._write_atomic_json(completed, {'via': 'fixture/author-1', 'response': text})
    monkeypatch.setattr(relay, '_emit_handoff_message', handoff)
    runner._phase_author(ctx, None, 0.2)
    manifest = terminal(owner, complete_fixture=True)
    report = audit_coverage(owner.records.store, manifest)
    assert manifest.capture_state == 'complete', report
    assert report['phase_runs'] == report['logical_calls'] == 1


def test_missing_grading_terminal_cannot_be_hidden_by_a_prior_run(grading, monkeypatch):
    _, _, owner = grading
    grade(grading)
    monkeypatch.setattr(GradingRun, 'finish', lambda self, *a, **k: self.evidence('final_evidence'))
    grade(grading)
    manifest = terminal(owner, complete_fixture=True)
    report = audit_coverage(owner.records.store, manifest)
    assert manifest.capture_state == 'incomplete' and report['grading_runs'] == 2
    assert any(item['reason'] == 'invocation_terminal_missing_or_duplicated' for item in report['issues'])


def test_grading_judge_call_cannot_cover_another_run(grading, monkeypatch):
    from researchwiki.grade import support
    _, _, owner = grading
    def observed_classifier(pairs):
        with observe_call('claim_support', uuid.uuid4().hex, {'model': 'fixture'}) as capture:
            capture.answer('supported', model='fixture')
        return ['supported'] * len(pairs)
    monkeypatch.setattr(support, 'llm_support_classifier', observed_classifier)
    run_support(grading, observed_classifier)
    def unobserved_classifier(pairs):
        return ['supported'] * len(pairs)
    monkeypatch.setattr(support, 'llm_support_classifier', unobserved_classifier)
    run_support(grading, unobserved_classifier)
    manifest = terminal(owner, complete_fixture=True)
    report = audit_coverage(owner.records.store, manifest)
    assert manifest.capture_state == 'incomplete'
    assert any(item['reason'] == 'support_call_unaccounted' for item in report['issues'])


def test_fully_observed_grading_failure_is_not_a_missing_capture(grading):
    _, ctx, owner = grading
    with pytest.raises(ValueError):
        run_support(grading, lambda pairs: [])
    for phase in PHASES:
        note_phase_omission(phase, 'standalone_grading_fixture')
    owner.finish({'scope': 'synthetic-grading-failure', 'execution': 'finished', 'outcome': 'failed',
                  'publication': 'not_published', 'projections': 'not_applicable'})
    manifest = owner.records.store.read_attempt(ctx.attempt_id)
    assert manifest.capture_state == 'complete'
    assert audit_coverage(owner.records.store, manifest)['status'] == 'complete'


@pytest.mark.parametrize('entry', ['operation', 'runner'])
def test_required_coverage_gap_preserves_publication_and_blocks_paid_replay(library, monkeypatch, entry):
    from researchwiki.operations.ingest import execute_ingest
    from researchwiki.operations.runtime import runner_options
    from researchwiki.records import lineage as capture_lineage
    root, request = library
    original = capture_lineage.note_call_omission
    def drop_author_reason(phase, *args, **kwargs):
        return None if phase == 'author' else original(phase, *args, **kwargs)
    monkeypatch.setattr(capture_lineage, 'note_call_omission', drop_author_reason)
    if entry == 'operation':
        result = execute_ingest(request).result
        assert result.publication == 'published' and result.error.code == 'capture_unavailable'
        assert result.error.retry == 'reconcile_first'
    else:
        with pytest.raises(RecordStoreUnavailable) as caught:
            runner.run_ingest(request.source_pdf, **runner_options(request.options))
        assert caught.value.publication == 'published'
    manifests = RecordStore(root / '.researchwiki/records').list_attempts()
    assert len(manifests) == 1 and manifests[0].capture_state == 'incomplete'
    assert manifests[0].terminal.publication == 'published'
    assert len(list((root / 'papers').glob('*.pdf'))) == 1


def test_coverage_is_persisted_and_recomputed_after_export(lineage, tmp_path, capsys):
    _, ctx, owner = lineage
    @observed_phase('author')
    def produce(ctx, conn, temperature):
        for index in range(2):
            with observe_call('author', uuid.uuid4().hex, {'model': 'fixture'}) as capture:
                if index == 0:
                    capture.answer('observed answer', model='fixture')
        return draft()
    produce(ctx, None, 0.2)
    manifest = terminal(owner, complete_fixture=True)
    item = next(i for i in manifest.inventory if i.slot == 'attempt.coverage')
    saved = next(a for a in manifest.artifacts if a.artifact_id in item.artifact_ids)
    assert decode_json(owner.records.store.read_blob(saved.payload))['status'] == 'incomplete'
    exported = export_attempt(owner.records.store, ctx.attempt_id, tmp_path / 'exported')
    store = RecordStore(exported)
    before = store.read_attempt(ctx.attempt_id)
    assert records_main(['--root', str(exported), 'verify', ctx.attempt_id, '--coverage', '--json']) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['coverage']['status'] == 'incomplete' and result['coverage']['logical_calls'] == 2
    assert store.read_attempt(ctx.attempt_id) == before


def test_old_producer_completeness_is_preserved_and_audit_limits_are_explicit(tmp_path, capsys):
    fixture = Path(__file__).with_name('record-fixtures') / 'attempt-spec-v1.json'
    spec = AttemptSpec.from_dict(decode_json(fixture.read_bytes()))
    store = RecordStore.create(tmp_path / 'old-archive')
    store.start_attempt(spec)
    for req in spec.requirements:
        store.record_artifact(req.slot, old_artifact(store, spec, req.slot))
    manifest = store.finish_attempt(spec.attempt_id, old_finish())
    before = {p: p.read_bytes() for p in store.root.rglob('*') if p.is_file()}
    assert records_main(['--root', str(store.root), 'verify', spec.attempt_id, '--coverage', '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['capture_state'] == 'complete'
    assert result['coverage']['status'] == 'limited'
    assert 'producer_did_not_declare_invocation_audit' in result['coverage']['limitations']
    assert store.read_attempt(spec.attempt_id) == manifest
    assert {p: p.read_bytes() for p in store.root.rglob('*') if p.is_file()} == before


def test_standalone_grading_acceptance_declares_its_scope(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, RESEARCHWIKI_DB_PATH=str(tmp_path / 'state.db'),
               RESEARCHWIKI_JOBS_ROOT=str(tmp_path / 'jobs'))
    completed = subprocess.run([sys.executable, str(repo / '.github/scripts/grading_acceptance.py')],
                               cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
