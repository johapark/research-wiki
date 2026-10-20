"""Durable local records and ownership rules, without paid provider calls."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
from threading import Barrier

import pytest

from researchwiki.jobs import InputIdentity, JobConflict, JobStore, JobStoreUnavailable, JobSupervisor
from researchwiki.jobs.events import pending_relays, read_events
from researchwiki.jobs.files import write_private_json
from researchwiki.operations import ContractError, IngestEvent, IngestOptions, IngestRequest, IngestResult


@pytest.fixture
def library(tmp_path):
    root = tmp_path / 'library'
    for folder in ('wiki/other', 'papers', 'inbox', 'prompts', 'config'):
        (root / folder).mkdir(parents=True)
    (root / 'prompts/author-system-research.md').write_text('Author instructions')
    (root / 'config/pricing.yaml').write_text('models: {}')
    source = root / 'inbox/paper.pdf'
    source.write_bytes(b'%PDF-1.4')
    request = IngestRequest(request_id='request-1', library_root=str(root), source_pdf=str(source),
                            options=IngestOptions(use_stub=True, n_drafts=1))
    return root, request, JobStore(root, store_root=tmp_path / 'runtime')


def failed_result(record):
    from researchwiki.operations import OperationError
    return IngestResult(request_id=record.request.request_id, job_id=record.job_id, execution='finished',
                        outcome='failed', publication='not_published', projections='not_applicable',
                        error=OperationError(code='invalid_request', category='validation', message='Correct input',
                                             retry='after_correction'))


def test_idempotent_submit_survives_restart_and_source_consumption(library):
    root, request, store = library
    record = store.submit_ingest(request)
    Path(request.source_pdf).unlink()
    again = JobStore(root, store_root=store.base).submit_ingest(request)
    assert again == record and record.execution == 'queued'
    assert record.request_sha256 and record.identity is None


def test_duplicate_submit_is_atomic_across_threads(library):
    _, request, store = library
    with ThreadPoolExecutor(8) as pool:
        records = list(pool.map(lambda _: store.submit_ingest(request), range(16)))
    assert len({record.job_id for record in records}) == 1
    with pytest.raises(JobConflict, match='different request'):
        store.submit_ingest(replace(request, options=replace(request.options, max_evolve=0)))


def test_runtime_namespace_is_independent_of_db_override(library, tmp_path, monkeypatch):
    root, request, store = library
    other = tmp_path / 'elsewhere/library'
    shutil.copytree(root, other)
    monkeypatch.setenv('RESEARCHWIKI_DB_PATH', str(tmp_path / 'shared.db'))
    other_store = JobStore(other, store_root=store.base)
    assert other_store.root != store.root
    record = store.submit_ingest(request)
    with pytest.raises(JobStoreUnavailable):
        other_store.get_job(record.job_id)
    with pytest.raises(JobConflict):
        other_store.submit_ingest(request)


def test_job_records_survive_derived_state_deletion(library):
    root, request, store = library
    record = store.submit_ingest(request)
    (root / 'state.db').write_bytes(b'derived')
    (root / '.tantivy-index').mkdir()
    (root / 'state.db').unlink()
    shutil.rmtree(root / '.tantivy-index')
    assert store.get_job(record.job_id) == record
    if os.name == 'posix':
        assert (store.path(record.job_id) / 'manifest.json').stat().st_mode & 0o777 == 0o600
        assert store.path(record.job_id).stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize('bad', ['../escape', 'x' * 32, '/absolute', '', 'a' * 31])
def test_job_ids_cannot_select_arbitrary_paths(library, bad):
    _, _, store = library
    with pytest.raises(ContractError):
        store.get_job(bad)


def test_override_cannot_put_jobs_in_synced_content(library):
    root, _, _ = library
    for path in (root, root / 'wiki', root / 'papers', root / 'inbox'):
        with pytest.raises(JobConflict):
            JobStore(root, store_root=path)


@pytest.mark.parametrize('damage', ['future', 'malformed', 'identity'])
def test_bad_manifest_is_never_rewritten_or_replayed(library, damage):
    _, request, store = library
    record = store.submit_ingest(request)
    path = store.path(record.job_id) / 'manifest.json'
    data = json.loads(path.read_text())
    if damage == 'future':
        data['schema_version'] = 99
    elif damage == 'identity':
        data['request']['source_pdf'] += '.pdf'
    path.write_text('{' if damage == 'malformed' else json.dumps(data))
    before = path.read_bytes()
    with pytest.raises((ContractError, JobStoreUnavailable)):
        store.get_job(record.job_id)
    assert path.read_bytes() == before


def test_cancel_wins_before_claim_and_does_not_start_worker(library):
    _, request, store = library
    record = store.submit_ingest(request)
    cancelled = store.cancel_queued_job(record.job_id)
    assert cancelled.execution == 'cancelled' and cancelled.result.publication == 'not_published'
    with pytest.raises(JobConflict):
        store.claim(record.job_id, 'late-owner')
    from researchwiki.jobs.worker import run_worker
    assert run_worker(store, record.job_id, operation=lambda *a, **k: pytest.fail('cancelled work ran')) == 1


def test_worker_generation_fences_terminal_writes(library):
    _, request, store = library
    record = store.submit_ingest(request)
    running = store.claim(record.job_id, 'owner-1')
    with pytest.raises(JobConflict):
        store.finish(record.job_id, 'owner-2', running.generation, failed_result(record))
    with pytest.raises(JobConflict):
        store.finish(record.job_id, 'owner-1', running.generation - 1, failed_result(record))
    with pytest.raises(JobConflict):
        store.cancel_queued_job(record.job_id)
    finished = store.finish(record.job_id, 'owner-1', running.generation, failed_result(record))
    assert finished.execution == 'finished' and finished.owner is None
    with pytest.raises(JobConflict):
        store.finish(record.job_id, 'owner-1', running.generation, failed_result(record))


@pytest.mark.parametrize('attempt', range(4))
def test_cancel_and_claim_race_has_one_winner(library, attempt):
    _, request, store = library
    record = store.submit_ingest(request)
    ready = Barrier(2)

    def claim():
        ready.wait(timeout=5)
        try:
            with store.owner_lock(record.job_id):
                return store.claim(record.job_id, 'owner').execution
        except JobConflict:
            return 'lost'

    def cancel():
        ready.wait(timeout=5)
        try:
            return store.cancel_queued_job(record.job_id).execution
        except JobConflict:
            return 'lost'

    with ThreadPoolExecutor(2) as pool:
        a, b = pool.submit(claim), pool.submit(cancel)
        outcomes = [a.result(timeout=10), b.result(timeout=10)]
    terminal = store.get_job(record.job_id)
    assert outcomes.count('lost') == 1
    assert terminal.execution in ('running', 'cancelled')


def test_explicit_retry_creates_a_new_linked_job_and_tracks_changed_inputs(library):
    _, request, store = library
    record = store.submit_ingest(request)
    running = store.claim(record.job_id, 'owner')
    original = InputIdentity(pdf_sha256='a', auxiliary_sha256='b', configuration_sha256='c')
    store.set_identity(record.job_id, 'owner', running.generation, original)
    store.finish(record.job_id, 'owner', running.generation, failed_result(record))
    retry = store.retry_job(record.job_id)
    assert retry.job_id != record.job_id and retry.request.request_id != request.request_id
    assert retry.parent_job_id == record.job_id and retry.execution == 'queued'
    running = store.claim(retry.job_id, 'new-owner')
    changed = store.set_identity(retry.job_id, 'new-owner', running.generation,
                                 replace(original, pdf_sha256='changed', configuration_sha256='different'))
    assert changed.source_changed is True and changed.configuration_changed is True


def test_lost_worker_stays_unknown_and_cannot_be_retried(library):
    root, request, store = library
    record = store.submit_ingest(request)
    store.claim(record.job_id, 'lost-owner')
    # A matching canonical path is deliberately insufficient publication proof.
    (root / 'papers/example.pdf').write_bytes(b'%PDF')
    supervisor = JobSupervisor(root, store_root=store.base)
    lost = supervisor.refresh_job(record.job_id)
    assert lost.execution == 'interrupted' and lost.result.publication == 'unknown'
    assert lost.result.error.retry == 'reconcile_first'
    with pytest.raises(JobConflict):
        supervisor.retry_job(record.job_id)


def test_live_owner_lock_prevents_loss_reconciliation(library):
    root, request, store = library
    record = store.submit_ingest(request)
    with store.owner_lock(record.job_id):
        store.claim(record.job_id, 'active')
        observed = JobSupervisor(root, store_root=store.base).refresh_job(record.job_id)
        assert observed.execution == 'running' and observed.owner == 'active'


def test_event_tail_can_be_partial_but_complete_corruption_is_an_error(library):
    _, request, store = library
    record = store.submit_ingest(request)
    event = IngestEvent(request_id=request.request_id, job_id=record.job_id, sequence=1,
                        timestamp='2026-10-19T00:00:00Z', kind='operation_started', call_id='scope')
    path = store.path(record.job_id) / 'events.jsonl'
    path.write_text(json.dumps(event.to_dict()) + '\n{"sequence":2')
    assert read_events(store, record.job_id) == (event,)
    assert read_events(store, record.job_id, after_sequence=1) == ()
    path.write_text(json.dumps(event.to_dict()) + '\n{"sequence":2}\n')
    with pytest.raises(JobStoreUnavailable):
        read_events(store, record.job_id)


def test_event_identity_and_sequence_must_match_the_job(library):
    _, request, store = library
    record = store.submit_ingest(request)
    event = IngestEvent(request_id='someone-else', job_id=record.job_id, sequence=1,
                        timestamp='2026-10-19T00:00:00Z', kind='operation_started', call_id='scope')
    path = store.path(record.job_id) / 'events.jsonl'
    path.write_text(json.dumps(event.to_dict()) + '\n')
    with pytest.raises(JobStoreUnavailable):
        read_events(store, record.job_id)


def test_pending_relay_handoffs_are_only_active_while_running(library):
    from researchwiki.operations import RelayHandoff

    _, request, store = library
    record = store.submit_ingest(request)
    store.claim(record.job_id, 'owner')
    event = IngestEvent(request_id=request.request_id, job_id=record.job_id, sequence=1,
                        timestamp='2026-10-19T00:00:00Z', kind='relay_waiting', call_id='model-1',
                        relay=RelayHandoff(op_id='op-1', prompt_path=str(store.root / 'prompt.json'),
                                           response_path=str(store.root / 'response.json'), timeout_seconds=10))
    (store.path(record.job_id) / 'events.jsonl').write_text(json.dumps(event.to_dict()) + '\n')
    assert pending_relays(store, record.job_id) == (event.relay,)
    store.mark_interrupted(record.job_id)
    assert pending_relays(store, record.job_id) == ()


def test_terminal_record_write_failure_preserves_running_uncertainty(library, monkeypatch):
    from researchwiki.jobs import store as store_module

    _, request, store = library
    record = store.submit_ingest(request)
    running = store.claim(record.job_id, 'owner')

    def fail(*args):
        raise OSError('disk full')

    monkeypatch.setattr(store_module, 'write_private_json', fail)
    with pytest.raises(OSError):
        store.finish(record.job_id, 'owner', running.generation, failed_result(record))
    assert store.get_job(record.job_id) == running


def test_submission_crash_leaves_no_visible_half_job(library, monkeypatch):
    from researchwiki.jobs import store as store_module

    _, request, store = library
    monkeypatch.setattr(store_module, 'write_private_json', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):
        store.submit_ingest(request)
    assert not list(store.root.iterdir())


def test_parent_profile_settings_are_not_inherited_by_another_library():
    from researchwiki.env_profiles import ACTIVE_ENV_FILE_VAR, DOTENV_PROVENANCE_VAR
    from researchwiki.jobs.environment import child_environment

    parent = {'PATH': '/bin', 'OPENAI_API_KEY': 'profile-owned-secret', 'RW_MODELS_CONFIG': '/library-a/models.yaml',
              'RESEARCHWIKI_DB_PATH': '/library-a/state.db', ACTIVE_ENV_FILE_VAR: '/library-a/.env',
              DOTENV_PROVENANCE_VAR: json.dumps({'path': '/library-a/.env', 'keys': ['OPENAI_API_KEY', 'RW_MODELS_CONFIG']})}
    before = dict(parent)
    child = child_environment(parent)
    assert child == {'PATH': '/bin'} and parent == before
    assert child_environment({'OPENAI_API_KEY': 'exported-key'}) == {'OPENAI_API_KEY': 'exported-key'}


def test_manifest_cannot_be_redirected_to_a_foreign_library(library, tmp_path):
    _, request, store = library
    record = store.submit_ingest(request)
    data = record.to_dict()
    data['request']['library_root'] = str(tmp_path / 'foreign')
    write_private_json(store.path(record.job_id) / 'manifest.json', data)
    with pytest.raises(JobStoreUnavailable):
        store.get_job(record.job_id)
