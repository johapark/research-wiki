"""Kill real native workers at transaction boundaries; never call a paid model."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from researchwiki import ingest_ledger
from researchwiki.jobs import JobConflict, JobStore, JobSupervisor
from researchwiki.jobs.files import JobStoreUnavailable, write_private_json
from researchwiki.jobs.recovery import reconcile_job
from researchwiki.operations import IngestResult, OperationError
from tests.test_job_workers import clean_environment, make_library, probe_process, until


CRASH_WORKER = r'''
import sys, time
from pathlib import Path
from researchwiki import ingest_ledger, mutation
from researchwiki.agents import promote
from researchwiki.jobs import JobStore
from researchwiki.jobs import worker
from researchwiki.operations import ingest
root, base, job_id, point = sys.argv[1:]
store = JobStore(root, store_root=base)
if point == 'after_cleanup_no_events':
    def failed_sink(self, event): raise OSError('event storage unavailable')
    worker.EventWriter.__call__ = failed_sink
def hold():
    (Path(root) / 'kill-ready').write_text(point)
    time.sleep(60)
original_begin = worker.begin
def begin(binding):
    original_begin(binding)
    if point == 'before_operation': hold()
worker.begin = begin
original_snapshot = mutation.snapshot
def snapshot(*args, **kwargs):
    snap = original_snapshot(*args, **kwargs)
    if point == 'after_snapshot': hold()
    return snap
mutation.snapshot = snapshot
original_move = promote._move_pdf
def move(*args, **kwargs):
    result = original_move(*args, **kwargs)
    if point == 'after_pdf_move': hold()
    return result
promote._move_pdf = move
original_commit = mutation.Snapshot.mark_committed
def commit(self):
    if point == 'before_commit': hold()
    original_commit(self)
    if point == 'after_commit': hold()
mutation.Snapshot.mark_committed = commit
original_discard = mutation.Snapshot.discard
def discard(self):
    original_discard(self)
    if self.committed and point in ('after_cleanup', 'after_cleanup_no_events', 'corrupt_events'): hold()
mutation.Snapshot.discard = discard
original_finish = store.finish
def finish(*args, **kwargs):
    if point == 'before_terminal': hold()
    if point == 'terminal_write_failure': raise OSError('terminal storage unavailable')
    return original_finish(*args, **kwargs)
store.finish = finish
def operation(request, *, event_sink):
    with (Path(root) / 'operation-count').open('a') as stream: stream.write('once\n')
    return ingest(request, event_sink=event_sink)
raise SystemExit(worker.run_worker(store, job_id, operation=operation))
'''


@pytest.mark.parametrize('point,published', [
    ('before_operation', False), ('after_snapshot', False), ('after_pdf_move', False),
    ('before_commit', False), ('after_commit', True), ('after_cleanup', True),
    ('before_terminal', True),
    ('after_cleanup_no_events', True), ('corrupt_events', True),
])
def test_native_worker_kill_reconciles_without_repeating_ingestion(tmp_path, point, published):
    root, request = make_library(tmp_path, 'crash-' + point)
    # Exercise inbox consumption and restoration, not just an external-file copy.
    source = root / 'inbox/paper.pdf'
    Path(request.source_pdf).rename(source)
    request = replace(request, source_pdf=str(source))
    env = clean_environment(tmp_path)
    supervisor = JobSupervisor(root, store_root=tmp_path / 'runtime', environment=env)
    job = supervisor.submit_ingest(request)
    process = subprocess.Popen([sys.executable, '-c', CRASH_WORKER, str(root),
                               str(supervisor.store.base), job.job_id, point],
                              cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        until(lambda: (root / 'kill-ready').exists() or process.poll() is not None, timeout=40)
        if process.poll() is not None:
            pytest.fail('worker exited before kill point: ' + process.communicate()[1])
        assert supervisor.refresh_job(job.job_id).execution == 'running'
        with pytest.raises(JobConflict, match='live owner'):
            supervisor.reconcile_job(job.job_id, timeout=20)
        if point == 'corrupt_events':
            with (supervisor.store.path(job.job_id) / 'events.jsonl').open('a') as stream:
                stream.write('{bad complete event}\n')
        process.kill()
        process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    lost = supervisor.refresh_job(job.job_id)
    assert lost.execution == 'interrupted' and lost.result.publication == 'unknown'
    count_path = root / 'operation-count'
    before = count_path.read_text() if count_path.exists() else ''
    recovered = supervisor.reconcile_job(job.job_id, timeout=40)
    assert recovered.execution == 'finished'
    assert recovered.result.publication == ('published' if published else 'not_published')
    assert (count_path.read_text() if count_path.exists() else '') == before
    assert not list((root / '.mutation').glob('*.json'))
    if published:
        assert recovered.result.outcome == 'published'
        assert Path(recovered.result.page_path).is_file() and Path(recovered.result.pdf_path).is_file()
        assert not source.exists()
        # Offline fixture has no embedding model; pending is honest, not a
        # fabricated current state. Local repair can be tried again explicitly.
        assert recovered.result.projections == 'maintenance_pending'
        checkpoint = ingest_ledger.read_ledger(supervisor.store.ledger_binding(recovered))['result']
        assert recovered.result.post_work == (checkpoint['post_work'] if point == 'before_terminal' else 'unknown')
        with pytest.raises(JobConflict):
            supervisor.retry_job(job.job_id)
    else:
        assert source.is_file() and not list((root / 'papers').glob('*.pdf'))
        assert recovered.result.error.retry == 'explicit_retry'
        retry = supervisor.retry_job(job.job_id)
        assert retry.parent_job_id == job.job_id and retry.execution == 'queued'
        supervisor.cancel_queued_job(retry.job_id)
    again = supervisor.reconcile_job(job.job_id, timeout=40)
    assert again.result.publication == recovered.result.publication
    if point == 'corrupt_events':
        with pytest.raises(JobStoreUnavailable): supervisor.read_events(job.job_id)
    else:
        assert [e.sequence for e in supervisor.read_events(job.job_id)] == list(
            range(1, len(supervisor.read_events(job.job_id)) + 1))
    if point == 'after_cleanup_no_events': assert supervisor.read_events(job.job_id) == ()


@pytest.fixture
def abandoned(tmp_path, monkeypatch):
    root, request = make_library(tmp_path, 'abandoned')
    store = JobStore(root, store_root=tmp_path / 'runtime')
    queued = store.submit_ingest(request)
    record = store.claim(queued.job_id, 'old-owner')
    binding = store.ledger_binding(record)
    ingest_ledger.begin(binding)
    monkeypatch.chdir(root)
    from researchwiki.jobs import recovery
    monkeypatch.setattr(recovery, 'configure_worker', lambda request: request)
    return root, store, record, binding


@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'future', 'wrong_generation', 'wrong_request'])
def test_missing_or_invalid_evidence_never_authorizes_retry(abandoned, damage):
    _, store, record, binding = abandoned
    path = Path(binding['path'])
    data = json.loads(path.read_text())
    if damage == 'missing':
        path.unlink()
    elif damage == 'corrupt':
        path.write_text('{')
    else:
        if damage == 'future': data['schema_version'] = 99
        if damage == 'wrong_generation': data['binding']['generation'] += 1
        if damage == 'wrong_request': data['binding']['request_id'] = 'different'
        write_private_json(path, data)
    before = path.read_bytes() if path.exists() else None
    result = reconcile_job(store, record.job_id)
    assert result.execution == 'interrupted' and result.result.publication == 'unknown'
    assert (path.read_bytes() if path.exists() else None) == before
    with pytest.raises(JobConflict): store.retry_job(record.job_id)


@pytest.mark.parametrize('journal', ['{', '{"version":999,"status":"active","operation":"promote"}'])
def test_corrupt_or_future_journal_blocks_unpublished_certification(abandoned, journal):
    root, store, record, _ = abandoned
    (root / '.mutation').mkdir()
    path = root / '.mutation/unresolved.json'
    path.write_text(journal)
    result = reconcile_job(store, record.job_id)
    assert result.execution == 'interrupted' and result.result.publication == 'unknown'
    assert path.read_text() == journal


def test_durable_checkpoint_recovers_gate_rejected_draft(abandoned):
    root, store, record, binding = abandoned
    draft = root / 'review.md'
    draft.write_text('Preserved review draft')
    result = IngestResult(request_id=record.request.request_id, job_id=record.job_id,
                          execution='finished', outcome='draft_ready', publication='not_published',
                          projections='not_applicable', draft_path=str(draft), gate_reasons=('coverage',))
    ingest_ledger.complete(binding, result)
    restored = reconcile_job(store, record.job_id)
    assert restored.result == result
    assert draft.read_text() == 'Preserved review draft'


def test_known_unpublished_recovery_preserves_partial_artifact_and_budget_error(abandoned):
    root, store, record, binding = abandoned
    partial = root / 'partial.md'
    partial.write_text('Best partial draft')
    result = IngestResult(request_id=record.request.request_id, job_id=record.job_id,
                          execution='finished', outcome='failed', publication='unknown',
                          draft_path=str(partial), error=OperationError(
                              code='budget_exhausted', category='budget', message='Budget exhausted',
                              retry='reconcile_first'))
    ingest_ledger.complete(binding, result)
    restored = reconcile_job(store, record.job_id)
    assert restored.result.publication == 'not_published'
    assert restored.result.draft_path == str(partial)
    assert restored.result.error.code == 'budget_exhausted'
    assert restored.result.error.retry == 'explicit_retry'


def test_event_or_matching_filename_is_not_publication_proof(abandoned):
    root, store, record, binding = abandoned
    data = ingest_ledger.read_ledger(binding)
    data['state'] = 'prepared'
    data['journal_path'] = str(root / '.mutation/missing.json')
    write_private_json(Path(binding['path']), data)
    (root / 'papers/matching.pdf').write_bytes(b'%PDF')
    restored = reconcile_job(store, record.job_id)
    assert restored.result.publication == 'unknown'
    with pytest.raises(JobConflict): store.retry_job(record.job_id)


def committed_publication(abandoned):
    from researchwiki.mutation import mutation
    from researchwiki.operations import CheckResult, EventRecorder
    from researchwiki.operations.lifecycle import IngestFacts, observe_lifecycle, publication_details
    from researchwiki.operations.observation import observe_models

    root, store, record, binding = abandoned
    page = root / 'wiki/other/lovelace-2026-a-test-of-local-jobs.md'
    pdf = root / 'papers'/ (page.stem + '.pdf')
    facts = IngestFacts(gate_reasons=('forced coverage exception',),
                        checks=(CheckResult(name='numeric', status='failed'),
                                CheckResult(name='entailment', status='not_requested')))
    recorder = EventRecorder(record.request.request_id, lambda e: None, job_id=record.job_id)
    recorder.bind_attempt('commit-attempt')
    with observe_models(recorder), observe_lifecycle(facts), ingest_ledger.observe_publication(binding):
        details = publication_details('commit-attempt', page.stem, 'other')
        with mutation([page, pdf, root/'wiki/index.md'], operation='promote', details=details) as snap:
            ingest_ledger.prepare(snap)
            page.write_text('---\ntitle: Local jobs\ntype: paper\ncategory: [other]\n---\n'
                            '## Key Contributions\n- The model predicts regulatory activity from DNA sequence.\n')
            pdf.write_bytes(Path(record.request.source_pdf).read_bytes())
            ingest_ledger.set_artifacts(snap, SimpleNamespace(wiki_path=page, pdf_path=pdf), page.stem)
            snap.mark_committed()
    return snap, page, pdf


def test_committed_journal_survives_archival_failure_and_preserves_gate_evidence(abandoned, monkeypatch):
    snap, page, _ = committed_publication(abandoned)
    _, store, record, binding = abandoned
    original = ingest_ledger.write_private_json

    def fail_archive(path, value):
        if value['state'] == 'committed': raise OSError('disk full')
        original(path, value)

    monkeypatch.setattr(ingest_ledger, 'write_private_json', fail_archive)
    restored = reconcile_job(store, record.job_id, repair=lambda *a: pytest.fail('repair must wait for journal handoff'))
    assert restored.result.publication == 'published' and restored.result.projections == 'maintenance_pending'
    assert restored.result.gate_reasons == ('forced coverage exception',)
    assert restored.result.checks[0].status == 'failed'
    assert restored.result.promotion == 'always'
    assert snap.journal_path.exists() and page.exists()
    assert ingest_ledger.read_ledger(binding)['state'] == 'prepared'
    with pytest.raises(JobConflict): store.retry_job(record.job_id)
    monkeypatch.setattr(ingest_ledger, 'write_private_json', original)
    repaired = reconcile_job(store, record.job_id, repair=lambda *a: None)
    assert repaired.result.publication == 'published' and repaired.result.projections == 'current'
    assert not snap.journal_path.exists()
    assert ingest_ledger.read_ledger(binding)['state'] == 'committed'


@pytest.mark.parametrize('kind', ['page', 'pdf', 'missing_pdf'])
def test_later_canonical_changes_are_never_overwritten_or_reauthored(abandoned, kind):
    snap, page, pdf = committed_publication(abandoned)
    _, store, record, _ = abandoned
    snap.reconcile()
    if kind == 'page': page.write_text(page.read_text() + '\nUser edit\n')
    if kind == 'pdf': pdf.write_bytes(pdf.read_bytes() + b'\n% replaced')
    if kind == 'missing_pdf': pdf.unlink()
    before = (page.read_bytes(), pdf.read_bytes() if pdf.exists() else None)
    restored = reconcile_job(store, record.job_id)
    assert restored.result.publication == 'published'
    assert restored.result.projections == 'maintenance_pending'
    assert (page.read_bytes(), pdf.read_bytes() if pdf.exists() else None) == before
    assert restored.result.error.retry == 'reconcile_first'
    with pytest.raises(JobConflict): store.retry_job(record.job_id)


def test_local_repair_rebuilds_deleted_db_indexes_and_required_claim_grades(abandoned, monkeypatch):
    import numpy as np
    from researchwiki.agents import llm
    from researchwiki.db.connection import db_path, get_connection
    from researchwiki.index import pages_semantic
    from researchwiki.db.verify import verify
    from researchwiki.search import get_default_backend

    snap, page, pdf = committed_publication(abandoned)
    _, store, record, binding = abandoned
    # Resolve and archive normally, then lose all derived DB content. Receipt
    # recovery must reconstruct it from current canonical files.
    snap.reconcile()
    db_path().unlink()
    monkeypatch.setattr(llm, 'call', lambda **k: pytest.fail('recovery called a model provider'))
    monkeypatch.setattr(pages_semantic.semantic, '_get_model', lambda *a: object())
    monkeypatch.setattr(pages_semantic.semantic, 'is_available', lambda: True)
    monkeypatch.setattr(pages_semantic.semantic, 'embed_texts',
                        lambda texts: np.ones((len(texts), 2), dtype=np.float32))
    before = page.read_bytes(), pdf.read_bytes()
    restored = reconcile_job(store, record.job_id)
    assert restored.result.publication == 'published' and restored.result.projections == 'current'
    assert restored.result.post_work == 'unknown'
    assert (page.read_bytes(), pdf.read_bytes()) == before
    assert verify().is_clean
    assert get_default_backend().exists() and pages_semantic.load_index() is not None
    conn = get_connection()
    try:
        rows = conn.execute('SELECT bm25_top1, source_pdf_sha256 FROM claims WHERE paper_stem = ?', (page.stem,)).fetchall()
        assert rows and all(row[0] is not None and row[1] for row in rows)
    finally:
        conn.close()
    assert ingest_ledger.read_ledger(binding)['state'] == 'committed'


def test_incomplete_rollback_keeps_backup_and_job_blocked(abandoned):
    from researchwiki.mutation import snapshot
    root, store, record, binding = abandoned
    source = Path(record.request.source_pdf)
    details = dict(repair_projections=True, ingest_ledger=binding,
                   request_id=record.request.request_id, job_id=record.job_id)
    snap = snapshot([source], operation='promote', details=details)
    ingest_ledger.prepare(snap)
    snap.entries[0].backup.unlink()
    source.write_bytes(b'Changed while interrupted')
    result = reconcile_job(store, record.job_id)
    assert result.result.publication == 'unknown' and snap.journal_path.exists()
    assert source.read_bytes() == b'Changed while interrupted'
    with pytest.raises(JobConflict): store.retry_job(record.job_id)


def test_disabled_journal_cannot_publish_a_durable_job(abandoned, monkeypatch):
    from researchwiki.mutation import mutation
    root, store, record, binding = abandoned
    monkeypatch.setenv('RW_MUTATION_JOURNAL', '0')
    page = root / 'wiki/other/never.md'
    with pytest.raises(ValueError, match='require mutation journalling'):
        with mutation([page], operation='promote', details={'ingest_ledger': binding}) as snap:
            ingest_ledger.prepare(snap)
            page.write_text('Must never reach canonical write')
    assert not page.exists()
    restored = reconcile_job(store, record.job_id)
    assert restored.result.publication == 'not_published'


def test_other_cli_recovery_archives_proof_before_discarding_journal(abandoned):
    from researchwiki.mutation import recover_pending
    snap, _, _ = committed_publication(abandoned)
    _, store, record, binding = abandoned
    assert ingest_ledger.read_ledger(binding)['state'] == 'prepared'
    recover_pending()  # no worker context or job supervisor in this call
    assert not snap.journal_path.exists()
    assert ingest_ledger.read_ledger(binding)['state'] == 'committed'
    restored = reconcile_job(store, record.job_id, repair=lambda *a: None)
    assert restored.result.publication == 'published'


def test_terminal_write_failure_recovers_operation_checkpoint(tmp_path):
    root, request = make_library(tmp_path, 'failed-terminal')
    supervisor = JobSupervisor(root, store_root=tmp_path / 'runtime', environment=clean_environment(tmp_path))
    job = supervisor.submit_ingest(request)
    process = subprocess.run([sys.executable, '-c', CRASH_WORKER, str(root), str(supervisor.store.base),
                              job.job_id, 'terminal_write_failure'], cwd=root,
                             env=clean_environment(tmp_path), capture_output=True, text=True, timeout=40)
    assert process.returncode != 0 and 'terminal storage unavailable' in process.stderr
    assert supervisor.get_job(job.job_id).execution == 'running'
    checkpoint = ingest_ledger.read_ledger(supervisor.store.ledger_binding(supervisor.get_job(job.job_id)))['result']
    assert checkpoint['publication'] == 'published'
    restored = supervisor.reconcile_job(job.job_id, timeout=40)
    assert restored.execution == 'finished' and restored.result.publication == 'published'
    assert restored.result.post_work == checkpoint['post_work']
    assert (root / 'operation-count').read_text() == 'once\n'


def test_reconciliation_cannot_overwrite_a_newer_terminal_state(abandoned):
    _, store, record, _ = abandoned
    lost = store.mark_interrupted(record.job_id)
    result = IngestResult(request_id=record.request.request_id, job_id=record.job_id,
                          execution='finished', outcome='failed', publication='not_published')
    store.finish_reconciliation(lost, result)
    with pytest.raises(JobConflict, match='changed during reconciliation'):
        store.finish_reconciliation(lost, result)


@pytest.mark.parametrize('mode', ['relay', 'mixed'])
def test_relay_and_mixed_worker_loss_never_replays_a_waiting_model_call(tmp_path, mode):
    root, request = make_library(tmp_path, mode + '-recovery',
                                 provider='chat-relay' if mode == 'relay' else 'openai-compatible',
                                 mixed=mode == 'mixed')
    supervisor = JobSupervisor(root, store_root=tmp_path/'runtime', environment=clean_environment(tmp_path))
    job = supervisor.submit_ingest(request)
    process = probe_process(root, supervisor.store, job.job_id, clean_environment(tmp_path), mode=mode)
    try:
        handoff, = until(lambda: supervisor.pending_relays(job.job_id))
        with pytest.raises(JobConflict): supervisor.reconcile_job(job.job_id, timeout=20)
        process.kill()
        process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    events = supervisor.read_events(job.job_id)
    recovered = supervisor.reconcile_job(job.job_id, timeout=30)
    assert recovered.execution == 'finished' and recovered.result.publication == 'not_published'
    assert recovered.result.error.retry == 'explicit_retry'
    assert supervisor.read_events(job.job_id) == events
    assert not Path(handoff.response_path).exists()
    assert supervisor.pending_relays(job.job_id) == ()


def test_receipt_paths_work_with_symlinked_library_directories(abandoned, tmp_path):
    root, store, record, _ = abandoned
    for name in ('wiki', 'papers'):
        target = tmp_path / ('synced-' + name)
        (root / name).rename(target)
        (root / name).symlink_to(target, target_is_directory=True)
    snap, page, pdf = committed_publication(abandoned)
    snap.reconcile()
    recovered = reconcile_job(store, record.job_id, repair=lambda *a: None)
    assert recovered.result.publication == 'published'
    assert Path(recovered.result.page_path) == page.resolve()
    assert Path(recovered.result.pdf_path) == pdf.resolve()


def test_journal_unlink_failure_keeps_rollback_backups_available(abandoned, monkeypatch):
    snap, _, _ = committed_publication(abandoned)
    original = Path.unlink

    def fail_journal(path, *args, **kwargs):
        if path == snap.journal_path: raise PermissionError('journal unlink denied')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', fail_journal)
    snap.discard()
    assert snap.journal_path.exists() and snap.backup_dir.exists()


@pytest.mark.parametrize('unavailable', ['unreadable', 'dangling_symlink'])
def test_unavailable_journal_directory_does_not_certify_nonpublication(abandoned, monkeypatch, unavailable):
    root, store, record, _ = abandoned
    directory = root / '.mutation'
    if unavailable == 'dangling_symlink':
        directory.symlink_to(root / 'offline-mutations', target_is_directory=True)
    else:
        directory.mkdir()
        original = Path.iterdir

        def deny(path):
            if path == directory: raise PermissionError('mutation storage offline')
            return original(path)

        monkeypatch.setattr(Path, 'iterdir', deny)
    restored = reconcile_job(store, record.job_id)
    assert restored.execution == 'interrupted' and restored.result.publication == 'unknown'
    log = store.path(record.job_id) / 'reconciliation.log'
    assert log.is_file() and ('offline' in log.read_text() or 'unavailable' in log.read_text())
