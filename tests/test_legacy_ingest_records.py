"""Historical inspection and backup preserve source data and expose metadata only."""

import json
from pathlib import Path
import sqlite3

import pytest

from researchwiki import ingest_ledger
from researchwiki.fsatomic import file_sha256
from researchwiki.jobs.events import EventWriter
from researchwiki.operations import IngestEvent, IngestResult, RelayHandoff
from researchwiki.records import RecordConflict, RecordStoreUnavailable
from researchwiki.records.legacy import inventory_legacy
from researchwiki.records.legacy_backup import backup_legacy_database
from researchwiki.tasks.records import main
from tests.test_jobs import library as library


def history(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE ingest_iterations (
            id INTEGER PRIMARY KEY, attempt_id TEXT, iteration INTEGER, role TEXT,
            created_at INTEGER, draft_text TEXT, critic_notes TEXT, grader_scores TEXT,
            gate_metrics TEXT, decision TEXT
        );
        CREATE TABLE claims (paper_stem TEXT, source_pdf_sha256 TEXT);
        INSERT INTO ingest_iterations VALUES
            (1, 'attempt-abc', 1, 'target_claims', 1792860000,
             'PRIVATE DRAFT', 'PRIVATE CRITIQUE', '{}',
             '{"target_claims":2,"keywords":3}', 'extracted');
        INSERT INTO claims VALUES ('synthetic', 'known-claim-row-hash');
    """)
    conn.commit()
    conn.close()


def fingerprint(root):
    return {str(path.relative_to(root)): file_sha256(path)
            for path in root.rglob('*') if path.is_file() and not path.is_symlink()}


def test_missing_inventory_never_initializes_databases_or_local_stores(tmp_path, monkeypatch, capsys):
    missing = tmp_path / 'missing/state.db'
    library_root = tmp_path / 'missing-library'
    runtime = tmp_path / 'runtime'
    monkeypatch.setenv('RESEARCHWIKI_JOBS_ROOT', str(runtime))
    assert main(['inventory', '--legacy', '--db-path', str(missing),
                 '--library-root', str(library_root), '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['databases'][0]['status'] == 'not_found'
    assert report['related_evidence']['jobs']['status'] == 'not_found'
    assert not missing.parent.exists() and not runtime.exists() and not library_root.exists()


def test_inventory_uses_explicit_library_and_keeps_payloads_private(tmp_path, monkeypatch):
    root = tmp_path / 'library'
    root.mkdir()
    db = tmp_path / 'legacy.db'
    history(db)
    monkeypatch.chdir(tmp_path)  # journal backup validation must use the selected library
    monkeypatch.setenv('RESEARCHWIKI_JOBS_ROOT', str(tmp_path / 'runtime'))
    monkeypatch.setattr('researchwiki.db.connection.get_connection',
                        lambda *a, **k: pytest.fail('inventory must not bootstrap state.db'))

    for name in ('.agent-output', '.grade-cache/synthetic', '.ingest/batch-demo',
                 '.llm-relay/pending', '.llm-relay/completed', '.mutation/backups'):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / '.agent-output/draft.md').write_text('PRIVATE SANDBOX')
    (root / '.ingest/batch-demo/checkpoint.json').write_text('{}')
    (root / '.llm-relay/pending/op.prompt.json').write_text('PRIVATE PROMPT')
    (root / '.llm-relay/completed/op.response.json').write_text('PRIVATE RESPONSE')
    backup = root / '.mutation/backups/page.md'
    backup.write_text('PRIVATE JOURNAL BACKUP')
    (root / '.mutation/journal.json').write_text(json.dumps({
        'version': 1, 'operation': 'promote', 'status': 'active',
        'backup_dir': str(backup.parent), 'details': {'attempt_id': 'attempt-abc'},
        'entries': [{'target': '/unused/target', 'backup': str(backup)},
                    {'target': '/unused/target', 'backup': str(tmp_path / 'outside')}],
    }))
    (root / '.mutation/corrupt.json').write_text('{')
    before = fingerprint(tmp_path)

    report = inventory_legacy(db_override=db, library_root=root)
    attempt = report['databases'][0]['attempts'][0]
    assert attempt['draft_rows'] == attempt['critic_note_rows'] == 1
    assert attempt['target_claim_history']['claim_count_total_when_recorded'] == 2
    related = report['related_evidence']
    journal = related['mutation_journals']['journals'][0]
    assert journal['legacy_history_match'] is True
    assert journal['backup_references_present'] == 1
    assert journal['backup_references_missing_or_outside'] == 1
    assert related['mutation_journals']['invalid_files'] == 1
    assert related['relay_mailbox']['matched_pairs'] == 1
    assert related['sandbox_outputs']['markdown_files'] == 1
    assert related['grading_cache']['paper_cache_directories'] == 1
    assert related['ingest_runs']['batch_checkpoints'] == 1
    assert 'PRIVATE' not in json.dumps(report)
    assert fingerprint(tmp_path) == before


def test_job_inventory_reads_ledgers_and_events_without_reconciling(library, monkeypatch):
    root, request, store = library
    monkeypatch.setenv('RESEARCHWIKI_JOBS_ROOT', str(store.base))
    db = root / 'legacy.db'
    history(db)
    record = store.submit_ingest(request)
    record = store.claim(record.job_id, 'fixture-owner')
    binding = store.ledger_binding(record)
    ingest_ledger.begin(binding)
    handoff = RelayHandoff(op_id='fixture-op', prompt_path=str(root / 'prompt.json'),
                           response_path=str(root / 'response.json'), timeout_seconds=60)
    writer = EventWriter(store.path(record.job_id) / 'events.jsonl')
    for sequence, kind in enumerate(('relay_waiting', 'relay_response_received'), 1):
        writer(IngestEvent(request_id=request.request_id, job_id=record.job_id,
                           sequence=sequence, timestamp='2026-10-24T00:00:00Z',
                           kind=kind, call_id='fixture-call', relay=handoff))
    writer.close()
    result = IngestResult(request_id=request.request_id, job_id=record.job_id,
                          execution='finished', outcome='draft_ready', publication='not_published',
                          attempt_id='attempt-abc')
    ingest_ledger.complete(binding, result)
    store.finish(record.job_id, 'fixture-owner', record.generation, result)
    invalid_job = store.root / ('f' * 32)
    invalid_job.mkdir()
    (invalid_job / 'manifest.json').write_text('{')
    before = fingerprint(store.root)

    report = inventory_legacy(db_override=db, library_root=root)['related_evidence']['jobs']
    assert report['job_count'] == 1 and report['invalid_job_directories'] == 1
    job = report['jobs'][0]
    assert job['legacy_history_match'] is True
    assert job['publication_ledger']['status'] == 'started'
    assert job['events']['complete_events'] == 2 and job['events']['open_relay_waits'] == 0
    assert job['publication'] == 'not_published'
    assert fingerprint(store.root) == before


def test_backup_includes_committed_wal_and_refuses_existing_destinations(tmp_path):
    source = tmp_path / 'source.db'
    history(source)
    writer = sqlite3.connect(source)
    try:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        writer.execute("INSERT INTO claims VALUES ('wal-only', 'wal-hash')")
        writer.commit()
        before = source.read_bytes()
        bundle = tmp_path / 'backup'
        result = backup_legacy_database(source, bundle)
        manifest = json.loads((bundle / 'manifest.json').read_text())
        assert manifest['database']['sha256'] == file_sha256(Path(result['database']))
        with sqlite3.connect(result['database']) as snapshot:
            assert snapshot.execute("SELECT COUNT(*) FROM claims WHERE paper_stem='wal-only'").fetchone()[0] == 1
            assert snapshot.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert source.read_bytes() == before
        sentinel = bundle / 'sentinel'
        sentinel.write_text('preserve me')
        with pytest.raises(RecordConflict, match='already exists'):
            backup_legacy_database(source, bundle)
        assert sentinel.read_text() == 'preserve me'
    finally:
        writer.close()


def test_corrupt_database_backup_preserves_source_and_removes_partial_bundle(tmp_path):
    source = tmp_path / 'corrupt.db'
    source.write_bytes(b'original non-SQLite bytes')
    bundle = tmp_path / 'backup'
    with pytest.raises(RecordStoreUnavailable):
        backup_legacy_database(source, bundle)
    assert not bundle.exists()
    assert source.read_bytes() == b'original non-SQLite bytes'
