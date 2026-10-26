"""Community job controls keep queueing, execution, inspection, and repair distinct."""

from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from researchwiki import __main__ as cli
from researchwiki.ingest_ledger import begin
from researchwiki.jobs import JobStore, JobSupervisor
from researchwiki.operations import IngestEvent, IngestResult, OperationError
from tests.test_job_workers import clean_environment, make_library, until


@pytest.fixture
def library(tmp_path):
    root, request = make_library(tmp_path, 'cli-library')
    store = JobStore(root, store_root=tmp_path/'runtime')
    return root, request, store


def invoke(library, capsys, arguments):
    root, _, store = library
    code = cli.main(['jobs', *arguments, '--library-root', str(root), '--store-root', str(store.base), '--json'])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result['schema_version'] == 1
    return code, result, captured.err


def test_help_and_inspection_ignore_caller_dotenv_and_do_not_initialize_models(library, tmp_path, monkeypatch, capsys):
    caller = tmp_path/'elsewhere'
    caller.mkdir()
    (caller/'.env').write_text('This is not a valid profile')
    monkeypatch.chdir(caller)
    monkeypatch.setattr(cli, '_load_dotenv', lambda *a: pytest.fail('jobs loaded a parent profile'))
    assert cli.main(['jobs', '--help']) == 0
    assert 'reconcile' in capsys.readouterr().out
    code, payload, _ = invoke(library, capsys, ['list'])
    assert code == 0 and payload['jobs'] == []
    assert not library[2].root.exists()


def test_direct_submission_is_idempotent_readable_and_queue_only(library, tmp_path, monkeypatch, capsys):
    root, request, store = library
    monkeypatch.setattr(JobSupervisor, 'start_job', lambda *a, **k: pytest.fail('submission started work'))
    options = tmp_path/'options.json'
    options.write_text(json.dumps(request.options.to_dict()))
    arguments = ['submit', 'source.pdf', '--request-id', 'cli-request', '--options-file', str(options)]
    code, payload, _ = invoke(library, capsys, arguments)
    job = payload['job']
    assert code == 0 and job['execution'] == 'queued'
    assert job['request']['source_pdf'] == str(root/'source.pdf')
    assert job['request']['options'] == request.options.to_dict()
    Path(request.source_pdf).unlink()
    again = invoke(library, capsys, arguments)[1]['job']
    assert again == job
    manifest = store.path(job['job_id'])/'manifest.json'
    before = manifest.read_bytes(), manifest.stat().st_mtime_ns
    for command in (['show', job['job_id']], ['list'], ['list', '--state', 'queued'], ['events', job['job_id']]):
        assert invoke(library, capsys, command)[0] == 0
    assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before
    assert not (store.path(job['job_id'])/'publication.json').exists()


def test_versioned_request_submission_and_global_profile_stays_unloaded(library, tmp_path, monkeypatch, capsys):
    root, request, store = library
    profile = root/'selected.env'
    profile.write_text('RW_MODELS_CONFIG=config/models.jobs.yaml\n')
    data = tmp_path/'request.json'
    data.write_text(json.dumps(request.to_dict()))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, '_load_dotenv', lambda *a: pytest.fail('profile loaded in parent'))
    code = cli.main(['--env-file', 'cli-library/selected.env', 'jobs', '--json',
                     '--library-root', str(root), '--store-root', str(store.base),
                     'submit', '--request-file', str(data)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload['job']['request']['provider_profile'] == str(profile)
    assert payload['job']['execution'] == 'queued'
    assert invoke(library, capsys, ['submit', '--request-file', str(data), '--request-id', 'conflict'])[0] == 1
    code = cli.main(['--env-file', str(profile), 'jobs', 'show', payload['job']['job_id'],
                     '--library-root', str(root), '--store-root', str(store.base), '--json'])
    assert code == 1 and 'stored profile' in json.loads(capsys.readouterr().out)['error']['message']


@pytest.mark.parametrize('arguments', [
    ['show', 'not-a-job'], ['show', 'a'*32], ['submit', 'source.pdf'],
    ['submit'], ['submit', 'missing.pdf', '--request-id', 'missing'],
    ['events', 'a'*32, '--after-sequence', '-1'], ['list', '--unknown'],
])
def test_usage_errors_have_clean_versioned_json(library, capsys, arguments):
    code, payload, _ = invoke(library, capsys, arguments)
    assert code == 1 and payload['error']['code'] == 'invalid_request'


def test_explicit_library_root_is_required(capsys):
    assert cli.main(['jobs', 'list', '--json']) == 1
    assert '--library-root' in json.loads(capsys.readouterr().out)['error']['message']


@pytest.mark.parametrize('damage', ['future', 'invalid_json'])
def test_corrupt_job_is_environment_failure_and_is_not_rewritten(library, capsys, damage):
    _, request, store = library
    record = store.submit_ingest(request)
    path = store.path(record.job_id)/'manifest.json'
    data = json.loads(path.read_text())
    data['schema_version'] = 999
    path.write_text('{' if damage == 'invalid_json' else json.dumps(data))
    before = path.read_bytes()
    for command in (['show', record.job_id], ['list'], ['submit', 'source.pdf', '--request-id', 'another']):
        code, payload, _ = invoke(library, capsys, command)
        assert code == 2 and payload['error']['category'] == 'environment'
    assert path.read_bytes() == before


def test_cancel_and_explicit_retry_do_not_start_a_worker(library, monkeypatch, capsys):
    _, request, store = library
    monkeypatch.setattr(JobSupervisor, 'start_job', lambda *a, **k: pytest.fail('queue control started work'))
    queued = store.submit_ingest(request)
    cancelled = invoke(library, capsys, ['cancel', queued.job_id])
    assert cancelled[0] == 0 and cancelled[1]['job']['execution'] == 'cancelled'
    assert invoke(library, capsys, ['retry', queued.job_id])[0] == 1
    first = store.submit_ingest(replace(request, request_id='failed-attempt'))
    running = store.claim(first.job_id, 'test-owner')
    store.finish(first.job_id, 'test-owner', running.generation, IngestResult(
        request_id=first.request.request_id, job_id=first.job_id, execution='finished',
        outcome='draft_ready', publication='not_published', gate_reasons=('coverage',)))
    code, retried, _ = invoke(library, capsys, ['retry', first.job_id])
    assert code == 0 and retried['job']['parent_job_id'] == first.job_id
    assert retried['job']['execution'] == 'queued'
    active = store.submit_ingest(replace(request, request_id='active'))
    store.claim(active.job_id, 'active-owner')
    assert invoke(library, capsys, ['cancel', active.job_id])[0] == 1


def test_event_cursor_preserves_schema_and_corrupt_events_do_not_break_show(library, capsys):
    _, request, store = library
    record = store.submit_ingest(request)
    event = IngestEvent(request_id=request.request_id, job_id=record.job_id, sequence=1,
                        timestamp='2026-10-20T12:00:00Z', kind='operation_started', call_id='operation')
    path = store.path(record.job_id)/'events.jsonl'
    path.write_text(json.dumps(event.to_dict())+'\n{"partial":')
    code, payload, _ = invoke(library, capsys, ['events', record.job_id])
    assert code == 0 and payload['events'] == [event.to_dict()] and payload['next_sequence'] == 1
    assert invoke(library, capsys, ['events', record.job_id, '--after-sequence', '1'])[1]['events'] == []
    path.write_text(path.read_text()+'}\n')
    assert invoke(library, capsys, ['events', record.job_id])[0] == 2
    assert invoke(library, capsys, ['show', record.job_id])[0] == 0


@pytest.mark.parametrize('outcome,publication,promotion,category,expected', [
    ('published', 'published', 'auto', None, 0),
    ('published', 'published', 'always', 'environment', 0),
    ('draft_ready', 'not_published', 'auto', None, 1),
    ('draft_ready', 'not_published', 'never', None, 0),
    ('failed', 'not_published', 'auto', 'validation', 1),
    ('failed', 'unknown', 'auto', 'environment', 2),
    ('failed', 'unknown', 'auto', 'internal', 3),
])
def test_run_exit_contract_preserves_terminal_receipt(library, monkeypatch, capsys,
                                                     outcome, publication, promotion, category, expected):
    _, request, store = library
    record = store.submit_ingest(request)
    result = IngestResult(request_id=request.request_id, job_id=record.job_id, execution='finished',
                          outcome=outcome, publication=publication, promotion=promotion,
                          error=OperationError(code='test_error', category=category, message='Test',
                                               retry='reconcile_first') if category else None)
    terminal = replace(record, execution='finished', result=result)
    monkeypatch.setattr(JobSupervisor, 'start_job', lambda *a, **k: object())
    monkeypatch.setattr(JobSupervisor, 'wait', lambda *a, **k: terminal)
    code, payload, _ = invoke(library, capsys, ['run', record.job_id])
    assert code == expected and payload['job']['result'] == result.to_dict()


def run_cli(root, base, arguments, *, cwd, env, timeout=45):
    return subprocess.run([sys.executable, '-m', 'researchwiki', 'jobs', *arguments,
                           '--library-root', str(root), '--store-root', str(base), '--json'],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def test_real_cli_worker_publishes_from_another_cwd_and_refuses_replay(library, tmp_path):
    root, request, store = library
    caller = tmp_path/'caller'
    caller.mkdir()
    (caller/'.env').write_text('Malformed caller profile must not be loaded')
    source = root/'inbox/paper.pdf'
    Path(request.source_pdf).rename(source)
    request = replace(request, source_pdf=str(source))
    path = caller/'request.json'
    path.write_text(json.dumps(request.to_dict()))
    env = clean_environment(tmp_path)
    from researchwiki.env_profiles import ACTIVE_ENV_FILE_VAR, DOTENV_PROVENANCE_VAR
    env.update(RW_MODELS_CONFIG=str(caller/'foreign-models.yaml'), OPENAI_API_KEY='foreign-profile-key')
    env[ACTIVE_ENV_FILE_VAR] = str(caller/'foreign.env')
    env[DOTENV_PROVENANCE_VAR] = json.dumps({'path': str(caller/'foreign.env'),
                                           'keys': ['RW_MODELS_CONFIG', 'OPENAI_API_KEY']})
    submitted = run_cli(root, store.base, ['submit', '--request-file', str(path)], cwd=caller, env=env)
    job = json.loads(submitted.stdout)['job']
    assert submitted.returncode == 0 and job['execution'] == 'queued'
    completed = run_cli(root, store.base, ['run', job['job_id']], cwd=caller, env=env)
    terminal = json.loads(completed.stdout)['job']
    assert completed.returncode == 0, completed.stderr
    assert terminal['execution'] == 'finished' and terminal['result']['outcome'] == 'published'
    assert not source.exists() and Path(terminal['result']['pdf_path']).is_file()
    for action in ('run', 'retry', 'cancel'):
        refused = run_cli(root, store.base, [action, job['job_id']], cwd=caller, env=env)
        assert refused.returncode == 1 and json.loads(refused.stdout)['error'] is not None
    assert json.loads(run_cli(root, store.base, ['show', job['job_id']], cwd=caller, env=env).stdout)['job'] == terminal
    repeat = run_cli(root, store.base, ['submit', '--request-file', str(path)], cwd=caller, env=env)
    assert json.loads(repeat.stdout)['job'] == terminal


@pytest.mark.parametrize('proof', [True, False])
def test_cli_reconciliation_distinguishes_proved_and_unknown_publication(library, tmp_path, proof):
    root, request, store = library
    job = store.submit_ingest(request)
    running = store.claim(job.job_id, 'exited-worker')
    if proof:
        begin(store.ledger_binding(running))
    completed = run_cli(root, store.base, ['reconcile', job.job_id], cwd=tmp_path, env=clean_environment(tmp_path))
    payload = json.loads(completed.stdout)
    assert completed.returncode == (0 if proof else 2), completed.stderr
    assert payload['job']['result']['publication'] == ('not_published' if proof else 'unknown')
    assert payload['job']['execution'] == ('finished' if proof else 'interrupted')


@pytest.mark.skipif(os.name != 'posix', reason='POSIX foreground signal test')
def test_foreground_relay_handoff_and_ctrl_c_preserve_recovery(library, tmp_path):
    root, request, store = library
    config = json.loads((root/'config/models.jobs.yaml').read_text())
    for role in config['roles'].values(): role['provider'] = 'chat-relay'
    (root/'config/models.jobs.yaml').write_text(json.dumps(config))
    (root/'.env').write_text(f'RW_MODELS_CONFIG=config/models.jobs.yaml\nRESEARCHWIKI_DB_PATH={root}/state.db\n')
    request = replace(request, options=replace(request.options, use_stub=False, use_llm_reconcile=True))
    job = store.submit_ingest(request)
    supervisor = JobSupervisor(root, store_root=store.base)
    process = subprocess.Popen([sys.executable, '-m', 'researchwiki', 'jobs', 'run', job.job_id,
                                '--library-root', str(root), '--store-root', str(store.base), '--json'],
                               cwd=tmp_path, env=clean_environment(tmp_path), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        until(lambda: supervisor.pending_relays(job.job_id) or process.poll() is not None, timeout=30)
        if process.poll() is not None:
            pytest.fail('relay worker exited early: '+process.communicate()[1])
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    payload = json.loads(stdout)
    assert process.returncode == 1 and payload['error']['code'] == 'interrupted'
    assert payload['job']['execution'] == 'interrupted'
    assert '📨 LLM relay pending' in stderr
    recovered = run_cli(root, store.base, ['reconcile', job.job_id], cwd=tmp_path, env=clean_environment(tmp_path))
    assert recovered.returncode == 0 and json.loads(recovered.stdout)['job']['result']['publication'] == 'not_published'


def test_metadata_cli_import_does_not_initialize_models_or_read_dotenv(library, tmp_path):
    root, _, store = library
    probe = f'''import sys
from researchwiki.__main__ import main
assert main(['jobs', 'list', '--library-root', {str(root)!r}, '--store-root', {str(store.base)!r}, '--json']) == 0
assert 'researchwiki.agents.model_config' not in sys.modules
assert 'torch' not in sys.modules
'''
    completed = subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, env=clean_environment(tmp_path),
                                capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr


def test_invalid_request_schema_and_options_never_queue_work(library, tmp_path, capsys):
    root, request, store = library
    path = tmp_path/'invalid.json'
    for data in (dict(request.to_dict(), schema_version=999), dict(request.to_dict(), unknown=True),
                 dict(request.to_dict(), library_root=str(tmp_path/'another-library'))):
        path.write_text(json.dumps(data))
        assert invoke(library, capsys, ['submit', '--request-file', str(path)])[0] == 1
    path.write_text('{"max_model_calls": 0}')
    assert invoke(library, capsys, ['submit', 'source.pdf', '--request-id', 'invalid', '--options-file', str(path)])[0] == 1
    assert not store.root.exists()
    assert Path(request.source_pdf).is_file() and (root/'wiki/index.md').is_file()


def test_internal_failures_keep_json_clean_and_tracebacks_on_stderr(library, monkeypatch, capsys):
    def fail(*a): raise RuntimeError('private diagnostic detail')
    monkeypatch.setattr(JobSupervisor, 'list_jobs', fail)
    code, payload, stderr = invoke(library, capsys, ['list'])
    assert code == 3 and payload['error']['code'] == 'internal_error'
    assert 'private diagnostic detail' not in json.dumps(payload)
    assert 'private diagnostic detail' in stderr


def test_human_output_distinguishes_queue_and_published_maintenance(library, capsys):
    root, request, store = library
    queued = store.submit_ingest(request)
    base = ['--library-root', str(root), '--store-root', str(store.base)]
    assert cli.main(['jobs', 'show', queued.job_id, *base]) == 0
    assert 'Run: researchwiki jobs run' in capsys.readouterr().out
    running = store.claim(queued.job_id, 'owner')
    store.finish(queued.job_id, 'owner', running.generation, IngestResult(
        request_id=request.request_id, job_id=queued.job_id, execution='finished', outcome='published',
        publication='published', projections='maintenance_pending', gate_reasons=('forced exception',)))
    assert cli.main(['jobs', 'show', queued.job_id, *base]) == 0
    output = capsys.readouterr().out
    assert 'publication: published' in output and 'projections: maintenance_pending' in output
    assert 'jobs reconcile' in output and 'forced exception' in output
