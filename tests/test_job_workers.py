"""Real process boundaries, with synthetic sources and deterministic providers."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from researchwiki.jobs import JobConflict, JobStore, JobSupervisor
from researchwiki.operations import IngestOptions, IngestRequest


REPO = Path(__file__).resolve().parents[1]


def make_library(parent, name, *, provider='openai-compatible', mixed=False):
    root = parent / name
    root.mkdir()
    shutil.copytree(REPO / 'config', root / 'config')
    shutil.copytree(REPO / 'prompts', root / 'prompts')
    for folder in ('wiki/other', 'papers', 'inbox'):
        (root / folder).mkdir(parents=True)
    (root / 'wiki/index.md').write_text('# Index\n\n## other\n')
    spec = importlib.util.spec_from_file_location('acceptance_fixture', REPO / '.github/scripts/fresh_install_acceptance.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = root / 'source.pdf'
    module.write_fixture_pdf(source, title='A Test of Local Jobs', author='Ada Lovelace')
    roles = {role: {'provider': provider, 'model': name, 'temperature': 0.1, 'max_tokens': 64}
             for role in ('author', 'critic', 'judge', 'extractor', 'proposer', 'classifier')}
    if mixed:
        roles['author']['provider'] = 'chat-relay'
    config = {'base_url': 'http://127.0.0.1:9/v1', 'roles': roles}
    (root / 'config/models.jobs.yaml').write_text(json.dumps(config))
    profile = root / '.env'
    profile.write_text(f'RW_MODELS_CONFIG=config/models.jobs.yaml\nRESEARCHWIKI_DB_PATH={root}/state.db\n'
                       + (f'OPENAI_API_KEY=secret-{name}\n' if provider != 'chat-relay' else ''))
    profile.chmod(0o600)
    options = IngestOptions(use_stub=True, use_semantic=False, use_llm_reconcile=False, n_drafts=1,
                            max_evolve=0, promotion='always', no_cross_link=True,
                            title='A Test of Local Jobs', year=2026, authors=('Ada Lovelace',))
    return root, IngestRequest(request_id=f'request-{name}', library_root=str(root), source_pdf=str(source), options=options)


def clean_environment(tmp_path):
    from researchwiki.env_profiles import ACTIVE_ENV_FILE_VAR, DOTENV_PROVENANCE_VAR, INHERITED_ENV_VAR
    env = dict(os.environ)
    for key in ('RW_MODELS_CONFIG', 'RW_LLM_PROVIDER', 'RW_LLM_BASE_URL', 'ANTHROPIC_BASE_URL',
                'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'RESEARCHWIKI_DB_PATH',
                ACTIVE_ENV_FILE_VAR, DOTENV_PROVENANCE_VAR, INHERITED_ENV_VAR):
        env.pop(key, None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(tmp_path / 'empty-hf-cache'))
    return env


def until(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    pytest.fail('worker did not reach the expected state')


PROBE_WORKER = r'''
import json, os, sys, time
from pathlib import Path
from researchwiki.jobs import JobStore
from researchwiki.jobs.worker import run_worker
from researchwiki.operations import IngestResult
from researchwiki.agents import llm
store = JobStore(sys.argv[1], store_root=sys.argv[2])
job_id, mode = sys.argv[3:5]
def operation(request, *, event_sink):
    event_sink.bind_attempt('probe-attempt')
    root = Path(request.library_root)
    if mode == 'hold':
        with (root / 'model-work.txt').open('a') as stream:
            stream.write('once\n')
        (root / 'ready').write_text('ready')
        time.sleep(30)
    def adapter(**kwargs):
        assert os.environ.get('OPENAI_API_KEY') == 'secret-' + kwargs['model'], 'credential crossover'
        return llm.LLMResponse('answer', 'api:' + kwargs['model'], 0.1, 2, 1)
    llm.call_openai_compatible = adapter
    replies = []
    if mode == 'mixed':
        replies.append(llm.call(phase='reconcile', prompt='metadata test', event_sink=event_sink).model)
    replies.append(llm.call(phase='author', prompt='worker test', event_sink=event_sink).model)
    (root / 'probe.json').write_text(json.dumps({'models': replies, 'profile': request.provider_profile,
                                               'db': os.environ.get('RESEARCHWIKI_DB_PATH')}))
    return IngestResult(request_id=request.request_id, execution='finished', outcome='draft_ready',
                        publication='not_published', projections='not_applicable', attempt_id='probe-attempt',
                        draft_path=str(root / 'probe.json'))
raise SystemExit(run_worker(store, job_id, operation=operation))
'''


def probe_process(root, store, job_id, env, *, mode='api'):
    return subprocess.Popen([sys.executable, '-c', PROBE_WORKER, str(root), str(store.base), job_id, mode],
                            cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_default_worker_runs_real_ingestion_and_persists_terminal_result(tmp_path):
    root, request = make_library(tmp_path, 'native-worker')
    supervisor = JobSupervisor(root, store_root=tmp_path / 'runtime', environment=clean_environment(tmp_path))
    job = supervisor.submit_ingest(request)
    handle = supervisor.start_job(job.job_id, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        terminal = supervisor.wait(handle, timeout=60)
    finally:
        if handle.process.poll() is None:
            handle.process.kill()
            handle.process.wait()
    assert handle.process.returncode == 0
    assert terminal.execution == 'finished' and terminal.result.outcome == 'published'
    assert terminal.result.job_id == job.job_id and terminal.identity is not None
    assert Path(terminal.result.pdf_path).read_bytes() == Path(request.source_pdf).read_bytes()
    events = supervisor.read_events(job.job_id)
    assert events[-1].kind == 'operation_finished' and events[-1].publication == 'published'
    assert {event.job_id for event in events} == {job.job_id}
    assert 'secret-native-worker' not in (supervisor.store.path(job.job_id) / 'manifest.json').read_text()
    assert (supervisor.store.path(job.job_id) / 'diagnostic.log').is_file()


def test_two_libraries_select_their_own_profiles_without_parent_mutation(tmp_path):
    from researchwiki.env_profiles import ACTIVE_ENV_FILE_VAR, DOTENV_PROVENANCE_VAR
    from researchwiki.jobs.environment import child_environment

    a, req_a = make_library(tmp_path, 'library-a')
    b, req_b = make_library(tmp_path, 'library-b')
    store_a, store_b = JobStore(a, store_root=tmp_path / 'runtime'), JobStore(b, store_root=tmp_path / 'runtime')
    job_a, job_b = store_a.submit_ingest(req_a), store_b.submit_ingest(req_b)
    parent = clean_environment(tmp_path)
    parent.update(OPENAI_API_KEY='secret-library-a', RW_MODELS_CONFIG=str(a / 'config/models.jobs.yaml'),
                  RESEARCHWIKI_DB_PATH=str(tmp_path / 'foreign.db'))
    parent[ACTIVE_ENV_FILE_VAR] = str(a / '.env')
    parent[DOTENV_PROVENANCE_VAR] = json.dumps({'path': str(a / '.env'), 'keys': ['OPENAI_API_KEY', 'RW_MODELS_CONFIG']})
    before = dict(parent)
    cwd_before = Path.cwd()
    processes = [probe_process(root, store, job.job_id, child_environment(parent))
                 for root, store, job in ((a, store_a, job_a), (b, store_b, job_b))]
    try:
        for process in processes:
            _, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    for root, store, job in ((a, store_a, job_a), (b, store_b, job_b)):
        terminal = store.get_job(job.job_id)
        assert terminal.result.outcome == 'draft_ready'
        data = json.loads((root / 'probe.json').read_text())
        assert data == {'models': ['api:' + root.name], 'profile': str(root / '.env'), 'db': str(root / 'state.db')}
        assert all(event.request_id == job.request.request_id for event in read_events(store, job.job_id))
    assert parent == before and Path.cwd() == cwd_before
    assert not (tmp_path / 'foreign.db').exists()


@pytest.mark.parametrize('mode', ['relay', 'mixed'])
def test_relay_worker_exposes_exact_handoffs_and_keeps_foreground_protocol(tmp_path, mode):
    root, request = make_library(tmp_path, mode + '-worker', provider='chat-relay' if mode == 'relay' else 'openai-compatible', mixed=mode == 'mixed')
    store = JobStore(root, store_root=tmp_path / 'runtime')
    job = store.submit_ingest(request)
    process = probe_process(root, store, job.job_id, clean_environment(tmp_path), mode=mode)
    supervisor = JobSupervisor(root, store_root=store.base)
    try:
        handoff, = until(lambda: supervisor.pending_relays(job.job_id))
        assert Path(handoff.prompt_path).is_file()
        assert supervisor.refresh_job(job.job_id).execution == 'running'
        Path(handoff.response_path).parent.mkdir(parents=True, exist_ok=True)
        Path(handoff.response_path).write_text(json.dumps({'via': 'test/relay-model-1', 'response': 'answer'}))
        _, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    terminal = store.get_job(job.job_id)
    assert terminal.result.outcome == 'draft_ready' and terminal.result.usage.status == 'unavailable'
    models = json.loads((root / 'probe.json').read_text())['models']
    assert models[-1] == 'chat-relay:test/relay-model-1'
    if mode == 'mixed':
        assert models[0] == 'api:mixed-worker'
    assert '📨 LLM relay pending' in stderr
    assert supervisor.pending_relays(job.job_id) == ()
    events = supervisor.read_events(job.job_id)
    assert any(event.kind == 'relay_waiting' for event in events)
    assert any(event.model == 'chat-relay:test/relay-model-1' for event in events)


def test_real_owner_excludes_second_worker_and_process_death_blocks_replay(tmp_path):
    root, request = make_library(tmp_path, 'lost-worker')
    store = JobStore(root, store_root=tmp_path / 'runtime')
    job = store.submit_ingest(request)
    process = probe_process(root, store, job.job_id, clean_environment(tmp_path), mode='hold')
    supervisor = JobSupervisor(root, store_root=store.base)
    try:
        until(lambda: (root / 'ready').exists())
        assert supervisor.refresh_job(job.job_id).execution == 'running'
        with pytest.raises(JobConflict):
            supervisor.cancel_queued_job(job.job_id)
        second = probe_process(root, store, job.job_id, clean_environment(tmp_path), mode='hold')
        _, stderr = second.communicate(timeout=20)
        assert second.returncode == 1, stderr
        assert (root / 'model-work.txt').read_text() == 'once\n'
        process.kill()
        process.communicate(timeout=20)
        interrupted = supervisor.refresh_job(job.job_id)
        assert interrupted.execution == 'interrupted' and interrupted.result.publication == 'unknown'
        with pytest.raises(JobConflict):
            supervisor.retry_job(job.job_id)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_worker_revalidates_queued_inputs_before_model_work(tmp_path):
    root, request = make_library(tmp_path, 'missing-input')
    supervisor = JobSupervisor(root, store_root=tmp_path / 'runtime', environment=clean_environment(tmp_path))
    job = supervisor.submit_ingest(request)
    Path(request.source_pdf).unlink()
    handle = supervisor.start_job(job.job_id, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    terminal = supervisor.wait(handle, timeout=20)
    assert terminal.execution == 'finished' and terminal.result.publication == 'not_published'
    assert terminal.result.error.code == 'invalid_request'
    assert not (supervisor.store.path(job.job_id) / 'events.jsonl').exists()


def test_job_api_import_does_not_initialize_model_configuration(tmp_path):
    probe = 'import sys; import researchwiki.jobs; assert "researchwiki.agents.model_config" not in sys.modules; assert "torch" not in sys.modules'
    process = subprocess.run([sys.executable, '-c', probe], cwd=tmp_path,
                             env=clean_environment(tmp_path), text=True, capture_output=True, timeout=20)
    assert process.returncode == 0, process.stderr


def test_retry_worker_reports_source_and_configuration_changes(tmp_path):
    root, request = make_library(tmp_path, 'retry-worker')
    store = JobStore(root, store_root=tmp_path / 'runtime')
    first = store.submit_ingest(request)
    process = probe_process(root, store, first.job_id, clean_environment(tmp_path))
    _, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, stderr
    original = store.get_job(first.job_id)
    assert original.result.outcome == 'draft_ready'
    Path(request.source_pdf).write_bytes(Path(request.source_pdf).read_bytes() + b'\n% updated')
    config_path = root / 'config/models.jobs.yaml'
    configuration = json.loads(config_path.read_text())
    configuration['roles']['author']['model'] = 'retry-worker-v2'
    config_path.write_text(json.dumps(configuration))
    profile = root / '.env'
    profile.write_text(profile.read_text().replace('secret-retry-worker', 'secret-retry-worker-v2'))
    retry = store.retry_job(first.job_id)
    process = probe_process(root, store, retry.job_id, clean_environment(tmp_path))
    _, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, stderr
    final = store.get_job(retry.job_id)
    assert final.source_changed is True and final.configuration_changed is True
    assert final.identity != original.identity and final.parent_job_id == first.job_id
    assert 'Source content changed since the parent job' in final.result.warnings
    assert 'Provider configuration changed since the parent job' in final.result.warnings
    assert 'secret-' not in (store.path(retry.job_id) / 'manifest.json').read_text()


def read_events(store, job_id):
    from researchwiki.jobs.events import read_events as read
    return read(store, job_id)
