import json

import pytest

from researchwiki.tasks import claim_overlap as co


@pytest.mark.parametrize('pending,args', [([], []), (['a', 'b'], ['--limit', '0'])])
def test_json_empty_and_limit_zero_parse_directly(monkeypatch, capsys, pending, args):
    monkeypatch.setattr(co, 'find_backlog', lambda: pending)
    assert co.main(['--backlog', '--dry-run', '--json', *args]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['results'] == []
    assert result['withheld'] == len(pending)


def test_progress_flushed_before_next_stem_and_failure_names_retryable(tmp_path, monkeypatch, capsys):
    from researchwiki.errors import EnvironmentFailure
    progress = tmp_path / 'progress.jsonl'
    monkeypatch.setattr(co, 'find_backlog', lambda: ['a', 'b', 'c'])
    calls = []
    def run(stem, **kwargs):
        calls.append(stem)
        if stem == 'b':
            # This read happens while the command is still running.
            first = json.loads(progress.read_text().splitlines()[0])
            assert first['stem'] == 'a' and first['completed'] == 1
            raise EnvironmentFailure('provider unavailable')
        return {'applied': [], 'edge_only': [], 'coincidence': [], 'judge_failed': [],
                'n_candidates': 1, 'judged_pairs': 1, 'reused_judgments': 0, 'status': 'preview'}
    monkeypatch.setattr(co, 'run', run)
    assert co.main(['--backlog', '--dry-run', '--json', '--progress-json', str(progress)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['retryable_stems'] == ['b', 'c']
    events = [json.loads(line) for line in progress.read_text().splitlines()]
    assert events[-1]['event'] == 'run_finished' and events[-1]['retryable_stems'] == ['b', 'c']
    assert calls == ['a', 'b']


def test_json_unknown_pair_and_provider_failures_parse(monkeypatch, capsys):
    from researchwiki.errors import EnvironmentFailure
    monkeypatch.setattr(co, 'review_pair', lambda *a, **k: (_ for _ in ()).throw(ValueError('unknown claim')))
    assert co.main(['--pair', 'a#x', 'b#y', '--json']) == 1
    assert json.loads(capsys.readouterr().out)['exit_code'] == 1
    monkeypatch.setattr(co, 'run', lambda *a, **k: (_ for _ in ()).throw(EnvironmentFailure('provider gone')))
    assert co.main(['a', '--json']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'incomplete'


def test_progress_is_visible_during_isolated_subprocess(tmp_path):
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path
    progress = tmp_path / 'progress.jsonl'
    release = tmp_path / 'release'
    script = '''
import sys, time
from pathlib import Path
from researchwiki.tasks import claim_overlap as co
from researchwiki.overlap.receipts import new_run_id
co.find_backlog=lambda: ['a','b']
progress,release=map(Path,sys.argv[1:])
def run(stem,**kwargs):
    if stem=='b':
        deadline=time.monotonic()+10
        while not release.exists():
            if time.monotonic()>deadline: raise RuntimeError('parent never observed progress')
            time.sleep(.01)
    return {'applied':[],'edge_only':[],'coincidence':[],'judge_failed':[],
            'n_candidates':1,'judged_pairs':1,'status':'preview'}
co.run=run
raise SystemExit(co.main(['--backlog','--dry-run','--json','--progress-json',str(progress)]))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    child = subprocess.Popen([sys.executable, '-u', '-c', script, str(progress), str(release)],
                             cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not progress.exists() or not progress.read_text():
            assert time.monotonic() < deadline
            assert child.poll() is None
            time.sleep(.01)
        first = json.loads(progress.read_text().splitlines()[0])
        assert first['stem'] == 'a' and child.poll() is None
        release.touch()
        out, err = child.communicate(timeout=10)
        assert child.returncode == 0, err
        assert len(json.loads(out)['results']) == 2
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_unknown_revert_with_equals_is_user_error(capsys):
    assert co.main(['--revert=nonexistent-run-id', '--json']) == 1
    assert json.loads(capsys.readouterr().out)['exit_code'] == 1
