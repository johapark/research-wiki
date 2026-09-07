"""Batch regressions using real child environments and controlled interruption."""

import subprocess
import sys
import threading

import pytest

from researchwiki.tasks import _ingest_batch as batch


def test_worker_preserves_profile_and_forwards_non_ascii_handoff(tmp_path, monkeypatch, capsys):
    from researchwiki.env_profiles import load_profile

    monkeypatch.chdir(tmp_path)
    # Register restoration of the variable inserted by the real loader.
    monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    monkeypatch.delenv("RW_LLM_PROVIDER")
    profile = tmp_path / ".env.relay"
    profile.write_text("RW_LLM_PROVIDER=chat-relay\n")
    (tmp_path / ".env").write_text("RW_MODELS_CONFIG=config/unselected.yaml\n")
    load_profile(profile, required=True)
    monkeypatch.setenv("PYTHONIOENCODING", "ascii:backslashreplace")
    real_popen = subprocess.Popen
    # Substitute a tiny diagnostic child for ingest, retaining all real worker
    # Popen arguments, environment inheritance, stderr decoding and forwarding.
    code = (
        "import json, os; from pathlib import Path; "
        "from researchwiki.__main__ import _load_dotenv; _load_dotenv(); "
        "print(json.dumps([os.environ.get('RW_LLM_PROVIDER'), "
        "os.environ.get('RW_MODELS_CONFIG')])); "
        "from researchwiki.agents.relay import _emit_handoff_message; "
        "_emit_handoff_message(Path('pending.json'), Path('response.json'), 'author', 600)"
    )
    monkeypatch.setattr(batch.subprocess, "Popen", lambda cmd, **kw: real_popen(
        [sys.executable, "-c", code], **kw,
    ))
    pdf = str(tmp_path / "paper.pdf")
    result = batch._worker(pdf, tmp_path, ["agent", "ingest"], [])
    assert result["returncode"] == 0
    log = batch._worker_log_path(tmp_path, pdf).read_text()
    assert '["chat-relay", null]' in log
    err = capsys.readouterr().err
    assert "📨 LLM relay pending" in err
    assert "pending.json" in err and "response.json" in err


@pytest.mark.parametrize("returncode", [0, 2])
def test_interrupt_checkpoints_running_worker_outcome(tmp_path, monkeypatch, returncode):
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"pdf")
    release, started = threading.Event(), threading.Event()

    def worker(pdf, *args):
        started.set()
        assert release.wait(5)
        if returncode == 0:
            source.rename(tmp_path / "landed.pdf")
        return {"input": pdf, "status": "completed" if returncode == 0 else "failed",
                "returncode": returncode}

    def interrupted(futures):
        assert started.wait(5)
        release.set()
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "_worker", worker)
    monkeypatch.setattr(batch.concurrent.futures, "as_completed", interrupted)
    batch._atomic_write_json(tmp_path / "plan.json", {
        "inputs": [str(source)], "workers": 1, "workers_explicit": True,
        "subcommand": ["agent", "ingest"], "extra_args": ["--stub"],
    })
    assert batch._run_batch(tmp_path, [str(source)], 1, ["agent", "ingest"], ["--stub"]) == 1
    state = batch._read_checkpoint(tmp_path)
    bucket = "completed" if returncode == 0 else "failed"
    assert state[bucket][str(source)]["returncode"] == returncode
    if returncode == 0:
        assert batch.resume_batch(tmp_path, no_retry=False) == 0
        assert not batch._read_checkpoint(tmp_path).get("unresumable")
