"""Resume must not mix different input bytes, prompts or model configurations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchwiki.tasks import _batch_identity as identity, _ingest_batch as batch
from researchwiki.tasks import agent, ingest


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "author-system-research.md").write_text("Original prompt\n")
    (tmp_path / "CLAUDE.md").write_text("Original contract\n")
    config = tmp_path / "models.yaml"
    config.write_text("base_url: https://example.test/v1\nroles:\n  author:\n    model: original\n")
    monkeypatch.setenv("RW_MODELS_CONFIG", str(config))
    inputs = [str(tmp_path / name) for name in ("finished.pdf", "pending.pdf")]
    for name in inputs:
        Path(name).write_bytes(b"%PDF-1.4 original")
    calls = []

    def worker(pdf, *args):
        calls.append(pdf)
        rc = 0 if pdf == inputs[0] else 2
        return {"input": pdf, "status": "completed" if rc == 0 else "failed", "returncode": rc}

    monkeypatch.setattr(batch, "_worker", worker)

    def start(*, extra_args=None, per_input_args=None, subcommand=None):
        assert batch.new_batch(inputs, subcommand or ["agent", "ingest"], extra_args or [],
                               workers=1, per_input_args=per_input_args) == 1
        directory, = (tmp_path / ".ingest").glob("batch-*")
        calls.clear()
        return directory

    return tmp_path, config, inputs, calls, start


def test_plan_freezes_identity_without_credentials(run, monkeypatch):
    root, _, inputs, calls, start = run
    monkeypatch.setenv("OPENAI_API_KEY", "secret-before-start")
    directory = start()
    raw = (directory / "plan.json").read_text()
    saved = json.loads(raw)["identity"]
    assert saved["inputs"][inputs[1]]["pdf_sha256"]
    assert saved["prompts"]["prompts/author-system-research.md"]
    assert saved["framework"]["version"] and saved["framework"]["source_sha256"]
    assert saved["models"]["phases"]["author"]["model"] == "original"
    assert saved["models"]["endpoints"]["openai"] == "https://example.test/v1"
    assert "secret-before-start" not in raw and "API_KEY" not in raw
    # Credential rotation doesn't change which workflow produced the outputs.
    monkeypatch.setenv("OPENAI_API_KEY", "secret-after-start")
    assert batch.resume_batch(directory, no_retry=False) == 1  # worker fails, not identity
    assert calls == [inputs[1]]
    assert (root / "pending.pdf").exists()


@pytest.mark.parametrize("change, field", [
    ("pdf", "pdf_sha256"), ("prompt", "author-system-research.md"),
    ("new_prompt", "author-system-review.md"), ("removed_prompt", "author-system-research.md"),
    ("contract", "CLAUDE.md"), ("model", "models.phases.author.model"),
    ("temperature", "models.phases.author.temperature"),
    ("reasoning", "models.phases.author.reasoning_effort"),
    ("provider", "models.phases.author.provider"), ("endpoint", "models.endpoints.openai"),
    ("flags", "command.extra_args"), ("per_input_flags", "command.per_input_args"),
    ("version", "framework.version"),
])
def test_drift_refuses_resume_before_workers_or_checkpoint_changes(
    run, monkeypatch, capsys, change, field,
):
    root, config, inputs, calls, start = run
    directory = start()
    checkpoint = (directory / "checkpoint.json").read_bytes()
    original_plan = (directory / "plan.json").read_bytes()
    prompt = root / "prompts" / "author-system-research.md"
    if change == "pdf":
        Path(inputs[1]).write_bytes(b"%PDF-1.4 replacement")
    elif change == "prompt":
        prompt.write_text("Changed instructions\n")
    elif change == "new_prompt":
        (prompt.parent / "author-system-review.md").write_text("New prompt\n")
    elif change == "removed_prompt":
        prompt.unlink()
    elif change == "contract":
        (root / "CLAUDE.md").write_text("Changed contract\n")
    elif change == "model":
        config.write_text(config.read_text().replace("original", "different"))
    elif change in {"temperature", "reasoning"}:
        setting = "temperature: 0.8" if change == "temperature" else "reasoning_effort: high"
        config.write_text(config.read_text() + f"    {setting}\n")
    elif change == "provider":
        monkeypatch.setenv("RW_LLM_PROVIDER", "chat-relay")
    elif change == "endpoint":
        monkeypatch.setenv("RW_LLM_BASE_URL", "https://different.test/v1")
    elif change in {"flags", "per_input_flags"}:
        plan = json.loads(original_plan)
        if change == "flags":
            plan["extra_args"].append("--no-semantic")
        else:
            plan["per_input_args"][inputs[1]] = ["--year", "2025"]
        (directory / "plan.json").write_text(json.dumps(plan))
    elif change == "version":
        monkeypatch.setattr(identity, "__version__", "future-version")
    plan_before_resume = (directory / "plan.json").read_bytes()
    capsys.readouterr()
    assert batch.resume_batch(directory, no_retry=False) == 1
    assert calls == []
    assert (directory / "checkpoint.json").read_bytes() == checkpoint
    assert (directory / "plan.json").read_bytes() == plan_before_resume
    err = capsys.readouterr().err
    assert "resume refused" in err and field in err
    assert "start a new batch" in err


@pytest.mark.parametrize("entry_point", [agent, ingest])
def test_both_cli_entry_points_enforce_identity(run, entry_point):
    _, _, inputs, calls, start = run
    directory = start()
    Path(inputs[1]).write_bytes(b"changed")
    argv = ["--resume", str(directory)]
    assert entry_point.main((["ingest"] if entry_point is agent else []) + argv) == 1
    assert calls == []


def test_unchanged_resume_skips_completed_pdf_after_promotion(run, monkeypatch):
    root, _, inputs, calls, start = run
    directory = start()
    Path(inputs[0]).rename(root / "canonical.pdf")

    def succeed(pdf, *args):
        calls.append(pdf)
        return {"input": pdf, "status": "completed", "returncode": 0}

    monkeypatch.setattr(batch, "_worker", succeed)
    assert batch.resume_batch(directory, no_retry=False, workers_override=2) == 0
    assert calls == [inputs[1]]
    assert set(batch._read_checkpoint(directory)["completed"]) == set(inputs)


def test_missing_pending_pdf_keeps_existing_recovery_behavior(run, capsys):
    _, _, inputs, calls, start = run
    directory = start()
    Path(inputs[1]).unlink()
    assert batch.resume_batch(directory, no_retry=False) == 1
    assert calls == []
    assert inputs[1] in batch._read_checkpoint(directory)["unresumable"]
    assert "resume refused" not in capsys.readouterr().err


@pytest.mark.parametrize("flag, per_input", [("--author-prompt-file", False),
                                           ("--supplementary", True)])
@pytest.mark.parametrize("remove", [False, True])
def test_override_and_supplementary_bytes_are_frozen(run, flag, per_input, remove, capsys):
    root, _, inputs, calls, start = run
    asset = root / "support.txt"
    asset.write_text("Original bytes")
    args = [flag, asset.name]
    directory = start(extra_args=[] if per_input else args,
                      per_input_args={inputs[1]: args} if per_input else None)
    plan = json.loads((directory / "plan.json").read_text())
    stored_args = plan["per_input_args"][inputs[1]] if per_input else plan["extra_args"]
    assert stored_args == [flag, str(asset)]
    if remove:
        asset.unlink()
    else:
        asset.write_text("Changed bytes")
    assert batch.resume_batch(directory, no_retry=False) == 1
    assert calls == []
    assert str(asset) in capsys.readouterr().err


def test_unversioned_framework_edit_is_detected(run, tmp_path, monkeypatch, capsys):
    # A package fixture proves the digest reads source bytes, not just a release tag.
    package = tmp_path / "package"
    (package / "tasks").mkdir(parents=True)
    source = package / "implementation.py"
    source.write_text("original = True\n")
    monkeypatch.setattr(identity, "__file__", str(package / "tasks" / "_batch_identity.py"))
    _, _, _, calls, start = run
    directory = start()
    source.write_text("original = False\n")
    assert batch.resume_batch(directory, no_retry=False) == 1
    assert calls == []
    assert "framework.source_sha256" in capsys.readouterr().err


def test_unreadable_new_configuration_stops_before_checkpoint_changes(run, capsys):
    _, config, _, calls, start = run
    directory = start()
    checkpoint = (directory / "checkpoint.json").read_bytes()
    config.write_text("roles: [broken yaml\n")
    assert batch.resume_batch(directory, no_retry=False) == 2
    assert calls == []
    assert (directory / "checkpoint.json").read_bytes() == checkpoint
    assert "could not verify batch identity" in capsys.readouterr().err


def test_restoring_configuration_allows_resume_without_restarting_completed_inputs(run):
    _, config, inputs, calls, start = run
    directory = start()
    original = config.read_bytes()
    config.write_text(config.read_text().replace("original", "different"))
    assert batch.resume_batch(directory, no_retry=False) == 1
    assert calls == []
    config.write_bytes(original)
    assert batch.resume_batch(directory, no_retry=False) == 1  # worker's recorded failure
    assert calls == [inputs[1]]


def test_creation_failure_leaves_no_partial_batch(run, capsys):
    root, config, inputs, calls, _ = run
    config.write_text("roles: [broken yaml\n")
    assert batch.new_batch(inputs, ["agent", "ingest"], [], workers=1) == 2
    assert calls == [] and not (root / ".ingest").exists()
    assert "could not record batch identity" in capsys.readouterr().err


@pytest.mark.parametrize("damage", ["missing", "unsupported", "incomplete"])
def test_modern_checkpoint_cannot_fall_back_to_unverified_resume(run, damage, capsys):
    _, _, _, calls, start = run
    directory = start()
    plan = json.loads((directory / "plan.json").read_text())
    if damage == "missing":
        del plan["identity"]
    elif damage == "unsupported":
        plan["identity"]["version"] = 999
    else:
        plan["identity"]["inputs"] = None
    (directory / "plan.json").write_text(json.dumps(plan))
    assert batch.resume_batch(directory, no_retry=False) == 1
    assert calls == []
    assert "resume refused" in capsys.readouterr().err


def test_legacy_checkpoint_warns_without_inventing_provenance(run, capsys):
    _, _, inputs, calls, start = run
    directory = start()
    plan = json.loads((directory / "plan.json").read_text())
    del plan["version"], plan["identity"]
    (directory / "plan.json").write_text(json.dumps(plan))
    plan_before = (directory / "plan.json").read_bytes()
    assert batch.resume_batch(directory, no_retry=False) == 1  # worker still fails
    assert calls == [inputs[1]]
    assert (directory / "plan.json").read_bytes() == plan_before
    assert "legacy checkpoint has no recorded identity" in capsys.readouterr().err
