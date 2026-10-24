"""Live retention lifecycle with disposable libraries and zero paid calls."""

from dataclasses import replace
from pathlib import Path
import json
import importlib
import shutil
from types import SimpleNamespace
import uuid

import pytest

from researchwiki.agents import model_config
from researchwiki.agents.context import Context
from researchwiki.agents.runner import run_ingest
from researchwiki.fsatomic import file_sha256
from researchwiki.ingest_ledger import prepare, set_artifacts
from researchwiki.mutation import mutation, pending_journals, recover_pending
from researchwiki.operations.ingest import execute_ingest
from researchwiki.operations.lifecycle import publication_details
from researchwiki.records import RecordStore, RecordStoreUnavailable
from researchwiki.records.export import export_attempt
from researchwiki.records.lifecycle import capture_operation, current_owner
from researchwiki.records.publication import publication_receipts
from researchwiki.records.settings import validate, validate_root
from tests.test_job_workers import make_library, clean_environment


@pytest.fixture
def library(tmp_path, monkeypatch):
    from researchwiki.env_profiles import ACTIVE_ENV_FILE_VAR

    root, request = make_library(tmp_path, "retention")
    config = root / "config/models.jobs.yaml"
    value = json.loads(config.read_text())
    value["ingest"] = {"retention": {"mode": "required", "profile": "extraction-v1", "root": ".researchwiki/records"}}
    config.write_text(json.dumps(value))
    monkeypatch.chdir(root)
    monkeypatch.setenv("RW_MODELS_CONFIG", str(config))
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(root / "state.db"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    for name in (ACTIVE_ENV_FILE_VAR, "RW_LLM_PROVIDER", "RW_LLM_BASE_URL", "ANTHROPIC_BASE_URL", "RW_MUTATION_JOURNAL"):
        monkeypatch.delenv(name, raising=False)
    model_config._load_routing_snapshot.cache_clear()
    return root, request


def archive(root):
    return RecordStore(root / ".researchwiki/records")


def test_operation_publishes_with_one_complete_attempt_and_portable_receipt(library):
    root, request = library
    execution = execute_ingest(request)
    assert execution.result.publication == "published", execution.result.to_dict()
    manifests = archive(root).list_attempts()
    assert len(manifests) == 1 and manifests[0].spec.attempt_id == execution.result.attempt_id
    assert manifests[0].capture_state == "complete"
    assert manifests[0].terminal.outcome == "published"
    receipts = publication_receipts(archive(root), manifests[0].spec.attempt_id)
    assert [r["state"] for r in receipts] == ["prepared", "staged", "committed"]
    assert receipts[-1]["artifacts"]["pdf_sha256"] == file_sha256(Path(execution.result.pdf_path))
    assert not pending_journals() and current_owner() is None
    result_ids = next(i.artifact_ids for i in manifests[0].inventory if i.slot == "attempt.result")
    terminal = next(a for a in manifests[0].artifacts if a.artifact_id in result_ids)
    value = json.loads(archive(root).read_blob(terminal.payload))
    assert value["scope"] == "operation" and value["post_work"] == execution.result.post_work


def test_direct_runner_owns_and_finishes_capture(library):
    root, request = library
    from researchwiki.operations.runtime import runner_options
    ctx = run_ingest(request.source_pdf, **runner_options(request.options))
    manifest = archive(root).read_attempt(ctx.attempt_id)
    assert manifest.capture_state == "complete" and manifest.terminal.outcome == "published"
    assert ctx.processing_pdf_path != ctx.pdf_path and current_owner() is None


def test_default_off_creates_no_archive(library):
    root, request = library
    config = root / "config/models.jobs.yaml"
    value = json.loads(config.read_text())
    del value["ingest"]
    config.write_text(json.dumps(value))
    model_config._load_routing_snapshot.cache_clear()
    assert execute_ingest(request).result.publication == "published"
    assert not (root / ".researchwiki/records").exists()


def test_duplicate_is_a_second_attempt_with_unexecuted_target_inventory(library):
    root, request = library
    first = execute_ingest(request).result
    assert first.publication == "published"
    inbox = root / "inbox/duplicate.pdf"
    shutil.copyfile(first.pdf_path, inbox)
    second = execute_ingest(replace(request, source_pdf=str(inbox))).result
    assert second.outcome == "already_present", second.to_dict()
    assert second.attempt_id != first.attempt_id and inbox.exists()
    manifest = archive(root).read_attempt(second.attempt_id)
    assert manifest.capture_state == "complete"
    assert all(i.status == "not_run" for i in manifest.inventory if i.slot.startswith("target."))


def test_rejected_early_metadata_records_unexecuted_phases(library, monkeypatch):
    root, request = library
    monkeypatch.setattr("researchwiki.agents.runner._phase_reconcile", lambda *args: ({}, None))
    result = execute_ingest(request).result
    assert result.outcome == "failed" and result.publication == "not_published"
    manifest = archive(root).read_attempt(result.attempt_id)
    assert manifest.capture_state == "complete"
    assert all(i.status == "not_run" for i in manifest.inventory if i.slot.startswith("target."))


def test_source_failure_prevents_reconcile_and_has_no_provider_retry(library, monkeypatch):
    _, request = library
    monkeypatch.setattr("researchwiki.records.extraction.ExtractionRecorder.snapshot_source",
                        lambda *args: (_ for _ in ()).throw(RecordStoreUnavailable("source archive unavailable")))
    monkeypatch.setattr("researchwiki.agents.runner._phase_reconcile", lambda *args: pytest.fail("must stop before reconcile"))
    result = execute_ingest(request).result
    assert result.publication == "not_published" and result.error.code == "capture_unavailable"
    assert result.error.retry == "reconcile_first"


def test_post_work_is_inside_owner_and_latched_failure_preserves_publication(library, monkeypatch):
    root, request = library
    def post(ctx, options):
        assert current_owner().ctx is ctx
        assert archive(root).read_attempt(ctx.attempt_id).terminal is None
        current_owner().records.failure = RecordStoreUnavailable("post-work capture failed")
        return SimpleNamespace(state="completed", warnings=[])
    monkeypatch.setattr(importlib.import_module("researchwiki.operations.ingest"), "run_post_work", post)
    result = execute_ingest(request).result
    assert result.publication == "published" and result.outcome == "published"
    assert result.error.code == "capture_unavailable" and result.error.retry == "reconcile_first"
    assert Path(result.page_path).exists()
    assert archive(root).read_attempt(result.attempt_id).terminal is None


def test_terminal_capture_failure_does_not_turn_publication_into_retry(library, monkeypatch):
    root, request = library
    original = RecordStore.finish_attempt
    monkeypatch.setattr(RecordStore, "finish_attempt", lambda *args: (_ for _ in ()).throw(RecordStoreUnavailable("terminal unavailable")))
    result = execute_ingest(request).result
    assert result.publication == "published" and result.error.retry == "reconcile_first"
    monkeypatch.setattr(RecordStore, "finish_attempt", original)
    assert publication_receipts(archive(root), result.attempt_id)[-1]["state"] == "committed"


def test_required_capture_refuses_disabled_journalling_before_runner(library, monkeypatch):
    root, request = library
    monkeypatch.setenv("RW_MUTATION_JOURNAL", "0")
    result = execute_ingest(request, runner=lambda *a, **k: pytest.fail("must not start")).result
    assert result.error.code == "capture_unavailable" and result.publication == "not_published"
    assert not (root / ".researchwiki/records").exists()


@pytest.mark.parametrize("value", [None, [], {"mode": "best_effort"}, {"mode": "required"},
                                  {"profile": "future"}, {"root": 123}, {"unexpected": True}])
def test_settings_reject_unknown_or_ambiguous_policy(value):
    with pytest.raises(ValueError):
        validate(value)


def test_archive_root_rejects_canonical_and_synced_content(library, tmp_path):
    root, _ = library
    for path in (root, root.parent, root / "inbox/archive", root / "wiki/records", root / ".mutation/records"):
        with pytest.raises(RecordStoreUnavailable):
            validate_root(path, root)
    synced = tmp_path / "synced"
    shutil.move(root / "wiki", synced)
    (root / "wiki").symlink_to(synced, target_is_directory=True)
    with pytest.raises(RecordStoreUnavailable):
        validate_root(synced / "records", root)


def test_export_survives_source_and_database_deletion(library, tmp_path):
    root, request = library
    result = execute_ingest(request).result
    output = tmp_path / "exported"
    exported = export_attempt(archive(root), result.attempt_id, output)
    before = RecordStore(exported).read_attempt(result.attempt_id)
    shutil.rmtree(root / ".researchwiki")
    Path(request.source_pdf).unlink()
    (root / "state.db").unlink(missing_ok=True)
    assert RecordStore(exported).verify(result.attempt_id).manifest == before
    assert publication_receipts(RecordStore(exported), result.attempt_id)[-1]["state"] == "committed"


def test_profile_and_job_identity_include_retention_and_generation(library, tmp_path):
    root, request = library
    from researchwiki.jobs import JobSupervisor
    supervisor = JobSupervisor(root, store_root=tmp_path / "runtime", environment=clean_environment(tmp_path))
    queued = supervisor.submit_ingest(request)
    handle = supervisor.start_job(queued.job_id)
    record = supervisor.wait(handle, timeout=40)
    assert record.result.publication == "published", record.result.to_dict()
    manifest = archive(root).read_attempt(record.result.attempt_id)
    assert manifest.spec.job_id == record.job_id and manifest.spec.generation == record.generation
    assert manifest.spec.request_id == request.request_id


@pytest.fixture
def prepared(library):
    root, request = library
    with capture_operation(None) as owner:
        ctx = Context(attempt_id=str(uuid.uuid4()), pdf_path=Path(request.source_pdf), pdf_filename="source.pdf")
        owner.bind(ctx)
        target = root / "papers/synthetic-2026-fixture.pdf"
        page = root / "wiki/other/synthetic-2026-fixture.md"
        result = SimpleNamespace(pdf_path=target, wiki_path=page)
        yield root, owner, ctx, result


def publish_fixture(owner, ctx, result, *, before_commit=None):
    paths = (ctx.pdf_path, result.pdf_path, result.wiki_path)
    with mutation(paths, operation="promote", details=publication_details(ctx.attempt_id, "synthetic-2026-fixture", "other")) as snap:
        prepare(snap)
        from researchwiki.agents.promote import _move_pdf
        _move_pdf(ctx.pdf_path, "synthetic-2026-fixture")
        result.wiki_path.write_text("synthetic publication")
        set_artifacts(snap, result, "synthetic-2026-fixture")
        if before_commit:
            before_commit(snap)
        snap.mark_committed()
        snap.discard()


def test_source_replacement_during_move_rolls_back_canonical_files(prepared, monkeypatch):
    root, owner, ctx, result = prepared
    from researchwiki.agents import promote
    original = promote.shutil.copy2
    def substituted(source, target):
        original(source, target)
        if Path(target) == result.pdf_path:
            Path(target).write_bytes(b"%PDF-replaced-during-copy")
    monkeypatch.setattr(promote.shutil, "copy2", substituted)
    with pytest.raises(RecordStoreUnavailable, match="canonical PDF differs"):
        publish_fixture(owner, ctx, result)
    assert not result.pdf_path.exists() and not result.wiki_path.exists()
    assert ctx.pdf_path.exists()


def test_displaced_canonical_pdf_is_retained_before_upgrade(prepared, monkeypatch):
    _, owner, ctx, result = prepared
    result.pdf_path.write_bytes(b"%PDF-displaced-canonical")
    from researchwiki import wiki
    monkeypatch.setattr(wiki, "find_stem_collision", lambda stem: result.wiki_path)
    monkeypatch.setattr(wiki, "classify_pdf_collision", lambda *args: "journal-upgrade")
    monkeypatch.setattr(wiki, "_doi_from_existing_page", lambda *args: "10.old/source")
    publish_fixture(owner, ctx, result)
    manifest = owner.records.store.read_attempt(ctx.attempt_id)
    old = next(a for a in manifest.artifacts if a.kind == "source_snapshot" and any(s.role == "displaced_canonical" for s in a.sources))
    assert owner.records.store.read_blob(old.payload) == b"%PDF-displaced-canonical"


def test_commit_handoff_failure_keeps_journal_and_recovers_without_models(prepared, monkeypatch):
    _, owner, ctx, result = prepared
    from researchwiki.records import publication
    original = publication._retain
    def fail_committed(document, state, artifacts=None):
        if state == "committed":
            raise RecordStoreUnavailable("receipt handoff failed")
        return original(document, state, artifacts)
    with monkeypatch.context() as patch:
        patch.setattr(publication, "_retain", fail_committed)
        with pytest.raises(RecordStoreUnavailable):
            publish_fixture(owner, ctx, result)
    assert pending_journals() and result.pdf_path.exists() and result.wiki_path.exists()
    assert owner.records.store.read_attempt(ctx.attempt_id).terminal is None
    monkeypatch.setattr("researchwiki.projections.reconcile", lambda *args: None)
    monkeypatch.setattr("researchwiki.agents.llm.call", lambda **kwargs: pytest.fail("recovery must not call a model"))
    recover_pending()
    assert not pending_journals()
    assert publication_receipts(owner.records.store, ctx.attempt_id)[-1]["state"] == "committed"


def test_displaced_snapshot_failure_prevents_overwrite(prepared, monkeypatch):
    _, owner, ctx, result = prepared
    old = b"%PDF-preserve-old-canonical"
    result.pdf_path.write_bytes(old)
    from researchwiki import wiki
    monkeypatch.setattr(wiki, "find_stem_collision", lambda stem: result.wiki_path)
    monkeypatch.setattr(wiki, "classify_pdf_collision", lambda *args: "journal-upgrade")
    monkeypatch.setattr(wiki, "_doi_from_existing_page", lambda *args: "10.old/source")
    original = owner.records.store.put_file
    def put(path, **kwargs):
        if Path(path) == result.pdf_path:
            raise RecordStoreUnavailable("cannot snapshot displaced PDF")
        return original(path, **kwargs)
    monkeypatch.setattr(owner.records.store, "put_file", put)
    with pytest.raises(RecordStoreUnavailable):
        publish_fixture(owner, ctx, result)
    assert result.pdf_path.read_bytes() == old and not result.wiki_path.exists()
    assert ctx.pdf_path.exists()


def test_original_source_change_before_move_prevents_publication(prepared):
    _, owner, ctx, result = prepared
    ctx.pdf_path.write_bytes(b"%PDF-new-user-version")
    with pytest.raises(RecordStoreUnavailable, match="original PDF changed"):
        publish_fixture(owner, ctx, result)
    assert not result.pdf_path.exists() and not result.wiki_path.exists()
    assert ctx.pdf_path.read_bytes() == b"%PDF-new-user-version"


def test_terminal_checkpoint_failure_is_explicitly_repairable_without_reingest(library, monkeypatch):
    root, request = library
    original = RecordStore._write_checkpoint
    def write(store, manifest):
        if manifest.terminal is not None:
            raise OSError("terminal checkpoint unavailable")
        return original(store, manifest)
    monkeypatch.setattr(RecordStore, "_write_checkpoint", write)
    result = execute_ingest(request).result
    assert result.publication == "published" and result.error.code == "capture_unavailable"
    monkeypatch.setattr(RecordStore, "_write_checkpoint", original)
    store = archive(root)
    assert store.verify(result.attempt_id).checkpoint == "stale"
    store.repair_checkpoint(result.attempt_id)
    assert store.read_attempt(result.attempt_id).terminal.outcome == "published"


def test_latched_capture_failure_blocks_later_optional_model_calls(prepared):
    _, owner, _, _ = prepared
    from researchwiki.agents import llm
    owner.records.failure = RecordStoreUnavailable("prior capture failure")
    with pytest.raises(RecordStoreUnavailable, match="prior capture"):
        llm.call(phase="author", prompt="do not dispatch", use_stub=True)


def test_worker_retention_selection_is_frozen_across_identity_and_execution(library, monkeypatch):
    root, _ = library
    from researchwiki.records.settings import freeze_selected, selected
    with freeze_selected() as frozen:
        monkeypatch.setattr(model_config, "_ingest_settings", lambda: {"retention": {
            "mode": "required", "root": str(root / "another-archive")}})
        assert selected() is frozen
        assert selected().root == root / ".researchwiki/records"
    assert selected().root == root / "another-archive"


def test_records_cli_read_only_verification_explicit_repair_and_export(library, tmp_path, capsys):
    root, request = library
    from researchwiki.tasks.records import main
    result = execute_ingest(request).result
    assert main(["show", result.attempt_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["manifest"]["capture_state"] == "complete"
    checkpoint = archive(root).root / "attempts" / result.attempt_id / "manifest.json"
    checkpoint.unlink()
    assert main(["verify", result.attempt_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["checkpoint"] == "missing"
    assert not checkpoint.exists()
    assert main(["verify", result.attempt_id, "--repair-checkpoint", "--json"]) == 0
    capsys.readouterr()
    output = tmp_path / "cli-export"
    assert main(["export", result.attempt_id, "--output", str(output), "--json"]) == 0
    assert RecordStore(output).read_attempt(result.attempt_id).terminal.publication == "published"
    capsys.readouterr()
    value = json.loads(checkpoint.read_text())
    value["schema_version"] = 99
    checkpoint.write_text(json.dumps(value))
    before = checkpoint.read_bytes()
    assert main(["verify", result.attempt_id, "--repair-checkpoint", "--json"]) == 2
    assert checkpoint.read_bytes() == before


def test_records_cli_does_not_initialize_an_unknown_archive(tmp_path, capsys):
    from researchwiki.tasks.records import main
    root = tmp_path / "missing"
    assert main(["--root", str(root), "list", "--json"]) == 2
    assert not root.exists()


def test_invalid_journal_binding_is_preserved_for_review(prepared, monkeypatch):
    _, owner, ctx, result = prepared
    def corrupt(snap):
        snap.details["records_binding"]["archive_id"] = uuid.uuid4().hex
    with pytest.raises(RecordStoreUnavailable):
        publish_fixture(owner, ctx, result, before_commit=corrupt)
    before = pending_journals()[0]
    journal = Path(before["journal_path"])
    original = journal.read_bytes()
    recover_pending()
    assert journal.read_bytes() == original
    assert result.pdf_path.exists() and result.wiki_path.exists()
    assert publication_receipts(owner.records.store, ctx.attempt_id)[-1]["state"] == "staged"


@pytest.mark.parametrize("valid_report", [True, False])
def test_batch_resume_does_not_replay_capture_failure_or_unknown_result(library, tmp_path, monkeypatch, valid_report):
    _, request = library
    from researchwiki.tasks import _ingest_batch
    from researchwiki.operations import IngestResult, OperationError
    def popen(command, **kwargs):
        if valid_report:
            path = Path(command[command.index("--result-file") + 1])
            value = IngestResult(request_id="captured-batch", execution="finished", outcome="failed",
                                 publication="not_published", projections="not_applicable",
                                 error=OperationError(code="capture_unavailable", category="environment",
                                                      message="Archive unavailable", retry="reconcile_first"))
            path.write_text(json.dumps(value.to_dict()))
        return SimpleNamespace(stderr=iter(()), wait=lambda: 2)
    monkeypatch.setattr(_ingest_batch.subprocess, "Popen", popen)
    record = _ingest_batch._worker(request.source_pdf, tmp_path, ["agent", "ingest"], [])
    assert record["returncode"] == 2 and record["retry"] == "reconcile_first"
    assert not _ingest_batch._should_retry(record)
    assert _ingest_batch._should_retry({"returncode": 2})


def test_records_recovery_hands_off_committed_journal_without_provider_work(prepared, monkeypatch, capsys):
    _, owner, ctx, result = prepared
    from researchwiki.records import publication
    from researchwiki.tasks.records import main
    original = publication._retain
    def interrupted(document, state, artifacts=None):
        if state == "committed":
            raise RecordStoreUnavailable("capture handoff interrupted")
        return original(document, state, artifacts)
    with monkeypatch.context() as patch:
        patch.setattr(publication, "_retain", interrupted)
        with pytest.raises(RecordStoreUnavailable):
            publish_fixture(owner, ctx, result)
    monkeypatch.setattr("researchwiki.projections.reconcile", lambda *args: None)
    monkeypatch.setattr("researchwiki.agents.llm.call", lambda **kwargs: pytest.fail("must not author again"))
    assert main(["recover", "--json"]) == 0
    assert not pending_journals()
    assert publication_receipts(owner.records.store, ctx.attempt_id)[-1]["state"] == "committed"
