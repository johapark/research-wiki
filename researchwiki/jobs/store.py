"""Durable single-library jobs, idempotent submit, and generation-fenced writes."""

from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import uuid

from filelock import FileLock

from ..fsatomic import exclusive_lock
from ..operations import IngestResult, OperationError
from ..runtime import repository_key, repository_runtime_dir
from .admission import normalize_request, validate_inputs
from .contracts import JobRecord, validate_job_id
from .files import JobConflict, JobStoreUnavailable, digest, outside_library, read_json, write_private_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, library_root, *, store_root=None):
        self.library_root = Path(library_root).expanduser().resolve()
        self.library_id = repository_key(self.library_root)
        base = store_root if store_root is not None else os.environ.get("RESEARCHWIKI_JOBS_ROOT")
        self.base = Path(base).expanduser().resolve() if base is not None else None
        self.root = repository_runtime_dir(self.library_root, base=self.base) / "jobs"
        outside_library(self.root, self.library_root)

    def path(self, job_id):
        path = self.root / validate_job_id(job_id)
        if path.is_symlink():
            raise JobStoreUnavailable("job directory cannot be a symlink")
        return path

    def owner_lock(self, job_id):
        # Kept for the entire worker run. The OS releases it after process loss;
        # it is never unlinked, so PID reuse cannot create another live owner.
        path = self.path(job_id)
        if not path.is_dir():
            raise JobConflict("unknown job")
        return FileLock(str(path / "owner.lock"), timeout=0)

    def get_job(self, job_id):
        record = JobRecord.from_dict(read_json(self.path(job_id) / "manifest.json"))
        if record.job_id != job_id or record.library_id != self.library_id or (
            Path(record.request.library_root).resolve() != self.library_root
        ) or record.request_sha256 != digest(record.request.to_dict()):
            raise JobStoreUnavailable("job record library/request identity mismatch")
        return record

    def submit_ingest(self, request, *, parent_job_id=None):
        request = normalize_request(request)
        if Path(request.library_root) != self.library_root:
            raise JobConflict("request belongs to a different library")
        fingerprint = digest(request.to_dict())
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with exclusive_lock(self.root / "submission"):
            for path in sorted(self.root.iterdir()):
                if len(path.name) != 32 or not path.is_dir() or path.name.startswith('.'):
                    continue
                existing = self.get_job(path.name)
                if existing.request.request_id == request.request_id:
                    if existing.request_sha256 != fingerprint:
                        raise JobConflict("request_id already names a different request")
                    return existing
            validate_inputs(request)
            now = timestamp()
            job_id = uuid.uuid4().hex
            record = JobRecord(job_id=job_id, library_id=self.library_id, request=request,
                               request_sha256=fingerprint, execution="queued", created_at=now,
                               updated_at=now, parent_job_id=parent_job_id)
            staged = self.root / f".pending-{job_id}"
            staged.mkdir(mode=0o700)
            try:
                write_private_json(staged / "manifest.json", record.to_dict())
                os.replace(staged, self.path(job_id))
            finally:
                if staged.exists():
                    shutil.rmtree(staged)
            return record

    def _change(self, job_id, fn):
        path = self.path(job_id) / "manifest.json"
        with exclusive_lock(path):
            record = fn(self.get_job(job_id))
            write_private_json(path, record.to_dict())
            return record

    def claim(self, job_id, owner):
        def apply(record):
            if record.execution != "queued":
                raise JobConflict("only a queued job can start")
            return replace(record, execution="running", owner=owner, worker_pid=os.getpid(),
                           generation=record.generation + 1, updated_at=timestamp())
        return self._change(job_id, apply)

    @staticmethod
    def _owned(record, owner, generation):
        if record.execution != "running" or record.owner != owner or record.generation != generation:
            raise JobConflict("worker no longer owns this job generation")

    def set_identity(self, job_id, owner, generation, identity):
        def apply(record):
            self._owned(record, owner, generation)
            parent = self.get_job(record.parent_job_id) if record.parent_job_id else None
            old = parent.identity if parent else None
            return replace(record, identity=identity, updated_at=timestamp(),
                           source_changed=None if old is None else (old.pdf_sha256 != identity.pdf_sha256 or
                                                                  old.auxiliary_sha256 != identity.auxiliary_sha256),
                           configuration_changed=None if old is None else old.configuration_sha256 != identity.configuration_sha256)
        return self._change(job_id, apply)

    def finish(self, job_id, owner, generation, result):
        def apply(record):
            self._owned(record, owner, generation)
            return replace(record, execution=result.execution, result=result, owner=None,
                           worker_pid=None, updated_at=timestamp())
        return self._change(job_id, apply)

    def cancel_queued_job(self, job_id):
        def apply(record):
            if record.execution == "cancelled":
                return record
            if record.execution != "queued":
                raise JobConflict("only queued jobs can be cancelled; running jobs require interruption")
            result = IngestResult(request_id=record.request.request_id, job_id=job_id, execution="cancelled",
                                  outcome=None, publication="not_published", projections="not_applicable",
                                  promotion=record.request.options.promotion)
            return replace(record, execution="cancelled", result=result, updated_at=timestamp())
        return self._change(job_id, apply)

    def retry_job(self, job_id):
        record = self.get_job(job_id)
        result = record.result
        if record.execution != "finished" or result is None or result.publication != "not_published":
            raise JobConflict("retry requires a finished, known-unpublished job; uncertain work needs reconciliation")
        if result.error and result.error.retry in ("not_retryable", "reconcile_first"):
            raise JobConflict("this failure requires correction/reconciliation before retry")
        request = replace(record.request, request_id=uuid.uuid4().hex)
        return self.submit_ingest(request, parent_job_id=job_id)

    def mark_interrupted(self, job_id):
        # Caller must hold the owner lock. Explicit reconciliation inspects
        # durable publication evidence; events and filenames cannot settle it.
        def apply(record):
            if record.execution != "running":
                return record
            result = IngestResult(
                request_id=record.request.request_id, job_id=job_id, execution="interrupted",
                outcome=None, publication="unknown", projections="unknown", post_work="unknown",
                promotion=record.request.options.promotion,
                error=OperationError(code="worker_lost", category="environment", message="Worker ownership lost",
                                     retry="reconcile_first"),
            )
            return replace(record, execution="interrupted", result=result, owner=None,
                           worker_pid=None, updated_at=timestamp())
        return self._change(job_id, apply)

    def ledger_binding(self, record):
        return dict(version=1, library_root=str(self.library_root), job_id=record.job_id,
                    request_id=record.request.request_id, request_sha256=record.request_sha256,
                    generation=record.generation, path=str(self.path(record.job_id) / "publication.json"))

    def finish_reconciliation(self, expected, result):
        """Caller holds owner lease; compare state/generation before settlement."""
        def apply(record):
            pending = record.execution == "interrupted" or (
                record.execution == "finished" and record.result.publication == "published"
                and record.result.projections == "maintenance_pending"
            )
            if record != expected or not pending:
                raise JobConflict("job changed during reconciliation")
            return replace(record, execution=result.execution, result=result, updated_at=timestamp())
        return self._change(expected.job_id, apply)
