"""Private process entry point for exactly one durably queued ingestion job."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import os
import sys
import traceback
import uuid

from filelock import Timeout

from ..operations import EventRecorder, IngestResult, OperationError, error_from_exception, ingest
from ..ingest_ledger import begin, complete, observe_publication
from .admission import validate_inputs
from .environment import configure_worker, execution_identity
from .events import EventWriter
from .files import JobConflict, JobStoreUnavailable
from .store import JobStore
from ..records.settings import freeze_selected


class Tee:
    """Retain diagnostics while keeping foreground relay handoffs visible."""
    def __init__(self, log, terminal):
        self.log, self.terminal = log, terminal

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        self.terminal.write(text)
        self.terminal.flush()
        return len(text)

    def flush(self):
        self.log.flush()
        self.terminal.flush()


def run_worker(store, job_id, *, operation=ingest):
    try:
        with store.owner_lock(job_id):
            owner = uuid.uuid4().hex
            record = store.claim(job_id, owner)
            return _execute_owned(store, record, owner, operation)
    except (Timeout, JobConflict):
        return 1  # another owner or queued cancellation won; never run models


def _execute_owned(store, record, owner, operation):
    job_id = record.job_id
    generation = record.generation
    request = record.request
    events = None
    recorder = None
    started = False
    fd = os.open(store.path(job_id) / "diagnostic.log", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(Tee(log, sys.stderr)):
            try:
                binding = store.ledger_binding(record)
                begin(binding)
                validate_inputs(request)
                effective = configure_worker(request)
                with freeze_selected():
                    identity = execution_identity(effective)
                    record = store.set_identity(job_id, owner, generation, identity)
                    events = EventWriter(store.path(job_id) / "events.jsonl")
                    recorder = EventRecorder(request.request_id, events, job_id=job_id)
                    started = True
                    with observe_publication(binding):
                        result = operation(effective, event_sink=recorder)
                warnings = result.warnings
                if record.source_changed:
                    warnings += ("Source content changed since the parent job",)
                if record.configuration_changed:
                    warnings += ("Provider configuration changed since the parent job",)
                result = replace(result, job_id=job_id, warnings=warnings)
            except (Exception, KeyboardInterrupt) as exc:
                traceback.print_exc()
                interrupted = isinstance(exc, KeyboardInterrupt)
                error = JobStoreUnavailable("Worker inputs or storage unavailable") if isinstance(exc, OSError) else exc
                result = IngestResult(
                    request_id=request.request_id, job_id=job_id,
                    attempt_id=recorder.attempt_id if recorder else None,
                    execution="interrupted" if interrupted else "finished",
                    outcome=None if interrupted else "failed", publication="unknown" if started else "not_published",
                    projections="unknown" if started else "not_applicable",
                    promotion=request.options.promotion,
                    error=OperationError(code="interrupted", category="environment", message="Worker interrupted",
                                         retry="reconcile_first" if started else "explicit_retry") if interrupted else error_from_exception(error),
                )
            finally:
                if events is not None:
                    try:
                        events.close()
                    except OSError:
                        print("job progress close failed; inspect the diagnostic log", file=sys.stderr)
            # One atomic manifest owns the result and terminal state. A write
            # failure leaves a running record; loss detection blocks replay.
            complete(store.ledger_binding(record), result)
            store.finish(job_id, owner, generation, result)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", required=True)
    parser.add_argument("--store-root")
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)
    return run_worker(JobStore(args.library_root, store_root=args.store_root), args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
