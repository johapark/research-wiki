"""Explicit isolated reconciliation. No provider preflight, model calls, or replay."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import traceback

from filelock import Timeout

from ..ingest_ledger import read_ledger
from ..operations import CheckResult, IngestResult, OperationError
from .environment import configure_worker
from .files import JobConflict
from .repair import repair_projections, validate_artifacts
from .store import JobStore


def _unresolved(record, code="reconciliation_unavailable"):
    return replace(record.result, error=OperationError(
        code=code, category="environment", message="Publication recovery needs review; no model work was replayed",
        retry="reconcile_first",
    ))


def _checkpoint(record, ledger):
    raw = ledger["result"]
    if raw is None:
        return None
    result = IngestResult.from_dict(raw)
    if result.request_id != record.request.request_id or result.job_id != record.job_id:
        raise ValueError("operation checkpoint belongs to a different job")
    if result.publication == "published" and ledger["state"] != "committed" and not (
        ledger["state"] == "started" and result.outcome == "already_present"
    ):
        raise ValueError("operation checkpoint lacks publication evidence")
    if ledger["state"] == "committed" and (
        result.publication != "published" or result.attempt_id != ledger["artifacts"]["attempt_id"]
    ):
        raise ValueError("operation checkpoint contradicts publication receipt")
    return result


def _note_failure(store, record, exc):
    """Detailed failures stay private; wire errors contain stable, safe guidance."""
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(store.path(record.job_id) / "reconciliation.log", flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except OSError:
        import sys
        print("reconciliation diagnostic storage unavailable", file=sys.stderr)


def _publication(record, ledger):
    artifacts = ledger["artifacts"]
    validate_artifacts(Path(record.request.library_root), artifacts)
    checkpoint = _checkpoint(record, ledger)
    result = checkpoint or IngestResult(
        request_id=record.request.request_id, job_id=record.job_id, execution="finished",
        outcome="published", publication="published", post_work="unknown",
        promotion=record.request.options.promotion,
    )
    return replace(result, execution="finished", outcome="published", publication="published",
                   projections="maintenance_pending", error=None,
                   attempt_id=artifacts["attempt_id"], paper_stem=artifacts["paper_stem"],
                   page_path=artifacts["page_path"], pdf_path=artifacts["pdf_path"],
                   gate_reasons=tuple(artifacts["gate_reasons"]),
                   checks=tuple(CheckResult.from_dict(check) for check in artifacts["checks"]),
                   pending_relays=(),
                   warnings=result.warnings + ("Publication recovered after worker loss; no model work was replayed",))


def _journal_files():
    from ..paths import mutation_dir

    root = mutation_dir()
    try:
        root.stat()
    except FileNotFoundError:
        if root.is_symlink():
            raise ValueError("mutation directory symlink is unavailable")
        return []
    # iterdir raises on an unreadable directory; glob may silently skip it.
    return [path for path in root.iterdir() if path.suffix == ".json"]


def _recover_journals():
    from ..mutation import recover_pending

    # pending_journals() intentionally skips corrupt JSON for legacy callers.
    # Such a skip cannot establish absence of publication for job reconciliation.
    for path in _journal_files():
        if path.is_symlink():
            raise ValueError("mutation journal cannot be a symlink")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("status") not in {"active", "committed"}:
            raise ValueError("invalid mutation journal")
    recover_pending()
    if _journal_files():
        raise ValueError("unresolved mutation journals remain")


def _journal_publication(record, ledger, binding):
    """A commit marker remains authoritative if the archival write itself fails."""
    from ..mutation import JOURNAL_VERSION, pending_journals

    found = None
    for document in pending_journals():
        details = document.get("details", {})
        if (document.get("version") == JOURNAL_VERSION and document.get("status") == "committed"
                and isinstance(details, dict) and details.get("ingest_ledger") == binding
                and not details.get("ingest_rolled_back")
                and all(details.get(key) == binding[key] for key in ("request_id", "job_id"))):
            evidence = dict(ledger, state="committed", artifacts=details.get("ingest_artifacts"),
                            paths=[entry["target"] for entry in document["entries"]])
            candidate = _publication(record, evidence)
            if found is not None:
                raise ValueError("multiple publication transactions for one job")
            found = candidate
    return found


def _reconcile_locked(store, record, *, repair):
    binding = store.ledger_binding(record)
    try:
        ledger = read_ledger(binding)
        # Validate checkpoint and commit evidence before trusting either.
        _checkpoint(record, ledger)
        published = (_publication(record, ledger) if ledger["state"] == "committed"
                     else _journal_publication(record, ledger, binding))
    except Exception as exc:
        _note_failure(store, record, exc)
        return _unresolved(record, "publication_evidence_unavailable")
    try:
        _recover_journals()
        ledger = read_ledger(binding)  # another CLI may have resolved this journal
        checkpoint = _checkpoint(record, ledger)
        if ledger["state"] == "committed":
            published = _publication(record, ledger)
            repair(record, ledger)
            return replace(published, projections="current")
        if checkpoint is not None and checkpoint.publication != "unknown":
            return replace(checkpoint, execution="finished", outcome=checkpoint.outcome or "failed", pending_relays=())
        if ledger["state"] not in {"started", "rolled_back"}:
            return _unresolved(record, "publication_evidence_incomplete")
        if checkpoint is not None:
            error = checkpoint.error
            if error is not None and error.retry == "reconcile_first":
                error = replace(error, retry="explicit_retry")
            return replace(checkpoint, execution="finished", outcome=checkpoint.outcome or "failed",
                           publication="not_published", projections="not_applicable", error=error,
                           pending_relays=())
        return IngestResult(
            request_id=record.request.request_id, job_id=record.job_id, execution="finished",
            outcome="failed", publication="not_published", projections="not_applicable",
            post_work="not_requested", promotion=record.request.options.promotion,
            error=OperationError(code="worker_lost_before_publication", category="environment",
                                 message="Interrupted before publication; explicit fresh retry is available",
                                 retry="explicit_retry"),
        )
    except Exception as exc:
        _note_failure(store, record, exc)
        # A committed journal can establish publication even if projection repair
        # or archival failed. It remains on disk until evidence is durable.
        try:
            latest = read_ledger(binding)
            if latest["state"] == "committed":
                published = _publication(record, latest)
        except Exception:
            pass
        if published is not None:
            return replace(published, error=OperationError(
                code="publication_maintenance_pending", category="environment",
                message="Published content needs local maintenance or review; no authoring was replayed",
                retry="reconcile_first",
            ))
        return _unresolved(record)


def reconcile_job(store, job_id, *, repair=repair_projections):
    """Run inside a selected library process; never switch the caller's runtime."""
    from ..mutation import corpus_lock

    if Path.cwd().resolve() != store.library_root:
        raise JobConflict("reconciliation requires an isolated selected-library process")
    with store.owner_lock(job_id):
        record = store.mark_interrupted(job_id)
        pending = record.execution == "interrupted" or (
            record.execution == "finished" and record.result.publication == "published"
            and record.result.projections == "maintenance_pending"
        )
        if not pending:
            return record
        with corpus_lock():
            try:
                configure_worker(record.request)
                result = _reconcile_locked(store, record, repair=repair)
            except Exception as exc:
                _note_failure(store, record, exc)
                result = _unresolved(record)
            return store.finish_reconciliation(record, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", required=True)
    parser.add_argument("--store-root")
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)
    # Repair can use already-installed local embedding weights, never download.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    try:
        reconcile_job(JobStore(args.library_root, store_root=args.store_root), args.job_id)
    except (Timeout, JobConflict):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
