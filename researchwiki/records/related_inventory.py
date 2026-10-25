"""Read-only inventory of filesystem evidence adjacent to historical ingests."""

from __future__ import annotations

import json
import re
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


_JOB_ID = re.compile(r"^[0-9a-f]{32}$")


def inventory_related_state(library_root: Path, historical_attempt_ids=()) -> dict:
    """Summarize local job, journal, mailbox, sandbox and cache availability.

    Only metadata and counts are returned. Drafts, event bodies, relay payloads,
    and journal paths are never included in the report.
    """
    root = Path(library_root).expanduser().resolve(strict=False)
    attempt_ids = set(historical_attempt_ids)
    return {
        "library_root": str(root),
        "scan_started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "consistency": "best_effort_per_file_reads; no cross-store snapshot",
        "jobs": _jobs(root, attempt_ids),
        "mutation_journals": _journals(root, attempt_ids),
        "relay_mailbox": _relay_mailbox(root),
        "sandbox_outputs": _flat_files(root / ".agent-output", markdown_only=True),
        "grading_cache": _child_directories(root / ".grade-cache"),
        "ingest_runs": _ingest_runs(root / ".ingest"),
    }


def _jobs(root: Path, attempt_ids: set[str]) -> dict:
    from ..jobs.events import read_events
    from ..jobs.files import JobStoreUnavailable
    from ..jobs.store import JobStore
    from ..ingest_ledger import read_ledger
    from .errors import RecordStoreUnavailable

    try:
        store = JobStore(root)
        entries = _directory_entries(store.root)
    except (OSError, ValueError) as exc:
        return {"status": "unavailable", "error_type": type(exc).__name__, "jobs": []}
    if entries is None:
        return {"status": _directory_failure_status(store.root),
                "root": str(store.root), "jobs": []}

    jobs = []
    invalid = 0
    ignored = 0
    event_totals = Counter()
    for path in entries:
        if path.name.startswith(".") or not _JOB_ID.fullmatch(path.name):
            ignored += 1
            continue
        if path.is_symlink() or not path.is_dir() or (path / "manifest.json").is_symlink():
            invalid += 1
            continue
        try:
            record = store.get_job(path.name)
        except (JobStoreUnavailable, OSError, ValueError):
            invalid += 1
            continue

        result = record.result
        attempt_id = _safe_identifier(result.attempt_id) if result else None
        event_file = path / "events.jsonl"
        event_state = "missing"
        events = ()
        if event_file.is_symlink():
            event_state = "symlink_skipped"
        elif event_file.is_file():
            try:
                events = read_events(store, record.job_id)
                event_state = "readable"
            except (JobStoreUnavailable, OSError, ValueError):
                event_state = "invalid_or_unavailable"

        relay_counts = Counter(event.kind for event in events if event.kind.startswith("relay_"))
        event_totals.update(relay_counts)
        open_relays = _open_relay_waits(events)

        ledger_state = "missing"
        ledger_artifacts = None
        ledger_paths = None
        ledger_file = path / "publication.json"
        if ledger_file.is_symlink():
            ledger_state = "symlink_skipped"
        elif ledger_file.is_file():
            try:
                ledger = read_ledger(store.ledger_binding(record))
                ledger_state = ledger["state"]
                artifacts = ledger.get("artifacts")
                ledger_artifacts = len(artifacts) if isinstance(artifacts, dict) else 0
                paths = ledger.get("paths")
                ledger_paths = len(paths) if isinstance(paths, list) else 0
                if attempt_id is None and isinstance(artifacts, dict):
                    attempt_id = _safe_identifier(artifacts.get("attempt_id"))
            except (OSError, ValueError, TypeError, RecordStoreUnavailable):
                ledger_state = "invalid_or_unavailable"

        jobs.append({
            "job_id": record.job_id,
            "execution": record.execution,
            "outcome": result.outcome if result else None,
            "publication": result.publication if result else None,
            "attempt_id": attempt_id,
            "legacy_history_match": attempt_id in attempt_ids if attempt_id else None,
            "events": {"status": event_state, "complete_events": len(events),
                       "relay_events": dict(sorted(relay_counts.items())),
                       "open_relay_waits": open_relays},
            "publication_ledger": {"status": ledger_state,
                                   "artifact_fields": ledger_artifacts,
                                   "path_references": ledger_paths},
        })
    jobs.sort(key=lambda item: (item["job_id"]))
    executions = Counter(job["execution"] for job in jobs)
    publications = Counter(job["publication"] for job in jobs if job["publication"] is not None)
    ledgers = Counter(job["publication_ledger"]["status"] for job in jobs)
    return {
        "status": "available",
        "root": str(store.root),
        "job_count": len(jobs),
        "invalid_job_directories": invalid,
        "ignored_entries": ignored,
        "executions": dict(sorted(executions.items())),
        "publications": dict(sorted(publications.items())),
        "publication_ledgers": dict(sorted(ledgers.items())),
        "relay_event_totals": dict(sorted(event_totals.items())),
        "jobs": jobs,
    }


def _open_relay_waits(events) -> int:
    pending = set()
    for event in events:
        if event.kind in ("model_call_finished", "model_call_failed"):
            pending = {key for key in pending if key[0] != event.call_id}
        if event.relay is None:
            continue
        key = (event.call_id, event.relay.op_id)
        if event.kind == "relay_waiting":
            pending.add(key)
        else:
            pending.discard(key)
    return len(pending)


def _journals(root: Path, attempt_ids: set[str]) -> dict:
    directory = root / ".mutation"
    entries = _directory_entries(directory)
    if entries is None:
        return {"status": _directory_failure_status(directory),
                "journal_count": 0, "journals": []}

    journals = []
    invalid = 0
    symlinks = 0
    for path in entries:
        if path.suffix != ".json":
            continue
        if path.is_symlink():
            symlinks += 1
            continue
        if not path.is_file():
            invalid += 1
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            invalid += 1
            continue
        if not isinstance(document, dict):
            invalid += 1
            continue
        details = document.get("details")
        details = details if isinstance(details, dict) else {}
        records_binding = details.get("records_binding")
        ledger_binding = details.get("ingest_ledger")
        attempt_id = details.get("attempt_id")
        if not isinstance(attempt_id, str) and isinstance(records_binding, dict):
            attempt_id = records_binding.get("attempt_id")
        if not isinstance(attempt_id, str) and isinstance(ledger_binding, dict):
            attempt_id = ledger_binding.get("attempt_id")
        safe_attempt_id = _safe_identifier(attempt_id)
        raw_rows = document.get("entries")
        rows = raw_rows if isinstance(raw_rows, list) else []
        backup_dir = _journal_backup_directory(document.get("backup_dir"), directory)
        backup_available = bool(backup_dir and backup_dir.is_dir() and not backup_dir.is_symlink())
        backup_refs = [row.get("backup") for row in rows
                       if isinstance(row, dict) and isinstance(row.get("backup"), str)
                       and row.get("backup")]
        backup_present = 0
        backup_missing = 0
        for raw_backup in backup_refs:
            try:
                candidate = Path(raw_backup).resolve(strict=False)
            except (OSError, RuntimeError, ValueError):
                backup_missing += 1
                continue
            if backup_dir is not None and candidate.is_relative_to(backup_dir) and candidate.exists():
                backup_present += 1
            else:
                backup_missing += 1
        journals.append({
            "operation": _safe_operation(document.get("operation")),
            "status": document.get("status") if document.get("status") in ("active", "committed") else "unknown",
            "version": document.get("version") if type(document.get("version")) is int else None,
            "entry_count": len(rows),
            "backup_reference_count": len(backup_refs),
            "backup_directory_available": backup_available,
            "backup_references_present": backup_present,
            "backup_references_missing_or_outside": backup_missing,
            "records_bound": isinstance(records_binding, dict),
            "job_ledger_bound": isinstance(ledger_binding, dict),
            "attempt_id": safe_attempt_id,
            "legacy_history_match": safe_attempt_id in attempt_ids if safe_attempt_id else None,
        })
    statuses = Counter(item["status"] for item in journals)
    return {
        "status": "available",
        "journal_count": len(journals),
        "invalid_files": invalid,
        "symlinks_skipped": symlinks,
        "statuses": dict(sorted(statuses.items())),
        "journals": sorted(journals, key=lambda item: (
            item["status"], item["operation"] or "", item["attempt_id"] or ""
        )),
    }


def _journal_backup_directory(raw, journal_root: Path) -> Path | None:
    if not isinstance(raw, str) or not raw or "\0" in raw:
        return None
    try:
        root = journal_root.resolve(strict=False)
        candidate = Path(raw).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate if candidate != root and root in candidate.parents else None


def _relay_mailbox(root: Path) -> dict:
    base = root / ".llm-relay"
    if base.is_symlink():
        return {"status": "symlink_skipped"}
    if not base.exists():
        return {"status": "not_found", "pending_prompts": 0, "completed_responses": 0}
    if not base.is_dir():
        return {"status": "not_a_directory"}

    pending = _mailbox_files(base / "pending", ".prompt.json")
    completed = _mailbox_files(base / "completed", ".response.json")
    prompt_ids = set(pending["ids"])
    response_ids = set(completed["ids"])
    return {
        "status": ("available" if pending["status"] in ("available", "not_found")
                   and completed["status"] in ("available", "not_found") else "partial"),
        "pending_directory_status": pending["status"],
        "completed_directory_status": completed["status"],
        "pending_prompts": len(prompt_ids),
        "completed_responses": len(response_ids),
        "matched_pairs": len(prompt_ids & response_ids),
        "prompt_only": len(prompt_ids - response_ids),
        "response_only": len(response_ids - prompt_ids),
        "symlinks_skipped": pending["symlinks"] + completed["symlinks"],
        "payloads_read": False,
        "consumed_mailboxes_recoverable_from_this_directory": False,
    }


def _mailbox_files(directory: Path, suffix: str) -> dict:
    entries = _directory_entries(directory)
    if entries is None:
        return {"ids": [], "symlinks": 0,
                "status": _directory_failure_status(directory)}
    ids = []
    symlinks = 0
    for path in entries:
        if path.is_symlink():
            symlinks += 1
        elif path.is_file() and path.name.endswith(suffix):
            ids.append(path.name[:-len(suffix)])
    return {"ids": ids, "symlinks": symlinks, "status": "available"}


def _flat_files(path: Path, *, markdown_only: bool = False) -> dict:
    entries = _directory_entries(path)
    if entries is None:
        return {"status": _directory_failure_status(path), "files": 0,
                "symlinks_skipped": 0}
    files = [item for item in entries if not item.is_symlink() and item.is_file()]
    markdown = sum(item.suffix.lower() == ".md" for item in files)
    return {
        "status": "available",
        "files": len(files),
        "markdown_files": markdown if markdown_only else None,
        "symlinks_skipped": sum(item.is_symlink() for item in entries),
    }


def _child_directories(path: Path) -> dict:
    entries = _directory_entries(path)
    if entries is None:
        return {"status": _directory_failure_status(path),
                "paper_cache_directories": 0, "symlinks_skipped": 0}
    return {
        "status": "available",
        "paper_cache_directories": sum(not item.is_symlink() and item.is_dir() for item in entries),
        "symlinks_skipped": sum(item.is_symlink() for item in entries),
    }


def _ingest_runs(path: Path) -> dict:
    entries = _directory_entries(path)
    if entries is None:
        return {"status": _directory_failure_status(path),
                "run_directories": 0, "batch_checkpoints": 0}
    directories = [item for item in entries if not item.is_symlink() and item.is_dir()]
    batches = [item for item in directories if item.name.startswith("batch-")]
    checkpoints = sum(
        not (item / "checkpoint.json").is_symlink()
        and (item / "checkpoint.json").is_file()
        for item in batches
    )
    return {
        "status": "available",
        "run_directories": len(directories),
        "batch_directories": len(batches),
        "batch_checkpoints": checkpoints,
        "symlinks_skipped": sum(item.is_symlink() for item in entries),
    }


def _directory_entries(path: Path) -> list[Path] | None:
    if _unavailable_directory_status(path) != "available":
        return None
    try:
        return sorted(path.iterdir())
    except OSError:
        return None


def _unavailable_directory_status(path: Path) -> str:
    if path.is_symlink():
        return "symlink_skipped"
    try:
        info = path.stat()
    except FileNotFoundError:
        return "not_found"
    except OSError:
        return "unavailable"
    if not stat.S_ISDIR(info.st_mode):
        return "not_a_directory"
    return "available"


def _directory_failure_status(path: Path) -> str:
    status = _unavailable_directory_status(path)
    return "unavailable" if status == "available" else status


def _safe_identifier(value) -> str | None:
    if isinstance(value, str) and len(value) <= 128 and re.fullmatch(r"[A-Za-z0-9._-]+", value):
        return value
    return None


def _safe_operation(value) -> str | None:
    if isinstance(value, str) and len(value) <= 80 and re.fullmatch(r"[A-Za-z0-9._-]+", value):
        return value
    return None
