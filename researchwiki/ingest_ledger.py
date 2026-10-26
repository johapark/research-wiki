"""Durable publication evidence, independent of progress delivery and derived DBs.

Workers opt in with a private ledger. Promotion binds its journal to that ledger
before touching canonical files. Commit/rollback resolution is copied there
before journal deletion, including when a different CLI process drains a journal.
The journal remains the recovery authority until this handoff succeeds. This is
process-crash recovery, not a distributed transaction or a power-loss guarantee.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import re

from .fsatomic import exclusive_lock, file_sha256, write_private_json


_current = ContextVar("researchwiki_ingest_ledger", default=None)
_STATES = {"started", "prepared", "committed", "rolled_back"}


def validate_binding(binding):
    if not isinstance(binding, dict) or set(binding) != {
        "version", "library_root", "job_id", "request_id", "request_sha256", "generation", "path",
    } or type(binding["version"]) is not int or binding["version"] != 1:
        raise ValueError("unsupported ingest ledger binding")
    if not isinstance(binding["job_id"], str) or not re.fullmatch("[a-f0-9]{32}", binding["job_id"]):
        raise ValueError("invalid ledger job ID")
    if type(binding["generation"]) is not int or binding["generation"] < 1:
        raise ValueError("invalid ledger generation")
    if not all(isinstance(binding[k], str) and binding[k] for k in (
        "library_root", "request_id", "request_sha256", "path",
    )):
        raise ValueError("invalid ledger identity")
    root, path = Path(binding["library_root"]), Path(binding["path"])
    if not root.is_absolute() or not path.is_absolute() or path.name != "publication.json":
        raise ValueError("invalid ledger path")
    if path.parent.name != binding["job_id"] or path.is_symlink():
        raise ValueError("ledger path does not identify its job")
    for content in (root, *(root / name for name in ("wiki", "papers", "inbox"))):
        if path.resolve().is_relative_to(content.resolve()):
            raise ValueError("ledger must be outside canonical content")


def read_ledger(binding):
    validate_binding(binding)
    data = json.loads(Path(binding["path"]).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {
        "schema_version", "binding", "state", "journal_path", "artifacts", "paths", "result",
    } or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported ingest ledger")
    if data["binding"] != binding or data["state"] not in _STATES:
        raise ValueError("ingest ledger identity/state mismatch")
    _validate_payload(data)
    return data


def _validate_payload(data):
    journal = data["journal_path"]
    if journal is not None and (not isinstance(journal, str) or not Path(journal).is_absolute()):
        raise ValueError("invalid ingest journal reference")
    if not isinstance(data["paths"], list) or any(
        not isinstance(path, str) or not Path(path).is_absolute() for path in data["paths"]
    ):
        raise ValueError("invalid ingest ledger paths")
    if data["result"] is not None and not isinstance(data["result"], dict):
        raise ValueError("invalid ingest operation checkpoint")
    if data["state"] == "committed":
        if not isinstance(data["artifacts"], dict):
            raise ValueError("committed ingest lacks artifacts")
    elif data["artifacts"] is not None:
        raise ValueError("uncommitted ingest contains publication artifacts")
    if data["state"] == "started" and (journal is not None or data["paths"]):
        raise ValueError("unstarted mutation has journal data")
    if data["state"] == "prepared" and (journal is None or data["result"] is not None):
        raise ValueError("unresolved mutation has invalid checkpoint data")


def begin(binding):
    validate_binding(binding)
    path = Path(binding["path"])
    if path.exists():
        raise FileExistsError("ingest ledger already exists")
    write_private_json(path, dict(schema_version=1, binding=binding, state="started",
                                 journal_path=None, artifacts=None, paths=[], result=None))


@contextmanager
def observe_publication(binding):
    token = _current.set(binding)
    try:
        yield
    finally:
        _current.reset(token)


def journal_binding():
    binding = _current.get()
    return {"ingest_ledger": binding} if binding else {}


def prepare(snap):
    binding = snap.details.get("ingest_ledger")
    if binding is None:
        return
    if not snap.enabled:
        raise ValueError("durable jobs require mutation journalling")
    with exclusive_lock(binding["path"]):
        data = read_ledger(binding)
        if data["state"] != "started":
            raise ValueError("one publication transaction per ingestion job")
        data.update(state="prepared", journal_path=str(snap.journal_path.resolve()),
                    paths=[str(e.target) for e in snap.entries])
        write_private_json(Path(binding["path"]), data)


def set_artifacts(snap, result, stem):
    """Captured before mark_committed, so its atomic journal write carries proof."""
    if "ingest_ledger" not in snap.details:
        return
    from .operations.lifecycle import current_facts

    facts = current_facts()
    snap.details["ingest_artifacts"] = {
        "paper_stem": stem, "attempt_id": snap.details["attempt_id"],
        "page_path": str(result.wiki_path.resolve()), "pdf_path": str(result.pdf_path.resolve()),
        "page_sha256": file_sha256(result.wiki_path), "pdf_sha256": file_sha256(result.pdf_path),
        "gate_reasons": list(facts.gate_reasons) if facts else [],
        "checks": [check.to_dict() for check in facts.checks] if facts else [],
    }


def require_backups(document):
    """A missing backup cannot certify that this job's mutation was rolled back."""
    if "ingest_ledger" in document.get("details", {}):
        for entry in document["entries"]:
            if entry["backup"] and not Path(entry["backup"]).exists():
                raise ValueError("ingest rollback backup is missing; receipt retained")


def archive(document, *, committed):
    binding = document.get("details", {}).get("ingest_ledger")
    if binding is None:
        return
    with exclusive_lock(binding["path"]):
        _archive_locked(document, binding, committed=committed)


def _archive_locked(document, binding, *, committed):
    data = read_ledger(binding)
    if Path(binding["library_root"]).resolve() != Path.cwd().resolve():
        raise ValueError("journal ledger belongs to a different library")
    if any(document["details"].get(key) != binding[key] for key in ("request_id", "job_id")):
        raise ValueError("journal ledger operation identity mismatch")
    outcome = "committed" if committed else "rolled_back"
    if data["state"] in {"committed", "rolled_back"} and data["state"] != outcome:
        raise ValueError("conflicting durable publication outcome")
    artifacts = document["details"].get("ingest_artifacts") if committed else None
    if committed and not isinstance(artifacts, dict):
        raise ValueError("committed ingestion journal lacks artifact evidence")
    data.update(state=outcome, artifacts=artifacts,
                paths=[entry["target"] for entry in document["entries"]])
    write_private_json(Path(binding["path"]), data)


def complete(binding, result):
    """An operation checkpoint is evidence; the job manifest still owns status."""
    with exclusive_lock(binding["path"]):
        data = read_ledger(binding)
        if data["state"] == "prepared":
            # A pending journal must resolve first; never erase uncertainty.
            return
        if data["state"] == "committed" and result.publication != "published":
            raise ValueError("operation result contradicts committed publication")
        data["result"] = result.to_dict()
        write_private_json(Path(binding["path"]), data)
