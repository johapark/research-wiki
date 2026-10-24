"""Validated journal-to-archive handoff for foreground and job ingestion."""

from pathlib import Path
from functools import wraps
import uuid

from ..fsatomic import file_sha256
from .contracts import Artifact, SourceRef, identifier
from .errors import RecordConflict, RecordCorrupt, RecordStoreUnavailable
from .jsonio import decode_json
from .lifecycle import current_owner
from .settings import validate_root
from .store import RecordStore, now

# Artifact pipeline stage metadata, independent of model routing phases.
_PUBLICATION_STAGE = "promotion"


def _latched(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (RecordStoreUnavailable, ValueError, OSError) as exc:
            error = exc if isinstance(exc, RecordStoreUnavailable) else RecordStoreUnavailable("publication capture unavailable")
            owner = current_owner()
            if owner is not None and owner.records is not None:
                owner.records.failure = error
            raise error from (None if error is exc else exc)
    return run


@_latched
def journal_binding():
    owner = current_owner()
    if owner is None or owner.records is None:
        return {}
    owner.check()
    records = owner.records
    return {"records_binding": {"schema_version": 1, "root": str(records.store.root),
                                "library_root": str(Path.cwd().resolve()),
                                "archive_id": records.store.metadata().archive_id,
                                "attempt_id": records.spec.attempt_id}}


def _bound(document):
    binding = document.get("details", {}).get("records_binding")
    if binding is None:
        return None
    if not isinstance(binding, dict) or set(binding) != {"schema_version", "root", "library_root", "archive_id", "attempt_id"}:
        raise RecordCorrupt("invalid journal archive binding")
    if type(binding["schema_version"]) is not int or binding["schema_version"] != 1:
        raise RecordCorrupt("unsupported journal archive binding")
    for key in ("archive_id", "attempt_id"):
        identifier(binding[key])
    if not all(isinstance(binding[k], str) and Path(binding[k]).is_absolute() for k in ("root", "library_root")):
        raise RecordCorrupt("invalid archive binding paths")
    library = Path(binding["library_root"]).resolve()
    if document.get("operation") != "promote":
        raise RecordCorrupt("archive binding requires an ingestion promotion journal")
    if library != Path.cwd().resolve() or document["details"].get("attempt_id") != binding["attempt_id"]:
        raise RecordCorrupt("journal archive binding belongs to a different library or attempt")
    root = validate_root(binding["root"], library)
    store = RecordStore(root)
    if store.metadata().archive_id != binding["archive_id"]:
        raise RecordCorrupt("journal archive identity changed")
    manifest = store.read_attempt(binding["attempt_id"])
    if manifest.spec.request_id != document["details"].get("request_id") or manifest.spec.job_id != document["details"].get("job_id"):
        raise RecordCorrupt("journal operation identity differs from the archived attempt")
    if manifest.spec.job_id and manifest.spec.generation != document["details"].get("ingest_ledger", {}).get("generation"):
        raise RecordCorrupt("journal job generation differs from the archived attempt")
    if not any(r.slot == "publication.receipt" and r.kind == "terminal_result" for r in manifest.spec.requirements):
        raise RecordCorrupt("journal attempt has no publication receipt inventory")
    return store, manifest


def _retain(document, state, artifacts=None):
    bound = _bound(document)
    if bound is None:
        return
    store, manifest = bound
    if artifacts is not None:
        originals = next((i.artifact_ids for i in manifest.inventory if i.slot == "source.original"), ())
        expected = {a.payload.sha256 for a in manifest.artifacts if a.artifact_id in originals}
        if artifacts.get("pdf_sha256") not in expected:
            raise RecordCorrupt("publication receipt differs from its retained source")
    value = {"schema_version": 1, "boundary": "publication_journal", "state": state,
             "journal": Path(document.get("journal_path") or "journal").name,
             "artifacts": artifacts}
    # Snapshot._journal_document has no journal_path. Its backup directory and
    # journal share the stable allocated ID, including during recovery.
    value["journal"] = Path(document["backup_dir"]).name
    blob = store.put_json(value)
    item_id = uuid.uuid5(uuid.NAMESPACE_URL,
                        f"{manifest.archive_id}/{manifest.spec.attempt_id}/{value['journal']}/{state}").hex
    existing = next((a for a in manifest.artifacts if a.artifact_id == item_id), None)
    item = Artifact(artifact_id=item_id, attempt_id=manifest.spec.attempt_id,
                    kind="terminal_result", payload=blob, producer=manifest.spec.producer,
                    created_at=existing.created_at if existing else now(), origin="observed", phase=_PUBLICATION_STAGE)
    store.record_artifact("publication.receipt", item)
    owner = current_owner()
    if owner is not None and owner.records is not None and owner.records.spec.attempt_id == manifest.spec.attempt_id:
        if state in ("committed", "rolled_back"):
            owner.receipt = state


@_latched
def prepare(snap):
    if "records_binding" not in snap.details:
        return
    if not snap.enabled:
        raise RecordStoreUnavailable("required capture needs a durable publication journal")
    owner = current_owner()
    if owner is None or owner.records is None:
        raise RecordStoreUnavailable("capture publication requires its active attempt owner")
    owner.check()
    owner.prepared = True
    _retain(snap._journal_document("active"), "prepared")


@_latched
def before_move(source, target):
    owner = current_owner()
    if owner is None or owner.records is None:
        return
    records = owner.records
    owner.check()
    if Path(source).resolve() != records.source.original_path or file_sha256(source) != records.source.source.document.sha256:
        raise RecordStoreUnavailable("original PDF changed after its processing snapshot")
    if Path(target).exists() and Path(source).resolve() != Path(target).resolve():
        blob = records._write(records.store.put_file, target)
        ref = SourceRef(document=blob, role="displaced_canonical")
        records.artifact("source.displaced", "source_snapshot", None, payload=blob,
                         sources=(ref,), parents=(records.source.artifact_id,))


@_latched
def set_artifacts(snap, result, stem):
    if "records_binding" not in snap.details:
        return
    owner = current_owner()
    if owner is None or owner.records is None:
        raise RecordStoreUnavailable("capture publication owner is missing")
    owner.check()
    records = owner.records
    records._write(records.store.read_attempt, records.spec.attempt_id)
    pdf_hash = file_sha256(result.pdf_path)
    if pdf_hash != records.source.source.document.sha256:
        raise RecordStoreUnavailable("canonical PDF differs from the retained processing source")
    from ..paths import papers_dir, wiki_dir
    snap.details["records_artifacts"] = {
        "paper_stem": stem, "pdf_sha256": pdf_hash, "page_sha256": file_sha256(result.wiki_path),
        "pdf_path": str(Path("papers") / result.pdf_path.resolve().relative_to(papers_dir().resolve())),
        "page_path": str(Path("wiki") / result.wiki_path.resolve().relative_to(wiki_dir().resolve())),
    }
    _retain(snap._journal_document("active"), "staged", snap.details["records_artifacts"])


@_latched
def archive(document, *, committed):
    if "records_binding" not in document.get("details", {}):
        return
    artifacts = document["details"].get("records_artifacts") if committed else None
    if committed:
        if document.get("status") != "committed":
            raise RecordCorrupt("uncommitted journal cannot prove publication")
        if not isinstance(artifacts, dict) or set(artifacts) != {"paper_stem", "pdf_sha256", "page_sha256", "pdf_path", "page_path"}:
            raise RecordCorrupt("committed capture journal lacks artifact evidence")
        from .contracts import sha256
        for key in ("pdf_sha256", "page_sha256"):
            sha256(artifacts[key])
        for key in ("pdf_path", "page_path"):
            path = Path(artifacts[key])
            if path.is_absolute() or ".." in path.parts:
                raise RecordCorrupt("unsafe publication artifact reference")
    try:
        _retain(document, "committed" if committed else "rolled_back", artifacts)
    except RecordConflict as exc:
        raise RecordStoreUnavailable("publication archive handoff is unresolved; preserve its journal") from exc


def publication_receipts(store, attempt_id):
    manifest = store.read_attempt(attempt_id)
    ids = next((i.artifact_ids for i in manifest.inventory if i.slot == "publication.receipt"), ())
    return tuple(decode_json(store.read_blob(a.payload)) for a in manifest.artifacts if a.artifact_id in ids)
