"""Opt-in archive primitives. Construction and reads never initialize a store.

The event history is authoritative. A checkpoint must equal a verified prefix of
that history before it may be repaired; corrupt/future checkpoints are preserved.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import uuid

from ..fsatomic import exclusive_lock
from ..operations._wire import ContractError
from .contracts import (
    Artifact, ArtifactRecorded, AttemptEvent, AttemptFinished, AttemptManifest,
    AttemptSpec, BlobRef, InventoryResolved, StoreMetadata, identifier,
)
from .errors import CheckpointStale, RecordConflict, RecordCorrupt, RecordStoreUnavailable
from .jsonio import canonical_bytes, digest
from .state import advance, project
from .storage import Storage, make_directory, storage_errors


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Verification:
    manifest: AttemptManifest
    checkpoint: str  # current | stale | missing
    event_count: int
    blob_count: int


class RecordStore:
    def __init__(self, root):
        self._storage = Storage(root)
        self.root = self._storage.root
        self._archive_id = None

    @classmethod
    def create(cls, root):
        store = cls(root)
        with storage_errors(), exclusive_lock(store._storage.path("store.json")):
            path = store._storage.path("store.json")
            if not path.exists():
                if store.root.exists() and any(store.root.iterdir()):
                    raise RecordConflict("refusing to initialize a nonempty unrecognized archive")
                make_directory(store.root)
                metadata = StoreMetadata(archive_id=uuid.uuid4().hex, created_at=now())
                store._storage.publish(path, canonical_bytes(metadata.to_dict()))
            store.metadata()
        return store

    def metadata(self):
        with storage_errors():
            try:
                value = self._storage.read_json(self._storage.path("store.json"))
            except FileNotFoundError as exc:
                raise RecordStoreUnavailable("archive is not initialized; create it explicitly") from exc
            try:
                metadata = StoreMetadata.from_dict(value)
            except ContractError as exc:
                raise RecordCorrupt("invalid or unsupported archive metadata; preserve the original store") from exc
            if self._archive_id is not None and metadata.archive_id != self._archive_id:
                raise RecordCorrupt("archive identity changed while open")
            self._archive_id = metadata.archive_id
            return metadata

    def put_blob(self, data, *, media_type="application/octet-stream", encoding=None):
        with storage_errors():
            self.metadata()
            return self._storage.put_bytes(data, media_type=media_type, encoding=encoding)

    def put_json(self, value):
        return self.put_blob(canonical_bytes(value), media_type="application/json", encoding="utf-8")

    def put_file(self, source, *, media_type="application/pdf"):
        with storage_errors():
            self.metadata()
            with Path(source).open("rb") as stream:
                return self._storage.put_stream(stream, media_type=media_type, encoding=None)

    def read_blob(self, ref):
        ref = BlobRef.from_dict(ref.to_dict())
        with storage_errors():
            self.metadata()
            path = self._storage.path(*ref.relative_path.split("/"))
            if path.exists() and not path.is_file():
                raise RecordCorrupt("archive blob must be a regular file")
            try:
                data = path.read_bytes()
            except FileNotFoundError as exc:
                raise RecordCorrupt("referenced archive blob is missing") from exc
            if len(data) != ref.size_bytes or digest(data) != ref.sha256:
                raise RecordCorrupt("archive blob digest/size mismatch")
            if ref.encoding == 'utf-8':
                try:
                    data.decode('utf-8')
                except UnicodeError as exc:
                    raise RecordCorrupt("archive text blob violates its declared encoding") from exc
            return data

    def _attempt_path(self, attempt_id):
        return self._storage.path("attempts", identifier(attempt_id))

    def _events(self, attempt_id):
        base = self._attempt_path(attempt_id)
        if not base.is_dir():
            raise RecordConflict("unknown archive attempt")
        for child in base.iterdir():
            self._storage.path(*child.relative_to(self.root).parts)
            if child.name not in ("events", "manifest.json") and not child.name.startswith(".write-"):
                raise RecordCorrupt("unexpected file in archive attempt")
        directory = self._storage.path("attempts", attempt_id, "events")
        if not directory.is_dir():
            raise RecordCorrupt("attempt event history is missing")
        events = []
        for path in directory.iterdir():
            self._storage.path(*path.relative_to(self.root).parts)
            if path.name.startswith(".write-") and path.is_file():
                continue  # An interrupted unpublished write is never a retained event.
            try:
                event = AttemptEvent.from_dict(self._storage.read_json(path))
            except (ContractError, IsADirectoryError) as exc:
                raise RecordCorrupt("invalid or unsupported archive event") from exc
            if path.name != f"{event.sequence:08d}-{event.event_id}.json":
                raise RecordCorrupt("event filename differs from its identity")
            events.append(event)
        return tuple(sorted(events, key=lambda event: event.sequence))

    def _inspect(self, attempt_id):
        metadata = self.metadata()
        events = self._events(attempt_id)
        manifest = project(events)
        if manifest.archive_id != metadata.archive_id or manifest.spec.attempt_id != attempt_id:
            raise RecordCorrupt("attempt belongs to a different archive/identity")
        seen = {}
        verified = set()
        for artifact in manifest.artifacts:
            for ref in artifact.blobs():
                if ref.sha256 in seen and seen[ref.sha256] != ref.size_bytes:
                    raise RecordCorrupt("conflicting sizes for the same blob")
                identity = (ref.sha256, ref.encoding)
                if identity not in verified:
                    self._storage.verify_blob(ref)
                    verified.add(identity)
                seen[ref.sha256] = ref.size_bytes
        path = self._storage.path("attempts", attempt_id, "manifest.json")
        if not path.exists():
            return Verification(manifest, "missing", len(events), len(seen)), events
        try:
            checkpoint = AttemptManifest.from_dict(self._storage.read_json(path))
        except ContractError as exc:
            raise RecordCorrupt("invalid or unsupported archive checkpoint; repair refused") from exc
        if checkpoint.last_sequence > len(events):
            raise RecordCorrupt("checkpoint references missing events")
        prefix = project(events[:checkpoint.last_sequence])
        if checkpoint != prefix:
            raise RecordCorrupt("checkpoint is not a valid history prefix; repair refused")
        status = "current" if checkpoint == manifest else "stale"
        return Verification(manifest, status, len(events), len(seen)), events

    def verify(self, attempt_id):
        with storage_errors(), exclusive_lock(self._attempt_path(attempt_id)):
            return self._inspect(attempt_id)[0]

    def read_attempt(self, attempt_id):
        report = self.verify(attempt_id)
        if report.checkpoint != "current":
            raise CheckpointStale("valid history has a stale/missing checkpoint; repair it explicitly")
        return report.manifest

    def repair_checkpoint(self, attempt_id):
        with storage_errors(), exclusive_lock(self._attempt_path(attempt_id)):
            report, _ = self._inspect(attempt_id)
            if report.checkpoint != "current":
                self._write_checkpoint(report.manifest)
            return report.manifest

    def _write_checkpoint(self, manifest):
        path = self._storage.path("attempts", manifest.spec.attempt_id, "manifest.json")
        self._storage.publish(path, canonical_bytes(manifest.to_dict()))

    def _commit_event(self, event, manifest):
        path = self._storage.path("attempts", event.attempt_id, "events",
                                  f"{event.sequence:08d}-{event.event_id}.json")
        if path.exists():
            raise RecordConflict("immutable event already exists")
        self._storage.publish(path, canonical_bytes(event.to_dict()))
        # Failure here leaves the event durable and inspectable, not rolled back.
        self._write_checkpoint(manifest)
        return manifest

    def start_attempt(self, spec):
        spec = AttemptSpec.from_dict(spec.to_dict())
        with storage_errors(), exclusive_lock(self._attempt_path(spec.attempt_id)):
            metadata = self.metadata()
            base = self._attempt_path(spec.attempt_id)
            if base.exists():
                checkpoint = self._storage.path("attempts", spec.attempt_id, "manifest.json")
                if any(base.iterdir()):
                    events = self._events(spec.attempt_id)
                    if events or checkpoint.exists():
                        report, _ = self._inspect(spec.attempt_id)
                        if report.manifest.spec != spec:
                            raise RecordConflict("attempt identity already has a different specification")
                        if report.checkpoint != "current":
                            raise CheckpointStale("attempt start persisted; repair its checkpoint before reuse")
                        return report.manifest
                # Empty event directories left before the first atomic publish
                # contain no authoritative observation and may be initialized.
            event = AttemptEvent(archive_id=metadata.archive_id, attempt_id=spec.attempt_id,
                                 event_id=uuid.uuid4().hex, sequence=1, previous_sha256=None,
                                 created_at=now(), kind="attempt_started", payload=spec)
            return self._commit_event(event, project((event,)))

    def _append(self, attempt_id, kind, payload):
        with storage_errors(), exclusive_lock(self._attempt_path(attempt_id)):
            report, events = self._inspect(attempt_id)
            if report.checkpoint != "current":
                raise CheckpointStale("history advanced; repair its checkpoint before another write")
            for previous in events:
                if previous.kind == kind and previous.payload == payload:
                    return report.manifest  # Stable-ID retry, without a duplicate event.
            event = AttemptEvent(archive_id=report.manifest.archive_id, attempt_id=attempt_id,
                                 event_id=uuid.uuid4().hex, sequence=len(events) + 1,
                                 previous_sha256=report.manifest.last_event_sha256,
                                 created_at=now(), kind=kind, payload=payload)
            manifest = advance(report.manifest, event)
            if kind == "artifact_recorded":
                for ref in payload.artifact.blobs():
                    self._storage.verify_blob(ref)
            return self._commit_event(event, manifest)

    def record_artifact(self, slot, artifact):
        artifact = Artifact.from_dict(artifact.to_dict())
        return self._append(artifact.attempt_id, "artifact_recorded", ArtifactRecorded(slot=slot, artifact=artifact))

    def resolve_slot(self, attempt_id, slot, status, reason):
        return self._append(identifier(attempt_id), "inventory_resolved",
                            InventoryResolved(slot=slot, status=status, reason=reason))

    def finish_attempt(self, attempt_id, terminal):
        terminal = AttemptFinished.from_dict(terminal.to_dict())
        return self._append(identifier(attempt_id), "attempt_finished", terminal)

    def list_attempts(self):
        with storage_errors():
            self.metadata()
            directory = self._storage.path("attempts")
            if not directory.exists():
                return ()
            manifests = []
            for path in sorted(directory.iterdir()):
                self._storage.path("attempts", path.name)
                try:
                    identifier(path.name)
                except ContractError as exc:
                    raise RecordCorrupt("unrecognized entry in archive attempts") from exc
                manifests.append(self.read_attempt(path.name))
            return tuple(manifests)
