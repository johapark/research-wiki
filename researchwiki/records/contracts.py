"""Version-one internal archive contracts. No provider, corpus, or filesystem I/O."""

from dataclasses import dataclass, fields
from datetime import datetime
import re
from typing import Literal
import uuid

from ..operations._wire import ContractError, WireModel, nonempty
from .jsonio import canonical_bytes, digest

ArtifactKind = Literal["source_snapshot", "representation", "model_request", "model_response",
                       "extraction_result", "parse_trace", "terminal_result"]
Resolution = Literal["not_run", "omitted_by_policy", "unavailable", "failed", "unknown"]
InventoryStatus = Literal["pending", "retained", "not_run", "omitted_by_policy", "unavailable", "failed", "unknown"]
EXTRACTION_SLOTS = (
    ("source.original", "source_snapshot"), ("source.representation", "representation"),
    ("target.request", "model_request"), ("target.response", "model_response"),
    ("target.result", "extraction_result"), ("target.trace", "parse_trace"),
)


def identifier(value):
    if not isinstance(value, str):
        raise ContractError("archive identifier must be a UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ContractError("archive identifier must be a UUID") from exc
    if value not in (parsed.hex, str(parsed)):
        raise ContractError("archive identifier must be canonical lowercase UUID text")
    return value


def sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ContractError("invalid archive SHA256")
    return value


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", value) or ".." in value:
        raise ContractError("invalid archive slot/phase token")


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            raise ValueError("timezone missing")
    except (ValueError, TypeError) as exc:
        raise ContractError("archive timestamp must include a timezone") from exc


class RecordModel(WireModel):
    @classmethod
    def from_dict(cls, data):
        # Every on-disk field, including the version and optional nulls, is explicit.
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
            raise ContractError(f"invalid {cls.__name__} fields")
        return super().from_dict(data)


@dataclass(frozen=True, kw_only=True)
class StoreMetadata(RecordModel):
    archive_id: str
    created_at: str
    schema_version: Literal[1] = 1

    def __post_init__(self):
        super().__post_init__()
        identifier(self.archive_id)
        timestamp(self.created_at)


@dataclass(frozen=True, kw_only=True)
class BlobRef(RecordModel):
    sha256: str
    size_bytes: int
    media_type: str
    encoding: Literal["utf-8"] | None = None
    schema_version: Literal[1] = 1

    def __post_init__(self):
        super().__post_init__()
        sha256(self.sha256)
        if self.size_bytes < 0 or not re.fullmatch(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+", self.media_type):
            raise ContractError("invalid archive blob size or media type")

    @property
    def relative_path(self):
        return f"blobs/sha256/{self.sha256[:2]}/{self.sha256}"


@dataclass(frozen=True, kw_only=True)
class Producer(RecordModel):
    name: str
    version: str
    revision: str | None = None

    def __post_init__(self):
        super().__post_init__()
        nonempty(self.name, "producer name")
        nonempty(self.version, "producer version")
        if self.revision is not None and not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ContractError("invalid producer revision")


@dataclass(frozen=True, kw_only=True)
class SourceRef(RecordModel):
    document: BlobRef
    role: Literal["original", "effective", "displaced_canonical", "supplementary"]
    representation: BlobRef | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.role != "supplementary" and self.document.media_type != "application/pdf":
            raise ContractError("paper source must reference a PDF blob")


@dataclass(frozen=True, kw_only=True)
class Artifact(RecordModel):
    artifact_id: str
    attempt_id: str
    kind: ArtifactKind
    payload: BlobRef
    producer: Producer
    created_at: str
    origin: Literal["observed", "historical_import", "reconstructed", "reprocessed", "synthetic"]
    sources: tuple[SourceRef, ...] = ()
    parents: tuple[str, ...] = ()
    phase: str | None = None
    call_id: str | None = None
    dispatch_id: str | None = None
    slot: int | None = None
    schema_version: Literal[1] = 1

    def __post_init__(self):
        super().__post_init__()
        for value in (self.artifact_id, self.attempt_id, *self.parents):
            identifier(value)
        for value in (self.call_id, self.dispatch_id):
            if value is not None:
                identifier(value)
        timestamp(self.created_at)
        if self.phase is not None:
            token(self.phase)
        if len(set(self.parents)) != len(self.parents) or self.artifact_id in self.parents:
            raise ContractError("duplicate or self-referencing artifact parents")
        if self.slot is not None and self.slot < 0:
            raise ContractError("negative author slot")
        if self.dispatch_id is not None and self.call_id is None:
            raise ContractError("dispatch requires a logical call")
        if self.kind == "source_snapshot" and not any(s.document == self.payload for s in self.sources):
            raise ContractError("source snapshot must bind its own document blob")
        if self.kind == "representation" and not any(s.representation == self.payload for s in self.sources):
            raise ContractError("representation artifact must bind its source representation")

    def blobs(self):
        yield self.payload
        for source in self.sources:
            yield source.document
            if source.representation is not None:
                yield source.representation


@dataclass(frozen=True, kw_only=True)
class CapturePolicy(RecordModel):
    mode: Literal["required", "off"]
    profile: Literal["extraction-v1"] = "extraction-v1"


@dataclass(frozen=True, kw_only=True)
class Requirement(RecordModel):
    slot: str
    kind: ArtifactKind
    required: bool = True

    def __post_init__(self):
        super().__post_init__()
        token(self.slot)


def extraction_requirements():
    """Minimum declared inventory; producers explicitly resolve unexecuted slots."""
    return tuple(Requirement(slot=slot, kind=kind) for slot, kind in EXTRACTION_SLOTS)


@dataclass(frozen=True, kw_only=True)
class AttemptSpec(RecordModel):
    attempt_id: str
    policy: CapturePolicy
    producer: Producer
    requirements: tuple[Requirement, ...]
    created_at: str
    request_id: str | None = None
    job_id: str | None = None
    generation: int | None = None
    schema_version: Literal[1] = 1

    def __post_init__(self):
        super().__post_init__()
        identifier(self.attempt_id)
        timestamp(self.created_at)
        if len({r.slot for r in self.requirements}) != len(self.requirements):
            raise ContractError("duplicate inventory slots")
        declared = {r.slot: r for r in self.requirements}
        if any(slot not in declared or declared[slot].kind != kind or not declared[slot].required
               for slot, kind in EXTRACTION_SLOTS):
            raise ContractError("extraction profile requires its complete minimum inventory")
        if self.request_id is not None:
            nonempty(self.request_id, "request ID")
        if self.job_id is not None:
            identifier(self.job_id)
        if (self.job_id is None) != (self.generation is None) or (self.generation is not None and self.generation < 1):
            raise ContractError("job identity requires a positive generation")


@dataclass(frozen=True, kw_only=True)
class ArtifactRecorded(RecordModel):
    slot: str
    artifact: Artifact

    def __post_init__(self):
        super().__post_init__()
        token(self.slot)


@dataclass(frozen=True, kw_only=True)
class InventoryResolved(RecordModel):
    slot: str
    status: Resolution
    reason: str

    def __post_init__(self):
        super().__post_init__()
        token(self.slot)
        nonempty(self.reason, "inventory resolution reason")


@dataclass(frozen=True, kw_only=True)
class AttemptFinished(RecordModel):
    execution: Literal["finished", "interrupted"]
    outcome: Literal["published", "draft_ready", "already_present", "failed"] | None
    publication: Literal["published", "not_published", "unknown"]
    projections: Literal["current", "maintenance_pending", "not_applicable", "unknown"]
    reason: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.execution == "finished" and self.outcome is None:
            raise ContractError("finished execution requires an outcome")
        if self.outcome in ("published", "already_present") and self.publication != "published":
            raise ContractError("published outcome requires published state")
        if self.outcome == "draft_ready" and self.publication != "not_published":
            raise ContractError("draft outcome cannot claim publication")
        if self.publication == "published" and self.projections == "not_applicable":
            raise ContractError("published state needs a projection observation")
        if self.publication == "not_published" and self.projections not in ("not_applicable", "unknown"):
            raise ContractError("unpublished state cannot claim projection maintenance")
        if self.reason is not None:
            nonempty(self.reason, "terminal reason")


@dataclass(frozen=True, kw_only=True)
class AttemptEvent(RecordModel):
    archive_id: str
    attempt_id: str
    event_id: str
    sequence: int
    previous_sha256: str | None
    created_at: str
    kind: Literal["attempt_started", "artifact_recorded", "inventory_resolved", "attempt_finished"]
    payload: AttemptSpec | ArtifactRecorded | InventoryResolved | AttemptFinished
    schema_version: Literal[1] = 1

    def __post_init__(self):
        super().__post_init__()
        for value in (self.archive_id, self.attempt_id, self.event_id):
            identifier(value)
        timestamp(self.created_at)
        types = dict(attempt_started=AttemptSpec, artifact_recorded=ArtifactRecorded,
                     inventory_resolved=InventoryResolved, attempt_finished=AttemptFinished)
        if type(self.payload) is not types[self.kind] or self.sequence < 1:
            raise ContractError("event kind/payload or sequence mismatch")
        if (self.sequence == 1) != (self.previous_sha256 is None):
            raise ContractError("only the first event has no predecessor")
        if self.previous_sha256 is not None:
            sha256(self.previous_sha256)

    @property
    def digest(self):
        return digest(canonical_bytes(self.to_dict()))


@dataclass(frozen=True, kw_only=True)
class InventoryItem(RecordModel):
    slot: str
    status: InventoryStatus
    artifact_ids: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self):
        super().__post_init__()
        token(self.slot)
        for value in self.artifact_ids:
            identifier(value)
        if len(set(self.artifact_ids)) != len(self.artifact_ids):
            raise ContractError("duplicate inventory artifact IDs")
        if self.status == "retained" and (not self.artifact_ids or self.reason is not None):
            raise ContractError("retained inventory requires artifacts without a failure reason")
        if self.status == "pending" and (self.artifact_ids or self.reason is not None):
            raise ContractError("pending inventory cannot claim observations")
        if self.status not in ("pending", "retained") and (self.reason is None or not self.reason.strip()):
            raise ContractError("resolved inventory requires a reason")
        if self.status in ("not_run", "omitted_by_policy") and self.artifact_ids:
            raise ContractError("unexecuted inventory cannot contain artifacts")


@dataclass(frozen=True, kw_only=True)
class AttemptManifest(RecordModel):
    archive_id: str
    spec: AttemptSpec
    inventory: tuple[InventoryItem, ...]
    artifacts: tuple[Artifact, ...]
    terminal: AttemptFinished | None
    last_sequence: int
    last_event_sha256: str
    capture_state: Literal["in_progress", "complete", "incomplete", "disabled"]
    schema_version: Literal[1] = 1

    def __post_init__(self):
        super().__post_init__()
        identifier(self.archive_id)
        sha256(self.last_event_sha256)
        if self.last_sequence < 1:
            raise ContractError("manifest requires a first event")
        if tuple(item.slot for item in self.inventory) != tuple(req.slot for req in self.spec.requirements):
            raise ContractError("manifest inventory differs from its specification")
        artifacts = {artifact.artifact_id: artifact for artifact in self.artifacts}
        if len(artifacts) != len(self.artifacts) or any(a.attempt_id != self.spec.attempt_id for a in self.artifacts):
            raise ContractError("manifest has duplicate or foreign artifacts")
        assigned = []
        for req, item in zip(self.spec.requirements, self.inventory):
            for artifact_id in item.artifact_ids:
                if artifact_id not in artifacts or artifacts[artifact_id].kind != req.kind:
                    raise ContractError("inventory references an unknown or wrong-kind artifact")
                assigned.append(artifact_id)
        if len(set(assigned)) != len(assigned) or set(assigned) != set(artifacts):
            raise ContractError("every artifact must belong to exactly one inventory slot")
        if self.capture_state != capture_state(self.spec, self.inventory, self.terminal):
            raise ContractError("capture state contradicts the declared inventory")


def capture_state(spec, inventory, terminal):
    if spec.policy.mode == "off":
        return "disabled"
    if terminal is None:
        return "in_progress"
    statuses = {item.slot: item.status for item in inventory}
    return "complete" if all(not req.required or statuses[req.slot] in ("retained", "not_run")
                             for req in spec.requirements) else "incomplete"
