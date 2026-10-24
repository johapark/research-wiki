"""Internal, explicit-root ingestion record primitives; not wired into live ingestion."""

from .contracts import (
    Artifact, AttemptFinished, AttemptSpec, BlobRef, CapturePolicy, Producer,
    Requirement, SourceRef, extraction_requirements,
)
from .errors import CheckpointStale, RecordConflict, RecordCorrupt, RecordStoreUnavailable
from .store import RecordStore, Verification

__all__ = [
    "Artifact", "AttemptFinished", "AttemptSpec", "BlobRef", "CapturePolicy",
    "Producer", "Requirement", "SourceRef", "CheckpointStale", "RecordConflict",
    "RecordCorrupt", "RecordStoreUnavailable", "RecordStore", "Verification",
    "extraction_requirements",
]
