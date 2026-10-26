"""Application ingestion contracts and synchronous single-library execution."""

from ._wire import ContractError
from .contracts import (
    CheckResult, IngestEvent, IngestOptions, IngestRequest, IngestResult,
    OperationError, RelayHandoff, UsageReport, transition_execution,
)
from .events import EventRecorder
from .results import error_from_exception, result_from_context
from .observation import observe_models
from .ingest import ingest

__all__ = [
    "CheckResult", "ContractError", "EventRecorder", "IngestEvent", "IngestOptions",
    "IngestRequest", "IngestResult", "OperationError", "RelayHandoff", "UsageReport",
    "transition_execution",
    "error_from_exception", "result_from_context",
    "observe_models",
    "ingest",
]
