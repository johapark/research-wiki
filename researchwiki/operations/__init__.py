"""Application ingestion contracts and opt-in model progress observation.

The synchronous operation service and durable worker/jobs layer are subsequent
increments. These types do not make the existing mutable runner Context public.
"""

from ._wire import ContractError
from .contracts import (
    CheckResult, IngestEvent, IngestOptions, IngestRequest, IngestResult,
    OperationError, RelayHandoff, UsageReport, transition_execution,
)
from .events import EventRecorder
from .results import error_from_exception, result_from_context
from .observation import observe_models

__all__ = [
    "CheckResult", "ContractError", "EventRecorder", "IngestEvent", "IngestOptions",
    "IngestRequest", "IngestResult", "OperationError", "RelayHandoff", "UsageReport",
    "transition_execution",
    "error_from_exception", "result_from_context",
    "observe_models",
]
