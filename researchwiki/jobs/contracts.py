"""Versioned machine-local job records; no provider state or credentials."""

from dataclasses import dataclass
from typing import Literal
import re

from ..operations import IngestRequest, IngestResult
from ..operations._wire import ContractError, WireModel, nonempty
from ..operations.contracts import ExecutionState


def validate_job_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ContractError("job_id must be a lowercase UUID hex value")
    return value


@dataclass(frozen=True, kw_only=True)
class InputIdentity(WireModel):
    pdf_sha256: str
    auxiliary_sha256: str
    configuration_sha256: str


@dataclass(frozen=True, kw_only=True)
class JobRecord(WireModel):
    job_id: str
    library_id: str
    request: IngestRequest
    request_sha256: str
    execution: ExecutionState
    created_at: str
    updated_at: str
    generation: int = 0
    owner: str | None = None
    worker_pid: int | None = None
    result: IngestResult | None = None
    identity: InputIdentity | None = None
    source_changed: bool | None = None
    configuration_changed: bool | None = None
    parent_job_id: str | None = None
    schema_version: Literal[1] = 1

    def __post_init__(self):
        super().__post_init__()
        validate_job_id(self.job_id)
        if self.parent_job_id is not None:
            validate_job_id(self.parent_job_id)
        nonempty(self.library_id, "library_id")
        if self.generation < 0:
            raise ContractError("generation must be nonnegative")
        if self.execution == "running":
            if not self.owner or self.worker_pid is None or self.worker_pid <= 0 or self.result is not None:
                raise ContractError("running job requires an owner and PID, without a terminal result")
        elif self.owner is not None or self.worker_pid is not None:
            raise ContractError("only running jobs have an owner")
        if self.execution in ("finished", "interrupted", "cancelled"):
            if self.result is None or self.result.execution != self.execution:
                raise ContractError("terminal job requires a matching result")
        elif self.result is not None:
            raise ContractError("unfinished job cannot have a result")
        if self.result is not None and (
            self.result.job_id != self.job_id or self.result.request_id != self.request.request_id
        ):
            raise ContractError("job/result identity mismatch")
