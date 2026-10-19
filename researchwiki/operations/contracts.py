"""Version 1 ingestion application contracts; no provider or filesystem I/O.

    Provider routing remains phase-based: a single attempt can mix API calls
    and chat-relay. A request never invents a job-wide API/relay toggle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ._wire import ContractError, WireModel, absolute_path, nonempty


ExecutionState = Literal["queued", "running", "finished", "interrupted", "cancelled"]
Outcome = Literal["published", "draft_ready", "already_present", "failed"]
PublicationState = Literal["not_published", "published", "unknown"]
ProjectionState = Literal["not_applicable", "current", "maintenance_pending", "unknown"]
CheckState = Literal["passed", "failed", "completed", "not_requested", "incomplete", "unavailable"]
RetryPolicy = Literal[
    "after_correction", "after_environment_recovery", "explicit_retry",
    "reconcile_first", "not_retryable",
]


@dataclass(frozen=True, kw_only=True)
class IngestOptions(WireModel):
    use_stub: bool = False
    use_semantic: bool = True
    verify_claim_entailment: bool = False
    n_drafts: int | None = None  # None preserves phase-config default resolution
    max_evolve: int = 1
    max_debug: int = 1
    promotion: Literal["auto", "always", "never"] = "auto"
    doi: str | None = None
    title: str | None = None
    year: int | None = None
    authors: tuple[str, ...] | None = None
    author_prompt_file: str | None = None
    supplementary: tuple[str, ...] = ()
    use_llm_reconcile: bool = True
    allow_rename: bool = False
    no_cross_link: bool = False
    claim_overlap: bool = False
    contradiction_alert: bool = False
    memory_evolve: bool = False
    max_model_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_wall_seconds: float | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("n_drafts", "max_model_calls", "max_tokens", "max_cost_usd", "max_wall_seconds"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ContractError(f"{name}: must be positive")
        for name in ("max_evolve", "max_debug"):
            if getattr(self, name) < 0:
                raise ContractError(f"{name}: must be nonnegative")
        if self.year is not None and not 1000 <= self.year <= 9999:
            raise ContractError("year: must have four digits")
        for path in self.supplementary:
            absolute_path(path, "supplementary")
        if self.author_prompt_file is not None:
            absolute_path(self.author_prompt_file, "author_prompt_file")
        if self.authors is not None:
            if not self.authors:
                raise ContractError("authors: must not be empty")
            for author in self.authors:
                nonempty(author, "author")


@dataclass(frozen=True, kw_only=True)
class IngestRequest(WireModel):
    request_id: str
    library_root: str
    source_pdf: str
    options: IngestOptions = field(default_factory=IngestOptions)
    provider_profile: str | None = None
    schema_version: Literal[1] = 1
    operation: Literal["ingest"] = "ingest"

    def __post_init__(self):
        super().__post_init__()
        nonempty(self.request_id, "request_id")
        absolute_path(self.library_root, "library_root")
        absolute_path(self.source_pdf, "source_pdf")
        if Path(self.source_pdf).suffix.lower() != ".pdf":
            raise ContractError("source_pdf: expected a .pdf path")
        if self.provider_profile is not None:
            absolute_path(self.provider_profile, "provider_profile")


@dataclass(frozen=True, kw_only=True)
class OperationError(WireModel):
    code: str
    category: Literal["validation", "environment", "gate", "budget", "internal"]
    message: str
    retry: RetryPolicy

    def __post_init__(self):
        super().__post_init__()
        nonempty(self.code, "code")
        nonempty(self.message, "message")


@dataclass(frozen=True, kw_only=True)
class CheckResult(WireModel):
    name: str
    status: CheckState


@dataclass(frozen=True, kw_only=True)
class UsageReport(WireModel):
    status: Literal["reported", "estimated", "unavailable"] = "unavailable"
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("input_tokens", "output_tokens", "estimated_cost_usd"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ContractError(f"{name}: must be nonnegative")
        if self.status == "unavailable" and any(
            value is not None for value in (self.input_tokens, self.output_tokens, self.estimated_cost_usd)
        ):
            raise ContractError("unavailable usage cannot report a zero or measured total")


@dataclass(frozen=True, kw_only=True)
class RelayHandoff(WireModel):
    op_id: str
    prompt_path: str
    response_path: str
    timeout_seconds: float
    phase: str | None = None
    retry_of: str | None = None
    response_reused: bool = False

    def __post_init__(self):
        super().__post_init__()
        nonempty(self.op_id, "op_id")
        absolute_path(self.prompt_path, "prompt_path")
        absolute_path(self.response_path, "response_path")
        if self.timeout_seconds <= 0:
            raise ContractError("timeout_seconds: must be positive")


@dataclass(frozen=True, kw_only=True)
class IngestResult(WireModel):
    request_id: str
    execution: ExecutionState
    outcome: Outcome | None
    publication: PublicationState
    projections: ProjectionState = "unknown"
    post_work: Literal["not_requested", "completed", "partial", "failed"] = "not_requested"
    job_id: str | None = None
    attempt_id: str | None = None
    paper_stem: str | None = None
    page_path: str | None = None
    pdf_path: str | None = None
    draft_path: str | None = None
    promotion: Literal["auto", "always", "never"] = "auto"
    gate_reasons: tuple[str, ...] = ()
    checks: tuple[CheckResult, ...] = ()
    warnings: tuple[str, ...] = ()
    error: OperationError | None = None
    usage: UsageReport = field(default_factory=UsageReport)
    pending_relays: tuple[RelayHandoff, ...] = ()
    schema_version: Literal[1] = 1

    def __post_init__(self):
        super().__post_init__()
        nonempty(self.request_id, "request_id")
        if self.execution == "finished" and self.outcome is None:
            raise ContractError("finished execution requires an outcome")
        if self.execution in ("queued", "running", "cancelled") and self.outcome is not None:
            raise ContractError("unfinished or cancelled execution cannot have an outcome")
        if self.execution == "cancelled" and self.publication != "not_published":
            raise ContractError("cancelled execution must not claim uncertain publication")
        if self.outcome in ("published", "already_present") and self.publication != "published":
            raise ContractError("published/already_present outcome requires published artifacts")
        if self.outcome == "draft_ready" and self.publication != "not_published":
            raise ContractError("draft_ready must not claim canonical publication")
        if self.publication == "published" and self.outcome == "failed":
            raise ContractError("post-publication failures must preserve the published outcome")
        if self.projections in ("current", "maintenance_pending") and self.publication != "published":
            raise ContractError("projection state requires known canonical publication")
        for name in ("page_path", "pdf_path", "draft_path"):
            if getattr(self, name) is not None:
                absolute_path(getattr(self, name), name)


def transition_execution(current: ExecutionState, target: ExecutionState) -> ExecutionState:
    """Retries create new jobs; interrupted jobs can only be reconciled."""
    allowed = {
        "queued": {"running", "cancelled", "finished"},
        "running": {"finished", "interrupted"},
        "interrupted": {"finished"},
        "finished": set(),
        "cancelled": set(),
    }
    if current not in allowed or target not in allowed[current]:
        raise ContractError(f"invalid execution transition: {current!r} -> {target!r}")
    return target


@dataclass(frozen=True, kw_only=True)
class IngestEvent(WireModel):
    request_id: str
    sequence: int
    timestamp: str
    kind: Literal[
        "model_call_started", "model_call_finished", "model_call_failed",
        "relay_waiting", "relay_response_received", "relay_response_rejected", "relay_timeout",
    ]
    call_id: str
    phase: str | None = None
    provider: str | None = None
    model: str | None = None
    attempt_id: str | None = None
    job_id: str | None = None
    relay: RelayHandoff | None = None
    usage: UsageReport | None = None
    error_code: str | None = None
    schema_version: Literal[1] = 1

    def __post_init__(self):
        super().__post_init__()
        nonempty(self.request_id, "request_id")
        nonempty(self.call_id, "call_id")
        if self.sequence < 1:
            raise ContractError("sequence: must be positive")
        from datetime import datetime
        try:
            stamp = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractError("timestamp: must be an ISO timestamp") from exc
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise ContractError("timestamp: must be in UTC")
        if self.kind.startswith("relay_") and self.relay is None:
            raise ContractError("relay events require exact handoff identity")
