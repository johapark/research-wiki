"""Translate existing engine outcomes without treating retrieval as proof."""

from __future__ import annotations

from pathlib import Path

from ._wire import ContractError
from .contracts import CheckResult, IngestRequest, IngestResult, OperationError


def _path(root: str, value) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str((path if path.is_absolute() else Path(root) / path).resolve())


def result_from_context(request: IngestRequest, context, *, projections="unknown") -> IngestResult:
    """Map a returned runner Context; projection success requires separate evidence.

    Do not inspect file existence to guess whether a partially failed attempt
    published. This adapter handles only the runner's explicit terminal outcomes.
    Usage and post-work remain unknown/unrequested until their owners supply
    authoritative results; runner completion alone does not establish either.
    """
    outcomes = {"promoted": "published", "sandboxed": "draft_ready", "already_present": "already_present"}
    outcome = outcomes.get(context.outcome)
    if outcome is None:
        raise ContractError("runner did not return a terminal ingestion outcome")
    artifact = _path(request.library_root, context.committed_path)
    if artifact is None or not context.paper_stem:
        raise ContractError("terminal ingestion outcome is missing its artifact identity")
    scores = getattr(getattr(context, "winner", None), "scores", {}) or {}
    checks = checks_from_scores(scores)
    published = outcome in ("published", "already_present")
    return IngestResult(
        request_id=request.request_id, attempt_id=context.attempt_id,
        execution="finished", outcome=outcome,
        publication="published" if published else "not_published",
        projections=projections if published else "not_applicable",
        paper_stem=context.paper_stem, page_path=artifact if published else None,
        pdf_path=_path(request.library_root, f"papers/{context.paper_stem}.pdf") if published else None,
        draft_path=None if published else artifact, promotion=context.promote_mode,
        gate_reasons=tuple(context.gate_reasons), checks=tuple(checks),
    )


def error_from_exception(error: BaseException) -> OperationError:
    """Classify failure and retry policy; preserve uncertain publication separately.

    The message names the exception class only. Provider/parser exceptions can
    contain confidential prompt/response content; details belong in local traces.
    """
    from ..agents.budget import BudgetExhausted
    from ..agents.context import PromoteFailed, ReconcileFailed, StemRenameRefused
    from ..agents.relay import RelayTimeout
    from ..errors import EnvironmentFailure

    code, category, retry = "internal_error", "internal", "not_retryable"
    if isinstance(error, RelayTimeout):
        code, category, retry = "relay_timeout", "environment", "after_environment_recovery"
    elif isinstance(error, BudgetExhausted):
        code, category, retry = "budget_exhausted", "budget", "after_correction"
    elif isinstance(error, ReconcileFailed):
        code, category, retry = "metadata_unresolved", "environment", "after_correction"
    elif isinstance(error, StemRenameRefused):
        code, category, retry = "stem_rename_refused", "environment", "after_correction"
    elif isinstance(error, PromoteFailed):
        code, category, retry = "publication_failed", "environment", "reconcile_first"
    elif isinstance(error, EnvironmentFailure):
        code, category, retry = "environment_unavailable", "environment", "after_environment_recovery"
    elif isinstance(error, (ContractError, FileNotFoundError)):
        code, category, retry = "invalid_request", "validation", "after_correction"
    return OperationError(code=code, category=category, retry=retry, message=type(error).__name__)


def checks_from_scores(scores):
    from ..grade.checks import entailment_check

    checks = []
    if scores:
        checks.append(CheckResult(name="retrieval", status=(
            "completed" if scores.get("n_graded", 0) else "unavailable"
        )))
        checks.append(CheckResult(name="numeric", status=(
            "failed" if scores.get("n_drift", 0) else
            "passed" if scores.get("n_graded", 0) and "n_drift" in scores else "unavailable"
        )))
    checks.append(CheckResult(name="entailment", status=entailment_check(scores)["status"]))
    return tuple(checks)
