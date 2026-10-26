"""Synchronous ingestion for one already-configured library process.

This is the shared operation behind the foreground CLI and isolated local workers. It
does not configure a process, schedule jobs, parse terminal output, or retry.
"""

from dataclasses import dataclass, field, replace

from .contracts import IngestRequest, IngestResult, OperationError
from .events import EventRecorder
from .observation import observe_models
from .post_work import PostWorkReport, run_post_work
from .results import error_from_exception, result_from_context
from .runtime import recover_ingest_mutations, runner_options, validate_runtime
from .lifecycle import IngestFacts, emit, observe_lifecycle, phase, result_with_facts


@dataclass
class IngestExecution:
    """Internal presentation adapter; mutable runner state stays out of the wire."""

    result: IngestResult
    context: object = None
    error: BaseException | None = None
    recovery_notes: tuple[str, ...] = ()
    post_work: PostWorkReport = field(default_factory=PostWorkReport)


def ingest(request: IngestRequest, *, event_sink=None) -> IngestResult:
    """Execute once and return a typed result, including known failure outcomes.

    The caller must start a process at ``library_root`` with the requested
    profile already selected through the existing env-profile loader. Progress
    accepts an EventRecorder or a callback receiving IngestEvent objects.
    Existing engine diagnostics and foreground relay handoffs remain visible.
    """
    return execute_ingest(request, event_sink=event_sink).result


def execute_ingest(request, *, event_sink=None, runner=None, recover=None) -> IngestExecution:
    """Internal hook for CLI rendering and deterministic engine integration tests."""
    recorder = event_sink if isinstance(event_sink, EventRecorder) else EventRecorder(
        request.request_id, event_sink if event_sink is not None else lambda event: None,
    )
    if recorder.request_id != request.request_id:
        from ._wire import ContractError

        error = ContractError("event recorder belongs to a different request")
        return IngestExecution(IngestResult(
            request_id=request.request_id, execution="finished", outcome="failed",
            publication="not_published", projections="not_applicable",
            error=error_from_exception(error), promotion=request.options.promotion,
        ), error=error)
    facts = IngestFacts()
    with observe_models(recorder), observe_lifecycle(facts):
        emit("operation_started", call_id=request.request_id)
        execution = _execute(request, recorder, facts, runner, recover)
        result = result_with_facts(execution.result, facts)
        if facts.publication and result.error is not None:
            result = replace(result, error=replace(result.error, retry="reconcile_first"), post_work="partial")
        if facts.publication and result.projections == "maintenance_pending":
            result = replace(result, warnings=result.warnings + ("Required projection maintenance is pending",))
        result = replace(result, job_id=recorder.job_id)
        emit("operation_finished", call_id=request.request_id,
             status="interrupted" if result.execution == "interrupted" else result.outcome,
             publication=result.publication, projections=result.projections,
             post_work=result.post_work, paper_stem=result.paper_stem,
             error_code=result.error.code if result.error else None)
        execution.result = _delivery_warnings(result, recorder)
        return execution


def _execute(request, recorder, facts, runner, recover):
    from ..agents.budget import BudgetExhausted

    ctx = None
    started = False
    notes = ()
    post = PostWorkReport()
    try:
        from ._wire import ContractError

        if recorder.request_id != request.request_id:
            raise ContractError("event recorder belongs to a different request")
        with phase("validation"):
            validate_runtime(request)
        if not request.options.use_stub:
            from ..agents.llm import preflight_providers

            with phase("preflight"):
                preflight_providers()
        options = runner_options(request.options)
        with phase("recovery"):
            notes = tuple((recover or recover_ingest_mutations)() or ())
        emit("recovery_finished", status="completed")
        if runner is None:
            from ..agents.runner import run_ingest

            runner = run_ingest
        with observe_models(recorder):
            started = True
            ctx = runner(request.source_pdf, event_sink=recorder, **options)
            result = result_from_context(request, ctx)
            try:
                with phase("post_work"):
                    post = run_post_work(ctx, request.options)
            except Exception as exc:
                # Canonical publication already happened. Never turn a hook
                # failure into an invitation to repeat authoring/publication.
                post = PostWorkReport(state="failed", warnings=[f"post-work: {type(exc).__name__}"])
            result = replace(result, post_work=post.state, warnings=tuple(post.warnings))
    except (Exception, BudgetExhausted, KeyboardInterrupt) as exc:
        if ctx is not None:
            # Preserve an explicit terminal runner outcome if a later adapter
            # fails. Do not infer canonical success from a matching filename.
            try:
                result = result_from_context(request, ctx)
            except Exception:
                result = _failure(request, recorder, exc, started)
            else:
                result = replace(
                    result, post_work="failed",
                    execution="interrupted" if isinstance(exc, KeyboardInterrupt) else "finished",
                    error=_interrupt_error() if isinstance(exc, KeyboardInterrupt) else error_from_exception(exc),
                )
        else:
            result = _failure(request, recorder, exc, started)
        return IngestExecution(result, ctx, exc, notes, post)
    return IngestExecution(result, ctx, None, notes, post)


def _failure(request, recorder, error, started):
    interrupted = isinstance(error, KeyboardInterrupt)
    operation_error = _interrupt_error() if interrupted else error_from_exception(error)
    partial = getattr(error, "partial_path", None)
    if partial is not None:
        from pathlib import Path
        partial = str((Path(request.library_root) / partial).resolve())
    return IngestResult(
        request_id=request.request_id, attempt_id=recorder.attempt_id,
        execution="interrupted" if interrupted else "finished",
        outcome=None if interrupted else "failed",
        publication="unknown" if started else "not_published",
        projections="unknown" if started else "not_applicable",
        paper_stem=getattr(error, "stem", None),
        draft_path=partial,
        promotion=request.options.promotion, error=operation_error,
    )


def _interrupt_error():
    return OperationError(
        code="interrupted", category="environment", message="KeyboardInterrupt", retry="reconcile_first",
    )


def _delivery_warnings(result, recorder):
    if not recorder.delivery_errors:
        return result
    return replace(result, warnings=result.warnings + (
        f"{recorder.delivery_errors} progress event delivery failure(s)",
    ))
