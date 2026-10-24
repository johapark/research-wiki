"""Explicit owner-reported ingestion facts and content-free lifecycle events.

Facts are in-process receipts, independent of event delivery. They do not replace
the mutation journal or prove publication after a worker process has disappeared.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import uuid

from .observation import current_recorder


@dataclass
class IngestFacts:
    publication: dict | None = None
    components: dict = field(default_factory=dict)
    memory_work: str = "unknown"
    gate_reasons: tuple[str, ...] = ()
    checks: tuple = ()

    @property
    def projections(self):
        if self.publication is None:
            return "unknown"
        required = ("database", "keyword_index", "semantic_index", "claim_grades")
        current = all(self.components.get(k) == "completed" for k in required)
        return "current" if current and self.components.get("post_hooks", "completed") == "completed" else "maintenance_pending"


_facts = ContextVar("researchwiki_ingest_facts", default=None)


def current_facts():
    return _facts.get()


@contextmanager
def observe_lifecycle(facts):
    token = _facts.set(facts)
    try:
        yield
    finally:
        _facts.reset(token)


def emit(kind, *, call_id=None, stage=None, **details):
    recorder = current_recorder()
    if recorder is not None:
        if stage is not None:
            details["phase"] = stage
        recorder.emit(kind, call_id=call_id or uuid.uuid4().hex, **details)


@contextmanager
def phase(name, *, ctx=None, slot=None):
    from ..records.lifecycle import current_owner, check_capture
    owner = current_owner()
    if owner is not None:
        owner.visited.add(name)
    identity = uuid.uuid4().hex
    details = {"phase": name, "iteration": getattr(ctx, "iteration", None), "slot": slot}
    emit("phase_started", call_id=identity, **details)
    try:
        yield
        check_capture()
    except BaseException as exc:
        emit("phase_failed", call_id=identity, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
             error_code=type(exc).__name__, **details)
        raise
    else:
        emit("phase_finished", call_id=identity, status="completed", **details)


def observed_phase(name):
    def decorate(fn):
        @wraps(fn)
        def run(ctx, *args, **kwargs):
            with phase(name, ctx=ctx, slot=kwargs.get("slot")):
                return fn(ctx, *args, **kwargs)
        return run
    return decorate


def observed_attempt(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        from .observation import observe_models

        recorder = kwargs.get("event_sink") or current_recorder()
        with observe_models(recorder):
            try:
                ctx = fn(*args, **kwargs)
            except BaseException as exc:
                emit("attempt_finished", status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error_code=type(exc).__name__, publication=(
                         "published" if current_facts() and current_facts().publication else "unknown"))
                raise
            emit("attempt_finished", status={"promoted": "published", "sandboxed": "draft_ready",
                 "already_present": "already_present"}.get(ctx.outcome, "unknown"))
            return ctx
    return run


def publication_details(attempt_id, stem, category):
    from ..ingest_ledger import journal_binding

    details = {"stem": stem, "category": category, "repair_projections": True, "attempt_id": attempt_id}
    details.update(journal_binding())
    recorder = current_recorder()
    if recorder is not None and recorder.attempt_id == attempt_id:
        details.update(request_id=recorder.request_id, job_id=recorder.job_id)
    return details


def canonical_committed(result, stem):
    """Called only after the canonical mutation's mark_committed succeeds."""
    identity = {"paper_stem": stem, "page_path": str(result.wiki_path.resolve()),
                "pdf_path": str(result.pdf_path.resolve())}
    facts = current_facts()
    if facts is not None:
        facts.publication = identity
    emit("publication_committed", publication="published", projections="maintenance_pending", **identity)


def maintenance(component, status):
    facts = current_facts()
    if facts is not None:
        facts.components[component] = status
    emit("maintenance_finished", phase=component, status=status,
         projections=facts.projections if facts and facts.publication else None)


def gate_evaluated(ctx, gate):
    facts = current_facts()
    if facts is not None:
        from .results import checks_from_scores

        facts.gate_reasons = tuple(gate.reasons)
        facts.checks = checks_from_scores(getattr(ctx.winner, "scores", {}) or {})
    emit("gate_evaluated", stage="promotion", promotion=ctx.promote_mode,
         status="not_requested" if ctx.promote_mode == "never" else "failed" if gate.reasons else "passed",
         gate_failures=0 if ctx.promote_mode == "never" else len(gate.reasons), paper_stem=ctx.paper_stem)


def memory_completed(status):
    facts = current_facts()
    if facts is not None:
        facts.memory_work = status
    emit("maintenance_finished", stage="memory_evolve", status=status)


def result_with_facts(result, facts):
    from dataclasses import replace

    if facts.publication is None:
        return result
    return replace(result, **facts.publication, outcome="published", publication="published",
                   projections=facts.projections, gate_reasons=facts.gate_reasons,
                   checks=facts.checks or result.checks)
