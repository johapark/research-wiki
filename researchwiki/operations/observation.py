"""Explicit attempt scopes, propagated to parallel author slots via contextvars.

No process-wide callback or stdout interception: two request scopes can observe
different recorders. Current engine provider/budget globals still require the
planned process-per-ingest isolation; this does not make the runner thread-safe.
"""

from contextlib import contextmanager
from contextvars import ContextVar, copy_context

from .events import EventRecorder


_recorder: ContextVar[EventRecorder | None] = ContextVar("researchwiki_model_events", default=None)


def current_recorder() -> EventRecorder | None:
    return _recorder.get()


@contextmanager
def observe_models(recorder: EventRecorder | None):
    token = _recorder.set(recorder)
    try:
        yield
    finally:
        _recorder.reset(token)


def activate_ingest_scope(tracker, recorder, *, attempt_id: str):
    if recorder is not None:
        recorder.bind_attempt(attempt_id)
    return _ingest_scope(tracker, recorder)


@contextmanager
def _ingest_scope(tracker, recorder):
    from ..agents.budget import activate

    with activate(tracker), observe_models(recorder):
        yield


def submit_observed(pool, fn, *args, **kwargs):
    """Each slot gets its own copied context, sharing the thread-safe recorder."""
    return pool.submit(copy_context().run, fn, *args, **kwargs)
