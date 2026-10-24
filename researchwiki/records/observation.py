"""Separate content-retention scopes, never the best-effort progress channel."""

from contextlib import contextmanager
from contextvars import ContextVar
import math
import uuid


_extraction = ContextVar("researchwiki_extraction_records", default=None)
_call = ContextVar("researchwiki_retained_call", default=None)


def current_call():
    return _call.get()


@contextmanager
def observe_extraction(recorder):
    token = _extraction.set(recorder)
    try:
        yield
    finally:
        _extraction.reset(token)


@contextmanager
def observe_call(phase, call_id, request):
    recorder = _extraction.get() if phase == "target_claims" else None
    if phase == "claim_support":
        from .grading import current_run, GradingCallRecorder
        if current_run() is not None:
            recorder = GradingCallRecorder(current_run())
    from .lineage import MODEL_PHASES, current_scope, PhaseCallRecorder
    scope = current_scope()
    if scope is not None and phase in MODEL_PHASES.get(scope.phase, set()) and recorder is None:
        recorder = PhaseCallRecorder(scope)
    capture = RetainedCall(recorder, call_id, request) if recorder else None
    token = _call.set(capture)
    try:
        yield capture
    finally:
        _call.reset(token)


def finite_setting(value):
    if value is None:
        return {"value": None, "status": "unknown"}
    if type(value) is float and not math.isfinite(value):
        return {"value": None, "status": "provider_controlled"}
    return {"value": value, "status": "requested_or_applied"}


class RetainedCall:
    def __init__(self, recorder, call_id, request):
        self.recorder = recorder
        self.call_id = call_id
        self.dispatch_id = None
        self.dispatched = False
        self.usage = None
        self.usage_model = request["model"]
        self.response_ids = []
        self.request_id = recorder.artifact(
            "target.request", "model_request", {"schema_version": 1, "boundary": "logical", **request},
            parents=recorder.dependencies(), call_id=call_id)

    def dispatch(self, request):
        self.recorder.check()
        self.dispatch_id = uuid.uuid4().hex
        self.request_id = self.recorder.artifact(
            "target.request", "model_request", {"schema_version": 1, "boundary": "client_dispatch", **request},
            parents=(self.request_id,), call_id=self.call_id, dispatch_id=self.dispatch_id)
        # A client-side dispatch observation is not proof of remote execution.
        self.dispatched = True

    def answer(self, answer, *, model=None, model_origin="provider_reported", usage=None, **details):
        # Keep known usage before a write can fail, for exact budget settlement.
        self.usage = usage
        if answer is None:
            self.recorder.answer_unavailable = True
        item = self.recorder.artifact(
            "target.response", "model_response",
            {"schema_version": 1, "boundary": "visible_answer", "answer": answer,
             "model": model, "model_origin": model_origin if model is not None else "unavailable", "usage": usage,
             "usage_status": "reported" if usage is not None else "unavailable", **details},
            parents=(self.request_id,), call_id=self.call_id, dispatch_id=self.dispatch_id)
        self.recorder.response_ids.append(item)
        self.response_ids.append(item)

    def rejected(self, error_code, status=None):
        self.recorder.artifact("target.trace", "parse_trace",
                               {"schema_version": 1, "boundary": "transport_rejection",
                                "error_code": error_code, "http_status": status},
                               parents=(self.request_id,), call_id=self.call_id, dispatch_id=self.dispatch_id)

    def finished(self, response, provider):
        # API adapters supply their reported metadata separately. LLMResponse.model
        # remains a configured identity on API paths; relay identities are self-reported.
        item = self.recorder.artifact(
            "target.response", "model_response",
            {"schema_version": 1, "boundary": "phase_input", "text": response.text,
             "model": response.model, "model_origin": "self_reported" if provider == "chat-relay" else "configured",
             "temperature": finite_setting(response.temperature)},
            parents=tuple(self.response_ids[-1:]) or (self.request_id,),
            call_id=self.call_id, dispatch_id=self.dispatch_id)
        self.recorder.response_ids.append(item)
        self.response_ids.append(item)


def dispatch(request):
    capture = current_call()
    if capture is not None:
        capture.dispatch(request)


def answer(value, **details):
    capture = current_call()
    if capture is not None:
        capture.answer(value, **details)


def rejected(error_code, status=None):
    capture = current_call()
    if capture is not None:
        capture.rejected(error_code, status)
