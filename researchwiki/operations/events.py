"""Explicit thread-safe progress recorder for API and relay calls.

This is an observation channel, not a job ledger. Sink failures are counted
and warned about without changing a model call's result or consuming another
relay response. Callers can poll ``pending_relays`` while author slots wait.
"""

from __future__ import annotations

import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Callable

from .contracts import IngestEvent, RelayHandoff, UsageReport
from ._wire import nonempty


class EventRecorder:
    def __init__(
        self, request_id: str, sink: Callable[[IngestEvent], None], *,
        job_id: str | None = None, attempt_id: str | None = None,
    ):
        nonempty(request_id, "request_id")
        if not callable(sink):
            raise TypeError("event sink must be callable")
        self.request_id = request_id
        self.job_id = job_id
        self.attempt_id = attempt_id
        self._sink = sink
        self._lock = threading.RLock()
        self._sequence = 0
        self._delivery_errors = 0
        self._pending: dict[tuple[str, str], RelayHandoff] = {}

    @property
    def delivery_errors(self) -> int:
        with self._lock:
            return self._delivery_errors

    @property
    def pending_relays(self) -> tuple[RelayHandoff, ...]:
        with self._lock:
            return tuple(self._pending.values())

    def bind_attempt(self, attempt_id: str) -> None:
        nonempty(attempt_id, "attempt_id")
        with self._lock:
            if self.attempt_id is not None and self.attempt_id != attempt_id:
                raise ValueError("an event recorder belongs to one ingest attempt")
            first = self.attempt_id is None
            self.attempt_id = attempt_id
            if first:
                self.emit("attempt_started", call_id=attempt_id)

    def emit(self, kind: str, *, call_id: str, **details) -> None:
        with self._lock:
            event = IngestEvent(
                request_id=self.request_id, job_id=self.job_id,
                attempt_id=self.attempt_id, sequence=self._sequence + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                kind=kind, call_id=call_id, **details,
            )
            self._sequence = event.sequence
            if kind in ("model_call_finished", "model_call_failed"):
                self._pending = {key: value for key, value in self._pending.items() if key[0] != call_id}
            if event.relay is not None:
                key = (call_id, event.relay.op_id)
                if kind == "relay_waiting":
                    self._pending[key] = event.relay
                else:
                    self._pending.pop(key, None)
            try:
                self._sink(event)
            except Exception:
                self._delivery_errors += 1
                if self._delivery_errors == 1:
                    print("[operations] progress delivery failed; ingest execution continues", file=sys.stderr)


class ModelCallObserver:
    """One correlation ID per dispatched call, including relay schema retries."""

    def __init__(self, recorder: EventRecorder | None, *, phase, provider: str, model: str):
        self.recorder = recorder
        self.phase = phase
        self.provider = provider
        self.model = model
        self.call_id = uuid.uuid4().hex

    def emit(self, kind: str, **details) -> None:
        if self.recorder is not None:
            self.recorder.emit(
                kind, call_id=self.call_id, phase=self.phase,
                provider=self.provider, **details,
            )

    def started(self) -> None:
        self.emit("model_call_started", model=self.model)

    def failed(self, error: BaseException) -> None:
        # Error messages can contain prompts or provider response bodies.
        self.emit("model_call_failed", model=self.model, error_code=type(error).__name__)

    def finished(self, response) -> None:
        if self.recorder is None:
            return
        usage = UsageReport() if self.provider == "chat-relay" else UsageReport(
            status="reported", input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        self.emit("model_call_finished", model=response.model, usage=usage)


def relay_event(recorder, kind: str, *, call_id: str, phase, model: str,
                op_id: str, prompt_path, response_path, timeout: float,
                retry_of=None, response_reused: bool = False) -> None:
    if recorder is not None:
        recorder.emit(
            kind, call_id=call_id, phase=phase, provider="chat-relay", model=model,
            relay=RelayHandoff(
                op_id=op_id, phase=phase, prompt_path=str(prompt_path.resolve()),
                response_path=str(response_path.resolve()), timeout_seconds=timeout,
                retry_of=retry_of, response_reused=response_reused,
            ),
        )
