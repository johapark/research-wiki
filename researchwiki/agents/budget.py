"""Thread-safe, optional resource budgets for one ingest attempt.

The tracker is process-scoped while an ingest runs. Batch workers are separate
processes, and the only in-process concurrency is the parallel author fan-out,
so a lock-protected shared tracker gives every draft one common budget.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

from . import model_config


@dataclass(frozen=True)
class IngestBudget:
    max_model_calls: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_wall_seconds: float | None = None

    def active(self) -> bool:
        return any(v is not None for v in (
            self.max_model_calls, self.max_tokens,
            self.max_cost_usd, self.max_wall_seconds,
        ))


class BudgetExhausted(BaseException):
    """Control-flow stop that must cross best-effort ``except Exception`` blocks.

    Several optional ingest phases deliberately downgrade provider failures to
    warnings. A budget stop is not a provider failure and must reach the runner
    so it can prevent promotion and persist the terminal event. Like
    KeyboardInterrupt, it is caught explicitly at the orchestration boundary.
    """
    def __init__(self, dimension: str, limit: float, used: float, snapshot: dict):
        self.dimension = dimension
        self.limit = limit
        self.used = used
        self.snapshot = snapshot
        super().__init__(
            f"ingest budget exhausted: {dimension} limit={limit:g}, "
            f"used_or_reserved={used:g}"
        )


@dataclass(frozen=True)
class _Reservation:
    tokens: int
    cost_usd: float


class BudgetTracker:
    def __init__(self, budget: IngestBudget):
        self.budget = budget
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self.calls = 0
        self.tokens = 0
        self.cost_usd = 0.0
        self.reserved_tokens = 0
        self.reserved_cost_usd = 0.0
        self.suspended = False

    def suspend(self) -> None:
        """Stop enforcing after an irreversible promotion has landed."""
        with self._lock:
            self.suspended = True

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict:
        return {
            "model_calls": self.calls,
            "tokens": self.tokens,
            "estimated_cost_usd": round(self.cost_usd, 8),
            "reserved_tokens": self.reserved_tokens,
            "reserved_cost_usd": round(self.reserved_cost_usd, 8),
            "wall_seconds": round(time.monotonic() - self.started, 3),
        }

    def check_wall(self) -> None:
        if self.suspended:
            return
        limit = self.budget.max_wall_seconds
        if limit is None:
            return
        elapsed = time.monotonic() - self.started
        if elapsed >= limit:
            raise BudgetExhausted("wall_seconds", limit, elapsed, self.snapshot())

    def reserve_call(
        self, *, model: str, provider: str, prompt_chars: int, max_tokens: int,
        free: bool = False,
    ) -> _Reservation:
        estimated_input = max(1, prompt_chars // 4)
        reserved_tokens = estimated_input + max(0, max_tokens)
        provider_key = provider.lower().strip()
        configured_url = model_config.base_url() or ""
        local_provider = (
            provider_key == "lmstudio"
            or (provider_key == "openai-compatible" and (
                not configured_url
                or "localhost" in configured_url
                or "127.0.0.1" in configured_url
            ))
        )
        priced = model_config.rate_for(model) is not None
        if (self.budget.max_cost_usd is not None and not free
                and not local_provider and not priced):
            raise BudgetExhausted(
                "cost_pricing_missing", self.budget.max_cost_usd, 0,
                self.snapshot(),
            )
        reserved_cost = 0.0 if free or local_provider else model_config.estimate_usd(
            model, estimated_input, max(0, max_tokens)
        )
        with self._lock:
            elapsed = time.monotonic() - self.started
            if (self.budget.max_wall_seconds is not None
                    and elapsed >= self.budget.max_wall_seconds):
                raise BudgetExhausted(
                    "wall_seconds", self.budget.max_wall_seconds, elapsed,
                    self._snapshot_unlocked(),
                )
            next_calls = self.calls + 1
            if (self.budget.max_model_calls is not None
                    and next_calls > self.budget.max_model_calls):
                raise BudgetExhausted(
                    "model_calls", self.budget.max_model_calls, next_calls,
                    self._snapshot_unlocked(),
                )
            next_tokens = self.tokens + self.reserved_tokens + reserved_tokens
            if (self.budget.max_tokens is not None
                    and next_tokens > self.budget.max_tokens):
                raise BudgetExhausted(
                    "tokens", self.budget.max_tokens, next_tokens,
                    self._snapshot_unlocked(),
                )
            next_cost = self.cost_usd + self.reserved_cost_usd + reserved_cost
            if (self.budget.max_cost_usd is not None
                    and next_cost > self.budget.max_cost_usd):
                raise BudgetExhausted(
                    "estimated_cost_usd", self.budget.max_cost_usd, next_cost,
                    self._snapshot_unlocked(),
                )
            self.calls = next_calls
            self.reserved_tokens += reserved_tokens
            self.reserved_cost_usd += reserved_cost
        return _Reservation(reserved_tokens, reserved_cost)

    def finish(self, reservation: _Reservation, *, model: str,
               input_tokens: int, output_tokens: int, free: bool = False) -> None:
        actual_tokens = max(0, input_tokens) + max(0, output_tokens)
        actual_cost = 0.0 if free else model_config.estimate_usd(
            model, max(0, input_tokens), max(0, output_tokens)
        )
        with self._lock:
            self.reserved_tokens -= reservation.tokens
            self.reserved_cost_usd -= reservation.cost_usd
            self.tokens += actual_tokens
            self.cost_usd += actual_cost

    def fail(self, reservation: _Reservation) -> None:
        """Charge the reservation when transport failure leaves usage unknown."""
        with self._lock:
            self.reserved_tokens -= reservation.tokens
            self.reserved_cost_usd -= reservation.cost_usd
            self.tokens += reservation.tokens
            self.cost_usd += reservation.cost_usd


_active_lock = threading.Lock()
_active: BudgetTracker | None = None


def current_tracker() -> BudgetTracker | None:
    return None if _active is not None and _active.suspended else _active


@contextmanager
def activate(tracker: BudgetTracker | None):
    global _active
    with _active_lock:
        previous = _active
        _active = tracker
    try:
        yield tracker
    finally:
        with _active_lock:
            _active = previous
