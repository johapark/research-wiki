"""Client-side RPM throttle in researchwiki.agents.llm._throttle.

Uses a fake clock (monkeypatched onto llm.time) so the tests exercise the
wait arithmetic without real sleeping.
"""
from __future__ import annotations

import pytest

from researchwiki.agents import llm


class FakeClock:
    """Stand-in for the `time` module: monotonic() reads a virtual clock,
    sleep() records the request and advances the clock by that amount."""
    def __init__(self, start: float = 1000.0):
        self.t = start
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(llm, "time", c)
    llm._last_call_at.clear()
    yield c
    llm._last_call_at.clear()


def test_no_throttle_when_rpm_none(clock):
    llm._throttle("m", None)
    llm._throttle("m", None)
    assert clock.slept == []


def test_no_throttle_when_rpm_zero_or_negative(clock):
    llm._throttle("m", 0)
    llm._throttle("m", -5)
    assert clock.slept == []


def test_first_call_never_sleeps(clock):
    llm._throttle("m", 60)  # no prior timestamp for this model
    assert clock.slept == []


def test_second_immediate_call_sleeps_one_interval(clock):
    # rpm=60 → interval 1.0s. Two back-to-back calls (clock hasn't advanced)
    # → the second must wait ~1.0s.
    llm._throttle("m", 60)
    llm._throttle("m", 60)
    assert clock.slept == [pytest.approx(1.0)]


def test_elapsed_time_reduces_the_wait(clock):
    # rpm=6 → interval 10s. If 4s already elapsed, only 6s remain.
    llm._throttle("m", 6)
    clock.t += 4.0
    llm._throttle("m", 6)
    assert clock.slept == [pytest.approx(6.0)]


def test_enough_elapsed_means_no_wait(clock):
    llm._throttle("m", 6)   # interval 10s
    clock.t += 12.0         # more than one interval passed
    llm._throttle("m", 6)
    assert clock.slept == []


def test_throttle_is_keyed_by_model(clock):
    # Distinct models keep independent budgets — one does not stall the other.
    llm._throttle("model-a", 60)
    llm._throttle("model-b", 60)
    assert clock.slept == []
