import sqlite3
import time
from types import SimpleNamespace

import pytest

from researchwiki.agents import budget, llm
from researchwiki.agents.runner_support import finalize_attempt_timing
from researchwiki.db.connection import get_connection


def test_parallel_reservations_cannot_oversubscribe_call_limit():
    tracker = budget.BudgetTracker(budget.IngestBudget(max_model_calls=1))
    first = tracker.reserve_call(
        model="unknown-local", provider="lmstudio", prompt_chars=40, max_tokens=10
    )
    with pytest.raises(budget.BudgetExhausted) as exc:
        tracker.reserve_call(
            model="unknown-local", provider="lmstudio", prompt_chars=40, max_tokens=10
        )
    assert exc.value.dimension == "model_calls"
    tracker.finish(first, model="unknown-local", input_tokens=5, output_tokens=4)
    assert tracker.snapshot()["model_calls"] == 1


def test_token_budget_is_reserved_before_call():
    tracker = budget.BudgetTracker(budget.IngestBudget(max_tokens=20))
    with pytest.raises(budget.BudgetExhausted) as exc:
        tracker.reserve_call(
            model="unknown-local", provider="lmstudio", prompt_chars=40, max_tokens=11
        )
    assert exc.value.dimension == "tokens"
    assert tracker.snapshot()["model_calls"] == 0


def test_budget_stop_is_not_swallowed_by_best_effort_exception_handlers():
    stop = budget.BudgetExhausted("tokens", 1, 2, {})
    with pytest.raises(budget.BudgetExhausted):
        try:
            raise stop
        except Exception:  # mirrors optional ingest phases
            pytest.fail("budget control flow was swallowed")


def test_cost_budget_refuses_unpriced_cloud_model(monkeypatch):
    monkeypatch.setattr(budget.model_config, "rate_for", lambda model: None)
    tracker = budget.BudgetTracker(budget.IngestBudget(max_cost_usd=1.0))
    with pytest.raises(budget.BudgetExhausted) as exc:
        tracker.reserve_call(
            model="unpriced-cloud", provider="anthropic",
            prompt_chars=40, max_tokens=10,
        )
    assert exc.value.dimension == "cost_pricing_missing"


def test_suspended_budget_does_not_block_post_promotion_housekeeping():
    tracker = budget.BudgetTracker(budget.IngestBudget(max_model_calls=1))
    tracker.suspend()
    with budget.activate(tracker):
        assert budget.current_tracker() is None
        tracker.check_wall()


def test_terminal_attempt_timer_survives_exhausted_wall_budget():
    rows = []
    ctx = SimpleNamespace(
        attempt_id="a", paper_stem="stem", pdf_filename="paper.pdf",
        iteration=4, budget_exhausted={"dimension": "wall_seconds"},
    )
    finalize_attempt_timing(
        ctx, None, started=time.monotonic() - 0.01,
        error=budget.BudgetExhausted("wall_seconds", 1, 2, {}),
        writer=lambda **kw: rows.append(kw),
    )
    assert ctx.iteration == 5
    assert rows[0]["role"] == "attempt"
    assert rows[0]["decision"] == "budget-exhausted"
    assert rows[0]["duration_ms"] >= 9


def test_llm_call_charges_actual_usage(monkeypatch):
    response = llm.LLMResponse("ok", "priced-model", 0.0, 7, 3)
    monkeypatch.setattr(llm, "call_anthropic", lambda **kwargs: response)
    tracker = budget.BudgetTracker(budget.IngestBudget(max_model_calls=1, max_tokens=100))
    with budget.activate(tracker):
        got = llm.call(
            prompt="small", model="priced-model", provider="anthropic", max_tokens=20
        )
    assert got.text == "ok"
    snap = tracker.snapshot()
    assert snap["model_calls"] == 1
    assert snap["tokens"] == 10
    assert snap["reserved_tokens"] == 0


def test_priced_local_call_stays_unmetered_after_finish(monkeypatch):
    monkeypatch.setattr(budget.model_config, "rate_for", lambda model: object())
    monkeypatch.setattr(
        budget.model_config, "estimate_usd", lambda model, input_tokens, output_tokens: 5.0
    )
    tracker = budget.BudgetTracker(budget.IngestBudget(max_cost_usd=0.1))
    reservation = tracker.reserve_call(
        model="priced-local",
        provider="lmstudio",
        prompt_chars=40,
        max_tokens=10,
        free=True,
    )

    tracker.finish(
        reservation, model="priced-local", input_tokens=1_000, output_tokens=1_000
    )

    assert tracker.snapshot()["estimated_cost_usd"] == 0.0


def test_llm_cost_budget_uses_the_resolved_local_endpoint(monkeypatch):
    response = llm.LLMResponse("ok", "priced-model", 0.0, 7, 3)
    monkeypatch.setattr(
        llm,
        "resolve_openai_endpoint",
        lambda: llm.EndpointResolution("http://localhost:1234/v1", "env"),
    )
    monkeypatch.setattr(llm, "call_openai_compatible", lambda **kwargs: response)
    monkeypatch.setattr(budget.model_config, "rate_for", lambda model: object())
    monkeypatch.setattr(
        budget.model_config, "estimate_usd", lambda model, input_tokens, output_tokens: 5.0
    )
    tracker = budget.BudgetTracker(budget.IngestBudget(max_cost_usd=0.1))

    with budget.activate(tracker):
        got = llm.call(
            prompt="small",
            model="priced-model",
            provider="openai-compatible",
            max_tokens=20,
        )

    assert got.text == "ok"
    assert tracker.snapshot()["estimated_cost_usd"] == 0.0


def test_remote_hostname_containing_localhost_is_not_unmetered(monkeypatch):
    monkeypatch.setattr(
        llm,
        "resolve_openai_endpoint",
        lambda: llm.EndpointResolution("https://localhost.invalid/v1", "env"),
    )
    monkeypatch.setattr(budget.model_config, "rate_for", lambda model: None)
    monkeypatch.setattr(
        llm,
        "call_openai_compatible",
        lambda **kwargs: pytest.fail("cost guard must run before transport"),
    )
    tracker = budget.BudgetTracker(budget.IngestBudget(max_cost_usd=1.0))

    with budget.activate(tracker), pytest.raises(budget.BudgetExhausted) as exc:
        llm.call(
            prompt="small",
            model="unpriced-remote",
            provider="openai-compatible",
            max_tokens=20,
        )

    assert exc.value.dimension == "cost_pricing_missing"


def test_local_anthropic_endpoint_is_unmetered(monkeypatch):
    response = llm.LLMResponse("ok", "priced-model", 0.0, 7, 3)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:8000")
    monkeypatch.setattr(llm, "call_anthropic", lambda **kwargs: response)
    monkeypatch.setattr(budget.model_config, "rate_for", lambda model: object())
    monkeypatch.setattr(
        budget.model_config, "estimate_usd", lambda model, input_tokens, output_tokens: 5.0
    )
    tracker = budget.BudgetTracker(budget.IngestBudget(max_cost_usd=0.1))

    with budget.activate(tracker):
        got = llm.call(
            prompt="small",
            model="priced-model",
            provider="anthropic",
            max_tokens=20,
        )

    assert got.text == "ok"
    assert tracker.snapshot()["estimated_cost_usd"] == 0.0


def test_throttle_failure_settles_the_reservation(monkeypatch):
    def fail_throttle(*args, **kwargs):
        raise RuntimeError("synthetic throttle failure")

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(llm, "_throttle", fail_throttle)
    tracker = budget.BudgetTracker(
        budget.IngestBudget(max_model_calls=1, max_tokens=100)
    )

    with budget.activate(tracker), pytest.raises(RuntimeError, match="throttle"):
        llm.call(
            prompt="small",
            model="test-model",
            provider="anthropic",
            max_tokens=20,
        )

    snapshot = tracker.snapshot()
    assert snapshot["model_calls"] == 1
    assert snapshot["reserved_tokens"] == 0

def test_old_batch_args_without_budgets_remain_valid():
    from researchwiki.tasks.agent import build_parser, _batch_passthrough_args
    args = build_parser().parse_args(["ingest", "a.pdf"])
    emitted = _batch_passthrough_args(args)
    assert not any(flag.startswith("--max-") for flag in emitted)


def test_budget_flags_propagate_to_batch_workers():
    from researchwiki.tasks.agent import build_parser, _batch_passthrough_args
    args = build_parser().parse_args([
        "ingest", "a.pdf", "--max-model-calls", "3", "--max-tokens", "1000",
        "--max-cost-usd", "0.5", "--max-wall-seconds", "60",
    ])
    emitted = _batch_passthrough_args(args)
    assert emitted[-8:] == [
        "--max-model-calls", "3", "--max-tokens", "1000",
        "--max-cost-usd", "0.5", "--max-wall-seconds", "60.0",
    ]


def test_legacy_iteration_table_gets_nullable_telemetry_columns(tmp_path):
    path = tmp_path / "legacy.db"
    raw = sqlite3.connect(path)
    raw.execute(
        "CREATE TABLE ingest_iterations ("
        "id INTEGER PRIMARY KEY, attempt_id TEXT, paper_stem TEXT, "
        "role TEXT, created_at INTEGER)"
    )
    raw.execute(
        "INSERT INTO ingest_iterations "
        "(attempt_id, paper_stem, role, created_at) VALUES ('old', NULL, 'reconcile', 1)"
    )
    raw.commit()
    raw.close()

    conn = get_connection(path)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(ingest_iterations)")}
    row = conn.execute(
        "SELECT duration_ms, gate_metrics FROM ingest_iterations WHERE attempt_id='old'"
    ).fetchone()
    assert {"duration_ms", "gate_metrics"} <= cols
    assert row["duration_ms"] is None and row["gate_metrics"] is None
    conn.close()
