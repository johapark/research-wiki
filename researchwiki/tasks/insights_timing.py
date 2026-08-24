"""Timing analytics for ``researchwiki insights``.

Durations are sparse by design on migrated databases. Every view therefore
keeps a measured/eligible denominator and never interprets NULL as zero.
Nested commit subphases are visible individually but excluded from attempt
totals because their wall time is already included in the parent ``commit``.
"""

from __future__ import annotations

import json
from collections import defaultdict


NESTED_COMMIT_ROLES = frozenset({
    "short_name", "keywords", "promote", "index_update",
    "memory_evolve", "grade_persist",
})
TIMED_ROLES = frozenset({
    "reconcile", "extract", "target_claims", "crosslinks", "author",
    "grade", "tournament", "critic", "debug", "claim_support", "commit",
    *NESTED_COMMIT_ROLES,
    "attempt",
})


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def _is_timing_event(row) -> bool:
    if row["role"] not in TIMED_ROLES:
        return False
    # Revision keep/reject rows reuse role=tournament but represent decisions,
    # not work. New rows identify themselves through gate_metrics.operator.
    if row["role"] == "tournament" and row["gate_metrics"]:
        try:
            if json.loads(row["gate_metrics"]).get("operator"):
                return False
        except (TypeError, json.JSONDecodeError):
            pass
    return True


def latency_distribution(rows) -> dict[str, dict]:
    eligible: dict[str, int] = defaultdict(int)
    values: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if not _is_timing_event(row):
            continue
        role = row["role"]
        eligible[role] += 1
        if row["duration_ms"] is not None:
            values[role].append(int(row["duration_ms"]))
    out = {}
    for role in sorted(eligible):
        vals = sorted(values.get(role, []))
        out[role] = {
            "samples": len(vals),
            "eligible": eligible[role],
            "min_ms": vals[0] if vals else None,
            "mean_ms": round(sum(vals) / len(vals)) if vals else None,
            "median_ms": _percentile(vals, 0.5),
            "p95_ms": _percentile(vals, 0.95),
            "max_ms": vals[-1] if vals else None,
            "nested_in_commit": role in NESTED_COMMIT_ROLES,
        }
    return out


def gather_attempts(conn, where_time: str, params: tuple, *,
                    stem: str | None = None,
                    attempt_id: str | None = None) -> list[dict]:
    filters = ""
    extra: list[str] = []
    if stem:
        filters += " AND paper_stem = ?"
        extra.append(stem)
    if attempt_id:
        filters += " AND attempt_id = ?"
        extra.append(attempt_id)
    rows = conn.execute(
        "SELECT id, attempt_id, paper_stem, pdf_filename, iteration, role, "
        "duration_ms, decision, decision_reason, gate_metrics, created_at "
        "FROM ingest_iterations WHERE 1=1" + where_time + filters +
        " ORDER BY id",
        (*params, *extra),
    ).fetchall()
    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[row["attempt_id"]].append(row)

    attempts = []
    for key, events in grouped.items():
        timing_events = [row for row in events if _is_timing_event(row)]
        phase_events = [row for row in timing_events if row["role"] != "attempt"]
        top_level = [row for row in phase_events
                     if row["role"] not in NESTED_COMMIT_ROLES]
        measured = [row for row in top_level if row["duration_ms"] is not None]
        measured_ms = sum(int(row["duration_ms"]) for row in measured)
        event_span_ms = max(0, (events[-1]["created_at"] - events[0]["created_at"]) * 1000)
        terminal = next((row for row in reversed(events)
                         if row["role"] == "attempt"), None)
        wall_ms = (int(terminal["duration_ms"])
                   if terminal and terminal["duration_ms"] is not None
                   else event_span_ms)
        budget_stop = next((row for row in events if row["role"] == "budget"), None)
        commit = next((row for row in reversed(events) if row["role"] == "commit"), None)
        outcome = (
            terminal["decision"] if terminal else
            "budget-exhausted" if budget_stop else
            commit["decision"] if commit else
            "incomplete"
        )
        attempts.append({
            "attempt_id": key,
            "paper_stem": next((row["paper_stem"] for row in reversed(events)
                                if row["paper_stem"]), None),
            "pdf_filename": events[0]["pdf_filename"],
            "started_at": events[0]["created_at"],
            "last_event_at": events[-1]["created_at"],
            "outcome": outcome,
            "measured_minutes": round(measured_ms / 60000, 3) if measured else None,
            "measured_duration_ms": measured_ms if measured else None,
            "wall_minutes": round(wall_ms / 60000, 3),
            "wall_duration_ms": wall_ms,
            "wall_source": "terminal-timer" if terminal else "event-span-fallback",
            "event_span_minutes": round(event_span_ms / 60000, 3),
            "event_span_ms": event_span_ms,
            "timing_samples": len(measured),
            "timing_eligible": len(top_level),
            "steps": [
                {
                    "iteration": row["iteration"],
                    "role": row["role"],
                    "duration_ms": row["duration_ms"],
                    "nested_in_commit": row["role"] in NESTED_COMMIT_ROLES,
                    "decision": row["decision"],
                }
                for row in phase_events
            ],
        })
    attempts.sort(key=lambda item: (item["last_event_at"], item["attempt_id"]),
                  reverse=True)
    return attempts
