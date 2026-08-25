"""Append-only writer + reader helpers for the ingest_iterations table.

The state-machine runner in researchwiki/agents/runner.py writes a row through
`write_iteration` after each phase / each LLM call. The LLM never inserts
directly — that's the framework's job.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass

from .connection import get_connection


VALID_ROLES = {
    "reconcile", "extract", "target_claims", "crosslinks", "author",
    "grade", "grade_persist", "tournament", "critic", "debug", "short_name",
    "keywords", "commit", "memory_evolve", "pdf_upgrade", "claim_support",
    "budget",
    "promote", "index_update",
    "attempt",
}


@dataclass
class Iteration:
    id: int
    attempt_id: str
    paper_stem: str | None
    pdf_filename: str
    iteration: int
    role: str
    section: str | None
    draft_text: str | None
    parent_iteration_id: int | None
    grader_scores: dict | None
    critic_notes: str | None
    decision: str | None
    decision_reason: str | None
    model_used: str | None
    temperature: float | None
    cost_input_tokens: int | None
    cost_output_tokens: int | None
    cost_cache_read_tokens: int | None
    cost_cache_write_tokens: int | None
    duration_ms: int | None
    gate_metrics: dict | None
    created_at: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Iteration":
        gs = row["grader_scores"]
        return cls(
            id=row["id"],
            attempt_id=row["attempt_id"],
            paper_stem=row["paper_stem"],
            pdf_filename=row["pdf_filename"],
            iteration=row["iteration"],
            role=row["role"],
            section=row["section"],
            draft_text=row["draft_text"],
            parent_iteration_id=row["parent_iteration_id"],
            grader_scores=json.loads(gs) if gs else None,
            critic_notes=row["critic_notes"],
            decision=row["decision"],
            decision_reason=row["decision_reason"],
            model_used=row["model_used"],
            temperature=row["temperature"],
            cost_input_tokens=row["cost_input_tokens"],
            cost_output_tokens=row["cost_output_tokens"],
            cost_cache_read_tokens=row["cost_cache_read_tokens"],
            cost_cache_write_tokens=row["cost_cache_write_tokens"],
            duration_ms=row["duration_ms"],
            gate_metrics=json.loads(row["gate_metrics"]) if row["gate_metrics"] else None,
            created_at=row["created_at"],
        )


def write_iteration(
    *,
    attempt_id: str,
    pdf_filename: str,
    iteration: int,
    role: str,
    paper_stem: str | None = None,
    section: str | None = None,
    draft_text: str | None = None,
    parent_iteration_id: int | None = None,
    grader_scores: dict | None = None,
    critic_notes: str | None = None,
    decision: str | None = None,
    decision_reason: str | None = None,
    model_used: str | None = None,
    temperature: float | None = None,
    cost_input_tokens: int | None = None,
    cost_output_tokens: int | None = None,
    cost_cache_read_tokens: int | None = None,
    cost_cache_write_tokens: int | None = None,
    duration_ms: int | None = None,
    gate_metrics: dict | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Insert one ingest_iterations row. Returns the new row's id.

    The runner passes a single connection through all phases of an attempt
    so they share a transaction; pass `conn` to reuse it. If omitted, opens
    a fresh connection (fine for one-off writes).
    """
    if role not in VALID_ROLES:
        raise ValueError(f"role={role!r} not in {sorted(VALID_ROLES)}")

    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    cur = conn.execute(
        """
        INSERT INTO ingest_iterations (
            attempt_id, paper_stem, pdf_filename, iteration, role, section,
            draft_text, parent_iteration_id, grader_scores, critic_notes,
            decision, decision_reason, model_used, temperature,
            cost_input_tokens, cost_output_tokens,
            cost_cache_read_tokens, cost_cache_write_tokens,
            duration_ms, gate_metrics,
            created_at
        ) VALUES (
            :attempt_id, :paper_stem, :pdf_filename, :iteration, :role, :section,
            :draft_text, :parent_iteration_id, :grader_scores, :critic_notes,
            :decision, :decision_reason, :model_used, :temperature,
            :cost_input_tokens, :cost_output_tokens,
            :cost_cache_read_tokens, :cost_cache_write_tokens,
            :duration_ms, :gate_metrics,
            :created_at
        )
        """,
        {
            "attempt_id": attempt_id,
            "paper_stem": paper_stem,
            "pdf_filename": pdf_filename,
            "iteration": iteration,
            "role": role,
            "section": section,
            "draft_text": draft_text,
            "parent_iteration_id": parent_iteration_id,
            "grader_scores": json.dumps(grader_scores) if grader_scores else None,
            "critic_notes": critic_notes,
            "decision": decision,
            "decision_reason": decision_reason,
            "model_used": model_used,
            "temperature": temperature,
            "cost_input_tokens": cost_input_tokens,
            "cost_output_tokens": cost_output_tokens,
            "cost_cache_read_tokens": cost_cache_read_tokens,
            "cost_cache_write_tokens": cost_cache_write_tokens,
            "duration_ms": duration_ms,
            "gate_metrics": json.dumps(gate_metrics) if gate_metrics is not None else None,
            "created_at": int(time.time()),
        },
    )
    # Always commit: ingest_iterations is append-only event log, every row
    # should be durable as soon as it's written. Closing the runner connection
    # without per-write commits would silently lose all rows on rollback.
    conn.commit()
    if own_conn:
        conn.close()
    return cur.lastrowid


def read_attempt(attempt_id: str) -> list[Iteration]:
    """Return every iteration row for a single attempt, in chronological order."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM ingest_iterations WHERE attempt_id = ? ORDER BY id",
        (attempt_id,),
    ).fetchall()
    conn.close()
    return [Iteration.from_row(r) for r in rows]


def update_paper_stem(
    attempt_id: str,
    paper_stem: str,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Backfill paper_stem on existing rows once the reconciler picks one.
    Pass `conn` to reuse the runner's connection (avoids SQLite write-write
    contention against the runner's open transaction).
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    cur = conn.execute(
        "UPDATE ingest_iterations SET paper_stem = ? WHERE attempt_id = ? AND paper_stem IS NULL",
        (paper_stem, attempt_id),
    )
    n = cur.rowcount
    if own_conn:
        conn.commit()
        conn.close()
    return n
