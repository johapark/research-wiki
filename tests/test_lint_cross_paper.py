"""Tests for the cross-paper contradiction lint (Pattern 1 — Starling-transferable).

Three guarantees worth pinning down without invoking the real LLM:
  1. Pair generation skips same-paper pairs (the rule is "claims across DIFFERENT
     papers"). A high-similarity intra-paper pair must not be returned.
  2. The judge function is called once per cross-paper pair above the threshold;
     `agree` and `different_topic` verdicts are filtered out.
  3. The function gracefully no-ops when no embeddings infrastructure is
     available, so default `researchwiki lint` (no --cross-paper) never breaks.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from researchwiki.tasks.lint import cross_paper as cp


def _make_db(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE papers (stem TEXT PRIMARY KEY);
        CREATE TABLE claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_stem TEXT NOT NULL,
            section TEXT NOT NULL,
            position INTEGER NOT NULL,
            text TEXT NOT NULL,
            claim_slug TEXT,
            is_cross_ref INTEGER NOT NULL DEFAULT 0,
            supporting_text TEXT,
            last_graded_at INTEGER
        );
        """
    )
    return conn


def _seed_claims(conn, rows: list[tuple[str, str, str]]) -> None:
    """rows = [(stem, text, supporting_text), ...]; section/position auto-generated."""
    for i, (stem, text, supp) in enumerate(rows):
        conn.execute(
            "INSERT OR IGNORE INTO papers (stem) VALUES (?)",
            (stem,),
        )
        conn.execute(
            """INSERT INTO claims
               (paper_stem, section, position, text, is_cross_ref,
                supporting_text, last_graded_at)
               VALUES (?, 'key_contributions', ?, ?, 0, ?, 1)""",
            (stem, i, text, supp),
        )
    conn.commit()


def test_skips_same_paper_pairs(tmp_path, monkeypatch):
    """Two near-identical claims from the same paper must not produce a pair."""
    conn = _make_db(tmp_path)
    _seed_claims(
        conn,
        [
            ("paperA", "near-paraphrase claim one", "ctx-a-1"),
            ("paperA", "near-paraphrase claim two", "ctx-a-2"),
        ],
    )

    monkeypatch.setattr(cp, "_NUMPY_AVAILABLE", True)
    monkeypatch.setattr(
        "researchwiki.index.embeddings.is_available", lambda: True, raising=True,
    )
    # Force both claims to identical unit vectors → cosine 1.0 — same-paper
    # filter is the only thing that should prevent the pair.
    fake_embs = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(
        "researchwiki.index.embeddings.embed_texts",
        lambda texts: fake_embs[: len(texts)],
        raising=True,
    )

    judge_calls: list[str] = []
    def _judge(prompt: str) -> dict:
        judge_calls.append(prompt)
        return {"verdict": "disagree_numeric", "rationale": "should not be reached"}

    out = cp.find_cross_paper_contradictions(judge_fn=_judge, db_conn=conn)
    assert out == []
    assert judge_calls == []  # judge never invoked because no cross-paper pair


def test_filters_agree_and_different_topic(tmp_path, monkeypatch):
    """Only disagree_numeric / disagree_direction verdicts pass through."""
    conn = _make_db(tmp_path)
    _seed_claims(
        conn,
        [
            ("paperA", "claim alpha", "ctx-a"),
            ("paperB", "claim beta", "ctx-b"),
            ("paperC", "claim gamma", "ctx-c"),
            ("paperD", "claim delta", "ctx-d"),
        ],
    )

    monkeypatch.setattr(cp, "_NUMPY_AVAILABLE", True)
    monkeypatch.setattr(
        "researchwiki.index.embeddings.is_available", lambda: True, raising=True,
    )
    # All four claims project onto the same vector → every pair has cosine 1.0,
    # so all 6 cross-paper pairs become candidates. Judge classifies them in a
    # mix of verdicts; the function should keep only the two disagreement kinds.
    fake_embs = np.array([[1.0, 0.0]] * 4, dtype=np.float32)
    monkeypatch.setattr(
        "researchwiki.index.embeddings.embed_texts",
        lambda texts: fake_embs[: len(texts)],
        raising=True,
    )

    verdicts = iter([
        {"verdict": "agree", "rationale": "same fact"},
        {"verdict": "disagree_numeric", "rationale": "A says 80, B says 60"},
        {"verdict": "different_topic", "rationale": "embedding artifact"},
        {"verdict": "disagree_direction", "rationale": "A says +, B says -"},
        {"verdict": "agree", "rationale": "same"},
        {"verdict": "agree", "rationale": "same"},
    ])
    out = cp.find_cross_paper_contradictions(
        judge_fn=lambda _p: next(verdicts), db_conn=conn,
    )
    kept_verdicts = sorted(c["verdict"] for c in out)
    assert kept_verdicts == ["disagree_direction", "disagree_numeric"]


def test_noop_when_numpy_missing(tmp_path, monkeypatch, capsys):
    """No numpy → empty result + warning, never raises."""
    conn = _make_db(tmp_path)
    _seed_claims(conn, [("paperA", "x", "y"), ("paperB", "x", "y")])
    monkeypatch.setattr(cp, "_NUMPY_AVAILABLE", False)
    out = cp.find_cross_paper_contradictions(db_conn=conn)
    assert out == []
    err = capsys.readouterr().err
    assert "numpy" in err.lower()


def test_noop_when_fewer_than_two_claims(tmp_path, monkeypatch):
    conn = _make_db(tmp_path)
    _seed_claims(conn, [("paperA", "lone claim", "ctx")])
    monkeypatch.setattr(cp, "_NUMPY_AVAILABLE", True)
    monkeypatch.setattr(
        "researchwiki.index.embeddings.is_available", lambda: True, raising=True,
    )
    out = cp.find_cross_paper_contradictions(db_conn=conn)
    assert out == []
