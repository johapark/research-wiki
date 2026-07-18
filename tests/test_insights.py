"""`researchwiki insights` — aggregation over the ingest telemetry log.

Seeds ingest_iterations rows (author + linked grade) in a temp DB and pins the
model-quality join (grade scores attributed to the authoring model), token
tallies, and section difficulty.
"""

import json
import time

import pytest

from researchwiki.db.connection import get_connection
from researchwiki.tasks import insights


def _add(conn, **kw):
    kw.setdefault("attempt_id", "att1")
    kw.setdefault("pdf_filename", "p.pdf")
    kw.setdefault("iteration", 0)
    kw.setdefault("created_at", int(time.time()))
    cols = ", ".join(kw)
    ph = ", ".join("?" for _ in kw)
    cur = conn.execute(f"INSERT INTO ingest_iterations ({cols}) VALUES ({ph})", tuple(kw.values()))
    return cur.lastrowid


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    conn = get_connection()
    # Two author drafts by different models, each with a linked grade row.
    a1 = _add(conn, role="author", section="results", model_used="big-model",
              temperature=0.5, cost_input_tokens=1000, cost_output_tokens=200, decision="committed")
    _add(conn, role="grade", section="results", parent_iteration_id=a1,
         grader_scores=json.dumps({"mean_semantic": 0.85, "n_drift": 1, "n_negation_mismatches": 0}))
    a2 = _add(conn, role="author", section="results", model_used="small-model",
              temperature=0.3, cost_input_tokens=500, cost_output_tokens=100, decision="discarded")
    _add(conn, role="grade", section="results", parent_iteration_id=a2,
         grader_scores=json.dumps({"mean_semantic": 0.70, "n_drift": 4, "n_negation_mismatches": 1}))
    # A non-model deterministic phase (should show in by_role, not skew models).
    _add(conn, role="extract", cost_input_tokens=0, cost_output_tokens=0)
    conn.commit()
    return conn


def test_model_quality_attribution(seeded):
    data = insights._gather(seeded, None)
    big = data["quality"]["big-model"]
    small = data["quality"]["small-model"]
    assert big["drafts"] == 1 and small["drafts"] == 1
    assert big["sem_sum"] / big["sem_n"] == pytest.approx(0.85)
    assert small["sem_sum"] / small["sem_n"] == pytest.approx(0.70)
    assert big["drift"] == 1 and small["drift"] == 4


def test_token_and_role_tallies(seeded):
    data = insights._gather(seeded, None)
    assert data["by_model"]["big-model"]["in_tok"] == 1000
    assert data["by_model"]["small-model"]["out_tok"] == 100
    assert "author" in data["by_role"] and "grade" in data["by_role"] and "extract" in data["by_role"]
    assert data["by_role"]["author"]["calls"] == 2


def test_section_difficulty(seeded):
    data = insights._gather(seeded, None)
    sec = data["by_section"]["results"]
    assert sec["graded"] == 2
    assert sec["drift"] == 5           # 1 + 4
    assert sec["neg"] == 1
    assert sec["sem_sum"] / sec["sem_n"] == pytest.approx((0.85 + 0.70) / 2)


def test_decisions_and_attempts(seeded):
    data = insights._gather(seeded, None)
    assert data["decisions"] == {"committed": 1, "discarded": 1}
    assert data["n_attempts"] == 1


def test_json_main_smoke(seeded, monkeypatch, capsys):
    # main() opens its own connection via the RESEARCHWIKI_DB_PATH override.
    assert insights.main(["--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_attempts"] == 1
    assert out["by_model"]["big-model"]["mean_semantic"] == pytest.approx(0.85)


def test_days_filter_excludes_old(seeded):
    # Everything seeded is "now"; a 1-day window still includes it, a negative
    # cutoff far in the future excludes all.
    future_cutoff = int(time.time()) + 86400
    data = insights._gather(seeded, future_cutoff)
    assert data["n_attempts"] == 0
    assert data["by_model"] == {}
