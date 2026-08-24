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


def test_revision_lineage_is_partial_record_safe(seeded):
    parent = _add(seeded, attempt_id="att2", role="author", section="page",
                  model_used="m", decision="kept")
    _add(seeded, attempt_id="att2", role="grade", parent_iteration_id=parent,
         grader_scores=json.dumps({"semantic_score": 0.5, "n_drift": 1}))
    revision = _add(seeded, attempt_id="att2", role="author", section="page",
                    parent_iteration_id=parent, model_used="m",
                    cost_input_tokens=80, cost_output_tokens=20)
    _add(seeded, attempt_id="att2", role="grade", parent_iteration_id=revision,
         grader_scores=json.dumps({"semantic_score": 0.7, "n_drift": 0}))
    # A historical revision whose grade never landed remains eligible but is
    # explicitly incomparable; it is never treated as a regression or zero.
    _add(seeded, attempt_id="att2", role="debug", section="page",
         parent_iteration_id=revision, model_used="m")
    seeded.commit()

    lineage = insights._gather(seeded, None)["lineage"]
    assert lineage["eligible"] == 2
    assert lineage["comparable"] == 1
    assert lineage["improved"] == 1
    assert lineage["incomparable"] == 1
    assert lineage["tokens_per_improvement"] == 100


def test_latency_and_gate_metrics_report_denominators(seeded):
    _add(seeded, attempt_id="att3", role="commit", duration_ms=250,
         gate_metrics=json.dumps({"promoted": True, "gate_failures": 0}),
         decision="committed-to-wiki")
    _add(seeded, attempt_id="att4", role="commit", duration_ms=None,
         gate_metrics=None, decision="committed-to-sandbox")
    seeded.commit()
    data = insights._gather(seeded, None)
    assert data["latency"]["commit"] == {
        "samples": 1, "eligible": 2,
        "min_ms": 250, "mean_ms": 250, "median_ms": 250,
        "p95_ms": 250, "max_ms": 250, "nested_in_commit": False,
    }
    assert data["gate_health"]["samples"] == 1
    # The two historical grade rows are also eligible and correctly remain
    # unmeasured rather than being interpreted as zero defects.
    assert data["gate_health"]["eligible"] == 4
    assert data["attempt_status"]["completed"] >= 2


def test_attempt_timings_exclude_nested_commit_subphases(seeded):
    _add(seeded, attempt_id="timed", role="reconcile", duration_ms=1000)
    _add(seeded, attempt_id="timed", role="author", duration_ms=2000)
    _add(seeded, attempt_id="timed", role="keywords", duration_ms=3000)
    _add(seeded, attempt_id="timed", role="commit", duration_ms=5000,
         decision="committed-to-wiki")
    seeded.commit()

    attempt = next(a for a in insights._gather(seeded, None)["attempts"]
                   if a["attempt_id"] == "timed")
    assert attempt["measured_duration_ms"] == 8000
    assert attempt["measured_minutes"] == pytest.approx(0.133)
    assert attempt["wall_source"] == "event-span-fallback"
    assert attempt["timing_samples"] == attempt["timing_eligible"] == 3
    keyword = next(s for s in attempt["steps"] if s["role"] == "keywords")
    assert keyword["nested_in_commit"] is True


def test_terminal_attempt_timer_supplies_exact_wall_time(seeded):
    _add(seeded, attempt_id="terminal", role="reconcile", duration_ms=1000)
    _add(seeded, attempt_id="terminal", role="attempt", duration_ms=72500,
         decision="failed", decision_reason="RuntimeError: nope")
    seeded.commit()

    attempt = next(a for a in insights._gather(seeded, None)["attempts"]
                   if a["attempt_id"] == "terminal")
    assert attempt["wall_duration_ms"] == 72500
    assert attempt["wall_minutes"] == pytest.approx(1.208)
    assert attempt["wall_source"] == "terminal-timer"
    assert attempt["outcome"] == "failed"
    assert [step["role"] for step in attempt["steps"]] == ["reconcile"]


def test_revision_decision_event_is_not_a_timing_gap(seeded):
    _add(seeded, attempt_id="decision", role="tournament", duration_ms=None,
         gate_metrics=json.dumps({"operator": "evolve", "revision_accepted": False}),
         decision="discarded")
    seeded.commit()
    data = insights._gather(seeded, None)
    attempt = next(a for a in data["attempts"] if a["attempt_id"] == "decision")
    assert attempt["timing_eligible"] == 0


def test_attempts_text_view_needs_no_sql(seeded, monkeypatch, capsys):
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(seeded.execute(
        "PRAGMA database_list"
    ).fetchone()["file"]))
    assert insights.main(["--attempts"]) == 0
    out = capsys.readouterr().out
    assert "Attempt timings:" in out
    assert "att1" in out


def test_attempt_detail_text_lists_steps(seeded, monkeypatch, capsys):
    _add(seeded, attempt_id="detail", role="extract", duration_ms=60000,
         decision="observed")
    _add(seeded, attempt_id="detail", role="attempt", duration_ms=75000,
         decision="completed")
    seeded.commit()
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(seeded.execute(
        "PRAGMA database_list"
    ).fetchone()["file"]))

    assert insights.main(["--attempt-id", "detail"]) == 0
    out = capsys.readouterr().out
    assert "wall" in out and "1.25m" in out
    assert "extract" in out and "1.00m" in out
    assert "By model" not in out
