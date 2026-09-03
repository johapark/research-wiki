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
        CREATE TABLE cross_paper_judgements (
            src_stem      TEXT NOT NULL,
            src_slug      TEXT NOT NULL,
            tgt_stem      TEXT NOT NULL,
            tgt_slug      TEXT NOT NULL,
            verdict       TEXT NOT NULL,
            similarity    REAL NOT NULL,
            sim_threshold REAL NOT NULL,
            judged_at     INTEGER NOT NULL,
            judge_phase   TEXT NOT NULL DEFAULT 'cross_paper_judge',
            PRIMARY KEY (src_slug, tgt_slug)
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
                supporting_text, last_graded_at, claim_slug)
               VALUES (?, 'key_contributions', ?, ?, 0, ?, 1, ?)""",
            (stem, i, text, supp, f"kc-{stem}-{i}"),
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


# ---------------------------------------------------------------------------
# Blocked scan — the full N x N product was 611 MB at 12.4k claims, on a path
# that runs every ingest. These pin that the cheap version is the same function.
# ---------------------------------------------------------------------------

def _brute_force_pairs(claims, embs, thr):
    """Reference implementation: the obvious full-matrix upper triangle."""
    sims = embs @ embs.T
    out = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            if claims[i]["paper_stem"] == claims[j]["paper_stem"]:
                continue
            s = float(sims[i, j])
            if s >= thr:
                out.append((i, j, s))
    out.sort(key=lambda t: -t[2])
    return out


def _synthetic(n=40, dim=8, seed=11):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    claims = [{"paper_stem": f"paper{i % 7}", "claim_slug": f"kc-{i}",
               "section": "key_contributions", "position": i,
               "text": f"claim {i}", "supporting_text": ""}
              for i in range(n)]
    return claims, v


@pytest.mark.parametrize("block", [1, 3, 7, cp._BLOCK])
def test_blocked_scan_matches_full_matrix_for_any_block_size(block, monkeypatch):
    """Block size is a memory knob, never a semantic one."""
    monkeypatch.setattr(cp, "_BLOCK", block)
    claims, embs = _synthetic()
    got = cp._candidate_pairs(claims, embs, 0.10)
    want = _brute_force_pairs(claims, embs, 0.10)
    # Endpoints and their order must match exactly; the similarities only to
    # float tolerance, since a different block size groups the matmul
    # differently and the last bits legitimately move.
    assert [(i, j) for i, j, _ in got] == [(i, j) for i, j, _ in want]
    assert [s for *_, s in got] == pytest.approx([s for *_, s in want], abs=1e-6)


def test_scan_returns_pairs_in_cosine_descending_order():
    """`max_pairs` slices off the front, so order decides which pairs get judged.

    Ported from claim_discover's blocked scan, which ranks by IDF mass and needs
    no cosine sort — transplanting that loop without re-adding the sort would
    silently change the judged sample rather than fail anything.
    """
    claims, embs = _synthetic()
    sims = [s for _, _, s in cp._candidate_pairs(claims, embs, 0.10)]
    assert sims == sorted(sims, reverse=True)
    assert len(sims) > 5, "fixture too sparse to be meaningful"


def test_only_stem_pushdown_matches_scan_then_filter():
    """Restricting rows up front == scanning everything and filtering after."""
    claims, embs = _synthetic()
    target = "paper3"
    pushed = cp._candidate_pairs(claims, embs, 0.10, only_stem=target)
    after = [
        (i, j, s) for i, j, s in cp._candidate_pairs(claims, embs, 0.10)
        if target in (claims[i]["paper_stem"], claims[j]["paper_stem"])
    ]
    # Compare as unordered endpoint sets: the pushdown always emits the target
    # paper's claim as the row, where the triangle scan emits the lower index.
    # Similarities are compared separately, to tolerance — the two paths group
    # the matmul differently, so the last bits legitimately differ.
    keys = lambda ps: sorted(tuple(sorted((i, j))) for i, j, _ in ps)
    assert keys(pushed) == keys(after)
    by_key = lambda ps: dict((tuple(sorted((i, j))), s) for i, j, s in ps)
    pa, aa = by_key(pushed), by_key(after)
    assert [pa[k] for k in sorted(pa)] == pytest.approx(
        [aa[k] for k in sorted(aa)], abs=1e-6)
    assert pushed, "fixture produced no pairs for the target paper"
    assert all(target in (claims[i]["paper_stem"], claims[j]["paper_stem"])
               for i, j, _ in pushed)


def test_only_stem_with_no_matching_claims_is_empty():
    claims, embs = _synthetic()
    assert cp._candidate_pairs(claims, embs, 0.10, only_stem="nonexistent") == []


# ---------------------------------------------------------------------------
# Resumability. Before the coverage table, the judge's only trace was a
# `contradicts` edge — written for disagreements alone — so a re-run re-paid for
# every pair it had already cleared, and "has this pool been judged?" had no
# answer. These pin both halves.
# ---------------------------------------------------------------------------

def _two_paper_db(tmp_path, monkeypatch):
    conn = _make_db(tmp_path)
    _seed_claims(conn, [
        ("paperA", "accuracy on benchmark Y was 91 percent", "ctx-a"),
        ("paperB", "accuracy on benchmark Y was 44 percent", "ctx-b"),
    ])
    monkeypatch.setattr(cp, "_NUMPY_AVAILABLE", True)
    monkeypatch.setattr(
        "researchwiki.index.embeddings.is_available", lambda: True, raising=True)
    fake = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(
        "researchwiki.index.embeddings.embed_texts",
        lambda texts: fake[: len(texts)], raising=True)
    return conn


def test_cleared_pair_is_recorded_and_skipped_on_rerun(tmp_path, monkeypatch):
    """An `agree` verdict must still cost exactly one judge call, ever."""
    conn = _two_paper_db(tmp_path, monkeypatch)
    calls = []

    def _judge(prompt):
        calls.append(prompt)
        return {"verdict": "agree", "rationale": "consistent"}

    first = {}
    assert cp.find_cross_paper_contradictions(
        judge_fn=_judge, db_conn=conn, stats=first) == []
    assert len(calls) == 1
    assert first == {"stopped_early": None,
                     "pool": 1, "judged": 1, "skipped_already_judged": 0,
                     "disagreements": 0, "sim_threshold": 0.85}
    # The clear is recorded even though it produced no edge and no report row.
    assert conn.execute(
        "SELECT verdict FROM cross_paper_judgements").fetchone()["verdict"] == "agree"

    second = {}
    assert cp.find_cross_paper_contradictions(
        judge_fn=_judge, db_conn=conn, stats=second) == []
    assert len(calls) == 1, "re-run must not re-judge a pair already cleared"
    assert second["pool"] == 1 and second["skipped_already_judged"] == 1
    assert second["judged"] == 0


def test_rejudge_overrides_the_skip(tmp_path, monkeypatch):
    conn = _two_paper_db(tmp_path, monkeypatch)
    calls = []

    def _judge(prompt):
        calls.append(prompt)
        return {"verdict": "different_topic", "rationale": "distinct cohorts"}

    cp.find_cross_paper_contradictions(judge_fn=_judge, db_conn=conn)
    cp.find_cross_paper_contradictions(judge_fn=_judge, db_conn=conn)
    assert len(calls) == 1
    cp.find_cross_paper_contradictions(judge_fn=_judge, db_conn=conn, rejudge=True)
    assert len(calls) == 2


def test_disagreement_records_both_judgement_row_and_edge(tmp_path, monkeypatch):
    """The edge is the finding; the row is the receipt that the pair was seen."""
    conn = _two_paper_db(tmp_path, monkeypatch)
    stats = {}
    out = cp.find_cross_paper_contradictions(
        judge_fn=lambda _p: {"verdict": "disagree_numeric", "rationale": "91 vs 44"},
        db_conn=conn, stats=stats)
    assert len(out) == 1 and out[0]["verdict"] == "disagree_numeric"
    assert stats["disagreements"] == 1 and stats["judged"] == 1

    row = conn.execute("SELECT * FROM cross_paper_judgements").fetchone()
    assert row["verdict"] == "disagree_numeric"
    # Canonical ordering: the same sort both the row and the edge key on, so a
    # symmetric relation can't be stored twice under opposite orientations.
    assert (row["src_stem"], row["tgt_stem"]) == ("paperA", "paperB")
    assert row["sim_threshold"] == 0.85

    from researchwiki.claim_graph import open_edges_db
    edges = open_edges_db()
    try:
        assert edges.execute(
            "SELECT COUNT(*) FROM edges WHERE relation='contradicts'"
        ).fetchone()[0] == 1
    finally:
        edges.close()


def test_pool_is_reported_when_max_pairs_is_zero(tmp_path, monkeypatch):
    """Sizing a sweep must cost no judge calls — the zero-cost dry run."""
    conn = _two_paper_db(tmp_path, monkeypatch)
    calls = []
    stats = {}
    out = cp.find_cross_paper_contradictions(
        judge_fn=lambda p: calls.append(p), db_conn=conn,
        max_pairs=0, stats=stats)
    assert out == [] and calls == []
    assert stats["pool"] == 1, "pool must be filled before the max_pairs slice"
    assert stats["judged"] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM cross_paper_judgements").fetchone()[0] == 0


def test_stats_are_wellformed_even_when_nothing_clears_the_threshold(tmp_path, monkeypatch):
    conn = _two_paper_db(tmp_path, monkeypatch)
    stats = {}
    assert cp.find_cross_paper_contradictions(
        judge_fn=lambda _p: {"verdict": "agree"}, db_conn=conn,
        sim_threshold=1.5, stats=stats) == []
    assert stats == {"stopped_early": None,
                     "pool": 0, "judged": 0, "skipped_already_judged": 0,
                     "disagreements": 0, "sim_threshold": 1.5}


# ---------- house rule 3 (errors.py): the sweep stops and reports ----------

def _all_same_direction(monkeypatch, n: int) -> None:
    """Force every claim onto one vector so all cross-paper pairs are candidates."""
    monkeypatch.setattr(cp, "_NUMPY_AVAILABLE", True)
    monkeypatch.setattr(
        "researchwiki.index.embeddings.is_available", lambda: True, raising=True,
    )
    fake = np.array([[1.0, 0.0]] * n, dtype=np.float32)
    monkeypatch.setattr(
        "researchwiki.index.embeddings.embed_texts",
        lambda texts: fake[: len(texts)],
        raising=True,
    )


def test_sweep_stops_on_an_environment_failure_and_keeps_its_verdicts(
    tmp_path, monkeypatch
):
    """A chat-relay responder who walks away mid-sweep used to unwind the whole
    of `lint`, discarding the ~30 free local checks already computed. Tolerating
    it per pair would be worse still: the failure modes are correlated, so each
    remaining pair would pay its own full RW_RELAY_TIMEOUT for nothing.
    """
    from researchwiki.agents.relay import RelayTimeout

    conn = _make_db(tmp_path)
    _seed_claims(conn, [
        ("paperA", "claim alpha", "ctx-a"),
        ("paperB", "claim beta", "ctx-b"),
        ("paperC", "claim gamma", "ctx-c"),
        ("paperD", "claim delta", "ctx-d"),
    ])
    _all_same_direction(monkeypatch, 4)

    calls = {"n": 0}

    def _judge(_prompt):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise RelayTimeout("chat-relay: no response in 600s for x.response.json")
        return {"verdict": "disagree_numeric", "rationale": "A 80 vs B 60"}

    stats: dict = {}
    out = cp.find_cross_paper_contradictions(
        judge_fn=_judge, db_conn=conn, stats=stats)

    assert calls["n"] == 3, "stopped at the failure, did not keep trying pairs"
    assert len(out) == 2, "the two verdicts reached are still returned"
    assert stats["judged"] == 2
    assert stats["stopped_early"], "a partial sweep must announce itself"
    assert "RelayTimeout" in stats["stopped_early"]


def test_verdicts_reached_before_the_halt_are_committed(tmp_path, monkeypatch):
    """The judge spend is not lost: `_record_judgement` commits per pair, so a
    re-run skips what the halted sweep already paid for."""
    from researchwiki.agents.relay import RelayTimeout

    conn = _make_db(tmp_path)
    _seed_claims(conn, [
        ("paperA", "claim alpha", "ctx-a"),
        ("paperB", "claim beta", "ctx-b"),
        ("paperC", "claim gamma", "ctx-c"),
    ])
    _all_same_direction(monkeypatch, 3)

    calls = {"n": 0}

    def _judge(_prompt):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RelayTimeout("no response in 600s")
        return {"verdict": "agree", "rationale": "same fact"}

    cp.find_cross_paper_contradictions(judge_fn=_judge, db_conn=conn, stats={})
    recorded = conn.execute("SELECT COUNT(*) FROM cross_paper_judgements").fetchone()[0]
    assert recorded == 1, "the pair judged before the halt is persisted"

    # A second run resumes rather than re-paying for it.
    second: dict = {}
    cp.find_cross_paper_contradictions(
        judge_fn=lambda _p: {"verdict": "agree"}, db_conn=conn, stats=second)
    assert second["skipped_already_judged"] == 1
