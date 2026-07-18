"""Call-count levers for memory_evolve: adaptive neighbor selection + the
judged-pair ledger. Pure-logic / isolated-DB tests (no LLM, no wiki)."""
from __future__ import annotations

from types import SimpleNamespace

from researchwiki.agents.phases.evolution import _adaptive_select
from researchwiki.agents.phases import evolve_ledger


def _hit(key, score):
    return SimpleNamespace(key=key, score=score)


# ---------- _adaptive_select ----------

def test_below_floor_dropped():
    hits = [_hit("a", 0.80), _hit("b", 0.60)]
    kept, dropped = _adaptive_select(hits, min_cosine=0.65, gap=0.12, min_keep=4, max_keep=12)
    assert [h.key for h in kept] == ["a"]
    assert ("b", "below_floor") in [(h.key, r) for h, r in dropped]


def test_marginal_tail_trimmed_by_gap():
    # top 0.83, a tail beyond the 0.12 gap (< 0.71) is dropped once min_keep met.
    hits = [_hit(k, s) for k, s in
            [("a", 0.83), ("b", 0.75), ("c", 0.73), ("d", 0.72), ("e", 0.69), ("f", 0.69)]]
    kept, dropped = _adaptive_select(hits, min_cosine=0.65, gap=0.12, min_keep=4, max_keep=12)
    assert [h.key for h in kept] == ["a", "b", "c", "d"]          # e,f are 0.69 < 0.71
    assert sorted(r for _, r in dropped) == ["gap", "gap"]


def test_accepted_at_gap_boundary_kept():
    # an accepted-at-0.73 neighbor (top 0.83) is exactly within the 0.12 gap.
    hits = [_hit("top", 0.83), _hit("edit", 0.73)]
    kept, _ = _adaptive_select(hits, min_cosine=0.65, gap=0.12, min_keep=1, max_keep=12)
    assert "edit" in [h.key for h in kept]


def test_min_keep_overrides_gap():
    # only 2 above floor; min_keep=4 keeps both even though the 2nd is beyond gap.
    hits = [_hit("a", 0.90), _hit("b", 0.66)]
    kept, dropped = _adaptive_select(hits, min_cosine=0.65, gap=0.05, min_keep=4, max_keep=12)
    assert {h.key for h in kept} == {"a", "b"} and dropped == []


def test_max_keep_caps_hub_paper():
    hits = [_hit(f"n{i}", 0.90) for i in range(15)]  # all tied, all within gap
    kept, dropped = _adaptive_select(hits, min_cosine=0.65, gap=0.12, min_keep=4, max_keep=12)
    assert len(kept) == 12
    assert [r for _, r in dropped] == ["cap"] * 3


def test_no_candidates_above_floor():
    kept, dropped = _adaptive_select([_hit("a", 0.5)], min_cosine=0.65, gap=0.12, min_keep=4, max_keep=12)
    assert kept == []
    assert [(h.key, r) for h, r in dropped] == [("a", "below_floor")]


# ---------- evolve_ledger ----------

def _ledger(tmp_path):
    return evolve_ledger.open_ledger(tmp_path / "judged.db")


def test_page_hash_deterministic_and_content_sensitive():
    assert evolve_ledger.page_hash("abc") == evolve_ledger.page_hash("abc")
    assert evolve_ledger.page_hash("abc") != evolve_ledger.page_hash("abd")


def test_record_then_cached_hit(tmp_path):
    conn = _ledger(tmp_path)
    evolve_ledger.record_none(conn, "src", "synthesis/t", "sh", "th", now=1)
    assert evolve_ledger.is_cached_none(conn, "src", "synthesis/t", "sh", "th")


def test_cache_miss_on_changed_target_hash(tmp_path):
    conn = _ledger(tmp_path)
    evolve_ledger.record_none(conn, "src", "synthesis/t", "sh", "th", now=1)
    assert not evolve_ledger.is_cached_none(conn, "src", "synthesis/t", "sh", "DIFFERENT")


def test_cache_miss_on_changed_source_hash(tmp_path):
    conn = _ledger(tmp_path)
    evolve_ledger.record_none(conn, "src", "synthesis/t", "sh", "th", now=1)
    assert not evolve_ledger.is_cached_none(conn, "src", "synthesis/t", "DIFFERENT", "th")


def test_clear_removes_entry(tmp_path):
    conn = _ledger(tmp_path)
    evolve_ledger.record_none(conn, "src", "synthesis/t", "sh", "th", now=1)
    evolve_ledger.clear(conn, "src", "synthesis/t")
    assert not evolve_ledger.is_cached_none(conn, "src", "synthesis/t", "sh", "th")


def test_record_none_upserts(tmp_path):
    conn = _ledger(tmp_path)
    evolve_ledger.record_none(conn, "src", "synthesis/t", "sh", "th1", now=1)
    evolve_ledger.record_none(conn, "src", "synthesis/t", "sh", "th2", now=2)  # target changed
    assert not evolve_ledger.is_cached_none(conn, "src", "synthesis/t", "sh", "th1")
    assert evolve_ledger.is_cached_none(conn, "src", "synthesis/t", "sh", "th2")
    n = conn.execute("SELECT COUNT(*) FROM judged_none").fetchone()[0]
    assert n == 1  # upsert, not duplicate
