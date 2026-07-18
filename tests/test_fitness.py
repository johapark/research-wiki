"""Per-operator fitness (agents/fitness.py) and the drafting-stance selector.

These pin the behavior that distinguishes the operator lenses:
  - tournament_key rewards coverage breadth on a near-tie (AUTHOR).
  - is_evolve_improvement credits lifting the weakest claim (EVOLVE).
  - is_strict_improvement does NOT (DEBUG / generic) — the contrast is the point.
"""

from types import SimpleNamespace

from researchwiki.agents.fitness import (
    is_evolve_improvement,
    is_strict_improvement,
    tournament_key,
)
from researchwiki.agents.phases.draft import DRAFT_STANCES, stance_for_slot


def _d(**scores):
    return SimpleNamespace(scores=scores)


# ---------- tournament_key (AUTHOR: coverage breadth) ----------

def test_higher_fidelity_wins_regardless_of_coverage():
    high_fidelity = _d(semantic_score=0.70, n_graded=3)
    broad = _d(semantic_score=0.60, n_graded=30)
    assert tournament_key(high_fidelity) > tournament_key(broad)


def test_coverage_breaks_a_fidelity_near_tie():
    # Same semantic bucket (~0.60) → the broader-coverage draft wins.
    narrow = _d(semantic_score=0.602, n_graded=5)
    broad = _d(semantic_score=0.604, n_graded=12)
    assert tournament_key(broad) > tournament_key(narrow)


def test_tournament_picks_broad_draft_on_tie():
    from researchwiki.agents.phases.draft import tournament
    narrow = SimpleNamespace(scores={"semantic_score": 0.601, "n_graded": 4},
                             iteration_id=1)
    broad = SimpleNamespace(scores={"semantic_score": 0.603, "n_graded": 11},
                            iteration_id=2)
    winner, _ = tournament([narrow, broad])
    assert winner is broad


# ---------- is_evolve_improvement (EVOLVE: floor lift) ----------

def test_evolve_accepts_floor_lift_on_tied_mean():
    new = _d(semantic_score=0.60, n_drift=0, weakest_score=0.50)
    old = _d(semantic_score=0.60, n_drift=0, weakest_score=0.40)
    assert is_evolve_improvement(new, old) is True


def test_evolve_rejects_floor_drop_on_tied_mean():
    new = _d(semantic_score=0.60, n_drift=0, weakest_score=0.40)
    old = _d(semantic_score=0.60, n_drift=0, weakest_score=0.50)
    assert is_evolve_improvement(new, old) is False


def test_evolve_mean_gain_still_wins():
    assert is_evolve_improvement(
        _d(semantic_score=0.70, weakest_score=0.1),
        _d(semantic_score=0.60, weakest_score=0.9),
    ) is True


def test_evolve_drift_veto_precedes_floor():
    # More drift is rejected even when the floor rose — integrity first.
    new = _d(semantic_score=0.60, n_drift=2, weakest_score=0.90)
    old = _d(semantic_score=0.60, n_drift=0, weakest_score=0.10)
    assert is_evolve_improvement(new, old) is False


# ---------- contrast: the strict (DEBUG) rule ignores the floor ----------

def test_strict_rule_does_not_credit_floor_lift():
    new = _d(semantic_score=0.60, n_drift=0, weakest_score=0.50, mean_bm25=4.0)
    old = _d(semantic_score=0.60, n_drift=0, weakest_score=0.40, mean_bm25=4.0)
    # Same inputs that is_evolve_improvement accepts → strict rule rejects.
    assert is_strict_improvement(new, old) is False
    assert is_evolve_improvement(new, old) is True


# ---------- stance selector ----------

def test_slot_zero_is_neutral_baseline():
    name, instruction = stance_for_slot(0)
    assert name == "balanced"
    assert instruction == ""


def test_non_baseline_stances_have_instructions():
    for slot in range(1, len(DRAFT_STANCES)):
        name, instruction = stance_for_slot(slot)
        assert name != "balanced"
        assert instruction.strip()


def test_stances_cycle_past_the_list():
    assert stance_for_slot(len(DRAFT_STANCES)) == stance_for_slot(0)


# ────────────────────────────────────────────────────────────────────
# Phase 1: combined_quality blends fidelity (semantic_score) and salience
# (salience_score) at equal weight. All three lenses key on this combined
# scalar instead of semantic_score alone, so a draft / revision that
# improves salience without changing semantic is correctly credited.
# ────────────────────────────────────────────────────────────────────

from researchwiki.agents.fitness import (
    BM25_MARGIN,
    FLOOR_MARGIN,
    SAL_WEIGHT,
    SEM_WEIGHT,
    SEMANTIC_EPSILON,
    combined_quality,
)


# ---------- combined_quality ----------

def test_combined_both_present_blends_at_configured_weights():
    q = combined_quality({"semantic_score": 0.8, "salience_score": 0.6})
    assert q == SEM_WEIGHT * 0.8 + SAL_WEIGHT * 0.6


def test_combined_falls_back_to_sem_when_sal_missing():
    assert combined_quality({"semantic_score": 0.75, "salience_score": None}) == 0.75


def test_combined_falls_back_to_sal_when_sem_missing():
    assert combined_quality({"semantic_score": None, "salience_score": 0.5}) == 0.5


def test_combined_returns_none_when_both_missing():
    assert combined_quality({"semantic_score": None, "salience_score": None}) is None
    assert combined_quality({}) is None


# ---------- tournament_key with the combined primary ----------

def test_tournament_high_salience_wins_when_combined_higher():
    """Low-sem high-sal beats high-sem low-sal when their combined
    qualities differ by more than the bucketing granularity."""
    high_sal = _d(semantic_score=0.60, salience_score=0.90,
                  coherence_score=0.5, n_drift=0, n_graded=10,
                  mean_bm25=20.0, weakest_score=5.0)
    high_sem = _d(semantic_score=0.80, salience_score=0.40,
                  coherence_score=0.5, n_drift=0, n_graded=10,
                  mean_bm25=20.0, weakest_score=5.0)
    # combined: 0.5*0.60 + 0.5*0.90 = 0.75 vs. 0.5*0.80 + 0.5*0.40 = 0.60.
    # 0.75 bucket > 0.60 bucket → high_sal wins.
    drafts = sorted([high_sem, high_sal], key=tournament_key, reverse=True)
    assert drafts[0] is high_sal


def test_tournament_coherence_breaks_combined_tie():
    a = _d(semantic_score=0.80, salience_score=0.60,
           coherence_score=0.95, n_drift=0, n_graded=10,
           mean_bm25=20.0, weakest_score=5.0)
    b = _d(semantic_score=0.80, salience_score=0.60,
           coherence_score=0.50, n_drift=0, n_graded=10,
           mean_bm25=20.0, weakest_score=5.0)
    drafts = sorted([b, a], key=tournament_key, reverse=True)
    assert drafts[0] is a


# ---------- is_evolve_improvement: the Phase-1 regression target ----------

def test_evolve_accepts_salience_only_gain():
    """The load-bearing test for Phase 1: a revision that lifts salience
    without changing semantic IS accepted. Pre-Phase-1 the rule keyed on
    semantic_score alone and discarded these gains."""
    prior = _d(semantic_score=0.80, salience_score=0.50,
               n_drift=0, weakest_score=5.0, mean_bm25=20.0)
    revised = _d(semantic_score=0.80, salience_score=0.70,
                 n_drift=0, weakest_score=5.0, mean_bm25=20.0)
    # combined: 0.65 → 0.75, gain 0.10 > EPSILON 0.01.
    assert is_evolve_improvement(revised, prior) is True


def test_evolve_rejects_salience_only_drop():
    prior = _d(semantic_score=0.80, salience_score=0.70,
               n_drift=0, weakest_score=5.0, mean_bm25=20.0)
    revised = _d(semantic_score=0.80, salience_score=0.50,
                 n_drift=0, weakest_score=5.0, mean_bm25=20.0)
    assert is_evolve_improvement(revised, prior) is False


def test_evolve_floor_lift_still_credited_when_combined_tied():
    """When the combined primary is tied and drift is tied, evolve still
    rewards a weakest-score floor lift — depth-of-fix signal preserved."""
    prior = _d(semantic_score=0.80, salience_score=0.60,
               n_drift=0, weakest_score=3.0, mean_bm25=20.0)
    revised = _d(semantic_score=0.80, salience_score=0.60,
                 n_drift=0, weakest_score=3.0 + FLOOR_MARGIN + 0.01,
                 mean_bm25=20.0)
    assert is_evolve_improvement(revised, prior) is True


# ---------- is_strict_improvement: salience credited too ----------

def test_strict_accepts_salience_only_gain():
    prior = _d(semantic_score=0.80, salience_score=0.50,
               n_drift=0, mean_bm25=20.0)
    revised = _d(semantic_score=0.80, salience_score=0.70,
                 n_drift=0, mean_bm25=20.0)
    assert is_strict_improvement(revised, prior) is True


def test_strict_combined_margin_below_epsilon_falls_through():
    """Gains within EPSILON don't trigger acceptance on the primary
    signal — they fall through to the drift / bm25 tail."""
    prior = _d(semantic_score=0.80, salience_score=0.60,
               n_drift=0, mean_bm25=20.0)
    revised = _d(semantic_score=0.805, salience_score=0.605,
                 n_drift=0, mean_bm25=20.0)
    # combined gain 0.005 < EPSILON 0.01; bm25 tied → no improvement.
    assert is_strict_improvement(revised, prior) is False
