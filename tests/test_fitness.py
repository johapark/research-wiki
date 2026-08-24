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


# ────────────────────────────────────────────────────────────────────
# Phase 2: the blend must not be less informative than either axis alone.
#
# Rounding the *output* of a weighted average halves any single-axis delta
# before bucketing, so with salience flat between drafts — the common case:
# median within-paper spread 0.017, and 46% of papers had every draft within
# 0.01 — a semantic difference had to exceed 0.02 to reach the key at all.
# Replaying 127 historical tournaments found 9 where semantic separated the
# drafts but the combined bucket tied. Quantizing the axes *before* blending
# restored 7 of those.
# ────────────────────────────────────────────────────────────────────

from researchwiki.agents.fitness import (
    ANCHOR_CONFIDENCE_FULL,
    TARGET_CONFIDENCE_FULL,
    TARGET_WEIGHT,
    salience_confidence,
    target_claim_confidence,
)


def test_quantized_blend_keeps_semantic_separation_when_salience_is_flat():
    """The core regression. Under the old key both drafts blended to 0.60 and
    the winner was decided by coherence; the 0.01 fidelity edge was discarded
    even though salience contributed nothing to the comparison."""
    better = _d(semantic_score=0.51, salience_score=0.70, n_anchors=20,
                coherence_score=0.10, n_drift=0, n_graded=10,
                mean_bm25=20.0, weakest_score=5.0)
    worse = _d(semantic_score=0.50, salience_score=0.70, n_anchors=20,
               # Higher coherence: under the old key this won the tie.
               coherence_score=0.90, n_drift=0, n_graded=10,
               mean_bm25=20.0, weakest_score=5.0)
    assert round(0.5 * 0.51 + 0.5 * 0.70, 2) == round(0.5 * 0.50 + 0.5 * 0.70, 2), \
        "premise: the old output-rounded blend tied these two"
    assert tournament_key(better) > tournament_key(worse)


def test_axes_within_one_bucket_still_tie_so_coherence_decides():
    """The complement: quantizing must not turn every float wobble into a
    separation, or coherence and coverage never get to break a real near-tie."""
    a = _d(semantic_score=0.802, salience_score=0.601, n_anchors=20,
           coherence_score=0.95, n_drift=0, n_graded=10,
           mean_bm25=20.0, weakest_score=5.0)
    b = _d(semantic_score=0.799, salience_score=0.604, n_anchors=20,
           coherence_score=0.10, n_drift=0, n_graded=10,
           mean_bm25=20.0, weakest_score=5.0)
    assert tournament_key(a)[0] == tournament_key(b)[0]
    assert tournament_key(a) > tournament_key(b)   # coherence breaks it


# ---------- salience confidence weighting ----------

def test_thin_denominator_dilutes_salience_toward_fidelity():
    """A salience score over 2 anchors shouldn't swing selection as hard as one
    over 40. Same scores, different denominators → different blends."""
    thin = {"semantic_score": 0.80, "salience_score": 0.20, "n_anchors": 2}
    thick = {"semantic_score": 0.80, "salience_score": 0.20, "n_anchors": 40}
    assert combined_quality(thin) > combined_quality(thick)
    # ...and the thin one sits nearer pure fidelity.
    assert abs(combined_quality(thin) - 0.80) < abs(combined_quality(thick) - 0.80)


def test_full_confidence_at_threshold_is_the_parity_blend():
    at = {"semantic_score": 0.80, "salience_score": 0.60,
          "n_anchors": ANCHOR_CONFIDENCE_FULL}
    assert combined_quality(at) == SEM_WEIGHT * 0.80 + SAL_WEIGHT * 0.60


def test_missing_n_anchors_means_unknown_not_zero():
    """Back-compat: a score dict without `n_anchors` (pre-dating the key, or
    hand-built) must keep full salience weight rather than silently collapsing
    to fidelity-only."""
    assert combined_quality({"semantic_score": 0.80, "salience_score": 0.60}) == \
        SEM_WEIGHT * 0.80 + SAL_WEIGHT * 0.60
    assert salience_confidence({}) == 1.0


def test_salience_confidence_is_clamped():
    assert salience_confidence({"n_anchors": 0}) == 0.0
    assert salience_confidence({"n_anchors": 1000}) == 1.0
    assert salience_confidence({"n_anchors": 5}) == 0.5


def test_zero_anchor_confidence_falls_back_to_fidelity():
    """conf 0 zeroes the salience weight; the blend must not divide by zero."""
    q = combined_quality({"semantic_score": 0.80, "salience_score": 0.20,
                          "n_anchors": 0})
    assert q == 0.80


# ---------- importance-weighted target-claim coverage ----------

def test_target_claim_axis_joins_primary_before_coherence():
    covered = _d(
        semantic_score=0.70, salience_score=0.70, n_anchors=20,
        target_claim_score=0.90, n_target_claims=12,
        coherence_score=0.10,
    )
    omitted = _d(
        semantic_score=0.70, salience_score=0.70, n_anchors=20,
        target_claim_score=0.40, n_target_claims=12,
        coherence_score=0.99,
    )
    assert tournament_key(covered) > tournament_key(omitted)


def test_three_confident_axes_receive_equal_base_weight():
    scores = {
        "semantic_score": 0.9,
        "salience_score": 0.6,
        "n_anchors": ANCHOR_CONFIDENCE_FULL,
        "target_claim_score": 0.3,
        "n_target_claims": TARGET_CONFIDENCE_FULL,
    }
    expected = (
        SEM_WEIGHT * 0.9 + SAL_WEIGHT * 0.6 + TARGET_WEIGHT * 0.3
    ) / (SEM_WEIGHT + SAL_WEIGHT + TARGET_WEIGHT)
    assert combined_quality(scores) == expected


def test_thin_target_claim_set_is_confidence_diluted():
    thin = {
        "semantic_score": 0.8,
        "target_claim_score": 0.2,
        "n_target_claims": 1,
    }
    thick = {**thin, "n_target_claims": TARGET_CONFIDENCE_FULL}
    assert combined_quality(thin) > combined_quality(thick)
    assert target_claim_confidence(thin) == 1 / TARGET_CONFIDENCE_FULL


def test_missing_target_axis_preserves_previous_two_axis_result():
    scores = {
        "semantic_score": 0.8,
        "salience_score": 0.6,
        "n_anchors": ANCHOR_CONFIDENCE_FULL,
    }
    assert combined_quality(scores) == SEM_WEIGHT * 0.8 + SAL_WEIGHT * 0.6
