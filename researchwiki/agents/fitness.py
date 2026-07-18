"""Per-operator fitness — the lens each operator's output is judged by.

The pipeline used to score every draft with one global rule regardless of which
operator produced it, even though the operators have different jobs and failure
modes. This module makes the lenses explicit:

  - `tournament_key`        — AUTHOR / selection. Rewards *coverage breadth*:
                              among drafts of equal combined quality, prefer
                              the one that grounds more of the paper's claims
                              and conforms structurally.
  - `is_evolve_improvement` — EVOLVE. Rewards *depth*: accept a revision when it
                              lifts the weakest-supported claim (the critic's
                              target) even if the page-mean is flat. The old
                              global rule discarded exactly these wins.
  - `is_strict_improvement` — DEBUG / generic. The conservative rule: a repair
                              must improve the primary signal (or cut drift) —
                              it isn't credited for raising the floor while the
                              mean slips.

All three lenses now read a **combined-quality** primary signal —
`combined_quality()` — that blends fidelity (`semantic_score`) and
salience-recall (`salience_score`) at equal weight (0.5/0.5). This replaces
`semantic_score` as the primary signal so a revision that improves PDF-anchor
recall without changing paraphrase fidelity is correctly credited as an
improvement; the prior single-axis rule discarded exactly those gains.

Aggregate keys consumed (from `phases.grade()`):
  semantic_score, salience_score, n_anchors, mean_bm25, n_graded, n_drift,
  weakest_score, coherence_score.

Thresholds (combined ε = 0.01, floor margin = 0.03, BM25 margin = 0.5) are
the same magnitudes the prior single rule used; behavior on clearly-separated
drafts is unchanged. Salience and coherence weights (`SAL_WEIGHT`, `SEM_WEIGHT`)
are first-cut and tunable — calibrate after observing real ingests.
"""

from __future__ import annotations

# Combined-quality margin to count as a real improvement. Same magnitude as
# the prior sem-only EPSILON; combined is the average of two 0..1 quantities,
# so 0.01 still corresponds to "noticeable" rather than "noise."
SEMANTIC_EPSILON = 0.01
FLOOR_MARGIN = 0.03
BM25_MARGIN = 0.5

# Default 0.5/0.5: salience and fidelity are coequal axes (one says "what you
# wrote is in the PDF," the other "what's in the PDF made it onto the page").
# Provisional — adjust after we see how the two signals correlate on real
# ingests. Tuning here changes all three fitness lenses simultaneously.
SAL_WEIGHT = 0.5
SEM_WEIGHT = 0.5


def combined_quality(scores: dict) -> float | None:
    """Primary fitness signal: equal-weighted blend of fidelity and salience.

    Returns None when neither axis is available; gracefully falls back when
    one is missing:

      both present  → SEM_WEIGHT*sem + SAL_WEIGHT*sal
      sem only      → sem    (no salience anchors recoverable from the PDF,
                              or `--no-salience` was set)
      sal only      → sal    (semantic disabled / model unavailable)
      neither       → None   (caller picks a fallback ordering)
    """
    sem = scores.get("semantic_score")
    sal = scores.get("salience_score")
    if sem is not None and sal is not None:
        return SEM_WEIGHT * sem + SAL_WEIGHT * sal
    if sem is not None:
        return sem
    if sal is not None:
        return sal
    return None


def tournament_key(draft) -> tuple[float, float, float, float, float, float]:
    """Author/selection fitness — argmax key for the tournament.

    Ordering (descending): combined quality (bucketed) → coherence → numeric
    integrity → coverage breadth → lexical match → weakest claim.

    Bucketing the combined score to 2 decimals (~0.01 granularity, matching
    SEMANTIC_EPSILON) lets coherence and coverage break a near-tie. Stuffing
    low-quality claims still can't game the primary because extra weak claims
    drag both `semantic_score` and `salience_score` down.
    """
    s = draft.scores
    q = combined_quality(s)
    q_bucket = round(q, 2) if q is not None else 0.0
    coh = s.get("coherence_score") or 0.0
    drift = s.get("n_drift") or 0
    n_graded = s.get("n_graded") or 0
    return (
        q_bucket,
        coh,
        -float(drift),
        float(n_graded),
        s.get("mean_bm25") or 0.0,
        s.get("weakest_score") or 0.0,
    )


def is_strict_improvement(new_draft, prior_winner) -> bool:
    """Generic / DEBUG fitness: replace the winner only on a strict gain on the
    primary signal. Combined quality first, then a drift veto, then a BM25
    margin. Avoids cycles where a revision is different-but-not-better.
    """
    a, b = new_draft.scores, prior_winner.scores
    qa, qb = combined_quality(a), combined_quality(b)
    if qa is not None and qb is not None:
        if qa > qb + SEMANTIC_EPSILON:
            return True
        if qa < qb - SEMANTIC_EPSILON:
            return False
    a_drift = a.get("n_drift") or 0
    b_drift = b.get("n_drift") or 0
    if a_drift < b_drift:
        return True
    if a_drift > b_drift:
        return False
    return (a.get("mean_bm25") or 0) > (b.get("mean_bm25") or 0) + BM25_MARGIN


def is_evolve_improvement(new_draft, prior_winner) -> bool:
    """EVOLVE fitness: like the strict rule, but also accepts a revision that
    lifts the *weakest* claim when the primary signal is tied.

    Evolve exists to fix the specific weak claims the critic flagged. When it
    rewrites one weak claim, the page-level combined score can stay flat (one
    claim out of many) while the floor rises — the strict rule discards that,
    defeating the loop's purpose. We credit a floor gain (> FLOOR_MARGIN) only
    after the primary is a tie and drift hasn't worsened, so a revision can
    never be accepted while making the primary or drift worse.
    """
    a, b = new_draft.scores, prior_winner.scores
    qa, qb = combined_quality(a), combined_quality(b)
    if qa is not None and qb is not None:
        if qa > qb + SEMANTIC_EPSILON:
            return True
        if qa < qb - SEMANTIC_EPSILON:
            return False
    # Primary is tied — drift veto still applies (never accept more drift).
    a_drift = a.get("n_drift") or 0
    b_drift = b.get("n_drift") or 0
    if a_drift < b_drift:
        return True
    if a_drift > b_drift:
        return False
    # Depth: reward lifting the weakest-supported claim.
    a_floor, b_floor = a.get("weakest_score"), b.get("weakest_score")
    if a_floor is not None and b_floor is not None:
        if a_floor > b_floor + FLOOR_MARGIN:
            return True
        if a_floor < b_floor - FLOOR_MARGIN:
            return False
    return (a.get("mean_bm25") or 0) > (b.get("mean_bm25") or 0) + BM25_MARGIN
