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
`combined_quality()` — that blends fidelity (`semantic_score`), structural
PDF-anchor recall (`salience_score`), and importance-weighted extracted-claim
recall (`target_claim_score`). The target axis is absent on legacy or failed
extractions, preserving the prior fidelity/salience behavior exactly.

Aggregate keys consumed (from `phases.grade_draft()`):
  semantic_score, salience_score, n_anchors, target_claim_score,
  n_target_claims, mean_bm25, n_graded, n_drift, weakest_score,
  coherence_score.

Thresholds (combined ε = 0.01, floor margin = 0.03, BM25 margin = 0.5) are
the same magnitudes the prior single rule used; behavior on clearly-separated
drafts is unchanged. Axis weights (`SEM_WEIGHT`, `SAL_WEIGHT`,
`TARGET_WEIGHT`) are first-cut and tunable — calibrate after observing real
ingests with target-score history.
"""

from __future__ import annotations

# Combined-quality margin to count as a real improvement. Same magnitude as
# the prior sem-only EPSILON; combined is an average of 0..1 quantities,
# so 0.01 still corresponds to "noticeable" rather than "noise."
SEMANTIC_EPSILON = 0.01
FLOOR_MARGIN = 0.03
BM25_MARGIN = 0.5

# Equal base weights: fidelity, structural salience, and triaged target claims
# are coequal axes when all have confident denominators. The target score is
# already importance-weighted critical=3/high=2/normal=1 by grade.scorer.
# With no target extraction, this reduces exactly to the prior 0.5/0.5 blend.
# Salience and fidelity say respectively "what's in the PDF made it onto the
# page" and "what you wrote is in the PDF"; target claims add "the
# load-bearing, paper-wide items made it onto the page."
# Measured on 533 historical drafts the two are essentially independent
# (Pearson +0.034), which is what justifies carrying both rather than folding
# them together. Kept at parity deliberately: nothing in the corpus data
# argues for a specific other split, so the number stays defensible rather than
# tuned to a sample. Tuning here changes all three fitness lenses at once.
SAL_WEIGHT = 0.5
SEM_WEIGHT = 0.5
TARGET_WEIGHT = 0.5

# Anchor count at which salience earns its full weight. Below it the score is
# a ratio over too small a denominator to trust — 9% of historical drafts had
# fewer than 10 anchors, 2% fewer than 5, and one draft's entire salience
# verdict rested on a single anchor. Salience is then down-weighted in
# proportion and the blend renormalized, so a thin recall estimate dilutes
# toward fidelity instead of swinging selection at full strength.
ANCHOR_CONFIDENCE_FULL = 10
TARGET_CONFIDENCE_FULL = 5
SOURCE_CONFIDENCE_FULL = 2


def salience_confidence(scores: dict) -> float:
    """Weight multiplier from anchor count and structural-source diversity.

    A *missing* `n_anchors` means "unknown", not "zero": callers that predate
    the key (and hand-built score dicts in tests) get full confidence, so
    behavior only changes where the grader actually reported a thin
    denominator. `phases.grade_draft` always writes `n_anchors` alongside
    `salience_score`.
    """
    n = scores.get("n_anchors")
    if n is None:
        return 1.0
    count_confidence = min(1.0, max(0.0, float(n) / ANCHOR_CONFIDENCE_FULL))
    sources = scores.get("n_anchor_sources")
    if sources is None:
        return count_confidence
    source_confidence = min(
        1.0, max(0.0, float(sources) / SOURCE_CONFIDENCE_FULL)
    )
    return count_confidence * source_confidence


def target_claim_confidence(scores: dict) -> float:
    """Weight multiplier from valid-claim count and location diversity.

    Five claims is enough for full confidence because the extraction prompt
    reserves roughly 3–6 critical claims for a paper's load-bearing core. As
    with salience, a missing count means unknown/legacy and retains full weight.
    """
    n = scores.get("n_target_claims")
    if n is None:
        return 1.0
    count_confidence = min(1.0, max(0.0, float(n) / TARGET_CONFIDENCE_FULL))
    sources = scores.get("n_target_claim_sources")
    if sources is None:
        return count_confidence
    source_confidence = min(
        1.0, max(0.0, float(sources) / SOURCE_CONFIDENCE_FULL)
    )
    return count_confidence * source_confidence


def combined_quality(scores: dict) -> float | None:
    """Confidence-weighted blend of fidelity and two coverage signals.

    Returns None when no axis is available; gracefully falls back when some
    are missing:

      all present   → normalized weighted blend, with denominator confidence
      one present   → that score
      none present  → None (caller picks a fallback ordering)

    Note for callers comparing against SEMANTIC_EPSILON: this is a weighted
    average, so a gain of d on one axis alone moves the result by w*d — at
    two-axis parity, an 0.01 single-axis gain registers as 0.005 (and less with
    three axes) and does NOT clear the epsilon. `tournament_key` sidesteps that
    by quantizing the axes
    before blending; the improvement rules keep the continuous blend, so they
    require roughly a 0.02 single-axis move. `researchwiki insights --lineage`
    now reconstructs revision pairs through author-parent and grade-author
    edges, providing the evidence needed for future epsilon calibration.
    """
    components = []
    sem = scores.get("semantic_score")
    sal = scores.get("salience_score")
    target = scores.get("target_claim_score")
    if sem is not None:
        components.append((float(sem), SEM_WEIGHT))
    if sal is not None:
        components.append((float(sal), SAL_WEIGHT * salience_confidence(scores)))
    if target is not None:
        components.append((
            float(target), TARGET_WEIGHT * target_claim_confidence(scores),
        ))
    if not components:
        return None
    total = sum(weight for _, weight in components)
    if total <= 0:
        # A score with an explicitly zero denominator is not trustworthy, but
        # retaining the only available value is a safer fallback than erasing
        # the primary signal entirely.
        return components[0][0]
    return sum(value * weight for value, weight in components) / total


def _quantized_quality(scores: dict) -> float | None:
    """`combined_quality` with each axis rounded to 0.01 *before* blending.

    Selection-only. Rounding the blend's *output* (what `tournament_key` used
    to do) makes the primary less informative than either axis alone: a
    weighted average halves a single-axis delta, so with salience flat — and it
    usually is flat between drafts of the same paper, median spread 0.017, with
    46% of papers showing every draft within 0.01 — a semantic difference had
    to exceed 0.02 to survive the 0.01 bucketing. Replaying 127 historical
    tournaments found 9 where semantic separated the drafts but the combined
    bucket tied, handing the decision to coherence; in 11 of 14 such cases
    salience contributed no signal at all, so the blend was purely discarding
    fidelity resolution.

    Quantizing the inputs instead keeps the 0.01 granularity that lets
    coherence break genuine ties, while a 0.01 move on *either* axis still
    reaches the key. On the same replay this restored a separation in 7 of the
    9 erased tournaments and changed the winner in 8 of 127 (6%) — versus 42%
    if the salience axis were dropped from selection entirely.
    """
    sem = scores.get("semantic_score")
    sal = scores.get("salience_score")
    target = scores.get("target_claim_score")
    quantized = dict(scores)
    if sem is not None:
        quantized["semantic_score"] = round(sem, 2)
    if sal is not None:
        quantized["salience_score"] = round(sal, 2)
    if target is not None:
        quantized["target_claim_score"] = round(target, 2)
    return combined_quality(quantized)


def tournament_key(draft) -> tuple[float, float, float, float, float, float]:
    """Author/selection fitness — argmax key for the tournament.

    Ordering (descending): combined quality (axis-quantized) → coherence →
    numeric integrity → coverage breadth → lexical match → weakest claim.

    The primary blends axes that have each been rounded to 2 decimals (~0.01
    granularity, matching SEMANTIC_EPSILON) rather than rounding the blend —
    see `_quantized_quality` for why that distinction decides real tournaments.
    Drafts genuinely within 0.01 on every available axis still tie here, so coherence and
    coverage break the near-tie. Stuffing low-quality claims can't game the
    primary because extra weak claims drag fidelity and/or coverage down.
    """
    s = draft.scores
    q = _quantized_quality(s)
    q_bucket = q if q is not None else 0.0
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
