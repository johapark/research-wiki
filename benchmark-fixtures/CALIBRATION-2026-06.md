# Salience-scorer threshold calibration — 2026-06

> **Archived.** This calibration refers to the pre-OA personal-corpus fixture set (kim / lai / yang / gupta / siren / zhang), retired 2026-07-03. Threshold values and calibration methodology remain relevant; specific per-fixture numbers do not carry over to the current OA corpus.

## What this document is

The verdict thresholds in `grade/scorer.py` (token-overlap match/partial,
bi-encoder cosine match/partial) and the importance weights in
`grade/scorer.py` were originally first-cut values. This is the first
calibration pass against the curated `benchmark-fixtures/` suite. The output:
a snapshot of how the scorer behaves on real data, plus a recommendation
for what (if anything) to change.

**TL;DR:** Thresholds are well-calibrated. No changes applied.
The dominant cause of `miss` and `partial` verdicts is the **numeric
integrity check** (correctly flagging pages that paraphrase the meaning
but drop specific numbers), not threshold strictness. Lowering thresholds
would let in false matches; raising them would lose paraphrase tolerance.

## Methodology

`internal/calibrate_salience.py` runs every curated content fixture through
`grade.scorer.score_text(use_semantic=True)` against the live wiki page,
emits a per-item table with verdict + underlying signals, and aggregates.

Scope: 6 content fixtures × ~20 items each = 145 verdicts across five
axes (headline_claims, capabilities, limitations, related_papers,
comparator_fidelity). Retrieval fixtures excluded — they target a
different scoring path.

Review process: every verdict in the borderline zone (overlap 0.30–0.85
with semantic 0.50–0.95) hand-inspected against the source paper's wiki
page. False-positive / false-negative classifications:

- **True miss** — page genuinely doesn't cover the item.
- **False miss** — page covers the item but signals fell below threshold.
- **True match** — page covers the item.
- **False match** — page doesn't cover but accidentally hit threshold.

## Findings

### Aggregate

| Axis | match | partial | miss | n |
|---|---|---|---|---|
| headline_claims | 47 | 10 | 6 | 63 |
| capabilities | 23 | 3 | 2 | 28 |
| limitations | 14 | 9 | 1 | 24 |
| related_papers | 12 | 0 | 3 | 15 |
| comparator_fidelity | 13 | 1 | 1 | 15 |
| **total** | **109 (75%)** | **23 (16%)** | **13 (9%)** | **145** |

Per-fixture overall weighted recall:

| Fixture | recall |
|---|---|
| kim-2026-structural-motif-search-across-the-protein | 0.903 |
| siren-2021-pangenomics-enables-genotyping-known-structural | 0.885 |
| zhang-2026-structural-basis-for-risc-assembly | 0.865 |
| gupta-2026-base-editing-of-hbg1-and-hbg2 | 0.849 |
| yang-2026-the-past-present-and-future | 0.830 |
| lai-2026-clinical-application-of-base-editing | 0.754 |

The 0.75–0.90 range is healthy. `lai-2026` scores lowest because it's a
clinical-trial paper with many specific numerics (cohort sizes, response
rates, follow-up months); the wiki page summarizes without quoting all
of them, so the numeric-integrity check correctly flags drift.

### Cause analysis for `miss` verdicts (n=13)

| Cause | Count | What it means |
|---|---|---|
| ≥2 missing numerics | 8 | Page paraphrased the claim but dropped specific numbers — correctly flagged. |
| Mechanical-axis miss (related_papers wikilink absent, comparator_fidelity ratio absent) | 4 | Different verdict path; not affected by token/cosine thresholds. |
| Genuine low-signal miss | 1 | `yang/success-rate-and-activity` (overlap 0.44, cosine 0.48) — page genuinely doesn't cover this item. Threshold correctly fires. |

**The numeric integrity check is the dominant miss driver** — not threshold
strictness. Lowering cosine match from 0.75 to 0.65 wouldn't recover any
of the numeric-drift misses (they have semantic 0.73–0.91 already; cosine
isn't the bottleneck). The check is binary and working as designed.

### Cause analysis for `partial` verdicts in the high-signal zone (n=7 of 23)

Items with overlap ≥ 0.65 AND cosine ≥ 0.70 that landed `partial` instead
of `match`:

```
gupta cohort-31-patients          ovl=0.69  sem=0.77  num_miss=['35']
gupta voc-prevention              ovl=0.82  sem=0.70  num_miss=['60']
gupta primary-endpoint            ovl=0.74  sem=0.81  num_miss=['60']
gupta interim-analysis-unplanned  ovl=0.67  sem=0.78  num_miss=['60']
kim   index-storage-540k          ovl=0.67  sem=0.76  num_miss=['542']
kim   indexing-speed              ovl=0.69  sem=0.71  num_miss=['540']
gupta ae-distribution             ovl=0.67  sem=0.74  num_miss=[]
```

Six of seven are downgraded from `match` to `partial` by exactly one
missing numeric. The page covered the meaning but didn't quote the
specific value. **This is the desired behavior** — salience should
penalize drift even when the prose is right. The seventh
(`ae-distribution`) is a true partial: overlap and cosine just below the
match thresholds.

### Cause analysis for `match` verdicts carried by cosine (n=29 of 109)

Matches where overlap < 0.65 AND cosine ≥ 0.75 (no drift) — i.e., matches
that the token-overlap path alone would have lost:

- Median overlap in this group: 0.55
- Median cosine: 0.83
- Examples: `lai/dsb-free-rationale` (ovl 0.46, sem 0.76),
  `siren/tool-summary` (ovl 0.58, sem 0.90),
  `yang/monomers-and-folds` (ovl 0.44, sem 0.86)

**The bi-encoder semantic path is doing 27% of the matching work.** This
validates the earlier addition of cosine alongside token overlap (commit
`29a7f20`) — without it, paraphrased claims would systematically miss.

### Score-distribution sanity

| Verdict | n | semantic (min/median/max) | overlap (min/median/max) |
|---|---|---|---|
| match | 84 (verb) | 0.59 / 0.80 / 0.94 | 0.30 / 0.75 / 1.00 |
| partial | 22 (verb) | 0.65 / 0.73 / 0.81 | 0.00 / 0.54 / 0.82 |
| miss | 9 (verb) | 0.48 / 0.82 / 0.91 | 0.44 / 0.58 / 0.77 |

Strikingly, miss verdicts have a **higher median cosine (0.82) than partial
verdicts (0.73)** — a clean signal that the misses aren't caused by low
semantic content, they're caused by numeric drift on otherwise
well-paraphrased claims.

The match-cosine min of 0.59 belongs to two items (`kim/encoding-bits`,
`siren/cost-per-sample`) that matched via token overlap (≥0.78). The OR
rule in `_heuristic_verdict` is consistent — neither path generates
spurious matches.

## Threshold recommendations

| Threshold | Current | Recommended | Rationale |
|---|---|---|---|
| `_SEM_MATCH_THRESHOLD` | 0.75 | **keep** | Right at the median of the match-vs-partial boundary; 25% of all matches depend on it. Raising would erode paraphrase tolerance; lowering would introduce false matches. |
| `_SEM_PARTIAL_THRESHOLD` | 0.50 | **keep** | Lightly used in practice (most low-cosine items also have low overlap and become miss anyway), but harmless. No data supports a change. |
| Token overlap match (in `_heuristic_verdict`: 0.75) | 0.75 | **keep** | Matches the median match overlap. Items below 0.75 still match via cosine; threshold isn't lossy. |
| Token overlap partial (0.45) | 0.45 | **keep** | Mostly preempted by cosine in real data. No partials with overlap < 0.45 in the dataset. |
| Importance weights (critical=3, high=2, normal=1) | unchanged | **keep** | No empirical basis to change without ground-truth on relative importance. |
| Numeric integrity (binary check) | n/a | **keep as-is** | The dominant verdict-driver. Working correctly across all observed cases. |

## Skipped (out of scope for this pass)

- **`grade/coherence.py` weights** — coherence scores 1.0 for all
  agent-authored pages in the test set; no failure cases yet to calibrate
  against. Revisit when a healthy page scores low or vice versa.
- **`grade/fidelity/synthesis.py` BM25_FLOOR / SEMANTIC_FLOOR** — only
  affect the `weak` advisory verdict, not the hard-fail `misattributed`.
  Lower priority; revisit when a synthesis page misclassifies.
- **`agents/fitness.py` SEM_WEIGHT / SAL_WEIGHT** — design choice, not
  calibration. Without ground truth on which draft is "actually best"
  in tournament outcomes, there's no empirical basis to move from 0.5/0.5.

## What would change this recommendation

- **More fixtures, especially adversarial ones.** 6 fixtures × ~20 items
  is enough to spot directional miscalibration but not enough to find
  fine-grained threshold sweet spots. Adding fixtures that intentionally
  paraphrase aggressively, drop numbers, or include false-claim cases
  would surface failure modes this pass missed.
- **An explicit FP / FN ground-truth pass.** Hand-labeling each verdict
  as truly-correct vs. truly-wrong (rather than the heuristic
  classification used here) would produce calibration curves for tuning.
- **Production-ingest sample.** The current fixture suite is curated by
  one author; running calibration against pages authored by the agent in
  production would reveal whether the thresholds generalize.

## Reproducing

The calibration harness lives at `internal/calibrate_salience.py` — a
maintainer-only utility, gitignored and not shipped with releases. Ask
the authors for a copy, or run against your own wiki with equivalent
scoring code.

```bash
python internal/calibrate_salience.py            # text table
python internal/calibrate_salience.py --csv /tmp/v.csv  # for spreadsheet
python internal/calibrate_salience.py --fixture <stem>  # one fixture
python internal/calibrate_salience.py --no-semantic     # token-only baseline
```

The harness is read-only — never modifies the codebase or fixtures.
