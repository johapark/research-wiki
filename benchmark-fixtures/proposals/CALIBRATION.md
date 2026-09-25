# Agent calibration — 2026-09-14

Status: **agent-calibrated, not human-validated**. Rubric 2, suite
`proposal-quality-v2`. The user explicitly delegated calibration to the active
chat agent. No additional API calls, generator tuning, wiki edits, or held-out
model runs were made. This is unblinded self-review, not independent judging or
measured agreement with the user's preferences.

## Material reviewed

All twelve questions and their evaluator criteria were audited. Five saved
`output/proposal-evaluation/*-contract-fix.json` receipts were inspected against
their **own original requests, evidence packets, existing-page summaries, and
feedback**, not retroactively against the new fixtures. Their four proposals and
one empty response were rechecked with the current structural validator; all
passed. Load-bearing evidence was checked through the supplied claim anchors;
the frozen PDF text was also consulted for the retrieval comparison and the
higher-order representation premise.

Detailed scores, reasons, source-receipt hashes, and evidence references are in
the local `output/proposal-calibration-20260914/reviews.json`. Original receipts
remain unchanged. This record is not a complete development split, so it must
not be fed to the suite scorer as if it were one. There are no current-suite
outputs for X02, R02, or F02, and the earlier five packets differ from this suite.

## Question audit

| Case | Decision | Reason / scoring boundary |
|---|---|---|
| S01 | Reduce answer hints | Ask for a consequential retrieval distinction without giving the three representation labels; credit the response's explanatory consequence, not the requested comparison itself. |
| I01 | Retain; tighten contribution check | Existing-page context makes novelty assessable. Require a specific additional decision or experiment, not a familiar architecture with a new title. |
| X01 | Retain; tighten baseline check | A source-to-target mapping is a legitimate requirement. Review-level evidence warrants an exploratory test, not assumed superiority. Require an advantage worth the complexity. |
| X02 | Retain identical request | Only the supplied source-method evidence changes. Abstention is about this packet's coverage, not whether transfer is impossible in principle. |
| R01 | Remove room-temperature qualifier | The intended test is irrelevant evidence, not recognition of an extraordinary physical premise from model knowledge. Still an easy restraint control. |
| R02 | Retain closed scope | Tests respecting an already-covered request; a narrower idea is disallowed here, unlike in I01 and the historical covered receipt. |
| F01 | Retain request; complete feedback checklist | Explicitly check material narrowing and separate, dimensionally meaningful cost accounting, in addition to baseline and rejection logic. |
| F02 | Retain; clarify retention boundary | Bounded correction is the requested constraint, not an answer. Ask what happens at the retention limit; do not reward merely naming a cap. |
| S02 | Retain | The substrate question intentionally sets an analytical axis. Credit an interpretation or boundary derived from evidence, not repetition of that axis. |
| S03 | Reduce answer hints | Remove the method distinction from the question so the response must derive it from supplied evidence. |
| I02 | Retain | Grid/palette requirements define the user's problem without prescribing an optimization. No credit for importing an unrelated speedup. |
| I03 | Reduce answer hints | Remove the already-specified two-factor experiment. Keep alternative supported failure-isolation tests admissible. |

The nine opportunity / three abstention labels remain unchanged. Holdout evidence
and criteria were curator-inspected, as before; no holdout responses were generated
or used to choose the policy. This suite measures **question-conditioned ideation**,
not open-ended discovery or general creativity. Development cases share one source
group; heldout still lacks restraint and feedback cases. Do not overstate coverage.

## Saved-output judgments

Scores below are in evidence / insight / contribution / decision value /
proportionality order. They are editorial judgments, not empirical truth.

| Saved scenario | Scores | Verdict | Decisive reason |
|---|---|---|---|
| mature: retrieval synthesis | 1 / 2 / 2 / 1 / 2 | Shortlist for scoping | The evidence-organization distinction contributes beyond a catalog. Drafting must keep design-fit advice conditional; a common comparative benchmark is explicitly absent. |
| covered: event ledger | 2 / 2 / 1 / 2 / 1 | Defer | The contrast with the existing survey is clear, but the delta from the supplied receipt/correction idea contexts is not. Resolve retention/immutability semantics and demonstrate that delta before building another architecture. |
| transfer: relationship sidecar | 2 / 2 / 2 / 1 / 1 | Defer | The operation mapping is concrete. However, its first experiment can accept parity with pairwise links without establishing an offsetting benefit, and adds genotyping before isolating the representation question. |
| rejected: allocation experiment | 2 / 2 / 2 / 0 / 1 | Defer | A substantive narrowed revision, but budget B combines operations, tokens, latency, and amortized costs without a common unit. Its rejection rule conflates lack of detectable interaction with ruling out a useful effect. |
| weak: unsupported fabrication topic | N/A | Appropriate abstention | The original packet has no fabrication/coherence evidence. Empty output passes validation; it receives no invented proposal-axis scores. |

No critical factual flag was assigned to these four proposals in this review.
That does not certify every sentence: the main identified blockers are editorial
or experimental-design defects. Structural validity is not usefulness. The
historical covered request did **not** prohibit narrower ideas; applying R02's
new closed-scope restriction to it would be an evaluator error. Historical
feedback is likewise judged only against the instructions actually supplied.

## Calibrated policy

- Judge the proposal as submitted; do not silently repair it before scoring.
- Name the response's contribution beyond both the prompt and existing pages.
- A shortlist authorizes a concrete scoping action, not automatic page creation.
- A real baseline must challenge the added mechanism. Equal performance alone
  does not justify added complexity unless another worthwhile benefit is shown.
- Rejection criteria must be interpretable; resource constraints need compatible
  units or separate caps. Distinguish inconclusive tests from negative results.
- Keep experimental flaws separate from unsupported factual assertions.
- Preserve reviewer identity and kind in scored output. Agent utility is never
  relabeled human utility, and repeated self-review is not inter-rater agreement.

The 8/10 threshold is **unchanged and remains diagnostic**. Both deferred
8/10 proposals would pass it; increasing the threshold would merely fit this
small convenience sample. Direct reviewer decisions remain primary. The two
boundary checks below are constructed rubric examples, not observed generations:

1. Replace the retrieval proposal's caveat with a claim of demonstrated
   head-to-head superiority without adding evidence: evidence becomes 0, the
   critical factual flag is true, and the decision becomes reject regardless of
   polish or other scores.
2. Repeat only the comparison requested in a prompt without adding an explanatory
   distinction: insight is 0. Rewording the prompt is not generated insight.

Arithmetic regression tests cover deferred high totals, critical failures that
cannot be compensated by other axes, empty outputs, and missing reviewer provenance.
They test implementation, not the correctness of subjective judgments.

## Completion and limits

Initial agent calibration is complete: every case has a disposition; the rubric
has explicit decision boundaries; each saved response has an evidence-referenced
judgment; reviewer provenance is retained; and old packets remain unchanged.
The rule can now support a development comparison. **Benchmark performance,
agreement with human preferences, and independent judge reliability remain
unmeasured.** Future disagreements may justify another version; they must not
be resolved by silently moving the current holdout criteria.
