# Impact-review held-out evaluation — 2026-10-01

## Question and decision rule

Does the default, model-free ingest impact scan recover at least 95% of
materially relevant paper-to-authored-page updates? A pair is **relevant** only
when the source paper could change an existing claim, comparison axis, caveat,
or proposed design. Shared subject words alone do not count.

## Design

- Froze the scanner and used `researchwiki check-coverage --top-n 15` as an
  independent nomination channel. It yielded 89 **uncited** paper–page pairs
  from 71 source papers and 12 synthesis, idea, or proposal pages. Seven papers
  used during earlier threshold tuning were excluded. One selected idea page
  had zero nominations.
- Labeled all 89 pairs from the wiki pages before viewing their impact scores:
  22 relevant, 62 not relevant, five uncertain. The five uncertain cases were
  excluded from binary metrics. Labels, reasons, source hooks, and file hashes
  are in [the frozen labels](impact-review-2026-10-01-labels.json). This is one
  Codex review, without an independent second adjudicator or PDF-level audit
  of every pair.
- Ran the unmodified `impact_review.scan` on all 71 sources, with receipts
  redirected to temporary storage. [The runner](run_impact_review_holdout.py)
  records every pair's score and candidate decision in
  [the baseline results](impact-review-2026-10-01-results.json). Full-file
  hashes guard against page changes. Its replay restores the single GraphOT
  metadata value corrected after baseline scoring. The
  [replay results](impact-review-2026-10-01-replay.json) match all 89 baseline
  pair measurements exactly.

## Baseline result

| Measure | Result |
|---|---:|
| Relevant pairs retrieved | 21/22 = **95.5% recall** |
| Wilson 95% interval for recall | 78.2–99.2% |
| Irrelevant pairs surfaced | 33/62 |
| Precision within independently nominated, adjudicated pairs | 21/54 = **38.9%** |
| Synthesis relevant pairs | 8/8 found |
| Idea relevant pairs | 9/10 found |
| Proposal relevant pairs | 4/4 found |

The 71 topically enriched sources produced a median of four review candidates
per paper (mean 4.87, maximum 13) across all 39 eligible authored pages. A
separate, seeded random sample of 24 other paper pages produced a median of two
(mean 3.12, maximum eight); source keys and counts are in
[the volume sample](impact-review-2026-10-01-random-volume.json). No wiki page,
index, state database, or live review receipt was changed by either scoring run.

### Miss and diagnosis

The missed pair was Luo 2024's mismatch-and-indel off-target scorer → the
GraphOT idea. Its similarity was 0.8263, below the unconditional 0.83 cutoff;
it shared eight terms but no qualifying phrase. The fallback at 0.80 required a
matching content category. GraphOT carried `category: [ideas]` instead of its
CRISPR content category, so the fallback could not fire. I corrected the page
to `category: [cgt]`; its grounding and fidelity gates pass. A focused scan
with temporary receipts now surfaces Luo → GraphOT. **The frozen baseline stays
21/22**; the fix does not retroactively improve the held-out result.

This exposed a broader metadata dependency: 19 of 39 eligible authored pages
had only a page-type value (`synthesis` or `ideas`) in `category:` before the
GraphOT fix; 18 remain. On those pages, same-category corroboration is
unavailable. The baseline found 20 of its 33 false positives through the high
semantic threshold, 12 through shared phrases, and one through category-plus-
terms. The queue is deliberately sensitive, but these counts show real review
workload and a weak precision signal.

## Interpretation

The **point-estimate** 95% recall target is met on this labeled set. The
evaluation does **not** establish population recall of at least 95%: only 22
positive pairs were labeled, the Wilson lower bound is 78.2%, and the
independent nomination channel can itself miss relevant pairs. These are
held-out **source papers relative to threshold tuning**, not prospectively
captured pre-ingest target-page snapshots. Precision is conditional on that
topically enriched nomination set, not an estimate over every paper–page pair.
The evaluator stores hashes and measurements, not copies of the 83 source and
target wiki pages; a future prose edit will stop a fresh replay until the
matching wiki revision is restored.

The operational conclusion is narrower: the default scan and durable queue
work across synthesis, idea, and proposal pages, and the tested recall point
estimate clears 95%; category metadata and review burden still need attention.
The next validation should accumulate prospectively labeled arrivals with
pre-edit page snapshots and a second adjudicator, then report recall and queue
size by page type. Treat the existing 18 page-type `category:` values as a
separate metadata repair or make the fallback independent of that field before
claiming robust cross-corpus recall.
