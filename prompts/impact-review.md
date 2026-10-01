# Post-ingest impact review

Trigger: `researchwiki status` reports a pending, stale, or unscanned impact
receipt, or an `agent ingest` receipt lists impact candidates.

The local scan compares a promoted paper with eligible synthesis pages, open,
scoping, or validated ideas, and active proposal records. It writes
`.ingest/impact-review/<paper-stem>.yaml`. This is a candidate queue, not wiki
evidence and not permission to edit an authored page. `audit:` retains every
scanned target, including those below the candidate threshold; `candidates:`
lists the pages requiring a decision. The scan makes no model calls.

For each candidate, read its wiki page and the new paper's graded claims first.
Use the PDF when the wiki lacks the deciding detail. Edit the candidate row's
`decision:` to `incorporated`, `not_relevant`, or `deferred`, and write a short,
specific `reason:`. A decision without a reason remains pending in `status`.
Only choose `incorporated` after any page edit passes that page type's grounding
and fidelity gates. Proposal records preserve their append-only `## Feedback`
history; do not rewrite it as part of this queue review.

An `unscanned` receipt means the local index or scan failed, not that there are
no candidates. Fix the reported cause and rerun the local scan through
`python -c 'from researchwiki.impact_review import scan; scan("<category>/<paper-stem>")'`
from the repository root. A `stale` receipt means the source or a target's substantive content
changed after the last scan; rerun it before deciding. The scan preserves
decisions only when the source and target fingerprints still match.

`--memory-evolve` remains an optional model-backed proposer for synthesis pages
only. Ideas and proposal records require page-type-specific review; the impact
receipt never writes their prose or status.
