# Real-paper synthesis release evaluation

This opt-in suite runs actual configured synthesis author/judge models against
the six bundled CC-BY-4.0 papers. Attribution and license evidence are in
[../LICENSES.md](../LICENSES.md). It creates an isolated library with minimal
retrieval records containing verbatim PDF quotations; it never writes the user's
wiki or copies private content. It exercises PDF-backed retrieval, not the full
graded-claim ingestion path. Synthetic unit tests cover that path separately.

```bash
python .github/scripts/evaluate_generated_content.py --output output/evaluations/synthesis
```

This spends model calls. `--prepare-only` builds the isolated library without
calls; `--case CASE_ID` limits a diagnostic run. Keep receipts, rendered drafts,
extracted page text, and the manifest under the ignored output directory. The
manifest records source and receipt hashes, framework version, model routing,
outcomes, and latency. Receipts record actual responder IDs and reported usage.
The runner refuses to overwrite an existing run containing model results.

## Frozen cases and review policy

[cases.yaml](cases.yaml) retains the two positive synthesis requests and one
required-abstention request from the original evaluation, without changing their
questions, evidence selection, or expectations. The questions are held out from
implementation unit tests; papers are existing benchmark fixtures, not guaranteed
unseen during training. Gold expectations and page locations remain reviewer-only.
Models receive the normal production evidence packet and question.

Review every synthesis prose unit against its cited passage and the full PDF.
Record support as supported, partial, or unsupported; check attribution,
quantities, qualifications, relevance, and required distinctions. The pipeline's
judge and `verified` status are not independent evaluation.

Score each editorial axis 0 (fails), 1 (partial), or 2 (clear and useful):

- Theme organization: ideas rather than consecutive paper summaries.
- Cross-paper reasoning: a specific, supported relationship between tasks.
- Coverage: selected papers substantively used, or exclusion justified.
- Boundaries: tasks, uncertainty, and validation scope remain distinct.

A positive case passes only with a verified synthesis, every required distinction,
no partially/unsupported factual unit, and at least 6/8 with no zero axis.
Operational errors and rejected drafts stay in request denominators; they are not
abstentions. The negative case passes only with `insufficient_evidence`, or a
verified explanation directly rejecting the premise without inventing evidence.
Report useful yield over two positives, appropriate restraint over one negative,
and fully supported/total authored units, separating verified and rejected drafts.
Zero denominators are unavailable. The release threshold requires all cases to
pass; do not relax it after viewing results or silently retry for favorable output.

Reviews by a chat agent must record `reviewer_kind: agent`. This small smoke
evaluation is not human validation, a statistically powered accuracy benchmark,
or proof of general reliability. Record correlated model families as a limitation.

## Historical evaluation

The automated `answer` command/module has been removed. The
[2026-10-28 report](EVALUATION-2026-10-28.md),
[original cases](historical-cases-2026-10-28.yaml), and
[original policy](historical-policy-2026-10-28.md) preserve the earlier combined
evaluation and its failures as provenance. These historical answer cases are not
supported by the current runner and do not describe a current product feature.
