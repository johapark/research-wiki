# Real-paper answer and synthesis release evaluation

This opt-in suite runs actual configured author/judge models against the six
bundled CC-BY-4.0 papers. Attribution and license evidence are in
[../LICENSES.md](../LICENSES.md). It creates an isolated library with minimal
retrieval records containing verbatim PDF quotations; it never writes the user's
wiki or copies private content. It exercises PDF-backed retrieval, not the full
graded-claim ingestion path. Synthetic unit tests cover that path separately.

```bash
python .github/scripts/evaluate_generated_content.py --output output/evaluations/v0.6.0
```

This spends model calls. `--prepare-only` builds the isolated library without
calls; `--case CASE_ID` limits a diagnostic run. Run from the checkout. Keep the
raw receipts, rendered drafts, extracted page text, and manifest under the ignored
output directory. The manifest records the fixture/PDF/receipt hashes, framework
version, selected model routing, outcomes, and latency. Receipts record actual
responder IDs and reported usage.

## Frozen cases and review policy

`cases.yaml` is frozen before the first model run. The questions are held out
from implementation unit tests; the papers are existing benchmark fixtures, not
guaranteed unseen during training. Older DeepCRISPR/CRISPECTOR papers in particular
may be familiar to models. Gold expectations/page locations are reviewer-only;
models receive only the normal production evidence packet and question.

Review every emitted answer statement and every synthesis prose unit against its
cited passage and the full PDF. Record support as supported, partial, or
unsupported, and check source attribution, quantities, and qualifications.
Independently assess relevance to the question and the required distinctions in
the fixture. A correct but irrelevant answer fails usefulness. Never treat the
pipeline's own judge or its `verified` status as independent evaluation.

For synthesis, score each axis 0 (fails), 1 (partial), or 2 (clear and useful):

- Theme organization: organized by ideas rather than consecutive paper summaries.
- Cross-paper reasoning: a specific, supported relationship between the tasks.
- Coverage: both selected papers substantively used, or exclusion justified.
- Boundaries: tasks, uncertainty, and validation scope remain distinct.

A useful positive synthesis must be verified, have no unsupported factual unit,
meet every required distinction, and score at least 6/8 with no zero axis. A useful
positive answer must be verified, fully source-supported, and meet every required
distinction. Partial/unsupported facts never count as accurate. Operational errors
and rejected drafts remain in the request denominator; they are not abstentions.

Negative cases pass only with `insufficient_evidence`, or a verified explanation
that directly rejects the false premise without inventing evidence. Report answer
useful yield over six positive cases, correct restraint over two negative cases,
synthesis useful yield over two positive cases, and synthesis restraint over one
negative case. Also report supported/total emitted factual units, separating
verified outputs from rejected drafts. A zero denominator is unavailable.

The release threshold for this small smoke suite is all positive/negative cases
meeting these rules and no unsupported verified factual unit. Freeze this rule
before running; retain initial failures if code is corrected and rerun under a
new run directory. Do not silently retry until a favorable output appears.

Reviews performed by the chat agent must say `reviewer_kind: agent`. This is an
agent-reviewed release smoke evaluation, not human validation, a statistically
powered accuracy benchmark, or proof of general scientific reliability. Record
same-model-family author/judge/reviewer correlation as a limitation when relevant.
