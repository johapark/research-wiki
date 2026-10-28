# v0.6.0 real-paper release evaluation

> **Historical scope:** this report predates removal of the automated `answer`
> command and module. Answer scores and corrections below describe that withdrawn
> implementation. Active [cases](cases.yaml) and the runner now evaluate synthesis
> only; synthesis findings remain relevant. The original
> [review policy](historical-policy-2026-10-28.md) and cases are preserved unchanged.

**Result: evaluation complete; the frozen release threshold is not met.**
The full real-model run produced five useful verified answers out of six
positive requests and correctly abstained on both negative answer requests.
Both positive synthesis drafts were rejected; the negative synthesis request
correctly produced no draft. A later numerical-format diagnostic verified two
answers, but does not replace or improve the primary run's suite score.

## Method and provenance

The [original frozen cases](historical-cases-2026-10-28.yaml) contain eight answer requests and three synthesis
requests spanning six bundled real papers. Four papers are from 2026; the two
CRISPR papers are older. Gold expectations were fixed before model execution
and never included in author/judge prompts. The library contains minimal records
with verbatim abstract quotations, copied PDFs, and isolated indexes/state. It
exercises PDF fallback rather than claiming full graded-claim ingestion coverage.
The user's wiki and PDFs were not edited.

Actual author and judge responses identify `gpt-6.1-sol`, through the configured
OpenAI-compatible provider. Review is **agent review**, not human validation.
Every primary-run answer statement and every synthesis unit, including hooks
and theme headings, was inspected against its cited passage and the extracted
full PDF. The review does not merely reuse the pipeline's verification flag.
See [reviews-2026-10-28.json](reviews-2026-10-28.json) for per-unit decisions,
required distinctions, editorial scores, all four run histories, fixture/PDF
hashes, receipt hashes, and available source-manifest fingerprints.

Raw receipts, rendered drafts, full extracted PDF text, manifests, and (after the
initial run) exact model-call transcripts remain in ignored local directories:

- `output/evaluations/v0.6.0-initial/`: original behavior.
- `output/evaluations/v0.6.0-corrected/`: explicit schemas and multiple keyword hits.
- `output/evaluations/v0.6.0-final/`: primary complete run with abstract context.
- `output/evaluations/v0.6.0-number-format/`: two numerical-case diagnostics.

Runs are not pooled. Previous failures are retained, and the runner refuses to
overwrite a directory that already contains model results. An output directory
named `final` identifies that run, not a claim that the release passed.

## Primary complete run

| Case | Pipeline outcome | Agent assessment |
| --- | --- | --- |
| DeepCRISPR scope/features | Verified | Correct prediction targets and sequence/epigenetic features |
| CRISPECTOR2.0 allele measurement | Verified | Correct de novo SNVs, statistical calling, and measurement role |
| VariantMedium architecture/counts | Verified | Correct model stages and training/evaluation totals |
| MGA assembly scope | Verified | Correct haplotype-mixed output and HiFi-only input |
| scHilda hierarchy | Verified | Correct global lineage decision and local subtype retrieval |
| Ibuprofen pilot | Rejected | Scientifically supported draft, but a spelled-out count was rendered as digits; exploratory qualification omitted |
| Drug question scoped to MGA | Insufficient evidence | Appropriate abstention |
| Invented VariantMedium cure trial | Insufficient evidence | Appropriate abstention |
| CRISPR prediction/measurement synthesis | Rejected | Partially grounded background definition and repetitive organization |
| Experimental labels/knowledge synthesis | Rejected | Background source mismatch and repetitive organization |
| Invented MGA/ibuprofen common trial | Insufficient evidence | Appropriate abstention |

Useful-answer yield is **5/6**. Appropriate answer restraint is **2/2**.
Useful-synthesis yield is **0/2**. Appropriate synthesis restraint is **1/1**.
All **22/22 statements in verified primary-run answers** are source-supported.
The rejected trial draft contains four scientifically supported statements,
but the engine correctly refuses a quantity absent in that literal written form
from its selected passage. No rejected draft is counted as a verified answer.

Of 22 authored units in the two rejected synthesis drafts, **20 are fully
supported by their selected evidence and two are partially supported**. The
problematic definitions are not necessarily false in the full papers; their
selected citations do not establish every clause. Both drafts pass deterministic
page grounding/fidelity gates, but the separate support/editorial judge rejects
them. This demonstrates why passing the page gates alone is insufficient.

Agent editorial scores are 7/8 for each synthesis: theme organization 1/2,
cross-paper reasoning 2/2, coverage 2/2, and boundaries 2/2. They organize around
ideas and distinguish tasks, but repeatedly restate the same comparison across
the fixed sections. Useful status still fails because the drafts are rejected
and contain a partially grounded unit; a numerical editorial score cannot
override that requirement.

## Failures found and corrections

The original run rejected all six positive answers because the judge response
did not match the required JSON envelope. `llm.call(schema=...)` enforces schemas
only for chat relay; API transports ignore that argument. Answer and synthesis
instructions now include the response schema explicitly. Decoding, citation,
numeric, and support checks remain mandatory.

Single best keyword chunks also omitted task-level context and aggregate counts.
Retrieval now keeps a bounded verbatim abstract alongside specific validated PDF
passages, within three passages per paper/twelve per answer/twenty-four per
synthesis. It does not turn a cached wiki summary into PDF evidence. Regression
tests cover multipart fallback, rejecting a stale second passage before any
model call, and checking abstract context against the extracted page.

The primary trial draft rendered the PDF's `Twenty- eight` as `28`, which failed
literal numerical validation. Answer instructions now preserve the source's
written number format. A separate two-case diagnostic verified the VariantMedium
and trial answers, with **11/11 source-supported statements**. The trial answer
still omitted the explicit exploratory-pilot qualification required by the
frozen review fixture, so that diagnostic does not clear the suite threshold.

## Software verification before removal

The final full suite passes: **4,366 tests**, in 170.71 seconds on Python 3.13.3.
It was run with `RESEARCHWIKI_LOCK_DIR=/tmp/researchwiki-v060-test-locks` because
the session sandbox denies writes to the default user-level lock directory.
An earlier default-directory run encountered permission failures and was stopped;
it is not counted as a software regression or as a passing run.

Ruff and `git diff --check` pass. The CLI reports `researchwiki 0.6.0`; the release
notes extractor accepts `v0.6.0`. Wheel and source distributions build, and the
wheel's version, PolyForm license expression, and included license were verified.
These checks do not substitute for the failed real-paper synthesis threshold.

After removing the answer operation, the full suite passes **4,335 tests** in
162.64 seconds with the same writable test-lock override. Lint and whitespace
checks pass. The built wheel excludes the answer package/command and includes
the relocated synthesis dependencies. Nine historical synthesis receipts still
round-trip unchanged, and the active synthesis-only evaluator prepares its
isolated library successfully without model calls. No new scientific quality
score is claimed from this dependency relocation.

## Remaining release work

- Improve and reevaluate synthesis citation selection: background definitions
  need passages establishing every clause, rather than general topical support.
- Improve and reevaluate synthesis progression: each section should develop the
  argument, rather than repeatedly summarize the same prediction/measurement or
  supervision/knowledge distinction.
- Run the active synthesis-only suite against the final release revision.
  Automated answer-generation work is withdrawn; research Q&A remains with the
  chat agent and existing grounding tools.

The threshold and required distinctions were not relaxed after viewing results.
No tag or GitHub release was created as part of this evaluation.

## Limits

This is a small, question-held-out smoke evaluation on existing benchmark papers,
not a statistically powered accuracy benchmark. The papers may be familiar from
training or other benchmarks. There is one run per code revision, no measured
variance, no blinded human review, and possible author/judge/reviewer family
correlation. Successful abstention and source checks establish behavior on these
cases, not comprehensive coverage or general scientific reliability.
