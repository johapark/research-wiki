# Proposal and ideation benchmark v2

Twelve corpus-grounded cases assess whether proposals support a worthwhile next
decision. Eight development cases include five families exercised in earlier
live testing. Four held-out cases use four separate source groups. No source
paper or paired case crosses that split. Holdout means withheld from generator
tuning; curators necessarily inspect its evidence and expectations.

The case labels and rubric are **agent-calibrated, not human-validated**.
The user delegated the initial editorial calibration; its findings and limits
are recorded in [CALIBRATION.md](./CALIBRATION.md). No current-suite model quality
score or agreement with the user's preferences has been established.
This is a small diagnostic benchmark, not a validated estimate of research
success. A shortlist-worthy hypothesis may subsequently be disproved.

## Contents

- `inputs.yaml`: questions, source selections, existing-page references, and
  explicitly synthetic feedback. No expected answers or scores.
- `expectations.yaml`: evaluator-only criteria, critical evidence, failure
  conditions, and non-exhaustive examples of acceptable approaches.
- `SCORING.md`: scoring anchors, denominators, calibration, and comparison rules.
- `CALIBRATION.md`: question audit, saved-output judgments, and policy changes.
- `review-template.json`: shape of one candidate/repeat review block. Nulls are
  deliberately incomplete judgments; they must not be interpreted as zero.

| ID | Split | Case |
|---|---|---|
| S01 | Development | Retrieval structures and question scope |
| I01 | Development | A new memory design question beyond an existing survey |
| X01 | Development | Relationship-preserving pangenome export using a source method |
| X02 | Development | Same transfer request with source-method evidence removed |
| R01 | Development | Unsupported qubit-fabrication question with lexical distractors |
| R02 | Development | Explicitly closed survey scope already covered by an existing page |
| F01 | Development | Narrow a rejected retrieval decision framework |
| F02 | Development | Revise memory correction under retention and update constraints |
| S02 | Held out | Cleavage-assay substrate and cellular interpretation |
| S03 | Held out | Fixed-reference projection versus query-parameter adaptation |
| I02 | Held out | Accelerate pixel abstraction without losing its constraints |
| I03 | Held out | Separate representation robustness from clustering-seed stability |

X01/X02 is a controlled evidence-removal pair. I01/R02 and I01/F02 are workflow
variants with changed instructions/context, not clean causal ablations. Report
them as such. All related development cases share one source group, so twelve
cases do not represent twelve independent topic groups. Each held-out case has
its own group, with no shared source papers with development or other holdouts.

## Prepare a frozen local pack

Run from the repository root with the project Python environment:

```bash
python -m researchwiki.benchmark.proposals prepare --out output/proposal-benchmark-v2
python -m researchwiki.benchmark.proposals check --pack output/proposal-benchmark-v2
```

Preparation reads the claims DB first and resolves every selected durable slug.
It freezes exact claim text, available supporting snippets, full existing-page
context, full source-page Markdown, and extracted PDF text with page numbers.
The manifest records PDF hashes, file hashes, rubric version, and the suite identity;
the scoring handbook and available calibration record are frozen alongside evaluator criteria. No wiki
writes, model calls, automatic proposals, or feedback changes occur. Missing
papers/claims make the suite unavailable; they are not scored as model failures.
Preparation refuses an existing output directory. Use a new directory/version
for changed data, and retain the old baseline pack.

`RESEARCHWIKI_DB_PATH` can select an existing local snapshot. A snapshot still
must agree with source-page anchors; do not regenerate grades/claims merely to
make a fixture pass. Missing or changed anchors require curator review.

These are **personal-corpus fixtures**. Tracked definitions contain source
identifiers and task specifications; the frozen corpus text stays in gitignored
`output/`. The complete suite is not self-contained on a fresh clone. Do not
commit personal corpus snapshots or PDFs. A redistributable subset requires
article-specific licenses and attribution through the existing bundled-fixture
process; no such subset is claimed here.

## Input separation and execution

Only `inputs/<case-id>.json` is passed to the generator in fixed-packet mode.
`evaluator/`, the manifest's case labels, scorer, and this handbook must remain
outside the generation context. The preparation/scoring module remains offline.
A separate `proposal_run` module provides the development-only comparison below;
there is no automatic judge or end-to-end runner.

For fixed-packet runs, pass the JSON packet to the existing `generate_proposals`
function. Cross-category packets already contain their curated source evidence;
do not call `expand_cross_category` or retrieve extra sources in this mode.
The negative twin intentionally lacks such evidence and should abstain.

### Development comparison runner

```bash
python -m researchwiki.benchmark.proposal_run plan \
  --pack output/proposal-benchmark-v2 --out output/proposal-comparison-dev
```

This is offline. It creates one request per candidate per development case (16
for this suite), freezing the exact prompts, schema, model settings, source
packet hashes, and effective endpoint. It does not copy held-out packets or
evaluator criteria into requests. Inspect `plan.json` and `requests/` before
approving transmission; `inputs/` contains the corresponding frozen packets.
These files contain personal corpus content and stay in gitignored `output/`.

```bash
python -m researchwiki.benchmark.proposal_run run \
  --plan output/proposal-comparison-dev --approve-plan COPY_EXACT_PLAN_ID
```

Only this command makes live calls. Both commands load the normal root `.env`;
`--env-file .env.NAME` selects a named profile with the standard precedence
checks. No credential values or complete env/config files enter the plan.
Execution rejects a changed endpoint/config, prompt, or proposal implementation.
Both candidates use the same model, temperature, output limit, evidence text,
user message, schema, parser, and validator. The **only prompt difference is the
system editorial policy**: a concise, format-compatible baseline versus the
production policy. Both are single-call generators; this does not compare agents,
retrieval systems, or search planners. Candidate identities are assigned random
A/B aliases; within-case call order alternates across cases.

The runner is serial with no generation retries. Existing provider retries and
parameter negotiation remain enabled: 16 generations are **not** a hard limit of
16 HTTP attempts or a monetary spending cap. Requested settings are recorded;
the wrapper does not expose final negotiated wire settings or per-attempt usage.
Receipts retain raw response text, returned usage, latency, and validation errors;
`run/summary.json` reports per-candidate operational counts and known token totals.
Failed-call usage is unknown, not zero cost. Latency includes provider retries and
rate-limit waits. Raw responses are saved before parsing, so bad JSON is retained.

Completed and interrupted run directories cannot be reused. A started receipt
without a returned response may already have incurred a charge; do not blindly
repeat it. To repeat intentionally, create and approve a new plan. Exit codes:
0 = all responses valid, 1 = bad input or validation failures, 2 = provider/environment
failures, 3 = internal error. An internal error stops the run and preserves available
receipts; no complete-run score should be inferred from partial results.

### Review without candidate identities

Give the reviewer `run/review/outputs-A.json`, `outputs-B.json`, and the frozen
`inputs/`, plus the pack's evaluator criteria. Keep `plan.json`, request prompts,
raw receipt metadata, and `summary.json` out of their context until judgments are
recorded. `run/review/reviews-A.json` and `reviews-B.json` are incomplete scoring
templates. Fill reviewer provenance, every proposal's scores/reasons/verdict,
and feedback checks in the order specified by the evaluator criteria. Interpret
every intelligible proposal in malformed raw responses too: add diagnostic review
entries while leaving `operational_success=false`. An empty template proposal list
after a parser failure does not establish that the raw response had no proposals.

The reviewer folder hides explicit identities, prompts, and usage; it cannot hide
stylistic clues or a model naming itself in its raw output. An agent that already
read the identity mapping must disclose that its review was not identity-blinded.
Do not let the generator read review files or evaluator instructions. Score each
completed candidate block separately with the offline `proposals score` command;
then unblind the mapping and inspect quality, operational, and cost differences.
No reviewer verdicts or wiki feedback are generated or persisted automatically.

For a future end-to-end runner, each case's `corpus_paths` is an allowlist, not
the union of every file in the pack. Stage only those Markdown files and that
case's existing-page context into an isolated temporary wiki/DB/index. Otherwise
the positive twin's source method would leak into X02. Grade-cache-dependent
retrieval must also be isolated. Capture the retrieved packet before generation
and report essential-claim recall separately from editorial quality. Fixed and
end-to-end results must not be combined into one score.

The frozen documents preserve input reproducibility, not privacy guarantees or
protection from a model's training knowledge. Judge attributions against the
supplied evidence; novel hypotheses may be original, but claimed external results
still require supporting evidence.

## Review and aggregate

Preserve raw responses, validation errors, exact model/provider/config, prompt
hashes, pack ID, repeat number, latency, and token usage. Label the five original
live cases as historical development evidence, not scores for this newly curated
suite: packets and task constraints have changed.

For one candidate and one repeat, populate a review block containing every case
in a single split and declare `execution_mode` (`fixed-packet` or `end-to-end`),
then run:

```bash
python -m researchwiki.benchmark.proposals score \
  --pack output/proposal-benchmark-v2 --reviews output/reviews.json
```

Use the direct reviewer shortlist decision for primary utility. Declare
`reviewer_kind` (`human` or `agent`) and `reviewer` identity; keep reviewer kinds
separate and label agent results agent-assessed. The scorer also reports the
provisional 8/10 threshold and disagreements. Do not
choose the decision after seeing the threshold result. Record candidate identity
outside the blinded reviewer interface.

Start with one run per development case. After freezing criteria, compare a
simple single-call prompt with the production proposal generator using identical
models and packets. This fixed-packet comparison tests generation, not retrieval
or cross-category search planning. Use three independent runs per case for a release comparison, retain
all failures, and report split-specific counts and per-case variation. Related
variants and repeats are not independent samples. No live evaluation is performed
by preparing this suite.

Version 2 reduces answer hints in S01/S03/I03 and removes R01's unnecessary
room-temperature qualifier. It also tightens evaluator checks. Keep version 1
packs unchanged; historical outputs and version 1 scores are not version 2 results.
