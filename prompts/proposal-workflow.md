# Synthesis and idea proposal workflow

Read this file when the user asks the wiki to suggest, remember, review, or act
on possible synthesis or idea pages. Proposal Markdown is a decision ledger,
not a finished evidence page.

## Generate a bounded proposal set

Start with a question, limitation, or target problem:

```bash
researchwiki proposals generate "<question>"
researchwiki proposals generate "<question>" --papers <stem> ...
```

The default is a preview. It retrieves a paper-diverse packet from the claims
DB, includes relevant prior proposal decisions and up to three existing synthesis/
idea summaries, makes one generation call, and returns at most three proposals.
The full proposals, evidence mappings, and author provenance are saved in a local
`.proposal-cache/<hash>.json` receipt. Read that file before choosing entries:

```bash
researchwiki proposals accept .proposal-cache/<hash>.json --select 1 3
```

Acceptance makes no model calls and saves those exact entries. Repeating acceptance
of the same receipt/entry returns the existing record without overwriting feedback.
`generate --write` remains a generate-and-save shortcut, not acceptance of an earlier
preview. Receipts are local review artifacts; accepted Markdown is the synced ledger.

Use `--prepare-only` to inspect retrieval without **any** model calls, including the
cross-category planner. Cross-category preparation contains target evidence only
and marks `planning_pending: true`; generation adds the planner and source search.
Explicit `--papers` must exist and have claims: at most eight, or four target-category
papers for cross-category generation. Claims from explicit papers follow the
existing query ranking before their per-paper quota is applied, with lexical
query matching as the fallback for claims outside the retrieved set. Each paper
enters the packet, but individual
proposals need not cite every selected paper. A proposal
must explain a tension, shared mechanism, complementary limitation, boundary
condition, or cross-category application. Prefer one discriminating thesis to
a broad topic summary. Evidence connections may use only the IDs in the packet;
the writer maps them to durable `[[stem#claim_slug]]` citations.

Existing-page and feedback context use the same conservative lexical matching:
multi-term questions require two distinct meaningful term matches, including
one in the title, question, or topic seed. Those fields outweigh summary/thesis
matches, and no matches is a valid result. This is an overlap hint, not semantic
duplicate detection or a topic-sufficiency gate on retrieved claims.

The exact JSON contract is included in both planner and author prompts on every
provider. An unsupported topic must return `{"proposals": []}` rather than a bare
array or an unrelated suggestion. Planning requires a target problem, explicit
capabilities, and one to three domain-independent search queries. Transfer mode
requires an idea with a named source-category method, operation-level mapping,
and citations from both source and target categories; abstain if the evidence
cannot support that connection. Revisions prompted by prior feedback must retain
the supplied `parent_proposal` ID. These are checked structurally where possible;
the quality of a mapping or experimental control still needs editorial review.

### Chat-authored proposals

Use the preparation JSON as `packet` in a receipt with `version: 1`, `proposals`
(the same structured fields used by generation), and `usage: {model: "<exact model>"}`.
Keep evidence IDs and citation mappings unchanged; cross-category ideas must cite
both target and source-category evidence. Review the complete receipt, then use
the same `accept <file> --select ...` command. Validation checks structure and
evidence references, not scientific fidelity; proposals remain hypotheses, not
citable evidence. Do not regenerate a reviewed receipt to accept it.

## Cross-category application

Start from a problem in a known content category, not from a favorite method:

```bash
researchwiki proposals generate "<target problem>" \
  --target-category <category> --cross-category
```

This adds one small planning call that translates the target problem into
capabilities, then searches claims outside the target category. Keep a result
only if it states the source method, target problem, transfer mapping,
mechanism, assumptions, adaptations, baseline, and first rejecting experiment.

## Remember decisions

Saved records live under `wiki/proposals/` and sync with the rest of the wiki.
Record every material decision with a reason:

```bash
researchwiki proposals list --status proposed
researchwiki proposals feedback <proposal-id> \
  --decision shortlisted --reason "<why>"
researchwiki proposals feedback <proposal-id> \
  --decision published --reason "<what held up>" \
  --resulting-page "[[ideas/<slug>]]"
```

Statuses are `proposed`, `shortlisted`, `deferred`, `rejected`, `drafted`, and
`published`. Later generation receives relevant prior statuses and the latest
feedback, reducing repeated weak suggestions. Use `parent_proposal` only when a
new proposal explicitly revises one of those supplied prior records.

## Rebuild and verify

Markdown is canonical. `state.db` tables `proposals` and `proposal_feedback`
are derived views only:

```bash
researchwiki db rebuild
researchwiki lint --json
```

Inspect `proposal_contract_violations`; fix malformed fields, sections, or
feedback in Markdown and rebuild. Once a proposal is chosen, draft the actual
synthesis or idea page using its own author prompt and run both grounding gates
plus `check-coverage`. Do not treat a proposal itself as citable evidence.

## Legacy concept hubs

Existing `wiki/concepts/` pages remain readable and the manual `concepts`
commands remain available. New wikis do not scaffold concept hubs, `status`
does not promote candidates, and ingest does not mutate them automatically.
Use proposals for new cross-paper discovery; maintain a legacy hub explicitly
only when a single-term spoke registry is still useful.
