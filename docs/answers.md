# Grounded answer operation

`researchwiki answer` generates a bounded answer from an explicit evidence scope.
This reusable operation is available in Community; projects and saved-output
management live in the separate commercial application.

```bash
researchwiki answer "What did the enzyme assay detect?" --scope-file selection.json
researchwiki answer "What did the enzyme assay detect?" --scope-file selection.json --json
```

Run from the library root. Use the [version 1 evidence selection](evidence-scopes.md)
format. A scope without source hashes is snapshotted before retrieval; a supplied
snapshot must still match. Missing/non-paper sources fail closed. An empty
selection produces insufficient evidence without a model call.

## Retrieval and checks

Claims nominate papers first, then scoped lexical wiki retrieval extends the
selection. When those hits are thin, selected PDFs are inspected in stem order.
The current bound is four papers, twelve passages, and eight atomic statements.
PDF extraction covers at most the engine's supported first eighty pages.
This operation does not establish exhaustive project coverage or research consensus.

Every retrieved passage must match fresh extraction of its reported PDF pages.
The author selects an evidence ID for each plain-text statement; the engine owns
the citation renderer. Claim anchors must be backed by both their current claim
text and their PDF passage. Direct PDF evidence uses a bare paper citation.

Every statement, including short ones, must pass numeric consistency against its
cited passage and a separate support-judge invocation. Claim-anchor citations also
require full entailment from the cited claim text. Partial/unsupported verdicts,
missing verdicts, malformed replies, invented citations, generic/stub model IDs,
and source changes prevent a verified answer. Prose output hides rejected drafts;
JSON retains their statements and check outcomes for inspection.

`verified` means these checks passed. A model judge can still make mistakes, and
retrieval can omit relevant evidence. Review source passages before consequential
use. A held-out real-paper accuracy evaluation remains necessary before release.

## Providers and receipts

The registered `answer` and `answer_judge` phases inherit the author and judge
roles. They use the existing provider configuration: API, chat-relay, or mixed
routing. The ChatGPT profile supplies reasoning headroom for both phases. Relay
handoffs remain visible on stderr and use the ordinary mailbox protocol; no new
model client or responder protocol is introduced.

Version 1 JSON receipts include the question, full scope and source SHA256 values,
selected excerpts/page ranges/citations, atomic statements and check results,
actual response model IDs, reported token fields, and creation time. Relay token
fields are zero because the relay does not measure usage, not evidence of zero
cost. Receipts are local provenance records, not signatures or access grants.

Exit codes: 0 verified; 1 rejected or insufficient evidence (also invalid input
through the normal CLI funnel); 2 unavailable sources/index/provider. No wiki
page, project metadata, or saved output is written. Normal retrieval caches may
be built. There is no automatic repair/retry after a rejected answer, though
existing transport-level retry behavior still applies.
