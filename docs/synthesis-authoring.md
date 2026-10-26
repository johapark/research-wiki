# Verified synthesis authoring

`compose` authors a complete synthesis from an explicit evidence selection. It
uses the same Community PDF, claim, provider, and page-gate implementations as
the existing workflows. The existing `synthesize` command remains the manual
scaffold workflow.

```bash
researchwiki compose "How do these assay results relate?" --title "Assay evidence" --scope-file selection.json
researchwiki compose "How do these assay results relate?" --title "Assay evidence" --scope-file selection.json --json > receipt.json
researchwiki publish-synthesis receipt.json --slug assay-evidence --scope-file selection.json --json
researchwiki check-coverage wiki/synthesis/assay-evidence.md --scope-file selection.json --json
```

Run from the selected library root. [Evidence selections](evidence-scopes.md) may
contain two to eight ingested papers for composition. Narrow larger selections
explicitly; the operation never silently truncates a project. Empty/single-paper
selections return insufficient evidence without a model call. Titles are plain
text up to 180 characters; questions up to 1,500 characters.

## Composition and verification

Every selected paper gets a bounded evidence card: claims are retrieved first,
then PDF passages are checked against fresh extraction and actual page ranges.
The packet holds at most twenty-four passages; PDF extraction covers the first
eighty pages supported by the existing engine. This is bounded retrieval, so the
result cannot establish exhaustive coverage of the papers or a field.

The author returns structured prose organized by ideas. The engine renders the
fixed Question → Short answer → Background → Organizing framework → Findings →
Cross-cutting insights → Tensions / open questions → Outlook → References spine.
Findings has themes connecting multiple papers, and Cross-cutting insights must
combine evidence from at least two papers. The framework table is derived from
the theme/source mapping. Prose uses one footnote per source paper and inline
durable anchors when a specific claim is cited. The exact author model is quoted
in YAML. The hook, theme headings, and all paragraphs are checked, including
units too short for the legacy structural claim heuristic.

Before a document is verified:

- Every unit must cite only evidence in the packet. All numbers must occur in
  its cited passages and, for anchored evidence, cited claim text.
- `check-grounding`'s structural logic must pass. Every durable anchor must
  resolve independently; a valid paper footnote cannot mask a broken anchor.
- The same fidelity check as `grade synthesis --no-semantic` must pass and
  grade more than zero claims. Section/footnote structure must also pass.
- A separate support-judge invocation must fully support every unit against its
  PDF evidence and, where present, its anchor evidence. Partial, unsupported,
  malformed, missing, or duplicate verdicts reject the document.
- The judge must approve theme organization, meaningful cross-paper reasoning,
  and the cite-or-exclude rationale for every selected paper. The scoped coverage
  check is advisory, but each omission must have an explicit reviewed exclusion.
- Selected source hashes and citation paths must remain current.

Outlook is strictly sourced in this automated workflow. The existing manual
authoring procedure's labelled inference/model-prior options remain available
outside this operation. A model judge can err; real-paper accuracy and editorial
quality evaluation remain release work. Synthetic tests check transport and gate
enforcement, not the quality of a real model's scientific review.

## Providers and provenance

`synthesis_author` and `synthesis_support` inherit author/judge roles and support
API, chat-relay, and mixed routing. Both phases preflight before the author call;
the exact response model and reported usage are recorded. Relay handoffs use the
ordinary stderr/mailbox protocol. Relay token zeros are unmeasured usage.

Version 1 receipts retain the title/question, scope and hashes, citation paths,
PDF excerpts/page ranges, structured draft, every unit's checks, coverage ledger,
page-gate counts, editorial review, exact model IDs, usage, and timestamps. A
verified flag cannot omit any required check. Receipts are provenance records,
not signatures or access-control grants.

`compose` previews only: it does not write the wiki, index, log, or application
saved outputs. Retrieval caches may be built. JSON exposes rejected drafts for
inspection; prose output hides them. Exit 0 means verified, 1 rejected/insufficient
or invalid input, 2 an unavailable source/index/provider, and 3 an internal bug.

## Explicit publication and recovery

`publish-synthesis` requires the current scope and a verified receipt. It reruns
the deterministic page gates and source/path checks before the commit. A generic
optional `--guard-file PATH --guard-sha256 HASH` binds an application's selection
metadata to this commit: the hash is checked under the engine's corpus lock,
without adding any project database or commercial dependency to Community.

Publication journals the page, catalog, and log as one canonical mutation. A
failure before commit rolls back; a projection failure after commit reports
`published` with `maintenance_pending` and retains the journal for existing
recovery. It never reports an already committed page as unpublished. DB and
keyword search are updated without a semantic-model download. Republishing the
same receipt/content is idempotent; an existing page with different content or
a slug collision is refused. There is no implicit overwrite or paid model replay.

The page records its source fingerprint and generation-receipt digest. Keep the
receipt if you need the full evidence/check history. The commercial application's
saved-output store supplies that durable lifecycle for project syntheses.
