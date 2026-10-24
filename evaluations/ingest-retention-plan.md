# Ingestion retention, provenance, and auditability plan

Status: Stage 0 foundation and Stage 1 extraction hooks implemented; lifecycle activation pending.
Created: 2026-10-22. Code review and implementation-scope refinement: 2026-10-23.
Code baseline inspected: `0ba18dc162fda2987ae2ba1f8ad7c7b28c0b9750`.

### Review outcome

The retention direction is sound. Implementation should proceed in small engine
changes, with the following corrections incorporated before runtime work:

- Required capture needs a minimal publication guard and outcome/retry integration
  in the first instrumented release; these cannot all wait for full-pipeline tracing.
- Initial deployment is explicit opt-in `required` capture. Switch the default only
  after the controlled-rollout gate; do not add a new default ingestion failure
  mode before source ownership, budget settlement, and recovery checks pass.
- Store exact source spans separately from whitespace-normalized retrieval text.
- Snapshot the displaced canonical PDF before promotion overwrites it. Metadata
  reconciliation alone does not perform that source upgrade.
- Normalize provider-controlled/unknown settings into strict JSON, and record where
  model/schema/usage information actually came from.
- A full capture scope encloses operation post-work, not only `run_ingest`.
- Freeze retention settings at worker execution and include them in job configuration
  identity, while preserving strict version-1 public wire contracts.

The archive foundation in §19 Stage 0 is implemented on
`feat/ingest-records-foundation`; see [its internal contract](../docs/ingest-records.md).
It uses explicit temporary-root fixtures and is not wired into live ingestion.
Stage 1's [extraction-observation slice](../docs/ingest-extraction-records.md) is
implemented on `feat/ingest-extraction-retention`, with explicit test injection.
The next coding change is its separate lifecycle integration. No default capture
behavior has been activated.

## 1. Purpose

Make ingestion leave a durable explanation of what happened, including work that
did not become a published page. An operator should be able to determine which
source version was read, what content reached each instrumented phase, what a
model returned, how that response was transformed, what evidence each check used,
and why a draft was selected, rejected, revised, or published.

This improves the ingestion process itself:

- Diagnose extraction omissions separately from drafting omissions.
- Investigate unsupported claims, lost qualifications, numeric drift, and poor
  evidence retrieval using the actual inputs and observations from the run.
- Compare parser, prompt, model, retrieval, and gate changes against frozen runs.
- Retain useful failed attempts and losing drafts instead of only successful pages.
- Recover audit records after database rebuild, cache cleanup, or library relocation.
- Reuse reviewed records for future evaluations without treating generated output
  or retrieval scores as established scientific evidence.

Published Markdown remains the research knowledge layer; original PDFs remain
its grounding sources. The retained archive is an audit layer. Unpublished model
outputs must not enter normal wiki search, the citable claims DB, or publication
automatically merely because they have been archived.

## 2. Scope and boundaries

### Included

Generic engine functionality shared by foreground CLI ingestion, the application
operation, batch workers, and durable local jobs:

1. Immutable source and extracted-representation snapshots.
2. Exact logical model inputs and complete visible outputs for captured phases.
3. Raw-to-parsed extraction lineage, including dropped and capped candidates.
4. Full verification inputs and independently identified check results.
5. Draft, revision, selection, and publication lineage.
6. Explicit capture policy, incomplete states, storage failures, and recovery.
7. Portable inspection, verification, export/import, and historical inventory.

### Not included in the initial implementation

- Changing scientific grounding rules, grading thresholds, or promotion overrides.
- Removing the existing claim-count or PDF-page limits without separate evaluation.
- Claiming exhaustive extraction, general OCR, figure understanding, or table parsing.
- Automatically rerunning old papers, making additional judge calls, or replaying
  paid requests to repair missing records.
- Replacing the current provider client, progress recorder, publication journal,
  machine-local job supervisor, or current wiki/claims database.
- A hosted archive service, distributed writer coordination, or a new GUI.

## 3. Current behavior and confirmed gaps

| Area | Inspected behavior | Consequence |
| --- | --- | --- |
| Target extraction | [`phase_target_claims`](../researchwiki/agents/runner_support.py) returns the parsed list but writes counts/status, usage, and timing to iteration history | The full extracted checklist is not recoverable from the history row |
| Extraction parsing | [`target_claims.py`](../researchwiki/agents/phases/target_claims.py) can salvage partial JSON, filter malformed items, deduplicate, sort by importance, and cap the parsed result at 35 | Persisting only the final object loses response and transformation history; the checklist is intentionally bounded |
| Source exposure | [`extract.py`](../researchwiki/agents/phases/extract.py) reads at most 80 PDF pages; section and prompt budgets further limit context | A successful extraction does not imply that the entire paper was processed |
| Wiki-derived claims | Database claims are parsed from selected wiki-page sections | They do not represent every finding proposed by the extractor or present in the PDF |
| Grading evidence | [`fidelity/paper.py`](../researchwiki/grade/fidelity/paper.py) saves the top BM25 passage truncated to 500 characters; checks can use wider context | The saved excerpt cannot explain all inputs to numeric, negation, semantic, or support checks |
| Draft history | [`runner.py`](../researchwiki/agents/runner.py) and [`iterations.py`](../researchwiki/db/iterations.py) retain some drafts, scores, notes, and decisions in SQLite | Valuable partial lineage exists, but not a complete portable attempt record |
| Progress events | [`events.py`](../researchwiki/operations/events.py) deliberately emits bounded observations; delivery failures are best-effort | It is unsuitable as the authoritative content-retention channel |
| Model responses | [`llm.py`](../researchwiki/agents/llm.py) returns responses to phase code; provider and phase transformations occur along the path | Capture must occur before visible content is discarded or transformed |
| Provider metadata | The Anthropic adapter can use `NaN` for provider-controlled temperature; both API adapters currently return the configured model in `LLMResponse.model`; schema enforcement differs by route | Strict archive JSON needs a normalized projection; do not label configured IDs as provider-reported or API schema intent as enforced output |
| Chunk text | [`_chunk_text`](../researchwiki/index/pdf_chunks.py) calculates source offsets but emits whitespace-joined text and currently retains page ranges rather than those exact offsets | Retrieval text is not necessarily a verbatim substring; preserve an exact source slice and its mapping separately |
| Attempt lifetime | The runner returns before [`run_post_work`](../researchwiki/operations/ingest.py) executes; its failure paths also do not always return a context | A runner-return finalizer alone misses later work and can lose failure identity |
| PDF upgrade | [`promote._move_pdf`](../researchwiki/agents/promote.py) can overwrite an older canonical PDF during a journalled upgrade | Retaining only the incoming PDF cannot preserve the displaced source version |
| Chat relay | [`relay.py`](../researchwiki/agents/relay.py) removes consumed replies and some rejected responses | The mailbox cannot serve as an ingest archive |
| Recovery and backup | [`WORKFLOW.md`](../WORKFLOW.md#what-gets-cached-what-gets-regenerated) distinguishes corpus files, indexes, histories, and machine-local jobs | Copying corpus directories or rebuilding indexes cannot recreate original model calls or missing history |

Historical source hashes and evidence provenance are incomplete. This plan does
not claim a fresh corpus-wide measurement or infer missing historical identity
from the currently available PDFs.

## 4. Design principles

1. **Retain observations before interpretation.** Store visible response content
   before JSON salvage, normalization, filtering, selection, or cleanup.
2. **Identify document versions by bytes.** Stem, title, DOI, and current path are
   metadata; none identifies the exact historical PDF by itself.
3. **Preserve the processing boundary.** Record what was read and what was omitted,
   including page caps, prompt allocation, unsupported modalities, and fallbacks.
4. **Separate retrieval from verification.** A candidate passage, a numeric match,
   and an entailment verdict are different observations with different limits.
5. **Make unknowns explicit.** Missing, unrequested, skipped, failed, unavailable,
   and response-unknown must not collapse into success, zero, or an empty list.
6. **Keep publication independent.** Retention cannot weaken a gate or fabricate
   publication proof. A capture failure after publication cannot undo that fact.
7. **Make the archive portable.** It must be usable without the original DB,
   original mount path, live relay mailbox, or disposable chunk cache.
8. **Keep capture provider-neutral.** API, chat-relay, and mixed routing use the
   same logical artifact contract and existing model-call implementation.
9. **Do not spend calls to repair bookkeeping.** Inspection, verification,
   reconciliation, and import must not silently dispatch models.
10. **Prefer incremental delivery.** Preserve the currently lost extraction
    output first, then expand to full evidence and full-pipeline lineage.

## 5. Success criteria

For any phase declared captured by an attempt's profile, an operator can:

- Retrieve the exact source bytes and the stored representation actually used.
- Inspect the exact logical prompt/schema and complete visible response content.
- Explain how raw output became accepted, dropped, merged, capped, or selected items.
- Resolve a retained text quote against its original versioned representation.
- Inspect each check's actual evidence set and completion/result separately.
- Trace the selected page through its draft/revision parents and gate decisions.
- Distinguish an interrupted attempt, a rejected draft, a published page needing
  maintenance, and incomplete capture.
- Move/export the record and validate it without the original database or caches.

Completeness is always relative to a named capture profile and an explicit artifact
inventory. The first extraction-focused release must not describe itself as full
tracing of every ingestion phase.

## 6. Storage model

### 6.1 Authoritative archive

Proposed default location:

```text
<library>/.researchwiki/ingest-records/v1/
  store.json
  blobs/sha256/<first-two-hex>/<full-digest>
  attempts/<attempt-id>/
    events/<sequence>-<event-id>.json
    manifest.json
  imports/<import-id>/
    manifest.json
```

- `store.json` identifies the schema and a stable archive UUID.
- Blobs contain immutable PDFs, text, JSON, and other declared payloads. Identical
  bytes can be reused within the archive; different source versions coexist.
- Attempt events record artifact references and state transitions in local order.
- `manifest.json` is an atomically replaced checkpoint of the attempt's current
  inventory/state. Its checkpoint can be reconstructed from valid attempt events.
- Import manifests preserve origin and original identifiers for historical or
  external bundles without rewriting their observations.

This store is retained data. It is not a sandbox, a progress log, an embedding
cache, or a replacement for the job store. Any searchable SQLite catalog over
these records is a rebuildable projection, not their sole copy.

Support an explicit archive-root override. Store relative internal references,
not absolute machine paths as payload identity. Allow a directory-level symlink
for the archive just as a library can use symlinked corpus directories.

When only `wiki/`, `papers/`, and `inbox/` are synced, this new directory will not
automatically follow them. Initialization/readiness output and backup instructions
must show its resolved location. Portability must not depend on syncing live SQLite
WAL files. Do not place extra historical PDFs inside canonical `papers/`.

### 6.2 Durable write order

For each retained observation:

1. Encode payload bytes deterministically where applicable; retain raw text bytes
   separately when normalization would change them.
2. Write to a temporary file on the destination filesystem while computing its digest.
3. Flush and synchronize the file, then publish the immutable blob atomically.
   Verify existing same-digest content rather than overwriting it blindly.
4. Under the attempt writer lock, publish an immutable event referencing only
   already-durable payloads; assign its sequence and parent event identity.
5. Atomically update the manifest checkpoint after the event becomes durable.
6. Acknowledge persistence to the ingest phase only after its required records exist.

Use supported directory synchronization where available and document the actual
platform guarantee. A local durable write is not a cloud-sync acknowledgment or
a distributed transaction. Hashes detect content mismatch; they do not authenticate
the scientific truth of a model output or the identity of its claimed author.

A crash before step 4 may leave an unreferenced blob; a crash after step 4 may leave
a stale checkpoint. Neither permits fabricating a response or replaying a model.
Verification must distinguish repairable checkpoints from missing/corrupt events.

### 6.3 Concurrency and immutability

- One recorder per executed attempt; parallel author slots share its serialized sink.
- Propagate recorder context using the engine's existing copied-context pattern.
- Use attempt-scoped host locks, not a corpus-wide lock around model calls.
- Preserve event order as observation order. Do not invent a linear phase ordering
  when draft slots run concurrently.
- A shared synced folder is not a distributed locking system. Separate machines
  create separate attempt UUIDs; conflicting same-attempt histories are rejected.
- Corrections, regrading, and reviews create new observations with parent references.
  Original payloads and decisions remain immutable.

Lock order is explicit: when an operation needs both the corpus guard and archive
write locks, acquire the corpus guard first. Regular capture must release archive
locks before requesting the corpus guard. Publish blobs before acquiring the attempt
event lock; do not hold an attempt lock across model I/O, child processes, or calls
back into corpus mutation. Readers of immutable blobs need no write lock. Exercise
promotion, export and reconciliation concurrently to detect lock inversions.

## 7. Identity and schema contracts

### 7.1 Identifiers

| Identifier | Definition and purpose |
| --- | --- |
| `archive_id` | Stable UUID for the archive; independent of checkout path and machine-derived repository key |
| `document_sha256` | SHA256 of exact source-file bytes; original and upgraded PDFs have distinct identities |
| `representation_id` | Versioned identity binding document digest, parser/repair settings, and stored text/object digests |
| `attempt_id` | Existing executed-ingest attempt identity; not a stable request ID or PDF stem |
| `request_id`, `job_id`, `generation` | Optional correlation to submission and worker ownership; preserve their existing semantics |
| `call_id` | One logical model call, correlated with progress where available |
| `dispatch_id` | One observable client dispatch/retry within a logical call |
| `response_id` | A particular received/reused response observation |
| `candidate_id` | A target-extraction item tied to call/response and original item index |
| `draft_id`, `revision_id` | Immutable draft/revision identity with explicit parent references |
| `check_id`, `evidence_set_id` | Exact verification invocation and the ordered context it used |

Normalized claim text or a SQLite row number is not a universal claim identity.
Preserve duplicates and their relationships; normalization hashes can nominate
duplicates but cannot silently merge different findings. Existing durable wiki
claim slugs remain useful downstream references to published claims.

### 7.2 Common artifact envelope

Each artifact descriptor must include:

- `schema_version`, `artifact_id`, and `kind`.
- Attempt, phase, call/dispatch/slot identifiers as applicable.
- Content reference, media type, byte count, digest, and encoding when textual.
- Producer identity/version, creation timestamp, and source/parent references.
- `origin`: observed, imported historical record, reconstructed observation, or
  newly reprocessed work.
- Completeness state and explicit omission/error reason where relevant.

Use strict versioned readers. Reject duplicate JSON keys, unknown required schema
versions, invalid digests, conflicting identity, non-finite numeric values, and
unsafe paths. Do not reset damaged records or silently discard unknown fields
while claiming to have imported the complete original record.

### 7.3 Attempt manifest

The manifest carries the following groups; field spelling is finalized in the
schema prototype before implementation spreads across modules.

| Group | Required information |
| --- | --- |
| Identity | Archive/attempt IDs and optional request/job/generation correlation |
| Policy | Capture mode/profile/version, resolved non-secret configuration, declared required artifact kinds |
| Producer | Engine package/revision, prompt/config projection digests, parser/check versions |
| Source | Original/effective document references; stem/category/metadata only when known, with observation time/provenance |
| Inventory | Each expected artifact's retained/not-run/omitted/unavailable/failed/unknown status and reason |
| State | Execution, scientific outcome, publication, projection maintenance, and capture state separately |
| Lineage | Parent attempts, retries, upgrades, historical imports, and recovery observations |
| Checkpoint | Last valid event sequence/hash and terminalization status |

Do not require a successful reconcile, a final stem, or a published page before
an attempt can be recorded. Validation failures should retain whatever identity
and input information actually became available.

## 8. Source snapshots and representations

### 8.1 Original versus effective input

Snapshot the submitted PDF before paid work or destructive source movement. Record
the effective processing source separately if it changes. In the current engine,
metadata reconciliation reads the incoming source; the preprint-to-journal swap
happens in `promote._move_pdf`. Under the existing corpus transaction guard, retain
the old canonical PDF as `displaced_canonical` before overwriting it and bind the
new canonical version to its already captured incoming bytes. If that required
snapshot fails, abort before replacing the old canonical file. Every later
prompt/representation must refer to the document version it actually used.

Copy source bytes into the archive; do not hardlink or symlink individual snapshots
to files that may be replaced or edited. Distinguish the original user-facing
source path/ownership from the stable processing input. Scientific extraction
should read the immutable snapshot or verify a read-stable equivalent. Promotion
must retain current inbox/external-file move/copy semantics and verify the expected
effective source instead of accidentally moving the archive's copy.

Introduce a distinct read-side processing-source reference rather than assigning
the archive blob path to `ctx.pdf_path`. Metadata extraction, target extraction and
pre-publication grading must use the same retained version. Promotion still owns
the original path and existing move/copy behavior. Validate source identity when
crossing that boundary; a changed original must not be published as though its
new bytes produced an older captured representation. Ordinary in-place extraction
with only an unrelated before/after file hash is insufficient proof of read identity.
Check the staged canonical PDF against the retained effective digest before
`mark_committed`, not only the original pathname before opening/moving it. A source
replacement between those steps must reject publication rather than bind old
extraction artifacts to new canonical bytes.

Retain supplied supplementary files with separate digests and parent-document
references. Record whether they were processed, partially processed, or merely
available. A supplement's existence does not establish that its contents reached
the extractor.

### 8.2 Text representation

Preserve:

- Total physical PDF pages and the pages actually processed.
- Page-level native/extracted text as available and the repaired/normalized text
  used by the engine, with transformation/version information.
- Exact joined text, page boundaries, section spans, and joining convention.
- PDFium/parser versions and relevant text-repair/wordlist/configuration identity.
- Extraction errors, empty pages, textless pages, and OCR status.

Prefer exposing a page-aware result from the existing extraction code and adapting
it back to the current `(sections, full_text)` interface. Avoid a second independent
extractor or inconsistent repeated PDF reads. Fixture tests must prove unchanged
scientific input text for existing processing paths, except documented fixes.

### 8.3 Source coordinates

For an exact text span, retain:

```json
{
  "document_sha256": "<source digest>",
  "representation_id": "<stored representation>",
  "coordinate_system": "page_unicode_codepoint_v1",
  "pdf_page": 1,
  "start": 120,
  "end": 186,
  "quote_ref": "<immutable quote blob>"
}
```

This is an illustrative field example, not an existing wire contract. Offsets
are zero-based, end-exclusive Unicode code-point positions in the declared page
text; physical PDF pages are one-based. Printed page labels are separate optional
metadata. Joined-text offsets require an explicit page map and coordinate label.

Validate that slicing the stored representation reproduces the exact quote.
Parser/repair/OCR changes create a new representation. Do not reuse old offsets
against newly extracted text without an explicit checked mapping.

Section names and model-supplied “Fig. 2” locations remain location hints until
resolved. Bounding boxes, table cells, and figure-object references are nullable
unless the relevant modality was actually processed. No inferred page coordinates.

### 8.4 Retrieval text is a transformed representation

Current chunking finds a window's `start_off`/`end_off` in joined source text but
sets `Chunk.text` to words joined by single spaces. Newline and spacing differences
mean that text may not equal the corresponding verbatim slice.

Retain both `source_span` (exact stored representation slice) and `retrieval_text`
(the bytes the retriever/check actually consumed), plus the transformation/version
that links them. A multi-page window needs joined offsets and page-map segments;
do not put the entire window under one page-local quote. Exact-quote verification
uses `source_span`; check reproduction uses `retrieval_text`.

Expose the offsets when chunks are built, rather than locating them later with
fuzzy matching or first-substring search. If persisted chunk metadata changes,
version the cache and rebuild stale entries locally; do not reinterpret old caches
as carrying coordinates they never stored. Reuse the existing per-page extraction,
`PAGE_SEPARATOR`, and `page_offsets` equality invariant in `pdf/text.py`.

## 9. Coverage and context exposure

Retain coverage at three distinct levels:

| Level | Record |
| --- | --- |
| Document extraction | Total/processed pages, omitted ranges and reason, unreadable pages, supplements/modality status |
| Phase input | Included sections/page spans, selection order, character/token budgets, clipping/fallback behavior, other source dependencies |
| Output transformation | Raw item count when parseable, valid/invalid/duplicate/capped items, final checklist size, keyword transformations |

The current 80-page extraction/indexing cap and 35-claim parsed checklist cap
remain unchanged in the first release. Record actual configured/effective values,
not just today's constants. A complete retained response from a bounded prompt
is not an exhaustive inventory of paper findings.

Each prompt should reference the source spans/context blocks used to assemble it.
The exact prompt remains authoritative about what the engine sent, including
fallback text that cannot be mapped precisely. Unresolved mappings stay explicit.

Classify input origin: source PDF text/captions, supplementary content, existing
wiki context, external metadata, user input, or another declared source. External
abstracts/TLDRs and discovery prose do not become PDF evidence through retention.
Keep existing grounding exceptions and provenance requirements unchanged.

## 10. Model-call retention

### 10.1 Logical request

Before dispatch, persist the exact engine-visible logical inputs:

- System/user messages and any cache-prefix content, with their ordering.
- Structured-output schema and relevant prompt-template references/digests.
- Phase, source/context dependencies, slot/iteration/parent IDs.
- Requested provider/model, temperature, maximum output tokens, reasoning settings,
  and other effective non-secret generation parameters.
- Adapter/config projection version and capture policy.

Do not store authorization headers, API keys, dotenv contents, credential-bearing
endpoint strings, or wholesale transport/debug dumps. Configuration identity is
based on a whitelisted non-secret projection. Content retention is a deliberate
local data channel; progress events continue to carry bounded references/status.

### 10.2 Visible response

Persist complete visible answer text or structured content before answer-format
stripping, JSON salvage, item filtering, or phase validation. The permitted answer
boundary excludes reasoning-only blocks as explained below. Within that boundary,
where a transport adapter normalizes answer content, retain the pre-normalized
content and the normalized value passed to phase code, with a transformation
reference. Do not claim byte-exact retention of the entire provider wire response.

Also record, when actually available:

- Requested and reported model identities separately, and how each was obtained.
- Usage totals/cache subsets with reported, estimated, or unavailable status.
- Finish/stop reason, structured-output validation outcome, and effective setting
  changes such as a supported-effort fallback.
- Dispatch/retry/reuse lineage and response arrival time.
- Typed failure details without copying secret-bearing raw exception/wire dumps.

Do not collect provider-internal hidden reasoning. Exact reported model names and
relay `via` values are provenance observations, not cryptographic verification of
which underlying model executed. Unknown token usage is not zero usage.

Archive normalization must handle current adapter behavior explicitly:

| Observation | Archive representation |
| --- | --- |
| Anthropic temperature omitted after negotiation (`NaN` sentinel today) | Preserve requested temperature; applied temperature is null with `provider_controlled`, never JSON `NaN` or invented zero |
| API `LLMResponse.model` populated from the request | `configured/requested` identity unless a separate provider-reported value was actually observed |
| Relay `via` | `self_reported` identity, stored separately from the configured hint |
| API `schema` argument currently ignored by dispatch | Retain requested schema and `transport_enforcement=not_applied`; record phase-parser validation separately |
| Relay schema validation | Record actual relay validation and retries, not provider-native structured-output enforcement |
| Stub target phase returns before calling `llm.call` | Synthetic phase output with no dispatched model request/response; do not invent a call to satisfy the profile |

Preserve byte-exact answer content at the permitted visible-output boundary, but
do not move capture ahead of the adapter's removal of reasoning-only blocks merely
to call an artifact “raw.” Distinguish answer-content retention, normalization, and
excluded reasoning; this archive is not a provider wire-response dump.

### 10.3 API, relay, and mixed parity

| Mode | Required integration |
| --- | --- |
| API | Shared `llm.call` recorder plus narrow adapter hooks where visible response content or retry details would otherwise be lost |
| Chat relay | Same logical request/response artifacts; capture reply content and validation/retry outcome before mailbox deletion; preserve exact reported `via` and reuse status |
| Mixed | Every phase identifies its actual route; do not infer a single attempt-wide provider |
| Stub | Same schema where exercised, explicitly marked synthetic/stub and excluded from real quality/usage claims |

Separate one logical call from observable dispatches. A client-side dispatch-start
record does not prove that a remote provider executed the request. Server-side
retries that the client cannot observe remain unknown.

Retain malformed and partial visible responses, including relay schema failures,
with their validation result. Preserve available content without wholesale copying
the mailbox or transport credentials. If a response must be redacted or part of it
is unavailable, record that limitation rather than claiming exact full capture.

Successful archival persistence must precede normal cleanup of a captured relay
response. A capture failure is not a transient provider error and must not trigger
an API retry, consume another reply, or conceal an already received response.
Budget reservations and reported usage still settle correctly when capture fails.

The budget paths require explicit treatment. `BudgetTracker.fail` currently charges
the reserved tokens/cost because transport usage is unknown; it is not a way to
cancel a call that capture prevented from dispatching. Persist request intent before
reservation where possible. If capture fails after reservation but before any I/O,
release/cancel that undispatched reservation exactly once without charging a call.
If a response already arrived, settle actual available usage even when the archive
write fails. Only uncertain dispatched work uses the conservative unknown-usage
path. Tests must assert call, token, cost, and reservation counters at each boundary.

## 11. Extraction candidates and transformation lineage

Preserve these layers separately:

1. Full visible response, including non-JSON or truncated material.
2. Parsed top-level object/items where parsing succeeds.
3. Salvaged items and parse errors, with original response positions where known.
4. Validation results for every recognizable candidate.
5. Deduplication groups and the representative-selection rule.
6. Importance sorting, cap decisions, and final `TargetClaimsOutput`.
7. Keywords before and after filtering.
8. What the author prompt actually included from that checklist.

Suggested per-item transformation fields:

| Field | Purpose |
| --- | --- |
| `candidate_id`, `response_id`, `raw_item_index` | Link the item to the observed response |
| `raw_item_ref`, `normalized_item_ref` | Preserve original fields and later normalized form |
| `parse_status`, `validation_errors` | Distinguish missing fields, invalid values, and partial salvage |
| `duplicate_of`, `normalization_version` | Explain duplicate handling without erasing the original |
| `selected_for_checklist`, `selection_reason` | Retain both accepted and capped/dropped items |
| `author_exposure` | Included, omitted by normal-item cap, unavailable mapping, or phase not run |
| `wiki_coverage_observation` | A later coverage check's match/partial/miss result, with check identity |

Not every malformed response has recoverable item boundaries. Preserve the raw
response and record item indexing as unavailable instead of inventing a list.
An omitted checklist item is not automatically an incorrect scientific claim;
author selection and scientific support require separate observations.

Make parsing observable through a pure response-to-result/trace helper. The wrapper
retains its original response before invoking it. Preserve current accepted output,
ordering, cap and exception behavior; newly tolerating a malformed field is a
separate parser fix, not an invisible retention side effect. When parsing itself
raises, keep the response and exception classification even if item-level lineage
could not be produced. Do not require a second extraction call to generate a trace.

## 12. Verification and evidence retention

### 12.1 Keep evidence roles distinct

- **Claim-stating span:** text that asserts the proposition.
- **Retrieval candidate:** context nominated by BM25, embeddings, or another retriever.
- **Evidence item:** a result, passage, table/figure object, analysis, or derivation
  assessed as bearing on the claim by an identified check/reviewer.

A sentence repeating a claim does not necessarily supply the underlying scientific
evidence. Text presence, qualitative support, numeric consistency, and scientific
truth must not be represented as the same flag.

### 12.2 Retain the context actually used

For each verification invocation, save an ordered evidence set containing:

- Claim text and its extraction/draft/published-claim references where applicable.
- Full candidate passage text by immutable reference, not only chunk IDs or scores.
- Exact source representation, page/span coordinates when available, and quote hash.
- Retrieval query, retriever/index/configuration version, rank, and score.
- The specific context subset supplied to each numeric, negation, semantic, or
  model-based support check, including any full-document fallback.
- Check completion state, result, error/unavailability reason, and producer/version.
- Model judge request/response references when that optional check ran.

The current top-1 display excerpt can remain <=500 characters for compatibility.
The archive should retain full passages and all context actually used. Current
grading uses top-3 context for some checks and can retrieve top-10 for semantic
scoring; record actual sets rather than guessing one shared `top_k`.

Preserve quantitative tokens and unmatched values with their context and relevant
normalization rules. Do not call a semantic score an entailment verdict or treat
an unrequested entailment check as passed.

### 12.3 Independent check status

Each check should distinguish:

```text
execution: not_requested | not_run | completed | incomplete | unavailable | failed
result: check-specific value, or null when no result exists
```

The exact enums require alignment with existing check contracts. These proposed
archive fields do not silently change existing public wire values. Capture the
actual result and method; do not retrospectively upgrade old retrieval-only
grades into stronger verification.

## 13. Draft, revision, and publication lineage

Retain every produced draft, not only the winner:

- Author request/response and emitted draft bytes; slot, stance, temperature and IDs.
- Each grade/coverage observation associated with the exact draft revision.
- Critic feedback and its exact inputs, including failed or empty critiques.
- Revised drafts with parent references and changed/retained observations.
- Tournament/ranking inputs, selection outcome, and available decision reason.
- Gate results, incomplete checks, and any explicit promotion overrides.
- Final rendered page bytes/hash, canonical PDF digest, and publication receipt.

Selection is an operational decision, not a factual correctness label. A losing
draft, a malformed response, and a scientifically unsupported assertion have
different meanings and must remain distinguishable.

Publication authority stays with the existing mutation journal and ingest ledger.
The archive records their outcome and references. It must never certify success
merely because a page path exists or a progress event mentioned publication.

## 14. Capture policy and failure semantics

### 14.1 Explicit profiles

Proposed initial policy:

| Setting | Behavior |
| --- | --- |
| Mode `required` | Required records must persist at the defined boundaries; storage errors become explicit ingest failures/incomplete outcomes |
| Mode `off` | Explicit operator opt-out; record/report that audit content was not retained using available operational channels |
| Profile `extraction-v1` | Source snapshots, representations, target-extraction request/responses/transforms, and attempt/terminal status |
| Later profile `full-ingest-v1` | Declares the additional verification and full phase/draft/publication inventory once implemented |

Develop and validate opt-in required retention first, with existing non-capturing
behavior remaining the rollout default until §19 Stage 5 passes. Once that gate
passes, change new-run defaults to required retention for the supported profile
with documented storage/error behavior and an explicit opt-out. A selected required
mode must never silently fall back to metadata-only capture when space is exhausted.

Freeze resolved policy and archive location for an executing job. Include their
non-secret identity in execution configuration; do not borrow a foreign parent
profile or silently change policy on retry. Proposed environment/config names
should be finalized with the schema rather than adding unversioned wire fields.

Resolve policy at worker execution after its selected profile loads. Persist that
effective policy in the attempt and include its safe projection (mode, profile,
resolved root/archive identity) in `jobs.environment.execution_identity`. The current
identity whitelist omits it. Historical job identities without these settings are
`not_recorded`, not evidence of a particular old policy; retries must report this
comparison limit and create new attempts. Do not retroactively rewrite queued or
completed version-1 request/manifest JSON to add fields.
The existing `configuration_changed` flag remains an aggregate comparison of opaque
configuration digests. Without an independently retained policy on both attempts,
report the policy-specific comparison as unavailable rather than claiming which
setting caused that aggregate hash to change.

### 14.2 Failure matrix

| Failure point | Required behavior |
| --- | --- |
| Archive initialization/source snapshot fails before paid work | Stop before model dispatch in required mode; report a typed environment/capture failure |
| Request persistence fails | Do not dispatch that call; settle any local reservation correctly |
| Request recorded, process/network dies around dispatch | Leave an incomplete/response-unknown observation; do not claim the provider did or did not execute |
| Response received, persistence fails | Report capture incomplete; retain mailbox/emergency material when possible; never automatically repeat paid work |
| Phase parser fails after response persistence | Keep response and parse failure; preserve existing scientific phase behavior and typed errors |
| Gate rejects a draft | Keep complete captured artifacts and rejection reasons; retain existing gate outcome/exit semantics |
| Process dies before manifest checkpoint | Verify retained events/blobs and repair only the checkpoint without model calls |
| Required pre-publication artifact is missing | Block normal publication until capture is corrected or an explicit supported policy decision is made; do not weaken a scientific gate |
| Canonical commit succeeds, archive finalization fails | Preserve `published`; report capture/maintenance separately and reconcile from journal/ledger evidence |
| Already-present source requires no extraction | Record the duplicate outcome and available source references; do not invent extraction artifacts or rerun a model to fill them |
| Unknown schema or corrupt/missing blob | Reject/flag the record; preserve original files for repair, never reset or silently re-fetch |

Emergency spooling is best effort, not a guarantee when storage is unavailable.
There is an unavoidable interval between a remote response and durable local
receipt. The design must expose ambiguity rather than promise exactly-once model
execution or losslessness across every possible process/power/network failure.

Capture persistence errors need a dedicated typed classification, proposed
`CaptureUnavailable(EnvironmentFailure)`, plus an in-memory latched failure on the
recorder. Existing best-effort `except Exception` paths must not erase required
capture failure. A pre-publication health check consults the latch even if an
optional phase caught the exception; post-publication handling reports it without
rewriting committed publication as a failure to publish. Capture callbacks stay
outside retryable provider/schema-error handlers.
Consult the latch before every later model dispatch as well as publication. After
publication, stop new optional paid work and report it incomplete, while allowing
required model-free projection/journal repair to finish.

Use existing wire error fields/enums for an initial implementation. Distinguish
`record_store_unavailable` before dispatch from `record_capture_incomplete` after
received/uncertain work. Classify these before the generic environment-error branch.
Use conservative `reconcile_first` where dispatch/publication is unresolved and
explicit fresh retry where required; neither classification authorizes paid replay.

### 14.3 Separate state dimensions

Maintain distinct attempt dimensions:

- Execution: running, finished, interrupted, or the existing applicable job state.
- Scientific outcome: published, draft ready, duplicate/already present, failure,
  or not yet resolved, using the existing operation's meanings.
- Publication and projection maintenance: journal/ledger-derived existing states.
- Capture: in progress, complete for profile, incomplete, disabled, or corrupt.

Queued/cancelled jobs need not fabricate an executed-ingest archive. Link any
available job record and explain that no model phase ran. Avoid one overall
`success` boolean that obscures successful publication with incomplete capture.

## 15. Compatibility and integration architecture

### 15.1 Separate recorder from progress

Add an artifact recorder/store boundary alongside the existing progress recorder.
Progress remains bounded and best-effort. The artifact recorder enforces required
persistence and returns typed artifact references to phase wrappers.

Suggested internal interfaces:

```text
begin_attempt(identity, policy)
retain_source(source, role)
retain_representation(document, representation)
retain_request(call, logical_input)
retain_response(call, dispatch, visible_output)
retain_phase_artifact(kind, payload, parents)
record_check(check, context, result)
finalize_attempt(result, capture_inventory)
```

These are design targets, not currently supported APIs. Keep persistence out of
pure parsing/scoring helpers where possible: return structured observations and
persist them in the shared runner/operation wrappers. Transport hooks are justified
only where otherwise-discarded content/retry details must be observed earlier.

Recorder lifetime has two entry paths:

- The shared operation owns a scope through validation/recovery, runner execution,
  post-publication work, and final result construction. Bind the runner's actual
  attempt ID once it exists; a pre-run validation failure must not be presented as
  a model-executed attempt. Inspect failures using the scope even when no context
  was returned by the runner.
- A direct `run_ingest` caller receives a fallback runner-owned scope if no parent
  scope exists. It must not create a second recorder/attempt under the operation.
  Its terminal inventory states that operation-level post-work was not invoked.

Do not close the complete-operation archive in the runner's `finally`: optional
`run_post_work` executes after it. At finalization retain original commit-time page
bytes separately from later observed page changes, including supplementary YAML
attachments. Finalization errors must not mask an earlier interruption, failure,
or the authoritative publication outcome.

### 15.2 Expected engine changes

| Module area | Planned work |
| --- | --- |
| New `researchwiki/records/` | Strict schemas, blob store, attempt journal/checkpoints, recorder, verification and export/import |
| `agents/context.py`, `operations/observation.py` | Attempt-scoped references/context propagation without a process-global callback |
| `agents/phases/extract.py`, `pdf/text.py` | Page-aware immutable representation using existing extraction/repair implementation |
| `agents/phases/target_claims.py`, `runner_support.py` | Expose parse/transform lineage and persist it without changing checklist selection behavior |
| `agents/llm.py`, provider adapters, `agents/relay.py` | Shared logical-call recording and narrow pre-normalization/pre-cleanup hooks |
| `grade/fidelity/paper.py`, `agents/phases/grade.py`, `index/pdf_chunks.py` | Full evidence-set observations with exact source/version identity |
| `agents/runner.py`, critic/revision/selection wrappers | Draft parent links, full phase outcomes and skipped-phase reasons |
| `operations/ingest.py`, `jobs/environment.py`, worker/recovery paths | Frozen policy, typed failure reporting, terminal inventory, safe publication reconciliation |
| `mutation.py`, `ingest_ledger.py` | Reuse journal/ledger handoff for archive publication observations; no second commit authority |
| New `tasks/records.py`, CLI dispatch | Local inspect/verify/export/import/inventory commands |
| `WORKFLOW.md`, setup/readiness/status docs | Retained-data location, backup/cleanup policy, coverage and incomplete-state reporting |

Minimum publication integration belongs in Stage 1, not exclusively Stage 3.
Before the first canonical mutation, check the recorder's latched health and the
profile's pre-publication required inventory. Within the existing corpus guard,
preserve any displaced PDF before replacement. Required post-work/publication
observations may still be pending at this boundary; do not demand a terminal
record for work that has not happened yet.

After `mark_committed`, archive write errors preserve journal/ledger authority and
the published result, with explicit capture incompleteness. Do not add the archive
root as an ordinary corpus mutation target: the mutation path restrictions and
cross-directory configuration require a small reference/handoff integration, not
a second multi-file transaction system. Stage 3 extends this bridge to full lineage.

The handoff must precede journal discard for captured direct CLI/operation runs as
well as durable jobs. The existing job-ledger binding requires a job identity and
an external machine-local `publication.json`; it cannot simply point inside the
new portable archive. Add a separate validated archive binding to journal details
and an idempotent publication/rollback handoff keyed by journal and attempt identity.
Keep the existing job binding and its path restrictions intact. Retain an unresolved
journal if required handoff persistence fails, so another process can finish it
without guessing from canonical filenames or replaying provider calls.

### 15.3 Wire contracts

Existing request/result/event/job contracts are strict version 1. Do not add unknown
fields under that schema version or change current enums silently.

The first release can use new archive schemas and dedicated inspection commands
while preserving current operation wire shapes. Human-readable capture summaries
belong on the appropriate diagnostic channel; do not contaminate `--json` stdout.
If typed archive pointers/policy are later added to public operation/job contracts,
make an explicit versioning/migration decision, update all readers, and run paired
consumer compatibility checks before deployment.

## 16. Proposed operator commands

The following commands do not exist yet; they define the intended operator surface.
They should use versioned JSON output, stable error categories, and explicit roots.

| Proposed command | Purpose and constraints |
| --- | --- |
| `researchwiki records list --json` | List retained attempts and completeness/profile; optional stem/request/job filters |
| `researchwiki records show ATTEMPT --json` | Inspect source, inventory, calls, checks and lineage without rewriting records |
| `researchwiki records verify ATTEMPT --json` | Verify blobs, source references, event continuity, expected inventory and exact text spans; no provider calls |
| `researchwiki records export ATTEMPT --out NEW_PATH` | Produce a checksum-verifiable bundle with selected attempts and declared included/omitted source payloads; refuse existing destination |
| `researchwiki records import BUNDLE` | Add immutable records idempotently; reject conflicting IDs/bytes and unsafe paths; never overwrite local observations |
| `researchwiki records inventory --legacy --json` | Report available historical rows/files and provenance gaps through framework readers |
| `researchwiki records reconcile ATTEMPT` | Repair an understood checkpoint or publication observation from durable evidence; no re-authoring |

Do not add a general “replay” command that silently dispatches paid calls. Offline
reconstruction of stored transformations and explicit new reprocessing are different
operations and must have different labels and behavior.

Exports must declare whether they are self-contained. An audit-only bundle that
omits PDFs cannot claim independent source verification. Import validates every
relative path and digest; unrelated library files and credentials are not part of
the export. Import/export does not change canonical wiki/PDF files.

## 17. Historical inventory and salvage

### 17.1 Audit before mutation

Add a read-only inventory that reports, per attempt where possible:

- Available iteration/draft/score/critic records and target-count-only history.
- Existing source-hash bindings, current source availability, and conflicting versions.
- Available sandbox outputs, grading candidates, and active/consumed mailbox gaps.
- Missing request/response artifacts and unknown parser/model/prompt identity.
- Job/publication evidence available independently of SQLite.

Use framework history/DB readers and the existing insights interface for timing
questions; do not rely on hand-written operator SQL as the migration contract.
Keep observed coverage counts separate from unsupported conclusions about corpus
quality or completeness.

### 17.2 Preserve originals

Before import or schema migration, take a consistent SQLite backup using its
backup API and preserve machine-local job/recovery records. Coordinate with active
ingests rather than copying a live database file while ignoring WAL state.
Record snapshot identity/time and any consistency limits; do not clear active
journals or mailboxes as a migration step.

Import original rows/files with explicit legacy origin and new import identifiers.
Keep source row IDs as metadata, not portable record identity. Archive known PDFs
without pretending that their current hashes were recorded by the historical run.

### 17.3 Honest reconstruction

- If an original PDF hash was retained and the matching bytes are available, bind
  that version and retain the basis for the binding.
- If original identity is absent, a current PDF match is a new compatibility
  observation, not proof of which version an old run used.
- A legacy 500-character snippet can be located in stored text. Preserve multiple
  matches as ambiguous; finding the text does not establish entailment.
- Missing target lists and prompts cannot be regenerated from counts or selected
  wiki bullets. Keep them missing.
- Fresh extraction/regrading creates a new attempt linked to the old one, with
  current source/parser/model identity and explicit cost authorization. It never
  overwrites the original or is described as recovered original output.

Historical salvage is optional parallel work. Preserving future runs must not wait
for every old attempt to be reconstructed.

## 18. Storage lifecycle, backup, and cleanup

Default behavior preserves referenced records and source versions. Existing cache,
sandbox, and completed-mailbox cleanup must never delete the archive implicitly.

- Measure archive bytes by source/representation/attempt and the effect of content
  deduplication. Preserve original/source-upgrade versions even if stems are reused.
- Admission checks can estimate required space and enforce configured quotas, but
  runtime writes must still handle disk exhaustion and permission changes.
- A quota failure must not silently truncate payloads or reduce capture scope.
- Any future garbage collection begins with a dry-run inventory of unreferenced
  blobs, excludes active attempts/imports/exports, and uses a conservative grace period.
- Referenced content cannot be pruned as an ordinary cache. Explicit record removal
  requires an auditable policy and must not leave a manifest claiming intact evidence.
- Exports can pin the retained records needed to reproduce an evaluation. A local
  catalog rebuild or corpus DB rebuild cannot alter those frozen source references.
- Backup/export verification should run on an independent location to prove that
  original mount paths and machine-local stores are not hidden dependencies.

Do not promise automatic full replay of a nondeterministic model. The archive
reproduces what the engine observed and supports replay of deterministic local
transformations where their code/configuration is available.

## 19. Implementation stages

### Stage 0 — freeze contract and fixtures

Implementation checkpoint (2026-10-23): `researchwiki/records/` now provides strict
version-1 contracts, immutable payload insertion, attempt event/checkpoint projection,
verification and explicit safe checkpoint repair. The extraction profile's six
minimum slots are frozen in `tests/record-fixtures/attempt-spec-v1.json` and documented
in [the foundation guide](../docs/ingest-records.md). Ingestion, provider dispatch,
public operation/job wire schemas and capture defaults remain unchanged.

First coding PR: new internal `researchwiki/records/` contract/store primitives
and disposable fixtures. Include only the common envelope, source/reference identity,
attempt event/inventory/checkpoint, and capture-policy types needed by extraction.
Use normalized strict JSON; do not create a speculative domain ontology.

Implement immutable blob insertion, attempt-scoped event/checkpoint publication,
strict reads, integrity verification, and repair of an understood stale checkpoint.
Fixture code exercises the store with synthetic payloads and explicit temporary
roots. Runtime ingestion defaults and provider dispatch remain untouched. Defer
import, legacy salvage, cleanup, a searchable catalog and general CLI ergonomics.

Meaningful tests cover byte identity/deduplication, immutable ID conflicts, corrupt
and future-version records, interrupted event/checkpoint order, concurrent writers,
relative reference containment, and moving the store without losing references.
Do not treat a passing store fixture as an instrumented ingestion attempt.

Acceptance: every required artifact has an identity, owner, persistence boundary,
and missing-state representation; API/relay capture maps to the same contract.
The store verifies independently of corpus databases/caches, with no writes outside
its selected test root and no public operation/job wire changes. The first review
should be limited to these foundational behaviors.

### Stage 1 — stop target-extraction loss

Extraction checkpoint (2026-10-23): immutable processing-source binding, page-aware
raw/repaired representations preserving legacy joined text, a pure target parser
with candidate selection traces, and scoped shared-client/API/relay capture hooks
are implemented. Synthetic fixtures exercise retry cleanup, missing usage,
provider-controlled settings, budget cancellation/settlement and archive failures.
Unreadable relay protocol envelopes are explicitly unavailable rather than copied
wholesale or represented as full answer capture. See the
[extraction guide](../docs/ingest-extraction-records.md). Lifecycle activation and
its operator tooling below remain unimplemented.

Deliver through two dependent, reviewable PRs after the store foundation:

**Extraction PR:** page-aware source snapshots preserving existing text, separate
processing/original ownership paths, pure parse-result/trace extraction, and exact
target-phase request/answer capture through the shared client and relay hooks.
Include adapter normalization and correctly identified synthetic/no-call output.
Keep the recorder test-injected until lifecycle integration is complete.

**Lifecycle PR:** opt-in required policy, selected-profile/job identity propagation,
latched typed failures and budget cancellation/settlement, operation/direct-run
scope ownership, early and duplicate exits, minimal publication health check and
displaced-source preservation, and publication-safe finalization. Add minimal
list/show/verify and self-contained export for independent acceptance. No default
switch, public wire-field additions, import/catalog/pruning, or paid backfill.

Acceptance: source, request, response and transformations remain inspectable after
relay cleanup, DB deletion, cache deletion and relocation; an unselected candidate
is still present; no additional model calls occur. Incomplete runs remain honestly
incomplete. Only the `extraction-v1` profile is advertised.

Stage 1 cannot ship required capture without its lifecycle PR: deferring all
publication/retry integration until Stage 3 would invalidate its failure guarantees.

### Stage 2 — make grading explainable

Deliverables: full ordered evidence sets, exact check inputs and source spans,
retriever/index identity, numeric/negation observations, optional support-judge
calls, and evidence-role/completion distinctions.

Acceptance: every retained score/verdict can be traced to the exact claim/context
used; long passages are not truncated in the archive; caches can be removed without
losing evidence. Retrieval-only results remain retrieval-only.

### Stage 3 — complete phase and publication lineage

Deliverables: remaining model/metadata observations, all author/critic/revision/
selection artifacts, parent links, conditional phase omissions, post-operation
terminal capture status, and expansion of Stage 1's publication bridge and safe
checkpoint reconciliation to full-pipeline lineage.

Acceptance: losing/rejected drafts and interrupted attempts survive; publication
status comes from durable commit evidence; post-commit capture failure cannot cause
paid re-authoring. Enable `full-ingest-v1` only after its declared inventory is covered.

### Stage 4 — historical import and operator lifecycle

Deliverables: read-only historical inventory, consistent backup/import tooling,
explicit reconstruction observations, full import/export validation, storage
inventory, backup guidance and a conservative cleanup design.

Acceptance: old and new observations cannot be confused; unresolved original hashes
remain unresolved; exports validate on a relocated machine; original DB/history
records are not destroyed or silently rewritten.

### Stage 5 — profile evaluation and controlled rollout

Deliverables: frozen fixture results, before/after compatibility report, storage/
latency measurements, readiness diagnostics, operator documentation, and a reviewed
sample of complete real ingests when separately authorized.

This is a gate applied per profile as soon as its implementation is ready, not a
dependency on completing historical Stage 4 or every full-pipeline feature first.
The extraction profile can pass it after Stage 1; the full profile passes separately
after Stage 3. Until then capture stays opt-in and the existing default stays off.

Acceptance: stage-specific requirements pass for API, relay and mixed routing;
existing ingestion/publication behavior remains compatible; documented overhead
and storage budgets are supported by measurements. Report new-run coverage and
legacy coverage separately.

Stage 1 is the first runtime implementation priority. Stages 2/3 enrich the same
versioned records. Historical work can proceed independently after the inventory
contract exists; it must not delay loss prevention for new ingests.

## 20. Verification matrix

| Scenario | Expected assertion |
| --- | --- |
| Valid extraction | Exact request/response, parsed items, source digest and final checklist are all retained |
| More than 35 valid response items | All observed raw items survive; capped checklist and reasons are distinct; no exhaustive-paper claim |
| Invalid/duplicate/partial JSON | Original visible content and parse/salvage/drop lineage survive unchanged |
| API, relay, mixed, stub | Same required schema and identity relations; actual route/usage origin and synthetic status remain explicit |
| Relay validation retry or reused reply | Original/next response lineage survives cleanup without double consumption or invented dispatches |
| Parallel author slots | Artifacts do not collide or cross attempts; observed event order is consistent |
| PDF replaced or upgraded | Original and effective source versions coexist; prompts/grades bind to the correct bytes |
| Long/textless/malformed PDF or missing supplement | Processed/omitted/unavailable content is explicit; no invented spans or silent claim of full coverage |
| Text normalization/parser change | Old quotes still resolve to old representations; new offsets do not overwrite them |
| Passage exceeds 500 characters | Full archived passage survives even if the legacy UI/DB excerpt stays truncated |
| Numeric/negation/semantic/support checks | Each records its actual context and completion; absent entailment is not a pass |
| Rejected/losing/revised draft | Full bytes and parent/check/selection lineage remain inspectable |
| Capture disk/quota/permission failure | Explicit pre-dispatch blocking or incomplete outcome; no silent downgrade or paid replay |
| Crash around request/response | Durable earlier records survive; ambiguous dispatch/response is marked unknown |
| Crash after canonical commit | Publication remains published; repair uses journal/ledger evidence without author calls |
| Unknown schema/corrupt event or blob | Original bytes preserved and validation fails clearly; no reset |
| DB rebuild/cache/sandbox cleanup | Archive references remain valid; normal corpus claims do not ingest archived drafts |
| Relocated export/import | Checksums, source references and exact quotes validate without original absolute paths or SQLite |
| Legacy count-only/unhashed history | Missing output/identity stays missing; current source hashes are not backdated |
| Recording disabled explicitly | Operator-visible policy; no completeness claim; existing scientific gates remain active |
| Credential handling | Logical content retained; authorization/profile secrets and hidden reasoning absent from captured protocol/config fields |
| Provider-controlled temperature and model identity | Strict JSON has no `NaN`; configured IDs and actually reported IDs have different provenance; unknowns stay null |
| API schema intent versus relay validation | Identical requested schema does not imply identical transport enforcement; parser and relay checks are recorded separately |
| Whitespace-normalized or cross-page chunk | Verbatim source segments reproduce exact quotes; retrieval/check text reproduces normalized input; no guessed single-page offset |
| Capture stops an undispatched reserved call | No provider request and no falsely charged call/token/cost; reservation released once |
| Response arrives before capture failure | Known model usage settles once; capture error does not enter the provider retry loop |
| Optional phase catches a capture error | Recorder latch prevents pre-publication success in required mode; published work remains published |
| Same job retried with changed retention settings | Safe configuration identity shows the change; older identities report retention comparison as unavailable |
| Direct runner versus shared operation | Exactly one recorder; no returned context needed on failure; full operation includes optional post-work |
| Canonical PDF upgrade | Old canonical bytes retained before replacement under the corpus guard; failed snapshot leaves old canonical source intact |
| Concurrent publication and archive operations | No archive-to-corpus lock inversion or parent lock held across provider/child work |
| Direct CLI crashes between canonical commit and archive handoff | Journal remains recovery authority without a job ledger; handoff is idempotent and completes before journal discard |

Use disposable synthetic PDFs and deterministic local API/relay responders for
automated coverage. Do not mutate the developer's corpus or contact a real model
provider. Run meaningful crash/failure and concurrency checks, relevant Community
regressions, package installation checks, and paired consumer checks where an
interface changes. A documentation-only change does not require running ingestion.

## 21. Measurements and release decision

Collect, by capture profile and processing outcome:

- Required-artifact completion and missing-artifact reasons.
- Source/representation/span identity resolution and checksum failures.
- Response/dispatch ambiguity and recovery outcomes.
- Raw/valid/duplicate/capped/author-exposed claim counts, with explicit denominators.
- Evidence-set completeness for each check type.
- Extra local write latency, peak memory, bytes per source/attempt, and deduplication.
- Model-call/usage differences against a matching baseline fixture run.
- Cases where a previously unexplainable omission/check/selection becomes inspectable.

Do not infer improved scientific precision/recall merely from improved retention.
Evaluate scientific quality separately against independently reviewed source coverage.
Choose numeric overhead/storage thresholds from the prototype measurements rather
than asserting an unmeasured percentage improvement.

Release the extraction profile when its records survive the failure, cleanup,
portability and mode-parity checks. Release full-pipeline capture only after its
larger inventory meets the same standard. Keep the rollout reversible through an
explicit policy setting without deleting existing archive data.

## 22. Decisions resolved by review and remaining proof points

| Decision | Implementation direction |
| --- | --- |
| First delivery scope | Internal schema/store PR, extraction-observation PR, then lifecycle/opt-in activation PR; no single large rewrite |
| Initial default | Existing off behavior until the implemented profile passes its rollout gate; selected required mode never silently downgrades |
| Source ownership | Stable processing snapshots are separate from original move/copy paths; preserve displaced canonical PDFs at promotion |
| Exact evidence | Store source spans and transformed retrieval/check text separately, with representation/transform versions |
| Provider metadata | Strict normalized projection with requested/applied/unknown settings, identity origin and actual schema-enforcement layer |
| Failure and cost | Typed latched capture failure; undispatched cancellation differs from unknown-use transport failure and from known-response settlement |
| Lifecycle | Parent operation scope includes post-work; direct runner fallback scope is explicit and never double-instantiated |
| Publication integration | Minimal health/snapshot/outcome bridge in Stage 1; full lineage in Stage 3 |
| Job compatibility | Resolve settings at execution and extend the non-secret configuration identity without rewriting version-1 requests or adding unknown fields |
| Locks | Corpus guard before archive writes when both needed; no archive lock across corpus acquisition or provider/child calls |
| Initial tooling | Minimal inspect/verify/export; defer import, catalog, pruning and historical salvage |

Stage 0 freezes the exact implemented internal field spelling, canonical JSON
encoding and fixture versions. The lifecycle PR chooses profile/configuration setting names
and their profile-loader integration. Prototype measurements determine storage
admission/overhead budgets; supported filesystems determine documented durability.
Any later public wire-schema change needs its own compatibility proposal. These
proof points do not require a new model or complete historical backfill.

## 23. Turn retained observations into ingestion improvements

Retention is useful only if it shortens diagnosis and supports better decisions.
After the first profile is reliable, use frozen attempts to separate causes rather
than changing several phases at once.

| Observed problem | Diagnostic distinction | Controlled follow-up |
| --- | --- | --- |
| An important finding is absent from the wiki | Source page omitted, prompt omitted, extractor missed it, parser dropped it, checklist capped it, or author omitted it | Change the responsible stage and measure important-claim coverage plus unsupported additions on an independently reviewed set |
| Long papers lose findings | Unread pages versus context-budget clipping versus bounded output | Evaluate explicit additional-page/section passes and merge rules; preserve source scope and extra-call costs before changing defaults |
| A plausible passage does not support the claim | Lexical similarity versus lost qualifications, wrong population/comparison, or missing table/figure evidence | Compare evidence selection/context against reviewed support cases; do not tune solely to BM25 or embedding scores |
| A correct number is attached to the wrong claim | Number presence versus the relevant metric, denominator, comparator, condition, or source version | Evaluate more specific numeric/context checks without treating any matching token anywhere in the paper as sufficient |
| Malformed responses recur | Provider truncation, schema incompatibility, phase parser behavior, or content filtering | Use retained stop/parse observations to adjust one setting or parser rule at a time; report salvaged versus clean output separately |
| A draft improves style but loses content | Extraction unchanged, author exposure changed, revision removed a claim, or selection preferred a weaker draft | Compare parent/child coverage and support observations; revise critic/selection criteria against source-reviewed cases |
| A source upgrade changes outcomes | New PDF bytes, new extracted representation, metadata correction, or a model/configuration change | Compare matched source versions and settings; do not attribute all differences to the model |
| A failure invites an unnecessary second ingest | Missing capture checkpoint versus unresolved publication versus stale derived indexes | Improve the specific reconciliation/error path and prove no extra authoring call or duplicate publication occurs |

For each follow-up, freeze the source versions, relevant observations, experiment
configuration, comparison rule, and evaluation set before interpreting results.
Report precision and coverage together, include difficult/negative cases, and keep
original baseline records unchanged. Offline inspection and deterministic replay
are free of provider calls; new extraction/judge experiments remain explicit paid
or local-model operations with their own attempt identities.

The first release should therefore deliver both retained records and practical
answers to two operator questions: “Where was this information lost?” and “What
exactly caused this check or publication decision?” Broader extraction and grading
changes follow from that evidence rather than being bundled invisibly into the
retention implementation.
