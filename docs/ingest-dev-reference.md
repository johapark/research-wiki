# Ingestion records developer reference

`researchwiki.records` implements immutable ingestion records, capture hooks and
portable export. The [user guide](ingest-retention.md) owns configuration,
inspection commands, publication recovery and user-visible limits. The
[retention plan](../evaluations/ingest-retention-plan.md) tracks remaining work.

This reference covers the archive contract, extraction capture, grading capture
and their validation. Retention is off by default. Enabled attempts use the
`extraction-v1` profile; new attempts declare additional grading and draft-lineage inventory while
existing archived attempts remain readable.

## Implemented contracts

`RecordStore(root)` selects an explicit archive directory and performs no writes.
`RecordStore.create(root)` initializes it explicitly or opens a valid existing
archive. A nonempty unrecognized directory, unsupported metadata, or a changed
archive identity is never reset. A root directory may be a symlink to a synced
folder; internal archive symlinks are rejected.

Payloads are immutable byte-addressed objects at
`blobs/sha256/<first-two-hex>/<digest>`. `put_blob`, `put_file`, and `put_json` return
a strict `BlobRef`; `read_blob` verifies the bytes it returns. JSON uses UTF-8,
deterministic key ordering and finite values. Duplicate JSON keys, unknown/missing
wire fields, future schema versions, invalid IDs and non-finite numbers are rejected.
Binary malformed responses remain retainable without pretending they are valid JSON
or UTF-8. Capture adapters normalize model settings before creating payloads.

`SourceRef` binds a document blob and its role to an optional representation blob.
An `Artifact` has a stable ID, actual attempt ID, kind, payload, source references,
producer, timestamp, origin, and optional earlier-artifact parents/call identifiers.
All referenced payloads are verified, including source/representation blobs that
differ from the artifact's primary payload. This store does not parse PDFs,
extract text, validate scientific support, redact user payloads, or authenticate a
producer's provenance assertions.

The `extraction-v1` profile freezes six minimum required inventory slots:

| Slot | Artifact kind |
| --- | --- |
| `source.original` | `source_snapshot` |
| `source.representation` | `representation` |
| `target.request` | `model_request` |
| `target.response` | `model_response` |
| `target.result` | `extraction_result` |
| `target.trace` | `parse_trace` |

Additional declared slots may use the supported kinds. Several artifacts, such as
response retries, can belong to one slot. The minimum slots cannot be removed,
made optional, or assigned different kinds to fabricate completeness. Both
hex UUIDs and canonical hyphenated ingestion attempt UUIDs are preserved.

## Attempt lifecycle

1. Construct `AttemptSpec` with the actual attempt identity, capture policy,
   producer, explicit requirements, and timezone-aware creation time. Optional
   request/job/generation fields correlate existing operations without replacing
   their identity. Use `extraction_requirements()` for the minimum inventory.
2. `start_attempt(spec)` publishes the initial event/checkpoint. Repeating an
   identical start is idempotent; reusing the ID for another specification fails.
3. Store payloads, then `record_artifact(slot, artifact)`. Parents must be earlier
   artifacts in the same attempt. Repeating the same descriptor is idempotent;
   changing bytes or metadata under the same artifact ID is a conflict.
4. `resolve_slot(attempt_id, slot, status, reason)` explicitly records `not_run`,
   `omitted_by_policy`, `unavailable`, `failed`, or `unknown`. Failure/unknown can
   preserve already-retained artifacts. An unexecuted slot cannot contain artifacts.
5. `finish_attempt(attempt_id, AttemptFinished(...))` records the supplied execution,
   outcome, publication, and projection observations separately. Remaining pending
   inventory becomes `unknown` with `not_recorded_before_terminal`, never an empty
   successful output. Terminal attempts cannot accept new observations; exact
   retries remain idempotent. Corrections need a new identified observation/attempt.

Capture state is derived: open required attempts are `in_progress`; terminal
attempts are `complete` only if all required slots are retained or explicitly
reported `not_run`, otherwise `incomplete`. Mode `off` reports `disabled` and
declares omitted inventory. A reasoned `not_run` observation is not retained source
data or proof that a model executed. Completeness describes the producer-declared
capture inventory, not extraction recall, scientific correctness, or publication
authority. The lifecycle layer supplies runtime observations without changing
those distinctions.

## Verification and recovery

Immutable attempt events are stored as
`attempts/<attempt-id>/events/<sequence>-<event-id>.json`. A predecessor digest
links their canonical JSON. `manifest.json` is an atomically replaced projection.

- `verify(attempt_id)` validates event identity/order/transitions, referenced blob
  sizes/checksums, and manifest correspondence. It returns `Verification` with
  `checkpoint` equal to `current`, `stale`, or `missing`.
- `read_attempt` requires a current checkpoint. A missing/stale one raises
  `CheckpointStale`; reads never repair records implicitly.
- `repair_checkpoint` replaces only a missing checkpoint or one that exactly
  represents an earlier valid prefix of the verified history. Invalid/future
  schemas, conflicting inventory, lost events, or corrupt payloads block repair.
- An event that persists before checkpoint failure remains authoritative. Repair
  and retry do not create a second artifact, response, or event. An interrupted
  write before the first event may leave empty directories; the same start can
  initialize them without replacing an authoritative observation.
- `list_attempts` validates retained attempts and surfaces damage rather than
  silently skipping it. Inspection/repair never calls a model, touches a corpus
  database, or infers publication from the existence of a wiki file.

`RecordConflict` describes an invalid requested mutation. `RecordStoreUnavailable`
is an environment failure; `RecordCorrupt` and `CheckpointStale` specialize it.
Callers must not interpret any of these as permission for an automatic paid retry.
An error after atomic event publication may mean that the event exists even when
its checkpoint did not finish. Verify before deciding what to do next.

## Storage and concurrency limits

Writers use the engine's existing host-local `exclusive_lock` mechanism and private
temporary files, file `fsync`, and atomic replacement. New files are created with
0600 permissions; new archive directories request 0700. Parent-directory sync is
attempted on supported POSIX filesystems; Windows and unsupported directory-fsync
operations have weaker directory durability. No power-loss or cloud-sync guarantee
is asserted. Framework host lock files remain in its normal temporary lock directory,
not in the retained archive or canonical corpus.

Blob publication and each attempt's event/checkpoint update are serialized. No
model calls, corpus guards, parent/child subprocesses, or SQLite transactions occur
under archive locks. The implementation coordinates cooperating processes on one
host. It is not a distributed-writer or hostile-filesystem isolation mechanism.

Payload insertion is streamed for files. Blob reads return verified bytes; full
attempt verification streams source blobs and deduplicates checksum work within
that verification. The current store replays and verifies the
attempt when mutating it; large-history/write-overhead measurements and optimization
are future work. Unreferenced payloads or unpublished `.write-*` files can remain
after a crash; no pruning or cleanup API is implemented.

Moving the complete archive preserves identities and relative references. No
original absolute mount path or database is needed. The implementation has no
general bundle import, legacy migration, or pruning. The lifecycle layer adds
publication handoff, runtime policy selection and self-contained attempt export.

## Source ownership and representations

`ExtractionRecorder(store, attempt_id)` reads an already-started required-capture
attempt. `snapshot_source(path)` copies the original PDF into an immutable blob
and returns a `ProcessingSource` with separate original and processing paths.
`extract_sections(processing_path, records=recorder)` reads that snapshot and
retains a versioned, page-aware representation.

The representation records raw PDFium pages, repaired page text, the exact repaired
joined text used by the existing extractor, extractor version, PDF
metadata, pages read, the 80-page limit, and unread page numbers. Unicode codepoint
offsets into the joined text are recorded only if independent page repair gives
the same join. A cross-page repair keeps the original joined transform and marks
page mapping unavailable; it never silently substitutes another text. The artifact
envelope retains the attempt's producer/version identity.

`bind_context(ctx)` keeps `ctx.pdf_path` as the original file owned by promotion
and selects `processing_pdf_path` for metadata, extraction, crosslinks and grading
reads. It verifies attempt/source identity. The lifecycle layer instantiates and
binds the recorder before metadata work, guards source replacement at promotion,
and preserves displaced canonical PDFs.

## Target extraction

Given a retained source/representation:

```python
from researchwiki.agents.phases.extract import extract_sections
from researchwiki.agents.phases.target_claims import extract_target_claims
from researchwiki.records.extraction import ExtractionRecorder

records = ExtractionRecorder(store, attempt_id)  # caller already started this attempt
source = records.snapshot_source(pdf_path)
sections, text = extract_sections(source.processing_path, records=records)
output = extract_target_claims(
    metadata=metadata, sections=sections, pdf_full_text=text, records=records,
)
```

The target phase verifies that supplied sections/text match its retained extraction.
An isolated context scope connects its call to the shared `llm.call` client;
unrelated phases and calls outside that scope retain their existing behavior.
It creates:

- Logical request: exact prompt/system/cache-prefix/schema, configured provider
  and model, normalized settings, and source dependencies. This persists before
  budget reservation.
- Dispatch requests: separate IDs under one logical call, actual non-secret adapter
  inputs and negotiated settings. A client dispatch observation does not prove
  remote execution; SDK-internal retries remain unobserved.
- Visible answer: complete permitted answer content, available usage/stop metadata,
  provider-reported or relay self-reported identity, validation/reuse observations.
- Phase input: normalized parser input text, with configured API model identity
  kept distinct from provider-reported identity.
- Extraction result and parse trace: final output and every recognizable candidate,
  including validation failures, duplicate representatives, importance sorting and
  the 35-item cap. Partial salvage records response character spans where known.
  Keyword inputs/filter results are separate.

`parse_target_claims(text)` is pure and returns `(output, trace)`. Existing valid
normalization, sorting and caps are preserved. Invalid field types are reported
and dropped rather than raising incidental `.strip()` errors. Strict JSON rejects
duplicate keys and non-finite values; malformed/truncated text remains retained
in answer artifacts even when only some items can be salvaged.

The optional `render_for_author_prompt(output, records=records, max_normal=5)`
records rendered exposure and the normal-item cap separately from the full target
result. The live author path selects the current attempt's recorder automatically
when rendering that attempt's target output.

## API, relay and failure boundaries

API request projections exclude credentials, headers and endpoint strings. Answer
projection excludes provider reasoning fields, Anthropic thinking blocks and the
existing OpenAI-compatible `<thought>` prefix. It retains the answer before
phase-level fence stripping, JSON salvage or filtering; it is not a wire dump.
Anthropic's provider-controlled temperature becomes null with an explicit status,
preserving the requested temperature separately. API schema intent is recorded as
`not_applied`; relay schema checking is recorded as local validation.

Relay answer fields are serialized as a whitelisted projection before rejection,
retry or accepted-response cleanup. Decoded answer values, untrimmed `via`, reuse
status and retry linkage survive mailbox deletion. Extra envelope fields are
excluded. If protocol JSON/UTF-8 is unreadable, no safe answer projection is known:
record `unavailable`, keep the mailbox for manual inspection, and leave capture
incomplete. This does not claim byte-exact retention of malformed protocol envelopes.

The target-phase stub creates synthetic results/traces and explicitly `not_run`
request/response inventory; it does not fabricate dispatched calls or real usage.
Unavailable relay/API usage is recorded independently of legacy numeric usage
defaults returned to callers.

Archive failures are typed and latched. Logical-request failure prevents reservation;
failure before any dispatch cancels its reservation without charging a call. A
failure after a known response settles available reported usage; uncertain dispatched
usage keeps conservative reservation charging. Response retention failure preserves
the relay mailbox and prevents automatic retry. Phase-input retention failure occurs
after settlement and cannot charge twice. Diagnostic capture stores error classes
and HTTP status rather than secret-bearing provider exception bodies.

The [user guide](ingest-retention.md) describes parent operation ownership through post-work,
terminal and early-exit inventory, profile/job identity, publication guards,
receipt handoff and operator inspection/export.


## Grading capture

The `observed_grading` scope covers draft grading, the requested support veto and
post-publication claim grading. Each invocation has a distinct `run_id`, role and
attempt iteration. `observed_claim` associates ordered candidates and individual
checks with the parsed claim. Original draft Markdown, parsed page bodies/options
and full fidelity/aggregate results are retained at their respective boundaries.

New attempts declare `grading.source`, `grading.input`, `grading.evidence` and
`grading.result` inventory; `grading.request`/`grading.response` are required when
verification is requested. These use existing representation, parse-trace and
model-call artifact kinds. Source references and earlier-artifact parents bind
evidence to the retained processing PDF and grading run.

BM25 top-five and semantic top-ten candidate sets record actual returned order,
ranks/scores, raw/effective query and parser fallback. Check inputs preserve the
actual context limits described in the [user guide](ingest-retention.md#evidence-and-check-limits).
Numeric traces distinguish raw/transformed inputs and retrieved/full-PDF matches;
negation traces retain the exact neighborhood, tokens, pattern and advisory role.
Coverage traces retain full salience/target fixtures and scoring results before
bounded missed-anchor transport.

Evidence retains its role: retrieval candidates, value-presence checks, advisory
negation parity and classifier observations. No hits, skipped cross-references,
unavailable semantic scoring, no eligible support context and failed/malformed
judgments remain distinct. Partial observations survive failed/interrupted runs.

Support capture persists evidence and eligible/skipped classifier inputs before
judge dispatch. `GradingCallRecorder` routes the existing shared-client API/relay
capture into grading inventory, preserving full visible replies, parser inputs,
validation/retry/reuse and reported usage without assigning them to target extraction.
The existing latched failure and usage-settlement rules apply. A known response
settles available usage once even if archival persistence fails; missing output
never automatically causes another call.

Implementation hashes identify the numeric, fidelity, retrieval, semantic and
support code available during the run. Configured model identity does not
establish a resolved weight revision; unknown revisions remain null.

## Source offsets and cache version

PDF chunk cache version 3 stores character starts/ends and a versioned text
representation alongside the Tantivy index. Old caches rebuild on their next use.
Chunk text and scoring rules retain their existing behavior; the added stored
fields do not participate in BM25 indexing.

Offsets count Unicode codepoints in `pdfium-page-repair-v1` text, not PDF bytes or
page coordinates. Raw source slices preserve whitespace. Retrieved text is the
separate `whitespace-word-join-v1` transformation. Capture verifies that each
declared slice reproduces its retrieval text and matches the index representation
identity. Unknown mappings are marked unavailable. A contradictory mapping or
source identity fails required capture and prevents publication.

The numeric checker also retains its own joined/repaired and normalized text
representations. It does not assume that their offsets equal the index text's
offsets across page boundaries.

## Semantic vectors and storage

Actual vectors used by the semantic calculation are retained with dtype,
endianness, shape, byte order and base64-encoded bytes. Evidence bundles deduplicate
them by content; each claim references its query vector and ordered candidate
vectors. These inputs allow the dot products and max/clamp aggregation to be
checked without the original embedding cache or model weights. They do not
reconstruct the text encoder or identify an unavailable model revision.

Candidate text is also deduplicated within each run. Numeric full-text
representations reuse content-addressed blobs. Evidence is persisted in bundles
rather than one archive event per claim, limiting the event/history overhead.
Retaining complete passages and vectors still adds storage and verification I/O;
representative overhead measurements remain part of the default-rollout gate.

## Draft, critique and selection lineage

`records.lineage.record_phase` wraps the author, grade, tournament, critic, evolve
and DEBUG content owners. It does not consume or trust best-effort progress events.
Each invocation persists an identified input before work and a completed, failed
or interrupted output. Failure diagnostics retain exception classes, not arbitrary
exception messages. Disabled retention takes the original path without writes.

New attempts declare `lineage.<phase>.input`, `.output`, `.request` and `.response`
slots plus draft/decision inventory. Unexecuted phases and no-call critics/stubs
resolve to `not_run`; missing output from executed work remains unknown. Unavailable
visible answers mark response inventory unavailable and capture incomplete. A phase
that fails can have completely observed failure records; completion does not imply
that the phase succeeded or that a draft is correct.

Draft artifacts retain exact emitted text, model/temperature, usage, stance, trailer
fields and legacy iteration IDs. Artifact UUIDs are their portable identities;
machine-local iteration IDs are correlation hints. The attempt-local registry keeps
strong object references only in memory and serializes access across parallel author
slots. Distinct drafts retain distinct identities even if their text is identical.
Changed draft content creates a new observation rather than overwriting an artifact.

Author/evolve outputs depend on their retained phase inputs and visible responses.
Revisions point to prior drafts; evolve inputs also point to retained critic output
when available. DEBUG drafts survive rejection, with their nested grading linked
to the repair input. Grading-run inputs point to the lineage input for the exact
draft, while grade outputs retain complete scores and claim details.

Reconciliation records retain source identity and operator overrides alongside
returned metadata. Crosslink records retain the source sections, metadata and final
candidate set. Commit input records the fully cleaned/verified draft, gate outcome
and crosslink checks before keyword or category calls. These phase inputs and results
use separate run IDs and parent links.

Tournament records retain ordered candidates, scores, actual ordering keys, selected
draft and the deterministic rationale. Revision decisions retain the candidate,
prior winner, both score sets, operator and accepted/rejected result. These are
operational decisions, not scientific labels. Source hashes and parent links make
losing drafts and rejected repairs inspectable after DB/source/cache deletion.

`PhaseCallRecorder` reuses shared `RetainedCall` capture for reconciliation,
crosslink, author/critic/revision/DEBUG, keyword and category API/relay calls.
Logical requests, dispatches, permitted visible replies, parser inputs and
usage/retry observations are assigned to the active phase/run;
they do not enter target-extraction inventory. Required input/call capture failures
stop further work using the existing latch and budget policy. No extra model calls
are introduced.

The canonical Markdown page is retained as one content-addressed representation
before the journal commit point. Its digest is checked against the bytes written
to the corpus. Staged and committed publication receipts link to that same page
artifact; the staged receipt also links back to the prepared receipt, and the
committed receipt links to the staged receipt. A rollback receipt links the page
artifact when promotion had reached that point. `records show --lineage` can
recover the exact Markdown bytes after the corpus page has changed or disappeared.
This archive snapshot preserves the ingest-time training text; it is not a claim
that the draft differs from the published page. The existing mutation journal
and ledger remain publication authorities.

This remains a partial Stage 3 implementation. Richer conditional omission
reasons are still required before a full-ingest profile.

## Inspection payloads

`records show --lineage` includes full draft/critique/selection payloads and their
artifact envelopes. It can be combined with `--grading`.
`records show --grading` includes decoded full grading payloads and their artifact
envelopes, parent links and source references. Ordinary `records show` keeps its
compact manifest and publication receipts. Exports include all referenced grading
representations, evidence, model calls and results. Verification and inspection
work without the original source, corpus, database, mailboxes or caches.

## Validation

All checks use disposable synthetic inputs and local responders, with no external
provider calls or writes to the developer's corpus. They validate retention and
portability, rather than scientific extraction quality.

| Layer | Tests / installed acceptance harness | Coverage |
| --- | --- | --- |
| Archive | `tests/test_ingest_records_contracts.py`, `tests/test_ingest_records_store.py`; `.github/scripts/records_acceptance.py` | Strict wire fixture (`tests/record-fixtures/attempt-spec-v1.json`), immutable IDs/deduplication, malformed bytes, corruption/future schemas, interrupted event/checkpoint writes, safe repair, symlink containment, concurrent writers and relocation |
| Extraction | `tests/test_ingest_extraction_retention.py`; `.github/scripts/extraction_acceptance.py` | Real synthetic PDF extraction, complete/discarded candidates, API/relay retry and cleanup, usage/setting normalization, budget and archive failures, relocation after source deletion |
| Lifecycle | `tests/test_ingest_retention_lifecycle.py`; `.github/scripts/retention_acceptance.py` | Real CLI with stub models and a zero-request local sentinel, portable export after corpus/DB deletion, explicit checkpoint repair and publication/capture failure boundaries |
| Grading | `tests/test_grading_retention.py`; `.github/scripts/grading_acceptance.py` | Long passages, exact offsets, deterministic semantic vector replay, mock API/relay judgments, failure handling and portable inspection after source/cache deletion |
| Draft lineage | `tests/test_draft_lineage_retention.py`; `.github/scripts/retention_acceptance.py` | API/relay calls, all parallel author slots, losing/revised/DEBUG drafts, critique/grade parents, actual selection decisions, capture failures and portable CLI inspection |
| Core ingestion | `.github/scripts/fresh_install_acceptance.py` | Fresh installed-package ingestion behavior |

The archive-only smoke check needs `filelock` without loading PDF, ML, provider or
database stacks. Extraction/grading harnesses use real PDFium extraction and
mocked model transports. The installed acceptance CI job runs these harnesses on
Linux, macOS and Windows.

```bash
python -m pytest -q tests/test_ingest_records_contracts.py tests/test_ingest_records_store.py tests/test_ingest_extraction_retention.py tests/test_ingest_retention_lifecycle.py tests/test_grading_retention.py tests/test_draft_lineage_retention.py
python .github/scripts/records_acceptance.py --expect-installed
python .github/scripts/extraction_acceptance.py --expect-installed
python .github/scripts/retention_acceptance.py --expect-installed
python .github/scripts/grading_acceptance.py --expect-installed
python .github/scripts/fresh_install_acceptance.py
```

`--expect-installed` asserts that the harness uses a wheel installed in a separate
virtual environment. Omit it for checks against a source checkout.
