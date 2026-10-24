# Ingestion-record foundation

The internal `researchwiki.records` package implements the archive foundation in
[the retention plan](../evaluations/ingest-retention-plan.md). It is not connected
to live ingestion yet. No archive is selected or initialized automatically, no
provider is called, and no operation/job wire schema or ingestion default changes.
There are no `researchwiki records` CLI commands in this increment.

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
or UTF-8. Model setting normalization remains the future capture adapter's job.

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
authority. The first implementation has no engine hook asserting any such events.

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
that verification. This conservative first implementation replays and verifies the
attempt when mutating it; large-history/write-overhead measurements and optimization
are future work. Unreferenced payloads or unpublished `.write-*` files can remain
after a crash; no pruning or cleanup API is implemented.

Moving the complete archive preserves identities and relative references. No
original absolute mount path or database is needed. The implementation has no
general bundle import/export, legacy migration, publication handoff, runtime
capture policy selection, or provider callback yet; those remain later increments.

## Validation

Frozen wire fixture: `tests/record-fixtures/attempt-spec-v1.json`.
Behavior tests: `tests/test_ingest_records_contracts.py` and
`tests/test_ingest_records_store.py`. They use opaque synthetic bytes and temporary
directories, including malformed response bytes, interrupted publication/checkpoint
writes, strict schema violations, changed source snapshots, internal symlink escapes,
concurrent process writers, and relocation without a database. No real scientific
paper, live provider, or developer-library write is needed.

Run the focused checks with:

```bash
python -m pytest -q tests/test_ingest_records_contracts.py tests/test_ingest_records_store.py
python .github/scripts/records_acceptance.py
```

The acceptance script also supports `--expect-installed` when run with a wheel
installed in a separate virtual environment. The archive-only smoke check needs
`filelock`, without loading the PDF, ML, provider, or database stacks. Existing
ingestion behavior is checked separately by
`.github/scripts/fresh_install_acceptance.py`; neither script demonstrates live
retention hooks, which remain the next implementation stage.
