# Opt-in ingestion retention

The `extraction-v1` profile retains source PDFs, page-aware representations,
target-extraction inputs, visible answers, candidate transformations and publication
receipts. It applies to `researchwiki add`, `agent ingest`, their batch workers,
the synchronous ingestion operation and durable local ingestion jobs. The legacy
digest-only `researchwiki ingest` workflow is outside this profile.

## Enable it in the selected models profile

Add this to the models YAML chosen by your current environment profile:

```yaml
ingest:
  retention:
    mode: required
    profile: extraction-v1
    root: .researchwiki/records
```

The default is `off`. Required mode needs an explicit archive root and mutation
journalling. Relative paths resolve from the selected library root. The archive
must be separate from `wiki/`, `papers/`, `inbox/` and `.mutation/`, including their
symlink targets. A root directory symlink to a separate synced folder is supported.
Internal archive symlinks are rejected. `.researchwiki/` is ignored by Git.

The runner starts one identified attempt and copies the PDF before metadata model
work. Read phases use the immutable processing copy; promotion continues to own
the original source path. Identical source bytes reuse one archive blob. Every
new PDF version has its own blob. Before overwriting a canonical PDF, promotion
retains its displaced bytes; before committing, it checks the new canonical PDF
against the processing snapshot. A mismatch aborts and rolls back publication.

Settings are frozen for an operation. Job workers freeze them across execution
identity and ingestion, and include the resolved policy/profile/root in the
non-secret configuration fingerprint. Attempts retain the actual request, job
and generation IDs; version-1 public request/result/job schemas are unchanged.

## Inspect and export

Use the selected profile's root, or pass an explicit archive:

```bash
researchwiki records list --json
researchwiki records show <attempt-id> --json
researchwiki records verify <attempt-id> --json
researchwiki records export <attempt-id> --output /path/to/new-export --json
researchwiki records --root /path/to/archive verify <attempt-id> --json
```

List/show/verify do not initialize an archive or repair records. Verification
checks event history, identity, payload hashes and checkpoint correspondence.
`show` includes publication receipts. Exports require a new destination and copy
only the chosen attempt, its events/checkpoint, archive identity and referenced
blobs. Exported attempts verify without the original corpus, databases, caches,
mailboxes or mount path. These commands can inspect an explicit archive outside
an initialized library.

## Capture and publication failures

Archive failure is typed and latched. Further model calls and promotion stop;
required capture does not silently downgrade to off. Undispatched reservations
are cancelled; known response usage settles once; uncertain dispatched work keeps
conservative budget charging. No capture error automatically requests another
answer or consumes another relay reply.

The operation owns capture through post-work. Direct runner calls have their own
scope when no operation already owns it. Terminal records include the operation's
post-work result, or explicitly identify the narrower direct-runner scope.
Duplicate and early exits resolve phases that never executed as `not_run`.
Executed work whose output was lost remains unknown/incomplete.

Publication writes a validated archive binding into the mutation journal. Prepared,
staged and committed/rolled-back observations are distinct. The final commit or
rollback receipt must reach the archive before journal cleanup. A failed handoff
keeps the journal and its backups. This also works for foreground ingestion that
has no job ledger. Existing job publication ledgers remain separate authorities.

A capture failure after commit preserves the known published result and reports
`capture_unavailable` with `reconcile_first`. Published CLI work exits successfully
with repair guidance, so a batch does not re-author it. Captured batch workers use
versioned result sidecars; capture failures and missing/unknown proof block batch
resume replay. Jobs likewise refuse retry while their result requires reconciliation.

## Repair without repeating model work

For a missing checkpoint or one that exactly matches an earlier verified event
prefix, repair it explicitly:

```bash
researchwiki records verify <attempt-id> --repair-checkpoint --json
```

Future schemas, conflicting checkpoints, missing events and corrupt payloads are
preserved for review. Repair never invents observations or fixes source bytes.

If the library still has a retained publication journal, run from that library:

```bash
researchwiki records recover --json
```

Recovery requires every pending journal to name the selected archive. It uses the
existing corpus transaction recovery: rollback uncommitted file mutations, or
finish the handoff/derived maintenance of committed mutations. Failed recovery
retains unresolved journals. It makes no model calls. Ordinary index/grade repair
commands remain available after journal recovery.

An interrupted attempt may stay open after its publication receipt is recovered.
That receipt proves the observed commit/rollback boundary; it does not reconstruct
missing extraction or post-work history. This increment does not synthesize a
successful terminal result for an interrupted process. Review such an attempt
before submitting a separate new request; do not use re-ingestion as archive repair.

## Storage, timing and limits

Retention adds full unique PDF versions, raw/repaired text, complete visible target
answers, traces and filesystem synchronization. The current store replays history
and verifies referenced payloads on updates. Storage and elapsed-time budgets on
representative papers remain a controlled-rollout prerequisite before changing the
default. This release deliberately leaves the choice explicit.

Only the extraction profile is advertised. Supplementary attachment retention
belongs to a later profile extension. Complete capture means its declared
inventory was observed or explicitly skipped; it does not establish scientific
truth or publication authority. Full grading evidence, remaining model phases,
losing drafts, import/pruning and general archive reconciliation remain later work.
Host-local locks and atomic writes provide process-crash recovery, with the
filesystem/sync durability limits in [the archive guide](ingest-records.md).

The live acceptance harness uses a synthetic PDF and stub models through the real
CLI. A local sentinel asserts zero provider requests. It exports the attempt,
deletes the corpus/database, verifies the portable archive and explicitly repairs
a missing checkpoint. The CI installed acceptance job runs it on Linux, macOS and
Windows:

```bash
python .github/scripts/retention_acceptance.py --expect-installed
```
