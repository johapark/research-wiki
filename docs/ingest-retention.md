# Ingestion retention guide

Opt-in retention preserves the source versions, extraction output and grading
evidence observed during ingestion. Use this guide to enable it, inspect/export
records and recover from capture failures. The [developer reference](ingest-dev-reference.md)
describes archive contracts and capture implementation.

The `extraction-v1` profile applies to `researchwiki add`, `agent ingest`, their
batch workers, the synchronous ingestion operation and durable local ingestion
jobs. The legacy digest-only `researchwiki ingest` workflow is outside this profile.

## What is saved

- Immutable original PDF snapshots and displaced canonical PDFs when replaced.
- Page-aware extracted text, full target-extraction answers and candidate/filter traces.
- Grading inputs, complete retrieved passages, check observations and optional support judgments, detailed below.
- Every produced author/evolve/DEBUG draft, critiques, grading links and selection decisions, including losing or rejected drafts.
- Attempt identities, completion states and publication receipts.

| Layer | Retained information |
| --- | --- |
| Page/draft input | Original draft Markdown when supplied, exact parsed page body, parsed claims and cross-reference flags, requested grading options |
| Source representations | Original PDF identity, the exact page-repaired text behind the retrieval index, raw full-text numeric input, and the transformed numeric full text |
| Retrieval | Ordered top-five candidates and top-ten semantic candidates actually returned; ranks/scores, chunk IDs, query text and parser fallback, chunk/page limits, index/cache/extractor versions |
| Source mapping | Full normalized retrieval text, full source slice, character offsets into the archived index representation, page ranges and section labels |
| Numeric check | Raw and transformed claim/neighborhood inputs, full-text input reference, extracted tokens, unmatched values and whether matches came from retrieval or elsewhere in the PDF |
| Negation check | Exact claim/neighborhood, matched tokens, rule/pattern and advisory mismatch result |
| Semantic check | Requested/available/completed status, configured model/device, unknown model revision, token limit when exposed, cache-use status, ordered candidate cosine scores, chosen index and actual vectors |
| Coverage | Salience and target fixtures, complete scoring inputs and results before bounded missed-anchor transport |
| Support check | Eligible/skipped claims, original positions and classifier item IDs, exact provided contexts, expected/checked counts, returned verdicts or failure class |
| Support model call | Logical inputs/schema, observable dispatches, complete visible answers, actual route/model-origin/usage metadata, relay validation/retry/reuse and parser-input text |
| Results | Full fidelity report and final aggregate, completed/failed/interrupted state and available partial observations |

Retention records what the engine observed. Retrieved passages and classifier
judgments are not automatically reviewed scientific labels. Completeness means
that the declared inventory was retained or explicitly skipped; it does not
establish scientific truth.

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
researchwiki records show <attempt-id> --grading --json
researchwiki records show <attempt-id> --lineage --json
researchwiki records verify <attempt-id> --json
researchwiki records export <attempt-id> --output /path/to/new-export --json
researchwiki records --root /path/to/archive verify <attempt-id> --json
```

List/show/verify do not initialize an archive or repair records. Verification
checks event history, identity, payload hashes and checkpoint correspondence.
`show` includes publication receipts. `--lineage` adds full draft/critique/decision
payloads; it can be combined with `--grading` for full grading evidence. Exports require a new destination and copy
only the chosen attempt, its events/checkpoint, archive identity and referenced
blobs. Exported attempts verify without the original corpus, databases, caches,
mailboxes or mount path. These commands can inspect an explicit archive outside
an initialized library.

## Audit historical SQLite ingest history

Before importing older history, inventory the selected database and the former
machine-wide `state.db` without initializing, migrating, or rewriting either:

```bash
researchwiki records inventory --legacy --json
researchwiki records inventory --legacy --db-path /path/to/state.db --json
researchwiki records inventory --legacy --library-root /path/to/library --json
researchwiki records backup --legacy --db-path /path/to/state.db \
  --output /path/to/new-backup-directory
```

The report summarizes per-attempt iteration roles, retained draft/score/critic
row counts, target-claim counts and decisions where those counters were recorded,
timestamps, source filenames, and source-hash coverage in the current claims
table. It does not print draft text, critic notes, or other prose. Claim-table
hashes are rebuildable state and are not proof of which PDF version a historical
attempt used; the history rows themselves do not contain the original PDF hash or
full target-claim list. `related_evidence` summarizes the selected library's job
manifests/events/publication ledgers, mutation journals, relay mailbox file counts,
and top-level sandbox, grading-cache, and ingest-run availability. It returns
metadata and counts only; draft, event, journal, and relay payload contents are
not included in the report. This is a best-effort per-file scan, not a single
cross-store snapshot. `--library-root` selects the library for this filesystem
inventory. Consumed relay payloads have already been deleted and cannot be
inventoried retrospectively. Preserve the original database and its job/recovery
state; the backup command captures one SQLite database only. It uses SQLite's online
backup API so committed WAL content visible to SQLite is included in a consistent
snapshot. It refuses an existing output directory and writes `state.db` plus a
`manifest.json` containing the snapshot method/time, database checksum, table
inventory, and integrity-check result. The manifest marks the backup complete;
an output folder without it is incomplete. This does not preserve external job
directories, relay mailboxes, or publication journals, which must be handled
separately before migration.

## Evidence and check limits

The archive keeps complete retrieval candidates. Existing database/report previews
remain capped at 500 characters, and the optional support judge currently receives
that preview. Its prompt builder also strips whitespace and caps contexts at 1,200
characters. Retention saves both the supplied context and generated request, without
extra retrieval or judge calls.

Numeric and negation checks retain their actual top-three neighborhood; numeric
checks also retain the separate full-PDF fallback. The existing 80-page text-read
limit is recorded. No hits, skipped cross-references, unavailable semantic scoring,
missing support context and failed judgments remain distinguishable.

Source spans, vectors and check contexts travel with an export, so inspecting them
does not depend on the original PDF, wiki, database or disposable grading cache.
PDF chunk cache version 3 adds offsets and text identity; old caches rebuild on
next use. Chunk text and scoring rules keep their existing behavior.

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

Candidate passages, source representations and semantic vectors add storage and
verification I/O. Candidate text/vectors are deduplicated within a grading run;
identical payloads reuse content-addressed blobs. There are no extra model calls
for retention.

Only `extraction-v1` is advertised, with grading inventory declared on new attempts.
Metadata reconciliation and crosslink results, plus author/critic/revision and
commit-time model calls are retained through API and chat-relay adapters. The
exact rendered Markdown page is retained once as a content-addressed representation;
staged and committed publication receipts bind it by artifact ID and digest, so
the ingest-time page remains recoverable after the corpus page changes or disappears.
Conditional phase and model-call omissions now have individual reason artifacts
in `records show --lineage`; an unexplained missing call remains unknown. Supplementary
attachments, import/pruning and general archive reconciliation remain later work.
No `full-ingest-v1` profile is enabled yet, pending acceptance coverage for the
conditional paths and post-operation terminal status.
Host-local locks and atomic writes support process-crash recovery, subject to the
[filesystem and sync limits](ingest-dev-reference.md#storage-and-concurrency-limits).
