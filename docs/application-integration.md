# Application integration

Developer reference for embedding the ingestion engine and managing local jobs.
For everyday use, start with the [agent-guided overview](../README.md).

## Ingestion operation

`researchwiki.operations.ingest(request, event_sink=...)` runs the same single-PDF
operation as `add` and `agent ingest`, returning a versioned `IngestResult`.
Construct requests with `IngestRequest` and `IngestOptions`; wire values round-trip
through `to_dict()` / `from_dict()`. Results distinguish publication, reviewable
drafts, failed attempts, check outcomes, and incomplete post-processing.

Run it in a process already configured for the request's absolute `library_root`
and provider profile. It validates that selection without changing CWD or loading
a different environment. Use separate processes for different libraries/profiles.
API and chat-relay routing remain phase-configured; relay handoffs stay visible
and still need a connected responder. An event callback receives `IngestEvent`
objects containing lifecycle and model/relay progress without paper prose or
credentials. Events use schema version 2; readers also accept version 1 model
events. Requests and terminal results remain at version 1. Phase events identify
each invocation and parallel draft slot. `publication_committed` is emitted only
after the canonical file transaction commits.

The operation performs no automatic retry. Uncertain publication requires
reconciliation; an optional-hook failure preserves a successful publication.
Projection status is `current` only when database repair, keyword/semantic
indexing, and required claim-grade persistence report completion. A saved paper
with unfinished maintenance reports `maintenance_pending`. Unobserved existing
papers still report `unknown`. After a known commit, an interrupted maintenance
step returns a published result with repair guidance rather than a retryable CLI
ingest failure. The local jobs layer below can explicitly reconcile publication
after worker loss without repeating authoring.

## Local jobs

`researchwiki.jobs.JobSupervisor` adds durable local submission, isolated worker
startup, status/events, queued cancellation, and explicit retry. Submission is
idempotent by library, request ID, and request fingerprint. Each worker selects
its own library/profile; the supervisor does not switch the application's CWD
or provider environment. Records live outside the checkout in the existing
per-library runtime location, independently of the rebuildable database.
`RESEARCHWIKI_JOBS_ROOT` selects an alternate base, still namespaced per library.

Use `submit_ingest(request)`, `start_job(job_id)`, and `get_job(job_id)` on the
supervisor. `read_events()` and `pending_relays()` expose progress and exact
active handoffs. `wait(handle)` blocks; relay work still needs a connected
responder while the process waits. Terminal stderr handoffs remain visible.
No daemon or automatic retry is installed. `cancel_queued_job()` prevents work
only before the worker claims it. `retry_job()` creates a new job for finished,
known-unpublished work; source/configuration changes are recorded.

`refresh_job()` detects lost ownership using the worker's OS lock, rather than
PID liveness alone. Worker loss leaves publication unknown and retry blocked.
Call `reconcile_job(job_id)` to inspect durable publication evidence and recover
mutation journals in an isolated library/profile process. A confirmed rollback
or pre-publication stop permits an explicit fresh retry. A confirmed commit
preserves publication and repairs the database, indexes, and claim grades;
unavailable maintenance remains pending. Missing/unsupported evidence stays
unknown. Events and matching filenames never establish publication. Recovery
does not repeat provider calls, answer relay prompts, run optional model hooks,
or overwrite changed canonical content.

The Community CLI exposes the same controls:

```bash
researchwiki jobs submit inbox/paper.pdf --library-root "$PWD" --request-id paper-001 --json
researchwiki jobs run JOB_ID --library-root "$PWD" --json
researchwiki jobs show JOB_ID --library-root "$PWD" --json
researchwiki jobs reconcile JOB_ID --library-root "$PWD" --json
```

`submit` and `retry` only enqueue; `run` owns a foreground worker with visible
relay handoffs. `list`, `show`, and cursor-based `events` inspect durable state;
`cancel` applies only to queued work. Explicit library selection works from
outside the checkout, and profiles load only in workers. See the
[job command guide](./jobs.md) for request/options files, profiles, JSON,
exit codes, and recovery behavior.

## Explicit evidence selections

Search, claims, PDF lookup, synthesis scaffolding, and bibliography export accept
`--scope-file selection.json` to restrict evidence before ranking and model context
construction. Empty selections remain empty; unscoped operations retain their
existing behavior. Applications can supply versioned selections without adding a
project database or commercial dependency to the engine. See the
[evidence-scope guide](./evidence-scopes.md).

Research Q&A is handled by the chat agent using the existing search, claims, and
PDF-reading tools, following the grounding rules in [CLAUDE.md](../CLAUDE.md).

`researchwiki compose "question" --title "Title" --scope-file selection.json`
authors a complete theme-based synthesis with mandatory page and support checks.
Preview first; `publish-synthesis` explicitly journals a verified receipt into
the wiki. See the [synthesis authoring guide](./synthesis-authoring.md).
