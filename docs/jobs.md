# Durable ingestion jobs

`researchwiki jobs` exposes the Community ingestion supervisor. It uses the
same native operation, verification gates, provider routing, and canonical
files as `add` and `agent ingest`. Existing commands remain available.

## Queue and run

Every command requires `--library-root` naming an initialized checkout with
its runtime assets. The command may run from another working directory.
`submit` queues one PDF; `run` starts its isolated worker in the foreground:

```bash
researchwiki jobs submit inbox/paper.pdf --library-root /path/to/research-wiki \
  --request-id paper-import-001 --json
researchwiki jobs run JOB_ID --library-root /path/to/research-wiki --json
```

Replace `JOB_ID` with the submitted response's `job.job_id`. The PDF path is
relative to the selected library unless absolute. A repeated submission with
the same request ID and payload returns the same job, including after a
successful ingest consumes its inbox source. A changed payload under the same
request ID is rejected. Submission does not start model work.

The caller owns the foreground process; there is no daemon or detached-worker
promise. API calls and chat-relay may be mixed by phase. Relay handoffs stay on
stderr and need a connected responder. Ctrl+C interrupts the owned worker and
returns its available state; it does not claim that canonical publication was
prevented. Reconcile that job before retrying. Completed model calls are not
resumed at their exact phase.

## Inspect, cancel, and recover

```bash
researchwiki jobs list --library-root /path/to/research-wiki --state queued --json
researchwiki jobs show JOB_ID --library-root /path/to/research-wiki --json
researchwiki jobs events JOB_ID --library-root /path/to/research-wiki --after-sequence 0 --json
researchwiki jobs cancel JOB_ID --library-root /path/to/research-wiki --json
researchwiki jobs reconcile JOB_ID --library-root /path/to/research-wiki --json
researchwiki jobs retry JOB_ID --library-root /path/to/research-wiki --json
```

`list`, `show`, and `events` read durable state without refreshing ownership or
modifying the library. List order is newest creation timestamp first. A running
record can outlive its worker; use explicit reconciliation to inspect its lease
and publication evidence. Events include a `next_sequence` cursor and exact
pending relay handoffs, but contain no prompt or draft body. Corrupt complete
event lines fail reading; an incomplete final line is ignored. A damaged event
stream does not prevent inspecting the manifest with `show`.

`cancel` applies only to queued jobs and is idempotent once cancelled. It
refuses running work. `retry` creates a new queued request/job linked to a
finished, known-unpublished parent; it never starts that job. Each explicit
retry creates a fresh record, so retain its returned ID. Published work and
uncertain publication cannot be retried as ingestion.

`reconcile` refuses a live owner. After worker loss it resolves durable
commit/rollback evidence and repairs local projections without replaying
provider calls, answering relay prompts, or rerunning optional model hooks.
A confirmed pre-publication stop permits an explicit retry; a confirmed commit
preserves publication. Missing/unsupported evidence stays unknown, and changed
or missing canonical artifacts require review. Reconciliation may leave
maintenance pending when local embedding weights or other required resources
are unavailable. Further details: [recovery](../WORKFLOW.md#what-gets-cached-what-gets-regenerated).

## Options, profiles, and storage

Direct PDF submission accepts `--options-file FILE`, containing the versioned
`IngestOptions` fields. Omitted fields keep their normal defaults. For example,
an intentional review run can use:

```json
{"promotion": "never", "n_drafts": 1}
```

For the complete request surface, use `submit --request-file FILE` instead of
a PDF argument. This accepts a version 1 `IngestRequest`:

```json
{
  "schema_version": 1,
  "operation": "ingest",
  "request_id": "paper-import-001",
  "library_root": "/absolute/path/to/research-wiki",
  "source_pdf": "/absolute/path/to/research-wiki/inbox/paper.pdf",
  "options": {"n_drafts": 1, "promotion": "auto"}
}
```

Request execution paths and auxiliary paths in options JSON must be absolute.
The request file's library must match `--library-root`. JSON file arguments
themselves resolve from the caller's working directory. Unknown fields,
unsupported versions, and invalid budgets are rejected.

Workers load the selected library's optional `.env`. At submission, use
`--provider-profile PROFILE` to select a profile relative to that library, or
the usual global `researchwiki --env-file PROFILE jobs submit ...` with a path
relative to the caller's directory. Choose one profile mechanism. Existing jobs
retain their stored selection; global `--env-file` is rejected for the other
job commands. The CLI parent never loads a library profile. Profile parsing,
credential precedence, and inherited-routing conflict rules remain the native
loader's responsibility. A valid profile can be queued without API credentials;
execution preflights its actual providers.

Job records default to
`~/.local/share/researchwiki/repos/<library-key>/jobs/`. `--store-root BASE`
selects another machine-local base, still namespaced by library; use the same
base for subsequent commands. `RESEARCHWIKI_JOBS_ROOT` may also select that
base. Job storage inside the checkout or its canonical content is refused.
Common flags (`--library-root`, `--store-root`, `--json`) may appear before or
after the subcommand. Records survive DB/index rebuilds. Preserve them when
retaining job history and recovery evidence.

## JSON and exit codes

`--json` emits one version 1 JSON envelope. Diagnostics, native worker progress,
and relay handoffs remain on stderr. Help output is ordinary text.

| Command | Success payload fields, besides `schema_version`, `command`, `error` |
| --- | --- |
| `submit`, `run`, `show`, `cancel`, `retry`, `reconcile` | `job`: version 1 `JobRecord`, including its typed request/result |
| `list` | `jobs`: array of job records |
| `events` | `job_id`, `events`, `next_sequence`, `pending_relays` |

`error` is null when the command action completes normally. Command failures
include `error.code`, `error.category`, and a safe `error.message`; a job's
operation result may independently carry its own error or gate reasons.
For example, inspecting a failed job succeeds as an inspection. A run's job
record remains available in stdout when ingestion fails or stops at a gate.
Parse errors may have `command: null`.

| Exit | Meaning |
| --- | --- |
| 0 | Successful action/inspection. `run`: published/already-present result or intentional `promotion: never` draft. Known publication remains successful even when maintenance needs repair. |
| 1 | Invalid request, incompatible action/state, automatic publication-gate rejection, validation/budget failure, or user interruption. |
| 2 | Environment failure or interrupted/unresolved execution. `reconcile` also returns 2 while publication is unknown or maintenance remains pending. |
| 3 | Internal failure; detailed traceback remains on stderr or in local worker diagnostics. |

An exit code alone does not describe publication. Read `job.result.publication`,
`projections`, `post_work`, `checks`, and `gate_reasons`. Retrieval/numeric
checks are distinct from entailment, and unavailable usage is not measured zero.
