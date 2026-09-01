# Beta compatibility and migration contract

This contract applies from the first beta release onward. It covers the surfaces
operators and automation persist: CLI command/flag spellings, exit-code meanings,
structured JSON, wiki frontmatter, and ingest phase-role identifiers. Python
modules remain internal.

## Deprecation window

A published CLI command or flag, JSON key, or frontmatter field is not removed or
repurposed immediately. A deprecation must:

1. be recorded in `researchwiki/data/deprecations.yaml` and the changelog;
2. remain usable for at least 90 days;
3. coexist with its replacement through the entire next minor release line; and
4. be removed no earlier than the second subsequent minor release.

For example, a surface deprecated in `0.4.x` remains throughout `0.5.x` and can
be removed only in `0.6.0` or later, after the 90-day date has also passed. Both
the version and date gates must be satisfied. Security or data-loss defects may
disable unsafe behavior sooner, but the release notes must identify the exception
and provide the safest available migration.

Deprecation notices go to stderr. Machine-readable stdout—including `--json`—must
remain parseable and byte-compatible apart from documented additive fields.

## Surface rules

### CLI and exit codes

Renamed commands and flags retain aliases for the full window. Aliases delegate
to the same implementation and preserve stdout, JSON shape, and exit behavior.
The general exit-code meanings remain stable during beta: `0` success, `1` user
input, `2` environment, and `3` internal failure. A command family with a
documented specialized gate result keeps that contract; its meanings cannot be
silently reassigned.

### JSON

Adding a key is compatible. Renaming a key requires dual emission of old and new
keys with equivalent values for the full window; removing or changing the type or
meaning of an existing key is breaking. Versioned artifacts use a new
`schema_version` for incompatible shapes rather than changing an old schema in
place. Deprecation warnings never enter JSON stdout.

### Frontmatter

Readers accept both deprecated and replacement fields for the full window.
Writers emit the replacement form only. A rename or semantic change ships with an
idempotent migration that plans first, refuses ambiguous values, preserves
unrelated fields, validates file identity before applying, and creates a backup.
Fields are never silently reinterpreted.

The beta provenance upgrade follows that rule through `researchwiki migrate
provenance`: planning creates an editable review manifest without changing wiki
pages; applying requires every page to be resolved. Exact telemetry is recovered
when available. Otherwise a maintainer either attests an exact model, skips the
page, or explicitly acknowledges that no model was recorded:

```yaml
author_provenance: legacy-unrecorded
provenance_acknowledged_at: 2026-09-01
```

Acknowledgment is an honest terminal state, not a fabricated `author_model`.
`lint` reports it separately from actionable missing provenance.

### Persisted phase-role identifiers

Values stored in `ingest_iterations.role` and public model-routing phase names are
append-only. They are never renamed or recycled, even after their implementation
function changes, because `db rebuild` cannot reconstruct historical telemetry.
New identifiers may be added; old identifiers remain readable permanently.

## Maintainer checklist

Before merging a public-surface change:

- update the deprecation ledger and changelog when replacing a surface;
- add compatibility tests for the old and new forms together;
- verify notices use stderr and JSON remains clean;
- provide a dry-run/plan and backup for persisted-data migrations; and
- treat removal as a breaking release under `CONTRIBUTING.md`.
