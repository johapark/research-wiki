# Extraction retention hooks

This is Stage 1's extraction slice from the
[retention plan](../evaluations/ingest-retention-plan.md), built on the
[archive foundation](ingest-records.md). The recorder is explicitly injected for
tests and adapter acceptance. CLI ingestion, jobs, profiles and library defaults
do not select it. Required capture needs the following lifecycle slice before
operator activation.

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
reads. It verifies attempt/source identity. This binding is test-injected; the
runner does not instantiate the recorder. Source replacement checks at promotion
and preservation of displaced canonical PDFs remain lifecycle work.

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
result. The live author path does not inject this argument yet.

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

Activation still needs parent operation ownership through post-work, terminal and
early-exit inventory, profile/job configuration identity, publication guards and
receipt handoff, and operator inspection/export. The lifecycle must prevent capture
failure from becoming a publishable success.

## Validation

```bash
python -m pytest -q tests/test_ingest_extraction_retention.py
python .github/scripts/extraction_acceptance.py
```

The acceptance harness processes a synthetic PDF through the real extractor,
shared client, API adapter and relay validation/retry path using deterministic
local responders. It checks discarded candidates, relay cleanup and relocation
after original-source deletion, without external calls or corpus databases.
`--expect-installed` verifies execution against a wheel installed in another venv.
Both archive and extraction harnesses run in the installed acceptance CI job
on Linux, macOS and Windows.
These checks validate retention behavior, not scientific extraction quality or
live CLI capture activation.
