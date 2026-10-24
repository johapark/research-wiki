# Retained grading evidence and verification inputs

With [opt-in ingestion retention](ingest-retention.md) enabled, new attempts also
declare grading source, input, evidence, result and optional model-call inventory.
This covers draft grading, the requested support veto and post-publication claim
grading. Each invocation has a distinct `run_id`, role and attempt iteration.
The selected profile remains `extraction-v1`; full author/revision/selection tracing
belongs to a later increment. Existing archived attempts remain readable.

## What is retained

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

Implementation hashes identify the numeric, fidelity, retrieval, semantic
and support code available during the run. Configured model identity
does not establish a resolved weight revision; unknown revisions remain null.

## Full passages and exact check context

The database and existing grader reports still expose their bounded 500-character
supporting-text preview. The archive preserves the full retrieval candidates.
Numeric/negation checks retain the actual joined top-three neighborhood and the
separate full-PDF fallback. The full-text input is bounded by the grader's existing
80-page read limit, which is recorded explicitly.

The optional support judge currently receives that 500-character preview. Its
prompt builder additionally strips whitespace and caps each context at 1,200
characters. Both the supplied context and the exact generated model request are
retained, so a short judge context cannot be mistaken for the full source passage.
Retention does not make extra retrieval or judge calls to enrich these inputs.

Evidence is labelled by role. A BM25 hit is a `retrieval_candidate`; numeric
matching establishes value presence; negation parity is advisory; a support
verdict is a classifier observation. None of these automatically becomes a
reviewed scientific label. No hits, skipped cross-references, unavailable semantic
scoring, no eligible support context and failed/malformed judgments remain distinct.

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

## Inspection and portability

```bash
researchwiki records show <attempt-id> --grading --json
researchwiki records export <attempt-id> --output /path/to/new-export --json
researchwiki records --root /path/to/new-export show <attempt-id> --grading --json
```

`--grading` includes full grading payloads and their artifact envelopes, parent
links and source references. Ordinary `records show` keeps its compact manifest
and publication receipts. Export includes all referenced grading representations,
evidence, model calls and results. Verification/inspection works after the original
PDF, wiki pages, database and grading cache are removed.

Archive write failures use the existing latched capture error and publication
recovery policy. Evidence and support inputs persist before the judge dispatches.
A response already received settles known usage once even if archival persistence
fails. Missing output does not trigger an automatic second model call.

Validation uses long synthetic passages, actual source offsets, deterministic
semantic vectors and mock API/relay responders. The installed acceptance harness
uses PDFium on a real synthetic PDF, exercises the shared support client, deletes
source/cache data and inspects the portable export:

```bash
python .github/scripts/grading_acceptance.py --expect-installed
```
