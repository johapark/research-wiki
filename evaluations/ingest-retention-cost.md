# Ingestion retention cost experiment

Date: 2026-10-25. Scope: R2 of the [retention plan](ingest-retention-plan.md).
Retention stays opt-in. These synthetic measurements do not set a default-on policy.

## Method and limits

Run from the checkout root:

```bash
python .github/scripts/retention_benchmark.py --output /tmp/retention-cost.json --repeats 3
```

The benchmark uses text-only synthetic PDFs with 2, 12 and 48 pages. Short and
typical cases have two author slots; the long case has four. Real engine wrappers
extract the PDF, extract targets, author and grade each candidate, select a winner,
request a critique, produce/grade a revision, and record its deliberate rejection.
Slots execute serially to isolate archive cost; parallel-writer correctness is
tested separately. Scientific results from this fixture have no research meaning.

OpenAI-compatible requests receive in-process HTTP response objects; chat-relay
requests receive local mailbox replies through the real adapter. Neither mode
contacts a model or external service. Fixed API usage and the relay's actual
unknown/zero usage convention are preserved, not estimated as training labels.
Capture-off/required pairs must have identical source text, targets, drafts,
scores, winner, model calls, iteration usage and budget settlements. All inputs,
archives, grading indexes and exports live in temporary directories; no corpus,
SQLite database or real mailbox is modified.

Each repetition starts a new subprocess per mode, excluding interpreter/module
imports from stage timers. Its first run has an empty grading cache; the second
reuses that cache with a new attempt in the same archive. "Cold" means cold
application cache, not a flushed OS cache or cold machine. Three repetitions
alternate mode order. Stage counters are instrumented in both modes; times are
wall-clock observations on one host, not CI performance assertions.

`bind` includes snapshot creation. `work` covers the scientific phases and their
capture. `finalize` includes omission records, closure and the invocation audit.
`verify` includes structural/blob verification plus the independent coverage audit;
`export` includes a portable copy and verification of the copied attempt. Source
parsing and grading index construction remain inside `work`; cold/warm results
separate them from repeated capture work. Publication, metadata discovery,
semantic embeddings, support judging, supplementary files, scanned/image-heavy
PDFs, slow/network storage and batch concurrency are outside this experiment.

Storage counts logical file bytes, not allocated blocks. Per-attempt unique blobs
are categorized once as source, text, model, evidence or other payload; events and
the checkpoint are separate. Referenced bytes include all repeated artifact/source
references. Warm archive totals include both attempts; added bytes measure the
second attempt's marginal growth. Peak RSS is the subprocess high-water mark,
including imports and both cache runs; it is not incremental per-attempt memory.
Windows reports RSS as unavailable through this harness.

Hash counters cover archive blob writes and verification/read hashes, not every
hash in the engine or event-digest computation. Event JSON reads and projection
events count logical work, not physical disk reads. Counters include destination
work during export. The raw JSON retains every repetition and stage counter.

## Evaluation budget

For this text-only fixture, the provisional manual opt-in targets are at most
10 seconds of additional warm processing/closure for short/typical inputs and
30 seconds for long inputs, plus at most 5 MiB of attempt data beyond unique PDF
bytes. These are engineering targets for this experiment, not user guarantees,
an automatic quota, or a justification for discarding observations. No extra
model calls or changed outputs/usage are acceptable. Broader routine-use approval
still needs declared-input/span audits and representative real-use measurements.

Any optimization must preserve event replay, payload verification, explicit stale
checkpoint repair, corruption detection and serialized writers. The existing
archive remains authoritative; a cache/catalog or async durability design is not
part of R2.

## Results and decision

The baseline and optimized versions used the same fixtures. The baseline store
source SHA-256 is
`75df1256015e51a76d4a8ebe6a63979e2977f4ec351e73725cccfdfc67a862f2`
(`33a432c`).

The bounded change reuses the already-validated full-history projection when a
checkpoint claims the current sequence. The checkpoint must still equal that
projection. Stale checkpoints still need their own exact prefix projection;
missing checkpoints still require explicit repair before writes. Every call
rereads/validates events and rehashes referenced payloads under the same writer
lock. Nothing is cached across calls and the integrity contract is unchanged.
The optimized store source SHA-256 is
`dab82ab66c068ce431625536d375271fa4581ac210ae08ada5f567830212f5d9`.

The count-based regression failed on the baseline at both tested history sizes,
then passed after the change. Archive regressions also check that a previously
verified blob/checkpoint can be corrupted and is detected on the next operation,
and that stale checkpoints are checked against the correct prefix. The existing
process-writer, interrupted-checkpoint, repair and relocation tests remain part
of validation.

Host: `macOS-26.6.2-arm64-arm-64bit`, Python **3.12.8**. Three repetitions per
case/route/mode, each with cold and warm cache runs: **144 phase-fixture runs**
across baseline and optimized versions. Every capture-off/required pair and every
before/after output fingerprint matched. API call counts were 5/5/7 for
short/typical/long; relay call counts were the same. Budget settlements and
iteration usage also matched within each route.

### Processing time

Warm processing includes bind + work + finalization. Values are median [min–max]
seconds; improvement compares required-mode medians. Off-mode figures are from
the optimized run. They exclude provider latency, so absolute overhead is more
useful than an off/on ratio.

| Case / route | Off, warm | Required baseline, warm | Required optimized, warm | Median improvement |
| --- | ---: | ---: | ---: | ---: |
| short / API | 0.03 [0.03–0.03] | 8.60 [8.49–8.64] | 7.29 [7.14–7.32] | 15.2% |
| short / relay | 0.04 [0.03–0.05] | 9.19 [9.17–9.37] | 8.99 [7.75–11.46] | 2.2% |
| typical / API | 0.11 [0.10–0.15] | 8.97 [8.97–9.27] | 7.80 [7.79–8.31] | 13.1% |
| typical / relay | 0.11 [0.11–0.11] | 9.42 [9.39–9.45] | 8.25 [8.23–8.27] | 12.4% |
| long / API | 0.57 [0.57–0.58] | 17.15 [17.10–17.22] | 14.92 [14.92–14.99] | 13.0% |
| long / relay | 0.58 [0.57–0.58] | 17.77 [17.75–17.94] | 15.57 [15.44–15.60] | 12.3% |

Optimized stage medians in seconds. Work cold/warm differs in application-cache
state; closure, verify and export columns use the warm run. Medians of components
need not add to the median of the total.

| Case / route | Bind | Work, cold / warm | Finalize | Verify + coverage | Export + checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| short / API | 0.024 | 3.959 / 3.871 | 3.336 | 0.124 | 0.422 |
| short / relay | 0.021 | 4.402 / 4.617 | 4.353 | 0.189 | 0.639 |
| typical / API | 0.022 | 4.747 / 4.208 | 3.565 | 0.135 | 0.448 |
| typical / relay | 0.022 | 4.653 / 4.503 | 3.723 | 0.138 | 0.465 |
| long / API | 0.022 | 9.853 / 9.687 | 5.215 | 0.196 | 0.644 |
| long / relay | 0.023 | 10.318 / 10.182 | 5.370 | 0.202 | 0.660 |

Cold off-mode work medians for short/typical/long were **0.166/0.265/0.767 s**
on API and **0.177/0.256/0.786 s** on relay. Cache warming saves a small amount
of parsing/index work; archive capture and closure dominate this fixture.

### Storage and memory

First-attempt logical bytes, in KiB, from one optimized repetition. Artifact
shape and sizes were stable across repetitions; these figures are not forecasts
for image-heavy PDFs or semantic vectors. Other means non-grading metadata,
parser/decision payloads and terminal/audit records.

| Case / route | PDF | Text | Model | Evidence | Other payload | Events | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short / API | 6.7 | 31.7 | 117.6 | 519.4 | 26.8 | 135.1 | 99.7 |
| short / relay | 6.7 | 31.7 | 136.3 | 512.4 | 21.8 | 139.3 | 102.9 |
| typical / API | 38.4 | 185.9 | 319.8 | 830.7 | 26.9 | 135.2 | 99.9 |
| typical / relay | 38.4 | 185.9 | 389.6 | 823.8 | 21.9 | 139.4 | 103.0 |
| long / API | 153.7 | 746.8 | 663.4 | 1393.6 | 37.2 | 194.3 | 144.5 |
| long / relay | 153.7 | 746.8 | 909.6 | 1382.0 | 30.6 | 198.5 | 147.6 |

MiB below. Referenced payload includes repeated source/artifact references; unique
payload excludes event/checkpoint files. First total includes metadata, events and
checkpoint. Warm added is actual marginal archive file growth for the second
attempt, sharing identical blobs. RSS values are medians of process peaks; imports
are included and variation prevents attributing small differences to this change.

| Case / route | Unique payload | Referenced payload | First total | Warm added | Peak RSS, off / required |
| --- | ---: | ---: | ---: | ---: | ---: |
| short / API | 0.69 | 2.52 | 0.92 | 0.85 | 95.2 / 108.3 |
| short / relay | 0.69 | 2.59 | 0.93 | 0.88 | 92.9 / 106.1 |
| typical / API | 1.37 | 12.09 | 1.60 | 1.30 | 96.5 / 113.4 |
| typical / relay | 1.43 | 12.53 | 1.66 | 1.40 | 95.4 / 109.9 |
| long / API | 2.92 | 66.84 | 3.26 | 2.12 | 100.2 / 122.0 |
| long / relay | 3.15 | 68.60 | 3.49 | 2.48 | 106.5 / 118.6 |

### Deterministic archive work

Warm bind + work + finalization, one repetition. Event counts, event JSON reads
and payload hash byte counts were unchanged by the optimization; projection work
was approximately halved. Hash/read counts are logical operations, not disk I/O.

| Case / route | Retained events | Payload hashed, MiB | Event JSON reads | Projection events, baseline → optimized |
| --- | ---: | ---: | ---: | ---: |
| short / API | 113 | 52.5 | 6,610 | 13,221 → 6,611 |
| short / relay | 116 | 54.0 | 6,961 | 13,923 → 6,962 |
| typical / API | 113 | 120.1 | 6,610 | 13,221 → 6,611 |
| typical / relay | 116 | 128.0 | 6,961 | 13,923 → 6,962 |
| long / API | 155 | 405.6 | 12,343 | 24,687 → 12,344 |
| long / relay | 158 | 445.5 | 12,820 | 25,641 → 12,821 |

### Decision

- Keep the bounded projection reuse. It cuts repeated projection work without
  weakening verification, changing archived bytes or altering scientific work.
  The typical/long timing improvement was approximately 12–13%; the short API
  improvement was 15%. Short relay was noisy (2% median improvement, overlapping
  ranges), so no reliable speedup is claimed for that case.
- Median warm overhead meets the provisional 10/30-second targets and each first
  attempt stays below 5 MiB beyond PDF bytes. One short relay repetition took
  11.46 seconds versus 0.05 seconds off, exceeding the 10-second target. These
  results support bounded manual experimentation, not a latency guarantee or
  routine/default-on approval. Retention remains off by default.
- Repeated full-history reads and payload checks remain the principal scaling
  limitation. Closure alone takes about 3–5 seconds here. Event reads still grow
  superlinearly with attempt size; this change removes a duplicate projection,
  not that overall scaling. Do not add more capture or promise cheap large runs
  based on this experiment.
- No catalog, unverified cross-call cache, event batching, background recorder,
  compression or pruning is introduced. Revisit the integrity/work boundary only
  if representative opt-in use demonstrates that the remaining cost is excessive.
  Any future batching/cache proposal needs an explicit integrity contract and
  corruption/concurrency/recovery evidence.
- Continue with R3: declare the input scope and independently audit known retained
  source/span formats. Real PDFs, semantic/support modes and slower storage still
  need a bounded release evaluation before recommending routine use.

To reproduce the baseline, use the same harness in a checkout of `33a432c`
(the harness must be copied there because it was added afterward). Keep other
load low and retain the JSON from each run; do not rerun selectively to remove
slow samples.

## Validation

- The replay count regression failed on the baseline at both tested sizes.
- All 31 focused archive tests passed after the optimization, including process
  writers, corrupt events/blobs/checkpoints, stale/missing checkpoints, interrupted
  writes and explicit repair.
- All four source acceptance harnesses passed: archive, extraction (API/relay),
  grading and live retention/publication with portable export.
- Full pytest: **4,391 passed, 10 skipped**. Ruff and whitespace checks passed.
- All 72 before/after run pairs preserved outputs, budgets, call/event counts and
  capture hash/read counts. Measurements use local/mock responders, not live models.

R1's installed-wheel acceptance remains historical evidence; this packet did not
repeat that wheel build or run the cross-platform CI matrix.
