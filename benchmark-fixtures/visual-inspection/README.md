# Portable visual inspection benchmark

This example dataset contains **14 inspection cases, 84 answer checks, three
nomination cases and 13 PNGs**, all derived from six CC-BY-4.0 PDFs
bundled in this repository. It runs directly from a fresh clone. The dataset can
be redistributed and adapted with attribution; see [LICENSES.md](LICENSES.md) for the grants, source
identities and changes. The Python utilities retain the framework's license.

Reference answers were checked by the active coding assistant against PDF
renders. **Independent human adjudication is pending.** These are diagnostic
fixtures, not a representative estimate of corpus performance. Report model
identity, usage and latency with any evaluation. Visual ingest remains optional
and off by default.

## Cases

| ID | Source | What it tests | Smoke |
|---|---|---|---|
| v01 | MGA Table 3, p17 | Runtime columns, h:mm:ss and literal printed memory units | yes |
| v02 | MGA Table 2, p10 | Grouped rows, N50 exception and missing-value dashes | |
| v08 | Clinical Figure 2, p29 + p35 | Separate caption/plate, risk-difference sign, CI and denominators | yes |
| v09 | Clinical Figure 3, p29 + p36 | Plotted medians, caption statistics and min/max whiskers | |
| v11 | MGA Table 2, altered p10 | Abstain on a masked cell while reading its neighbor | |
| v12 | Clinical caption p29 only | Detect absent artwork; retain caption enrollment count | |
| v13 | Tangermeme Figure 1, p2 | Printed timing bars, seconds and run count | yes |
| v14 | Tangermeme Figure 1, p2 | Logarithmic axis, diagram legend and unprinted median | |
| v15 | Cell-identification comparison Figure 1, p5 | Dense heatmaps, F1 versus unlabeled percentage and gray cells | yes |
| v16 | Cell-identification comparison Figure 2, p7 | Dataset order, grayscale encoding and unlabeled bar | |
| v17 | DeepCRISPR Figure 3, p9 | ROC/PR legend values and axis binding | yes |
| v18 | DeepCRISPR Figure 2, p6 | Independent-dataset bars, training axes and exact-value abstention | |
| v19 | scAnnotate Figure 3, p7 | Rank colors versus accuracy sizes; unprinted exact accuracy | yes |
| v20 | scAnnotate Figure 6, p8 | Binary mosaic legend, selected cells and absent row counts | |

Pages are 1-based physical PDF pages. Every inspection input is a full page at
110 DPI, with at most two images and three million total pixels per case. The
masked image is an explicitly labeled controlled occlusion.

| Nomination ID | Input | Expected behavior |
|---|---|---|
| n01 | Results prose already supplies both medians | No extra visual request |
| n02 | Caption-only Table 2 excerpt | Request `table:2` |
| n03 | Figure 2 and 3 caption titles only | Request both figures; do not invent artwork pages |

The nomination cases are bounded controlled excerpts testing selection behavior,
not full claim-extraction replays. They include a no-extra-call case to test
restraint.

## Files

- [cases.json](cases.json) and [nomination-cases.json](nomination-cases.json):
  questions, source context and inventory; no reference answers.
- [manifest.json](manifest.json): source PDF hashes, image hashes/dimensions,
  page numbers, license-evidence pages/DOIs and the six smoke IDs.
- `assets/`: 13 model-input PNGs, totaling about 2.7 MB.
- `reference/`: separate [answers](reference/expectations.json),
  [nomination expectations](reference/nomination-expectations.json),
  [mask transformations](reference/transformations.json) and
  [annotation notes](reference/REVIEW.md). Never send these to the candidate model.
- `build.py`: rebuilds the data from the six bundled, hash-verified PDFs.
- `benchmark.py`: offline integrity validation, packet preparation and scoring;
  it has no network or model transport.

## Prepare and score

From the repository root, validate the shipped data and prepare six smoke
requests into gitignored development space:

```sh
python benchmark-fixtures/visual-inspection/benchmark.py validate
python benchmark-fixtures/visual-inspection/benchmark.py prepare \
  --suite smoke --output dev/visual-benchmark/packets/smoke-vision
```

`requests.jsonl` contains OpenAI-compatible text/image messages with embedded PNG
data URLs. `limits` is harness metadata, not an API field. A separate runner must
choose an image-capable model, supply the endpoint-appropriate completion limit,
enforce the 30-second deadline and record exact model/configuration. Use one
outbound request per case, at most 1,200 completion tokens and no automatic retry
or re-prompt. Packet preparation makes no model calls.

Save parsed responses as JSONL or a JSON array. The prepared
`response-template.json` gives every requested question. An answer has this shape:

```json
{
  "case_id": "v01",
  "answers": [{
    "question_id": "human_mga_elapsed",
    "status": "observed",
    "value": "<value read from the image>",
    "unit": "h:mm:ss",
    "evidence": [{
      "image_id": "mga-p17",
      "pdf_page": 17,
      "ref_id": "table:3",
      "region": "Table 3 elapsed-time row, HUMAN/MGA column"
    }]
  }]
}
```

Return every requested question exactly once. Status is `observed`, `not_shown`
or `unreadable`; absent or masked values use JSON `null`. Preserve the requested
unit label. Attach actual stopwatch/API telemetry outside model output as
`telemetry: {wall_ms, input_tokens, output_tokens, request_count, cache_hit}`;
missing telemetry stays unknown rather than zero. Retain timeouts and missing
responses in the denominator.

```sh
python benchmark-fixtures/visual-inspection/benchmark.py score \
  --suite smoke --responses dev/visual-benchmark/responses.jsonl \
  --output dev/visual-benchmark/score.json
```

Use `--suite full` on preparation and scoring for all 14 inspection cases.
Use `prepare --track nomination` and `score --track nomination` for the three
selection cases. Nomination responses contain `case_id` and `visual_requests`,
each with `ref_id`, `priority`, `reason` and `question`; the prepared template and
system prompt specify the contract.

## Scoring and overhead

The scorer checks values, units, statuses, question identity and evidence-image
binding. A correct number attributed to the caption instead of the forest plate
fails. Dashes are not zero; hidden values require abstention. The report includes
atom accuracy, case pass rates, unsupported-value assertions, measured latency,
recorded token usage, missing responses and recorded budget violations.

Manually verify region descriptions and any extra scientific assertions: the
mechanical scorer cannot certify their semantics. Nomination questions also need
manual review for relevance and invented locations. See the
[review notes](reference/REVIEW.md) before using the scores to choose a default.

Start with the six smoke requests, one per source paper. A text-only packet can be prepared with
`--mode text-only`, but it is an image-grounding negative control, not today's
complete ingest baseline. A real incremental-utility comparison must supply the
exact extracted text today's ingest would see, then compare recovery and actual
cost against inspection of the same source regions. Values already settled by
text do not establish a benefit from an extra call.

## Rebuild and limitations

Install the project's dependencies, then run:

```sh
PYTHONPATH=. python benchmark-fixtures/visual-inspection/build.py
python benchmark-fixtures/visual-inspection/benchmark.py validate
python -m pytest tests/test_visual_inspection_benchmark.py -q
```

Rebuilding refuses changed source PDF hashes, because old annotations must not
silently be applied to a new version. Reprepare any evaluation packets after
changing or rebuilding the fixtures. Keep packets, responses and run logs under
gitignored `dev/`; only fixed dataset artifacts belong here.

The cases share pages and cover six papers. The suite does not validate
automatic artwork routing, full ingest, caches or promotion gates. The clinical
paper's captions and unlabeled appended plates need explicit routing tests before
automatic rollout. Add independently reviewed, openly licensed papers before
making broad coverage or default-setting decisions.
