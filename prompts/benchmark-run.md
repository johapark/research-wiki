# Benchmarking a model on a fixture

**Trigger**: user asks to test/benchmark an LLM (a provider or `config/models.*.yaml`)
by ingesting a `benchmark-fixtures/` paper and assessing the result.

The goal is to **measure a model's ingest quality without mutating the real wiki**.
A benchmark ingest is a throwaway — it must never leave a page in `wiki/`, a line in
`index.md`, or a row that outlives the run.

## Golden rule — always `--force-sandbox`

```bash
RW_MODELS_CONFIG=models.<name>.yaml \
  researchwiki agent ingest benchmark-fixtures/pdfs/<stem>.pdf -n 1 --force-sandbox
```

- `--force-sandbox` writes the authored page to `.agent-output/<stem>.md` and
  **never promotes to `wiki/` and never updates `index.md`**. This is the whole
  point — do NOT run a plain `agent ingest` for a benchmark.
- `RW_MODELS_CONFIG=…` inline selects the config under test. Do **not** edit `.env`
  (its `RW_LLM_BASE_URL` / key must already point at the right endpoint for that
  config — e.g. the Gemini OpenAI-compatible URL for any `models.gemini*`/`gemma`).
- `-n 1` (one author draft) keeps call volume low for rate-limited free tiers.
  Add `-w 1` only for multi-PDF batches.

## Assess

The authored page lands at `.agent-output/<stem>.md`. Judge it against
`benchmark-fixtures/<stem>.yaml` (the fixture declares the headline_claims,
capabilities, limitations, comparator ratios, and related_papers a thorough page
should capture). Score coverage per item (HIT / PARTIAL / MISS), weight by
`importance` (critical/high/normal), and check comparator_fidelity ratios
verbatim.

Record the `attempt_id` printed at startup and include
`researchwiki insights --attempt-id <id>` in the benchmark result when latency
matters. Report terminal wall time separately from summed phase work (drafts
may run in parallel), and use the per-step rows for phase comparisons. Do not
reconstruct timings with `researchwiki db query`.

**Prefer judging manually** (read both files, score directly). The automated
`researchwiki benchmark-fixture <stem> --llm` scorer resolves the page from
`wiki/`, so it does **not** see a sandboxed page — and its LLM judge inherits the
active `RW_MODELS_CONFIG`, so pointing it at a rate-limited free tier (e.g.
`gemini-3.5-flash`: 5 RPM / 20 RPD) 429s it into all-MISS noise. If you must use
the automated scorer, run it under a non-rate-limited judge config.

## Never leave artifacts

- Sandbox runs leave only `.agent-output/<stem>.md` — harmless, overwritten next
  run. Leave it or delete it; it is not in `wiki/`.
- **If you ever promoted by mistake** (forgot `--force-sandbox`), clean up fully:
  1. `rm wiki/<cat>/<stem>.md papers/<stem>.pdf`
  2. remove the `[[<cat>/<stem>]]` line from `wiki/index.md`
  3. **`researchwiki db rebuild && researchwiki reindex`** — BOTH are required.
     `db rebuild` alone purges only `state.db`; the deleted page still lingers in
     the tantivy index **and** the `.semantic-cache/`, so the next ingest's
     crosslink phase does semantic-neighbour lookup, hits the ghost, and crashes
     with `FileNotFoundError: wiki/<cat>/<stem>.md`. `reindex` rebuilds both.
