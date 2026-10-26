# Retrieval fixtures

Hand-curated query/answer fixtures for evaluating *retrieval quality* —
complementary to the page-content fixtures in the parent directory.

**Page-content fixtures** ask: did the ingest pipeline extract the right
things from the PDF into the wiki page?
**Retrieval fixtures** ask: given a natural-language query, does the
embedding put the right items in the top-K?

Both run through the same harness (`researchwiki benchmark-fixture <stem>`).
Dispatch is by the YAML's `fixture_type:` field:

```
benchmark-fixtures/
  ├── {stem}.yaml               # fixture_type omitted → coverage (default)
  └── retrieval/
      ├── claims/<slug>.yaml    # fixture_type: claims  → claim-level retrieval
      └── pages/<slug>.yaml     # fixture_type: pages   → page-level retrieval
```

**Prerequisite for the bundled retrieval fixtures**: unlike content
fixtures (which score against a freshly-authored draft), retrieval
fixtures need the corpus indexed. Ingest the bundled OA papers first:

```bash
for f in benchmark-fixtures/pdfs/*.pdf; do
  researchwiki agent ingest "$f"
done
researchwiki db rebuild && researchwiki reindex
```

The bundled fixtures anchor **only** to CC-BY-4.0 stems in
`benchmark-fixtures/pdfs/` so they remain redistributable and portable — no
dependency on a personal wiki corpus. The per-paper license grant and
article-bound evidence are recorded in `../LICENSES.md`.

Corpus-specific fixtures may also be committed when they capture recurring
real-world queries. The harness preflights their anchors: if their source papers
are not present in a checkout, it reports the fixture as unavailable rather than
misreporting retrieval quality as zero.

## Comparing BGE and Harrier

Install the optional modern-model dependencies, then compare against the current
corpus without changing the production search indexes or claim grader:

```bash
pip install -e '.[harrier]'
researchwiki benchmark-retrieval --candidate-embedding harrier-270m \
  --out output/embedding-comparison/harrier-270m.json
researchwiki benchmark-retrieval --candidate-embedding harrier-0.6b \
  --out output/embedding-comparison/harrier-0.6b.json
```

Aliases resolve to `BAAI/bge-small-en-v1.5`,
`microsoft/harrier-oss-v1-270m`, and `microsoft/harrier-oss-v1-0.6b`.
Both Harrier checkpoints use their packaged `web_search_query` prompt for
queries, unprompted documents, packaged last-token pooling, and normalized
vectors. An explicit `--candidate-query-prefix` replaces the automatic prompt.
The existing single-fixture A/B command accepts these aliases too.
An optional live text query uses the same model and isolated cache:

```bash
researchwiki search "adaptive retrieval with reflection tokens" \
  --embedding harrier-270m --mode hybrid --json
```

`search --embedding` defaults to a 2,048-token cap and CPU float32. It affects
only that text query; ordinary search and claim grading keep their existing
defaults. Use `--embedding-max-seq-length N` to change the cap. See-Also stays
on the default backend. A first query may build the selected model's page cache.

The suite runs BM25, semantic, and hybrid retrieval on the same available
fixtures. Its JSON records skipped expected anchors, inactive negatives,
per-fixture ranking changes, corpus and fixture fingerprints, model revision,
dimensions, dtype, context limit, parameter bytes, cache reuse, warm query
encoding median/p95, and document-sample throughput. Warm timings exclude model
loading, corpus embedding, and search. Build durations are labelled as new
encoding or cache reuse so they cannot be mistaken for comparable throughput.

Inference defaults to CPU and float32 for both models. Use `--dtype auto` to
retain a checkpoint's native dtype; CPU bfloat16 can be slower on some hosts.
Harrier batches default to 8 and BGE to 32; `--batch-size N` overrides both.
Each model retains its native context limit. A second run with
`--max-seq-length 512` caps both to compare under the same token limit (the
tokenizers still differ). `--device`, `--cache-folder`, `--timing-repeats`,
`--fixtures`, and `--backends` control the run explicitly. No LLM calls occur.
For a practical CPU run over research pages, `--max-seq-length 2048` keeps
BGE at its native 512-token limit and caps Harrier at 2,048 tokens.
Bookkeeping pages (`meta` and `dashboard`, including the full log and catalogue)
are excluded from both lexical and semantic page results.

Benchmark vectors live in separate atomic `.retrieval-cache-*/` bundles. Their
signatures include the full model ID, resolved revision, dtype, context limit,
document prefix, and every current claim/page input. A page edit or claim
addition/deletion invalidates its cache. Legacy matrix/JSON pairs are ignored.
If only the context cap changes, known-model vectors can be reused after
tokenization verifies that every document fits both caps without truncation.
Current BGE production vectors may seed the baseline read-only when their text
hashes and default representation match; altered contexts and instructions
cannot reuse them. The suite refuses a corpus change between comparison arms.

Eight additional corpus-native fixtures cover late interaction, reflection
tokens, off-target enumeration, scientific novelty detection, long-read
alignment, human pangenomes, single-cell integration, and genome language
models. They need the corresponding papers ingested locally; a fresh clone
reports them as unavailable rather than inventing misses.

## When to add a retrieval fixture

- **Before changing the embedding.** Curate ~10 fixtures spanning the query
  intents you actually issue (`claims "topic"`, `search "topic"`,
  `search --like X`). Run the existing embedding to set a baseline; then
  swap and rerun.
- **When a query keeps returning the wrong thing.** Encode the failure
  mode as a fixture so it becomes a regression test.
- **When you ship a new retrieval architecture** (late-interaction, graph
  RAG, query rewriting). Same fixture set lets you compare cleanly.

Aim for ~10–15 fixtures across query intents. Fewer than 5 is too thin to
detect a swap-effect above noise.

## Claim-level fixture format

```yaml
fixture_id: crispr-cas9-off-target-ml-prediction
fixture_type: claims
query: "machine learning prediction of CRISPR Cas9 off-target activity"
notes: |
  What the query is really asking for, and what the failure mode looks
  like. Free text — not scored, but the curator's voice helps later
  triage when scores move.
k: 10  # top-k cutoff for nDCG / MRR

# Anchors keyed on (paper_stem, section, position) — matches the claims
# DB schema. Lookup with `researchwiki claims --by-stem <stem>` to grab
# anchors quickly. Importance gain weights for nDCG: critical=3, high=2,
# normal=1.
expected_claims:
  - paper_stem: lin-2018-off-target-predictions-in-crispr-cas9-gene
    section: key_contributions
    position: 0
    importance: critical
    rationale: "Why this anchor matters — for the curator and the diff log."

# Negative anchors. Stems that share tokens with the query but don't match
# the intent. Caught false positives are diagnostic of token-overlap-without-
# semantics — exactly what a domain-trained embedding should disambiguate.
must_not_appear:
  - paper_stem: brixi-2026-genome-modelling-and-design-across-all-domains
    rationale: "Evo 2 — DNA foundation model; mentions CRISPR but isn't an off-target predictor."
```

## Page-level fixture format

Same shape; `expected_pages` instead of `expected_claims` (no
section/position). Anchors may include category prefix
(`synthesis/foo`) for synthesis/idea pages. Optional `expected_rank` on
an entry enforces an exact rank — use sparingly (only when a single
paper is the unambiguous correct answer for the query).

```yaml
fixture_id: late-interaction-neural-retrieval
fixture_type: pages
query: "late interaction neural retrieval BERT"
k: 5
expected_pages:
  - paper_stem: khattab-2020-colbert-efficient-and-effective-passage
    importance: critical
    expected_rank: 1
    rationale: "ColBERT IS late interaction. Must be rank 1."
must_not_appear:
  - paper_stem: luo-2024-interpretable-crisprcas9-off-target-activities-with-mismatches
    rationale: "Cross-domain false positive from the 'BERT' token overlap."
```

## Importance levels

- **critical** — a thorough top-K must include this. Weighted 3 in nDCG.
- **high** — should appear; missing is a quality issue. Weight 2.
- **normal** — nice-to-have. Weight 1.

A fixture without any `critical` anchors is too thin to detect retrieval
regressions. Aim for 2–4 critical anchors per fixture.

## Scoring

`researchwiki benchmark-fixture <fixture-stem>` reports:

| Metric | What it measures |
|---|---|
| **MRR** | Reciprocal rank of the highest-importance hit |
| **nDCG@K** | Importance-weighted ranking quality |
| **Expected recall (all)** | Fraction of expected anchors appearing anywhere in top-K |
| **Expected recall (critical)** | Critical-tier-only recall |
| **must_not hits** | Count of negative anchors that landed in top-K (lower = better) |

A/B mode (`--baseline-embedding X --candidate-embedding Y`) emits per-item
rank changes so you can see *which* anchors moved, not just the aggregate Δ.

## Authoring workflow

1. Pick a real query you've issued (or want to). Note what you wanted back.
2. Run it under the current embedding (`researchwiki claims "..."` or
   `search "..."`) and record the actual top-K.
3. Identify which results are right (→ `expected_*`) and which are wrong
   but plausible (→ `must_not_appear`). Tag missing-but-should-be-there
   results from your wiki knowledge as `expected_*` too.
4. Look up anchors with `researchwiki claims --by-stem <stem>` — grab
   the relevant `(section, position)` tuples for claim-level fixtures.
5. Write rationales for each anchor. They're not scored, but they age
   well: future you will not remember why "this paper, this position"
   was the right call.

Time per fixture: ~15 min once you know the topic.
