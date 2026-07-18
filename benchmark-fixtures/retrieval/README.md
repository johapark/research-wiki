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

The bundled fixtures anchor **only** to stems in `benchmark-fixtures/pdfs/`
so they remain portable — no dependency on a personal wiki corpus.

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

## Scoring (when implemented)

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
