# Benchmark fixtures — self-contained corpus

Hand-curated fixtures that score the `researchwiki agent ingest` pipeline. Each fixture pairs a paper PDF with a structured YAML declaring what a thorough wiki page **should** capture: headline claims, capabilities, limitations, related-paper links.

The harness is **portable**: six CC-BY-4.0 OA papers are committed under `pdfs/` so anyone with a fresh clone can run every benchmark without a pre-existing corpus. Attribution: see [`LICENSES.md`](./LICENSES.md).

```
benchmark-fixtures/
├── LICENSES.md                # per-paper CC-BY-4.0 attribution
├── CALIBRATION-2026-06.md     # archived — pre-OA threshold calibration
├── pdfs/                      # bundled CC-BY-4.0 OA papers (committed)
│   └── {stem}.pdf
├── {stem}.yaml                # content-coverage fixtures (this dir)
└── retrieval/
    ├── claims/{slug}.yaml     # claim-level retrieval fixtures
    └── pages/{slug}.yaml      # page-level retrieval fixtures
```

## Running the harness

**Content fixtures** — score a fresh ingest of the bundled PDF against the fixture.

**When benchmarking a model (agent-driven), always `--force-sandbox`** so the
throwaway ingest never lands in `wiki/` or `index.md` — see
[`../prompts/benchmark-run.md`](../prompts/benchmark-run.md):

```bash
RW_MODELS_CONFIG=models.<name>.yaml \
  researchwiki agent ingest benchmark-fixtures/pdfs/muslu-2026-variantmedium-*.pdf -n 1 --force-sandbox
# → .agent-output/<stem>.md ; judge it against benchmark-fixtures/<stem>.yaml (manually, or a non-rate-limited scorer)
```

The automated scorer path below **promotes** the page into `wiki/` — use it only
for calibrating the harness itself, not for one-off model comparisons, and clean
up afterwards (`rm` the page/PDF, drop the `index.md` line, then
`db rebuild && reindex` — both, or the next ingest's crosslink phase crashes on the
ghost page):

```bash
# From a fresh clone (mutates the real wiki):
researchwiki agent ingest benchmark-fixtures/pdfs/muslu-2026-variantmedium-*.pdf
researchwiki db rebuild && researchwiki reindex
researchwiki benchmark-fixture muslu-2026-variantmedium-sensitive-and-generalizable-somatic --llm
```

The scorer resolves the PDF via `researchwiki.paths.resolve_pdf(stem)`: it checks `papers/{stem}.pdf` first (canonical user copy after ingest), falling back to `benchmark-fixtures/pdfs/{stem}.pdf` (bundled copy). Once you've ingested, the personal copy wins and the bundled copy is inert.

**Retrieval fixtures** presume the bundled papers are ingested and the claims DB / semantic cache are built. One-time setup:

```bash
for f in benchmark-fixtures/pdfs/*.pdf; do
  researchwiki agent ingest "$f"
done
researchwiki db rebuild && researchwiki reindex
researchwiki benchmark-fixture retrieval/claims/<slug>
```

**Replication** — the author phase is stochastic; single-run scores have ~±10pp variance on prose-heavy papers. For lever-comparison validation:

```bash
researchwiki benchmark-fixture <stem> --repeat 5 --llm   # ~$0.93/fixture; mean±SD reported
```

## Bundled corpus

| Stem | Type | Venue | Domain |
|---|---|---|---|
| muslu-2026-variantmedium-sensitive-and-generalizable-somatic | method-with-benchmarks | Genome Medicine | somatic variant calling |
| zhang-2026-mga-a-tool-for-haplotype-mixed | method-with-benchmarks | Genome Biology | genome assembly |
| li-2026-schilda-hierarchical-integration-of-llm | method-with-benchmarks | PLoS Comp Bio | single-cell annotation |
| fonseca-2026-adjunctive-ibuprofen-in-pre-extensively-drug-resistant | clinical-trial | Nat Comms | tuberculosis phase IIA |
| chuai-2018-deepcrispr-optimized-crispr-guide-rna | method-with-benchmarks | Genome Biology | CRISPR guide prediction |
| assa-2024-quantifying-allele-specific-crispr-editing-activity | method-with-benchmarks | Nucleic Acids Research | allele-specific editing measurement |

Content-coverage fixtures declare `published_at:` for automatic contamination flagging by the scorer.

## Adding a new fixture

Two paths, matching the two fixture kinds.

### Content-coverage fixture (whole-paper recall)

1. **Add the PDF.** If the paper is CC-BY-4.0 or CC0, drop it under `pdfs/{stem}.pdf` (canonical stem via `researchwiki.stems.derive_stem`) and append its attribution to `LICENSES.md`. The license statement must belong to that article: bind it to the article title or DOI within the article boundary, since an issue-extracted PDF can begin or end with material from an adjacent paper. A CC block merely appearing somewhere in the file is not sufficient. Otherwise, keep the PDF in your personal `papers/{stem}.pdf` — `resolve_pdf` finds it there — but the fixture is user-local and won't run on a fresh clone.
2. **Write `{stem}.yaml` at this directory's root** using the schema below. Aim for 3–6 `critical` headline claims and 2–3 acknowledged limitations. `related_papers: []` is honest when there's no intra-corpus edge.
3. **Sanity-check**: `researchwiki benchmark-fixture --list` should include the new stem; `researchwiki benchmark-fixture <stem>` should return a valid ContentFixture.

### Retrieval fixture (claims or pages ranking)

Put it under `retrieval/claims/{slug}.yaml` or `retrieval/pages/{slug}.yaml` with `fixture_type: claims | pages`. See [`retrieval/README.md`](./retrieval/README.md) for the schema. Anchors must point only at bundled OA stems for the fixture to be portable; otherwise document it as user-local.

## File format (content-coverage)

```yaml
paper_stem: kim-2026-structural-motif-search-across-the-protein
paper_type: method-with-benchmarks   # method-with-benchmarks | theory | dataset | review | clinical-trial | position
title: "Structural motif search across the protein universe with Folddisco"
published_at: "2026-05-14"             # YYYY-MM-DD; scorer flags contamination when < LLM cutoff
notes: |
  Free-text — what's tricky about scoring this paper, what to watch for.

# Claims a thorough page MUST capture. Each is graded independently.
headline_claims:
  - id: speedup-pyscomotif-20k             # stable identifier; used in regression diffs
    importance: critical                    # critical | high | normal
    verbalization: |
      Folddisco's full query pipeline is 20× faster than pyScoMotif at the
      20K-structure index scale (median); 18× faster at 500K.
    # Optional structured form. When present, the scorer verifies the page pairs
    # ratio with comparator (catches "20× vs RCSB" when the paper says "vs pyScoMotif").
    relation:
      subject: "Folddisco full-pipeline query latency"
      ratio: "20×"
      comparator: "pyScoMotif"
    location: "p.3 main text; Fig. 1h"     # for the curator's reference; not scored

# Demonstrated capabilities — things the paper proves the tool can do.
capabilities:
  - id: gpcr-state-discrimination
    importance: critical
    verbalization: |
      Distinguishes active from inactive β-adrenergic receptor structures
      using CWxP/NPxxY/DRY motifs from CXCR2.

# Limitations the authors actually acknowledge (Discussion / Limitations subsection).
limitations:
  - id: 20a-cutoff
    importance: high
    verbalization: |
      The 20 Å connectivity cutoff precludes detection of motifs whose
      elements span longer distances (e.g., distant allosteric pockets).

# Wikilinks the page should include — cross-corpus edges only.
related_papers:
  - link: <category>/<related-paper-stem>
    importance: critical
    rationale: "Why the page must link this — e.g. direct comparator, prior baseline, or cited method."
```

## Importance levels

- **critical** — page omitting this is materially wrong. Heavily weighted.
- **high** — page should cover this; missing it is a quality issue.
- **normal** — nice-to-have; small deduction.

A fixture without `critical` claims is too thin to detect regressions. Aim for 3–6 critical claims per fixture.

## Scoring

`researchwiki benchmark-fixture <stem>` scores along five axes:

| Axis | What it measures | Scorer |
|---|---|---|
| **Headline-claim recall** | Fixture's headline claims present on the page? | LLM judge per claim |
| **Comparator fidelity** | When the page asserts a comparison, is the comparator the one in the fixture? | Mechanical (`relation`) + LLM verification |
| **Capability coverage** | Demonstrated capabilities present? | LLM judge per capability |
| **Limitations coverage** | Limitations from fixture mentioned? | LLM judge per limitation |
| **Related-papers recall** | Wikilinks from fixture present in Related Papers? | Mechanical string match |

Each axis returns `(matched, partial, missing)` triples weighted by importance. Aggregate = weighted recall percentage. The report names individual missing items so regressions are diagnosable.

Two modes:

- **heuristic** (default) — token overlap + numeric presence, mechanical `relation` and wikilink checks. Offline, deterministic, directional but imperfect.
- **`--llm` judge** — one LLM call per verbalization item (~30 calls per fixture). Tolerant of paraphrase, better recall. Recommended for lever-comparison scoring.

Add `--with-grader` for reference-free per-claim faithfulness (BM25 + bi-encoder + numeric-integrity + negation-parity). Add `--with-style` for compression + extractiveness. Add `--repeat N` for author-stochasticity replication.

See `CALIBRATION-2026-06.md` for the archived threshold-calibration exercise.
