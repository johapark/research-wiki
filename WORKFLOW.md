# Workflow — how the framework operates end-to-end

Companion to `README.md` (the gist) and `CLAUDE.md` (the LLM contract). This
doc walks through what actually happens when you use the framework, with real
output excerpts. Read top-to-bottom for the flow; jump to a section if you
want a specific operation.

If `README.md` describes the destination, this doc describes the journey.

---

## What the framework does, in one paragraph

You drop research-paper PDFs into `inbox/`. An LLM authors a wiki page for
each one (Summary, Key Contributions, Methodology, Results, Limitations,
Related Papers). A four-axis grader scores the page — **fidelity** (per-claim
BM25 + bi-encoder cosine against the source PDF), **salience-recall**
(synthetic anchors extracted from the PDF's abstract / figure captions /
results lead-ins, checked against the page), **coherence** (page-shape
contract — required sections, word count, bullet density), and **grounding**
(every claim-shaped unit carries a `[[wikilink]]` or `[^id]` citation). The
ingest agent's tournament selects the winning draft via a combined
fidelity-plus-salience scalar; coherence and numeric drift form the
lexicographic tail. New ingests then trigger evolution proposals against
neighboring synthesis pages — when paper P arrives, the framework asks an
LLM whether existing synthesis pages should be edited in
light of P, and writes structured proposals to
`.ingest/{stem}-evolution-proposals/` for human review. The result is a
markdown wiki on disk that compounds: every new paper makes related synthesis
pages slightly more correct.

---

## The shape of one ingest

```
inbox/raw-paper.pdf
        │
        ▼
┌────────────────────────────────────────────────────────────────────┐
│ researchwiki agent ingest inbox/raw-paper.pdf                      │
│                                                                    │
│  1. reconcile     PDF → DOI / title / year / venue / authors       │
│                   (LLM extractor on first 1-2 pages → Semantic      │
│                    Scholar lookup → Crossref + regex fallbacks)     │
│  2. extract       PDF → introduction / methods / results sections   │
│  3. crosslinks    citation-graph candidates ∪ semantic-KNN         │
│                   candidates → LLM-judged topical list              │
│  4. author × N    parallel drafts at differing temperatures         │
│  5. grade × N     fidelity (BM25 + bi-encoder cosine) per claim     │
│                   + salience-recall (PDF anchors → fixture →        │
│                   token-overlap + cosine) + coherence (structural   │
│                   conformance)                                      │
│  6. tournament    deterministic argmax on combined-quality scalar   │
│                   (0.5·semantic + 0.5·salience), bucketed; tail     │
│                   axes coherence → drift → coverage → bm25          │
│  7. critic        translate weak-claim flags into revision notes    │
│  8. evolve        revise the winning draft against critic notes;    │
│                   keep iff combined-quality improves (the same      │
│                   primary scalar; salience-only gains accepted)     │
│  9. (debug)       repair structural-gate failures if any            │
│ 10. promote       move PDF → papers/{stem}.pdf,                     │
│                   write wiki/{category}/{stem}.md,                  │
│                   add back-links, append to index.md and log.md     │
│ 11. shortname     LLM proposes a 1–4 word handle for the index entry│
│ 12. keywords      LLM proposes 5–10 retrieval tokens for YAML       │
│ 13. memory evolve KNN against existing synthesis                    │
│                   pages → per-neighbor LLM judgment → proposals     │
│                   written to .ingest/{stem}-evolution-proposals/    │
└────────────────────────────────────────────────────────────────────┘
        │
        ▼
papers/{stem}.pdf                     # canonical filename
wiki/{category}/{stem}.md             # the new wiki page
.ingest/{stem}-evolution-proposals/   # proposals for neighbor pages
ingest_iterations table               # per-phase audit trail in state DB
```

Each phase is a pure function over the previous phases' outputs.
Persistence to the `ingest_iterations` table happens in thin runner
wrappers, never inside the phase functions.

---

## A worked example

Here's what an ingest looks like when you run it. The output excerpts below
are from a real run on `du-2025-a-versatile-crisprcas9-system-off-target` —
copy-pasted, not hand-crafted, so what you see is what you'll see.

### 1. Drop a PDF

```bash
cp ~/Downloads/some-paper.pdf inbox/
```

Filename doesn't matter. The framework derives a canonical stem from the
paper's first-page metadata (`{author}-{year}-{first-five-title-words}`).

### 2. Run the agent

```bash
researchwiki agent ingest inbox/some-paper.pdf                 # single PDF
researchwiki agent ingest inbox/*.pdf                          # ≥2 PDFs — auto-batch, 4 workers, checkpoint
researchwiki agent ingest --resume .ingest/batch-<ts>/         # resume after a crash / Ctrl-C
```

Passing multiple PDFs to a single invocation enters crash-safe batch mode:
workers run in parallel (default 4, tune with `-w N`) and every completion
is recorded atomically under `.ingest/batch-<ts>/checkpoint.json` — so a
mid-batch crash is recoverable with `--resume <batch-dir>` and rerun
picks up exactly where it stopped. Don't fan out one background Bash per
file: that bypasses the checkpoint, uncaps concurrency, and multiplies
`state.db` write contention.

You'll see something like (excerpted, real output from a recent ingest):

```
[agent] attempt_id=8a3b...
[agent] pdf=some-paper.pdf
[agent] mode=real (openai-compatible/gpt-5.6-luna)
[agent] n_drafts=2  use_semantic=True  max_evolve=1
[agent] reconcile → stem=du-2025-a-versatile-crisprcas9-system-off-target year=2025 type=research
[agent] extract   → sections=['introduction', 'methods', 'results', 'discussion'] pdf_claims=11
[agent] target-claims → 23 extracted (5 critical, 11 high)
[agent] crosslinks → 8 verified candidate(s)
[agent] author #1  → stance=balanced t=0.2 7183 chars
[agent] author #2  → stance=skeptical t=0.6 7637 chars
[agent] grade   → draft 2890 sem=0.71 sal=0.34 coh=1.00 bm25=29.87
[agent] grade   → draft 2891 sem=0.74 sal=0.30 coh=1.00 bm25=33.72
[agent] tournament → winner draft 2891
[agent] critic     → flagged 1 weak claims (low semantic in Results)
[agent] evolve     → revised draft, sem=0.78 sal=0.34 (combined-quality improved; kept)
[agent] gate       → passed (combined-quality ≥ 0.55, no drift, KC≥4, n_graded≥6)
[agent] promote    → wiki/cgt/du-2025-a-versatile-crisprcas9-system-off-target.md (3 back-links added)
[agent] shortname  → 'CCLMoff'
[agent] keywords   → ['CRISPR off-target prediction', 'transformer', 'RNA language model', ...]
[agent] evolve     → 1 proposal(s) at .ingest/du-2025-...-evolution-proposals/  (knn=2 above_thr=1 judged=1 actionable=1)
```

Each grade line carries four axes: `sem` is the page-level mean of per-claim
bi-encoder cosines (fidelity); `sal` is salience-recall against PDF-anchor
synthetic fixture; `coh` is the structural-conformance score (0..1, sum of
weights of passing checks); `bm25` is the page-level mean of per-claim top-1
BM25 retrieval scores. The tournament keys on `combined-quality = 0.5·sem +
0.5·sal` bucketed to 2 decimals — so a draft with low semantic but high
salience can beat a high-semantic-low-salience peer if the combined number
wins. The lexicographic tail (coherence → -drift → n_graded → bm25 →
weakest_score) decides ties.

Each `[agent] X → Y` line is one phase committing one row to the
`ingest_iterations` table — the audit trail is durable as the run
progresses, so a crash mid-way leaves a partial trace you can inspect.

### 3. Inspect the new wiki page

`wiki/cgt/du-2025-a-versatile-crisprcas9-system-off-target.md`:

```yaml
---
title: "A versatile CRISPR/Cas9 system off-target prediction tool using language model"
authors: Yi-Hong Du, Xiao-Yan Tang, ...
year: 2025
doi: 10.1093/bib/...
venue: Briefings in Bioinformatics
type: paper
category: [cgt]
pdf_path: "[[du-2025-a-versatile-crisprcas9-system-off-target.pdf]]"   # Obsidian wikilink → click-to-open; real file at papers/{stem}.pdf
short_name: CCLMoff
keywords: [CRISPR off-target prediction, transformer, RNA language model, ...]
tags: [ingested-via-agent]
---

## Summary
**CCLMoff** is a transformer-based deep learning framework ...

## Key Contributions
- ...

## Methodology and Architecture
...

## Results
...

## Limitations
- ...

## Related Papers
- [[cgt/lin-2020-crisprnet-a-recurrent-convolutional-network]] — direct successor; CCLMoff outperforms CRISPR-Net on shared benchmarks
- [[cgt/chuai-2018-deepcrispr-optimized-crispr-guide-rna]] — explicitly benchmarks against and surpasses DeepCRISPR
- ...
```

YAML carries the full audit (DOI, source-PDF wikilink, ingest tags). The
body is six standard sections capped to budget. Every wikilink in
`Related Papers` was verified — either citation-graph confirmed or
LLM-judged topical with a one-line rationale.

### 4. Review the evolution proposals

`.ingest/du-2025-...-evolution-proposals/extend__synthesis__crispr-cas9-off-target-methods-2026.md`:

```yaml
---
source: cgt/du-2025-a-versatile-crisprcas9-system-off-target
target: synthesis/crispr-cas9-off-target-methods-2026
verdict: extend
confidence: 0.82
---

# EXTEND — [[synthesis/crispr-cas9-off-target-methods-2026]]

**Rationale:** CCLMoff is a new off-target scoring/ranking method that
belongs in the Prediction section alongside CRISOT and CRISPRme, with a
specific claim about outperforming prior learning-based methods using an
RNA language model backbone and the largest benchmark dataset to date.

## Patch

- Add `[[cgt/du-2025-a-versatile-crisprcas9-system-off-target]]` into the
  `## Approaches` / equivalent section as an inline `[[wikilink]]`, plus a
  matching `[^id]: [[cgt/du-2025-…]]` footnote under `## References`
  (synthesis pages cite via the body — there is no `referenced_papers:` field).
- Section: **### Prediction — computational ranking of candidate off-targets**
- New bullet:
  > **CCLMoff** [[cgt/du-2025-a-versatile-crisprcas9-system-off-target]] is a
  > transformer-based scorer that applies a pretrained RNA language model
  > (RNAcentral) to the off-target ranking problem ...
```

The proposal carries everything a reviewer needs: source paper, target
page, verdict type, confidence, structured patch with verbatim section
heading and pre-formed bullet text including the source wikilink.

### 5. Apply the proposal (LLM-mediated, human-gated)

Reviewing and applying proposals is the LLM's job; the human's job is to
say yes or no. The flow (formalized in `CLAUDE.md` Step 6 of the Ingest
section):

1. The LLM reads each proposal file and the target synthesis page.
2. It verifies the patch claim against the source paper's wiki page —
   numbers, framing, and "first/largest/only" claims drift between the
   LLM that wrote the proposal and the actual page text. Mismatches get
   fixed before the patch lands; unsupported superlatives get softened.
3. Proposals can also be **stale**: the source paper may have been
   recategorized after the proposal was generated (path prefix drift), or
   the source may not even have a wiki page (e.g. the proposal came from
   a `--repeat` benchmark run against `benchmark-fixtures/pdfs/`). The
   LLM detects both and either fixes the path in-flight or rejects the
   proposal outright.
4. The LLM surfaces a one-paragraph verdict per proposal: target page,
   what the patch does, what it'd change before applying, apply/skip
   recommendation.
5. You answer yes or no — a single answer covers all proposals from one
   ingest unless you specify otherwise.
6. On approval the LLM applies the edits and `rm -rf`'s the proposal
   directory; on skip, the directory stays and `lint` flags it as
   `stale_evolution_proposals` after 7 days for revisit.

The reason editing is gated by explicit permission: mutating an existing
synthesis page is the highest-blast-radius action the framework can take.
The current posture is "LLM does the substantive review, human is the
final go/no-go" — keeps load-bearing pages from drifting silently.

---

## Querying the wiki

Three retrieval modes via `researchwiki search`:

```bash
researchwiki search "memory evolution"           # default: hybrid (RRF over BM25 + semantic)
researchwiki search "memory evolution" --mode bm25      # keyword only
researchwiki search "memory evolution" --mode semantic  # bi-encoder only
researchwiki search --like ai/xu-2025-...               # See-Also on a page
```

Hybrid output shows the per-ranker provenance inline:

```
# Search results for: 'agentic memory long-context'  (hybrid RRF; 5 hits)

 1. rrf=0.0328  [BM25 #1, sem #1 (0.82)]  [[ai/xu-2025-a-mem-agentic-memory-for-llm]]  (paper)
      A-Mem: Agentic Memory for LLM Agents
 2. rrf=0.0306  [BM25 #3, sem #8 (0.70)]  [[references/anthropic-2025-building-effective-ai-agents-architecture]]
      Building Effective AI Agents: Architecture Patterns and Implementation Frameworks
 3. rrf=0.0306  [BM25 #7, sem #4 (0.74)]  [[ai/toledo-2025-ai-research-agents-for-machine]]
      AI Research Agents for Machine Learning ...
```

Reading: hit #3 was BM25 rank 7 (modest keyword match) but semantic rank 4
(strong topical match) — RRF correctly promoted it. The `[BM25 #N, sem #M]`
annotation makes the fusion auditable: you can tell at a glance whether a
result fused well (top in both) or one-sided (good keywords, weak topic
match — likely noise).

The same content is reachable from the CLI without any server: topic search
is `researchwiki search`, a page is a plain file `Read`, and structural
questions go through `researchwiki db query`. For grounded citations use
`researchwiki claims "<query>"` — each hit prints a durable
`[[stem#claim_slug]]` anchor (content-addressed, survives `db rebuild`) that
you paste straight into a page. The bare `claim_id:NNN` shown alongside is a
session-local row handle, reassigned on rebuild — never a citation token.
Use `researchwiki pdf-search <stem> "<query>"` to pull an exact passage the
wiki page didn't quote.

---

## Maintaining the wiki

Three commands cover the maintenance loop:

| Command | Cadence | Cost |
|---|---|---|
| `researchwiki status` | Run after every ingest session | Local only, sub-second |
| `researchwiki lint` (or `--json`) | Weekly, or after batch ingests | Local only, sub-second |
| `researchwiki audit` | After batch ingests, or weekly | Calls Semantic Scholar — minutes for hundreds of papers |

`status` is the dashboard. After the recent updates it includes:

```
Index health:
  Tantivy BM25:        built 2m ago
  Semantic page idx:   built 2m ago (87 pages, dim=384)

Pending evolution proposals: 1 dir(s), 1 total file(s)
  du-2025-...                                              1 file(s), 32m old
  Review and apply, then `rm -rf` the directory.

Ingest cost (last 7 days):
  attempts:           7
  total tokens:       20K input + 31K output
  mean per attempt:   7K tokens
  estimated cost:     $0.00  (default models unpriced in the estimator)
```

`lint` reports orphans, broken wikilinks, missing back-links, stale
syntheses (by mtime, by content via topic-seed search, and by audit-count
drift), missing keywords, missing DOIs, stem↔YAML year drift, stale
evolution proposals (≥7 days old), concept candidates, page-type
mismatches, invalid YAML frontmatter, and supplementary-file consistency
(missing-on-disk + orphaned files in `papers/{stem}.supp/`). Pass `--fix`
to auto-insert missing back-links (the only auto-fix; everything else
needs human judgment). Pass `--cross-paper` to opt into the
LLM-call-heavy cross-paper contradiction check: claim pairs across
different papers at high embedding cosine are sent to a judge that flags
numeric or directional disagreements. Off by default — the bulk of lint
is local and sub-second, and we keep the LLM cost out of the default
path. `lint --json` exposes every category as a structured field for
downstream tooling.

`audit` calls Semantic Scholar to verify the citation graph: every
`[[wikilink]]` should be backed by a real citation either way, and
high-citation papers OUTSIDE the wiki that are cited by ≥2 wiki papers
get flagged as ingest candidates.

---

## Module map

Where things live in the package:

```
researchwiki/
├── index/                  # Indexing primitives
│   ├── embeddings.py       #   Bi-encoder model singleton (BAAI/bge-small)
│   ├── claim_embeddings.py #   Cached bi-encoder embeddings for claims
│   ├── pdf_chunks.py       #   Per-PDF Tantivy chunk index + chunk embeddings
│   ├── pages_bm25.py       #   Wiki-page BM25 index (Tantivy)
│   ├── pages_semantic.py   #   Wiki-page dense embedding store
│   ├── graph.py            #   Weighted paper-graph edges + modularity clustering
│   └── types.py            #   Document, SearchHit, SearchBackend ABC
├── search/                 # Query orchestration over index/
│   ├── __init__.py         #   suggest_category, build_documents_from_wiki
│   ├── hybrid.py           #   RRF fusion (BM25 + semantic)
│   ├── refs.py             #   Durable [[stem#slug]] citation form for a claim hit
│   └── tools.py            #   Read-only primitives behind `claims` + `pdf-search`
├── grade/                  # All page-scoring / quality evaluation
│   ├── fidelity/           #   Per-claim fidelity, paired by page-type
│   │   ├── paper.py        #     Paper page vs. its OWN PDF (continuous floats)
│   │   └── synthesis.py    #     Synthesis/idea page vs. CITED PDFs (categorical
│   │                       #     verdicts: supported/weak/composite/misattributed)
│   ├── salience.py         #   PDF-anchor recall (synthetic ContentFixture from
│   │                       #     abstract / Results / captions, fed through scorer)
│   ├── coherence.py        #   Page-shape contract (sections, word count,
│   │                       #     bullets, wikilink density). No PDF, no LLM.
│   ├── grounding.py        #   Citation-presence check on every claim-shaped unit
│   ├── support.py          #   Per-claim entailment (qualitative analogue of fidelity)
│   ├── claim_overlap.py    #   Near-paraphrase claim overlap across papers (crosslink)
│   ├── scorer.py           #   Fixture-based scorer (token-overlap + bi-encoder
│   │                       #     cosine + LLM-judge verdict paths). Used by both
│   │                       #     grade/salience.py and benchmark/benchmark-fixture.
│   ├── primitives.py       #   Deterministic primitives (numeric drift, negation)
│   └── parser.py           #   Extract claim units from wiki markdown
├── benchmark/              # Benchmark methodology (renamed from eval/)
│   ├── fixture.py          #   ContentFixture / RetrievalFixture loaders
│   ├── retrieval.py        #   Retrieval-quality benchmarks
│   ├── retrieval_reports.py#   Retrieval-fixture scoring dispatch + prose reports
│   ├── content_reports.py  #   Rendering for content-fixture scoring output
│   ├── replicate.py        #   Author-N-times replication driver (NOT auto-imported
│   │                       #     from the package — depends on agents.phases)
│   └── style.py            #   Style report
├── agents/                 # The ingest agent
│   ├── runner.py           #   13-phase state-machine driver
│   ├── context.py          #   Shared phase Context (each phase reads/writes it)
│   ├── fitness.py          #   Tournament + improvement-rule lenses;
│   │                       #     `combined_quality` = 0.5·semantic + 0.5·salience
│   ├── llm.py              #   LLM API wrapper (provider-routed; real + stub)
│   ├── model_config.py     #   Per-role model assignments from config/models.*.yaml
│   ├── relay.py            #   chat-relay provider client
│   ├── judge.py            #   LLM-judge helpers (structural verdicts)
│   ├── prompt_lib.py       #   Load prompts from prompts/*.md (A/B-able)
│   ├── promote.py          #   Move PDF + write wiki page + back-links + index.md
│   └── phases/             #   Each phase as its own module
│       ├── reconcile.py    #     PDF → DOI/title/year/venue/authors
│       ├── extract.py      #     PDF → sections + pdf_claims
│       ├── target_claims.py#     Identify the load-bearing claims to author against
│       ├── crosslinks.py   #     Citation-graph + semantic crosslink candidates
│       ├── draft.py        #     Author + tournament (combined-quality argmax)
│       ├── grade.py        #     Wraps grade.fidelity.paper + grade.salience
│       │                   #     + grade.coherence, packs aggregate scores dict
│       ├── revise.py       #     Critic + evolve + debug
│       ├── commit.py       #     Sandbox write + propose_short_name + keywords
│       ├── grade_persist.py#     Post-commit fidelity grading on the promoted page
│       ├── evolution.py    #     Memory-evolution proposals (orchestration)
│       ├── memory_evolve.py#     Propose edits to existing synthesis pages
│       └── evolve_ledger.py#     Judged-pair idempotency cache for memory_evolve
├── providers/              # External-API wrappers (S2, Crossref, PubMed, bioRxiv, ORCID)
├── db/                     # State DB (sqlite) — derived from wiki/, rebuildable
├── tasks/                  # CLI subcommands — one module per command, auto-discovered
│   ├── ingest.py           #   Digest-only path (manual page authoring)
│   ├── agent.py            #   Full agent path (auto-authoring)
│   ├── grade.py            #   Per-paper fidelity + salience report
│   ├── grade_synthesis.py  #   Synthesis/idea fidelity (misattribution check)
│   ├── check_grounding.py  #   Structural citation check
│   ├── check_coverage.py   #   Recall surface — unreferenced top-N hits
│   │                       #     for a synthesis/idea page's topic_seed
│   ├── benchmark_fixture.py #  Hand-curated-fixture benchmark
│   ├── evolve.py           #   Standalone memory-evolution proposals
│   ├── search.py reindex.py status.py lint.py audit.py claims.py pdf_search.py ...
├── concepts/               # Concept-hub surfacing + scaffold + reciprocal linking
├── claim_graph/            # Content-addressed claim identity + edge cache
├── synthesis_candidates/   # Detect paper clusters lacking a synthesis page
└── pdf/                    # pypdfium2-backed PDF text/structure extraction
```

**Recent layering cleanup** (worth knowing because the docstrings still
refer to the old paths in places):

- `grade/coverage.py` → `grade/fidelity/paper.py` (rename for honesty —
  the file does fidelity, not coverage; the dataclass `CoverageReport`
  became `PaperFidelityReport`).
- `grade/fidelity.py` → `grade/fidelity/synthesis.py` (paired sibling;
  `FidelityReport` → `SynthesisFidelityReport`).
- `agents/coherence.py` → `grade/coherence.py` (it's a deterministic
  page-shape grader, not agent-specific).
- `eval/scorer.py` → `grade/scorer.py` (the fixture-based scorer is
  consumed by `grade/salience.py` as production grader infrastructure,
  not benchmark-only).
- `eval/` → `benchmark/` (after the scorer left, what remains is
  genuinely benchmark methodology — replication driver, retrieval
  benchmarks, style report).
- `grade/scoring.py` → `grade/primitives.py` (the module holds
  deterministic helpers — numeric integrity, negation parity — not
  scoring; the rename also disambiguates from `grade/scorer.py`, which
  is one letter away and does something completely different).

These moves also broke a circular import: `benchmark/__init__.py` no
longer auto-loads `replicate.py` (which depends upward on
`agents.phases`). Consumers of the replication driver import it directly
via `from researchwiki.benchmark.replicate import replicate_score`. The
lazy imports that previously worked around this in `grade/salience.py`
are gone.

**Naming history (2026-06):** the CLI was `eval-coverage` and the data
directory `eval-fixtures/`; both renamed to `benchmark-fixture` /
`benchmark-fixtures/` to align with the `benchmark/` package and make the
purpose obvious (it's the hand-curated fixture benchmark, not a coverage
metric). Wiki-content paths (`wiki/`, `papers/`, `inbox/`) are unchanged.

The `index/` package was carved out specifically to share indexing
primitives between `grade/` (which scores claims and pages) and `search/`
(which retrieves pages). Anyone adding a new "embed-and-query something"
feature should reach for `index/`.

---

## What gets cached, what gets regenerated

The wiki has clear canonicalness:

- **Markdown is canonical.** `wiki/*.md` is the source of truth.
  Everything else can be regenerated from it.
- **The state DB** (sqlite, at `~/.local/share/researchwiki/state.db`) is
  a derived index over `wiki/` + `papers/` + caches. Run `researchwiki db
  rebuild` to reconcile drift.
- **`.tantivy-index/` and `.semantic-cache/`** are search indexes built
  from `wiki/` by `researchwiki reindex`. Both gitignored.
- **`.grade-cache/{stem}/`** is per-paper PDF chunk index + embeddings.
  Gitignored. Built lazily on first grade call per paper.
- **`.s2-cache/`, `.crossref-cache/`, `.web-cache/`** are external-API
  responses. Gitignored. Built lazily.
- **`.ingest/`** is per-attempt transient state (digests, evolution
  proposals). Cleared as you act on them.

If you wipe everything except `wiki/`, `papers/`, `inbox/`, and the
framework code, you can rebuild every cache and re-derive every index.
This is by design — the markdown layer is what survives.

---

## Costs and trade-offs

### Per-ingest cost

Absolute cost is **config-dependent** — it rides on whichever
`config/models.*.yaml` file `RW_MODELS_CONFIG` selects and that provider's
token pricing. The current active default is `models.chatgpt.yaml`:
`gpt-5.6-luna` drives the quality-sensitive roles (author / critic / judge /
proposer), and the cheaper `gpt-5.4-mini` handles the deterministic
short-output roles (classifier / extractor). The committed fallback when no
override is set is `config/models.yaml` (Upstage Solar: `solar-pro3` /
`solar-mini`).

At OpenAI standard rates (per 1M tokens: `gpt-5.6-luna` $1.00 in / $6.00
out, `gpt-5.4-mini` $0.75 in / $4.50 out), a single-draft ingest runs
roughly ~20K input + ~6K output tokens across all roles (from `researchwiki
insights`), with the **author** phase dominating and everything else a long
tail — so a typical paper lands around **~$0.05–$0.08**, and a 2-draft
author tournament (`-n 2`) closer to **~$0.08–$0.12**. The two grader runs
are semantic-only (no LLM) and free; memory-evolution proposals cost in
proportion to how many synthesis neighbors clear the cosine prefilter.

Treat these as order-of-magnitude — token counts are config-independent
(same prompts), but dollars move with the model assignment. Note the
framework's built-in cost estimator only prices models already in its
internal rate table; the current default models aren't in it, so
`researchwiki insights` and `status` report **`$0.00`** for them today —
read that as "unpriced," and multiply the per-role token counts from
`insights` by your provider's rates for a real figure. The agent is
calibrated to spend on *fidelity* (claim-grading, critic, evolve), not on
speed — that hasn't changed across model swaps.

### When to opt out of which phases

- **`--use-stub`** — full offline / deterministic mode. No API calls. Use
  for harness tests and CI.
- **`--no-llm-reconcile`** — skip the LLM metadata extractor (`gpt-5.4-mini`
  under the current default), fall back to the regex+S2 path. Use for
  offline/stub mode or to shave the (tiny) reconcile cost. The regex path is
  still maintained but accumulated
  format-specific patches; LLM-reconcile is the structurally robust path.
- **`--no-semantic` on reindex** — skip the bi-encoder pass, BM25 only.
  Use when sentence-transformers isn't installed.
- **Agent runner skips memory-evolution in stub mode** automatically.

### When the agent gets it wrong

- **Reconcile lands wrong metadata** — rare with LLM-reconcile (default).
  Common adversarial cases: Science First-release PDFs with ~2K
  extractable chars, NEJM Perspectives bundled with research articles.
  Recovery: `--doi <correct-doi>` and/or `--title <real-title>` override.
  See CLAUDE.md "Recovery — re-ingest after a broken ingest" for the
  full step-by-step.
- **Author hallucinates a number.** The grader's `n_drift > 0` gates the
  page; the DEBUG operator runs to repair before commit. If DEBUG can't
  fix it, the page goes to `.agent-output/` (sandbox) for manual review
  rather than `wiki/`. The most common cause is a PDF with sparse
  extractable text (a few thousand chars rather than full body) — the
  author has nothing to ground its numbers in, the grader catches the
  drift, and the gate correctly refuses to promote. The fix is to find a
  better-quality source PDF (publisher version vs. preprint scan) and
  re-ingest.
- **Cross-link verifier strips a real wikilink.** Means the candidate
  wasn't on the verified list. Either accept it as a topical-not-cited
  paper, or update the candidate list manually.

---

## When to do what (quick reference)

| You did this | Then run |
|---|---|
| Dropped PDF in `inbox/` | `researchwiki agent ingest inbox/<file>.pdf` |
| Dropped ≥2 PDFs in `inbox/` | `researchwiki agent ingest inbox/*.pdf` (auto-batches with checkpoint/resume) |
| Batch ingest crashed mid-run | `researchwiki agent ingest --resume .ingest/batch-<ts>/` |
| Edited a wiki page manually | `researchwiki db rebuild && researchwiki reindex` |
| Want to find synthesis pages affected by a recent paper | `researchwiki evolve <category/stem>` |
| Want to know what to ingest next | `researchwiki audit --json` |
| Want to find pages with sparse keywords | `researchwiki lint --json \| jq .missing_keywords` |
| Want to know how much you've spent ingesting lately | `researchwiki status` (last section) |
| Got curious whether your wiki has anything on X | `researchwiki search "X"` (or `--like` from a page) |
| Lost track of what's in the inbox | `researchwiki status` (top section) |
| Want to score a paper page against its PDF | `researchwiki grade paper <stem>` |
| Want to backfill grading for un-graded papers only | `researchwiki grade regression --missing-only` |
| Want to re-grade every paper and detect drift | `researchwiki grade regression` (or `--no-persist` for diff-only) |
| Want to verify a synthesis page's claims trace to cited papers | `researchwiki grade synthesis <page>` |
| Want to check that every claim has a citation | `researchwiki check-grounding <page>` |
| Want to benchmark page authoring against a curated fixture | `researchwiki benchmark-fixture <stem>` |

---

## Where to go from here

- `CLAUDE.md` — the contract for LLM operations (the Four Rules,
  file-naming convention, ingest steps, query workflow).
- `config/models.yaml` — central model assignments for each role
  (author, critic, judge, classifier, etc.). Edit this file to A/B test
  models without code changes.
- Closed planning docs that informed the current architecture (Phase A–D
  semantic index / hybrid retrieval / memory evolution; v1 agent design
  history) were removed from `plans/` as they closed out — see git
  history (`git log --diff-filter=D --name-only -- plans/`) if you need
  the historical context.

For new contributors: read `CLAUDE.md` first (it's load-bearing for the
LLM-facing semantics), then this doc. For the pipeline internals, the
phase functions in `researchwiki/agents/phases/` and the state machine in
`researchwiki/agents/runner.py` are the source of truth.
