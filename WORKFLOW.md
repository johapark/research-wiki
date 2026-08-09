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
│                   (0.5·semantic + 0.5·salience, each axis bucketed  │
│                   to 0.01 first, salience down-weighted when its    │
│                   anchor count is thin); tail axes coherence →      │
│                   drift → coverage → bm25                           │
│  7. critic        translate weak-claim flags into revision notes,   │
│                   plus triage of uncovered critical PDF anchors     │
│                   (recall gaps; ≥2 eligible fires the loop alone)   │
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
[agent] critic     → flagged 1 weak claims, 2 coverage gaps
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
0.5·sal` — so a draft with low semantic but high salience can beat a
high-semantic-low-salience peer if the combined number wins. The lexicographic
tail (coherence → -drift → n_graded → bm25 → weakest_score) decides ties.

Two details of that blend matter when you're reading a tournament outcome that
looks wrong:

- **Each axis is rounded to 0.01 before blending**, not the blend afterwards.
  Rounding the output made the primary *less* informative than either axis
  alone: a weighted average halves a single-axis delta, and salience is usually
  flat between drafts of the same paper (median spread 0.017; 46% of papers have
  every draft within 0.01), so a semantic difference had to exceed 0.02 to
  survive bucketing. Quantizing the inputs keeps the 0.01 granularity that lets
  coherence break genuine ties while a 0.01 move on *either* axis still reaches
  the key.
- **Salience is confidence-weighted by `n_anchors`**, ramping to full weight at
  10 anchors (`fitness.ANCHOR_CONFIDENCE_FULL`) and the blend renormalized. 9%
  of historical drafts had fewer than 10 anchors and 2% fewer than 5, where the
  score is a ratio over too small a denominator to swing selection at full
  strength; those dilute toward fidelity instead.

`salience_score` values are **not comparable across the 2026-07 abstract-anchor
guards** (below) — the guards changed the denominator, so `insights` history
straddling that change mixes two scales.

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

## Migrating an existing corpus

The ingest above starts from a PDF. If you already have LLM-generated paper pages
— from an older release of this framework, or a simpler "PDF in → summary page
out" generator — `researchwiki migrate` brings them in *without* re-authoring the
prose. Re-ingesting would overwrite the text you migrated in order to keep.

Scope, and it's enforced rather than advised: **one paper-derived page per PDF,
and the PDF must be present.** A page that isn't one-paper-shaped has nothing to
extract claims from and no PDF to grade against, so `migrate` blocks it instead
of landing an unciteable stub. Arbitrary note vaults don't belong here.

### Why heading names decide everything

Claim extraction is a **markdown parse, not a PDF read**. `parse_claims` splits
the body and looks up four *exact* H2 names from `grade.parser.SECTION_KEYS`:

```
## Key Contributions    ## Results    ## Limitations    ## Methodology and Architecture
```

A page whose findings live under `## Findings` yields **zero claims**. It then
can't be cited (no `[[stem#slug]]` anchor exists), `grade synthesis` can't verify
a citation to it — and `lint`'s `ungraded_papers` can't see it either, because
that check JOINs `claims` and a page with none is invisible. Meanwhile
`backfill hook` succeeds on it and everything else stays quiet. That combination
is why the command exists, and why `lint` grew `zero_claim_papers`.

So `migrate` normalizes headings **before** the first commit. That ordering isn't
stylistic: `claim_slug` is content-addressed on `(section, normalized text)`, and
`db rebuild` NULLs every grader column whose claim text changed. Rewrite a graded
page's body and you've silently thrown away its grading, with no way to mark
claims graded again short of re-running the grader.

### The phases

| Phase | Writes | Cost |
|---|---|---|
| `migrate preflight [src]` | nothing | local; hard-fails if the embedding model is unavailable |
| `migrate inspect <src>` | run dir only | local; per-page classification |
| `migrate apply [--dry-run]` | `wiki/` + `papers/` | **zero tokens** |
| `migrate verify` | nothing | local |

```bash
researchwiki migrate preflight ~/old-wiki/pages
researchwiki migrate inspect  ~/old-wiki/pages --category compbio
researchwiki migrate apply --dry-run
diff -u ~/old-wiki/pages/<name>.md .ingest/migrate-*/staged/<stem>.md
researchwiki migrate apply
researchwiki db rebuild && researchwiki reindex
researchwiki migrate verify
researchwiki grade regression --missing-only --no-salience    # free; the long run
# ------------------------- token boundary -------------------------
researchwiki backfill doi && researchwiki backfill keywords
researchwiki backfill hook -w 6
researchwiki db rebuild && researchwiki reindex
```

Everything above the boundary costs **no API tokens** — including grading, which
is pypdfium2 + Tantivy BM25 + a local CPU bi-encoder. For ~200 papers budget
20–40 minutes of local CPU and ~76 MB of `.grade-cache`, against roughly $0.30 of
Haiku calls for hooks below the line. Cross-linking (`claim-overlap`, ~200–2000
judge calls at the `--top 10` default) is the dominant cost and is deliberately
*not* a phase — run it later, `--top 4` for a bulk import.

`preflight` refuses to proceed when the local embedding model is missing. The BGE
weights aren't installed by `pip install -e .` — they're a ~133 MB HuggingFace
download on first use — and when they're absent grading degrades to BM25-only
*silently*, which would mean re-grading every paper afterwards.

### Reading `inspect` before you apply

```
## fixable (1)

- doe2023.md
    stem   doe-2023-deep-mutational-scanning-of-a-receptor
    claims 0 → 2
    graded renames: Key Findings→Key Contributions, Benchmarks→Results
```

`claims N → M` is the number that matters: the page is parsed as-is and again
after the rewrite, in memory. `0 → 2` means the rename is doing real work. A
`type: paper` page still reading `0 → 0` gets blocked rather than imported.

Verdicts are `compliant` (no rewrite needed) · `fixable` (confident renames) ·
`needs-human` (ambiguous heading, or frontmatter that disagrees with itself) ·
`blocked` (no frontmatter, no PDF, not one-paper-shaped) · `duplicate` (DOI
already in the corpus).

Two things `migrate` refuses to guess. **Ambiguous headings**: mapping
`## Results and Discussion` onto `Results` imports discussion prose as graded
claims, and renaming `## References` to `Related Papers` fills it with entries
that aren't wikilinks — both reported, neither rewritten without
`--accept-ambiguous`. **Conflicting frontmatter**: `year: 2024` alongside
`date: 2023-11-02` is a real disagreement, so it goes to `needs-human` rather
than a coin flip. Required values (`title`/`authors`/`year`) are never invented;
`doi`/`venue` are flagged for the lookup path, which is `backfill doi`.

### Rollback

`wiki/` and `papers/` are gitignored, so git can't undo a migration. Four layers
instead: the source corpus is never mutated (rewrites happen on copies under
`.ingest/migrate-<ts>/staged/`), `--dry-run` writes only there, `apply` tars any
target it would overwrite into the run dir, and the journal records each
completed step so `apply` resumes rather than redoing. That run directory is the
rollback path and it lives under gitignored `.ingest/` — move it somewhere durable
for a large run.

Full procedure, failure-mode table and the manual fallback:
[`prompts/migration-backfill.md`](./prompts/migration-backfill.md).

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
  attempts:           11
  total tokens:       603K input + 108K output
  mean per attempt:   65K tokens
  estimated cost:     $1.91  (rates as of 2026-08-03; upper bound — ignores prompt-cache hits)
    claude-haiku-4-5-20251001     300K in,      15K out
    claude-sonnet-5            304K in,      92K out
```

Rates come from [`config/pricing.yaml`](./config/pricing.yaml), which carries an
`as_of:` date printed beside every dollar figure. A model absent from the table
counts as $0.00 — correct for a local backend, so `status` separately names any
*cloud* model that's missing rather than letting a stale table quietly understate
the bill.

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
│   │                       #     abstract / Results / captions, fed through scorer).
│   │                       #     `anchor_is_substantive` is shared with the critic —
│   │                       #     anchors are both denominator and author instruction
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
│   ├── runner.py           #   State-machine driver — one `_phase_*` wrapper per
│   │                       #     phase over the phases/ modules below. The roles
│   │                       #     persisted to ingest_iterations are enumerated in
│   │                       #     `db/iterations.VALID_ROLES` (never rename those)
│   ├── commentary.py       #   Commentary-shaped-PDF guard (refuses to promote a
│   │                       #     Research Highlight / News & Views as type: paper)
│   ├── context.py          #   Shared phase Context (each phase reads/writes it)
│   ├── fitness.py          #   Tournament + improvement-rule lenses;
│   │                       #     `combined_quality` = 0.5·semantic + 0.5·salience,
│   │                       #     salience down-weighted below 10 anchors; selection
│   │                       #     quantizes each axis to 0.01 before blending
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
│   ├── _grade_synthesis.py #   Synthesis/idea fidelity (misattribution check)
│   ├── lint/               #   One module per check family (link, yaml, staleness,
│   │                       #     claim_anchors, concept_contract, db_checks, …)
│   ├── check_grounding.py  #   Structural citation check
│   ├── check_coverage.py   #   Recall surface — unreferenced top-N hits
│   │                       #     for a synthesis/idea page's topic_seed
│   ├── benchmark_fixture.py #  Hand-curated-fixture benchmark
│   ├── evolve.py           #   Standalone memory-evolution proposals
│   ├── search.py reindex.py status.py lint.py audit.py claims.py pdf_search.py ...
├── concepts/               # Concept-hub surfacing + scaffold + reciprocal linking
├── claim_graph/            # Content-addressed claim identity + edge cache
├── synthesis_candidates/   # Detect paper clusters lacking a synthesis page
├── migrate/                # Import one-paper-per-PDF pages from an older/simpler wiki
│                           #   (preflight → inspect → apply → verify; zero tokens)
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
- **The state DB** (sqlite, at
  `~/.local/share/researchwiki/repos/<name>-<hash>/state.db`) is a derived
  index over `wiki/` + `papers/` + caches. Run `researchwiki db rebuild` to
  reconcile drift. It lives outside the repo on purpose, so a sync daemon
  never sees its WAL files. `<hash>` comes from the checkout's absolute path,
  which keeps separate wikis on one machine from sharing a `claims` table —
  but also means **moving or renaming the checkout points the tooling at a
  fresh, empty DB**. `RESEARCHWIKI_DB_PATH` pins it if that is a risk
  ([below](#pinning-the-state-db-researchwiki_db_path)). One table,
  `ingest_iterations`, is the only thing here not reconstructable from
  markdown; it is a per-machine record of what this machine ran.
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

Rates live in [`config/pricing.yaml`](./config/pricing.yaml) (Anthropic + OpenAI,
USD per million tokens) with an `as_of:` date and the `sources:` URLs they were
read from. `agents/model_config.py` — the same module that routes phases to
models — resolves a rate by **longest prefix**, so the
dated build IDs the APIs return resolve to their family — the estimator used to
key on bare family names and silently priced those at $0.00. Time-boxed rates are
expressed with `until:` (Sonnet 5's introductory pricing lapses 2026-08-31).

Every printed figure is an **upper bound**: `ingest_iterations` records only
input/output totals, and a prompt-cache hit costs 0.1× input, so runs with
`cache_prompt=True` really cost less than shown. To refresh, correct the rates and
bump `as_of:` in the same edit.

Absolute cost is **config-dependent** — it rides on whichever
`config/models.*.yaml` file `RW_MODELS_CONFIG` selects and that provider's
token pricing. The current active default is `models.chatgpt.yaml`:
`gpt-5.6-luna` drives every role. `gpt-5.4-mini` used to hold the
deterministic short-output roles (classifier / extractor) as the cheap
option, which it is not — mini is a 5.4-generation model and the 5.6 line cut
prices, so it costs 3.75x luna per token with no offsetting quality
argument. When no `config/models.yaml` is present the loader falls back to a
hardcoded table in `agents/model_config.py` that mirrors this template.

At the rates in `config/pricing.yaml` (per 1M tokens: `gpt-5.6-luna` $0.20 in
/ $1.20 out), a single-draft ingest runs
roughly ~26K input + ~3.5K output tokens across all roles, with **author**
and **target_claims** together accounting for nearly all of it and everything
else a long tail — so a typical paper lands around **~$0.01** (measured mean
over 13 ingests on this corpus), and a 2-draft author tournament (`-n 2`)
roughly double. Swapping the quality roles to `gpt-5.6-terra` ($2.00 / $12.00)
moves that to **~$0.07/paper** — 10x the rate, ~9x the bill once normalized
for paper size. The two grader runs
are semantic-only (no LLM) and free; memory-evolution proposals cost in
proportion to how many synthesis neighbors clear the cosine prefilter.

Treat these as order-of-magnitude — token counts are config-independent
(same prompts), but dollars move with the model assignment. The cost
estimator prices only models present in `config/pricing.yaml`; the current
GPT-5.6 defaults *are* listed, so `insights` and `status` report real figures
for them. A model missing from that table resolves to `$0.00`, which is
correct for a local backend but reads as "unpriced" for a cloud one —
`status` names any cloud model it could not price, so a stale table is
visible rather than a silently understated bill. The agent is
calibrated to spend on *fidelity* (claim-grading, critic, evolve), not on
speed — that hasn't changed across model swaps.

### When to opt out of which phases

- **`--use-stub`** — full offline / deterministic mode. No API calls. Use
  for harness tests and CI.
- **`--no-llm-reconcile`** — skip the LLM metadata extractor (`gpt-5.6-luna`
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
- **`sal` is low on a page that reads complete.** Usually the anchor set,
  not the page. `salience_score` is recall against a fixture synthesized from
  PDF structure, and the `critical` tier — weight 3 in
  `scorer.IMPORTANCE_WEIGHTS` — is the abstract, a median 75% of the weighted
  denominator (p90 94%). So whatever `pdf.sections.extract_abstract`
  over-reaches into becomes the score, and it over-reaches at both ends: 42% of
  corpus abstracts hit its 4000-char cap (median 17 sentences / 467 words where
  a real abstract is 5–12 / 150–350), bleeding into the introduction, and
  occasionally at the front into the masthead and author list. Front-matter is
  a *guaranteed* miss — a page should omit a funding disclaimer — so it inflated
  the denominator with content no page could earn credit for: measured at 62%
  of papers carrying at least one, median 4.0% of the weighted denominator,
  p90 14.5%, max 79%.

  Three guards in `grade/salience.py` bound this, and the **order matters**:

  1. `_abstract_is_prose` rejects the whole region when it averages under 10
     words per sentence — the bibliography-slab and title-fragment shapes,
     whose implied ceiling was ~0.21 (unreachable, not merely low).
  2. `anchor_is_substantive` drops per-sentence artifacts: under 60 chars,
     boilerplate regex hits, lowercase-initial mid-clause fragments, two or
     more academic credentials, three or more semicolons.
  3. `_MAX_ABSTRACT_ANCHORS = 12`, applied to the *survivors of step 2 in
     document order* — **not** to the raw sentence index. `bjornsson-2020` is
     the case that decides this: its abstract region runs 28 sentences whose
     leading 12 are masthead and author list, with `BACKGROUND:` at index 12,
     so a raw positional cap keeps only junk and discards every finding
     (0.24 → 0.06). Filtering first inverts it to 0.24 → 0.34.

  12 is calibrated against the hand-curated `benchmark-fixtures/`: of curated
  headline items that localize to an abstract sentence at all, 100% land in the
  first 10 (max index 8), so the cap discards nothing a human judged
  load-bearing. Against an all-sentences baseline on a 58-paper sample, the
  three guards together move median salience 0.322 → 0.344 (mean 0.319 → 0.340;
  29 papers up, 11 down, 18 unchanged). The gains are the
  structurally-penalized tail — `chen-2025` +0.16, `christian-2026` +0.16,
  `jaganathan-2019` +0.16, `bjornsson-2020` +0.10. The 11 that scored lower
  moved little (worst -0.07) and are pages that had been earning credit against
  introduction bleed.

  Anchors reach the author as well as the grader — `phases.revise.coverage_gaps`
  turns an uncovered critical anchor into an *additive* instruction — so the
  same substance test filters both. A bad anchor there makes the page worse
  rather than merely mis-scored. It still runs on read because pages graded
  before the guards landed carry the old junk in their stored `missed_anchors`.

  Tightening `extract_abstract` itself was tried and rejected: preferring a
  ≤400-word paragraph in its path-2 (largest-paragraph) branch changed the
  abstract pick on 13/58 papers, and on `chen-2025` picked a *different*
  paragraph (543 → 52 words) rather than a shorter correct one. The anchor
  filter covers the same ground without touching extraction that other
  consumers share.

---

## Provider setup in depth

Companion to `README.md`'s Providers/Model-config tables, which cover the
default path (`config/models.chatgpt.yaml` + `OPENAI_API_KEY`, nothing else
to configure). This section is for the paths beyond that default: switching
configs without copying, mixing providers per role, running fully local, and
chat-relay for users with no API key at all.

### Switching configs without copying (`RW_MODELS_CONFIG`)

Instead of copying a template over your active `config/models.yaml`, point
the loader at any config file via the `RW_MODELS_CONFIG` env var. A bare
filename resolves under `config/`; an absolute/path-separated value is used
verbatim. This is the clean way to A/B a backend or run a one-off without
disturbing your default:

```bash
RW_MODELS_CONFIG=models.glm.yaml researchwiki agent ingest inbox/paper.pdf   # one-off
# or in .env to make it your persistent default backend
```

Precedence: `RW_MODELS_CONFIG` → `config/models.yaml` → hardcoded defaults.
`researchwiki status` prints the active config (with a `[RW_MODELS_CONFIG]`
marker when the env var is in effect), so "which config am I using?" is
always answerable. Unlike `RW_LLM_PROVIDER` (which forces one provider
across every phase and defeats per-role mixing), `RW_MODELS_CONFIG` selects
a whole file that keeps its own per-role mixing.

### Pinning the state DB (`RESEARCHWIKI_DB_PATH`)

`state.db` is not stored in the repo — it sits under
`~/.local/share/researchwiki/repos/<name>-<hash>/`, where `<hash>` is derived
from the checkout's absolute path. That keeps two wikis on one machine from
sharing a `papers`/`claims` table. The cost is that the DB is bound to *where
the checkout is*: move it, rename a parent directory, or clone the same wiki
to a second location, and the tooling silently opens a new, empty database.

The failure is quiet rather than loud, which is what makes it worth knowing.
`researchwiki db rebuild` cheerfully repopulates `papers` and `claims` from
your markdown, so `status` and `search` look healthy — while two things that
are *not* derivable from markdown stay missing: **claim grades** (until you
re-grade, every claim is uncitable, so no synthesis page can ground against
it) and **`ingest_iterations`** (the cost/quality telemetry behind
`researchwiki insights`).

Set `RESEARCHWIKI_DB_PATH` to an explicit file to opt out of path-keying:

```bash
# in .env, or exported in your shell
RESEARCHWIKI_DB_PATH=~/.local/share/researchwiki/my-wiki.db
```

It overrides the per-repo path entirely, so the DB follows the *wiki* rather
than the directory the code happens to live in. Worth setting before you
reorganize a checkout; it takes precedence over everything else.

If you are already stranded, nothing is lost — the old database is still on
disk under its previous key. Compare and copy the richer one over the new:

```bash
ls -la ~/.local/share/researchwiki/repos/
sqlite3 <old>/state.db "SELECT COUNT(*) FROM ingest_iterations;"   # richer wins
cp <new>/state.db <new>/state.db.bak
sqlite3 <old>/state.db ".backup '<new>/state.db'"
researchwiki db rebuild        # grades survive: claims upsert by slug
```

Failing that, `researchwiki grade regression --missing-only` re-derives the
grades locally in a few minutes with no API calls. Telemetry is not
re-derivable — see the note on multiple machines below.

### Working from more than one machine

A common setup is one wiki, two computers, with `wiki/` and `papers/` on a
sync service. What syncs, and what each machine derives for itself:

| | Where it lives | How the second machine gets it |
|---|---|---|
| Pages, PDFs | `wiki/`, `papers/` | Sync service — this is the source of truth |
| Claims, grades, indexes | state DB, `.tantivy-index/`, `.semantic-cache/` | Re-derived locally, identically |
| LLM-judged caches | `.claim-graph/`, `.evolve-cache/` | Re-judged if needed; the *outcome* already synced as wikilinks |
| Cost/quality telemetry | `ingest_iterations` | **Stays on the machine that ran the ingest — by design** |

After the sync service delivers new pages, bring the second machine to parity:

```bash
researchwiki db rebuild && researchwiki reindex
researchwiki grade regression --missing-only    # no API calls
```

Grading needs only the PDFs (which sync) and the local bi-encoder, so it is
free and deterministic — both machines compute the same scores. Everything
that constitutes knowledge therefore converges without copying any database.

**Telemetry is deliberately not shared.** `ingest_iterations` records what a
given machine actually ran — which model, how many tokens, which drafts were
discarded. That is a per-machine operational log, not part of the wiki, and
`insights` is most meaningful read that way. Expect each machine to report
only its own ingests, and read `status`'s cost rollup as local spend.

**Do not put the state DB on the sync service.** Two machines writing one
SQLite file through a sync daemon corrupts it — WAL sidecars are exactly the
files these daemons handle worst. Keeping the DB outside the synced tree is
why it lives under `~/.local/share/` in the first place.

### Chat-relay (subscription users — no API key)

If your only model access is a **chat subscription** (Claude.ai Pro, ChatGPT
Plus, Cursor Pro), the framework still runs end-to-end. The chat-relay
provider delegates each LLM call to whatever chat agent is already in your
terminal via a filesystem protocol — no API key, server, or per-paper cost.

**How it works.** `agent ingest` emits a prompt at
`.llm-relay/pending/{op_id}.prompt.json` and blocks; the chat agent reads
it, writes `.llm-relay/completed/{op_id}.response.json`; the CLI moves on.
One ingest is 5–8 handoffs — a few minutes if the agent watches for
prompts. Protocol spec: [`prompts/chat-relay.md`](./prompts/chat-relay.md).

```bash
export RW_LLM_PROVIDER=chat-relay      # or add to .env
researchwiki agent ingest inbox/some-paper.pdf
```

Then tell your chat agent *"watch `.llm-relay/pending/` and respond to each
prompt as it appears."* Schema validation + retry-with-feedback is built in
(up to 3 attempts), so you don't babysit format drift.

**Caveats:**
- **Wall clock is bounded by your attention** — each phase blocks on the
  agent; times out at 10 min/phase if it walks away.
- **Cost dashboards show $0** — tokens aren't measurable through the relay.
- **One ingest at a time** — parallel ingests queue to the same agent, which
  responds serially.
- **Cache reuse on re-runs** — `op_id = sha1(phase|prompt)[:12]`, so a crash
  mid-ingest reuses completed phases. `RW_RELAY_FRESH=1` forces re-prompting.

### Per-role mixing

`RW_LLM_PROVIDER` is a **global override** — it forces *every* phase to one
provider and **silently defeats per-role mixing** whenever set (**including
in `.env`**). To mix (e.g. chat-relay `author`, local everything else):
**unset `RW_LLM_PROVIDER`**, then set `provider:` per role in
`config/models.yaml`:

```yaml
roles:
  author:     {provider: chat-relay, model: claude-via-relay,    temperature: 0.5, max_tokens: 16000}
  critic:     {provider: lmstudio,   model: qwen3.6-35b-a3b-mlx, temperature: 0.3, max_tokens: 12000}
  judge:      {provider: lmstudio,   model: qwen3.6-35b-a3b-mlx, temperature: 0.2, max_tokens: 12000}
  classifier: {provider: lmstudio,   model: qwen3.6-35b-a3b-mlx, temperature: 0.1, max_tokens: 6000}
  proposer:   {provider: lmstudio,   model: qwen3.6-35b-a3b-mlx, temperature: 0.3, max_tokens: 6000}
  extractor:  {provider: lmstudio,   model: qwen3.6-35b-a3b-mlx, temperature: 0.0, max_tokens: 8000}
```

`author`/`evolve`/`debug` follow the `author` role. For this mix,
**`RW_LLM_BASE_URL` must be set** (e.g. `http://localhost:1234/v1`) so local
roles pass the readiness gate — **even on the default port** — and the
relay needs an active servicer for the `author` phase.

### Local LLMs (LM Studio / vLLM / llama.cpp / ollama)

Any OpenAI-compatible server works alongside or instead of Anthropic. For a
**fully local** setup, the model we dogfood is **Qwen3.6-35B-A3B** (a
35B-param MoE, ~3B active — runs on a 32 GB Apple-silicon laptop via LM
Studio's MLX build): across our ingest history it authored drafts at mean
claim-fidelity ≈ 0.78, within a hair of Gemini 3.5 Flash (≈ 0.80) and above
Solar (≈ 0.75) — good enough to keep **every** role local, not just the
cheap ones. `config/models.lmstudio.yaml` points all six roles at one such
model. If you'd rather mix, a common split is to keep **author** on
Anthropic for peak fidelity and route **classifier** / **proposer** /
**reconcile** to a smaller local model to drop marginal cost toward zero.

**Start a server** (LM Studio is simplest — download a model, click **Start
Server**; default `http://localhost:1234/v1`):

```bash
curl http://localhost:1234/v1/models | jq '.data[].id'   # confirm it's up
```

vLLM: `vllm serve <model> --port 1234`. llama.cpp:
`llama-server -m <gguf> --port 1234`. ollama: `http://localhost:11434/v1`.
Real OpenAI: `RW_LLM_BASE_URL=https://api.openai.com/v1` + `OPENAI_API_KEY`.

**Route a role** in `config/models.yaml`:

```yaml
roles:
  author:     {provider: anthropic, model: claude-sonnet-4-6,          temperature: 0.5, max_tokens: 2500}
  classifier: {provider: lmstudio,  model: meta-llama-3.1-8b-instruct, temperature: 0.1, max_tokens: 200}
  proposer:   {provider: lmstudio,  model: meta-llama-3.1-8b-instruct, temperature: 0.3, max_tokens: 200}
```

**Sizing:** a **7–8B** model handles the short roles well —
`classifier`, `short_name`/`keywords`, `reconcile` (verify against S2) —
but writes shallow `author` drafts and weak `judge` verdicts, so on small
hardware keep those two on Anthropic and route the rest local. A **~30B+
MoE like Qwen3.6-35B-A3B** closes that gap (near-cloud author fidelity in
our runs) while still fitting a 32 GB laptop, which is why it's our
recommended all-roles-local model; dense 70B+ raises the ceiling further
but needs 40+ GB VRAM.

**Caveats:** no prompt caching (author is ~free on local anyway); token
counts may be approximate/zero (dashboard shows $0 — accurate); pure-local
is supported (readiness checks are provider-aware — leave
`ANTHROPIC_API_KEY` unset).

---

## CLI reference

Your LLM picks among these based on what you ask; you can run any directly.

```bash
cp ~/Downloads/some-paper.pdf inbox/                 # 1. drop a PDF

researchwiki agent ingest inbox/some-paper.pdf       # 2. default: LLM authors + grades + promotes + back-links + evolve
researchwiki agent ingest inbox/*.pdf                #    ≥2 PDFs auto-batch with checkpoint/resume

researchwiki ingest inbox/some-paper.pdf --category cgt   # 2-fallback: digest-only (recovery / unextractable / custom voice)

researchwiki synthesize --title "CRISPR off-target strategies" \
    --slug crispr-off-target-strategies --topic-seed "CRISPR off-target prediction" \
    --papers cgt/smith-2024-... cgt/jones-2025-...   # 3. scaffold a synthesis page (idea pages are manual)

researchwiki reindex                                 # 4. rebuild Tantivy + semantic index (~10s)

researchwiki search "CRISPR off-target"              # 5. hybrid (BM25 + semantic RRF); --mode bm25|semantic
researchwiki search --like compbio/smith-2024-...    #    See-Also on a page; --see-also adds 2 related/hit

researchwiki evolve cgt/some-stem                    # 6. synthesis-evolution proposals for a paper
researchwiki neighbors compbio/some-stem --needs-ingest --year 2024-2026   # 7. what to ingest next
researchwiki attach compbio/some-stem ~/Downloads/Methods.pdf              # 8. attach supplementary files

researchwiki status                                  # 9. health: index, costs, pending proposals
researchwiki lint --fix                              #    consistency report; --fix auto-inserts back-links
researchwiki audit > /tmp/audit.md                   #    citation-graph audit (Semantic Scholar)
```

`researchwiki status` on a populated wiki prints per-category counts,
cross-link density, orphans, and inbox backlog; on an empty wiki it prints
`Pages: 0` cleanly.

### All commands

| Command | Purpose |
|---|---|
| `agent ingest <pdf> [--supplementary <f>...]` | Full pipeline: reconcile → extract → crosslink → parallel-author → grade → critic/evolve/debug → promote → propose evolutions. Provider API key required (`OPENAI_API_KEY` by default). |
| `ingest <pdf>... [--category] [--supplementary]` | Digest-only (no LLM authoring): DOI, S2 metadata, stem, crosslinks, anchoring → `.ingest/{stem}-digest.md`. Author the page yourself. |
| `attach <category/stem> <file>` | Attach a supplementary file to an existing page; copies into `papers/{stem}.supp/`, updates YAML. |
| `neighbors <doi-or-stem>` | S2 citation-graph neighbors. `--mode references\|citations\|recommendations\|all`, `--year`, `--needs-ingest`. Structured fields only. |
| `evolve <category/stem>` | Neighboring synthesis pages to edit in light of a paper → proposals in `.ingest/{stem}-evolution-proposals/`. |
| `backfill <hook\|keywords\|doi>` | One-shot: populate the named field on existing pages (hook + keywords via LLM from page prose; doi via Semantic Scholar → Crossref with a sanity check). |
| `migrate <preflight\|inspect\|apply\|verify>` | Bulk-import one-paper-per-PDF markdown from an older release or a simpler LLM wiki. Zero tokens; normalizes H2 headings and frontmatter keys before committing so claim extraction works. See `prompts/migration-backfill.md`. |
| `synthesize --title [...] [--papers]` | Scaffold `wiki/synthesis/{slug}.md`. Idea/reference pages are manual. |
| `candidates <concepts\|synthesis>` | Surface opportunity signals: un-scaffolded concept hubs (concepts) or uncovered paper clusters warranting a synthesis page (synthesis). |
| `reindex [--no-semantic]` | Rebuild Tantivy + semantic index from `wiki/`. |
| `search "<query>" [--mode ...]` or `--like <stem>` | Hybrid retrieval (RRF over BM25 + semantic) by default. |
| `status` | Dashboard: counts, density, orphans, backlog, index health, pending proposals, 7-day cost. |
| `lint [--fix]` | Orphans, broken/missing wikilinks (auto-fixable), stale syntheses, missing keywords/DOIs, year drift, stale proposals. |
| `audit` | Citation-graph audit vs Semantic Scholar: wikilinks without a real citation, and vice versa. |
| `retraction-check`, `preprint-check`, `orcid-lookup` | Structured PubMed / bioRxiv / ORCID queries. |
| `claims "<query>" [--k N]` | Grounded-citation search over the pre-graded claims table (atomic bullets + `[[stem#slug]]` citation anchors + support scores). |
| `pdf-search <stem> "<query>" [--k N]` | BM25 inside one paper's PDF chunks — pull an exact number/passage the page didn't quote. |
| `check-grounding <page> [--strict]` | Structural gate: flag claim-bearing units lacking a `[[wikilink]]`. `--strict` also fails `*(model prior)*`-marked units. Exits 1 when it finds something. |
| `check-coverage <page>` | Advisory recall gate: wiki papers ranking high on the page's `topic_seed` that the page never cites. Run after `check-grounding` + `grade synthesis` on any synthesis/idea/concept page. |
| `grade <paper\|synthesis\|regression>` | Fidelity scoring against source PDFs: `paper <stem>` (fidelity + salience), `synthesis <page>` (cross-paper misattribution), `regression [--missing-only]` (re-grade all + drift diff). |
| `concepts <term> --thesis "…"` | Scaffold a concept hub from a recurring term; refuses without a thesis. `--upgrade-spokes` backfills `[[stem#slug]]` citations on existing hubs; `concepts refresh <slug>` drafts a hub's *Cross-domain connections* from typed claim edges (review-gated, never auto-applied). |
| `claim-graph [--tensions\|--contradicting\|--neighbors]` | Query the LLM-judged claim-edge graph — including the typed edges `claim-overlap` records without writing a Related-Papers bullet. `claim-graph promote [--apply]` transitions candidate edges. |
| `bootstrap-categories` | Propose a category taxonomy from `inbox/` papers; `--apply` creates the dirs. |
| `suggest-splits [--category <cat>\|--all]` | Cluster `wiki/other/` or a populated category and propose taxonomy changes. Review-gated; nothing auto-creates a category. |
| `db <rebuild\|verify>` | Rebuild the structured mirror from markdown after any manual page edit; `verify` reports drift without writing. (`db papers` / `db query` below.) |
| `init [--scaffold-only]` | First-time setup wizard, or just create the directory scaffold. |
| `mcp-serve` | Read-only MCP server (search / claims / check-grounding) for Claude Desktop and IDE clients. |
| `eval-classifier` | Leave-one-out evaluation of the Tantivy-backed category classifier. |
| `benchmark-fixture <stem> [--repeat N] [--llm]` | Score page authoring against a hand-curated `benchmark-fixtures/` fixture. `--repeat` keeps drafts in memory; for a single authored page use `agent ingest … --force-sandbox`, never a bare `agent ingest` (it would promote a fixture paper into your corpus). |
| `claim-overlap <stem> [--sim N] [--top N] [--dry-run] [--json]` | Proactively cross-link a newly-ingested paper: finds existing papers with near-paraphrase claims, LLM-judges each as a real relationship vs coincidence, and auto-adds reciprocal Related-Papers `[[wikilinks]]` for confirmed matches. Run after `db rebuild`. |
| `db papers [--year/--category/--page-type/--no-doi/--venue/--author/--status] [--count] [--json]` | Structured lookups over the frontmatter mirror — counts/filters ("cgt papers from 2024", "papers missing a DOI") without re-reading markdown. `db query "SELECT…"` for ad-hoc read-only SQL. |
| `insights [--days N] [--json]` | Analytics over the ingest telemetry log: draft quality + cost by model, section difficulty, token spend by role, draft decisions. Read-only, no LLM. |

Every operation appends a parseable entry to `wiki/log.md` (inside `wiki/`
so an Obsidian vault opened there can browse it).

---

## When to do what (quick reference)

| You did this | Then run |
|---|---|
| Dropped PDF in `inbox/` | `researchwiki agent ingest inbox/<file>.pdf` |
| Dropped ≥2 PDFs in `inbox/` | `researchwiki agent ingest inbox/*.pdf` (auto-batches with checkpoint/resume) |
| Batch ingest crashed mid-run | `researchwiki agent ingest --resume .ingest/batch-<ts>/` |
| Edited a wiki page manually | `researchwiki db rebuild && researchwiki reindex` |
| Have paper pages from an older/simpler wiki | `researchwiki migrate preflight <src>`, then `inspect` (see *Migrating an existing corpus*) |
| A paper page yields no citable claims | `researchwiki lint --json \| jq .zero_claim_papers` — almost always non-canonical H2 headings |
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
