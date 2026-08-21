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
`.ingest/{stem}-evolution-proposals/`. Those are reviewed by an LLM — which
verifies each patch against the source paper before recommending it — and applied
only on human approval; a human can of course review them directly instead. The
result is a markdown wiki on disk that compounds: every new paper makes related
synthesis pages slightly more correct.

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

Here's what an ingest looks like when you run it. The excerpts below are from a
real run on `kim-2026-structural-motif-search-across-the-protein` (Folddisco) —
values taken from that attempt's `ingest_iterations` rows, not hand-crafted. The
run predates the `sal`/`coh` columns in the grade line, so those two axes are
absent from the excerpt and documented separately below.

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
[agent] attempt_id=ac715918...
[agent] pdf=some-paper.pdf
[agent] mode=real (anthropic/claude-sonnet-4-6)
[agent] n_drafts=2  use_semantic=True  max_evolve=1
[agent] reconcile → stem=kim-2026-structural-motif-search-across-the-protein year=2026 type=research
[agent] extract   → sections=['references', 'methods']
[agent] crosslinks → 6 verified candidate(s)   (3 citation-graph + 3 topical)
[agent] author #1  → stance=balanced  t=0.2
[agent] author #2  → stance=skeptical t=0.6
[agent] grade   → draft 101 sem=0.77 graded=12 drift=0 bm25=12.1
[agent] grade   → draft 102 sem=0.77 graded=15 drift=0 bm25=15.4
[agent] tournament → winner draft 102 (tied on sem; decided on the tail)
[agent] critic     → no weak claims; pass-through
[agent] promote    → wiki/compbio/kim-2026-structural-motif-search-across-the-protein.md (6 back-links added)
[agent] shortname  → 'Folddisco'
[agent] keywords   → ['structural motif search', 'Folddisco', 'pairwise geometric features', ...]
[agent] evolve     → no actionable proposals (knn=8 above_thr=6 judged=6 actionable=0)
```

A current grade line carries four axes: `sem` is the page-level mean of per-claim
bi-encoder cosines (fidelity); `sal` is salience-recall against the PDF-anchor
synthetic fixture; `coh` is the structural-conformance score (0..1, sum of
weights of passing checks); `bm25` is the page-level mean of per-claim top-1
BM25 retrieval scores. The tournament keys on `combined-quality = 0.5·sem +
0.5·sal` — so a draft with low semantic but high salience can beat a
high-semantic-low-salience peer if the combined number wins. The lexicographic
tail (coherence → -drift → n_graded → bm25 → weakest_score) decides ties.

The run above is a tie broken by that tail, which is the case worth seeing: both
drafts scored `sem=0.77`, so the primary scalar could not separate them, and the
winner was the draft that graded more claims (15 vs 12) at a higher BM25 (15.4 vs
12.1). A tournament outcome that looks arbitrary on the primary axis usually isn't
— check the tail before assuming the selection is noise.

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

`wiki/compbio/kim-2026-structural-motif-search-across-the-protein.md`:

```yaml
---
title: "Structural motif search across the protein universe with Folddisco"
authors: Hyunbin Kim, Rachel Seongeun Kim, Milot Mirdita, Jaewon Yoon, Martin Steinegger
senior_authors: Martin Steinegger
year: 2026
doi: 10.1038/s41587-026-03162-9
venue: Nature Biotechnology
type: paper
category: [compbio]
pdf_path: "[[kim-2026-structural-motif-search-across-the-protein.pdf]]"   # Obsidian wikilink → click-to-open; real file at papers/{stem}.pdf
publication_status: published
short_name: Folddisco
hook: "Structural motif search across 53M AlphaFold structures in seconds using position-independent geometric indexing with novel side-chain dihedral angles; 20× faster querying and 4× smaller indices than pyScoMotif while improving sensitivity."
keywords: [structural motif search, Folddisco, pairwise geometric features, side-chain dihedral angles, rarity-based scoring, ...]
ingested_at: 2026-06-05T15:43:46
---

## Summary
**Folddisco** indexes pairwise geometric features of residue pairs ...

## Key Contributions
- ...

## Methodology and Architecture
...

## Results
...

## Limitations
- ...

## Related Papers
- [[compbio/van-kempen-2024-fast-and-accurate-protein-structure]] — Foldseek is cited as the closest related structural search tool; Folddisco extends its infrastructure but addresses the nonlinear motif-matching problem Foldseek cannot handle
- [[compbio/abramson-2024-accurate-structure-prediction-of-biomolecular]] — AlphaFold2/3-scale predicted structure databases are the primary target corpus motivating Folddisco's scalability requirements
- ...
```

YAML carries the full audit (DOI, venue, source-PDF wikilink, the catalog gloss
in `hook:`, and the ingest timestamp). The body is six standard sections capped
to budget. Every wikilink in `Related Papers` was verified — either
citation-graph confirmed or LLM-judged topical with a one-line rationale. This
run's six candidates split 3 citation-graph (MMseqs2, Foldseek, AlphaFold 3 — all
cited by the source) and 3 topical.

### 4. Review the evolution proposals

The Folddisco run above produced **none** — `knn=8 above_thr=6 judged=6
actionable=0`, meaning six synthesis neighbours cleared the cosine prefilter, all
six were judged, and none warranted an edit. That is the common outcome and worth
seeing: the judge declining six times is the prefilter doing its job, not a
failure. The format below is therefore from a different run (`wei-2026`, an
EVO-based promoter predictor, against the Hyena/Evo lineage page):

`.ingest/wei-2026-...-evolution-proposals/refine__synthesis__dna-foundation-models-in-the-hyena-evo-lineage.md`:

```yaml
---
source: compbio/wei-2026-evosnr-prom-predicting-promoters-at-single-nucleotide
target: synthesis/dna-foundation-models-in-the-hyena-evo-lineage
verdict: refine
confidence: 0.82
---

# REFINE — [[synthesis/dna-foundation-models-in-the-hyena-evo-lineage]]

**Rationale:** applies Evo 7B via LoRA to prokaryotic promoter prediction at
single-nucleotide resolution, so it belongs in the Evidence section as a
downstream application of the lineage's byte-level tokenization ...

## Patch

- Cite `[[compbio/wei-2026-evosnr-prom-predicting-promoters-at-single-nucleotide]]`
  in the body (inline, plus a matching `[^id]:` footnote under `## References` if
  the page uses footnotes — synthesis pages cite via the body, there is no
  `referenced_papers:` field).
- Section: **## Evidence from the wiki**
- New bullet:
  > [[compbio/wei-2026-evosnr-prom-predicting-promoters-at-single-nucleotide]] —
  > applies Evo 7B via LoRA + lexicon-enhanced embeddings to prokaryotic promoter
  > prediction at single-nucleotide resolution ...
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

## Importing a reference-manager library

`migrate` above imports *pages*. If what you have is a library in Zotero,
Paperpile, Mendeley or ReadCube — PDFs and metadata, no prose —
`researchwiki import` is the sibling command, and the two are not
interchangeable: `migrate` deliberately refuses this case, because there is no
authored text to preserve. Pages here are authored normally, by `agent ingest`.

**The export is the asset.** A reference manager already holds a curated DOI,
title, authors and year for every record — exactly the fields `agent ingest`
otherwise rediscovers through its most failure-prone stretch (PDF extract → DOI
hunt → S2 lookup → LLM reconcile → `metadata_sanity`), which is where every
`unknown-` stem and wrong-but-resolving DOI comes from. So `inspect` records the
exact `--doi/--title/--authors/--year` argv each record contributes, and `apply`
feeds them through batch mode: the hard part becomes a lookup.

Export **BibTeX or RIS**, not CSL-JSON — the first two carry attachment paths and
CSL-JSON carries none from any exporter seen, which costs you the `declared`
pairing rung. Parsing is deliberately tolerant, because the files that break a
strict parser are the ones people actually have: a real 532-item ReadCube export
carries a 4-character `PMID` tag where the convention is 2, an always-empty `XX`
tag on 385 records, citekeys with `:` and non-ASCII that strict BibTeX forbids,
and CRLF endings that make a naive `"\r\n"` split return one giant record.

### Pairing, and why it reports its own confidence

`<pdf-root>` is optional. When you give one, every PDF under it is read **once**
(the naive record × PDF loop is O(n²) and turns a free phase into an overnight
one), then three rungs run as three passes over all records — so a confident DOI
match always wins a file over a merely plausible title match:

| Rung | Basis |
|---|---|
| `declared` | a path the export itself named |
| `doi` | the DOI printed inside the PDF equals the record's |
| `title` | the PDF's opening text covers the record's title |

The title rung publishes a *margin*, not just a score, because a near-tie means
the score came from vocabulary two records share rather than from identity.
Measured against 313 DOI-confirmed pairs (the DOI gives ground truth, so title
matching can be scored against it), requiring a 0.05 margin removed every wrong
pairing at the cost of four correct ones — and those four land in `review` with
their candidates listed, rather than silently attaching the wrong PDF.

### Triage: three verdicts, the detail in reason strings

`apply` acts on `ready` only. Four gates are worth knowing:

- **`no-text-layer`** — the silent one. A scanned PDF extracts to nothing, ingest
  logs a warning nobody reads, and the page then passes every later gate on
  grounding that does not exist.
- **`superseded-by-journal`** — a preprint whose published version is also in the
  export. Invisible to DOI dedupe, since the pair carries two different DOIs; the
  real library held 10 such pairs and zero duplicate DOIs.
- **`duplicate-doi`** — its complement, the same DOI twice. That library had none,
  but concatenated or merged exports produce them readily.
- **`maybe-commentary`** — a Research Highlight would otherwise be ingested as the
  paper it describes. See CLAUDE.md → Page Types §7.

Both dedupe gates pick their survivor by a total order rather than by input
position, because the same library exported twice listed its records in different
orders and position would import differently from identical data.

With no PDFs at all the run is still worth doing: the report lists every record
that clears every gate *except* having a file, with its DOI, deduplicated — on a
cloud-hosted library that fetch list is the most useful thing the command
produces.

### The phases

| Phase | Writes | Cost |
|---|---|---|
| `import preflight <export>` | nothing | local; parse and count only |
| `import inspect <export> [pdf-root]` | run dir only | local; pairing + gates |
| `import apply --run <dir> [--limit N]` | `inbox/` → `wiki/` + `papers/` | **the only phase that spends** |
| `import verify --run <dir>` | nothing | local |

```bash
researchwiki import preflight ~/lib/library.ris
researchwiki import inspect   ~/lib/library.ris ~/lib/pdfs
less .ingest/import-*/report.md                      # read this before applying
researchwiki import apply --run .ingest/import-<ts> --limit 30 --dry-run
researchwiki import apply --run .ingest/import-<ts> --limit 30
researchwiki db rebuild && researchwiki reindex
researchwiki import verify --run .ingest/import-<ts>
```

**Stage it with `--limit`.** `apply` re-checks per record whether a paper is
already in the wiki, or already sitting in `inbox/` from a wave that failed — the
one set of facts deliberately *not* frozen in the manifest, because it is a fact
about now. That is what makes the next `--limit 30` mean "the next 30 still
pending" rather than "the first 30, again", so waves compose instead of repeating.

Everything else *is* frozen, so `apply` cannot reach a different conclusion than
the `inspect` you read. There is no journal and no staging directory here, unlike
`migrate`: the only mutation is copying a PDF, and everything after it belongs to
`_ingest_batch`, which already keeps a crash-safe checkpoint. Recovery is
`agent ingest --resume <batch-dir>` — the path you already know.

Per-paper cost is ordinary `agent ingest` cost (see *Costs and trade-offs*), so
budget by wave size. Cross-linking is deliberately not a phase: a bulk import
arrives as N disconnected nodes, and `verify` names the follow-ups
(`claim-overlap --backlog`, `candidates concepts --bridges`) rather than spending
a judge call per pair inline.

There is deliberately **no `--category`**. Category is chosen per paper by
promote's neighbour-vote classifier, which is the better answer for a mixed
library anyway — a reference manager's collections rarely map onto wiki
categories, and one global value would flatten the corpus into a single bin.

Full procedure, the reason→verdict table and failure modes:
[`prompts/import-reference-manager.md`](./prompts/import-reference-manager.md).

---

## Exporting the corpus as a bibliography

`researchwiki export` is the inverse of the above, and the answer to "can I get my
data out". One phase, zero tokens, no network, and byte-identical across runs — so
a `.bib` can live in version control and diff meaningfully.

```bash
researchwiki export --format bibtex --category cgt > cgt.bib   # summary to stderr
researchwiki export --format ris --out library.ris             # for Zotero
researchwiki export --json                                     # the report only
```

### The citekey is the page stem, and that is the whole argument

`\cite{bae-2014-cas-offinder-a-fast-and-versatile}` is long. It is also the only
key that cannot change under you. A short key (`bae2014`) has to disambiguate
collisions with a letter suffix — and on this corpus 47 pages collide across 19
such keys — but that suffix is recomputed every run, so ingesting one more 2026
Wang paper renumbers keys already sitting in a manuscript. A stem never changes,
by the stability rule in CLAUDE.md § *Disambiguation & updates*. There is no
`--citekey` flag, because shipping an unsafe alternative alongside the safe one
just invites the failure.

### Names are not parsed for BibTeX or RIS

Both formats understand `First von Last` themselves, so the transformation is
"replace the separator with ` and `" and nothing else. That is not laziness — it
is what makes 58 nobiliary-particle names (`A. van der Graaf`) and 76 four-token
names impossible to corrupt, since no boundary is ever guessed. Only CSL-JSON
wants structured `family`/`given`, and there anything ambiguous becomes a CSL
`literal`, which is the format's own construct for a name with no given/family
structure.

### Gaps are downgraded and reported, never invented

| Corpus reality | What the export does |
|---|---|
| 53 papers with no `venue:` | `@misc`, not `@article` with an empty `journal`. An `@article` missing `journal` makes `bibtex` merely *warn* — surfacing weeks later in a LaTeX log rather than in the bibliography and the report. |
| 3 furniture venues (`Journal of LaTeX Class Files`, `preprint`) | suppressed. The one place this command could print a falsehood. |
| 10 entries with no DOI | emitted anyway; no format requires one. `no_doi_reason` becomes a `note`, because it explains a citation gap a reader of the `.bib` wants. |
| 8 DOIs recorded as `https://doi.org/…` | normalized through the importer's own `clean_doi`, so they are neither emitted as URLs nor dropped by a stricter validator. |
| `document_id: "Nature 654:324-326"` | passed to `note` verbatim. It *contains* a volume and page range, but the field is free text and FDA guidance numbers use it differently, so mining it is judgement. |
| 248 of 421 titles containing `CRISPR-Cas9`, `DNA`, `Cas9` | brace-protected per word, or `plain.bst` emits `Crispr-cas9`. |

Synthesis, idea and concept pages are **excluded, with no flag to include them.**
They have no DOI, venue or year of record, so an entry for one would assert a
publication that does not exist once pasted into a manuscript — a
citation-integrity problem rather than a formatting one. Sharing analysis is
[`prompts/share-page.md`](./prompts/share-page.md), which produces a document for
a human reader instead.

### It round-trips

The strongest available check, and it is cheap because the importer already
exists: 421 records, all three formats, **zero mismatches** on title, authors,
DOI, venue or year after parsing back through `refimport`. `import preflight` on
the emitted `.bib` reports the same counts the export did.

One documented loss: the 149 `@misc` entries come back as `preprint` (96, via
`eprint`) and `other` (53, the venue-less papers) rather than as `article`. That
information genuinely is not in the corpus, so the asymmetry is asserted by a test
rather than hidden.

Output is UTF-8 with no LaTeX macro conversion — 86 author fields carry non-ASCII,
and a macro table would have to be perfect or it corrupts names. A pdfLaTeX-only
pipeline with bare `bibtex` needs `\usepackage[utf8]{inputenc}`, or biber.

Full procedure and how to act on the report:
[`prompts/export-bibliography.md`](./prompts/export-bibliography.md).

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

## Bottom-up discovery — when the wiki proposes the page

Everything above is top-down: you ask, the wiki answers. This is the inverse —
the corpus surfacing a cluster nobody went looking for. Four tiers propose work
for a human to accept or decline; none edits authored wiki prose automatically.

| Tier | Command | Cost | What it ranks |
|---|---|---|---|
| Concept hubs | `candidates concepts [--bridges]` | local, sub-second | recurring terms in ≥3 papers' contribution claims with no hub yet |
| Claim pairs | `candidates pairs [--cross-category]` | local, sub-second | cross-paper claim pairs *below* the auto-link threshold |
| Paper clusters | `candidates synthesis [--judge] [--write-proposals]` | local preview; configured-model calls only with `--judge` | dense clusters no synthesis page covers |
| Typed relations | `claim-graph [--tensions]` | local (edges already judged) | how papers relate, not merely that they overlap |

`status` auto-surfaces the first two once they cross a threshold (bridge terms;
15 unreviewed cross-category pairs, 14-day decay). The third is noisier and is
run deliberately. Each pair row prints the judged exact-claim path,
`claim-overlap --pair A#slug B#slug`; decline one permanently with
`candidates pairs --decline A B --reason "…"`.

### Why the thresholds are what they are

These numbers are the reason the tiers are shaped this way, and every one of them
cost a measurement. Two provenance notes before trusting them:

- **The 2026-08-19 figures come from a different corpus** — 117 papers / 3,027
  claims, on another machine. The specific papers they cite
  (`van-iterson-2017`, `sarthi-2024`, `rose-1998`, `parks-2018`) and the
  `mixture-model` hub they were measured on **are not in this wiki**, so those
  cases are not reproducible here. The *mechanisms* transfer; the absolute
  numbers describe that corpus, not this one.
- **The 2026-08-20 figures are this corpus** — 419 papers, 12,504 claims, 10,527
  of them in contribution sections.

**Literal member search loses exactly the bridges it exists to find.**
`find_members("mixture model")` returned 4 spokes spanning 2 categories; the
finished hub had 7 spanning 4. All three misses were vocabulary, not absence —
"Gaussian Mixture Models (GMMs)", "three-component *normal* mixture",
"normal-mixture mean estimation". This is systematic rather than unlucky: a term
is a *bridge* precisely when fields name the same thing differently, so literal
matching fails hardest on the highest-value candidates, and fails silently. Hence
the semantic recall tier in `concepts`, and its alias suggestions.

**But cosine cannot decide membership, only propose it.** Ranking every
contribution claim against the embedded term put all 7 true members above the
first false positive — by a margin of **0.003**. No threshold survives that
across terms. Candidate volume at a 0.70 floor says the same thing from the other
side: 5 papers for `off-target activity`, 17 for `mixture model`, 31 for
`ATAC-seq`. Auto-adding at any usable floor turns a 7-spoke entry note into a
category listing. **The semantic pass proposes; the human decides.** That rule is
why membership still flows through the lexical path, via `--aliases`.

**Lowering the cosine threshold is not how you find more.** Measured across
thresholds, the share of *all possible* paper pairs that qualify:

| threshold | paper pairs | % of all possible |
|---|---|---|
| 0.83 (auto-link) | 99 | 1.5% |
| 0.78 | 989 | 15% |
| 0.75 | 2,487 | 37% |
| 0.72 | 4,372 | 64% |
| 0.70 | 5,415 | **80%** |

At any floor low enough to be interesting, most of the corpus is "related to"
most of the corpus. The decisive test: the single most valuable relation on that
hub — two papers disagreeing about whether a mixture can serve as an empirical
null — peaks at cosine **0.743**, so reaching it by threshold means accepting
~2,400 paper pairs to find it.

**Rare-term overlap is the signal cosine is missing.** Ranking the same band by
IDF-weighted shared-term mass instead put that relation at **#210 of 54,792 —
top 0.4%**, with genuinely related pairs above it. A 384-dim embedding compresses
away exactly the rare vocabulary that marks two claims as being about the same
specific thing: two claims sit at 0.73 because both are methods prose, while two
claims sharing "empirical null" and "null distribution" are about one topic.
**Cosine measures register, rare-term overlap measures subject** — the same
hybrid `search` already trusts via RRF, applied to claim pairs instead of
queries. This is what `candidates pairs` ranks by, inside a 0.72–0.83 band whose
ceiling is the auto-link threshold, so the tier never re-surfaces a judged pair.

**Semantic proposals at ingest time would return nothing.** Not on cost — it is
zero tokens and the encoder is already loaded during grading — but because the
pass reads the claim-embedding cache and the cache's only writer is
`claim-overlap`, which is off by default at ingest. For a newly promoted paper,
coverage was **0 of 25 claims**: the pass would silently return nothing for
exactly the paper it was asked about. Hence proposals happen at scaffold time and
in `check-coverage`, where a reviewer is present, rather than in an automatic
hook whose output goes to a batch worker's log file.

**Contradiction density is not the gate it looks like.** The cross-paper judge's
pool at its 0.85 floor is 1,106 claim pairs across 643 paper pairs, and per paper
that is median 4 / p90 17 / max 44 — so only 19 of 305 papers exceed
`alert_after_ingest`'s `max_pairs=20`, and **≥85% of that pool has already been
judged** across the corpus's ingests, for exactly one disagreement (~1-in-900).
More judging at 0.85 buys almost nothing.

What that judge is *for* matters more than its volume: it keeps only
`disagree_numeric` and `disagree_direction`, routing anything with a different
cohort, dataset or run to `different_topic`. That finds **errors** — one of the
two papers must be wrong, and the one edge it found is real (232 vs 47 phased
diploid assemblies for the same assembly). It does not find **arguments**. Two
papers taking incompatible methodological positions score `different_topic`,
correctly by its own definition. A tier that proposes pages because there is an
*argument* therefore needs its own verdict vocabulary, evaluated in a lower
cosine band — not more calls to this one.

### Decided not to build: a tension-hunting tier

There is no fifth tier that proposes a page because two papers *argue*, and this
is a decision rather than a gap.

The contradiction judge finds errors at roughly 1 in 900 (above). Softening the
target to "these two papers pull in opposite directions" raises the rate to about
1 in 15 — measured by hand-classifying the top 60 cross-category pairs, zero
tokens. But the four survivors were then read and rejected: in none of them did
one paper say the other was wrong. They leaned different ways and both were true.
Two of the four dissolved entirely once the scope was checked — one paper supplied
exactly the evidence the other had called for, and another compared different
methods for a different purpose. Three of the four near-misses turned out to be
papers *agreeing*, with their critical remarks aimed at some third method neither
of them used.

Genuine disagreement between papers is simply rare. A tier built to find it would
spend a judge call per pair to surface mostly shared vocabulary and polite
non-overlap, and the reviewer would carry the cost of telling those apart. The
existing tiers propose pages from what the corpus *has* — recurring terms and
shared rare vocabulary — which is abundant. That is the right thing to lean on.

If this is ever revisited, the two leads worth keeping are that critique lives in
`limitations` sections while the thing critiqued lives in contribution sections
(no tier pairs those two directions), and that the relation being sought is not
`contradicts` at all. Neither changes the rarity finding, which is the reason not
to.

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
  wei-2026-...                                             1 file(s), 32m old
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
│   ├── claim_embeddings.py #   Cached bi-encoder embeddings for claims. Three readers,
│   │                       #     two writers: `get_claim_embeddings` rewrites the cache
│   │                       #     to its own row set (so a narrow caller evicts the rest),
│   │                       #     `warm_claim_embeddings` persists the union instead, and
│   │                       #     `load_cached_*` never loads the model at all.
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
│       ├── extract.py      #     PDF → sections + full text
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
├── tasks/                  # CLI subcommands, auto-discovered from module names.
│                           #   Not every module here is a command: `claim_discover.py`
│                           #   and `pair_dismissals.py` are libraries behind
│                           #   `candidates pairs`. A command is a module exposing
│                           #   `main()`; the leading `_` convention is the hint and
│                           #   `__main__._is_entry_point` is the enforced invariant.
│   ├── ingest.py           #   Digest-only path (manual page authoring)
│   ├── agent.py            #   Full agent path (auto-authoring)
│   ├── grade.py            #   Per-paper fidelity + salience report
│   ├── _grade_synthesis.py #   Synthesis/idea fidelity (misattribution check)
│   ├── lint/               #   One module per check family (link, yaml, staleness,
│   │                       #     claim_anchors, concept_contract, db_checks,
│   │                       #     cross_paper — the one LLM-costing check — …);
│   │                       #     `__init__` is the dispatcher and decides nothing,
│   │                       #     `report.py` owns the --json contract + prose report
│   ├── visualize.py        #   Thin CLI over visualize.py → output/graph.html
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
├── refimport/              # Import a reference-manager library (Zotero/Paperpile/
│                           #   Mendeley/ReadCube) from its own BibTeX/RIS/CSL-JSON
│                           #   export. Sibling to migrate/: that one imports pages,
│                           #   this one imports PDFs + metadata and authors nothing.
│   ├── parse.py            #   Tolerant BibTeX/RIS/CSL-JSON reader → ExportItem
│   ├── pair.py             #   Record → PDF, three rungs (declared/doi/title);
│   │                       #     one extraction pass over the tree, ever
│   ├── triage.py           #   The gates → ready/review/skip + reason strings
│   ├── apply.py            #   Plan a wave, copy into inbox/, hand to _ingest_batch
│   └── manifest.py         #   Run dir + manifest.json (frozen pairing/verdicts/argv)
├── refexport.py            # The inverse of refimport/: corpus → BibTeX/RIS/CSL-JSON.
│                           #   One module, not a package — escaping, one type table,
│                           #   one wiki walk, three renderers. Zero tokens.
├── okfexport.py            # Corpus → an Open Knowledge Format bundle (OKF v0.2),
│                           #   behind the same `export` command. Separate module
│                           #   from refexport because the scope differs: a
│                           #   bibliography carries only published documents, an
│                           #   OKF bundle carries every page type (its unit is a
│                           #   "concept", abstract ideas included). Also emits a
│                           #   tree, not a stream, so `--format okf` needs `--out`.
├── visualize.py            # Corpus → a self-contained interactive graph (the
│                           #   renderer is templates/graph.html). Draws wikilinks
│                           #   AND typed claim edges; contradictions drawn loud.
├── names.py                # Author-name parsing, shared by stems (which surname
│                           #   goes in a stem) and refexport (family/given for CSL)
├── templates/              # Shipped assets read via importlib.resources
│   └── graph.html          #   The visualize renderer: no CDN, no build step
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

researchwiki visualize --open                         # 10. corpus → output/graph.html (self-contained)
researchwiki export --format bibtex > refs.bib        # 11. corpus → bibliography (published pages only)
researchwiki export --format okf --out output/okf     #     corpus → OKF bundle (every page type; needs a dir)
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
| `import <preflight\|inspect\|apply\|verify>` | Bulk-import a reference-manager library from its BibTeX/RIS/CSL-JSON export, which supplies each paper's DOI/title/authors/year instead of rediscovering them. Only `apply` spends tokens or writes pages; `<pdf-root>` is optional, and a metadata-only run still returns a fetch list of DOIs. Stage it with `--limit N`. See `prompts/import-reference-manager.md`. |
| `export [--format bibtex\|ris\|csl-json]` | The inverse: emit the corpus as a bibliography for a reference manager or a manuscript. Zero tokens, no network, byte-identical across runs. Citekey is the page stem. Only page types describing someone else's publication are emitted — a synthesis page would assert a publication that does not exist. `--json` gives the report, which doubles as a page-defect to-do list. See `prompts/export-bibliography.md`. |
| `synthesize --title [...] [--papers]` | Scaffold `wiki/synthesis/{slug}.md`. Idea/reference pages are manual. |
| `candidates <concepts\|synthesis\|pairs>` | Surface opportunity signals: un-scaffolded concept hubs (concepts), uncovered paper clusters warranting a synthesis page (synthesis), or cross-paper claim pairs sitting below the auto-link threshold (pairs). Concepts/pairs are local; synthesis is a local preview unless `--judge` opts into configured-model scope checks, and writes artifacts only with `--write-proposals`. Each pair prints an exact `claim-overlap --pair` review command. `--decline A B --reason` suppresses a pair permanently. See *Bottom-up discovery* above. |
| `reindex [--no-semantic]` | Rebuild Tantivy + semantic index from `wiki/`. |
| `search "<query>" [--mode ...]` or `--like <stem>` | Hybrid retrieval (RRF over BM25 + semantic) by default. |
| `status` | Dashboard: counts, density, orphans, backlog, index health, pending proposals, 7-day cost. |
| `lint [--fix] [--cross-paper]` | Orphans, broken/missing wikilinks (auto-fixable), stale syntheses, missing keywords/DOIs, year drift, stale proposals. All local except `--cross-paper`, which opts into the LLM contradiction judge over high-cosine claim pairs; `--cross-paper-max-pairs 0` sizes that pool for zero calls, and every verdict is recorded so a repeat run only judges what the last one missed. |
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
| `eval` | `classifier`: leave-one-out accuracy of the category auto-suggester (free). `triggers`: whether CLAUDE.md's prompt pointers fire (costs tokens). `eval-classifier` is a deprecated alias. |
| `benchmark-fixture <stem> [--repeat N] [--llm]` | Score page authoring against a hand-curated `benchmark-fixtures/` fixture. `--repeat` keeps drafts in memory; for a single authored page use `agent ingest … --force-sandbox`, never a bare `agent ingest` (it would promote a fixture paper into your corpus). |
| `claim-overlap <stem> [--sim N] [--top N] [--dry-run] [--json]` | Proactively cross-link a newly-ingested paper: finds existing papers with near-paraphrase claims, LLM-judges each as a real relationship vs coincidence, and auto-adds reciprocal Related-Papers `[[wikilinks]]` for confirmed matches. `--pair A#slug B#slug` instead judges exactly one bottom-up discovery pair, bypassing threshold/top-k retrieval. Run after `db rebuild`. |
| `db papers [--year/--category/--page-type/--no-doi/--venue/--author/--status] [--count] [--json]` | Structured lookups over the frontmatter mirror — counts/filters ("cgt papers from 2024", "papers missing a DOI") without re-reading markdown. `db query "SELECT…"` for ad-hoc read-only SQL. |
| `remove <stem> [--apply] [--keep-pdf]` | Retract a paper: page, PDF, caches, back-link bullets, `index.md` entry, concept spokes and DB rows. Dry run by default, runs inside the mutation journal. Reports but never edits authored `[[stem#slug]]` citations on synthesis/idea/concept pages — that list is the to-do queue. See `prompts/remove-paper.md`. |
| `visualize [--open] [--json]` | Self-contained interactive graph of the corpus to `output/graph.html` — `[[wikilinks]]` plus typed claim edges, `contradicts` drawn loud. Zero tokens, no network. Shows structure, not claims. |
| `figures <stem> [--figure N]` | List a paper's figure captions, or render one page to `.figures-cache/` to read when the evidence is in the figure rather than the prose. Captions are free; render one page at a time — a PNG costs context in proportion to its pixel area. |
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
| Want the library in Zotero/Paperpile | `researchwiki export --format ris --out library.ris` |
| Writing a manuscript from wiki papers | `researchwiki export --format bibtex --out refs.bib`, then `\cite{<page-stem>}` |
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
| Want the wiki to propose a page instead of answering one | `researchwiki candidates concepts --bridges`, then `candidates pairs --cross-category` (see *Bottom-up discovery*) |
| `status` printed a bridge-term or claim-pair count | That line is the trigger — `researchwiki candidates <concepts --bridges\|pairs --cross-category>` |
| Want to act on a proposed claim pair | Copy that row's `researchwiki claim-overlap --pair A#slug B#slug` command (the exact judged path); `candidates pairs --decline A B --reason "…"` to reject it for good |
| Want to see which papers disagree | `researchwiki claim-graph --tensions`; `researchwiki visualize --open` to see whether tensions cluster on one paper |
| Want to retract a paper | `researchwiki remove <stem>` (dry run), then `--apply` (see `prompts/remove-paper.md`) |
| The evidence is in a figure, not the prose | `researchwiki figures <stem>` for captions; `--figure N` to render just that page |

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
