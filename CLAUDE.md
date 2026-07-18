# LLM Wiki — Research

Personal knowledge base of research papers, inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/1dd0294ef9567971c1e4348a90d69285), simplified to two tiers:

```
Original PDF (immutable) → wiki/{category}/*.md (the single LLM-authored page)
```

**Language**: wiki content is English; conversation English or Korean.

---

## THE FOUR RULES (do not violate)

1. **No web search for prose content.** Never `WebSearch`/`WebFetch` to pull summaries or facts into the wiki. Every claim grounds in a PDF we have.

   **Exception — structured-metadata APIs.** Whitelist for *structural metadata only*, all mediated through `researchwiki` CLI wrappers (never raw `WebFetch`/`WebSearch`). Responses cache under `.s2-cache/` / `.crossref-cache/` / `.web-cache/`.

   | API | Allowed fields | Used by |
   |---|---|---|
   | Semantic Scholar Graph | title, authors, year, venue, externalIds, references, citations, `/recommendations`, `abstract` (verbatim), `tldr` (draft/cross-check only) | `ingest`, `audit` |
   | Crossref | title, authors, year, container-title, ISSN, `reference` list, `type` | `ingest` (S2 fallback) |
   | PubMed E-utilities | PMID, `pubtype`, `pubdate`, retraction linkage | `retraction-check` |
   | bioRxiv/medRxiv | `server`, `version`, `date`, `category`, `type`, `published` DOI. `abstract` **not re-exposed** | `preprint-check` |
   | ORCID Public API | names, latest employment. `biography`/`keywords`/`email` **not re-exposed** | `orcid-lookup` |

   **Prose fields:** S2 `abstract` verbatim OK; S2 `tldr` draft/cross-check only (record `tldr_source: semantic-scholar`). Everything else prose-y (venue blurbs, retraction reasons, ORCID bio) banned.

   **Provenance.** Whitelist claims carry source + fetch date (YAML or inline `(PubMed, fetched 2026-05-07)`).

   **Extraction, not ingestion.** Fields inform *decisions*, never paraphrased into prose.

   **Source credibility (non-whitelist).** Peer-reviewed journals + author-hosted PDFs OK; blogs, Wikipedia, aggregators, social, press releases — do not paraphrase. Preprints only via `inbox/`.

   **User-provided URL exception.** User explicitly hands you a URL/repo → authorized to `WebFetch` (or `gh`) *that exact URL* for the **conversational answer**. Bounds: only the given URL (no transitive links); wiki prose still needs the PDF (Rule 3); cite inline (`per the README at github.com/…`), no `[[wikilink]]`; GitHub via `gh` preferred; doesn't apply to URLs Claude generated.

2. **Answer from the wiki first.** `wiki/` is the only source of truth.
3. **Wiki insufficient → re-read the PDF.** `papers/{stem}.pdf`; use `pypdfium2` or `Read`. Supplementary at `papers/{stem}.supp/{filename}`.
4. **No paper on the topic → say so.** *"I don't have a paper on this — please give me the PDF."* Don't improvise.

These apply to every response, including synthesis pages.

**Corollary — don't let model priors leak.** Never infer affiliations, funding, dates, DOIs, or numerical results beyond what the PDF states. When in doubt, omit.

**Corollary — cross-links must be source-supported.** `[[wikilink]]` only when the source explicitly cites/builds on/contrasts the other paper. Topical adjacency alone is not enough.

**Corollary — flag adjacent gaps.** After a substantive cross-paper answer with partial coverage, close with a concrete "What's missing" line (specific paper types, not vague topics) and ask user to drop PDFs in `inbox/`. Skip when coverage is comprehensive.

**Corollary — ground through the claims DB.** Wiki pages are truth (Rule 2); the **claims DB is the grounding layer**. Before authoring any page with factual claims (synthesis, idea, filed query), `researchwiki claims "<topic>"` (or `--by-stem <stem>`) is the **first stop** — pre-graded units anchored to paper + section, scored against the PDF. Cite at the claim level with `[[stem#claim_slug]]` anchors (durable, content-addressed — a slug like `kc-9f3a2b1c` survives `db rebuild` and its identity is verified by `check-grounding`); the `claims` CLI prints the exact citation form to copy. Fall back to bare `[[stem]]` when the paragraph refers to the paper as a whole. On prose-heavy pages, use **academic footnotes** — one per source paper, `[^id]: [[category/stem]]` at bottom (exact form + grading gotcha in [`prompts/synthesis-page-author.md`](./prompts/synthesis-page-author.md)). Paper-page *Related Papers* stay inline `[[wikilink]]`. **In markdown tables**, footnotes don't render — use bare `[[stem]]` (or `Short-name [[stem]]`), never `[[stem\|alias]]`. **Never write `claim_id:NNN`** into a page (they're `AUTOINCREMENT` row keys, reassigned on `db rebuild`). Verify synthesis + idea pages with **both** gates, both must exit 0:
- `researchwiki check-grounding <page>` — structural (every claim carries a citation)
- `researchwiki grade synthesis <page>` — fidelity (each cited claim holds in the paper it cites)

Then `researchwiki check-coverage <page>` — recall of wiki papers ranking high on the page's `topic_seed` that the page doesn't cite. Advisory, but every unreferenced hit should be a deliberate decision.

---

## Editing this file

CLAUDE.md loads every turn — keep it lean. Trigger-gated procedures live in `prompts/{slug}.md`; leave a one-line pointer here with the trigger condition. Don't inline workflows that won't run on the current turn.

---

## Repository Structure

```
Research/
├── CLAUDE.md, README.md, pyproject.toml
├── inbox/                  # Raw drops awaiting processing
├── papers/                 # Canonically-named PDFs only (cp, never symlink)
├── researchwiki/           # Framework package (CLI, tasks/, providers/)
├── .ingest/                # Per-paper digests (delete after use)
├── .s2-cache/              # Cached Semantic Scholar responses
└── wiki/                   # Wiki pages (English)
    ├── {category}/
    ├── synthesis/          # Retrospective field maps
    ├── ideas/              # Forward-looking design proposals
    ├── concepts/           # Single-term hub notes (bridge nodes)
    ├── references/         # Regulatory guidance, protocols, whitepapers, books
    └── index.md, log.md, pdfs-failed-parsing.md
```

**Invariant**: `papers/` holds only canonically-named PDFs. Raw drops live in `inbox/`. A non-empty `inbox/` is the backlog.

---

## File Naming Convention

Stem shared by both tiers: `{first-author-lastname}-{year}-{first-5-title-words}.{ext}`

**Source of truth**: derive from the PDF's first page text, not `reader.metadata`.

**Author rules:**
- First author's surname as printed on p.1.
- Hyphenated given names dropped (`Wen-Wei Liao` → `liao`); hyphenated surnames kept (`García-López` → `garcia-lopez`).
- Diacritics → ASCII (`García` → `garcia`).
- Consortium papers: hyphenated consortium slug (`1000 Genomes Project` → `1000-genomes-project`).

**Issuer rules (non-paper reference docs):**
- Government bodies collapsed to parent: FDA → `fda`, EMA → `ema`, ICH → `ich`, `fda-cber` → `fda`.
- Companies: IDT → `idt`, `10x Genomics` → `10x-genomics`.
- Trial protocols: sponsor slug or registered trial ID.
- Books: normal Author rules.

**Year**: 4-digit publication year from the paper (header/footer/first-page citation). Preprints use the version year on the document.

**Title rules:**
- First 5 words; **all** words count (including stop words).
- Trailing stop word? Extend until stem ends on a content word. Stop words: `a an the of for with and or in on at to from by as across over all that this these those`.
- Colons skipped — keep counting into the subtitle.
- Hyphenated terms = one word (`Cas-OFFinder` is 1 word).
- Numbers kept as-is (`Evo 2` → `evo-2`).
- Strip all other punctuation; lowercase; spaces → hyphens; diacritics → ASCII.

**Examples:**

| Input | Stem |
|---|---|
| Bae (2014), "Cas-OFFinder: a fast and versatile algorithm…" | `bae-2014-cas-offinder-a-fast-and-versatile` |
| Liao (2023), "A draft human pangenome reference" | `liao-2023-a-draft-human-pangenome-reference` |
| Brixi (2026), "Genome modelling and design across all domains of life with Evo 2" | `brixi-2026-genome-modelling-and-design-across-all-domains` |

**Disambiguation & updates:**
- Same author/year → append BibTeX letter (second Smith 2024 → `smith-2024b-…`); first keeps bare year; never re-letter.
- Preprint → journal: keep stem. Update YAML (`title`, `doi`, venue); filename and `[[wikilinks]]` stay.
- First-author change between versions: keep earliest stem; note in YAML `authors`.

---

## Categories

Categories are **local and derived from your papers, never predefined.** Valid iff `wiki/<category>/` exists.

- **Scaffold (committed)**: `synthesis`, `ideas`, `concepts`, `references`, `other` — via `.gitkeep`. First four are **page-type dirs**, not content categories; `other` is the classifier's abstention bucket.
- **Growth is explicit**: a new category exists only once `wiki/<category>/` is created. `--category X` rejected if dir doesn't exist; classifier abstains to `other`. Typos can't spawn categories.
- **Cold start**: `researchwiki bootstrap-categories` reads `inbox/` and proposes+creates dirs. See Operations → Initialization.
- **At ingest**: classifier picks existing category or abstains. `wiki/other/` ≥10 papers → `status` flags it, `suggest-splits` proposes splits.

Tip: classify by **method**, not topic. If removing the domain guts the contribution, group with the domain-grounded papers; else group with methods papers.

---

## Page Types

### 1. Paper page — `wiki/{category}/{stem}.md`

One per paper in `papers/`. Produced by `researchwiki ingest`.

### 2. Synthesis page — `wiki/synthesis/{slug}.md`

Cross-paper analytical page — trajectory, comparison, recurring concept.

```yaml
---
title: "CRISPR off-target strategies"
type: synthesis
category: [cgt]                        # CONTENT category (not "synthesis") — typically the dominant category of the cited papers
generated_at: 2026-05-28
topic_seed: "CRISPR off-target prediction"
tags: [crispr, off-target]
---
```

**No `referenced_papers:` field.** Synthesis (and idea) pages cite via the body — inline `[[wikilink]]`s and `## References` footnotes (`[^id]: [[category/stem]]`), which are the single source of truth every gate/tool reads. (Concept pages *do* keep `referenced_papers:` — there it's the functional spoke registry.)

### 3. Reference document — `wiki/references/{stem}.md`

Non-peer-reviewed: guidance, protocols, whitepapers, books.

```yaml
---
title: "..."
type: guidance                        # guidance | protocol | whitepaper | book
category: [references]
issuer: FDA                           # required for guidance/whitepaper/protocol
issuance_date: "April 2026"           # optional
status: draft                         # optional: draft | final | active | superseded
document_id: ""                       # optional: docket #, guidance #, NCT #, ISBN
authors: ""                           # required for `book`
pdf_path: /.../papers/fda-2026-....pdf
pdf_filename: fda-2026-....pdf
source_collection: external
source_url: ""                        # optional: canonical URL for blog posts / online whitepapers
author_model: "claude-opus-4-7"       # optional: LLM that authored the page (manual whitepaper path — mirrors the field agent ingest writes on paper pages)
keywords: []                          # required: 6–10 short phrases describing the doc's content, parallel to the keywords field on paper pages. Enforced by `researchwiki lint`
ingested_at: "2026-04-15T14:30:00"    # ISO 8601 local. Manual path: stamp with `date +%Y-%m-%dT%H:%M:%S`
tags: []
---
```

**Manual workflow**: reference docs skip `researchwiki ingest`. Move PDF into `papers/`, write the page directly, stamp `ingested_at`, update `index.md`, append to `log.md`.

**Cross-links**: regulatory docs prescribe method *categories*, not individual papers — `[[wikilink]]` describes methodological alignment ("aligns with §N.X of"), not citation.

### 4. Idea page — `wiki/ideas/{slug}.md`

Forward-looking design proposals grounded in wiki content. Distinct from synthesis (which retrospectively maps a field).

```yaml
---
title: "..."
type: idea
category: [single-cell]               # CONTENT category — dominant field of the design
status: open                          # open | scoping | validated | superseded | abandoned
verdict: incremental                  # strong | incremental | weak — mirrors Verdict section
companion_synthesis: ["[[…]]"]        # optional; quote each wikilink so Obsidian renders it (unquoted [[..]] parses as a nested list → "?")
generated_at: 2026-06-08
topic_seed: "..."
tags: [idea, ...]
---
```

**Required H2 order — Verdict → Background → Opportunities → Plans → Caveats.**

- **Verdict**: one-line strength label (`strong`/`incremental`/`weak`) + one-para tl;dr citing the load-bearing anchors. Written last, placed first. Strict-grounded (no `*(model prior)*`).
- **Background**: strictly grounded, cited via `[^id]` footnotes.
- **Opportunities**: design proposal. Model priors allowed — mark each with `*(model prior)*`; numbers/benchmark results/named-entity attributions still need a wiki citation.
- **Plans**: staged implementation, one load-bearing assumption per phase. Model priors allowed with same rules.
- **Caveats**: strictly grounded — real invalidators, cited like Background. Includes "what would update this page" (papers whose ingestion would shift `status:`).

**Sourcing rule** (extends Rule 1): idea pages are the one place model priors are allowed, but only in Opportunities/Plans, marked `*(model prior)*`. Marker doubles as a linter signal: `check-grounding` reports marker-tagged units as `model_prior` (warning) rather than `ungrounded` (failure). `--strict` treats the marker as ungrounded. Anywhere else the marker has no effect. Cross-link rules unchanged.

**Status lifecycle**: `open` → `scoping` → `validated` | `superseded` | `abandoned`.

**Manual workflow** — see [`prompts/idea-page-author.md`](./prompts/idea-page-author.md) (section-by-section guidance, verdict-label criteria, verify steps). Skip `researchwiki ingest`; move drafts into `wiki/ideas/`, update `index.md`, append to `log.md`.

After authoring, run **both** gates (both must exit 0):
1. `researchwiki check-grounding wiki/ideas/<slug>.md` — structural
2. `researchwiki grade synthesis wiki/ideas/<slug>.md` — fidelity

Then `researchwiki check-coverage wiki/ideas/<slug>.md` — advisory recall.

### 5. Concept page — `wiki/concepts/{slug}.md`

Single-term **hub note** — a mini-synthesis around one recurring concept (surfaced by `researchwiki candidates concepts [--bridges]` at ≥3 papers). Ties every wiki paper that instantiates the term into one bridge node; most valuable when `concept_span ≥ 2` (spans categories the citation graph and semantic-KNN don't bridge). Strictly grounded (no model priors). YAML: `type: concept`, `category:` (content), `referenced_papers:` (the spokes), `concept_span:`, `generated_at:`, `topic_seed:`. Required H2s: **Definition** → **How it appears across the corpus**; optional **Cross-domain connections**, **What would update this page**. Each member paper gets a reciprocal `[[concepts/<slug>]]` back-link. Verify with **both** gates (`check-grounding` + `grade synthesis`) + advisory `check-coverage`.

**Manual workflow** — see [`prompts/concept-page-author.md`](./prompts/concept-page-author.md). Scaffold-first: `researchwiki concepts <term>` builds a grounded stub; fill Definition + spokes, then verify. Update `index.md` (`## concepts`), append to `log.md`.

### 6. Log / meta pages

- `index.md` — page catalogue, by category.
- `log.md` — chronological record.
- `pdfs-failed-parsing.md` — PDFs needing replacement.
- `wiki/synthesis/suggested-additions.md` — gap map from `researchwiki audit`.

### Page-type discipline

- Every page declares `type:` — `paper`, `synthesis`, `guidance`, `protocol`, `whitepaper`, `book`, `idea`, `concept`.
- Synthesis, idea + concept pages must cite their sources and pass **both** `check-grounding` + `grade synthesis`. Synthesis/idea cite via the body (inline `[[wikilink]]`s + `## References` footnotes); concept pages use the `referenced_papers:` spoke list.
- **Exemption**: the `## What would update this page` H2 is skipped by `check-grounding` (name-narrow, exact heading, case-insensitive). Other "next steps"-style headings aren't exempt.
- Idea pages must follow Verdict → Background → Opportunities → Plans → Caveats.
- Concept pages must follow Definition → How it appears across the corpus (Cross-domain connections optional).

---

## Operations

### Initialization — cold start

If every `wiki/{category}/` is empty AND user signals they're new (*"set this up"*, *"initialize"*), read [`prompts/init.md`](./prompts/init.md). Proactively offer init; alternatively suggest `researchwiki init` (interactive wizard).

### Model providers — routing and mixed mode

Per-phase provider comes from `config/models.yaml`. **`RW_LLM_PROVIDER`** is a global override that forces every phase to one provider and **silently defeats per-role mixing** — including via `.env`. If mixing isn't taking effect, check `.env` for a stray `RW_LLM_PROVIDER`. The OpenAI-compatible endpoint (`openai-compatible`/`lmstudio`/`openai` providers) is declared **per config** via a top-level `base_url:` key — so switching backends rides on `RW_MODELS_CONFIG` alone, no paired env edit. Precedence: `RW_LLM_BASE_URL` env (ad-hoc override, if set) → config `base_url:` → LM Studio localhost default. Note the API **key** is still one global `OPENAI_API_KEY` — switching to a backend that needs a different key still requires an `.env` key change (LM Studio needs none; the Gemini configs share one key).

**`RW_MODELS_CONFIG`** selects which *file* the loader reads (default `config/models.yaml`). Bare filename resolves under `config/`; absolute/path-separated used verbatim. Safe to park in `.env` (no per-role-mixing footgun). `researchwiki status` prints the active path.

**Anthropic-compatible third parties (e.g. z.ai GLM)**: use `provider: anthropic` with `ANTHROPIC_BASE_URL` stopping at the host root (z.ai: `https://api.z.ai/api/anthropic`, **no** trailing `/v1`). Free tiers may 429 the parallel author phase — drop to `-n 1`.

**Gemini free tier (`config/models.gemini.yaml`)**: observed ceiling is ~5 requests/minute on the Flash model, shared per-project (not per-key) — `agent ingest`'s default 4 batch workers × 2 parallel author drafts blows through it. Pass `-w 1` to serialize the batch; add `-n 1` if 429s persist.

### Ingest — add a new paper

**Default path: `researchwiki agent ingest`.** Handles pypdfium2 extraction, DOI detection, S2 lookup, LLM-reconcile, stem derivation, `mv` to `papers/`, page authoring, atomic back-link apply, and `researchwiki evolve`. Tags `ingested-via-agent`; logs `ingest_iterations` to the state DB.

**Step 0** — drop PDF into `inbox/`.

**Step 1** — run agent ingest. **For ≥2 PDFs pass them all to a single `agent ingest` invocation** — it auto-enters crash-safe batch mode (4 workers by default, atomic `.ingest/batch-<ts>/checkpoint.json` per completion, `--resume <batch-dir>` picks up after a crash). Do NOT fan out one background Bash per file: it bypasses the checkpoint, uncaps concurrency, and multiplies `state.db` write contention. Omit `--category` to let the classifier suggest unless every paper obviously shares one.

```bash
researchwiki agent ingest inbox/<raw-filename>.pdf              # single PDF
researchwiki agent ingest inbox/*.pdf                           # ≥2 PDFs — batch, 4 workers, checkpoint
researchwiki agent ingest inbox/*.pdf -w 2                      # cap workers (e.g. rate-limited providers)
researchwiki agent ingest --resume .ingest/batch-<ts>/          # resume after a crash / Ctrl-C
```

**Never write ad-hoc scripts to ingest PDFs, and never fan out one Bash task per file.** Always use `agent ingest` (or digest path for recovery); rely on its built-in batch mode for multi-PDF runs.

**Digest-path fallback — `researchwiki ingest`** — recovery, unextractable PDFs, special page types, custom-voice cases. Workflow + page-contract template in [`prompts/ingest-digest.md`](./prompts/ingest-digest.md).

#### After ingest (both paths)

**Step 2** — `researchwiki db rebuild && researchwiki reindex`. Required after every batch.

**Step 2.5 — Proactive claim cross-linking.** For each newly-ingested stem, run `researchwiki claim-overlap <stem>`. Finds existing papers with near-paraphrase claims, LLM-judges each for real relationship vs vocabulary overlap, and auto-adds reciprocal `[[wikilink]]`s to both pages' Related Papers on confirmed matches (tagged `auto-added; claim-overlap`). Use `--dry-run` to preview. Skip for reference/idea/synthesis pages (no graded paper claims).

**Step 2.6 — Concept-hub attachment.** Agent path auto-runs `concepts.attach_after_ingest` (right after claim-overlap): the new paper joins any existing `wiki/concepts/` hub whose `topic_seed` term appears in a **contribution claim** (key_contributions / results / methodology sections — a body-prose-only mention isn't enough; those log as near-misses). The spoke bullet cites the specific matching claim via `[[stem#slug]]`; `referenced_papers`/`concept_span` refresh on the hub, and a reciprocal `[[concepts/<slug>]]` lands on the paper (tagged `auto-added; concept-link`). No-ops until concept pages exist. Digest path: run nothing here (attachment is agent-only); new bridge concepts instead surface via `researchwiki candidates concepts --bridges` — span-≥2 terms are labeled `concept-ready (bridge)`. Scaffold one with `researchwiki concepts "<term>"` (see [`prompts/concept-page-author.md`](./prompts/concept-page-author.md)); creation stays review-gated (it writes prose → both gates). To backfill slug citations on existing hubs after new claims land, `researchwiki concepts --upgrade-spokes` rewrites bare `[[stem]]` spokes to `[[stem#slug]]` (idempotent).

**Step 3 — Update `index.md`** — `[[category/stem]] — **Short name** (*Venue* year): one-sentence hook.` (Neither ingest path auto-updates `index.md`.)

**Step 4 — Append to `log.md`** (auto-handled).

**Step 5 — Check stale syntheses.** `researchwiki lint --json`; inspect `stale_synthesis`, `stale_by_content`, `p2_entries_with_anchor_hits`. Refresh or leave a one-liner in `log.md`.

**Step 6 — Memory-evolution proposals.** Agent path auto-runs `researchwiki evolve`; digest path doesn't (run manually). Actionable proposals land under `.ingest/{stem}-evolution-proposals/`. **You are the reviewer** — read each proposal + target synthesis, verify patches against the source paper (numbers, framing, superlatives drift), one-paragraph verdict per proposal, ask user permission (one yes/no covers all from a single ingest unless specified). On approval: apply, `rm -rf` the proposal dir, update synthesis `generated_at:` (the citation lands in the body — synthesis/idea have no `referenced_papers:`). Skip when `evolve` returned zero verdicts or paper is a reference doc.

### Recovery — re-ingest after a broken ingest

When `lint --json` flags `missing_doi`/`stem_year_drift`/`unknown-` stem or agent landed bad metadata: re-ingest with overrides (not YAML-patching). Full workflow in [`prompts/recovery.md`](./prompts/recovery.md).

### Recategorize — move a paper to a different category

Follow [`prompts/recategorize.md`](./prompts/recategorize.md). Directory is canonical (`db rebuild` ignores YAML `category:`); procedure repoints inbound links, updates YAML, verifies via `lint`.

### Benchmark — test a model on a fixture

When asked to benchmark/test an LLM by ingesting a `benchmark-fixtures/` paper: **always `agent ingest … --force-sandbox`** (writes to `.agent-output/`, never promotes to `wiki/` or touches `index.md`), select the config under test with an inline `RW_MODELS_CONFIG=` override, judge `.agent-output/<stem>.md` against `benchmark-fixtures/<stem>.yaml`. Full procedure + the mandatory `db rebuild && reindex` cleanup (if you ever promote by mistake) in [`prompts/benchmark-run.md`](./prompts/benchmark-run.md).

### Query — ask a cross-paper question and file the answer back

1. Answer from `wiki/` first. `researchwiki claims "<topic>"` is the **first stop** for factual claims (pre-graded, BM25+semantic-scored). Each hit prints its `[[stem#claim_slug]]` citation form — copy that directly into prose. `claim_id:NNN` is a session-local handle, never a citation token. `researchwiki search` for page-level discovery.

   **Structural/bibliometric questions go to the DB.** Corpus counts/filters — "how many cgt papers from 2024?", "which lack a DOI?", "everything in *Nature*" — via `researchwiki db papers [--year/--category/--page-type/--no-doi/--venue/--author/--status] [--count] [--json]` or `researchwiki db query "SELECT …"` for ad-hoc. Ingest telemetry (model quality/cost, hardest sections, token spend) via `researchwiki insights`.
2. Insufficient (Rule 3): re-read PDFs; update paper pages if worth keeping.
3. No paper covers it (Rule 4): say so.
4. Cite facts with `[[wikilink]]`; mention sections in prose, not `claim_id`s.
5. Non-trivial cross-paper → create a synthesis page. **This is how the wiki compounds.**

| User question shape | Location |
|---|---|
| "What is X?" (aggregated) | `wiki/synthesis/{x-slug}.md` |
| "Approaches to Z?" | `wiki/synthesis/{z-slug}.md` |
| "How does A compare to B?" | `wiki/synthesis/{a-vs-b}.md` |
| "Trajectory of field F?" | `wiki/synthesis/{f-trajectory}.md` |

Use `researchwiki synthesize --title "…" --topic-seed "…" --papers <stems>` to scaffold. With `--topic-seed` + `--papers` set, the stub's *Evidence from the wiki* section pre-populates with `claim_lookup` hits and each paper's claims. `researchwiki claims --by-stem <stem>` dumps one paper's citable surface (`--include-context` adds source-PDF chunks).

After authoring, run both gates (both exit 0): `check-grounding` (structural) + `grade synthesis` (fidelity). Then `check-coverage` (advisory recall).

**Rules for filed answers:**
- Cite only wiki papers; every claim backed by a paper in `papers/`.
- Wikilink every claim.
- YAML: `generated_at: YYYY-MM-DD`, `topic_seed: "4–8-word query"` (no `referenced_papers:` — cite in the body via `[[wikilink]]`s + `## References` footnotes).
- Append to `log.md`: `## [YYYY-MM-DD] query | <question> → <page>`.

Before trusting an existing synthesis, check `lint --json` for `stale_by_content` / `stale_synthesis`.

**Refreshing `wiki/synthesis/suggested-additions.md`**: triggered by `stale_by_audit_count`, ingestion of a listed paper, or ≥3 ingests since last refresh. Workflow in [`prompts/audit-refresh.md`](./prompts/audit-refresh.md).

### Cross-link discovery — manual page writes

After writing or substantially editing a **multi-topic page** (synthesis, whitepaper, broad reference doc — ≥3 distinct named tools/methods/concepts), grep `wiki/` for each named entity and add source-supported cross-links. Agent ingest covers paper pages via `propose_crosslinks`; manual writes have no analogue. Template in [`prompts/cross-link-discovery.md`](./prompts/cross-link-discovery.md).

### Export a wiki page as a shareable

When user asks to share/export a synthesis or idea page, produce a self-contained markdown at `share/<slug>.md` (gitignored). Strip `[[wikilinks]]`, framework-specific YAML, and self-referential phrasing; rewrite footnotes to full academic citations with DOI links. Procedure in [`prompts/export-shareable.md`](./prompts/export-shareable.md).

### Lint — periodic health check

- **`researchwiki status`** — instant, local dashboard: paper count per category, cross-link graph, `inbox/` backlog, recent additions. Run after every ingest session. Auto-surfaces the concept-hub bridge count (see `candidates concepts` below) — if it prints a nonzero `Concept-hub candidates: N bridge term(s) …` line, that's your trigger to scaffold one.
- **`researchwiki audit`** — S2 citation-graph audit. Unsupported wikilinks, missing cross-links, external papers cited by ≥2 wiki papers. Weekly or after batches. `--json` writes `.s2-cache/audit-{date}.json`.
- **`researchwiki lint`** — local consistency checks: orphans, broken wikilinks, stale syntheses (mtime, content, audit-count), missing back-links, YAML schema violations, page-type mismatches, category YAML↔dir drift, P2 entries with anchor hits. `--fix` auto-inserts missing back-links tagged `(auto-added; refine)`.

**Opportunity signals (not defects; user-initiated cadence):**

- **`researchwiki candidates concepts [--bridges] [--json]`** — recurring vocabulary terms mentioned by ≥3 wiki papers with no `wiki/concepts/{slug}.md` yet. Bridge candidates (span ≥2 categories) are the highest-leverage ones — `status` auto-surfaces the bridge count, so run this whenever that line is nonzero. Scaffold with `researchwiki concepts "<term>"` (see [`prompts/concept-page-author.md`](./prompts/concept-page-author.md)).
- **`researchwiki candidates synthesis`** — dense paper clusters (wikilinks + semantic cosine + keyword Jaccard, connected components) not yet covered by any existing synthesis page. Higher noise rate than concepts, so **not** auto-surfaced; run after ≥5 ingests since last, or when the user asks a cross-paper question that lands in an unfamiliar cluster. Output is proposal stubs in `.ingest/synthesis-candidates/{slug}.md`; human picks a topic and runs `researchwiki synthesize`.

**Category-YAML↔dir drift** catches a recategorized paper whose frontmatter wasn't updated. Page-type dirs (`ideas/`/`synthesis/`/`references/`) accept either the page-type name or a valid content category as YAML `category:`.

### Whitelist-API lookups — `retraction-check`, `preprint-check`, `orcid-lookup`

CLI wrappers around PubMed / bioRxiv / ORCID. Usage, YAML-recording rules, and workflow split for each in [`prompts/lookups.md`](./prompts/lookups.md).

### Agent output — prefer `--json`

- `search --json` → `[{stem, category, page_type, title, score, snippet, key}]`; `--see-also` adds `see_also`.
- `lint --json` → `{pages_scanned, orphans, broken_wikilinks, missing_backlinks, page_type_mismatches, category_yaml_drift, stale_synthesis, stale_by_content, stale_by_audit_count, p2_entries_with_anchor_hits, dangling_claim_anchors, concept_contract_violations, fix_applied}`. Concept-hub candidates: `candidates concepts --json` → `[{term, pages, categories}]`. Contract violations are advisory (Definition ≥40 words, span≥2 hubs need Cross-domain connections, Definition shouldn't paraphrase a spoke).
- `audit --json` → `{papers, cross_wiki_citations, edge_summary, recommended_additions, shared_citation_anchors, anchor_groups, categories, category_breadth, count_normalized}`.

### Exit-code contract

| Code | Meaning |
|---|---|
| **0** | Success. Zero results still `0` for read-only tools. |
| **1** | User-input error / no-result-where-expected. |
| **2** | Environment error: missing index, provider unreachable, disk unreadable. |
| **3** | Reserved: internal bug / uncaught exception. |

---

## Search — full-text + See-Also

Tantivy-backed (`.tantivy-index/`, gitignored).

```bash
researchwiki reindex                              # rebuild from wiki/
researchwiki search "CRISPR off-target"           # BM25 keyword
researchwiki search --like compbio/smith-2024-... # See-Also on a page
researchwiki search "prime editing" --see-also    # keyword + 2 related per hit
```

Query syntax: Tantivy's (`"quoted phrases"`, `field:value`, `+required`, `-excluded`).

**Category auto-suggest**: `--category` omitted → ingest votes among top-5 paper-type neighbors; writes `# auto-suggested: N/5 neighbors agree`. Falls back to `category: [TODO]` if <3/5 agree or index missing.

---

## PDF Management Rules

- **Copy, never symlink** from external locations into `inbox/`.
- **Move (`mv`) from `inbox/` → `papers/{stem}.pdf`** during processing.
- `pdf_path` always inside `papers/`; `pdf_filename` matches `basename(pdf_path)`.
- **One canonical PDF per page**. Stem collisions classified by DOI prefix:
  - `journal-upgrade` (preprint page, incoming journal): PDF auto-swapped; manually update YAML `doi:`/`venue:` (run `preprint-check --doi <preprint-doi>`). Body + `[[wikilinks]]` preserved. Logs `pdf_upgrade`.
  - `duplicate` / `preprint-downgrade` / `unclear`: PDF stays in `inbox/`; agent path raises rather than overwriting.

---

## log.md — chronological record

Append-only; one H2 per entry (parseable-prefix — `grep '^## '`):

```markdown
## [2026-05-03] ingest | Smith 2024 — Example paper title
Category: compbio. DOI: 10.1038/example. Cross-links: 4 outgoing, 2 incoming.

## [2026-05-03] query | Four strategies for reducing off-target effects → wiki/synthesis/off-target-strategies.md
Referenced papers: smith-2024, jones-2025, lee-2026, chen-2024.
```

`wiki/log.md` is gitignored. `ingest`, `synthesize`, and `lint` auto-append; manual entries match the prefix format.

---

## Browsing with Obsidian

Open `wiki/` as an [Obsidian](https://obsidian.md/) vault for visual navigation, graph view, full-text search. Read-only — doesn't interfere with agent edits.

---

**When in doubt, follow Rule 1 — lean toward not fetching.**
