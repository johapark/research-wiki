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
   | Semantic Scholar Graph | title, authors, year, venue, externalIds, references, citations, `/recommendations`, `abstract` (verbatim), `tldr` (draft/cross-check only) | `ingest`, `scout`, `neighbors` |
   | Crossref | title, authors, year, container-title, ISSN, `reference` list, `type` | `ingest` (S2 fallback) |
   | PubMed E-utilities | PMID, `pubtype`, `pubdate`, retraction linkage | `retraction-check` |
   | bioRxiv/medRxiv | `server`, `version`, `date`, `category`, `type`, `published` DOI. `abstract` **not re-exposed** | `preprint-check` |
   | ORCID Public API | names, latest employment. `biography`/`keywords`/`email` **not re-exposed** | `orcid-lookup` |

   **Prose fields:** S2 `abstract` verbatim OK; S2 `tldr` draft/cross-check only (record `tldr_source: semantic-scholar`). Everything else prose-y (venue blurbs, retraction reasons, ORCID bio) banned.

   **Provenance.** Whitelist claims carry source + fetch date (YAML or inline `(PubMed, fetched 2026-05-07)`).

   **Extraction, not ingestion.** Fields inform *decisions*, never paraphrased into prose.

   **Exception — agent-native web scouting.** Only when the user explicitly asks for `researchwiki scout web`, the active chat agent may use its native web-search harness for broad discovery. The CLI emits a bounded handoff but has no search provider of its own; the agent answers conversationally with native citations, while the repository caches only a minimal, self-attested, `discovery-only` source receipt under `.scout-cache/`—never research prose or a formal report. Web results cannot support wiki prose, claims, or wikilinks until the underlying PDF is ingested. On a resumed session use `researchwiki scout web list` / `show <run-id> --json` rather than duplicating a request. Read [`prompts/scout-web.md`](./prompts/scout-web.md) before running it.

   **Source credibility (non-whitelist).** For wiki evidence, peer-reviewed journals + author-hosted PDFs OK; blogs, Wikipedia, aggregators, social, press releases — do not paraphrase. Preprints only via `inbox/`. Web-scout discovery may surface those source types as leads, but does not promote them to evidence.

   **User-provided URL exception.** User explicitly hands you a URL/repo → authorized to `WebFetch` (or `gh`) *that exact URL* for the **conversational answer**. Bounds: only the given URL (no transitive links); wiki prose still needs the PDF (Rule 3); cite inline (`per the README at github.com/…`), no `[[wikilink]]`; GitHub via `gh` preferred; doesn't apply to URLs Claude generated.

2. **Answer from the wiki first.** `wiki/` is the only source of truth.
3. **Wiki insufficient → re-read the PDF.** `papers/{stem}.pdf`; use `pypdfium2` or `Read`. Supplementary at `papers/{stem}.supp/{filename}`.
4. **No paper on the topic → say so.** *"I don't have a paper on this — please give me the PDF."* Don't improvise.

These apply to every response, including synthesis pages.

**Corollary — don't let model priors leak.** Never infer affiliations, funding, dates, DOIs, or numerical results beyond what the PDF states. When in doubt, omit.

**Corollary — cross-links must be source-supported.** `[[wikilink]]` only when the source explicitly cites/builds on/contrasts the other paper. Topical adjacency alone is not enough.

**Corollary — flag adjacent gaps.** After a substantive cross-paper answer with partial coverage, close with a concrete "What's missing" line (specific paper types, not vague topics) and ask user to drop PDFs in `inbox/`. Skip when coverage is comprehensive.

**Corollary — ground through the claims DB.** Wiki pages are truth (Rule 2); the **claims DB is the grounding layer**. Before authoring any page with factual claims (synthesis, idea, filed query), `researchwiki claims "<topic>"` (or `--by-stem <stem>`) is the **first stop** — pre-graded units anchored to paper + section, scored against the PDF. Cite at the claim level with `[[stem#claim_slug]]` anchors (durable, content-addressed — a slug like `kc-9f3a2b1c` survives `db rebuild` and its identity is verified by `check-grounding`); the `claims` CLI prints the exact citation form to copy. Fall back to bare `[[stem]]` when the paragraph refers to the paper as a whole. On prose-heavy pages, use **academic footnotes** — one per source paper, `[^id]: [[category/stem]]` at bottom (exact form + grading gotcha in [`prompts/synthesis-page-author.md`](./prompts/synthesis-page-author.md)). Paper-page *Related Papers* stay inline `[[wikilink]]`. **In markdown tables**, footnotes don't render — use bare `[[stem]]` (or `Short-name [[stem]]`), never `[[stem\|alias]]`. **Never write `claim_id:NNN`** (legacy row-id form; `check-grounding` still tolerates it on old pages, but no current tool emits it — row ids are `AUTOINCREMENT` and reassigned on `db rebuild`, so cite the slug instead). Verify synthesis + idea pages with **both** gates, both must exit 0:
- `researchwiki check-grounding <page>` — structural (every claim carries a citation)
- `researchwiki grade synthesis <page>` — fidelity (each cited claim holds in the paper it cites)

Then `researchwiki check-coverage <page>` — recall of wiki papers ranking high on the page's `topic_seed` that the page doesn't cite. Advisory, but every unreferenced hit should be a deliberate decision. Hits whose **contribution claims** also match the seed are marked `← claim match`, print the claim as evidence, and sort first — a page-text hit with no claim behind it is usually shared vocabulary (`foundation model`), so read that ordering as the gate's own confidence.

---

## Editing this file

This file loads every turn — keep it lean. Trigger-gated procedures live in `prompts/{slug}.md`; leave a one-line pointer here with the trigger condition. Don't inline workflows that won't run on the current turn.

**`AGENTS.md` is a symlink to this file** — don't "fix" it into a separate document. Codex CLI (and Cursor, Aider, Continue, Gemini CLI, Cody, …) auto-load a repo-level instruction file but do *not* follow markdown links out of it, so a pointer file would leave every non-Claude agent running without the Four Rules. The symlink is what guarantees each one gets this contract verbatim instead of a suggestion to go read it. Same reason applies to any future tool-specific filename (`.cursorrules`, `GEMINI.md`): symlink it here rather than copying, so the contract can't drift per-tool.

---

## Repository Structure

```
Research/
├── CLAUDE.md, README.md, pyproject.toml
├── prompts/                # Trigger-gated procedures + LLM system prompts
├── config/                 # models.*.yaml (per-phase routing), pricing.yaml
├── inbox/                  # Raw drops awaiting processing
├── papers/                 # Canonically-named PDFs only (cp, never symlink)
├── researchwiki/           # Framework package (CLI, tasks/, providers/)
├── .ingest/                # Per-paper digests (delete after use)
├── .s2-cache/              # Cached Semantic Scholar responses
├── .scout-cache/           # Quarantined agent-native web leads/results
└── wiki/                   # Wiki pages (English)
    ├── {category}/
    ├── synthesis/          # Retrospective field maps
    ├── ideas/              # Forward-looking design proposals
    ├── concepts/           # Single-term hub notes (bridge nodes)
    ├── references/         # Regulatory guidance, protocols, whitepapers, books
    └── index.md, log.md, views.md
```

**Invariant**: `papers/` holds only canonically-named PDFs. Raw drops live in `inbox/`. A non-empty `inbox/` is the backlog.

`wiki/`, `papers/`, `inbox/` are gitignored in full (no `.gitkeep`) and **may be directory symlinks** into a synced folder — the recommended layout when the user syncs their library ([`prompts/migration-backfill.md`](./prompts/migration-backfill.md#keep-the-checkout-out-of-the-synced-folder)). Deliberate; don't "fix" them. Unrelated to the per-PDF *"copy, never symlink"* rule below, which governs files moved into `inbox/`. Missing entirely on a fresh clone — `researchwiki init --scaffold-only` creates them. Always run `researchwiki` from the repo root; paths resolve from `Path.cwd()`.

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
- Hyphenated terms = one word (`Cas-OFFinder` is 1 word). Unicode dashes count as hyphens (`ATAC‐seq` → `atac-seq`); a suspended compound's dangling hyphen doesn't survive (`epigenome- and` → `epigenome`).
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

- **Scaffold**: `synthesis`, `ideas`, `concepts`, `references`, `other` — created by `researchwiki init` (`categories.DEFAULT_DIRS`), not committed. First four are **page-type dirs**, not content categories; `other` is the classifier's abstention bucket.
- **Growth is explicit**: a new category exists only once `wiki/<category>/` is created. `--category X` rejected if dir doesn't exist; classifier abstains to `other`. Typos can't spawn categories.
- **Cold start**: `researchwiki bootstrap-categories` reads `inbox/` and proposes+creates dirs. See Operations → Initialization.
- **At ingest**: classifier picks existing category or abstains. `wiki/other/` ≥10 papers → `status` flags it, `suggest-splits` proposes splits (new category / reassign / stay).
- **Divergence (populated category)**: a category can grow a sub-cluster distinct enough to speciate into a sibling. `status` surfaces a cluster-verified, decay-stamped nudge; `researchwiki suggest-splits --category <cat>` (or `--all`) judges each separable sub-cluster `split_out` vs `stay` and prints migration steps. Same review-gated, human-applied model as the `other`-bucket splits — nothing auto-creates a category.

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

**H2 spine — `Question` → `Short answer` → … → `Tensions / open questions` → `What would update this page` → `References`.** Unlike idea pages, this is a **spine with permitted extension, not a closed list**: idea sections close because grounding policy differs per section (Background/Caveats strict, Opportunities/Plans allow `*(model prior)*`), so a sixth *content* section would have undefined policy. (`## References` and `## What would update this page` are the two exceptions — they carry no claims of their own, and `lint`'s `idea_contract` allows both.) Synthesis grounding is uniform, so extra H2s are free — and better. `researchwiki synthesize` scaffolds the spine.

- **Required**: `Question`, `Short answer`, `What would update this page` — and `References` whenever the page uses `[^id]` footnotes, since that's where they're defined.
- **Recommended**: `Tensions / open questions` (that exact spelling — three variants had drifted). Skip it on a page with no genuine disagreement rather than padding an empty section.
- **The middle is yours.** `Evidence from the wiki` is the scaffold's placeholder, not a required slot: 10 of 23 pages replace it with thematic H2s (`Positions on the axis`, `Architectural lineage`, `When to use which`), which is what the scaffold's own comment asks for. Comparison and axis shapes are first-class. One cost to know: `index.pages_semantic._INDEX_SECTIONS` matches synthesis pages on `Short answer` / `Question` / `Evidence from the wiki` / `Summary`, and its whole-body fallback fires only when **no** section matched — so a page keeping the two required H2s and renaming the middle has that middle excluded from the semantic index, below `thin_index_text`'s floor. Worth a `Short answer` that carries the page's full vocabulary.
- **Why the names are exact**: `check-grounding` exempts `## What would update this page` by exact string, so a variant silently switches the exemption off and the gate then demands citations for content designed not to have them. `index.pages_semantic._INDEX_SECTIONS` also matches by name, and `Short answer` must come first because the embedder truncates at 512 tokens.

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
pdf_path: "[[fda-2026-....pdf]]"       # Obsidian wikilink → click-to-open (papers/ sits beside wiki/ in the vault); quote it
source_url: ""                        # optional: canonical URL for blog posts / online whitepapers
author_model: "claude-opus-4-7"       # optional: LLM that authored the page (manual whitepaper path — mirrors the field agent ingest writes on paper pages)
keywords: []                          # required: 5–10 short phrases describing the doc's content, parallel to the keywords field on paper pages. Fewer than 5 is a `lint` finding (`missing_keywords`) and the writer won't emit a shorter list at all
ingested_at: 2026-04-15T14:30:00      # ISO 8601 local, UNQUOTED (a real YAML timestamp — Dataview's date column needs it). Manual path: stamp with `date +%Y-%m-%dT%H:%M:%S`
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

**Required H2 order — Verdict → Background → Opportunities → Plans → Caveats**, plus `## References` whenever the page uses `[^id]` footnotes (that's where they're defined; `lint` reports `idea_footnotes_without_references` without it).

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

Single-term **hub note** — a mini-synthesis around one recurring concept (surfaced by `researchwiki candidates concepts [--bridges]` at ≥3 papers). Ties every wiki paper that instantiates the term into one bridge node; most valuable when `concept_span ≥ 2` (spans categories the citation graph and semantic-KNN don't bridge). Strictly grounded (no model priors). YAML: `type: concept`, `category:` (content), `referenced_papers:` (the spokes), `concept_thesis:` (**required** — one sentence on why this is a concept and not glossary/synthesis; the scaffolder refuses without it), `concept_span:`, `generated_at:`, `topic_seed:`. Required H2s: **Definition** → **How it appears across the corpus**; optional **Cross-domain connections**, **What would update this page**. Each member paper gets a reciprocal `[[concepts/<slug>]]` back-link. Verify with **both** gates (`check-grounding` + `grade synthesis`) + advisory `check-coverage`.

**Manual workflow** — see [`prompts/concept-page-author.md`](./prompts/concept-page-author.md). Scaffold-first: `researchwiki concepts <term> --thesis "<one sentence>"` builds a grounded stub; fill Definition + spokes, then verify. Update `index.md` (`## concepts`), append to `log.md`.

### 6. Log / meta pages

- `index.md` — page catalogue, by category. Bullet prose comes from each page's `hook:` field (Step 3), so the file is regenerable rather than hand-maintained.
- `log.md` — chronological record.
- `views.md` — Dataview dashboard (`type: dashboard`), scaffolded by `researchwiki init`. Renders only inside Obsidian. Root bookkeeping pages carry no `category:` — the parent dir is `wiki`, so adding one trips category-drift.
- `wiki/synthesis/suggested-additions.md` — gap map from `researchwiki scout`.

**PDFs needing replacement** are not a file. A page whose PDF wasn't text-extractable records that in its own YAML `pdf_extraction_note:` (plus `abstract_source:` for what it was built from instead); `researchwiki status` lists them under *Workflow state*.

### 7. Commentary page — `wiki/{category}/{stem}.md`

A piece *about* someone else's paper: Research Highlight, News & Views, Comment, editorial. It lives beside the paper it discusses (a category dir, **not** `references/` — that's for guidance and books) and carries `type: commentary` plus `primary_paper:`.

```yaml
type: commentary
primary_paper: "[[single-cell/gold-2026-scoring-gene-importance-by-interpreting]]"
# …or a YAML list when the piece covers several, with a citation string for any
# primary that isn't in the wiki:
# primary_paper:
#   - "[[cgt/orosco-2026-dna-guided-crispr-cas12-for-cellular-rna]]"
#   - "Wu et al., *Nature Biotechnology* (2026) — companion study; not in this wiki"
```

**Why the type exists: attribution.** Typed `paper`, a commentary's claims are extracted and credited to the commentator — the primary authors' results asserted under the wrong stem, and citable via `[[stem#slug]]` anchors from synthesis pages. Neither fidelity gate objects, because the claims genuinely *are* in the commentary's PDF; the gates check faithfulness to the cited source, not entitlement to the claim. `type: commentary` is the structural fix: `db.rebuild` extracts claims only from `type: paper`, so retyping retracts them on the next `db rebuild` and they cannot come back.

**Contract.** Open with a blockquote banner above `## Summary` saying the page is a commentary, naming the primary paper, and directing the reader to cite it. Write `hook:` about *the page* — what it covers and that it carries no findings of its own — never a restatement of the primary paper's contribution. Body sections may stay as authored (they describe the primary work accurately); what changes is the attribution frame. A commentary is **not** citable evidence: cite the primary paper, and where only the commentary is in the wiki, attribute in prose to the original authors and footnote the commentary as the route ("…as summarized by[^marchal-2026]").

Multi-item Research Highlights columns are the sharpest case — one *Nature Genetics* column can summarize four unrelated studies by four groups, so the single stem otherwise asserts all four groups' work.

**Detection is local, not upstream.** Crossref returns `type: journal-article` and PubMed `Journal Article` for highlight and primary alike, so no metadata lookup can catch this. `agent ingest` blocks it structurally instead — see *Ingest → Commentary guard*.

### Page-type discipline

- Every page declares `type:` — `paper`, `synthesis`, `guidance`, `protocol`, `whitepaper`, `book`, `idea`, `concept`, `commentary`, plus `meta` / `dashboard` for the wiki-root bookkeeping pages. The last two are exempt from `hook:` and skipped by `db rebuild`.
- Synthesis, idea + concept pages must cite their sources and pass **both** `check-grounding` + `grade synthesis`. Synthesis/idea cite via the body (inline `[[wikilink]]`s + `## References` footnotes); concept pages use the `referenced_papers:` spoke list.
- **Exemption**: the `## What would update this page` H2 is skipped by `check-grounding` (name-narrow, exact heading, case-insensitive). Other "next steps"-style headings aren't exempt.
- Idea pages must follow Verdict → Background → Opportunities → Plans → Caveats.
- Concept pages must follow Definition → How it appears across the corpus (Cross-domain connections optional).
- Commentary pages need `primary_paper:` and a banner blockquote, produce no claims, and are never cited as evidence — cite the primary paper.
- **`tags:` is for concept / idea / synthesis pages only.** Paper and commentary pages don't carry it: the field was provenance there (`ingested-via-agent` was the only tag 334 of 391 paper pages had) and its topical remainder was near-singletons, while `keywords:` — required at 5–10, `lint`-checked, indexed — already carried the vocabulary. On the other three types `keywords:` is *exempt*, so `tags:` is the only keyword-like field they have and it is embedded for them (`index.pages_semantic._TAGS_CARRY_SIGNAL`).

---

## Operations

### Initialization — cold start

If every `wiki/{category}/` is empty AND user signals they're new (*"set this up"*, *"initialize"*), read [`prompts/init.md`](./prompts/init.md). Proactively offer init; alternatively suggest `researchwiki init` (interactive wizard).

### Model providers — routing and mixed mode

Per-phase provider comes from the selected model config: **`RW_MODELS_CONFIG`** when set, otherwise `config/models.yaml`. A bare selector resolves under `config/`; an absolute/path-separated value is used verbatim. Only an *absent implicit* `config/models.yaml` activates the built-in OpenAI endpoint and all-Luna role table. `researchwiki status` prints the active path. **`RW_LLM_PROVIDER`** is a global override that forces every phase to one provider and silently defeats per-role mixing. It replaces the provider but **not** the model, so forcing it over a config chosen for another backend mints pairs like `anthropic/gpt-5.6-terra` that no API serves; this mismatch is warned about once per process. Prefer `RW_MODELS_CONFIG` for whole-backend swaps.

**Dotenv profiles.** The optional root `.env` loads automatically and parent-shell variables take precedence. For multiple provider setups, copy `.env.template` to `.env.NAME`, restrict it to mode `0600`, and run every command as `researchwiki --env-file .env.NAME <command>` — the global option must come first and an explicitly selected file must exist. A named profile replaces the root `.env`; it does not merge with it. Exported credentials may still supply `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, but inherited routing variables (`RW_MODELS_CONFIG`, `RW_LLM_PROVIDER`, `RW_LLM_BASE_URL`, `ANTHROPIC_BASE_URL`) are rejected; put routing in the selected profile and do not `source` it as a shell script. `researchwiki --env-file .env.NAME init` writes selectors for known providers using the tracked templates and leaves checkout-global `config/models.yaml` unchanged. A custom OpenAI-compatible backend asks for an explicit writable config path such as `config/profiles/litellm.yaml`; no path is inferred from `.env.NAME`.

The OpenAI-compatible endpoint (`openai-compatible`/`lmstudio`/`openai` providers) is declared per config via top-level `base_url:`, so a backend switch normally rides on `RW_MODELS_CONFIG` alone. Precedence is `RW_LLM_BASE_URL` (ad-hoc override) → config `base_url:`. **Any selected config with an effective OpenAI-compatible route must provide one of those endpoints** or routing fails closed with environment exit code 2. OpenAI-compatible routes consume the process-level `OPENAI_API_KEY`; named profiles may carry different values, so switching backends does not require editing the root `.env` (LM Studio needs none; Gemini uses `OPENAI_API_KEY`).

**Anthropic-compatible third parties (e.g. z.ai GLM)**: use `provider: anthropic` with `ANTHROPIC_BASE_URL` stopping at the host root (z.ai: `https://api.z.ai/api/anthropic`, **no** trailing `/v1`). Free tiers may 429 the parallel author phase — drop to `-n 1`.

**Thinking controls are transport-specific.** `reasoning_effort:` is sent only on the OpenAI-compatible transport. Providers disagree on its vocabulary, so a rejection that names `reasoning_effort` / `reasoning.effort` is negotiated to the nearest advertised value (preferring less thinking on a tie), or the field is omitted when the endpoint supports none; the result is cached per endpoint+model for the process. LiteLLM-wrapped 500 responses are handled the same as direct 400s. Anthropic models never receive `reasoning_effort`: calls that request no thinking use Anthropic's native `thinking: {type: disabled}` instead. Thus the intent works on both transports, but the wire field does not.

**Gemini free tier (`config/models.gemini.yaml`)**: observed ceiling is ~5 requests/minute on the Flash model, shared per-project (not per-key) — `agent ingest`'s default 4 batch workers × 2 parallel author drafts blows through it. Pass `-w 1` to serialize the batch; add `-n 1` if 429s persist.

**`chat-relay` provider** — select it with `researchwiki init`, which puts `RW_LLM_PROVIDER=chat-relay` in the active dotenv profile; under `--env-file`, do not export that routing override in the parent shell. `researchwiki` then has no API key of its own and relays each prompt to *you* to fill. Read [`prompts/chat-relay.md`](./prompts/chat-relay.md) for that protocol; it's a specialized worker role and fires only under that env var.

### API pricing — `config/pricing.yaml`

Per-million-token rates for Anthropic + OpenAI models, carrying an **`as_of:` date** and its `sources:` URLs. Read by `agents/model_config.py` (which already owns the per-phase model routing, so one module holds all the facts about a model); used by `status`'s cost rollup and `insights`' per-model spend, both of which print the date beside the figure.

`model_config.rate_for(model)` matches by **longest prefix**, so the dated build IDs the API returns (`claude-haiku-4-5-20251001`) resolve to their family. Unknown model → `None` → `$0.00`, which is right for a local backend; `status` separately names any *cloud* model missing from the table so a stale file is visible rather than a silently understated bill. Sonnet 5's introductory rate is time-boxed via `until:` and lapses on the stated date.

New ingest rows record cache-read/write token subsets, and reports apply the table's Anthropic cache rates. Legacy rows keep unknown cache detail rather than guessing it. To refresh pricing: open both `sources:` URLs, correct rates, **bump `as_of:` in the same edit**, never delete retired models (old rows still reference them), then `pytest tests/test_pricing.py`.

### Ingest — add a new paper

**Default path: `researchwiki agent ingest`.** Handles pypdfium2 extraction, DOI detection, S2 lookup, LLM-reconcile, stem derivation, `mv` to `papers/`, page authoring, atomic back-link apply, and `researchwiki evolve`. Logs `ingest_iterations` to the state DB; writes no `tags:` (see Page Types).

**Step 0** — drop PDF into `inbox/`. **Read the inbox with `researchwiki status`, not `ls inbox/`** — same answer plus the aggregate signals (stale index / DB, orphans, cost rollup, the three decay-stamped nudges) that the every-~5-ingests cadence under *Lint* would otherwise be the only thing surfacing.

**Step 1** — run agent ingest. **For ≥2 PDFs pass them all to a single `agent ingest` invocation** — it auto-enters crash-safe batch mode (4 workers by default, atomic `.ingest/batch-<ts>/checkpoint.json` per completion, `--resume <batch-dir>` picks up after a crash). Do NOT fan out one background Bash per file: it bypasses the checkpoint, uncaps concurrency, and multiplies `state.db` write contention. `agent ingest` takes **no `--category`** — the classifier picks one per paper and abstains to `other`. (`--category` is the digest path's flag, on `researchwiki ingest`.)

**Run it backgrounded** — one invocation, backgrounded, whether it's 1 PDF or 20, so the conversation continues; this is not the banned per-file fan-out, and it needs no subagent (a subagent spends a whole context waiting on stdout you have to read yourself anyway). While it runs the wiki is **read-only**: no `reindex` / `db rebuild` / `lint` / `grade` until the completion notification lands, or they contend with promote's writes to `state.db`, `wiki/`, `index.md` and `log.md` — and a `reindex` that wins the race indexes a half-landed tree.

**Keep monitoring until the process reaches a terminal state.** Silence or a long provider retry is not completion. Retain the background session, poll it without starting competing wiki writes, and do not begin After ingest until the command has exited and every batch input is terminal in the checkpoint. Report failures and sandboxed papers alongside successes; never stop after the first completion in a multi-paper batch.

```bash
researchwiki agent ingest inbox/<raw-filename>.pdf              # single PDF
researchwiki agent ingest inbox/*.pdf                           # ≥2 PDFs — batch, 4 workers, checkpoint
researchwiki agent ingest inbox/*.pdf -w 2                      # cap workers (e.g. rate-limited providers)
researchwiki agent ingest --resume .ingest/batch-<ts>/          # resume after a crash / Ctrl-C
```

Optional per-PDF guardrails are `--max-model-calls`, `--max-tokens`, `--max-cost-usd`, and `--max-wall-seconds`; batch mode forwards them to every worker. A limit is reserved before a model call, so parallel drafts cannot oversubscribe it. Exhaustion writes a terminal `budget-exhausted` event and preserves the best graded partial under `.agent-output/` when one exists. Cost budgets refuse an unpriced cloud model instead of pretending it is free. Once promotion has irreversibly succeeded, the guard is suspended so indexing and grade persistence finish rather than leaving a half-maintained page.

**Timing and telemetry questions use `researchwiki insights`, not ad-hoc SQL.** `researchwiki insights --attempts` lists attempts (combine with `--days N` for a recent window); `--attempt-id <full-id>` prints that ingest's phase breakdown; `--stem <stem>` filters the attempt/lineage views; add `--json` for machine-readable milliseconds. The phase table reports min/mean/median/p95/max plus `samples/eligible`, so migrated and interrupted rows with NULL timing are never treated as zero. `commit` is the end-to-end parent; rows prefixed `↳` are nested maintenance and excluded from phase-work totals. New runs have an exact terminal wall timer. Older or partial attempts show `wall≈` from their event span, explicitly marked as a fallback.

**Never write ad-hoc scripts to ingest PDFs, and never fan out one Bash task per file.** Always use `agent ingest` (or digest path for recovery); rely on its built-in batch mode for multi-PDF runs. **One exception: `chat-relay`** — batch mode redirects each worker's stderr into `.ingest/batch-*/worker-*.log`, which is where the relay prints its pending-prompt notice, so a batch run under chat-relay looks like a hang and then times out. Parallelize it with one foreground single-PDF invocation per subagent instead ([`prompts/chat-relay.md`](./prompts/chat-relay.md#parallel-ingests--fan-out-but-not-with-batch-mode)).

**Metadata adoption gate.** A DOI is adopted only if the resolved record *is this paper* (`metadata_sanity`: first-author surname present, year within ±1, ≥50% title-content-word overlap) — resolvability alone is not enough, and the URL-DOI hunt no longer reads past the References heading, where every DOI belongs to somebody else. When the PDF prints no usable DOI, an S2 title match is tried under the same gate. A masthead-derived venue naming typesetting furniture is dropped rather than recorded; `lint`'s `venue_suspect` and `backfill doi --verify` are the backstops. Verified cross-links are written in **both** directions at promote time, and a `topical` label is upgraded to a citation when the source PDF's reference list proves one. Thin extraction (no results/discussion section, or zero PDF-side claims) logs a warning — the page can still pass every gate on a thinner grounding corpus than it appears to have.

**Commentary guard.** `agent ingest` refuses to auto-promote a commentary-shaped PDF as `type: paper`: a strong masthead label (`Research Highlight`, `News & Views`, or a line-anchored `Editorial`) fires alone, as does a publisher news-DOI namespace (Springer Nature's `10.1038/d…`, its own type declaration); otherwise `page_count == 1` **and** Crossref `reference-count == 0`, or a line-anchored weak label (`Comment(ary)`, `Books & Arts`, `In this issue`) plus one structural signal. Page lands in `.agent-output/` typed `commentary`, fired signals named in the gate reason and its frontmatter; `--auto-promote` overrides but still writes `type: commentary`. Unlike other gate failures this one is **not** DEBUG-repairable — no rewrite of the prose changes what the PDF is. Cost is near-zero: the Crossref lookup is gated on a local pre-trigger that fires on ~3% of PDFs, so the common path adds no network call. See `researchwiki/agents/commentary.py` and Page Types §7. The digest path has no guard — check the PDF's masthead yourself.

**Promote is transactional.** Its five steps (page + DB row → PDF move → back-links → `index.md` → `log.md`) run inside a write-ahead journal under `.mutation/`, so a failure rolls the whole tree back rather than half-landing a paper. A failure still exits **2** naming the cause, and `--resume` records an input whose PDF already moved as `unresumable` rather than re-queueing a vanished path. A crash mid-promote leaves a journal: `status` reports it, the next `agent ingest` drains it automatically, and only a rollback that fails 5× needs hands ([`prompts/recovery.md` § Half-landed promote](./prompts/recovery.md#half-landed-promote)). `RW_MUTATION_JOURNAL=0` bypasses journalling — escape hatch, not a mode. One asymmetry to know: file state is journalled, the `state.db` row is not, so a *crash*-recovered rollback removes the page and leaves the row for the next `db rebuild`.

**Digest-path fallback — `researchwiki ingest`** — recovery, unextractable PDFs, special page types, custom-voice cases. Workflow + page-contract template in [`prompts/ingest-digest.md`](./prompts/ingest-digest.md).

#### After ingest (both paths)

**Step 2** — `researchwiki reindex`. Required after every batch: the Tantivy and semantic indexes are rebuilt wholesale, never incrementally.

`researchwiki db rebuild` is **not** required here. Every wiki-page writer in the package calls `wiki.commit_page`, which upserts that one page's row at write time, so `papers`/`claims` already track markdown. Rebuild remains the reconciler of last resort, for the three things per-page commits can't do: deletion detection, `claim-graph reconcile`, and edits made **outside** the package — hand edits, `Edit`/`Write` tool calls, `git` operations. Run it when you've edited pages yourself (as before any gate run on a hand-edited page), and periodically alongside `lint`. `researchwiki db verify` reports whether it's actually needed; it costs <1 s on a ~500-page corpus, so when unsure, just run it.

**Step 2.5 — Proactive claim cross-linking. Batched, not per-ingest.** Finds existing papers with near-paraphrase claims, LLM-judges each for real relationship vs vocabulary overlap, and auto-adds reciprocal `[[wikilink]]`s to both pages' Related Papers on confirmed matches (tagged `auto-added; claim-overlap`). Only the verdicts where one paper *engages* the other earn a bullet — `builds_on` / `refines` / `corroborates`. A `measures_same` verdict (same quantity, different cohorts — in practice shared methodology) records a typed claim-graph edge and **no bullet**: it's a real relation but not the source citing/building on/contrasting the other paper, so it fails the cross-link corollary. Edge-only matches are reported separately as `edge_only`. Read those edges back with `researchwiki claim-graph` (`--tensions` for unresolved contradictions, `--neighbors <stem#slug>`, `--contradicting`); `researchwiki claim-graph promote [--apply]` transitions candidate edges to confirmed. Neither subcommand shows up in the parent `--help` usage line — they dispatch on the first positional. **Off by default at ingest** — it spends a judge call per candidate pair and confirms a link on roughly 1 paper in 10, so paying per ingest buys little. Pass `agent ingest --claim-overlap` to run it inline anyway; otherwise the stem lands in the `claim_overlap_runs` backlog and you drain it in one pass:

```bash
researchwiki claim-overlap --backlog --dry-run     # preview the whole backlog
researchwiki claim-overlap --backlog --limit 20    # drain 20 stems
researchwiki claim-overlap <stem>                  # one stem, on demand
```

Coverage is tracked per stem with a fingerprint of the claims compared, so a regrade or re-ingest re-opens that stem rather than leaving it marked done against a stale comparison. `status` surfaces the pending count once ≥10 accumulate (decay-stamped, so it won't nag); `lint --json` carries the full list as `stems_missing_claim_overlap`. Skip for reference/idea/synthesis pages (no graded paper claims).

**Discovery tier — `researchwiki candidates pairs`.** The judged path above is tuned for *writing* bullets (precision); this is the layer below it, for *finding* what 0.83 never sees, and it lives with the other opportunity signals rather than on `claim-overlap`. **Lowering `--sim` is not the alternative** — at 0.70, 80% of all possible paper pairs qualify. It ranks a cosine *band* (0.72–0.83) by IDF-weighted shared-term mass: cosine measures register, rare-term overlap measures subject. Zero tokens, no judging, no writes. `--cross-category` narrows to pairs nothing else connects; `--decline <A> <B> --reason "…"` suppresses one (same vocabulary as `candidates concepts`). Act on an entry with the exact `researchwiki claim-overlap --pair A#slug B#slug` command printed on that row. `status` nudges once ≥15 cross-category pairs are unreviewed (14-day decay).

**Step 2.6 — Concept-hub attachment.** Agent path auto-runs `concepts.attach_after_ingest` (this one *is* automatic — unlike Step 2.5 it makes no LLM call): the new paper joins any existing `wiki/concepts/` hub whose `topic_seed` term appears in a **contribution claim** (key_contributions / results / methodology sections — a body-prose-only mention isn't enough; those log as near-misses). The spoke bullet cites the specific matching claim via `[[stem#slug]]`; `referenced_papers`/`concept_span` refresh on the hub, and a reciprocal `[[concepts/<slug>]]` lands on the paper (tagged `auto-added; concept-link`). No-ops until concept pages exist. Digest path: run nothing here (attachment is agent-only); new bridge concepts instead surface via `researchwiki candidates concepts --bridges` — span-≥2 terms are labeled `concept-ready (bridge)`. Scaffold one with `researchwiki concepts "<term>" --thesis "<one sentence>"` — the thesis is a hard gate, so the bare form exits 1 when stdin is not a TTY (see [`prompts/concept-page-author.md`](./prompts/concept-page-author.md)); creation stays review-gated (it writes prose → both gates). To backfill slug citations on existing hubs after new claims land, `researchwiki concepts --upgrade-spokes` rewrites bare `[[stem]]` spokes to `[[stem#slug]]` (idempotent). `researchwiki concepts refresh <slug>` drafts a hub's `## Cross-domain connections` from typed claim edges among its members — review-gated, emitted under `.ingest/`, never auto-applied.

**Step 3 — Set `hook:` on the new page.** The catalog gloss `index.md` renders after the citation: `[[category/stem]] — **Short name** (*Venue* year): {hook}`. Write it **result-first** — method + scale + the distinguishing finding — because its job is to separate this paper from the ~40 others in its category section; restating the paper's *question* (what sentence 1 of `## Summary` gives you) fails that job. Quote the value: hooks routinely contain `[[wikilinks]]` and `:`, both of which break unquoted. Advisory ceilings, `lint`-warned and **never auto-truncated**: **paper / commentary 400** chars (1–2 sentences), **concept / synthesis / reference 1000**, **idea 2000**.

**Agent ingest writes `hook:` automatically.** The author phase emits a `HANDLE:`/`HOOK:` trailer after the six sections; `phases.draft.split_gloss_trailer` parses it off the body (so the critic and graders only ever see the sections) and the *same* string lands in both YAML `hook:` and the `index.md` bullet — they can't drift. Costs no extra LLM call, and the author has the source sections in context. Malformed or missing → field omitted, page lands on `lint`'s `missing_hook`; nothing is ever salvaged from a Summary slice.

**Rule 1**: the hook is generated from PDF-grounded page prose, never from an S2 `tldr` — that's why a persisted `hook:` needs no provenance field. The digest path's *draft* gloss **is** `tldr`-seeded, so rewrite it before it becomes `hook:`; never paste it through.

**Backfill** — `researchwiki backfill hook` for pages that predate the field. See Operations → Backfill.

**Step 4 — Append to `log.md`** (auto-handled).

**Step 5 — Check stale syntheses.** `researchwiki lint --json`; inspect `stale_synthesis`, `stale_by_content`, `p2_entries_with_anchor_hits`. Refresh or leave a one-liner in `log.md`.

**Step 6 — Memory-evolution proposals.** Agent path auto-runs `researchwiki evolve`; digest path doesn't (run manually). Actionable proposals land under `.ingest/{stem}-evolution-proposals/`. **You are the reviewer** — read each proposal + target synthesis, verify patches against the source paper (numbers, framing, superlatives drift), one-paragraph verdict per proposal, ask user permission (one yes/no covers all from a single ingest unless specified). On approval: apply, `rm -rf` the proposal dir, update synthesis `generated_at:` (the citation lands in the body — synthesis/idea have no `referenced_papers:`). Skip when `evolve` returned zero verdicts or paper is a reference doc.

### Import — bring in a reference-manager library

`researchwiki import` ingests an existing corpus from Zotero / Paperpile / Mendeley / ReadCube, using the manager's own **export** (BibTeX / RIS / CSL-JSON) as authoritative metadata. Trigger: user has a library elsewhere, or drops a `.ris`/`.bib` into `inbox/`. Phases `preflight` → `inspect` → `apply` → `verify`; the first two and the last cost zero tokens. Their export already holds curated DOI/title/authors/year, so `inspect` records the exact per-paper `--doi/--title/--authors/--year` and `apply` feeds them through batch mode — turning ingest's most failure-prone stretch into a lookup. **Stage it** (`--limit 30`); `apply` re-checks liveness per record, so the next wave takes the next N. Read [`prompts/import-reference-manager.md`](./prompts/import-reference-manager.md) before running it.

### Export — emit the corpus as a bibliography, or as an OKF bundle

`researchwiki export` is the inverse: it writes the corpus as BibTeX / RIS / CSL-JSON for a reference manager or a manuscript's reference list, or as an **Open Knowledge Format** bundle (`--format okf`). Trigger: *"can I get a bib file"*, *"export my library"*, *"put this in Zotero"*; for OKF, *"export as OKF"*, *"make a portable bundle"*. Zero tokens, no network, deterministic — two runs are byte-identical, and nothing is written without `--out`. Filters `--category`/`--year`/`--stem`; the bibliography goes to stdout and the summary to stderr, so `researchwiki export --category cgt > cgt.bib` gives a clean file.

**The two formats have different scope, and that is not configurable.** The bibliography carries only pages describing something somebody else published (`paper`/`commentary`/`whitepaper`/`guidance`/`book`) — a BibTeX entry for a synthesis page would assert a publication that does not exist. OKF's unit is a *concept*, explicitly including abstract ideas with no underlying resource, so it carries **every** page type and simply omits `resource` where nothing is published. A page absent from the `.bib` and present in the bundle is correct in both; don't "align" them.

`--format okf` **requires `--out <dir>`** (a bundle is a directory tree whose file paths are the concept identities) and refuses a non-empty directory that isn't already one of its bundles — exit 2. Its `--json` report is a *different* shape from the bibliography's: `{format, okf_version, concepts, by_type, links_rewritten, links_unresolved, sources_emitted, verified_emitted, verified_absent_no_gate_record, generated_missing_actor, description_missing, skipped, stale_files}`.

Mapping worth knowing: `hook:` → `description` (wikilinks flattened to plain text), `doi:` → `resource`, `tags:` ∪ `keywords:` → `tags`, `author_model:`+`ingested_at:` → `generated`, footnote `[^id]:`/`referenced_papers:` → `sources[]` (the footnote label *is* OKF's `sources[].id`, so per-claim attribution carries over unchanged), idea `status:` → OKF `draft|stable|deprecated` with the native value kept in `x_researchwiki_status`. Unmapped frontmatter is preserved under `x_researchwiki_*` rather than dropped. **`verified` is emitted only for graded paper pages** (`process:researchwiki-grade-paper`, machine-confirmed tier); synthesis/idea/concept pages get none, because `check-grounding`/`grade synthesis` persist nothing and a trust claim without a record behind it is exactly the falsehood the gates exist to prevent. Full rationale in `researchwiki/okfexport.py`.

**The citekey is the page stem**, because a key already in a manuscript must never change; any short scheme recomputes its disambiguating letters at export time. Only page types describing someone else's publication are emitted (`paper`, `commentary`, `whitepaper`, `guidance`, `book`) — synthesis/idea/concept pages are the user's own unpublished analysis, and an entry for one would assert a publication that does not exist, so there is no flag to include them (use the share workflow instead). Gaps are downgraded and reported, never invented: a paper with no venue becomes `@misc` rather than an `@article` with no `journal`, and a furniture venue is suppressed. Read [`prompts/export-bibliography.md`](./prompts/export-bibliography.md) — the `--json` report doubles as a page-defect to-do list.

### Migrate — import paper pages from an older or simpler wiki

`researchwiki migrate` brings in **one-paper-per-PDF markdown** produced by an earlier release of this framework or by a simpler "PDF in → summary page out" generator, without re-authoring prose. Phases: `preflight` (deps/disk; hard-fails if the local embedding model is missing) → `inspect` (read-only classification) → `apply` (`--dry-run` stages first) → `verify`. All zero-token; `apply` prints the paid `backfill` follow-ups rather than running them.

Not for arbitrary note vaults — a page that isn't one-paper-shaped has no claims and no PDF to grade against, so `migrate` blocks it instead of landing a stub. Read [`prompts/migration-backfill.md`](./prompts/migration-backfill.md) before running it; the trap it exists for is that claim extraction matches H2 names **exactly** (`## Key Contributions` / `## Results` / `## Limitations` / `## Methodology and Architecture`), so a page with different headings yields zero claims, is uncitable, and is invisible to `ungraded_papers` — `lint`'s `zero_claim_papers` is the check that catches it.

### Backfill — fill missing YAML on pages that already exist

`researchwiki backfill <target>` populates a field without re-authoring prose. Targets: **`hook`** (catalog gloss + `short_name`, `-w N`), **`keywords`** (batched), **`doi`** (S2 → Crossref, sanity-checked). **`doi --verify`** inverts the work list — it checks the DOIs already recorded rather than filling missing ones, reporting `MISMATCH` when a DOI resolves to a *different* paper (and, free from the same lookup, venues that disagree with the provider). Read-only, exits 1 on any mismatch. Worth a pass after any ingest-metadata bug, because a wrong-but-resolving DOI is invisible to every other check — `lint` only reports `missing_doi`. All take `--dry-run`, `--limit N`, `--reindex`; each selects its work list from the matching `lint` check. Always `--dry-run --limit 5` first, and `db rebuild && reindex` after.

For a bulk import use `migrate` (above), which runs these for you in the right order. Backfilling more than a handful of existing pages: same prompt, [`prompts/migration-backfill.md`](./prompts/migration-backfill.md).

### Recovery — re-ingest after a broken ingest

When `lint --json` flags `missing_doi`/`stem_year_drift`/`unknown-` stem or agent landed bad metadata: re-ingest with overrides (not YAML-patching). Full workflow in [`prompts/recovery.md`](./prompts/recovery.md).

### Remove — retract a page

Trigger: ingested in error, retracted upstream, superseded under a different stem, or a mis-typed commentary. `researchwiki remove <stem>` (dry run by default; `--apply` writes, `--keep-pdf` leaves the PDF re-ingestable). Removes the page, PDF, caches, back-link bullets, `index.md` entry, concept spokes and DB rows; **reports but never edits** authored `[[stem#slug]]` / footnote citations on synthesis + idea + concept pages — expect `dangling_claim_anchors` on exactly those pages afterwards, which is the to-do queue, not a defect. Runs inside the mutation journal. **The target resolves by filename stem, not `type:`** — a synthesis, idea, concept or reference page is removable the same way, with the paper-shaped machinery (PDF, caches, `claims` rows) simply finding nothing; `index.md`/`log.md` are excluded from the scan. Note the asymmetry: authored prose is protected on *citing* pages, never on the target, so removing a synthesis page deletes twice-gated prose with only the dry run as a guard. Full procedure in [`prompts/remove-paper.md`](./prompts/remove-paper.md). **Not** for re-ingesting with corrected metadata — that's `prompts/recovery.md`.

### Recategorize — move a paper to a different category

Follow [`prompts/recategorize.md`](./prompts/recategorize.md). Directory is canonical (`db rebuild` ignores YAML `category:`); procedure repoints inbound links, updates YAML, verifies via `lint`.

### Benchmark — test a model on a fixture

When asked to benchmark/test an LLM by ingesting a `benchmark-fixtures/` paper: **always `agent ingest … --force-sandbox`** (writes to `.agent-output/`, never promotes to `wiki/` or touches `index.md`), select the config under test with an inline `RW_MODELS_CONFIG=` override, judge `.agent-output/<stem>.md` against `benchmark-fixtures/<stem>.yaml`. Full procedure + the mandatory `db rebuild && reindex` cleanup (if you ever promote by mistake) in [`prompts/benchmark-run.md`](./prompts/benchmark-run.md).

### Query — ask a cross-paper question and file the answer back

1. Answer from `wiki/` first. `researchwiki claims "<topic>"` is the **first stop** for factual claims (pre-graded, BM25+semantic-scored). Each hit prints its `[[stem#claim_slug]]` citation form — copy that directly into prose. `researchwiki search` for page-level discovery.

   **Structural/bibliometric questions go to the DB.** Corpus counts/filters — "how many cgt papers from 2024?", "which lack a DOI?", "everything in *Nature*" — via `researchwiki db papers [--year/--category/--page-type/--no-doi/--venue/--author/--status] [--count] [--json]` or `researchwiki db query "SELECT …"` for ad-hoc. Ingest telemetry — model quality/cost, hardest sections, token spend, phase distributions, attempt wall time, and exact per-step timings — goes through `researchwiki insights`; do not bypass that interface with `db query`.
2. Insufficient (Rule 3): re-read PDFs; update paper pages if worth keeping. `researchwiki pdf-search <stem> "<query>"` for a raw passage. When the evidence is *in a figure* — the passage says "see Fig. 4" and Fig. 4 is where the number lives — `researchwiki figures <stem>` lists captions (free, and often answers it), and `--figure N` renders that one page to `.figures-cache/` for you to `Read`. Render one page, only when the caption doesn't settle it: the PNG costs context in proportion to its pixel area.
3. No paper covers it (Rule 4): say so.
4. Cite facts with `[[wikilink]]`; mention sections in prose.
5. Non-trivial cross-paper → create a synthesis page. **This is how the wiki compounds.**

| User question shape | Location |
|---|---|
| "What is X?" (aggregated) | `wiki/synthesis/{x-slug}.md` |
| "Approaches to Z?" | `wiki/synthesis/{z-slug}.md` |
| "How does A compare to B?" | `wiki/synthesis/{a-vs-b}.md` |
| "Trajectory of field F?" | `wiki/synthesis/{f-trajectory}.md` |

Use `researchwiki synthesize --title "…" --topic-seed "…" --papers <stems>` to scaffold. With `--topic-seed` + `--papers` set, the stub's *Evidence from the wiki* section pre-populates with `claim_lookup` hits and each paper's claims. `researchwiki claims --by-stem <stem>` dumps one paper's citable surface (`--include-context` adds source-PDF chunks, each tagged with where in the PDF it sits — `§results, p. 7`; blank on claims graded before that existed, filled on the next `grade`).

After authoring, run both gates (both exit 0): `check-grounding` (structural) + `grade synthesis` (fidelity). Then `check-coverage` (advisory recall).

**Rules for filed answers:**
- Cite only wiki papers; every claim backed by a paper in `papers/`.
- Wikilink every claim.
- YAML: `generated_at: YYYY-MM-DD`, `topic_seed: "4–8-word query"` (no `referenced_papers:` — cite in the body via `[[wikilink]]`s + `## References` footnotes).
- Append to `log.md`: `## [YYYY-MM-DD] query | <question> → <page>`.

Before trusting an existing synthesis, check `lint --json` for `stale_by_content` / `stale_synthesis`.

**Refreshing `wiki/synthesis/suggested-additions.md`**: triggered by `stale_by_audit_count`, ingestion of a listed paper, or ≥3 ingests since last refresh. Workflow in [`prompts/scout-refresh.md`](./prompts/scout-refresh.md).

### Cross-link discovery — manual page writes

After writing or substantially editing a **multi-topic page** (synthesis, whitepaper, broad reference doc — ≥3 distinct named tools/methods/concepts), grep `wiki/` for each named entity and add source-supported cross-links. Agent ingest covers paper pages via `propose_crosslinks`; manual writes have no analogue. Template in [`prompts/cross-link-discovery.md`](./prompts/cross-link-discovery.md).

### Share a wiki page as a standalone document

When user asks to share a synthesis or idea page, produce a self-contained markdown at `output/share/<slug>.md` (gitignored — `output/` is the umbrella for everything emitted for an outside reader). Strip `[[wikilinks]]`, framework-specific YAML, and self-referential phrasing; rewrite footnotes to full academic citations with DOI links. Procedure in [`prompts/share-page.md`](./prompts/share-page.md). **Not** `researchwiki export`, which emits the corpus as a bibliography or an OKF bundle — "export" belongs to that command.

### Lint — periodic health check

- **`researchwiki status`** — instant, local dashboard: paper count per category, cross-link graph, `inbox/` backlog, resumable web-scout handoffs, orphaned PDFs in `papers/` (a PDF no page claims — workflow state, not a defect: `remove --keep-pdf` produces it on purpose), recent additions. Run after every ingest session, and at minimum every ~5 ingests since the last run — the per-ingest checks (Step 2.5/2.6, `other`-saturation, divergence nudge) are per-paper, so several signals only surface in aggregate: index staleness (`reindex` skipped), structured-DB drift (`db rebuild` skipped), cumulative orphans, cross-session backlog (stale `.ingest/*-evolution-proposals/` dirs, forgotten `inbox/` files or `scout web` runs), and the 7-day cost rollup. Auto-surfaces the concept-hub bridge count (see `candidates concepts` below) — if it prints a nonzero `Concept-hub candidates: N bridge term(s) …` line, that's your trigger to scaffold one. Same for `Claim-overlap backlog: N stem(s) pending` (≥10, decay-stamped) → `researchwiki claim-overlap --backlog`, and `Claim-pair discovery: N unreviewed cross-category pair(s)` (≥15, 14-day decay) → `researchwiki candidates pairs --cross-category`.
- **`researchwiki scout`** — bare `scout` / `scout citations` performs structured S2 citation scouting: cross-wiki citation edges, recommendations, and external papers cited by ≥2 wiki papers. Weekly or after batches. `--json` retains the legacy `.s2-cache/audit-{date}.json` snapshot contract; `researchwiki audit` remains a deprecated alias during the compatibility window. `scout web` is a separate, explicit agent-handoff workflow whose quarantined results are discovery-only; read [`prompts/scout-web.md`](./prompts/scout-web.md).
- **`researchwiki lint`** — local consistency checks: orphans, broken wikilinks, pages with no `type:` at all (invisible to every other check, because each consumer reads the field as `fm.get("type", "paper")` — so a page that lost it behaves as a paper and a commentary's claims would get extracted and misattributed), stale syntheses (ingest-date, content, audit-count — **never** file mtime, which moves on any edit and made every maintenance pass look like a source change; a paper whose ingest date nothing recorded is skipped rather than guessed at), missing back-links, YAML schema violations, page-type mismatches, category YAML↔dir drift, P2 entries with anchor hits, and the reachability of `prompts/*.md` from this file (`orphan_prompts` — a trigger-gated prompt nothing links, so no condition would ever make an agent read it; `broken_prompt_pointers` — a pointer whose file is gone). `--fix` applies deterministic repairs only: it auto-inserts missing back-links tagged `(auto-added; refine)`, recovers `ingested_at` / `author_model` for `missing_author_model` pages from the `ingest_iterations` telemetry log, and reconciles structured-DB drift by upserting current pages and deleting rows whose pages are gone. Provenance recovery is a recorded fact, not an inference, and is marked in YAML with `# recovered from ingest_iterations`; it fills a blank only, never corrects a recorded value, and skips any page the log has never seen (a migrated or hand-written page has no ingest date or model to recover — file timestamps are not a fallback, since back-link splicing resets them).

**Opportunity signals (not defects; user-initiated cadence):**

- **`researchwiki candidates concepts [--bridges] [--json]`** — recurring vocabulary terms with no `wiki/concepts/{slug}.md` yet, mentioned in a **contribution claim** (key_contributions / results / methodology) by ≥3 wiki papers — the same sections `find_members` matches over, so an advertised count is a floor on what will actually scaffold rather than a promise the scaffolder then refuses. A `limitations`-only mention still shows in `sections` and still moves `weighted`, but is not membership. Bridge candidates (span ≥2 categories) are the highest-leverage ones — `status` auto-surfaces the bridge count, so run this whenever that line is nonzero. Scaffold with `researchwiki concepts "<term>" --thesis "<one sentence>"` (see [`prompts/concept-page-author.md`](./prompts/concept-page-author.md)). At corpus scale the list fills with generic-bigram noise; `--triage` batch-LLM-classifies the whole candidate set against the concept-vs-glossary thesis test and auto-declines the noise verdicts (tagged `source=llm-triage`, reversible, `--dry-run` to preview). A single candidate can also be suppressed by hand with `--decline TERM --reason TEXT` (`--undecline`/`--list-declined` manage the list), so it stops resurfacing here and in `status`'s bridge count.
- **`researchwiki neighbors <page-or-doi>`** — what sits around a paper in the S2 citation graph (cited-by / citing / similar), each hit marked `[in-wiki]` or `[needs-ingest]`. Structural fields only, so it stays inside Rule 1. Use it for *is this still the current work?* and *what next?*; `--needs-ingest` narrows to the gap.
- **`researchwiki candidates pairs [--cross-category] [--json]`** — cross-paper claim pairs sitting *below* `claim-overlap`'s auto-link threshold, ranked by IDF-weighted shared rare-term mass inside a cosine band (0.72–0.83). Local, sub-second, no LLM, writes nothing. **Lowering `claim-overlap --sim` is not the alternative** — at 0.70, 80% of all possible paper pairs qualify. `--cross-category` narrows to the pairs no other structure in the wiki connects; `status` surfaces that count. Act on one with the exact `researchwiki claim-overlap --pair A#slug B#slug` command printed on that row; reject one permanently with `--decline <A> <B> --reason TEXT` (`--undecline`/`--list-declined`), which is fingerprinted on both papers' claim slugs so a regrade re-opens the pair.
- **`researchwiki candidates synthesis [--judge] [--write-proposals]`** — Louvain communities in the weighted paper graph (wikilinks + semantic cosine + keyword Jaccard) not yet covered by any existing synthesis page. Higher noise rate than concepts, so **not** auto-surfaced; run after ≥5 ingests since last, or when the user asks a cross-paper question that lands in an unfamiliar cluster. The default is a local, non-mutating preview; `--judge` opts into configured-model scope checks and `--write-proposals` persists review stubs under `.ingest/synthesis-candidates/`. Human picks a topic and runs `researchwiki synthesize`.

**Category-YAML↔dir drift** catches a recategorized paper whose frontmatter wasn't updated. Page-type dirs (`ideas/`/`synthesis/`/`references/`/`concepts/`) accept either the page-type name or a valid content category as YAML `category:`.

### Visualize — see the corpus structure

`researchwiki visualize` writes a self-contained interactive graph to `output/graph.html` (gitignored; `--open` launches it, `--json` emits the data instead). Zero tokens, no network — layout runs client-side.

Two edge kinds: `[[wikilinks]]` (what authors wrote) and typed claim edges from `.claim-graph/edges.db`, collapsed to one edge per page pair per relation. Only live claim statuses are drawn (`candidate`/`confirmed`/`promoted`) — `stale` means the claim the edge was judged against changed, so nobody currently stands behind it; `--claim-status stale` widens, `--no-claims` drops them. `contradicts` is drawn loud — the point is seeing that several tensions cluster on one paper, which `claim-graph --tensions` can only list.

Use it to decide whether a cluster deserves a synthesis page (pairs with `candidates synthesis`), or to spot an isolated component. **Not** for factual questions — it shows structure, not claims.

### Whitelist-API lookups — `retraction-check`, `preprint-check`, `orcid-lookup`

CLI wrappers around PubMed / bioRxiv / ORCID. Usage, YAML-recording rules, and workflow split for each in [`prompts/lookups.md`](./prompts/lookups.md).

### Agent output — prefer `--json`

- `search --json` → `[{key, stem, category, page_type, title, score, rrf_score, bm25_rank, bm25_score, semantic_rank, semantic_score, snippet}]`; `--see-also` adds `see_also`.
- `lint --json` → `{pages_scanned, invalid_frontmatter, orphans, broken_wikilinks, broken_index_bullets, orphan_pdfs, missing_backlinks, missing_type, page_type_mismatches, category_yaml_drift, stale_synthesis, stale_by_content, stale_by_audit_count, stale_evolution_proposals, missing_keywords, missing_hook, hook_too_long, missing_author_model, missing_doi, stem_year_drift, unquoted_wikilink_lists, supplementary_missing_on_disk, supplementary_orphaned_files, dangling_claim_anchors, orphan_prompts, broken_prompt_pointers, concept_contract_violations, idea_contract_violations, venue_suspect, none_placeholders, thin_index_text, ungraded_papers, zero_claim_papers, stems_missing_claim_overlap, duplicate_claim_sets, db_drift, cross_paper_contradictions, cross_paper_stats, p2_entries_with_anchor_hits, fix_applied}`. `broken_index_bullets` and `orphan_pdfs` are the two blind spots a **hand-deleted page** leaves: `broken_wikilinks` excludes root meta pages (so `log.md`'s historical template fragments don't drown it), which took `index.md` with it, and every other check starts from the page corpus, so a PDF whose page is gone is reachable from nothing. Neither is auto-fixed — an `orphan_pdfs` entry is the intended state right after `remove --keep-pdf`, and a stale bullet looks the same mid-recategorize as after a deletion. `orphan_pdfs` is *also* a `status` line (under *Workflow state*, beside the `inbox/` and `.ingest/` counters), which is where a human meets it; `lint --json` is the full stem list for an agent — the same split `stems_missing_claim_overlap` uses. `missing_author_model` is scoped to `type: paper` pages that carry `ingested_at:` — i.e. pages the pipeline wrote, where the field should always be present; pages predating it are out of scope, so the check stays a defect list rather than a legacy backlog. Repaired by `lint --fix` where the ingest log has the run; unrecoverable otherwise. `ungraded_papers` is drained by `researchwiki grade regression --missing-only` (one grader pass per paper — the expensive maintenance command); until a paper's claims are graded they can't ground a synthesis page and OKF export emits no `verified` for it. `duplicate_claim_sets` is advisory — page pairs whose claims are each other's nearest neighbours (reciprocal top-1 share ≥ 0.25), the structural signature of a commentary ingested as `type: paper`; `null` means skipped (claim-embedding cache cold or <50% covered — warm it with any `claim-overlap` run). Legitimate near-duplication is common (two trials of one therapy, a paper and its preprint), so the pair is reported and the call is the reviewer's. `cross_paper_stats` is `null` unless the opt-in `--cross-paper` check ran, and then carries `{pool, judged, skipped_already_judged, disagreements, sim_threshold}`. `pool` is filled before the `max_pairs` slice, so `--cross-paper-max-pairs 0` sizes a sweep for zero LLM calls. Every verdict — including the `agree` / `different_topic` clears — is recorded in `cross_paper_judgements`, so a repeat run judges only the pairs the previous one never reached (`--cross-paper-rejudge` overrides). Read the pool figures before planning a sweep: at 0.85 the great majority has already been judged across the corpus's ingests, and this judge finds *errors* (same experiment, different number), not methodological disagreements. `candidates pairs --json` → `{cos_lo, cos_hi, cross_category_only, pairs[]}`, each pair `{stem_a, stem_b, category_a, category_b, cross_category, citation_a, citation_b, cosine, idf_mass, shared_terms, text_a, text_b}` — `citation_*` are ready-to-paste `[[stem#slug]]` anchors. Concept-hub candidates: `candidates concepts --json` → `[{term, slug, pages, categories, weighted, label, source, sections}]` — `pages`/`categories` count contribution-section papers only (`sections` spans every section, so the two disagree by design); `label` is the triage signal (`concept-ready (bridge)` / `concept-ready (deep)` / `candidate` / `glossary-suspect`). Contract violations are advisory (Definition ≥40 words, span≥2 hubs need Cross-domain connections, Definition shouldn't paraphrase a spoke). `idea_contract_violations` is the idea-page analogue, also advisory: heading presence, the `Verdict → Background → Opportunities → Plans → Caveats` order, unexpected H2s (`## Related Papers` belongs on paper pages), YAML `verdict:`-vs-section-label agreement, and footnote definition/`## References` hygiene. **Neither page gate reads headings** — `check-grounding` and `grade synthesis` both parse paragraphs — so this is the only check that sees a missing `## Verdict`.
- `export --json` → `{format, records, by_entry_type, venue_missing, venue_furniture, doi_missing, authors_unparseable, skipped}` — the *report*, not the bibliography, so it claims stdout and suppresses the payload unless `--out` is also given. **`--format okf --json` returns a different contract** (see Export above): `{format, okf_version, concepts, by_type, links_rewritten, links_unresolved, sources_emitted, verified_emitted, verified_absent_no_gate_record, generated_missing_actor, description_missing, skipped, stale_files}`. Dispatch on `format`. `records + len(skipped)` equals the number of pages selected, and every list except `by_entry_type` is a page-defect to-do list rather than a statistic.
- `scout --json` (and deprecated `audit --json`) → `{papers_skipped_no_doi, papers_intentional_no_doi, total_paper_pages, papers, cross_wiki_citations, edge_summary, recommended_additions, shared_citation_anchors, anchor_groups, s2_missing, duplicate_dois}`; `categories`/`category_breadth`/`count_normalized` are per-entry fields nested inside `recommended_additions`/`shared_citation_anchors`, not top-level. `duplicate_dois` lists ambiguous DOI-to-page assignments whose graph edges are skipped. `scout web` uses a separate versioned request/receipt/manifest contract documented in [`prompts/scout-web.md`](./prompts/scout-web.md), never this citation snapshot.
- `scout web list --json` → `{schema_version, runs[]}`; each run carries `{run_id, state, query, created_at, source_count, discovery_method, next_command, error}`. States are `requested`, `recorded`, `invalid`. `scout web show <run-id> --json` re-emits the versioned request plus `{state, request_path, cached_result}` so a different chat-agent host can resume it exactly; `cached_result` is `null` while requested and otherwise carries the exact cached `{receipt, manifest, receipt_path, manifest_path}`. Receipts and requests are `schema_version: 3`; a version-2 artifact (no `discovery_method`) is hard-rejected rather than migrated. No formal report is generated.

### Exit-code contract

| Code | Meaning |
|---|---|
| **0** | Success. Zero results still `0` for read-only tools. |
| **1** | User-input error / no-result-where-expected. |
| **2** | Environment error: missing index, provider unreachable, disk unreadable. |
| **3** | Reserved: internal bug / uncaught exception. |

**Page-gate exception.** For the three page gates — `check-grounding`, `grade synthesis`, `check-coverage` — **1 means "the gate found something"** (the review-triggering outcome), so bad input (missing page) is **2**, not 1. All three agree; don't "align" one to the table above (`tests/test_exit_codes.py::test_page_gates_agree_on_missing_path` pins it).

### Releasing the framework

Bumping the version, promoting `CHANGELOG.md`'s `[Unreleased]` section, and tagging → [`CONTRIBUTING.md` § Releasing](./CONTRIBUTING.md#releasing). Two things worth knowing before touching anything: the version lives **only** in `researchwiki/__init__.py` (`pyproject.toml` resolves it from there — adding a literal back fails `tests/test_version.py`), and removing a `--json` key is a breaking change because agents parse those contracts.

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
- **Supplementary files** live at `papers/{stem}.supp/{filename}`. `researchwiki attach <page> <file>` puts one there and declares it in the page's YAML (a source under `inbox/` is moved, not copied); `agent ingest --supplementary` does the same at ingest time. `lint` reports both directions of drift (`supplementary_missing_on_disk`, `supplementary_orphaned_files`).
- `pdf_path` is an Obsidian wikilink to the source PDF (`"[[{stem}.pdf]]"`) — click-to-open in the vault. The real file always lives at `papers/{stem}.pdf`; `db rebuild` derives that path from the stem.
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
