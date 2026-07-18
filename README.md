# Research Wiki

A markdown-first wiki for research papers that compounds as you add to it. **You don't operate the CLI — your LLM does.** You converse with Claude Code (or any LLM with file + shell access), and it authors each wiki page from the source PDF, grades claims against it, surfaces synthesis opportunities, and refines neighboring pages when related work arrives. Your job: drop PDFs in `inbox/`, ask cross-paper questions, review the LLM's edits. Local, Obsidian-compatible, Git-trackable.

This README is for **you**, the human. [CLAUDE.md](./CLAUDE.md) is the contract your LLM reads — you don't need to. [WORKFLOW.md](./WORKFLOW.md) has an end-to-end walkthrough with real agent output.

## How it works

The usage pattern is conversational. You say what you want; the LLM picks the right command, runs it, and reports back grounded in the wiki:

| You say | LLM does (under the hood) |
| --- | --- |
| "I dropped two PDFs in inbox/. Ingest both." | Runs `researchwiki agent ingest` for each; reports stems, gleaning, memory-evolution proposals. |
| "What does the wiki have on sequence-to-expression models?" | `search` → reads pages → answers with `[[wikilinks]]` (Rule 2: never improvise from priors). |
| "We have 7 pangenome-graph papers — should we synthesize?" | `candidates synthesis` → scaffolds via `synthesize` → authors the body grounded in cited papers. |
| "File this design idea back as an idea page." | Scaffolds `wiki/ideas/{slug}.md` (**Background → Opportunities → Plans → Caveats**), `status: open`. Manual trigger only. |
| "Is this paper the most up-to-date in its field?" | `neighbors <stem> --mode all --needs-ingest` — references/citations/recommendations annotated with in-wiki status. |
| "Reconcile got the year wrong on a page." | Follows the recovery procedure: strip back-links, delete page, re-ingest with `--doi`, `db rebuild` + `reindex`. |

You can run any command directly, but most users won't. The CLI is a contract between the LLM and the framework; the LLM is the operator.

## Getting started

Three paths to a running wiki — pick one:

- **Talk to your LLM.** Clone, drop PDFs in `inbox/`, open the directory in Claude Code (or any agent that reads `CLAUDE.md` / `AGENTS.md`), say *"initialize this for me — I have N PDFs in inbox/."* The agent walks the steps in [`prompts/init.md`](./prompts/init.md).
- **Run the wizard.** `researchwiki init` — interactive terminal wizard (provider → categories → dashboard → confirm). Prompts you for each decision and writes `.env` / `config/models.yaml`.
- **Do it manually.** The sub-sections below walk the same steps as a reference.

Taxonomy comes from *your* papers: `researchwiki bootstrap-categories` derives categories from what's in `inbox/`, **not** the biology+ML defaults listed below.

### Prerequisites

- **Python 3.10+** and **git**.
- An **LLM provider** (four options, mixable per role — see below). The digest path (`researchwiki ingest`) and offline tools (`status`, `lint`, `search`) need none.
- ~2 GB of disk for **torch + sentence-transformers + the BAAI/bge-small-en-v1.5 bi-encoder** (the claim grader, cached under `~/.cache/huggingface/` on first `reindex`). Bundled by default; skip loading at runtime with `--no-semantic` on `agent ingest` / `reindex`.

### Install

```bash
git clone git@github.com:johapark/research-wiki.git
cd research-wiki
pip install -e .                          # core + claim grader + Anthropic SDK
```

This exposes `researchwiki` on your `PATH`. Everything most users need — pypdfium2, tantivy, the bi-encoder grader, and the Anthropic SDK — is in the default install. Extras: **`[mcp]`** (MCP server for Claude Desktop), **`[dev]`** (pytest). To skip the ~2 GB grader weights at runtime, pass `--no-semantic` to `agent ingest` (draft selection falls back to BM25; pages land in the sandbox for manual promotion) or use the digest path (no grading).

### Providers

Set credentials/routing via a gitignored **`.env`** at the project root (loaded automatically every invocation — no `source` needed) or inline shell exports (which take precedence). Put your **API key** in `.env`; keep **`RW_LLM_BASE_URL`** as a session export, since it changes when you swap backends.

```bash
# .env
OPENAI_API_KEY="sk-..."                # OpenAI cloud (default, Bearer)
# ANTHROPIC_API_KEY="sk-ant-xxxx"      # Anthropic cloud
# RW_LLM_PROVIDER="chat-relay"         # chat-relay (no API key)
```

| Provider | Setup | Cost |
| --- | --- | --- |
| **OpenAI** (default) | Set `OPENAI_API_KEY`. The default config (and the zero-config fallback) already routes to `gpt-5.6-luna` + `gpt-5.4-mini` — nothing to copy. | ~$0.05/paper |
| **Anthropic** | `cp config/models.anthropic.yaml config/models.yaml`, set `ANTHROPIC_API_KEY`. Routes to Sonnet 4.6 + Haiku 4.5. | ~$0.10/paper |
| **Other OpenAI-compatible** (Gemini, Groq, OpenRouter, …) | `cp config/models.gemini.yaml config/models.yaml` (Gemini — ready-made) or `config/models.openai-compatible.yaml` (generic template), set `OPENAI_API_KEY` + `RW_LLM_BASE_URL`. | provider-dependent |
| **Local LLM** (LM Studio / vLLM / llama.cpp / ollama) | Any OpenAI-compatible server. `provider: lmstudio` on a role; base URL defaults to `http://localhost:1234/v1` (override with `RW_LLM_BASE_URL`). | ~free after download |
| **Chat-relay** (no API key/server) | `export RW_LLM_PROVIDER=chat-relay`; the chat agent in your terminal fills each prompt in `.llm-relay/pending/`. | chat subscription |

### Model config

Which model runs each role is read from `config/models.yaml` — **gitignored** (your local, mutable copy). The repo ships committed templates; copy one:

| Template | Routes to |
|---|---|
| `config/models.chatgpt.yaml` | **Default (recommended)** — GPT-5-class **gpt-5.6-luna** for the quality-sensitive roles (author/critic/judge/proposer), cheaper **gpt-5.4-mini** for the deterministic classifier/extractor; ~$0.05/paper. Zero-config fallback when no `config/models.yaml` is present. Set `OPENAI_API_KEY` to an OpenAI key and go. |
| `config/models.anthropic.yaml` | Sonnet 4.6 + Haiku 4.5. Highest fidelity; ~$0.10/paper. |
| `config/models.gemini.yaml` | **Recommended free API** — Google Gemini via its OpenAI-compatible endpoint. **Gemini 3.5 Flash** for author/critic/judge, **Gemini 3.1 Flash-Lite** for classifier/proposer/extractor. Best-grading free option in our dogfooding (see below); set `OPENAI_API_KEY` to a Gemini key and go. |
| `config/models.lmstudio.yaml` | **Recommended local** — pure-local, every role on one LM Studio model. Runs **Qwen3.6-35B-A3B** in our setup (see [Local LLMs](#local-llms-lm-studio--vllm--llamacpp--ollama)): no API key, nothing leaves the machine, ~free after the one-time download. |
| `config/models.openai-compatible.yaml` | Generic template for any OpenAI-compatible cloud (OpenAI, Groq, Together, Fireworks, DeepInfra, OpenRouter, Cerebras, Anyscale, …); worked examples in the file's comments. |
| `config/models.glm.yaml` | GLM-4.7-Flash via z.ai's Anthropic-compatible endpoint (free tier). |

```bash
cp config/models.chatgpt.yaml config/models.yaml     # the default; or the others
```

**Which one?** The **default is `config/models.chatgpt.yaml`** — OpenAI's GPT-5-class models across the pipeline, with the cheaper mini on the mechanical roles to hold cost near ~$0.05/paper; in benchmarking it captured every critical headline claim with verbatim comparator figures. For a **zero-cost cloud** start, use `config/models.gemini.yaml` — in our ingest history Gemini 3.5 Flash produced the highest-grading drafts of any free provider (mean claim-fidelity ≈ 0.80, edging Qwen's ≈ 0.78 and Solar's ≈ 0.75). Two caveats to weigh: the free tier is rate-limited (~5 requests/min — the config already serializes drafting to stay under it) and **Google may train on free-tier prompts/responses**, so for unpublished or confidential PDFs stay on OpenAI/Anthropic or the local path. For a **fully private, no-key** setup, use `config/models.lmstudio.yaml` with Qwen3.6-35B (below) — it trades a little fidelity for keeping every paper on your own hardware.

With no `config/models.yaml` present, the loader falls back to a hardcoded table mirroring `models.chatgpt.yaml` — so the OpenAI default works with no copy at all (just set `OPENAI_API_KEY`). Copy a template only when you want to *change* something; delete the file to reset.

**Switch backends without copying — `RW_MODELS_CONFIG`.** Instead of copying a template over your active `config/models.yaml`, point the loader at any config file via the `RW_MODELS_CONFIG` env var. A bare filename resolves under `config/`; an absolute/path-separated value is used verbatim. This is the clean way to A/B a backend or run a one-off without disturbing your default:

```bash
RW_MODELS_CONFIG=models.glm.yaml researchwiki agent ingest inbox/paper.pdf   # one-off
# or in .env to make it your persistent default backend
```

Precedence: `RW_MODELS_CONFIG` → `config/models.yaml` → hardcoded defaults. `researchwiki status` prints the active config (with a `[RW_MODELS_CONFIG]` marker when the env var is in effect), so "which config am I using?" is always answerable. Unlike `RW_LLM_PROVIDER` (which forces one provider across every phase and defeats per-role mixing), `RW_MODELS_CONFIG` selects a whole file that keeps its own per-role mixing.

### Point your LLM at the directory

Open the clone in **Claude Code** (primary tested harness; Cursor / Aider / Codex should work but get less testing). The LLM reads **CLAUDE.md** on first prompt — the Four Rules, naming convention, ingest workflow, recovery procedure. No server or plugin: it drives the wiki through the `researchwiki` CLI and file reads. The three content dirs (`inbox/`, `papers/`, `wiki/`) ship as `.gitkeep` shells, so a clone is immediately functional; their contents stay gitignored.

### Categories

No fixed taxonomy — you propose categories from your actual papers. Two ways:

1. **`researchwiki bootstrap-categories`** — drop 3+ PDFs in `inbox/`, run it; an LLM proposes a 2-to-N taxonomy grounded in them (size scales with corpus). Print-only; `--apply` atomically rewrites the CLAUDE.md table + `researchwiki/categories.py` and `mkdir`s each `wiki/<slug>/`.
2. **Edit by hand** — CLAUDE.md's Categories table and `categories.py`'s `VALID_CATEGORIES` set (both must agree).

Always keep **`other`** — the abstention bucket the classifier falls back to; `status` flags it past 10 papers and `suggest-splits` proposes promotions.

**Shipped defaults** (biology + ML starting point — replace via bootstrap):

| Category | Includes |
| --- | --- |
| `genomics` | Genomes, GWAS, biostatistics, NGS methods |
| `compbio` | AI/ML applied to biology — structure prediction, sequence foundation models, multi-omics |
| `cgt` | Cell & Gene Therapy — CRISPR-Cas, AAV, ASO, prime/base editing |
| `ai` | Pure CS / AI / ML — agent frameworks, LLM tooling, ML methodology |
| `synthesis` | Cross-paper analytical pages |
| `references` | Non-peer-reviewed: guidance, protocols, whitepapers, textbooks |
| `other` | Cross-cutting + abstention bucket |

**Tip**: pick **durable** cuts — methods (`prime-editing`, `transformer-models`) or fields (`immunology`, `rna-biology`), not transient topic-surface slugs (`alphafold-class-papers`) that age when the vocabulary moves.

### Your first ingest (~5 min, ~$0.05)

Drop a PDF in `inbox/` and say *"Ingest the paper I just dropped in `inbox/`."* The LLM runs `researchwiki agent ingest`. ~3–5 min later: metadata reconciled (the extractor cross-checks Semantic Scholar), a draft written and graded against the PDF, promoted to `wiki/{category}/{stem}.md` with back-links added, and any synthesis-evolution proposals surfaced for review. Then ask *"What did we just learn?"* for a summary.

**Backlog?** Say *"ingest everything in inbox/"* — the LLM loops over each file (~$0.05/paper on the default OpenAI config). **If something breaks mid-ingest**, tell the LLM; it follows the recovery procedure in CLAUDE.md (don't hand-patch YAML). Then the workflow is conversational: batch ingest, cross-paper Q&A, `neighbors` discovery, synthesis detection, idea pages, page fixes.

## CLI reference (what your LLM runs)

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

`researchwiki status` on a populated wiki prints per-category counts, cross-link density, orphans, and inbox backlog; on an empty wiki it prints `Pages: 0` cleanly.

### All commands

| Command | Purpose |
|---|---|
| `agent ingest <pdf> [--supplementary <f>...]` | Full pipeline: reconcile → extract → crosslink → parallel-author → grade → critic/evolve/debug → promote → propose evolutions. Provider API key required (`OPENAI_API_KEY` by default). |
| `ingest <pdf>... [--category] [--supplementary]` | Digest-only (no LLM authoring): DOI, S2 metadata, stem, crosslinks, anchoring → `.ingest/{stem}-digest.md`. Author the page yourself. |
| `attach <category/stem> <file>` | Attach a supplementary file to an existing page; copies into `papers/{stem}.supp/`, updates YAML. |
| `neighbors <doi-or-stem>` | S2 citation-graph neighbors. `--mode references\|citations\|recommendations\|all`, `--year`, `--needs-ingest`. Structured fields only. |
| `evolve <category/stem>` | Neighboring synthesis pages to edit in light of a paper → proposals in `.ingest/{stem}-evolution-proposals/`. |
| `backfill <keywords\|doi>` | One-shot: populate the named field on paper pages predating it (keywords via LLM; doi via Semantic Scholar → Crossref). |
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
| `check-grounding` | Lint a markdown file: flag claim-bearing units lacking a `[[wikilink]]` or `claim_id`. |
| `claim-overlap <stem> [--sim N] [--top N] [--dry-run] [--json]` | Proactively cross-link a newly-ingested paper: finds existing papers with near-paraphrase claims, LLM-judges each as a real relationship vs coincidence, and auto-adds reciprocal Related-Papers `[[wikilinks]]` for confirmed matches. Run after `db rebuild`. |
| `db papers [--year/--category/--page-type/--no-doi/--venue/--author/--status] [--count] [--json]` | Structured lookups over the frontmatter mirror — counts/filters ("cgt papers from 2024", "papers missing a DOI") without re-reading markdown. `db query "SELECT…"` for ad-hoc read-only SQL. |
| `insights [--days N] [--json]` | Analytics over the ingest telemetry log: draft quality + cost by model, section difficulty, token spend by role, draft decisions. Read-only, no LLM. |

Every operation appends a parseable entry to `wiki/log.md` (inside `wiki/` so an Obsidian vault opened there can browse it).

## Navigating the wiki

Directories represent *page types*, each with its own contract and trigger:

```
wiki/
├── {category}/    ← paper pages (auto-generated by `agent ingest`)
├── synthesis/     ← cross-paper field maps (triggered by a query)
├── ideas/         ← forward-looking design proposals (manual only)
├── references/    ← regulatory guidance, protocols, whitepapers (manual)
├── index.md, log.md, pdfs-failed-parsing.md   ← bookkeeping
```

- **`{category}/` — paper pages.** One per PDF, `{author}-{year}-{first-5-title-words}.md`. Strict contract (Summary, Key Contributions, Methodology, Results, Limitations, Related Papers); every claim source-supported (Rule 1). Q&A starts here.
- **`synthesis/` — cross-paper field maps (retrospective).** Trajectories, status snapshots, method comparisons. Triggered when a non-trivial cross-paper answer is worth keeping — the primary way the wiki compounds. Lint flags stale ones; `candidates synthesis` finds uncovered clusters.
- **`ideas/` — design proposals (manual).** Propose **what could be built** (vs synthesis's what *exists*). Structure: **Background** (source-supported motivation) → **Opportunities** (the design, from wiki-grounded principles) → **Plans** (staged, with checkpoints) → **Caveats** (failure modes). `status:` lifecycle `open → scoping → validated | superseded | abandoned`. No auto-generation — ask explicitly.
- **`references/` — non-peer-reviewed.** Guidance (FDA/EMA/ICH), protocols, whitepapers, textbooks. Manual (no S2 metadata). Cross-links describe **methodological alignment**, not citation.
- **Bookkeeping.** `index.md` (curated catalog), `log.md` (auto-appended history), `pdfs-failed-parsing.md` (extraction-failure ledger). All inside `wiki/` so `[[wikilinks]]` resolve in Obsidian.
- **`papers/{stem}.pdf` + `papers/{stem}.supp/`.** Canonical PDFs + supplementary attachments (listed under `supplementary:` YAML). The LLM `Read`s these on demand (Rule 3).

See [CLAUDE.md Page Types](./CLAUDE.md) for the full contracts.

## Chat-relay (subscription users — no API key)

If your only model access is a **chat subscription** (Claude.ai Pro, ChatGPT Plus, Cursor Pro), the framework still runs end-to-end. The chat-relay provider delegates each LLM call to whatever chat agent is already in your terminal via a filesystem protocol — no API key, server, or per-paper cost.

**How it works.** `agent ingest` emits a prompt at `.llm-relay/pending/{op_id}.prompt.json` and blocks; the chat agent reads it, writes `.llm-relay/completed/{op_id}.response.json`; the CLI moves on. One ingest is 5–8 handoffs — a few minutes if the agent watches for prompts. Protocol spec: [`prompts/chat-relay.md`](./prompts/chat-relay.md).

```bash
export RW_LLM_PROVIDER=chat-relay      # or add to .env
researchwiki agent ingest inbox/some-paper.pdf
```

Then tell your chat agent *"watch `.llm-relay/pending/` and respond to each prompt as it appears."* Schema validation + retry-with-feedback is built in (up to 3 attempts), so you don't babysit format drift.

**Caveats:**
- **Wall clock is bounded by your attention** — each phase blocks on the agent; times out at 10 min/phase if it walks away.
- **Cost dashboards show $0** — tokens aren't measurable through the relay.
- **One ingest at a time** — parallel ingests queue to the same agent, which responds serially.
- **Cache reuse on re-runs** — `op_id = sha1(phase|prompt)[:12]`, so a crash mid-ingest reuses completed phases. `RW_RELAY_FRESH=1` forces re-prompting.

### Per-role mixing

`RW_LLM_PROVIDER` is a **global override** — it forces *every* phase to one provider and **silently defeats per-role mixing** whenever set (**including in `.env`**). To mix (e.g. chat-relay `author`, local everything else): **unset `RW_LLM_PROVIDER`**, then set `provider:` per role in `config/models.yaml`:

```yaml
roles:
  author:     {provider: chat-relay, model: claude-via-relay,    temperature: 0.5, max_tokens: 16000}
  critic:     {provider: lmstudio,   model: qwen3.6-35b-a3b-mlx, temperature: 0.3, max_tokens: 12000}
  judge:      {provider: lmstudio,   model: qwen3.6-35b-a3b-mlx, temperature: 0.2, max_tokens: 12000}
  classifier: {provider: lmstudio,   model: qwen3.6-35b-a3b-mlx, temperature: 0.1, max_tokens: 6000}
  proposer:   {provider: lmstudio,   model: qwen3.6-35b-a3b-mlx, temperature: 0.3, max_tokens: 6000}
  extractor:  {provider: lmstudio,   model: qwen3.6-35b-a3b-mlx, temperature: 0.0, max_tokens: 8000}
```

`author`/`evolve`/`debug` follow the `author` role. For this mix, **`RW_LLM_BASE_URL` must be set** (e.g. `http://localhost:1234/v1`) so local roles pass the readiness gate — **even on the default port** — and the relay needs an active servicer for the `author` phase.

## Local LLMs (LM Studio / vLLM / llama.cpp / ollama)

Any OpenAI-compatible server works alongside or instead of Anthropic. For a **fully local** setup, the model we dogfood is **Qwen3.6-35B-A3B** (a 35B-param MoE, ~3B active — runs on a 32 GB Apple-silicon laptop via LM Studio's MLX build): across our ingest history it authored drafts at mean claim-fidelity ≈ 0.78, within a hair of Gemini 3.5 Flash (≈ 0.80) and above Solar (≈ 0.75) — good enough to keep **every** role local, not just the cheap ones. `config/models.lmstudio.yaml` points all six roles at one such model. If you'd rather mix, a common split is to keep **author** on Anthropic for peak fidelity and route **classifier** / **proposer** / **reconcile** to a smaller local model to drop marginal cost toward zero.

**Start a server** (LM Studio is simplest — download a model, click **Start Server**; default `http://localhost:1234/v1`):

```bash
curl http://localhost:1234/v1/models | jq '.data[].id'   # confirm it's up
```

vLLM: `vllm serve <model> --port 1234`. llama.cpp: `llama-server -m <gguf> --port 1234`. ollama: `http://localhost:11434/v1`. Real OpenAI: `RW_LLM_BASE_URL=https://api.openai.com/v1` + `OPENAI_API_KEY`.

**Route a role** in `config/models.yaml`:

```yaml
roles:
  author:     {provider: anthropic, model: claude-sonnet-4-6,          temperature: 0.5, max_tokens: 2500}
  classifier: {provider: lmstudio,  model: meta-llama-3.1-8b-instruct, temperature: 0.1, max_tokens: 200}
  proposer:   {provider: lmstudio,  model: meta-llama-3.1-8b-instruct, temperature: 0.3, max_tokens: 200}
```

**Sizing:** a **7–8B** model handles the short roles well — `classifier`, `short_name`/`keywords`, `reconcile` (verify against S2) — but writes shallow `author` drafts and weak `judge` verdicts, so on small hardware keep those two on Anthropic and route the rest local. A **~30B+ MoE like Qwen3.6-35B-A3B** closes that gap (near-cloud author fidelity in our runs) while still fitting a 32 GB laptop, which is why it's our recommended all-roles-local model; dense 70B+ raises the ceiling further but needs 40+ GB VRAM.

**Caveats:** no prompt caching (author is ~free on local anyway); token counts may be approximate/zero (dashboard shows $0 — accurate); pure-local is supported (readiness checks are provider-aware — leave `ANTHROPIC_API_KEY` unset).

## Optional Dataview dashboard

`wiki/index.md` is the curated catalog, but for **structured cuts** — recent additions, papers by year, citation hubs, orphans, syntheses by member count — install [Dataview](https://blacksmithgu.github.io/obsidian-dataview/) and use `wiki/views.md`. Vault-only (renders as code blocks on GitHub). The cold-start setup already scaffolds `wiki/views.md` (recent papers / synthesis / ideas); ask your LLM for more cuts.

**Installing Dataview** (a community plugin — enable once per vault):

1. Open `wiki/` as a vault (**Open folder as vault**).
2. **Settings → Community plugins** → **Turn on community plugins** (leaves Restricted Mode).
3. **Browse** → search **Dataview** → **Install**, then toggle it on.
4. Open `wiki/views.md` in **Reading view** — the `dataview` blocks render as live tables.

## What's tracked

Tracked: `researchwiki/` (package + CLI), `prompts/`, `config/` **templates** (your `config/models.yaml` is gitignored), `tests/`, docs (`CLAUDE.md`, `README.md`, `WORKFLOW.md`), `pyproject.toml`, `.gitignore`, and the `inbox/` `papers/` `wiki/` **shells** (`.gitkeep` only).

Gitignored (your library stays local): `papers/*.pdf` + `.supp/`, `wiki/{category}/*.md`, `inbox/*.pdf`, the `wiki/` bookkeeping files (`index.md`, `log.md`, `pdfs-failed-parsing.md`), and repo-root caches (`.ingest/`, `.s2-cache/`, `.crossref-cache/`, `.web-cache/`, `.tantivy-index/`, `.semantic-cache/`, `.grade-cache/`, `.agent-output/`, `.suggest-splits-stamp`).

## Development & validation

Primary validation is **dogfooding** against a private research corpus spanning compbio / cgt / genomics / ai categories. On top: a **hermetic pytest suite** (500+ tests, no network/LLM) covering the load-bearing operators — stem derivation, promotion gate, numeric-drift detection, per-operator fitness/selection, DEBUG classifier — plus `tests/grade-fixtures/` (curated PDFs + `run-eval.py` grader regression).

```bash
# Cold-install verification (~30s)
git clone git@github.com:johapark/research-wiki.git && cd research-wiki
python -m venv .venv && source .venv/bin/activate
pip install -e .
researchwiki --version           # → researchwiki 0.1.0
researchwiki status              # works on empty wiki, no API key

# Unit tests (hermetic)
pip install -e '.[dev]' && pytest    # 500+ tests, a few seconds

# Grader regression
cd tests/grade-fixtures && python run-eval.py
```

If cold-install fails on a fresh clone, that's a packaging bug — please file an issue.

## Design principles

- **Two-tier**: raw PDF (immutable) → single LLM-authored wiki page. No separate "summary" layer.
- **No prose from the web**: only Semantic Scholar structural metadata + verbatim abstract + draft-only TLDR (Rule 1).
- **Git-trackable framework, gitignored content**: share the workflow, keep your library.
- **Obsidian-native**: `[[wikilinks]]` + plain markdown, no lock-in.
- **Query → File loop**: non-trivial cross-paper answers are persisted back as synthesis pages. This is how the wiki compounds.

## Inspiration

Inspired by [Andrej Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/1dd0294ef9567971c1e4348a90d69285), simplified to two tiers.
