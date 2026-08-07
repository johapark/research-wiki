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
| "File this design idea back as an idea page." | Scaffolds `wiki/ideas/{slug}.md` (**Verdict → Background → Opportunities → Plans → Caveats**), `status: open`. Manual trigger only. |
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
- An **LLM provider** (four options, mixable per role — see [Provider setup in depth](./WORKFLOW.md#provider-setup-in-depth)). The digest path (`researchwiki ingest`) and offline tools (`status`, `lint`, `search`) need none.
- ~2 GB of disk for **torch + sentence-transformers + the BAAI/bge-small-en-v1.5 bi-encoder** (the claim grader, cached under `~/.cache/huggingface/` on first `reindex`). Bundled by default; skip loading at runtime with `--no-semantic` on `agent ingest` / `reindex`.

### Install

```bash
git clone git@github.com:johapark/research-wiki.git
cd research-wiki
python3 -m venv ~/.venvs/research-wiki    # outside the repo — see below
~/.venvs/research-wiki/bin/pip install -e .   # core + claim grader + Anthropic SDK
```

Activate the venv (or call `~/.venvs/research-wiki/bin/researchwiki`) to get the CLI — a venv is not on `PATH` otherwise. Everything most users need — pypdfium2, tantivy, the bi-encoder grader, and the Anthropic SDK — is in the default install. Extras: **`[mcp]`** (MCP server for Claude Desktop), **`[dev]`** (pytest). To skip the ~2 GB grader weights at runtime, pass `--no-semantic` to `agent ingest` (draft selection falls back to BM25; pages land in the sandbox for manual promotion) or use the digest path (no grading).

**Why a venv, and why not `.venv/` in the repo.** Bare `pip install -e .` lands in whatever interpreter is active — often a shared conda `base` — and this project's ~2 GB of pins are platform-specific (on x86_64 macOS torch caps at 2.2.2, forcing `transformers<5` and `numpy<2`). And a venv is ~34,000 files, which matters most if you sync your library — see just below.

### Syncing your library (and reading it on your phone)

Two directories are worth syncing: **`papers/`** and **`wiki/`**. They are also exactly the two the repo does *not* carry — both are gitignored, since the framework is public and your corpus isn't. So git and your sync service (iCloud, Dropbox, Drive, Syncthing) have disjoint jobs, and the recommended layout just lets each do its own.

**Keep the checkout outside the synced folder, and symlink the content dirs in.**

```bash
SYNC="$HOME/<your-synced-folder>/research-wiki"   # holds wiki/ and papers/
git clone git@github.com:johapark/research-wiki.git ~/src/research-wiki
cd ~/src/research-wiki
for d in wiki papers inbox; do rm -rf "$d"; ln -sfn "$SYNC/$d" "$d"; done
```

Your sync daemon then sees markdown and PDFs only — no virtualenv, no `.git/` rewritten on every command, no SQLite or search index. Keeping the checkout *inside* the synced folder instead makes the venv alone ~96% of the sync load, which starves the daemon into corrupting `.git/index` and spawning `<name> 2.md` conflict copies.

**The payoff: `$SYNC` becomes purely a vault, so the Obsidian mobile app can open it** — your whole wiki readable on a phone or tablet, `[[wikilinks]]` and all, with none of the checkout in the way. Keep `wiki/` and `papers/` as siblings there so `pdf_path:` links (`[[{stem}.pdf]]`) resolve and PDFs open on tap.

Run `researchwiki init --scaffold-only` afterwards to create the page-type dirs inside `$SYNC`, and relocate the caches to `~/.cache/research-wiki`. No git configuration is needed — these three dirs are gitignored in full, so git doesn't care whether they're directories or symlinks. The copy-paste block, the no-symlink fallback, and the sync-specific failure modes (including why a stale sync looks exactly like uncommitted work) are in [`prompts/migration-backfill.md`](./prompts/migration-backfill.md#keep-the-checkout-out-of-the-synced-folder).

**One thing does not travel with the checkout: the state DB.** It lives under `~/.local/share/researchwiki/repos/<name>-<hash>/`, keyed by the checkout's absolute path — so moving the checkout opens a fresh, empty database. `db rebuild` repopulates claims from your markdown and everything *looks* fine, while claim grades and ingest telemetry quietly aren't there. Set **`RESEARCHWIKI_DB_PATH`** in `.env` to pin the DB to one file and opt out of path-keying; if you are already stranded, the old DB is still on disk and recoverable. Details in [`WORKFLOW.md` → Pinning the state DB](./WORKFLOW.md#pinning-the-state-db-researchwiki_db_path).

**On a second computer**, let the sync service carry `wiki/` and `papers/` and re-derive the rest locally — `researchwiki db rebuild && researchwiki reindex && researchwiki grade regression --missing-only` (no API calls). Everything that counts as knowledge converges that way, with no database copying. Cost/quality telemetry is the deliberate exception: it stays on whichever machine ran the ingest, since it records what that machine actually did. Never put the state DB itself on the sync service — concurrent SQLite writes through a sync daemon corrupt it.

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
| **OpenAI** (default) | Set `OPENAI_API_KEY`. The default config (and the zero-config fallback) already routes every role to `gpt-5.6-luna` — nothing to copy. | ~$0.01/paper |
| **Anthropic** | `cp config/models.anthropic.yaml config/models.yaml`, set `ANTHROPIC_API_KEY`. Routes to Sonnet 4.6 + Haiku 4.5. | ~$0.10/paper |
| **Other OpenAI-compatible** (Gemini, Groq, OpenRouter, …) | `cp config/models.gemini.yaml config/models.yaml` (Gemini — ready-made) or `config/models.openai-compatible.yaml` (generic template), set `OPENAI_API_KEY` + `RW_LLM_BASE_URL`. | provider-dependent |
| **Local LLM** (LM Studio / vLLM / llama.cpp / ollama) | Any OpenAI-compatible server. `provider: lmstudio` on a role; base URL defaults to `http://localhost:1234/v1` (override with `RW_LLM_BASE_URL`). [Details](./WORKFLOW.md#local-llms-lm-studio--vllm--llamacpp--ollama). | ~free after download |
| **Chat-relay** (no API key/server) | `export RW_LLM_PROVIDER=chat-relay`; the chat agent in your terminal fills each prompt in `.llm-relay/pending/`. [Details](./WORKFLOW.md#chat-relay-subscription-users--no-api-key). | chat subscription |

### Model config

Which model runs each role is read from `config/models.yaml` — **gitignored** (your local, mutable copy). The repo ships committed templates; copy one:

| Template | Routes to |
|---|---|
| `config/models.chatgpt.yaml` | **Default (recommended)** — GPT-5-class **gpt-5.6-luna** across every role; ~$0.01/paper measured over 13 ingests. Zero-config fallback when no `config/models.yaml` is present. Set `OPENAI_API_KEY` to an OpenAI key and go. |
| `config/models.anthropic.yaml` | Sonnet 4.6 + Haiku 4.5. Highest fidelity; ~$0.10/paper. |
| `config/models.gemini.yaml` | **Recommended free API** — Google Gemini via its OpenAI-compatible endpoint. **Gemini 3.5 Flash** for author/critic/judge, **Gemini 3.1 Flash-Lite** for classifier/proposer/extractor. Best-grading free option in our dogfooding (see below); set `OPENAI_API_KEY` to a Gemini key and go. |
| `config/models.lmstudio.yaml` | **Recommended local** — pure-local, every role on one LM Studio model. Runs **Qwen3.6-35B-A3B** in our setup (see [Local LLMs](./WORKFLOW.md#local-llms-lm-studio--vllm--llamacpp--ollama)): no API key, nothing leaves the machine, ~free after the one-time download. |
| `config/models.openai-compatible.yaml` | Generic template for any OpenAI-compatible cloud (OpenAI, Groq, Together, Fireworks, DeepInfra, OpenRouter, Cerebras, Anyscale, …); worked examples in the file's comments. |
| `config/models.glm.yaml` | GLM-4.7-Flash via z.ai's Anthropic-compatible endpoint (free tier). |

```bash
cp config/models.chatgpt.yaml config/models.yaml     # the default; or the others
```

**Which one?** The **default is `config/models.chatgpt.yaml`** — OpenAI's GPT-5-class models across the pipeline, holding cost near ~$0.01/paper (measured mean over 13 ingests: 25.8K input / 3.5K output tokens); in benchmarking it captured every critical headline claim with verbatim comparator figures. For a **zero-cost cloud** start, use `config/models.gemini.yaml` — in our ingest history Gemini 3.5 Flash produced the highest-grading drafts of any free provider (mean claim-fidelity ≈ 0.80, edging Qwen's ≈ 0.78 and Solar's ≈ 0.75). Two caveats to weigh: the free tier is rate-limited (~5 requests/min — the config already serializes drafting to stay under it) and **Google may train on free-tier prompts/responses**, so for unpublished or confidential PDFs stay on OpenAI/Anthropic or the local path. For a **fully private, no-key** setup, use `config/models.lmstudio.yaml` with Qwen3.6-35B ([details](./WORKFLOW.md#local-llms-lm-studio--vllm--llamacpp--ollama)) — it trades a little fidelity for keeping every paper on your own hardware.

With no `config/models.yaml` present, the loader falls back to a hardcoded table mirroring `models.chatgpt.yaml` — so the OpenAI default works with no copy at all (just set `OPENAI_API_KEY`). Copy a template only when you want to *change* something; delete the file to reset.

**Want to A/B backends, mix providers per role, run fully local, or use a chat subscription with no API key?** See [`WORKFLOW.md` → Provider setup in depth](./WORKFLOW.md#provider-setup-in-depth).

### Point your LLM at the directory

Open the clone in **Claude Code** (primary tested harness; Cursor / Aider / Codex should work but get less testing). The LLM reads **CLAUDE.md** on first prompt — the Four Rules, naming convention, ingest workflow, recovery procedure. No server or plugin: it drives the wiki through the `researchwiki` CLI and file reads. The three content dirs (`inbox/`, `papers/`, `wiki/`) are gitignored in full and so are absent from a fresh clone — `researchwiki init` creates them, or `researchwiki init --scaffold-only` if you just want the directories without the wizard. Until then every other command exits with a pointer to it.

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
| `other` | Cross-cutting + abstention bucket |

Plus the four **page-type dirs** every wiki gets regardless of domain — `synthesis`, `ideas`, `concepts`, `references`. They hold their own page types and are never content categories (the classifier can't target them).

**Tip**: pick **durable** cuts — methods (`prime-editing`, `transformer-models`) or fields (`immunology`, `rna-biology`), not transient topic-surface slugs (`alphafold-class-papers`) that age when the vocabulary moves.

### Your first ingest (~5 min, ~$0.05)

Drop a PDF in `inbox/` and say *"Ingest the paper I just dropped in `inbox/`."* The LLM runs `researchwiki agent ingest`. ~3–5 min later: metadata reconciled (the extractor cross-checks Semantic Scholar), a draft written and graded against the PDF, promoted to `wiki/{category}/{stem}.md` with back-links added, and any synthesis-evolution proposals surfaced for review. Then ask *"What did we just learn?"* for a summary.

**Backlog?** Say *"ingest everything in inbox/"* — the LLM loops over each file (~$0.01/paper on the default OpenAI config). **If something breaks mid-ingest**, tell the LLM; it follows the recovery procedure in CLAUDE.md (don't hand-patch YAML). Then the workflow is conversational: batch ingest, cross-paper Q&A, `neighbors` discovery, synthesis detection, idea pages, page fixes.

Most users won't run `researchwiki` commands directly — the LLM picks among them based on what you ask. For the full command table and a copy-paste-able walkthrough, see [`WORKFLOW.md` → CLI reference](./WORKFLOW.md#cli-reference).

## Navigating the wiki

Directories represent *page types*, each with its own contract and trigger:

```
wiki/
├── {category}/    ← paper pages (auto-generated by `agent ingest`)
│                    …and commentary pages, which live beside the paper they discuss
├── synthesis/     ← cross-paper field maps (triggered by a query)
├── concepts/      ← single-term hub notes bridging categories (scaffolded, review-gated)
├── ideas/         ← forward-looking design proposals (manual only)
├── references/    ← regulatory guidance, protocols, whitepapers (manual)
├── index.md, log.md                          ← bookkeeping
```

- **`{category}/` — paper pages.** One per PDF, `{author}-{year}-{first-5-title-words}.md`. Strict contract (Summary, Key Contributions, Methodology and Architecture, Results, Limitations, Related Papers); every claim source-supported (Rule 1). Q&A starts here. **Commentary pages** (Research Highlights, News & Views, editorials) also live here, typed `commentary` with a `primary_paper:` pointer — they produce no claims of their own, and you cite the paper they discuss rather than the commentary. `agent ingest` detects the shape and refuses to promote one as a paper.
- **`synthesis/` — cross-paper field maps (retrospective).** Trajectories, status snapshots, method comparisons. Triggered when a non-trivial cross-paper answer is worth keeping — the primary way the wiki compounds. Lint flags stale ones; `candidates synthesis` finds uncovered clusters.
- **`concepts/` — single-term hub notes.** A mini-synthesis around one recurring concept, tying every paper that instantiates the term into one bridge node — most valuable when it spans categories the citation graph and semantic search don't connect. `candidates concepts` surfaces terms at ≥3 papers; scaffolding one requires a **thesis** (one sentence on why it's a concept and not a glossary entry), which is what keeps the directory from filling with acronyms. Strictly grounded.
- **`ideas/` — design proposals (manual).** Propose **what could be built** (vs synthesis's what *exists*). Structure: **Verdict** (`strong`/`incremental`/`weak` + tl;dr, written last and placed first) → **Background** (source-supported motivation) → **Opportunities** (the design, from wiki-grounded principles) → **Plans** (staged, with checkpoints) → **Caveats** (failure modes). The one page type where model priors are allowed, and only in Opportunities/Plans, marked `*(model prior)*`. `status:` lifecycle `open → scoping → validated | superseded | abandoned`. No auto-generation — ask explicitly.
- **`references/` — non-peer-reviewed.** Guidance (FDA/EMA/ICH), protocols, whitepapers, textbooks. Manual (no S2 metadata). Cross-links describe **methodological alignment**, not citation.
- **Bookkeeping.** `index.md` (curated catalog), `log.md` (auto-appended history). Both inside `wiki/` so `[[wikilinks]]` resolve in Obsidian. A PDF that wouldn't text-extract is recorded on the page itself via YAML `pdf_extraction_note:` and listed by `researchwiki status` — there's no separate ledger to keep current.
- **`papers/{stem}.pdf` + `papers/{stem}.supp/`.** Canonical PDFs + supplementary attachments (listed under `supplementary:` YAML). The LLM `Read`s these on demand (Rule 3).

See [CLAUDE.md Page Types](./CLAUDE.md) for the full contracts.

## Optional Dataview dashboard

`wiki/index.md` is the curated catalog, but for **structured cuts** — recent additions, papers by year, citation hubs, orphans, syntheses by member count — install [Dataview](https://blacksmithgu.github.io/obsidian-dataview/) and use `wiki/views.md`. Vault-only (renders as code blocks on GitHub). The cold-start setup already scaffolds `wiki/views.md` (recent papers / synthesis / ideas); ask your LLM for more cuts.

**Installing Dataview** (a community plugin — enable once per vault):

1. Open `wiki/` as a vault (**Open folder as vault**).
2. **Settings → Community plugins** → **Turn on community plugins** (leaves Restricted Mode).
3. **Browse** → search **Dataview** → **Install**, then toggle it on.
4. Open `wiki/views.md` in **Reading view** — the `dataview` blocks render as live tables.

## What's tracked

Tracked: `researchwiki/` (package + CLI), `prompts/`, `config/` **templates** (your `config/models.yaml` is gitignored), `tests/`, docs (`CLAUDE.md`, `README.md`, `WORKFLOW.md`), `pyproject.toml`, `.gitignore`. Nothing under `inbox/`, `papers/`, `wiki/` — not even a placeholder; `researchwiki init` creates those dirs locally.

Gitignored (your library stays local): `papers/*.pdf` + `.supp/`, `wiki/{category}/*.md`, `inbox/*.pdf`, the `wiki/` bookkeeping files (`index.md`, `log.md`), and repo-root caches (`.ingest/`, `.s2-cache/`, `.crossref-cache/`, `.web-cache/`, `.tantivy-index/`, `.semantic-cache/`, `.grade-cache/`, `.agent-output/`, `.suggest-splits-stamp`).

## Development & validation

Contributing, running the test suite, and the conventions enforced by CI: see [`CONTRIBUTING.md`](./CONTRIBUTING.md). Credential/data-flow surface — notably that ingesting a paper sends its text to your configured LLM provider: see [`SECURITY.md`](./SECURITY.md).

## License

The framework — everything under `researchwiki/`, `tests/`, `prompts/`, `config/`, and the docs — is **MIT** ([`LICENSE`](./LICENSE)).

Two carve-outs, neither covered by that MIT grant:

- **`benchmark-fixtures/pdfs/`** — five published papers redistributed under **CC-BY-4.0**, each with per-paper attribution in [`benchmark-fixtures/LICENSES.md`](./benchmark-fixtures/LICENSES.md). If you fork or redistribute, keep that attribution intact.
- **Your own library** — `papers/`, `inbox/`, and `wiki/` are gitignored and never distributed. The PDFs you add stay under their publishers' terms; this project claims nothing over them.

## Design principles

- **Two-tier**: raw PDF (immutable) → single LLM-authored wiki page. No separate "summary" layer.
- **No prose from the web**: only Semantic Scholar structural metadata + verbatim abstract + draft-only TLDR (Rule 1).
- **Git-trackable framework, gitignored content**: share the workflow, keep your library.
- **Obsidian-native**: `[[wikilinks]]` + plain markdown, no lock-in.
- **Query → File loop**: non-trivial cross-paper answers are persisted back as synthesis pages. This is how the wiki compounds.

## Inspiration

Inspired by [Andrej Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/1dd0294ef9567971c1e4348a90d69285), simplified to two tiers.
