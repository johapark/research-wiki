# Research Wiki

A local, markdown-first research wiki that compounds as you add papers. Drop PDFs into `inbox/`, ask your chat agent to ingest or analyze them, and browse the resulting knowledge base in Obsidian.

The agent operates the CLI for you: it reads the source PDF, writes and grades the page, connects related work, and keeps indexes current. [CLAUDE.md](./CLAUDE.md) and its `AGENTS.md` symlink contain the agent contract; [WORKFLOW.md](./WORKFLOW.md) is the detailed human reference.

## Features

- **Grounded in local PDFs.** Wiki prose comes from your papers, not web summaries. Answers start from the wiki and return to the PDFs when more detail is needed.
- **Claim-level verification.** Claims are graded against their source and exposed through durable `[[paper#claim]]` anchors. Cross-paper pages (syntheses, concepts, and ideas) are also checked for complete citations and source fidelity.
- **Connected knowledge.** Citation-supported links, concept hubs, synthesis pages, claim relationships, and evolution proposals turn isolated summaries into a research map.
- **Agent-operated workflows.** The agent coordinates ingestion, synthesis, verification, linking, indexing, and recovery through the CLI.
- **Search and discovery.** BM25 and semantic search, citation/recommendation neighbors, gap detection, and an interactive graph help you find papers and missing literature.
- **Portable and local-first.** Your wiki is plain Markdown, your library is gitignored, and the content directories can sync independently of the framework.
- **Provider-flexible.** Use OpenAI, Anthropic, Gemini, another OpenAI-compatible service, a local model, or chat relay; roles can use different providers.

## How it works

You describe the outcome; the agent chooses the commands:

| You say | The agent does |
| --- | --- |
| “Ingest everything in `inbox/`.” | Runs a crash-safe batch ingest and reports every success or failure. |
| “What does the wiki have on sequence-to-expression models?” | Searches the wiki, reads relevant pages, and answers with `[[wikilinks]]`. |
| “Should these pangenome papers become a synthesis?” | Finds the cluster, scaffolds a synthesis, grounds it in claims, and runs both gates. |
| “File this design as an idea.” | Creates a reviewable idea page with grounded background and clearly marked model priors. |
| “Is this paper current?” | Compares its references, citations, and recommendations with the local library. |

You can run `researchwiki` directly, but the intended interface is conversation with an agent that has file and shell access.

## Quick start

### Requirements

- Python 3.10+
- A shell-enabled chat agent (Claude Code, Codex, etc.)
- An API key for a supported provider (optional but recommended; not required for local-model or chat-relay setups)

### Install

```bash
git clone https://github.com/johapark/research-wiki.git
cd research-wiki
python3 -m venv ~/.venvs/research-wiki
~/.venvs/research-wiki/bin/pip install -e .
source ~/.venvs/research-wiki/bin/activate
```

Keep the virtual environment outside the checkout, especially when syncing your library. The default install includes PDF extraction, Tantivy search, semantic grading, and the Anthropic SDK. Optional extras are `[mcp]` and `[dev]`.

Setup is clone-first: the framework reads `config/`, prompts, and agent instructions from the checkout. A standalone `pip install` from GitHub without retaining the clone is not a supported installation path.

### Initialize

Choose one guided setup path:

- Open the clone in Claude Code, Codex, or another compatible shell-enabled chat agent, then say: **“Initialize this research wiki.”**
- Run the interactive wizard: `researchwiki init`.

Both paths configure the provider and create the wiki directories and dashboard. For non-interactive setup, `researchwiki init --scaffold-only` creates the directory and dashboard scaffold without configuring a provider or asking questions. On an existing wiki, `researchwiki init --refresh-dashboard` adopts the current dashboard template and backs your copy up under `.ingest/`; nothing overwrites a dashboard unless you ask.

Then run the local, free readiness check:

```bash
researchwiki doctor
```

It checks paths, dependencies, provider configuration, the state DB, search state, and the semantic-model cache without contacting the provider or downloading anything. `researchwiki doctor --probe` is an explicit one-call connectivity test and may spend tokens.

No taxonomy is predefined, and you do not need to invent one before the first paper. New papers use `other/` until at least three PDFs are available; then `researchwiki bootstrap-categories` can propose categories from the corpus. As it grows, the agent suggests useful splits for review; nothing changes automatically.

### First ingest

Give the agent a PDF path and say **“Add this paper to my research wiki.”** The file can already be anywhere on disk; copying it into `inbox/` first is optional. The agent will create a canonical PDF in `papers/`, a grounded page in `wiki/{category}/`, reciprocal supported links, and updated indexes.

The direct CLI equivalent is:

```bash
researchwiki add /path/to/paper.pdf
```

Pass several paths—or ask the agent to ingest the whole inbox—to use the checkpointed batch workflow. If your papers already live in Zotero, Paperpile, Mendeley, or ReadCube, use the [library import workflow](#import-and-export) instead.

## Providers

Initialization configures the default provider, using the root `.env` for credentials and `config/models.yaml` when custom routing is needed. The root `.env` loads automatically; variables already exported by the parent shell take precedence.

Use a named profile when you want several provider setups in one checkout. The global `--env-file` option must come before the command, and the named file must exist:

```bash
cp .env.template .env.openai
chmod 600 .env.openai
researchwiki --env-file .env.openai init
researchwiki --env-file .env.openai status
```

A named profile replaces rather than merges with the root `.env`; use the same `--env-file` option for every command that should use it. Exported credentials may still supply its API key, but parent-shell routing variables are rejected. Put routing in the selected profile and do not `source` it as a shell script. Known providers reuse tracked `config/models.*.yaml` templates; custom OpenAI-compatible backends ask for an explicit writable config path.

| Provider | Minimal setup |
| --- | --- |
| **OpenAI** | Select OpenAI in `researchwiki init`; set `OPENAI_API_KEY`. |
| **Anthropic** | Select Anthropic in `researchwiki init`; set `ANTHROPIC_API_KEY`. |
| **Gemini** | Select Other OpenAI-compatible in `researchwiki init`; provide the Gemini endpoint, model IDs, and key. |
| **Other OpenAI-compatible** | Select it in `researchwiki init`; provide the exact models, endpoint, and API key. |
| **Local model** | Select Local LLM in `researchwiki init`; no API key is required. |
| **Chat relay** | Select Chat-relay in `researchwiki init`; no API key or server is required. |

API-based providers are called directly by each ingest subprocess. They need a
provider endpoint and usually an API key, but they require no interactive chat
agent: multi-paper ingest defaults to four concurrent workers, and each request
returns to the subprocess that issued it.

Chat-relay does not call a model API. It writes each model request under
`.llm-relay/pending/` and waits for an active Codex, Claude Code, or compatible
chat agent to write the matching response under `.llm-relay/completed/`. A
multi-paper chat-relay batch therefore defaults to one worker and forwards
requests from its own child processes to the parent terminal. Run it in the
foreground — backgrounding hides those handoffs. Passing `-w N` explicitly permits concurrent
relay requests, but the CLI creates only isolated ingest subprocesses—not `N`
isolated chat contexts. When native subagents are available, the supervising chat
agent should use one foreground single-PDF ingest per subagent and maintain a
bounded rolling pool. See [the chat-relay protocol](./prompts/chat-relay.md).
Batch resumes re-evaluate the active provider while preserving an explicit
`-w N`. A relay timeout keeps its own pending prompt and is retryable with
`agent ingest --resume` once that prompt is answered — the checkpoint is
per-PDF, so the paper restarts from the top rather than from the phase that
timed out.

`RW_MODELS_CONFIG` selects a model config without defeating its per-role routing; `RW_LLM_PROVIDER` globally overrides every role and should normally remain unset. Keep credentials in mode-`0600` dotenv files and use only endpoints you trust. Model roles, endpoint precedence, profile isolation, local setup, and provider-specific caveats are documented in [Provider setup in depth](./WORKFLOW.md#provider-setup-in-depth).

## Organizing and browsing the wiki

Open the directory containing both `wiki/` and `papers/` as an [Obsidian](https://obsidian.md/) vault—normally the repository root—and start at `wiki/views.md`. This keeps each page's source-PDF link inside the vault. Enable the Dataview community plugin to render the dashboard's live tables; without it, every page remains ordinary readable Markdown.

| Location | Purpose |
| --- | --- |
| `wiki/views.md` | Dashboard and default landing page |
| `wiki/index.md` | Full catalog grouped by category |
| `wiki/log.md` | Chronological ingest and authoring history |
| `wiki/{category}/` | Paper and commentary pages |
| `wiki/synthesis/` | Retrospective cross-paper analyses |
| `wiki/concepts/` | Grounded hub notes spanning recurring concepts |
| `wiki/ideas/` | Forward-looking, reviewable design proposals |
| `wiki/references/` | Guidance, protocols, whitepapers, and books |

Paper pages are generated from PDFs. Commentary pages are explicitly attributed and carry no findings of their own. Synthesis and concept pages are strictly grounded; idea pages allow marked model priors only in their design sections. See [CLAUDE.md](./CLAUDE.md) for the full page contracts.

Useful discovery commands include `researchwiki search`, `researchwiki neighbors`, `researchwiki candidates synthesis`, `researchwiki candidates concepts`, and `researchwiki claim-graph`. `researchwiki visualize` writes a self-contained interactive graph to `output/graph.html`.

## Import and export

For an existing Zotero, Paperpile, Mendeley, or ReadCube library, export BibTeX or RIS and ask the agent to import it. `researchwiki import` pairs records with PDFs, reports duplicates and extraction problems before spending tokens, and ingests in controlled waves. See the [import guide](./prompts/import-reference-manager.md).

Export options:

```bash
researchwiki export --format bibtex > refs.bib
researchwiki export --format ris > refs.ris
researchwiki export --format okf --out output/okf
```

Bibliography exports include published resources; OKF also carries synthesis, concept, and idea pages as portable knowledge. See the [export guide](./prompts/export-bibliography.md).

## Sync across computers

The durable library consists of three directories:

- `wiki/` — pages, dashboard, index, and log
- `papers/` — canonical PDFs and supplementary files
- `inbox/` — the ingest backlog

> **Tip:** keep the Git checkout outside your synced folder. Put these three directories under iCloud, Dropbox, Drive, Syncthing, or a similar service, then symlink them into the checkout.

For a new installation, follow the [Install](#install) steps through virtual-environment activation. Before initialization, link the empty content directories into that checkout:

```bash
SYNC="$HOME/<your-synced-folder>/research-wiki"
mkdir -p "$SYNC"/{wiki,papers,inbox}
for d in wiki papers inbox; do ln -s "$SYNC/$d" "$d"; done

researchwiki init
```

This keeps `.git/`, virtual environments, SQLite, and derived indexes out of the sync service. Open `$SYNC` as the Obsidian vault so its sibling `wiki/` and `papers/` directories—and therefore PDF links—remain inside the vault.

On each additional computer:

1. Clone the same Git revision and follow the [Install](#install) steps.
2. Create a machine-local dotenv profile (`.env` or `.env.NAME`) with its credentials and model routing.
3. Link the same three synced directories.
4. Wait for sync to finish, then rebuild local derived state:

```bash
researchwiki db rebuild
researchwiki reindex
researchwiki grade regression --missing-only
```

Do **not** sync the state DB or cache/index directories. Avoid ingesting or editing on two computers simultaneously, and wait for file sync to settle before switching machines. Knowledge converges from Markdown and PDFs; ingest telemetry and cost history remain machine-local. See the [migration and sync guide](./prompts/migration-backfill.md#keep-the-checkout-out-of-the-synced-folder) for existing libraries and no-symlink setups.

## Data, privacy, and validation

Git tracks the framework: `researchwiki/`, `prompts/`, config templates, tests, and documentation. Your `wiki/`, `papers/`, `inbox/`, dotenv profiles, active/custom model configs, databases, and caches are gitignored.

Cloud-backed model workflows may send prompts, extracted PDF text, and wiki content to the configured provider. Use a local model to keep those calls on your machine. See [SECURITY.md](./SECURITY.md) for the complete data-flow surface.

For contribution and test guidance, see [CONTRIBUTING.md](./CONTRIBUTING.md).
The [beta compatibility contract](./COMPATIBILITY.md) defines deprecation and
persisted-data migration guarantees. For the complete CLI walkthrough and
recovery procedures, see [WORKFLOW.md](./WORKFLOW.md).

## License

The framework is [MIT licensed](./LICENSE). PDFs under `benchmark-fixtures/pdfs/` are redistributed under CC-BY-4.0 with attribution in [`benchmark-fixtures/LICENSES.md`](./benchmark-fixtures/LICENSES.md). Your own PDFs remain under their original terms and are never included in the repository.

## Inspiration

Inspired by [Andrej Karpathy’s LLM Wiki pattern](https://gist.github.com/karpathy/1dd0294ef9567971c1e4348a90d69285) and [joonan30’s llm-wiki](https://gist.github.com/joonan30/cbce305684d079dbe9a3fbaefe4e3959), simplified to two tiers: immutable source PDF → one LLM-authored wiki page.
