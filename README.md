# Research Wiki

A local, markdown-first research wiki that compounds as you add papers. Drop PDFs into `inbox/`, ask your coding agent to ingest or analyze them, and browse the resulting knowledge base in Obsidian.

The agent operates the CLI for you: it reads the source PDF, writes and grades the page, connects related work, and keeps indexes current. [CLAUDE.md](./CLAUDE.md) and its `AGENTS.md` symlink contain the agent contract; [WORKFLOW.md](./WORKFLOW.md) is the detailed human reference.

## Features

- **Grounded in local PDFs.** Wiki prose comes from your papers, not web summaries. Answers start from the wiki and return to the PDFs when more detail is needed.
- **Claim-level verification.** Claims are graded against their source and exposed through durable `[[paper#claim]]` anchors. Cross-paper pages (syntheses, concepts, and ideas) are also checked for complete citations and source fidelity.
- **Connected knowledge.** Citation-supported links, concept hubs, synthesis pages, claim relationships, and evolution proposals turn isolated summaries into a research map.
- **Agent-operated ingestion.** Metadata reconciliation, naming, classification, drafting, grading, backlinking, indexing, and crash recovery run as one transactional workflow.
- **Search and discovery.** BM25 and semantic search, citation/recommendation neighbors, gap detection, and an interactive graph help you find papers and missing literature.
- **Portable and private by default.** Your wiki is plain Markdown, your library is gitignored, and the content directories can sync independently of the framework.
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
- Git
- An LLM provider for agent ingestion; offline tools such as `status`, `lint`, and `search` need none
- About 2 GB for the bundled semantic grader and its model cache

### Install

```bash
git clone git@github.com:johapark/research-wiki.git
cd research-wiki
python3 -m venv ~/.venvs/research-wiki
~/.venvs/research-wiki/bin/pip install -e .
source ~/.venvs/research-wiki/bin/activate
```

Keep the virtual environment outside the checkout, especially when syncing your library. The default install includes PDF extraction, Tantivy search, semantic grading, and the Anthropic SDK. Optional extras are `[mcp]` and `[dev]`.

### Initialize

Choose one:

- Open the clone in Claude Code, Codex, Cursor, Aider, or another agent that reads `CLAUDE.md`/`AGENTS.md`, then say: **“Initialize this research wiki.”**
- Run the interactive wizard: `researchwiki init`.
- For directories only: `researchwiki init --scaffold-only`.

No taxonomy is predefined. During setup, enter category names yourself (the default) or, with at least three PDFs in `inbox/`, let the agent propose them from your papers. As the corpus grows, the agent suggests useful category splits for your review; nothing changes automatically. `other/` is always available as the classifier’s abstention bucket.

### First ingest

Place a PDF in `inbox/` and tell the agent: **“Ingest the new paper.”** It will create a canonical PDF in `papers/`, a grounded page in `wiki/{category}/`, reciprocal supported links, and updated indexes.

For multiple PDFs, ask it to ingest the whole inbox in one batch. If your papers already live in Zotero, Paperpile, Mendeley, or ReadCube, use the [library import workflow](#import-and-export) instead.

## Sync across computers

The durable library consists of three directories:

- `wiki/` — pages, dashboard, index, and log
- `papers/` — canonical PDFs and supplementary files
- `inbox/` — the ingest backlog

> **Tip:** keep the Git checkout outside your synced folder. Put these three directories under iCloud, Dropbox, Drive, Syncthing, or a similar service, then symlink them into the checkout. This is the layout used by the project author.

For a fresh checkout:

```bash
SYNC="$HOME/<your-synced-folder>/research-wiki"
mkdir -p "$SYNC"/{wiki,papers,inbox}

git clone git@github.com:johapark/research-wiki.git ~/src/research-wiki
cd ~/src/research-wiki
for d in wiki papers inbox; do ln -s "$SYNC/$d" "$d"; done

researchwiki init --scaffold-only
```

This keeps `.git/`, virtual environments, SQLite, and derived indexes out of the sync service. Open `$SYNC/wiki` as an Obsidian vault; keep `$SYNC/wiki` and `$SYNC/papers` as siblings so PDF links resolve.

On each additional computer:

1. Clone the same Git revision and install the package.
2. Create a local `.env` with that machine’s credentials and model routing.
3. Link the same three synced directories.
4. Wait for sync to finish, then rebuild local derived state:

```bash
researchwiki db rebuild
researchwiki reindex
researchwiki grade regression --missing-only
```

Do **not** sync the state DB or cache/index directories. Avoid ingesting or editing on two computers simultaneously, and wait for file sync to settle before switching machines. Knowledge converges from Markdown and PDFs; ingest telemetry and cost history remain machine-local. See the [migration and sync guide](./prompts/migration-backfill.md#keep-the-checkout-out-of-the-synced-folder) for existing libraries and no-symlink setups.

## Providers

The root `.env` loads automatically; shell variables take precedence. For an isolated profile, run `researchwiki --env-file .env.NAME init`. This creates a gitignored `config/profiles/NAME.yaml`, leaving tracked config templates unchanged. Use the same `--env-file` option for later commands. Mode `0600` is recommended for env files; broader permissions warn but do not block execution.

| Provider | Minimal setup |
| --- | --- |
| **OpenAI** | Set `OPENAI_API_KEY`; no model config is required. |
| **Anthropic** | Select Anthropic in `researchwiki init`; set `ANTHROPIC_API_KEY`. |
| **Gemini** | Select Gemini in `researchwiki init`; set `OPENAI_API_KEY` to the Gemini key. |
| **Other OpenAI-compatible** | Select it in `researchwiki init`; provide the exact models, endpoint, and API key. |
| **Local model** | Use `config/models.lmstudio.yaml` with LM Studio or another compatible local server. |
| **Chat relay** | Set `RW_LLM_PROVIDER=chat-relay`; no API key or server is required. |

Example:

```bash
# .env
OPENAI_API_KEY="sk-..."
# RW_MODELS_CONFIG=models.anthropic.yaml
# RW_LLM_BASE_URL=https://your-compatible-endpoint/v1
```

`RW_MODELS_CONFIG` selects a file under `config/`; `RW_LLM_BASE_URL` overrides its endpoint. Explicit configs fail closed when missing, malformed, unsupported, or ambiguous. HTTP and HTTPS endpoints are accepted—use only hosts you trust because the API key and paper content are sent there. Model roles, mixed-provider routing, local setup, and provider-specific caveats are documented in [Provider setup in depth](./WORKFLOW.md#provider-setup-in-depth).

## Organizing and browsing the wiki

Open `wiki/` as an [Obsidian](https://obsidian.md/) vault and start at `views.md`. Enable the Dataview community plugin to render its live tables; without it, every page remains ordinary readable Markdown.

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

Useful discovery commands include `search`, `neighbors`, `candidates synthesis`, `candidates concepts`, and `claim-graph`. `researchwiki visualize` writes a self-contained interactive graph to `output/graph.html`.

## Import and export

For an existing Zotero, Paperpile, Mendeley, or ReadCube library, export BibTeX or RIS and ask the agent to import it. `researchwiki import` pairs records with PDFs, reports duplicates and extraction problems before spending tokens, and ingests in controlled waves. See the [import guide](./prompts/import-reference-manager.md).

Export options:

```bash
researchwiki export --format bibtex > refs.bib
researchwiki export --format ris > refs.ris
researchwiki export --format okf --out output/okf
```

Bibliography exports include published resources; OKF also carries synthesis, concept, and idea pages as portable knowledge. See the [export guide](./prompts/export-bibliography.md).

## Data, privacy, and validation

Git tracks the framework: `researchwiki/`, `prompts/`, config templates, tests, and documentation. Your `wiki/`, `papers/`, `inbox/`, local model config, databases, and caches are gitignored.

Ingestion sends extracted paper text to the configured LLM provider unless you use a local model. See [SECURITY.md](./SECURITY.md) for the data-flow surface.

For contribution and test guidance, see [CONTRIBUTING.md](./CONTRIBUTING.md). For the complete CLI walkthrough and recovery procedures, see [WORKFLOW.md](./WORKFLOW.md).

## License

The framework is [MIT licensed](./LICENSE). PDFs under `benchmark-fixtures/pdfs/` are redistributed under CC-BY-4.0 with attribution in [`benchmark-fixtures/LICENSES.md`](./benchmark-fixtures/LICENSES.md). Your own PDFs remain under their original terms and are never included in the repository.

## Inspiration

Inspired by [Andrej Karpathy’s LLM Wiki pattern](https://gist.github.com/karpathy/1dd0294ef9567971c1e4348a90d69285) and [joonan30’s llm-wiki](https://gist.github.com/joonan30/cbce305684d079dbe9a3fbaefe4e3959), simplified to two tiers: immutable source PDF → one LLM-authored wiki page.
