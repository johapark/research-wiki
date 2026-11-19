# Research Wiki

A local, markdown-first research wiki that compounds as you add papers. Give your chat agent PDFs to ingest or questions to investigate, then explore the knowledge base in the browser dashboard or Obsidian.

The agent handles setup, ingestion, verification, search, and maintenance. You describe the outcome; it runs the necessary commands. Your source PDFs and Markdown pages remain the durable library.

## Features

- **Standalone dashboard.** Browse corpus status, search and filter papers, save pages, follow connections, and read wiki content in your browser. Live and offline modes are available.
- **Grounded in local PDFs.** Wiki prose comes from your papers. Answers start from the wiki and return to the source PDFs when more detail is needed.
- **Claim-level verification.** Claims are checked against their sources; cross-paper analyses and ideas are checked for citations and source fidelity.
- **Connected knowledge.** Citation-supported links, synthesis pages, claim relationships, and proposals turn isolated summaries into a research map.
- **Agent-operated workflows.** Ask the agent to ingest, compare, organize, troubleshoot, or maintain your library.
- **Search and discovery.** Keyword and semantic search, citation and recommendation neighbors, gap detection, and an interactive graph help you find relevant work.
- **Portable and local-first.** The library is plain Markdown and PDFs, kept separate from the framework and able to sync across computers.
- **Provider-flexible.** Choose OpenAI, Anthropic, Gemini, another compatible service, a local model, or chat relay.

## Quick start

### Requirements

- Python 3.10+
- A chat agent with file and shell access, such as Claude Code or Codex
- An API key for a cloud provider, or a local-model or chat-relay setup

### Install

Clone or download this repository and open its folder in your chat agent. Keep the checkout outside any synced folder; the agent can configure library syncing separately. Retain the checkout because the framework uses its prompts, configuration, and agent instructions.

Tell the agent: **“Initialize this research wiki.”** It checks the installation, installs missing dependencies in a virtual environment outside the checkout, settles the storage layout, configures your chosen provider, and runs a local readiness check. Tell it whether you want to sync across devices before initialization.

The agent contract is in [CLAUDE.md](./CLAUDE.md), also loaded through the `AGENTS.md` symlink. Detailed setup steps are in the [initialization guide](./prompts/init.md).

### Add your first paper

Give the agent a PDF path and say **“Add this paper to my research wiki.”** For a batch, place PDFs in `inbox/` and say **“Ingest everything in the inbox.”** If you want the original file to stay at its current location, ask the agent to copy it into the inbox first.

The agent reports which papers were published, which need review, and which failed. A reviewable draft is not a published wiki page. Exact duplicate inbox files are archived and remain recoverable. For an existing reference-manager library, start with [import and export](#import-and-export).

Then say **“Open the dashboard.”**

## How it works

| You say | The agent does |
| --- | --- |
| “Open the dashboard.” | Starts or reuses the local server and opens the browser workspace. |
| “Ingest everything in the inbox.” | Processes the PDFs, verifies the pages, and reports outcomes and remaining work. |
| “What does the wiki have on sequence-to-expression models?” | Searches the library, reads relevant pages, and answers from the available evidence. |
| “Should these pangenome papers become a synthesis?” | Reviews coverage and recommends a cross-paper analysis; writes it when asked. |
| “File this design as an idea.” | Creates an idea page with cited background and clearly marked model assumptions. |
| “What related papers am I missing?” | Finds literature leads; new evidence still requires an ingested PDF. |
| “Check the library and fix incomplete indexing.” | Inspects workflow state and repairs local derived state. |

## Organizing and browsing the wiki

### Browser dashboard

Ask the agent to **“Open the dashboard”** or **“Stop the dashboard server.”** The live dashboard runs on localhost, shows corpus totals and workflow queues, and refreshes after library changes. Search and combined category, page-type, year, and status filters help navigate the library. Table and card layouts, saved pages, backlinks, and source-PDF links support reading and exploration.

The reader offers adjustable text size and a font selector: Source Serif 4, SNUJaha with Korean support, or the system font. Bookmarks, theme, and reading preferences are saved in your browser.

For a portable snapshot, say **“Export an offline dashboard.”** The agent creates `output/dashboard.html` with the library, search, reader, and fonts embedded. Regenerate it after library changes. Source-PDF links still need the local PDFs alongside the export's relative paths.

Server management details are in the [dashboard guide](./prompts/dashboard.md).

### Obsidian

You can also open the directory containing both `wiki/` and `papers/` as an [Obsidian](https://obsidian.md/) vault and start at `wiki/views.md`. Enable the Dataview community plugin for its live tables. Every wiki page remains readable as ordinary Markdown.

### Library organization

Categories emerge from your papers. The agent proposes an initial grouping when enough PDFs are available and suggests refinements as the corpus grows; category changes are reviewed before application.

- `papers/` holds canonical source PDFs and supplementary files.
- `wiki/` holds paper summaries, cross-paper syntheses, ideas, proposals, and reference documents.
- `inbox/` holds PDFs awaiting processing.
- `wiki/index.md` catalogs pages; `wiki/log.md` records ingestion and authoring history.

Ask **“Suggest evidence-backed synthesis or idea topics”** to explore proposals, or **“Show the library graph”** to view connections. The agent records your proposal decisions so they can inform later work. Page contracts are documented in [CLAUDE.md](./CLAUDE.md).

## Providers

Choose a provider during initialization, or ask the agent to switch later. It manages the provider configuration and guides credential setup.

| Provider | What you need |
| --- | --- |
| **OpenAI or Anthropic** | An API key for the selected service. |
| **Gemini or another OpenAI-compatible service** | The service endpoint, model names, and API key. |
| **Local model** | A running compatible local model endpoint. |
| **Chat relay** | An active supported chat agent to answer model requests; no separate model API key or server. |

Cloud providers may charge for model calls. Chat-relay workflows need the connected agent while processing. You can use separate provider profiles and different models for different roles; ask the agent to configure them. Details are in [provider setup](./WORKFLOW.md#provider-setup-in-depth) and the [chat-relay guide](./prompts/chat-relay.md).

## Import and export

For Zotero, Paperpile, Mendeley, or ReadCube, export BibTeX or RIS and give the agent that file and the PDF directory. Say **“Import this library.”** It checks metadata, PDF matches, duplicates, and extraction problems before ingestion. See the [import guide](./prompts/import-reference-manager.md).

Ask **“Export my bibliography as BibTeX”** or **“Export it as RIS”** for reference-manager formats. An Open Knowledge Format (OKF) export also carries synthesis, idea, proposal, and legacy concept pages. See the [export guide](./prompts/export-bibliography.md).

## Sync across computers

Tell the agent which sync service and folder to use before initialization. It keeps the checkout and virtual environment outside that folder and links the three durable directories—`wiki/`, `papers/`, and `inbox/`—into the checkout.

On an additional computer, open a separate checkout and ask **“Connect this checkout to my existing synced library.”** The agent configures that machine's provider, links the content directories, and rebuilds local state after syncing finishes. Databases, search indexes, caches, credentials, and processing telemetry stay machine-local.

Wait for file sync to settle before switching computers, and avoid ingesting or editing on two computers simultaneously. The [migration and sync guide](./prompts/migration-backfill.md#keep-the-checkout-out-of-the-synced-folder) covers existing libraries and systems without symlinks.

## Data, privacy, and validation

Git tracks the framework, prompts, configuration templates, tests, and documentation. Your PDFs, wiki pages, inbox, credentials, active model configurations, databases, and caches are gitignored. Back up the durable library independently of Git.

Cloud-backed workflows may send prompts, extracted PDF text, and wiki content to your configured provider. A local model keeps those model calls on your machine. See [SECURITY.md](./SECURITY.md) for the complete data flow.

## Further documentation

- [WORKFLOW.md](./WORKFLOW.md): technical workflows, command reference, costs, and recovery procedures.
- [Application integration](./docs/application-integration.md): ingestion APIs, local jobs, and evidence selections for developers.
- [Verified synthesis authoring](./docs/synthesis-authoring.md): composing and publishing cross-paper analyses.
- [CONTRIBUTING.md](./CONTRIBUTING.md): development and test guidance.
- [COMPATIBILITY.md](./COMPATIBILITY.md): deprecation and persisted-data migration guarantees.

## License

The framework is licensed under [PolyForm Noncommercial 1.0.0](./LICENSE), permitting noncommercial use, modification, and redistribution. Commercial use requires a separate license from the copyright holder. Copies previously distributed under MIT retain their MIT permissions.

Bundled Source Serif 4 and SNUJaha fonts use SIL Open Font License 1.1; their notices are included in [the font assets](./researchwiki/templates/fonts/README.md).

[Benchmark PDFs](./benchmark-fixtures/LICENSES.md) and the [visual inspection dataset](./benchmark-fixtures/visual-inspection/LICENSES.md) use CC-BY-4.0 with attribution. Your own PDFs retain their original terms and are not included in the repository.

## Inspiration

Inspired by [Andrej Karpathy’s LLM Wiki pattern](https://gist.github.com/karpathy/1dd0294ef9567971c1e4348a90d69285) and [joonan30’s llm-wiki](https://gist.github.com/joonan30/cbce305684d079dbe9a3fbaefe4e3959), simplified to two tiers: immutable source PDF → one LLM-authored wiki page.
