# Initialization — cold-start workflow

Read this file when a fresh clone has no `wiki/`, `papers/`, or `inbox/` and
the user asks to initialize it. Keep the path to first value short. Ask only
three consequential questions when their answers are not already known:

1. Should the library sync across devices?
2. Which model provider should author the pages?
3. Which PDF should be added first?

Categories, dashboard design, and Obsidian graph cosmetics are not first-run
decisions. Make reasonable defaults and explain them after the first paper.

## 1. Check the install and decide the storage layout

Run `researchwiki --version`. Do not use `python -m researchwiki` as the install
check: from the checkout it can import an uninstalled package while optional
dependencies are missing.

If installation is needed, use a virtual environment outside the checkout:

```bash
python3 -m venv ~/.venvs/research-wiki
~/.venvs/research-wiki/bin/pip install -e .
source ~/.venvs/research-wiki/bin/activate
```

Ask about sync **before creating content directories**. The answer determines
where `wiki/`, `papers/`, and `inbox/` live.

- No sync: leave all three as ordinary directories in the checkout.
- Sync: keep the Git checkout and virtual environment outside the synced
  folder; create the three directories inside the synced folder and symlink
  them into the checkout. Follow
  [`migration-backfill.md`](./migration-backfill.md#keep-the-checkout-out-of-the-synced-folder).

Do not put the checkout, `.git`, state DB, or derived caches in the synced
folder. If the platform cannot use symlinks, use the documented fallback.

## 2. Scaffold and run the free readiness check

After the storage layout is settled, run:

```bash
researchwiki init --scaffold-only
researchwiki doctor
```

The scaffold command is non-interactive and idempotent. It creates the content
directories, page-type directories, `wiki/index.md`, and `wiki/views.md`; an
existing dashboard is preserved. The dashboard is static and renders in
Obsidian when the Dataview plugin is enabled.

The dashboard contract is automatic: Keep this dashboard JavaScript-free. It
uses stamped dates, including `WHERE type = "concept" AND generated_at`.
There is **No synthesis member-count column**; concept membership comes from
its canonical spoke registry, `referenced_papers`.

`researchwiki doctor` is local and free. It checks Python, dependencies,
content paths and symlinks, model routing and credentials, state DB access,
search state, curl, and whether the semantic model is already cached. It does
not download the model or contact the provider. Never run
`researchwiki doctor --probe` without telling the user: `--probe` makes one
small provider call and may spend tokens.

## 3. Configure a provider only when needed

If `doctor` reports a provider blocker, ask which provider the user wants and
use `researchwiki init` in their terminal, or configure the equivalent profile
directly. The root `.env` loads automatically; named profiles are selected with
the global `--env-file` option before the command.

Use the provider table in [`README.md`](../README.md#providers) rather than
repeating all endpoint and model details here. The important defaults are:

- OpenAI uses the built-in routing and needs `OPENAI_API_KEY`.
- Anthropic and other backends need their matching `config/models.*.yaml`
  selection as well as their credential.
- Local OpenAI-compatible endpoints need no key.
- Chat relay needs no key or server; read [`chat-relay.md`](./chat-relay.md)
  before the first ingest.

Rerun `researchwiki doctor` after configuration. Offer `doctor --probe` only if
the user wants an explicit network/model check.

## 4. Let categories emerge from the corpus

A new wiki uses `wiki/other/` until there is enough evidence for a taxonomy.
Do not ask the user to invent categories for one or two papers.

At three or more PDFs, offer:

```bash
researchwiki bootstrap-categories            # preview
researchwiki bootstrap-categories --apply    # create approved directories
```

Categories are local and review-gated. `other/` remains the abstention bucket;
later, `researchwiki suggest-splits` can propose refinements. Manual category
creation (`mkdir wiki/<slug>/`) is an advanced option, not the default path.

## 5. Add the first paper

Accept the user's PDF wherever it already lives; copying it into `inbox/` first
is optional:

```bash
researchwiki add /path/to/paper.pdf
```

`add` is the discoverable front door to the complete `agent ingest` workflow.
It derives the canonical stem, copies or moves the source into `papers/`,
writes the grounded page, indexes its claims, and updates search and catalog
state. Multiple paths activate the existing checkpointed batch workflow.

After success, read the new page and summarize its Summary and key
contributions in 2–3 sentences. Surface any evolution proposals. Then run
`researchwiki status`; the receipt plus status should show:

- the canonical PDF in `papers/`;
- the page in `wiki/other/` or a corpus-derived category;
- indexed claims and current derived indexes;
- no inbox backlog for a PDF that came from `inbox/`.

If the user already has a reference-manager library, stop after readiness and
use [`import-reference-manager.md`](./import-reference-manager.md) instead of
adding papers one by one.

## 6. Optional Obsidian polish after first value

Only after the first page exists, explain that the repository root can be
opened as an Obsidian vault and Dataview renders `wiki/views.md`. If the graph
is noisy, offer this reversible `.obsidian/graph.json` filter:

```json
{ "search": "-path:\"index.md\" -path:\"log.md\" -path:\"views.md\" -path:\"prompts/\" -path:\"benchmark-fixtures/\" -path:\"tests/\" -path:\"CLAUDE.md\" -path:\"README.md\" -path:\"AGENTS.md\" -path:\"WORKFLOW.md\"" }
```

This is graph-only cosmetics. Do not delay the first ingest to configure it.
