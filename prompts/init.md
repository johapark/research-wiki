# Initialization — cold-start workflow

Read this file when a fresh clone needs setup: every `wiki/{category}/` is empty, no PDFs in `papers/`, and the user signals they're new (*"set this up"*, *"initialize"*, *"how do I start"*). Walk through the steps below; surface each decision to the user — do not run silently.

> **Scripted alternative:** `researchwiki init` is an interactive terminal wizard that performs these same steps (provider → categories → dashboard → confirm) by prompting the user directly. It's the human-run complement to this LLM-guided path — point the user to it if they'd rather drive setup themselves; the steps below are the conversational equivalent and stay in sync with it.

Skip steps the user has already completed: if `researchwiki --version` succeeds, skip Step 1; if a provider is configured and reachable, skip Step 2; if `wiki/{category}/` directories were already populated by a prior init attempt, skip Step 3.

## Step 1 — Install

`researchwiki --version` first. If missing:

```bash
pip install -e .                             # core + claim grader + Anthropic SDK
```

The default install includes torch + sentence-transformers + BGE bi-encoder (~2 GB) and the Anthropic SDK. If the user wants to skip loading the grader weights at runtime, tell them to pass `--no-semantic` to `agent ingest` / `reindex` — draft selection falls back to BM25 and pages land in the sandbox for manual promotion.

## Step 2 — Pick a provider

Ask the user. Verify the env var is set in their shell after they choose (`echo $ANTHROPIC_API_KEY` / `echo $RW_LLM_PROVIDER`).

- **Anthropic cloud** (default, ~$0.10/paper) — `ANTHROPIC_API_KEY` must be set. Best fidelity for the `author` and `judge` phases.
- **OpenAI-compatible cloud** (OpenAI, Google Gemini, Groq, OpenRouter, etc.) — `cp config/models.openai-compatible.yaml config/models.yaml`, then set `RW_LLM_BASE_URL` to the provider's base URL and `OPENAI_API_KEY` to its key (forwarded as Bearer). Without this copy step, the loader's hardcoded fallback routes every role to `provider: anthropic` regardless of which key is set — an OpenAI/Google-only user who skips this will hit `ANTHROPIC_API_KEY not set` mid-pipeline, not a clean upfront error. Edit the `model:` strings to match the provider's catalog.
- **Local LLM** (free per paper after setup) — server on `localhost:1234` (or override `RW_LLM_BASE_URL`); per-role routing in `config/models.yaml`. 7–8B models work for `classifier` / `keywords` / `reconcile`; keep `author` and `judge` on a larger model.
- **Chat-relay** (no API key, no local server) — `export RW_LLM_PROVIDER=chat-relay`. The chat agent fills each prompt in `.llm-relay/pending/`. Read `prompts/chat-relay.md` for the protocol before the first ingest.

## Step 3 — Categories (run bootstrap by default)

**Default action when `inbox/` has any PDFs: run `researchwiki bootstrap-categories` (print-only) and surface the proposed taxonomy to the user.** Do NOT default to the shipped categories table in `CLAUDE.md` — it is tuned for biology + ML and will be wrong for other domains. Empty `wiki/{category}/` directories mean the taxonomy has not been chosen yet, regardless of what the table shows.

```bash
researchwiki bootstrap-categories            # print-only
researchwiki bootstrap-categories --apply    # rewrite CLAUDE.md table + categories.py atomically
```

Skip bootstrap *only* when:
- The user explicitly asks for the shipped defaults (*"use the defaults"*, *"keep the defaults"*).
- `inbox/` is empty — defer until they drop PDFs.
- The user wants to hand-edit (`CLAUDE.md` Categories table + `researchwiki/categories.py` `VALID_CATEGORIES` set must agree).

`other` must always be present (abstention bucket).

## Step 4 — Scaffold the dashboard

Create `wiki/views.md` — a [Dataview](https://blacksmithgu.github.io/obsidian-dataview/) dashboard surfacing recent additions (top 15 papers, top 10 synthesis pages, top 5 ideas). The file is static (the queries render against whatever pages exist), so generate it now regardless of whether any papers are ingested yet. Skip only if `wiki/views.md` already exists from a prior init.

Write the three `dataview` blocks filtering on `type` (`paper` / `synthesis` / `idea`) and sorting on `default(ingested_at, file.ctime)` / `default(generated_at, file.mtime)` DESC — the `default(...)` fallback keeps pages ingested before those stamps existed in the ranking. Filter on `type` rather than a folder-based `FROM` so the queries work whether the Obsidian vault is opened at `wiki/` or the repo root. Tell the user the tables render only inside Obsidian with the Dataview plugin enabled; on GitHub they appear as code blocks.

**Table schema** — each block is a `TABLE WITHOUT ID` with `FROM ""` (whole vault). Use exactly these columns and sort keys:

*Recent papers (LIMIT 15, `WHERE type = "paper"`):*
```dataview
TABLE WITHOUT ID
  file.link as Page,
  short_name as "Short name",
  category[0] as Cat,
  default(ingested_at, dateformat(file.ctime, "yyyy-MM-dd")) as "Added"
FROM ""
WHERE type = "paper"
SORT default(ingested_at, file.ctime) DESC
LIMIT 15
```

*Recent synthesis (LIMIT 10, `WHERE type = "synthesis"`):* columns `file.link as Page`, `length(referenced_papers) as Members`, `topic_seed as "Topic seed"`, `default(generated_at, dateformat(file.mtime, "yyyy-MM-dd")) as Generated`; `SORT default(generated_at, file.mtime) DESC`.

*Recent ideas (LIMIT 5, `WHERE type = "idea"`):* columns `file.link as Page`, `verdict as Verdict`, `status as Status`, `default(generated_at, dateformat(file.mtime, "yyyy-MM-dd")) as Generated`; `SORT default(generated_at, file.mtime) DESC`.

**Frontmatter — no `category:` field.** `views.md` sits at the wiki root (`wiki/views.md`), whose parent dir is `wiki`, not a content category or page-type dir. Give it only `type: dashboard` and `tags:` (mirror the other root bookkeeping files — `index.md`, `log.md` — which carry no `category:`). Adding `category: [...]` here trips `lint`'s category-drift check, which reads the parent dir as the canonical category and sees `wiki` ≠ whatever you wrote. Same rule for any other page you author directly at the wiki root.

## Step 5 — Filter non-corpus markdown from Obsidian's graph view

The graph view should render paper ↔ paper relationships, not the scaffolding around them. Everything below is either a hub that links to nearly every paper (bookkeeping files) or a static reference doc that has no place in the paper-relationships graph (workflow prompts, harness docs, repo READMEs). Filter it all out at the vault level.

Files to exclude:

- **Wiki root bookkeeping** — `index.md` (catalogs every paper), `log.md` (mentions every stem in ingest entries), `views.md` (Dataview blocks list every paper), `pdfs-failed-parsing.md` (parse-failure stubs).
- **Workflow prompts** — everything under `prompts/` (init, ingest-digest, idea-page-author, recategorize, recovery, lookups, cross-link-discovery, export-shareable, audit-refresh, plus system prompts). Reference docs about how to run the framework; not corpus content.
- **Benchmark harness** — everything under `benchmark-fixtures/` (fixture YAMLs, README, PLAN, CALIBRATION, LICENSES, retrieval/README). Test infrastructure, not knowledge.
- **Test fixtures** — everything under `tests/` (HALLUCINATED grader-test pages). Deliberately-wrong pages used for grader-regression testing.
- **Root repo docs** — `CLAUDE.md`, `README.md`, `AGENTS.md`, `WORKFLOW.md`. Project instructions and READMEs.

Dot-prefixed dirs (`.agent-output/`, `.eval-*/`, `.pytest_cache/`) are ignored by Obsidian's file explorer by default and don't need an explicit filter.

The fix is one line in the vault's graph config. Open `.obsidian/graph.json` (create it if Obsidian hasn't written one yet — this file is per-user gitignored, so seeding it is fine) and set the `search` field to a combined exclusion:

```json
{ "search": "-path:\"index.md\" -path:\"log.md\" -path:\"views.md\" -path:\"pdfs-failed-parsing.md\" -path:\"prompts/\" -path:\"benchmark-fixtures/\" -path:\"tests/\" -path:\"CLAUDE.md\" -path:\"README.md\" -path:\"AGENTS.md\" -path:\"WORKFLOW.md\"" }
```

Use `path:` (substring match on the full path) rather than `file:` (basename match) so a single `-path:"prompts/"` filters everything under that directory, and so the filter works whether the vault is opened at the repo root or at `wiki/`. Every filtered file stays browsable via the file tree, global search, and cmd-click on wikilinks — they just don't render as nodes in the graph pane. Reversible by clearing the `search` field.

Nothing programmatic depends on this; it's a graph-only cosmetic change. Apply the same filter to any secondary vault the user maintains (e.g. `wiki/.obsidian/graph.json` if they open the wiki subdirectory as its own vault).

## Step 6 — First ingest

```bash
researchwiki agent ingest inbox/<file>.pdf
```

Summarize the new page (Summary section + key contributions in 2–3 sentences) and surface any memory-evolution proposals per Step 6 of the Ingest workflow in `CLAUDE.md`.

## Step 7 — Confirm

`researchwiki status` should show non-zero `Pages:` and zero `inbox/ PDFs awaiting ingest`. If both check out, setup is done — tell the user what they can do next (drop more PDFs, ask cross-paper questions, file ideas).
