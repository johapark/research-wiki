# Initialization — cold-start workflow

Read this file when a fresh clone needs setup: no `wiki/`, `papers/` or `inbox/` yet (they are gitignored in full, so a clone has none), and the user signals they're new (*"set this up"*, *"initialize"*, *"how do I start"*). Walk through the steps below; surface each decision to the user — do not run silently.

**Create the content dirs first** — `researchwiki init --scaffold-only`. Every command except `init` exits 2 while `wiki/` is missing, so nothing below works until this runs. It is non-interactive (the wizard needs a TTY you don't have) and idempotent. Do it after Step 1's install, before anything else; if the user is setting up the synced-folder layout, symlink the dirs *first* so the scaffold lands in the synced folder.

> **Scripted alternative:** `researchwiki init` is an interactive terminal wizard covering **Steps 2, 3, 4 and 7** (provider → categories → dashboard → confirm) by prompting the user directly. It's the human-run complement to this LLM-guided path — point the user to it if they'd rather drive setup themselves.
>
> It is not a substitute for this file. Step 1 (install) it cannot do — the wizard is itself part of the package being installed — and it does not touch Step 5 (graph filter) or Step 6 (first ingest). Note also that the wizard's own on-screen numbering starts at its provider step, so its "Step 1" is this file's Step 2.

Skip steps the user has already completed: if `researchwiki --version` succeeds, skip Step 1; if a provider is configured and reachable, skip Step 2; if `wiki/{category}/` directories were already populated by a prior init attempt, skip Step 3.

## Step 1 — Install

`researchwiki --version` first — and trust only that. `python -m researchwiki` *appears* to work from the repo root even with nothing installed, because the package imports from the working directory; it then prints a wall of `could not load task 'X': No module named 'yaml'` warnings, which reads like a broken install rather than a missing one.

If it's missing, **install into a virtualenv, and put that virtualenv outside the repo**:

```bash
python3 -m venv ~/.venvs/research-wiki
~/.venvs/research-wiki/bin/pip install -e .   # core + claim grader + Anthropic SDK
```

Both halves of that matter.

- **A venv, not the ambient interpreter.** Bare `pip install -e .` installs into whatever Python is active — frequently a shared conda `base`. This project pulls ~2 GB (torch + transformers + sentence-transformers), and its constraints are platform-specific: on x86_64 macOS, torch caps at 2.2.2, which forces `transformers<5` and `numpy<2` (see the install-time failure table in [`migration-backfill.md`](./migration-backfill.md)). Those pins have no business in an environment other projects share.
- **Outside the repo, not `.venv/` inside it.** A venv is ~34,000 files, so an in-repo venv becomes 96% of the sync load if the tree is synced, starving the daemon until it loses races against concurrent writes: stale `.git/index`, `<name> 2.md` conflict copies.

Venvs bake absolute paths into `bin/` and `pyvenv.cfg`, so pick the location now — moving one later means recreating it.

**Then ask whether they want `wiki/` and `papers/` synced** — across machines, or to read on a phone. Raise it here rather than later: the answer decides where the checkout itself goes, and moving a tree afterwards means redoing the venv and the caches.

If yes, recommend the split layout — **checkout outside the synced folder, with `wiki/`, `papers/`, `inbox/` symlinked into it.** Git and the sync service then have disjoint jobs (`wiki/*` and `papers/*` are gitignored, so the repo never carried them anyway), and the synced folder is left as a clean Obsidian vault that the **mobile app can open**. Walk them through [`migration-backfill.md`](./migration-backfill.md) § *Keep the checkout out of the synced folder*: copy-paste block, the cache relocations, and the fallback for platforms without usable symlinks. No git configuration is needed — those dirs are gitignored in full, so git is indifferent to whether they're directories or symlinks.

Warn against the inverse — checkout *inside* the synced folder with `.git` and caches relocated out. Those pointers hold absolute paths but live in the synced tree, so they reach every other machine and resolve nowhere there. Same section documents the observed failures.

That install pulls every Python dependency: `pypdfium2`, `tantivy`, `pyyaml`, `numpy`, torch + transformers + sentence-transformers, and the Anthropic SDK. Optional extras: `pip install -e '.[mcp]'` for `mcp-serve`, `.[dev]` for pytest.

`researchwiki` will not be on `PATH` in new shells — invoke it as `~/.venvs/research-wiki/bin/researchwiki`, activate the venv, or alias it.

**Two things `pip install` does *not* give you:**

- **The BGE bi-encoder weights.** `BAAI/bge-small-en-v1.5` is a ~133 MB HuggingFace download fetched on **first use** and cached under `~/.cache/huggingface/`. `index/embeddings.py` only switches to offline mode once that cache exists, so the first run needs network. If it fails, `_get_model()` swallows the error and every claim grades **BM25-only, silently** — a corpus graded that way has to be re-graded once the model lands. Warm it up and verify with `researchwiki migrate preflight`, which hard-fails when the model is unavailable.
- **`curl`.** `backfill doi`'s Crossref fallback shells out to it (`tasks/backfill.py`), inside an `except Exception: return None` — so on a slim container it just quietly stops matching. `which curl` to check.

If the user genuinely wants to skip the grader weights, `--no-semantic` on `agent ingest` / `reindex` / `grade` falls back to BM25 — draft selection degrades and pages land in the sandbox for manual promotion.

## Step 2 — Pick a provider

Ask the user, then verify the key is actually visible in their shell (`echo $OPENAI_API_KEY` / `echo $ANTHROPIC_API_KEY` / `echo $RW_LLM_PROVIDER`).

**The zero-config path is OpenAI-compatible.** With no `config/models.yaml` present, `agents/model_config._FALLBACK_ROLES` routes every role to an OpenAI-compatible endpoint at `https://api.openai.com/v1` — so an OpenAI user sets `OPENAI_API_KEY` and is done: no copy step, no `RW_LLM_BASE_URL`.

**Every other backend needs a config file, Anthropic included.** That's the failure that actually happens: a user with an Anthropic key skips the copy, believes they're on the default path, and hits a request for `OPENAI_API_KEY` mid-pipeline.

Which template pairs with which backend, and what each costs per paper, lives in README's *Providers* and *Model config* tables — send the user there instead of restating it, so a moved default only has to be corrected once. What matters at init time:

| Choice | What to do |
|---|---|
| OpenAI | Nothing but `OPENAI_API_KEY`. |
| Anthropic | `cp config/models.anthropic.yaml config/models.yaml`, set `ANTHROPIC_API_KEY`. |
| Other OpenAI-compatible (Gemini, Groq, OpenRouter, …) | Copy the matching `config/models.*.yaml` — Gemini has a ready-made one — and set `OPENAI_API_KEY` to *that* provider's key. The endpoint rides in the config's own `base_url:`; `RW_LLM_BASE_URL` is only an ad-hoc override. |
| Local (LM Studio / vLLM / llama.cpp / ollama) | `cp config/models.lmstudio.yaml config/models.yaml`, start the server, no key needed. |
| Chat-relay (no key, no server) | `export RW_LLM_PROVIDER=chat-relay`, then read [`chat-relay.md`](./chat-relay.md) before the first ingest. |

For the local path: 7–8B models handle `classifier` / `keywords` / `reconcile` fine, but write shallow `author` drafts and weak `judge` verdicts — keep those two larger. `WORKFLOW.md` § *Local LLMs* names the all-roles-local model we dogfood.

## Step 3 — Categories (run bootstrap by default)

**Default action when `inbox/` has any PDFs: run `researchwiki bootstrap-categories` (print-only) and surface the proposed taxonomy to the user.** A fresh wiki has no content categories at all — `init` creates only `other` and the four page-type dirs — so the taxonomy genuinely has to be chosen here. Any category table you may have seen in a README is one author's example, not a default anyone inherits.

```bash
researchwiki bootstrap-categories            # print-only (needs ≥3 PDFs in inbox/)
researchwiki bootstrap-categories --apply    # mkdir each wiki/<slug>/
```

Skip bootstrap *only* when:
- `inbox/` has fewer than 3 PDFs — defer until they drop more (that's `MIN_INBOX_FOR_BOOTSTRAP`).
- The user wants to name the categories themselves: `mkdir wiki/<slug>/` is the entire operation, since a category is valid iff its directory exists. Then `researchwiki reindex`.

`other` must always be present (abstention bucket).

## Step 4 — Scaffold the dashboard

Create `wiki/views.md` — a [Dataview](https://blacksmithgu.github.io/obsidian-dataview/) dashboard surfacing recent additions (top 15 papers, top 10 synthesis pages, top 5 ideas). The file is static (the queries render against whatever pages exist), so generate it now regardless of whether any papers are ingested yet. Skip only if `wiki/views.md` already exists from a prior init.

Write the three `dataview` blocks filtering on `type` (`paper` / `synthesis` / `idea`) and sorting on the YAML stamps — `ingested_at` DESC for papers, `generated_at` DESC for synthesis and ideas. Filter on `type` rather than a folder-based `FROM` so the queries work whether the Obsidian vault is opened at `wiki/` or the repo root. Tell the user the tables render only inside Obsidian with the Dataview plugin enabled; on GitHub they appear as code blocks.

**Never fall back to a filesystem time.** Do *not* write `default(ingested_at, file.ctime)` or `default(generated_at, file.mtime)`. Both fallbacks answer "when was this file last touched", not "when was this page added", and the two diverge routinely: back-link splicing rewrites a page and resets its **birthtime**, while **mtime** moves on any edit at all — the same reason `lint`'s staleness checks refuse mtime, and why provenance recovery refuses to treat a file timestamp as an ingest date. Observed failure: one ingest spliced 12 reciprocal back-links, and the 7 targets that had no `ingested_at` were stamped with the ingest second, so seven papers from 2017–2025 displaced the actual new paper at the top of *Recent papers*.

Instead, require the stamp: `WHERE type = "paper" AND ingested_at`, and format it directly (`dateformat(ingested_at, "yyyy-MM-dd")`). A page with no recorded date is then excluded rather than mis-ranked — the honest outcome, since nothing on disk records when it was added. `researchwiki lint --fix` recovers real stamps from the `ingest_iterations` log where a run exists; migrated and hand-authored pages have nothing to recover and stay out of the table. Requiring the stamp also removes the `dateformat`/`default` nesting trap that the fallback created, where stamped pages rendered as Dataview date objects and unstamped ones as `yyyy-MM-dd` strings in the same column.

*Recent papers (LIMIT 15, `WHERE type = "paper"`):*
```dataview
TABLE WITHOUT ID
  file.link as Page,
  short_name as "Short name",
  category[0] as Cat,
  dateformat(ingested_at, "yyyy-MM-dd") as "Added"
FROM ""
WHERE type = "paper" AND ingested_at
SORT ingested_at DESC
LIMIT 15
```

*Recent synthesis (LIMIT 10, `WHERE type = "synthesis" AND generated_at`):* columns `file.link as Page`, `topic_seed as "Topic seed"`, `dateformat(generated_at, "yyyy-MM-dd") as Generated`; `SORT generated_at DESC`.

**No member-count column here** — do not add `length(referenced_papers) as Members`. Synthesis pages carry no `referenced_papers:` field (CLAUDE.md §2: they cite via the body, inline `[[wikilink]]`s plus `## References` footnotes), so the column renders empty for every row. Dataview cannot count body citations, and there is no frontmatter substitute to point it at. The field is real only on **concept** pages, where it is the functional spoke registry — a `type = "concept"` block could legitimately use it.

*Recent ideas (LIMIT 5, `WHERE type = "idea" AND generated_at`):* columns `file.link as Page`, `verdict as Verdict`, `status as Status`, `dateformat(generated_at, "yyyy-MM-dd") as Generated`; `SORT generated_at DESC`.

**Frontmatter — no `category:` field.** `views.md` sits at the wiki root (`wiki/views.md`), whose parent dir is `wiki`, not a content category or page-type dir. Give it only `type: dashboard` and `tags:` (mirror the other root bookkeeping files — `index.md`, `log.md` — which carry no `category:`). Adding `category: [...]` here trips `lint`'s category-drift check, which reads the parent dir as the canonical category and sees `wiki` ≠ whatever you wrote. Same rule for any other page you author directly at the wiki root.

## Step 5 — Filter non-corpus markdown from Obsidian's graph view

The graph view should render paper ↔ paper relationships, not the scaffolding around them. Everything below is either a hub that links to nearly every paper (bookkeeping files) or a static reference doc that has no place in the paper-relationships graph (workflow prompts, harness docs, repo READMEs). Filter it all out at the vault level.

Files to exclude:

- **Wiki root bookkeeping** — `index.md` (catalogs every paper), `log.md` (mentions every stem in ingest entries), `views.md` (Dataview blocks list every paper).
- **Workflow prompts** — everything under `prompts/` (init, ingest-digest, idea-page-author, recategorize, recovery, lookups, cross-link-discovery, share-page, export-bibliography, audit-refresh, plus system prompts). Reference docs about how to run the framework; not corpus content.
- **Benchmark harness** — everything under `benchmark-fixtures/` (fixture YAMLs, README, PLAN, CALIBRATION, LICENSES, retrieval/README). Test infrastructure, not knowledge.
- **Test fixtures** — everything under `tests/` (HALLUCINATED grader-test pages). Deliberately-wrong pages used for grader-regression testing.
- **Root repo docs** — `CLAUDE.md`, `README.md`, `AGENTS.md`, `WORKFLOW.md`. Project instructions and READMEs.

Dot-prefixed dirs (`.agent-output/`, `.eval-*/`, `.pytest_cache/`) are ignored by Obsidian's file explorer by default and don't need an explicit filter.

The fix is one line in the vault's graph config. Open `.obsidian/graph.json` (create it if Obsidian hasn't written one yet — this file is per-user gitignored, so seeding it is fine) and set the `search` field to a combined exclusion:

```json
{ "search": "-path:\"index.md\" -path:\"log.md\" -path:\"views.md\" -path:\"prompts/\" -path:\"benchmark-fixtures/\" -path:\"tests/\" -path:\"CLAUDE.md\" -path:\"README.md\" -path:\"AGENTS.md\" -path:\"WORKFLOW.md\"" }
```

Use `path:` (substring match on the full path) rather than `file:` (basename match) so a single `-path:"prompts/"` filters everything under that directory, and so the filter works whether the vault is opened at the repo root or at `wiki/`. Every filtered file stays browsable via the file tree, global search, and cmd-click on wikilinks — they just don't render as nodes in the graph pane. Reversible by clearing the `search` field.

Nothing programmatic depends on this; it's a graph-only cosmetic change. Apply the same filter to any secondary vault the user maintains (e.g. `wiki/.obsidian/graph.json` if they open the wiki subdirectory as its own vault).

## Step 6 — First ingest

```bash
researchwiki agent ingest inbox/<file>.pdf
```

Summarize the new page (Summary section + key contributions in 2–3 sentences) and surface any memory-evolution proposals per Step 6 of the Ingest workflow in `CLAUDE.md`.

## Step 7 — Confirm

`researchwiki status` should show non-zero `Pages:`, zero `inbox/ PDFs awaiting ingest`, and `Structured DB: in sync`. Also confirm the semantic index built — `status` prints its model and page count; if it's missing, `reindex` was skipped or the embedding model isn't available (Step 1). `researchwiki migrate preflight` is the direct check.

If those hold, setup is done — tell the user what they can do next (drop more PDFs, ask cross-paper questions, file ideas). If they have an existing corpus of paper pages from an older wiki, point them at [`migration-backfill.md`](./migration-backfill.md).
