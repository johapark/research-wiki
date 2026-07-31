# Contributing

Thanks for looking at this. It's a solo-maintained alpha project, so please open
an issue to discuss anything substantial before writing code — it may already be
intentional, or deliberately out of scope.

## Setup

```bash
git clone https://github.com/johapark/research-wiki.git && cd research-wiki
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q          # should be green before you change anything
```

Optional extras: `.[mcp]` for the read-only MCP server. The semantic grader pulls
`sentence-transformers` and downloads a ~133 MB model on first use — not needed
for the test suite.

A fresh clone has no `config/models.yaml` (it's gitignored, per-user). The loader
falls back to hardcoded defaults, so the CLI works immediately; copy one of the
`config/models.*.yaml` templates when you want to pick a backend.

## Tests

**The suite is hermetic and must stay that way**: no network calls, no LLM calls,
no model downloads. That's what keeps it a few seconds and lets CI run without
secrets. CI runs it on Python 3.10 / 3.11 / 3.12.

When touching an LLM path, monkeypatch `llm.call` (see
`tests/test_llm_retry.py`, `tests/test_keywords_parse_failure.py`) or use the
`--stub` provider. Never let a test reach a real endpoint.

Bug fixes should come with a regression test that **fails before the fix**. If a
test guards an invariant, verify it can actually fail — temporarily introduce the
violation and confirm it's caught. A guard that can't fail isn't a guard.

## Conventions that will trip you up

These are enforced by tests, and each exists because of a real bug:

**Phase functions are `verb_object`; modules keep topic names.**
`grade.py` defines `grade_draft()`, `reconcile.py` defines `reconcile_metadata()`.
The rule exists so no re-exported name equals a submodule name — otherwise the
function shadows the module and `phases.grade` silently becomes a function while
`phases.draft` stays a module. `tests/test_phases_namespace.py` enforces this as
an invariant.

**Register new LLM phases in `_FALLBACK_PHASES`** (`agents/model_config.py`).
An unregistered phase raises `PhaseNotRegistered`, and callers that tolerate
transient failures would otherwise turn it into a silent no-op.
`tests/test_phase_registration.py` scans every `phase="..."` literal and fails CI
if one isn't registered.

**Never rename phase-name *strings*.** The values in `db/iterations.py`
`VALID_ROLES`, `role=` arguments, `phase=` model-config keys, and the `reconcile:`
style keys in `config/models.*.yaml` are persisted or user-facing. They're
deliberately decoupled from the Python function names. `ingest_iterations` is the
one table `db rebuild` **cannot** regenerate, so renaming a role orphans all
historical cost and analytics data.

**Fail loudly, not emptily.** Don't return `[]`, `{}`, or `None` on a parse or
config failure without logging it. Silent degradation here once left 38% of the
corpus missing required metadata, because a malformed LLM response looked
identical to "nothing to report". Log it, retry when a retry could plausibly
help, and point the operator at the remediation command.

**Exit codes** — `0` success (zero results still counts), `1` user-input error,
`2` environment error (missing index, provider unreachable), `3` internal bug.
Prefer a `--json` mode on anything an agent might parse.

## Content vs framework

`wiki/`, `papers/`, and `inbox/` are **gitignored user content** — someone's
personal library. Never commit them, and stage named paths (`git add <path>`)
rather than `git add -A`.

If you add a benchmark fixture PDF under `benchmark-fixtures/pdfs/`, it **must**
be openly licensed (the existing five are CC-BY-4.0) and recorded with full
attribution in `benchmark-fixtures/LICENSES.md`. Do not add publisher PDFs that
aren't redistributable.

## Docs to keep in sync

- **`CLAUDE.md`** — the behavioural contract for LLMs operating in the repo
  (grounding rules, page types, ingest/query workflows). Update it when you
  change how pages are authored or verified.
- **`AGENTS.md`** — a **symlink to `CLAUDE.md`**, not a separate document. Tools
  that auto-load `AGENTS.md` (Codex CLI, Cursor, Aider, Gemini CLI, …) don't
  follow markdown links out of it, so they'd otherwise run without the
  grounding rules. Edit `CLAUDE.md`; never replace the symlink with a file.
- **`WORKFLOW.md`** — end-to-end pipeline walkthrough and module map. Update the
  module map when you add or move a module.

## Commits and PRs

Conventional commits with a scope, matching the existing history:

```
fix(llm): retry transient 401s from OpenAI-compatible endpoint
refactor(phases): rename phase functions to verb_object
docs(workflow): correct citation guidance
```

Explain **why** in the body, not just what — the history is the main design
record here. Keep PRs focused; run `python -m pytest -q` before pushing.

## Licensing

Contributions are accepted under the [MIT License](./LICENSE), the project's
license. See the README's License section for the CC-BY-4.0 carve-out covering
bundled benchmark PDFs.
