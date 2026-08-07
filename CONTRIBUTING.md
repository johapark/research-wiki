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

**If you sync `wiki/` and `papers/`** (iCloud/Dropbox — the maintainer does), keep the checkout itself *outside* the synced folder and symlink those two dirs in. Both are gitignored, so git and the sync service have disjoint jobs and neither has to carry the other's content. If your checkout does live in a synced folder, at minimum put the venv elsewhere: `python3 -m venv ~/.venvs/research-wiki`. A venv is ~34,000 files and will dominate the sync queue, at which point the daemon starts losing races against your own writes — a stale `.git/index` that reports every tracked file as modified, and `<name> 2.py` conflict copies of files you just edited. See [`prompts/migration-backfill.md`](./prompts/migration-backfill.md#keep-the-checkout-out-of-the-synced-folder) § *Keep the checkout out of the synced folder*.

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
- **`CHANGELOG.md`** — add user-visible changes under `## [Unreleased]` as you go,
  not in a scramble at release time. Internal refactors that change nothing for a
  user don't need an entry.

## Commits and PRs

Conventional commits with a scope, matching the existing history:

```
fix(llm): retry transient 401s from OpenAI-compatible endpoint
refactor(phases): rename phase functions to verb_object
docs(workflow): correct citation guidance
```

Explain **why** in the body, not just what — the history is the main design
record here. Keep PRs focused; run `python -m pytest -q` before pushing.

## Releasing

**The version lives in exactly one place**: `__version__` in
`researchwiki/__init__.py`. `pyproject.toml` declares it `dynamic` and resolves it
from there, so a built artifact and `researchwiki --version` can't disagree.
Re-adding a `version = "…"` literal to `pyproject.toml` fails
`tests/test_version.py`.

### What counts as breaking

The public surface here isn't a Python API — everything under `researchwiki/` is
internal, and renaming a function is a PATCH. What consumers actually depend on:

| Surface | Breaking change |
|---|---|
| CLI | Removing or renaming a command or flag |
| Exit codes | Changing what a code means (see the table in CLAUDE.md) |
| `--json` contracts | **Removing a key** — agents parse `lint --json`, `search --json`, `audit --json`. Adding one is MINOR |
| Page frontmatter | Removing or repurposing a field the CLAUDE.md contract specifies |
| Phase-role strings | `db/iterations.VALID_ROLES` and the `phase=`/`role=` keys — never rename these at all; `ingest_iterations` is the one table `db rebuild` can't regenerate |

While on `0.x`, a breaking change takes the **MINOR** slot (0.2.0 → 0.3.0) and says
so at the top of its changelog entry.

### Cutting a release

1. **Pick the number** from the commits since the last tag —
   `git log v<prev>..HEAD --format='%s'`. Any breaking change per the table above →
   MINOR (while 0.x); any `feat` → MINOR; otherwise PATCH.
2. **Promote the changelog.** Rename `## [Unreleased]` to
   `## [x.y.z] - YYYY-MM-DD`, open a fresh empty `## [Unreleased]` above it, and add
   the compare link at the bottom. Entries are curated prose, not generated subject
   lines — the *why* lives in commit bodies and a generator discards it.
3. **Bump** `__version__` in `researchwiki/__init__.py`.
4. **`python -m pytest -q`.** `tests/test_version.py` fails unless steps 2 and 3
   agree, which is the point: the bump and its notes ship together.
5. **Commit** as `chore(release): vX.Y.Z`, then tag it:
   `git tag -a vX.Y.Z -m "researchwiki vX.Y.Z"` — annotated, matching `v0.1.0`.
6. **Push the commit, then the tag.** `git push origin main && git push origin vX.Y.Z`.

Pushing the tag fires `.github/workflows/release.yml`, which runs the full test
matrix, re-checks that the tag equals `__version__`, extracts that version's
changelog section as the release notes, builds an sdist + wheel, and creates the
GitHub Release. Any of those failing means no release is published — so a bad tag
costs you a `git tag -d` and a re-push, not a bad artifact.

**Not published to PyPI.** Releases are installable from a tag or a clone; there is
no package-index presence to keep in sync.

## Licensing

Contributions are accepted under the [MIT License](./LICENSE), the project's
license. See the README's License section for the CC-BY-4.0 carve-out covering
bundled benchmark PDFs.
