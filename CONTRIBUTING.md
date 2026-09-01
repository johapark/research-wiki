# Contributing

Thanks for looking at this. It's a solo-maintained alpha project, so please open
an issue to discuss anything substantial before writing code — it may already be
intentional, or deliberately out of scope.

## Setup

```bash
git clone https://github.com/johapark/research-wiki.git && cd research-wiki
python3 -m venv ~/.venvs/research-wiki                    # outside the repo — see below
~/.venvs/research-wiki/bin/pip install -e '.[dev,mcp]'
~/.venvs/research-wiki/bin/python -m ruff check .
~/.venvs/research-wiki/bin/python -m pytest -q            # green before you change anything
```

**The venv goes outside the checkout.** A venv is tens of thousands of files,
and this one carries roughly 2 GB of platform-specific wheels and installed
packages. In-repo `.venv/` is gitignored so git doesn't care — but if the tree
is ever synced it dominates the sync load, and the daemon starts losing races
against your own writes: a stale `.git/index` reporting every tracked file as
modified, `<name> 2.py` conflict copies of files you just edited. Venvs also bake
absolute paths into `bin/`, so relocating one later means recreating it.

**If you sync `wiki/` and `papers/`** (iCloud/Dropbox — the maintainer does), also keep the checkout itself *outside* the synced folder and symlink those two dirs in. Both are gitignored, so git and the sync service have disjoint jobs and neither has to carry the other's content. See [`prompts/migration-backfill.md`](./prompts/migration-backfill.md#keep-the-checkout-out-of-the-synced-folder) § *Keep the checkout out of the synced folder*.

**The install is not small.** torch, transformers and sentence-transformers are
*core* dependencies, not extras — the claim grader is part of the default
product — so expect ~2 GB. Its model weights are a separate ~133 MB download on
first use, which the test suite doesn't need.

`[mcp]` is in the command above to match CI. Without it
`tests/test_mcp_serve.py` `importorskip`s and you'd be passing a smaller suite
than CI runs — which is exactly how a broken `mcp-serve` once shipped unnoticed.

A fresh clone has no `config/models.yaml` (it's gitignored, per-user). The loader
falls back to hardcoded defaults that are **OpenAI-compatible**, so
`OPENAI_API_KEY` alone gets you a working CLI; copy one of the
`config/models.*.yaml` templates to pick any other backend — including Anthropic,
which is not the zero-config path.

## Tests

**The suite is hermetic and must stay that way**: no network calls, no LLM calls,
no model downloads. That keeps it fast enough for routine local runs and lets CI
run without secrets. CI runs it on Python 3.10 through 3.14.

CI also runs Ruff's correctness checks (`E9`, selected `F` rules, `B023`) and a
McCabe ceiling (`C901`, maximum 40) across the package and tests. Run
`python -m ruff check .`
before the full suite.

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

**Keep modules under 800 lines of code.** Past some volume of logic a module is
doing more than one job, so this is a cohesion budget rather than a style
preference — `tests/test_module_size.py` fails the build when a new file crosses
it.

**Code, not lines**: docstrings, comment-only lines and blanks don't count, and
that is the point rather than a technicality. This package is ~57% code and ~29%
prose, so a physical-line budget taxes documentation, and it did: `agents/llm.py`
was flagged at 902 lines while holding 470 of code (well explained, not complex)
while `tasks/lint/report.py` passed at 603 lines holding 549. Write as much
explanation as the decision deserves; the budget only counts what you have to
hold in your head. A multi-line string bound to a name still counts, so the rule
can't be dodged by moving code-adjacent text into a constant.

**Two bounds, not one.** The 800 ceiling says how large any module may ever be,
and it is permissive because the metric is what makes the gate rank modules
honestly. The `_DEBT` dict in that test is a separate ratchet: nine modules are
pinned **at their current size** (502-817 code lines) and may shrink freely but
cannot gain a line — whether or not they sit under the ceiling. A bare exemption
list is how `agents/runner.py` reached 817 with nothing objecting.

The two retire on different rules, which matters: a pin is released when its
module drops below `RATCHET_RELEASE` (500), *not* when the ceiling moves. Keying
retirement on the ceiling meant raising it from 500 to 800 deleted eight pins and
handed those modules ~1,700 lines of unratcheted growth, though none had shrunk
by a line.

Split cohesive groups into focused siblings rather than adding an entry — the
list is for existing debt, not an escape hatch for new code. When a module drops
below the release point, delete its entry (a test insists, because a stale
exemption is indistinguishable from a passing gate).

**Keep functions under 250 lines of code and McCabe complexity at 40 or less.**
The same test applies the code-line metric per function and exact-ratchets any
inherited exception; Ruff enforces complexity. A coordinator that crosses either
limit should delegate cohesive collection, validation, or rendering phases.

## Content vs framework

`wiki/`, `papers/`, and `inbox/` are **gitignored user content** — someone's
personal library. Never commit them, and stage named paths (`git add <path>`)
rather than `git add -A`.

If you add a benchmark fixture PDF under `benchmark-fixtures/pdfs/`, it **must**
be openly licensed (the existing four are CC-BY-4.0) and recorded with full
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
- **`prompts/*.md` + their CLAUDE.md pointers** — a trigger-gated procedure is
  only reachable through the sentence in CLAUDE.md that gates it, so a prompt
  whose trigger doesn't fire may as well not exist. After editing a trigger (or
  adding a prompt), run `researchwiki eval triggers --slug <slug>`: it generates
  should-fire / should-not-fire requests, routes them using only the triggers,
  and names the misses. `--dry-run` prices the run and spends nothing.
  Costs tokens, so it's on-demand — not part of `pytest`. The free half (a prompt
  no pointer reaches, a pointer with no file) is reported by `researchwiki lint`
  as `orphan_prompts` / `broken_prompt_pointers`.

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
| `--json` contracts | **Removing a key** — agents parse `lint --json`, `search --json`, `scout --json` (and its deprecated `audit --json` alias). `scout web` artifacts are separately versioned with `schema_version`; incompatible schema changes require a new version. Adding one is MINOR |
| Page frontmatter | Removing or repurposing a field the CLAUDE.md contract specifies |
| Phase-role strings | `db/iterations.VALID_ROLES` and the `phase=`/`role=` keys — never rename these at all; `ingest_iterations` is the one table `db rebuild` can't regenerate |

From beta onward, replacements also follow [COMPATIBILITY.md](./COMPATIBILITY.md):
deprecated CLI/JSON/frontmatter surfaces coexist for at least 90 days, through
the next minor line, and cannot be removed before the second subsequent minor.
The machine-readable inventory is `researchwiki/data/deprecations.yaml`.

### Which slot, while on 0.x

**SemVer does not govern this project yet, and saying otherwise caused a bug in
this section.** Clauses 6, 7 and 8 of SemVer 2.0.0 — the PATCH, MINOR and MAJOR
rules — are each guarded `| x > 0`. At `0.y.z` the only clause that applies is 4:
*"Anything MAY change at any time."* What follows is therefore a house convention
filling a gap the spec leaves open on purpose, not a reading of the spec.

The convention is the usual 0.x one: **the whole line shifts down a slot.**

| Change | 0.x slot | at 1.0+ |
|---|---|---|
| Breaking (removes/renames a surface in the table above) | **MINOR** — 0.4.x → 0.5.0 | MAJOR |
| Additive (new command, flag, `--json` key, frontmatter field, page type) | **PATCH** — 0.4.0 → 0.4.1 | MINOR |
| Everything else (diagnostics, warnings, error messages, prompt edits, heuristics, internal refactors) | **PATCH** | PATCH |

A breaking change says so at the top of its changelog entry.

**Shift the whole line or none of it.** An earlier version of this section moved
breaking changes to MINOR but left additions there too, so MINOR carried both
meanings and `0.4.0 → 0.5.0` could no longer tell a consumer whether something
broke or something was added — which is the one signal the shift exists to
preserve. It also fought the packaging tools: `^0.4.0` (npm, cargo) and `~=0.4.0`
(pip) both mean *0.4.z only*, so a purely additive release under the old rule
broke every caret pin for nothing.

What has not changed is that the **table decides**, not the commit label. A `feat`
that adds no listed surface is a PATCH; so is a `fix` that adds none. Check what
the diff exposed, not what the subject line called it.

Both rules here replace "any `feat` → MINOR", borrowed from Conventional Commits.
That keyed the version to a label chosen per commit, before anyone knew what the
release would contain, and it over-fired on precisely the shape it couldn't see:
the `feat` in the 0.3.1 range added 96 lines to one internal module, two stderr
warnings, and no public surface whatsoever. Deciding from the table keeps the
choice mechanical, with one arbiter and no per-release argument about whether a
feature was big enough.

### Cutting a release

1. **Pick the number** from the commits since the last tag —
   `git log v<prev>..HEAD --format='%s'`, read against the table above. While 0.x:
   breaking → MINOR, everything else (additions included) → PATCH. Commit types are
   evidence, not the rule: check what the diff added, not whether the subject said
   `feat`.
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
