# Recategorize — move a paper to a different category

**Trigger:** moving `<stem>` from `wiki/<old>/` to `wiki/<new>/`. Target dir must already exist (see Categories in CLAUDE.md).

The **directory is the canonical category** — `db rebuild` derives a page's category from its parent dir and ignores YAML `category:`.

1. `mv wiki/<old>/<stem>.md wiki/<new>/<stem>.md`. The PDF in `papers/` does **not** move (`pdf_path` is unchanged).
2. Update the page's own YAML to `category: [<new>]`. Directory wins, but a stale field misleads anyone reading the source or Obsidian's property view — and `lint` flags it (see step 5).
3. **Repoint every inbound full-path link** `[[<old>/<stem>]]` → `[[<new>/<stem>]]` across `wiki/` (`grep -rl '<old>/<stem>' wiki/`), and move the paper's line in `index.md` to the new category section. Bare-stem links `[[<stem>]]` resolve regardless of category and need no change.
4. `researchwiki db rebuild && researchwiki reindex`.
5. `researchwiki lint --json` → confirm **`broken_wikilinks` and `category_yaml_drift` are both empty**. The first catches a missed inbound link (a stale `[[<old>/<stem>]]`); the second catches a missed YAML update in step 2.
6. Append a `## [YYYY-MM-DD] recategorize | <stem> (<old> → <new>)` line to `log.md`.
