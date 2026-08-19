# Export the corpus — as a bibliography, or as an OKF bundle

**Trigger:** the user wants their wiki library in a reference manager, or a `.bib` for a manuscript in progress. Signals: *"can I get a bib file"*, *"export my library"*, *"put this in Zotero"*, *"I'm writing a paper and want the references"*. Also *"export as OKF"* / *"make a portable bundle"* — a different output with a different scope, covered in its own section below.

**Not this file:**
- [`share-page.md`](./share-page.md) — one synthesis/idea page → a document for a human reader.
- [`import-reference-manager.md`](./import-reference-manager.md) — the inverse: a manager's export → wiki pages.

Zero tokens, no network, deterministic. Nothing is written unless you pass `--out`.

## Which format

| Target | Command |
|---|---|
| LaTeX manuscript | `researchwiki export --format bibtex --out refs.bib` |
| Zotero / Mendeley / EndNote | `researchwiki export --format ris --out library.ris` |
| Anything CSL-aware, or further processing | `researchwiki export --format csl-json --out library.json` |
| A portable bundle of the whole wiki, analysis included | `researchwiki export --format okf --out bundle/` |

Filters: `--category`, `--year YYYY|YYYY-YYYY`, `--stem` (all repeatable except `--year`). Without `--out` the bibliography goes to stdout and the summary to stderr, so `researchwiki export --category cgt > cgt.bib` gives a clean file.

## Cite with the stem

The citekey **is** the page stem — `\cite{bae-2014-cas-offinder-a-fast-and-versatile}`. Long, and deliberately so: it is the only key that cannot change under you. Any short scheme (`bae2014`) has to disambiguate collisions with a letter suffix, and that suffix is recomputed at export time — so ingesting one more 2026 Wang paper would renumber keys already sitting in a manuscript. The stem never changes, per CLAUDE.md § *Disambiguation & updates*.

## What is not in there, and why

*(This section is about the bibliography formats. OKF's scope is different — see below.)*

Synthesis, idea and concept pages are **excluded from the bibliography and there is no flag to include them.** They are the user's own unpublished analysis: no DOI, no venue, no year of record. An entry for one would assert, once pasted into a manuscript, a publication that does not exist — a citation-integrity problem rather than a formatting one. If the user wants to share their analysis, that is `share-page.md`.

Commentary pages *are* included. They are real publications a reference manager should hold — but remember CLAUDE.md's rule that a commentary is not citable evidence: cite the primary paper it discusses.

No `volume`, `issue`, `pages`, `publisher`, `issn` or `abstract` is emitted, because no page carries them. Getting them would mean a Crossref/S2 lookup, which would make the command non-deterministic and network-dependent.

## `--format okf` — the other output

Open Knowledge Format is a portable bundle, not a bibliography, and its unit is a
*concept* — explicitly including abstract ideas with nothing published behind them.
So it carries **every** page type, synthesis / idea / concept included, and simply
omits `resource` where there is no publication. A page absent from the `.bib` and
present in the bundle is correct in both; don't try to align them.

```bash
researchwiki export --format okf --out bundle/          # --out is mandatory
researchwiki export --format okf --out bundle/ --json   # the report as well
```

- **`--out <dir>` is required** — a bundle is a directory tree whose file paths
  *are* the concept identities, so there is no stdout form. A non-empty directory
  that isn't already one of its bundles is refused, exit 2.
- **`--json` returns a different shape** from the bibliography's report:
  `{format, okf_version, concepts, by_type, links_rewritten, links_unresolved,
  sources_emitted, verified_emitted, verified_absent_no_gate_record,
  generated_missing_actor, description_missing, skipped, stale_files}`. Dispatch on
  `format` rather than assuming the bibliography keys.
- **`verified` is emitted only for graded paper pages.** `check-grounding` and
  `grade synthesis` persist nothing, so a synthesis page has no gate record to point
  at and gets no trust claim — asserting one would be exactly the falsehood those
  gates exist to prevent. `verified_absent_no_gate_record` counts them.
- `links_unresolved` and `description_missing` are page-defect lists like the
  bibliography's, not statistics: an unresolved link is a `[[wikilink]]` with no page
  behind it, and a missing description is a page with no `hook:`.

Field mapping worth knowing: `hook:` → `description`, `doi:` → `resource`,
`tags:` ∪ `keywords:` → `tags`, `author_model:` + `ingested_at:` → `generated`,
footnote `[^id]:` / `referenced_papers:` → `sources[]` (the footnote label *is* OKF's
`sources[].id`, so per-claim attribution survives), idea `status:` → `draft|stable|deprecated`
with the native value kept in `x_researchwiki_status`. Unmapped frontmatter is preserved
under `x_researchwiki_*` rather than dropped.

## Read the report, then go fix the pages

`--json` prints the report instead of the bibliography (combine with `--out` to get both). These are the *bibliography* report's keys — OKF's are above. Each is a to-do list, not just a statistic:

| Key | What it means | What to do |
|---|---|---|
| `venue_furniture` | A `venue:` that is typesetting furniture (`Journal of LaTeX Class Files`, `preprint`). Suppressed from output — the one place this command could print a falsehood. | Fix the page's `venue:`; `researchwiki lint` reports these as `venue_suspect`. |
| `venue_missing` | A paper with no `venue:`. Emitted as `@misc` rather than an `@article` with no `journal`. | Fix the page's `venue:` if the paper has one. Preprints legitimately have none. |
| `doi_missing` | No usable DOI. Emitted anyway — no format requires one. `reason` carries `no_doi_reason` when the page explains itself. | A `null` reason on a `type: paper` page is worth `researchwiki backfill doi`. |
| `authors_unparseable` | The `authors:` field is prose, not a name list, so no author field was emitted. | Rewrite the page's `authors:` as comma- or semicolon-separated names. |
| `skipped` | An exportable page that produced no entry (in practice: no `title`). | Fix the page. |

`records + len(skipped)` always equals the number of pages selected, so nothing goes missing silently.

## Notes

- **Output is UTF-8** and Unicode is emitted as-is, not as LaTeX macros. biber, XeLaTeX, LuaLaTeX, Pandoc and Zotero all read it. A pdfLaTeX-only pipeline with bare `bibtex` needs `\usepackage[utf8]{inputenc}`, or switch to biber.
- **Titles are brace-protected per word** (`{CRISPR-Cas9}`), because `plain`/`unsrt`/`abbrv` lowercase titles and would otherwise emit `Crispr-cas9`. Over half the corpus needs this.
- **Author names are passed through unmodified** for BibTeX and RIS, which parse `First von Last` themselves. Only CSL-JSON gets a structured `family`/`given`, and only where the split is unambiguous — a corporate or consortium name becomes a CSL `literal`.
- Two runs produce byte-identical output, so a `.bib` can live in version control and diff meaningfully.
- Nothing is appended to `log.md`: that is for operations that mutate the corpus.
