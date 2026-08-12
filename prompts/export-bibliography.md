# Export the corpus as a bibliography

**Trigger:** the user wants their wiki library in a reference manager, or a `.bib` for a manuscript in progress. Signals: *"can I get a bib file"*, *"export my library"*, *"put this in Zotero"*, *"I'm writing a paper and want the references"*.

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

Filters: `--category`, `--year YYYY|YYYY-YYYY`, `--stem` (all repeatable except `--year`). Without `--out` the bibliography goes to stdout and the summary to stderr, so `researchwiki export --category cgt > cgt.bib` gives a clean file.

## Cite with the stem

The citekey **is** the page stem — `\cite{bae-2014-cas-offinder-a-fast-and-versatile}`. Long, and deliberately so: it is the only key that cannot change under you. Any short scheme (`bae2014`) has to disambiguate collisions with a letter suffix, and that suffix is recomputed at export time — so ingesting one more 2026 Wang paper would renumber keys already sitting in a manuscript. The stem never changes, per CLAUDE.md § *Disambiguation & updates*.

## What is not in there, and why

Synthesis, idea and concept pages are **excluded and there is no flag to include them.** They are the user's own unpublished analysis: no DOI, no venue, no year of record. An entry for one would assert, once pasted into a manuscript, a publication that does not exist — a citation-integrity problem rather than a formatting one. If the user wants to share their analysis, that is `share-page.md`.

Commentary pages *are* included. They are real publications a reference manager should hold — but remember CLAUDE.md's rule that a commentary is not citable evidence: cite the primary paper it discusses.

No `volume`, `issue`, `pages`, `publisher`, `issn` or `abstract` is emitted, because no page carries them. Getting them would mean a Crossref/S2 lookup, which would make the command non-deterministic and network-dependent.

## Read the report, then go fix the pages

`--json` prints the report instead of the bibliography (combine with `--out` to get both). Each key is a to-do list, not just a statistic:

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
