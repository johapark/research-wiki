# `refimport` fixtures

Bibliographic exports for the `researchwiki import-library` parser tests.

**Every record here is synthetic.** The quirks are real — each one was observed
in a 532-item ReadCube library exported as both RIS and BibTeX — but the papers,
authors and DOI suffixes are invented. `wiki/` and `papers/` are gitignored
because this repo is public and a personal corpus is not; excerpting real library
records into `tests/` would commit part of one. DOI suffixes sit under real
registrant prefixes (`10.1101/` for bioRxiv, `10.1038/d41586-` for Nature
news/comment) **only** where a prefix is the thing under test, and none resolve.

## What each fixture covers

| File | Format | Records |
|---|---|---|
| `readcube-sample.ris` | RIS, **CRLF** | 12 |
| `readcube-sample.bib` | BibTeX | 12 |
| `zotero-sample.json` | CSL-JSON | 4 |

The RIS and BibTeX files describe the *same* 12 items, so a test can assert the
two parsers agree — the real exports did, field for field.

`zotero-sample.json` is deliberately small and exists for one thing the ReadCube
files cannot test: a **populated `type` field**. Zotero types a book as `book`
and a blog post as `webpage`, so the typed non-paper gate actually fires. In the
ReadCube exports 531 of 532 items are `JOUR`/`@article` — books included — which
is why that gate is kept but never relied on.

### Quirks, and why each is here

| Quirk | Where | Why it matters |
|---|---|---|
| CRLF line endings | `.ris` (whole file) | Python text-mode reads translate newlines, so splitting on a literal `"\r\n"` yields **one** record. Silent whole-file failure. |
| 4-character `PMID` tag | `.ris` records 1–2 | RIS tags are conventionally 2 chars. A fixed-width parser drops it, or worse, mis-slices the line. |
| Junk `XX  - ` tag, always empty | `.ris` records 1–2 | 385 of 532 real records carry it. Unknown tags must be ignored, not fatal. |
| Wrapped continuation line | `.ris` record 11 | Dropping the continuation truncates a title, which silently changes the derived stem. |
| `L1` attachment path | `.ris` record 12 | The only rung-1 exercise: ReadCube writes no file paths at all. |
| Citekey containing `:` | `.bib` several | 55 of 532 real citekeys. Illegal in strict BibTeX — a validating parser rejects the whole file. |
| Citekey containing non-ASCII | `.bib` `grünewald…` | 16 of 532. Same. |
| LaTeX escapes in a title | `.bib` record 9 (`\'{e}lan`) | ReadCube emits raw Unicode and *no* escapes, but Zotero/JabRef do. `latex.py` exists for those. |
| Brace-protected word in a title | `.bib` record 9 (`{CRISPR}`) | The value scanner must handle a nested brace group without ending the field early, and the braces must not survive into the title. |
| `file = {desc:path:mime}` | `.bib` record 12 | BBT's triple format for rung 1. |
| U+2010 HYPHEN in a title | both, record 2 | Folds to ASCII `-` in `stems.strip_diacritics`; unfolded it welds `ATAC-seq` into `atacseq`. 15 of 516 real stems. |
| Suspended compound (`epigenome- and`) | both, record 3 | Left a dangling `-` in the stem. 3 of 516 real stems. |
| Preprint + published, identical title | both, records 4–5 | **10 such pairs in 532 real records, and 0 duplicate DOIs** — title-level dedupe is the only thing that finds them. |
| No author/year but a DOI | both, record 6 | Still importable: the DOI alone is a sufficient override. 11 of 532. |
| No DOI, no author, no year | both, record 7 | Genuinely unresolvable. 5 of 532. |
| Book typed as `JOUR`/`@article` | both, record 7 | 531 of 532 real records are `JOUR`, including two books — proof the type field cannot be trusted for triage. |
| `10.1038/d41586-` prefix | both, record 8 | Nature news/comment. A free, precise commentary signal needing no network call. |
| Non-ASCII + Unicode-dash surname | both, record 9 | `Grünewald-López` must survive as `grunewald-lopez`. |
| `ELEC` / `@misc` type | both, record 10 | The one non-`JOUR` record in the real export. |

## Regenerating

These are hand-written and checked in. If you extend them, extend **both** the
RIS and the BibTeX so the cross-format agreement test keeps meaning something,
and add a row above saying what the new record is for.
