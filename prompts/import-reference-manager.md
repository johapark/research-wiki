# Importing a reference-manager library

Trigger: the user has an existing corpus in **Zotero, Paperpile, Mendeley,
ReadCube/Papers** or similar and wants it in the wiki. Signals: *"I have 500
papers in Zotero"*, *"can I import my library"*, a `.ris` / `.bib` / `.json`
export appearing in `inbox/`.

Not this file: markdown pages from an older wiki → [`migration-backfill.md`](./migration-backfill.md).
Bad metadata on a page that already exists → [`recovery.md`](./recovery.md).

---

## Why the export matters more than the PDFs

A reference manager already holds curated **DOI, title, authors and year**.
Those are exactly the fields `agent ingest` otherwise rediscovers through its
most failure-prone stretch — pypdfium2 extract → DOI hunt → S2 lookup →
LLM-reconcile → `metadata_sanity` — and the stretch that produces every
`unknown-` stem, `stem_year_drift`, and wrong-but-resolving DOI in
`recovery.md`. Supplying them turns that stretch into a lookup.

This matters more the larger the library. At one paper a wrong stem is a
two-minute fix. At 300 it is a cleanup project, because by then other pages
link to it.

**Never hand-write an import loop.** `agent ingest` refuses per-PDF flags in
batch mode, and fanning out one Bash call per file bypasses the checkpoint and
multiplies `state.db` contention (CLAUDE.md → Ingest). `import apply` is the
supported path: it passes per-record overrides through the batch runner.

---

## The four phases

```bash
researchwiki import preflight <export>                 # parse only, no PDFs
researchwiki import inspect   <export> [pdf-root]      # pair + triage
researchwiki import apply --run <dir> --limit 30       # copy + ingest a wave
researchwiki import verify --run <dir>                 # did it land?
```

`preflight` and `inspect` cost **zero tokens** and write nothing outside their
run directory under `.ingest/import-<stamp>/`. `apply` is the only phase that
spends money or writes pages.

**The manifest is the import's only durable record**, and `.ingest/` is
gitignored scratch that users are told to clear. It holds which PDF paired to
which record and why — the one artifact that can answer "was this page built
from the right file?" months later. For anything beyond a trial run, put it
somewhere that survives: `inspect --run-dir <path>`. Pages themselves carry no
import-provenance field, deliberately — see the note at the end of this file.

**Stage the import.** Reference libraries are aspirational — a large share is
skimmed-once PDFs that will never be cited, and the wiki's unit of value is a
graded, citable claim that costs real money and real review attention. Run
`--limit 30`, read a few pages, then decide about the rest. `apply` re-checks
per record whether a paper is already in the wiki, so a second `--limit 30`
takes the **next** thirty, not the same thirty.

---

## Step 1 — get the export

Ask for **BibTeX or RIS in preference to CSL-JSON**: only those two can carry
attachment paths. (Zotero: *File → Export Library*, tick "Export Files".)

Then locate the PDFs. This is where users get stuck, and the answer is
tool-specific — some keep a local store, some are cloud-only and need a bulk
download first. `inspect` runs fine without them, so do not block on it.

## Step 2 — `preflight`

Reports format, record count, DOI/title/year coverage, and the item-type
histogram. Read the DOI percentage: it predicts how much of the import will
pair cleanly, because the DOI printed in a PDF is the strongest pairing signal.

If it reports **"This export names no attachment paths"**, that is normal —
ReadCube and every CSL-JSON exporter omit them. Pairing falls back to content.

## Step 3 — `inspect`

The phase that earns the design. Pairs records to PDFs, runs every gate, writes
`manifest.json` and a human `report.md`. `--json` for programmatic use;
`--limit N` to sample a big library first.

**`<pdf-root>` is optional.** With no PDFs, the report still lists every record
that clears every gate except having a file, with its DOI — the fetch list. On
a cloud-hosted library that list *is* the deliverable.

Read `report.md` before applying. Three things deserve attention:

| Section | What to do |
|---|---|
| `review` items | Adjudicate by hand — each names its candidate PDFs and scores. This is free and tells you whether pairing is trustworthy before you spend. |
| `no-text-layer` | Scanned PDFs. OCR them or leave them out; importing one produces a page grounded on nothing. |
| Missing PDFs | The fetch list, emitted as plain DOIs so it can be piped. |
| Reference material | Books, guidance, theses the exporter typed as non-papers. They are real `wiki/references/` pages — hand-written, not ingested. Listed so a count doesn't become a dead end. |

## Step 4 — `apply`

```bash
researchwiki import apply --run <dir> --limit 30 --dry-run   # argv per paper
researchwiki import apply --run <dir> --limit 30             # spends money
```

`--run` is required — a bare `apply` silently choosing among several inspect
runs is a footgun, and `inspect` prints the exact command.

Dry-run first and read the argv: it shows the `--doi/--title/--authors/--year`
each paper will get, which is the whole point of importing from an export
rather than a folder.

If it crashes mid-wave, recover with `researchwiki agent ingest --resume
.ingest/batch-<stamp>/` — the batch runner owns the checkpoint.

## Step 5 — `verify`, then the free follow-ups

```bash
researchwiki import verify --run <dir>
researchwiki db rebuild && researchwiki reindex
researchwiki grade regression --no-salience      # free; claims are uncitable until graded
```

`verify` reports landed / sandboxed / not-yet-imported per record, plus the
`lint` keys an import can break. **Sandboxed** means the gates held a page in
`.agent-output/` — it is finished work awaiting review, not a failure.

`grade regression` without `--no-semantic`. That flag degrades grading to
BM25-only and every claim would need re-grading later; the salience pass is the
one worth skipping.

## Step 6 — wire the new pages into the graph

A bulk import arrives as N disconnected nodes, and nothing in the checks above
measures that. This step is the difference between a pile of pages and a wiki:

```bash
researchwiki claim-overlap --backlog --dry-run   # reciprocal links, LLM-judged
researchwiki audit --json                        # citation-graph gaps (needs S2)
researchwiki candidates concepts --bridges       # cross-category hub notes
researchwiki candidates synthesis                # dense clusters
```

Bridge concepts (span ≥ 2 categories) are the highest-leverage pages to write
after an import — they connect categories the citation graph doesn't.

---

## Cost

**Measure it on the first wave rather than quoting a figure.** README's
~$0.01/paper is for a config routing every role to the cheap model; a config
that sends `author`/`critic`/`judge` to a frontier model costs several times
that. After wave one:

```bash
researchwiki db query "SELECT model_used, SUM(cost_input_tokens), SUM(cost_output_tokens)
  FROM ingest_iterations WHERE created_at > strftime('%s','now','-6 hours') GROUP BY model_used"
```

against `config/pricing.yaml`. Multiply out before committing to the rest.
Observed on one real import: **$0.058/paper** on a mixed frontier/cheap config —
six times the README figure.

---

## What the gates mean

`inspect` assigns one of three verdicts, with the detail in reason strings.
`apply` acts on `ready` only.

| Reason | Verdict | Meaning |
|---|---|---|
| `no-text-layer` | skip | Scanned PDF, < 200 chars/page. **The silent one** — ingest would log a warning nobody reads and the page would pass every later gate on grounding that isn't there. |
| `superseded-by-journal` | skip | A preprint whose published version is also in the export. Invisible to DOI dedupe — same title, different DOIs. |
| `already-present` | skip | DOI or stem already in the wiki. Makes top-up imports idempotent. |
| `no-pdf` | skip | Metadata-only record. Never a failure; it lands on the fetch list. |
| `not-a-paper` | skip | Typed as book/webpage/thesis by the exporter. Only fires where the exporter populates `type`. |
| `pdf-unreadable` | skip | Encrypted or corrupt. |
| `weak-pairing` | review | Title matched below the confidence bar. |
| `ambiguous-pairing` | review | Matched confidently but *not distinctively* — another record scored nearly as well against the same PDF, so the score came from shared vocabulary rather than identity. |
| `unresolvable` | review | No DOI, no author, no year. In practice: books and notes the exporter typed as journal articles. |
| `thin-metadata` | review | No DOI and either no title or no year — nothing to override with. |
| `maybe-commentary` | review | Nature news/comment DOI prefix, or a one-page PDF with no DOI. See CLAUDE.md → Page Types §7. |
| `stem-collision` | review | Two importable records derive one stem. Which keeps the bare year is a judgement about the corpus. |

A record with **no DOI, author or year but a valid DOI** is still `ready` — a
DOI alone is a sufficient override, and reconcile resolves the rest.

---

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `cannot identify the export format` | not BibTeX/RIS/CSL-JSON, or truncated | check the file actually exported |
| Everything is `no-pdf` | `<pdf-root>` omitted or wrong | pass the directory holding the PDFs |
| Many `weak-pairing` | PDFs whose first page is a masthead, pushing the title past the match window | usually recovered by the DOI rung; adjudicate the rest by hand |
| Many PDFs "matched no record" | duplicate copies, or files whose record is paired from another copy | compare counts: unclaimed PDFs ≫ unpaired records means duplicates, not misses |
| `apply` exits 2 | embedding model unusable | see the install table in [`migration-backfill.md`](./migration-backfill.md); grading depends on it |
| A page has unparseable frontmatter | a field containing `: ` written unquoted | `lint`'s `invalid_frontmatter`; fix in place, don't re-ingest — the prose is fine |

## Don't use this for

- **Notes and annotations.** A user's own highlights aren't PDF-grounded prose
  under Rule 1 and have no home in the page contract.
- **Records with no PDF.** Rule 3 makes it non-optional: no PDF → no claims →
  a page that can never ground a citation. They belong on the fetch list.
- **Books and guidance documents.** Those are `wiki/references/` pages written
  by hand (CLAUDE.md → Page Types §3), not ingested.

---

## Why pages carry no import-provenance field

A page produced by an import looks exactly like a page produced by a normal
`agent ingest`, and that is on purpose.

The tempting field is something like `imported_from: zotero`. It was considered
and rejected for three reasons. The frontmatter contract is a
breaking-change surface (`CHANGELOG.md`), so a field is cheap to add and
expensive to remove. Nothing branches on it — no gate, no query, no workflow
would read it. And the precedent is explicit: `tags:` was *removed* from paper
pages because provenance-in-frontmatter proved to be dead weight, with
`ingested-via-agent` the only tag on 334 of 391 pages.

The question it would answer — *"the import may have mis-paired some PDFs;
which pages are suspect?"* — is already answerable three ways, each more precise
than a boolean flag:

- `ingested_at` clusters tightly: one wave lands inside a few minutes.
- The **manifest** records the exact PDF, pairing rung and confidence per record.
- `ingest_iterations` in the state DB keeps `pdf_filename`, which for an import
  is the exporter's own name (`Nature-2026.3.pdf`) rather than a normal drop.

So the provenance exists; it just doesn't live in the page. Keep the manifest
(`--run-dir`) and you keep the answer.
