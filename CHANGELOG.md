# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**What counts as a breaking change here** is not a Python API — it's the CLI surface,
the exit-code contract, the `--json` key contracts, the page frontmatter contract, and
the persisted phase-role strings. See [`CONTRIBUTING.md` § Releasing](./CONTRIBUTING.md#releasing)
for the policy and the bump procedure.

Entries are curated, not generated: commit bodies in this repo carry the design
rationale, and a subject-line dump would throw exactly that away. Read `git log` for
the reasoning behind any line below.

## [Unreleased]

### Fixed

- `agent ingest` now checks that the configured provider has usable credentials
  *before* extracting the PDF and reconciling metadata, instead of discovering it
  in the author phase at the end of that work. The old failure was the worst one
  a new user could hit on their first command: an uncaught `RuntimeError` escaped
  as exit **3** — "internal bug, file a report" — for what is a plain
  configuration error, and because `call_openai_compatible` substitutes the
  literal string `lm-studio` for an unset `OPENAI_API_KEY`, the diagnostic read
  `Incorrect API key provided: lm-studio`, quoting a value the user had never
  typed. It is now exit **2** with the missing variable, the endpoint that needs
  it, and the config file actually in force.

  The check resolves the endpoint through the same precedence as `llm.call`
  (`RW_LLM_BASE_URL` → the config's `base_url:` → the LM Studio default), so a
  preflight verdict can't disagree with what the call site would do. Loopback
  endpoints and `chat-relay` need no credentials and pass; `--stub` skips the
  check entirely. It is deliberately stricter than `has_synchronous_llm()`,
  which answers "is any key set anywhere" and so waves through the case README
  and `prompts/init.md` both name as the one that actually happens — an
  Anthropic key set, the config copy skipped, every role still routed to
  OpenAI. That case now fails with the `cp config/models.anthropic.yaml` hint.

  Argv is still validated first: a path that doesn't exist remains exit 1, so a
  typo isn't reported as missing credentials.

## [0.3.0] - 2026-08-11

### Added

- `researchwiki import`, for the corpus most new users actually arrive with:
  an existing library in Zotero, Paperpile, Mendeley or ReadCube. Four phases —
  `preflight` → `inspect` → `apply` → `verify` — of which only `apply` spends tokens
  or writes pages; the rest write nothing outside their run directory. `migrate`
  explicitly refuses this case — it imports markdown pages from an older wiki, while
  these users have PDFs and metadata and no prose — so until now they were told what
  the tool was *not* for and given nowhere to go.

  The export is the asset. A reference manager carries curated DOI, title, authors
  and year, which is exactly what `agent ingest` otherwise rediscovers through its
  most failure-prone stretch (PDF extract → DOI hunt → S2 lookup → LLM reconcile →
  `metadata_sanity`) and the stretch that produces every `unknown-` stem and
  wrong-but-resolving DOI. `inspect` records the exact `--doi/--title/--authors/
  --year` argv each record contributes, so importing becomes a lookup rather than a
  rediscovery.

  Parsing is tolerant by design, because the files that break a strict parser are the
  ones users have. Validated against a real 532-item ReadCube library exported in both
  RIS and BibTeX, reproducing every count exactly from each: 532 records, 521 usable
  DOIs, 529 titles, 517 author lists, 521 years. That file carries a 4-character
  `PMID` RIS tag where the convention is 2, an always-empty `XX` tag on 385 records,
  citekeys containing `:` (55) and non-ASCII (16) that strict BibTeX forbids outright,
  and CRLF endings that make a literal `"\r\n"` split return one giant record.

  Triage is three verdicts — `ready`/`review`/`skip` — with the detail in reason
  strings. Two gates are worth naming. `no-text-layer` is the silent one: a scanned
  PDF extracts to nothing, ingest logs a warning nobody reads, and the page then
  passes every later gate on grounding that does not exist. `superseded-by-journal`
  is invisible to DOI-level dedupe — the real library held 10 preprint/published
  pairs and **zero** duplicate DOIs, so only title comparison finds them, and the
  survivor is chosen deliberately because the two exports listed those pairs in
  different orders. `duplicate-doi` is its complement, for the same paper under one
  DOI twice: that library had none, but a concatenated or merged export produces
  exact duplicates readily, and nothing downstream would catch them — `apply`
  re-checks the wiki, not the manifest, so both records dispatch in one wave.
  Both gates pick their survivor by a total order rather than by input position.

  `<pdf-root>` is optional, and the report always lists every record that clears
  every gate except having a file, with its DOI. For a cloud-hosted library that
  fetch list is the most useful thing the command produces: without it, a
  metadata-only run reports a count and nothing actionable.

  `researchwiki import apply --run <dir> --limit N` copies a wave into `inbox/` and
  hands it to `agent ingest` with each paper's own `--doi/--title/--authors/--year`.
  It is the only phase that spends tokens or writes pages, and the only one where
  "nothing to do" is a failure rather than a result.

  There is no journal and no staging directory, because nothing here can be left
  half-done: the only mutation is copying a PDF, and everything after belongs to
  `_ingest_batch`, which already keeps a crash-safe checkpoint. Recovery is
  `agent ingest --resume <batch-dir>`, the path users already know.

  One fact is deliberately **not** frozen in the manifest. Pairing, verdicts, stems
  and argv are, so `apply` cannot conclude something different from the `inspect` a
  user read — but whether a paper is *already in the wiki* is a fact about now.
  Re-checking it per record is what makes `--limit 30` mean "the next 30 still
  pending" instead of "the first 30, again", so waves compose instead of repeating.

  `researchwiki import verify --run <dir>` closes the loop: landed / sandboxed /
  not-yet-imported per record, plus the `lint` keys an import can break and the
  graph-wiring follow-ups. It reads the manifest rather than the batch checkpoint,
  because the question is about records ("did this paper reach the wiki") and the
  checkpoint only knows about files — a record can finish its ingest and still sit
  in `.agent-output/`, which is finished work awaiting review, not a failure.

- `researchwiki export` — the corpus as BibTeX / RIS / CSL-JSON, which is the
  inverse of `import` and the answer to "can I get my data out". Zero tokens, no
  network, and byte-identical across runs, so a `.bib` can live in version control
  and diff meaningfully. One phase; nothing is written without `--out`.

  **The citekey is the page stem.** Long, and the only key that cannot change
  under you: a short key has to disambiguate collisions with a letter suffix — 47
  pages collide across 19 naive `surname+year` keys on the real corpus — and that
  suffix is recomputed every run, so ingesting one more paper by the same author
  in the same year renumbers keys already sitting in a manuscript. There is no
  `--citekey` flag, because shipping an unsafe alternative beside the safe one
  invites exactly that failure.

  **Names are not parsed for BibTeX or RIS**, which both understand
  `First von Last` themselves. That is what makes 58 nobiliary-particle names
  (`A. van der Graaf`) and 76 four-token names impossible to corrupt: no boundary
  is ever guessed. Only CSL-JSON wants structured `family`/`given`, and there
  anything ambiguous becomes a CSL `literal` — the format's own construct for a
  name with no given/family structure, so declining to split is a faithful record
  rather than a fallback.

  Gaps are downgraded and reported, never invented. 53 venue-less papers become
  `@misc` rather than `@article` with an empty `journal`, because that shape makes
  `bibtex` merely *warn* — surfacing weeks later in a LaTeX log instead of in the
  bibliography and the report. Three venue values that are really typesetting
  furniture (`Journal of LaTeX Class Files`, `preprint`) are suppressed; that is
  the only place this command could have printed a falsehood. Eight DOIs recorded
  as `https://doi.org/…` normalize through the importer's own `clean_doi`, so they
  are neither emitted as URLs nor dropped by a stricter validator. `document_id`
  values like `Nature 654:324-326` *contain* a volume and page range and are still
  passed through to `note` whole, because the field is free text and FDA guidance
  numbers use it differently. 248 of 421 titles carry a token a title-lowercasing
  style would destroy, so titles are brace-protected per word.

  Synthesis, idea and concept pages are excluded with no flag to include them.
  They have no DOI, venue or year of record, so an entry for one would assert a
  publication that does not exist once pasted into a manuscript — a
  citation-integrity problem rather than a formatting one. Sharing analysis is
  what `prompts/share-page.md` is for.

  Round-trips: 421 records, all three formats, zero mismatches on title, authors,
  DOI, venue or year back through `refimport`, and `import preflight` on the
  emitted file reports the same counts. The one documented loss is that the 149
  `@misc` entries return as `preprint` (96) and `other` (53) rather than
  `article`; that information genuinely is not in the corpus, and a test asserts
  the asymmetry rather than hiding it.

- Author-name parsing lives in one module (`researchwiki/names.py`), shared by
  stem derivation and the exporter. `stems.first_author_surname` held the only
  real parser — `et al.` stripping, consortium detection, and a
  nobiliary-particle walk whose floor is what keeps `Bin Liu` and `Di Liu` correct
  even though `bin` and `di` are particles — and the exporter needed the same
  boundary for CSL. Behaviour-preserving, gated on it rather than asserted: output
  is byte-identical across all 432 distinct author strings in the corpus and the
  documented test shapes.

- `refimport.clean_doi` is public. DOI wrappers arrive from both directions, so
  the normalizer belongs to the package rather than to `parse`'s internals.

- `prompts/export-shareable.md` → `prompts/share-page.md`. It triggered on the
  word "export", which now names a command that does something else; the output
  has always gone to `share/`, so this aligns the name with the destination.

- `_ingest_batch.new_batch` accepts `per_input_args`, mapping an absolute input path
  to flags for that PDF alone. The CLI still refuses `--doi`/`--title`/`--authors`/
  `--year` in batch mode and should: one `--doi` has no meaning across N PDFs. But a
  programmatic caller holding a *different* DOI for every input is the case that
  guard was never about, and it is exactly what importing a reference-manager
  library is. Persisted in `plan.json` as an additive key, read back with a default,
  so batch directories written before this still resume.

### Fixed

- Stem derivation no longer drops a nobiliary particle from a name written
  `Family, Given`. `van der Graaf, A.` derived `der-graaf`: the part before the
  comma is *already* the surname, but the particle walk ran on it anyway, and its
  floor — the rule that keeps `Bin Liu` from being read as particle + surname —
  stopped it one token early. Also affected `De Winter, S.` and
  `van den Berg, L.`.

  The comma is the hard part, not the walk. It separates `Family, Given` in a
  bibliographic export and it separates *authors* in this wiki's own `authors:`
  field, so `van der Graaf, A.` is one person and `Akari Asai, Zeqiu Wu` is two.
  Treating every comma as an inversion is the obvious fix and it changes **349**
  first-author surnames on the real corpus, because a byline's leading given name
  is so often an initial or a particle lookalike (`Di Liu, Bin Wang`). Two signals
  are now required together: the part before the comma must be surname-shaped
  (every token but the last a particle), and the part after must be a given-name
  run. Measured across every corpus byline *and* every raw `authors:` field —
  756 distinct inputs — exactly the three broken cases change and no existing
  stem does. Callers are still expected to split a byline before asking for its
  first surname; this keeps the function robust when one doesn't, because the
  failure mode is a silent wrong filename.

- Stem derivation folds Unicode dashes to ASCII `-`, so a paper whose title is set
  with U+2010 no longer derives a different stem than the same paper spelled with a
  plain hyphen. Publisher-set titles use the Unicode forms freely, and the
  `[^a-z0-9-]` pass *deletes* an unfolded dash instead of preserving the word
  boundary — welding `ATAC‐seq` into `atacseq` while the PDF's own spelling gives
  `atac-seq`. The fold already existed in `slugify_phrase` and was missing from
  `normalize_title_word`, which is the shape of the bug: a normalization every
  caller had to remember separately. It now lives in `strip_diacritics`, which both
  paths share, so slugs and stems cannot drift apart again. Measured against a real
  532-item reference-manager library: 15 of 516 stems were affected. U+2212 MINUS
  SIGN is deliberately excluded — it is category `Sm`, and titles use it as a
  mathematical operator (`CD4−`), not as punctuation.
- Suspended compounds no longer leave a dangling separator in a stem. A title like
  *"epigenome- and transcriptome-wide"* produced `…-in-epigenome-`, and
  *"long- and short-read"* produced `…-long--and-short-read` — trailing and doubled
  hyphens no other stem carries. Interior hyphens are untouched, so CLAUDE.md's rule
  that a hyphenated term counts as one word (`Cas-OFFinder`) still holds. 3 of 516
  stems in the same library. The defect is cosmetic rather than breaking:
  `STEM_PREFIX_RE` only anchors the `{surname}-{year}` prefix, so it matches a
  doubled-hyphen stem, and the one such page in this corpus
  (`chang-2025-rapid-accurate-long--and-short-read`) parses and grades normally with
  41 claims.

  Neither fix renames anything: existing pages keep the stems they were created
  with, per the stem-stability rule in CLAUDE.md § *Disambiguation & updates*. Seven
  pages in this corpus would derive a different title-part if re-ingested — six from
  the dash fold (`…-buchwaldhartwig`, `…-crisprnet`, and four `…-crisprcas9` stems,
  all from en dashes or U+2010 in the published title) and one from the hyphen repair
  (`…-long--and-short-read`). Leaving them is the deliberate call: a stem is the
  filename and every inbound `[[wikilink]]`, so the corpus will carry both
  conventions until a paper is re-ingested for some other reason. Worth knowing
  before a recovery re-ingest of any of the seven, which will land the new form.

## [0.2.1] - 2026-08-10

### Added

- `RW_RELAY_TIMEOUT` (seconds) overrides the chat-relay poll deadline, previously a
  hard-coded 600 s. The deadline starts when the prompt is written rather than when
  the responder notices it, so it does not survive concurrency: with several
  ingests in flight each holds its own 600 s clock, and a responder answering them
  serially can lose workers it has not reached. Settable per run instead of raising
  the floor for everyone, which would also make a genuinely abandoned run hang ten
  times longer. Resolved per call, not as a default argument, so the value is not
  frozen at import; an unusable value falls back to 600 rather than failing.
- `agent ingest` warns when batch mode is combined with the `chat-relay` provider,
  the one provider/mode pairing that fails silently: each worker's stderr is
  redirected into `.ingest/batch-<ts>/worker-*.log`, and that is where the relay
  prints its pending-prompt notice, so the responder is never told a prompt is
  waiting and every worker eventually fails its 600 s timeout. The warning names
  the fix (one foreground invocation per responder) and the alternative (keep batch
  mode, poll `.llm-relay/pending/`). It warns rather than refuses, because batch
  mode is legitimate here for a responder that polls. Detection is
  `model_config.uses_chat_relay()`, which checks the `RW_LLM_PROVIDER` override
  *and* the resolved per-phase providers — a `models.yaml` can route individual
  phases to chat-relay while the rest stay on an API provider.
- Chat-relay prompt payloads carry `stem` and `pdf`, so a responder answering
  several ingests at once can tell whose prompt it is holding without parsing the
  prompt body. `stem` is null on the `reconcile` prompt — the phase that derives
  it — and `pdf` is set before the first call, so every prompt is attributable.
  Additive and nullable, so `schema_version` stays `1` and the response shape is
  unchanged; neither field feeds the op_id hash, because op_id is the cache key
  and folding identity into it would invalidate responses already on disk.

### Fixed

- `researchwiki init --scaffold-only` now creates `wiki/index.md`. It created every
  content directory but not the catalog, and `promote._append_index_entry` returns
  False when that file is absent — so on a fresh clone the *first* paper ingested
  never got its catalog line while every later one did. Existence is the whole
  requirement: a missing `## <category>` section is created by the splice, not
  refused. Idempotent, and it never rewrites an existing catalog. The
  accompanying warning no longer offers "category section absent" as a possible
  cause, since that case is handled — it now names the missing path and the fix.
- Reconcile no longer adopts a PDF's `/Title` when that field holds production
  furniture rather than a title. Oxford stamps it with an internal job code and the
  page range — minimap2's is `OP-CBIO180195 3094..3100`, 24 characters beginning with
  nothing on the banned-prefix list, so the length-and-prefix check took it and
  short-circuited the first-page text scan. It then became the Semantic Scholar
  title-match query (three 404s, retried) and the last-resort page title. 9 of 345
  `/Title` values in the corpus are furniture of this shape. A deny-list cannot cover
  it because the failure is a *shape*, so titles are now checked for one: at least
  three tokens that look like words. Rejection falls through to the text scan, which
  CLAUDE.md already names as the source of truth ("the PDF's first page text, not
  `reader.metadata`"). Measured over every corpus PDF: 336 of 345 kept, and all 9
  rejected are exactly the furniture cases — no real title is refused.
- The first-page title scan skips journal mastheads. With furniture `/Title` values
  now falling through to it, the first qualifying line was often the masthead instead
  — bae-2014's reads "Vol. 30 no. 10 2014, pages 1473–1475 BIOINFORMATICS
  APPLICATIONS NOTE doi:10.1093/bioinformatics/btu048", which passes both the prefix
  and authorish checks. Volume/issue citations, explicit page ranges, inline DOIs,
  "Advance Access publication" running heads and bare `19: 1655-1664` volume:page
  citations never occur inside a title, so those lines are skipped. 7 of the 8
  affected PDFs still on disk now yield their real title.
- Reconcile no longer takes Semantic Scholar's year over the document's when S2 has
  merged a preprint into the journal record. The existing `_s2_record_is_preprint`
  guard needs S2 to *admit* it is describing a preprint by naming a preprint venue;
  S2 also merges the other way, keeping the journal's venue and the preprint's year.
  For minimap2's journal DOI `10.1093/bioinformatics/bty191` it returns `year=2017`
  `venue='Bioinform.'` (its `ArXiv: 1708.01492` was posted 2017-08) while the PDF
  prints "accepted on May 4, 2018" throughout — and since S2 outranks the LLM's
  reading of the document by two places in the chain, a correctly-extracted 2018 was
  discarded and the stem came out `li-2017-…`. When S2 and the document disagree,
  Crossref now arbitrates: its record for a journal DOI is the journal's own, with no
  preprint to merge. Fires only on disagreement, so the common path adds no request;
  no Crossref answer leaves the prior behaviour intact. Does **not** cover the
  distinct case of an arXiv-DOI preprint whose version year differs from S2's (arXiv
  DOIs are not in Crossref) — CLAUDE.md's "version year on the document" rule still
  governs there, by hand.
- `prompts/chat-relay.md` no longer documents a lock that does not exist. It
  claimed the relay grabs `.llm-relay/lock` so parallel ingests serialize on
  chat-relay phases; there is no such lock (the only `flock`s guard `index.md` and
  back-link writes), and relay calls are isolated per op_id, so concurrent ingests
  can be answered in parallel. It also gave the op_id formula as
  `sha1(phase|prompt|stem)`, where the code hashes `phase|prompt` plus a
  `retry_of` discriminator. Documented the real constraint in its place: batch mode
  parallelizes correctly but runs each worker with `stderr=subprocess.STDOUT` into
  a log file, which is where the relay's pending-prompt notice goes — so under
  chat-relay a batch run looks like a hang and then times out. Parallelize
  chat-relay with one foreground single-PDF invocation per responder instead; noted
  as the documented exception to CLAUDE.md's "never fan out one Bash task per file".
- Numeric-drift no longer glues adjacent numeric table columns into one impossible
  number. `parse_claims._extract_table_rows` joins cells with spaces, so a Results
  row arrives as prose like `74,514 559`; `collapse_spaced_thousands` then read
  `514` as a fresh thousands lead and produced `74,514559`, a value nobody wrote and
  nothing can match, so the claim drifted by construction. `_SPACED_THOUSANDS_RE`'s
  lookbehind now also excludes a comma — the mirror of the trailing `(?!,\d)` guard
  added for superscript markers, on the same reasoning: a lead group immediately
  preceded by a comma is the tail of an already-comma-delimited number and cannot
  also begin a space-delimited one. Punctuation is unaffected (`In 2020, 300 000`
  still joins, since the character before `300` is the space). Measured across all
  11,907 claims: 7 claims cleared, 0 newly flagged, 11,899 untouched. Rows of three
  or more numeric columns still corrupt (`1,173 864 619 486` → `864619486`); that
  case is genuinely ambiguous against a real space-delimited 864,619,486 and needs
  claim provenance threaded into the grader, which is separate work.
- Numeric-drift no longer fires on values a paper only ever writes with a letter
  prefix. `NUMERIC_TOKEN_RE`'s `(?<![\w.])` lookbehind — which correctly stops
  `K562` from contributing `562` — also hid Phred/QV notation, so Hansen 2026's
  "QV increased from Q63.1 ... to Q68.9" put neither value in the evidence set and a
  page correctly claiming "QV from 63.1 to 68.9" was flagged as drift and refused
  auto-promotion. `63.1` escaped only because that PDF also says "the initial QV was
  63.1" unprefixed, so whether the veto fired came down to luck. Evidence-side
  matching now also admits `<letter><decimal>` forms; the decimal point is the
  discriminator, so letter+integer identifiers (`K562`, `HG002`, `GRCh38`,
  `rs45512696`) stay excluded. Additive to the value-set, so it can only suppress
  false drift, never mask a rounding — `grade paper` on the affected page goes from
  1 drift claim to 0 with every other value still checked.
- `pip install 'researchwiki[mcp]'` no longer installs a broken `mcp-serve`. The
  extra was unbounded (`mcp>=1.0`) and mcp 2.0 removed `mcp.server.fastmcp`, which
  the server is built on, so a fresh install of the extra resolved to a version the
  server couldn't import. Capped to `<2`; the port to the 2.x server API is
  separate work.

### Changed

- The ingest log's `pdf_claims=N` counter is gone, along with the
  `extract ⚠ zero PDF-side claims extracted` warning it drove. It counted lines
  starting with `-`/`*` in the PDF's Methods/Results — markdown bullets, in typeset
  prose, which essentially never appear: it read 0 for 6 of 8 papers in one session,
  and the sole nonzero reading was two samtools/pbmm2 command-line flags
  (`--secondary=no -s 25000 -K 15G`) mistaken for claims. Nothing consumed it
  (`ctx.claims_count` was assigned and never read), so its only effect was a warning
  that fired on nearly every ingest and implied the drift check had been weakened
  when that check reads the full PDF text and was never affected. The warning now
  fires from the `target_claims` phase, which actually extracts PDF-side claims
  (18–35 on those same papers). `extract_sections()` returns
  `(sections, full_text)`; `ctx.claims_count` is removed.
- CI installs the `mcp` extra, so the MCP server's tests actually run. They had
  been `importorskip`-ing on every CI run — which is how the breakage above stayed
  invisible.

## [0.2.0] - 2026-08-07

### Added

- `migrate` — bulk-import one-paper-per-PDF markdown from an older release or a simpler
  LLM wiki, without re-authoring prose. Phases `preflight` → `inspect` → `apply` →
  `verify`, all zero-token; normalizes H2 headings and frontmatter so claim extraction
  works on the imported pages.
- **Commentary page type** (`type: commentary`) with a structural ingest guard: a
  Research Highlight or News & Views is no longer auto-promoted as the paper it
  describes, so the primary authors' results stop being credited to the commentator.
- Concept-hub tooling: a permanent suppression list for declined candidates
  (`candidates concepts --decline/--undecline/--list-declined`), batch LLM triage, and
  aggregation of near-duplicate candidates.
- `backfill hook` (catalog glosses + short names) and `backfill doi --verify`, which
  checks the DOIs already recorded rather than filling missing ones — a wrong-but-
  resolving DOI was previously invisible to every other check.
- `lint` checks: `duplicate_claim_sets`, `zero_claim_papers`, `venue_suspect`,
  `none_placeholders`, `thin_index_text`.
- Dated API pricing table (`config/pricing.yaml`, carrying `as_of:` and its sources)
  behind the cost rollups in `status` and `insights`.
- A recall signal for the critic, so omission stops being rewarded.
- Versioning system: this changelog, a single source of truth for the version, release
  invariants pinned by `tests/test_version.py`, and a GitHub Release built on tag push.

### Changed

- **DOI adoption is gated on identity, not resolvability**: a DOI is adopted only when
  the resolved record *is* this paper (first-author surname, year within ±1, ≥50% title
  overlap), and the URL-DOI hunt no longer reads past the References heading.
- Verified cross-links are written in **both** directions at promote time.
- `claim-overlap` is batched with per-stem coverage tracking instead of running per
  ingest — it confirmed a link on roughly one paper in ten, so per-ingest cost bought
  little.
- Catalog glosses (`hook:`) are derived from page YAML, so `index.md` is regenerable
  rather than hand-maintained.
- `tags:` dropped from paper and commentary frontmatter; `keywords:` already carried the
  vocabulary, and `tags:` was provenance noise on 334 of 391 pages.
- `init` creates the content directories in code instead of shipping `.gitkeep`
  scaffolding.
- The exit-code contract is now decidable, not merely consistent.
- Model routing: the chatgpt config's quality roles (author/critic/judge) moved to
  `gpt-5.6-terra`; `target_claims` routed to the cheaper extractor role.
- Docs consolidated — `docs/` removed, its one file folded into
  `prompts/concept-page-author.md`.

### Fixed

- **Cross-link integrity**: back-links no longer assert citations that were never
  checked, `lint --fix` mirrors the citation direction when inserting one, and a
  `measures_same` verdict no longer earns a Related Papers bullet.
- **Staleness detection** ages synthesis pages on source ingest date, never file mtime —
  which had made every maintenance pass look like a source change.
- **Counting and graph coverage**: DOI-less papers and cross-page citations are no
  longer dropped from the cross-link graph; root meta pages are excluded from orphan,
  db-drift and broken-link noise; `status` counts every page.
- **Stem derivation**: nobiliary particles stay in the first-author surname, and Latin
  letters NFKD cannot decompose are transliterated instead of dropped.
- **Grading and numeric parsing**: space-separated thousands, superscript citation
  markers no longer glued into numbers, claim-anchor citations routed correctly, and the
  abstract tier no longer scores salience against extraction junk.
- **Providers**: Semantic Scholar cache writes are atomic; ORCID and PubMed negative
  caches have a TTL.
- LLM calls negotiate `reasoning_effort` instead of failing on HTTP 400.
- `page_key` no longer crashes when `wiki/` is a directory symlink.
- `attach` consumes its `inbox/` source instead of leaving it as backlog.
- Year is no longer taken from Semantic Scholar when its record is the preprint's.
- Test-suite hygiene: the suite no longer writes to the real `state.db`, and passes from
  any working directory.

### Removed

- The unused `source_collection` frontmatter field.
- `gpt-5.4-mini` from model routing.

## [0.1.0] - 2026-07-18

Initial tagged release.

[Unreleased]: https://github.com/johapark/research-wiki/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/johapark/research-wiki/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/johapark/research-wiki/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/johapark/research-wiki/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/johapark/research-wiki/releases/tag/v0.1.0
