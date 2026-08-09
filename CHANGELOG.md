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

- `pip install 'researchwiki[mcp]'` no longer installs a broken `mcp-serve`. The
  extra was unbounded (`mcp>=1.0`) and mcp 2.0 removed `mcp.server.fastmcp`, which
  the server is built on, so a fresh install of the extra resolved to a version the
  server couldn't import. Capped to `<2`; the port to the 2.x server API is
  separate work.

### Changed

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

[Unreleased]: https://github.com/johapark/research-wiki/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/johapark/research-wiki/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/johapark/research-wiki/releases/tag/v0.1.0
