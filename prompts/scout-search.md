# Literature search — `scout search` and `scout search fetch`

Read this when the user asks to search the literature, find papers or trials on
a topic, or download a lead's PDF. It is discovery, not a new evidence tier:
every result is a lead until its PDF is ingested.

## Contract

- **Sources.** PubMed (E-utilities), arXiv (API), bioRxiv/medRxiv (searched
  through Europe PMC, because api.biorxiv.org has no keyword search), and —
  opt-in with `--source clinicaltrials` — ClinicalTrials.gov. Searching them
  is always fine (Rule 1 governs grounding, not discovery); use this command
  rather than raw `WebFetch`/`WebSearch` so caching, rate limits and wiki
  dedupe apply.
- **Leads, not evidence.** A result may not support wiki prose, a claim, a
  `[[wikilink]]`, or the substance of an answer. In conversation, say what
  exists and why it looks relevant ("a 2025 bioRxiv preprint on X, not in the
  wiki"), not what it found. The next corpus action is always the PDF.
- **Abstracts and trial summaries are for triage.** Every row carries the
  source's verbatim text (`abstract`; for trials the registry's brief
  summary) in `--json` and the snapshot; `--abstracts` also prints it in the
  terminal. Read it to decide whether to ingest; never paraphrase it into a
  page or present it as the paper's result.
- **Trials.** Status, phase, sponsor, dates, enrollment, linked PMIDs and
  posted documents ride along with the summary. A trial's linked PMIDs are
  paper leads in their own right — search or fetch those.
- **Fetch only what metadata says is open.** arXiv and bioRxiv/medRxiv are free
  to read; anything else needs Europe PMC `isOpenAccess: Y` and an OA PDF link.
  A refused or blocked download is a `manual` entry with its URL — hand it to
  the user. Never route around a paywall or a bot challenge.

## Search

```bash
researchwiki scout search "prime editing off-target"            # pubmed, arxiv, biorxiv, medrxiv
researchwiki scout search "base editing" --source clinicaltrials --source pubmed
researchwiki scout search "single-cell foundation model" --days 90 --sort fit
researchwiki scout search "CAR-T persistence" --abstracts      # print abstracts in the terminal view
researchwiki scout search --query fetch                          # a query that is literally "fetch"
```

- `--limit N` is per source (default 20, max 100). Source query syntax passes
  through: PubMed field tags (`crispr[tiab]`), arXiv prefixes (`ti:`, `cat:`).
- Results already in the wiki (DOI, journal DOI, retained `arxiv_id`, or a
  reference page's `document_id: NCT…`) and declined ones are dropped. A
  matching title counts only when the first author agrees and nothing
  contradicts it (two different journal DOIs, years far apart), so two
  papers that are both called "Editorial" stay distinct.
- Leads near a synthesis/idea page's *What would update this page* bullet
  appear first, exactly as in `scout recent`. A match means *on-topic*, not
  *confirmed update* — judge it yourself.
- `--max-age-days` (default 1) bounds how stale a cached source answer may be.
  Responses cache under `.web-cache/search/`; each run's snapshot lands under
  `.web-cache/search/runs/`.
- **Exit codes.** 0 complete; 1 bad arguments or a source rejected the query;
  2 a source was unreachable — the rest still printed, so report what came back
  and name the failed source rather than retrying blindly.

## Decline

```bash
researchwiki scout search --decline pmid:12345678 --reason "veterinary, out of scope"
researchwiki scout search --list-declined
researchwiki scout search --undecline pmid:12345678
```

Keys: a DOI, `arxiv:<id>`, `pmid:<n>`, `pmcid:PMC<n>`, `nct:NCT<n>`. The ledger
is shared with `scout recent`; search declines do not steer its S2 seeds.

## Fetch

```bash
researchwiki scout search fetch 10.48550/arxiv.2401.01234 10.1101/2025.11.03.686307 --dry-run
researchwiki scout search fetch arxiv:2401.01234 pmcid:PMC6907074
researchwiki scout search fetch arxiv:2401.01234 10.1101/2025.11.03.686307 --ingest
```

- Pass a row's `fetch_key` (shown as *fetch as …*): for a journal paper with an
  arXiv preprint or a PMC copy, that is the downloadable version.
- Files land as `inbox/arxiv-<id>.pdf`, `inbox/<server>-<doi-tail>v<N>.pdf`,
  `inbox/PMC<n>.pdf`. An existing file is never overwritten; a paper already
  in the wiki is skipped.
- Europe PMC PDF links usually refuse scripted downloads (HTTP 403); expect a
  `manual` entry. bioRxiv may rate-limit (429) a burst — the same.
- The output prints one `agent ingest <pdf> --doi <DOI>` command per file,
  because batch mode refuses `--doi`. For ≥2 files prefer `--ingest`, which runs
  one checkpointed batch with per-file DOI overrides — it spends model calls,
  so use it only when the user asked to ingest. Then follow CLAUDE.md
  *Ingest → After ingest* as for any other PDF.
- **Exit codes.** 0 everything fetched or already present; 1 something was
  skipped or needs a manual download; 2 the network failed — fetch stopped at
  `stopped_on` and what landed before it is listed.

## Provenance

A page built from a fetched PDF records nothing special: the PDF is the source,
and ingest's own metadata path (S2 → Crossref) fills the YAML. If you cite a
lead's structured fact in conversation before ingest (a trial's phase or
status, a preprint's licence), attribute it inline with the source and fetch
date, e.g. `(ClinicalTrials.gov, fetched 2026-10-03)`.
