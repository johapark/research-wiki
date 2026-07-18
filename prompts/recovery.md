# Recovery — re-ingest after broken metadata

Trigger: `lint --json` flags `missing_doi` / `stem_year_drift` / `unknown-` stem, or the agent landed bad metadata at ingest. Re-ingest with overrides rather than YAML-patching — re-derives stem, picks up authoritative S2 metadata, re-creates back-links. ~$0.10 per recovery.

## Workflow

```bash
# Strip inbound back-links, delete page, reindex BEFORE re-running
# (else crosslinks phase crashes on stale semantic index)
grep -rln "<stem>" wiki/                           # find inbound links
sed -i '' "/\[\[<cat>\/<stem>\]\]/d" <each-file>   # strip them
rm wiki/<cat>/<stem>.md                            # PDF stays; auto-renamed if new stem
researchwiki reindex
researchwiki agent ingest papers/<stem>.pdf --doi <correct-doi>
researchwiki db rebuild && researchwiki reindex
```

## Override flags

LLM-reconcile is on by default since R3, so most overrides are cold paths now.

- `--doi <doi>` — missed DOI on adversarial PDFs (Science First-release with ~2K extractable chars). Still occasionally needed.
- `--title "<full>"` — only when LLM-reconcile *also* failed on title. Cold path.
- `--authors "A; B; C"` — cold path; LLM-reconcile handles affiliation glyphs semantically.
- `--year YYYY` — cold path; LLM-reconcile distinguishes paper year from citation year.

## Skip re-ingest if

- Page is correct except one minor field (patch YAML directly).
- Drift is a legitimate preprint→journal year shift.
- Duplicate-detection failure (different paper, same stem) → see *PDF Management Rules* in `CLAUDE.md`.

## No-DOI-by-design papers

NeurIPS posters, workshop papers, internal tech reports, OpenReview-only entries with no DOI: declare `no_doi_reason: "<short why>"` in YAML. Lint's `missing_doi` and audit's no-DOI WARN skip these. **Don't** set this for papers that DO have a DOI you haven't found yet.

## Notes

Stem may change (corrected title → different 5-word window); PDF auto-renames, back-links re-derive from the citation graph, old row drops on `db rebuild`. No manual back-link fix-up beyond the initial strip.
