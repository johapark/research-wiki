# Recovery — re-ingest after broken metadata

Trigger: `lint --json` flags `missing_doi` / `stem_year_drift` / `unknown-` stem, or the agent landed bad metadata at ingest. Re-ingest with overrides rather than YAML-patching — re-derives stem, picks up authoritative S2 metadata, re-creates back-links. ~$0.10 per recovery.

## Workflow

```bash
# Strip inbound back-links, delete page, then rebuild BOTH derived stores
# before re-running.
grep -rln "<stem>" wiki/                           # find inbound links
sed -i '' "/\[\[<cat>\/<stem>\]\]/d" <each-file>   # strip them
rm wiki/<cat>/<stem>.md                            # PDF stays; auto-renamed if new stem
researchwiki db rebuild                            # REQUIRED — see below
researchwiki reindex
researchwiki agent ingest papers/<stem>.pdf --doi <correct-doi>
researchwiki db rebuild && researchwiki reindex
```

**Both rebuilds before the re-ingest are load-bearing, for different reasons.**

- `reindex` — else the crosslinks phase works off a stale semantic index that
  still contains the deleted page.
- `db rebuild` — else the `papers` row for the deleted page survives, reconcile's
  `find_by_doi` still resolves to the old stem, and when the corrected metadata
  derives a *different* stem the runner raises `StemRenameRefused` and aborts.
  (Reconcile's own extractor call is already spent by then; what it saves is the
  author/critic/grade phases, which are the expensive part.) That refusal is correct — it is the guard
  against silently orphaning a page — but during recovery the old page is already
  gone, so the row is a ghost and the abort is pure friction. Observed
  2026-08-06 recovering `dauparas-2023-…` → `dauparas-2025-…`: the run stopped
  with `prior page → compbio/dauparas-2023-… (indexed 2026-08-06)` after only
  `reindex` had been run.

  `--allow-rename` also gets past it, but prefer clearing the row: the flag
  suppresses the guard for the whole run, including any collision you did *not*
  intend.

## Override flags

LLM-reconcile is on by default since R3, so most overrides are cold paths now.

- `--doi <doi>` — missed DOI on adversarial PDFs (Science First-release with ~2K extractable chars). Still occasionally needed.
- `--title "<full>"` — only when LLM-reconcile *also* failed on title. Cold path.
- `--authors "A; B; C"` — cold path; LLM-reconcile handles affiliation glyphs semantically.
- `--year YYYY` — cold path; LLM-reconcile distinguishes paper year from citation year, and since 2026-08-06 reconcile also rejects S2's year when S2's record is the *preprint's* under a journal DOI (`_s2_record_is_preprint`), which was the one year failure that reached the stem.

## Skip re-ingest if

- Page is correct except one minor field (patch YAML directly).
- Drift is a legitimate preprint→journal year shift.
- Duplicate-detection failure (different paper, same stem) → see *PDF Management Rules* in `CLAUDE.md`.

## No-DOI-by-design papers

NeurIPS posters, workshop papers, internal tech reports, OpenReview-only entries with no DOI: declare `no_doi_reason: "<short why>"` in YAML. Lint's `missing_doi` and audit's no-DOI WARN skip these. **Don't** set this for papers that DO have a DOI you haven't found yet.

## Notes

Stem may change (corrected title → different 5-word window); PDF auto-renames and back-links re-derive from the citation graph. No manual back-link fix-up beyond the initial strip.

The stale `papers` row drops on `db rebuild` — which is why that command appears **twice** above. The pre-ingest run is what lets a stem change through at all; the post-ingest run is what records the new one. Running only the second leaves the recovery blocked at reconcile.
