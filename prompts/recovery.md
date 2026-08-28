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

## Half-landed promote

Trigger: `agent ingest` exited **2** with *"promote did not complete — the paper is PARTIALLY landed"*, or a `--resume` reported inputs **"no longer on disk"**.

Promote runs inside a write-ahead journal (`.mutation/`), so the usual outcome is that nothing landed at all — the rollback already undid it and the exit code is just telling you *why*. The usual cause is a duplicate: `_move_pdf` refuses a stem collision that isn't a preprint→journal upgrade. Fix the cause and re-run; there is normally nothing to clean up.

Check by hand only when one of these is true:

- **`status` still lists an interrupted mutation.** The process died before the rollback finished. The next `agent ingest` drains it automatically; run one, or call the drain directly. A journal reporting 5 failed attempts has given up and needs a look.
- **`RW_MUTATION_JOURNAL=0` was set.** Journalling was off, so the half-landed state below is real.
- **The page exists but its DB row doesn't, or vice versa.** File state is journalled; the `state.db` row is not. A crash-recovered rollback removes the page and leaves the row for `db rebuild` to reconcile — run `researchwiki db rebuild`.

In those three cases, check the four things in order:

```bash
ls wiki/*/<stem>.md                       # 1. page written?
ls papers/<stem>.pdf inbox/               # 2. which side is the PDF on?
grep -n "\[\[.*/<stem>\]\]" wiki/index.md # 3. index bullet?
grep -n "<stem>" wiki/log.md              # 4. log entry?
```

All four absent → nothing landed; just re-run. A page with no bullet and no log entry is the half-landed shape, and **the page is the thing to remove** — nothing else in the wiki points at it yet:

```bash
rm wiki/<cat>/<stem>.md
researchwiki db rebuild                   # drops the claims + papers rows
```

Then resolve the cause and re-ingest from wherever the PDF actually is:

- **Duplicate** (`papers/<stem>.pdf` already existed) — the honest question is which copy to keep. Same paper → delete the incoming one from `inbox/` and stop. Genuinely different paper on the same stem → see *PDF Management Rules* in `CLAUDE.md` for disambiguation.
- **Preprint→journal upgrade that wasn't classified as one** — fix the DOI on the existing page first (`researchwiki preprint-check --doi <preprint-doi>`), then re-run.
- **Anything else** — re-run `researchwiki agent ingest <path-to-pdf>`; note the path is `papers/<stem>.pdf` if the move had already happened.

Batch note: an input the resume calls unresumable is recorded terminal in `checkpoint.json` under `unresumable` and is never re-queued, so finishing it by hand is the only path. `worker_started: true` means the subprocess ran and died mid-promote (check the four things above); `false` means the file left `inbox/` some other way.

For the failed run's official trace, copy its full `attempt_id` from the agent
output and run `researchwiki insights --attempt-id <id>`. This prints every
measured phase, the failure outcome, and timing coverage without a raw DB query.
New attempts include an exact terminal wall timer even on failure or budget
exhaustion; a killed historical process without that terminal row is labeled
with the approximate event-span fallback instead.

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

NeurIPS posters, workshop papers, internal tech reports, OpenReview-only entries with no DOI: declare `no_doi_reason: "<short why>"` in YAML. Lint's `missing_doi` and scout's no-DOI warning skip these. **Don't** set this for papers that DO have a DOI you haven't found yet.

## Notes

Stem may change (corrected title → different 5-word window); PDF auto-renames and back-links re-derive from the citation graph. No manual back-link fix-up beyond the initial strip.

The stale `papers` row drops on `db rebuild` — which is why that command appears **twice** above. The pre-ingest run is what lets a stem change through at all; the post-ingest run is what records the new one. Running only the second leaves the recovery blocked at reconcile.
