# Remove — retract a paper from the wiki

Trigger: a paper was ingested in error, retracted upstream (`retraction-check` flagged it), superseded by a version under a different stem, or landed as a mis-typed commentary. **Not** for re-ingesting with corrected metadata — that's [`recovery.md`](./recovery.md), which keeps the PDF and the back-link graph.

```bash
researchwiki remove <stem>                      # dry run (default) — writes nothing
researchwiki remove <stem> --apply              # do it
researchwiki remove <stem> --apply --keep-pdf   # keep papers/<stem>.pdf for re-ingest
researchwiki db rebuild && researchwiki reindex # always, afterwards
```

## What it removes, and what it refuses to touch

**Removed — generated bookkeeping.** `promote` wrote these; no human judgement is embedded in them, so removing them restores the state the wiki would have had:

- `wiki/{category}/{stem}.md` and `papers/{stem}.pdf`
- `papers/{stem}.supp/`, `.grade-cache/{stem}/`, `.figures-cache/{stem}/`, `.ingest/{stem}-evolution-proposals/`
- inbound *Related Papers* bullets on every citing page
- the `index.md` bullet
- a concept hub's `referenced_papers:` entry, its spoke bullet, and a recomputed `concept_span`
- `papers` / `claims` / `ingest_iterations` / `claim_overlap_runs` rows, and claim-graph edges

**Reported, never edited — authored prose.** A sentence on a synthesis, idea or concept page citing `[[stem#slug]]` or a `[^id]` footnote was written by a human and has passed `check-grounding` and `grade synthesis`. No rewrite rule is safe: stripping the citation leaves a claim with no support, and deleting the sentence can remove a conclusion several papers jointly carried. The command lists each one with its file and line number; **you** decide.

A concept hub is the one page that gets both treatments: its spoke registry is generated and is cleaned, its Definition prose is authored and is not.

## After `--apply`

1. `researchwiki db rebuild && researchwiki reindex` — the command prints this reminder.
2. **Expect `lint` to report `dangling_claim_anchors` and `broken_wikilinks` on exactly the pages it listed.** That is the intended state — a visible to-do queue, not a defect. Anything it *didn't* list is a real problem worth reporting.
3. Edit each listed page, then re-run `researchwiki grade synthesis <page>` on it. The page must pass both gates again before it's done.
4. A commentary page whose `primary_paper:` pointed at the removed stem is flagged but not rewritten — retype or remove it by hand.

## Safety

Dry run is the default; `--apply` is required to write anything. The whole removal runs inside a mutation journal (`researchwiki/mutation.py`), so a failure part-way through rolls the tree back rather than leaving a half-removed paper. If the process is killed mid-removal, `status` reports the journal and the next `agent ingest` drains it.

`--keep-pdf` when the page is wrong but the paper should be re-ingested: everything else goes, `papers/{stem}.pdf` stays, and `researchwiki agent ingest papers/{stem}.pdf` starts clean.
