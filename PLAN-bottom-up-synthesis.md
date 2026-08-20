# Plan — strengthening bottom-up synthesis

Tracking doc for framework work on the *discovery* side of the pipeline: making
the wiki propose pages rather than only answer questions put to it.

Opened 2026-08-19, after building `wiki/concepts/mixture-model.md` (the first
concept hub) surfaced three concrete failures in one sitting.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` dropped

---

## Why this workstream exists

Measured against the corpus on 2026-08-19:

| Layer | State |
|---|---|
| Atomic claims | 3,027 across 117 papers, slug-addressable, graded |
| Page-to-page links | 4.8 mean outbound, 0 orphans |
| Concept hubs | 1 (built this session; 0 before) |
| Typed claim edges | 10 (1 before the backlog drain) |
| Synthesis pages | 1, against 134 ingests |
| Claim-level citations on authored pages | 2 anchors vs 86 bare `[[stem]]` |

The substrate is rich and the connective layer is nearly empty. Every authored
page traces to a question the user asked or a project they run — none traces to
the corpus surfacing a cluster nobody went looking for. That inversion is the
target.

---

## Evidence base — what the mixture-model build actually showed

Building one hub produced three reproducible failures. These are the
justification for everything below; keep them here so the rationale outlives
the conversation.

### E1 — literal member search loses exactly the bridges it exists to find

`find_members("mixture model")` returned **4 spokes, span 2**. The finished page
has **7 spokes, span 4**. All three recovered members were invisible to literal
matching:

| Missed member | Its wording | Why it mattered |
|---|---|---|
| `sarthi-2024-raptor` (ai) | "Gaussian Mixture Models (GMMs)" | the RAG ↔ biology bridge |
| `van-iterson-2017` (compbio) | "three-component **normal** mixture" | the page's central disagreement |
| `rose-1998` (ai) | "normal-mixture mean estimation" | the theoretical anchor |

This is systematic, not unlucky. A term is a *bridge* precisely when different
fields name the same thing differently, so literal matching fails hardest on the
highest-value candidates — and fails silently.

`check-coverage` did flag `van-iterson` (advisory, rank 2). It was only caught
because the advisory output got read.

### E2 — cosine alone cannot decide membership

Ranking all 2,538 contribution claims against the embedded term `"mixture model"`:

```
0.774  van-iterson   three-component normal mixture        TRUE  (lexical miss)
0.766  han-2026      two-Gaussian mixture model            TRUE
0.765  sarthi-2024   Gaussian Mixture Models (GMMs)        TRUE  (lexical miss)
0.749  kang-2020     Harmony linear mixture model          TRUE
0.744  rose-1998     Gibbs-form association probabilities  TRUE  (lexical miss)
0.738  parks-2018    mixture of unaffected and affected    TRUE
0.736  xu-2019       jointly model >20 datasets            TRUE
─────────────────────────────────────────────────────────────── all 7 true members
0.733  frauen-2026   "Pretraining Mixture proportions"     FALSE (different sense)
0.726  arbab-2020    "combines two predictive models"      FALSE
```

Every true member ranks above the first false positive — but the margin is
**0.003**. That is not a threshold anyone can rely on across terms.

Candidate volume at a 0.70 floor confirms it:

| Term | distinct papers ≥ 0.70 |
|---|---|
| off-target activity | 5 |
| attention mechanism | 12 |
| mixture model | 17 |
| chromatin accessibility | 25 |
| ATAC-seq | 31 |

Auto-adding at any usable floor would turn a 7-spoke entry note into a 31-spoke
category listing.

**Design consequence: the semantic pass proposes, it does not decide.**

### E3 — the discovery threshold is tuned for a different job

The full `claim-overlap --backlog` drain (117 stems) produced 190 candidates from
3,027 claims → **5 confirmed links, 4 edge-only**. Six of the resulting ten edges
are `builds_on`/`corroborates` — the only relations that can carry a chain.

At 0.83, precision is right for *writing bullets onto pages* (a wrong bullet is a
visible defect) but wrong for *discovery* (a missed connection is invisible and
permanent, a false candidate costs one judge call). One threshold is doing two
jobs with opposite cost asymmetries.

### E4 — no candidate generator reads the claim graph

`candidates concepts` ranks by term frequency across papers. `candidates
synthesis` uses wikilinks + cosine + keyword Jaccard. Neither looks at typed
claim edges — the only structure encoding *how* papers relate rather than *that*
they overlap.

The most valuable content on the mixture-model page is a `contradicts`-shaped
relation (Parks 2018 vs van Iterson 2017 on whether a mixture can serve as an
empirical null). No candidate generator would have surfaced it.

---

## Work items

### (1) Semantic member discovery — `[x]` done (2026-08-19)

**Problem:** E1. **Constraint:** E2 — propose, never auto-add.

Add a semantic recall tier to concept member discovery. After the existing
lexical/keyword steps run unchanged, embed the term (+ aliases) and rank
contribution claims by cosine; report papers **not** already found lexically as
*candidates*, with claim text, score, and the wording that matched.

Design decisions, all driven by E2:

- **Additive only.** Existing lexical membership is untouched, so no page's
  spoke list changes without a human acting. No regression risk.
- **Candidates are printed, not written.** They land in stdout (and `--json`),
  never in the hub file.
- **Capped and floored.** Top-N above a floor, so `ATAC-seq`'s 31 papers don't
  become a wall of text.
- **Suggests the alias.** The matching claim text *reveals* the vocabulary
  ("three-component normal mixture" → propose `normal mixture`). This folds
  work item (4) into (1) for free — the author re-runs with `--aliases` and the
  candidates become real members through the existing, already-trusted path.
- **Cache-only by default** where possible, so a cold embedding cache degrades
  to today's behavior rather than a surprise model load.

Checklist:

- [x] Calibrate floor against a known-good case (0.70; all 3 missed members land 0.744–0.774)
- [x] Confirm candidate volume is reviewable at that floor (5–31 papers/term)
- [x] `semantic_member_candidates()` — `researchwiki/concepts/semantic_members.py`
- [x] Wire into `run()` as a reported candidate list (`semantic_candidates`, `suggested_aliases`, `semantic_span_gain` in the result dict / `--json`)
- [x] Alias suggestion derived from matched claim wording (acronym + qualifier-bigram)
- [x] Surface candidates in the `min_members` failure message — the moment a thin hub is usually a vocabulary problem, not an absence
- [x] Tests — `tests/test_concepts_semantic_members.py`, 11 passing
- [x] Acceptance re-run against `mixture model` (below)
- [-] `attach_after_ingest` parity — **decided against 2026-08-19** (see E7 and the decisions log). Scaffold-time and `check-coverage` only.
- [x] Surface the same pass in `check-coverage` — see (5) below

**Acceptance — met.** `find_members("mixture model")` with no aliases returns 4
members / span 2; the semantic pass proposes `van-iterson-2017` (0.774),
`sarthi-2024` (0.765), `rose-1998` (0.744) as its top 3 candidates and suggests
`normal mixture` + `GMM` — the two aliases found by hand, which lift the hub to
7 members / span 4.

**Cache-safety note.** The pass reads `load_cached_claim_embeddings`, never
`get_claim_embeddings`. The latter rewrites the shared cache to whatever row set
it receives: calling it with contribution-sections-only during calibration
silently evicted 489 limitations claims (3,027 → 2,538 rows) and would have made
the next `claim-overlap` run re-embed them. Restored, and the constraint is now
load-bearing in the module.

### (2) Hybrid discovery tier — `[x]` done (2026-08-19, redesigned mid-flight)

**Problem:** E3. **Originally scoped as** "lower the threshold to ~0.72 and queue
the extra candidates." **Measurement killed that** — see E5. Replaced with a
hybrid ranker.

- [x] Measure candidate volume across thresholds (E5)
- [x] Test whether threshold-lowering would surface a known-valuable relation (E5 — no)
- [x] Identify a ranking signal that does (E6 — IDF-weighted shared-term mass)
- [x] `discover_pairs()` — `researchwiki/tasks/claim_discover.py`
- [x] CLI: `candidates pairs [--cross-category] [--limit N] [--json]`, zero-token
- [x] Tests — `tests/test_claim_discover.py`, 8 passing
- [x] Default band 0.72–0.83 (ceiling = the auto-link threshold, so the tier never re-surfaces judged pairs); default limit 40
- [x] Hyphenation fix — compounds now also emit their parts; `empirical-null` vs `empirical`+`null` had been disjoint, which cost the motivating pair 295 ranks (#369 → #74)
- [x] Wire into `status` as an opportunity signal — `discovery_warning()`, threshold 15 cross-category pairs, 14-day decay stamp (`.claim-discovery-stamp`). Deliberately a higher bar and longer decay than the claim-overlap backlog nudge (10 / 7 days): a coverage gap means something is *wrong*, an opportunity queue does not. Pinned by a test.
- [x] Persist dismissed pairs — `candidates pairs --decline A B --reason "…"` / `--undecline` / `--list-declined`, stored in `.pair-dismissals.json`, order-independent, filtered out of the queue. Mirrors `.concept-declines.json`.
- [x] Remove the O(N²) memory cliff — the ranker built a full similarity matrix (37 MB at 3k claims, **676 MB at 13k**), and `status` calls it every run. Now a blocked upper-triangle scan bounded at `512 × N`; a parametrized test asserts identical output to the brute-force reference at four block sizes.
- [x] Fingerprint dismissals on the evidence — resolved by an observation that killed the original objection: claim slugs are **already content-addressed** (`blake2s(normalize(text))`), so the slug set of both papers' contribution claims is a fingerprint with exactly the right sensitivity — stable across `db rebuild`, changed by a real edit, an added claim or a removed one. No page-level hashing, no churn. A stale entry stops suppressing and is marked `[STALE]` in `--list-dismissed`; entries predating the field, and any pair whose fingerprint can't be computed, stay valid (suppression is the safe failure mode for human decisions).

**Runtime:** 0.25 s over 3,027 claims, no network, no LLM.

**Sample output** (`--cross-category`, all previously unlinked):

```
idf=37.8 cos=0.763  yan-2020 ↔ merrill-2022     distal, enhancer-associated, promoters
idf=35.0 cos=0.791  kazemian-2022 ↔ gillmore-2021  d-dimer, elevations, transient, five
idf=34.3 cos=0.806  tsai-2016 ↔ stortz-2023     pams, off-targets, genome-wide, mismatches
```

The `kazemian ↔ gillmore` hit is the clearest proof the signal is real: Kazemian
discusses the NTLA-2001 interim six-patient report, and Gillmore *is* that trial
— same six patients, same transient D-dimer elevations. Neither page linked the
other. Cosine 0.791 put it below the 0.83 auto-link tier; rare-term overlap put
it at rank 3.

**Resolved open question:** the tier queues unjudged. Judging 54,792 band pairs
is impossible; ranking them costs nothing and puts the known-good pair in the
top 0.4%.

### (5) Claim-level evidence in `check-coverage` — `[x]` done (2026-08-19)

**Problem:** the gate ranks whole *pages* against `topic_seed`, mixing a paper's
contribution with its intro, discussion and title vocabulary — so its score says
nothing about *why* a paper was retrieved. On the mixture-model hub it put
`van-iterson-2017` at rank 2 (4.64) among hits scoring 4.47 and 4.24, and the
only way to learn it was the page's most valuable omission was to query its
claims by hand.

Now a second pass ranks contribution claims against the seed, annotates each
page-level hit with the matching claim, and sorts claim-backed hits first. It
only annotates rows the page-level pass already produced — never adds one — so
it cannot introduce a false positive. `--json` gains optional per-hit
`claim_*` fields; the contract is otherwise unchanged.

**Acceptance — met.** Replaying the pre-alias page (van-iterson and rose-1998
uncited), the gate now reports both **first, with their claims**:

```
score=4.92  van-iterson-2017   ← claim match
   0.774 #met-7842a3d3 (methodology)
     › BACON models observed association z-statistics as a three-component normal mixture…
score=6.50  rose-1998          ← claim match
   0.744 #met-244bcf2e (methodology)
score=8.55  rosen-2026         (no claim match — demoted despite ranking highest)
score=7.17  cui-2024           (no claim match — "foundation model" vocabulary)
```

**One guard, and a feature cut.** Cosine alone made this noisy: at the
scaffolder's 0.70 floor the synthesis page gained 16 speculative rows, and at
0.74 the six survivors were *all* false positives. The floor stayed (0.74, in
the calibration gap between the lowest true member at 0.744 and the highest
false positive at 0.733); the speculative `claim_only` section was **removed** —
across all six authored pages it produced 0 results, including on its own
acceptance case, where both recoveries came through annotation instead. The
lexical guard written to denoise it went with it.

Net: 2 claim-backed annotations across the corpus, recall 2/2 on the case that
motivated the work, and no new report section or JSON key.

### E7 — semantic proposals at ingest time would return nothing

Cost of adding the semantic pass to `attach_after_ingest`, measured 2026-08-19.

**Resources are not the objection.** Zero tokens (embedding + cosine only), and
the bi-encoder is already constructed during ingest grading (`grade/scorer.py`
calls `embed_texts`/`score_claim`), so the 2.16 s cold-call cost is already
paid. Marginal cost is ~0.02 s per hub — 3 warm calls took 0.06 s — plus one
4.6 MB cache read per process.

**The objection is that it would not work at all:**

```
contribution claims in DB      : 2538
covered by cache               : 2538
cache coverage for a NEW paper : 0 of 25
```

`semantic_member_candidates` reads `load_cached_claim_embeddings` and never
`get_claim_embeddings` — a load-bearing constraint, since the latter rewrites the
shared cache to whatever rows it receives (the bug that evicted 489 claims during
this workstream's own calibration). The only writer of that cache is
`grade/claim_overlap.py`, and claim-overlap is **off by default at ingest**. So
at `attach_after_ingest` time the new paper's claims have never been embedded,
and the pass silently returns nothing for precisely the paper it was asked about.

Every fix has a catch:

| Option | Catch |
|---|---|
| Call `get_claim_embeddings` at ingest | Reintroduces the cache-clobbering bug |
| Enable claim-overlap at ingest | Deliberately off — a judge call per candidate, ~1 in 10 confirms |
| Add an append-only cache-warming path | ~30 lines, no LLM, genuinely useful — but new infrastructure for an unused feature |

**Two further costs**, independent of the cache:

- **No destination for the output.** `attach_after_ingest` is automatic and
  *writes* spokes; a proposal cannot be written to the hub. In batch mode each
  worker's stderr goes to `.ingest/batch-*/worker-*.log`
  (`_ingest_batch.py:111`), so a printed proposal is invisible during exactly
  the multi-PDF runs where it would fire most. Writing to `.ingest/` instead
  feeds a backlog `status` already nags about.
- **No reviewer attached.** The `check-coverage` guards (0.74 floor, ≥2 shared
  terms) reached usable precision because that gate is run deliberately, with
  attention on its output. An ingest hook fires while you are doing something
  else, and scales as ingests × hubs.

### E5 — lowering the cosine threshold does not work

Paper-pairs above each threshold (117 papers → 6,786 possible pairs):

| threshold | claim pairs | paper pairs | % of all possible |
|---|---|---|---|
| 0.83 (current) | 168 | 99 | 1.5% |
| 0.80 | 981 | 417 | 6% |
| 0.78 | 2,988 | 989 | 15% |
| 0.75 | 14,629 | 2,487 | 37% |
| 0.72 | 55,773 | 4,372 | 64% |
| 0.70 | 117,470 | 5,415 | **80%** |

At any threshold low enough to be interesting, most of the corpus is "related to"
most of the corpus. That is not a discovery signal.

The decisive test: **the Parks 2018 ↔ van Iterson 2017 relation — the most
valuable content on the mixture-model hub — peaks at cosine 0.743.** Reaching it
by threshold means accepting ~2,400 paper-pairs to find it. Yield would be far
worse than the 2.6% the 0.83 tier already achieves.

### E6 — rare-term overlap is the signal cosine is missing

Ranking the 54,792 pairs in the 0.72–0.80 band by **IDF-weighted shared-term
mass** instead of cosine puts Parks ↔ van Iterson at **#210 of 54,792 — top
0.4%**, and the pairs ranked above it are genuinely related:

```
idf=45.4  fastslic ↔ power-slic       shared: bsds, undersegmentation, boundary
idf=35.1  donno-2022 ↔ yan-2026       shared: learnable, one-hot, embeddings
idf=29.3  generative-agents ↔ reflexion  shared: reflection, memory, scores
idf=29.0  kazemian LNP ↔ gillmore CRISPR shared: d-dimer, elevations, transient
idf=28.4  donno-2022 ↔ lotfollahi-2023 shared: cvae, variational, autoencoder
```

Why it works: a 384-dim embedding compresses away exactly the rare, distinctive
vocabulary that marks two claims as being about *the same specific thing*. Two
claims can sit at 0.73 because both are methods prose; two claims sharing
"empirical null" and "null distribution" are about one topic. Cosine measures
register, IDF overlap measures subject.

This is the same hybrid the framework already trusts in `search` (Tantivy BM25
fused with semantic via RRF) — applied to claim pairs rather than queries.

### (3) Edge-driven candidate generation — `[ ]` not started

**Problem:** E4. Highest conceptual value, largest unknown.

A generator that clusters on typed claim edges — especially `contradicts` and
`refines` — and proposes pages because there is an *argument*, not because
vocabulary overlaps.

- [ ] Wait for edge density to justify it (10 edges today; revisit past ~50)
- [ ] Prototype: cluster `contradicts`/`refines` edges, emit proposal stubs like `candidates synthesis` does
- [ ] Decide surface: new `candidates tensions`, or a mode on the existing commands

**Blocked on (2)** — needs more edges than the corpus currently has, and (2) is
what produces them.

### (4) Alias suggestion at scaffold time — `[x]` folded into (1)

Originally separate; falls out of (1)'s candidate reporting at no extra cost.

### (6) `concepts --rescan` — `[ ]` parked, not scheduled

Corpus-wide spoke proposals for existing hubs, run deliberately like
`claim-overlap --backlog` rather than automatically at ingest. This is the shape
the `attach_after_ingest` question resolved *into* (E7): same signal, a reviewer
attached, embedding cache warmed in the same pass, no per-ingest tax.

Parked because nothing has asked for it. `attach_after_ingest`'s lexical
attachment already covers the common case, and (5) catches omissions on the
pages that carry citations. Revisit if hubs grow past a handful and spokes start
going stale between scaffolds.

- [ ] Needs the append-only claim-embedding cache path from E7 (~30 lines, no LLM) — which would also let `claim-overlap` warm incrementally instead of rewriting

---

## Sequencing

```
(1) semantic member discovery   ← done; fixed a demonstrated failure
 └─ (4) alias suggestion        ← free byproduct
 └─ (5) claim evidence in check-coverage   ← same machinery, second surface
 └─ (6) concepts --rescan       ← parked; what (E7) ruled the ingest hook out in favour of
(2) hybrid discovery tier       ← done; produces the edges (3) needs
 └─ (3) edge-driven candidates  ← blocked until edge density supports it
```

---

## Decisions log

- **2026-08-20** — **Simplification pass.** Three things cut on the grounds that
  they were surface without yield: the `claim_only` section in `check-coverage`
  (0 results across all six authored pages, including its own acceptance case —
  both recoveries came through annotation) and the lexical guard written only to
  denoise it; and `concepts --no-semantic`, an escape hatch nobody asked for
  guarding a 0.02 s advisory print.

  Separately, the pair-discovery feature was **moved off `claim-overlap`**, whose
  job is cross-linking *one paper*. Corpus-wide discovery and a curation list are
  a different question, and bolting them on had taken that command from 8 flags
  to 14 with three unrelated modes. They now live as `candidates pairs`, beside
  `candidates concepts`/`synthesis`, reusing the sibling's exact
  `--decline`/`--reason`/`--undecline`/`--list-declined` vocabulary — no new flag
  names anywhere. `claim-overlap` is back to 7.

- **2026-08-19** — **No semantic proposals in `attach_after_ingest`.** Not on
  cost — it is nearly free (zero tokens, model already loaded, ~0.02 s/hub) —
  but because the pass is cache-only by design and the new paper's claims are
  never in that cache at ingest time (E7: 0 of 25 covered). Making it work needs
  an append-only cache-warming path that does not exist, and even then the
  output has no destination an automatic hook can use: batch workers log to
  files nobody reads, and there is no reviewer in the loop.

  The same signal already reaches the author at both moments it matters —
  scaffold time via `concepts --aliases`, and `check-coverage` (5). If corpus-wide
  spoke coverage is wanted later, the right shape is a **deliberate rescan**
  (`concepts --rescan`, run like `claim-overlap --backlog`): same signal, a
  reviewer attached, cache warmed in the same pass, no per-ingest tax. Parked,
  not scheduled — nothing has asked for it yet.

- **2026-08-19** — Semantic membership will *propose*, never auto-add. Driven by
  E2: the true/false margin on the calibration case is 0.003, and a usable floor
  admits 31 papers for `ATAC-seq`. Precision at the page level is preserved by
  routing candidates through the existing alias path rather than around it.
- **2026-08-19** — Discovery does not come from lowering the cosine threshold.
  E5 measured it: at 0.70, 80% of all possible paper pairs qualify, and the
  known-valuable relation still sits at 0.743 among ~2,400 others. The signal is
  **IDF-weighted shared-term overlap inside a cosine band** (E6) — cosine
  measures register, rare-term overlap measures subject. Same hybrid the
  framework already trusts in `search`.
- **2026-08-19** — The discovery tier queues *unjudged*. Judging 54,792 band
  pairs is impossible; ranking them is free and puts the known-good pair in the
  top 18%. Confirmation routes through the existing judged path per stem.
- **2026-08-19** — Dismissals are permanent, not decay-stamped, and keyed on the
  paper pair rather than a claims fingerprint. Two papers do not become related
  because time passed, so a decay window would just re-ask a settled question.
  The fingerprint alternative (as `claim_overlap_runs` uses) would invalidate on
  any edit to either page, which for a hand-reviewed queue is near-permanent
  churn. Cost: a dismissal survives a rewrite of the evidence it was based on —
  `--undismiss` is the manual remedy.
- **2026-08-19** — Concept hubs should stay *sparse*. Luhmann's keyword index
  carried 1–3 addresses per term deliberately; `attach_after_ingest` has no cap
  and grows hubs monotonically. Not fixed here, but any auto-add proposal must
  answer it. See the Zettelkasten discussion that opened this workstream.

---

## Related

- `wiki/ideas/framework-upgrades-from-agent-memory-and-rag-literature.md` — the
  fidelity-axis companion; HippoRAG's synonymy-detection mechanism (retrieval
  encoders adding edges between similar noun phrases) is the published analogue
  of item (1).
- `wiki/ideas/supervision-and-correction-authority-in-the-wiki-pipeline.md` —
  governance axis.
- `prompts/concept-page-author.md` — the authoring contract items (1)/(4) change
  the inputs to.
