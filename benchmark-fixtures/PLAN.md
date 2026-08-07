# Ingestion improvement plan

The harness in this directory exists to make pipeline changes testable. This document is the running plan for which change to make next, why, and how we'll know if it worked.

---

## Baseline (2026-07, OA corpus)

Fixture set was overhauled 2026-07-03 — the previous personal-corpus fixtures (kim/lai/yang/gupta/siren/zhang, six papers behind gitignored PDFs) were retired and replaced with five CC-BY-4.0 open-access papers committed under `pdfs/`. The harness now runs end-to-end on a fresh clone, so anyone can reproduce scoring — via `researchwiki benchmark-fixture <stem> --repeat N --llm`, which keeps drafts in memory, or `agent ingest … --force-sandbox` for a single authored page. **Never a bare `agent ingest` on these PDFs**: it promotes fixture papers into `papers/`/`wiki/` and `index.md`, violating the corpus-isolation rule invoked below. See [`prompts/benchmark-run.md`](../prompts/benchmark-run.md).

Baseline established 2026-07-03 via `researchwiki benchmark-fixture <stem> --repeat 5 --llm` per fixture. Author phase runs 5 times against the fixture's bundled PDF (drafts in memory, no `papers/`/`wiki/` writes — per corpus-isolation rule); each draft scored independently by the Opus 4.7 cross-family judge. Config: `config/models.yaml` shipped Anthropic tier — author = `claude-sonnet-4-6@anthropic`, reconcile = `claude-haiku-4-5-20251001@anthropic`, **eval_judge = `claude-opus-4-7@anthropic`** (deliberately cross-family to avoid same-model self-grading bias).

The pre-OA scoreboard is preserved for historical reference under the "Historical" section further down — it can't be directly compared to OA numbers (different papers, different domains), so it functions as an archive, not a rolling baseline.

| Fixture | Type | LLM-judge mean ± SD (N=5) | Range | Per-run |
|---|---|---|---|---|
| muslu-2026 — VariantMedium | method-with-benchmarks | 24.5% ± 4.2% | [20%, 30%] | 30 27 26 20 20 |
| zhang-2026 — MGA | method-with-benchmarks | 48.8% ± 5.8% | [41%, 56%] | 56 41 52 46 48 |
| li-2026 — scHilda | method-with-benchmarks | 36.4% ± 3.1% | [32%, 40%] | 36 36 40 38 32 |
| fonseca-2026 — TB ibuprofen trial | clinical-trial | 87.8% ± 3.6% | [84%, 93%] | 89 86 86 93 84 |
| nohel-2026 — BEAMSTER | dataset | 66.2% ± 3.2% | [64%, 71%] | 71 68 64 65 64 |
| **pooled mean** | | **52.7%** (fixture-wise SD 25.0%) | | |

Per-axis mean across fixtures (LLM-judge):

| Axis | Pooled mean | muslu | zhang | li | fonseca | nohel |
|---|---|---|---|---|---|---|
| headline_claims | 68.8% | 44% | 62% | 52% | 100% | 86% |
| capabilities | 59.9% | 31% | 60% | 33% | 95% | 80% |
| limitations | 51.2% | 16% | 54% | 27% | 76% | 83% |
| related_papers | 0.0% | 0% | 0% | 0% | 0% | 0% |
| comparator_fidelity | 21.2% | 0% | 18% | 12% | 69% | 7% |

Notes:
- **Noise floor is smaller than the tool's docstring suggested.** Per-fixture SD at N=5 lands 3–6pp, not ±10pp. Δmean effects ≥ ~5pp on a single fixture (or ≥ ~3pp on the pooled mean) are attributable to a config change, below that lives in noise.
- **muslu-2026 remains a low outlier** — 24.5% mean with `comparator_fidelity: 0%` across all 5 replicates. The pipeline reliably produces the numbers and the comparator names but doesn't co-locate them inside the 100-char window the mechanical check requires. First lever target.
- **`related_papers` is vestigial** on this corpus — 3 of 5 fixtures declare `[]`; the other 2 don't have edges to bundled stems. Every replicate scores 0%. Not a pipeline defect; the axis has nothing to score against. Either add real intra-corpus edges to the fixtures where they exist, or treat this axis as inactive.
- **Cross-family judge changed absolute levels only modestly** vs the retired single-shot Sonnet-judged run: muslu +1pp, zhang -1pp, li -8pp, fonseca +4pp, nohel -3pp; pooled mean 54.0% → 52.7%. The 5–15pp self-grading bias I hypothesized didn't materialize on this Sonnet/Opus pair — either the same-family effect is smaller than the literature suggests, or `--repeat 5` averaging masks it. Either way, from this point forward the cross-family judge is the methodology of record.
- Raw per-axis JSON archived under `.benchmark-runs/baseline-2026-07-03-repeat5/` (gitignored). Regenerate with the same command; each fixture takes ~5–8 min at N=5 and ~$3 in Opus judge tokens.
- **Reproducing the baseline requires a local config edit.** `config/models.yaml` is gitignored (per-user state); the tracked `config/models.anthropic.yaml` template leaves `eval_judge: {role: judge}` at Sonnet. To match the numbers above, edit your local `config/models.yaml` and set `eval_judge: {role: judge, model: claude-opus-4-7, temperature: 0.0, max_tokens: 400}`. Alternatively, run with `RW_MODELS_CONFIG=<your-anthropic-plus-opus-judge.yaml>` pointing at a locally-authored file.

### Retired single-shot readings (2026-07-03)

Kept for provenance only. These were captured before the judge was flipped to Opus and before `--repeat` mode was used; they aren't methodologically comparable to the pooled baseline above and should not be diffed against future config runs. Muslu was 4.3% / 23.4% (heuristic / LLM), zhang 31.2% / 50.0%, li 26.2% / 44.0%, fonseca 64.9% / 83.8%, nohel 53.8% / 68.8%; pooled means 36.1% / 54.0%. Raw JSON at `.benchmark-runs/baseline-2026-07-03/`.

---

## Historical (pre-2026-07-03, personal corpus)

The scoreboard, calibration notes, and lever trajectory below refer to the retired personal-corpus fixture set (kim / lai / yang / gupta / siren / zhang). Preserved as a record of the harness's development history and methodology audits; not comparable to the current OA baseline.

## Goal

The agent ingest produces wiki pages of materially different quality across paper types. Tracked against three hand-curated fixtures with the LLM judge.

### Running scoreboard (LLM-judged weighted recall)

| Fixture | Venue | Type | Initial baseline | post-L1 | post-L4 (failed) | post-rollback | post-L2 (Current) |
|---|---|---|---|---|---|---|---|
| Folddisco (kim-2026) | Nat Biotech | method-with-benchmarks | 95.8% (curated) | 95.8% | 95.8% | 95.8% | **95.8%** |
| Yang (yang-2026) | Nature | review | 56.0% (agent) | 84.0% | 76.0% | 84.0% | **90.0%** |
| Lai (lai-2026) | Nature | clinical-trial | 55.1% (agent) | 86.2% | 82.6% | 81.2% | **82.6%** |
| Giraffe (siren-2021) | **Science** | method-with-benchmarks | 35.4% (old ingest) | — | — | — | **93.8%** ⚠ pre-cutoff |
| Gupta (gupta-2026) | **NEJM** | clinical-trial | 92.1% (existing) | — | — | — | **84.9%** |

Folddisco's near-ceiling score validates the harness across every lever (no regression on the curated calibration anchor). Cumulative trajectory:

- L1 (full PDF text to author): Yang +28pp, Lai +31pp — biggest single win.
- L4 (paper-type templates): rolled back, net-negative.
- L2 (figure-caption / ED-figure structured extraction): Yang +6pp, Lai +1pp.
- L6 (grade Limitations + Methodology): infrastructure win — surfaces drift signal on agent pages (Yang/Lai limitations BM25 means 13.7/14.0 vs Folddisco's 25.5).
- L3 (target-claims extraction): N=5 Yang +6.6pp (variance halved), Lai null, Gupta +1.4pp — pooled +2.4pp at d=0.60. Headline axis +5.7pp; capabilities -10pp (redistribution).

**Mean post-L2 across all five fixtures ≈ 89.4%.** Remaining ~10pp headroom is dominated by synthesis-side compression (the agent saw the content but didn't preserve detail), not access-side starvation.

### Cross-journal generalization (validated 2026-06-13)

Two of the five fixtures (Giraffe / Science, Gupta / NEJM) are non-Nature-family, deliberately chosen to test whether L1+L2 generalize. Findings:

- **L1 is journal-agnostic.** Both non-Nature papers score in the 83–94% band, comparable to the three Nature fixtures. Recall depends on the agent having access to the paper's content, not on a specific journal's section structure.
- **L2's caption regex is Nature-family-specific by design.** Pipe-style (`Fig. N | …`) captions are common in Nature, Cell, and some Science articles; period-style (`Fig. N. …`, `Figure N. …`) in NEJM, older Science, Bioinformatics, PLoS. On Giraffe and Gupta, the regex extracted zero captions — both papers got L1-only context. They scored similarly to Nature fixtures anyway because L1's full PDF text already includes caption content unstructured.
- **Generalizing L2** to period-style captions would expose captions as labeled blocks for non-Nature journals, but the upside is bounded by what L1 already supports. Worth doing if a future fixture surfaces a caption-specific failure on a non-Nature paper; not urgent at the current ~90% mean recall.

### Pre-cutoff caveat

Giraffe (Dec 2021) precedes the LLM training cutoff and is contamination-flagged. The 93.8% may partly reflect memorization of the original Science paper rather than pure pipeline quality. **Resolved 2026-06-13** by adding Zhang (Molecular Cell, June 2026) as a contamination-clean non-Nature fixture.

## Scoring methodology assessment (2026-06-13)

Audit of fixtures + scorer surfaced one bug fix and several methodological limits worth recording.

### Bug fixed

**Comparator-fidelity 300-char window produced false matches** when a page listed multiple comparators in nearby prose. Tightened to ±100 chars + closest-comparator-wins logic. Means earlier reported `comparator_fidelity` scores were too lenient — Folddisco's 81.2% on that axis is closer to actual after the fix.

### Bugs known but unfixed (low priority)

- Number regex mishandles `×` suffix (drops the `×`), superscript scientific notation (`10⁶` → no match), scientific notation with multiplication (`1×10⁶` → `1×`), and dates (`2026-01-01` → fragments). Means numerical-fidelity check is unreliable for these forms.
- LLM judge returns `partial` for direct numerical contradictions (probed adversarially: page says "<25 hours", claim says "<5 hours" → judge said partial, should be miss).

### Methodological caveats

1. **Curator-judge correlation.** Fixture verbalizations were written by me (Claude) reading the PDFs, sometimes also looking at existing wiki pages. The LLM judge is the same model family. Both correlations bias scores upward — current numbers are an *optimistic* upper bound.
2. **Single-run variance is ~±10pp on prose-heavy papers.** Author-side tournament between two stance-varied drafts plus sampling drift. Single-run lever comparisons below this magnitude are inconclusive — most of the L4 / post-L2gen / completeness-review effects fell in this band.
3. **No inter-rater reliability.** Fixtures haven't been independently re-curated. A subset re-curated by a different agent or human would quantify the curator bias.
4. **Importance weights (3:2:1) are subjective.** No principled rubric for `critical` vs `high`.

### Recommendations

- **R1 (DONE 2026-06-13)**: `--repeat N` flag added to `benchmark-fixture`. Runs reconcile + extract + crosslinks ONCE (deterministic given source PDF + caches), then runs the author phase N times against that shared upstream state. Each draft scored independently; report aggregates mean / SD / min / max overall and per-axis.
  - **Day-to-day ingest is unaffected.** `researchwiki agent ingest <pdf>` stays one-shot. Replication is opt-in for methodology validation only.
  - Cost per replicate: ~$0.06 author + per-item judge calls when `--llm`. N=5 with judging ≈ $0.93/fixture.
  - Replicate scores will sit slightly below the committed-page score (~5-10pp) because replicate drafts skip the critic+evolve refinement that runs in the full ingest. The replicate captures author-side variance, which is what we want to measure.
- **R2**: Fix the number regex.
- **R3**: Tighten judge prompt to mark numerical contradictions as `miss`.
- **R4**: Re-curate one fixture independently (different LLM session or human) to measure inter-rater agreement.

These don't invalidate the L1 win — that's +28pp to +31pp, well above the noise floor. They do mean the L4-failed / L2gen-mixed / completeness-mixed conclusions shouldn't be treated as final without replication.

Stochastic-variance caveat: each ingest produces a fresh tournament between two stance-varied drafts. Single-run scores have ±5% noise on prose-heavy papers. Treat individual numbers as noisy estimates; treat trends across multiple papers and multiple lever runs as signal.

## Diagnosis from the calibration

The LLM judge's rationales name exactly what the agent dropped. Mapped to architectural causes:

| Missing claim | Cause | Lever |
|---|---|---|
| Yang: "no specific kcat/KM values" (Fig. 3d) | Section parser doesn't index figure captions | L2 |
| Yang: "never describes the diffusion mechanism" | Methodology prose past 4000-char anchor cap | **L1** |
| Yang: "never lists targets like SARS-CoV-2 Spike, HA, integrins" | Fig. 2 caption content unreachable | L2 |
| Lai: "no mention of pancellular HbF or ≥94.7%" | Results detail past truncation | **L1** |
| Lai: "does not assert 145 sites or 87/36/22 split" | Multi-paragraph claim spanning Methods + Results | **L1** |
| Lai: "no mention of 62.1% editing, 45.3% HbF, 44.9% F-cells" | Preclinical Results past anchor cap | **L1** |
| Lai: missing wikilink to parallel base-editing trial (gupta-2026) | Cross-link proposer didn't surface same-target paper | Separate |

Most stubborn misses are content the author never saw. Section-cap truncation in `agents/phases/extract.py` (4000 chars/section, capped further to 2500/2500/1500 in the prompt) hides extractable content. L1 (pass full PDF text alongside curated excerpts) is the lever this diagnosis points to.

## Lever sequence

### L1 — pass `pdf_full_text` to the author prompt (VALIDATED 2026-06-13)

**Change:** in `agents/phases/draft.py:_build_author_prompt`, add a "Full PDF (supplementary context)" block after the curated section excerpts, sized to a budget (~30K chars). The full text is already extracted in `phases/extract.py` and stored in `ctx.pdf_full_text`; only the author phase needs to start receiving it.

**Predicted flips on re-ingestion:**

| Fixture / item | Current verdict | Predicted |
|---|---|---|
| Yang `enzyme-kcat-km-spread` | miss | match (Fig. 3d caption is in full text) |
| Yang `rfdiffusion-mechanism` | miss | match (Methodology prose now visible) |
| Yang `protein-binders` (capabilities) | miss | match (Fig. 2 captions visible) |
| Lai `pancellular-hbf` | miss | match |
| Lai `off-target-145-sites` | miss | match |
| Lai `preclinical-editing-rates` | miss | match |
| Lai `tbe-architecture` (capabilities) | miss | match |

**Predicted aggregate:** Yang/Lai overall recall jumps from ~55% to ~75–80%. Folddisco does not regress (its curated page already covers everything).

**Falsifier:** if the predicted flips don't land, the lever is wrong — model can't use the extra context (attention dilution, budget pressure, format issues), and we should look at L2/L4 instead.

**Outcome (2026-06-13):** Validated. Yang 56.0% → 84.0% (+28%); Lai 55.1% → 86.2% (+31%). 5/7 predicted flips landed fully (miss → match), 2/7 partial. Folddisco unchanged (regression-anchor sanity-check passed). Side benefit: Lai related_papers improved by +28.6% as the author saw parallel-trial citations in the full-text block. Files: `phases/draft.py:_build_author_prompt`, `runner.py:_phase_author`, `prompts/author-system-{research,review}.md`.

### L2 — extract figure / ED captions as structured blocks (VALIDATED 2026-06-13)

**Hypothesis:** even with L1's full-text fallback, the author was failing to surface caption-bound content (named instances, kcat/KM tables, ED-fig results). Adding `figure_captions` and `extended_data` as labeled blocks in the author prompt should let the author target them when the fixture grades on caption-resident content.

**Implementation:**
- `researchwiki/pdf/sections.py`: new `extract_caption_blocks` function. Pipe-style caption regex (`(Extended Data\s+)?(Fig.|Figure|Table)\s+\d+\w?\s*\|`) catches Nature-family captions reliably; period-style captions fall through to L1's full text.
- Per-caption cap 1500 chars, total cap 12K per side. Yang: 6K main caption text, 0 ED. Lai: 9K main + 8.6K ED. Folddisco: 3K main + 4.6K ED.
- `phases/draft.py:_build_author_prompt`: surfaces both new blocks when present, with explicit instruction text pointing the author to them as primary sources for quantitative anchors.
- `prompts/author-system-{research,review}.md`: updated to describe four-block input order (curated sections > captions > ED captions > full text).

**Outcome:**
- Yang 84.0% → 90.0% (+6pp). Capabilities axis +14.7pp specifically — the targeted axis.
- Lai 81.2% → 82.6% (+1.4pp). Headline_claims +4pp, capabilities +5pp; limitations −8pp (likely draft-to-draft variance, net positive aggregate).
- Folddisco 95.8% → 95.8% (no regression).

**Predicted-flip outcomes:** 4 of 7 caption-targeted items improved (3 of 4 Yang capabilities partial→match, 1 Lai capability miss→partial). The 3 that didn't flip:
- Yang `dna-rna-binders`: miss→miss. Content IS in Fig. 4a caption (now in the prompt) but author chose not to include it — `normal` importance, de-prioritized at synthesis time.
- Yang `rfdiffusion-mechanism`: partial→partial. Mechanism is in Methodology prose, not captions; L2 isn't designed to fix this.
- Lai `editing-spectrum-disrupts-bcl11a`: miss→miss. Content is in Fig. 4c caption + ED Fig. 8d caption (both now surfaced) but author still didn't include — synthesis-side compression issue, not access.

**What this teaches:** L2 closes access-side gaps for caption-bound content (the targeted use case). Items that L2 didn't fix point at remaining synthesis-side issues — content in front of the author that gets dropped during writing. Those are downstream of L2 and would need a different lever (importance-aware writer, longer Results budget, or fixture-aligned rubric).

### L4 — paper-type-aware author templates (REPLICATED N=5 2026-06-13; nuanced verdict)

**Original hypothesis:** prescriptive type-specific templates (strengthened review prompt + new clinical-trial prompt) would close the remaining ~15–25 points of headroom by forcing structural completeness.

**Single-run finding (initial):** Yang 84.0% → 76.0% (−8%); Lai 86.2% → 82.6% (−4%) — appeared net-negative.

**N=5 replicated finding (2026-06-13, after `--repeat` shipped):**

| Fixture | Baseline (N=5) | L4 (N=5) | Δ Mean | Cohen's d | Verdict |
|---|---|---|---|---|---|
| Yang (review) | 79.4% ± 7.0% | 74.2% ± 7.0% | −5.2pp | −0.74 | trend negative, not sig. at N=5 |
| Lai (clinical-trial) | 80.3% ± 1.7% | 80.6% ± 0.9% | +0.3pp | +0.22 | null at aggregate |

**The earlier "L4 fails by −8pp" was largely noise.** Yang's baseline variance (SD=7%) means single-run differences ≤14pp can't be cleanly attributed. The replicated effect is closer to −5pp on Yang and ~zero on Lai.

**But L4 is still not a clear win.** The per-axis pattern at N=5 reveals the real story:

| Axis | Yang Δ | Lai Δ |
|---|---|---|
| headline_claims | −4.7pp | −7.7pp |
| capabilities | −1.2pp | **+6.7pp** |
| **limitations** | **−16.7pp** ▼ | **+21.7pp** ◀ |
| related_papers | 0 | 0 |
| comparator_fidelity | 0 | −5.0pp |

The limitations axis shows opposite signs across the two fixtures: the strengthened *review* prompt regresses limitations on Yang (real, replicated, −16.7pp) while the dedicated *clinical-trial* template improves limitations on Lai (+21.7pp). The L4 templates redistribute the word budget — gains in some axes paid for by losses in others.

**Conclusion (revised):**
- The strengthened review prompt is a regression (limitations axis, replicated). **Stay rolled back.**
- The clinical-trial template is null at aggregate, but the redistribution is favorable on Lai (gains in limitations + capabilities, losses in headline_claims + comparator_fidelity). Worth a second look against Gupta + a third clinical-trial fixture before committing. **Stay rolled back for now**, revisit if a third trial fixture validates.

**Methodology lesson:** the earlier "L4 fails" rollback was correct as a decision but partly for the wrong reason. The actual failure was per-axis redistribution, not aggregate regression. The aggregate summary hid the real signal; per-axis with SDs across replicates surfaces it. Keep replicating.

#### Clinical-trial template — Lai + Gupta combined N=5 (2026-06-13 follow-up)

| | Baseline | L4 | Δ | Cohen's d |
|---|---|---|---|---|
| Lai | 80.3% ± 1.7% | 80.6% ± 0.9% | +0.3pp | +0.22 |
| Gupta | 82.1% ± 2.1% | 85.4% ± **6.2%** | +3.3pp | +0.72 |
| **Pooled** | 81.2% ± 2.0% | 83.0% ± 4.9% | **+1.8pp** | +0.48 |

Welch's |t| pooled = 1.08 (df=12) — directionally positive but below conventional significance.

**Limitations axis is the consistent winner**: +21.7pp Lai, +13.0pp Gupta. Both replicated, both well above noise. The template pushes weak axes (limitations) up at the cost of strong ones (whichever of headline_claims / capabilities scored highest at baseline).

**Concerning signal: variance increase on Gupta** (SD 2.1% → 6.2%, 3×). L4 runs split bimodally on Gupta — same prompt, same input, both 92% drafts and 79% drafts. The template increases output unpredictability on at least this fixture. Lai didn't show this, so it's fixture-specific.

**Decision: stay rolled back.** Mean effect (+1.8pp pooled) too small to justify the variance penalty and not statistically significant at N=10. If the limitations axis specifically becomes a priority later, revisit with N=10 per fixture (~$3) for confidence intervals.

**Why it failed:** the author has a finite word budget (even after raising max_tokens to 4000, it's bounded by section caps). New prescriptive directives **redistributed** the budget toward category-enumeration at the cost of qualifier-specificity — turning matches into partials rather than misses into matches. Examples:

- Yang `catalysis-high-barrier`: match → partial. Page kept "high-barrier reactions are unsolved" framing but dropped "designed enzymes lag natural enzymes by orders of magnitude" specificity.
- Yang `success-rate-and-activity`: match → partial. Mentioned "largely solved" but dropped the "success rate and activity still need improvement" qualifier.
- Lai `editing-spectrum-disrupts-bcl11a`: match → miss. The clinical-trial template's prescribed Methodology structure crowded out the BCL11A-disruption EMSA result.

The infrastructure pieces all stayed:

- `_detect_paper_type` (reconcile.py) now identifies clinical-trial papers via NCT/EudraCT IDs + phase-N trial language. Lai correctly classifies as `type=clinical-trial`.
- `prompt_lib.load_author_system` resolves `prompts/author-system-{paper_type}.md` dynamically — new types ship by file, no code edit.
- `config/models.yaml` author `max_tokens: 2500 → 4000` (real bug fix; first L4 run was truncating mid-bullet).

**Rolled back:** the prompt content (`author-system-review.md` strengthening, `author-system-clinical-trial.md` removal). Returned to L1's 84%/86% baseline.

**What this teaches:** "prescribe more structure" is the wrong shape of intervention for word-budget-bounded outputs. Future paper-type templates should be **subtractive** (specify what to drop) or **constraint-based** (specify invariants the output must preserve), not additive bullet-templates that compete with discretionary content. Possible v2 forms:

- *Constraint-based*: "If the paper has a primary endpoint, the page MUST state both the endpoint definition AND the success rate." No bullet template; just an invariant.
- *Subtractive*: "For clinical-trial papers, omit per-patient detail unless the cohort is n ≤ 5." Frees budget for what matters at this n.
- *Anchored to fixture-style claims*: write directives that mirror what the fixtures grade on, not what feels structurally complete.

**Trigger for revisiting:** if remaining type-specific gaps after L1 + L2 land are large enough to justify another iteration. Currently the gap is ~10–14 points on Yang/Lai which may not be worth another prompt-engineering cycle.

### Cross-link discovery for parallel-trial papers

**Trigger:** orthogonal to L1–L4. The proposer in `agents/phases/crosslinks.py` should surface same-target / same-modality papers as topical neighbors. Diagnose against the Lai fixture once L1 lands.

## Validation protocol

1. Save pre-change pages to `.eval-sandbox/{stem}.pre-Lx.md`.
2. Run `researchwiki benchmark-fixture --llm` against pre-change pages — confirm baseline matches the recorded baseline.
3. Re-ingest with the change in place.
4. Run `researchwiki benchmark-fixture --llm` against the new pages.
5. Diff per-axis recall and per-item verdicts. Compare to predictions.
6. If predictions land: lever validated, advance to next.
7. If predictions don't land: document why, revise the diagnosis.

## Folddisco's role as the regression anchor

Folddisco's manually-curated page sits at 95.8%. Any pipeline change that drops it materially (say, below 90%) is regressing the calibration ceiling — usually a sign the change is over-correcting (e.g., adding context that pushes the author away from the precise prose the curated page achieved). Always re-score Folddisco alongside Yang and Lai.

## Open items / backlog

The L1+L2 wins are validated and shipped; the L4 attempt is documented as a failed-hypothesis-with-lessons; the eval harness exists. Below is everything from the original lever framework + scoring methodology assessment that is **not yet attempted** and not deferred. Ordered by leverage × cost; pick from the top of each tier when picking up the work.

### Cheap fixes — DONE 2026-06-13

- **R2 ✅** — Number regex in `researchwiki/grade/scorer.py` (then `eval/scorer.py`) now handles `×` suffix, Unicode superscript exponents (`10⁶`), and other forms the audit surfaced. Trailing `\b` removed; superscript digits added inline. Verified `'20×'` → `'20×'`, `'10⁶'` → `'10⁶'`, `'10²³'` → `'10²³'`. Folddisco anchor unchanged at 95.8% (LLM judge), so calibration-neutral.
- **R3 ✅** — `_JUDGE_SYSTEM` clarified: `partial` reserved for SILENCE about detail; direct CONTRADICTION (different number, opposite direction, different named entity) is `miss`. Adversarial re-probe verified: "<25 hours" claim against "<5 hours" page → `miss` (was `partial`). Folddisco anchor unchanged.

### Reporting gains — DONE 2026-06-13

- **U1 ✅** — `--with-grader` flag added to benchmark-fixture. Runs `researchwiki/grade/fidelity/paper.py` (then `grade/coverage.py`) alongside fixture scoring, reports BM25 + bi-encoder + numeric-drift + negation-mismatch in a separate pane. Two complementary signals; disagreements are diagnostic. Single-shot only (warns when used with `--repeat`).
- **U2 ✅** — `--with-style` flag added. Reports compression (page-tokens / paper-tokens; verdict tier `compressed | normal | verbose`) and extractiveness (fraction of page sentences with verbatim ≥10-word spans from PDF; verdict tier `paraphrased | normal | extractive`). Mechanical, no LLM calls, ~50ms per page. New `researchwiki/benchmark/style.py` (then `eval/style.py`) module. Calibrated thresholds against committed pages — typical wiki page is 5–25% compression and 5–25% extractiveness; flags fire only at extremes. Smoke-tested across all six fixtures: 5 of 6 agent-output pages flag for low extractiveness (the agent paraphrases everything; almost zero verbatim spans), the one in-band agent page (Yang at 17%) and the curated page (Folddisco at 6%) bracket the normal range.

### Methodology calibration (medium cost) — quantify the curator-judge correlation

- **R4 — inter-rater reliability on one fixture.** Re-curate one fixture (Yang or Lai) via a different Claude session or a human, blind to the existing fixture. Compute item-level agreement (κ or %). If agreement is high (>90% match-on-match), trust the current scoreboard. If lower, calibrate aggregate scores down by the disagreement rate. Quantifies the documented "curator-judge correlation" upward bias. ~2 hours of curating + 1 hour analysis.

### Structural levers (medium–high cost) — chase the remaining ~10pp headroom

The mean post-L2 ceiling on prose-heavy fixtures is ~85%. The residual gap is **synthesis-side compression**: content the author saw (post-L1 has full PDF text in prompt) but didn't preserve in the draft. L3 and L6 attack this directly; L4 v2 is a different shape of attempt at type-specific structure.

- **L3 ✅ DONE 2026-06-13** — Target-claims extraction phase added between `extract` and `crosslinks`. New `researchwiki/agents/phases/target_claims.py` runs one LLM call (Haiku 4.5, ~$0.02/ingest) producing a structured list of `{type, content, importance, location}` tuples. The author phase consumes the list as a coverage target — directive is "preserve critical/high items," not "include every one verbatim." Empty list on any failure → graceful fallback to pre-L3 prompt shape.

  N=5 across three fixtures: pooled +2.4pp, Cohen's d=0.60 (medium effect), |t|=1.66 at df=28 (below conventional significance, would clear at higher N). Per-fixture: Yang (review) +6.6pp with SD halved (7.0→3.2 — most striking secondary signal); Lai (clinical-trial) −0.7pp null; Gupta (clinical-trial) +1.4pp.

  Per-axis pattern (consistent across fixtures): headline_claims +5.7pp pooled (the targeted axis); related_papers +4.8pp; **capabilities −10.0pp pooled** — the L4-style redistribution where the agent over-prioritizes the headline-biased target list and drops capability content. Limitations +1.0pp / comparator_fidelity +2.5pp essentially unchanged.

  **Caveat for future iterations**: extractor prompt biases toward specific (numbers + names) claims, which over-represents headline type. To balance, future L3 iterations could (a) require minimum N capability + N limitation claims per extraction, (b) soften the author-prompt framing to "supplementary coverage targets" rather than primary spec, or (c) enforce a per-type cap.

  **Decision: shipped.** Net positive, headline_claims gain is consistent, Yang variance reduction is unique and valuable. Capabilities-axis loss is real but smaller absolute weight than the headline gain. Re-validate after any future tuning.
- **L6 ✅ DONE 2026-06-13** — Parser (`researchwiki/grade/parser.py`) extended to grade Limitations bullets and Methodology bullets + prose sentences. New `_extract_prose_sentences` function splits paragraphs/sentences with min-length filters (≥40 chars, ≥5 words) and strips bold inline section labels. Folddisco anchor: 57 graded claims (was ~30); Limitations BM25 mean 25.5 (highest — curated limitations well-grounded), Methodology BM25 mean 26.2. No regression on KC/Results scoring. Surfaces drift signal on agent pages: Yang limitations BM25 mean 13.71 — catches agent-inferred limitations like "Future projections are speculative and not grounded" (meta-commentary, not in the review). Lai limitations BM25 mean 13.98 — catches added specificity not in the paper (e.g. "patients recruited from one institution in China" — country detail invented). The hypothetical "specific numbers not extractable" cop-out would now hit low BM25 here and be flagged.
- **L4 v2 — constraint-based or subtractive type-specific templates** (not additive bullet templates). The L4 retro identified the failure mode: prescriptive directives redistribute a fixed word budget rather than expand coverage. Possible v2 forms documented in the L4 entry above ("If primary endpoint, page MUST state definition AND success rate"; "Omit per-patient detail unless n≤5"). Revisit only if a third clinical-trial fixture motivates further work; current null-aggregate-effect doesn't justify a second attempt without new evidence.
- **Cross-link discovery improvements** — orthogonal to ingest quality. The Lai fixture loses points on `related_papers` because the proposer (k=8 + strict judge) misses parallel-trial papers like `gupta-2026`. A targeted fix would add a "same-therapeutic-target / same-method-category" auto-flag heuristic in `agents/phases/crosslinks.py:propose_crosslinks` — bypass the LLM judge for high-confidence parallel-work pairings. The naïve k=12 + judge-loosening attempted this session was indeterminate single-shot; needs a more targeted heuristic or N=5 validation.

### Methodology assumptions worth retesting

- **N=5 may not generalize to small effects.** Yang's baseline SD=7% on the LLM-judged scoreboard means Cohen's d=0.74 (medium-large) didn't clear |t|>2 at N=5. For lever decisions on changes in the ±5pp range, N=10 per fixture is the realistic floor (~$2 per fixture per condition with `--llm`).
- **Curator-judge correlation likely inflates scores by 5–10pp.** Pending R4 validation. Current numbers should be read as "a Claude-curated fixture judged by Claude" — favorable upper bound.

### Bulk re-ingest gotchas (2026-06-14)

Re-ingested 50 idea-page-referenced papers onto the L1+L2+L3+L6 pipeline. 48/50 stems preserved naturally; 2 (avsec, linder) had silent year-drift renames that left orphan pages and moved PDFs. Manually recovered.

Diagnosis of the 2 renames: the `--year` override flag works correctly and DOES lock stem derivation. My recovery wrapper had a bug — it extracted the override year from the existing page's YAML, which had drifted from the stem during earlier in-session re-ingests. Lesson: the durable identity of an existing page is the STEM (encoded in the filename), not the YAML.

One real fix shipped: `promote.py` now refuses to commit when reconcile detected a prior page AND the new derived stem differs. `--allow-rename` opts in. Catches the silent-orphan failure mode at write-time before any state mutation.

### Local hygiene items

- `.researchwiki.db` is a stray copy of the local state DB in the repo root; canonical path is `~/.local/share/researchwiki/state.db`. Add to `.gitignore` so it doesn't get accidentally committed.

## Out of scope for this plan

- Adding more fixtures beyond the current six (a dataset paper, a theory paper, a Bioinformatics-style methods paper) — wait until R4 quantifies inter-rater reliability or until a specific failure type surfaces that the current six don't catch.
- **L5** — adversarial reader pass at runtime. Deferred — the eval scorer's `--llm` mode is a fixture-anchored version of the same idea, and the rolled-back completeness-review experiment didn't show a clear signal at single-shot. Revisit only if a runtime quality gap is observed that the eval harness can't catch post-hoc.
- **L7** — paper types beyond research/review/clinical-trial (dataset, theory, position). Premature without a specific paper-type-failure signal; the dynamic prompt loader infrastructure already supports it when needed.
- Reference-free SummaC/QAFactEval upgrade to the grader — defer; existing grader signals (BM25 + bi-encoder + numeric integrity + negation parity) are sufficient for current decisions, and adding model dependencies should wait for a clear motivating failure.
