# Idea-page authoring — procedure

Trigger: when writing a new idea page (`wiki/ideas/<slug>.md`) or substantially editing an existing one (rewriting Opportunities/Plans, adding new design components, regenerating from a deeper conversation). Use this alongside CLAUDE.md §4 — that's the canonical reference; this file walks the actual authoring workflow.

## Section contract

Every idea page has five H2 sections in this order (CLAUDE.md §4):

1. **Verdict — strength + tl;dr.** A one-line strength label (strong / incremental / weak) and a one-paragraph tl;dr summary so a reader can decide in 30 seconds whether to read the full page. Strict-grounded: the tl;dr cites the load-bearing wiki anchors that make the design work.
2. **Background — why this idea is worth considering.** Source-supported. Establishes the current state, the limitations, what makes the direction possible now.
3. **Opportunities — the design proposal.** What could be built. Wiki-grounded principles + LLM-contributed extensions.
4. **Plans — how to actually build this.** Staged implementation path with stop-go checkpoints per phase.
5. **Caveats — where the optimistic case could break.** Source-supported pessimism. What would invalidate the design, what would change the conclusion.

Sub-headings inside any section use H3, never H2 — with one exception: a `## References` H2 at the end, required whenever the page uses `[^id]` footnotes (step 2 below), since that's where they're defined. `## Related Papers` is a *paper*-page section and does not belong on an idea page; some existing idea pages carry it as an empty trailing stub, which is a defect to clean up rather than a pattern to copy.

**Neither page gate enforces this contract** — both parse *units* (paragraphs and bullets) and never read a heading, so a page whose Verdict prose sits above the first H2 with no `## Verdict` heading passes both green. `researchwiki/grade/grounding.py`'s `_PERMISSIVE_IDEA_SECTION_RE` matches only `^(opportunities|plans)\b`; its job is to locate the model-prior-eligible ranges, not to validate the five section names. The check that *does* see headings is `researchwiki lint` → `idea_contract_violations` (`researchwiki/tasks/lint/idea_contract.py`): heading presence, order, unexpected H2s, `verdict:`-vs-label agreement, and footnote/`## References` hygiene. It's **advisory** — reported, never exit-code-flipping — so read it rather than relying on lint's exit status.

Because those permissive ranges run from each H2 to the next, H3 sub-sections inherit their parent section's sourcing rules.

### Verdict labels

The label is a holistic judgment over three axes — **impact**, **novelty**, and **feasibility** — not about how wiki-grounded the design is. Wiki-grounding is already tracked separately (check-grounding's wiki-cited-vs-model-prior count, the `*(model prior)*` markers). The verdict answers a different question: *if this got built, how much would the field care, how new would it actually be, and could it realistically be done?*

The three axes:

- **Impact** — would the field meaningfully care if this worked? Solves an open problem, beats the frontier on benchmarks the field tracks, unlocks downstream work, or just a marginal improvement?
- **Novelty** — is this genuinely new, or known-technique-in-new-clothes? A novel composition of established components counts; a novel framing of an old problem can count; a slight reparameterization of an existing model usually doesn't.
- **Feasibility** — given current data, hardware, and methods, can this actually be built? An idea that requires unobtainable data, a 10× compute jump, or a method that doesn't exist yet is not feasible regardless of impact.

The three axes interact — a novel, impactful idea that's not feasible is **weak** (it won't be built); a non-novel but high-impact, very feasible project is **incremental** (good engineering, not research); only when all three are at least moderate-to-high does an idea earn **strong**.

| Label | When |
|---|---|
| **strong** | High on all three axes (or two high, one solidly moderate). Genuinely novel, the field will care, and it can plausibly be built with available resources. The page should be able to point at concrete wiki anchors for each of impact, novelty, and feasibility. |
| **incremental** | Mixed across axes — at least one axis is moderate-to-high, but at least one is weaker. Examples: high-impact but mostly-known technique (engineering rather than research); novel and feasible but the impact case is conditional on an unresolved open question; novel and high-impact but the feasibility hinges on a not-yet-available piece (data, hardware, prerequisite method). The default label when the design is real but doesn't clearly clear all three bars. |
| **weak** | At least one axis is *blocking* — not feasible (key data/hardware/method missing), not novel (the wiki already contains a paper that does this or did something equivalent), or not impactful (niche use case, marginal improvement, no readers would change their workflow). Worth keeping as a record so the negative-result reasoning isn't lost. |

The verdict is independent of how well-supported the design is on paper. A design can be **strong** while leaning heavily on `*(model prior)*` clauses (high-upside, high-risk research bet whose key premises are LLM-contributed); a design can be **weak** while every premise is wiki-cited (well-supported but uninteresting, infeasible, or already-done). Be honest about all four axes — impact, novelty, feasibility, and grounding — readers calibrate against the verdict.

## Sourcing rules per section

Detailed table in CLAUDE.md §4. Distilled here:

| Section | Sourcing |
|---|---|
| Verdict | Strict-grounded. The strength label is the LLM's synthesis judgment (no citation needed for the label itself), but the tl;dr cites the load-bearing wiki anchors via footnotes so a reader can audit *why* the verdict came out the way it did. No `*(model prior)*` allowed in the tl;dr. |
| Background | Strictly grounded. Every claim cites a wiki paper via `[[wikilink]]` or footnote. |
| Opportunities | **Actively contribute design ideas from training knowledge.** Each wiki-grounded principle gets `[[wikilink]]`; each LLM-contributed component gets `*(model prior)*`. Numbers / benchmark results / named-entity attributions still need a wiki citation. |
| Plans | Same as Opportunities. Phase goals and deliverables that tie to specific wiki papers cite them; tactics that are LLM-knowledge contributions get `*(model prior)*`. |
| Caveats | Strictly grounded. Speculative pessimism is uninformed; cite the real invalidators. |

## When to use which mark

Three citation/exemption mechanisms exist. They cover disjoint cases — pick by what kind of sentence you're writing:

| Sentence is… | Mechanism |
|---|---|
| A subject-matter claim (about CRISPR / ML / methods) supported by a wiki paper | `[[wikilink]]` to the paper |
| A subject-matter claim sourced from the model's training knowledge (idea-page Opportunities/Plans only) | `*(model prior)*` after the sentence |
| About the page itself (page-meta) — the page contract, status lifecycle, authoring instructions | Don't write it on the page. Either delete (it's already in CLAUDE.md §4) or move to this prompt file. |

**Skip-grounding is rarely needed.** The `<!-- skip-grounding-start/end -->` mechanism exists for genuine page-meta that has no other home, but most page-meta on idea pages was duplicated from CLAUDE.md and should just be deleted. If you find yourself reaching for skip-grounding, ask first whether the wrapped sentence belongs on the page at all.

## Don't duplicate CLAUDE.md content on the page

The five-section contract, the `status:` lifecycle (open / scoping / validated / superseded / abandoned), the validated/superseded transition rules — all of those live in CLAUDE.md §4 and load every turn. **Don't restate them on the page.** A reader who needs them has CLAUDE.md; an LLM authoring/editing the page already has CLAUDE.md in its context. Repeating them creates page-meta that the linter doesn't know what to do with and obscures the page-specific framing.

What stays on the page:

- The page-specific premise — why *this* idea is worth a page.
- The page-specific (a)/(b)/(c) `status:` triggers — *what would shift this particular page* between lifecycle states (a different formulation than the generic lifecycle definition).
- The page-specific gap-listing in Caveats — *which* missing papers would change *this* conclusion, not the generic instruction to ingest-and-update-status.

## Authoring workflow

1. **Survey the wiki.** First, check for existing **synthesis pages** that already map this field — `researchwiki search "<topic>"` and `ls wiki/synthesis/`. If a relevant synthesis exists (it retrospectively maps the same landscape this idea proposes to extend), it's the cheapest grounding source: its body citations (inline `[[wikilink]]`s + `## References` footnotes) are the candidate citation list, its narrative often supplies the Background's current-state-and-limitations framing, and its tensions/open-questions section often pre-maps Caveats. **Reuse, don't re-derive.** Add the synthesis to the page's `companion_synthesis:` YAML field (quote each wikilink — `- "[[synthesis/…]]"` — so Obsidian renders it as a link rather than showing "?"), cite it via `[[wikilink]]` where its framing is load-bearing, and skip re-surveying papers it already covers. Then run `researchwiki claims "<topic>"` for any aspect the synthesis doesn't cover, and `researchwiki claims --by-stem <stem>` for each paper you plan to cite directly. While surveying, jot down *gaps* — claims you'd want to cite but no wiki paper or synthesis supports — for the ask step (§ Ask the user for missing papers).
2. **Draft Background.** Lead with current state and limitations, source every claim. Footnotes (`[^id]` / `[^id]: [[wikilink]]`) keep the prose readable when one paper supports many sentences.

   When a load-bearing claim rests on figure data the caption doesn't settle — a trend, an axis range, a panel the text only gestures at — `researchwiki figures <stem>` lists captions for free and `--figure N` renders that one page. One page, and only when the caption falls short; the render is free but reading it spends context.
3. **Draft Opportunities.** Build the design table — one row per design principle, each tied to the wiki paper that established it. Then add a `### Beyond-wiki design extensions` H3 with the LLM-contributed components — alternative architectures, training strategies, calibration techniques, etc. — each marked `*(model prior)*`.
4. **Draft Plans.** Stage the implementation to fit the design's natural shape — the existing example page uses six phases, but adjust the count to the work. Cite the wiki paper that grounds each phase goal; mark LLM-knowledge tactics within phases with `*(model prior)*`. Each phase needs a stop-go checkpoint.
5. **Draft Caveats.** Two H3 sub-sections work well: limitations of the load-bearing dataset/method (cite invalidators) and *What would change the conclusion* (page-specific gap-listing — which missing papers would shift this idea's status, not the generic ingest-and-update-status instruction).
6. **Draft Verdict last but place at the top.** After Caveats is settled (so you have an honest read on premises and invalidators), write the strength label and the tl;dr paragraph. Place the section *first* in the page, before Background. The label is the LLM's synthesis judgment over the whole page; the tl;dr summarizes the design in one paragraph and cites the load-bearing wiki anchors via the page's existing footnotes. Writing it last avoids "aspirational verdict" — the label has to honestly reflect the rest of the page. If you'd write **strong** but Caveats list real unresolved invalidators, the right label is **incremental** or **weak**. **Mirror the label into the YAML `verdict:` property** (`strong` | `incremental` | `weak`) so the index/log and any Dataview cut can triage on it without parsing prose — keep the frontmatter value and the section's label identical.
7. **Classify the content category.** Set YAML `category:` to the design's dominant **content** field — the same category an ingested paper on this topic would get — *not* `ideas` (the directory already carries the page type). Classify it the way ingest does: pick the existing content category (`ls wiki/` for the valid set, e.g. `ai`, `single-cell`, `compbio`) that the design most centrally sits in, abstaining to `other` only when nothing fits. In practice the dominant category among the papers the idea cites (its body `[[wikilink]]`s / `## References` footnotes) is the answer — an idea whose principles all trace to `ai/` papers classifies as `ai`; one built on `single-cell/` papers classifies as `single-cell`. The page still lives in `wiki/ideas/` and `db rebuild` still records its DB category as `ideas`; the YAML value is content-grouping bookkeeping (Obsidian property view, `index.md`, `views.md`). Must be a valid content category or `lint`'s category-drift check flags it.
8. **Ask the user for missing papers** — see § below. Do this *before* index/log so the user can iterate on the draft with new sources before it's wired in.
9. **Update `wiki/index.md`** — append the page under `## ideas`. The index entry should lead with the verdict label so the index acts as a triage view.
10. **Append to `wiki/log.md`** — `## [YYYY-MM-DD] idea | <title> [<verdict>] → wiki/ideas/<slug>.md`.

## Ask the user for missing papers

Idea pages compound when their design components and caveats trace to specific wiki papers. After the first draft is written, surface the gaps to the user and ask them to drop relevant PDFs into `inbox/`. This is the idea-page analogue of CLAUDE.md's *flag adjacent gaps* corollary: don't silently fall back to model priors or soft pessimism when a missing paper would let you make a stronger claim.

**When to ask** (each maps to a specific page weakness the ingestion would fix):

| Symptom on the draft | What the missing paper would fix |
|---|---|
| A Background claim is load-bearing but ungrounded — model priors aren't allowed in Background, so you've either left it cited weakly or omitted it. | Lets the claim move from omitted/weak to a `[[wikilink]]` citation that anchors the design's premise. |
| An Opportunities clause carries `*(model prior)*` but the user clearly wanted a stronger argument (or the marker is doing too much load-bearing work for the section). | Lets the clause move from `*(model prior)*` to a `[[wikilink]]`, strengthening Opportunities and shrinking strict-mode ungrounded count. |
| A Plans phase's tactic is `*(model prior)*` but a real published technique exists — naming it would replace LLM speculation with a citation. | Same as above for Plans. |
| A Caveats invalidator is asserted but uncited (Caveats is strict — no `*(model prior)*` allowed). | Lets the invalidator be cited, restoring the section's "real-pessimism, not speculation" contract. |
| The design is "X but better"; a recent paper directly competes and the user hasn't ingested it yet. | Either supersedes the design (move status to `superseded` after ingest) or sharpens its differentiation. |
| A `What would change the conclusion` bullet names a paper class (e.g., "an MoE genomic LM paper") but no such paper is in the wiki — the bullet is an open invitation. | Pre-ingesting one of those papers either confirms the listed risk or removes the bullet. |

**How to ask** — at the end of the first draft (or after `check-grounding`), present a short, specific list:

- One bullet per paper or paper-class. Name a representative paper or technique class — *"a recent MoE-on-byte-tokenized-genomes paper, e.g. {specific candidate if you can name one}"* beats *"more papers on efficient training."*
- Tie each ask to where on the page it would land. *"If you drop X, the clause Y in Opportunities can move from `*(model prior)*` to a `[[wikilink]]`,"* or *"would let me cite a real invalidator in Caveats §Limitations rather than asserting it."*
- Cap at ~3–5 asks per draft. Past that, the user is being asked to ingest a small literature; pick the highest-leverage gaps.
- Optionally re-rank: which ingest would most change the design, vs. which would just add citations. Lead with the design-changers.

**When NOT to ask:**

- If the design is already well-supported — every load-bearing Background/Caveats claim is cited, every Opportunities/Plans `*(model prior)*` clearly belongs in the model-priors set (architectural composition tactics, training tricks not specific to a domain).
- For tangentially-relevant topics. The page is the unit of focus; don't pad the ask list with adjacent literature the user might find interesting but that wouldn't change the page.
- For paper classes the wiki already covers densely — re-ingestion is friction.
- During substantive edits where the user has already responded to a prior ask. Don't re-ask for things they declined or deferred.

**Iterating after the user ingests:** after each new ingest, re-run `researchwiki check-grounding` and audit which `*(model prior)*` clauses can now move to `[[wikilink]]` citations. Move them — then re-run `researchwiki grade synthesis` to confirm the new citations *hold* (fidelity catches a clause moved to a `[[wikilink]]` whose paper doesn't actually support it). The default-mode → strict-mode delta is the metric: each ingestion that lets a model-prior clause become a wiki citation tightens the page.

## Verify before committing

```bash
researchwiki check-grounding wiki/ideas/<slug>.md          # MANDATORY structural gate — must exit 0
researchwiki check-grounding wiki/ideas/<slug>.md --strict  # audit: every model-prior clause
researchwiki grade synthesis wiki/ideas/<slug>.md           # MANDATORY fidelity gate — must exit 0
researchwiki check-coverage wiki/ideas/<slug>.md            # ADVISORY recall surface — review unreferenced hits
```

Both grading gates are **mandatory before the page is done** and they're orthogonal — neither subsumes the other, so run both:

- **Structural** (`check-grounding`) must hit 0 ungrounded. Every claim is either wiki-cited or marker-tagged in an eligible section. The annotated output shows `⚠ model prior` next to each marker-tagged clause — review them as a final pass. Catches a *missing* citation.
- **Fidelity** (`grade synthesis`) must hit 0 `misattributed`. Each claim that *cites* a paper is graded against that paper's PDF; a `misattributed` verdict means a number in the claim appears in **none** of the cited papers — fix the number or the citation (you cited the wrong paper). Catches a *present-but-wrong* citation that structural passes green. `*(model prior)*` clauses carry no citation, so fidelity skips them (`uncited`); `weak`/`composite` are advisory and never fail the run.
- **Strict mode** (structural-only) will flag every model-prior clause as ungrounded — that's by design (strict surfaces every spot the page leans on training knowledge). The strict report is a *map of model-prior contributions*, not a failure list. Use it to audit: are these really LLM-contributable claims, or has an empirical claim drifted into Opportunities without a citation?
- **Recall** (`check-coverage`) is an advisory third surface, run after both gates pass. It re-ranks paper pages against the page's `topic_seed` and surfaces top-N hits the page doesn't cite. Treat each one as a deliberate scoping decision — cite it, narrow the `topic_seed` so it stops landing in the top-N, or leave it as a known exclusion. Exit 1 doesn't fail the page; an *unreviewed* exit 1 does.

**Neither gate reads headings.** Both parse *units* (paragraphs and bullets), so a missing or misordered H2 — including a missing `## Verdict` — is invisible to them. Add a third command for the structure:

```bash
researchwiki lint --json | jq '.idea_contract_violations[] | select(.page | endswith("<slug>"))'
```

Advisory, so it won't fail lint — read the findings. It covers heading presence and order, unexpected H2s, YAML `verdict:`-vs-section-label agreement, and footnote/`## References` hygiene.

If the structural pass flags an ungrounded unit, the fix is one of:
- Add a `[[wikilink]]` (the claim is supportable by an existing wiki paper).
- Add `*(model prior)*` (the claim is an LLM-knowledge contribution AND it's in Opportunities or Plans).
- Rephrase or delete (the sentence is page-meta or a meta-statement that doesn't belong on the page).

## When the wiki ingests something that affects the page

- Cited paper updates: rerun `researchwiki check-grounding` and `researchwiki grade synthesis` after the citation graph rebuild. The marker-recognition is robust to renamed wikilinks as long as the linter sees a `[[…]]`.
- A paper realizes the proposed design: update `status:` to `validated`, ingest the paper, add a "Defining paper" link near the top, prune `Beyond-wiki design extensions` items the published design covers (their content is now wiki-supportable).
- A paper supersedes the design: update `status:` to `superseded`, ingest theirs, leave the page as historical record.
- A paper invalidates a load-bearing premise: update `status:` to `abandoned`, leave the page as historical record.

The CLAUDE.md `status:` lifecycle definition is the canonical rule for these transitions; this prompt only describes the authoring side.
