# Concept-page authoring — procedure

Trigger: when writing a new concept hub note (`wiki/concepts/<slug>.md`) or refreshing an existing one after new papers join the concept. A concept page is a **mini-synthesis around a single recurring term** — a hub that ties together every wiki paper that instantiates the concept, so the graph gains a bridge node the citation graph and semantic-KNN both miss. Use alongside CLAUDE.md §Page Types.

A concept page earns its existence when a term recurs across **≥3 papers** (surfaced by `researchwiki lint`'s concept-candidate check) — and is most valuable when those papers **span categories** (`concept_span ≥ 2`), because then the hub is the only thing linking otherwise-siloed domains. A term confined to one category is a weaker candidate; prefer a synthesis page there.

## What a concept page is *not*

- **Not a synthesis page.** Synthesis maps a *field* or compares approaches; a concept page defines *one term* and enumerates where it appears. If you're writing more than a paragraph of cross-paper *argument*, you want `wiki/synthesis/`.
- **Not an idea page.** No forward-looking design, no model priors. Concept pages are **strictly grounded** end to end — every claim cites a wiki paper.
- **Not a glossary stub.** A one-line dictionary definition with no member papers is noise. The value is the spoke list.

## YAML contract

```yaml
---
title: "Retrieval-augmented generation"
type: concept
category: [ai]                 # dominant CONTENT category of the member papers
referenced_papers:             # the hub spokes — every member paper
  - [[ai/gutierrez-2024-hipporag-neurobiologically-inspired-long-term-memory]]
  - [[ai/edge-2025-from-local-to-global-a-graphrag]]
concept_span: 3                # distinct categories the term spans (from lint)
generated_at: 2026-07-03
topic_seed: "retrieval-augmented generation"
tags: [concept, rag]
---
```

- `type: concept` and living under `wiki/concepts/` are both required — `lint`'s page-type check flags a mismatch either way.
- `category:` is the dominant **content** category of the member papers (`ai`, `single-cell`, …), *not* `concepts` (the directory already carries the page type). Must be a valid content category or category-drift flags it. Mirror the idea/synthesis convention.
- `referenced_papers:` lists the member papers — the same set the body links to. Keep it in sync with the body's spokes (used by the `views.md` dashboard and as the coverage baseline).
- `concept_span:` is provenance from the extractor — how many categories the term bridges. Bridges (span ≥ 2) are the reason the page exists; record it.

## Section contract

Two required H2s, two optional. Sub-headings are H3.

1. **Definition** *(required, strict-grounded)* — 1–3 sentences: what the concept is, as the member papers use it. Every sentence cites a member paper via footnote (`[^id]`). Don't import a textbook definition the papers don't state — ground it in how *these* papers frame it.
2. **How it appears across the corpus** *(required)* — the spoke list. One bullet (or H3) per member paper: how *that* paper instantiates, uses, or extends the concept, with an inline `[[wikilink]]` to the paper. This is the hub's load-bearing content. Order by category so cross-domain span is visible at a glance.
3. **Cross-domain connections** *(optional, strict-grounded)* — when `concept_span ≥ 2`, a short paragraph on how the concept manifests *differently* across domains (e.g. retrieval-augmentation as graph traversal in `ai` vs. reference-based nearest-neighbor lookup in `single-cell`). This is the bridge payoff — the insight a same-category reader wouldn't see. Strictly grounded; if you can't cite the cross-domain contrast, omit the section rather than speculate.
4. **What would update this page** *(optional)* — which paper classes, once ingested, would extend or reframe the concept. Exempt from `check-grounding` by the exact heading (name-narrow). Keep it to concrete paper types, matching the CLAUDE.md gap-flag corollary.

`## What would update this page` is the only heading `check-grounding` skips — every other section is held to strict grounding.

## Sourcing

Strictly grounded throughout — **no `*(model prior)*`** (that marker is idea-page-only; here it has no effect and signals the wrong page type). Two citation forms, same as synthesis pages:

| Where | Form |
|---|---|
| Prose (Definition, Cross-domain connections) | Academic footnotes — `[^id]` in the sentence, `[^id]: [[wikilink]]` at the bottom. Keeps prose readable when one paper backs several sentences. |
| Spoke list (How it appears) | Bare inline `[[wikilink]]` per bullet, as on paper-page *Related Papers*. Footnotes are for prose, not list rows. |

`check-grounding` resolves both forms. Never write `claim_id:NNN` into the page (row keys, reassigned on `db rebuild`).

## Reciprocal linking

A hub is only half-built if the spokes don't point back. Each member paper gets a reciprocal `[[concepts/<slug>]]` link in its *Related Papers* section, tagged so provenance is clear:

```markdown
- [[concepts/retrieval-augmented-generation]] — instantiates this concept (auto-added; concept-link)
```

The `researchwiki concepts` task applies these reciprocally (like `claim-overlap` does for paper-paper edges); when authoring by hand, add them yourself or `lint` will flag `missing_backlinks` for every spoke. This bidirectional wiring is the point — it's what drains the one-way-link debt and makes the hub navigable from any member.

## Authoring workflow (scaffold-first)

1. **Scaffold.** `researchwiki concepts <term>` gathers the member papers (pages mentioning the term), pulls each one's top graded claims, and writes a grounded stub with the spoke list pre-populated. (Digest-style manual path: create the file, fill the YAML, and assemble the spoke list from `researchwiki lint` + `researchwiki search "<term>"`.)
2. **Write the Definition** from the members' own framing — `researchwiki claims "<term>"` surfaces the pre-graded, citable units. Cite each sentence to the paper it came from.
3. **Fill the spoke list.** For each member, one line on how that paper uses the concept. `researchwiki claims --by-stem <stem>` dumps a paper's citable surface. Keep bullets specific ("uses PPR over a passage graph" beats "uses retrieval").
4. **Add Cross-domain connections** only if the span is real and you can cite the contrast.
5. **Set `concept_span:`** from the extractor's report and order the spoke list by category.
6. **Update `wiki/index.md`** — append under a `## concepts` section: `[[concepts/<slug>]] — one-line what-it-bridges.`
7. **Append to `wiki/log.md`** — `## [YYYY-MM-DD] concept | <title> (span N) → wiki/concepts/<slug>.md`.

## Verify before committing

Both gates mandatory (must exit 0), same as synthesis/idea pages:

```bash
researchwiki check-grounding wiki/concepts/<slug>.md   # structural — every claim carries a citation
researchwiki grade synthesis wiki/concepts/<slug>.md   # fidelity — each cited claim holds in its paper
researchwiki check-coverage  wiki/concepts/<slug>.md   # advisory — wiki papers ranking high on topic_seed the page omits
```

- **Structural** must hit 0 ungrounded. A concept page has no model-prior escape hatch, so every flagged unit is a real fix: add a `[[wikilink]]`, or delete the sentence.
- **Fidelity** must hit 0 `misattributed` — a spoke bullet that credits a paper with something it doesn't say fails here even when structural passes.
- **Coverage** is advisory: it surfaces papers that rank high on the concept's `topic_seed` but aren't spokes. Each hit is a candidate member — add it (and its reciprocal link) or record why it's excluded.

## When the wiki ingests something that affects the page

- A new paper uses the concept → add it as a spoke (+ reciprocal link), update `referenced_papers:`, bump `concept_span:` if it's a new category, refresh `generated_at:`. The post-ingest concept-attachment step does this automatically; verify and re-run the gates.
- The concept fragments into distinct sub-meanings across new papers → consider splitting into narrower concept pages, or promoting to a synthesis page if the cross-paper story has grown past enumeration into argument.
- Staleness: concept pages are tracked by `lint` like synthesis/idea pages (via `generated_at` vs. member mtimes). A `stale_by_content` flag means a member paper changed after the last refresh — re-verify the spoke.
