# Synthesis-page authoring — procedure

Trigger: when writing a new synthesis page (`wiki/synthesis/<slug>.md`), folding newly ingested papers into one, or upgrading a page that predates the fixed structure. This file is the procedure and the citation-format reference; CLAUDE.md §2 is the short contract.

Claude Code reaches this file through the `synthesis-page` skill (`.claude/skills/synthesis-page/SKILL.md`); every other agent reaches it through the pointer in CLAUDE.md. Both read this same file, so the contract cannot drift per tool.

A synthesis page is the wiki's review article on a subject. The bar is a review in a good journal: a newcomer can follow it, an expert learns something from it, and it is organized around *ideas* rather than around the papers it cites. A page that walks the corpus one paper per bullet has failed even if every bullet is accurate — the reader could have read the paper pages.

## Contents

1. [The fixed structure](#the-fixed-structure)
2. [Writing so it argues instead of lists](#writing-so-it-argues-instead-of-lists)
3. [Citations and source labels](#citations-and-source-labels)
4. [Mode A — new page](#mode-a--new-page)
5. [Mode B — update with new papers](#mode-b--update-with-new-papers)
6. [Mode C — upgrade a legacy page](#mode-c--upgrade-a-legacy-page)
7. [Verify, then land](#verify-then-land)
8. [When a page gets too big](#when-a-page-gets-too-big)

---

## The fixed structure

Every synthesis page has these H2 sections, with these exact names, in this order. Nothing else at H2. Sub-structure goes in H3 (and H4 if a theme genuinely needs it).

| H2 | What it is for | Grounding |
|---|---|---|
| `## Question` | Scope. What the page covers, what it deliberately leaves out and why, and links to neighbouring synthesis pages that cover the excluded parts. One or two short paragraphs. | strict |
| `## Short answer` | The page's argument in miniature: one framing sentence, then 4–6 takeaways a reader could repeat to a colleague. Keep the page's key vocabulary here — the semantic index embeds this section first and truncates. | strict |
| `## Background` | What a newcomer needs before the rest makes sense: the problem, the key terms, the measurement or experimental setup the field shares. Define every term the later sections lean on. | strict |
| `## Organizing framework` | How this page sorts the evidence, and why that sorting explains the field. Then a table placing every cited paper. | strict |
| `## Findings` | One H3 per theme the framework defines. This is the body of the review. | strict |
| `## Cross-cutting insights` | What appears only when the themes are read together: where they agree or conflict, mechanisms they share, trade-offs no single paper names. On a comparative topic, also when to use which. | strict |
| `## Tensions / open questions` | Genuine disagreements and unresolved questions, with the evidence on each side. | strict |
| `## Outlook` | Where the field is heading and what would settle the open questions. The one section that may go beyond what the papers state — see [source labels](#citations-and-source-labels). | labelled |
| `## References` | One footnote definition per cited paper. | — |

### Why one structure for every topic

The headings are deliberately neutral so they fit any kind of synthesis. What changes from topic to topic is the *organizing scheme* — the thing you choose in `Organizing framework` — not the sections:

| Kind of topic | Organizing scheme | What a `Findings` H3 is |
|---|---|---|
| Methods landscape ("approaches to off-target prediction") | approach families, or the design decisions every method must make | one family / one decision |
| A single question ("do DNA LMs use evolutionary signal?") | lines of evidence | one line of evidence and what it shows |
| A collection of facts ("LDL-C-associated variants") | categories of fact, by mechanism or by type of evidence | one category, with what its facts show together |
| One finding seen from several angles | the angles (genetic, clinical, mechanistic, …) | one angle |
| A field's trajectory | eras or turning points | one era and what it unlocked |

So a fact-collection page does not need a "comparative assessment", and a methods page does not need eras. If you catch yourself wanting an extra H2 ("Benchmarks", "When to use which", "Regulatory anchor"), it belongs as an H3 inside `Findings` or as part of `Cross-cutting insights`.

### `Organizing framework` in detail

This is the page's most abstract section and the one readers struggle with most, because a scheme is an idea about other ideas. Write it **concrete first**, in this order:

1. **Open with one real case from the corpus**, in two or three sentences: two papers that look like they disagree, or two results a reader would wrongly compare. Name them, give the numbers, say what a reader would conclude from them unaided. No axis vocabulary yet.
2. **Show the scheme resolving that case.** "They disagree because one assay has chromatin and the other has none" — the explanation comes before the category it illustrates.
3. **Then name the axes**, now that each one has an instance attached: "That difference — the substrate a study used — is the first of two axes this page sorts on."
4. **Then one worked example per remaining axis**, same shape: case, then the axis it names.
5. **Say what the scheme buys**, in a sentence a reader can test: what it predicts, what it stops you doing. This is the page's thesis about the field — the best existing page frames scRNA-seq analysis as "five coupled decisions", which is what lets every later section argue rather than list.
6. **The placement table.** One row per cited paper (or per tool, when a paper contributes several), columns for the axes plus one short note. Use bare `[[stem]]` links in tables — footnotes don't render inside markdown tables, and never use `[[stem|alias]]` (the pipe breaks the table). Keep table cells to names, categories and short labels; put numbers and claims in prose, where each one can be graded against its source.

The table is also the page's index for later updates (Mode B): a new paper is placed by adding a row.

Keep this section to the scheme. Caveats about which papers are only adjacent to the topic belong in `Question` (scope) or on the table rows, not here — a reader meeting the scheme for the first time cannot use an exception to it.

**Before and after.** The abstract-first version a reviewer called hard to read:

> The second axis is the one that makes the evidence comparable, and it cuts across the first: **which condition a study varied**. The candidates are sequence alone, donor genotype, chromatin state, DNA topology, editor class, cell type, and delivery format. This is the right second cut because the corpus's apparent disagreements are almost all condition mismatches. A biochemical assay and a cellular assay "disagree" about a site list because one has chromatin and one does not.[^change-seq]

Concrete first, same content:

> CHANGE-seq, run on purified DNA, finds a site list that cellular GUIDE-seq largely rejects: targeted sequencing confirmed 98.3% of the sites the two assays shared, but only 18.3% of the sites CHANGE-seq found alone.[^change-seq] Read as a disagreement about which sites are real, that is a contradiction. It isn't one. Purified DNA has no chromatin, so the two assays are answering different questions, and the 18.3% is the price of asking the easier one.
>
> **What a study varied is therefore the second axis**, cutting across the workflow stage: sequence alone, donor genotype, chromatin, DNA topology, editor class, cell type, delivery format. Most apparent conflicts in this corpus resolve the same way the CHANGE-seq one does — two high-fidelity variants rank in opposite orders because one was tested as plasmid and the other as a ribonucleoprotein.[^hifi] Sorting by stage alone makes these look like contradictions; sorting by stage *and* condition makes them look like results from different experiments.

### `Findings` in detail

Each H3 is a theme, never a paper. Within it, follow the shape **point → evidence → conditions → limits**:

- Open with the theme's claim in one sentence.
- Give the evidence, combining papers in the same paragraph when they bear on the same point.
- State the conditions each result holds under (see the writing rules below).
- Close with where the theme stops working or what it cannot yet show.

One concrete example per theme — a worked case, a number with its context, a named failure — does more for a newcomer than three abstract sentences.

---

## Writing so it argues instead of lists

These rules are why the page exists. Each comes with its reason so you can apply it to cases the rule doesn't name.

**Organize by idea, not by paper.** A paragraph makes one point about the field, and papers are its evidence. If a paragraph is centred on one paper, it has to say what that paper changed in the argument — what was believed before, what it showed, what that rules in or out. *Why:* the reader already has one page per paper; the synthesis earns its place only by saying what the papers mean together.

**No catalog bullets.** The pattern `- **ToolName**[^x] is a … that …` is the single most common failure in this wiki (37 of 42 bullets on one existing page). Write prose. Use bullets only for content that is genuinely a list: decision rules, criteria, a set of parallel independent facts. Even then, the sentence before the list says what the items show together. *Why:* a bullet per tool reads as complete while saying nothing about how the tools relate.

**State the conditions behind every result.** For each number or finding, say what system or cohort it came from, what it was measured or compared against, and who ran the comparison (the method's own authors, or an independent benchmark). *Why:* results in this corpus are rarely comparable as stated — one benchmark tuned the cluster count, another fixed it — and a reader who sees two numbers side by side will compare them unless told not to. This habit is what separates the best existing page from the rest.

**Quote a number in the paper's own form.** Write "4% of wild-type activity" if that is what the paper says, not the equivalent "0.04×"; "8.345 × 10⁻¹⁶", not a rounded "8.3 × 10⁻¹⁶". *Why:* `grade synthesis` matches numeric tokens against the PDF text, so a correct paraphrase fails as `misattributed` and costs a round trip to the PDF to clear. Converting units or re-deriving a percentage has the same effect. If the number you want isn't printed in the paper, either cite the printed one or make the claim qualitative.

**Open each H2 with its takeaway — and cite it.** The first sentence of a section is its conclusion; the rest supports it. *Why:* someone skimming section openings should get the whole argument, and putting the conclusion first forces you to have one. These framing sentences are the most common `check-grounding` failure: a summary of the section's cited claims is itself a claim, so carry the footnotes of the claims it summarizes rather than deleting the sentence.

**Explain before you evaluate.** Background defines terms before Findings uses them. Spell out an acronym at first use. Prefer the plain word ("cut", "edit", "reads") to the jargon when both are accurate. *Why:* "easy to understand" is half of the brief, and a review that only experts can read is a literature dump with better formatting.

**Show an instance before you name an abstraction.** This is the readability rule that matters most, because the abstract passages are where these pages lose readers. Whenever a sentence's subject is a category rather than a thing — *an axis, a tier, a trade-off, a failure mode, a mechanism, a methodological problem* — the reader must already have seen one concrete case of it. So: the case, then the name for the class of cases, then the generalization. Never the reverse, and never a chain of three abstract sentences before the first example.

The rule applies everywhere, not just in `Organizing framework`:

| Section | What goes wrong | The fix |
|---|---|---|
| `Background` | defining a term with other terms the reader also doesn't have | define it by what it does in one concrete study |
| `Organizing framework` | naming axes, then asserting they're the right cut | lead with the case the scheme resolves (see above) |
| `Findings` | a theme opening with its abstract claim and no instance for a paragraph | one worked case right after the theme's claim sentence |
| `Cross-cutting insights` | an insight stated as a general property of the field, evidence after | name the two results that collide, then the pattern they show |
| `Outlook` | an inference whose premises are abstractions | point at the specific findings it is drawn from |

*Why:* a reader can hold one concrete case and generalize from it, but cannot instantiate an abstraction they have no example of — they read the words and keep nothing. This is also the test for whether you actually have the insight: an insight you can't attach a case to is usually a restatement of the section heading.

**Prefer short sentences for abstract content.** A claim about a category, carrying two subordinate clauses and three citations, is unreadable even when correct. One idea per sentence; put the qualifications in the next one.

**Make the synthesis visible in the prose.** Sentences that no single paper could have written are the point: "the most principled fix rests on an assumption most of the pipeline does not make"; "those figures cannot both describe the same protocol". Put such a sentence in the same paragraph as the cited facts it combines, so the fidelity grader reads it as a cross-paper (`composite`) claim. *Why:* this is the insight the user asked for, and it stays grounded because its premises are cited right next to it.

**No word targets.** Length follows the question and the evidence. A focused question with 12 papers may need 2,000 words; a field with 60 may need 10,000. Pages grow as papers are folded in — that is expected. What is not acceptable at any length is padding: a sentence that restates the previous one, or a theme with one paper and nothing to say about it.

### Before and after — a methods page

Catalog style (from `crispr-cas9-off-target-methods-2026`):

> - **CFD**[^doench-2015] is the canonical hypothesis-driven baseline: a per-position-per-mismatch scoring matrix derived from systematic profiling of 27,897 sgRNA mismatch/indel variants at the CD33 locus. Used as the comparator across most subsequent learned scorers…
> - **CRISTA**[^abadi-2017] is the early Random Forest regressor (2017) that unifies on-target and off-target cleavage-efficiency prediction…

Argued style (from `scrna-seq-clustering-and-marker-gene-detection`):

> Testing a gene against clusters built from the same counts is circular, and the corpus documents the problem repeatedly. The 2019 tutorial warns against validating clusters with marker-gene *p* values; SC3's authors advise using post-clustering *p* values only to rank genes; singleCellHaystack's authors give the same caution; and Pullin and McCarthy call for correcting marker *p* values for double dipping…[^luecken][^sc3][^haystack][^pullin]

The second opens with a claim about the field, uses four papers as evidence for one point, and only then goes into each response. The first could be reordered at random without losing anything — the test of a list.

### Before and after — a fact-collection page

A fact page can still list facts; what it must add is what they show together. Instead of a run of per-gene table rows followed by the next section, a theme paragraph should read like:

> Loss-of-function variants that lower LDL cholesterol keep turning up in the same small set of genes, and in each case the human genetics arrived before the drug: PCSK9 loss-of-function carriers have lower LDL-C and less coronary disease, and the same holds for … The pattern suggests …[^a][^b][^c]

The facts are still there, with their numbers and their sources; the paragraph says why they belong in one theme.

---

## Citations and source labels

**Citation form.** This section also serves pages that aren't synthesis pages: the prose sections of idea and concept pages follow the same footnote rules.

- **Marker — one per paper, not per claim.** `[^cas-offinder]`, `[^memgpt]`, reused every place that paper is cited. Obsidian and GitHub both handle a named footnote referenced many times, and it keeps inline markers terse (`…retrieved by relevance.[^memgpt][^amem]`) instead of stacking anchor lists in the prose.
- **Definition — a paper-level link plus a short label**, one per line under `## References`:

  ```
  [^cas-offinder]: [[cgt/bae-2014-cas-offinder-a-fast-and-versatile]] — Bae 2014, Cas-OFFinder
  [^memgpt]:       [[ai/packer-2023-memgpt-towards-llms-as-operating]] — MemGPT
  ```

- Inline claim anchors `[[stem#claim_slug]]` (copied from `researchwiki claims`) are good in prose when the citation should point at one specific claim. A claim anchor inside a footnote *definition* also grades correctly now (the grader strips the fragment, fixed in 718e009 and re-verified 2026-10-04), but the paper-level form reads better and is what the rest of the wiki uses.
- In tables, bare `[[stem]]` only — footnotes don't render inside a markdown table, and `[[stem|alias]]` breaks it.
- Obsidian renders footnotes as clickable superscripts, with a ↩ back-link, only in **Reading view** (Cmd/Ctrl+E). Raw `[^id]` in Source mode is not a formatting bug.

**Frontmatter provenance.** `author_model:` carries the exact model id that wrote the page's current prose, quoted — never `TODO`, a provider name, or a family alias. `researchwiki synthesize` stamps a placeholder because a scaffold cannot know which model will fill it; replace it before the page lands. Update it whenever a substantial rewrite changes the authorship. Only mechanically maintained `meta` / `dashboard` pages are exempt, and `lint`'s `missing_author_model` is the backstop.

**Every section except `Outlook` is strictly grounded.** Each claim-bearing paragraph or bullet carries a wiki citation. There is no model-prior allowance in Background, Findings or Cross-cutting insights, however obvious the fact seems.

**`Outlook` may use three kinds of source, and labels each one.** Outlook is where the page takes a view on what comes next, which needs more than the papers' own future-work paragraphs. Each paragraph or bullet carries exactly one source, because the label applies to the whole unit:

| Source | How to write it | What the gates do |
|---|---|---|
| A paper's own stated limitation or next step | ordinary citation | graded like any other claim |
| A conclusion you draw by combining the papers | `*(inference)*` **plus** citations to the papers it follows from | `check-grounding`: grounded, counted as `inference_claims`. `grade synthesis`: verdict `inference`, not graded. An `*(inference)*` with no citation **fails**. |
| Background knowledge not in any wiki paper | `*(model prior)*` | `check-grounding`: reported as a warning (`model_prior`), not a failure. |

Rules that still hold in Outlook:

- Numbers, benchmark results and attributions to named groups always need a paper citation, and `check-grounding` enforces it here: a `*(model prior)*` unit carrying a quantity stays **ungrounded**. A model prior can say "long-read assays are likely to matter here"; it cannot say "long-read assays reach 99% sensitivity". Cite the paper the figure comes from, or make the sentence qualitative. (Digits inside a name — `Cas9`, `ABE8e` — don't count as quantities.)
- A number inside an `*(inference)*` unit is a premise, not part of the conclusion, so `grade synthesis` still requires it to appear in a cited paper and reports `misattributed` when it doesn't. Keep a figure you can't source out of the inference, or cite the paper it came from. An inference whose citations all point at pages without PDFs — another synthesis page, say — grades as `uncited` rather than `inference`, because there is nothing to check its premises against.
- Don't put a cited fact and an inference in the same paragraph. The label applies to the whole unit, so the fact's wording stops being graded. Split them.
- Make Outlook commit to a view. "More research is needed" is not an outlook. Say which open question matters most, which inference you are most and least confident in, and what kind of study would settle it.
- Outside `Outlook` the labels do nothing — the unit is graded (or flagged) as ordinary prose. Don't use them there.

**Never infer what a PDF does not state** — affiliations, dates, DOIs, numbers. The labels license *reasoning* in Outlook; they never license facts.

---

## Mode A — new page

The steps are staged so that the page scales: you never hold every paper's claims in context at once. For a small topic (under ~15 papers) you can collapse steps 3–4 and read claims directly; the structure of the page stays the same.

Working directory for this page: `.ingest/synthesis/<slug>/` (gitignored, safe to delete afterwards).

### 1. Scope the question

Write the question in one sentence and decide what is out of scope. Check `ls wiki/synthesis/` and `researchwiki search "<topic>" --mode auto --json` for synthesis pages that already cover part of it. If one does:

- If it covers the same question, this is Mode B or C, not a new page.
- If it covers a neighbouring question, link it from `Question` and leave that ground to it.

### 2. Gather candidates

Run, and keep the outputs:

```bash
researchwiki search "<topic>" --mode auto --limit 80 --json   > .ingest/synthesis/<slug>/search.json
researchwiki claims "<topic>" --k 40 --mode hybrid --json      > .ingest/synthesis/<slug>/claims-topic.json
```

Run two or three reformulations of the topic as well — different vocabulary finds different papers. Collect the paper stems (`page_type: paper`) into a candidate list, and note any `commentary` pages: cite the primary paper they discuss, never the commentary as evidence.

If you have scaffolded the page (step 6), `researchwiki check-coverage <page> --top-n <N>` is a third recall channel; set `N` to roughly the candidate count, not the default 20.

### 3. Build an evidence card for every candidate

For each stem:

```bash
researchwiki claims --by-stem <stem> --json > .ingest/synthesis/<slug>/claims/<stem>.json
```

Then write a card to `.ingest/synthesis/<slug>/cards.md`, about 120 words each:

```markdown
### <stem>
- In scope: yes | no (reason)
- Theme guess: <one phrase>
- Key results: <1–2 results with their conditions>, [[stem#slug]], [[stem#slug]]
- Stated limitation: <the paper's own>, [[stem#slug]]
- Builds on / contrasts: <stems this paper explicitly cites or argues against, from its page>
```

Read the paper's wiki page (`Summary`, `Key Contributions`, `Limitations`, `Related Papers`) for the card, not just the claim dump. `researchwiki claim-graph --stem <stem> --json` lists judged relations to other papers' claims (`contradicts`, `refines`, …) — useful for "Builds on / contrasts" and for Tensions.

**With many candidates, parallelize.** If your environment has subagents, give each a batch of about 10 stems and the card template, and have them write their cards to separate files you then concatenate. Let each subagent run on the default model — naming one risks an "invalid model name" failure and a silent relaunch. Without subagents, work through the list in order and append each card as you finish it — the card file is the memory, not the conversation.

Cards that say `In scope: no` are kept: they are your exclusion record.

### 4. Choose the organizing scheme and build the coverage ledger

Read the cards only — not the full claim dumps. Choose the scheme (see the table in *Why one structure for every topic*) that best explains why the in-scope papers differ. Then write `.ingest/synthesis/<slug>/ledger.md`:

- the scheme and its 2–4 axes or categories;
- the themes (future `Findings` H3s) and which stems go in each;
- every candidate stem marked `cite` or `exclude: <reason>`.

A theme with one paper is usually a sign the scheme is wrong, or that the paper belongs inside another theme.

### 5. Confirm the outline (interactive sessions)

If a person is in the loop, show them the scheme, the themes with their papers, and the exclusions, and wait for agreement before drafting. Changing the scheme after drafting means rewriting most of the page. In a non-interactive run, record the outline in the ledger and continue.

### 6. Scaffold and draft

```bash
researchwiki synthesize --title "<title>" --topic-seed "<4–8 word query>" --papers <stem> <stem> ...
```

Replace the scaffold body with the fixed structure if the scaffold predates it. Then draft in this order, loading the full claim dump only for the theme you are writing:

1. `Background`
2. `Organizing framework`
3. `Findings`, one H3 at a time
4. `Cross-cutting insights`
5. `Tensions / open questions` — seed it with `researchwiki claim-graph --tensions` and with contradictions you noted on the cards
6. `Outlook`
7. `Short answer`, then `Question` — last, because you only know the argument once it is written
8. YAML: `title`, `type: synthesis`, `category: [<dominant content category of the cited papers>]`, `generated_at: YYYY-MM-DD`, `author_model: "<exact model id>"`, `topic_seed`, `tags`, and a quoted `hook:` (≤1000 characters, result-first: the page's main finding, not its question)

When a load-bearing claim rests on a figure the caption does not settle, `researchwiki figures <stem>` lists captions for free and `--figure N` renders one page.

### 7. Verify and land — see [Verify, then land](#verify-then-land).

### 8. Ask for missing papers

After the gates pass, list up to 3–5 specific papers or paper types whose absence weakened the page, each tied to where it would land ("would let the second Outlook inference cite its premise directly"). Ask the user to drop PDFs into `inbox/`. Skip this when coverage is already comprehensive.

---

## Mode B — update with new papers

Use when one or more papers were ingested after the page was written — typically because an impact-review receipt (`.ingest/impact-review/<stem>.yaml`) names this page, or the user asks.

Don't rewrite the page. Rewriting churns prose that already passed both gates.

1. Build cards (Mode A step 3) for the new papers only. If `.ingest/synthesis/<slug>/` from the original run is gone, the page's `Organizing framework` table is the record of where existing papers sit.
2. Place each new paper:
   - **Fits an existing theme** → revise that theme's argument where the new evidence changes it (don't just append a sentence), and add its row to the framework table.
   - **Starts a new theme** → add the H3, add its rows, and revisit `Cross-cutting insights`.
   - **Breaks the scheme** (it needs an axis the page doesn't have) → stop and tell the user; this is a restructure, done as Mode C.
3. Recheck `Outlook` every time: a new paper often confirms or refutes one of its inferences. A confirmed inference becomes an ordinary cited claim (move it to Findings if it now belongs there); a refuted one is removed and the refutation noted in Tensions.
4. Update `Short answer` if a takeaway changed, `generated_at:`, and `author_model:` if a different model wrote the revision.
5. Verify and land. Record the impact-review decision (`incorporated`, with a reason) for each receipt you acted on.

For a one-sentence addition, `researchwiki evolve <category/stem>` can propose the patch instead.

---

## Mode C — upgrade a legacy page

A legacy page has the old shape — usually `Evidence from the wiki` in the middle and `What would update this page` at the end. `researchwiki lint --json` will list it once the structure check exists.

Run Mode A with these changes:

- Seed the candidate list with every paper the page already cites (its footnote definitions and inline links), then add search results.
- Keep prose that already argues and fits a theme; move it under the new headings rather than rewriting it.
- Rewrite catalog-style bullets into theme paragraphs.
- Fold each `What would update this page` item into `Outlook`, as the open question it implies, and drop the old section. **This is where migrations fail the gate:** the old heading was exempt from `check-grounding` by name, and `Outlook` is not. An uncited wish list ("a head-to-head benchmark of X and Y") was legal there and is ungrounded here, so each item now needs the paper whose stated limitation leaves it open, an `*(inference)*` plus its premises, or a `*(model prior)*`.
- Keep the slug. Inbound `[[synthesis/<slug>]]` links stay valid.

---

## Verify, then land

**Gates — both must exit 0:**

```bash
researchwiki db rebuild          # the page was written outside the package
researchwiki check-grounding wiki/synthesis/<slug>.md
researchwiki grade synthesis wiki/synthesis/<slug>.md --weak
researchwiki check-coverage wiki/synthesis/<slug>.md --top-n <candidate count>
```

- `check-grounding` must report 0 ungrounded. Its summary line also reports the labelled inferences and model priors — read them: if Outlook is mostly model priors, the page is leaning on you rather than the corpus.
- `grade synthesis` must report 0 misattributed and **more than 0 graded**. Read the graded count every time: `0 graded` with exit 0 means the fidelity gate silently did nothing — the citations aren't in a form it can resolve, or no cited stem has a PDF — which is worse than a failure, because the page looks checked. Look at the `weak` units it lists too: each is a sentence the cited paper barely supports.
- `check-coverage` is advisory, but every hit must be dealt with: cited, or excluded with a reason. An unreviewed hit is a failure of the review; a reviewed exclusion is fine.

**Land the page:**

1. `researchwiki reindex`.
2. Add a bullet under `## synthesis` in `wiki/index.md`: `- [[synthesis/<slug>]] — **<Title>** (<year>): <hook>`.
3. Append to `wiki/log.md`:

   ```markdown
   ## [YYYY-MM-DD] synthesis | <Title>
   Category: <category>. <One sentence on the scheme and scope.> Grounding N/N units (I labelled inference, M model prior); fidelity S supported, C composite, W weak, 0 misattributed.
   Coverage review: <each check-coverage hit not cited, with its reason>.
   ```

---

## When a page gets too big

Suggest a split to the user — never split on your own — when one theme passes about 12 papers or about 2,500 words, or the whole page passes about 12,000 words. These are starting heuristics, not rules.

A split moves that theme into a child synthesis page with the same fixed structure. The parent keeps a cited summary paragraph in place of the theme and links the child from `Question` and from the theme's H3. Each page then passes both gates on its own.
