---
name: synthesis-page
description: Write, update, or upgrade a synthesis page in this research wiki (wiki/synthesis/*.md) — a review-article-quality survey of a subject, grounded in the ingested papers, in the wiki's fixed section structure (Question, Short answer, Background, Organizing framework, Findings, Cross-cutting insights, Tensions / open questions, Outlook, References). Use this whenever the user asks to write, draft, file, or save a synthesis, survey, review, literature review, state-of-the-field, landscape, comparison, or evidence-summary page on a topic; to fold newly ingested papers into an existing synthesis page (including after an impact-review receipt names one); or to upgrade or restructure an older synthesis page. Use it even when the user just says "write this up as a page" after a cross-paper answer. Do not use it to answer a question without filing a page, for idea pages (wiki/ideas/), for proposal records, or for exporting a page to share outside the wiki.
---

# Synthesis page

A synthesis page is this wiki's review article on one subject: organized around ideas rather than papers, readable by a newcomer, and in the same structure every time.

**Read [`prompts/synthesis-page-author.md`](../../../prompts/synthesis-page-author.md) before writing anything** (relative to the repo root: `prompts/synthesis-page-author.md`). It holds the fixed structure, the writing rules that keep a page from turning into a list of papers, the citation and source-label rules, and the step-by-step procedure for all three modes. That file is the canonical procedure — every agent working in this repo reads the same copy, so nothing here restates it.

## Pick the mode

| Situation | Mode |
|---|---|
| No synthesis page covers this question yet | **A — new page** |
| A page exists and papers were ingested since (often named by an impact-review receipt) | **B — update** |
| A page exists in the old shape (`Evidence from the wiki`, `What would update this page`) | **C — upgrade** |

Check `ls wiki/synthesis/` and `researchwiki search "<topic>" --mode auto` first; writing a second page on a question the wiki already covers is the most expensive mistake available.

## What never changes

- The four rules in CLAUDE.md still hold: every claim grounds in a PDF the wiki has, and nothing comes from web search.
- Every section except `Outlook` is strictly cited. `Outlook` may add the author's own inferences (`*(inference)*` plus the papers they follow from) and background knowledge (`*(model prior)*`), each labelled.
- The page is done only when `check-grounding` and `grade synthesis` both exit 0 — with a non-zero graded count — and every `check-coverage` hit is cited or excluded with a reason.
