# Synthesis-page citation format (footnotes)

Applies to prose-heavy pages that cite the claims DB — **synthesis**, and the
prose sections of **idea** / **concept** pages. Governs how footnote citations
are written so they read cleanly, click through in Obsidian, and pass
`grade synthesis`.

## Frontmatter provenance

Synthesis pages must carry `author_model:` in YAML. Use the exact model
identifier that wrote the current page (for example `gpt-5.6-luna`), quoted as
a string; never leave `TODO`, a provider name, or a generic label.
`researchwiki synthesize` emits a placeholder because the scaffold cannot know
which model will fill the prose; replace it before committing. Idea pages are
the deliberate exception because they are living documents and do not carry
`author_model:`.

## The format: one footnote per source paper

- **Marker** — id the footnote by the paper, not the claim: `[^memgpt]`,
  `[^amem]`, `[^yang]`. Reuse the same marker every place that paper is cited;
  Obsidian and GitHub both handle a named footnote referenced many times.
- **Definition** — a single **paper-level** `[[category/stem]]` link plus a
  short label, one per line at the bottom:

  ```
  [^memgpt]: [[ai/packer-2023-memgpt-towards-llms-as-operating]] — MemGPT
  [^amem]:   [[ai/xu-2025-a-mem-agentic-memory-for-llm]] — A-MEM
  ```

- Keeps inline markers terse (`…retrieved by relevance.[^memgpt][^amem]`)
  instead of stacking claim-anchor lists in the prose.

## Two things that trip people up

1. **Clickability is a view-mode thing.** Footnotes render as clickable
   superscripts (with a ↩ back-link) only in Obsidian **Reading view**
   (toggle Cmd/Ctrl+E). **Source mode** shows the raw `[^id]` text — that is
   not a formatting bug.
2. **`grade synthesis` only resolves the paper-level form.** A definition
   holding a **comma-separated list of `[[stem#slug]]` claim anchors** parses
   as *uncited* — the grader skips every such unit and reports `0 graded`
   (check-grounding still passes, so the fidelity gate silently no-ops). Use
   one `[[category/stem]]` link per footnote instead. Claim-level `[[stem#slug]]`
   anchors are for **inline** citations (non-footnote), not footnote defs.

## Verify (both must exit 0)

```
researchwiki check-grounding wiki/synthesis/<slug>.md   # structural
researchwiki grade synthesis wiki/synthesis/<slug>.md   # fidelity — confirm N graded, not 0
```

Then `researchwiki check-coverage wiki/synthesis/<slug>.md` (advisory recall).
Confirm `grade synthesis` reports a non-zero `graded` count — `0 graded` means
the citations aren't in a form the grader reads (see gotcha #2).
