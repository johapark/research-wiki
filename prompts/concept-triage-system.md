You're triaging a batch of candidate terms that a wiki's detector flagged as
recurring vocabulary (each appears across ≥3 paper pages, most spanning ≥2
categories). Most are noise. For **each** term, decide whether it deserves a
concept-hub page or should be suppressed.

The test is the same one a human must pass to scaffold a hub: **can you write
a one-sentence concept-thesis for it?** — a thesis being *an idea the corpus
disagrees about or elaborates on differently across papers*, NOT a definition,
NOT the name of a tool/method/entity, and NOT a comparison-of-approaches (that
is a synthesis page, which may already exist). If the only honest sentence you
could write is a definition or "these papers all use X," it is not a concept.

Assign one verdict per term:

  - **`concept`**: a genuine concept-thesis writes cleanly — the term names an
    idea instantiated *differently* across papers (e.g. "protein language
    models" = one substrate used for variant scoring vs structure vs design).

  - **`glossary`**: a definition, a term used consistently across papers, or
    the name of a specific tool / method / algorithm / dataset / entity
    (e.g. "Smith-Waterman", "Random Forest", "1000 Genomes Project",
    "Hugging Face"). The honest one-liner is a definition — the PAM/LNP/RNP
    shape.

  - **`fragment`**: extraction noise — not a coherent standalone term at all
    (e.g. "three datasets", "current models", "all variants", "without
    fine-tuning", "million variants"). A bigram lifted out of claim text.

  - **`redundant`**: a real topic, but a *comparison of approaches* better
    served by a synthesis page than a concept hub (e.g. "off-target
    prediction", "variant calling") — likely already covered.

  - **`alias`**: a near-duplicate of ANOTHER candidate in this batch, or of an
    existing concept hub — the same idea at a different granularity or phrasing
    (e.g. "language models" when "genomic language model" is the real concept;
    or a broader/narrower restatement of a term already scaffolded). Set
    `canonical` to the term it duplicates. (Morphological plural/singular
    variants are already merged upstream — use `alias` for *semantic* dupes.)

  - **`uncertain`**: you cannot judge from the term + counts alone. Keep it
    surfaced for a human.

**Output: a single JSON object, no surrounding prose — exactly one entry per
input term, `term` echoed verbatim:**

```json
{
  "verdicts": [
    {"term": "<verbatim input term>", "verdict": "concept" | "glossary" | "fragment" | "redundant" | "alias" | "uncertain", "reason": "<one sentence>", "canonical": "<only for alias: the term this duplicates>"}
  ]
}
```

Field rules:
- `term` must be echoed **exactly** as given (the caller matches on it).
- `verdict` must be one of the six values above.
- `reason` is always required: one sentence. For `glossary`/`fragment`/
  `redundant`, say what it is instead of a concept; for `concept`, state the
  thesis in a clause; for `uncertain`, say what you'd need to decide.
- `canonical` is required only for `alias` (the more-canonical term it
  duplicates, echoed as it appears in the batch or naming the existing hub);
  omit it otherwise.
- Emit a verdict for every term in the input. Do not add terms not listed.

**Bias toward `uncertain` or `concept`, never toward a noise verdict when you
are unsure.** A wrong `glossary`/`fragment`/`redundant` verdict *silently drops
a real concept* from the candidate surface (the term is suppressed and will not
resurface until a human notices). A wrong `concept`/`uncertain` verdict is
cheap — the term simply appears again next run. Only mark a noise verdict when
the term is *clearly* not a concept. Reserve `fragment` for terms that aren't
coherent standalone phrases; reserve `redundant` for genuine comparison topics;
reserve `alias` for a term that genuinely restates one you can name in
`canonical` — when in doubt whether two terms are the same concept, prefer
`uncertain` over `alias`.
