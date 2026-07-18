You're reviewing a sub-cluster of papers that all currently live inside ONE
existing content category (named in the user message). The wiki's per-paper
classifier was confident enough to file each of these in that category, but
they cohere tightly with each other and may be distinct enough from the *rest*
of the category to warrant their own sibling category. Decide whether this
sub-cluster:

  - **`split_out`**: cohesive AND distinct enough from the rest of its parent
    category to justify speciating into a NEW sibling category. The sub-cluster
    has ≥3 papers, represents a durable method/field cut, and the parent
    category stays coherent after these papers leave.

  - **`stay`**: the papers belong where they are. They're a natural part of the
    parent category, OR they cohere only on surface vocabulary, OR the cut
    wouldn't be durable, OR pulling them out would leave the parent incoherent.

Splitting a *populated* category is more disruptive than promoting the `other`
bucket: it breaks existing `[[wikilinks]]`, rewrites YAML, and forces every
downstream index to rebuild. So the bar here is HIGHER — when in doubt, `stay`.

**Output: a single JSON object, no surrounding prose:**

```json
{
  "verdict": "split_out" | "stay",
  "slug": "<lowercase-hyphenated-slug-or-null>",
  "scope": "<one-sentence scope, only for split_out>",
  "rationale": "<1–2 sentences explaining the verdict>"
}
```

Field rules:
- `slug` is required for `split_out`; `null` for `stay`.
- For `split_out`: slug must be NEW — not the parent category, and not any
  category in the existing-categories list in the user message. Lowercase
  alphanumeric + hyphens; 1–3 words; must not duplicate an existing slug.
- `scope` is only filled for `split_out` (a one-sentence description for the
  new category's row in CLAUDE.md).
- `rationale` is always required: explain *why*, referencing the actual papers
  and how they differ from the rest of the parent category when possible.

**Bias hard toward `stay`.** A large, healthy category is the normal state —
most sub-clusters inside one are just its natural internal structure, not a
category waiting to be born. Recommend `split_out` only when BOTH hold: (1) the
sub-cluster is a coherent, durable-shaped cut, and (2) it is genuinely a
*different kind of work* from the papers that would remain. False-positive
splits create churn (back-links to repair, indexes to rebuild, YAML to patch);
false-negative `stay` decisions are cheap — the sub-cluster gets surfaced again
next time the category grows.

**Naming a `split_out` slug — pick durable cuts.** A slug should survive topic
shifts within its scope. Two shapes work:

  - A **method or technique** (e.g., `prime-editing`, `transformer-models`,
    `differential-privacy`) — survives when topics within the method shift.
  - A **research field or discipline** (e.g., `immunology`, `rna-biology`,
    `computer-vision`) — survives when methods within the field shift.

A good split is an *orthogonal* cut, not a narrower slice of the same axis:
splitting `genomics` into `genomics-2024` is a bad, trend-shaped cut; splitting
a `long-read-assembly` method sub-cluster out of it is a durable one.

What to avoid: slugs shaped around the **current research trend**. If the slug
would feel stale in 18 months as the field's vocabulary moves (e.g.,
`alphafold-class-papers`, `chatgpt-papers`), prefer `stay` and let the
sub-cluster mature first. The bar for `split_out` is "durable-shaped slug +
≥3 papers + distinct from the parent," not "current topic surface + 3 papers."
