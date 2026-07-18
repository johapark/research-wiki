You're reviewing a cluster of papers that the wiki's per-paper classifier
landed in `other` (the structured "uncategorized backlog" bucket). Decide
whether this cluster:

  - **`new_category`**: cohesive enough and distinct enough from all existing
    categories to justify its own. The cluster has ≥3 papers and the LLM
    classifier was wrong to abstain — these papers form a real cluster the
    user should promote.

  - **`reassign`**: belongs in an existing category. The classifier abstained
    incorrectly; the cluster fits one of the categories already in the user's
    schema.

  - **`stay`**: genuinely cross-cutting or topical noise. Papers don't form
    a coherent cluster, OR they form one but it's too narrow to warrant a
    category. They should remain in `other`.

**Output: a single JSON object, no surrounding prose:**

```json
{
  "verdict": "new_category" | "reassign" | "stay",
  "slug": "<lowercase-hyphenated-slug-or-null>",
  "scope": "<one-sentence scope, only for new_category>",
  "rationale": "<1–2 sentences explaining the verdict>"
}
```

Field rules:
- `slug` is required for `new_category` and `reassign`; `null` for `stay`.
- For `new_category`: slug must be new (not in the existing-categories list);
  must be lowercase alphanumeric + hyphens; should be 1–3 words; must not
  duplicate an existing slug.
- For `reassign`: slug must be one of the existing categories listed in
  the user message.
- `scope` is only filled for `new_category` (a one-sentence description for
  the new category's row in CLAUDE.md).
- `rationale` is always required: explain *why* this cluster fits the chosen
  verdict, referencing the actual papers when possible.

**Bias toward `stay`** when the cluster is small (2 papers), borderline, or
when the rationale would just be "they share some keywords." False-positive
splits create churn (back-links to repair, indexes to rebuild, YAML to
patch); false-negative `stay` decisions are cheap (the cluster gets surfaced
again next time `other` grows).

**Naming a `new_category` slug — pick durable cuts.** A slug should survive
topic shifts within its scope. Two shapes work:

  - A **method or technique** (e.g., `prime-editing`, `transformer-models`,
    `differential-privacy`) — survives when topics within the method shift.
  - A **research field or discipline** (e.g., `immunology`, `rna-biology`,
    `computer-vision`) — survives when methods within the field shift.

Both are good; the user's existing taxonomy (listed in the user message)
was designed under this lens — some slugs may be field-shaped, others
method-shaped. Apply whatever shape fits the cluster.

What to avoid: slugs shaped around the **current research trend**. If the
slug would feel stale in 18 months as the field's vocabulary moves
(e.g., `alphafold-class-papers`, `chatgpt-papers`), prefer `stay` and
let the cluster mature first. The bar for `new_category` is "durable-shaped
slug + ≥3 papers + distinct from existing taxonomy," not "current topic
surface + 3 papers."

**For `reassign`**, the cluster needs to fit *all* listed papers reasonably
well into the proposed existing category. If only 2 of 4 fit and the rest
are cross-cutting, prefer `stay` for the whole cluster — partial reassignment
is the user's call, not the tool's.
