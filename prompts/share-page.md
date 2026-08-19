# Share a wiki page as a standalone document

**Trigger:** user asks to "share", "make shareable", or "send" a synthesis/idea page to someone outside this repo. Output goes to `output/share/<slug>.md` (gitignored).

**Not this file:** `researchwiki export` emits the *corpus* — as a bibliography (BibTeX/RIS/CSL-JSON) for a reference manager, or as an OKF bundle — see [`prompts/export-bibliography.md`](./export-bibliography.md). The word "export" belongs to that command; this one produces a document for a human reader, which is why it is "share".

The wiki's `[[category/stem]]` wikilinks, framework-specific YAML fields (`type`, `category`, `referenced_papers`, `topic_seed`, `author_model`), and self-referential phrasings ("in the wiki", "this synthesis page") are private context. A shareable strips all of it and replaces wikilinks with full academic citations — the result should render correctly in any standard markdown viewer (GitHub, Obsidian, pandoc) with no internal dependencies.

## Procedure

1. **Read the source page**: `wiki/synthesis/<slug>.md` or `wiki/ideas/<slug>.md`.

2. **Pull citation metadata** for every wikilink in the body — inline `[[wikilink]]`s and the `## References` footnote definitions, which are the whole citation set on a synthesis or idea page (neither carries `referenced_papers:`; a concept page does). For each `[[<cat>/<stem>]]`:
   ```bash
   awk '/^---$/{n++; if(n==2)exit} n==1' "wiki/<cat>/<stem>.md" | grep -E "^(title|authors|year|doi|venue):"
   ```
   Capture: first-author surname, title (in quotes), year, venue (italicized), DOI.

3. **Rewrite YAML frontmatter** to the neutral set only:
   ```yaml
   ---
   title: "..."
   date: YYYY-MM-DD
   tags: [...]
   ---
   ```
   Drop `type`, `category`, `referenced_papers`, `topic_seed`, `author_model`, `companion_synthesis`, `generated_at`, `status` (idea pages), and any other framework-internal fields.

4. **Replace inline `[[wikilinks]]`** in prose with neutral citations:
   - Wikilinks in **prose** (subject-position, parenthetical, footnote-reference) → drop the wikilink, keep or insert `Author Year[^id]` style. The `[^id]` retains the footnote pointer for the reference list.
   - Wikilinks in **table cells** → replace with bare "Author Year" plain text (no footnote — footnotes don't render inside markdown tables in many viewers). Citation is still preserved via the table's prose context or row content.
   - Wikilinks in **YAML lists** → strip (the new YAML drops `referenced_papers:` entirely).

5. **Rewrite footnote definitions** from `[^id]: [[<cat>/<stem>]]` to full academic citations:
   ```markdown
   [^id]: Author A. et al. "Title of paper." *Venue* (YEAR). DOI: [10.x/y](https://doi.org/10.x/y).
   ```
   - Use `et al.` after the first author when there are ≥3 authors.
   - Venue in italics. Year in parentheses.
   - DOI as a markdown link to `https://doi.org/<doi>`.

6. **Strip wiki-self-references** in prose. Search for and rephrase:
   - `the wiki`, `in the wiki`, `in this wiki`, `the wiki's papers` → `the published literature`, `to date`, or drop entirely
   - `this synthesis page`, `this page`, `the synthesis` → `this analysis`, `this survey`, or just `this`
   - `What would update this page` (the gate-skip-section heading) → `What would update this analysis` (or similar)
   - `claim_id:NNN` references → drop entirely (these are repo-internal row keys)

7. **Save** to `output/share/<slug>.md`.

8. **Verify** the output:
   ```bash
   # No bare wikilinks remain
   grep -nE "\[\[" output/share/<slug>.md
   # No framework leakage
   grep -niE "\bwiki\b|claim_id|referenced_papers|topic_seed|category_suggestion|model_prior" output/share/<slug>.md
   # Footnote refs/defs reconcile
   python3 -c "
   import re
   t = open('output/share/<slug>.md').read()
   refs = set(re.findall(r'\[\^([a-z0-9-]+)\](?!:)', t))
   defs = set(re.findall(r'^\[\^([a-z0-9-]+)\]:', t, re.MULTILINE))
   print(f'orphan refs (no def): {refs - defs or \"(none)\"}')
   print(f'unused defs: {defs - refs or \"(none)\"}')
   "
   ```
   First two greps should return nothing. Footnote reconciliation should show no orphan refs. (Unused defs flagged by the simple Python regex may be false positives when an inline ref is followed by a sentence colon — `Luo et al. (2024)[^luo-2024]:` — markdown parses these correctly as refs, only column-0 lines are real defs.)

9. **Do not commit** the artifact. It lives under `output/`, the umbrella for everything this repo emits for an outside reader (`output/graph.html`, `output/okf/`, `output/share/`), which is gitignored wholesale — the artifact is per-task, not durable repo state.

## Notes

- A shareable is one-way. Future edits to the source wiki page do not propagate; if the wiki page changes materially, regenerate the share file rather than hand-patching it.
- If the source page cites a paper that lacks a DOI (e.g., conference paper, unpublished report), use the arXiv/bioRxiv/medRxiv URL or a `(Author, Year, in preparation)` placeholder — but flag the gap to the user.
- Avoid disclosing repo or framework identity in the shareable. The reader should see a self-contained academic survey, not a dump from a personal knowledge base.
