# Export a wiki page as a shareable artifact

**Trigger:** user asks to "share", "export", "make sharable" a synthesis/idea page, or to send a wiki page to someone outside this repo. Output goes to `share/<slug>.md` (gitignored).

The wiki's `[[category/stem]]` wikilinks, framework-specific YAML fields (`type`, `category`, `referenced_papers`, `topic_seed`, `author_model`), and self-referential phrasings ("in the wiki", "this synthesis page") are private context. A shareable strips all of it and replaces wikilinks with full academic citations — the result should render correctly in any standard markdown viewer (GitHub, Obsidian, pandoc) with no internal dependencies.

## Procedure

1. **Read the source page**: `wiki/synthesis/<slug>.md` or `wiki/ideas/<slug>.md`.

2. **Pull citation metadata** for every wikilink in the body and in `referenced_papers:`. For each `[[<cat>/<stem>]]`:
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

7. **Save** to `share/<slug>.md`.

8. **Verify** the output:
   ```bash
   # No bare wikilinks remain
   grep -nE "\[\[" share/<slug>.md
   # No framework leakage
   grep -niE "\bwiki\b|claim_id|referenced_papers|topic_seed|category_suggestion|model_prior" share/<slug>.md
   # Footnote refs/defs reconcile
   python3 -c "
   import re
   t = open('share/<slug>.md').read()
   refs = set(re.findall(r'\[\^([a-z0-9-]+)\](?!:)', t))
   defs = set(re.findall(r'^\[\^([a-z0-9-]+)\]:', t, re.MULTILINE))
   print(f'orphan refs (no def): {refs - defs or \"(none)\"}')
   print(f'unused defs: {defs - refs or \"(none)\"}')
   "
   ```
   First two greps should return nothing. Footnote reconciliation should show no orphan refs. (Unused defs flagged by the simple Python regex may be false positives when an inline ref is followed by a sentence colon — `Luo et al. (2024)[^luo-2024]:` — markdown parses these correctly as refs, only column-0 lines are real defs.)

9. **Do not commit** `share/`. The directory is gitignored (`.gitignore` carries `share/`) and the artifact is per-task, not durable repo state.

## Notes

- A shareable is a one-way export. Future edits to the source wiki page do not propagate; if the wiki page changes materially, regenerate the share file rather than hand-patching it.
- If the source page cites a paper that lacks a DOI (e.g., conference paper, unpublished report), use the arXiv/bioRxiv/medRxiv URL or a `(Author, Year, in preparation)` placeholder — but flag the gap to the user.
- Avoid disclosing repo or framework identity in the shareable. The reader should see a self-contained academic survey, not a personal-knowledge-base export.
