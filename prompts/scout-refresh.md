# Refreshing wiki/synthesis/suggested-additions.md from citation scouting

Triggers:
- `lint --json` reports under `stale_by_audit_count` (`wiki_papers_at_audit: N` drifted > `max(5, 20% of N)`).
- A listed paper gets ingested (mark ✅ INGESTED + `[[wikilink]]`).
- ≥3 ingests since last refresh.

## Workflow

Run `researchwiki scout --json`, diff against the existing page, insert into the right tier:

- `recommended_additions` `multi_paper_count ≥ 3` → Priority 3.
- `shared_citation_anchors ≥ 2` → Priority 1.

**Preserve** ✅ / ~~strikethrough~~ / stars / inline notes — never regenerate. Update `wiki_papers_at_audit: N`.

## Structural signals at large N

- `anchor_groups.multi_category` → cross-category anchors as individual Priority 1 rows (rare, high-leverage).
- `anchor_groups.single_category` → batch cluster-internal anchors into grouped paragraph notes.
- Use `count_normalized` (= `multi_paper_count / total_papers`), not raw counts, to compare across scouting runs.
- Before refreshing Priority 2, check `p2_entries_with_anchor_hits` — LLM decides action (promote, re-annotate, leave).
- When `edge_summary.total > ~80`, drop full edge dump; render `top_hubs_incoming` + `by_category_flow` as a compact summary.

## Tool vs. LLM boundary

`anchor_groups`, `count_normalized`, `p2_entries_with_anchor_hits` give structural data, not verdicts. ★★★/★★/★ stars and inline notes remain LLM editorial.
