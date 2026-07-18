# Cross-link discovery — for manual page writes

Trigger: after writing or substantially editing a **multi-topic page** (synthesis, combined whitepaper, broad reference doc — anything covering ≥3 distinct named tools/methods/concepts), grep `wiki/` for each named entity and review for cross-link candidates per the *Cross-links must be source-supported* corollary in `CLAUDE.md`.

```bash
grep -rln -E "\b(Tool1|Tool2|Concept1|...)\b" wiki/ --include="*.md" \
  | grep -v "log\.md\|index\.md\|<this-page-stem>"
```

## Why needed

Neither `lint --fix` nor `audit` catches cross-links that *should* exist but don't on either side — `missing_backlinks` only flags asymmetric existing links. The agent ingest path's `propose_crosslinks` covers paper pages, but the manual workflow for synthesis + reference docs has no analogue, and the failure mode is silent.

## When most useful

When a page expands in scope: one-article reference doc folded into a multi-article compilation, two-paper synthesis stretched to seven, paper page where you patched in a previously-omitted method/tool.

## Bounded scope

Skip for single-topic pages, minor edits (typos, date updates), or pages whose named entities are too generic to grep usefully (synthesis pages titled around abstract concepts like "uncertainty" or "scaling"). Trigger is *named entities the page now covers* — if no distinctive search-grade tokens, skip.
