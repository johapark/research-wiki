"""Canonical Obsidian Dataview dashboard scaffold used by ``researchwiki init``."""

VIEWS_MD_TEMPLATE = """\
---
title: "Wiki Dashboard — Recent Additions"
type: dashboard
tags: [dashboard, dataview]
---

# Wiki Dashboard

Live views of recent additions across the wiki. Rendered by the [Dataview](https://blacksmithgu.github.io/obsidian-dataview/) community plugin — **install and enable Dataview in Obsidian** for the tables below to render. On GitHub (or without the plugin) they show as inert code blocks.

Recent tables use the frontmatter stamps (`ingested_at` / `generated_at`); pages without the relevant stamp are omitted.

Concept membership comes directly from its canonical `referenced_papers` spoke registry. Synthesis pages intentionally have no Members column because body citations, not duplicate frontmatter, are their source registry.

## Recent papers (top 15)

```dataview
TABLE WITHOUT ID
  link(file.link, file.name) AS "Stem",
  join(category, ", ") AS "Category",
  venue AS "Journal",
  year AS "Year",
  dateformat(ingested_at, "yyyy-MM-dd") AS "Added"
FROM ""
WHERE type = "paper" AND ingested_at
SORT ingested_at DESC
LIMIT 15
```

## Recent ideas (top 10)

```dataview
TABLE WITHOUT ID
  file.link AS "Idea",
  verdict AS "Verdict",
  status AS "Status",
  dateformat(generated_at, "yyyy-MM-dd") AS "Filed"
FROM ""
WHERE type = "idea" AND generated_at
SORT generated_at DESC
LIMIT 10
```

## Recent synthesis pages (top 10)

```dataview
TABLE WITHOUT ID
  file.link AS "Synthesis",
  topic_seed AS "Topic seed",
  dateformat(generated_at, "yyyy-MM-dd") AS "Updated"
FROM ""
WHERE type = "synthesis" AND generated_at
SORT generated_at DESC
LIMIT 10
```

## Recent concept hubs (top 10)

```dataview
TABLE WITHOUT ID
  file.link AS "Concept",
  length(referenced_papers) AS "Members",
  concept_span AS "Categories",
  concept_thesis AS "Thesis",
  dateformat(generated_at, "yyyy-MM-dd") AS "Filed"
FROM ""
WHERE type = "concept" AND generated_at
SORT generated_at DESC
LIMIT 10
```
"""
