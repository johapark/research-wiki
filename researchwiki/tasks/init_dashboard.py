"""Canonical Obsidian Dataview dashboard scaffold used by ``researchwiki init``."""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from pathlib import Path

from ..fsatomic import write_text_atomic
from ..paths import ingest_dir, wiki_dir, wiki_root

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


def refresh_dashboard() -> int:
    """Rewrite ``wiki/views.md` from the template, backing up an existing copy.

    The ordinary scaffold never overwrites this hand-editable page. This
    explicit upgrade action lets an older wiki adopt template changes while
    retaining its customized dashboard under ``.ingest/``.
    """
    views = wiki_dir() / "views.md"
    if views.is_file():
        current = views.read_text(encoding="utf-8")
        if current == VIEWS_MD_TEMPLATE:
            print("wiki/views.md already matches the current template — nothing to do.")
            return 0
        backup_dir = ingest_dir()
        backup_dir.mkdir(parents=True, exist_ok=True)
        # Reserve the name atomically. A timestamp alone can collide when the
        # command runs twice within one second or the wall clock repeats.
        fd, raw_backup = tempfile.mkstemp(
            prefix=f"views-{dt.datetime.now():%Y%m%dT%H%M%S}-",
            suffix=".md.bak",
            dir=backup_dir,
        )
        os.close(fd)
        backup = Path(raw_backup)
        try:
            write_text_atomic(backup, current)
        except Exception:
            backup.unlink(missing_ok=True)
            raise
        print(f"Backed up your dashboard to {backup.relative_to(wiki_root())}")
    else:
        wiki_dir().mkdir(parents=True, exist_ok=True)
    write_text_atomic(views, VIEWS_MD_TEMPLATE)
    print("Wrote wiki/views.md from the current template.")
    print("It renders only inside Obsidian with the Dataview plugin enabled; on "
          "GitHub the blocks show as inert code.")
    return 0
