"""Checks on what the semantic index will actually see for each page.

Separate from `db_checks` (sqlite) and `yaml_checks` (frontmatter shape) because
it reads a third thing: the *assembled embedding text*, which is a function of
page type, section names, and body prose together. It was written into
`db_checks` first, which was wrong on both counts — that module documents itself
as "Sqlite-backed checks" and this one touches no database.
"""

from __future__ import annotations

from pathlib import Path

from ...wiki import Page
from .walk import page_key


def find_thin_index_text(
    pages: list[Path],
    pages_fm: dict[Path, dict],
    pages_body: dict[Path, str],
) -> list[dict]:
    """Pages whose semantic-index text is too thin to retrieve on.

    Mirrors the warning `reindex` prints and shares its one implementation
    (`index.pages_semantic.thin_index_reason`), so the two cannot disagree. It is
    surfaced here as well because `reindex`'s output scrolls away while
    `lint --json` is what gets scanned, and the failure it guards is silent by
    nature: on 2026-08-06 a section-name mismatch reduced every synthesis, idea
    and concept page to its title and nothing noticed.

    Advisory. A thin page is not malformed; it is unreachable by semantic
    retrieval, which is a quieter problem than a broken link.

    Root bookkeeping pages (`index.md`, `log.md`, `views.md` — a slashless
    `page_key`) are skipped: they are catalogues, never retrieval targets. The
    first version of this check tried to express that as
    `md.parent.name == md.parent.parent.name`, which compares a directory name to
    its *grandparent's* and is simply false for `wiki/index.md` (`'wiki' != ''`),
    so root pages were scanned rather than skipped — and would have been reported
    under a `wiki/index`-shaped key that exists in no namespace.

    Takes `lint`'s already-cached frontmatter and body rather than re-reading
    every page: `thin_index_reason` needs a `Page`, and constructing one from the
    cache costs nothing where a second `read_page` pass over 447 files is most of
    a second.
    """
    try:
        from ...index.pages_semantic import thin_index_reason
    except Exception:
        # numpy / sentence-transformers absent — degrade to "not checked",
        # matching how the other optional-dependency checks behave.
        return []

    out: list[dict] = []
    for md in pages:
        key = page_key(md)
        if "/" not in key:          # root bookkeeping — not a retrieval target
            continue
        fm = pages_fm.get(md) or {}
        page = Page(
            path=md,
            stem=md.stem,
            category=md.parent.name,
            fm=fm,
            body=pages_body.get(md, ""),
        )
        try:
            reason = thin_index_reason(page)
        except Exception:
            continue
        if reason:
            out.append({
                "page": key,
                "page_type": str(fm.get("type") or "paper"),
                "reason": reason,
            })
    out.sort(key=lambda d: d["page"])
    return out
