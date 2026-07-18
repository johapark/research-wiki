"""Page enumeration + link / YAML parsing helpers shared across lint checks.

These primitives are read-only: they walk `wiki/`, extract markdown
constructs, and normalize frontmatter values. No side effects, no
network. Every check in the lint subpackage imports from here rather
than re-implementing page enumeration.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...paths import wiki_dir


WIKILINK_RE = re.compile(r"\[\[([^\]\|#]+?)(?:#[^\]\|]*)?(?:\|[^\]]+)?\]\]")

# Concept-candidate primitives (ACRONYM_RE, PHRASE_RE, STOP_ACRONYMS,
# STRUCTURAL_TOKENS, STOP_PHRASES) live in `researchwiki.concepts.candidates`.
# Candidates are opportunities, not defects, so they live with the task
# that scaffolds hubs rather than in lint.


def all_pages() -> list[Path]:
    """Sorted list of every `*.md` file under `wiki/`. Empty when the
    directory is missing (fresh clone)."""
    root = wiki_dir()
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.md"))


def page_key(md: Path) -> str:
    """`wiki/category/stem.md` → `category/stem`. Used as the canonical
    identifier across the link graph."""
    return md.relative_to(wiki_dir()).with_suffix("").as_posix()


def extract_links(text: str, known: set[str]) -> set[str]:
    """Resolve every `[[...]]` in `text` to a known page key.

    Bare-stem links (no slash) resolve to the matching `category/stem` key
    when unambiguous — mirrors how Obsidian's wikilink parser walks the
    vault. Unknown targets are dropped silently here; `broken_links` below
    is the inverse pass that surfaces them.
    """
    hits: set[str] = set()
    bare_known = {k.split("/", 1)[1]: k for k in known if "/" in k}
    for m in WIKILINK_RE.finditer(text):
        raw = m.group(1).strip()
        if raw in known:
            hits.add(raw)
        elif "/" not in raw and raw in bare_known:
            hits.add(bare_known[raw])
    return hits


def broken_links(text: str, known: set[str]) -> list[str]:
    """Inverse of `extract_links` — return raw link targets that don't
    resolve. Root-level pages (wiki/index.md, wiki/log.md) produce
    slashless keys; check `known` first so bare links to root-level pages
    resolve before falling through to the bare-stem index."""
    bare_known = {k.split("/", 1)[1] for k in known if "/" in k}
    broken: list[str] = []
    for m in WIKILINK_RE.finditer(text):
        raw = m.group(1).strip()
        if "/" in raw:
            if raw not in known:
                broken.append(raw)
        else:
            if raw not in known and raw not in bare_known:
                broken.append(raw)
    return broken


def first_category(raw) -> str:
    """Normalize a frontmatter `category:` value to its first category name.

    The line-based frontmatter parser keeps the YAML list as a string literal
    (e.g. ``'[single-cell]'``); a PyYAML-parsed frontmatter yields a real list.
    Handle both, plus a bare scalar. Returns ``""`` when empty/unparseable.
    """
    if isinstance(raw, (list, tuple)):
        return str(raw[0]).strip().strip("'\"") if raw else ""
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if "," in s:
        s = s.split(",", 1)[0]
    return s.strip().strip("'\"")


def count_keywords(raw) -> int:
    """Count items in a YAML `keywords:` value.

    Handles both frontmatter shapes: a real list (PyYAML) and the line
    parser's literal string (a list `[a, b, c]` arrives as the 9-character
    string `[a, b, c]`). Strip brackets, split on commas, drop empties.
    """
    if isinstance(raw, (list, tuple)):
        return sum(1 for x in raw if str(x).strip())
    s = (raw or "").strip()
    if not s:
        return 0
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return sum(1 for part in s.split(",") if part.strip())
