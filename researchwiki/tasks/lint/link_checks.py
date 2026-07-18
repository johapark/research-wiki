"""Wikilink-graph checks: orphans, broken links, missing back-links.

All three derive from the same out_links/in_links graph computed once
over the page corpus. Missing-backlinks is the only check with an
auto-fix path (`--fix`), so the writer (`apply_backlink_fixes`) lives
here too.
"""

from __future__ import annotations

from pathlib import Path

from ...paths import wiki_dir
from .walk import broken_links, extract_links, page_key


def build_link_graph(
    pages: list[Path],
    pages_prose: dict[Path, str],
    known: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]], list[tuple[str, list[str]]]]:
    """Walk every page's prose and build:

      out_links  : page → set of resolved targets
      in_links   : page → set of pages that link to it
      broken     : list of (page, [unresolved targets])

    Computed from prose only — HTML comments and code blocks hold
    template examples like `[[category/page]]` that are not real links.
    """
    out_links: dict[str, set[str]] = {}
    in_links: dict[str, set[str]] = {}
    broken: list[tuple[str, list[str]]] = []
    for md in pages:
        key = page_key(md)
        prose = pages_prose[md]
        tgts = extract_links(prose, known) - {key}
        out_links[key] = tgts
        for t in tgts:
            in_links.setdefault(t, set()).add(key)
        bad = broken_links(prose, known)
        if bad:
            broken.append((key, bad))
    return out_links, in_links, broken


def find_orphans(
    pages: list[Path],
    out_links: dict[str, set[str]],
    in_links: dict[str, set[str]],
) -> list[str]:
    """Paper pages with zero in-links and zero out-links.

    Synthesis pages excluded — they're catalog-shaped and legitimately
    can have no inbound links until someone references them.
    """
    orphans: list[str] = []
    for md in pages:
        key = page_key(md)
        if md.parent.name == "synthesis":
            continue
        if not out_links.get(key) and not in_links.get(key):
            orphans.append(key)
    orphans.sort()
    return orphans


def find_missing_backlinks(out_links: dict[str, set[str]]) -> list[tuple[str, str]]:
    """Pairs (src, tgt) where src→tgt exists but tgt→src doesn't.

    Only genuine reciprocal edges are surfaced — a symmetric backlink is
    required only between pages that are content nodes in the graph. These
    are excluded both as src and tgt because forcing symmetric links there
    is wrong or noisy:

      - root meta pages (`index`, `log`, `views`, …): a slashless key is a
        catalogue / log, not a content node. The index links *every* paper;
        papers must not carry a `- [[index]]` backlink.
      - `synthesis/`, `ideas/`: page-type pages that *reference* many papers
        as grounding (synthesis cites; an idea leans on 20+ papers). Forcing
        a backlink bullet onto every referenced paper is the noise this guard
        prevents.
      - `references/`: document-shaped, different linking semantics.

    `concepts/` is intentionally NOT excluded: a hub↔member edge is meant to
    be reciprocal (the `concepts` task adds the member→hub back-link), so a
    one-way concept edge is a real gap worth surfacing.
    """
    EXCLUDED_PREFIXES = ("synthesis/", "references/", "ideas/")

    def _excluded(key: str) -> bool:
        return "/" not in key or key.startswith(EXCLUDED_PREFIXES)

    missing: list[tuple[str, str]] = []
    for src, tgts in out_links.items():
        if _excluded(src):
            continue
        for t in tgts:
            if t == src or _excluded(t):   # never a self-pair
                continue
            if src not in out_links.get(t, set()):
                missing.append((src, t))
    return missing


def apply_backlink_fixes(missing_back: list[tuple[str, str]]) -> dict[str, int]:
    """Write back-links into target pages via the shared helper.

    Returns {tgt: bullets_added}. Each inserted bullet is marked
    `(auto-added; refine)` so a future LLM pass can rewrite the
    one-liner with a real citation relationship.
    """
    from ...backlinks import append_related_paper

    by_tgt: dict[str, list[str]] = {}
    for src, tgt in missing_back:
        by_tgt.setdefault(tgt, []).append(src)
    written: dict[str, int] = {}
    for tgt, srcs in sorted(by_tgt.items()):
        target_path = wiki_dir() / f"{tgt}.md"
        n = sum(1 for src in sorted(set(srcs)) if append_related_paper(target_path, src))
        if n:
            written[tgt] = n
    return written
