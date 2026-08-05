"""Wikilink-graph checks: orphans, broken links, missing back-links.

All three derive from the same out_links/in_links graph computed once
over the page corpus. Missing-backlinks is the only check with an
auto-fix path (`--fix`), so the writer (`apply_backlink_fixes`) lives
here too.
"""

from __future__ import annotations

import re
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
        # Root meta pages (`index`, `log`, `views`, … — a slashless key) are
        # catalogues/logs, not authored content: `log.md` in particular
        # accumulates historical entries with template fragments like
        # `[[stem]]` / `[[category/…]]` that are documentation, not real
        # links. Their out-links still feed the graph above (the index links
        # every paper), but their "broken" links are noise — exclude them,
        # consistent with the root-meta exclusion in `find_orphans`.
        bad = broken_links(prose, known)
        if bad and "/" in key:
            broken.append((key, bad))
    return out_links, in_links, broken


def find_orphans(
    pages: list[Path],
    out_links: dict[str, set[str]],
    in_links: dict[str, set[str]],
) -> list[str]:
    """Paper pages with zero in-links and zero out-links.

    Synthesis pages excluded — they're catalog-shaped and legitimately
    can have no inbound links until someone references them. Root meta
    pages (`index`, `log`, `views`, … — a slashless key) excluded too:
    they're catalogues/dashboards, not content nodes with citation
    relationships. Same exclusion as `find_missing_backlinks._excluded`.
    """
    orphans: list[str] = []
    for md in pages:
        key = page_key(md)
        if md.parent.name == "synthesis":
            continue
        if "/" not in key:
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


def _mirrored_note(src_prose: str, tgt_key: str) -> str:
    """Direction-aware note for the back-link mirroring src→tgt.

    The source page already states the relationship in its own bullet, so the
    direction is recoverable locally — no citation lookup needed. Find the
    line where src links tgt and invert whatever it claims.

    Falls back to the weakest claim when src's prose is unavailable, when the
    link sits outside a relationship bullet, or when the bullet is editorial
    prose rather than a canonical phrasing. First match wins if src links tgt
    more than once.
    """
    from ...backlinks import TOPICAL_NOTE, invert_relationship_note

    if not src_prose:
        return TOPICAL_NOTE
    stem = tgt_key.rsplit("/", 1)[-1]
    link_re = re.compile(r"\[\[(?:[^\]\|#]*/)?" + re.escape(stem) + r"[\]\|#]")
    for line in src_prose.splitlines():
        if link_re.search(line):
            return invert_relationship_note(line)
    return TOPICAL_NOTE


def apply_backlink_fixes(
    missing_back: list[tuple[str, str]],
    pages_prose: dict[Path, str] | None = None,
) -> dict[str, int]:
    """Write back-links into target pages via the shared helper.

    `pages_prose` is the prose-only page text `build_link_graph` already
    computed; it is what makes the inserted bullet state the *right*
    direction (see `_mirrored_note`). Omitting it is supported but degrades
    every bullet to the weakest claim.

    Returns {tgt: bullets_added}. Each inserted bullet is marked
    `(auto-added; refine)` so a future LLM pass can rewrite the
    one-liner with a real citation relationship.
    """
    from ...backlinks import append_related_paper

    prose_by_key = {
        page_key(p): text for p, text in (pages_prose or {}).items()
    }
    by_tgt: dict[str, list[str]] = {}
    for src, tgt in missing_back:
        by_tgt.setdefault(tgt, []).append(src)
    written: dict[str, int] = {}
    for tgt, srcs in sorted(by_tgt.items()):
        target_path = wiki_dir() / f"{tgt}.md"
        n = 0
        for src in sorted(set(srcs)):
            note = _mirrored_note(prose_by_key.get(src, ""), tgt)
            if append_related_paper(target_path, src, note=note):
                n += 1
        if n:
            written[tgt] = n
    return written


def find_none_placeholders(pages_body: dict) -> list[str]:
    """Pages whose `## Related Papers` holds a `(none…)` placeholder AND bullets.

    The placeholder is written by the page author when it had nothing to link;
    it becomes a lie the moment a bullet is added, and `append_related_paper`
    used to insert underneath it rather than clearing it. 62 pages carried the
    contradiction when this check was added — every one of them produced by that
    path, and invisible to every other check because the links themselves are
    fine.

    Cosmetic-only, so advisory: nothing downstream reads the placeholder. The
    insertion path now clears it (`backlinks._drop_none_placeholder`), making
    this the backstop for pages that predate the fix or that were hand-edited.
    """
    from ...backlinks import NONE_PLACEHOLDER_RE
    out: list[str] = []
    for md, body in pages_body.items():
        m = re.search(r"^## Related Papers\s*$", body, re.MULTILINE)
        if not m:
            continue
        section = body[m.end():]
        nxt = re.search(r"^## ", section, re.MULTILINE)
        if nxt:
            section = section[:nxt.start()]
        has_placeholder = any(NONE_PLACEHOLDER_RE.match(ln) for ln in section.split("\n"))
        has_bullet = bool(re.search(r"^- \[\[", section, re.MULTILINE))
        if has_placeholder and has_bullet:
            out.append(page_key(md))
    out.sort()
    return out
