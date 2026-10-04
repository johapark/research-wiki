"""Back-link bullets whose citation direction the publication years rule out.

A *Related Papers* bullet reading `[[T]] — cites this paper` on page S asserts
that T cites S ("this paper" is the page the bullet sits on). If T was
published before S, that cannot be true, and the inverse phrasing has the
mirrored problem.

This is the one part of CLAUDE.md's cross-link corollary — a link must rest on
the source explicitly citing, building on or contrasting the target — that is
checkable without opening a PDF. The rest needs the reference list, which is
why the corollary is otherwise enforced at authoring time.

Why it is worth a check rather than a one-off cleanup: `lint --fix` and
`promote` both write these bullets from an edge whose direction they inferred,
and a wrong direction is invisible to every other check — `missing_backlinks`
is satisfied by the bullet's *presence*, and both grading gates parse
paragraphs, never a bullet's claim. A sample of three flagged pairs confirmed
in 2026-10 that the older paper's PDF does not mention the newer one at all.

  crosslink_impossible_citation
    One bullet. Names the pair, both years, and which phrasing is wrong.

Years come from each page's YAML `year:`. A pair is reported only when both
years are integers and strictly ordered, so a missing or malformed year is
skipped rather than guessed at. Same-year pairs are never reported: a paper
can legitimately cite a preprint from earlier the same year.

Warn-only, like the other contract checks.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...backlinks import _CITED_BY_CLAIM, _CITES_CLAIM

# A Related Papers bullet: `- [[cat/stem]] — note` or `- [[cat/stem]]: note`.
# The note may be empty; the link may carry an `|alias` or `#anchor`.
_BULLET_RE = re.compile(r"^-\s+\[\[([^\]|#]+)[^\]]*\]\]\s*[—:-]?\s*(.*)$", re.MULTILINE)
_RELATED_HEADING_RE = re.compile(r"^##\s+Related Papers\s*$", re.MULTILINE | re.IGNORECASE)
_NEXT_H2_RE = re.compile(r"^##\s+", re.MULTILINE)


def _related_section(body: str) -> str:
    m = _RELATED_HEADING_RE.search(body)
    if not m:
        return ""
    rest = body[m.end():]
    nxt = _NEXT_H2_RE.search(rest)
    return rest[: nxt.start()] if nxt else rest


def _year(fm: dict) -> int | None:
    raw = fm.get("year")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None


def find_impossible_citation_directions(
    pages: list[Path], pages_body: dict[Path, str], pages_fm: dict[Path, dict],
) -> list[dict]:
    """Bullets asserting a citation the two pages' years make impossible."""
    year_by_stem: dict[str, int] = {}
    for md in pages:
        y = _year(pages_fm.get(md, {}) or {})
        if y is not None:
            year_by_stem[md.stem] = y

    out: list[dict] = []
    for md in pages:
        own_year = year_by_stem.get(md.stem)
        if own_year is None:
            continue
        section = _related_section(pages_body.get(md, ""))
        if not section:
            continue
        for target, note in _BULLET_RE.findall(section):
            tgt_stem = target.rsplit("/", 1)[-1].strip()
            tgt_year = year_by_stem.get(tgt_stem)
            if tgt_year is None:
                continue
            lowered = note.lower()
            # "cites this paper" → the target cites this page, so the target
            # must be the newer of the two.
            if _CITES_CLAIM in lowered and tgt_year < own_year:
                claim, older, newer = _CITES_CLAIM, tgt_stem, md.stem
            # "cited by this paper" → this page cites the target, so the
            # target must be the older one.
            elif _CITED_BY_CLAIM in lowered and tgt_year > own_year:
                claim, older, newer = _CITED_BY_CLAIM, md.stem, tgt_stem
            else:
                continue
            out.append({
                "page": md,
                "kind": "crosslink_impossible_citation",
                "detail": f"`[[{target}]] — {claim}` needs {older} ({year_by_stem[older]}) "
                          f"to cite {newer} ({year_by_stem[newer]})",
            })
    return out
