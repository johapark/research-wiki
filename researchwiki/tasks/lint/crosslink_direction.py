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

Years come from each page's YAML `year:`, and a pair is reported only when
both are integers and the gap is at least `MIN_YEAR_GAP` (2). A missing or
malformed year is skipped rather than guessed at.

The two-year floor exists because a one-year "impossibility" is usually a
**preprint citation**, not an error: a 2019 paper citing the 2020-published
version of a preprint it read in 2019 looks backwards against journal years.
Measured on this corpus, 174 of 177 one-year findings were that pattern and 11
were confirmed real citations by reading the PDFs — e.g. `chin-2019` genuinely
cites `zook-2020` ("a robust benchmark for detection of germline large
insertions and deletions"), because Chin was submitted in 2019 and published
in 2020. `lint`'s own `stem_year_drift` reports the same preprint/journal
skew from the other direction. Raising the floor drops that whole class and
keeps the findings whose gap no publication lag explains.

Warn-only, like the other contract checks.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...backlinks import _CITED_BY_CLAIM, _CITES_CLAIM

# A Related Papers bullet: `- [[cat/stem]] — note` or `- [[cat/stem]]: note`.
# The note may be empty; the link may carry an `|alias` or `#anchor`.
#
# Tolerant about the bullet's shape on purpose. `backlinks.py` always writes
# `- [[…]] — …`, but a wrong direction is likeliest in *hand-edited* prose,
# which is exactly where `*` markers, an indented continuation bullet and a
# bolded `**[[…]]**` link turn up — all three parsed to nothing before.
_BULLET_RE = re.compile(
    r"^[ \t]*[-*+]\s+\*{0,2}\[\[([^\]|#]+)[^\]]*\]\]\*{0,2}\s*[—:-]?\s*(.*)$",
    re.MULTILINE,
)
_RELATED_HEADING_RE = re.compile(r"^##\s+Related Papers\s*$", re.MULTILINE | re.IGNORECASE)
_NEXT_H2_RE = re.compile(r"^##\s+", re.MULTILINE)

#: Minimum year gap before a reversed citation claim is reported. See the
#: module docstring: at a gap of 1 the finding is dominated by papers citing
#: the preprint of a work whose journal year is later.
MIN_YEAR_GAP = 2


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
            if _CITES_CLAIM in lowered and own_year - tgt_year >= MIN_YEAR_GAP:
                claim, older, newer = _CITES_CLAIM, tgt_stem, md.stem
            # "cited by this paper" → this page cites the target, so the
            # target must be the older one.
            elif _CITED_BY_CLAIM in lowered and tgt_year - own_year >= MIN_YEAR_GAP:
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
