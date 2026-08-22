"""Prose rendering for the two checks that cover a hand-deleted page.

Split out of `report.py` at its size pin, along the same seam
`report_cross_paper.py` took: these two sections are a pair, and they are the
only ones whose subject is not a page at all. Every other section in the report
names a page and says what is wrong inside it. These name what a page left
behind after it stopped existing — a catalogue bullet in `index.md`, a PDF in
`papers/` — which is why both need a sentence of orientation that the rest of
the report never does, and why neither is auto-fixed.
"""

from __future__ import annotations


def print_broken_index_bullets(entries: list[dict]) -> None:
    """`index.md` lines whose wikilinks no longer resolve.

    Silent when clean: unlike `broken_wikilinks` this is not a standing count
    anybody watches, and a zero line every run would be noise.
    """
    if not entries:
        return
    n = sum(len(e["targets"]) for e in entries)
    print(f"## Broken index.md bullets ({n} across {len(entries)} line(s))")
    print("   The catalogue advertises pages that no longer exist. "
          "`broken_wikilinks` cannot see these — root meta pages are excluded "
          "from that scan so `log.md`'s historical template fragments don't "
          "drown it. Delete the bullet, or restore the page.")
    for e in entries[:20]:
        print(f"- **index.md:{e['line']}** → {', '.join(e['targets'])}")
    if len(entries) > 20:
        print(f"_... +{len(entries) - 20} more_")
    print()


def print_orphan_pdfs(stems: list[str]) -> None:
    """PDFs in `papers/` no page claims.

    Not always a defect, so the section says which of the two states it is
    rather than leaving the reader to guess.
    """
    if not stems:
        return
    print(f"## PDFs in papers/ with no page ({len(stems)})")
    print("   Unreachable from the wiki: not searchable, not citable, not in "
          "`index.md`. Either re-ingest "
          "(`researchwiki agent ingest papers/<stem>.pdf`) or delete. Expected "
          "right after `remove --keep-pdf` — that is the re-ingest queue, not "
          "a defect.")
    for stem in stems[:20]:
        print(f"- papers/{stem}.pdf")
    if len(stems) > 20:
        print(f"_... +{len(stems) - 20} more_")
    print()
