"""What sits in `papers/` that no page claims.

`papers/` holds exactly one canonically-named PDF per wiki page, and the stem
is the join key — `db rebuild` derives `papers/{stem}.pdf` from it. So a PDF
whose stem matches no page is a file nothing in the wiki can reach: not
searchable, not citable, not listed in `index.md`, and invisible to every other
check, all of which start from the page corpus and walk outwards.

That asymmetry is the reason this module exists. `researchwiki remove` deletes
the page and the PDF together, but a page deleted by hand (an `rm`, an Obsidian
delete, a `git` operation) strands the PDF silently — `db rebuild` drops the
row, `lint` reports the broken wikilinks on citing pages, and nothing at all
mentions the file still sitting on disk.

**Not always a defect.** `remove --keep-pdf` produces this state deliberately,
so the paper can be re-ingested clean. Read the list as a queue with two exits:
`researchwiki agent ingest papers/{stem}.pdf` to bring the paper back, or `rm`
to finish the job.
"""

from __future__ import annotations

from pathlib import Path

from ...paths import papers_dir


def find_orphan_pdfs(pages: list[Path]) -> list[str]:
    """PDF stems in `papers/` with no `wiki/**/{stem}.md`. Sorted.

    Non-recursive glob on purpose: `papers/{stem}.supp/*.pdf` is supplementary
    material, which belongs to its parent page and is covered by
    `find_supplementary_issues`. A recursive walk would report every one of
    them here as well.
    """
    pdir = papers_dir()
    if not pdir.is_dir():
        return []
    page_stems = {md.stem for md in pages}
    return sorted(
        pdf.stem for pdf in pdir.glob("*.pdf")
        if pdf.is_file() and pdf.stem not in page_stems
    )
