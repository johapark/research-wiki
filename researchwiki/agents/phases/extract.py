"""Extract phase — section + claim extractor over the PDF.

Runs after reconcile and before author. Produces (sections_dict, claim_count,
full_text) that the author phase consumes for grounding and the runner uses
to bound expectations on graded-claim count. `full_text` is the unbounded
extracted PDF text — retained so downstream phases that need a wider sampling
window (notably keyword extraction) can rescan without re-reading the PDF.
"""

from __future__ import annotations

from pathlib import Path

from ...pdf.text import extract_pdf
from ...pdf.sections import anchor_sections


def extract_sections(pdf_path: Path) -> tuple[dict, int, str]:
    """Run the structured section + claim extractors over the PDF.

    Returns (sections_dict, claim_count, full_text). `sections` has whichever
    of {introduction, methods, results, discussion, references} the anchor
    parser found, each capped at 4000 chars from its start. `full_text` is
    the full extracted PDF text (no cap), so downstream phases needing a
    larger window for named-entity extraction (keyword proposal, where
    primary tools like RUFUS often live past the 4000-char Methods cut)
    can sample without re-extracting.
    """
    text, _ = extract_pdf(pdf_path, max_pages=80)
    sections = anchor_sections(text, max_chars=4000)
    # `claim_count` is a rough signal — number of bullet-shaped lines we'd
    # parse out of Methods + Results. Real claim parsing happens once the
    # wiki page is committed and the DB rebuild picks it up.
    claim_count = 0
    for name, body in sections.items():
        if name in ("methods", "results"):
            for line in body.splitlines():
                s = line.strip()
                if (s.startswith("-") or s.startswith("*")) and len(s) > 10:
                    claim_count += 1
    return sections, claim_count, text
