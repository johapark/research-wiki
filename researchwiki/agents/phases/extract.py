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


def extract_sections(pdf_path: Path) -> tuple[dict, str]:
    """Run the structured section extractor over the PDF.

    Returns (sections_dict, full_text). `sections` has whichever
    of {introduction, methods, results, discussion, references} the anchor
    parser found, each capped at 4000 chars from its start. `full_text` is
    the full extracted PDF text (no cap), so downstream phases needing a
    larger window for named-entity extraction (keyword proposal, where
    primary tools like RUFUS often live past the 4000-char Methods cut)
    can sample without re-extracting.

    Returned a third value, `claim_count`, until 2026-08-10. It counted lines
    in Methods/Results that start with "-" or "*" — markdown bullets, in
    typeset PDF prose, which essentially never contain any. So it read 0 for
    almost every real paper (6 of 8 in one ingest session), and the two it
    did find on the one exception were samtools/pbmm2 command-line flags
    (`--secondary=no -s 25000 -K 15G`), not claims. Nothing consumed the
    number except a warning that therefore fired on nearly every ingest, and
    a `ctx.claims_count` field that was assigned and never read.

    The PDF-side claim count that phase 2.4 (`target_claims`) produces is the
    real one — an LLM extraction that returns 18–35 claims on these same
    papers, with per-claim importance. `runner._warn_thin_extraction` now
    keys the zero-claims warning off that instead.
    """
    text, _ = extract_pdf(pdf_path, max_pages=80)
    sections = anchor_sections(text, max_chars=4000)
    return sections, text
