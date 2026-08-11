"""A PDF's `/Title` is often production furniture, not the title.

Oxford stamps it with an internal job code plus the page range. minimap2's PDF
carries `OP-CBIO180195 3094..3100`, which is 24 characters and begins with nothing
on the banned-prefix list, so the old length-and-prefix check adopted it and
short-circuited the first-page text scan. That string then became the Semantic
Scholar title-match query — three 404s, retried — and the last-resort page title.
9 of 345 `/Title` values in the corpus were furniture of this shape (Oxford's
`OP-CBIO*`, Genome Research's `genome*`, one Liebert `CMB-*`).

CLAUDE.md already names first-page text, "not `reader.metadata`", as the source of
truth for the naming fields, so falling through to the scan is the documented
behaviour as well as the correct one. Observed 2026-08-10.
"""

from __future__ import annotations

import pytest

from researchwiki.agents.phases import reconcile as R


# ---------- _looks_like_title ----------

@pytest.mark.parametrize("furniture", [
    "OP-CBIO180195 3094..3100",              # Oxford job code + page range
    "OP-CBIO140048 1473..1475",
    "genome94052 1655..1664",                # Genome Research
    "CMB-2017-0251-ver9-Paten_3P 649..663",  # Mary Ann Liebert
])
def test_rejects_production_furniture(furniture):
    assert R._looks_like_title(furniture) is False


@pytest.mark.parametrize("title", [
    "Minimap2: pairwise alignment for nucleotide sequences",
    "Cas-OFFinder: a fast and versatile algorithm that searches for potential off-target sites",
    "A draft human pangenome reference",
    "Superbubbles, Ultrabubbles, and Cacti",
    "lDDT: a local superposition-free score for comparing protein structures",
    "GBZ file format for pangenome graphs",
    "We Still Don't Understand High-Dimensional Bayesian Optimization",
])
def test_accepts_real_titles(title):
    assert R._looks_like_title(title) is True


def test_empty_and_none_are_not_titles():
    assert R._looks_like_title(None) is False
    assert R._looks_like_title("") is False
    assert R._looks_like_title("   ") is False


# ---------- masthead lines in the text scan ----------

@pytest.mark.parametrize("masthead", [
    "Vol. 30 no. 10 2014, pages 1473–1475 BIOINFORMATICS APPLICATIONS NOTE",
    "Sequence analysis Advance Access publication January 24, 2014",
    "Genome Res. 2009 19: 1655-1664 originally published online July 31",
    "doi:10.1093/bioinformatics/btu048",
    "Nature Methods 10.1038/s41592-025-02626-1",
])
def test_masthead_lines_are_recognized(masthead):
    assert R._MASTHEAD_LINE_RE.search(masthead) is not None


@pytest.mark.parametrize("title", [
    "Minimap2: pairwise alignment for nucleotide sequences",
    "Superbubbles, Ultrabubbles, and Cacti",
    "A complete diploid human genome benchmark for personalized genomics",
    # A title may legitimately carry numbers and a colon; none of that is masthead.
    "Evo 2: genome modelling and design across all domains of life",
])
def test_real_titles_are_not_mistaken_for_masthead(title):
    assert R._MASTHEAD_LINE_RE.search(title) is None


# ---------- the real extractor ----------

def test_furniture_title_falls_through_to_the_text_scan():
    # Drives `_extract_title_from_pdf` itself rather than a reimplementation.
    meta = {"/Title": "OP-CBIO180195 3094..3100"}
    text = (
        "Vol. 34 no. 18 2018, pages 3094–3100 BIOINFORMATICS ORIGINAL PAPER\n"
        "doi:10.1093/bioinformatics/bty191\n"
        "Sequence analysis Advance Access publication May 10, 2018\n"
        "Minimap2: pairwise alignment for nucleotide sequences\n"
        "Heng Li*\n"
        "Abstract\n"
    )
    assert R._extract_title_from_pdf(meta, text).startswith(
        "Minimap2: pairwise alignment for nucleotide sequences")


def test_a_good_meta_title_is_still_preferred():
    # No regression for well-behaved PDFs: 336 of 345 corpus titles take this path.
    good = "A complete diploid human genome benchmark for personalized genomics"
    assert R._extract_title_from_pdf({"/Title": good}, "irrelevant body text") == good


def test_clean_pdf_meta_value_drops_furniture_for_titles_only():
    assert R._clean_pdf_meta_value("OP-CBIO180195 3094..3100", kind="title") is None
    # Author strings legitimately fail a prose-shape test, so the check is
    # title-only — an author deny-list is a different problem.
    assert R._clean_pdf_meta_value("H. Li", kind="author") == "H. Li"
