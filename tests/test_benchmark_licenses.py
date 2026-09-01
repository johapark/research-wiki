"""Article-bound license evidence for bundled benchmark additions.

Issue-extracted PDFs can contain the end of the preceding article. Merely
finding a Creative Commons block anywhere in such a file is therefore not
enough: the bundled article's identity must precede its license grant within
the same article boundary. These first-page cases make that check mechanical.
"""

from pathlib import Path

import pytest

from researchwiki.pdf.text import extract_pdf_page_texts


PDFS = Path(__file__).resolve().parent.parent / "benchmark-fixtures" / "pdfs"


@pytest.mark.parametrize(
    ("stem", "article_identity", "license_identity"),
    [
        (
            "chuai-2018-deepcrispr-optimized-crispr-guide-rna",
            "DeepCRISPR: optimized CRISPR guide RNA design by deep learning",
            "Creative Commons Attribution 4.0 International License",
        ),
        (
            "assa-2024-quantifying-allele-specific-crispr-editing-activity",
            "Quantifying allele-specific CRISPR editing activity with CRISPECTOR2.0",
            "Creative Commons Attribution License",
        ),
    ],
)
def test_article_identity_precedes_its_first_page_cc_by_grant(
    stem: str, article_identity: str, license_identity: str
):
    first_page = " ".join(
        extract_pdf_page_texts(PDFS / f"{stem}.pdf", max_pages=1)[0].split()
    )

    article_at = first_page.find(article_identity)
    license_at = first_page.find(license_identity)

    assert article_at >= 0, f"article identity not found on page 1 of {stem}"
    assert license_at >= 0, f"CC-BY grant not found on page 1 of {stem}"
    assert article_at < license_at, (
        f"CC block precedes {stem}'s article identity; it may license an "
        "adjacent article instead"
    )
