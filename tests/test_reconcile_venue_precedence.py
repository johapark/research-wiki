"""A preprint-only record must not inherit S2's merged formal venue."""

from researchwiki import metadata_sanity


def test_arxiv_doi_rejects_merged_conference_venue():
    s2 = {
        "venue": "Annual Meeting of the Association for Computational Linguistics",
    }

    assert metadata_sanity.trusted_s2_venue(
        s2["venue"], "10.48550/arXiv.2501.10120"
    ) is None


def test_arxiv_doi_keeps_preprint_venue():
    assert metadata_sanity.trusted_s2_venue(
        "arXiv.org", "10.48550/arXiv.2501.10120"
    ) == "arXiv.org"


def test_formal_doi_keeps_formal_venue():
    assert metadata_sanity.trusted_s2_venue(
        "Nucleic Acids Research", "10.1093/nar/gkae235"
    ) == "Nucleic Acids Research"
