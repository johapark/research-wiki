"""Reference-list scoping: what's past the References heading is somebody else's
paper, and what's in it can prove a citation.

Both halves come from the DeepSpCas9 ingest — the URL-DOI hunt scavenging a
cited paper's DOI, and three real citation relationships shipped as merely
"topical" because nobody looked at the reference list.
"""

from __future__ import annotations

import pytest

from researchwiki.pdf.text import REFS_HEADING_RE, find_url_doi_candidates


BODY = """A Paper About Things
https://doi.org/10.1126/sciadv.aax9249

We trained with Adam and evaluated on held-out data.
Data are at ssrn.com/abstract=1234567

REFERENCES AND NOTES
1. D. P. Kingma, J. Ba, arxiv.org/abs/1412.6980 (2014).
2. Someone Else, ssrn.com/abstract=9999999 (2020).
"""


class TestUrlHuntStopsAtReferences:
    def test_a_cited_arxiv_url_is_not_a_candidate(self):
        """The exact scavenge that offered the Adam optimizer as the paper's DOI."""
        got = dict((d, p) for p, d in find_url_doi_candidates(BODY))
        assert "10.48550/arXiv.1412.6980" not in got

    def test_a_body_ssrn_footer_is_still_found(self):
        """The case this function exists for must keep working."""
        got = dict((d, p) for p, d in find_url_doi_candidates(BODY))
        assert got.get("10.2139/ssrn.1234567") == "ssrn-url"

    def test_a_cited_ssrn_url_is_excluded(self):
        got = dict((d, p) for p, d in find_url_doi_candidates(BODY))
        assert "10.2139/ssrn.9999999" not in got

    def test_opt_out_restores_whole_document_scanning(self):
        got = dict((d, p) for p, d in find_url_doi_candidates(BODY, stop_at_references=False))
        assert "10.48550/arXiv.1412.6980" in got

    def test_no_references_heading_scans_everything(self):
        # SSRN ids must be 4-12 digits to match (pre-existing `_SSRN_ABSTRACT_RE`).
        text = "Title\nssrn.com/abstract=5551234\narxiv.org/abs/2101.00001\n"
        got = [d for _, d in find_url_doi_candidates(text)]
        assert "10.2139/ssrn.5551234" in got and "10.48550/arXiv.2101.00001" in got


class TestRefsHeadingCoverage:
    @pytest.mark.parametrize("heading", [
        "References",
        "REFERENCES",
        "Bibliography",
        "Literature cited",
        "REFERENCES AND NOTES",   # Science / AAAS — the form that used to miss
        "References and Notes",
        "Reference",              # singular
        "References:",
        "78REFERENCES",           # page number glued on by extraction
        "9References",
        "12. References",
    ])
    def test_matches(self, heading):
        assert REFS_HEADING_RE.search(f"body text\n{heading}\n1. A. Author\n")

    @pytest.mark.parametrize("line", [
        "See the references for details.",
        "references to prior work are collected below",
        "This bibliography is incomplete because",
    ])
    def test_does_not_match_prose(self, line):
        assert not REFS_HEADING_RE.search(f"body\n{line}\nmore\n")


class TestCitesReference:
    """`cites_reference` reads a real PDF, so it's exercised against the corpus
    paper that motivated it when that file is present."""

    PDF = "papers/kim-2019-spcas9-activity-prediction-by-deepspcas9.pdf"

    @pytest.fixture(autouse=True)
    def _skip_without_pdf(self):
        from pathlib import Path
        if not Path(self.PDF).exists():
            pytest.skip("corpus PDF not available")

    @pytest.mark.parametrize("surname,year,expected", [
        ("doench", "2014", True),           # its ref. 14
        ("chuai", "2018", True),            # DeepCRISPR
        ("hsu", "2013", True),
        ("sherkatghanad", "2023", False),   # postdates the paper; cites IT instead
        ("nosuchauthor", "1999", False),
    ])
    def test_against_the_real_reference_list(self, surname, year, expected):
        from pathlib import Path
        from researchwiki.pdf.text import cites_reference
        assert cites_reference(Path(self.PDF), surname, year) is expected

    def test_empty_inputs_are_false_not_an_error(self):
        from pathlib import Path
        from researchwiki.pdf.text import cites_reference
        assert cites_reference(Path(self.PDF), "", "2014") is False
        assert cites_reference(Path(self.PDF), "doench", "") is False

    def test_unreadable_pdf_is_false(self):
        from pathlib import Path
        from researchwiki.pdf.text import cites_reference
        assert cites_reference(Path("/nonexistent.pdf"), "doench", "2014") is False


class TestVenueAgreement:
    @pytest.mark.parametrize("a,b,want", [
        ("NeurIPS 2025", "Neural Information Processing Systems", True),
        ("Nature Biotechnology", "Nat Biotechnol", True),
        ("ICML", "International Conference on Machine Learning", True),
        ("Proc. Natl. Acad. Sci.", "Proceedings of the National Academy of Sciences", True),
        ("Nature", "Nature Genetics", True),
        ("Genetics", "Science Advances", False),          # the kim-2019 defect
        ("Journal of LaTeX Class Files", "IEEE Transactions on Pattern Analysis", False),
    ])
    def test_agreement(self, a, b, want):
        from researchwiki.tasks.backfill import _venue_agrees
        assert _venue_agrees(a, b) is want

    def test_a_missing_side_never_reports_drift(self):
        from researchwiki.tasks.backfill import _venue_agrees
        assert _venue_agrees("", "Nature") is True
        assert _venue_agrees("Nature", "") is True


class TestEtAlBylines:
    """`Guohui Chuai et al.` is an abbreviated byline, not a name ending in `al`."""

    @pytest.mark.parametrize("byline,want", [
        ("Guohui Chuai et al.", "chuai"),
        ("John G Doench et al", "doench"),
        ("Karl Petri, et al.", "petri"),
        ("Weian Du and others", "du"),
    ])
    def test_abbreviation_is_stripped(self, byline, want):
        from researchwiki.stems import first_author_surname
        assert first_author_surname([byline]) == want

    @pytest.mark.parametrize("byline,want", [
        ("Hui Kwon Kim", "kim"),
        ("García-López, Ana", "garcia-lopez"),
        ("S. De Winter", "de-winter"),
        ("1000 Genomes Project Consortium", "1000-genomes-project-consortium"),
        ("Artur Szałata", "szalata"),
    ])
    def test_normal_bylines_are_untouched(self, byline, want):
        from researchwiki.stems import first_author_surname
        assert first_author_surname([byline]) == want

    def test_a_byline_that_is_only_et_al_is_unknown(self):
        from researchwiki.stems import first_author_surname
        assert first_author_surname(["et al."]) == "unknown"
