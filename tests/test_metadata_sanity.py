"""The adoption gate for metadata about to be written into a page.

Centred on the failure it was built for: `reconcile`'s URL-DOI hunt adopting a
DOI on resolvability alone, which on the DeepSpCas9 ingest meant offering the
Adam optimizer's arXiv DOI — scavenged from the reference list — as the paper's
own.
"""

from __future__ import annotations

import pytest

from researchwiki import metadata_sanity as ms


DEEPSPCAS9 = "SpCas9 activity prediction by DeepSpCas9, a deep learning-based model with high generalization performance"
ADAM = "Adam: A Method for Stochastic Optimization"


class TestTheMotivatingCase:
    def test_the_adam_optimizer_is_rejected_for_the_deepspcas9_paper(self):
        """The exact record `reconcile` used to accept on resolvability alone."""
        assert not ms.sanity_ok(
            2014, ["Diederik P. Kingma", "Jimmy Ba"], ADAM,
            "Kim", 2019, DEEPSPCAS9,
        )

    def test_and_says_which_check_killed_it(self):
        why = ms.reject_reason(
            2014, ["Diederik P. Kingma", "Jimmy Ba"], ADAM,
            "Kim", 2019, DEEPSPCAS9,
        )
        assert why and "not among candidate authors" in why

    def test_the_real_record_is_accepted(self):
        assert ms.sanity_ok(
            2019,
            ["Hui Kwon Kim", "Younggwang Kim", "Sungroh Yoon", "Hyongbum Henry Kim"],
            DEEPSPCAS9, "Kim", 2019, DEEPSPCAS9,
        )
        assert ms.reject_reason(
            2019, ["Hui Kwon Kim"], DEEPSPCAS9, "Kim", 2019, DEEPSPCAS9,
        ) is None

    def test_a_same_author_different_paper_is_still_rejected(self):
        """The author check alone is not enough — prolific authors defeat it."""
        assert not ms.sanity_ok(
            2019, ["Hui Kwon Kim"],
            "Deep learning improves prediction of CRISPR-Cpf1 guide RNA activity",
            "Kim", 2019, DEEPSPCAS9,
        )


class TestDegradation:
    """Unknown inputs must weaken the test, not silently fail or pass it."""

    def test_missing_first_author_refuses(self):
        assert not ms.sanity_ok(2019, ["Someone"], DEEPSPCAS9, "", 2019, DEEPSPCAS9)

    def test_unknown_years_do_not_veto(self):
        assert ms.sanity_ok(None, ["Kim"], DEEPSPCAS9, "Kim", None, DEEPSPCAS9)

    def test_unjudgeable_title_does_not_veto(self):
        """A PDF whose title didn't extract must not veto every candidate."""
        assert ms.title_overlap("", "anything") is None
        assert ms.sanity_ok(2019, ["Kim"], DEEPSPCAS9, "Kim", 2019, "")

    def test_overlap_none_is_distinct_from_zero(self):
        assert ms.title_overlap("the of and", "a an") is None
        assert ms.title_overlap("genome editing", "protein folding") == 0.0

    @pytest.mark.parametrize("cand,wiki,ok", [
        (2019, 2019, True),
        (2019, 2020, True),    # preprint → journal shifts by one
        (2019, 2018, True),
        (2019, 2021, False),
    ])
    def test_year_tolerance(self, cand, wiki, ok):
        assert ms.sanity_ok(cand, ["Kim"], DEEPSPCAS9, "Kim", wiki, DEEPSPCAS9) is ok

    def test_diacritics_fold(self):
        assert ms.sanity_ok(
            2020, ["Ana García-López"], "A study of things",
            "Garcia-Lopez", 2020, "A study of things",
        )


class TestVenueFurniture:
    def test_latex_boilerplate_is_furniture(self):
        """A live corpus defect: ai/wang-2024-a-comprehensive-survey-of-continual."""
        assert ms.is_venue_furniture("Journal of LaTeX Class Files")

    @pytest.mark.parametrize("v", ["Preprint", "preprint", "Submitted to Nature",
                                   "Manuscript submitted to ACM", "In preparation"])
    def test_other_furniture(self, v):
        assert ms.is_venue_furniture(v)

    @pytest.mark.parametrize("v", [
        "Genetics",              # a REAL journal — genomics/li-2003-… carries it
        "Science Advances", "Nature Biotechnology", "bioRxiv", "", None,
    ])
    def test_real_venues_are_not_furniture(self, v):
        """The deny-list must never reach a name a real journal has.

        `Genetics` is the case that rules out a subject-word deny-list: it was
        *wrong* on kim-2019 (Science Advances' section label) and *right* on
        li-2003 (the Genetics Society of America journal), so it cannot be
        decided from the string alone.
        """
        assert not ms.is_venue_furniture(v)


class TestPlaceholderDoi:
    def test_the_acm_template_placeholder(self):
        """ai/formal-2021-splade-v2-… shipped with this; the PDF prints it."""
        assert ms.is_placeholder_doi("10.1145/nnnnnnn.nnnnnnn")

    @pytest.mark.parametrize("d", ["10.1145/XXXXXXX.XXXXXXX", "10.1145/0000000.0000000",
                                   "10.1234/nnnnnnn"])
    def test_other_placeholders(self, d):
        assert ms.is_placeholder_doi(d)

    @pytest.mark.parametrize("d", [
        "10.1126/sciadv.aax9249", "10.1038/s41586-024-10493-9",
        "10.48550/arXiv.2109.10086", "10.18653/v1/2022.findings-naacl.6",
        "", None,
    ])
    def test_real_dois_are_not_placeholders(self, d):
        assert not ms.is_placeholder_doi(d)


class TestEmptyCandidateAuthors:
    """A record with no author list is unjudgeable on author, not failed.

    Crossref returns exactly this for some ACL Anthology deposits, and treating
    it as a failure reported `ai/wadden-2022-multivers-…` as a wrong DOI when
    Crossref's own title for that DOI was the paper's.
    """
    T = "MULTIVERS: Improving scientific claim verification with weak supervision"

    def test_no_authors_but_matching_title_passes(self):
        assert ms.sanity_ok(2022, [], self.T, "Wadden", 2022, self.T)

    def test_no_authors_and_wrong_title_fails(self):
        why = ms.reject_reason(2022, [], "Something else entirely", "Wadden", 2022, self.T)
        assert why and "no candidate authors" in why

    def test_no_authors_and_no_title_refuses(self):
        why = ms.reject_reason(2022, [], "", "Wadden", 2022, self.T)
        assert why == "candidate record has neither authors nor a comparable title"

    def test_blank_author_strings_count_as_no_authors(self):
        assert ms.sanity_ok(2022, ["", "  "], self.T, "Wadden", 2022, self.T)


class TestMalformedAuthorsField:
    def test_et_al_is_reported_as_a_page_defect_not_a_doi_mismatch(self):
        why = ms.reject_reason(2018, ["Guohui Chuai"], "DeepCRISPR", "al", 2018, "DeepCRISPR")
        assert why and "malformed" in why
