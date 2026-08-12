"""Title-scoring behaviour, isolated from PDF generation.

Split from `test_refimport_pair.py` because these assert the *metric* rather
than the pairing pipeline, and the metric was wrong once already: symmetric F1
scored an exact title match at 0.615 against a real first page, because the page
carries authors and an abstract the title does not.
"""

import pytest

from researchwiki.refimport.pair import TITLE_ACCEPT, TITLE_FLOOR, _coverage, _tokens


def score(title: str, page_text: str) -> float:
    return _coverage(_tokens(title), _tokens(page_text))


def test_exact_title_on_a_realistic_page_scores_one():
    """The case F1 got wrong: everything after the title is page furniture and
    must not count against the match."""
    page = ("A draft synthetic pangenome reference\n"
            "Ada L. Fixture, Brian Second, Carol Hyphen\n"
            "Department of Testing, University of Examples\n"
            "Abstract Pangenome references capture variation that a single "
            "linear reference cannot represent.")
    assert score("A draft synthetic pangenome reference", page) == 1.0


def test_unrelated_title_scores_zero():
    assert score("Quantum chromodynamics on a lattice",
                 "A draft synthetic pangenome reference") == 0.0


def test_partial_title_lands_between_the_floor_and_accept():
    """A truncated or reworded title should be *reported* for a human, not
    silently accepted or silently dropped."""
    page = "Machine learning for protein structure and function"
    s = score("Machine learning for protein folding dynamics", page)
    assert TITLE_FLOOR <= s < TITLE_ACCEPT


@pytest.mark.parametrize("short_title", ["Evo 2", "GPT-4", "On AI"])
def test_short_titles_are_refused_rather_than_matched_by_chance(short_title):
    """Coverage of a 1-2 token title is satisfied by accident, and a chance
    match assigns the wrong PDF to a record — the most expensive error here."""
    page = "Evo 2 GPT-4 On AI and everything else in this unrelated document"
    assert score(short_title, page) == 0.0


def test_unicode_and_ascii_spellings_score_identically():
    """Folded through `stems.strip_diacritics`, so two spellings of one paper
    do not score as two papers."""
    page = "ATAC-seq: A Method for Assaying Chromatin Accessibility Genome-Wide"
    assert score("ATAC‐seq: A Method for Assaying Chromatin Accessibility Genome‐Wide",
                 page) == 1.0


def test_empty_inputs_score_zero_rather_than_raising():
    assert score("", "some text") == 0.0
    assert score("a real title with words", "") == 0.0
