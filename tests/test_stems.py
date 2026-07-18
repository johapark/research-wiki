"""Canonical-stem derivation rules (CLAUDE.md "File Naming Convention").

These rules are intricate (stop-word extension, colon skipping, diacritic
stripping, hyphen preservation, the 5-word window) and break silently if a
refactor nudges them. The examples here mirror the ones in CLAUDE.md so the
table and the code can't drift apart unnoticed.
"""

from researchwiki.stems import (
    derive_stem,
    derive_title_part,
    first_author_surname,
    stem_author_year,
)


# ---------- first_author_surname ----------

def test_surname_simple():
    assert first_author_surname(["Sangsu Bae"]) == "bae"


def test_surname_hyphenated_given_name_dropped():
    # "Wen-Wei Liao" → the hyphenated given name is not the surname.
    assert first_author_surname(["Wen-Wei Liao"]) == "liao"


def test_surname_hyphenated_surname_kept():
    assert first_author_surname(["García-López"]) == "garcia-lopez"


def test_surname_diacritics_stripped():
    assert first_author_surname(["Jürgen Müller"]) == "muller"


def test_surname_empty_list_is_unknown():
    assert first_author_surname([]) == "unknown"


# ---------- derive_title_part ----------

def test_title_first_five_words():
    assert derive_title_part("A draft human pangenome reference") == \
        "a-draft-human-pangenome-reference"


def test_title_colon_is_skipped():
    # Colon dropped; counting continues through the subtitle.
    title = "Cas-OFFinder: a fast and versatile algorithm that searches"
    assert derive_title_part(title) == "cas-offinder-a-fast-and-versatile"


def test_title_trailing_stopword_extends():
    # First 5 end on "across" (stop word) → extend until a content word.
    title = "Genome modelling and design across all domains of life with Evo 2"
    assert derive_title_part(title) == "genome-modelling-and-design-across-all-domains"


def test_title_numbers_kept():
    assert derive_title_part("Evo 2") == "evo-2"


def test_title_hyphenated_term_is_one_word():
    # "Cas-OFFinder" counts as a single word, hyphen preserved.
    assert derive_title_part("Cas-OFFinder fast versatile algorithm searches sites") == \
        "cas-offinder-fast-versatile-algorithm-searches"


def test_title_empty_is_untitled():
    assert derive_title_part("") == "untitled"


# ---------- derive_stem ----------

def test_derive_stem_full_example():
    stem = derive_stem(
        ["Sangsu Bae"], 2014,
        "Cas-OFFinder: a fast and versatile algorithm that searches",
    )
    assert stem == "bae-2014-cas-offinder-a-fast-and-versatile"


def test_derive_stem_accepts_string_year():
    stem = derive_stem(["Wen-Wei Liao"], "2023", "A draft human pangenome reference")
    assert stem == "liao-2023-a-draft-human-pangenome-reference"


# ---------- stem_author_year ----------

def test_author_year_strips_title():
    assert stem_author_year("brixi-2026-genome-modelling-and-design") == "brixi-2026"


def test_author_year_hyphenated_surname():
    assert stem_author_year("garcia-lopez-2024-some-long-title") == "garcia-lopez-2024"


def test_author_year_bibtex_suffix():
    assert stem_author_year("smith-2024b-second-paper-this-year") == "smith-2024b"


def test_author_year_non_canonical_passthrough():
    assert stem_author_year("not-a-stem") == "not-a-stem"
