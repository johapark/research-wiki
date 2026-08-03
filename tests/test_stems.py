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
    slugify_phrase,
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


def test_surname_comma_order():
    # "Last, First" (bibliographic export order) → surname is before the comma.
    assert first_author_surname(["Liao, Wen-Wei"]) == "liao"
    assert first_author_surname(["García-López, Ana"]) == "garcia-lopez"


def test_surname_consortium_slugged_whole():
    # Consortium bylines slug the whole name per CLAUDE.md, not a trailing token.
    assert first_author_surname(["1000 Genomes Project"]) == "1000-genomes-project"
    assert first_author_surname(["The ENCODE Consortium"]) == "the-encode-consortium"


def test_surname_blank_string_is_unknown():
    assert first_author_surname([""]) == "unknown"
    assert first_author_surname(["   "]) == "unknown"


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


# ---------- slugify_phrase (shared by synthesize + concept scaffolding) ----------
#
# `tasks/synthesize._slugify` and `concepts.candidates._term_slug` are both thin
# aliases over this. They used to be two independent regexes, which is how the
# two bugs below survived: a scaffolded page's filename and the candidate edge
# pointing at it were computed by different code.

def test_slugify_folds_diacritics_to_ascii_base():
    # The pre-fix regex deleted the accented character outright, so an author's
    # name lost a letter: "garca". CLAUDE.md's naming rule is García → garcia,
    # and `first_author_surname` above already obeys it — page slugs now agree.
    assert slugify_phrase("García-López pangenome methods") == \
        "garcia-lopez-pangenome-methods"
    assert slugify_phrase("Jürgen Müller prime editing") == \
        "jurgen-muller-prime-editing"


def test_slugify_maps_unicode_dashes_to_ascii_hyphen():
    # Publisher-set titles use en/em dashes and U+2011 non-breaking hyphens.
    # Deleting them welded the neighbouring words together.
    assert slugify_phrase("k‑mers in assembly") == "k-mers-in-assembly"      # non-breaking hyphen
    assert slugify_phrase("CRISPR–Cas9 specificity") == "crispr-cas9-specificity"  # en dash
    assert slugify_phrase("Evo 2 — genome design") == "evo-2-genome-design"  # em dash


def test_slugify_deletes_other_punctuation_without_splitting():
    # Deliberately *deletes* rather than separator-replaces: this is what keeps
    # possessives and decimals reading naturally, and it's the behavior every
    # slug already on disk was generated with.
    assert slugify_phrase("Claude's view of PubTator 3.0") == "claudes-view-of-pubtator-30"


def test_slugify_collapses_runs_and_trims():
    assert slugify_phrase("  spaced   out --- title  ") == "spaced-out-title"
    assert slugify_phrase("") == ""
    assert slugify_phrase("!!!") == ""


def test_synthesize_and_concept_slugs_agree():
    """The invariant the shared helper exists to enforce. If these ever diverge,
    `researchwiki concepts "<term>"` writes a hub whose filename doesn't match
    the candidate edge that pointed at it, and the hub reads as uncreated."""
    from researchwiki.concepts.candidates import _term_slug
    from researchwiki.tasks.synthesize import _slugify

    for phrase in ["García-López pangenome", "CRISPR–Cas9", "k‑mers",
                   "Claude's PubTator 3.0", "plain english phrase"]:
        assert _slugify(phrase) == _term_slug(phrase) == slugify_phrase(phrase)
