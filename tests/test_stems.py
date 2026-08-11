"""Canonical-stem derivation rules (CLAUDE.md "File Naming Convention").

These rules are intricate (stop-word extension, colon skipping, diacritic
stripping, hyphen preservation, the 5-word window) and break silently if a
refactor nudges them. The examples here mirror the ones in CLAUDE.md so the
table and the code can't drift apart unnoticed.
"""

import re

import pytest

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


def test_surname_nobiliary_particle_kept():
    """`surname as printed on p.1` includes the tussenvoegsel. Regression: the
    trailing-token rule gave `winter` for De Winter while the corpus already
    had `van-kempen-2024-...`, so the convention contradicted itself."""
    assert first_author_surname(["S. De Winter"]) == "de-winter"
    assert first_author_surname(["Seppe De Winter"]) == "de-winter"
    assert first_author_surname(["Michel van Kempen"]) == "van-kempen"
    assert first_author_surname(["Florestan De Moor"]) == "de-moor"
    assert first_author_surname(["Teven Le Scao"]) == "le-scao"


def test_surname_stacked_particles_kept():
    assert first_author_surname(["Laurens Van Den Berg"]) == "van-den-berg"


def test_surname_two_token_byline_is_never_particle_split():
    """`bin` and `di` are particles in Arabic and Italian names but ordinary
    given names in Chinese ones. A two-token byline is given name + surname, so
    the walk-left must not consume the first token."""
    assert first_author_surname(["Bin Liu"]) == "liu"
    assert first_author_surname(["Di Liu"]) == "liu"
    assert first_author_surname(["Da Chen"]) == "chen"


def test_surname_empty_list_is_unknown():
    assert first_author_surname([]) == "unknown"


def test_surname_non_decomposable_letters_transliterate():
    """NFKD folds `í` to `i`, but has nothing to say about `ł`, `ø` or `đ`.

    Those carry the stroke inside the codepoint, so they survived NFKD and were
    then deleted by the ASCII-only pass — removing a letter from the middle of
    a name. Observed 2026-08-04: Szałata was ingested as `szaata-2024-…`.
    """
    assert first_author_surname(["Artur Szałata"]) == "szalata"
    assert first_author_surname(["Kari Løken"]) == "loken"
    # Leading letter: dropping this one is the most damaging case, since it is
    # what a reader (and an alphabetical listing) scans for.
    assert first_author_surname(["Ivan Đurić"]) == "duric"


def test_surname_ligatures_expand_to_two_letters():
    assert first_author_surname(["Hans Straße"]) == "strasse"
    assert first_author_surname(["Æsa Þórsdóttir"]) == "thorsdottir"


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


# ---------- Unicode dashes (regression, 2026-08-11) ----------
#
# Publisher-set titles use U+2010 and friends freely. `slugify_phrase` folded
# them from the start; `normalize_title_word` did not, so the same paper
# derived two different stems depending on whether its metadata came from the
# PDF (ASCII hyphen) or from a reference-manager export (U+2010). Measured on a
# real 532-item library: 15 stems differed. The fold now lives in
# `strip_diacritics`, which both paths share.

@pytest.mark.parametrize("dash,name", [
    ("‐", "HYPHEN"),
    ("‑", "NON-BREAKING HYPHEN"),   # NFKD-decomposes to U+2010 first
    ("‒", "FIGURE DASH"),
    ("–", "EN DASH"),
    ("—", "EM DASH"),
])
def test_title_unicode_dashes_fold_to_ascii_hyphen(dash, name):
    assert derive_title_part(f"ATAC{dash}seq a method for assaying") == \
        "atac-seq-a-method-for-assaying", f"{name} was not folded"


def test_unicode_dash_and_ascii_hyphen_give_the_same_stem():
    """The whole point: one paper, two metadata sources, one stem."""
    from_pdf = derive_stem(["Jason D. Buenrostro"], 2015,
                           "ATAC-seq: A Method for Assaying Chromatin")
    from_export = derive_stem(["Buenrostro, Jason D."], 2015,
                              "ATAC‐seq: A Method for Assaying Chromatin")
    assert from_pdf == from_export == \
        "buenrostro-2015-atac-seq-a-method-for-assaying"


def test_hyphenated_surname_with_unicode_dash_is_kept():
    assert first_author_surname(["García‐López"]) == "garcia-lopez"


def test_minus_sign_is_not_a_dash():
    """U+2212 is category Sm — a mathematical operator, not punctuation. Titles
    use it as one (`CD4− cells`), so it is deleted like other symbols rather
    than folded into a word boundary."""
    assert derive_title_part("CD4− cells respond to antigen") == \
        "cd4-cells-respond-to-antigen"


def test_nbsp_separates_words():
    assert derive_title_part("a draft human pangenome reference") == \
        "a-draft-human-pangenome-reference"


# ---------- dangling hyphens (regression, 2026-08-11) ----------
#
# Suspended compounds ("epigenome- and transcriptome-wide") left a trailing `-`
# on the word, which the join carried into the stem. 3 of 516 real records.

def test_suspended_compound_does_not_leave_a_trailing_hyphen():
    assert derive_title_part(
        "Controlling bias and inflation in epigenome- and transcriptome-wide"
    ) == "controlling-bias-and-inflation-in-epigenome"


def test_suspended_compound_does_not_double_the_separator():
    assert derive_title_part(
        "Rapid, accurate long- and short-read mapping to pangenome graphs"
    ) == "rapid-accurate-long-and-short-read"


def test_interior_hyphen_still_survives():
    """The repair must not flatten legitimate hyphenated terms — that rule
    (`Cas-OFFinder` is one word) is what the edge-strip could easily break."""
    assert derive_title_part("Cas-OFFinder a fast and versatile") == \
        "cas-offinder-a-fast-and-versatile"


@pytest.mark.parametrize("title", [
    "epigenome- and transcriptome-wide association studies",
    "long- and short-read mapping",
    "ATAC‐seq: a method",
    "— leading dash",
    "trailing dash –",
    "Evo 2",
    "",
])
def test_stem_never_has_an_edge_or_doubled_separator(title):
    """The invariant both fixes exist to hold, stated once."""
    stem = derive_stem(["Ann Author"], 2024, title)
    assert "--" not in stem
    assert not stem.endswith("-") and not stem.startswith("-")
    assert re.fullmatch(r"[a-z0-9][-a-z0-9]*", stem), stem


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


def test_slugify_transliterates_what_nfkd_cannot():
    """Same failure as the surname path — page slugs must agree with stems."""
    assert slugify_phrase("Szałata transformers review") == \
        "szalata-transformers-review"
    assert slugify_phrase("Đurić Løken methods") == "duric-loken-methods"


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
