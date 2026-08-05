"""`grade synthesis` must route claim-anchor citations to their paper's PDF.

CLAUDE.md's recommended claim-level citation form is `[[stem#claim_slug]]`.
`_wikilink_to_stem` stripped `|alias` and the category prefix but not the
`#fragment`, so the stem came back as `stem#kc-9f3a2b1c`, `resolve_pdf` raised
FileNotFoundError, and the unit was recorded `uncited` and skipped.

Why this is worth a test rather than a one-line fix and move on: the gate still
exited 0. A page citing entirely by claim anchor — following the documented
practice — reported success while almost nothing had been verified. Measured
2026-08-05 on a 22-unit synthesis page: 3 units graded before the fix, 22 after,
against 35-43 on comparable footnote-cited pages. A gate that passes without
checking is worse than one that fails.

`_extract_anchor_pairs` already stripped the fragment, so the two halves of the
module disagreed about the same citation form — which is the shape of bug a
round-trip test catches and a spot check does not.
"""
from __future__ import annotations

import pytest

from researchwiki.grade.fidelity.synthesis import (
    _extract_anchor_pairs,
    _wikilink_to_stem,
)

_STEM = "nagai-2026-toward-generalizable-and-interpretable-ai"


@pytest.mark.parametrize("link", [
    _STEM,                                  # bare
    f"compbio/{_STEM}",                     # category-prefixed
    f"{_STEM}#kc-578ed7f4",                 # claim anchor, the documented form
    f"compbio/{_STEM}#kc-578ed7f4",         # both
    f"compbio/{_STEM}|Nagai 2026",          # aliased
    f"  compbio/{_STEM}#kc-578ed7f4  ",     # whitespace
])
def test_every_citation_form_yields_the_bare_stem(link):
    """All five forms must resolve identically — `resolve_pdf` takes a stem."""
    assert _wikilink_to_stem(link) == _STEM


def test_fragment_is_stripped_not_merely_tolerated():
    """The regression itself: a `#slug` must never survive into the stem."""
    assert "#" not in _wikilink_to_stem(f"{_STEM}#kc-578ed7f4")


def test_alias_containing_a_hash_does_not_confuse_the_split():
    """`|` is split first, so a `#` inside an alias can't leak into the stem."""
    assert _wikilink_to_stem(f"compbio/{_STEM}|see #3") == _STEM


def test_stem_extraction_agrees_with_anchor_extraction():
    """The two halves of the module must read one citation the same way.

    `_extract_anchor_pairs` was already fragment-aware; `_wikilink_to_stem` was
    not, so identity-checking and PDF-routing disagreed and only the silent one
    (routing) failed.
    """
    unit = f"Some claim.[[{_STEM}#kc-578ed7f4]]"
    (anchor_stem, slug), = _extract_anchor_pairs(unit)
    assert slug == "kc-578ed7f4"
    assert anchor_stem == _wikilink_to_stem(f"{_STEM}#kc-578ed7f4")


# ---------- space-separated thousands ----------
#
# `NUMERIC_TOKEN_RE` allows `,` inside a number but not whitespace, so a PDF
# typesetting "1 in 300 000" (ISO 31-0 / SI style, common in European journals,
# often with a non-breaking space) tokenized as two numbers and a page citing
# "300,000" was reported as numeric drift against a paper stating it plainly.
# Rounding is deliberately NOT tolerated here — `check_numerics`' own docstring
# says a rounded "510,000" must not match "510,495" — so these tests pin the
# boundary between "same value, different typesetting" and "different value".

from researchwiki.grade.primitives import (          # noqa: E402
    check_numerics, collapse_spaced_thousands,
)


@pytest.mark.parametrize("text,expected", [
    ("1 in 300 000.", "1 in 300000."),
    ("1 158 017 individuals", "1158017 individuals"),
    ("1 000 genomes", "1000 genomes"),          # non-breaking space
    ("8 600 000 reads", "8600000 reads"),  # narrow no-break space
    # A 1-3 digit lead is valid, so "22 333" is 22,333 and joins; the stray
    # leading "1" is unrelated text and stays where it is.
    ("1 22 333", "1 22333"),
])
def test_spaced_thousands_are_collapsed(text, expected):
    assert collapse_spaced_thousands(text) == expected


@pytest.mark.parametrize("text", [
    "in 2024 the cohort",        # a 4-digit year must not seed a match
    "8 heads and 128 samples",   # unrelated small numbers stay apart
    "300,000 already",           # comma form untouched
])
def test_unrelated_numbers_are_not_glued(text):
    assert collapse_spaced_thousands(text) == text


def test_collapse_is_idempotent():
    once = collapse_spaced_thousands("1 158 017")
    assert collapse_spaced_thousands(once) == once


def test_page_comma_matches_pdf_space():
    _, unmatched = check_numerics("HoFH ~1 in 300,000", "", "prevalence is 1 in 300 000.")
    assert unmatched == []


def test_rounding_is_still_flagged():
    """The boundary: 0.893 must NOT match a paper's 0.8926."""
    _, unmatched = check_numerics("F1 of 0.893", "", "F1 reached 0.8926")
    assert unmatched == ["0.893"]


def test_trailing_zero_decimals_still_match():
    """Pre-existing tolerance must survive the new normalization."""
    _, unmatched = check_numerics("F1 of 0.778", "", "F1 was 0.7780")
    assert unmatched == []


def test_superscript_citation_does_not_glue_to_a_comma_grouped_number():
    """Regression: PDF extraction glues a citation marker to the preceding word.

    A table row extracts as "Borzoi (2023)73 524,000". Without the `(?!,\\d)`
    guard, "73" is a valid 1-3 digit lead and "524" a valid 3-digit group, so the
    two joined into "73524,000" and the real figure 524,000 left the evidence
    set — producing a false MISATTRIBUTED on a page whose number was verifiably
    in the cited PDF. A following ",<digit>" proves the group was already
    comma-delimited, so it cannot also be space-delimited.
    """
    s = "Borzoi (2023)73 524,000 bp"
    assert collapse_spaced_thousands(s) == s
    _, unmatched = check_numerics("receptive field of 524,000 bp", "", s)
    assert unmatched == []


def test_comma_as_punctuation_still_joins():
    """The guard must key on ',<digit>', not on any comma."""
    assert collapse_spaced_thousands("300 000, and more") == "300000, and more"
