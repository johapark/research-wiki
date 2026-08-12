"""Author-name parsing (`researchwiki.names`).

The module exists because three callers were answering the same question
separately. `tests/test_stems.py` already pins the surname-boundary rules through
`first_author_surname`, which is now a wrapper over `surname_span`; what this file
covers is the surface only the exporter uses, plus the invariant that the two
callers cannot drift apart.

The extraction was additionally verified against every author string in the real
corpus (432 distinct values) with byte-identical output before and after. That
check is not committed as a fixture: it is corpus content, and `wiki/` is
gitignored in full for exactly that reason.
"""

import pytest

from researchwiki.names import (
    as_family_given,
    is_consortium,
    looks_like_prose,
    split_author_field,
    strip_et_al,
    surname_span,
)
from researchwiki.stems import first_author_surname


# ---------- the shared boundary ----------

@pytest.mark.parametrize("byline,family", [
    ("Sangsu Bae", "Bae"),
    ("Wen-Wei Liao", "Liao"),                 # hyphenated *given* name
    ("S. De Winter", "De Winter"),            # particle belongs to the surname
    ("Laurens Van Den Berg", "Van Den Berg"), # stacked particles
    ("Bin Liu", "Liu"),                       # `bin` is a given name here
    ("Di Liu", "Liu"),
    ("Teven Le Scao", "Le Scao"),
])
def test_surname_span_agrees_with_the_stem_it_produces(byline, family):
    """The invariant the shared helper exists to enforce: if these diverge, a
    stem and a bibliography entry disagree about who wrote the paper — which
    nobody notices until a manuscript reviewer does."""
    tokens = byline.split()
    assert " ".join(tokens[surname_span(tokens):]) == family
    # …and the stem is that same span, folded and slugged.
    assert first_author_surname([byline]) == family.lower().replace(" ", "-")


# ---------- splitting the frontmatter field ----------

def test_comma_is_the_default_delimiter():
    assert split_author_field("Akari Asai, Zeqiu Wu, Yizhong Wang") == \
        ["Akari Asai", "Zeqiu Wu", "Yizhong Wang"]


def test_semicolon_wins_when_present():
    """Three real pages use it, and it is unambiguous where a comma is not."""
    assert split_author_field("Mikhail Burtsev; Yang-Hui He; Evgeny Sobko") == \
        ["Mikhail Burtsev", "Yang-Hui He", "Evgeny Sobko"]


def test_a_yaml_list_is_already_split():
    assert split_author_field(["Ada Fixture", "Brian Second"]) == \
        ["Ada Fixture", "Brian Second"]


def test_et_al_is_stripped_not_treated_as_a_name():
    """Left in, the surname walk returns `al`, which matches no real author —
    four pages recorded that way were reported as wrong-DOI mismatches when the
    DOIs were fine."""
    assert split_author_field("Guohui Chuai et al.") == ["Guohui Chuai"]
    assert split_author_field("Ada Fixture, et al") == ["Ada Fixture"]
    assert strip_et_al("Smith and others") == "Smith"


def test_empty_and_blank_yield_nothing():
    assert split_author_field("") == []
    assert split_author_field(None) == []
    assert split_author_field("   ") == []


# ---------- prose is refused, not split ----------

def test_a_prose_byline_yields_no_authors():
    """The real case, from `anthropic-2026-paving-the-way-for-agents`. Splitting
    on commas invents ten authors, one of them named `Based on research by
    Ferdous Nasri`. Emitting nothing is correct: every entry type that carries a
    byline like this (`@techreport`, `@misc`) permits an absent author."""
    prose = ("Laura Luebbert (Anthropic Science). Based on research by "
             "Ferdous Nasri, Sarah Gurev, Patrick Varilly")
    assert split_author_field(prose) == []
    assert looks_like_prose(prose)


@pytest.mark.parametrize("byline", [
    "Anthropic (enterprise team)",
    "Brianna (Anthropic discovery team)",
])
def test_parenthetical_bylines_are_prose(byline):
    assert looks_like_prose(byline)


@pytest.mark.parametrize("byline", [
    "A. van der Graaf",          # period + lowercase *particle*
    "Abner Fernandes da Silva",
    "Allan dos Santos Costa",
    "Aäron van den Oord",
])
def test_particle_names_are_not_mistaken_for_prose(byline):
    """The tempting rule — 'a period followed by a lowercase word' — false-
    positives on four real pages where that lowercase word is a nobiliary
    particle. Measured, hence this test."""
    assert not looks_like_prose(byline)
    assert split_author_field(byline) == [byline]


# ---------- family/given, and declining to guess ----------

@pytest.mark.parametrize("raw,expected", [
    ("Sangsu Bae", ("Bae", "Sangsu")),
    ("V. Mirrokni", ("Mirrokni", "V.")),                  # initial kept verbatim
    ("Christopher Ré", ("Ré", "Christopher")),            # diacritic survives
    ("S. De Winter", ("De Winter", "S.")),                # particle in the family
    ("A. van der Graaf", ("van der Graaf", "A.")),
    ("Smith, John", ("Smith", "John")),                   # comma states the boundary
    ("van der Graaf, A.", ("van der Graaf", "A.")),
])
def test_family_given_split(raw, expected):
    assert as_family_given(raw) == expected


@pytest.mark.parametrize("raw", [
    "DeepSeek-AI",                        # corporate, single token
    "1000 Genomes Project Consortium",    # consortium
    "The ENCODE Project Consortium",
    "Abner Fernandes da Silva Junior X",  # >4 tokens, no particle to anchor on
])
def test_declines_to_split_when_it_cannot_be_sure(raw):
    """None means 'emit CSL `literal`', which is a faithful record of a name with
    no given/family structure — not a fallback. BibTeX and RIS never reach this
    code at all, since they parse `First von Last` themselves."""
    assert as_family_given(raw) is None


def test_consortium_detection():
    assert is_consortium("1000 Genomes Project")
    assert not is_consortium("Sangsu Bae")


def test_diacritics_are_never_folded_in_output():
    """86 corpus author fields carry non-ASCII. Folding is a stem concern; a
    bibliography has to show the reader the name as printed."""
    family, given = as_family_given("Aäron van den Oord")
    assert family == "van den Oord" and given == "Aäron"
