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
    looks_inverted,
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


def test_a_parenthesized_preferred_name_is_kept():
    """`Xuefei (Julie) Wang` is a real author here, inside a 42-name byline. A
    parenthesis test would have thrown away all 42 — which is why length is the
    only prose signal."""
    field = "Eser Aygün, Xuefei (Julie) Wang, Lai Wei"
    assert split_author_field(field) == \
        ["Eser Aygün", "Xuefei (Julie) Wang", "Lai Wei"]


def test_a_long_author_list_is_ordinary():
    """The ceiling applies per name, not to the whole field. Applied to the field
    it rejected ~300 real pages, because five authors is already a dozen
    whitespace tokens."""
    field = ", ".join(f"Given{i} Family{i}" for i in range(42))
    assert len(split_author_field(field)) == 42


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
    "Anthropic (enterprise team)",        # `team` marks an organisation
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


# ---------- `Family, Given` vs the author-list delimiter ----------
#
# A comma means two different things: it separates `Family, Given` in a
# bibliographic export, and it separates *authors* in this wiki's `authors:`
# field. `van der Graaf, A.` is one person; `Akari Asai, Zeqiu Wu` is two.


@pytest.mark.parametrize("raw", [
    "van der Graaf, A.",        # particles + surname, then an initial
    "De Winter, S.",
    "van den Berg, L.",
    "Smith, John",              # bare surname, then one given name
    "Liu, Bin",
])
def test_inverted_single_names_are_recognized(raw):
    assert looks_inverted(raw)


@pytest.mark.parametrize("raw", [
    "Akari Asai, Zeqiu Wu",              # two authors
    "A. Backhaus, J. Quiroz-Chávez",     # two authors, initialed
    "Di Liu, Bin Wang",                  # leading given name is a particle lookalike
    "Liu, Bin Wang",                     # `Liu` then a full name = two authors
    "Sangsu Bae",                        # no comma at all
])
def test_author_lists_are_not_read_as_inverted_names(raw):
    """Requiring only the surname-shape signal changes 349 first-author surnames
    on the real corpus, because a leading given name is so often an initial or a
    particle lookalike. Both signals are required."""
    assert not looks_inverted(raw)


@pytest.mark.parametrize("raw,surname", [
    ("van der Graaf, A.", "van-der-graaf"),
    ("De Winter, S.", "de-winter"),
    ("van den Berg, L.", "van-den-berg"),
])
def test_an_inverted_name_keeps_its_particle(raw, surname):
    """The bug this fixes: the pre-comma part is *already* the surname, so the
    particle walk must not run on it. Its floor exists to protect a leading given
    name, and applied here it stopped early and dropped the `van`."""
    assert first_author_surname([raw]) == surname


@pytest.mark.parametrize("raw,surname", [
    ("Akari Asai, Zeqiu Wu, Yizhong Wang", "asai"),
    ("A. Backhaus, J. Quiroz-Chávez", "backhaus"),
    ("Di Liu, Bin Wang", "liu"),
])
def test_a_whole_author_field_still_yields_the_first_surname(raw, surname):
    """Callers are supposed to split the byline first, but this used to be robust
    to being handed the whole field and staying robust is worth a test: the
    failure mode is a silent wrong filename."""
    assert first_author_surname([raw]) == surname


@pytest.mark.parametrize("raw", [
    "Akari Asai, Zeqiu Wu",
    "A. Backhaus, J. Quiroz-Chávez",
    "Di Liu, Bin Wang",
])
def test_an_author_list_is_never_split_into_one_persons_name(raw):
    """Both functions read the comma through `looks_inverted`, so neither can
    decide it inverts a name while the other decides it separates two. Without
    this, `as_family_given` reported a person named `Akari Asai` whose given name
    was `Zeqiu Wu` — the same defect as the stem bug, in the other direction.

    Declining is the honest answer: a comma that is not an inversion means the
    input is not the single name this function takes, so there is no boundary
    inside it to find."""
    assert as_family_given(raw) is None


def test_diacritics_are_never_folded_in_output():
    """86 corpus author fields carry non-ASCII. Folding is a stem concern; a
    bibliography has to show the reader the name as printed."""
    family, given = as_family_given("Aäron van den Oord")
    assert family == "van den Oord" and given == "Aäron"
