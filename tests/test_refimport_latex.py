"""LaTeX unescaping for BibTeX values (`refimport.latex`).

Defensive rather than central: the real ReadCube BibTeX export contains zero
backslashes, `$` or braces in its titles. Zotero and JabRef do escape, so the
assertions that matter are the ones checking a title survives all the way to a
correct *stem* — a title is only ever wrong in a way that costs something once
it has renamed a file.
"""

import pytest

from researchwiki.refimport.latex import delatex
from researchwiki.stems import derive_stem, derive_title_part


@pytest.mark.parametrize("raw,expected", [
    (r"Gr{\"u}newald", "Grunewald"),
    (r"Gr\"unewald", "Grunewald"),
    (r"{\'e}lan vital", "elan vital"),
    (r"\'{e}lan", "elan"),
    (r"Se\~nor", "Senor"),
    (r"H{\aa}kan", "Håkan"),
    (r"Wei{\ss}", "Weiß"),
    (r"{\L}ukasz", "Łukasz"),
    (r"\O{}rsted", "Ørsted"),
])
def test_accents_and_letter_commands(raw, expected):
    assert delatex(raw) == expected


def test_letter_commands_resolve_through_stems_not_here():
    """`\\l` becomes `ł`, not `l`. `stems._TRANSLITERATE` is the single place
    this project decides how `ł` romanizes; duplicating that judgement here
    would let the two drift."""
    assert delatex(r"Sza{\l}ata") == "Szałata"
    assert derive_title_part(delatex(r"Sza{\l}ata studies")) == "szalata-studies"


@pytest.mark.parametrize("raw,expected", [
    (r"Cas9 \& Cas12a", "Cas9 & Cas12a"),
    (r"50\% efficiency", "50% efficiency"),
    (r"a\_b", "a_b"),
    (r"\#hashtag", "#hashtag"),
])
def test_escaped_literals_are_unescaped(raw, expected):
    assert delatex(raw) == expected


def test_brace_protection_is_removed_but_content_kept():
    assert delatex("{CRISPR} off-target effects") == "CRISPR off-target effects"
    assert delatex("The {DNA} of {RNA}") == "The DNA of RNA"


def test_math_greek_is_spelled_out_not_deleted():
    """`$\\alpha$-synuclein` must not stem as `-synuclein`."""
    assert delatex(r"$\alpha$-synuclein aggregation") == "alpha-synuclein aggregation"
    assert derive_title_part(delatex(r"$\alpha$-synuclein aggregation in cells")) == \
        "alpha-synuclein-aggregation-in-cells"


def test_math_superscripts_lose_their_markers():
    assert delatex(r"a $10^{-9}$ M binder") == "a 10-9 M binder"


def test_text_commands_become_punctuation():
    assert delatex(r"long\textendash read sequencing") == "long-read sequencing"
    assert delatex(r"the cell\textquoteright s fate") == "the cell's fate"


def test_tilde_is_a_space_not_a_character():
    assert delatex(r"Fig.~1 shows") == "Fig. 1 shows"


def test_unknown_command_keeps_its_name_rather_than_vanishing():
    """Deleting an unrecognized command can silently drop a real word. Keeping
    the name is wrong-but-visible, which is the better failure."""
    assert delatex(r"\textbf{Important} finding") == "textbf Important finding"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_empty_input_is_empty_output_not_an_error(value):
    """A missing title is a triage decision (`thin-metadata`), not a crash."""
    assert delatex(value) == ""


def test_plain_unicode_passes_through_untouched():
    """The real ReadCube export's actual shape: raw Unicode, no escapes. This
    module must not corrupt it — the dash fold belongs to `stems`."""
    assert delatex("ATAC‐seq: A Method for Assaying") == "ATAC‐seq: A Method for Assaying"


def test_escaped_title_and_unicode_title_reach_the_same_stem():
    """Two exporters, two spellings of one paper, one stem."""
    zotero = derive_stem(["Jonas Gr{\\\"u}newald"], 2018,
                         delatex(r"Transcriptome-wide off-target assessment"))
    readcube = derive_stem(["Jonas Grünewald"], 2018,
                           delatex("Transcriptome-wide off-target assessment"))
    assert zotero == readcube == "grunewald-2018-transcriptome-wide-off-target-assessment"
