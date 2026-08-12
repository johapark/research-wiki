"""Bibliographic export, pure logic (`researchwiki.refexport`).

Everything here works from strings and hand-built `Page` objects — no wiki on
disk. The end-to-end behaviour of the command lives in
`tests/test_export_phases.py`.

The load-bearing property is that our own importer can read back what we emit,
so the type table and the round-trip are asserted rather than assumed.
"""

import json

import pytest

from researchwiki.refexport import (
    ENTRY_TYPES,
    EXPORTABLE_TYPES,
    Record,
    _needs_protection,
    _PAGE_KIND,
    bibtex_value,
    latex_escape,
    render_bibtex,
    render_csl_json,
    render_ris,
)
from researchwiki.refimport.parse import (
    _BIBTEX_TYPE_MAP,
    _CSL_TYPE_MAP,
    _RIS_TYPE_MAP,
    parse_bibtex,
    parse_csl_json,
    parse_ris,
)


def rec(**kw) -> Record:
    base = dict(key="bae-2014-cas-offinder", kind="article",
                title="Cas-OFFinder: a fast algorithm",
                authors=["Sangsu Bae", "Jeongbin Park"],
                year=2014, doi="10.1093/bioinformatics/btu048",
                venue="Bioinformatics")
    base.update(kw)
    return Record(**base)


# ---------- the type tables ----------

def test_every_exportable_page_type_has_a_kind():
    """Adding a page type to one table and not the other would emit `misc` for it
    silently, or raise a KeyError. Neither is acceptable."""
    assert set(_PAGE_KIND) == set(EXPORTABLE_TYPES)


@pytest.mark.parametrize("kind", sorted(ENTRY_TYPES))
def test_every_outbound_type_is_one_our_own_parser_reads(kind):
    """The guard that keeps export and import from drifting apart. It fails the
    moment someone adds an outbound type `refimport` has no key for — which would
    silently break the round-trip."""
    bib, ris, csl = ENTRY_TYPES[kind]
    assert bib in _BIBTEX_TYPE_MAP
    assert ris in _RIS_TYPE_MAP
    assert csl in _CSL_TYPE_MAP


def test_non_bibliographic_types_are_not_exportable():
    """A synthesis page has no DOI, venue or year of record. An entry for one
    would assert a publication that does not exist."""
    for t in ("synthesis", "idea", "concept", "meta", "dashboard"):
        assert t not in EXPORTABLE_TYPES


# ---------- LaTeX escaping ----------

@pytest.mark.parametrize("raw,escaped", [
    ("50% efficiency", r"50\% efficiency"),
    ("Smith & Jones", r"Smith \& Jones"),
    ("cost $5", r"cost \$5"),
    ("issue #3", r"issue \#3"),
    ("snake_case", r"snake\_case"),
    ("{braced}", r"\{braced\}"),
    ("a~b", r"a\textasciitilde{}b"),
    ("x^2", r"x\textasciicircum{}2"),
])
def test_specials_are_escaped(raw, escaped):
    assert latex_escape(raw) == escaped


def test_a_backslash_does_not_have_its_own_replacement_re_escaped():
    """The `{}` in `\\textbackslash{}` is LaTeX syntax. A separate brace pass
    escapes it into `\\textbackslash\\{\\}`, which is why escaping is one pass."""
    assert latex_escape(r"a\b") == r"a\textbackslash{}b"


def test_unicode_survives_untouched():
    """86 corpus author fields carry non-ASCII. biber, XeLaTeX, Pandoc and Zotero
    all read UTF-8; a macro table would have to be perfect or it corrupts names."""
    assert latex_escape("Christopher Ré") == "Christopher Ré"
    assert latex_escape("CRISPR–Cas9") == "CRISPR–Cas9"     # en-dash preserved


def test_control_characters_and_nbsp_are_normalized():
    assert latex_escape("a\x07b") == "ab"
    assert latex_escape("a\xa0b") == "a b"


# ---------- brace protection ----------

@pytest.mark.parametrize("token", ["CRISPR", "DNA", "AI", "AlphaFold", "mRNA",
                                   "PLoS", "Cas9", "SARS-CoV-2", "CRISPR-Cas9"])
def test_tokens_a_style_would_destroy_are_protected(token):
    assert _needs_protection(token)


@pytest.mark.parametrize("token", ["machine", "learning", "A", "The", "Genome",
                                   "a", "of"])
def test_ordinary_words_are_not_protected(token):
    """A token whose only capital is initial is sentence case or a proper noun the
    style may legitimately re-case."""
    assert not _needs_protection(token)


def test_protection_wraps_the_whole_token_after_escaping():
    """Decide on the raw token, escape, then wrap. A brace mid-word breaks
    hyphenation, and deciding after escaping would test introduced braces."""
    assert bibtex_value("CRISPR-Cas9 editing", protect=True) == \
        "{CRISPR-Cas9} editing"
    # `100%` carries no letter a style could lowercase, so it needs no brace —
    # and the escape still happens.
    assert bibtex_value("100% of DNA", protect=True) == r"100\% of {DNA}"


def test_protection_is_off_by_default():
    assert bibtex_value("CRISPR") == "CRISPR"


# ---------- rendering, and reading it back ----------

def test_bibtex_joins_authors_with_and_and_does_not_invert_them():
    """BibTeX parses `First von Last` itself, so inverting here could only
    introduce errors on the 58 particle names and 76 four-token names."""
    out = render_bibtex([rec(authors=["A. van der Graaf", "Christopher Ré"])])
    assert "author = {A. van der Graaf and Christopher Ré}" in out


def test_bibtex_omits_absent_fields_rather_than_emitting_them_empty():
    """An empty `journal = {}` in a `.bib` is worse than no field."""
    out = render_bibtex([rec(kind="misc", venue=None, doi=None, year=None)])
    for absent in ("journal", "doi", "year", "volume", "pages", "publisher"):
        assert f"{absent} = " not in out


def test_a_preprint_carries_eprint_so_the_importer_recognizes_it():
    """`parse._normalize_type` promotes `other → preprint` only when `eprint` is
    present, so this field is what makes the type round-trip."""
    out = render_bibtex([rec(kind="preprint", venue=None, eprint="2606.15357")])
    assert "@misc{" in out
    assert "eprint = {2606.15357}" in out
    assert "archivePrefix = {arXiv}" in out
    assert parse_bibtex(out)[0].item_type == "preprint"


@pytest.mark.parametrize("fmt,render,parse", [
    ("bibtex", render_bibtex, parse_bibtex),
    ("ris", render_ris, parse_ris),
    ("csl-json", render_csl_json, parse_csl_json),
])
def test_round_trips_through_our_own_importer(fmt, render, parse):
    """Verified over the whole real corpus too (421 records, three formats, zero
    mismatches on title/authors/doi/venue/year); this pins it in the suite."""
    r = rec(authors=["A. van der Graaf", "Christopher Ré", "DeepSeek-AI"],
            title="CRISPR-Cas9 editing at 50% efficiency")
    (item,) = parse(render([r]))
    assert item.key == r.key
    assert item.title == r.title          # braces and escapes both undone
    assert item.authors == r.authors
    assert item.year == r.year
    assert item.doi == r.doi
    assert item.venue == r.venue


def test_csl_uses_literal_for_a_name_it_cannot_split():
    """`literal` is CSL's own construct for a name with no given/family
    structure — a faithful record, not a fallback."""
    items = json.loads(render_csl_json([rec(authors=["DeepSeek-AI", "Sangsu Bae"])]))
    assert items[0]["author"] == [
        {"literal": "DeepSeek-AI"},
        {"family": "Bae", "given": "Sangsu"},
    ]


def test_ris_emits_one_au_line_per_author():
    out = render_ris([rec(authors=["Sangsu Bae", "Jeongbin Park"])])
    assert out.count("AU  - ") == 2
    assert "ER  - " in out


def test_renderers_are_empty_for_no_records():
    assert render_bibtex([]) == ""
    assert render_ris([]) == ""
    assert json.loads(render_csl_json([])) == []
