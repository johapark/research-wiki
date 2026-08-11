"""Bibliographic export parsing (`refimport.parse`).

Every quirk asserted here was observed in a real 532-item ReadCube library
exported as both RIS and BibTeX. `tests/refimport-fixtures/README.md` maps each
fixture record to the quirk it covers and to how often that quirk occurred.

The fixtures are synthetic: `wiki/` and `papers/` are gitignored because this
repo is public and a personal corpus is not, so excerpting real records into
`tests/` would commit part of one.
"""

import json
from pathlib import Path

import pytest

from researchwiki.refimport.parse import (
    ExportItem,
    parse_bibtex,
    parse_csl_json,
    parse_export,
    parse_ris,
    sniff_format,
)
from researchwiki.stems import derive_stem

FIXTURES = Path(__file__).parent / "refimport-fixtures"
RIS = FIXTURES / "readcube-sample.ris"
BIB = FIXTURES / "readcube-sample.bib"
CSL = FIXTURES / "zotero-sample.json"


def _by_title(items, fragment):
    hits = [i for i in items if i.title and fragment.lower() in i.title.lower()]
    assert hits, f"no fixture record matching {fragment!r}"
    return hits[0]


# ---------- format sniffing ----------

@pytest.mark.parametrize("path,expected", [
    (RIS, "ris"), (BIB, "bibtex"), (CSL, "csl-json"),
])
def test_sniff_identifies_each_fixture(path, expected):
    assert sniff_format(path.read_text(encoding="utf-8-sig")) == expected


def test_sniff_ignores_the_file_extension():
    """Users rename exports and some tools write `.txt` by default, so the
    extension is the least reliable signal available."""
    assert sniff_format("TY  - JOUR\r\nTI  - x\r\nER  - \r\n") == "ris"
    assert sniff_format("@article{k, title = {x}}") == "bibtex"


@pytest.mark.parametrize("junk", ["", "   ", "hello world", "<?xml version='1.0'?>"])
def test_sniff_returns_none_for_unidentifiable_content(junk):
    assert sniff_format(junk) is None


def test_parse_export_raises_valueerror_on_unknown_format(tmp_path):
    """ValueError, not a crash: the caller maps it onto exit 1 (bad argument),
    not exit 2 (broken environment)."""
    p = tmp_path / "notes.txt"
    p.write_text("just some prose")
    with pytest.raises(ValueError, match="cannot identify"):
        parse_export(p)


# ---------- RIS ----------

def test_ris_parses_every_record_despite_crlf():
    """The fixture is CRLF, as ReadCube writes. Splitting on a literal "\\r\\n"
    after a text-mode read returns ONE record — a silent whole-file failure."""
    items = parse_ris(RIS.read_text(encoding="utf-8-sig"))
    assert len(items) == 12


def test_ris_four_character_pmid_tag_does_not_break_the_record():
    """RIS tags are conventionally 2 chars; ReadCube emits `PMID`. A
    fixed-width parser mis-slices the line or drops the record."""
    item = _by_title(parse_ris(RIS.read_text(encoding="utf-8-sig")), "draft synthetic")
    assert item.raw.get("PMID") == "30000001"
    assert item.doi == "10.1234/jtg.2023.0001"


def test_ris_junk_xx_tag_is_ignored_not_fatal():
    """385 of 532 real records carry an always-empty `XX  - `."""
    items = parse_ris(RIS.read_text(encoding="utf-8-sig"))
    assert _by_title(items, "draft synthetic").title == "A draft synthetic pangenome reference"


def test_ris_wrapped_continuation_line_is_joined():
    """Dropping the continuation truncates the title, which silently changes
    the derived stem — the failure this whole feature exists to prevent."""
    item = _by_title(parse_ris(RIS.read_text(encoding="utf-8-sig")), "very long title")
    assert item.title == ("A very long title that some exporters wrap across "
                          "two physical lines mid-sentence")


def test_ris_declared_file_path_is_captured():
    item = _by_title(parse_ris(RIS.read_text(encoding="utf-8-sig")), "attachment path")
    assert item.declared_files == ["files/declared-paper.pdf"]


def test_ris_http_urls_are_not_mistaken_for_file_paths():
    """The ELEC record's only `UR` is a chrome-extension URL. A URL is not an
    attachment path, and treating one as such makes rung 1 match nothing."""
    item = _by_title(parse_ris(RIS.read_text(encoding="utf-8-sig")), "Rapid DNA unwinding")
    assert item.declared_files == []


def test_ris_authors_are_flipped_to_display_order():
    item = _by_title(parse_ris(RIS.read_text(encoding="utf-8-sig")), "draft synthetic")
    assert item.authors == ["Ada L. Fixture", "Brian Second"]


def test_ris_record_without_er_is_still_parsed():
    """An exporter that truncates its last record should cost that record's
    tail, not the record."""
    items = parse_ris("TY  - JOUR\nTI  - Dangling record\nPY  - 2024\n")
    assert len(items) == 1 and items[0].title == "Dangling record"


def test_ris_empty_input_yields_no_items_and_no_error():
    assert parse_ris("") == []


# ---------- BibTeX ----------

def test_bibtex_parses_every_record():
    assert len(parse_bibtex(BIB.read_text(encoding="utf-8-sig"))) == 12


def test_bibtex_citekey_with_colon_is_accepted():
    """55 of 532 real citekeys contain `:`, which strict BibTeX forbids — a
    validating parser rejects the entire file."""
    keys = [i.key for i in parse_bibtex(BIB.read_text(encoding="utf-8-sig"))]
    assert any(":" in k for k in keys)


def test_bibtex_citekey_with_non_ascii_is_accepted():
    """16 of 532 real citekeys, e.g. `störtz2023picrispr:-dba`."""
    keys = [i.key for i in parse_bibtex(BIB.read_text(encoding="utf-8-sig"))]
    assert any(any(ord(c) > 127 for c in k) for k in keys)


def test_bibtex_brace_protected_word_does_not_end_the_field_early():
    """`{CRISPR}` nests inside the title value. A non-greedy `\\{.*?\\}` would
    end the entry at the first inner close brace."""
    item = _by_title(parse_bibtex(BIB.read_text(encoding="utf-8-sig")), "off-target assessment")
    assert item.title == "Transcriptome-wide off-target assessment in CRISPR test cells with elan"


def test_bibtex_file_field_triple_is_split_to_the_path():
    """Better BibTeX writes `description:path:mimetype`."""
    item = _by_title(parse_bibtex(BIB.read_text(encoding="utf-8-sig")), "attachment path")
    assert item.declared_files == ["files/declared-paper.pdf"]


@pytest.mark.parametrize("value,expected", [
    ("Full Text PDF:files/a.pdf:application/pdf", ["files/a.pdf"]),
    ("files/a.pdf", ["files/a.pdf"]),
    ("PDF:files/a.pdf:application/pdf;PDF:files/b.pdf:application/pdf",
     ["files/a.pdf", "files/b.pdf"]),
])
def test_bibtex_file_field_shapes(value, expected):
    items = parse_bibtex("@article{k,\n title = {T},\n file = {%s}\n}" % value)
    assert items[0].declared_files == expected


def test_bibtex_malformed_entry_does_not_take_down_the_file():
    """Skip what you cannot understand; keep what you can."""
    text = ("@article{good1, title = {First}}\n"
            "@article{broken, title = {Unclosed\n"
            "@article{good2, title = {Second}}\n")
    titles = [i.title for i in parse_bibtex(text)]
    assert "First" in titles


def test_bibtex_string_and_comment_entries_are_skipped():
    text = ('@string{jtg = "Journal of Test Genomics"}\n'
            "@comment{ignore me}\n"
            "@article{real, title = {Real}}\n")
    items = parse_bibtex(text)
    assert len(items) == 1 and items[0].title == "Real"


@pytest.mark.parametrize("raw", ['title = {Braced}', 'title = "Quoted"', 'year = 2024'])
def test_bibtex_value_delimiters(raw):
    items = parse_bibtex("@article{k, %s}" % raw)
    assert items and (items[0].title or items[0].year)


# ---------- cross-format agreement ----------

@pytest.mark.parametrize("fragment", [
    "draft synthetic", "ATAC", "epigenome-", "Homotrimer", "Circling back",
])
def test_ris_and_bibtex_agree_on_the_same_library(fragment):
    """The two real exports agreed field for field. If a parser change makes
    them disagree, one of the two is wrong and this says so immediately."""
    r = _by_title(parse_ris(RIS.read_text(encoding="utf-8-sig")), fragment)
    b = _by_title(parse_bibtex(BIB.read_text(encoding="utf-8-sig")), fragment)
    assert (r.title, r.year, r.doi) == (b.title, b.year, b.doi)
    assert r.authors == b.authors


# ---------- field normalization ----------

@pytest.mark.parametrize("raw,expected", [
    ("10.1234/abc", "10.1234/abc"),
    ("https://doi.org/10.1234/abc", "10.1234/abc"),
    ("http://dx.doi.org/10.1234/abc", "10.1234/abc"),
    ("doi:10.1234/abc", "10.1234/abc"),
    ("10.1234/ABC", "10.1234/abc"),
    ("10.1234/abc.", "10.1234/abc"),
])
def test_doi_is_unwrapped_and_normalized(raw, expected):
    items = parse_bibtex("@article{k, title = {T}, doi = {%s}}" % raw)
    assert items[0].doi == expected


@pytest.mark.parametrize("bad", ["", "not-a-doi", "10.x/abc", "12.1234/abc", "10.1234"])
def test_unusable_doi_becomes_none_rather_than_being_passed_on(bad):
    """`--doi` with a malformed value costs a failed lookup. Better to have no
    DOI than a broken one."""
    items = parse_bibtex("@article{k, title = {T}, doi = {%s}}" % bad)
    assert items[0].doi is None
    assert items[0].has_usable_doi is False


@pytest.mark.parametrize("raw,expected", [
    ("2015", 2015), ("2015/03/01", 2015), ("c2015", 2015),
    ("in press", None), ("", None), ("2015-2016", 2015),
])
def test_year_extraction(raw, expected):
    items = parse_bibtex("@article{k, title = {T}, year = {%s}}" % raw)
    assert items[0].year == expected


def test_page_range_is_not_mistaken_for_a_year():
    items = parse_bibtex("@article{k, title = {T}, year = {1473--1475}}")
    assert items[0].year is None


# ---------- item types ----------

def test_readcube_types_everything_as_an_article_including_a_book():
    """531 of 532 real records are JOUR/@article, books included. This is why
    the typed non-paper gate is kept but never relied on."""
    items = parse_ris(RIS.read_text(encoding="utf-8-sig"))
    book = _by_title(items, "Packt.Testing")
    assert book.item_type == "article"


def test_csl_json_types_are_populated_and_usable():
    """Zotero's CSL-JSON does carry a real type — the one format where the
    typed gate actually fires."""
    items = parse_csl_json(CSL.read_text(encoding="utf-8-sig"))
    types = {i.title: i.item_type for i in items}
    assert types["The Test Framework Manual"] == "book"
    assert types["A blog post about testing"] == "webpage"
    assert types["A draft synthetic pangenome reference"] == "article"


def test_biorxiv_doi_prefix_promotes_an_item_to_preprint():
    """The preprint/published dedupe needs to know which row is the preprint,
    and the DOI prefix says so even when the type field doesn't."""
    items = parse_ris(RIS.read_text(encoding="utf-8-sig"))
    pre = [i for i in items if i.doi and i.doi.startswith("10.1101/")]
    assert pre and all(i.item_type == "preprint" and i.is_preprint_doi for i in pre)


def test_misc_entry_with_eprint_is_a_preprint():
    items = parse_bibtex("@misc{k, title = {T}, eprint = {2401.00001}}")
    assert items[0].item_type == "preprint"


# ---------- CSL-JSON ----------

def test_csl_json_authors_and_year():
    item = _by_title(parse_csl_json(CSL.read_text(encoding="utf-8-sig")), "draft synthetic")
    assert item.authors == ["Ada L. Fixture", "Brian Second"]
    assert item.year == 2023


def test_csl_json_carries_no_attachment_paths():
    """True of every CSL-JSON exporter seen. Pairing must fall through to the
    content-based rungs for this format."""
    items = parse_csl_json(CSL.read_text(encoding="utf-8-sig"))
    assert all(i.declared_files == [] for i in items)


def test_csl_json_invalid_json_raises_valueerror(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text('[{"id": "a", ')
    with pytest.raises(ValueError, match="does not parse"):
        parse_export(p)


# ---------- end-to-end, and the reason any of this matters ----------

def test_parse_export_returns_format_and_items():
    fmt, items = parse_export(RIS)
    assert fmt == "ris" and len(items) == 12


def test_unicode_dash_title_reaches_the_corrected_stem():
    """The whole point of the feature: an export-derived stem must equal the
    one the PDF path derives. Guards the `stems.py` dash fold from here too."""
    item = _by_title(parse_ris(RIS.read_text(encoding="utf-8-sig")), "ATAC")
    assert derive_stem(item.authors, item.year, item.title) == \
        "hyphen-2015-atac-seq-a-method-for-assaying"


def test_suspended_compound_title_reaches_a_clean_stem():
    item = _by_title(parse_bibtex(BIB.read_text(encoding="utf-8-sig")), "epigenome-")
    stem = derive_stem(item.authors, item.year, item.title)
    assert stem == "dangling-2017-controlling-bias-and-inflation-in-epigenome"
    assert "--" not in stem and not stem.endswith("-")


def test_item_without_metadata_still_yields_an_item():
    """5 of 532 real records have no DOI, author or year. They are a triage
    decision, never a parse failure."""
    item = _by_title(parse_ris(RIS.read_text(encoding="utf-8-sig")), "Packt.Testing")
    assert isinstance(item, ExportItem)
    assert item.doi is None and item.year is None and item.authors == []


# ---------- CSL `issued` shapes (regression) ----------

@pytest.mark.parametrize("issued,expected", [
    ({"date-parts": [[2015, 3, 1]]}, 2015),
    ({"date-parts": [[2015]]}, 2015),
    ({"raw": "2015"}, 2015),
    ({"literal": "c2015"}, 2015),
    ("2015", 2015),
    (2015, 2015),
    ({}, None),
    ({"date-parts": []}, None),
    (None, None),
    ([], None),
])
def test_csl_issued_shapes(issued, expected):
    """The spec says a mapping with `date-parts`; exporters also emit a bare
    string and the `{"raw": ...}` form. Assuming the mapping raised
    AttributeError, which escaped `parse_export` as exit 3 — against a module
    whose rule is that one bad record never takes down the file."""
    rec = {"id": "x", "type": "article-journal", "title": "T"}
    if issued is not None:
        rec["issued"] = issued
    items = parse_csl_json(json.dumps([rec]))
    assert items[0].year == expected


def test_a_malformed_csl_record_does_not_take_down_the_file():
    items = parse_csl_json(json.dumps([
        {"id": "bad", "type": "article-journal", "title": "Bad", "issued": "nonsense"},
        {"id": "good", "type": "article-journal", "title": "Good",
         "issued": {"date-parts": [[2020]]}},
    ]))
    assert [i.title for i in items] == ["Bad", "Good"]
    assert items[1].year == 2020
