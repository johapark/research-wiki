"""Type-tolerant Page frontmatter accessors.

These pin the contract that `str_field` / `list_field` / `year_int` read a
field identically whether `fm` holds legacy line-parser strings (every value a
str, lists as the literal "[a, b]") or real-YAML native types (list / int /
date). That tolerance is what lets `read_page` switch parsers underneath the
call sites without breaking them.
"""

from datetime import date
from pathlib import Path

from researchwiki.wiki import Page, read_page


def _page(fm: dict) -> Page:
    return Page(path=Path("x/y.md"), stem="y", category="x", fm=fm, body="")


# --- read_page (PyYAML parser, fm={} on malformed) ----------------------

def test_read_page_parses_native_yaml_types(tmp_path):
    md = tmp_path / "compbio" / "foo-2025-bar.md"
    md.parent.mkdir()
    md.write_text(
        "---\n"
        'title: "Foo: a study"\n'
        "year: 2025\n"
        "category: [compbio]\n"
        "keywords: [retrieval, tokens]\n"
        "referenced_papers:\n"
        "  - [[compbio/baz-2024-qux]]\n"
        "generated_at: 2026-05-28\n"
        "---\n"
        "## Summary\nbody text\n"
    )
    p = read_page(md)
    assert p is not None
    assert p.fm["title"] == "Foo: a study"          # colon-bearing scalar survives
    assert p.fm["year"] == 2025                      # native int
    assert p.fm["category"] == ["compbio"]           # native list
    assert p.list_field("keywords") == ["retrieval", "tokens"]
    # YAML reads an unquoted `- [[cat/stem]]` entry as nested flow lists,
    # NOT the literal "[[...]]" string. _fm_referenced_papers handles this.
    assert p.fm["referenced_papers"] == [[["compbio/baz-2024-qux"]]]
    assert isinstance(p.fm["generated_at"], date)
    assert p.body.startswith("## Summary")


def test_read_page_malformed_yaml_yields_empty_fm_not_none(tmp_path):
    # A typo'd block must not drop the page from read_pages() entirely.
    md = tmp_path / "x.md"
    md.write_text("---\ntitle: \"unterminated\n: : :\n---\nbody\n")
    p = read_page(md)
    assert p is not None
    assert p.fm == {}
    assert p.body == "body\n"


def test_read_page_no_frontmatter_returns_none(tmp_path):
    md = tmp_path / "x.md"
    md.write_text("no frontmatter here\n")
    assert read_page(md) is None


# --- _fm_referenced_papers (YAML nested-list vs quoted-string shapes) ----

def test_fm_referenced_papers_handles_both_yaml_shapes():
    from researchwiki.synthesis_candidates.detect import _fm_referenced_papers
    # unquoted `- [[cat/stem]]` → nested lists
    nested = [[["single-cell/cui-2024-scgpt"]], [["ai/zandieh-2025-turboquant"]]]
    assert _fm_referenced_papers(nested) == {
        "single-cell/cui-2024-scgpt", "ai/zandieh-2025-turboquant"}
    # quoted `- "[[cat/stem]]"` → strings
    strings = ["[[single-cell/cui-2024-scgpt]]", "[[ai/zandieh-2025-turboquant]]"]
    assert _fm_referenced_papers(strings) == {
        "single-cell/cui-2024-scgpt", "ai/zandieh-2025-turboquant"}
    assert _fm_referenced_papers(None) == set()


# --- str_field ----------------------------------------------------------

def test_str_field_passthrough_and_default():
    p = _page({"title": "Foo"})
    assert p.str_field("title") == "Foo"
    assert p.str_field("missing") == ""
    assert p.str_field("missing", "fallback") == "fallback"


def test_str_field_coerces_non_strings():
    # native YAML: int year, date, list
    p = _page({"year": 2025, "generated_at": date(2026, 5, 28),
               "tags": ["a", "b"]})
    assert p.str_field("year") == "2025"
    assert p.str_field("generated_at") == "2026-05-28"
    assert p.str_field("tags") == "a, b"


# --- list_field ---------------------------------------------------------

def test_list_field_native_yaml_list():
    p = _page({"keywords": ["retrieval", "tokens"]})
    assert p.list_field("keywords") == ["retrieval", "tokens"]


def test_list_field_legacy_literal_string():
    # line parser renders `keywords: [a, b]` as the literal string "[a, b]"
    p = _page({"keywords": "[retrieval, tokens]"})
    assert p.list_field("keywords") == ["retrieval", "tokens"]


def test_list_field_bare_comma_string_and_empty():
    assert _page({"k": "a, b ,c"}).list_field("k") == ["a", "b", "c"]
    assert _page({"k": ""}).list_field("k") == []
    assert _page({}).list_field("k") == []
    assert _page({"k": "[]"}).list_field("k") == []


# --- year_int -----------------------------------------------------------

def test_year_int_handles_int_and_string():
    assert _page({"year": 2025}).year_int() == 2025      # native YAML
    assert _page({"year": "2025"}).year_int() == 2025    # line parser
    assert _page({"year": "TODO"}).year_int() is None
    assert _page({}).year_int() is None
    assert _page({"year": True}).year_int() is None      # bool is not a year
