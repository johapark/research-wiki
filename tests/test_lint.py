"""Lint helpers.

`_first_category` normalizes the frontmatter `category:` value for the
category-YAML-vs-directory drift check. The line-based frontmatter parser keeps
the YAML list as a string literal (e.g. ``'[single-cell]'``), while a
PyYAML-parsed frontmatter yields a real list — both forms must normalize to the
same bare category name, or the drift check would fire false positives.
"""

from researchwiki.tasks.lint.walk import first_category as _first_category


def test_string_literal_list_one_item():
    assert _first_category("[single-cell]") == "single-cell"


def test_string_literal_list_is_page_type_dir():
    assert _first_category("[ideas]") == "ideas"


def test_real_list():
    assert _first_category(["single-cell"]) == "single-cell"


def test_bare_scalar():
    assert _first_category("single-cell") == "single-cell"


def test_string_literal_multi_item_takes_first():
    assert _first_category("[single-cell, other]") == "single-cell"


def test_real_list_multi_item_takes_first():
    assert _first_category(["single-cell", "other"]) == "single-cell"


def test_quoted_items_are_unwrapped():
    assert _first_category('["single-cell"]') == "single-cell"
    assert _first_category("['single-cell']") == "single-cell"


def test_empty_forms_return_empty_string():
    assert _first_category("") == ""
    assert _first_category("[]") == ""
    assert _first_category([]) == ""
