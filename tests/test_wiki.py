"""Frontmatter parsing in `wiki.read_page`, the reader every command walks with."""

from __future__ import annotations

import datetime as dt

import pytest
import yaml

from researchwiki import wiki


_FRONTMATTER = """\
title: "A: colon-bearing title"
type: paper
year: 2024
doi: 10.1038/example
keywords: [alpha, beta, "gamma, delta"]
authors:
  - Smith, J.
  - Jones, K.
ingested_at: 2026-04-15T14:30:00
generated_at: 2026-06-08
hook: "Uses [[cgt/other-2020-x]] and reports 2x."
empty:
flag: true
"""


def test_the_fast_loader_parses_exactly_like_safe_load():
    """The C loader must be a pure speedup: same types, same values, including
    the timestamp and date scalars Dataview depends on."""
    parsed = wiki.load_frontmatter_yaml(_FRONTMATTER)
    assert parsed == yaml.safe_load(_FRONTMATTER)
    assert isinstance(parsed["ingested_at"], dt.datetime)
    assert isinstance(parsed["generated_at"], dt.date)
    assert parsed["year"] == 2024 and parsed["empty"] is None


def test_the_fast_loader_stays_safe():
    """`CSafeLoader`, never the full loader: arbitrary object tags are refused."""
    with pytest.raises(yaml.YAMLError):
        wiki.load_frontmatter_yaml("x: !!python/object/apply:os.system ['true']\n")


def test_the_c_loader_is_used_when_available():
    if not hasattr(yaml, "CSafeLoader"):
        pytest.skip("PyYAML built without libyaml")
    assert wiki._SafeLoader is yaml.CSafeLoader


def test_malformed_yaml_still_yields_a_page_with_empty_frontmatter(tmp_path):
    page = tmp_path / "broken.md"
    page.write_text("---\ntitle: [unclosed\n---\n\nbody\n", encoding="utf-8")
    parsed = wiki.read_page(page)
    assert parsed is not None and parsed.fm == {} and "body" in parsed.body
