"""`researchwiki backfill hook` — candidate selection and frontmatter insertion.

The LLM call itself (`_propose_one_hook`) isn't covered here; the parsing it
depends on is pinned in `test_hook_field.py`. What matters at this layer is that
the target picks the same pages `lint` would and writes YAML that survives a
round trip — including the migration case, where an imported page has neither
the framework's section headings nor a `short_name:` anchor.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from researchwiki.tasks.backfill import (
    _find_hook_candidates,
    _insert_hook_lines,
)


@pytest.fixture
def tmp_wiki(tmp_path, monkeypatch):
    """Synthetic wiki. Patch every module that bound `wiki_dir` at import time."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    fake = lambda: wiki  # noqa: E731
    monkeypatch.setattr("researchwiki.paths.wiki_dir", fake)
    monkeypatch.setattr("researchwiki.tasks.lint.walk.wiki_dir", fake)
    monkeypatch.chdir(tmp_path)
    return wiki


def _mkpage(wiki: Path, key: str, fm: str, body: str = "x" * 300) -> Path:
    p = wiki / f"{key}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\n{fm}\n---\n\n## Summary\n\n{body}\n")
    return p


# ---------- candidate selection ----------

def test_only_pages_without_a_hook_are_candidates(tmp_wiki):
    _mkpage(tmp_wiki, "genomics/a-2026-x", 'type: paper\ntitle: A\nhook: "Has one."')
    _mkpage(tmp_wiki, "genomics/b-2026-y", "type: paper\ntitle: B")
    assert [p.name for p in _find_hook_candidates()] == ["b-2026-y.md"]


def test_non_paper_page_types_are_candidates_too(tmp_wiki):
    """Hooks are broader than keywords: synthesis / concept / reference pages all
    get an index.md bullet, so all need the field."""
    _mkpage(tmp_wiki, "synthesis/a-topic", "type: synthesis\ntitle: S")
    _mkpage(tmp_wiki, "concepts/a-term", "type: concept\ntitle: C")
    _mkpage(tmp_wiki, "references/fda-2026-g", "type: guidance\ntitle: G")
    assert len(_find_hook_candidates()) == 3


def test_root_bookkeeping_is_never_a_candidate(tmp_wiki):
    (tmp_wiki / "index.md").write_text("---\ntype: meta\n---\n\n# Index\n")
    (tmp_wiki / "log.md").write_text("---\ntype: meta\n---\n\n# Log\n")
    assert _find_hook_candidates() == []


# ---------- frontmatter insertion ----------

def test_hook_is_written_quoted_and_round_trips(tmp_wiki):
    p = _mkpage(tmp_wiki, "genomics/a-2026-x", "type: paper\ntitle: A")
    hook = 'Sharpens [[genomics/b-2026-y]] — ratio 3:1, "quoted", back\\slash.'
    assert _insert_hook_lines(p, hook, "") is True
    fm = yaml.safe_load(p.read_text().split("---", 2)[1])
    assert fm["hook"] == hook
    assert fm["title"] == "A"


def test_short_name_rides_along_when_supplied(tmp_wiki):
    p = _mkpage(tmp_wiki, "genomics/a-2026-x", "type: paper\ntitle: A")
    _insert_hook_lines(p, "A gloss.", "DOCKS")
    fm = yaml.safe_load(p.read_text().split("---", 2)[1])
    assert (fm["hook"], fm["short_name"]) == ("A gloss.", "DOCKS")


def test_existing_hook_is_replaced_not_duplicated(tmp_wiki):
    p = _mkpage(tmp_wiki, "genomics/a-2026-x",
                'type: paper\ntitle: A\nhook: "Old gloss."')
    _insert_hook_lines(p, "New gloss.", "")
    text = p.read_text()
    assert text.count("hook:") == 1
    assert yaml.safe_load(text.split("---", 2)[1])["hook"] == "New gloss."


def test_insert_anchors_after_short_name_when_present(tmp_wiki):
    p = _mkpage(tmp_wiki, "genomics/a-2026-x",
                "type: paper\ntitle: A\nshort_name: DOCKS")
    _insert_hook_lines(p, "A gloss.", "")
    fm_lines = p.read_text().split("---")[1].strip().splitlines()
    assert fm_lines[fm_lines.index("short_name: DOCKS") + 1].startswith("hook:")


def test_body_is_preserved_exactly(tmp_wiki):
    p = _mkpage(tmp_wiki, "genomics/a-2026-x", "type: paper\ntitle: A", body="Body text.")
    before = p.read_text().split("---", 2)[2]
    _insert_hook_lines(p, "A gloss.", "")
    assert p.read_text().split("---", 2)[2] == before


def test_block_scalar_in_frontmatter_survives(tmp_wiki):
    """Concept pages carry `concept_thesis: |`. Anchoring on title: must not
    drop the insert inside the block."""
    p = _mkpage(tmp_wiki, "concepts/a-term",
                "type: concept\ntitle: C\nconcept_thesis: |\n  line one\n  line two")
    assert _insert_hook_lines(p, "A gloss.", "") is True
    fm = yaml.safe_load(p.read_text().split("---", 2)[1])
    assert fm["concept_thesis"].strip() == "line one\nline two"
    assert fm["hook"] == "A gloss."


# ---------- migration cases ----------

def test_page_with_no_frontmatter_is_refused_not_corrupted(tmp_wiki):
    """An imported page with no YAML block has nowhere to put the field. Refuse
    and leave the file byte-identical rather than inventing a header."""
    p = tmp_wiki / "genomics" / "imported.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Some imported page\n\nProse without frontmatter.\n")
    before = p.read_text()
    assert _insert_hook_lines(p, "A gloss.", "") is False
    assert p.read_text() == before


def test_page_with_no_title_anchor_is_refused(tmp_wiki):
    p = _mkpage(tmp_wiki, "genomics/a-2026-x", "type: paper\nyear: 2026")
    before = p.read_text()
    assert _insert_hook_lines(p, "A gloss.", "") is False
    assert p.read_text() == before


def test_unterminated_frontmatter_is_refused(tmp_wiki):
    p = tmp_wiki / "genomics" / "broken.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("---\ntitle: A\n\n## Summary\n\nNo closing fence.\n")
    before = p.read_text()
    assert _insert_hook_lines(p, "A gloss.", "") is False
    assert p.read_text() == before
