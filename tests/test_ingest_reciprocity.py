"""Cross-links must be written in both directions, with the direction the PDF
supports — the two failures the DeepSpCas9 ingest produced on 2026-08-05.

1. `promote` wrote back-links onto the three verified targets but left the new
   page's own `## Related Papers` as `(none)`, so `lint` reported three
   `missing_backlinks` that nobody had introduced by hand.
2. All three were labelled "topically related" although the source PDF's
   reference list proves two of them are citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from researchwiki import backlinks as bl
from researchwiki.agents import promote


@dataclass
class FakeCandidate:
    wikilink: str
    kind: str = "topical"


PAGE = """---
type: paper
---

## Summary

text

## Related Papers

(none)
"""

TARGET = """---
type: paper
---

## Summary

other

## Related Papers

- [[cgt/unrelated-2001-thing]] — topically related (auto-added; refine)
"""


@pytest.fixture()
def wiki(tmp_path, monkeypatch):
    w = tmp_path / "wiki"
    (w / "cgt").mkdir(parents=True)
    (w / "cgt" / "src-2019-a-source-paper.md").write_text(PAGE, encoding="utf-8")
    (w / "cgt" / "doench-2014-rational-design-of-highly-active.md").write_text(TARGET, encoding="utf-8")
    monkeypatch.setattr(promote, "wiki_dir", lambda: w)
    return w


def test_both_directions_are_written(wiki):
    src = wiki / "cgt" / "src-2019-a-source-paper.md"
    added, skipped = promote._append_backlinks(
        candidates=[FakeCandidate("cgt/doench-2014-rational-design-of-highly-active")],
        source_category="cgt", source_stem="src-2019-a-source-paper",
        source_page=src, source_pdf=None,
    )
    assert added == ["cgt/doench-2014-rational-design-of-highly-active"]
    tgt_body = (wiki / "cgt" / "doench-2014-rational-design-of-highly-active.md").read_text()
    assert "[[cgt/src-2019-a-source-paper]]" in tgt_body
    # ...and the reciprocal, which is what used to be missing.
    src_body = src.read_text()
    assert "[[cgt/doench-2014-rational-design-of-highly-active]]" in src_body


def test_the_none_placeholder_is_cleared_not_appended_under(wiki):
    """62 corpus pages carried `(none)` sitting above real bullets."""
    src = wiki / "cgt" / "src-2019-a-source-paper.md"
    promote._append_backlinks(
        candidates=[FakeCandidate("cgt/doench-2014-rational-design-of-highly-active")],
        source_category="cgt", source_stem="src-2019-a-source-paper",
        source_page=src, source_pdf=None,
    )
    body = src.read_text()
    assert "(none)" not in body
    assert body.count("- [[") == 1


def test_reciprocal_note_is_the_inverse(wiki):
    """Source side says the opposite of target side, via one shared inverter."""
    src = wiki / "cgt" / "src-2019-a-source-paper.md"
    promote._append_backlinks(
        candidates=[FakeCandidate("cgt/doench-2014-rational-design-of-highly-active",
                                  kind="cited_by_source")],
        source_category="cgt", source_stem="src-2019-a-source-paper",
        source_page=src, source_pdf=None,
    )
    # Target page: "the linked (source) paper cites this one".
    tgt = (wiki / "cgt" / "doench-2014-rational-design-of-highly-active.md").read_text()
    line = next(l for l in tgt.splitlines() if "src-2019" in l)
    assert "cites this paper" in line and "cited by this paper" not in line
    # Source page: the inverse.
    line = next(l for l in src.read_text().splitlines() if "doench-2014" in l)
    assert "cited by this paper" in line


def test_source_side_is_not_counted_as_a_backlink(wiki):
    """`backlinks_added` and log.md's count mean 'written onto other pages'."""
    src = wiki / "cgt" / "src-2019-a-source-paper.md"
    added, _ = promote._append_backlinks(
        candidates=[FakeCandidate("cgt/doench-2014-rational-design-of-highly-active")],
        source_category="cgt", source_stem="src-2019-a-source-paper",
        source_page=src, source_pdf=None,
    )
    assert len(added) == 1


def test_omitting_source_page_keeps_the_old_one_directional_behaviour(wiki):
    """Back-compat: callers that don't pass source_page still work."""
    added, _ = promote._append_backlinks(
        candidates=[FakeCandidate("cgt/doench-2014-rational-design-of-highly-active")],
        source_category="cgt", source_stem="src-2019-a-source-paper",
    )
    assert added == ["cgt/doench-2014-rational-design-of-highly-active"]
    assert "(none)" in (wiki / "cgt" / "src-2019-a-source-paper.md").read_text()


class TestStemSurnameYear:
    @pytest.mark.parametrize("stem,want", [
        ("doench-2014-rational-design-of-highly-active", ("doench", "2014")),
        ("garcia-lopez-2020-a-hyphenated-surname-paper", ("garcia-lopez", "2020")),
        ("smith-2024b-a-disambiguated-second-paper", ("smith", "2024")),
        ("1000-genomes-project-2015-a-global-reference", ("1000-genomes-project", "2015")),
    ])
    def test_year_is_the_anchor_not_the_first_hyphen(self, stem, want):
        assert promote._stem_surname_year(stem) == want

    def test_unparseable_returns_none(self):
        assert promote._stem_surname_year("no-year-here") is None


class TestKindUpgrade:
    def test_no_pdf_leaves_the_kind_alone(self):
        c = FakeCandidate("cgt/doench-2014-rational-design-of-highly-active")
        assert promote._upgrade_kind_from_references(c, None) == "topical"

    def test_missing_pdf_leaves_the_kind_alone(self):
        c = FakeCandidate("cgt/doench-2014-rational-design-of-highly-active")
        assert promote._upgrade_kind_from_references(c, Path("/nonexistent.pdf")) == "topical"

    def test_an_asserted_citation_kind_is_never_downgraded(self):
        c = FakeCandidate("cgt/x-2014-y", kind="cites_source")
        assert promote._upgrade_kind_from_references(c, Path("/nonexistent.pdf")) == "cites_source"


class TestPlaceholderNeverEatsProse:
    """`_drop_none_placeholder` runs on every `append_related_paper` call — every
    ingest, every `lint --fix`, every claim-overlap link. A false positive there
    is silent prose loss in the framework's hottest write path.

    The first pattern was `^\\s*\\(?\\s*none\\b.*$`: optional paren, then anything.
    It deleted "None of the three replicates agreed, so this link is tentative."
    The original test used a bullet starting with `-`, so it never saw this.
    """

    @pytest.mark.parametrize("line", [
        "(none)",
        "(none — no overlapping wiki papers)",
        "(None yet.)",
        "none",
        "None.",
    ])
    def test_placeholders_are_dropped(self, line):
        assert bl._drop_none_placeholder(line + "\n") == ""

    @pytest.mark.parametrize("line", [
        "None of the three replicates agreed, so this link is tentative.",
        "None the less, the effect held.",
        "Nonetheless the result stands.",
        "- [[a/b]] — none of the three replicated",
        "None of these papers measure the same quantity",
    ])
    def test_prose_survives(self, line):
        assert bl._drop_none_placeholder(line + "\n").strip() == line

    def test_a_placeholder_mixed_with_prose_drops_only_the_placeholder(self):
        body = "(none)\nNone of the three replicates agreed.\n"
        assert bl._drop_none_placeholder(body).strip() == "None of the three replicates agreed."
