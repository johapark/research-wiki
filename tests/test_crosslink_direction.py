"""Back-link bullets whose citation direction the publication years rule out.

The motivating finding: 177 bullets corpus-wide claim the target cites the
page while the target is the older paper. Three spot-checks confirmed the
older paper's PDF does not mention the newer one at all. `lint --fix` and
`promote` both write these from an inferred edge direction, and no other check
sees a bullet's claim — `missing_backlinks` is satisfied by its presence.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.tasks.lint.contracts import LINT_JSON_KEYS
from researchwiki.tasks.lint.crosslink_direction import find_impossible_citation_directions

A = Path("wiki/cgt/abadi-2017-a-paper.md")       # older
B = Path("wiki/cgt/chuai-2020-b-paper.md")       # newer, 3 years on


# Default years are 3 apart: `MIN_YEAR_GAP` exempts a 1-year gap, since that is
# where preprint-vs-journal ambiguity lives (see test_one_year_gaps_*). Tests
# about anything other than the gap itself need a reportable pair.


def _run(bullets_on_a="", bullets_on_b="", year_a=2017, year_b=2020):
    pages = [A, B]
    body = {
        A: f"## Summary\n\ntext\n\n## Related Papers\n\n{bullets_on_a}\n",
        B: f"## Summary\n\ntext\n\n## Related Papers\n\n{bullets_on_b}\n",
    }
    fm = {A: {"year": year_a}, B: {"year": year_b}}
    return find_impossible_citation_directions(pages, body, fm)


def test_lint_json_exposes_the_key():
    assert "crosslink_impossible_citations" in LINT_JSON_KEYS


def test_older_page_claiming_the_newer_one_cites_it_is_fine():
    """2020 can cite 2017, so this bullet on the 2017 page is plausible."""
    assert _run(bullets_on_a="- [[cgt/chuai-2020-b-paper]] — cites this paper (auto-added; refine)") == []


def test_newer_page_claiming_the_older_one_cites_it_is_flagged():
    got = _run(bullets_on_b="- [[cgt/abadi-2017-a-paper]] — cites this paper (auto-added; refine)")
    assert [v["kind"] for v in got] == ["crosslink_impossible_citation"]
    assert "abadi-2017-a-paper (2017)" in got[0]["detail"]
    assert "chuai-2020-b-paper (2020)" in got[0]["detail"]


def test_cited_by_direction_is_checked_too():
    """`cited by this paper` on the 2017 page asserts 2017 cites 2020."""
    got = _run(bullets_on_a="- [[cgt/chuai-2020-b-paper]] — cited by this paper (auto-added; refine)")
    assert len(got) == 1
    assert got[0]["page"] == A


def test_cited_by_in_the_possible_direction_passes():
    assert _run(bullets_on_b="- [[cgt/abadi-2017-a-paper]] — cited by this paper") == []


def test_topical_bullets_are_never_flagged():
    """A topical note makes no citation claim, so year order says nothing."""
    assert _run(bullets_on_b="- [[cgt/abadi-2017-a-paper]] — topically related (auto-added; refine)") == []


def test_same_year_is_never_flagged():
    """A paper can cite a preprint from earlier the same year."""
    assert _run(bullets_on_b="- [[cgt/abadi-2017-a-paper]] — cites this paper",
                year_a=2026, year_b=2026) == []


def test_missing_or_malformed_year_is_skipped_not_guessed():
    for bad in (None, "n.d.", True):
        assert _run(bullets_on_b="- [[cgt/abadi-2017-a-paper]] — cites this paper",
                    year_a=bad) == []


def test_string_year_is_accepted():
    got = _run(bullets_on_b="- [[cgt/abadi-2017-a-paper]] — cites this paper", year_a="2017")
    assert len(got) == 1


def test_bullets_outside_related_papers_are_ignored():
    pages = [A, B]
    body = {A: "## Summary\n\ntext\n", B: "## Summary\n\n- [[cgt/abadi-2017-a-paper]] — cites this paper\n"}
    assert find_impossible_citation_directions(pages, body, {A: {"year": 2017}, B: {"year": 2018}}) == []


def test_alias_and_anchor_links_still_resolve():
    got = _run(bullets_on_b="- [[cgt/abadi-2017-a-paper|Abadi 2017]] — cites this paper")
    assert len(got) == 1


def test_one_year_gaps_are_not_reported():
    """Adversarial review: at a gap of 1 the finding is dominated by papers
    citing the *preprint* of a work whose journal year is later. 174 of 177
    corpus findings were one-year gaps and 11 were confirmed real citations by
    reading the PDFs — e.g. chin-2019 (submitted 2019, published 2020) really
    does cite zook-2020. The floor keeps the gaps no publication lag explains."""
    assert _run(bullets_on_b="- [[cgt/abadi-2017-a-paper]] — cites this paper",
                year_a=2019, year_b=2020) == []


def test_two_year_gaps_are_still_reported():
    got = _run(bullets_on_b="- [[cgt/abadi-2017-a-paper]] — cites this paper",
               year_a=2018, year_b=2020)
    assert [v["kind"] for v in got] == ["crosslink_impossible_citation"]


def test_min_year_gap_constant_is_two():
    from researchwiki.tasks.lint.crosslink_direction import MIN_YEAR_GAP
    assert MIN_YEAR_GAP == 2


@pytest.mark.parametrize("bullet", [
    "- [[cgt/abadi-2017-a-paper]] — cites this paper",
    "* [[cgt/abadi-2017-a-paper]] — cites this paper",
    "  - [[cgt/abadi-2017-a-paper]] — cites this paper",
    "- **[[cgt/abadi-2017-a-paper]]** — cites this paper",
    "+ [[cgt/abadi-2017-a-paper]]: cites this paper",
])
def test_hand_edited_bullet_shapes_are_parsed(bullet):
    """A wrong direction is likeliest in hand-edited prose, which is where `*`
    markers, indented bullets and bolded links appear — all parsed to nothing
    before, so the check silently skipped exactly the riskiest input."""
    got = _run(bullets_on_b=bullet, year_a=2017, year_b=2020)
    assert [v["kind"] for v in got] == ["crosslink_impossible_citation"], bullet
