"""The detector must count membership over the same sections the scaffolder does.

`candidates concepts` advertised `direction of effect` as "4 paper pages, 3
categories ← concept-ready (bridge)"; `researchwiki concepts "direction of
effect"` then found 2 papers and refused it as not concept-worthy. Both read
claims, not prose — the difference was one section filter. `find_members` routes
every check through `_matching_claims`, restricted to `_CONTRIBUTION_SECTIONS`,
because a term in a paper's *limitations* is a mention and not an instantiation.
The detector applied no filter, so two `limitations` hits counted fully toward
`pages`, `categories` and the bridge label, and the bridge tier advertised a
term that could not be scaffolded.

Membership counts now use the contribution sections. `weighted` and `sections`
still see every section, so the 0.5 `SECTION_WEIGHTS` signal survives and the
author can still see why a term looks bigger than its member count.

Fixtures use one sentence shape across all rows: the term extractor is
phrasing-sensitive (`The X matches y` does not yield `X` where `A X is z` does),
and varying the wording per row silently drops rows and makes these tests pass
for the wrong reason.

See PLAN-bottom-up-synthesis.md and researchwiki/concepts/term_claims.py.
"""

from __future__ import annotations

from researchwiki.concepts.candidates import find_candidates_from_claims

TERM = "Frozen Backbone"
SENTENCE = f"A {TERM} is reused across tasks."


def _rows(*section_category_pairs: tuple[str, str]) -> list[dict]:
    return [
        {"paper_stem": f"p-{i}", "section": sec, "text": SENTENCE,
         "category": cat, "claim_slug": f"{sec[:3]}-{i}"}
        for i, (sec, cat) in enumerate(section_category_pairs)
    ]


def _find(rows: list[dict]) -> dict | None:
    got = find_candidates_from_claims(rows, existing_slugs=set())
    return next((r for r in got if r["term"] == TERM), None)


def test_fixture_shape_is_actually_extracted():
    """Guard the guard: if the extractor stops seeing SENTENCE, every
    assertion below would pass vacuously."""
    hit = _find(_rows(("key_contributions", "ai"), ("results", "compbio"),
                      ("methodology", "single-cell")))
    assert hit is not None and hit["pages"] == 3


def test_limitations_papers_do_not_count_toward_pages_or_categories():
    """The `direction of effect` shape: 2 contribution + 2 limitations papers
    was advertised as 4 pages / 3 categories and was un-scaffoldable."""
    hit = _find(_rows(("key_contributions", "compbio"), ("key_contributions", "genetics"),
                      ("limitations", "genomics"), ("limitations", "single-cell")))
    # Two contribution papers is below the >= 3 floor, so the term is not
    # advertised at all — which is what the scaffolder would have told you.
    assert hit is None


def test_contribution_papers_qualify_and_limitations_stay_visible():
    hit = _find(_rows(("key_contributions", "ai"), ("methodology", "compbio"),
                      ("results", "single-cell"), ("limitations", "genomics")))
    assert hit is not None
    # Three contribution papers across three categories; the limitations paper
    # is neither a member nor a fourth category.
    assert hit["pages"] == 3
    assert hit["categories"] == 3
    # ...but it is still reported as evidence the term recurs, and still moves
    # the ranking signal at its 0.5 weight.
    assert hit["sections"].get("limitations") == 1
    assert hit["weighted"] > 3.0


def test_a_term_only_ever_in_limitations_is_not_a_candidate():
    hit = _find(_rows(("limitations", "ai"), ("limitations", "compbio"),
                      ("limitations", "genomics"), ("limitations", "single-cell")))
    assert hit is None
