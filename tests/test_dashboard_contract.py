"""Tests for the advisory ``wiki/views.md`` semantic contract."""

from pathlib import Path

import pytest

from researchwiki.tasks.init_dashboard import VIEWS_MD_TEMPLATE
from researchwiki.tasks.lint.dashboard_contract import (
    find_dashboard_contract_violations,
)


def _check(tmp_path: Path, text: str) -> list[dict]:
    views = tmp_path / "views.md"
    views.write_text(text, encoding="utf-8")
    return find_dashboard_contract_violations(views)


def _kinds(violations: list[dict]) -> set[str]:
    return {violation["kind"] for violation in violations}


def test_canonical_dashboard_passes(tmp_path):
    assert _check(tmp_path, VIEWS_MD_TEMPLATE) == []


def test_custom_prose_and_extra_columns_are_allowed(tmp_path):
    customized = VIEWS_MD_TEMPLATE.replace(
        "# Wiki Dashboard\n",
        "# Wiki Dashboard\n\nA personal note about how I use these tables.\n",
    ).replace(
        '  venue AS "Journal",\n  year AS "Year",',
        '  venue AS "Journal",\n  file.folder AS "Folder",\n  year AS "Year",',
    )

    assert _check(tmp_path, customized) == []


@pytest.mark.parametrize(
    ("mutate", "expected_kind"),
    [
        (
            lambda text: text.replace(
                'link(file.link, file.name) AS "Stem"',
                'short_name AS "Stem"',
            ),
            "dashboard_paper_columns",
        ),
        (
            lambda text: text.replace(
                '  link(file.link, file.name) AS "Stem",',
                '  title AS "Title",\n  link(file.link, file.name) AS "Stem",',
            ),
            "dashboard_paper_columns",
        ),
        (
            lambda text: text.replace("LIMIT 10", "LIMIT 5", 1),
            "dashboard_idea_query",
        ),
        (
            lambda text: text.replace(
                'WHERE type = "proposal" AND created_at',
                'WHERE type = "proposal"',
            ),
            "dashboard_proposal_query",
        ),
        (
            lambda text: text.replace(
                '  file.link AS "Synthesis",',
                '  file.link AS "Synthesis",\n  length(referenced_papers) AS "Members",',
            ),
            "dashboard_synthesis_members",
        ),
        (
            lambda text: text.replace(
                "SORT ingested_at DESC",
                "SORT file.mtime DESC",
            ),
            "dashboard_filesystem_time",
        ),
        (
            lambda text: text.replace("```dataview\n", "```dataviewjs\n", 1),
            "dashboard_dataviewjs",
        ),
    ],
)
def test_contract_drift_is_reported(tmp_path, mutate, expected_kind):
    assert expected_kind in _kinds(_check(tmp_path, mutate(VIEWS_MD_TEMPLATE)))


def test_table_order_drift_is_reported(tmp_path):
    prefix, after_idea = VIEWS_MD_TEMPLATE.split("## Recent ideas", 1)
    idea, after_synthesis = after_idea.split("## Recent synthesis pages", 1)
    synthesis, proposals = after_synthesis.split("## Recent proposals", 1)
    reordered = (
        prefix
        + "## Recent synthesis pages" + synthesis
        + "## Recent ideas" + idea
        + "## Recent proposals" + proposals
    )

    assert "dashboard_section_order" in _kinds(_check(tmp_path, reordered))


def test_missing_dashboard_is_reported(tmp_path):
    violations = find_dashboard_contract_violations(tmp_path / "views.md")

    assert _kinds(violations) == {"dashboard_missing"}


#: The fourth section a dashboard scaffolded before proposals carried.
_LEGACY_CONCEPT_SECTION = """## Recent concept hubs (top 10)

```dataview
TABLE WITHOUT ID
  file.link AS "Concept",
  length(referenced_papers) AS "Members",
  concept_span AS "Categories",
  concept_thesis AS "Thesis",
  dateformat(generated_at, "yyyy-MM-dd") AS "Filed"
FROM ""
WHERE type = "concept" AND generated_at
SORT generated_at DESC
LIMIT 10
```
"""


def _legacy_dashboard() -> str:
    head = VIEWS_MD_TEMPLATE[: VIEWS_MD_TEMPLATE.index("## Recent proposals")]
    return head + _LEGACY_CONCEPT_SECTION


def test_a_legacy_concept_hub_dashboard_still_passes(tmp_path):
    """A wiki scaffolded before proposals ends its dashboard with a concept-hub
    table. That is a valid layout for a wiki that still has hubs, not a defect
    that only `init --refresh-dashboard` could clear."""
    assert _check(tmp_path, _legacy_dashboard()) == []


def test_a_legacy_concept_table_is_still_checked(tmp_path):
    broken = _legacy_dashboard().replace(
        'length(referenced_papers) AS "Members"', 'concept_span AS "Members"')
    assert "dashboard_concept_members" in _kinds(_check(tmp_path, broken))


def test_a_proposal_table_is_required_once_concepts_are_gone(tmp_path):
    head = VIEWS_MD_TEMPLATE[: VIEWS_MD_TEMPLATE.index("## Recent proposals")]
    assert "dashboard_section_missing" in _kinds(_check(tmp_path, head))
