"""Tests for the synthesis-page heading contract lint.

Covers every check in `researchwiki.tasks.lint.synthesis_contract`. The one
that matters most for the rollout is `synthesis_legacy_spine`: every synthesis
page written before the fixed structure must collapse to exactly one finding,
so the 30 existing pages read as a to-do list rather than ~250 missing-section
complaints.
"""

from __future__ import annotations

from pathlib import Path

from researchwiki.tasks.lint.contracts import LINT_JSON_KEYS
from researchwiki.tasks.lint.synthesis_contract import (
    REQUIRED_SECTIONS,
    check_page,
    find_synthesis_contract_violations,
    is_legacy,
)

PATH = Path("wiki/synthesis/x.md")
TITLES = ("Question", "Short answer", "Background", "Organizing framework", "Findings",
          "Cross-cutting insights", "Tensions / open questions", "Outlook", "References")


def _body(sections: tuple[str, ...] = TITLES, findings: str = "### Theme A\n\nText.[^a]\n",
          outlook: str = "Next step for the field.[^a]\n") -> str:
    parts = []
    for title in sections:
        if title == "Findings":
            text = findings
        elif title == "Outlook":
            text = outlook
        elif title == "References":
            text = "[^a]: [[cgt/a-2024-x]]\n"
        else:
            text = f"{title} prose.[^a]\n"
        parts.append(f"## {title}\n\n{text}")
    return "\n".join(parts)


def _kinds(violations: list[dict]) -> list[str]:
    return [v["kind"] for v in violations]


def test_canonical_page_passes():
    assert check_page(PATH, _body()) == []


def test_required_sections_constant_matches_claude_md():
    assert REQUIRED_SECTIONS == tuple(t.lower() for t in TITLES)


def test_lint_json_exposes_the_key():
    assert "synthesis_contract_violations" in LINT_JSON_KEYS


# ---------- legacy pages ----------


def test_legacy_page_is_one_finding():
    """Old-shape page: one actionable finding, not one per new section."""
    body = ("## Question\n\nQ.\n\n## Short answer\n\nA.[^a]\n\n## Evidence from the wiki\n\n"
            "E.[^a]\n\n## What would update this page\n\n- x\n\n## References\n\n[^a]: [[a]]\n")
    assert _kinds(check_page(PATH, body)) == ["synthesis_legacy_spine"]


def test_thematic_legacy_page_is_still_legacy():
    """The 17 pages that already replaced `Evidence from the wiki` with argued
    H2s still lack `Organizing framework`, so they are legacy too."""
    body = ("## Question\n\nQ.\n\n## Short answer\n\nA.\n\n## Decision 1: inputs\n\nD.\n\n"
            "## Tensions / open questions\n\nT.\n\n## References\n\n[^a]: [[a]]\n")
    assert is_legacy(body)


def test_new_page_keeping_the_old_update_section_is_legacy():
    """`What would update this page` was replaced by `Outlook`; keeping it means
    the upgrade isn't finished."""
    body = _body() + "\n## What would update this page\n\n- x\n"
    assert _kinds(check_page(PATH, body)) == ["synthesis_legacy_spine"]


# ---------- structure ----------


def test_each_missing_section_reported():
    body = _body(sections=tuple(t for t in TITLES if t not in ("Background", "Outlook")))
    missing = [v["detail"] for v in check_page(PATH, body)
               if v["kind"] == "synthesis_missing_section"]
    assert missing == ["no `## Background` H2 found", "no `## Outlook` H2 found"]


def test_references_required_only_with_footnotes():
    body = _body(sections=TITLES[:-1]).replace("[^a]", "[[cgt/a-2024-x]]")
    assert check_page(PATH, body) == []


def test_misordered_sections_are_caught():
    order = list(TITLES)
    order[3], order[4] = order[4], order[3]
    assert "synthesis_section_order" in _kinds(check_page(PATH, _body(sections=tuple(order))))


def test_extra_h2_is_flagged_and_points_to_findings():
    body = _body() + "\n## Benchmarks\n\nB.[^a]\n"
    (v,) = [v for v in check_page(PATH, body) if v["kind"] == "synthesis_unexpected_h2"]
    assert "`## Benchmarks`" in v["detail"] and "Findings" in v["detail"]


def test_heading_names_are_exact():
    """`Outlook` is matched by the grounding gate, so a variant would silently
    switch its labels off; the contract flags it."""
    body = _body().replace("## Outlook", "## Outlook and future directions")
    assert "synthesis_unexpected_h2" in _kinds(check_page(PATH, body))


def test_findings_without_h3_is_flagged():
    body = _body(findings="One undivided body of prose.[^a]\n")
    assert "synthesis_findings_without_themes" in _kinds(check_page(PATH, body))


# ---------- labels and footnotes ----------


def test_labels_in_outlook_pass():
    body = _body(outlook="A guess about the field *(model prior)*.\n\n"
                         "A conclusion from both *(inference)*.[^a]\n")
    assert check_page(PATH, body) == []


def test_label_outside_outlook_is_flagged():
    body = _body(findings="### Theme A\n\nAn aside *(model prior)*.\n")
    (v,) = [v for v in check_page(PATH, body) if v["kind"] == "synthesis_label_outside_outlook"]
    assert "findings" in v["detail"]


def test_undefined_footnote_is_flagged():
    body = _body(findings="### Theme A\n\nText.[^missing]\n")
    assert "synthesis_footnotes_undefined" in _kinds(check_page(PATH, body))


def test_headings_in_code_and_comments_do_not_count():
    body = _body() + "\n```\n## Benchmarks\n```\n\n<!-- ## Notes -->\n"
    assert check_page(PATH, body) == []


# ---------- finder ----------


def test_finder_checks_only_synthesis_typed_pages_in_synthesis_dir():
    legacy = "## Question\n\nQ.\n\n## What would update this page\n\n- x\n"
    pages = [Path("wiki/synthesis/a.md"), Path("wiki/synthesis/suggested-additions.md"),
             Path("wiki/ideas/b.md")]
    body = {p: legacy for p in pages}
    fm = {pages[0]: {"type": "synthesis"}, pages[1]: {"type": "meta"},
          pages[2]: {"type": "synthesis"}}
    got = find_synthesis_contract_violations(pages, body, fm)
    assert [(v["page"], v["kind"]) for v in got] == [(pages[0], "synthesis_legacy_spine")]


def test_duplicate_section_is_flagged():
    """Review finding: duplicate headings were de-duplicated into `seen`, so a
    second `## Outlook` passed. It matters because the grounding gate's
    labelled range runs to the *next* H2 — labels in the second copy are
    silently inert."""
    body = _body().replace("## References", "## Outlook\n\nA second one.[^a]\n\n## References")
    (v,) = [x for x in check_page(PATH, body) if x["kind"] == "synthesis_duplicate_section"]
    assert "`## Outlook` appears 2 times" in v["detail"]


def test_duplicate_references_is_flagged():
    body = _body() + "\n## References\n\n[^b]: [[cgt/b-2024-y]]\n"
    assert "synthesis_duplicate_section" in _kinds(check_page(PATH, body))


def test_a_single_copy_of_each_section_is_not_flagged():
    assert "synthesis_duplicate_section" not in _kinds(check_page(PATH, _body()))
