"""Tests for the idea-page heading contract lint.

Covers every check in `researchwiki.tasks.lint.idea_contract`:
  - idea_missing_section
  - idea_section_order
  - idea_unexpected_h2
  - idea_missing_verdict_field
  - idea_verdict_label_mismatch
  - idea_verdict_label_unparseable
  - idea_footnotes_undefined
  - idea_references_section_missing

The motivating case for the whole module is `test_unheaded_verdict_prose_is_caught`:
an idea page whose Verdict prose sits above the first H2 with no `## Verdict`
heading passes both mandatory page gates, because they parse paragraphs and
never read headings.
"""

from __future__ import annotations

from pathlib import Path

from researchwiki.tasks.lint.idea_contract import (
    REQUIRED_SECTIONS,
    check_page,
    find_idea_contract_violations,
)


# ---------- fixture builders ----------


def _body(
    sections: tuple[str, ...] = ("Verdict", "Background", "Opportunities",
                                 "Plans", "Caveats"),
    verdict_label: str = "**Strength: incremental.** The design is real but "
                         "assembled from published parts.",
    extra: str = "",
) -> str:
    """Assemble an idea-page body from H2 sections in the given order."""
    parts = []
    for name in sections:
        filler = verdict_label if name.lower().startswith("verdict") else (
            f"Prose for the {name} section, long enough to look like real content."
        )
        parts.append(f"## {name}\n{filler}\n")
    return "\n".join(parts) + extra


def _fm(verdict: str | None = "incremental") -> dict:
    fm = {"type": "idea", "title": "Test idea"}
    if verdict is not None:
        fm["verdict"] = verdict
    return fm


def _kinds(violations: list[dict]) -> set[str]:
    return {v["kind"] for v in violations}


PATH = Path("wiki/ideas/test-idea.md")


# ---------- the happy path ----------


def test_canonical_page_passes():
    assert check_page(PATH, _body(), _fm()) == []


def test_descriptive_heading_suffix_is_accepted():
    """`## Plans — how to actually build this` is the prompt's own example."""
    body = _body(sections=("Verdict", "Background — why this matters",
                           "Opportunities", "Plans — how to build this",
                           "Caveats"))
    assert check_page(PATH, body, _fm()) == []


def test_references_h2_is_allowed():
    body = _body() + "\n## References\n[^a]: [[statistics/x-2020-y]]\n"
    assert check_page(PATH, body, _fm()) == []


def test_what_would_update_this_page_h2_is_allowed():
    body = _body() + "\n## What would update this page\nA paper on X.\n"
    assert check_page(PATH, body, _fm()) == []


# ---------- heading presence and order ----------


def test_unheaded_verdict_prose_is_caught():
    """The regression this module exists for: Verdict content present as
    unheaded prose above the first H2, no `## Verdict` heading."""
    body = ("Unheaded verdict prose that reads like a tl;dr but carries no "
            "heading at all.\n\n"
            + _body(sections=("Background", "Opportunities", "Plans", "Caveats")))
    violations = check_page(PATH, body, _fm())
    assert "idea_missing_section" in _kinds(violations)
    missing = [v for v in violations if v["kind"] == "idea_missing_section"]
    assert len(missing) == 1
    assert "Verdict" in missing[0]["detail"]


def test_each_missing_section_reported_separately():
    body = _body(sections=("Verdict", "Background"))
    missing = [v for v in check_page(PATH, body, _fm())
               if v["kind"] == "idea_missing_section"]
    assert len(missing) == 3  # Opportunities, Plans, Caveats


def test_misordered_sections_are_caught():
    """Verdict written last — the pre-contract shape — with all five present."""
    body = _body(sections=("Background", "Opportunities", "Plans",
                           "Caveats", "Verdict"))
    violations = check_page(PATH, body, _fm())
    order = [v for v in violations if v["kind"] == "idea_section_order"]
    assert len(order) == 1
    assert "background → opportunities → plans → caveats → verdict" \
        in order[0]["detail"]


def test_order_not_reported_when_sections_missing():
    """Missing-section violations are the actionable report; an order
    complaint on an incomplete page is noise."""
    body = _body(sections=("Background", "Verdict"))
    assert "idea_section_order" not in _kinds(check_page(PATH, body, _fm()))


def test_required_sections_constant_matches_claude_md():
    assert REQUIRED_SECTIONS == (
        "verdict", "background", "opportunities", "plans", "caveats",
    )


# ---------- unexpected H2s ----------


def test_related_papers_is_flagged():
    """`## Related Papers` is a paper-page section; several idea pages carry
    it as an empty trailing stub."""
    body = _body() + "\n## Related Papers\n"
    violations = check_page(PATH, body, _fm())
    unexpected = [v for v in violations if v["kind"] == "idea_unexpected_h2"]
    assert len(unexpected) == 1
    assert "Related Papers" in unexpected[0]["detail"]


def test_unexpected_h2_detail_preserves_author_case():
    body = _body() + "\n## Related Papers\n"
    detail = [v for v in check_page(PATH, body, _fm())
              if v["kind"] == "idea_unexpected_h2"][0]["detail"]
    assert "## Related Papers" in detail
    assert "related papers" not in detail


# ---------- verdict label mirror ----------


def test_missing_yaml_verdict_is_caught_and_names_section_label():
    violations = check_page(PATH, _body(), _fm(verdict=None))
    field = [v for v in violations if v["kind"] == "idea_missing_verdict_field"]
    assert len(field) == 1
    assert "incremental" in field[0]["detail"]


def test_label_mismatch_is_caught():
    violations = check_page(PATH, _body(), _fm(verdict="strong"))
    mismatch = [v for v in violations
                if v["kind"] == "idea_verdict_label_mismatch"]
    assert len(mismatch) == 1
    assert "strong" in mismatch[0]["detail"]
    assert "incremental" in mismatch[0]["detail"]


def test_invalid_yaml_label_is_caught():
    assert "idea_verdict_label_mismatch" in _kinds(
        check_page(PATH, _body(), _fm(verdict="promising"))
    )


def test_bare_label_without_strength_prefix_parses():
    """`**incremental** — high impact…` is in use on a real page."""
    body = _body(verdict_label="**incremental** — high impact, high feasibility.")
    assert check_page(PATH, body, _fm(verdict="incremental")) == []


def test_parenthetical_qualifier_does_not_hijack_the_label():
    """`**Strength: incremental (with strong upside…)**` must read as
    incremental, not strong — a naive substring search picks up the later word."""
    body = _body(
        verdict_label="**Strength: incremental (with strong upside contingent "
                      "on Phase 5)** — the design is a clean composition."
    )
    assert check_page(PATH, body, _fm(verdict="incremental")) == []


def test_label_after_leading_blockquote_parses():
    """Two real pages open the Verdict section with a `> *Strategic context…*`
    banner before the label line."""
    body = _body(
        verdict_label="> *Sibling design axis: this page proposes the "
                      "generation capability.*\n\n"
                      "**Strength: weak.** The evidence base lies outside biology."
    )
    assert check_page(PATH, body, _fm(verdict="weak")) == []


def test_unparseable_label_is_caught():
    body = _body(verdict_label="This section forgot to state a strength at all.")
    assert "idea_verdict_label_unparseable" in _kinds(
        check_page(PATH, body, _fm(verdict=None))
    )


def test_quoted_yaml_verdict_is_tolerated():
    assert check_page(PATH, _body(), _fm(verdict='"incremental"')) == []


# ---------- footnotes ----------


def test_undefined_footnote_ref_is_caught():
    body = _body() + "\nA claim.[^ghost]\n\n## References\n[^real]: [[a/b-2020-c]]\n"
    violations = check_page(PATH, body, _fm())
    undef = [v for v in violations if v["kind"] == "idea_footnotes_undefined"]
    assert len(undef) == 1
    assert "[^ghost]" in undef[0]["detail"]


def test_definitions_without_references_h2_is_convention_drift_not_breakage():
    """A real page has 25 definitions and no `## References` heading. The
    footnotes still resolve, so this must NOT report as undefined."""
    body = _body() + "\nA claim.[^a]\n\n[^a]: [[statistics/x-2020-y]]\n"
    kinds = _kinds(check_page(PATH, body, _fm()))
    assert "idea_references_section_missing" in kinds
    assert "idea_footnotes_undefined" not in kinds


def test_defined_footnotes_under_references_pass():
    body = _body() + "\nA claim.[^a]\n\n## References\n[^a]: [[statistics/x-2020-y]]\n"
    assert check_page(PATH, body, _fm()) == []


# ---------- robustness ----------


def test_fenced_code_headings_do_not_count():
    """An H2-shaped line inside a fenced block must not satisfy the contract."""
    body = ("```markdown\n## Verdict\n## Background\n## Opportunities\n"
            "## Plans\n## Caveats\n```\n")
    missing = [v for v in check_page(PATH, body, _fm())
               if v["kind"] == "idea_missing_section"]
    assert len(missing) == len(REQUIRED_SECTIONS)


def test_html_comment_headings_do_not_count():
    body = _body() + "\n<!--\n## Related Papers\n-->\n"
    assert "idea_unexpected_h2" not in _kinds(check_page(PATH, body, _fm()))


def test_commented_out_label_does_not_satisfy_the_mirror():
    """A provenance comment mentioning a label must not be read as the label."""
    body = _body(verdict_label="No label here.\n\n<!-- **Strength: strong** -->")
    assert "idea_verdict_label_unparseable" in _kinds(
        check_page(PATH, body, _fm(verdict=None))
    )


# ---------- page selection ----------


def test_non_idea_pages_are_skipped():
    pages = [Path("wiki/synthesis/x.md"), Path("wiki/statistics/y-2020-z.md")]
    bodies = {p: "no headings at all" for p in pages}
    fms = {p: {"type": "synthesis"} for p in pages}
    assert find_idea_contract_violations(pages, bodies, fms) == []


def test_idea_typed_page_outside_ideas_dir_is_skipped():
    """Selection is dir + type, mirroring concept_contract."""
    p = Path("wiki/statistics/stray.md")
    assert find_idea_contract_violations([p], {p: ""}, {p: {"type": "idea"}}) == []


def test_ideas_dir_page_without_idea_type_is_skipped():
    p = Path("wiki/ideas/readme.md")
    assert find_idea_contract_violations([p], {p: ""}, {p: {"type": "note"}}) == []


def test_finder_collects_across_pages():
    a, b = Path("wiki/ideas/a.md"), Path("wiki/ideas/b.md")
    bodies = {a: _body(), b: _body(sections=("Verdict", "Background"))}
    fms = {a: _fm(), b: _fm()}
    out = find_idea_contract_violations([a, b], bodies, fms)
    assert {v["page"] for v in out} == {b}
