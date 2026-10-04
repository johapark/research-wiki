"""`grade synthesis` skips `*(inference)*` units and counts them.

An inference in a synthesis page's Outlook is the author's conclusion from the
papers it cites, so no single cited PDF is expected to state it. Grading it
would either flag a false `weak` or, if it carries a number, a false
`misattributed`. The count is reported so the unchecked share of a page is
visible rather than silently absorbed into `n_claims`.
"""
from __future__ import annotations

from researchwiki.grade import grounding
from researchwiki.grade.fidelity import synthesis as fidelity


PAGE = """---
type: synthesis
author_model: "claude-opus-5-5"
---

## Outlook

Taken together, both assays imply a shared floor near 0.1% that predictors inherit *(inference)*.[^foo][^bar]

[^foo]: [[compbio/foo-2024-bar]]
[^bar]: [[compbio/bar-2025-baz]]
"""


def _inference_unit():
    units = grounding.parse_units(PAGE, permissive=True)
    return next(u for u in units if u.is_claim)


def test_inference_unit_gets_its_own_verdict_without_retrieval(monkeypatch):
    """The verdict is set before any PDF is indexed; the number in the
    sentence (0.1%) is never checked against the cited papers."""
    monkeypatch.setattr(fidelity, "resolve_pdf", lambda stem: stem)

    def _no_index(stem):
        raise AssertionError(f"PDF index built for an inference unit: {stem}")

    monkeypatch.setattr(fidelity, "build_pdf_index", _no_index)
    unit = _inference_unit()
    assert unit.is_inference
    claim = fidelity._grade_claim(unit, fidelity._footnote_targets(PAGE),
                                  use_semantic=False, fulltext_cache={})
    assert claim.verdict == "inference"
    assert claim.numeric_unmatched == []
    assert sorted(claim.cited_stems) == ["bar-2025-baz", "foo-2024-bar"]


def test_report_counts_inference_separately(monkeypatch, tmp_path):
    """`n_inference` is reported, and inference units are excluded from
    `n_claims` (graded) so a page can't look fully checked when it isn't."""
    monkeypatch.setattr(fidelity, "resolve_pdf", lambda stem: stem)
    page = tmp_path / "x.md"
    page.write_text(PAGE, encoding="utf-8")
    report = fidelity.grade_synthesis(page, semantic=False)
    assert report.n_inference == 1
    assert report.n_claims == 0
    assert report.ok
    assert report.to_dict()["n_inference"] == 1


def test_label_on_a_paper_page_is_graded_normally(monkeypatch):
    """Paper pages have no labelled section, so the label is inert there and
    the unit takes the ordinary grading path."""
    page = PAGE.replace("type: synthesis", "type: paper")
    assert not grounding.has_labelled_sections(page)
    units = grounding.parse_units(page, permissive=True)
    assert not any(u.is_inference for u in units)
