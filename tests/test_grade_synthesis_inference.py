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
    """No PDF index is built for an inference: the conclusion isn't graded.
    (Its numbers are checked against full text — see the tests below.)"""
    monkeypatch.setattr(fidelity, "resolve_pdf", lambda stem: stem)
    monkeypatch.setattr(fidelity, "_full_text", lambda stem, cache: "a floor near 0.1%")

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
    monkeypatch.setattr(fidelity, "_full_text", lambda stem, cache: "a floor near 0.1%")
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


def test_numbers_in_an_inference_must_come_from_a_cited_paper(monkeypatch):
    """The label exempts the conclusion, not its premises. A figure that no
    cited paper contains is misattributed, exactly as it would be unlabelled —
    otherwise *(inference)* would let any number past the fidelity gate."""
    monkeypatch.setattr(fidelity, "resolve_pdf", lambda stem: stem)
    monkeypatch.setattr(fidelity, "_full_text",
                        lambda stem, cache: "a detection floor of 0.5% in both assays")
    unit = _inference_unit()
    claim = fidelity._grade_claim(unit, fidelity._footnote_targets(PAGE),
                                  use_semantic=False, fulltext_cache={})
    assert claim.verdict == "misattributed"
    assert claim.numeric_unmatched


def test_inference_whose_numbers_are_in_a_cited_paper_stays_unchecked(monkeypatch):
    monkeypatch.setattr(fidelity, "resolve_pdf", lambda stem: stem)
    monkeypatch.setattr(fidelity, "_full_text",
                        lambda stem, cache: "both assays share a floor near 0.1% indels")
    unit = _inference_unit()
    claim = fidelity._grade_claim(unit, fidelity._footnote_targets(PAGE),
                                  use_semantic=False, fulltext_cache={})
    assert claim.verdict == "inference"


PAGE_NO_PDF = """---
type: synthesis
author_model: "claude-opus-5"
---

## Outlook

Both families will converge on a shared floor near 0.17% indel frequency *(inference)*.[^syn]

[^syn]: [[synthesis/variant-aware-crispr-off-target]] — a synthesis page, no PDF
"""


def _graded_no_pdf(tmp_path, monkeypatch, text=PAGE_NO_PDF):
    monkeypatch.setattr(fidelity, "resolve_pdf",
                        lambda stem: (_ for _ in ()).throw(FileNotFoundError(stem)))
    page = tmp_path / "x.md"
    page.write_text(text, encoding="utf-8")
    return fidelity.grade_synthesis(page, semantic=False)


def test_inference_with_no_gradable_pdf_fails_the_gate(tmp_path, monkeypatch):
    """Review finding, second pass. Classifying this `uncited` made the verdict
    honest but left the gate green, so a numeric inference citing only a
    PDF-less page still cleared both required gates with its figure unchecked.

    It gets its own verdict and fails. The distinction from plain `uncited`
    prose is what the label asserts: `*(inference)*` says the conclusion
    follows from the papers cited, so with no gradable paper that claim is
    unverifiable rather than merely unsourced — and the label also suppresses
    the retrieval and negation checks, so there is nothing left looking at it.
    """
    report = _graded_no_pdf(tmp_path, monkeypatch)
    assert report.n_inference_ungradable == 1
    assert report.n_inference == 0
    assert [c.verdict for c in report.claims] == ["inference_ungradable"]
    assert not report.ok                      # exit 1
    assert report.n_claims == 0               # not counted as graded
    assert report.to_dict()["n_inference_ungradable"] == 1


def test_uncited_prose_without_the_label_stays_advisory(tmp_path, monkeypatch):
    """The intentional skip the review asked to preserve: an unlabelled unit
    citing only a PDF-less page asserts no provenance, so it stays `uncited`
    and advisory. `check-grounding` is the gate that owns "a citation must be
    present"; duplicating that here would fail every legitimate scope
    paragraph that points at a sibling synthesis page."""
    plain = PAGE_NO_PDF.replace(" *(inference)*", "")
    report = _graded_no_pdf(tmp_path, monkeypatch, plain)
    assert [c.verdict for c in report.claims] == ["uncited"]
    assert report.ok
