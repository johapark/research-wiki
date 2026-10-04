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


def test_inference_with_one_missing_pdf_fails_the_gate(tmp_path, monkeypatch):
    """One available source cannot substantiate an inference from two papers."""
    def resolve(stem):
        if stem == "bar-2025-baz":
            raise FileNotFoundError(stem)
        return tmp_path / f"{stem}.pdf"

    monkeypatch.setattr(fidelity, "resolve_pdf", resolve)
    page = tmp_path / "x.md"
    page.write_text(PAGE, encoding="utf-8")
    report = fidelity.grade_synthesis(page, semantic=False)

    assert not report.ok
    assert report.n_inference_ungradable == 1
    assert report.n_inference == 0
    assert report.claims[0].unresolved_citations == ["bar-2025-baz"]


def test_qualitative_inference_with_unreadable_pdf_fails(tmp_path, monkeypatch):
    """A file's existence alone does not make its contents checkable."""
    page = tmp_path / "x.md"
    page.write_text(PAGE.replace("a shared floor near 0.1%", "a shared mechanism"),
                    encoding="utf-8")
    bad_pdf = tmp_path / "unreadable.pdf"
    bad_pdf.write_bytes(b"not a PDF")
    monkeypatch.setattr(fidelity, "resolve_pdf", lambda stem: bad_pdf)

    def cannot_extract(*args, **kwargs):
        raise ValueError("unreadable PDF")

    monkeypatch.setattr(fidelity, "extract_pdf", cannot_extract)
    report = fidelity.grade_synthesis(page, semantic=False)

    assert not report.ok
    assert report.n_inference_ungradable == 1
    assert set(report.claims[0].unresolved_citations) == {
        "foo-2024-bar", "bar-2025-baz",
    }


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


# ---------- a page cross-reference is not a premise ----------

CROSSREF_PAGE = """---
type: synthesis
author_model: "claude-opus-5"
---

## Outlook

Variant-aware prediction is blocked on modelling rather than data *(inference)*.[^real]
The case is argued at length on [[synthesis/variant-aware-crispr-off-target]].

[^real]: [[cgt/foo-2024-bar]] — a paper with a PDF
"""


def test_a_sibling_page_link_does_not_make_an_inference_ungradable(tmp_path, monkeypatch):
    """Requiring every cited source to have a readable PDF flagged the landed
    CRISPR page: its inference cited three real papers and also pointed at a
    sibling synthesis page for the longer argument. A `[[synthesis/…]]` link is
    a cross-reference, not a premise, so it must not count toward the check."""
    monkeypatch.setattr(fidelity, "resolve_pdf", lambda stem: stem)
    monkeypatch.setattr(fidelity, "_full_text", lambda stem, cache: "modelling work remains")
    page = tmp_path / "x.md"
    page.write_text(CROSSREF_PAGE, encoding="utf-8")
    report = fidelity.grade_synthesis(page, semantic=False)
    assert report.n_inference_ungradable == 0
    assert report.n_inference == 1
    assert report.ok


def test_reference_documents_still_count_as_sources():
    """`references/` is NOT a cross-reference dir: a guidance document or
    whitepaper has a real PDF in `papers/` — all 14 reference pages do — and
    citing one is citing a source. Excluding the whole page-type-dir set
    dropped FDA guidance from an inference's premises."""
    assert not fidelity._is_page_crossref("references/fda-2026-safety-assessment-of-genome-editing")
    assert not fidelity._is_page_crossref("cgt/bae-2014-cas-offinder-a-fast-and-versatile")
    for authored in ("synthesis/foo", "ideas/bar", "proposals/baz", "concepts/qux"):
        assert fidelity._is_page_crossref(authored), authored


def test_crossref_detection_handles_alias_anchor_and_case():
    for link in ("synthesis/foo|see also", "synthesis/foo#kc-abcd1234", "Synthesis/Foo"):
        assert fidelity._is_page_crossref(link), link


def test_an_inference_citing_only_a_page_crossref_still_fails(tmp_path, monkeypatch):
    """The dropped link must not become a free pass: with nothing else cited
    there are no premises at all, which is the original P1."""
    monkeypatch.setattr(fidelity, "resolve_pdf", lambda stem: stem)
    only_ref = CROSSREF_PAGE.replace("[^real]", "").replace(
        "[^real]: [[cgt/foo-2024-bar]] — a paper with a PDF", "")
    page = tmp_path / "y.md"
    page.write_text(only_ref, encoding="utf-8")
    report = fidelity.grade_synthesis(page, semantic=False)
    assert report.n_inference_ungradable == 1
    assert not report.ok


def test_bare_stem_crossref_is_recognised_like_the_prefixed_form(monkeypatch):
    """Adversarial review: judging on the written prefix alone made a
    *documented* citation form fail where the prefixed one passed. CLAUDE.md
    allows a bare `[[stem]]` when referring to a paper as a whole, and the
    authoring prompt requires bare stems inside tables, so the two forms must
    resolve identically. Checked against the real page corpus."""
    monkeypatch.setattr(fidelity, "_AUTHORED_STEMS",
                        frozenset({"variant-aware-crispr-off-target"}))
    assert fidelity._is_page_crossref("synthesis/variant-aware-crispr-off-target")
    assert fidelity._is_page_crossref("variant-aware-crispr-off-target")
    assert fidelity._is_page_crossref("variant-aware-crispr-off-target|see also")
    # A paper stem is still a source in either form.
    assert not fidelity._is_page_crossref("cgt/bae-2014-cas-offinder")
    assert not fidelity._is_page_crossref("bae-2014-cas-offinder")


def test_authored_stem_lookup_degrades_quietly(monkeypatch):
    """A filesystem problem must not fail a grade run: the lookup falls back to
    prefix-only detection rather than raising."""
    monkeypatch.setattr(fidelity, "_AUTHORED_STEMS", None)
    monkeypatch.setattr("researchwiki.paths.wiki_dir",
                        lambda: (_ for _ in ()).throw(OSError("boom")))
    assert fidelity._authored_page_stems() == frozenset()
    assert fidelity._is_page_crossref("synthesis/foo")      # prefix still works


NEGATING_PAGE = """---
type: synthesis
author_model: "claude-opus-5"
---

## Outlook

Neither assay detects sites below the floor, so the question cannot be settled with present tools *(inference)*.[^a]

[^a]: [[cgt/foo-2024-bar]] — a paper
"""


def test_negation_parity_is_computed_on_the_inference_path(tmp_path, monkeypatch):
    """It was hard-coded False. Negation parity is the one signal that survives
    the retrieval skip, because it compares the claim's polarity against the
    premises instead of expecting the conclusion to appear in them. Advisory,
    as on the graded path."""
    monkeypatch.setattr(fidelity, "resolve_pdf", lambda stem: stem)
    monkeypatch.setattr(fidelity, "_full_text",
                        lambda stem, cache: "both assays detect sites at this frequency")
    page = tmp_path / "n.md"
    page.write_text(NEGATING_PAGE, encoding="utf-8")
    report = fidelity.grade_synthesis(page, semantic=False)
    (claim,) = report.claims
    assert claim.verdict == "inference"
    assert claim.negation_mismatch is True      # surfaced, not swallowed
    assert report.ok                            # advisory, not a failure
