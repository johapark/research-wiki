"""Commentary-shaped-PDF guard — the ingest check that refuses to auto-promote
a Research Highlight as `type: paper`.

The regression this pins: a one-page Nature Genetics **Research highlight**
about `gold-2026-scoring-gene-importance-by-interpreting` (32 pages, Nature
Biotechnology) was promoted as a paper, and claim extraction then credited
Gold et al.'s contributions to the highlight's author. Both fidelity gates
passed and were correct to — the claims *are* in the highlight's PDF, they're
just not its contributions. No upstream type lookup helps: Crossref reports
`type: journal-article` / `subtype: None` and PubMed `['Journal Article']` for
the highlight and the primary alike.

Precision is the design target, so the tests are weighted toward proving what
does NOT fire. The two false positives that a 401-PDF corpus sweep actually
produced during development each get a named regression test:
`test_cell_press_correspondence_contact_block_is_not_a_signal` and
`test_article_number_page_field_is_not_a_single_page_extent`.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from researchwiki.agents.commentary import (
    COMMENTARY_PAGE_TYPE,
    crossref_lookup_worthwhile,
    detect_commentary,
    is_single_page_extent,
)
from researchwiki.agents.phases.draft import _wrap_with_frontmatter
from researchwiki.agents.phases.revise import detect_structural_gate_issues
from researchwiki.agents.promote import (
    MIN_GRADED_CLAIMS,
    MIN_KEY_CONTRIBUTIONS,
    _build_frontmatter,
    should_auto_promote,
)
from researchwiki.pdf.text import pdf_shape

# Verbatim head of the offending PDF's only page (control bytes already
# stripped by `pdf_shape`, line endings normalized to \n).
LI_2026_FIRST_PAGE = (
    "nature genetics Volume 58 | July 2026 | 1458 | 1458\n"
    "Research highlights\n"
    "Machine learning\n"
    "Quantifying \ngene importance \nwith SIGnature\n"
    "Computational methods are essential for analyzing gene signatures in "
    "single-cell RNA sequencing (scRNA-seq) datasets. To enable scalable "
    "cross-dataset analyses of gene importance, Gold et al. present "
    "SIGnature, a framework for scoring the importance of genes by combining "
    "explainable artificial intelligence with foundation model attributions.\n"
)

# A plausible ordinary research-article first page: no section label anywhere.
RESEARCH_FIRST_PAGE = (
    "nature biotechnology\nArticle\nhttps://doi.org/10.1038/s41587-026-01234-5\n"
    "Scoring gene importance by interpreting single-cell foundation models\n"
    "Ada Gold, Brian Silver, Cara Bronze\n"
    "Abstract\nWe present SIGnature, a framework for scoring gene importance.\n"
)


# ---------- the real failure ----------

def test_the_li_2026_research_highlight_is_detected():
    """All four measured signals fire on the exact case that motivated this."""
    v = detect_commentary(
        first_page_text=LI_2026_FIRST_PAGE,
        page_count=1,
        reference_count=0,
        crossref_page="1458-1458",
    )
    assert v.is_commentary is True
    assert v.page_type == COMMENTARY_PAGE_TYPE
    assert v.signals == [
        "label:research-highlight",
        "page-count-1",
        "reference-count-0",
        "single-page-extent-1458-1458",
    ]


def test_the_verdict_reason_names_the_signals_and_the_suggested_type():
    v = detect_commentary(
        first_page_text=LI_2026_FIRST_PAGE, page_count=1,
        reference_count=0, crossref_page="1458-1458",
    )
    reason = v.reason()
    for signal in v.signals:
        assert signal in reason
    assert COMMENTARY_PAGE_TYPE in reason
    assert "--auto-promote" in reason      # tells the operator how to override


@pytest.mark.skipif(
    not Path("papers/li-2026-quantifying-gene-importance-with-signature.pdf").exists(),
    reason="corpus PDF not present in this checkout",
)
def test_end_to_end_on_the_real_pdf():
    """`pdf_shape` + the detector agree on the on-disk PDF, not just a fixture."""
    n, first = pdf_shape(
        Path("papers/li-2026-quantifying-gene-importance-with-signature.pdf")
    )
    assert n == 1
    assert "Research highlights" in first
    v = detect_commentary(
        first_page_text=first, page_count=n,
        reference_count=0, crossref_page="1458-1458",
    )
    assert v.is_commentary is True


# ---------- what must NOT fire (precision) ----------

def test_an_ordinary_research_article_is_not_commentary():
    v = detect_commentary(
        first_page_text=RESEARCH_FIRST_PAGE, page_count=32,
        reference_count=61, crossref_page="1120-1151",
    )
    assert v.is_commentary is False
    assert v.signals == []


def test_page_count_one_alone_never_fires():
    """Genuine one-page Correspondence and Matters Arising exist and belong in
    the wiki as papers. Page count is only ever half of a pair."""
    v = detect_commentary(
        first_page_text="Matters arising is not in the label set.\n",
        page_count=1,
        reference_count=7,
        crossref_page="42-42",
    )
    assert v.is_commentary is False
    assert "page-count-1" in v.considered      # noticed, not acted on


def test_zero_references_alone_never_fires():
    v = detect_commentary(
        first_page_text=RESEARCH_FIRST_PAGE, page_count=14, reference_count=0
    )
    assert v.is_commentary is False
    assert v.considered == ["reference-count-0"]


def test_unknown_reference_count_is_not_read_as_zero():
    """A Crossref cache miss must not manufacture the structural pair."""
    v = detect_commentary(
        first_page_text="No label here.\n", page_count=1, reference_count=None
    )
    assert v.is_commentary is False


def test_no_inputs_at_all_is_not_commentary():
    assert detect_commentary(first_page_text=None).is_commentary is False


def test_strong_label_in_the_body_is_out_of_the_masthead_zone():
    """A research article citing a News & Views in its introduction must not be
    blocked. Only the top of page 1 counts for strong labels."""
    body = ("x" * 4000) + "\nas a recent News & Views argued, the field has moved on\n"
    v = detect_commentary(
        first_page_text=RESEARCH_FIRST_PAGE + body,
        page_count=18, reference_count=52,
    )
    assert v.is_commentary is False


def test_cell_press_correspondence_contact_block_is_not_a_signal():
    """Regression: `Correspondence` on its own line is Cell Press's author-
    contact label, present on page 1 of 8 genuine primaries in the corpus
    (17-39 pages). The label was removed from the weak set for this reason."""
    page = (
        "Anusri Pampari, Anshul Kundaje, Jonathan K. Pritchard, Joanna Wysocka\n"
        "Correspondence\nsahin.naqvi@childrens.harvard.edu (S.N.)\n"
    )
    v = detect_commentary(
        first_page_text=page, page_count=21, reference_count=0, crossref_page="100780"
    )
    assert v.is_commentary is False


def test_article_number_page_field_is_not_a_single_page_extent():
    """Regression: Cell Press puts the article number in Crossref's `page`
    field, so a bare `"100762"` on a 17-page paper must not read as one page."""
    assert is_single_page_extent("100762") is False
    assert is_single_page_extent("1458") is False        # bare, ambiguous
    assert is_single_page_extent("1120-1151") is False   # real multi-page range
    assert is_single_page_extent(None) is False
    assert is_single_page_extent("1458-1458") is True    # explicit same-page
    assert is_single_page_extent("e123–e123") is True    # en dash, alpha prefix


def test_editorial_board_mention_does_not_match_the_weak_label():
    """Weak labels are line-anchored, so prose containing the word is inert."""
    v = detect_commentary(
        first_page_text="Members of the editorial board declared no interest.\n",
        page_count=1, reference_count=0,
    )
    # The structural pair still fires on its own — but not via the label.
    assert not any("label:" in s for s in v.signals)


# ---------- the tiers that DO fire ----------

def test_strong_label_alone_is_sufficient():
    """A 2-page Nature Biotech News & views with references still gets caught."""
    page = (
        "nature biotechnology | https://doi.org/10.1038/s41587-026-03138-9\n"
        "News & views\nGene editing\nGuide DNA - not RNA - expands the CRISPR toolkit\n"
    )
    v = detect_commentary(first_page_text=page, page_count=2, reference_count=9)
    assert v.is_commentary is True
    assert v.signals == ["label:news-and-views"]


def test_news_and_views_spelled_out_and_split_across_lines():
    v = detect_commentary(first_page_text="News and\nViews\nGene editing\n", page_count=2)
    assert v.is_commentary is True


def test_structural_pair_fires_without_any_label():
    v = detect_commentary(
        first_page_text="An unlabelled one-pager that cites nothing.\n",
        page_count=1, reference_count=0,
    )
    assert v.is_commentary is True
    assert v.signals == ["page-count-1", "reference-count-0"]


def test_weak_label_fires_only_once_corroborated():
    """A weak label alone never fires; one structural signal is enough to arm it.

    Uses `Books & Arts` rather than `Editorial`: the latter was promoted to the
    masthead-scoped strong tier (see test_editorial_is_a_strong_masthead_label),
    so it is no longer an example of this rule.
    """
    page = "nature chemical biology Volume 22 | June 2026\nBooks & Arts\nA review\n"
    alone = detect_commentary(first_page_text=page, page_count=2)
    assert alone.is_commentary is False
    assert alone.considered == ["label:books-and-arts"]

    corroborated = detect_commentary(
        first_page_text=page, page_count=2, reference_count=0
    )
    assert corroborated.is_commentary is True
    assert corroborated.signals == ["label:books-and-arts", "reference-count-0"]


def test_editorial_is_a_strong_masthead_label():
    """`editorial-2026-circling-back-to-rna-vaccines`: 2 pages, 9 references,
    extent 673-674, Crossref `journal-article`. No structural signal can ever
    corroborate it, so a weak `Editorial` would miss it forever."""
    page = (
        "nature biotechnology Volume 44 | May 2026 | 673-674 | 673\n"
        "https://doi.org/10.1038/s41587-026-03155-8\n"
        "Editorial\n"
        "Circling back to RNA vaccines\n"
    )
    v = detect_commentary(
        first_page_text=page, page_count=2, reference_count=9, crossref_page="673-674"
    )
    assert v.is_commentary is True
    assert v.signals == ["label:editorial"]


def test_editorial_only_counts_inside_the_masthead_zone():
    """The reason promotion is safe: a body-text `Editorial` line is out of scope.

    The weak tier searched the whole page, which is why it could not be strong
    there. Padding past `_MASTHEAD_CHARS` must leave the guard silent.
    """
    from researchwiki.agents.commentary import _MASTHEAD_CHARS
    body = "A" * (_MASTHEAD_CHARS + 50) + "\nEditorial\n"
    v = detect_commentary(first_page_text=body, page_count=18, reference_count=61)
    assert v.is_commentary is False
    assert v.signals == []


# ---------- the crossref pre-trigger (the "cheap" contract) ----------

def test_ordinary_article_does_not_justify_a_crossref_request():
    """The common ingest path must add zero network calls."""
    assert crossref_lookup_worthwhile(RESEARCH_FIRST_PAGE, 32) is False


@pytest.mark.parametrize("first_page,pages", [
    (RESEARCH_FIRST_PAGE, 1),                      # single page
    (LI_2026_FIRST_PAGE, 1),                       # strong label
    ("News & views\nGene editing\n", 2),           # strong label, multi-page
    ("Editorial\nSomething\n", 2),                 # weak label
])
def test_a_local_pre_trigger_unlocks_the_crossref_lookup(first_page, pages):
    assert crossref_lookup_worthwhile(first_page, pages) is True


def test_structural_signals_are_cache_only_by_default(tmp_path, monkeypatch):
    """`allow_fetch=False` must never reach the network — a cache miss returns
    all-None so the guard degrades to "signal unknown" rather than blocking."""
    import researchwiki.providers.crossref as cr

    monkeypatch.setattr(cr, "crossref_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        cr, "_fetch_crossref_work",
        lambda *a, **k: pytest.fail("network fetch attempted with allow_fetch=False"),
    )
    assert cr.crossref_structural_signals("10.1038/nothing-cached") == {
        "reference_count": None, "page": None, "type": None, "subtype": None,
    }


def test_structural_signals_read_the_fields_the_guard_needs(tmp_path, monkeypatch):
    """`type`/`subtype` are carried for logging only — they are identical for
    the highlight and its 32-page primary, which is why they can't be the rule."""
    import json

    import researchwiki.providers.crossref as cr

    monkeypatch.setattr(cr, "crossref_cache_dir", lambda: tmp_path)
    doi = "10.1038/s41588-026-02698-5"
    payload = {"message": {
        "DOI": doi, "type": "journal-article", "subtype": None,
        "reference-count": 0, "page": "1458-1458",
    }}
    (tmp_path / f"crossref__{cr.safe_cache_key(doi)}.json").write_text(json.dumps(payload))
    got = cr.crossref_structural_signals(doi)
    assert got["reference_count"] == 0
    assert got["page"] == "1458-1458"
    assert got["type"] == "journal-article"
    assert got["subtype"] is None


# ---------- the promotion gate ----------

def _passing_scores():
    return {
        "semantic_available": True, "semantic_score": 0.70,
        "n_drift": 0, "n_graded": MIN_GRADED_CLAIMS,
    }


def _no_broken():
    return SimpleNamespace(broken=[])


def test_commentary_signals_block_a_page_that_passes_every_other_gate():
    gate = should_auto_promote(
        _passing_scores(), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS,
        commentary_signals=["label:research-highlight", "page-count-1"],
    )
    assert gate.promoted is False
    assert len(gate.reasons) == 1
    assert "label:research-highlight" in gate.reasons[0]
    assert COMMENTARY_PAGE_TYPE in gate.reasons[0]


@pytest.mark.parametrize("signals", [None, []])
def test_absent_commentary_signals_leave_the_gate_untouched(signals):
    gate = should_auto_promote(
        _passing_scores(), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS,
        commentary_signals=signals,
    )
    assert gate.promoted is True
    assert gate.reasons == []


def test_debug_does_not_try_to_repair_a_commentary_rejection():
    """No prose rewrite changes what the PDF *is*, so the reason must not land
    in DEBUG's repairable set (which is an allow-list — this pins that)."""
    gate = should_auto_promote(
        _passing_scores(), _no_broken(),
        n_key_contributions=MIN_KEY_CONTRIBUTIONS,
        commentary_signals=["label:news-and-views"],
    )
    assert detect_structural_gate_issues(gate.reasons) == []


# ---------- frontmatter steering ----------

BODY = "## Summary\nText.\n\n## Key Contributions\n- a\n- b\n- c\n- d\n"


def test_page_type_commentary_lands_in_the_frontmatter():
    """Even under `--auto-promote` the page is typed correctly — and because
    `db rebuild` only extracts claims from `type: paper`, this is what stops
    the misattributed claims at the source."""
    page = _build_frontmatter(
        {"title": "Quantifying gene importance with SIGnature", "year": 2026,
         "page_type": COMMENTARY_PAGE_TYPE},
        "li-2026-quantifying-gene-importance-with-signature",
        "single-cell", BODY,
    )
    assert yaml.safe_load(page.split("---", 2)[1])["type"] == COMMENTARY_PAGE_TYPE


def test_frontmatter_still_defaults_to_paper():
    page = _build_frontmatter({"title": "T", "year": 2026}, "x-2026-t", "genomics", BODY)
    assert yaml.safe_load(page.split("---", 2)[1])["type"] == "paper"


def test_sandbox_page_carries_the_type_and_the_signals():
    """The sandbox is where a blocked highlight actually lands, so the verdict
    has to be readable from the file itself, not only from the run log."""
    page = _wrap_with_frontmatter(
        BODY, {"title": "Quantifying gene importance with SIGnature", "year": 2026},
        "li-2026-quantifying-gene-importance-with-signature",
        category="single-cell",
        page_type=COMMENTARY_PAGE_TYPE,
        commentary_signals=["label:research-highlight", "page-count-1"],
    )
    fm = page.split("---", 2)[1]
    assert yaml.safe_load(fm)["type"] == COMMENTARY_PAGE_TYPE
    assert "# commentary guard fired: label:research-highlight, page-count-1" in fm


def test_grader_temp_page_frontmatter_is_unchanged():
    """`grade.py` wraps its scratch file with the same helper and must keep
    `type: paper` — re-typing the grader's input would change what it scores."""
    page = _wrap_with_frontmatter(BODY, {"title": "T", "year": 2026}, "x-2026-t")
    fm = page.split("---", 2)[1]
    assert yaml.safe_load(fm)["type"] == "paper"
    assert "commentary guard" not in fm


# --- Tier 1b: publisher news DOI namespace ---------------------------------
# Added after the guard missed `koralov-2026-hard-to-detect-mutations-in-
# autoimmune-diseases`: a 2-page Nature News & Views opening directly on body
# prose, so no label was in the masthead zone and page-count-1 didn't hold.

from researchwiki.agents.commentary import is_news_namespace_doi


class TestNewsNamespaceDoi:
    def test_nature_news_doi_fires_alone(self):
        """The koralov case: body prose, 2 pages, no Crossref — DOI carries it."""
        v = detect_commentary(
            first_page_text=(
                "Mutations that are acquired throughout life, known as somatic "
                "mutations, are perhaps best known for their role in cancer "
                "development. Writing in Nature, Nicola et al. use a high-fidelity "
                "DNA-sequencing technique to pinpoint hard-to-detect somatic mutations."
            ),
            page_count=2,
            reference_count=None,
            doi="10.1038/d41586-026-01415-w",
        )
        assert v.is_commentary
        assert "doi-news-namespace" in v.signals

    def test_research_namespace_does_not_fire(self):
        """`10.1038/s…` is the research namespace — must stay inert."""
        v = detect_commentary(
            first_page_text="An ordinary research article about genome editing.",
            page_count=14,
            reference_count=62,
            doi="10.1038/s41588-026-02653-4",
        )
        assert not v.is_commentary
        assert v.signals == []

    def test_no_doi_is_not_a_signal(self):
        assert not is_news_namespace_doi(None)
        assert not is_news_namespace_doi("")

    @pytest.mark.parametrize("doi", [
        "10.1038/d41586-025-02975-z",
        "10.1038/D41586-025-02975-Z",
        "https://doi.org/10.1038/d41586-026-01415-w",
        "https://dx.doi.org/10.1038/d41586-026-01415-w",
    ])
    def test_accepted_forms(self, doi):
        assert is_news_namespace_doi(doi)

    @pytest.mark.parametrize("doi", [
        "10.1038/s41586-025-02975-z",      # research namespace
        "10.1101/2024.11.21.24317744",     # medRxiv preprint
        "10.1016/j.cell.2024.01.001",      # Elsevier
        "10.1126/science.abc1234",         # Science
        "10.1371/journal.pgen.1001333",    # PLoS — letter-prefixed suffix, not news
        "10.1038/nbt.3900",                # legacy Nature research DOI
        "10.1038/d4158-026-01415-w",       # too few digits to be a namespace
    ])
    def test_rejected_forms(self, doi):
        """A letter after the slash is NOT sufficient — only verified namespaces."""
        assert not is_news_namespace_doi(doi)

    def test_doi_signal_composes_with_structural(self):
        """DOI fires, and structure we saw is still recorded for the log."""
        v = detect_commentary(
            first_page_text="News & views\nGene editing\nSomething about Cas12.",
            page_count=1,
            reference_count=0,
            doi="10.1038/d41586-026-01415-w",
        )
        assert v.is_commentary
        assert "doi-news-namespace" in v.signals
        assert "page-count-1" in v.signals
        assert "reference-count-0" in v.signals
