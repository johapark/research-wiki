"""URL→DOI candidate hunt and Crossref-validation helpers.

These exercise the slice-1 reconcile-reliability work: `find_url_doi_candidates`
recognizes preprint-server URL forms not in canonical `10.X/Y` shape, and
`verify_doi_via_crossref` confirms a candidate resolves on Crossref before
adoption. Together they fill the gap that produced the Jang 2025 silent
reconcile failure (SSRN preprint with `ssrn.com/abstract=NNNN` in the page
footer but no DOI form anywhere in the PDF).
"""

from __future__ import annotations

import json

import pytest

from researchwiki.pdf.text import find_url_doi_candidates


# ---------- find_url_doi_candidates ----------


def test_ssrn_url_promotes_to_canonical_doi():
    text = (
        "Some abstract text on the first page.\n"
        "This preprint research paper has not been peer reviewed. "
        "Electronic copy available at: https://ssrn.com/abstract=5736492\n"
    )
    candidates = find_url_doi_candidates(text)
    assert ("ssrn-url", "10.2139/ssrn.5736492") in candidates


def test_ssrn_url_dedupes_repeated_footer():
    # Real SSRN PDFs typeset the URL in every page footer — must not
    # explode into 16 separate candidates.
    footer = "Electronic copy available at: https://ssrn.com/abstract=5736492\n"
    text = footer * 16
    candidates = find_url_doi_candidates(text)
    assert candidates == [("ssrn-url", "10.2139/ssrn.5736492")]


def test_arxiv_url_promotes_to_canonical_doi():
    text = "Cite this work: https://arxiv.org/abs/2604.05018v2 (preprint)."
    candidates = find_url_doi_candidates(text)
    assert ("arxiv-url", "10.48550/arXiv.2604.05018") in candidates


def test_arxiv_bare_id_also_promotes():
    # The same regex handles `arXiv:2604.05018` style headers.
    text = "arXiv:2604.05018 [cs.LG]"
    candidates = find_url_doi_candidates(text)
    assert ("arxiv-url", "10.48550/arXiv.2604.05018") in candidates


def test_no_false_positive_on_canonical_doi_text():
    # Text with a real `10.X/Y` DOI shouldn't trigger the URL hunt — that
    # case is handled by `detect_doi`, not `find_url_doi_candidates`.
    text = "https://doi.org/10.1038/s41586-025-09584-w"
    candidates = find_url_doi_candidates(text)
    assert candidates == []


def test_empty_text_returns_empty_list():
    assert find_url_doi_candidates("") == []


def test_multiple_candidates_preserve_pdf_order():
    text = (
        "Reference 1: https://arxiv.org/abs/2401.11111\n"
        "Reference 2: https://ssrn.com/abstract=5000001\n"
        "Reference 3: https://arxiv.org/abs/2402.22222\n"
    )
    candidates = find_url_doi_candidates(text)
    # SSRN block emits first (its regex runs first in the helper); within
    # each block, PDF order is preserved.
    dois = [doi for _, doi in candidates]
    assert "10.2139/ssrn.5000001" in dois
    assert "10.48550/arXiv.2401.11111" in dois
    assert "10.48550/arXiv.2402.22222" in dois
    # arXiv candidates appear in PDF order.
    arxiv_only = [doi for prov, doi in candidates if prov == "arxiv-url"]
    assert arxiv_only == ["10.48550/arXiv.2401.11111", "10.48550/arXiv.2402.22222"]


# ---------- verify_doi_via_crossref ----------


def test_verify_doi_via_crossref_uses_cache(tmp_path, monkeypatch):
    """Verify reads from the on-disk cache when present, never hitting the
    network. Lets us drive the function deterministically without curl.
    """
    from researchwiki.providers import crossref

    # Redirect the cache directory to a tmp_path the test owns.
    monkeypatch.setattr(crossref, "crossref_cache_dir", lambda: tmp_path)

    # Pre-seed a hit-shape payload mimicking Crossref's response for an SSRN DOI.
    doi = "10.2139/ssrn.5736492"
    cache_file = tmp_path / f"crossref__{crossref.safe_cache_key(doi.lower())}.json"
    cache_file.write_text(json.dumps({
        "message": {
            "DOI": "10.2139/ssrn.5736492",
            "title": ["Accurate detection of sub-1% frequency somatic mutations by WGS"],
            "posted": {"date-parts": [[2025]]},
            "type": "posted-content",
        }
    }))

    out = crossref.verify_doi_via_crossref(doi)
    assert out is not None
    assert out["doi"] == "10.2139/ssrn.5736492"
    assert out["title"] is not None and "somatic mutations" in out["title"].lower()
    assert out["year"] == 2025


def test_verify_doi_via_crossref_returns_none_for_404_cache(tmp_path, monkeypatch):
    """Cached 404 sentinel (`{"message": {}}`) → verifier returns None."""
    from researchwiki.providers import crossref

    monkeypatch.setattr(crossref, "crossref_cache_dir", lambda: tmp_path)

    doi = "10.9999/this-doi-does-not-exist"
    cache = tmp_path / f"crossref__{crossref.safe_cache_key(doi.lower())}.json"
    cache.write_text(json.dumps({"message": {}}))

    assert crossref.verify_doi_via_crossref(doi) is None


def test_verify_doi_via_crossref_raises_on_provider_outage(tmp_path, monkeypatch):
    import researchwiki.providers.crossref as crossref
    from researchwiki.providers._http import StructuredProviderUnavailable

    class FailedRequest:
        returncode = 0
        stdout = "\n503"
        stderr = ""

    monkeypatch.setattr(crossref, "crossref_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(crossref.subprocess, "run", lambda *a, **k: FailedRequest())
    with pytest.raises(StructuredProviderUnavailable, match="HTTP 503"):
        crossref._fetch_crossref_work("10.9999/outage", retries=1)


def test_verify_doi_via_crossref_empty_doi_returns_none():
    from researchwiki.providers.crossref import verify_doi_via_crossref
    assert verify_doi_via_crossref("") is None


def test_verify_doi_via_crossref_picks_first_available_year(tmp_path, monkeypatch):
    """When `posted` is absent, fall through to `published-print`/`-online`."""
    from researchwiki.providers import crossref

    monkeypatch.setattr(crossref, "crossref_cache_dir", lambda: tmp_path)
    doi = "10.1038/test-doi"
    cache = tmp_path / f"crossref__{crossref.safe_cache_key(doi.lower())}.json"
    cache.write_text(json.dumps({
        "message": {
            "DOI": "10.1038/test-doi",
            "title": ["A paper with no `posted` field"],
            "published-online": {"date-parts": [[2026, 4]]},
            "type": "journal-article",
        }
    }))
    out = crossref.verify_doi_via_crossref(doi)
    assert out is not None
    assert out["year"] == 2026
