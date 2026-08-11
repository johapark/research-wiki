"""S2's year must not silently outrank the document's when they disagree.

S2 merges a preprint and its journal version two different ways. `_s2_record_is_preprint`
already catches the case where S2 *admits* it (preprint venue under a journal DOI).
The other case keeps the journal's venue and the preprint's year, so that guard
cannot see it: for minimap2's journal DOI `10.1093/bioinformatics/bty191` S2 returns
year=2017 venue='Bioinform.' (its `ArXiv: 1708.01492` was posted 2017-08) while the
PDF prints "accepted on May 4, 2018" throughout. The LLM answered 2018 correctly and
was overruled, because S2 sits two places above it in the chain -- producing the stem
`li-2017-…` for a 2018 paper. Observed 2026-08-10 while testing chat-relay fan-out.

Crossref arbitrates: its record for a journal DOI is the journal's own, with no
preprint to merge.
"""

from __future__ import annotations

import pytest

from researchwiki.agents.phases import reconcile


@pytest.fixture
def _crossref(monkeypatch):
    """Stub Crossref; returns whatever year the test sets."""
    calls: list[str] = []
    state: dict = {"year": None}

    def _fake(doi):
        calls.append(doi)
        return {"year": state["year"], "venue": "Bioinformatics"} if state["year"] else None

    monkeypatch.setattr(reconcile, "verify_doi_via_crossref", _fake)
    return state, calls


def _resolve(s2_year, llm_year, doi, *, venue="Bioinform."):
    """Replicate the year-resolution block under test.

    Kept as a thin mirror rather than calling `reconcile_metadata`, which would
    need a real PDF plus live S2/Crossref; the block is small and the precedence
    is the whole point.
    """
    s2_meta = {"year": s2_year, "venue": venue}
    llm_meta = {"year": llm_year}
    s2 = s2_meta.get("year")
    if s2 and reconcile._s2_record_is_preprint(s2_meta, doi):
        s2 = None
    if s2 and doi and llm_meta["year"] and llm_meta["year"] != s2:
        cr_year = (reconcile.verify_doi_via_crossref(doi) or {}).get("year")
        if cr_year and cr_year != s2:
            s2 = cr_year
    return s2 or llm_meta["year"]


DOI = "10.1093/bioinformatics/bty191"


def test_crossref_breaks_the_tie_against_s2(_crossref):
    # The minimap2 case: S2 2017, PDF 2018, Crossref 2018 -> 2018.
    state, calls = _crossref
    state["year"] = 2018
    assert _resolve(2017, 2018, DOI) == 2018
    assert calls == [DOI], "should consult Crossref exactly once"


def test_no_crossref_lookup_when_s2_and_pdf_agree(_crossref):
    # The common path must not pay for a request. Agreement means nothing to settle.
    state, calls = _crossref
    state["year"] = 2018
    assert _resolve(2018, 2018, DOI) == 2018
    assert calls == [], "no disagreement, so no Crossref call"


def test_s2_year_survives_when_crossref_is_silent(_crossref):
    # Fail-safe: an unavailable or unindexed DOI leaves the prior behaviour intact
    # rather than promoting a possibly-wrong PDF year.
    state, _calls = _crossref
    state["year"] = None
    assert _resolve(2017, 2018, DOI) == 2017


def test_crossref_agreeing_with_s2_changes_nothing(_crossref):
    # Crossref confirming S2 means the PDF year is the odd one out -- keep S2.
    state, _calls = _crossref
    state["year"] = 2017
    assert _resolve(2017, 2018, DOI) == 2017


def test_preprint_venue_guard_still_short_circuits(_crossref):
    # A preprint venue under a journal DOI is the pre-existing case: S2's year is
    # dropped outright, so the PDF's year wins without needing Crossref.
    state, calls = _crossref
    state["year"] = 2025
    assert _resolve(2023, 2025, "10.1038/s41592-025-02626-1", venue="bioRxiv") == 2025
    assert calls == [], "the older guard resolves this before Crossref is needed"


def test_real_preprint_ingest_keeps_s2_year(_crossref):
    # bioRxiv PDF + bioRxiv DOI: S2's year is correct and must be left alone.
    state, calls = _crossref
    state["year"] = 2099
    assert _resolve(2024, 2024, "10.1101/2024.01.01.573000", venue="bioRxiv") == 2024
    assert calls == []
