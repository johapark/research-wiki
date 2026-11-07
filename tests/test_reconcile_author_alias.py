"""An unavailable metadata provider must not turn a bilingual byline into an unknown stem."""

from types import SimpleNamespace

from researchwiki.agents.phases import reconcile


def test_reconcile_uses_romanized_surname_when_s2_is_unavailable(tmp_path, monkeypatch):
    title = "In vivo CAR-T double knockout screen maps synergistic antitumour gene pairs"
    doi = "10.1038/s41586-026-10956-z"
    llm_meta = {
        **reconcile._empty_metadata(),
        "title": title,
        "first_author_surname": "Zhou",
        "all_authors": ["Liqun Zhou (周立群)", "Paul A. Renauer"],
        "year": 2026,
        "doi": doi,
    }
    monkeypatch.setattr(
        reconcile, "extract_pdf",
        lambda *_args, **_kwargs: (f"{title}\nLiqun Zhou (周立群)\n{doi}", {}),
    )
    monkeypatch.setattr(reconcile, "propose_metadata_llm", lambda _text: llm_meta)
    monkeypatch.setattr(
        reconcile, "SemanticScholarProvider",
        lambda: SimpleNamespace(get_by_doi=lambda _doi: None),
    )
    monkeypatch.setattr(
        reconcile, "_optional_crossref_metadata",
        lambda _doi: {"venue": "Nature"},
    )
    monkeypatch.setattr(
        reconcile, "_detect_commentary_shape",
        lambda *_args: (SimpleNamespace(page_type=None, signals=[]), 46),
    )

    result = reconcile.reconcile_metadata(tmp_path / "paper.pdf", use_llm=True)

    assert result["stem"] == "zhou-2026-in-vivo-car-t-double-knockout"
    assert result["authors"].startswith("Liqun Zhou (周立群), Paul A. Renauer")
    assert result["authors_origin"] == "llm-extractor"
    assert "s2" not in result["sources"]
