"""Regression coverage for page-level index upserts."""

from pathlib import Path

import numpy as np

from researchwiki.index.pages_bm25 import TantivySearchBackend
from researchwiki.index.types import Document
from researchwiki.index import pages_semantic
from researchwiki.wiki import Page


def _doc(*, category="methods", title="Old title", body="olduniquetoken"):
    return Document(
        stem="smith-2026-example-paper",
        category=category,
        page_type="paper",
        title=title,
        authors="Smith",
        year=2026,
        summary=body,
        body=body,
    )


def _page(tmp_path: Path, stem: str, text: str, *, category="methods") -> Page:
    return Page(
        path=tmp_path / category / f"{stem}.md",
        stem=stem,
        category=category,
        fm={"type": "paper", "title": f"Title {stem}", "keywords": ["testing"]},
        body=f"## Summary\n\n{text}\n\n## Key Contributions\n\n{text}\n",
    )


def test_bm25_add_replaces_same_stem_instead_of_duplicating(tmp_path):
    backend = TantivySearchBackend(tmp_path / "tantivy")
    backend.build([_doc()])

    backend.add(_doc(
        category="new-category",
        title="Replacement title",
        body="replacementuniquetoken",
    ))

    assert backend.query("olduniquetoken") == []
    hits = backend.query("replacementuniquetoken")
    assert len(hits) == 1
    assert hits[0].key == "new-category/smith-2026-example-paper"


def test_semantic_upsert_embeds_only_changed_pages(tmp_path, monkeypatch):
    cache = tmp_path / "semantic"
    monkeypatch.setattr(pages_semantic, "semantic_cache_dir", lambda: cache)
    monkeypatch.setattr(pages_semantic.semantic, "is_available", lambda: True)

    calls: list[list[str]] = []

    def fake_embed(texts):
        calls.append(list(texts))
        return np.asarray([
            [float(len(text)), float("replacementuniquetoken" in text)]
            for text in texts
        ], dtype=np.float32)

    monkeypatch.setattr(pages_semantic.semantic, "embed_texts", fake_embed)

    changed_old = _page(tmp_path, "changed", "olduniquetoken source material")
    unchanged = _page(tmp_path, "unchanged", "stable source material")
    pages_semantic.build_index([changed_old, unchanged])
    calls.clear()

    changed_new = _page(tmp_path, "changed", "replacementuniquetoken source material")
    result = pages_semantic.upsert_pages([changed_new, unchanged])

    assert result["mode"] == "incremental"
    assert result["n_embedded"] == 1
    assert len(calls) == 1 and len(calls[0]) == 1
    assert "replacementuniquetoken" in calls[0][0]

    arr, rows = pages_semantic.load_index()
    assert arr.shape == (2, 2)
    assert {row["stem"] for row in rows} == {"changed", "unchanged"}
    changed_i = next(i for i, row in enumerate(rows) if row["stem"] == "changed")
    assert arr[changed_i, 1] == 1.0


def test_semantic_upsert_removes_old_category_key_for_same_stem(tmp_path, monkeypatch):
    cache = tmp_path / "semantic"
    monkeypatch.setattr(pages_semantic, "semantic_cache_dir", lambda: cache)
    monkeypatch.setattr(pages_semantic.semantic, "is_available", lambda: True)
    monkeypatch.setattr(
        pages_semantic.semantic,
        "embed_texts",
        lambda texts: np.ones((len(texts), 2), dtype=np.float32),
    )

    old = _page(tmp_path, "moving", "same content", category="old-category")
    pages_semantic.build_index([old])
    moved = _page(tmp_path, "moving", "same content", category="new-category")
    pages_semantic.upsert_pages([moved])

    _, rows = pages_semantic.load_index()
    assert [row["key"] for row in rows] == ["new-category/moving"]


def test_corrupt_semantic_metadata_is_treated_as_rebuildable_cache_miss(
    tmp_path, monkeypatch,
):
    cache = tmp_path / "semantic"
    cache.mkdir()
    (cache / pages_semantic.PAGES_META).write_text("{broken", encoding="utf-8")
    np.save(cache / pages_semantic.PAGES_NPY, np.ones((1, 2), dtype=np.float32))
    monkeypatch.setattr(pages_semantic, "semantic_cache_dir", lambda: cache)

    assert pages_semantic.load_index() is None
    assert pages_semantic.index_exists() is False
