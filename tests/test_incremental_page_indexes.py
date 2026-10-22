"""Regression coverage for page-level index upserts."""

from pathlib import Path

import numpy as np

from researchwiki.index import pages_bm25
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


def test_skipped_maintenance_refresh_reuses_unchanged_vectors_and_prunes_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(pages_semantic.semantic, "is_available", lambda: True)
    calls = []
    def embed(texts):
        calls.append(list(texts))
        return np.ones((len(texts), 2), dtype=np.float32)
    monkeypatch.setattr(pages_semantic.semantic, "embed_texts", embed)
    stable = _page(tmp_path, "stable", "stable content")
    changed = _page(tmp_path, "changed", "old content")
    removed = _page(tmp_path, "removed", "deleted content")
    pages_semantic.build_index([stable, changed, removed])
    pages_semantic.mark_stale()
    assert pages_semantic.load_index() is None
    current = _page(tmp_path, "changed", "new content")
    monkeypatch.setattr(pages_semantic, "read_pages", lambda: [stable, current])
    calls.clear()
    result = pages_semantic.upsert_pages([current])
    assert result["n_embedded"] == 1
    assert len(calls) == 1 and len(calls[0]) == 1
    assert "new content" in calls[0][0]
    _, rows = pages_semantic.load_index()
    assert {row["stem"] for row in rows} == {"stable", "changed"}


def test_mutation_reconcile_keeps_page_vectors_available(tmp_path, monkeypatch):
    """A promote/remove must not switch page KNN off for the rest of a batch.

    Removed pages and changed text lose their rows; an untouched neighbour and a
    back-link splice (which edits no indexed section) keep theirs, and the index
    stays loadable. Only an indexed page whose vector went out of date marks it
    stale, so the next upsert re-embeds it.
    """
    monkeypatch.setattr(pages_semantic.semantic, "is_available", lambda: True)
    monkeypatch.setattr(pages_semantic.semantic, "embed_texts",
                        lambda texts: np.ones((len(texts), 2), dtype=np.float32))
    stable = _page(tmp_path, "stable", "stable content")
    spliced = _page(tmp_path, "spliced", "spliced content")
    removed = _page(tmp_path, "removed", "removed content")
    pages_semantic.build_index([stable, spliced, removed])

    spliced.body += "\n## Related Papers\n\n- [[methods/new]] — cites this paper\n"
    fresh = _page(tmp_path, "fresh", "brand new paper")
    pages_semantic.drop_outdated([spliced, fresh], {"removed"})
    _, rows = pages_semantic.load_index()
    assert {row["stem"] for row in rows} == {"stable", "spliced"}

    edited = _page(tmp_path, "stable", "rewritten summary")
    pages_semantic.drop_outdated([edited], set())
    assert pages_semantic.load_index() is None, "an outdated vector must not be served"


def test_hand_edited_frontmatter_scalars_keep_the_index_loadable(tmp_path, monkeypatch):
    """A blank `type:` (None) or an unquoted `title: 2021` (int) is one page's
    typo, not a reason to discard every page's vector.

    `load_bundle` rejects the whole bundle when any row field is not a string,
    so one such page used to disable page KNN for the wiki and make every later
    upsert re-embed the corpus.
    """
    monkeypatch.setattr(pages_semantic.semantic, "is_available", lambda: True)
    batches = []

    def embed(texts):
        batches.append(len(texts))
        return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(pages_semantic.semantic, "embed_texts", embed)
    normal = _page(tmp_path, "normal", "ordinary content")
    blank_type = _page(tmp_path, "blank-type", "content with a blank type")
    blank_type.fm["type"] = None
    numeric_title = _page(tmp_path, "numeric-title", "content titled by a year")
    numeric_title.fm["title"] = 2021
    pages_semantic.build_index([normal, blank_type, numeric_title])

    _, rows = pages_semantic.load_index()
    by_stem = {row["stem"]: row for row in rows}
    assert by_stem["blank-type"]["page_type"] == "paper"
    assert by_stem["numeric-title"]["title"] == "2021"

    batches.clear()
    result = pages_semantic.upsert_pages([normal, blank_type, numeric_title])
    assert result["mode"] == "incremental" and batches == []


# ---------- a rebuild must not expose a partial index to readers ----------
#
# `build` used to unlink the directory's contents in place. Writers hold
# `exclusive_lock(search_index_dir())`, but readers take none: category suggest
# at ingest, `search`, and the MCP server all just open the directory. So an
# in-place wipe gave a concurrent reader a window where the index looked absent
# — which ingest treats as "abstain", filing the paper under `other` — or,
# mid-unlink, a directory that opened and then failed every query.


def test_rebuild_keeps_the_old_index_queryable_until_the_swap(tmp_path):
    """The reader sees the previous index or the new one, never neither."""
    path = tmp_path / "tantivy"
    backend = TantivySearchBackend(path)
    backend.build([_doc(body="olduniquetoken")])
    assert backend.exists()

    observed = []

    # Observe what a *separate* reader sees at the moment the writer is mid-build.
    reader = TantivySearchBackend(path)
    real_swap = pages_bm25.swap_in_directory

    def observing_swap(staged, target):
        observed.append(("before-swap", reader.exists(),
                         len(reader.query("olduniquetoken", limit=5))))
        real_swap(staged, target)
        observed.append(("after-swap", reader.exists(), None))

    import pytest
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(pages_bm25, "swap_in_directory", observing_swap)
        backend.build([_doc(body="newuniquetoken")])
    finally:
        monkeypatch.undo()

    assert observed[0] == ("before-swap", True, 1), (
        "the old index must still answer queries while the new one is building")
    assert observed[1][1] is True
    # And the new content is what is published.
    fresh = TantivySearchBackend(path)
    assert len(fresh.query("newuniquetoken", limit=5)) == 1
    assert fresh.query("olduniquetoken", limit=5) == []


def test_a_crashed_rebuild_leaves_the_old_index_intact(tmp_path):
    path = tmp_path / "tantivy"
    backend = TantivySearchBackend(path)
    backend.build([_doc(body="olduniquetoken")])

    import pytest
    monkeypatch = pytest.MonkeyPatch()

    def boom(staged, target):
        raise OSError("disk full at the worst moment")

    try:
        monkeypatch.setattr(pages_bm25, "swap_in_directory", boom)
        with pytest.raises(OSError):
            backend.build([_doc(body="newuniquetoken")])
    finally:
        monkeypatch.undo()

    reader = TantivySearchBackend(path)
    assert reader.exists(), "a failed rebuild must not destroy the live index"
    assert len(reader.query("olduniquetoken", limit=5)) == 1


def test_rebuild_leaves_no_staging_directories_behind(tmp_path):
    path = tmp_path / "tantivy"
    backend = TantivySearchBackend(path)
    backend.build([_doc()])
    backend.build([_doc()])
    leftovers = [p.name for p in path.parent.iterdir() if "-building-" in p.name
                 or "-retired-" in p.name]
    assert leftovers == []


def test_a_symlinked_index_dir_keeps_its_symlink(tmp_path):
    """A user who relocates the index with a symlink (the same thing CLAUDE.md
    documents for wiki/, papers/ and inbox/) must keep that layout: renaming
    the link aside would replace it with a real directory and strand the
    link's own target."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    link = tmp_path / "tantivy"
    link.symlink_to(elsewhere, target_is_directory=True)

    backend = TantivySearchBackend(link)
    backend.build([_doc(body="olduniquetoken")])
    assert link.is_symlink(), "the index path must still be a symlink"
    assert len(TantivySearchBackend(link).query("olduniquetoken", limit=5)) == 1
    # The content really lives at the link's destination.
    assert any(elsewhere.iterdir())

    backend.build([_doc(body="newuniquetoken")])
    assert link.is_symlink()
    assert len(TantivySearchBackend(link).query("newuniquetoken", limit=5)) == 1


def test_a_failed_publish_puts_the_old_index_back(tmp_path):
    """The gap between the two renames is the only moment the target is
    absent. If the publish fails there, the old index is restored rather than
    leaving the caller with nothing."""
    import pytest

    from researchwiki.index import types as idx_types

    path = tmp_path / "tantivy"
    backend = TantivySearchBackend(path)
    backend.build([_doc(body="olduniquetoken")])

    real_replace = idx_types.Path.replace
    calls = {"n": 0}

    def fail_on_publish(self, target):
        calls["n"] += 1
        if calls["n"] == 2:          # 1 = retire the old, 2 = publish the new
            raise OSError("publish failed")
        return real_replace(self, target)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(idx_types.Path, "replace", fail_on_publish)
        with pytest.raises(OSError):
            backend.build([_doc(body="newuniquetoken")])
    finally:
        monkeypatch.undo()

    reader = TantivySearchBackend(path)
    assert reader.exists(), "the old index must be restored, not left absent"
    assert len(reader.query("olduniquetoken", limit=5)) == 1
