"""Canonical commits survive failed maintenance; recovery never needs a model."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from researchwiki import mutation as mut
from researchwiki.db.connection import get_connection
from researchwiki.db.rebuild import rebuild
from researchwiki.index import incremental, pages_semantic


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    page = tmp_path / "wiki/other/example.md"
    page.parent.mkdir(parents=True)
    page.write_text("---\ntitle: Example\ntype: paper\n---\n## Key Contributions\n- Original meaningful claim.\n")
    monkeypatch.setattr(pages_semantic.semantic, "is_available", lambda: False)
    return page


def test_postcommit_failure_retains_files_and_retryable_receipt(corpus, monkeypatch):
    from researchwiki import projections
    original = projections.reconcile
    monkeypatch.setattr(projections, "reconcile", lambda *a, **kw: (_ for _ in ()).throw(OSError("repair failed")))
    with pytest.raises(OSError, match="repair failed"):
        with mut.mutation([corpus], operation="promote", details={"repair_projections": True}) as snap:
            corpus.write_text(corpus.read_text().replace("Original", "Published"))
            snap.mark_committed()
            snap.reconcile()
    assert "Published" in corpus.read_text()
    assert mut.pending_journals()[0]["status"] == "committed"
    assert "failed" in " ".join(mut.recover_pending())
    assert mut.pending_journals()
    # Recovery uses the current page, even when an older receipt is pending.
    corpus.write_text(corpus.read_text().replace("Published", "Newer"))
    monkeypatch.setattr(projections, "reconcile", original)
    mut.recover_pending()
    assert not mut.pending_journals()
    conn = get_connection()
    try:
        assert "Newer" in conn.execute("SELECT text FROM claims").fetchone()[0]
    finally:
        conn.close()
    assert mut.recover_pending() == []


def test_legacy_active_recovery_restores_then_repairs(corpus):
    rebuild()
    snap = mut.snapshot([corpus], operation="promote", details={"stem": "example"})
    corpus.write_text(corpus.read_text().replace("Original", "Interrupted"))
    rebuild()
    mut.recover_pending()
    conn = get_connection()
    try:
        assert "Original" in conn.execute("SELECT text FROM claims").fetchone()[0]
    finally:
        conn.close()
    assert not snap.journal_path.exists()


def test_reindex_waits_for_rollback_before_reading(corpus):
    entered, release, started = threading.Event(), threading.Event(), threading.Event()
    def mutation():
        with mut.mutation([corpus], operation="test"):
            corpus.write_text(corpus.read_text().replace("Original", "Rolledbacktoken"))
            entered.set()
            assert release.wait(5)
    def reindex():
        started.set()
        return incremental.rebuild_page_indexes(semantic=False)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(mutation)
        assert entered.wait(5)
        second = pool.submit(reindex)
        assert started.wait(5)
        release.set()
        first.result(timeout=10)
        second.result(timeout=10)
    backend = incremental.get_default_backend()
    assert backend.query("Rolledbacktoken") == []
    assert backend.query("Original")


def test_empty_reindex_clears_hits_and_skipped_semantic_is_unavailable(corpus):
    incremental.rebuild_page_indexes(semantic=False)
    corpus.unlink()
    result = incremental.rebuild_page_indexes(semantic=False)
    assert result["n_pages"] == 0
    assert incremental.get_default_backend().query("Original") == []
    assert not pages_semantic.index_exists()


def test_failed_commit_marker_restores_canonical_files(corpus, monkeypatch):
    original = mut.Snapshot._persist
    def persist(snap, status):
        if status == "committed":
            raise OSError("journal write failed")
        original(snap, status)
    monkeypatch.setattr(mut.Snapshot, "_persist", persist)
    with pytest.raises(OSError, match="journal write"):
        with mut.mutation([corpus], operation="promote") as snap:
            corpus.write_text("replacement")
            snap.mark_committed()
    assert "Original" in corpus.read_text()


def test_removal_file_failure_preserves_old_database(corpus):
    from researchwiki.removal import scan, apply
    rebuild()
    plan = scan("example")
    with pytest.raises(RuntimeError, match="file failure"):
        with mut.mutation(plan.touched_paths, operation="remove"):
            apply(plan, sync_db=False)
            raise RuntimeError("file failure")
    assert corpus.exists()
    conn = get_connection()
    try:
        assert conn.execute("SELECT text FROM claims").fetchone()[0] == "Original meaningful claim."
    finally:
        conn.close()


def test_journal_bypass_still_updates_derived_views(corpus, monkeypatch):
    monkeypatch.setenv("RW_MUTATION_JOURNAL", "0")
    with mut.mutation([corpus], operation="promote", details={"repair_projections": True}) as snap:
        snap.mark_committed()
        snap.reconcile()
    conn = get_connection()
    try:
        assert conn.execute("SELECT text FROM claims").fetchone()[0] == "Original meaningful claim."
    finally:
        conn.close()
    assert mut.pending_journals() == []


def test_incomplete_database_scan_cannot_authorize_deletion(corpus, monkeypatch):
    import importlib
    module = importlib.import_module("researchwiki.db.rebuild")
    rebuild()
    corpus.unlink()
    def walk(root, *, onerror):
        onerror(PermissionError("incomplete scan"))
        yield str(root), [], []
    monkeypatch.setattr(module.os, "walk", walk)
    with pytest.raises(PermissionError, match="incomplete scan"):
        rebuild()
    conn = get_connection()
    try:
        assert conn.execute("SELECT count(*) FROM claims").fetchone()[0] == 1
    finally:
        conn.close()


def test_status_recognizes_atomic_bundle_and_stale_marker(corpus, tmp_path, monkeypatch):
    import numpy as np
    from researchwiki.index.bundle import save_bundle
    from researchwiki.tasks import status
    cache = tmp_path / "semantic"
    monkeypatch.setattr(status, "semantic_cache_dir", lambda: cache)
    save_bundle(cache / pages_semantic.PAGES_BUNDLE, np.ones((1, 2)), {
        "model": pages_semantic.semantic.DEFAULT_MODEL, "dim": 2, "rows": [{}],
    })
    _, semantic = status._index_status()
    assert semantic["exists"] and semantic["n_pages"] == 1
    (cache / pages_semantic.PAGES_STALE).write_text("stale")
    _, semantic = status._index_status()
    assert semantic["stale"] and not semantic["exists"]
