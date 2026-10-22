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


def _embedding_model(monkeypatch, *, started=None, release=None):
    """A stub model; with `started`/`release`, the first call blocks."""
    import numpy as np
    calls = []

    def embed(texts):
        calls.append(list(texts))
        if started is not None and len(calls) == 1:
            started.set()
            assert release.wait(5)
        return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(pages_semantic.semantic, "is_available", lambda: True)
    monkeypatch.setattr(pages_semantic.semantic, "embed_texts", embed)
    return calls


def test_embedding_runs_outside_the_corpus_guard(corpus, monkeypatch):
    """A long `reindex` must not block promotes and grade persistence.

    While the model encodes, another writer takes the corpus guard and commits.
    Before, embedding ran inside the guard, so that writer waited out the whole
    embedding pass (minutes on a full rebuild).
    """
    started, release, committed = threading.Event(), threading.Event(), threading.Event()
    _embedding_model(monkeypatch, started=started, release=release)

    def writer():
        with mut.corpus_lock():
            committed.set()

    with ThreadPoolExecutor(2) as pool:
        indexing = pool.submit(incremental.rebuild_page_indexes, semantic=True)
        assert started.wait(5)
        pool.submit(writer)
        assert committed.wait(2), "the corpus guard was held through embedding"
        release.set()
        indexing.result(timeout=10)
    assert pages_semantic.load_index() is not None


def test_page_edited_during_embedding_publishes_its_current_text(corpus, monkeypatch):
    """Vectors embedded outside the guard are matched to the text they encode.

    A page rewritten while the model runs is re-embedded under the guard, so
    the published row never pairs new metadata with a vector for old text.
    """
    started, release = threading.Event(), threading.Event()
    calls = _embedding_model(monkeypatch, started=started, release=release)

    with ThreadPoolExecutor(1) as pool:
        indexing = pool.submit(incremental.rebuild_page_indexes, semantic=True)
        assert started.wait(5)
        with mut.corpus_lock():
            corpus.write_text(corpus.read_text().replace("Original", "Rewritten"))
        release.set()
        indexing.result(timeout=10)

    assert len(calls) == 2, "the edited page must be embedded again"
    assert "Rewritten" in calls[1][0] and "Original" not in calls[1][0]
    from researchwiki.wiki import read_page
    _, rows = pages_semantic.load_index()
    current = pages_semantic._index_row(read_page(corpus), pages_semantic.page_index_text(read_page(corpus)))
    assert rows == [current]


def test_prepared_vectors_make_an_unchanged_upsert_embed_nothing_under_the_guard(corpus, monkeypatch):
    """The common ingest path: changed pages are embedded once, outside the guard."""
    calls = _embedding_model(monkeypatch)
    incremental.rebuild_page_indexes(semantic=True)
    calls.clear()
    corpus.write_text(corpus.read_text().replace("Original", "Updated"))
    incremental.update_page_indexes([corpus])
    assert len(calls) == 1 and "Updated" in calls[0][0]


# ---------- rebuild must not hold the write lock while reading papers/ ----------
#
# `_upsert_claims` needs each paper's real content hash to decide whether a
# recorded grade still describes the current PDF. Hashing inside the write
# transaction held one SQLite lock for the whole walk plus a full read of
# `papers/` — 2.4 s for 667 PDFs here, far worse on a synced folder — while
# parallel ingest workers call `write_iteration`, whose only protection was a
# 5 s busy_timeout.


def test_pdf_digests_are_computed_before_the_write_transaction(tmp_path, monkeypatch):
    """No file_sha256 call may happen while the transaction is open."""
    from researchwiki.db import rebuild as rb

    monkeypatch.chdir(tmp_path)
    from researchwiki.paths import ensure_scaffold
    ensure_scaffold()
    (tmp_path / "wiki" / "methods").mkdir(parents=True, exist_ok=True)
    (tmp_path / "wiki" / "methods" / "smith-2024-a-paper.md").write_text(
        "---\ntype: paper\nyear: 2024\n---\n\n## Summary\nbody\n", encoding="utf-8")
    (tmp_path / "papers" / "smith-2024-a-paper.pdf").write_bytes(b"%PDF-1.4\n")

    events = []
    real_hash = rb.file_sha256

    def tracking_hash(path):
        events.append("hash")
        return real_hash(path)

    real_get = rb.get_connection

    def tracking_get():
        events.append("connect")
        return real_get()

    monkeypatch.setattr(rb, "file_sha256", tracking_hash)
    monkeypatch.setattr(rb, "get_connection", tracking_get)
    rb._rebuild_locked(verbose=False)

    assert "connect" in events, "sanity: the rebuild opened a connection"
    assert "hash" in events, "sanity: the rebuild hashed the PDF"
    assert events.index("hash") < events.index("connect"), (
        "every PDF must be hashed before the write connection is opened")


def test_write_iteration_retries_a_busy_commit_instead_of_dying(monkeypatch):
    import sqlite3

    from researchwiki.db import iterations as it

    sleeps = []
    monkeypatch.setattr(it.time, "sleep", sleeps.append)

    class Conn:
        def __init__(self):
            self.attempts = 0

        def execute(self, *_a, **_k):
            class Cur:
                lastrowid = 7
            return Cur()

        def commit(self):
            self.attempts += 1
            if self.attempts < 3:
                raise sqlite3.OperationalError("database is locked")

        def close(self):
            pass

    conn = Conn()
    row_id = it.write_iteration(
        attempt_id="a", paper_stem="s", pdf_filename="s.pdf", iteration=1,
        role="commit", conn=conn,
    )
    assert row_id == 7
    assert conn.attempts == 3, "retried until the lock cleared"
    assert sleeps, "backed off between attempts"


def test_write_iteration_does_not_retry_a_real_error(monkeypatch):
    import sqlite3

    import pytest as _pytest

    from researchwiki.db import iterations as it

    monkeypatch.setattr(it.time, "sleep", lambda *_a: None)

    class Conn:
        attempts = 0

        def execute(self, *_a, **_k):
            class Cur:
                lastrowid = 1
            return Cur()

        def commit(self):
            Conn.attempts += 1
            raise sqlite3.OperationalError("no such column: nonsense")

        def close(self):
            pass

    with _pytest.raises(sqlite3.OperationalError, match="no such column"):
        it.write_iteration(
            attempt_id="a", paper_stem="s", pdf_filename="s.pdf", iteration=1,
            role="commit", conn=Conn(),
        )
    assert Conn.attempts == 1, "a schema error must fail immediately"
