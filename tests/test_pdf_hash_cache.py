"""Read-path PDF hashes persist across processes and never outlive the bytes."""

import os
import sqlite3

import pytest

from researchwiki import fsatomic
from researchwiki.db import get_connection
from researchwiki.db import pdf_hashes
from researchwiki.db.pdf_hashes import PdfHashCache
from researchwiki.db.rebuild import rebuild


@pytest.fixture
def counted(monkeypatch):
    """Count real hashes; the cache must avoid them, not change their result."""
    calls = []
    real = fsatomic.file_sha256

    def hashing(path):
        calls.append(str(path))
        return real(path)

    monkeypatch.setattr(pdf_hashes, "file_sha256", hashing)
    return calls


def lookup(path, conn=None):
    own = conn is None
    conn = conn or get_connection()
    try:
        cache = PdfHashCache(conn)
        digest = cache.sha256(path)
        cache.flush()
        return digest
    finally:
        if own:
            conn.close()


def test_hash_is_reused_by_a_later_connection(tmp_path, counted):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF original")
    first = lookup(pdf)
    second = lookup(pdf)  # a new connection: what a later CLI run sees
    assert first == second == fsatomic.file_sha256(pdf)
    assert len(counted) == 1


@pytest.mark.parametrize("change", ["content", "same-size-content", "replaced-file"])
def test_any_stat_change_forces_a_rehash(tmp_path, counted, change):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF original")
    lookup(pdf)
    before = pdf.stat()
    if change == "content":
        pdf.write_bytes(b"%PDF a longer replacement")
    elif change == "same-size-content":
        pdf.write_bytes(b"%PDF ORIGINAL")
        # Same size; force a different mtime so the change is visible on
        # filesystems with coarse timestamps.
        os.utime(pdf, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    else:
        replacement = tmp_path / "incoming.pdf"
        replacement.write_bytes(b"%PDF original")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, pdf)  # same bytes and mtime, new inode
    assert lookup(pdf) == fsatomic.file_sha256(pdf)
    assert len(counted) == 2


def test_file_changed_while_hashing_is_not_cached(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF original")
    real = fsatomic.file_sha256

    def hash_then_replace(path):
        digest = real(path)
        pdf.write_bytes(b"%PDF replaced during the read")
        return digest

    monkeypatch.setattr(pdf_hashes, "file_sha256", hash_then_replace)
    lookup(pdf)
    conn = get_connection()
    try:
        assert conn.execute("SELECT count(*) FROM pdf_hashes").fetchone()[0] == 0
    finally:
        conn.close()


def test_unusable_database_falls_back_to_hashing(tmp_path, counted):
    """A connection without the table, or no connection, still answers."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF original")
    bare = sqlite3.connect(":memory:")
    try:
        assert lookup(pdf, bare) == fsatomic.file_sha256(pdf)
    finally:
        bare.close()
    cache = PdfHashCache(None)
    assert cache.sha256(pdf) == fsatomic.file_sha256(pdf)
    cache.flush()


def test_claim_queries_hash_each_pdf_once_across_runs(tmp_path, monkeypatch, counted):
    """The behaviour the cache exists for: repeated freshness checks."""
    from researchwiki.search.tools import claims_by_stem
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki/other").mkdir(parents=True)
    (tmp_path / "papers").mkdir()
    for stem in ("alpha", "beta"):
        (tmp_path / f"wiki/other/{stem}.md").write_text(
            f"---\ntitle: {stem}\ntype: paper\n---\n## Key Contributions\n- {stem} finding.\n")
        (tmp_path / f"papers/{stem}.pdf").write_bytes(f"%PDF {stem}".encode())
    rebuild()
    counted.clear()
    for _ in range(3):
        assert claims_by_stem("alpha") and claims_by_stem("beta")
    assert sorted(os.path.basename(p) for p in counted) == ["alpha.pdf", "beta.pdf"]
