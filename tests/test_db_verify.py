"""Cross-platform contracts for ``db verify`` filesystem discovery."""

from __future__ import annotations

import pytest

from researchwiki.db.rebuild import rebuild
from researchwiki.db.verify import verify


def test_verify_recognizes_crlf_frontmatter(tmp_path, monkeypatch):
    """Windows CRLF pages must agree between rebuild and verify."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    page = tmp_path / "wiki" / "other" / "lovelace-2026-acceptance.md"
    page.parent.mkdir(parents=True)
    page.write_bytes(
        b"---\r\n"
        b"title: Acceptance\r\n"
        b"type: paper\r\n"
        b"category: [other]\r\n"
        b"---\r\n"
        b"## Key Contributions\r\n"
        b"- The acceptance path is repeatable.\r\n"
    )

    stats = rebuild()
    report = verify()

    assert stats.pages_scanned == 1
    assert report.pages_scanned == 1
    assert report.is_clean


@pytest.mark.parametrize("scan", ["wiki-missing", "subdir-unreadable"])
def test_incomplete_scan_cannot_make_lint_fix_delete_rows(tmp_path, monkeypatch, scan):
    """Absence from a partial scan is not deletion evidence for `lint --fix`."""
    from researchwiki.db import get_connection
    from researchwiki.db import verify as verify_mod
    from researchwiki.tasks.lint.db_checks import db_drift_check_and_fix
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    page = tmp_path / "wiki" / "other" / "example.md"
    page.parent.mkdir(parents=True)
    page.write_text("---\ntitle: Example\ntype: paper\n---\n## Key Contributions\n- A claim.\n")
    rebuild()
    if scan == "wiki-missing":
        (tmp_path / "wiki").rename(tmp_path / "wiki-offline")
    else:
        def walk(root, *, onerror):
            onerror(PermissionError("unreadable category"))
            yield str(root), [], []
        monkeypatch.setattr(verify_mod.os, "walk", walk)

    with pytest.raises(OSError):
        verify()
    drift, fixed = db_drift_check_and_fix(apply_fix=True)

    assert fixed == {}
    conn = get_connection()
    try:
        assert conn.execute("SELECT count(*) FROM papers").fetchone()[0] == 1
    finally:
        conn.close()
