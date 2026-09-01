"""Cross-platform contracts for ``db verify`` filesystem discovery."""

from __future__ import annotations

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
