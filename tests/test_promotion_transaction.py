"""Real file/DB boundaries, with no PDF extraction or provider calls."""

from types import SimpleNamespace

import pytest

from researchwiki.agents import promote
from researchwiki.db import get_connection
from researchwiki.db.rebuild import rebuild


STEM = "smith-2026-a-test-paper"
OLD_DOI = "10.1101/2026.01.01.123456"
NEW_DOI = "10.1038/test.2026.123"
BODY = "## Key Contributions\n- Alpha improves speed.\n"


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    for folder in ("wiki/other", "papers", "inbox"):
        (tmp_path / folder).mkdir(parents=True)
    page = tmp_path / "wiki/other" / f"{STEM}.md"
    original = f"---\ntitle: Test\ntype: paper\ndoi: {OLD_DOI}\n---\n{BODY}"
    page.write_text(original)
    (tmp_path / "wiki/index.md").write_text("# Index\n")
    (tmp_path / "papers" / f"{STEM}.pdf").write_bytes(b"old pdf")
    incoming = tmp_path / "inbox/journal.pdf"
    incoming.write_bytes(b"new pdf")
    rebuild()
    conn = get_connection()
    with conn:
        conn.execute("UPDATE claims SET semantic_score=0.9, last_graded_at=123")
    conn.close()
    monkeypatch.setattr(promote, "_suggest_category", lambda *a, **k: ("other", "strong"))
    return SimpleNamespace(root=tmp_path, page=page, original=original, incoming=incoming)


def publish(library, doi=NEW_DOI):
    return promote.promote_to_wiki(
        stem=STEM, draft_text=BODY,
        metadata={"title": "Test", "authors": "Smith", "year": 2026, "doi": doi},
        candidates=[], source_pdf_path=library.incoming, attempt_id="review",
        author_model="test-model", use_llm_classify=False,
    )


def test_journal_upgrade_uses_previous_doi(library):
    result = publish(library)
    assert result.promoted, result.warnings
    assert result.pdf_upgrade == {"old_doi": OLD_DOI, "new_doi": NEW_DOI}
    assert (library.root / "papers" / f"{STEM}.pdf").read_bytes() == b"new pdf"


def test_rejected_duplicate_keeps_previous_rows_and_grades(library):
    result = publish(library, doi=OLD_DOI)
    assert not result.promoted
    assert library.page.read_text() == library.original
    conn = get_connection()
    try:
        row = conn.execute("SELECT semantic_score, last_graded_at FROM claims").fetchone()
        assert row is not None
        assert tuple(row) == (0.9, 123)
    finally:
        conn.close()


def test_failure_after_pdf_move_restores_files_and_grades(library, monkeypatch):
    def fail(**kwargs):
        raise OSError("backlink write failed")
    monkeypatch.setattr(promote, "_finish_promote", fail)
    with pytest.raises(OSError, match="backlink write failed"):
        publish(library)
    assert library.page.read_text() == library.original
    assert library.incoming.read_bytes() == b"new pdf"
    assert (library.root / "papers" / f"{STEM}.pdf").read_bytes() == b"old pdf"
    conn = get_connection()
    try:
        assert tuple(conn.execute("SELECT semantic_score, last_graded_at FROM claims").fetchone()) == (0.9, 123)
    finally:
        conn.close()


def test_reclassification_does_not_duplicate_existing_page(library, monkeypatch):
    (library.root / "wiki/new-category").mkdir()
    monkeypatch.setattr(promote, "_suggest_category", lambda *a, **k: ("new-category", "strong"))
    result = publish(library)
    assert result.promoted, result.warnings
    assert result.wiki_path == library.page
    assert len(list((library.root / "wiki").rglob(f"{STEM}.md"))) == 1


def test_invalid_page_rebuild_preserves_but_does_not_ground(library):
    from researchwiki.search.tools import claims_by_stem
    library.page.write_text("---\ntitle: [broken\n---\n" + BODY)
    stats = rebuild()
    assert stats.parse_errors
    assert stats.papers_deleted == 0
    conn = get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
    finally:
        conn.close()
    assert claims_by_stem(STEM) == []
    library.page.write_text(library.original)
    rebuild()
    assert claims_by_stem(STEM)
