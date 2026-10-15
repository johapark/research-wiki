"""A filename is not evidence identity; all fixtures are disposable."""

from researchwiki.db import get_connection
from researchwiki.db.rebuild import rebuild
from researchwiki.index import pdf_chunks
import pytest


def test_pdf_replacement_rebuilds_chunks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.pdf"
    source.write_text("Alpha improves speed.")
    monkeypatch.setattr(pdf_chunks, "extract_pdf_page_texts", lambda path, **kw: [path.read_text()])
    monkeypatch.setattr(pdf_chunks, "_build_embeddings_cache", lambda *a: None)
    pdf_chunks.build_pdf_index("example", pdf_path=source)
    source.write_text("Beta reduces memory.")
    pdf_chunks.build_pdf_index("example", pdf_path=source)
    hits = pdf_chunks.query_pdf("example", "Beta")
    assert hits and hits[0].text == "Beta reduces memory."


def test_pdf_changed_during_extraction_is_not_published(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.pdf"
    source.write_text("old input")
    def extract(path, **kwargs):
        path.write_text("replacement input")
        return ["old input"]
    monkeypatch.setattr(pdf_chunks, "extract_pdf_page_texts", extract)
    monkeypatch.setattr(pdf_chunks, "_build_embeddings_cache", lambda *a: None)
    import pytest
    with pytest.raises(RuntimeError, match="changed"):
        pdf_chunks.build_pdf_index("example", pdf_path=source)


def test_pdf_replacement_invalidates_unchanged_claim_grade(tmp_path, monkeypatch):
    from researchwiki.fsatomic import file_sha256
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    (tmp_path / "wiki/other").mkdir(parents=True)
    (tmp_path / "papers").mkdir()
    (tmp_path / "wiki/other/example.md").write_text(
        "---\ntitle: Test\ntype: paper\n---\n## Key Contributions\n- Alpha improves speed.\n"
    )
    source = tmp_path / "papers/example.pdf"
    source.write_text("old input")
    rebuild()
    conn = get_connection()
    with conn:
        conn.execute("UPDATE claims SET last_graded_at=123, semantic_score=0.9, "
                     "supporting_provenance='p. 1', source_pdf_sha256=?", (file_sha256(source),))
    conn.close()
    source.write_text("replacement input")
    rebuild()
    conn = get_connection()
    try:
        row = conn.execute("SELECT last_graded_at, semantic_score, supporting_provenance FROM claims").fetchone()
        assert tuple(row) == (None, None, None)
    finally:
        conn.close()


@pytest.mark.parametrize("edit", ["page", "pdf"])
def test_concurrent_source_edit_cannot_receive_old_grades(tmp_path, monkeypatch, edit):
    from researchwiki.grade.fidelity import paper
    monkeypatch.chdir(tmp_path)
    page = tmp_path / "wiki/other/example.md"
    page.parent.mkdir(parents=True)
    page.write_text("---\ntype: paper\n---\n## Key Contributions\n- Original meaningful claim.\n")
    pdf = tmp_path / "papers/example.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"old PDF")
    rebuild()
    monkeypatch.setattr(paper, "build_pdf_index", lambda *a, **kw: None)
    monkeypatch.setattr(paper, "extract_pdf", lambda *a, **kw: ("old PDF", 1))
    def score(stem, claim, *args, **kwargs):
        if edit == "page":
            page.write_text(page.read_text().replace("Original", "Changed"))
        else:
            pdf.write_bytes(b"new PDF")
        return paper.ClaimScore(claim.section, claim.position, claim.text, False,
                               4., 3., 0, "old PDF", "p. 1", [], [], None, False, True)
    monkeypatch.setattr(paper, "_score_claim", score)
    with pytest.raises(RuntimeError, match="changed"):
        paper.grade_page("example", semantic=False, include_salience=False)
    conn = get_connection()
    try:
        assert conn.execute("SELECT last_graded_at FROM claims").fetchone()[0] is None
    finally:
        conn.close()


def test_legacy_grade_provenance_migration_is_idempotent(tmp_path):
    import sqlite3
    from pathlib import Path
    db = tmp_path / "legacy.db"
    schema = Path(__file__).resolve().parents[1] / "researchwiki/db/schema.sql"
    with sqlite3.connect(db) as conn:
        legacy = "\n".join(line for line in schema.read_text().splitlines()
                           if not line.strip().startswith("source_pdf_sha256 "))
        conn.executescript(legacy)
        assert "source_pdf_sha256" not in {row[1] for row in conn.execute("PRAGMA table_info(claims)")}
    for _ in range(2):
        conn = get_connection(db)
        try:
            columns = [row["name"] for row in conn.execute("PRAGMA table_info(claims)")]
            assert columns.count("source_pdf_sha256") == 1
        finally:
            conn.close()
