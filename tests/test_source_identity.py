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


@pytest.mark.parametrize("pdf_state", ["legacy-unhashed", "pdf-absent"])
def test_rebuild_keeps_grades_without_proof_of_a_different_source(tmp_path, monkeypatch, pdf_state):
    """Only a recorded hash that differs from a present PDF clears a grade.

    A grade written before hashes existed, or one whose PDF is momentarily
    missing (a synced `papers/` offline), stays for retrieval to withhold.
    """
    from researchwiki.fsatomic import file_sha256
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    (tmp_path / "wiki/other").mkdir(parents=True)
    (tmp_path / "papers").mkdir()
    (tmp_path / "wiki/other/example.md").write_text(
        "---\ntitle: Test\ntype: paper\n---\n## Key Contributions\n- Alpha improves speed.\n"
    )
    source = tmp_path / "papers/example.pdf"
    source.write_text("input")
    rebuild()
    recorded = None if pdf_state == "legacy-unhashed" else file_sha256(source)
    conn = get_connection()
    with conn:
        conn.execute("UPDATE claims SET last_graded_at=123, semantic_score=0.9, "
                     "source_pdf_sha256=?", (recorded,))
    conn.close()
    if pdf_state == "pdf-absent":
        source.unlink()
    rebuild()
    conn = get_connection()
    try:
        row = conn.execute("SELECT last_graded_at, semantic_score, source_pdf_sha256 FROM claims").fetchone()
        assert tuple(row) == (123, 0.9, recorded)
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


def test_draft_grade_scores_the_override_pdf_not_the_canonical_one(tmp_path, monkeypatch):
    """Re-ingest/journal-upgrade: the old PDF sits in papers/, the new one in inbox/.

    Every per-claim query revalidates the chunk index by source hash, so the
    override has to reach each one — otherwise the draft is re-indexed from the
    canonical PDF and graded against text it was never written from.
    """
    from pathlib import Path
    from researchwiki.grade.fidelity import paper
    monkeypatch.chdir(tmp_path)
    built_from = []

    def extract(path, **kwargs):
        built_from.append(Path(path).parent.name)
        return [Path(path).read_text()]

    monkeypatch.setattr(pdf_chunks, "extract_pdf_page_texts", extract)
    monkeypatch.setattr(pdf_chunks, "_build_embeddings_cache", lambda *a: None)
    monkeypatch.setattr(paper, "extract_pdf", lambda p, **kw: (Path(p).read_text(), 1))
    for folder in ("papers", "inbox", ".agent-output"):
        (tmp_path / folder).mkdir()
    (tmp_path / "papers/example.pdf").write_text("Preprint reports seventy percent accuracy.")
    incoming = tmp_path / "inbox/journal.pdf"
    incoming.write_text("Journal version reports zebrafish regeneration benchmark results.")
    draft = tmp_path / ".agent-output/draft.md"
    draft.write_text("---\ntype: paper\n---\n## Key Contributions\n"
                     "- Zebrafish regeneration benchmark results reported.\n")

    report = paper.grade_page("example", page_path=draft, pdf_path=incoming,
                              semantic=False, persist=False, include_salience=False)

    assert "zebrafish" in report.claims[0].supporting_text.lower()
    assert built_from == ["inbox"]


def _counting_index(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    builds = []

    def extract(path, **kwargs):
        builds.append(1)
        return [Path(path).read_text()]

    monkeypatch.setattr(pdf_chunks, "extract_pdf_page_texts", extract)
    monkeypatch.setattr(pdf_chunks, "_build_embeddings_cache", lambda *a: None)
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers/example.pdf").write_text("Alpha improves speed. " * 50)
    return builds


def test_index_missing_its_segments_is_rebuilt_not_reused(tmp_path, monkeypatch):
    """Readiness markers alone do not prove the index is readable.

    A crash during the old in-place `rmtree` could keep `index_meta.json` and
    `.managed.json` while deleting the segment files; the stem then failed every
    query with `FileDoesNotExist` until someone forced a rebuild.
    """
    builds = _counting_index(tmp_path, monkeypatch)
    target = pdf_chunks.build_pdf_index("example")
    for path in target.iterdir():
        if path.name not in {"index_meta.json", ".managed.json", "meta.json"}:
            path.unlink()

    hits = pdf_chunks.query_pdf("example", "alpha")

    assert hits and "Alpha" in hits[0].text
    assert len(builds) == 2
    pdf_chunks.query_pdf("example", "alpha")
    assert len(builds) == 2, "a healthy index must still be reused"


def test_crash_while_removing_the_old_index_leaves_a_clean_miss(tmp_path, monkeypatch):
    """The failure the in-place `rmtree(target)` allowed: dying part-way through
    deleting the old index, after its segments went but before its markers did.

    Dangerous on a forced rebuild of an unchanged PDF (`grade --rebuild-index`):
    the surviving markers still name the current hash, so the old code reused
    the gutted directory and every query failed. The old index is now renamed
    aside before anything is deleted, so `target` is whole or absent.
    """
    import shutil
    from pathlib import Path
    builds = _counting_index(tmp_path, monkeypatch)
    pdf_chunks.build_pdf_index("example")
    real_rmtree = shutil.rmtree

    def die_mid_delete(path, *args, **kwargs):
        doomed = Path(path)
        if doomed.name.startswith(".example-") and "-retired-" not in doomed.name:
            return real_rmtree(path, *args, **kwargs)   # the staging dir's cleanup
        for entry in doomed.iterdir():                   # segments go, markers stay
            if entry.name not in {"index_meta.json", ".managed.json", "meta.json"}:
                entry.unlink()
        raise OSError("simulated crash part-way through deleting the old index")

    monkeypatch.setattr(pdf_chunks.shutil, "rmtree", die_mid_delete)
    try:
        pdf_chunks.build_pdf_index("example", force=True)
    except OSError:
        pass
    monkeypatch.setattr(pdf_chunks.shutil, "rmtree", real_rmtree)

    hits = pdf_chunks.query_pdf("example", "alpha")
    assert hits and "Alpha" in hits[0].text
    assert len(builds) >= 2
