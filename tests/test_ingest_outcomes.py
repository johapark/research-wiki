"""Terminal ingest outcomes and exact-duplicate short circuit."""

from pathlib import Path

from researchwiki.agents.context import Context
from researchwiki.agents.runner import _find_exact_duplicate


def _ctx(path: Path, stem: str = "smith-2026-example") -> Context:
    ctx = Context(attempt_id="a", pdf_path=path, pdf_filename=path.name)
    ctx.paper_stem = stem
    return ctx


def test_identical_deposited_pdf_is_already_present(tmp_path, monkeypatch):
    from researchwiki import paths, wiki

    papers = tmp_path / "papers"
    pages = tmp_path / "wiki" / "methods"
    papers.mkdir()
    pages.mkdir(parents=True)
    incoming = tmp_path / "inbox.pdf"
    canonical = papers / "smith-2026-example.pdf"
    incoming.write_bytes(b"same-pdf")
    canonical.write_bytes(b"same-pdf")
    page = pages / "smith-2026-example.md"
    page.write_text("---\ntype: paper\n---\n", encoding="utf-8")
    monkeypatch.setattr(paths, "papers_dir", lambda: papers)
    monkeypatch.setattr(wiki, "find_stem_collision", lambda stem: page)

    assert _find_exact_duplicate(_ctx(incoming), None) == (
        "smith-2026-example", page,
    )


def test_different_pdf_and_deliberate_canonical_reingest_continue(tmp_path, monkeypatch):
    from researchwiki import paths

    papers = tmp_path / "papers"
    papers.mkdir()
    canonical = papers / "smith-2026-example.pdf"
    canonical.write_bytes(b"canonical")
    incoming = tmp_path / "incoming.pdf"
    incoming.write_bytes(b"new-version")
    monkeypatch.setattr(paths, "papers_dir", lambda: papers)

    assert _find_exact_duplicate(_ctx(incoming), None) is None
    assert _find_exact_duplicate(_ctx(canonical), None) is None


def test_orphan_canonical_pdf_is_not_reported_as_ingested(tmp_path, monkeypatch):
    from researchwiki import paths, wiki

    papers = tmp_path / "papers"
    papers.mkdir()
    canonical = papers / "smith-2026-example.pdf"
    incoming = tmp_path / "incoming.pdf"
    canonical.write_bytes(b"same-pdf")
    incoming.write_bytes(b"same-pdf")
    monkeypatch.setattr(paths, "papers_dir", lambda: papers)
    monkeypatch.setattr(wiki, "find_stem_collision", lambda stem: None)

    assert _find_exact_duplicate(_ctx(incoming), None) is None


def test_authorized_stem_rename_is_not_short_circuited(tmp_path, monkeypatch):
    from researchwiki import paths

    papers = tmp_path / "papers"
    papers.mkdir()
    old = papers / "smith-2025-old-stem.pdf"
    old.write_bytes(b"same-pdf")
    incoming = tmp_path / "incoming.pdf"
    incoming.write_bytes(b"same-pdf")
    monkeypatch.setattr(paths, "papers_dir", lambda: papers)
    ctx = _ctx(incoming, stem="smith-2026-new-stem")
    ctx.allow_rename = True

    assert _find_exact_duplicate(ctx, "smith-2025-old-stem") is None
