"""Terminal ingest outcomes and exact-duplicate short circuit."""

from pathlib import Path
from types import SimpleNamespace
import errno
import os

import pytest

from researchwiki.agents import runner
from researchwiki.agents.context import Context
from researchwiki.agents.runner import _find_exact_duplicate
from researchwiki.agents.runner_support import _archive_duplicate_inbox_pdf
from researchwiki.db.iterations import read_attempt
from researchwiki.tasks._ingest_batch import _resolve_inputs


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


@pytest.fixture
def duplicate_library(tmp_path, monkeypatch):
    """Real local files and telemetry, with metadata/API work isolated."""
    monkeypatch.chdir(tmp_path)
    for directory in ("inbox", "papers", "wiki/methods"):
        (tmp_path / directory).mkdir(parents=True)
    incoming = tmp_path / "inbox" / "incoming.pdf"
    canonical = tmp_path / "papers" / "smith-2026-example.pdf"
    page = tmp_path / "wiki" / "methods" / "smith-2026-example.md"
    incoming.write_bytes(b"same-pdf")
    canonical.write_bytes(b"same-pdf")
    page.write_text("---\ntype: paper\n---\n", encoding="utf-8")
    monkeypatch.setattr(
        runner, "_phase_reconcile",
        lambda ctx, conn: ({"stem": canonical.stem, "year": 2026}, None),
    )
    monkeypatch.setattr(
        runner.model_config, "for_phase",
        lambda phase: SimpleNamespace(provider="stub", model="stub"),
    )
    monkeypatch.setattr(runner, "set_relay_identity", lambda **kwargs: None)

    def stop_at_extraction(ctx, conn):
        raise _ExtractionReached

    monkeypatch.setattr(runner, "_phase_extract", stop_at_extraction)
    return incoming, canonical, page


class _ExtractionReached(Exception):
    """A nonduplicate reaches extraction rather than finishing early."""


def test_ingest_archives_exact_inbox_duplicate_by_default(duplicate_library):
    incoming, canonical, page = duplicate_library
    page_before = page.read_bytes()

    ctx = runner.run_ingest(incoming, use_stub=True)

    assert ctx.outcome == "already_present"
    archive = ctx.duplicate_archive_path
    assert archive is not None
    assert archive.name == incoming.name
    assert archive.parent.parent == incoming.parent.parent / ".duplicate-pdfs"
    assert archive.read_bytes() == b"same-pdf"
    assert not incoming.exists()
    assert canonical.read_bytes() == b"same-pdf"
    assert page.read_bytes() == page_before
    events = read_attempt(ctx.attempt_id)
    assert [event.role for event in events] == ["duplicate", "attempt"]
    assert events[0].gate_metrics == {"duplicate_archive_path": str(archive)}
    assert "duplicate inbox PDF archived" in events[0].decision_reason


def test_external_duplicate_is_preserved(duplicate_library):
    incoming, canonical, page = duplicate_library
    external = incoming.parent.parent / "external.pdf"
    incoming.rename(external)

    ctx = runner.run_ingest(Path("inbox/../external.pdf"), use_stub=True)

    assert ctx.outcome == "already_present"
    assert ctx.duplicate_archive_path is None
    assert external.read_bytes() == canonical.read_bytes()
    assert page.exists()


def test_duplicate_cleanup_supports_a_symlinked_inbox(duplicate_library):
    incoming, canonical, page = duplicate_library
    inbox = incoming.parent
    library_inbox = inbox.with_name("synced-inbox")
    inbox.rename(library_inbox)
    inbox.symlink_to(library_inbox, target_is_directory=True)

    ctx = runner.run_ingest(Path("inbox/incoming.pdf"), use_stub=True)

    assert ctx.duplicate_archive_path.read_bytes() == b"same-pdf"
    assert inbox.is_symlink()
    assert not (library_inbox / incoming.name).exists()
    assert canonical.exists() and page.exists()


@pytest.mark.parametrize("kind", ["different_bytes", "canonical", "orphan"])
def test_nonduplicates_are_preserved_and_continue(duplicate_library, kind):
    incoming, canonical, page = duplicate_library
    if kind == "different_bytes":
        incoming.write_bytes(b"new-version")
    elif kind == "orphan":
        page.unlink()
    input_path = canonical if kind == "canonical" else incoming
    before = input_path.read_bytes()

    with pytest.raises(_ExtractionReached):
        runner.run_ingest(input_path, use_stub=True)

    assert input_path.read_bytes() == before
    assert canonical.read_bytes() == b"same-pdf"


@pytest.mark.parametrize("target_location", ["papers", "external", "inbox"])
def test_file_symlinks_and_their_targets_are_preserved(duplicate_library, target_location):
    incoming, canonical, page = duplicate_library
    if target_location == "papers":
        target = canonical
    elif target_location == "external":
        target = incoming.parent.parent / "external.pdf"
        target.write_bytes(b"same-pdf")
    else:
        target = incoming
    alias = incoming.with_name("alias.pdf")
    alias.symlink_to(target)

    if target_location == "papers":
        with pytest.raises(_ExtractionReached):
            runner.run_ingest(alias, use_stub=True)
    else:
        # Batch workers must receive the alias, not its resolved target.
        batch_input, = _resolve_inputs([str(alias), str(target)])
        assert Path(batch_input) == alias
        ctx = runner.run_ingest(batch_input, use_stub=True)
        assert ctx.outcome == "already_present"
        assert ctx.duplicate_archive_path is None

    assert alias.is_symlink()
    assert target.read_bytes() == b"same-pdf"
    assert canonical.exists() and page.exists()


def test_cleanup_rechecks_bytes_before_archiving_input(duplicate_library):
    incoming, canonical, _ = duplicate_library
    assert _find_exact_duplicate(_ctx(incoming), None) is not None
    incoming.write_bytes(b"new-version")

    assert _archive_duplicate_inbox_pdf(incoming, canonical.stem) is None
    assert incoming.read_bytes() == b"new-version"
    assert canonical.read_bytes() == b"same-pdf"


def test_cleanup_failure_reports_retained_input(duplicate_library, monkeypatch, capsys):
    incoming, canonical, page = duplicate_library

    def deny_move(*args, **kwargs):
        raise PermissionError("read-only archive")

    monkeypatch.setattr("researchwiki.agents.runner_support.shutil.move", deny_move)
    ctx = runner.run_ingest(incoming, use_stub=True)

    assert ctx.outcome == "already_present"
    assert ctx.duplicate_archive_path is None
    assert incoming.exists() and canonical.exists() and page.exists()
    output = capsys.readouterr()
    assert "inbox archive failed; input retained" in output.out + output.err
    assert read_attempt(ctx.attempt_id)[0].gate_metrics == {"duplicate_archive_path": None}


def test_repeated_filenames_keep_both_archived_copies(duplicate_library):
    incoming, canonical, _ = duplicate_library
    first = runner.run_ingest(incoming, use_stub=True).duplicate_archive_path
    incoming.write_bytes(canonical.read_bytes())
    second = runner.run_ingest(incoming, use_stub=True).duplicate_archive_path

    assert first != second
    assert first.name == second.name == incoming.name
    assert first.read_bytes() == second.read_bytes() == canonical.read_bytes()
    assert not incoming.exists()


def test_archiving_works_across_filesystems(duplicate_library, monkeypatch):
    incoming, canonical, page = duplicate_library

    def cross_device_rename(*args, **kwargs):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "rename", cross_device_rename)
    ctx = runner.run_ingest(incoming, use_stub=True)

    assert ctx.duplicate_archive_path.read_bytes() == b"same-pdf"
    assert not incoming.exists()
    assert canonical.read_bytes() == b"same-pdf"
    assert page.exists()
