"""`stage_supplementary` — copy from anywhere, but consume an `inbox/` source.

The asymmetry is the point. The primary PDF is `shutil.move`d out of `inbox/`
during ingest, but supplementary files were `shutil.copy2`d and the source left
behind. `inbox/` is the ingest backlog, so a leftover supplementary `.pdf` gets
swept up by the next `agent ingest inbox/*.pdf` and ingested as a standalone
paper — its own stem, its own page, its own citable claim set — while a leftover
`.xlsx` just accumulates unseen (`status` only globs `*.pdf`). Neither is
reported by `lint`'s supplementary checks, which never look at `inbox/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.paths import ensure_scaffold, supp_dir
from researchwiki.tasks.attach import stage_supplementary


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_scaffold()
    return tmp_path


def test_inbox_source_is_moved_not_copied(repo):
    src = repo / "inbox" / "Smith_Supplementary.pdf"
    src.write_bytes(b"%PDF-1.4 supp")

    staged = stage_supplementary("smith-2024-a-paper", src)

    assert staged["consumed"] is True
    assert not src.exists(), "inbox source should not survive as backlog"
    landed = supp_dir("smith-2024-a-paper") / staged["filename"]
    assert landed.read_bytes() == b"%PDF-1.4 supp"


def test_external_source_is_left_alone(repo):
    """The `attach ~/Downloads/Table_S4.xlsx` case — not our file to consume."""
    src = repo / "downloads" / "Table_S4.xlsx"
    src.parent.mkdir()
    src.write_bytes(b"xl")

    staged = stage_supplementary("smith-2024-a-paper", src)

    assert staged["consumed"] is False
    assert src.exists(), "a source outside inbox/ must be copied, not moved"
    assert (supp_dir("smith-2024-a-paper") / staged["filename"]).exists()


def test_symlinked_inbox_dir_still_consumes(tmp_path, monkeypatch):
    """`inbox/` is routinely a directory symlink into a synced folder, so the
    membership test has to compare resolved paths on both sides."""
    external = tmp_path / "synced" / "inbox"
    external.mkdir(parents=True)
    repo = tmp_path / "repo"
    (repo / "papers").mkdir(parents=True)
    (repo / "inbox").symlink_to(external, target_is_directory=True)
    monkeypatch.chdir(repo)

    src = external / "supp.pdf"
    src.write_bytes(b"%PDF")

    staged = stage_supplementary("smith-2024-a-paper", repo / "inbox" / "supp.pdf")

    assert staged["consumed"] is True
    assert not src.exists()


def test_file_symlink_inside_inbox_spares_the_target(repo):
    """A symlinked entry resolves outside `inbox/`, so we copy and leave the
    real file alone rather than reaching into wherever it was linked from."""
    real = repo / "library" / "supp.pdf"
    real.parent.mkdir()
    real.write_bytes(b"%PDF")
    link = repo / "inbox" / "supp.pdf"
    link.symlink_to(real)

    staged = stage_supplementary("smith-2024-a-paper", link)

    assert staged["consumed"] is False
    assert real.exists(), "must never delete a file outside inbox/"


def test_failed_staging_leaves_the_inbox_source_intact(repo):
    """Consumption is downstream of a successful copy — a rejected stage must
    not eat the only copy of the file."""
    src = repo / "inbox" / "supp.pdf"
    src.write_bytes(b"%PDF")
    stage_supplementary("smith-2024-a-paper", src)

    dupe = repo / "inbox" / "supp.pdf"
    dupe.write_bytes(b"%PDF second")
    with pytest.raises(FileExistsError):
        stage_supplementary("smith-2024-a-paper", dupe)

    assert dupe.exists()


def test_unlink_failure_is_reported_not_raised(repo, monkeypatch):
    """A read-only inbox costs a stale entry, never a failed ingest — the file
    is already staged by then."""
    src = repo / "inbox" / "supp.pdf"
    src.write_bytes(b"%PDF")

    def _boom(self, *a, **kw):
        raise PermissionError("read-only")

    monkeypatch.setattr(Path, "unlink", _boom)

    staged = stage_supplementary("smith-2024-a-paper", src)

    assert staged["consumed"] is False
    assert (supp_dir("smith-2024-a-paper") / staged["filename"]).exists()
