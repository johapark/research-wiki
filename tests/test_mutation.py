"""Transactional multi-file mutations.

`promote_to_wiki` writes a page, moves a PDF, splices back-links into N pages,
and appends to `index.md` and `log.md`. Each step was individually atomic and
nothing bound them, so a failure after the PDF move left a paper half-landed.

The properties that matter, and are pinned here:

  - rollback restores every declared path byte-for-byte, including files the
    mutation had already modified;
  - a path that did not exist before is *removed* on rollback, not left behind;
  - the commit point is explicit, and `discard()` runs after it — so a failure
    while cleaning up can never undo committed work;
  - a crash leaves a journal that the next run drains;
  - a rollback that keeps failing stops retrying instead of spinning forever.

Hermetic: tmp dirs only, no PDFs, no DB, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchwiki import mutation as mut


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A wiki root whose `.mutation/` is inside tmp_path."""
    monkeypatch.setattr(mut, "mutation_dir", lambda: tmp_path / ".mutation")
    monkeypatch.delenv("RW_MUTATION_JOURNAL", raising=False)
    return tmp_path


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------- rollback restores ----------

def test_rollback_restores_modified_files(root):
    page = _write(root / "wiki" / "a.md", "original a")
    index = _write(root / "wiki" / "index.md", "original index")

    with pytest.raises(RuntimeError):
        with mut.mutation([page, index], operation="promote"):
            page.write_text("clobbered", encoding="utf-8")
            index.write_text("clobbered too", encoding="utf-8")
            raise RuntimeError("boom")

    assert page.read_text(encoding="utf-8") == "original a"
    assert index.read_text(encoding="utf-8") == "original index"


def test_rollback_removes_files_that_did_not_exist(root):
    created = root / "wiki" / "new.md"

    with pytest.raises(RuntimeError):
        with mut.mutation([created], operation="promote"):
            _write(created, "brand new")
            raise RuntimeError("boom")

    assert not created.exists()


def test_rollback_restores_a_moved_file(root):
    """The PDF move is the step that made the old failure irrecoverable: the
    input left `inbox/` and nothing could put it back."""
    src = _write(root / "inbox" / "paper.pdf", "PDF BYTES")
    dest = root / "papers" / "stem.pdf"

    with pytest.raises(RuntimeError):
        with mut.mutation([src, dest], operation="promote"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
            raise RuntimeError("boom after the move")

    assert src.read_text(encoding="utf-8") == "PDF BYTES"
    assert not dest.exists()


def test_early_return_without_commit_rolls_back(root):
    """`promote` returns early on a PDF-move failure; that must undo, not
    strand. The context manager treats "left without committing" as failure."""
    page = _write(root / "wiki" / "a.md", "original")

    def run():
        with mut.mutation([page], operation="promote"):
            page.write_text("half-landed", encoding="utf-8")
            return "returned early"

    assert run() == "returned early"
    assert page.read_text(encoding="utf-8") == "original"


def test_commit_keeps_the_changes(root):
    page = _write(root / "wiki" / "a.md", "original")
    with mut.mutation([page], operation="promote") as snap:
        page.write_text("new content", encoding="utf-8")
        snap.mark_committed()
    assert page.read_text(encoding="utf-8") == "new content"


def test_commit_clears_the_journal(root):
    page = _write(root / "wiki" / "a.md", "x")
    with mut.mutation([page], operation="promote") as snap:
        snap.mark_committed()
    assert mut.pending_journals() == []


# ---------- registered undos ----------

def test_undo_runs_on_rollback_after_files_are_restored(root):
    page = _write(root / "wiki" / "a.md", "original")
    order: list[str] = []

    with pytest.raises(RuntimeError):
        with mut.mutation([page], operation="promote") as snap:
            page.write_text("changed", encoding="utf-8")
            snap.also_undo(lambda: order.append(
                "undo saw: " + page.read_text(encoding="utf-8")))
            raise RuntimeError("boom")

    assert order == ["undo saw: original"], \
        "files must be restored before the undo callbacks run"


def test_undo_does_not_run_on_commit(root):
    page = _write(root / "wiki" / "a.md", "x")
    calls: list[int] = []
    with mut.mutation([page], operation="promote") as snap:
        snap.also_undo(lambda: calls.append(1))
        snap.mark_committed()
    assert calls == []


def test_a_failing_undo_does_not_mask_the_original_error(root):
    page = _write(root / "wiki" / "a.md", "original")

    def bad_undo():
        raise ValueError("undo exploded")

    with pytest.raises(RuntimeError, match="the real failure"):
        with mut.mutation([page], operation="promote") as snap:
            snap.also_undo(bad_undo)
            raise RuntimeError("the real failure")
    assert page.read_text(encoding="utf-8") == "original"


# ---------- commit-point ordering ----------

def test_discard_failure_never_undoes_committed_work(root, monkeypatch):
    """The subtle part of the design: `discard()` runs *after* the commit
    point, so its failure must leave the committed changes alone."""
    page = _write(root / "wiki" / "a.md", "original")

    def boom(*a, **k):
        raise OSError("cleanup failed")

    with mut.mutation([page], operation="promote") as snap:
        page.write_text("committed content", encoding="utf-8")
        snap.mark_committed()
        monkeypatch.setattr(mut.shutil, "rmtree", boom)

    assert page.read_text(encoding="utf-8") == "committed content"


# ---------- crash recovery ----------

def test_crash_leaves_an_active_journal(root):
    page = _write(root / "wiki" / "a.md", "original")
    snap = mut.snapshot([page], operation="promote", details={"stem": "s"})
    page.write_text("half-written", encoding="utf-8")
    # Process dies here — no rollback, no commit.

    journals = mut.pending_journals()
    assert len(journals) == 1
    assert journals[0]["status"] == "active"
    assert journals[0]["operation"] == "promote"
    assert journals[0]["details"]["stem"] == "s"
    assert snap.journal_path.exists()


def test_recovery_rolls_back_an_active_journal(root):
    page = _write(root / "wiki" / "a.md", "original")
    mut.snapshot([page], operation="promote")
    page.write_text("half-written", encoding="utf-8")

    notes = mut.recover_pending()

    assert page.read_text(encoding="utf-8") == "original"
    assert any("rolled back" in n for n in notes)
    assert mut.pending_journals() == []


def test_recovery_discards_a_committed_journal(root):
    page = _write(root / "wiki" / "a.md", "original")
    snap = mut.snapshot([page], operation="promote")
    page.write_text("committed", encoding="utf-8")
    snap.mark_committed()
    # Crash between commit and cleanup.

    notes = mut.recover_pending()

    assert page.read_text(encoding="utf-8") == "committed", \
        "a committed journal must be discarded, never rolled back"
    assert any("discarded" in n for n in notes)
    assert mut.pending_journals() == []


def test_recovery_is_idempotent(root):
    page = _write(root / "wiki" / "a.md", "original")
    mut.snapshot([page], operation="promote")
    page.write_text("half", encoding="utf-8")
    mut.recover_pending()
    assert mut.recover_pending() == []
    assert page.read_text(encoding="utf-8") == "original"


def test_recovery_gives_up_after_the_attempt_cap(root):
    page = _write(root / "wiki" / "a.md", "original")
    snap = mut.snapshot([page], operation="promote")
    data = json.loads(snap.journal_path.read_text(encoding="utf-8"))
    data["attempts"] = mut.MAX_ROLLBACK_ATTEMPTS
    snap.journal_path.write_text(json.dumps(data), encoding="utf-8")

    notes = mut.recover_pending()

    assert any("left in place" in n for n in notes)
    assert snap.journal_path.exists(), "the journal is kept for a human to inspect"


def test_recovery_survives_a_corrupt_journal(root):
    (root / ".mutation").mkdir(parents=True, exist_ok=True)
    (root / ".mutation" / "broken.json").write_text("{not json", encoding="utf-8")
    assert mut.recover_pending() == []


def test_missing_backup_leaves_the_target_alone(root):
    """Better to keep the current file than delete content we cannot restore."""
    page = _write(root / "wiki" / "a.md", "original")
    snap = mut.snapshot([page], operation="promote")
    page.write_text("modified", encoding="utf-8")
    for backup in snap.backup_dir.iterdir():
        backup.unlink()

    mut.recover_pending()
    assert page.read_text(encoding="utf-8") == "modified"


def test_a_journal_with_no_backup_dir_never_deletes_the_cwd(root, monkeypatch):
    """The regression this pins was catastrophic: `document.get("backup_dir", "")`
    fell back to `Path("")` — which is `.` — and `_clean_up` then rmtree'd the
    entire working directory, wiki and papers included, silently
    (`ignore_errors=True`). One stray or truncated `.json` under `.mutation/`
    was the whole trigger, and `recover_pending` auto-runs at ingest start."""
    monkeypatch.chdir(root)
    canary = _write(root / "canary.txt", "still here")
    _write(root / "wiki" / "a.md", "wiki content")
    mdir = root / ".mutation"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "stray.json").write_text(
        json.dumps({"version": 1, "operation": "promote", "status": "committed"}),
        encoding="utf-8")

    notes = mut.recover_pending()

    assert canary.read_text(encoding="utf-8") == "still here"
    assert (root / "wiki" / "a.md").exists()
    assert not (mdir / "stray.json").exists(), "the journal itself is still drained"
    assert any("discarded" in n for n in notes)


def test_recovery_refuses_a_backup_dir_outside_mutation_dir(root):
    """Only the bulk delete is gated: the journal is drained, but a
    `backup_dir` pointing anywhere outside `.mutation/` is never rmtree'd."""
    precious = _write(root / "wiki" / "precious.md", "irreplaceable")
    mdir = root / ".mutation"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "hostile.json").write_text(json.dumps({
        "version": 1, "operation": "promote", "status": "committed",
        "backup_dir": str(root / "wiki"),
    }), encoding="utf-8")

    mut.recover_pending()

    assert precious.read_text(encoding="utf-8") == "irreplaceable"
    assert not (mdir / "hostile.json").exists()


def test_recovery_leaves_a_newer_schema_journal_in_place(root):
    """Old code draining a journal it cannot fully read could lose its
    rollback; report it and step aside instead."""
    page = _write(root / "wiki" / "a.md", "current")
    mdir = root / ".mutation"
    mdir.mkdir(parents=True, exist_ok=True)
    journal = mdir / "future.json"
    journal.write_text(json.dumps({
        "version": mut.JOURNAL_VERSION + 1, "operation": "promote",
        "status": "active", "backup_dir": str(mdir / "future"),
        "entries": [{"target": str(page), "backup": None}],
    }), encoding="utf-8")

    notes = mut.recover_pending()

    assert journal.exists()
    assert page.exists(), "a v-newer journal's entries must not be replayed"
    assert any("newer" in n for n in notes)


def test_recovery_reports_an_orphan_backup_dir_without_deleting_it(root):
    """A backup dir with no `.json` beside it is residue of a snapshot that died
    mid-copy. Deleting it could race a concurrent snapshot, so: report only."""
    orphan = root / ".mutation" / "20260814T000000-cafef00d"
    _write(orphan / "000-page.md", "backup bytes")

    notes = mut.recover_pending()

    assert orphan.exists()
    assert any("orphaned backup directory" in n for n in notes)


# ---------- snapshot mechanics ----------

def test_duplicate_paths_are_collapsed(root):
    page = _write(root / "wiki" / "a.md", "original")
    snap = mut.snapshot([page, page, str(page)], operation="promote")
    assert len(snap.entries) == 1


def test_none_entries_are_ignored(root):
    page = _write(root / "wiki" / "a.md", "x")
    snap = mut.snapshot([page, None], operation="promote")
    assert len(snap.entries) == 1


def test_two_snapshots_do_not_share_a_journal(root):
    page = _write(root / "wiki" / "a.md", "x")
    a = mut.snapshot([page], operation="promote")
    b = mut.snapshot([page], operation="promote")
    assert a.journal_path != b.journal_path
    assert a.backup_dir != b.backup_dir
    assert len(mut.pending_journals()) == 2


# ---------- the bypass flag ----------

def test_disabled_by_env_is_a_passthrough(root, monkeypatch):
    monkeypatch.setenv("RW_MUTATION_JOURNAL", "0")
    page = _write(root / "wiki" / "a.md", "original")

    with pytest.raises(RuntimeError):
        with mut.mutation([page], operation="promote"):
            page.write_text("not rolled back", encoding="utf-8")
            raise RuntimeError("boom")

    assert page.read_text(encoding="utf-8") == "not rolled back"
    assert not (root / ".mutation").exists()



@pytest.mark.parametrize("value,expected", [
    ("0", False), ("1", True), ("", True), ("yes", True),
])
def test_journalling_enabled_flag(monkeypatch, value, expected):
    monkeypatch.setenv("RW_MUTATION_JOURNAL", value)
    assert mut.journalling_enabled() is expected


# ---------------------------------------------------------------- on-disk contract

#: A journal exactly as v0.4.0 wrote one, key-for-key. Hard-coded rather than
#: produced by the current `_journal_document`, which is the whole point: a
#: round-trip through today's writer would still pass if both ends drifted
#: together, and the file that matters is the one already sitting in somebody's
#: `.mutation/` when they upgrade mid-mutation.
_V0_4_0_JOURNAL = {
    "version": 1,
    "operation": "promote",
    "status": "active",
    "attempts": 0,
    "created_at": "2026-08-14T12:00:00",
    "details": {"stem": "smith-2024-a-paper-about-things"},
}


def test_a_journal_written_by_the_previous_release_still_recovers(root):
    """The rollback path is only worth having if it survives an upgrade.

    An interrupted promote leaves a journal on disk. If a release changes how that
    file is read, the user's next run silently cannot undo the mutation the
    previous run abandoned — and the failure mode is a permanently half-landed
    paper, not an error anybody sees.
    """
    mdir = root / ".mutation"
    backup_dir = mdir / "20260814T120000-deadbeef"
    backup_dir.mkdir(parents=True)

    modified = _write(root / "wiki" / "cgt" / "page.md", "MUTATED\n")
    _write(backup_dir / "000-page.md", "ORIGINAL\n")
    created = _write(root / "wiki" / "cgt" / "brand-new.md", "created by the mutation\n")

    journal = mdir / "20260814T120000-deadbeef.json"
    journal.write_text(json.dumps({
        **_V0_4_0_JOURNAL,
        "backup_dir": str(backup_dir),
        "entries": [
            {"target": str(modified), "backup": str(backup_dir / "000-page.md")},
            {"target": str(created), "backup": None},
        ],
    }), encoding="utf-8")

    notes = mut.recover_pending()

    assert modified.read_text(encoding="utf-8") == "ORIGINAL\n"
    assert not created.exists(), "a path the mutation created must be removed"
    assert not journal.exists() and not backup_dir.exists()
    assert notes and "rolled back interrupted promote" in notes[0]


def test_the_journal_we_write_matches_the_documented_key_set(root):
    """Pin the serialized shape, since `recover_pending` in a *future* release
    will be reading files this one wrote."""
    page = _write(root / "wiki" / "cgt" / "page.md", "before\n")
    snap = mut.snapshot([page], operation="promote", details={"stem": "s"})

    document = json.loads(snap.journal_path.read_text(encoding="utf-8"))
    assert set(document) == {
        "version", "operation", "status", "attempts", "created_at",
        "backup_dir", "details", "entries",
    }
    assert document["version"] == mut.JOURNAL_VERSION == 1
    assert document["status"] == "active"
    assert set(document["entries"][0]) == {"target", "backup"}

    snap.mark_committed()
    assert json.loads(snap.journal_path.read_text(encoding="utf-8"))["status"] == "committed"
    snap.discard()


def test_a_created_path_is_recorded_without_a_backup(root):
    """`BackedUpPath.existed_before` is what rollback turns on, so the two cases
    have to be distinguishable in the record itself."""
    existing = _write(root / "wiki" / "cgt" / "here.md", "content\n")
    absent = root / "wiki" / "cgt" / "not-yet.md"

    snap = mut.snapshot([existing, absent], operation="promote")
    by_target = {e.target: e for e in snap.entries}

    assert by_target[existing.resolve()].existed_before is True
    assert by_target[absent.resolve()].existed_before is False
    snap.discard()
