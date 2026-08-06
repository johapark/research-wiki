"""`paths.ensure_scaffold()` — creates the gitignored content dirs.

Nothing under `wiki/`, `papers/`, `inbox/` is committed (no `.gitkeep`), so a
fresh clone has none of these and this function is what puts them there.
"""

from __future__ import annotations

import pytest

from researchwiki.categories import DEFAULT_DIRS
from researchwiki.paths import ensure_scaffold


def test_creates_content_dirs_and_page_type_scaffold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    created = ensure_scaffold()

    for name in ("wiki", "papers", "inbox"):
        assert (tmp_path / name).is_dir(), f"{name}/ not created"
    for d in DEFAULT_DIRS:
        assert (tmp_path / "wiki" / d).is_dir(), f"wiki/{d}/ not created"
    assert created, "should report what it created"


def test_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_scaffold()

    assert ensure_scaffold() == [], "second run should create nothing"


def test_creates_through_a_symlinked_wiki(tmp_path, monkeypatch):
    """The synced-folder layout: `wiki/` is a symlink to somewhere else, and
    the scaffold must land in the target, not replace the link."""
    external = tmp_path / "synced" / "wiki"
    external.mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "wiki").symlink_to(external)
    monkeypatch.chdir(repo)

    ensure_scaffold()

    assert (repo / "wiki").is_symlink(), "must not replace the symlink"
    for d in DEFAULT_DIRS:
        assert (external / d).is_dir(), f"wiki/{d}/ should land in the link target"


def test_dangling_symlink_is_reported_not_clobbered(tmp_path, monkeypatch):
    """An unmounted synced folder leaves a dangling link. Creating a directory
    over it would strand the real content, and `mkdir` on the *child* would
    otherwise fail with a bare FileExistsError naming the parent."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "wiki").symlink_to(tmp_path / "not-mounted" / "wiki")
    monkeypatch.chdir(repo)

    with pytest.raises(FileExistsError, match="dangling symlink"):
        ensure_scaffold()

    assert (repo / "wiki").is_symlink(), "link must survive the failure"
    assert not (repo / "wiki").exists(), "must not have been materialized"
