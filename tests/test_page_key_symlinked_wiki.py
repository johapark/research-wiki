"""`page_key()` must work when `wiki/` is a symlink into a synced folder.

That layout (prompts/migration-backfill.md) is the recommended one for users who
sync their library, and it makes callers that resolve paths disagree with
callers that don't — `check-coverage` resolves its argument, `all_pages()` does
not. Before the resolved-path fallback, the mismatch raised ValueError out of
`relative_to` and the CLI reported it as an internal error (exit 3).
"""

from __future__ import annotations

from researchwiki.tasks.lint.walk import page_key


def _make_vault(tmp_path):
    """Repo at `repo/` whose `wiki/` is a symlink to `vault/wiki/`."""
    external = tmp_path / "vault" / "wiki" / "ideas"
    external.mkdir(parents=True)
    page = external / "some-idea.md"
    page.write_text("# x\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "wiki").symlink_to(tmp_path / "vault" / "wiki")
    return repo, page


def test_resolved_path_through_symlinked_wiki(tmp_path, monkeypatch):
    repo, page = _make_vault(tmp_path)
    monkeypatch.chdir(repo)

    # What check-coverage does: Path(arg).resolve() — lands in the vault.
    assert page_key(page.resolve()) == "ideas/some-idea"


def test_unresolved_path_through_symlinked_wiki(tmp_path, monkeypatch):
    repo, _ = _make_vault(tmp_path)
    monkeypatch.chdir(repo)

    # What all_pages() yields: built from the unresolved wiki_dir().
    assert page_key(repo / "wiki" / "ideas" / "some-idea.md") == "ideas/some-idea"


def test_plain_directory_layout_still_works(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "wiki" / "ai").mkdir(parents=True)
    page = repo / "wiki" / "ai" / "smith-2024-a-paper.md"
    page.write_text("# x\n")
    monkeypatch.chdir(repo)

    assert page_key(page) == "ai/smith-2024-a-paper"
    assert page_key(page.resolve()) == "ai/smith-2024-a-paper"
