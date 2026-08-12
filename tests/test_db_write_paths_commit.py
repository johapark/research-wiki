"""Wiki-page writers must reconcile the DB, not defer to the next `db rebuild`.

The DB is a derived index, and `wiki.commit_page` exists so a caller that just
wrote one page can refresh that page's row for the cost of one transaction.
Several writers historically called `write_text_atomic` directly and skipped
it, so every one of them left `papers.indexed_at` behind the file's mtime —
which `db verify` reports as `stale`, and which makes `status` / `db query` /
`claims` read a stale row until somebody remembers to rebuild.

The frontmatter these paths touch is not incidental to the DB: `backfill doi`
writes the dedicated `doi` column, `attach` writes `supplementary:` into
`raw_frontmatter`, and both the memory-evolve and claim-graph appliers rewrite
`generated_at:`.

These tests pin the reconcile behaviour per write path. The negative control
(`test_raw_write_is_detected_as_stale`) is what keeps them honest — without it
they'd pass just as well against a `db verify` that reported nothing.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from researchwiki.db.connection import get_connection
from researchwiki.db.rebuild import rebuild
from researchwiki.db.verify import verify
from researchwiki.fsatomic import write_text_atomic

STEM = "smith-2024-a-test-paper"

# Carries `doi:` so `_replace_or_insert` takes its *replace* branch. Without an
# existing key that helper delegates to `_insert_after_key`, and a test aimed at
# the replace branch would silently exercise the insert branch instead.
PAGE = (
    "---\n"
    "title: Test Paper\n"
    "type: paper\n"
    "category: [compbio]\n"
    "year: 2024\n"
    "doi: 10.0000/placeholder\n"
    "short_name: Test\n"
    "---\n"
    "## Key Contributions\n"
    "- A claim.\n"
)


@pytest.fixture
def page(tmp_path, monkeypatch) -> Path:
    """A rebuilt single-page wiki, with the DB already in step with the file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    d = tmp_path / "wiki" / "compbio"
    d.mkdir(parents=True)
    p = d / f"{STEM}.md"
    p.write_text(PAGE, encoding="utf-8")
    rebuild()
    assert verify().stale == []
    # mtime has 1s granularity on some filesystems; make any bump observable.
    time.sleep(1.1)
    return p


def _indexed_at() -> int:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT indexed_at FROM papers WHERE stem = ?", (STEM,)
        ).fetchone()["indexed_at"]
    finally:
        conn.close()


def test_raw_write_is_detected_as_stale(page):
    """Negative control: without a commit, the write goes stale.

    If this ever fails, the assertions below stop proving anything.
    """
    write_text_atomic(page, page.read_text(encoding="utf-8"))

    stale = verify().stale
    assert [s[0] for s in stale] == [STEM]


def _doi_in_db() -> str | None:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT doi FROM papers WHERE stem = ?", (STEM,)
        ).fetchone()["doi"]
    finally:
        conn.close()


def test_backfill_replace_branch_commits(page):
    """`doi:` already present → the in-place replace branch, which writes alone."""
    from researchwiki.tasks.backfill import _replace_or_insert

    assert _doi_in_db() == "10.0000/placeholder", "fixture must start indexed"

    _replace_or_insert(page, "doi", "10.1038/example")

    assert verify().stale == []
    # The dedicated column, not merely the mtime, is refreshed.
    assert _doi_in_db() == "10.1038/example"


def test_backfill_insert_branch_commits(page):
    """Key absent → `_replace_or_insert` delegates, and the delegate commits."""
    from researchwiki.tasks.backfill import _replace_or_insert

    _replace_or_insert(page, "venue", "Nature")

    assert verify().stale == []
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT venue FROM papers WHERE stem = ?", (STEM,)
        ).fetchone()
    finally:
        conn.close()
    assert row["venue"] == "Nature"


def test_backfill_insert_after_key_commits(page):
    """`_insert_after_key` called directly (the `keywords` / `hook` paths)."""
    from researchwiki.tasks.backfill import _insert_after_key

    _insert_after_key(page, "venue: Nature", "year")

    assert verify().stale == []


def test_backfill_insert_keywords_line_commits(page):
    """`backfill keywords` writes through its own helper, not the shared one."""
    from researchwiki.tasks.backfill import _insert_keywords_line

    _insert_keywords_line(page, "keywords: [alpha, beta, gamma, delta, epsilon]")

    assert verify().stale == []
    assert "keywords: [alpha, beta, gamma, delta, epsilon]" in page.read_text(
        encoding="utf-8"
    )


def test_backfill_missing_anchor_does_not_commit_or_write(page):
    """No anchor → no write, so nothing to reconcile and no spurious refresh."""
    from researchwiki.tasks.backfill import _insert_after_key

    before_text = page.read_text(encoding="utf-8")
    before_indexed = _indexed_at()

    _insert_after_key(page, "venue: Nature", "no-such-key")

    assert page.read_text(encoding="utf-8") == before_text
    assert _indexed_at() == before_indexed


def test_attach_supplementary_commits(page):
    from researchwiki.tasks.attach import _insert_supplementary

    _insert_supplementary(page, "table_s1.xlsx", "table")

    assert verify().stale == []
    assert "supplementary:" in page.read_text(encoding="utf-8")


def test_evolution_auto_apply_commits(tmp_path, monkeypatch):
    """The memory-evolve applier rewrites `generated_at:` — it must reconcile.

    Spy rather than `db verify` here: this path takes `wiki_root_dir` directly
    and never builds a DB, so the call itself is the observable behaviour.
    """
    from researchwiki.agents.phases import evolution

    target = tmp_path / "synthesis" / "target-page.md"
    target.parent.mkdir(parents=True)
    target.write_text(
        "---\ntitle: Target Page\ntype: synthesis\n"
        "referenced_papers:\n  - [[other/existing-paper]]\n"
        "generated_at: 2026-01-01\ntags: [synthesis]\n---\n\n"
        "## Evidence\n- existing bullet\n",
        encoding="utf-8",
    )

    committed: list[Path] = []
    monkeypatch.setattr(evolution, "commit_page", committed.append)

    proposal = evolution.EvolutionProposal(
        source_key="ai/source-paper",
        target_key="synthesis/target-page",
        verdict="refine",
        confidence=0.95,
        rationale="test",
        patch={
            "add_bullet_under": "## Evidence",
            "bullet_text": "[[ai/source-paper]] — new contribution",
        },
    )

    ok, reason = evolution.auto_apply_proposal(proposal, wiki_root_dir=tmp_path)

    assert ok, reason
    assert "generated_at: 2026-01-01" not in target.read_text(encoding="utf-8")
    assert committed == [target]


def test_attach_supplementary_extends_existing_block_and_commits(page):
    """Second attach takes the block-extension branch, which writes separately."""
    from researchwiki.tasks.attach import _insert_supplementary

    _insert_supplementary(page, "table_s1.xlsx", "table")
    time.sleep(1.1)
    _insert_supplementary(page, "figure_s2.pdf", "figure")

    assert verify().stale == []
    text = page.read_text(encoding="utf-8")
    assert "table_s1.xlsx" in text and "figure_s2.pdf" in text
