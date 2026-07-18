"""`db rebuild` must not treat wiki/-root bookkeeping files as parse errors.

`index.md`, `log.md`, and `pdfs-failed-parsing.md` live directly under
wiki/ and carry no YAML frontmatter — they aren't pages. Before this fix,
rebuild() walked them via wiki/**/*.md, `_parse_frontmatter` returned None
for each, and every single rebuild reported nonzero parse_errors — which
`tasks/db.py` maps to exit code 2, permanently, regardless of actual page
health, and which also short-circuits before the post-rebuild claim-graph
reconcile step runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.db.rebuild import rebuild


def _write_paper(wiki_dir: Path, category: str, stem: str) -> None:
    d = wiki_dir / category
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.md").write_text(
        "---\n"
        "title: Test Paper\n"
        "type: paper\n"
        "category: [" + category + "]\n"
        "---\n"
        "## Key Contributions\n"
        "- A claim.\n"
    )


@pytest.fixture
def wiki_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    return wiki_dir


def test_root_meta_files_are_not_parse_errors(wiki_root):
    (wiki_root / "index.md").write_text("# Wiki Index\n\nNo frontmatter here.\n")
    (wiki_root / "log.md").write_text("\n## [2026-01-01] ingest | x\nfoo\n")
    (wiki_root / "pdfs-failed-parsing.md").write_text("# Failed PDFs\n\n- none yet\n")
    _write_paper(wiki_root, "compbio", "smith-2024-a-test-paper")

    stats = rebuild()

    assert stats.parse_errors == []
    assert stats.papers_upserted == 1
    # Meta files aren't counted as scanned pages either.
    assert stats.pages_scanned == 1


def test_genuinely_malformed_page_still_reported(wiki_root):
    """A real parse failure (frontmatter missing/malformed) inside a category
    dir must still surface — only the three known root-level meta files are
    exempt."""
    bad = wiki_root / "compbio" / "broken.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("no frontmatter at all\n")

    stats = rebuild()

    assert len(stats.parse_errors) == 1
    assert str(bad) in stats.parse_errors[0]


def test_frontmattered_root_dashboard_still_indexed(wiki_root):
    """views.md sits at wiki/ root too, but carries real frontmatter
    (type: dashboard) — it's a page, not a meta file, and must still be
    upserted."""
    (wiki_root / "views.md").write_text(
        "---\ntitle: Dashboard\ntype: dashboard\n---\nbody\n"
    )

    stats = rebuild()

    assert stats.parse_errors == []
    assert stats.papers_upserted == 1


def test_same_section_text_swap_does_not_abort_rebuild(wiki_root):
    """Two claims in one section that swap text between rebuilds must not trip
    UNIQUE(paper_stem, claim_slug) and abort the whole rebuild (review T1.4)."""
    d = wiki_root / "compbio"
    d.mkdir(parents=True, exist_ok=True)
    page = d / "smith-2024-swap.md"

    def write(c1: str, c2: str) -> None:
        page.write_text(
            "---\ntitle: T\ntype: paper\ncategory: [compbio]\n---\n"
            "## Key Contributions\n- " + c1 + "\n- " + c2 + "\n",
            encoding="utf-8",
        )

    write("Alpha improves speed.", "Beta improves memory.")
    rebuild()
    # Swap the two claims' text between positions.
    write("Beta improves memory.", "Alpha improves speed.")
    stats = rebuild()  # must not raise IntegrityError
    assert stats.parse_errors == []
