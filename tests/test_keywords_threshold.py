"""The `keywords:` minimum must be one number everywhere.

It wasn't. Four sites carried three different thresholds:

    render_keywords_yaml        refuses to write below 5
    lint find_missing_keywords  flagged below 3
    backfill candidates         selected below 3
    CLAUDE.md reference template said 6-10

That left a dead zone at 3-4 keywords where a page was simultaneously **not
flagged** by lint, **not selected** by `backfill keywords`, and **unwritable**
even if it had been selected — three locks, no key. A page could sit there
forever and every surface reported it as fine.

`agents.phases.commit.MIN_KEYWORDS` is now the canonical owner (the writer is
what enforces it). `lint` mirrors the value rather than importing it, because
`tasks.lint` pulls in no `agents` module and importing one costs ~107 ms on a
~1 s command — so this file is what forbids the copy from drifting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.agents.phases import MAX_KEYWORDS, MIN_KEYWORDS
from researchwiki.agents.phases.commit import render_keywords_yaml
from researchwiki.tasks.lint import yaml_checks


def test_lint_mirror_matches_the_canonical_constant():
    """The whole reason this file exists."""
    assert yaml_checks.MIN_KEYWORDS == MIN_KEYWORDS, (
        "tasks/lint/yaml_checks.py MIN_KEYWORDS has drifted from "
        "agents/phases/commit.py MIN_KEYWORDS — they must agree, or the dead "
        "zone this fixed comes back"
    )


def test_backfill_selects_exactly_what_the_writer_can_fix():
    """Selection floor and write floor must be the same number: a page selected
    for backfill must be one `render_keywords_yaml` would actually write."""
    import researchwiki.tasks.backfill as backfill
    assert backfill.MIN_KEYWORDS == MIN_KEYWORDS


def test_writer_refuses_below_the_floor_and_accepts_at_it():
    below = [f"kw{i}" for i in range(MIN_KEYWORDS - 1)]
    at = [f"kw{i}" for i in range(MIN_KEYWORDS)]
    assert render_keywords_yaml(below) is None
    assert render_keywords_yaml(at) == f"keywords: [{', '.join(at)}]"


def test_the_dead_zone_is_closed(tmp_path, monkeypatch):
    """A page with MIN_KEYWORDS-1 keywords must be *flagged*, not silently OK.

    Under the old thresholds a 4-keyword page passed lint entirely.
    """
    wiki = tmp_path / "wiki"
    (wiki / "compbio").mkdir(parents=True)
    fake = lambda: wiki  # noqa: E731
    monkeypatch.setattr("researchwiki.paths.wiki_dir", fake)
    monkeypatch.setattr("researchwiki.tasks.lint.walk.wiki_dir", fake)

    from researchwiki.tasks.lint.walk import all_pages
    from researchwiki.wiki import read_page

    kw = ", ".join(f"kw{i}" for i in range(MIN_KEYWORDS - 1))
    (wiki / "compbio" / "a-2026-x.md").write_text(
        f"---\ntype: paper\ntitle: A\nkeywords: [{kw}]\n---\n\n## Summary\n\nBody.\n"
    )
    pages = all_pages()
    fm = {p: (read_page(p).fm if read_page(p) else {}) for p in pages}

    found = yaml_checks.find_missing_keywords(pages, fm)
    assert found == [("compbio/a-2026-x", MIN_KEYWORDS - 1)]


def test_a_page_at_the_floor_is_not_flagged(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    (wiki / "compbio").mkdir(parents=True)
    fake = lambda: wiki  # noqa: E731
    monkeypatch.setattr("researchwiki.paths.wiki_dir", fake)
    monkeypatch.setattr("researchwiki.tasks.lint.walk.wiki_dir", fake)

    from researchwiki.tasks.lint.walk import all_pages
    from researchwiki.wiki import read_page

    kw = ", ".join(f"kw{i}" for i in range(MIN_KEYWORDS))
    (wiki / "compbio" / "b-2026-y.md").write_text(
        f"---\ntype: paper\ntitle: B\nkeywords: [{kw}]\n---\n\n## Summary\n\nBody.\n"
    )
    pages = all_pages()
    fm = {p: (read_page(p).fm if read_page(p) else {}) for p in pages}
    assert yaml_checks.find_missing_keywords(pages, fm) == []


def test_bounds_are_sane():
    assert 0 < MIN_KEYWORDS <= MAX_KEYWORDS


#: Repo root, derived from this file — never from the cwd. A relative
#: `Path("CLAUDE.md")` only resolves when pytest happens to run from the repo
#: root, which is not what IDE test runners do.
_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("filename,needle", [
    ("CLAUDE.md", "5–10 short phrases"),
])
def test_docs_state_the_same_range(filename, needle):
    """CLAUDE.md said 6-10 while every prompt and the writer said 5-10."""
    text = (_REPO_ROOT / filename).read_text(encoding="utf-8")
    assert needle in text, (
        f"{filename} no longer states the {MIN_KEYWORDS}-{MAX_KEYWORDS} range"
    )
    assert "6–10 short phrases" not in text
