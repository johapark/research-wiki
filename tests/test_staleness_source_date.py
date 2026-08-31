"""`stale_synthesis` must never read filesystem mtime.

mtime moves for any edit at all, including edits that change nothing a page
*says*. Three times in two days a mechanical maintenance pass — rewriting
back-link bullets, then stripping one YAML key from 402 files — spiked this
check into double digits with pure artifacts, and the artifacts are permanent:
mtimes never move back, and the only way to clear one is to bump `generated_at`,
which falsely claims a review that never happened.

So `_source_change_date` has three tiers and no mtime tier. These tests pin that.
"""

from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path

import pytest

from researchwiki.tasks.lint.staleness import _source_change_date, find_stale_synthesis


def _page(p: Path, body: str = "## Summary\n\ntext\n", **fm) -> Path:
    lines = ["---", "type: paper"]
    lines += [f"{k}: {v}" for k, v in fm.items()]
    lines += ["---", "", body]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


class TestTiers:
    def test_tier1_yaml_ingested_at_wins(self, tmp_path):
        p = _page(tmp_path / "a.md", ingested_at="2026-03-04T10:00:00")
        assert _source_change_date(p, {"ingested_at": "2026-03-04T10:00:00"}) == date(2026, 3, 4)

    def test_tier1_tolerates_quotes(self, tmp_path):
        p = _page(tmp_path / "a.md")
        assert _source_change_date(p, {"ingested_at": '"2026-03-04T10:00:00"'}) == date(2026, 3, 4)

    def test_tier2_db_date_when_yaml_absent(self, tmp_path):
        p = _page(tmp_path / "steinegger-2017-mmseqs2.md")
        got = _source_change_date(p, {}, {"steinegger-2017-mmseqs2": date(2026, 1, 9)})
        assert got == date(2026, 1, 9)

    def test_tier1_beats_tier2(self, tmp_path):
        p = _page(tmp_path / "a.md")
        got = _source_change_date(
            p, {"ingested_at": "2026-03-04T10:00:00"}, {"a": date(2020, 1, 1)}
        )
        assert got == date(2026, 3, 4)

    def test_tier3_is_none_not_a_guess(self, tmp_path):
        """Unknown must be unknown. `None` means callers skip the page."""
        p = _page(tmp_path / "a.md")
        assert _source_change_date(p, {}) is None
        assert _source_change_date(p, {}, {}) is None
        assert _source_change_date(p, {}, {"someone-else": date(2026, 1, 1)}) is None

    def test_unparseable_yaml_falls_through_rather_than_raising(self, tmp_path):
        p = _page(tmp_path / "a.md")
        assert _source_change_date(p, {"ingested_at": "not a date"}) is None
        assert _source_change_date(p, {"ingested_at": "nope"}, {"a": date(2026, 2, 2)}) == date(2026, 2, 2)


class TestMtimeIsIgnored:
    """The regression guard for the actual bug."""

    def test_a_future_mtime_produces_no_date(self, tmp_path):
        p = _page(tmp_path / "eddy-2011-hmm.md")
        future = time.time() + 86400 * 365
        os.utime(p, (future, future))
        assert _source_change_date(p, {}) is None

    def test_a_future_mtime_does_not_override_yaml(self, tmp_path):
        p = _page(tmp_path / "a.md")
        future = time.time() + 86400 * 365
        os.utime(p, (future, future))
        assert _source_change_date(p, {"ingested_at": "2026-03-04T10:00:00"}) == date(2026, 3, 4)


@pytest.fixture()
def tmp_wiki(tmp_path, monkeypatch):
    w = tmp_path / "wiki"
    (w / "compbio").mkdir(parents=True)
    (w / "synthesis").mkdir(parents=True)
    fake = lambda: w
    monkeypatch.setattr("researchwiki.paths.wiki_dir", fake)
    monkeypatch.setattr("researchwiki.tasks.lint.walk.wiki_dir", fake)
    return w


def _synthesis(w: Path, slug: str, generated_at: str, refs: list[str]) -> Path:
    body = "## Overview\n\n" + "\n".join(f"- [[{r}]]" for r in refs) + "\n"
    p = w / "synthesis" / f"{slug}.md"
    p.write_text(
        "---\ntype: synthesis\ncategory: [compbio]\n"
        f"generated_at: {generated_at}\n---\n\n{body}",
        encoding="utf-8",
    )
    return p


class TestFindStaleSynthesis:
    def _run(self, w, monkeypatch, db_dates=None):
        from researchwiki.tasks.lint import staleness
        from researchwiki.wiki import read_page
        from researchwiki.tasks.lint.walk import all_pages, page_key
        monkeypatch.setattr(staleness, "_db_ingest_dates", lambda: db_dates or {})
        pages = all_pages()
        fms = {p: (read_page(p).fm or {}) for p in pages}
        return find_stale_synthesis(pages, fms, {page_key(p) for p in pages})

    def test_fires_when_a_referenced_paper_is_genuinely_newer(self, tmp_wiki, monkeypatch):
        """The check must still work — de-noising it is not the same as disabling it."""
        _page(tmp_wiki / "compbio" / "new-2026-a-recent-paper.md",
              ingested_at="2026-08-04T12:00:00")
        _synthesis(tmp_wiki, "map", "2026-07-01", ["compbio/new-2026-a-recent-paper"])
        out = self._run(tmp_wiki, monkeypatch)
        assert [p.stem for p, _ in out] == ["map"]
        assert out[0][1] == ["compbio/new-2026-a-recent-paper"]

    def test_quiet_when_the_page_postdates_its_sources(self, tmp_wiki, monkeypatch):
        _page(tmp_wiki / "compbio" / "old-2026-an-earlier-paper.md",
              ingested_at="2026-08-04T12:00:00")
        _synthesis(tmp_wiki, "map", "2026-08-05", ["compbio/old-2026-an-earlier-paper"])
        assert self._run(tmp_wiki, monkeypatch) == []

    def test_same_day_is_not_stale(self, tmp_wiki, monkeypatch):
        _page(tmp_wiki / "compbio" / "x-2026-same-day-paper.md",
              ingested_at="2026-08-05T23:59:00")
        _synthesis(tmp_wiki, "map", "2026-08-05", ["compbio/x-2026-same-day-paper"])
        assert self._run(tmp_wiki, monkeypatch) == []

    def test_a_touched_but_unchanged_paper_does_not_age_the_page(self, tmp_wiki, monkeypatch):
        """The maintenance-pass scenario: file rewritten, substance identical."""
        p = _page(tmp_wiki / "compbio" / "old-2026-an-earlier-paper.md",
                  ingested_at="2026-01-02T12:00:00")
        _synthesis(tmp_wiki, "map", "2026-07-01", ["compbio/old-2026-an-earlier-paper"])
        future = time.time() + 86400 * 30
        os.utime(p, (future, future))
        assert self._run(tmp_wiki, monkeypatch) == []

    def test_a_dateless_paper_is_skipped_not_assumed_new(self, tmp_wiki, monkeypatch):
        p = _page(tmp_wiki / "compbio" / "eddy-2011-legacy-page.md")
        future = time.time() + 86400 * 30
        os.utime(p, (future, future))
        _synthesis(tmp_wiki, "map", "2026-07-01", ["compbio/eddy-2011-legacy-page"])
        assert self._run(tmp_wiki, monkeypatch) == []

    def test_db_date_can_still_flag_a_dateless_paper(self, tmp_wiki, monkeypatch):
        """Tier 2 keeps real signal that a pure skip would have lost."""
        _page(tmp_wiki / "compbio" / "nodate-2026-no-yaml-stamp.md")
        _synthesis(tmp_wiki, "map", "2026-07-01", ["compbio/nodate-2026-no-yaml-stamp"])
        out = self._run(tmp_wiki, monkeypatch,
                        db_dates={"nodate-2026-no-yaml-stamp": date(2026, 8, 4)})
        assert [p.stem for p, _ in out] == ["map"]

    def test_concept_and_idea_pages_are_covered_too(self, tmp_wiki, monkeypatch):
        (tmp_wiki / "concepts").mkdir()
        _page(tmp_wiki / "compbio" / "new-2026-a-recent-paper.md",
              ingested_at="2026-08-04T12:00:00")
        (tmp_wiki / "concepts" / "hub.md").write_text(
            "---\ntype: concept\ngenerated_at: 2026-07-01\n---\n\n"
            "## Definition\n\n- [[compbio/new-2026-a-recent-paper]]\n", encoding="utf-8")
        out = self._run(tmp_wiki, monkeypatch)
        assert [p.stem for p, _ in out] == ["hub"]
