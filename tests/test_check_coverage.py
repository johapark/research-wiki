"""Tests for check-coverage (Pattern 2 — Starling-transferable).

The CLI shape mirrors check-grounding/grade synthesis: takes a page, exits 0
when the topic_seed retrieves no unreferenced hits, exits 1 otherwise, exits 2
on missing/malformed input. Internally it reuses
`researchwiki.tasks.lint.staleness.unreferenced_top_hits` — the helper that
also powers `lint`'s `stale_by_content` check — so the unit-level coverage
already exists. These tests pin the CLI surface itself.
"""

from __future__ import annotations

from pathlib import Path

from researchwiki.tasks import check_coverage


def _write(p: Path, body: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def test_missing_path_returns_2(tmp_path, capsys):
    rc = check_coverage.main([str(tmp_path / "does-not-exist.md")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a file" in err


def test_missing_topic_seed_returns_2(tmp_path, capsys):
    md = _write(tmp_path / "no-seed.md", "---\ntitle: x\n---\n\nbody\n")
    rc = check_coverage.main([str(md)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "topic_seed" in err


def test_malformed_frontmatter_returns_2(tmp_path, capsys):
    md = _write(tmp_path / "bad.md", "---\ntitle: [unterminated\n---\n\nbody\n")
    rc = check_coverage.main([str(md)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "malformed frontmatter" in err


def test_returns_0_when_seed_present_and_no_unreferenced_hits(tmp_path, monkeypatch, capsys):
    """If the backend returns nothing, the page is trivially complete — exit 0."""
    md = _write(
        tmp_path / "page.md",
        "---\ntitle: t\ntopic_seed: alpha beta\n---\n\nbody [[other/cited]]\n",
    )

    class _Backend:
        def query(self, q, limit=1):
            return []
        def more_like_text(self, seed, limit, page_type=None):
            return []

    def _fake_backend():
        return _Backend()

    # Patch the search backend, the page-walker, and the wiki_dir-bound
    # `page_key` (which would otherwise reject tmp paths).
    import researchwiki.search as search_mod
    monkeypatch.setattr(search_mod, "get_default_backend", _fake_backend, raising=False)
    monkeypatch.setattr("researchwiki.tasks.check_coverage.all_pages", lambda: [md], raising=True)
    monkeypatch.setattr(
        "researchwiki.tasks.lint.staleness.page_key",
        lambda p: f"synthesis/{p.stem}",
        raising=True,
    )
    monkeypatch.setattr(
        "researchwiki.tasks.check_coverage.page_key",
        lambda p: f"synthesis/{p.stem}",
        raising=True,
    )

    rc = check_coverage.main([str(md)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no unreferenced top-N hits" in out


def test_returns_1_when_unreferenced_hit_surfaces(tmp_path, monkeypatch, capsys):
    md = _write(
        tmp_path / "page.md",
        "---\ntitle: t\ntopic_seed: alpha beta\n---\n\nbody [[cited/foo]]\n",
    )
    other_md = _write(tmp_path / "other" / "uncited.md", "x")

    class _Hit:
        def __init__(self, key, stem, score, title):
            self.key = key
            self.stem = stem
            self.score = score
            self.title = title

    class _Backend:
        def query(self, q, limit=1):
            return []
        def more_like_text(self, seed, limit, page_type=None):
            return [_Hit("other/uncited", "uncited", 12.5, "Uncited paper")]

    def _fake_backend():
        return _Backend()

    import researchwiki.search as search_mod
    monkeypatch.setattr(search_mod, "get_default_backend", _fake_backend, raising=False)
    monkeypatch.setattr(
        "researchwiki.tasks.check_coverage.all_pages",
        lambda: [md, other_md],
        raising=True,
    )
    def _key(p):
        return "synthesis/page" if p == md else f"other/{p.stem}"
    monkeypatch.setattr("researchwiki.tasks.lint.staleness.page_key", _key, raising=True)
    monkeypatch.setattr("researchwiki.tasks.check_coverage.page_key", _key, raising=True)

    rc = check_coverage.main([str(md)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "unreferenced hit" in out
    assert "other/uncited" in out
