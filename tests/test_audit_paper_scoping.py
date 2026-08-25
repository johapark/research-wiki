"""Paper-only consumers must not absorb DOI-bearing books or commentaries.

`read_wiki_papers` is the shared boundary for audit, preprint-check, and
retraction-check.  When it filtered only on DOI, five books plus one commentary
made the live audit report more auditable papers than total paper pages and
produced `papers_skipped_no_doi: -6`.

All provider calls here are fakes; the tests perform no network requests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchwiki import wiki as wiki_module
from researchwiki.providers import ScholarlyArticle
from researchwiki.tasks import audit, preprint_check, retraction_check


def _write_page(
    root: Path,
    relative: str,
    *,
    page_type: str | None,
    doi: str | None,
    no_doi_reason: str | None = None,
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = ["---", f'title: "{path.stem}"']
    if page_type is not None:
        frontmatter.append(f"type: {page_type}")
    if doi is not None:
        frontmatter.append(f'doi: "{doi}"')
    if no_doi_reason is not None:
        frontmatter.append(f'no_doi_reason: "{no_doi_reason}"')
    frontmatter.extend(["year: 2026", "---", "", "## Summary", "", "Body.", ""])
    path.write_text("\n".join(frontmatter), encoding="utf-8")


@pytest.fixture
def scoped_wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "wiki"
    _write_page(
        root, "genomics/paper.md", page_type="paper", doi="10.1101/paper.1"
    )
    # Missing type intentionally retains the framework-wide legacy default of
    # `paper`; lint reports the schema defect without silently dropping it.
    _write_page(
        root, "genomics/legacy.md", page_type=None, doi="10.1000/legacy"
    )
    # A stale explanatory field can survive after a DOI is added. It must not
    # be double-counted as both audited and intentionally DOI-less.
    _write_page(
        root,
        "genomics/doi-with-stale-reason.md",
        page_type="paper",
        doi="10.1000/resolved",
        no_doi_reason="historically unavailable",
    )
    _write_page(root, "genomics/no-doi.md", page_type="paper", doi=None)
    _write_page(
        root, "references/book.md", page_type="book", doi="10.1000/book"
    )
    _write_page(
        root,
        "references/legacy-reference.md",
        page_type=None,
        doi="10.1000/legacy-reference",
    )
    _write_page(
        root,
        "genomics/commentary.md",
        page_type="commentary",
        doi="10.1101/commentary.1",
    )
    _write_page(
        root,
        "synthesis/field-map.md",
        page_type="synthesis",
        doi="10.1000/synthesis",
    )
    # Structural directories never enter paper scope, even when malformed
    # frontmatter claims otherwise. In particular, audit's denominator must use
    # the same PAGE_TYPE_DIRS boundary as read_wiki_papers (which includes ideas).
    _write_page(
        root,
        "ideas/malformed-paper.md",
        page_type="paper",
        doi="10.1000/not-a-paper",
    )
    monkeypatch.setattr(wiki_module, "wiki_dir", lambda: root)
    return root


def test_read_wiki_papers_keeps_only_paper_page_types(scoped_wiki: Path) -> None:
    papers = wiki_module.read_wiki_papers()
    assert {p["stem"] for p in papers} == {
        "paper", "legacy", "doi-with-stale-reason",
    }


class _AuditProvider:
    def __init__(self) -> None:
        self.batched: list[str] = []

    def get_batch_metadata(self, dois: list[str]) -> None:
        self.batched.extend(dois)

    def get_by_doi(self, doi: str) -> ScholarlyArticle:
        return ScholarlyArticle(doi=doi, reference_count=0, citation_count=0)

    def get_references(self, article: ScholarlyArticle) -> list[ScholarlyArticle]:
        return []

    def get_citations(self, article: ScholarlyArticle) -> list[ScholarlyArticle]:
        return []

    def get_recommendations(self, article: ScholarlyArticle) -> list[ScholarlyArticle]:
        return []


def test_audit_denominator_cannot_go_negative_from_non_paper_dois(
    scoped_wiki: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = _AuditProvider()
    monkeypatch.setattr(audit, "get_default_provider", lambda **kwargs: provider)
    monkeypatch.setattr(audit, "s2_cache_dir", lambda: tmp_path / "audit-cache")

    assert audit.main(["--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["total_paper_pages"] == 4
    assert report["papers_skipped_no_doi"] == 1
    assert {p["stem"] for p in report["papers"]} == {
        "paper", "legacy", "doi-with-stale-reason",
    }
    assert set(provider.batched) == {
        "10.1101/paper.1", "10.1000/legacy", "10.1000/resolved",
    }


def test_preprint_scan_excludes_commentary_with_preprint_doi(
    scoped_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[str] = []

    def fake_lookup(doi: str) -> dict:
        seen.append(doi)
        return {
            "doi": doi,
            "server": "biorxiv",
            "title": "Paper",
            "version": 1,
            "date_posted": "2026-01-01",
            "category": "genomics",
            "type": "research_article",
            "published_doi": None,
            "source": "biorxiv",
            "fetched_at": "2026-08-20",
        }

    monkeypatch.setattr(preprint_check, "lookup", fake_lookup)
    assert preprint_check.main(["--all", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["scanned"] == 1
    assert seen == ["10.1101/paper.1"]


def test_retraction_scan_excludes_all_non_paper_page_types(
    scoped_wiki: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[str] = []

    def fake_status(doi: str) -> dict:
        seen.append(doi)
        return {
            "doi": doi,
            "pmid": "1",
            "retracted": False,
            "is_retraction_notice": False,
        }

    monkeypatch.setattr(retraction_check, "retraction_status", fake_status)
    assert retraction_check.main(["--all", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["scanned"] == 3
    assert set(seen) == {
        "10.1101/paper.1", "10.1000/legacy", "10.1000/resolved",
    }
