"""Two fixes that shipped on 2026-08-06 with no test pinning them.

Both were found by reconciling `status` against the DB and against `lint`, and
both were silent under-counting rather than errors — the kind of defect a green
suite is least likely to notice, which is exactly why they need tests.

  1. `status` filtered on `fm.get("type") == "paper"`, dropping `commentary`
     pages and the pages carrying no `type:` at all (which `db rebuild` records
     as `paper`). Those pages also fell out of the cross-link graph, silently
     deleting every edge touching them.
  2. `lint`'s orphan graph counted `index.md` bullets as in-links. Since the
     catalogue lists every page, the check could not fire once the catalogue was
     complete — it read 0 while `status` read 18.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.tasks.lint.link_checks import build_link_graph, find_orphans
from researchwiki.tasks.lint.walk import all_pages, page_key


@pytest.fixture()
def tmp_wiki(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    (wiki / "compbio").mkdir(parents=True)
    fake = lambda: wiki  # noqa: E731
    monkeypatch.setattr("researchwiki.paths.wiki_dir", fake)
    monkeypatch.setattr("researchwiki.tasks.lint.walk.wiki_dir", fake)
    return wiki


def _page(wiki: Path, rel: str, body: str = "", **fm) -> Path:
    p = wiki / f"{rel}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    head = "\n".join(f"{k}: {v}" for k, v in fm.items())
    p.write_text(f"---\n{head}\n---\n\n{body}\n", encoding="utf-8")
    return p


def _graph(wiki: Path):
    from researchwiki.wiki import read_page, strip_non_prose
    pages = all_pages()
    known = {page_key(p) for p in pages}
    prose = {}
    for md in pages:
        pg = read_page(md)
        prose[md] = strip_non_prose(pg.body if pg else md.read_text(encoding="utf-8"))
    return pages, known, build_link_graph(pages, prose, known)


class TestIndexDoesNotDeOrphan:
    def test_an_index_bullet_alone_leaves_a_page_orphaned(self, tmp_wiki):
        """The exact shape of `genomics/oberlin-2025-…` on 2026-08-06: its only
        inbound link was the `index.md` bullet added to it."""
        _page(tmp_wiki, "compbio/lonely-2024-a-paper-nothing-cites", type="paper")
        _page(tmp_wiki, "index", "- [[compbio/lonely-2024-a-paper-nothing-cites]] — **X**: gloss",
              type="meta")
        pages, known, (out_links, in_links, _) = _graph(tmp_wiki)
        assert find_orphans(pages, out_links, in_links) == [
            "compbio/lonely-2024-a-paper-nothing-cites"
        ]

    def test_a_real_citation_does_de_orphan(self, tmp_wiki):
        _page(tmp_wiki, "compbio/cited-2024-a-paper-with-an-inbound-link", type="paper")
        _page(tmp_wiki, "compbio/citer-2025-a-paper-that-cites-it",
              "## Related Papers\n\n- [[compbio/cited-2024-a-paper-with-an-inbound-link]] — cites this paper",
              type="paper")
        _page(tmp_wiki, "index",
              "- [[compbio/cited-2024-a-paper-with-an-inbound-link]] — **A**: g\n"
              "- [[compbio/citer-2025-a-paper-that-cites-it]] — **B**: g", type="meta")
        pages, known, (out_links, in_links, _) = _graph(tmp_wiki)
        # citer has an out-link, cited has a real in-link: neither is an orphan.
        assert find_orphans(pages, out_links, in_links) == []

    def test_root_out_links_are_still_recorded(self, tmp_wiki):
        """Only *in*-links from root pages are dropped. `find_missing_backlinks`
        applies its own root exclusion and the broken-link scan needs coverage."""
        _page(tmp_wiki, "compbio/target-2024-some-target-paper", type="paper")
        _page(tmp_wiki, "index", "- [[compbio/target-2024-some-target-paper]] — **X**: g",
              type="meta")
        _, _, (out_links, in_links, _) = _graph(tmp_wiki)
        assert out_links["index"] == {"compbio/target-2024-some-target-paper"}
        assert "index" not in in_links.get("compbio/target-2024-some-target-paper", set())


class TestStatusCountsEveryPage:
    """`status`'s paper filter must agree with `db rebuild`'s default."""

    def _page(self, category, **fm):
        from researchwiki.wiki import Page
        return Page(path=Path("wiki") / category / "x.md", stem="x-2024-a-paper",
                    category=category, fm=fm, body="")

    def test_a_missing_type_counts_as_paper(self, tmp_wiki):
        """`db rebuild` uses `fm.get("type", "paper")`. Reading a missing field as
        None dropped the 23 pages predating the `type:` requirement — 359 papers
        reported against the database's 382."""
        from researchwiki.tasks.status import is_paper_like
        assert is_paper_like(self._page("compbio"))                    # no type at all
        assert is_paper_like(self._page("compbio", type="paper"))
        assert is_paper_like(self._page("compbio", type=None))
        assert is_paper_like(self._page("compbio", type='"paper"'))    # quoted

    def test_commentary_counts(self, tmp_wiki):
        """9 pages, plus every cross-link edge touching them."""
        from researchwiki.tasks.status import is_paper_like
        assert is_paper_like(self._page("cgt", type="commentary"))

    def test_page_type_dirs_and_root_do_not_count(self, tmp_wiki):
        from researchwiki.tasks.status import is_paper_like
        for cat in ("synthesis", "references", "ideas", "concepts"):
            assert not is_paper_like(self._page(cat, type="paper"))
        # Root housekeeping is excluded by DIRECTORY, not by type — relying on the
        # type field is what let a type-less page through.
        assert not is_paper_like(self._page("wiki"))
        assert not is_paper_like(self._page("wiki", type="meta"))

    def test_non_paper_types_in_a_content_dir_do_not_count(self, tmp_wiki):
        from researchwiki.tasks.status import is_paper_like
        for t in ("synthesis", "concept", "idea", "whitepaper", "guidance",
                  "dashboard", "meta", "protocol", "book"):
            assert not is_paper_like(self._page("compbio", type=t)), t

    def test_commentary_is_paper_like(self):
        from researchwiki.tasks.status import PAPER_LIKE_TYPES
        assert "paper" in PAPER_LIKE_TYPES
        assert "commentary" in PAPER_LIKE_TYPES

    def test_non_paper_types_are_excluded(self):
        from researchwiki.tasks.status import PAPER_LIKE_TYPES
        for t in ("synthesis", "concept", "idea", "whitepaper", "guidance", "meta",
                  "dashboard", "protocol", "book"):
            assert t not in PAPER_LIKE_TYPES
