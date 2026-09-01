"""Per-check unit tests for the lint subpackage.

After splitting `tasks/lint.py` into `tasks/lint/` (one module per check
group), each check is a pure function over already-walked page state
and is independently testable. These tests construct synthetic wiki
trees under tmp_path, monkey-patch `wiki_dir()` to point at them, and
call each check directly.

Coverage focus is on checks that previously had zero tests:
  - link graph (orphans, missing back-links, broken links)
  - yaml-shape checks (type mismatches, category drift, missing DOI,
    stem/year drift, missing keywords)
  - staleness (synthesis, audit count)
  - concepts
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.concepts.candidates import find_concept_candidates
from researchwiki.tasks.lint.link_checks import (
    build_link_graph,
    find_broken_index_bullets,
    find_missing_backlinks,
    find_orphans,
)
from researchwiki.wiki import find_orphan_pdfs
from researchwiki.tasks.lint.staleness import (
    find_stale_by_audit_count,
    find_stale_synthesis,
)
from researchwiki.tasks.lint.walk import (
    broken_links,
    count_keywords,
    extract_links,
    first_category,
    page_key,
)
from researchwiki.tasks.lint.yaml_checks import (
    MIN_KEYWORDS,
    find_category_drift,
    find_missing_doi,
    find_missing_keywords,
    find_missing_type,
    find_page_type_mismatches,
    find_stem_year_drift,
)


# ---------- fixtures ----------

@pytest.fixture
def tmp_wiki(tmp_path, monkeypatch):
    """Synthetic wiki rooted at tmp_path/wiki.

    The lint subpackage modules use `from ...paths import wiki_dir`, which
    binds the symbol at import time — patching `researchwiki.paths.wiki_dir`
    alone wouldn't reach those bindings. Patch every module that consumes
    it so synthetic wiki paths flow through `page_key()` cleanly.
    """
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    fake = lambda: wiki  # noqa: E731
    monkeypatch.setattr("researchwiki.paths.wiki_dir", fake)
    monkeypatch.setattr("researchwiki.tasks.lint.walk.wiki_dir", fake)
    return wiki


def _mkpage(wiki: Path, key: str, body: str = "") -> Path:
    """Create wiki/{key}.md with the given body. Returns the Path."""
    p = wiki / f"{key}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


# ---------- walk ----------

def test_extract_links_resolves_full_keys(tmp_wiki):
    a = _mkpage(tmp_wiki, "cgt/foo-2024-bar")
    b = _mkpage(tmp_wiki, "compbio/baz-2025-qux")
    known = {page_key(a), page_key(b)}
    text = "See [[cgt/foo-2024-bar]] and [[compbio/baz-2025-qux]]."
    assert extract_links(text, known) == known


def test_extract_links_resolves_bare_stems(tmp_wiki):
    a = _mkpage(tmp_wiki, "cgt/foo-2024-bar")
    known = {page_key(a)}
    # Bare stem with no category should still resolve.
    text = "See [[foo-2024-bar]] for context."
    assert extract_links(text, known) == known


def test_broken_links_flags_unresolved(tmp_wiki):
    a = _mkpage(tmp_wiki, "cgt/foo-2024-bar")
    known = {page_key(a)}
    text = "Real [[cgt/foo-2024-bar]] but broken [[cgt/missing-page]] and [[also-missing]]."
    bad = broken_links(text, known)
    assert "cgt/missing-page" in bad
    assert "also-missing" in bad
    assert "cgt/foo-2024-bar" not in bad


def test_first_category_handles_three_shapes():
    assert first_category("[single-cell]") == "single-cell"
    assert first_category(["compbio"]) == "compbio"
    assert first_category("ai") == "ai"
    assert first_category("") == ""


def test_count_keywords_handles_three_shapes():
    assert count_keywords(["one", "two", "three"]) == 3
    assert count_keywords("[a, b, c]") == 3
    assert count_keywords("a, b") == 2
    assert count_keywords("") == 0
    assert count_keywords(None) == 0


# ---------- link graph: orphans + missing back-links ----------

def test_find_orphans_flags_zero_link_papers(tmp_wiki):
    a = _mkpage(tmp_wiki, "cgt/orphan", "## Page\n\nNo links.")
    b = _mkpage(tmp_wiki, "cgt/linked-from", "## Page\n\nSee [[cgt/orphan]]")
    pages = [a, b]
    pages_prose = {a: "No links.", b: "See [[cgt/orphan]]"}
    known = {page_key(a), page_key(b)}
    out_links, in_links, _ = build_link_graph(pages, pages_prose, known)
    orphans = find_orphans(pages, out_links, in_links)
    # 'a' has an inbound link → not orphan. 'b' has an outbound → not orphan.
    assert orphans == []


def test_find_orphans_real_orphan(tmp_wiki):
    a = _mkpage(tmp_wiki, "cgt/orphan", "")
    b = _mkpage(tmp_wiki, "cgt/connected-1", "See [[cgt/connected-2]]")
    c = _mkpage(tmp_wiki, "cgt/connected-2", "See [[cgt/connected-1]]")
    pages = [a, b, c]
    pages_prose = {a: "", b: "See [[cgt/connected-2]]", c: "See [[cgt/connected-1]]"}
    known = {page_key(p) for p in pages}
    out_links, in_links, _ = build_link_graph(pages, pages_prose, known)
    assert find_orphans(pages, out_links, in_links) == ["cgt/orphan"]


def test_find_orphans_excludes_synthesis(tmp_wiki):
    """Synthesis pages can have zero inbound links and still not be orphans."""
    s = _mkpage(tmp_wiki, "synthesis/empty-synth", "")
    pages = [s]
    pages_prose = {s: ""}
    known = {page_key(s)}
    out_links, in_links, _ = build_link_graph(pages, pages_prose, known)
    assert find_orphans(pages, out_links, in_links) == []


def test_find_missing_backlinks_paper_to_paper(tmp_wiki):
    a = _mkpage(tmp_wiki, "cgt/a", "[[cgt/b]]")
    b = _mkpage(tmp_wiki, "cgt/b", "")  # b doesn't link back to a
    pages = [a, b]
    pages_prose = {a: "[[cgt/b]]", b: ""}
    known = {page_key(a), page_key(b)}
    out_links, _, _ = build_link_graph(pages, pages_prose, known)
    missing = find_missing_backlinks(out_links)
    assert ("cgt/a", "cgt/b") in missing


def test_find_missing_backlinks_excludes_synthesis(tmp_wiki):
    """A → synthesis is asymmetric by design; not a missing-backlink."""
    a = _mkpage(tmp_wiki, "cgt/a", "[[synthesis/topic]]")
    s = _mkpage(tmp_wiki, "synthesis/topic", "")
    pages = [a, s]
    pages_prose = {a: "[[synthesis/topic]]", s: ""}
    known = {page_key(a), page_key(s)}
    out_links, _, _ = build_link_graph(pages, pages_prose, known)
    assert find_missing_backlinks(out_links) == []


def test_find_missing_backlinks_excludes_index_and_ideas(tmp_wiki):
    """The index catalogue and idea-page grounding links are asymmetric by
    design — a paper must not be forced to back-link them."""
    idx = _mkpage(tmp_wiki, "index", "[[cgt/a]]")
    idea = _mkpage(tmp_wiki, "ideas/plan", "[[cgt/a]]")
    a = _mkpage(tmp_wiki, "cgt/a", "")
    pages = [idx, idea, a]
    pages_prose = {idx: "[[cgt/a]]", idea: "[[cgt/a]]", a: ""}
    known = {page_key(idx), page_key(idea), page_key(a)}
    out_links, _, _ = build_link_graph(pages, pages_prose, known)
    assert find_missing_backlinks(out_links) == []


def test_find_missing_backlinks_never_self_pair(tmp_wiki):
    """A page that links itself must not be flagged as owing itself a backlink."""
    a = _mkpage(tmp_wiki, "cgt/a", "[[cgt/a]]")
    pages = [a]
    pages_prose = {a: "[[cgt/a]]"}
    known = {page_key(a)}
    out_links, _, _ = build_link_graph(pages, pages_prose, known)
    assert find_missing_backlinks(out_links) == []


def test_find_missing_backlinks_flags_concept_hub(tmp_wiki):
    """A concept hub → member edge IS meant to be reciprocal, so a one-way
    concept link is a real gap (unlike synthesis/ideas)."""
    c = _mkpage(tmp_wiki, "concepts/rag", "[[cgt/a]]")
    a = _mkpage(tmp_wiki, "cgt/a", "")
    pages = [c, a]
    pages_prose = {c: "[[cgt/a]]", a: ""}
    known = {page_key(c), page_key(a)}
    out_links, _, _ = build_link_graph(pages, pages_prose, known)
    assert ("concepts/rag", "cgt/a") in find_missing_backlinks(out_links)


# ---------- yaml-shape checks ----------

def test_find_missing_type_flags_absent_and_blank(tmp_wiki):
    a = _mkpage(tmp_wiki, "cgt/a", "")
    b = _mkpage(tmp_wiki, "cgt/b", "")
    c = _mkpage(tmp_wiki, "cgt/c", "")
    fm = {a: {"title": "no type at all"}, b: {"type": "   "}, c: {"type": "paper"}}
    assert find_missing_type([a, b, c], fm) == ["cgt/a", "cgt/b"]


def test_find_missing_type_exempts_root_bookkeeping(tmp_wiki):
    # index.md / log.md carry no frontmatter by design; demanding `type` of a
    # catalogue would make the check permanently noisy and get it ignored.
    idx = _mkpage(tmp_wiki, "index", "")
    log = _mkpage(tmp_wiki, "log", "")
    assert find_missing_type([idx, log], {idx: {}, log: {}}) == []


def test_find_missing_type_is_what_page_type_mismatches_cannot_see(tmp_wiki):
    """The two checks are not redundant.

    `find_page_type_mismatches` reads `fm.get("type", "paper")`, so a commentary
    page that lost its `type` looks like a conforming paper to it. That default
    is why this check has to exist separately.
    """
    p = _mkpage(tmp_wiki, "cgt/lost-its-type", "")
    fm = {p: {"primary_paper": "[[cgt/other]]"}}   # commentary shape, no type
    assert find_page_type_mismatches([p], fm) == []
    assert find_missing_type([p], fm) == ["cgt/lost-its-type"]


def test_missing_type_is_wired_into_the_lint_orchestrator():
    """A check nobody renders is a check nobody acts on.

    Pins both ends: the dispatcher computes it, and `report` emits it in the
    JSON contract and the prose report.
    """
    from researchwiki.tasks import lint as lint_pkg
    from researchwiki.tasks.lint import report as lint_report
    assert lint_pkg.find_missing_type is find_missing_type
    dispatcher = open(lint_pkg.__file__, encoding="utf-8").read()
    assert "missing_type = find_missing_type(pages, pages_fm)" in dispatcher
    assert dispatcher.count("missing_type=missing_type") == 2   # json + prose
    rendered = open(lint_report.__file__, encoding="utf-8").read()
    assert '"missing_type": kw["missing_type"]' in rendered
    assert "Pages with no `type:`" in rendered


def test_metadata_report_separates_missing_supplementary_section(capsys):
    """A missing-only supplementary report still ends before the next H2."""
    from researchwiki.tasks.lint import report as lint_report

    lint_report._emit_metadata_sections({
        "missing_keywords": [],
        "missing_doi": [],
        "missing_hook": [],
        "missing_author_model": [],
        "acknowledged_legacy_provenance": [],
        "hook_too_long": [],
        "unquoted_wikilinks": [],
        "supp_yaml_missing": [{"page": "cgt/paper", "missing": ["data.csv"]}],
        "supp_orphans": [],
    })

    assert capsys.readouterr().out.endswith(
        "- **cgt/paper** → missing: data.csv\n\n"
    )


def test_find_page_type_mismatches_synthesis_with_paper_type(tmp_wiki):
    s = _mkpage(tmp_wiki, "synthesis/x", "")
    pages = [s]
    fm = {s: {"type": "paper"}}
    out = find_page_type_mismatches(pages, fm)
    assert any("synthesis" in r for _, r in out)


def test_find_page_type_mismatches_references_with_paper_type(tmp_wiki):
    r = _mkpage(tmp_wiki, "references/some-doc", "")
    pages = [r]
    fm = {r: {"type": "paper"}}
    out = find_page_type_mismatches(pages, fm)
    assert any("references" in r for _, r in out)


def test_find_category_drift(tmp_wiki):
    """YAML category disagrees with parent dir."""
    p = _mkpage(tmp_wiki, "cgt/x", "")
    pages = [p]
    fm = {p: {"category": ["compbio"]}}  # frontmatter says compbio, dir says cgt
    drift = find_category_drift(pages, fm)
    assert drift == [("cgt/x", "compbio", "cgt")]


def test_find_category_drift_no_drift(tmp_wiki):
    p = _mkpage(tmp_wiki, "cgt/x", "")
    pages = [p]
    fm = {p: {"category": ["cgt"]}}
    assert find_category_drift(pages, fm) == []


def test_find_missing_doi(tmp_wiki):
    a = _mkpage(tmp_wiki, "cgt/has-doi", "")
    b = _mkpage(tmp_wiki, "cgt/no-doi", "")
    c = _mkpage(tmp_wiki, "cgt/todo-doi", "")
    d = _mkpage(tmp_wiki, "cgt/no-doi-by-design", "")
    s = _mkpage(tmp_wiki, "synthesis/topic", "")  # excluded by dir
    pages = [a, b, c, d, s]
    fm = {
        a: {"type": "paper", "doi": "10.1/x"},
        b: {"type": "paper", "doi": ""},
        c: {"type": "paper", "doi": "TODO"},
        d: {"type": "paper", "no_doi_reason": "workshop poster, no DOI"},
        s: {"type": "synthesis"},
    }
    missing = find_missing_doi(pages, fm)
    assert missing == ["cgt/no-doi", "cgt/todo-doi"]
    assert "cgt/has-doi" not in missing
    assert "cgt/no-doi-by-design" not in missing  # escape hatch
    assert not any(k.startswith("synthesis/") for k in missing)


def test_root_bookkeeping_is_not_treated_as_a_paper(tmp_wiki):
    """`index.md` / `log.md` carry no frontmatter at all.

    Both checks default a missing `type:` to "paper", so without the
    root-bookkeeping skip they demand a DOI and keywords from the catalogue
    and the changelog — noise that never clears, because there is nothing
    correct to write there.
    """
    idx = _mkpage(tmp_wiki, "index", "# Wiki Index\n")
    log = _mkpage(tmp_wiki, "log", "## [2026-01-01] ingest | x\n")
    views = _mkpage(tmp_wiki, "views", "")
    paper = _mkpage(tmp_wiki, "cgt/real-paper", "")
    pages = [idx, log, views, paper]
    fm = {idx: {}, log: {}, views: {"type": "dashboard"}, paper: {"type": "paper"}}

    assert find_missing_doi(pages, fm) == ["cgt/real-paper"]
    assert [k for k, _ in find_missing_keywords(pages, fm)] == ["cgt/real-paper"]


def test_find_stem_year_drift(tmp_wiki):
    p = _mkpage(tmp_wiki, "cgt/foo-2024-bar", "")
    pages = [p]
    fm = {p: {"type": "paper", "year": 2025}}  # YAML says 2025, stem says 2024
    drift = find_stem_year_drift(pages, fm)
    assert drift == [{"page": "cgt/foo-2024-bar", "stem_year": 2024, "yaml_year": 2025}]


def test_find_stem_year_drift_lettered_year_match(tmp_wiki):
    """smith-2024b-... stem year matches YAML year=2024."""
    p = _mkpage(tmp_wiki, "cgt/smith-2024b-thing", "")
    pages = [p]
    fm = {p: {"type": "paper", "year": 2024}}
    assert find_stem_year_drift(pages, fm) == []


def test_find_missing_keywords(tmp_wiki):
    a = _mkpage(tmp_wiki, "cgt/has-keywords", "")
    b = _mkpage(tmp_wiki, "cgt/sparse-keywords", "")
    pages = [a, b]
    fm = {
        # Sized off MIN_KEYWORDS so this fixture tracks the floor instead of
        # encoding whatever it happened to be when the test was written.
        a: {"type": "paper", "keywords": [f"kw{i}" for i in range(MIN_KEYWORDS)]},
        b: {"type": "paper", "keywords": ["one"]},
    }
    out = find_missing_keywords(pages, fm)
    assert out == [("cgt/sparse-keywords", 1)]


def test_find_missing_keywords_covers_reference_types(tmp_wiki):
    """`keywords:` is required on reference-document pages too (whitepaper,
    guidance, protocol, book) per CLAUDE.md Page Types §3. Regression guard
    on the extension: a whitepaper without enough keywords must surface here."""
    ok = _mkpage(tmp_wiki, "references/anthropic-2026-good", "")
    thin = _mkpage(tmp_wiki, "references/anthropic-2026-thin", "")
    missing = _mkpage(tmp_wiki, "references/fda-2026-blank", "")
    guidance_ok = _mkpage(tmp_wiki, "references/fda-2026-good-guidance", "")
    book_thin = _mkpage(tmp_wiki, "references/kutz-2024-thin-book", "")
    pages = [ok, thin, missing, guidance_ok, book_thin]
    fm = {
        ok: {"type": "whitepaper",
             "keywords": [f"kw{i}" for i in range(MIN_KEYWORDS + 1)]},
        thin: {"type": "whitepaper", "keywords": ["only-one", "two"]},
        missing: {"type": "guidance"},  # no keywords key at all
        guidance_ok: {"type": "guidance",
                      "keywords": [f"kw{i}" for i in range(MIN_KEYWORDS)]},
        book_thin: {"type": "book", "keywords": []},
    }
    out = find_missing_keywords(pages, fm)
    assert ("references/anthropic-2026-thin", 2) in out
    assert ("references/fda-2026-blank", 0) in out
    assert ("references/kutz-2024-thin-book", 0) in out
    # Well-populated ones don't trip.
    assert all(k != "references/anthropic-2026-good" for k, _ in out)
    assert all(k != "references/fda-2026-good-guidance" for k, _ in out)


def test_find_missing_keywords_ignores_synthesis_idea_concept(tmp_wiki):
    """Synthesis / idea / concept pages don't need `keywords:` (their body
    carries the search substrate). Exempt from the check even when empty."""
    syn = _mkpage(tmp_wiki, "synthesis/no-kw-syn", "")
    idea = _mkpage(tmp_wiki, "ideas/no-kw-idea", "")
    concept = _mkpage(tmp_wiki, "concepts/no-kw-concept", "")
    pages = [syn, idea, concept]
    fm = {p: {"type": p.parent.name.rstrip("s"), "keywords": []} for p in pages}
    assert find_missing_keywords(pages, fm) == []


# ---------- staleness ----------

def test_find_stale_by_audit_count_threshold(tmp_wiki):
    """A page caching wiki_papers_at_audit:10 should flag once corpus grows by ≥5."""
    audit_page = _mkpage(tmp_wiki, "synthesis/suggested-additions", "")
    pages = [audit_page]
    # Pad with 16 paper pages to make corpus >= cached + threshold.
    for i in range(16):
        pages.append(_mkpage(tmp_wiki, f"cgt/paper-{i:02d}", ""))
    fm = {audit_page: {"wiki_papers_at_audit": "10"}}
    for p in pages[1:]:
        fm[p] = {"type": "paper"}
    out = find_stale_by_audit_count(pages, fm)
    assert len(out) == 1
    assert out[0][0] == audit_page
    assert out[0][1] == 10


def test_find_stale_by_audit_count_below_threshold(tmp_wiki):
    audit_page = _mkpage(tmp_wiki, "synthesis/suggested-additions", "")
    pages = [audit_page]
    for i in range(13):
        pages.append(_mkpage(tmp_wiki, f"cgt/p-{i}", ""))
    fm = {audit_page: {"wiki_papers_at_audit": "10"}}
    for p in pages[1:]:
        fm[p] = {"type": "paper"}
    # Only 13 papers vs cached 10 → delta=3 < threshold=5; should not fire.
    assert find_stale_by_audit_count(pages, fm) == []


def test_find_stale_synthesis_no_generated_at(tmp_wiki):
    """Pages without generated_at are silently skipped."""
    s = _mkpage(tmp_wiki, "synthesis/x", "[[cgt/foo]]")
    p = _mkpage(tmp_wiki, "cgt/foo", "")
    pages = [s, p]
    fm = {s: {}, p: {"type": "paper"}}
    known = {page_key(s), page_key(p)}
    assert find_stale_synthesis(pages, fm, known) == []


# ---------- concepts ----------

def test_find_concept_candidates_acronym_threshold():
    """Acronym appearing in 3 distinct pages should surface."""
    pages_body = {
        Path("cgt/a.md"): "We use FOOBAR everywhere.",
        Path("cgt/b.md"): "FOOBAR is the key tool here.",
        Path("cgt/c.md"): "Adopting FOOBAR for analysis.",
        Path("cgt/d.md"): "Unrelated content.",
    }
    out = find_concept_candidates(pages_body, existing_slugs=set())
    tokens = {tok for tok, *_ in out}
    assert "FOOBAR" in tokens


def test_find_concept_candidates_filters_existing_slugs():
    """If a slug exists, the acronym is filtered."""
    pages_body = {
        Path("cgt/a.md"): "FOOBAR is here.",
        Path("cgt/b.md"): "FOOBAR is here too.",
        Path("cgt/c.md"): "Yet another mention of FOOBAR.",
    }
    out = find_concept_candidates(pages_body, existing_slugs={"foobar"})
    assert all(tok != "FOOBAR" for tok, *_ in out)


def test_find_concept_candidates_filters_stop_acronyms():
    """Common stop-acronyms (DOI, YAML, ...) are filtered even if frequent."""
    pages_body = {
        Path("cgt/a.md"): "DOI YAML PDF API",
        Path("cgt/b.md"): "DOI YAML PDF API",
        Path("cgt/c.md"): "DOI YAML PDF API",
    }
    assert find_concept_candidates(pages_body, existing_slugs=set()) == []


def test_find_concept_candidates_filters_english_and_primitives():
    """All-caps English words and ubiquitous primitives are filtered."""
    pages_body = {
        Path("cgt/a.md"): "THIS AND RNA DNA are frequent.",
        Path("cgt/b.md"): "THIS AND RNA DNA again.",
        Path("cgt/c.md"): "THIS AND RNA DNA once more.",
    }
    assert find_concept_candidates(pages_body, existing_slugs=set()) == []


def test_find_concept_candidates_filters_structural_phrases():
    """Figure/table cross-references never surface as concepts."""
    pages_body = {
        Path("cgt/a.md"): "See Extended Data Fig for details.",
        Path("cgt/b.md"): "As in Extended Data Fig again.",
        Path("cgt/c.md"): "Extended Data Fig shows this.",
    }
    tokens = {tok for tok, *_ in find_concept_candidates(pages_body, set())}
    assert "Extended Data Fig" not in tokens
    assert "Extended Data" not in tokens


def test_find_concept_candidates_filters_venue_names():
    """Venue names in STOP_PHRASES are filtered."""
    pages_body = {
        Path("ai/a.md"): "Published in Nature Machine Intelligence.",
        Path("ai/b.md"): "Also Nature Machine Intelligence.",
        Path("ai/c.md"): "Cited from Nature Machine Intelligence.",
    }
    tokens = {tok for tok, *_ in find_concept_candidates(pages_body, set())}
    assert "Nature Machine Intelligence" not in tokens


def test_find_concept_candidates_reports_category_span():
    """A term spanning two categories reports n_categories=2."""
    pages_body = {
        Path("ai/a.md"): "RAPTOR is a hierarchy.",
        Path("single-cell/b.md"): "RAPTOR-style trees here.",
        Path("compbio/c.md"): "We adopt RAPTOR.",
    }
    out = find_concept_candidates(pages_body, existing_slugs=set())
    span = {tok: c for tok, _, c in out}
    assert span["RAPTOR"] == 3


# ---------- broken index.md bullets ----------
#
# `build_link_graph` excludes root meta pages from the broken-link scan, which
# is right for `log.md` (historical entries carry template fragments that were
# never links) and wrong for `index.md` (every line is a generated catalogue
# entry pointing at a page that is supposed to exist). These pin the narrow
# scan that covers the gap: a page deleted by hand leaves its bullet behind and
# no other check sees it.

@pytest.fixture
def tmp_index(tmp_wiki, monkeypatch):
    """`find_broken_index_bullets` reads `index_path()`, bound at import."""
    idx = tmp_wiki / "index.md"
    monkeypatch.setattr("researchwiki.tasks.lint.link_checks.index_path", lambda: idx)
    return idx


def test_index_bullet_pointing_at_a_deleted_page_is_reported(tmp_wiki, tmp_index):
    live = _mkpage(tmp_wiki, "cgt/foo-2024-bar")
    tmp_index.write_text(
        "# Wiki index\n\n## cgt\n\n"
        "- [[cgt/foo-2024-bar]] — **Foo** (*Nature* 2024): still here.\n"
        "- [[cgt/gone-2023-deleted-by-hand]] — **Gone** (*Cell* 2023): not here.\n"
    )
    assert find_broken_index_bullets({page_key(live)}) == [
        {"line": 6, "targets": ["cgt/gone-2023-deleted-by-hand"]}
    ]


def test_only_bullet_lines_are_scanned(tmp_wiki, tmp_index):
    """A heading or an intro paragraph is hand-written prose, not catalogue."""
    tmp_index.write_text(
        "# Wiki index\n\n"
        "Prose mentioning [[cgt/not-a-bullet]] in passing.\n\n"
        "## cgt\n"
    )
    assert find_broken_index_bullets(set()) == []


def test_a_hook_quoting_link_syntax_is_not_a_broken_link(tmp_wiki, tmp_index):
    """Regression: the first run of this check on the real wiki flagged
    `` `[[stem#slug]]` `` inside an idea page's hook — the hook describes the
    wiki's own anchor syntax. `strip_non_prose` is the same guard
    `build_link_graph` applies."""
    live = _mkpage(tmp_wiki, "ideas/supervision")
    tmp_index.write_text(
        "## ideas\n"
        "- [[ideas/supervision]] — content-addressed slugs orphan "
        "`[[stem#slug]]` anchors, which lint reports.\n"
    )
    assert find_broken_index_bullets({page_key(live)}) == []


def test_a_bare_stem_bullet_still_resolves(tmp_wiki, tmp_index):
    live = _mkpage(tmp_wiki, "cgt/foo-2024-bar")
    tmp_index.write_text("- [[foo-2024-bar]] — bare-stem form.\n")
    assert find_broken_index_bullets({page_key(live)}) == []


def test_no_index_file_is_not_an_error(tmp_wiki, tmp_index):
    assert find_broken_index_bullets(set()) == []


# ---------- orphan PDFs ----------

@pytest.fixture
def tmp_papers(tmp_wiki, tmp_path, monkeypatch):
    """`find_orphan_pdfs` walks both trees itself, so both bindings are patched.

    `tmp_wiki` patches `wiki_dir` for `paths` and `lint.walk`; `researchwiki.wiki`
    holds its own module-level binding, which is the one this function reads.
    """
    papers = tmp_path / "papers"
    papers.mkdir()
    monkeypatch.setattr("researchwiki.wiki.papers_dir", lambda: papers)
    monkeypatch.setattr("researchwiki.wiki.wiki_dir", lambda: tmp_wiki)
    return papers


def test_a_pdf_with_no_page_is_reported(tmp_wiki, tmp_papers):
    _mkpage(tmp_wiki, "cgt/foo-2024-bar")
    (tmp_papers / "foo-2024-bar.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_papers / "gone-2023-deleted-by-hand.pdf").write_bytes(b"%PDF-1.4\n")
    assert find_orphan_pdfs() == ["gone-2023-deleted-by-hand"]


def test_supplementary_pdfs_are_not_orphans(tmp_wiki, tmp_papers):
    """They belong to their parent page and are `find_supplementary_issues`'
    business; a recursive walk would report every one of them here."""
    _mkpage(tmp_wiki, "cgt/foo-2024-bar")
    (tmp_papers / "foo-2024-bar.pdf").write_bytes(b"%PDF-1.4\n")
    supp = tmp_papers / "foo-2024-bar.supp"
    supp.mkdir()
    (supp / "supplementary-tables.pdf").write_bytes(b"%PDF-1.4\n")
    assert find_orphan_pdfs() == []


def test_a_page_in_any_category_claims_its_pdf(tmp_wiki, tmp_papers):
    """The join key is the stem, not the path — a reference doc under
    `references/` claims `papers/{stem}.pdf` exactly as a paper page does."""
    _mkpage(tmp_wiki, "references/fda-2026-a-guidance-about-things")
    (tmp_papers / "fda-2026-a-guidance-about-things.pdf").write_bytes(b"%PDF-1.4\n")
    assert find_orphan_pdfs() == []


def test_no_papers_dir_is_not_an_error(tmp_papers, tmp_path, monkeypatch):
    monkeypatch.setattr("researchwiki.wiki.papers_dir",
                        lambda: tmp_path / "nonexistent")
    assert find_orphan_pdfs() == []


def test_a_page_with_no_frontmatter_fence_still_claims_its_pdf(tmp_wiki, tmp_papers):
    """Regression: `status` used to pass `read_pages()`, which returns None for
    a file with no leading `---` fence and drops it. That page's PDF is not
    orphaned — and `lint`, which walks every `*.md`, disagreed. The function now
    walks the tree itself so both callers get the same answer."""
    (tmp_wiki / "cgt").mkdir(parents=True, exist_ok=True)
    (tmp_wiki / "cgt" / "foo-2024-bar.md").write_text("# Hand-written, no YAML\n")
    (tmp_papers / "foo-2024-bar.pdf").write_bytes(b"%PDF-1.4\n")
    assert find_orphan_pdfs() == []


def test_status_reports_orphan_pdfs_in_workflow_state(tmp_path, monkeypatch, capsys):
    """`status` is the human-facing home for this one: an orphan PDF is
    workflow state (a file awaiting an action), the same shape as an `inbox/`
    drop, and `remove --keep-pdf` produces it on purpose. `lint --json` keeps
    the full stem list; `status` carries the count — the split the
    claim-overlap backlog already uses."""
    from researchwiki import paths
    from researchwiki.tasks import status as status_task

    (tmp_path / "wiki" / "cgt").mkdir(parents=True)
    (tmp_path / "papers").mkdir()
    (tmp_path / "wiki" / "cgt" / "foo-2024-bar.md").write_text(
        "---\ntype: paper\ncategory: [cgt]\n---\n\n## Summary\nbody\n")
    (tmp_path / "papers" / "foo-2024-bar.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "papers" / "gone-2023-deleted-by-hand.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "wiki" / "index.md").write_text("# index\n")
    (tmp_path / "wiki" / "log.md").write_text("# log\n")
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    status_task.main([])

    out = capsys.readouterr().out
    assert "papers/ PDFs with no page:      1" in out
    assert "gone-2023-deleted-by-hand.pdf" in out
    assert "foo-2024-bar.pdf" not in out, "the claimed PDF is not a finding"
